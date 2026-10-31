"""Contract checks for the rigged GLB / params export."""
from __future__ import annotations

import json
import os
import struct
import unittest
from pathlib import Path

import numpy as np

from core.model import ModelCache


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"


def _read_glb(path):
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack("<III", data[:12])
    if magic != 0x46546C67 or version != 2 or total != len(data):
        raise AssertionError("bad GLB header")
    json_length, json_type = struct.unpack("<II", data[12:20])
    if json_type != 0x4E4F534A:
        raise AssertionError("bad JSON chunk")
    payload = json.loads(data[20:20 + json_length])
    binary_offset = 20 + json_length + 8
    bin_length, bin_type = struct.unpack("<II", data[20 + json_length:binary_offset])
    if bin_type != 0x004E4942 or bin_length != len(data) - binary_offset:
        raise AssertionError("bad BIN chunk")
    return payload, data, binary_offset


def _accessor(payload, data, binary_offset, index):
    accessor = payload["accessors"][index]
    view = payload["bufferViews"][accessor["bufferView"]]
    dtype = {
        5126: np.float32,
        5123: np.uint16,
        5125: np.uint32,
        5121: np.uint8,
    }[accessor["componentType"]]
    components = {
        "SCALAR": 1,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[accessor["type"]]
    return np.frombuffer(
        data,
        dtype=dtype,
        count=accessor["count"] * components,
        offset=binary_offset + view["byteOffset"],
    ).reshape(accessor["count"], components)


class GlbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(MODEL_DIR, Path("/tmp/smplx_glb_test"), device="cpu")
        cls.model, cls.rest_vertices, cls.rest_joints, _ = cls.cache.rest_geometry(GENDER)
        cls.glb = cls.cache.glb(GENDER)
        cls.payload, cls.data, cls.binary_offset = _read_glb(cls.glb)
        cls.primitive = cls.payload["meshes"][0]["primitives"][0]

    def setUp(self):
        self.cache.reset(GENDER)

    def test_glb_has_skin_and_weights(self):
        payload = self.payload
        self.assertEqual(payload["asset"]["version"], "2.0")
        self.assertEqual(len(payload["skins"]), 1)
        skin = payload["skins"][0]
        self.assertEqual(len(skin["joints"]), 55)
        self.assertEqual(skin["joints"], list(range(55)))

        attributes = self.primitive["attributes"]
        for key in ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0"):
            self.assertIn(key, attributes)

        positions = _accessor(
            payload, self.data, self.binary_offset, attributes["POSITION"]
        )
        self.assertEqual(positions.shape, (len(self.rest_vertices), 3))
        np.testing.assert_allclose(positions, self.rest_vertices, atol=2e-6)

        joints = _accessor(
            payload, self.data, self.binary_offset, attributes["JOINTS_0"]
        )
        self.assertEqual(joints.shape, (len(self.rest_vertices), 4))
        self.assertGreaterEqual(int(joints.min()), 0)
        self.assertLess(int(joints.max()), 55)

        weights = _accessor(
            payload, self.data, self.binary_offset, attributes["WEIGHTS_0"]
        )
        self.assertEqual(weights.shape, (len(self.rest_vertices), 4))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)

        inverse_bind = _accessor(
            payload, self.data, self.binary_offset, skin["inverseBindMatrices"]
        )
        self.assertEqual(inverse_bind.shape, (55, 16))
        # The bind matrix is the inverse of the joint's rest translation.
        for index in range(55):
            # glTF stores matrices column-major; transpose for row-major
            # indexing of the translation column.
            matrix = inverse_bind[index].reshape(4, 4).T
            np.testing.assert_allclose(
                matrix[:3, 3], -self.rest_joints[index], atol=2e-6
            )

    def test_glb_skin_reconstructs_rest_mesh(self):
        payload, data, offset = self.payload, self.data, self.binary_offset
        attributes = self.primitive["attributes"]
        positions = _accessor(payload, data, offset, attributes["POSITION"])
        weights = _accessor(payload, data, offset, attributes["WEIGHTS_0"])
        joints = _accessor(payload, data, offset, attributes["JOINTS_0"])
        inverse_bind = _accessor(
            payload, data, offset, payload["skins"][0]["inverseBindMatrices"]
        ).reshape(55, 4, 4).transpose(0, 2, 1)

        parents = self.model.parents.detach().cpu().numpy()
        nodes = payload["nodes"]
        globals_ = [None] * 55
        for index in range(55):
            local = np.eye(4, dtype=np.float64)
            local[:3, 3] = nodes[index]["translation"]
            parent = int(parents[index])
            globals_[index] = (
                local if parent < 0 else globals_[parent] @ local
            )
        skin_matrices = np.stack(
            [globals_[index] @ inverse_bind[index] for index in range(55)]
        )

        homogeneous = np.concatenate(
            [positions.astype(np.float64), np.ones((len(positions), 1))], axis=1
        )
        blended = np.zeros((len(positions), 4), dtype=np.float64)
        for influence in range(4):
            matrices = skin_matrices[joints[:, influence]]
            blended += weights[:, influence, None] * np.einsum(
                "vab,vb->va", matrices, homogeneous
            )
        np.testing.assert_allclose(blended[:, :3], positions, atol=1e-4)

    def test_glb_is_rest_pose_and_contains_identity(self):
        self.cache.update_segment(GENDER, 3, 1.2, [1.0, 1.0])
        self.cache.update_head(
            GENDER, {"head_length": 1.1, "head_width": 0.95, "jaw": 1.05}
        )
        import torch

        torch.manual_seed(3)
        self.cache.poses[GENDER] = torch.randn(1, 63) * 0.12
        self.cache.global_orients[GENDER] = torch.randn(1, 3) * 0.05
        self.cache._invalidate_geometry(GENDER)

        target = self.cache.glb(GENDER)
        payload, data, offset = _read_glb(target)
        positions = _accessor(
            payload,
            data,
            offset,
            payload["meshes"][0]["primitives"][0]["attributes"]["POSITION"],
        )
        _model, rest_vertices, rest_joints, _parents = self.cache.rest_geometry(GENDER)
        np.testing.assert_allclose(positions, rest_vertices, atol=2e-6)

        _model, posed_vertices, _posed_joints, _parents = self.cache._geometry(GENDER)
        self.assertGreater(
            float(np.abs(posed_vertices - rest_vertices).max()), 1e-4
        )

        params = json.loads(self.cache.params(GENDER).read_text(encoding="utf-8"))
        self.assertEqual(params["schema"], "smplx-avatar/1")
        self.assertEqual(params["gender"], GENDER)
        self.assertEqual(params["body_pose"], [0.0] * 63)
        self.assertTrue(params["tpose"])
        self.assertEqual(params["head"]["head_length"], 1.1)
        self.assertEqual(params["segments"]["3"]["length"], 1.2)
        self.assertEqual(len(params["skeleton"]["joints"]), 55)
        self.assertEqual(len(params["skeleton"]["parents"]), 55)

        # The exported joint node hierarchy must carry the edited rest joints.
        node0 = payload["nodes"][0]
        np.testing.assert_allclose(
            node0["translation"], rest_joints[0], atol=2e-6
        )
        node3 = payload["nodes"][3]
        parent = int(np.asarray(self.model.parents.cpu())[3])
        np.testing.assert_allclose(
            node3["translation"], rest_joints[3] - rest_joints[parent], atol=2e-6
        )


if __name__ == "__main__":
    unittest.main()
