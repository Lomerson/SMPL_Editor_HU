"""Uploaded SMPL-X-topology models replace the current gender model."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import trimesh

from core.model import ModelCache


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
RESOURCES = (
    RESOURCE_DIR / "smplx_fit_7",
    RESOURCE_DIR / "smplx_left_forearm_fixed_7",
)
RESOURCE = RESOURCES[0] / "fitted_smplx.npz"
GENDER = "female"


class CustomModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        if not RESOURCE.is_file():
            raise unittest.SkipTest(f"missing uploaded-model fixture {RESOURCE}")
        cls.cache = ModelCache(
            MODEL_DIR, Path("/tmp/smplx_custom_model_test"), device="cpu"
        )

    def setUp(self):
        self.cache.switch_standard_model(GENDER, record=False)
        self.cache.histories.pop(GENDER, None)

    def test_uploaded_surface_matches_each_folders_obj(self):
        for folder in RESOURCES:
            with self.subTest(folder=folder.name):
                npz = folder / "fitted_smplx.npz"
                obj = folder / "fitted_smplx.obj"
                if not npz.is_file() or not obj.is_file():
                    self.skipTest(f"missing fixture {folder}")
                self.cache.replace_custom_model(GENDER, npz.read_bytes())
                _model, vertices, joints, _parents = self.cache._geometry(GENDER)
                reference = trimesh.load(obj, process=False)
                self.assertLess(
                    float(np.abs(np.asarray(vertices) - np.asarray(reference.vertices)).max()),
                    1e-5,
                )
                self.assertEqual(np.asarray(joints).shape, (55, 3))
                self.assertEqual(
                    float(np.abs(self.cache.poses[GENDER].detach().cpu().numpy()).max()),
                    0.0,
                )

    def test_upload_replaces_model_starts_tpose_and_undoes(self):
        payload = RESOURCE.read_bytes()
        result = self.cache.replace_custom_model(GENDER, payload)
        self.assertEqual(result["label"], "替换为：自选模型")
        self.assertTrue(self.cache.available(GENDER))
        self.assertTrue(self.cache.model_identities[GENDER].startswith("custom:"))
        model = self.cache._model(GENDER)
        self.assertTrue(getattr(model, "is_custom", False))
        _model, vertices, joints, parents = self.cache._geometry(GENDER)
        self.assertEqual(np.asarray(vertices).shape, (10475, 3))
        self.assertEqual(np.asarray(joints).shape, (55, 3))
        self.assertEqual(len(parents), 55)
        self.assertEqual(
            float(np.abs(self.cache.poses[GENDER].detach().cpu().numpy()).max()),
            0.0,
        )
        history = self.cache.history_state(GENDER)
        self.assertEqual(history["entries"][-1]["label"], "替换为：自选模型")

        undo = self.cache.undo(GENDER)
        self.assertTrue(undo["ok"])
        self.assertEqual(undo["label"], "替换为：自选模型")
        self.assertEqual(self.cache.model_identities[GENDER], "standard")
        self.assertEqual(type(self.cache._model(GENDER)).__name__, "SMPLX")

    def test_custom_pose_control_changes_geometry(self):
        self.cache.replace_custom_model(GENDER, RESOURCE.read_bytes())
        _model, vertices, joints, _parents = self.cache._geometry(GENDER)
        target = np.asarray(joints[18], dtype=np.float64) + np.array([0.0, 0.2, 0.0])
        self.cache.pose_joint(GENDER, 18, target.tolist(), full_body=False, record=False)
        pose_max = float(np.abs(self.cache.poses[GENDER].detach().cpu().numpy()).max())
        _model, posed, _joints, _parents = self.cache._geometry(GENDER)
        delta = float(np.linalg.norm(np.asarray(posed) - np.asarray(vertices), axis=1).mean())
        self.assertGreater(pose_max, 0.0)
        self.assertGreater(delta, 1e-4)

    def test_switch_standard_records_model_label(self):
        self.cache.replace_custom_model(GENDER, RESOURCE.read_bytes())
        result = self.cache.switch_standard_model(GENDER)
        self.assertEqual(result["label"], "替换为：女性")
        self.assertEqual(self.cache.model_identities[GENDER], "standard")
        self.assertEqual(
            self.cache.history_state(GENDER)["entries"][-1]["label"],
            "替换为：女性",
        )
        self.assertEqual(len(self.cache._segments(GENDER)), 0)

    def test_invalid_npz_is_rejected(self):
        with self.assertRaises(ValueError):
            self.cache.replace_custom_model(GENDER, b"not an npz")
        self.assertEqual(self.cache.model_identities.get(GENDER, "standard"), "standard")


if __name__ == "__main__":
    unittest.main()
