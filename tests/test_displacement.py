"""Checks for the Blender displacement round-trip input."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

from core.model import ModelCache


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"


class DisplacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(MODEL_DIR, Path("/tmp/smplx_displacement_test"), device="cpu")
        cls.model, cls.base_vertices, cls.base_joints, _ = cls.cache.rest_geometry(GENDER)

    def setUp(self):
        self.cache.reset(GENDER)

    def test_displacement_is_applied_to_rest_and_glb(self):
        displacement = np.zeros_like(self.base_vertices)
        displacement[:37, 1] = 0.0125
        self.cache.update_displacement(GENDER, displacement)

        _model, vertices, joints, _parents = self.cache.rest_geometry(GENDER)
        np.testing.assert_allclose(vertices - self.base_vertices, displacement, atol=1e-6)
        np.testing.assert_allclose(joints, self.base_joints, atol=1e-6)

        glb = self.cache.glb(GENDER)
        self.assertTrue(glb.is_file())
        params = json.loads(self.cache.params(GENDER).read_text(encoding="utf-8"))
        self.assertIsNotNone(params["displacement"])
        self.assertEqual(params["displacement"]["shape"], [len(vertices), 3])
        npy = self.cache.cache_dir / f"avatar_{GENDER}_displacement.npy"
        self.assertTrue(npy.is_file())
        loaded = np.load(npy)
        np.testing.assert_allclose(loaded, displacement, atol=1e-6)

    def test_displacement_survives_pose_and_can_be_cleared(self):
        displacement = np.zeros_like(self.base_vertices)
        displacement[:, 2] += 0.004
        self.cache.update_displacement(GENDER, displacement)
        import torch

        torch.manual_seed(5)
        self.cache.poses[GENDER] = torch.randn(1, 63) * 0.08
        self.cache.global_orients[GENDER] = torch.randn(1, 3) * 0.04
        self.cache._invalidate_geometry(GENDER)
        _model, posed_vertices, _joints, _parents = self.cache._geometry(GENDER)
        self.assertGreater(
            float(np.abs(posed_vertices - (self.base_vertices + displacement)).max()),
            1e-4,
        )
        _model, rest_vertices, _joints, _parents = self.cache.rest_geometry(GENDER)
        np.testing.assert_allclose(
            rest_vertices, self.base_vertices + displacement, atol=1e-6
        )

        self.cache.reset_displacement(GENDER)
        _model, vertices, _joints, _parents = self.cache.rest_geometry(GENDER)
        np.testing.assert_allclose(vertices, self.base_vertices, atol=1e-6)

    def test_displacement_shape_is_validated(self):
        with self.assertRaises(ValueError):
            self.cache.update_displacement(GENDER, np.zeros((10, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.cache.update_displacement(
                GENDER, np.full((len(self.base_vertices), 3), np.nan, dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
