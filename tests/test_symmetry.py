"""Symmetric segment and pose editing checks."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from core.model import ModelCache
from core.symmetry import SYMMETRY_JOINTS, SYMMETRY_SEGMENTS


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"


class SymmetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(MODEL_DIR, Path("/tmp/smplx_symmetry_test"), device="cpu")

    def setUp(self):
        self.cache.reset(GENDER)

    def test_mapping_excludes_head_neck_spine(self):
        self.assertEqual(SYMMETRY_SEGMENTS[18], 19)
        self.assertEqual(SYMMETRY_JOINTS[21], 20)
        for excluded in (3, 6, 9, 12, 15):
            self.assertNotIn(excluded, SYMMETRY_SEGMENTS)
            self.assertNotIn(excluded, SYMMETRY_JOINTS)

    def test_symmetric_segment_writes_both_and_resets_both(self):
        self.cache.update_segment(GENDER, 18, 0.9, [0.8, 0.9], symmetric=True)
        segments = self.cache._segments(GENDER)
        self.assertIn("18", segments)
        self.assertIn("19", segments)
        self.assertAlmostEqual(segments["18"]["length"], segments["19"]["length"])
        self.assertEqual(segments["18"]["radii"], segments["19"]["radii"])
        self.assertNotIn("3", segments)

        self.cache.reset_segment(GENDER, 18, symmetric=True)
        self.assertNotIn("18", self.cache._segments(GENDER))
        self.assertNotIn("19", self.cache._segments(GENDER))

    def test_unpaired_segment_is_not_mirrored(self):
        self.cache.update_segment(GENDER, 3, 1.05, [0.9, 0.95], symmetric=True)
        segments = self.cache._segments(GENDER)
        self.assertIn("3", segments)
        self.assertNotIn("6", segments)

    def test_symmetric_pose_reaches_mirrored_targets(self):
        _model, _vertices, base_joints, _parents = self.cache.rest_geometry(GENDER)
        target_right = np.asarray(base_joints[21], dtype=np.float64) + np.array([0.0, 0.06, 0.0])
        mirror_left = np.array([-target_right[0], target_right[1], target_right[2]])
        self.cache.pose_joint(
            GENDER, 21, target_right.tolist(), full_body=True, symmetric=True
        )
        _model, _vertices, joints, _parents = self.cache.rest_geometry(GENDER)
        # The geometry is the *rest* pose; pose is a view state, so check the
        # rendered skeleton instead.
        _model, _vertices, posed_joints, _parents = self.cache._geometry(GENDER)
        self.assertLess(float(np.linalg.norm(posed_joints[21] - target_right)), 0.04)
        self.assertLess(float(np.linalg.norm(posed_joints[20] - mirror_left)), 0.04)


if __name__ == "__main__":
    unittest.main()
