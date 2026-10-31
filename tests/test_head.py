"""Regression checks for the dedicated head deformer."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from core.model import ModelCache


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"


def _edges(model):
    faces = np.asarray(model.faces)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


class HeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(MODEL_DIR, Path("/tmp/smplx_head_test"), device="cpu")
        cls.model, cls.base_vertices, cls.base_joints, _ = cls.cache._geometry(GENDER)
        cls.weights = cls.model.lbs_weights.detach().cpu().numpy()
        cls.head = cls.weights[:, 15] > 0.9
        cls.jaw = cls.weights[:, 22] > 0.9
        cls.eye_left = cls.weights[:, 23] > 0.99
        cls.eye_right = cls.weights[:, 24] > 0.99
        cls.edges = _edges(cls.model)
        cls.base_edge_lengths = np.linalg.norm(
            cls.base_vertices[cls.edges[:, 0]] - cls.base_vertices[cls.edges[:, 1]],
            axis=1,
        )

    def setUp(self):
        self.cache.reset(GENDER)

    def _geometry(self):
        return self.cache._geometry(GENDER)

    def _bbox(self, vertices, mask):
        points = vertices[mask]
        return points.min(axis=0), points.max(axis=0)

    def _eye_radius(self, vertices, joints, mask, joint):
        return float(np.linalg.norm(vertices[mask] - joints[joint], axis=1).mean())

    def test_default_head_matches_legacy_output(self):
        _model, expected = self.cache._output(GENDER)
        self.cache.update_head(GENDER, {})
        _model, vertices, joints, _parents = self._geometry()
        self.assertTrue(np.array_equal(vertices, expected.vertices[0].cpu().numpy()))
        self.assertTrue(
            np.array_equal(joints, expected.joints[0, : len(self.model.parents)].cpu().numpy())
        )

    def test_head_length_is_plus_y_and_keeps_thickness(self):
        base_min, base_max = self._bbox(self.base_vertices, self.head)
        self.cache.update_head(GENDER, {"head_length": 1.2})
        _model, vertices, joints, _parents = self._geometry()
        head_min, head_max = self._bbox(vertices, self.head)

        self.assertGreater(head_max[1], base_max[1] + 0.02)
        self.assertLess(head_min[1], base_min[1])
        # X/Z are width/depth; the length slider must not change them.
        np.testing.assert_allclose(
            [head_min[0], head_max[0], head_min[2], head_max[2]],
            [base_min[0], base_max[0], base_min[2], base_max[2]],
            atol=2e-5,
        )

        pivot = self.base_joints[15]
        expected_center = pivot + np.array([1.0, 1.2, 1.0]) * (
            self.base_joints[23] - pivot
        )
        np.testing.assert_allclose(joints[23], expected_center, atol=2e-5)
        # A length edit must not change the eyeball radius.
        base_radius = self._eye_radius(
            self.base_vertices, self.base_joints, self.eye_left, 23
        )
        radius = self._eye_radius(vertices, joints, self.eye_left, 23)
        self.assertAlmostEqual(radius / base_radius, 1.0, delta=0.01)

    def test_head_width_scales_eye_centres_and_spheres(self):
        base_min, base_max = self._bbox(self.base_vertices, self.head)
        self.cache.update_head(GENDER, {"head_width": 0.85})
        _model, vertices, joints, _parents = self._geometry()
        head_min, head_max = self._bbox(vertices, self.head)

        self.assertLess(head_max[0] - head_min[0], base_max[0] - base_min[0])
        np.testing.assert_allclose(
            [head_min[1], head_max[1], head_min[2], head_max[2]],
            [base_min[1], base_max[1], base_min[2], base_max[2]],
            atol=2e-5,
        )
        pivot = self.base_joints[15]
        expected = pivot + np.array([0.85, 1.0, 1.0]) * (self.base_joints[23] - pivot)
        np.testing.assert_allclose(joints[23], expected, atol=2e-5)

        base_radius = self._eye_radius(
            self.base_vertices, self.base_joints, self.eye_left, 23
        )
        radius = self._eye_radius(vertices, joints, self.eye_left, 23)
        self.assertAlmostEqual(radius / base_radius, float(np.sqrt(0.85)), delta=0.01)

    def test_head_depth_scales_front_back_and_eye_radius(self):
        base_min, base_max = self._bbox(self.base_vertices, self.head)
        self.cache.update_head(GENDER, {"head_depth": 0.85})
        _model, vertices, joints, _parents = self._geometry()
        head_min, head_max = self._bbox(vertices, self.head)

        self.assertLess(head_max[2] - head_min[2], base_max[2] - base_min[2])
        np.testing.assert_allclose(
            [head_min[0], head_max[0], head_min[1], head_max[1]],
            [base_min[0], base_max[0], base_min[1], base_max[1]],
            atol=2e-5,
        )
        base_radius = self._eye_radius(
            self.base_vertices, self.base_joints, self.eye_left, 23
        )
        radius = self._eye_radius(vertices, joints, self.eye_left, 23)
        self.assertAlmostEqual(radius / base_radius, float(np.sqrt(0.85)), delta=0.01)

    def test_jaw_scales_around_edited_jaw_joint(self):
        self.cache.update_head(GENDER, {"jaw": 1.2})
        _model, vertices, joints, _parents = self._geometry()
        np.testing.assert_allclose(joints[22], self.base_joints[22], atol=1e-6)
        before = np.linalg.norm(self.base_vertices[self.jaw] - self.base_joints[22], axis=1)
        after = np.linalg.norm(vertices[self.jaw] - joints[22], axis=1)
        self.assertAlmostEqual(float((after / before).mean()), 1.2, delta=0.01)

        edge_lengths = np.linalg.norm(
            vertices[self.edges[:, 0]] - vertices[self.edges[:, 1]], axis=1
        )
        ratios = edge_lengths / np.maximum(self.base_edge_lengths, 1e-9)
        self.assertLess(float(ratios.max()), 2.0)
        self.assertTrue(np.isfinite(vertices).all())

    def test_eye_scale_changes_radius_not_center(self):
        self.cache.update_head(GENDER, {"eye": 0.85})
        _model, vertices, joints, _parents = self._geometry()
        np.testing.assert_allclose(joints[23], self.base_joints[23], atol=1e-6)
        np.testing.assert_allclose(joints[24], self.base_joints[24], atol=1e-6)
        base_radius = self._eye_radius(
            self.base_vertices, self.base_joints, self.eye_left, 23
        )
        radius = self._eye_radius(vertices, joints, self.eye_left, 23)
        self.assertAlmostEqual(radius / base_radius, 0.85, delta=0.01)

    def test_combined_head_and_spine_edit(self):
        self.cache.update_segment(GENDER, 3, 1.1, [1.0, 1.0])
        self.cache.update_head(
            GENDER, {"head_length": 1.15, "head_width": 0.9, "jaw": 1.1, "eye": 1.0}
        )
        _model, vertices, joints, _parents = self._geometry()
        self.assertTrue(np.isfinite(vertices).all())
        self.assertGreater(joints[15][1], self.base_joints[15][1])
        # The head edit must not disturb the arms; they still follow the spine.
        arm = (self.weights[:, 16] + self.weights[:, 18]) > 0.95
        self.assertGreater(
            np.linalg.norm(vertices[arm] - self.base_vertices[arm], axis=1).max(),
            0.005,
        )


if __name__ == "__main__":
    unittest.main()
