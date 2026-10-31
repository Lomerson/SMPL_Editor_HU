"""Regression checks for the rest-pose geometry kernel.

Run from the project root with the torch3d environment:

    python -m unittest tests.test_deform -v

The model path can be overridden with ``SMPLX_MODEL_DIR``.
"""
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


class DeformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(MODEL_DIR, Path("/tmp/smplx_deform_test"), device="cpu")
        cls.model, cls.base_vertices, cls.base_joints, _ = cls.cache._geometry(GENDER)
        cls.weights = cls.model.lbs_weights.detach().cpu().numpy()
        cls.edges = _edges(cls.model)
        cls.base_edge_lengths = np.linalg.norm(
            cls.base_vertices[cls.edges[:, 0]] - cls.base_vertices[cls.edges[:, 1]],
            axis=1,
        )

    def setUp(self):
        self.cache.reset(GENDER)

    def _geometry(self):
        return self.cache._geometry(GENDER)

    def test_zero_edit_matches_legacy_smplx_output(self):
        # The old viewer returned SMPL-X's forward output (not the template
        # tensor) for an unedited T-pose; the shortcut in core.geometry keeps
        # that byte-for-byte.
        _model, expected = self.cache._output(GENDER)
        _model, vertices, joints, _parents = self._geometry()
        self.assertTrue(np.array_equal(vertices, expected.vertices[0].cpu().numpy()))
        self.assertTrue(
            np.array_equal(joints, expected.joints[0, : len(self.model.parents)].cpu().numpy())
        )

    def test_pose_matches_smplx_forward(self):
        import torch

        torch.manual_seed(7)
        body_pose = torch.randn(1, 63) * 0.15
        global_orient = torch.randn(1, 3) * 0.08
        self.cache.poses[GENDER] = body_pose
        self.cache.global_orients[GENDER] = global_orient
        with torch.no_grad():
            expected = self.model(body_pose=body_pose, global_orient=global_orient)
        _model, vertices, joints, _parents = self._geometry()
        self.assertLess(
            np.abs(vertices - expected.vertices[0].detach().cpu().numpy()).max(), 1e-5
        )
        self.assertLess(
            np.abs(
                joints - expected.joints[0, : len(self.model.parents)].detach().cpu().numpy()
            ).max(),
            1e-5,
        )

    def test_spine_length_moves_branches_without_gap(self):
        for factor in (1.3, 0.7):
            with self.subTest(factor=factor):
                self.setUp()
                self.cache.update_segment(GENDER, 3, factor, [1.0, 1.0])
                _model, vertices, joints, _parents = self._geometry()

                # The whole upper body must move, not only the torso surface.
                for joint in (6, 9, 12, 15):
                    self.assertGreater(
                        np.linalg.norm(joints[joint] - self.base_joints[joint]), 1e-4
                    )

                # Dominant arm/head vertices follow their joints rigidly.  A
                # hard seam or a missed branch would leave hundreds of them
                # behind by centimetres.
                arm = (self.weights[:, 17] + self.weights[:, 19]) > 0.95
                head = self.weights[:, 15] > 0.95
                arm_delta = vertices[arm] - self.base_vertices[arm]
                head_delta = vertices[head] - self.base_vertices[head]
                self.assertLess(
                    np.abs(arm_delta - (joints[17] - self.base_joints[17])).max(), 1e-3
                )
                self.assertLess(
                    np.abs(head_delta - (joints[15] - self.base_joints[15])).max(), 1e-3
                )

                edge_lengths = np.linalg.norm(
                    vertices[self.edges[:, 0]] - vertices[self.edges[:, 1]], axis=1
                )
                ratios = edge_lengths / np.maximum(self.base_edge_lengths, 1e-9)
                self.assertLess(ratios.max(), 1.55)
                self.assertGreater(ratios.min(), 0.5)

    def test_all_non_head_segments_accept_length(self):
        for child in range(1, 22):
            if child == 15:
                continue
            with self.subTest(child=child):
                self.cache.update_segment(GENDER, child, 1.05, [1.0, 1.0])
        # A head length request is deliberately ignored (girth still applies).
        self.cache.update_segment(GENDER, 15, 1.2, [1.0, 1.0])
        self.assertEqual(self.cache._segments(GENDER)["15"]["length"], 1.0)

    def test_girth_does_not_bleed_across_the_midline(self):
        # Thinning the left thigh must not pull the right thigh or the crotch.
        self.cache.update_segment(GENDER, 4, 1.0, [0.7, 0.7])
        _model, vertices, _joints, _parents = self._geometry()
        left = self.weights[:, 1] > 0.8
        right = self.weights[:, 2] > 0.8
        crotch = (
            (self.weights[:, 1] > 0.25)
            & (self.weights[:, 2] > 0.25)
            & (self.weights[:, 4] < 0.01)
            & (self.weights[:, 5] < 0.01)
        )
        self.assertGreater(np.linalg.norm(vertices[left] - self.base_vertices[left], axis=1).max(), 0.005)
        self.assertLess(np.linalg.norm(vertices[right] - self.base_vertices[right], axis=1).max(), 1e-4)
        self.assertLess(np.linalg.norm(vertices[crotch] - self.base_vertices[crotch], axis=1).max(), 1e-4)

        # The same rule protects the opposite arm.
        self.setUp()
        self.cache.update_segment(GENDER, 18, 1.0, [0.7, 0.7])
        _model, vertices, _joints, _parents = self._geometry()
        left_arm = self.weights[:, 16] > 0.8
        right_arm = self.weights[:, 17] > 0.8
        self.assertGreater(
            np.linalg.norm(vertices[left_arm] - self.base_vertices[left_arm], axis=1).max(),
            0.005,
        )
        self.assertLess(
            np.linalg.norm(vertices[right_arm] - self.base_vertices[right_arm], axis=1).max(),
            1e-4,
        )

    def test_elbow_girth_has_no_ring(self):
        axis = self.base_joints[19] - self.base_joints[17]
        length = float(np.linalg.norm(axis))
        axis = axis / length
        relative = self.base_vertices - self.base_joints[17]
        axial = relative @ axis
        radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)

        # Thinning the distal upper arm (100%) and the proximal forearm (0%)
        # is the exact case that used to leave an unscaled ring at joint 19.
        self.cache.update_segment(GENDER, 19, 1.0, [1.0, 0.7])
        self.cache.update_segment(GENDER, 21, 1.0, [0.7, 1.0])
        _model, vertices, _joints, _parents = self._geometry()

        relative2 = vertices - self.base_joints[17]
        axial2 = relative2 @ axis
        radial2 = np.linalg.norm(relative2 - axial2[:, None] * axis, axis=1)
        ratio = radial2 / np.maximum(radial, 1e-9)

        arm = (self.weights[:, 17] + self.weights[:, 19] + self.weights[:, 21]) > 0.5
        near_elbow = (np.abs(axial - length) < 0.012) & (radial < 0.09) & arm
        self.assertGreater(int(near_elbow.sum()), 20)
        self.assertLess(ratio[near_elbow].max(), 0.90)
        self.assertGreater(ratio[near_elbow].min(), 0.65)
        self.assertLess(float(ratio[near_elbow].std()), 0.05)

        # Scan across the joint: the radius must keep tapering rather than
        # jumping back towards 1.0 in an unscaled ring.
        band = (
            (axial > length - 0.03)
            & (axial < length + 0.03)
            & (radial < 0.09)
            & arm
        )
        bins = np.linspace(length - 0.03, length + 0.03, 7)
        means = []
        for start, end in zip(bins[:-1], bins[1:]):
            selected = band & (axial >= start) & (axial < end)
            if selected.any():
                means.append(float(ratio[selected].mean()))
        self.assertGreater(len(means), 4)
        self.assertLess(max(means), 0.85)
        self.assertGreater(min(means), 0.65)
        for previous, current in zip(means[:-1], means[1:]):
            self.assertLess(current, previous + 0.05)

        # Both bones must expose the same elbow value to the UI.
        segments = self.cache._segments(GENDER)
        self.assertAlmostEqual(segments["19"]["radii"][-1], segments["21"]["radii"][0])


if __name__ == "__main__":
    unittest.main()
