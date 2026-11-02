"""Timeline undo/redo checks for the editor history."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from core import autofit
from core.model import ModelCache


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"
FRONT = Path("/home/lomerson/MyProjects/ui/front.jpg")


class HistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(
            MODEL_DIR, Path("/tmp/smplx_history_test"), device="cpu",
            detector_dir="/home/lomerson/MyProjects/AnimePoseDetector",
        )

    def setUp(self):
        self.cache.reset(GENDER, record=False)
        self.cache.histories.pop(GENDER, None)

    def test_segment_edit_undo_and_jump_redo(self):
        history = self.cache.history_state(GENDER)
        self.assertEqual(len(history["entries"]), 1)
        self.assertFalse(history["can_undo"])

        self.cache.update_segment(GENDER, 18, 0.9, [0.8, 0.9], symmetric=True)
        history = self.cache.history_state(GENDER)
        self.assertEqual(len(history["entries"]), 2)
        self.assertTrue(history["can_undo"])
        self.assertIn("对称", history["entries"][-1]["label"])
        entry_id = history["entries"][-1]["id"]

        result = self.cache.undo(GENDER)
        self.assertTrue(result["ok"])
        self.assertNotIn("18", self.cache._segments(GENDER))
        self.assertNotIn("19", self.cache._segments(GENDER))
        self.assertTrue(result["history"]["can_redo"])

        redo = self.cache.jump_history(GENDER, entry_id)
        self.assertTrue(redo["ok"])
        self.assertIn("18", self.cache._segments(GENDER))
        self.assertIn("19", self.cache._segments(GENDER))

    def test_new_edit_after_undo_truncates_future(self):
        self.cache.update_head(GENDER, {"head_length": 1.1})
        first_id = self.cache.history_state(GENDER)["entries"][-1]["id"]
        self.cache.update_head(GENDER, {"head_length": 1.2})
        self.cache.jump_history(GENDER, first_id)
        self.assertTrue(self.cache.history_state(GENDER)["can_redo"])

        self.cache.update_segment(GENDER, 6, 1.05, [1.0, 1.0])
        history = self.cache.history_state(GENDER)
        self.assertFalse(history["can_redo"])
        labels = [item["label"] for item in history["entries"]]
        self.assertIn("头部编辑", labels)
        self.assertIn("骨段", labels[-1])
        self.assertEqual(len(history["entries"]), 3)

    def test_head_displacement_and_reset_are_undoable(self):
        self.cache.update_head(GENDER, {"head_length": 1.1})
        self.cache.update_displacement(GENDER, np.zeros((10475, 3), dtype=np.float32))
        self.cache.reset(GENDER)
        self.assertIsNone(self.cache.displacement(GENDER))

        undo = self.cache.undo(GENDER)
        self.assertTrue(undo["ok"])
        self.assertIsNotNone(self.cache.displacement(GENDER))
        undo = self.cache.undo(GENDER)
        self.assertTrue(undo["ok"])
        self.assertIsNone(self.cache.displacement(GENDER))
        self.assertAlmostEqual(self.cache.head(GENDER)["head_length"], 1.1)

    def test_pose_edit_is_undoable(self):
        _model, _vertices, base_joints, _parents = self.cache._geometry(GENDER)
        target = (np.asarray(base_joints[21], dtype=np.float64) + np.array([0.0, 0.05, 0.0])).tolist()
        self.cache.pose_joint(GENDER, 21, target, full_body=True)
        history = self.cache.history_state(GENDER)
        self.assertEqual(len(history["entries"]), 2)
        self.assertIn("姿态", history["entries"][-1]["label"])

        result = self.cache.undo(GENDER)
        self.assertTrue(result["ok"])
        _model, _vertices, joints, _parents = self.cache._geometry(GENDER)
        self.assertLess(float(np.linalg.norm(joints[21] - base_joints[21])), 1e-5)

    def test_auto_fit_is_one_history_entry(self):
        if not FRONT.is_file():
            self.skipTest("missing front reference")
        before = len(self.cache.history_state(GENDER)["entries"])
        autofit.run(self.cache, GENDER, path=str(FRONT))
        history = self.cache.history_state(GENDER)
        self.assertEqual(len(history["entries"]), before + 1)
        self.assertEqual(history["entries"][-1]["label"], "Auto-fit v0")
        self.assertTrue(history["can_undo"])


if __name__ == "__main__":
    unittest.main()
