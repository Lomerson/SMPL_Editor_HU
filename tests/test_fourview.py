"""Checks for the four-view T-pose reference zip."""
from __future__ import annotations

import io
import json
import os
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from core.model import ModelCache


MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"


class FourViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        cls.cache = ModelCache(MODEL_DIR, Path("/tmp/smplx_fourview_test"), device="cpu")
        cls.target = cls.cache.fourview(GENDER)
        cls.archive = zipfile.ZipFile(cls.target)
        cls.names = set(cls.archive.namelist())
        cls.metadata = json.loads(cls.archive.read("views.json"))
        cls.images = {
            name: Image.open(io.BytesIO(cls.archive.read(f"{name}.png"))).convert("RGB")
            for name in ("front", "back", "left", "right")
        }

    def setUp(self):
        self.cache.reset(GENDER)

    def test_zip_contains_four_views_and_metadata(self):
        self.assertEqual(
            self.names,
            {"front.png", "back.png", "left.png", "right.png", "views.json"},
        )
        self.assertEqual(self.metadata["schema"], "smplx-fourview/1")
        self.assertTrue(self.metadata["tpose"])
        self.assertEqual(self.metadata["resolution"], [1024, 1024])
        self.assertGreater(self.metadata["scale_px_per_m"], 0.0)
        for name in ("front", "back", "left", "right"):
            self.assertEqual(self.images[name].size, (1024, 1024))
            self.assertEqual(self.images[name].mode, "RGB")
            info = self.metadata["views"][name]
            self.assertEqual(info["file"], f"{name}.png")
            for key in ("left_eye_px", "right_eye_px"):
                x, y = info[key]
                self.assertGreaterEqual(x, 0)
                self.assertLess(x, 1024)
                self.assertGreaterEqual(y, 0)
                self.assertLess(y, 1024)
            self.assertGreater(info["eye_radius_px"], 0.0)

        self.assertTrue(self.metadata["views"]["front"]["eyes_visible"])
        self.assertFalse(self.metadata["views"]["back"]["eyes_visible"])
        self.assertTrue(self.metadata["views"]["left"]["eyes_visible"])
        self.assertTrue(self.metadata["views"]["right"]["eyes_visible"])

        self.assertFalse(self.metadata["views"]["front"]["arms_hidden"])
        self.assertFalse(self.metadata["views"]["back"]["arms_hidden"])
        for name in ("left", "right"):
            info = self.metadata["views"][name]
            self.assertTrue(info["arms_hidden"])
            self.assertEqual(len(info["arm_sections"]), 2)
            self.assertEqual(
                [section["style"] for section in info["arm_sections"]],
                ["solid", "dashed"],
            )
            for section in info["arm_sections"]:
                x, y = section["center_px"]
                self.assertGreaterEqual(x, 0)
                self.assertLess(x, 1024)
                self.assertGreaterEqual(y, 0)
                self.assertLess(y, 1024)
                self.assertGreater(section["radius_px"], 0.0)
        self.assertEqual(self.metadata["views"]["front"]["arm_sections"], [])
        self.assertEqual(self.metadata["views"]["back"]["arm_sections"], [])

    def test_side_views_hide_the_arm_silhouette(self):
        front = np.asarray(self.images["front"])
        side = np.asarray(self.images["left"])
        front_mask = np.any(front < 250, axis=2)
        side_mask = np.any(side < 250, axis=2)
        front_width = np.ptp(np.where(front_mask)[1])
        side_width = np.ptp(np.where(side_mask)[1])
        # The T-pose front view includes the outstretched arms; the side
        # silhouette must not carry that mass (nor the hand/finger bump).
        self.assertGreater(front_width - side_width, 300)

    def test_silhouette_has_margin_and_is_not_empty(self):
        pixels = np.asarray(self.images["front"])
        mask = np.any(pixels < 250, axis=2)
        ys, xs = np.where(mask)
        self.assertGreater(int(mask.sum()), 10000)
        self.assertGreaterEqual(int(xs.min()), 40)
        self.assertGreaterEqual(int(ys.min()), 40)
        self.assertLessEqual(int(xs.max()), 984)
        self.assertLessEqual(int(ys.max()), 984)

    def test_pose_is_not_baked_into_fourview(self):
        baseline = np.asarray(self.images["front"]).copy()
        import torch

        torch.manual_seed(11)
        self.cache.poses[GENDER] = torch.randn(1, 63) * 0.15
        self.cache.global_orients[GENDER] = torch.randn(1, 3) * 0.08
        self.cache._invalidate_geometry(GENDER)

        target = self.cache.fourview(GENDER)
        with zipfile.ZipFile(target) as archive:
            image = Image.open(io.BytesIO(archive.read("front.png"))).convert("RGB")
            metadata = json.loads(archive.read("views.json"))
        np.testing.assert_array_equal(np.asarray(image), baseline)
        self.assertTrue(metadata["tpose"])

    def test_identity_edits_are_reflected_in_metadata(self):
        self.cache.update_head(GENDER, {"head_length": 1.1, "jaw": 1.05})
        target = self.cache.fourview(GENDER)
        with zipfile.ZipFile(target) as archive:
            metadata = json.loads(archive.read("views.json"))
        self.assertAlmostEqual(metadata["head"]["head_length"], 1.1)
        self.assertAlmostEqual(metadata["head"]["jaw"], 1.05)


if __name__ == "__main__":
    unittest.main()
