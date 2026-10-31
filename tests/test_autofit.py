"""Workflow 1 auto-fit v0 input handling and coarse fit checks."""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core import autofit
from core.model import ModelCache


PROJECT = Path("/home/lomerson/MyProjects")
MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
DETECTOR_DIR = os.environ.get(
    "ANIME_POSE_DETECTOR_DIR", "/home/lomerson/MyProjects/AnimePoseDetector"
)
GENDER = "female"
FRONT = PROJECT / "ui" / "front.jpg"
SIDE = PROJECT / "ui" / "side.jpg"
COMBINED = PROJECT / "ui.jpg"
ARCHIVE = PROJECT / "ShigureUi.zip"


class AutofitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPL-X model under {MODEL_DIR}")
        if not (FRONT.is_file() and SIDE.is_file() and COMBINED.is_file()):
            raise unittest.SkipTest("missing ui reference assets")
        cls.cache = ModelCache(
            MODEL_DIR, Path("/tmp/smplx_autofit_test"), device="cpu",
            detector_dir=DETECTOR_DIR,
        )
        cls.temp = tempfile.TemporaryDirectory(prefix="autofit_test_")
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.cache.reset(GENDER)

    def _compose(self, name, vertical=False, diagonal=False):
        front = Image.open(FRONT).convert("RGB")
        side = Image.open(SIDE).convert("RGB")
        if vertical:
            canvas = Image.new("RGB", (max(front.width, side.width), front.height + side.height + 100), (255, 255, 255))
            canvas.paste(front, (0, 0))
            canvas.paste(side, (0, front.height + 100))
        else:
            canvas = Image.new("RGB", (front.width + side.width + 200, max(front.height, side.height) + 100), (255, 255, 255))
            canvas.paste(side, (0, 0))
            canvas.paste(front, (side.width + 200, 100))
        path = self.root / name
        canvas.save(path)
        return path

    def _assert_two_views(self, result):
        self.assertTrue(result["ok"])
        self.assertEqual(result["num_figures"], 2)
        self.assertIsNotNone(result["side"])
        front_box = result["front"]["figure"]["bbox"]
        side_box = result["side"]["bbox"]
        front_aspect = front_box[2] / max(front_box[3], 1)
        side_aspect = side_box[2] / max(side_box[3], 1)
        self.assertGreater(front_aspect, side_aspect)

    def test_combined_image_layouts(self):
        result = autofit.run(self.cache, GENDER, path=str(COMBINED))
        self._assert_two_views(result)

        for name, vertical in (("vertical.png", True), ("diagonal.png", False)):
            path = self._compose(name, vertical=vertical, diagonal=not vertical)
            result = autofit.run(self.cache, GENDER, path=str(path))
            self._assert_two_views(result)

    def test_directory_zip_and_targz_without_filename_hints(self):
        random_dir = self.root / "random_names"
        random_dir.mkdir(exist_ok=True)
        shutil.copyfile(FRONT, random_dir / "a.png")
        shutil.copyfile(SIDE, random_dir / "b.png")
        result = autofit.run(self.cache, GENDER, path=str(random_dir))
        self._assert_two_views(result)
        self.assertEqual(result["front"]["figure"]["name"], "a.png")

        if ARCHIVE.is_file():
            result = autofit.run(self.cache, GENDER, path=str(ARCHIVE))
            self._assert_two_views(result)

        tar_path = self.root / "nested.tar.gz"
        with tarfile.open(tar_path, "w:gz") as archive:
            archive.add(FRONT, arcname="deep/nested/one.png")
            archive.add(SIDE, arcname="deep/nested/two.png")
        result = autofit.run(self.cache, GENDER, path=str(tar_path))
        self._assert_two_views(result)

    def test_single_front_view(self):
        result = autofit.run(self.cache, GENDER, path=str(FRONT))
        self.assertTrue(result["ok"])
        self.assertEqual(result["num_figures"], 1)
        self.assertIsNone(result["side"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["applied_girth"], {})
        self.assertGreater(result["front"]["height_after_m"], 1.5)

    def test_upload_bytes_and_applied_identity(self):
        result = autofit.run(
            self.cache, GENDER, data=COMBINED.read_bytes(), filename="anything.jpg"
        )
        self._assert_two_views(result)
        segments = self.cache._segments(GENDER)
        self.assertEqual(len(segments), 21)
        self.assertTrue(all(0.25 <= item["length"] <= 4.0 for item in segments.values()))
        self.assertTrue(self.cache._pose(GENDER, self.cache._model(GENDER)).abs().max() == 0)


if __name__ == "__main__":
    unittest.main()
