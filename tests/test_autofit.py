"""Workflow 1 auto-fit v0 input handling and coarse fit checks."""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
import unittest
from pathlib import Path

import numpy as np
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
        for key in ("front", "side"):
            display = result[key].get("display")
            self.assertIsNotNone(display)
            self.assertTrue(display["preview"].startswith("data:image/png;base64,"))
            self.assertEqual(len(display["image_size"]), 2)
            self.assertGreater(display["person_height_px"], 0)
            self.assertGreater(display["ppm"], 0)
            original_size = result[key]["image_size"]
            scale = display["image_size"][0] / max(original_size[0], 1)
            self.assertAlmostEqual(
                display["ppm"], result[key]["ppm"] * scale, delta=0.05
            )
        self.assertIn("eye_px", result["front"]["display"])
        self.assertNotIn("eye_px", result["side"]["display"])
        self.assertGreater(result["model_height_m"], 1.0)

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

    def test_single_image_folder_and_archive(self):
        # A folder/archive with exactly one image is still allowed to be a
        # combined front+side image; a single front image stays front-only.
        combined_dir = self.root / "combined_only"
        combined_dir.mkdir(exist_ok=True)
        shutil.copyfile(COMBINED, combined_dir / "only.png")
        result = autofit.run(self.cache, GENDER, path=str(combined_dir))
        self._assert_two_views(result)

        front_dir = self.root / "front_only"
        front_dir.mkdir(exist_ok=True)
        shutil.copyfile(FRONT, front_dir / "only.png")
        result = autofit.run(self.cache, GENDER, path=str(front_dir))
        self.assertEqual(result["num_figures"], 1)
        self.assertIsNone(result["side"])

        combined_zip = self.root / "combined.zip"
        with zipfile.ZipFile(combined_zip, "w") as archive:
            archive.write(COMBINED, "deep/only.png")
        result = autofit.run(self.cache, GENDER, path=str(combined_zip))
        self._assert_two_views(result)

        front_tgz = self.root / "front.tgz"
        with tarfile.open(front_tgz, "w:gz") as archive:
            archive.add(FRONT, arcname="deep/only.png")
        result = autofit.run(self.cache, GENDER, path=str(front_tgz))
        self.assertEqual(result["num_figures"], 1)
        self.assertIsNone(result["side"])

    def test_combined_matches_directory_experiment(self):
        combined = autofit.run(self.cache, GENDER, path=str(COMBINED))
        directory = autofit.run(self.cache, GENDER, path=str(PROJECT / "ui"))
        cf, df = combined["front"], directory["front"]
        # Same source pixels; split must not shift the eye anchor much.
        self.assertLess(abs(cf["eye_px"][0] - df["eye_px"][0]), 8.0)
        self.assertLess(abs(cf["eye_px"][1] - df["eye_px"][1]), 8.0)
        self.assertLess(abs(cf["ppm"] - df["ppm"]) / max(df["ppm"], 1e-6), 0.02)
        self.assertLess(abs(cf["rms_px"] - df["rms_px"]), 20.0)

    def test_scale_rejection_keeps_better_fit(self):
        result = autofit.run(self.cache, GENDER, path=str(FRONT))
        front = result["front"]
        self.assertIn("scale_rejected", front)
        self.assertIn("rms_before_scale_px", front)
        self.assertIn("rms_after_scale_px", front)
        self.assertLess(front["rms_px"], 35.0)
        if front["scale_rejected"]:
            self.assertEqual(front["global_scale"], 1.0)
            self.assertLessEqual(front["rms_px"], front["rms_before_scale_px"] + 1.0)
            self.assertGreater(front["rms_after_scale_px"], front["rms_before_scale_px"])
        else:
            self.assertLessEqual(front["rms_px"], front["rms_after_scale_px"] + 1.0)

    def test_leg_joint_weights_reduce_leg_errors(self):
        images, _source, _single = autofit._collect_images(
            None, filename="", path=str(PROJECT / "ui")
        )
        candidates = autofit._split_candidates(self.cache, images, True)
        front = max(
            [item for item in candidates if item["aspect"] >= autofit.FRONT_ASPECT_MIN],
            key=lambda item: item["area"],
        )
        reference = self.cache.reference_eyes(autofit._png_bytes(front["rgb"]))
        bbox = front["bbox"]
        model_height = float(
            self.cache.rest_geometry(GENDER)[1][:, 1].max()
            - self.cache.rest_geometry(GENDER)[1][:, 1].min()
        )
        ppm = float(bbox[3]) / max(model_height, 1e-6)

        def leg_rms(joint_weights):
            self.cache.reset(GENDER, record=False)
            self.cache.solve_reference_lengths(
                GENDER, ppm, reference["eye_midpoint"], reference["joints"],
                z_weight=3.0, joint_weights=joint_weights,
            )
            _model, _vertices, joints, _parents = self.cache.rest_geometry(GENDER)
            joints = np.asarray(joints)
            eye = (joints[23] + joints[24]) / 2.0
            errors = []
            for joint in (4, 5, 7, 8, 10, 11):
                x, y = reference["joints"][str(joint)]
                predicted = np.array([
                    reference["eye_midpoint"][0]
                    + (joints[joint][0] - eye[0]) * ppm,
                    reference["eye_midpoint"][1]
                    - (joints[joint][1] - eye[1]) * ppm,
                ])
                errors.append(float(np.hypot(predicted[0] - x, predicted[1] - y)))
            return float(np.mean(errors))

        plain = leg_rms(None)
        weighted = leg_rms({
            0: 1.2, 1: 1.2, 2: 1.2,
            4: 2.0, 5: 2.0,
            7: 2.5, 8: 2.5,
            10: 1.5, 11: 1.5,
        })
        self.assertLess(weighted, plain - 1.0)

    def test_single_front_view(self):
        result = autofit.run(self.cache, GENDER, path=str(FRONT))
        self.assertTrue(result["ok"])
        self.assertEqual(result["num_figures"], 1)
        self.assertIsNone(result["side"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["applied_girth"], {})
        self.assertGreater(result["front"]["height_after_m"], 1.5)

    def test_cutout_keeps_white_pixels_inside_the_outline(self):
        rgb = np.full((200, 200, 3), 255, dtype=np.uint8)
        rgb[40:164, 40:44] = 0
        rgb[40:164, 156:160] = 0
        rgb[40:44, 40:160] = 0
        rgb[160:164, 40:160] = 0
        alpha = autofit._silhouette_mask(rgb)
        self.assertEqual(int(alpha[100, 100]), 255)
        self.assertEqual(int(alpha[10, 10]), 0)

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
