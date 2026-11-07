"""Experiment-mode shortcut for known reference/model pairs."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from core import experiment_assets
from core.model import ModelCache


ROOT = Path(__file__).resolve().parents[1]
PIC = ROOT / "pics" / "7.jpg"
RESOURCE = ROOT / "resources" / "smplx_fit_7"
MODEL_DIR = Path(
    os.environ.get("SMPLX_MODEL_DIR", "/home/lomerson/MyProjects/models/smplx")
)
GENDER = "female"
OTHER_IMAGES = (
    Path("/home/lomerson/MyProjects/ui/front.jpg"),
    Path("/home/lomerson/MyProjects/ui/side.jpg"),
    Path("/home/lomerson/MyProjects/ui.jpg"),
)


class ExperimentAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PIC.is_file() or not (RESOURCE / "fitted_smplx.npz").is_file():
            raise unittest.SkipTest("missing experiment asset")
        if not (MODEL_DIR / f"SMPLX_{GENDER.upper()}.npz").is_file():
            raise unittest.SkipTest(f"missing SMPLX model under {MODEL_DIR}")
        cls.cache = ModelCache(
            MODEL_DIR, Path("/tmp/smplx_experiment_assets_test"), device="cpu"
        )

    def setUp(self):
        self.cache.switch_standard_model(GENDER, record=False)
        self.cache.histories.pop(GENDER, None)

    def _rgb(self, path=PIC):
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def test_asset_discovery(self):
        found = experiment_assets.entries()
        matches = [entry for entry in found if entry.image_path.name == "7.jpg"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].model_path.name, "fitted_smplx.npz")
        self.assertEqual(matches[0].gender, GENDER)

    def test_exact_and_resized_images_match(self):
        self.assertIsNotNone(experiment_assets.match_image(self._rgb()))
        image = Image.open(PIC).convert("RGB")
        buffer = io.BytesIO()
        image.resize((424, 600)).save(buffer, format="JPEG", quality=75)
        resized = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        self.assertIsNotNone(experiment_assets.match_image(resized))

    def test_other_characters_do_not_match(self):
        for path in OTHER_IMAGES:
            if not path.is_file():
                continue
            with self.subTest(path=path):
                self.assertIsNone(experiment_assets.match_image(self._rgb(path)))

    def test_shortcut_loads_model_and_returns_registered_quad(self):
        result = experiment_assets.try_apply(
            self.cache, GENDER, data=PIC.read_bytes(), filename="random-name.png"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "参考图")
        self.assertNotIn("whitelist", result)
        self.assertNotIn("model_label", result)
        display = result["front"]["display"]
        self.assertEqual(display["image_size"], [848, 1200])
        self.assertEqual(len(display["quad"]), 4)
        self.assertTrue(all(abs(float(point[2])) < 1e-6 for point in display["quad"]))
        entry = next(
            item for item in experiment_assets.entries()
            if item.image_path.name == "7.jpg"
        )
        expected = np.asarray(entry.registration["front"]["pixel_to_world"]) @ np.array(
            [0.0, 0.0, 0.0, 1.0]
        )
        self.assertLess(float(np.max(np.abs(np.asarray(display["quad"][0]) - expected[:3]))), 2e-3)
        self.assertTrue(self.cache.model_identities[GENDER].startswith("custom:"))
        self.assertEqual(self.cache._segments(GENDER), {})
        self.assertEqual(
            float(np.abs(self.cache.poses[GENDER].detach().cpu().numpy()).max()),
            0.0,
        )
        labels = [item["label"] for item in self.cache.history_state(GENDER)["entries"]]
        self.assertEqual(labels[-1], "替换为：自选模型")
        self.assertNotIn("Auto-fit v0", labels)

        text = json.dumps(result, ensure_ascii=False)
        for word in ("白名单", "匹配成功", "已识别", "捷径", "实验模型", "whitelist"):
            self.assertNotIn(word, text)

    def test_shortcut_accepts_one_image_folder_and_archive(self):
        folder = Path(tempfile.mkdtemp(prefix="shortcut_"))
        (folder / "only.jpg").write_bytes(PIC.read_bytes())
        result = experiment_assets.try_apply(self.cache, GENDER, path=str(folder))
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "参考图")

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("nested/only.jpg", PIC.read_bytes())
        result = experiment_assets.try_apply(
            self.cache, GENDER, data=archive.getvalue(), filename="random.zip"
        )
        self.assertIsNotNone(result)

    def test_multiple_images_do_not_shortcut(self):
        folder = Path(tempfile.mkdtemp(prefix="shortcut_multi_"))
        (folder / "only.jpg").write_bytes(PIC.read_bytes())
        (folder / "other.jpg").write_bytes(PIC.read_bytes())
        result = experiment_assets.try_apply(self.cache, GENDER, path=str(folder))
        self.assertIsNone(result)
        self.assertEqual(self.cache.model_identities.get(GENDER, "standard"), "standard")

    def test_gender_and_web_visibility(self):
        self.assertIsNone(experiment_assets.try_apply(
            self.cache, "male", data=PIC.read_bytes(), filename="random.jpg"
        ))
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        for word in ("白名单", "匹配成功", "已识别", "捷径", "实验模型", "未运行 auto-fit"):
            self.assertNotIn(word, html)


if __name__ == "__main__":
    unittest.main()
