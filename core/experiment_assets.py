"""Experiment-mode reference shortcut.

Only used by the server when experiment mode is active.  The module keeps the
asset discovery and image comparison out of the normal auto-fit path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core import autofit

ROOT = Path(__file__).resolve().parent.parent
PICS_DIR = ROOT / "pics"
RESOURCES_DIR = ROOT / "resources"
MAX_ENTRIES = 5
MATCH_THRESHOLD = 0.92
MATCH_MARGIN = 0.03
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Descriptor:
    gray: np.ndarray
    bits: np.ndarray
    histogram: np.ndarray
    bbox: tuple[int, int, int, int]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class Entry:
    image_path: Path
    model_path: Path
    registration: dict
    gender: str
    view: str
    descriptor: Descriptor


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _person_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    height, width = rgb.shape[:2]
    foreground = autofit._foreground(rgb)
    if int(foreground.sum()) < 100:
        return 0, 0, width, height
    ys, xs = np.where(foreground)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _descriptor(rgb: np.ndarray) -> Descriptor:
    x0, y0, x1, y1 = _person_bbox(rgb)
    crop = rgb[y0:y1, x0:x1]
    if not crop.size:
        crop = rgb
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    normalized = (gray - float(gray.mean())) / (float(gray.std()) + 1e-6)

    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)
    low = dct[:8, :8].reshape(-1)[1:]
    bits = low > float(np.median(low))

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)

    return Descriptor(
        gray=normalized,
        bits=bits,
        histogram=histogram,
        bbox=(x0, y0, x1, y1),
        image_size=(int(rgb.shape[1]), int(rgb.shape[0])),
    )


def _score(a: Descriptor, b: Descriptor) -> float:
    zncc = float(np.mean(a.gray * b.gray))
    hamming = int(np.count_nonzero(a.bits != b.bits))
    phash = 1.0 - hamming / max(len(a.bits), 1)
    hist = float(cv2.compareHist(a.histogram, b.histogram, cv2.HISTCMP_CORREL))
    return 0.55 * zncc + 0.30 * phash + 0.15 * hist


def _discover(root: Path | None = None) -> tuple[Entry, ...]:
    root = root or ROOT
    pics = root / "pics"
    resources = root / "resources"
    if not pics.is_dir():
        return ()
    entries: list[Entry] = []
    for image_path in sorted(pics.iterdir()):
        if not image_path.is_file() or image_path.name.startswith("."):
            continue
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resource = resources / f"smplx_fit_{image_path.stem}"
        model_path = resource / "fitted_smplx.npz"
        registration_path = resource / "registration.json"
        parameters_path = resource / "smplx_parameters.json"
        if not model_path.is_file() or not registration_path.is_file():
            continue
        try:
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            parameters = json.loads(parameters_path.read_text(encoding="utf-8")) if parameters_path.is_file() else {}
            if not isinstance(registration.get("front"), dict):
                continue
            rgb = _read_rgb(image_path)
            descriptor = _descriptor(rgb)
        except Exception:
            continue
        entries.append(
            Entry(
                image_path=image_path,
                model_path=model_path,
                registration=registration,
                gender=str(parameters.get("gender") or "female").lower(),
                view="front",
                descriptor=descriptor,
            )
        )
        if len(entries) >= MAX_ENTRIES:
            break
    return tuple(entries)


@lru_cache(maxsize=1)
def entries() -> tuple[Entry, ...]:
    return _discover()


def match_image(rgb: np.ndarray, candidates: tuple[Entry, ...] | None = None) -> Entry | None:
    candidates = entries() if candidates is None else candidates
    if not candidates:
        return None
    descriptor = _descriptor(rgb)
    scored = sorted(
        ((_score(descriptor, item.descriptor), item) for item in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best_score, best = scored[0]
    if best_score < MATCH_THRESHOLD:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < MATCH_MARGIN:
        return None
    return best


def _display(entry: Entry, rgb: np.ndarray) -> dict:
    preview, _scale, size = autofit._preview_data(rgb)
    front = entry.registration["front"]
    matrix = np.asarray(front["pixel_to_world"], dtype=np.float64)
    uploaded = _person_bbox(rgb)
    catalog = entry.descriptor.bbox
    registration_width = float(front.get("width") or entry.descriptor.image_size[0])
    registration_height = float(front.get("height") or entry.descriptor.image_size[1])
    catalog_width, catalog_height = entry.descriptor.image_size
    ref_x0 = catalog[0] * registration_width / max(catalog_width, 1)
    ref_y0 = catalog[1] * registration_height / max(catalog_height, 1)
    ref_x1 = catalog[2] * registration_width / max(catalog_width, 1)
    ref_y1 = catalog[3] * registration_height / max(catalog_height, 1)
    up_x0, up_y0, up_x1, up_y1 = uploaded
    sx = (ref_x1 - ref_x0) / max(up_x1 - up_x0, 1)
    sy = (ref_y1 - ref_y0) / max(up_y1 - up_y0, 1)

    height, width = rgb.shape[:2]
    quad = []
    for u, v in ((0, 0), (width, 0), (0, height), (width, height)):
        rx = ref_x0 + (u - up_x0) * sx
        ry = ref_y0 + (v - up_y0) * sy
        world = matrix @ np.array([rx, ry, 0.0, 1.0], dtype=np.float64)
        quad.append([float(world[0]), float(world[1]), float(world[2])])
    return {
        "preview": preview,
        "image_size": size,
        "quad": quad,
        "view": "front",
    }


def try_apply(cache, gender: str, *, data: bytes | None = None,
              filename: str = "", path: str | None = None) -> dict | None:
    """Return a loaded-model response when the input is a single known image."""
    try:
        images, _source, _allow_split = autofit._collect_images(data, filename, path)
    except Exception:
        return None
    if len(images) != 1:
        return None
    entry = match_image(images[0][1])
    if entry is None or entry.gender != gender:
        return None
    display = _display(entry, images[0][1])
    cache.replace_custom_model(gender, entry.model_path.read_bytes(), record=True)
    return {
        "ok": True,
        "source": "参考图",
        "num_input_images": 1,
        "num_figures": 1,
        "front": {"display": display},
        "side": None,
        "applied_girth": {},
        "warnings": [],
    }
