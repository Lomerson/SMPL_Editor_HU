"""Auto-fit v0: T-pose front + side reference -> identity edits.

The input side is deliberately forgiving:

* one image containing both figures (any left/right, top/bottom or diagonal
  layout, as long as they do not overlap);
* a directory of images (filenames are *not* used to decide front/side);
* a zip / tar.gz / tgz archive with nested folders;
* a single front image (no side extrapolation).

The fit is intentionally coarse.  The front view supplies 2D joints and the
pixel/height calibration; the side view supplies torso depth ratios.  It is a
starting point for the manual editor, not a differentiable silhouette fit.
"""
from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import GENDERS

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
}
# The T-pose front is wide because of the outstretched arms; a side profile is
# narrow.  The gap is large (front ~0.83, side ~0.22 in the supplied assets).
FRONT_ASPECT_MIN = 0.52

# Conservative torso levels (fraction from the top of the figure), chosen below
# the T-pose arms so the side hand does not pollute the chest depth.
TORSO_LEVELS = (0.30, 0.38, 0.46)


def _load_rgb(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _is_image_name(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(":zone.identifier") or "__macosx" in lower:
        return False
    return Path(lower).suffix in IMAGE_SUFFIXES


def _archive_kind(name: str, data: bytes | None = None) -> str | None:
    lower = name.lower()
    if lower.endswith(".zip") or (data is not None and data[:4] == b"PK\x03\x04"):
        return "zip"
    if lower.endswith((".tar", ".tgz", ".tar.gz")):
        return "tar"
    if data is not None and data[:2] == b"\x1f\x8b":
        return "tar"
    return None


def _extract_archive(data: bytes, kind: str, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(target)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            archive.extractall(target)


def _images_in_directory(directory: Path):
    results = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and _is_image_name(path.name):
            try:
                results.append((path.name, _load_rgb(path.read_bytes())))
            except Exception:
                continue
    return results


def _collect_images(data: bytes | None, filename: str, path: str | None):
    if path:
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(path)
        if source.is_dir():
            images = _images_in_directory(source)
            if not images:
                raise ValueError(f"目录里没有图片: {path}")
            return images, f"目录 {path}"
        if source.is_file():
            payload = source.read_bytes()
            kind = _archive_kind(source.name, payload)
            if kind:
                with tempfile.TemporaryDirectory(prefix="autofit_arc_") as tmp:
                    _extract_archive(payload, kind, Path(tmp))
                    images = _images_in_directory(Path(tmp))
                if not images:
                    raise ValueError(f"压缩包里没有图片: {path}")
                return images, f"压缩包 {path}"
            return [(source.name, _load_rgb(payload))], f"图片 {path}"
        raise ValueError(f"不支持的输入路径: {path}")

    if data is None:
        raise ValueError("没有输入图片或路径")
    kind = _archive_kind(filename, data)
    if kind:
        with tempfile.TemporaryDirectory(prefix="autofit_arc_") as tmp:
            _extract_archive(data, kind, Path(tmp))
            images = _images_in_directory(Path(tmp))
        if not images:
            raise ValueError("压缩包里没有图片")
        return images, f"压缩包 {filename or '(upload)'}"
    return [((filename or "upload.png"), _load_rgb(data))], f"图片 {filename or '(upload)'}"


def _foreground(rgb: np.ndarray) -> np.ndarray:
    # The supplied references have a white background.  A low threshold keeps
    # light hair/clothes while ignoring the paper.
    return (np.min(rgb, axis=2) < 245).astype(np.uint8)


def _figure_bboxes(rgb: np.ndarray):
    foreground = _foreground(rgb)
    if foreground.sum() < 100:
        raise ValueError("图片前景为空")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, 8)
    components = []
    total = int(foreground.sum())
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < max(1000, int(total * 0.01)):
            continue
        components.append((int(area), int(x), int(y), int(w), int(h)))
    components.sort(reverse=True)
    if len(components) >= 2:
        first, second = components[0], components[1]
        if second[0] >= first[0] * 0.15 and second[4] >= rgb.shape[0] * 0.25:
            return [(x, y, w, h) for _, x, y, w, h in (first, second)]
    if not components:
        ys, xs = np.where(foreground)
        return [(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))]

    # Fallback for touching figures: split at the largest empty projection gap
    # on the axis that best separates them.
    best = None
    for axis in (0, 1):
        occupied = (foreground.sum(axis=axis) > max(3, int(foreground.shape[axis] * 0.003)))
        indices = np.where(occupied)[0]
        if not len(indices):
            continue
        gaps = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right - left > 1:
                gaps.append((right - left, int(left), int(right)))
        if not gaps:
            continue
        gap, left, right = max(gaps)
        if gap > (foreground.shape[axis] * 0.06) and left > 0 and right < foreground.shape[axis] - 1:
            best = (axis, left, right)
            break
    if best is None:
        x, y, w, h = components[0][1:]
        return [(x, y, w, h)]
    axis, left, right = best
    x, y, w, h = components[0][1:]
    if axis == 0:  # columns
        return [(x, y, left - x, h), (right, y, x + w - right, h)]
    return [(x, y, w, left - y), (x, right, w, y + h - right)]


def _crop(rgb: np.ndarray, box, padding: int = 16):
    x, y, w, h = box
    height, width = rgb.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    return rgb[y0:y1, x0:x1]


def _bbox_and_aspect(rgb: np.ndarray, apd):
    mask = apd.detect_person_mask(rgb) > 0
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("图中没有人物")
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return (int(xs.min()), int(ys.min()), width, height), width / max(height, 1), mask


def _split_candidates(cache, images):
    apd = cache._detector_module()
    candidates = []
    for name, rgb in images:
        try:
            boxes = _figure_bboxes(rgb)
        except Exception:
            boxes = [(0, 0, rgb.shape[1], rgb.shape[0])]
        for box in boxes:
            crop = _crop(rgb, box)
            try:
                bbox, aspect, mask = _bbox_and_aspect(crop, apd)
            except Exception:
                continue
            candidates.append(
                {
                    "name": name,
                    "rgb": crop,
                    "bbox": bbox,
                    "aspect": float(aspect),
                    "area": int(bbox[2] * bbox[3]),
                    "mask": mask,
                }
            )
    if not candidates:
        raise ValueError("没有检测到可用人物")
    return candidates


def _pick_views(candidates):
    fronts = [item for item in candidates if item["aspect"] >= FRONT_ASPECT_MIN]
    sides = [item for item in candidates if item["aspect"] < FRONT_ASPECT_MIN]
    if not fronts:
        raise ValueError("没有检测到正视图（需要 T-pose 正视图）")
    front = max(fronts, key=lambda item: item["area"])
    side = max(sides, key=lambda item: item["area"]) if sides else None
    return front, side


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _row_width(mask: np.ndarray, y: int) -> int:
    row = np.where(mask[y])[0]
    return int(row.max() - row.min() + 1) if len(row) else 0


def _model_band(vertices, y: float, half_height: float):
    band = np.abs(vertices[:, 1] - y) < half_height
    if not band.any():
        return None
    return vertices[band]


def _scale_lengths(cache, gender: str, scale: float):
    with cache.lock:
        config = cache._segments(gender)
        for child in range(1, 22):
            entry = config.get(str(child))
            if entry is None:
                continue
            value = float(entry.get("length", 1.0)) * scale
            entry["length"] = float(np.clip(value, 0.25, 4.0))
        cache._invalidate_geometry(gender)


def _fit_front(cache, gender: str, front, model_height: float):
    reference = cache.reference_eyes(_png_bytes(front["rgb"]))
    bbox = front["bbox"]
    ppm = float(bbox[3]) / max(model_height, 1e-6)
    fit = cache.solve_reference_lengths(
        gender,
        ppm,
        reference["eye_midpoint"],
        reference["joints"],
        z_weight=3.0,
    )
    _model, vertices, _joints, _parents = cache.rest_geometry(gender)
    height_before = float(vertices[:, 1].max() - vertices[:, 1].min())
    global_scale = float(np.clip(model_height / max(height_before, 1e-6), 0.85, 1.15))
    if abs(global_scale - 1.0) > 0.005:
        _scale_lengths(cache, gender, global_scale)
    rms_px, max_px = cache._reference_residual(
        gender, ppm, reference["eye_midpoint"], reference["joints"]
    )
    _model, vertices, _joints, _parents = cache.rest_geometry(gender)
    height_after = float(vertices[:, 1].max() - vertices[:, 1].min())
    return {
        "ppm": ppm,
        "bbox": list(map(int, bbox)),
        "num_joints": int(fit["num_joints"]),
        "rms_px": float(rms_px),
        "max_px": float(max_px),
        "lengths": {
            str(child): float(cache._segments(gender).get(str(child), {}).get("length", 1.0))
            for child in range(1, 22)
            if str(child) in cache._segments(gender)
        },
        "global_scale": global_scale,
        "height_before_m": height_before,
        "height_after_m": height_after,
        "detector_ppm": float(reference["pixels_per_meter"]),
        "image_size": [int(front["rgb"].shape[1]), int(front["rgb"].shape[0])],
    }


def _fit_side(cache, gender: str, side, model_height: float, front_height_px: int):
    bbox = side["bbox"]
    ppm = float(bbox[3]) / max(model_height, 1e-6)
    mask = side["mask"]
    y0 = bbox[1]
    height = bbox[3]
    _model, vertices, _joints, _parents = cache.rest_geometry(gender)
    y_top = float(vertices[:, 1].max())
    y_bottom = float(vertices[:, 1].min())
    model_span = max(y_top - y_bottom, 1e-6)

    measurements = []
    ratios = []
    for fraction in TORSO_LEVELS:
        image_y = int(np.clip(y0 + fraction * height, 0, mask.shape[0] - 1))
        width_px = _row_width(mask, image_y)
        if width_px <= 0:
            continue
        target_depth = width_px / ppm
        model_y = y_top - fraction * model_span
        band = _model_band(vertices, model_y, model_span * 0.012)
        if band is None:
            continue
        model_depth = float(band[:, 2].max() - band[:, 2].min())
        if model_depth < 1e-5:
            continue
        ratio = target_depth / model_depth
        measurements.append(
            {
                "fraction": float(fraction),
                "target_depth_m": float(target_depth),
                "model_depth_m": float(model_depth),
                "ratio": float(ratio),
            }
        )
        ratios.append(ratio)
    ratio = float(np.median(ratios)) if ratios else 1.0
    # Clothing makes the side silhouette wider than the body; only use it as
    # a gentle torso scale, never as an aggressive shrink.
    torso_girth = float(np.clip(0.5 + 0.5 * ratio, 0.88, 1.10))
    return {
        "ppm": ppm,
        "bbox": list(map(int, bbox)),
        "image_size": [int(side["rgb"].shape[1]), int(side["rgb"].shape[0])],
        "depth_ratio": ratio,
        "torso_girth": torso_girth,
        "measurements": measurements,
        "height_ratio_to_front": float(bbox[3]) / max(front_height_px, 1),
    }


def _apply_torso_girth(cache, gender: str, girth: float):
    if abs(girth - 1.0) < 0.02:
        return {}
    applied = {}
    config = cache._segments(gender)
    for child in (3, 6, 9):
        entry = config.get(str(child), {})
        length = float(entry.get("length", 1.0))
        cache.update_segment(gender, child, length, [girth, girth], record=False)
        applied[str(child)] = float(girth)
    return applied


def run(cache, gender: str, *, data: bytes | None = None,
        filename: str = "", path: str | None = None) -> dict:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    images, source = _collect_images(data, filename, path)
    candidates = _split_candidates(cache, images)
    front, side = _pick_views(candidates)

    # Auto-fit is one undoable operation even though it writes reset + lengths +
    # girth internally.
    history_base = cache.begin_edit(gender, "Auto-fit v0")
    try:
        # Start from the canonical rest identity; auto-fit replaces the previous
        # coarse edit rather than compounding it.
        cache.reset(gender, record=False)
        model, base_vertices, _joints, _parents = cache.rest_geometry(gender)
        model_height = float(base_vertices[:, 1].max() - base_vertices[:, 1].min())

        front_fit = _fit_front(cache, gender, front, model_height)
        side_fit = _fit_side(cache, gender, side, model_height, front_fit["bbox"][3]) if side else None
        girth = _apply_torso_girth(cache, gender, side_fit["torso_girth"]) if side_fit else {}
        cache._invalidate_geometry(gender)
    except Exception:
        cache.rollback_edit(gender, history_base)
        raise
    cache.commit_edit(gender, "Auto-fit v0", history_base)

    warnings = []
    if side is None:
        warnings.append("只检测到正视图，侧视厚度使用默认值")
    elif side_fit and abs(side_fit["height_ratio_to_front"] - 1.0) > 0.02:
        warnings.append(
            "正/侧视图人物像素高度相差 "
            f"{(side_fit['height_ratio_to_front'] - 1.0) * 100:.1f}%，已分别标定"
        )
    if side_fit and side_fit["measurements"]:
        worst = max(side_fit["measurements"], key=lambda item: abs(item["ratio"] - 1.0))
        if abs(worst["ratio"] - 1.0) > 0.25:
            warnings.append("侧视厚度与模型差异较大，已做保守缩放")

    return {
        "ok": True,
        "source": source,
        "num_input_images": len(images),
        "num_figures": len(candidates),
        "front": {key: value for key, value in front_fit.items() if key != "lengths"} | {
            "figure": {
                "name": front["name"],
                "bbox": front_fit["bbox"],
                "aspect": front["aspect"],
            },
            "lengths": front_fit["lengths"],
        },
        "side": None if side_fit is None else side_fit,
        "applied_girth": girth,
        "warnings": warnings,
    }
