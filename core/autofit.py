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

import base64
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
PREVIEW_MAX_SIDE = 1600

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
            return images, f"目录 {path}", len(images) == 1
        if source.is_file():
            payload = source.read_bytes()
            kind = _archive_kind(source.name, payload)
            if kind:
                with tempfile.TemporaryDirectory(prefix="autofit_arc_") as tmp:
                    _extract_archive(payload, kind, Path(tmp))
                    images = _images_in_directory(Path(tmp))
                if not images:
                    raise ValueError(f"压缩包里没有图片: {path}")
                return images, f"压缩包 {path}", len(images) == 1
            return [(source.name, _load_rgb(payload))], f"图片 {path}", True
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
        return images, f"压缩包 {filename or '(upload)'}", len(images) == 1
    return [((filename or "upload.png"), _load_rgb(data))], f"图片 {filename or '(upload)'}", True


def _foreground(rgb: np.ndarray) -> np.ndarray:
    # The supplied references have a white background.  A low threshold keeps
    # light hair/clothes while ignoring the paper.
    return (np.min(rgb, axis=2) < 245).astype(np.uint8)


def _projection_split(foreground: np.ndarray, box):
    """Last-resort split for two touching figures."""
    x, y, w, h = box
    crop = foreground[y:y + h, x:x + w]
    if w <= 0 or h <= 0:
        return None
    for axis in (0, 1):
        length = crop.shape[axis]
        other = crop.shape[1 - axis]
        if axis == 0 and w < h * 1.25:
            continue
        if axis == 1 and h < w * 1.25:
            continue
        occupied = crop.sum(axis=axis) > max(3, int(length * 0.003))
        indices = np.where(occupied)[0]
        if not len(indices):
            continue
        gaps = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right - left > 1:
                gaps.append((int(right - left), int(left), int(right)))
        if not gaps:
            continue
        gap, left, right = max(gaps)
        if gap < max(8, int(length * 0.05)):
            continue
        if axis == 0:
            first_w, second_w = left, w - right
            if min(first_w, second_w) < max(8, w * 0.1):
                continue
            return [(x, y, first_w, h), (x + right, y, w - right, h)]
        first_h, second_h = left, h - right
        if min(first_h, second_h) < max(8, h * 0.1):
            continue
        return [(x, y, w, first_h), (x, y + right, w, h - right)]
    return None


def _figure_bboxes(rgb: np.ndarray):
    """Return one complete-person box, or two for a combined image.

    The old version simply took the two largest connected components; a hat,
    hand or hair clump could win that contest.  This version requires full-
    height components, similar heights, and only then accepts a pair.
    """
    foreground = _foreground(rgb)
    if foreground.sum() < 100:
        raise ValueError("图片前景为空")
    ys, xs = np.where(foreground)
    fg_height = int(ys.max() - ys.min() + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, 8)
    components = []
    total = int(foreground.sum())
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < max(1000, int(total * 0.01)):
            continue
        components.append((int(area), int(x), int(y), int(w), int(h)))
    components.sort(reverse=True)

    # Two stacked figures are each ~0.46 of the combined image height; a
    # single figure is ~0.9.  Use the smaller threshold so vertical layouts
    # are not rejected.
    full = [item for item in components if item[4] >= rgb.shape[0] * 0.30]
    if len(full) >= 2:
        first, second = full[0], full[1]
        height_ratio = second[4] / max(first[4], 1)
        if (
            second[0] >= first[0] * 0.15
            and 0.6 <= height_ratio <= 1.6
            and second[2] + second[4] <= rgb.shape[0] + 2
        ):
            return [(x, y, w, h) for _, x, y, w, h in (first, second)]
        return [(full[0][1], full[0][2], full[0][3], full[0][4])]

    if len(full) == 1:
        box = (full[0][1], full[0][2], full[0][3], full[0][4])
        split = _projection_split(foreground, box)
        return split or [box]

    if components:
        _, x, y, w, h = components[0]
        return [(int(x), int(y), int(w), int(h))]
    return [(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))]


def _normalize_figure(cache, rgb: np.ndarray):
    """Re-crop to the detector's person bbox, keeping preview metadata local."""
    apd = cache._detector_module()
    mask = apd.detect_person_mask(rgb) > 0
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    crop = _crop(rgb, (x, y, w, h), padding=16)
    x0 = max(0, x - 16)
    y0 = max(0, y - 16)
    local_bbox = (x - x0, y - y0, w, h)
    mask_crop = mask[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]]
    return crop, local_bbox, float(w) / max(h, 1), mask_crop

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


def _split_candidates(cache, images, allow_split: bool):
    candidates = []
    for name, rgb in images:
        if allow_split:
            try:
                boxes = _figure_bboxes(rgb)
            except Exception:
                boxes = [(0, 0, rgb.shape[1], rgb.shape[0])]
        else:
            boxes = [(0, 0, rgb.shape[1], rgb.shape[0])]
        for index, box in enumerate(boxes):
            rough = _crop(rgb, box)
            normalized = _normalize_figure(cache, rough)
            if normalized is None:
                continue
            crop, bbox, aspect, mask = normalized
            candidates.append(
                {
                    "name": name if len(boxes) == 1 else f"{name}#{index + 1}",
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


def _fill_enclosed(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(padded, holes)[1:-1, 1:-1]


def _silhouette_mask(rgb: np.ndarray, person_mask=None) -> np.ndarray:
    """Return the filled character outline, preserving white interior pixels."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    silhouette = np.zeros((height, width), np.uint8)
    minimum_area = max(64.0, height * width * 0.0005)
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        cv2.drawContours(silhouette, [contour], -1, 255, -1)

    # Canny can miss faint white-on-white clothing.  The detector's person mask
    # is a useful backup, but fill its holes too so white details stay opaque.
    if (
        person_mask is not None
        and getattr(person_mask, "shape", None) == (height, width)
        and person_mask.any()
    ):
        silhouette = cv2.bitwise_or(silhouette, (person_mask > 0).astype(np.uint8) * 255)
    if not silhouette.any():
        silhouette = _foreground(rgb) * 255
    return _fill_enclosed(silhouette)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _preview_data(rgb: np.ndarray, person_mask=None):
    height, width = rgb.shape[:2]
    scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height, 1))
    # Keep the complete outer silhouette, not just non-white pixels: white
    # clothing, pom-poms and gaps between hair strands must stay opaque.
    alpha = _silhouette_mask(rgb, person_mask)
    image = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    if scale < 1.0:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        scale,
        [int(image.width), int(image.height)],
    )


def _display_payload(rgb, bbox, person_height_px, eye_px, model_height, person_mask=None):
    preview, scale, size = _preview_data(rgb, person_mask)
    box = [float(value) * scale for value in bbox]
    payload = {
        "preview": preview,
        "image_size": size,
        "figure_bbox": box,
        "person_height_px": float(person_height_px) * scale,
        "ppm": (float(person_height_px) * scale) / max(model_height, 1e-6),
    }
    if eye_px is not None:
        payload["eye_px"] = [float(value) * scale for value in eye_px]
    return payload


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
    eye_px = [float(reference["eye_midpoint"][0]), float(reference["eye_midpoint"][1])]
    joints = dict(reference["joints"])
    eye_frac = (eye_px[1] - bbox[1]) / max(bbox[3], 1.0)
    eye_source = "detector"
    if not 0.08 <= eye_frac <= 0.22:
        # The anime face cascade can lock onto a beret/pom-pom on a tight
        # combined-image crop.  Fall back to the stable T-pose eye height
        # fraction and drop the bogus eye joints from the length fit.
        eye_px = [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] * 0.13]
        joints = {key: value for key, value in joints.items() if key not in ("23", "24")}
        eye_source = "bbox_fraction"
    fit = cache.solve_reference_lengths(
        gender,
        ppm,
        eye_px,
        joints,
        z_weight=3.0,
    )
    _model, vertices, _joints, _parents = cache.rest_geometry(gender)
    height_before = float(vertices[:, 1].max() - vertices[:, 1].min())
    global_scale = float(np.clip(model_height / max(height_before, 1e-6), 0.85, 1.15))
    if abs(global_scale - 1.0) > 0.005:
        _scale_lengths(cache, gender, global_scale)
    rms_px, max_px = cache._reference_residual(
        gender, ppm, eye_px, joints
    )
    _model, vertices, _joints, _parents = cache.rest_geometry(gender)
    height_after = float(vertices[:, 1].max() - vertices[:, 1].min())
    return {
        "ppm": ppm,
        "bbox": list(map(int, bbox)),
        "eye_px": eye_px,
        "eye_source": eye_source,
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
    images, source, allow_split = _collect_images(data, filename, path)
    candidates = _split_candidates(cache, images, allow_split)
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

    model_height_m = float(front_fit["height_after_m"])
    try:
        front_fit["display"] = _display_payload(
            front["rgb"], front_fit["bbox"], front_fit["bbox"][3],
            front_fit.get("eye_px"), model_height_m, front["mask"],
        )
        front_fit["preview"] = front_fit["display"]["preview"]
        if side_fit is not None:
            side_fit["display"] = _display_payload(
                side["rgb"], side_fit["bbox"], side_fit["bbox"][3],
                None, model_height_m, side["mask"],
            )
            side_fit["preview"] = side_fit["display"]["preview"]
    except Exception as exc:
        warnings.append(f"预览生成失败：{type(exc).__name__}: {exc}")

    return {
        "ok": True,
        "source": source,
        "num_input_images": len(images),
        "num_figures": len(candidates),
        "model_height_m": model_height_m,
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
