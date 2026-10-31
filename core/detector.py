"""AnimePoseDetector adapter used by the reference-image upload."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

def detector_module(cache):
    """Import AnimePoseDetector lazily (it needs OpenCV, never torch)."""
    if cache._detector is None:
        if not cache.detector_dir or not Path(cache.detector_dir).is_dir():
            raise RuntimeError(f"AnimePoseDetector 目录不可用: {cache.detector_dir!r}")
        if cache.detector_dir not in sys.path:
            sys.path.insert(0, cache.detector_dir)
        import anime_pose_detector as apd
        cache._detector = apd
    return cache._detector


def reference_eyes(cache, image_bytes: bytes) -> dict:
    """Run AnimePoseDetector on an upload and return the eye anchor.

    The detector always writes ``<stem>_result.json`` beside its input, so
    the upload is decoded inside a throw-away directory that is removed on
    the way out: the user's picture never reaches the filesystem.
    """
    apd = cache._detector_module()
    with tempfile.TemporaryDirectory(prefix="smplx_ref_") as tmp:
        source = Path(tmp) / "upload.png"
        source.write_bytes(image_bytes)
        result = apd.estimate(
            str(source),
            smplx_dir=str(cache.model_dir),
            output_dir=tmp,
            write_vis=False,
        )
    eyes = result.get("eyes") or {}
    midpoint = result.get("eye_midpoint")
    left = (eyes.get("left") or {}).get("center")
    right = (eyes.get("right") or {}).get("center")
    if not midpoint or not left or not right:
        raise ValueError("未检测到双眼")
    smplx_block = result.get("smplx") or {}
    fit = smplx_block.get("fit_core_torso") or smplx_block.get("image_to_model_2d") or {}
    scale = float(fit.get("scale") or 0.0)
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("检测到的像素/米比例无效")
    joints = {}
    for entry in (result.get("joints") or {}).values():
        index = entry.get("smplx_index")
        xy = entry.get("xy")
        if index is None or xy is None or not (0 <= int(index) <= 24):
            continue
        joints[str(int(index))] = [float(xy[0]), float(xy[1])]
    return {
        "ok": True,
        "eye_midpoint": [float(midpoint[0]), float(midpoint[1])],
        "eye_left": [float(left[0]), float(left[1])],
        "eye_right": [float(right[0]), float(right[1])],
        "pixels_per_meter": scale,
        "image_width": int(result["image_width"]),
        "image_height": int(result["image_height"]),
        "joints": joints,
        "elapsed_ms": result.get("elapsed_ms"),
    }
