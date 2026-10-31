"""Four-view T-pose reference export.

The user draws a costume/character over these images, so the output is
deliberately simple: an orthographic silhouette with a dark outline and the
two eye positions.  No shading, texture or perspective is involved.

Views (world axes; SMPL-X is Y-up, +Z forward):

* front: camera +Z, screen right +X
* back:  camera -Z, screen right -X
* left:  camera +X, screen right -Z (character's left side)
* right: camera -X, screen right +Z

All four share one ``scale_px_per_m`` so proportions match.  The result is
``fourview_<gender>.zip`` containing ``front.png``, ``back.png``,
``left.png``, ``right.png`` and ``views.json``.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from core.config import GENDERS

SIZE = 1024
MARGIN = 48
FILL = (205, 205, 205)
OUTLINE = (34, 34, 34)
BACKGROUND = (255, 255, 255)
EYE = (20, 20, 20)
SECTION = (64, 64, 64)

# name, camera direction, screen-right axis, screen-up axis, eyes visible
VIEWS = (
    ("front", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), True),
    ("back", (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), False),
    ("left", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), True),
    ("right", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), True),
)


def _numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=dtype)


def _eye_radius(vertices, weights, joints):
    left = weights[:, 23] > 0.9
    right = weights[:, 24] > 0.9
    radii = []
    if left.any():
        radii.append(float(np.linalg.norm(vertices[left] - joints[23], axis=1).mean()))
    if right.any():
        radii.append(float(np.linalg.norm(vertices[right] - joints[24], axis=1).mean()))
    return float(np.mean(radii)) if radii else 0.0125


def _dashed_ellipse(draw, box, color, width=2, dash=10.0, gap=8.0):
    angle = 0.0
    while angle < 360.0:
        draw.arc(box, angle, min(angle + dash, 360.0), fill=color, width=width)
        angle += dash + gap


def _subtree(parents, root):
    nodes = {int(root)}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(parents):
            if int(parent) in nodes and index not in nodes:
                nodes.add(index)
                changed = True
    return nodes


def _section_radius(vertices, weights, arm_weight, joint):
    slab = np.abs(vertices[:, 0] - joint[0]) < 0.035
    selected = slab & (arm_weight > 0.5)
    if int(selected.sum()) < 4:
        return 0.04
    radial = vertices[selected][:, 1:3] - joint[1:3]
    return float(np.percentile(np.linalg.norm(radial, axis=1), 80.0))


def _project(vertices, joints, view, scale, size, margin):
    _name, _camera, right_axis, up_axis, _eyes = view
    right = np.asarray(right_axis, dtype=np.float32)
    up = np.asarray(up_axis, dtype=np.float32)
    u = vertices @ right
    w = vertices @ up
    u_min, u_max = float(u.min()), float(u.max())
    w_min, w_max = float(w.min()), float(w.max())
    offset_u = (size - (u_max - u_min) * scale) / 2.0 - u_min * scale
    offset_w = (size - (w_max - w_min) * scale) / 2.0 - w_min * scale

    def to_pixels(points):
        points = np.asarray(points, dtype=np.float32)
        x = points @ right * scale + offset_u
        y = size - (points @ up * scale + offset_w)
        return np.stack([x, y], axis=-1)

    return to_pixels(vertices), to_pixels(joints)


def _render_view(vertices, faces, joints, view, scale, arm_weight=None):
    pixels, joint_pixels = _project(vertices, joints, view, scale, SIZE, MARGIN)
    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)
    for triangle in faces:
        # In a side view the arms point at the camera and would otherwise
        # leave a hand/finger bump on the torso silhouette.  Drop triangles
        # that are fully arm; keep the mixed shoulder seam.
        if arm_weight is not None and float(arm_weight[triangle].min()) > 0.5:
            continue
        points = pixels[triangle]
        draw.polygon(
            [(float(point[0]), float(point[1])) for point in points], fill=255
        )
    dilated = mask.filter(ImageFilter.MaxFilter(3))
    edge = ImageChops.subtract(dilated, mask)

    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    image.paste(FILL, mask=mask)
    image.paste(OUTLINE, mask=edge)
    return image, joint_pixels


def build_fourview(cache, gender: str) -> bytes:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    model, vertices, joints, parents = cache.rest_geometry(gender)
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)
    weights = _numpy(model.lbs_weights)
    parents = _numpy(model.parents, dtype=np.int64)
    left_arm = np.zeros(len(vertices), dtype=np.float32)
    right_arm = np.zeros(len(vertices), dtype=np.float32)
    left_arm[:] = weights[:, list(_subtree(parents, 16))].sum(axis=1)
    right_arm[:] = weights[:, list(_subtree(parents, 17))].sum(axis=1)
    both_arms = np.clip(left_arm + right_arm, 0.0, 1.0)

    # One common scale: the largest projected extent over all four views.
    extents = []
    for view in VIEWS:
        _name, _camera, right_axis, up_axis, _eyes = view
        u = vertices @ np.asarray(right_axis, dtype=np.float32)
        w = vertices @ np.asarray(up_axis, dtype=np.float32)
        extents.append(max(float(u.max() - u.min()), float(w.max() - w.min())))
    scale = (SIZE - 2.0 * MARGIN) / max(max(extents), 1e-6)
    eye_radius = _eye_radius(vertices, weights, joints)
    eye_radius_px = max(2.0, eye_radius * scale)

    images = {}
    view_info = {}
    for view in VIEWS:
        name, _camera, right_axis, up_axis, eyes_visible = view
        side_view = name in ("left", "right")
        image, joint_pixels = _render_view(
            vertices, faces, joints, view, scale,
            arm_weight=both_arms if side_view else None,
        )
        sections = []
        if side_view:
            draw = ImageDraw.Draw(image)
            root = 16 if name == "left" else 17
            for joint, label in ((root, "肩部截面"), (root + 2, "手臂截面")):
                radius_m = _section_radius(
                    vertices, weights,
                    left_arm if name == "left" else right_arm,
                    joints[joint],
                )
                radius_px = float(np.clip(radius_m * scale, 4.0, 40.0))
                x, y = joint_pixels[joint]
                box = [
                    x - radius_px,
                    y - radius_px,
                    x + radius_px,
                    y + radius_px,
                ]
                style = "solid" if joint == root else "dashed"
                if style == "solid":
                    draw.ellipse(box, outline=SECTION, width=2)
                else:
                    _dashed_ellipse(draw, box, SECTION, width=2)
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=SECTION)
                sections.append(
                    {
                        "joint": int(joint),
                        "label": label,
                        "style": style,
                        "center_px": [float(x), float(y)],
                        "radius_px": radius_px,
                    }
                )
        if eyes_visible:
            draw = ImageDraw.Draw(image)
            for joint in (23, 24):
                x, y = joint_pixels[joint]
                radius = eye_radius_px
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline=EYE,
                    width=2,
                )
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=EYE)
        images[name] = image
        view_info[name] = {
            "file": f"{name}.png",
            "orientation": name,
            "camera_axis": "".join(
                ("+" if value > 0 else "-") + axis
                for value, axis in zip(_camera, "XYZ")
                if value != 0
            ),
            "screen_right_axis": "".join(
                ("+" if value > 0 else "-") + axis
                for value, axis in zip(right_axis, "XYZ")
                if value != 0
            ),
            "screen_up_axis": "".join(
                ("+" if value > 0 else "-") + axis
                for value, axis in zip(up_axis, "XYZ")
                if value != 0
            ),
            "eyes_visible": bool(eyes_visible),
            "left_eye_px": [float(v) for v in joint_pixels[23]],
            "right_eye_px": [float(v) for v in joint_pixels[24]],
            "eye_radius_px": float(eye_radius_px),
            "arms_hidden": bool(side_view),
            "arm_sections": sections,
        }

    metadata = {
        "schema": "smplx-fourview/1",
        "gender": gender,
        "tpose": True,
        "projection": "orthographic",
        "resolution": [SIZE, SIZE],
        "scale_px_per_m": float(scale),
        "margin_px": MARGIN,
        "head": cache.head(gender),
        "views": view_info,
        "notes": "Silhouette + eye positions only; side views hide both arms "
                 "and mark the near shoulder/arm cross-sections.  All four views "
                 "share one pixel scale and are rendered from the edited rest "
                 "mesh, not from the temporary view pose.",
    }

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("front", "back", "left", "right"):
            buffer = io.BytesIO()
            images[name].save(buffer, format="PNG", optimize=False)
            archive.writestr(f"{name}.png", buffer.getvalue())
        archive.writestr(
            "views.json",
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return output.getvalue()


def export_fourview(cache, gender: str) -> Path:
    with cache.lock:
        data = build_fourview(cache, gender)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache.cache_dir / f"fourview_{gender}.zip"
    fd, temporary = tempfile.mkstemp(suffix=".zip", dir=cache.cache_dir)
    os.close(fd)
    try:
        Path(temporary).write_bytes(data)
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target
