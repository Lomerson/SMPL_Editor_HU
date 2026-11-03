"""Four-view T-pose reference export.

The output is a set of full shaded orthographic skin screenshots (not
silhouettes).  The side views hide both arms, because in a T-pose the arms
point at the camera and would leave a hand/finger bump on the torso; the near
shoulder/arm cross-sections are still marked.

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
from PIL import Image, ImageDraw

from core.config import GENDERS

SIZE = 1024
MARGIN = 48
BACKGROUND = (255, 255, 255)
SKIN = (198, 172, 150)
AMBIENT = 0.38
SECTION = (44, 44, 44)

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


def _vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _render_view(vertices, faces, joints, view, scale, normals, arm_weight=None):
    """Z-buffered orthographic skin render, one view."""
    pixels, joint_pixels = _project(vertices, joints, view, scale, SIZE, MARGIN)
    camera = np.asarray(view[1], dtype=np.float32)
    right = np.asarray(view[2], dtype=np.float32)
    up = np.asarray(view[3], dtype=np.float32)
    depth = vertices @ camera
    light = camera * 0.70 + up * 0.28 + right * 0.18
    light /= max(float(np.linalg.norm(light)), 1e-6)

    z_buffer = np.full((SIZE, SIZE), -np.inf, dtype=np.float32)
    canvas = np.ones((SIZE, SIZE, 3), dtype=np.float32) * 255.0
    base = np.asarray(SKIN, dtype=np.float32)

    for triangle in faces:
        # A side view looks down the arms; drop fully-arm triangles so the
        # hand/finger silhouette cannot create a bump on the torso.
        if arm_weight is not None and float(arm_weight[triangle].min()) > 0.5:
            continue
        points = pixels[triangle]
        depths = depth[triangle]
        x0 = max(0, int(np.floor(float(points[:, 0].min()))))
        x1 = min(SIZE - 1, int(np.ceil(float(points[:, 0].max()))))
        y0 = max(0, int(np.floor(float(points[:, 1].min()))))
        y1 = min(SIZE - 1, int(np.ceil(float(points[:, 1].max()))))
        if x1 < x0 or y1 < y0:
            continue
        den = (
            (points[1, 1] - points[2, 1]) * (points[0, 0] - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (points[0, 1] - points[2, 1])
        )
        if abs(float(den)) < 1e-9:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        l0 = (
            (points[1, 1] - points[2, 1]) * (grid_x - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (grid_y - points[2, 1])
        ) / den
        l1 = (
            (points[2, 1] - points[0, 1]) * (grid_x - points[2, 0])
            + (points[0, 0] - points[2, 0]) * (grid_y - points[2, 1])
        ) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0.0) & (l1 >= 0.0) & (l2 >= 0.0)
        if not inside.any():
            continue
        pixel_depth = l0 * depths[0] + l1 * depths[1] + l2 * depths[2]
        region = z_buffer[y0:y1 + 1, x0:x1 + 1]
        mask = inside & (pixel_depth > region)
        if not mask.any():
            continue

        n0, n1, n2 = normals[triangle]
        face_normal = np.cross(
            vertices[triangle[1]] - vertices[triangle[0]],
            vertices[triangle[2]] - vertices[triangle[0]],
        )
        if float(np.dot(face_normal, camera)) < 0.0:
            n0, n1, n2 = -n0, -n1, -n2
        i0 = max(float(np.dot(n0, light)), 0.0)
        i1 = max(float(np.dot(n1, light)), 0.0)
        i2 = max(float(np.dot(n2, light)), 0.0)
        shade = np.clip(l0 * i0 + l1 * i1 + l2 * i2, 0.0, 1.0)
        rgb = (AMBIENT + (1.0 - AMBIENT) * shade)[..., None] * base
        region[mask] = pixel_depth[mask]
        canvas[y0:y1 + 1, x0:x1 + 1][mask] = rgb[mask]

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)), joint_pixels


def build_fourview(cache, gender: str) -> bytes:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    model, vertices, joints, parents = cache.rest_geometry(gender)
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)
    normals = _vertex_normals(vertices, faces)
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
            vertices, faces, joints, view, scale, normals,
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
            "render_mode": "shaded",
        }

    metadata = {
        "schema": "smplx-fourview/1",
        "gender": gender,
        "tpose": True,
        "projection": "orthographic",
        "render_mode": "shaded",
        "resolution": [SIZE, SIZE],
        "scale_px_per_m": float(scale),
        "margin_px": MARGIN,
        "head": cache.head(gender),
        "views": view_info,
        "notes": "Full shaded skin screenshots; side views hide both arms and "
                 "mark the near shoulder/arm cross-sections.  All four views "
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
