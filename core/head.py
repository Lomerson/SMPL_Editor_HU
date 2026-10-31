"""Dedicated head deformer.

The generic segment editor is deliberately kept away from the head.  SMPL-X's
head is not a single bone with a clean child axis: joint 15 owns the skull and
face, joint 22 owns the jaw, and joints 23/24 are independent eyeballs.  The
old viewer folded all of that into one 15->22 segment, took the mean child
direction as the "length" axis, and moved the eyes without scaling them.

This module implements the head as its own rest-pose edit:

* ``head_length`` is the chin-to-crown **+Y** axis (not front/back thickness);
* ``head_width`` is X and ``head_depth`` is Z;
* the jaw is scaled about the *edited* joint 22;
* the eyes are uniform spheres scaled about their own edited centres, with
  radius ``sqrt(width * depth)`` so a narrower/shallower head cannot make the
  eyeballs protrude;
* the neck is blended by the SMPL-X head weights, so there is no hard step.

Everything is applied to the edited rest mesh before the LBS pose, so the
skeleton, jaw pose and eye pose keep working.
"""
from __future__ import annotations

import numpy as np

HEAD_DEFAULTS = {
    "head_length": 1.0,
    "head_width": 1.0,
    "head_depth": 1.0,
    "jaw": 1.0,
    "eye": 1.0,
}

HEAD_RANGES = {
    "head_length": (0.80, 1.25),
    "head_width": (0.85, 1.15),
    "head_depth": (0.85, 1.15),
    "jaw": (0.80, 1.20),
    "eye": (0.85, 1.15),
}

HEAD_LABELS = {
    "head_length": "头长",
    "head_width": "头宽",
    "head_depth": "头厚",
    "jaw": "下颚",
    "eye": "眼球",
}

_WEIGHT_TOLERANCE = 1e-9


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def normalize_head_params(params) -> dict:
    """Clamp a request to the supported head ranges."""
    result = {}
    for key, default in HEAD_DEFAULTS.items():
        raw = (params or {}).get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
        low, high = HEAD_RANGES[key]
        result[key] = float(np.clip(value, low, high))
    return result


def is_default(params) -> bool:
    if not params:
        return True
    return all(
        abs(float(params.get(key, default)) - default) < _WEIGHT_TOLERANCE
        for key, default in HEAD_DEFAULTS.items()
    )


def _smooth_head_weights(weights, faces, iterations=3):
    """Diffuse the head/jaw weights over the mesh.

    SMPL-X's jaw weight changes from 0 to ~0.6 across a sub-millimetre mouth
    edge.  Applying a 20% jaw scale on top of that step creates a visible
    pinch.  Three neighbour-averaging passes spread the transition over a few
    rings without moving the eye spheres (they are separate components).
    """
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] < 3:
        return weights
    count = len(weights)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    degree = (
        np.bincount(edges[:, 0], minlength=count)
        + np.bincount(edges[:, 1], minlength=count)
    ).astype(np.float32)
    degree = np.maximum(degree, 1.0)
    result = {}
    for joint in (15, 22, 23, 24):
        values = weights[:, joint].astype(np.float32).copy()
        for _ in range(iterations):
            sums = np.zeros(count, dtype=np.float32)
            np.add.at(sums, edges[:, 0], values[edges[:, 1]])
            np.add.at(sums, edges[:, 1], values[edges[:, 0]])
            values = 0.5 * values + 0.5 * sums / degree
        result[joint] = values
    return result


def _smooth_scalar(values, faces, iterations=3):
    """Neighbour-average one per-vertex scalar over the mesh."""
    faces = np.asarray(faces)
    count = len(values)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    degree = (
        np.bincount(edges[:, 0], minlength=count)
        + np.bincount(edges[:, 1], minlength=count)
    ).astype(np.float32)
    degree = np.maximum(degree, 1.0)
    result = values.astype(np.float32).copy()
    for _ in range(iterations):
        sums = np.zeros(count, dtype=np.float32)
        np.add.at(sums, edges[:, 0], result[edges[:, 1]])
        np.add.at(sums, edges[:, 1], result[edges[:, 0]])
        result = 0.5 * result + 0.5 * sums / degree
    return result


class HeadDeformer:
    """Apply the five head parameters to an edited rest mesh."""

    def __init__(self, model, joints, params):
        self.params = normalize_head_params(params)
        self.joints = np.asarray(joints, dtype=np.float32)
        weights = model.lbs_weights
        if hasattr(weights, "detach"):
            weights = weights.detach()
        if hasattr(weights, "cpu"):
            weights = weights.cpu()
        self.weights = np.asarray(weights, dtype=np.float32)
        faces = model.faces
        if hasattr(faces, "detach"):
            faces = faces.detach()
        if hasattr(faces, "cpu"):
            faces = faces.cpu()
        self.faces = np.asarray(faces)
        self.pivot = self.joints[15].copy()

    def apply(self, vertices):
        if is_default(self.params):
            return np.asarray(vertices, dtype=np.float32).copy(), self.joints.copy()

        vertices = np.asarray(vertices, dtype=np.float32)
        length = self.params["head_length"]
        width = self.params["head_width"]
        depth = self.params["head_depth"]
        jaw = self.params["jaw"]
        eye = self.params["eye"]

        pivot = self.pivot
        scale = np.asarray((width, length, depth), dtype=np.float32)
        head_vertices = pivot + (vertices - pivot) * scale

        # Joint 22 follows the head scale; the jaw edit then scales the jaw
        # surface about that edited joint.
        jaw_joint = pivot + (self.joints[22] - pivot) * scale
        jaw_vertices = jaw_joint + jaw * (head_vertices - jaw_joint)

        # Eyeballs are independent spheres.  Their centres follow the full 3D
        # head scale; their radius uses sqrt(width * depth) so head length
        # does not change the eye size.
        eye_left = pivot + (self.joints[23] - pivot) * scale
        eye_right = pivot + (self.joints[24] - pivot) * scale
        eye_radius = float(np.sqrt(width * depth) * eye)
        eye_left_vertices = eye_left + eye_radius * (vertices - self.joints[23])
        eye_right_vertices = eye_right + eye_radius * (vertices - self.joints[24])

        smooth = _smooth_head_weights(self.weights, self.faces)
        w15 = smooth[15]
        w22 = smooth[22]
        w23 = smooth[23]
        w24 = smooth[24]
        head_weight = np.clip(w15 + w22 + w23 + w24, 0.0, 1.0)
        mask = _smooth_scalar(_smoothstep(head_weight), self.faces, iterations=8)

        head_total = np.maximum(w15 + w22 + w23 + w24, 1e-6)
        # The jaw weight itself changes too abruptly at the lip seam for a
        # 20% scale.  Use a smooth *spatial* lower-face mask instead: below
        # the mouth and in front of the ears.  This keeps the jaw and the
        # adjacent lower face moving together.
        # Relative to the (possibly spine-moved) head pivot, so a spine
        # length edit cannot shift the jaw region out of the mask.
        jaw_lower = _smoothstep((0.005 - (vertices[:, 1] - pivot[1])) / 0.035)
        jaw_front = _smoothstep(((vertices[:, 2] - pivot[2]) - 0.010) / 0.050)
        jaw_weight = _smooth_scalar(jaw_lower * jaw_front, self.faces, iterations=3)
        eye_weight = _smoothstep((w23 + w24) / head_total)
        eye_total = w23 + w24
        eye_vertices = head_vertices.copy()
        has_eye = eye_total > 1e-6
        eye_vertices[has_eye] = (
            w23[has_eye, None] * eye_left_vertices[has_eye]
            + w24[has_eye, None] * eye_right_vertices[has_eye]
        ) / eye_total[has_eye, None]

        base = head_vertices * (1.0 - jaw_weight[:, None]) + jaw_vertices * jaw_weight[:, None]
        base = base * (1.0 - eye_weight[:, None]) + eye_vertices * eye_weight[:, None]
        result = vertices * (1.0 - mask[:, None]) + base * mask[:, None]

        joints = self.joints.copy()
        joints[22] = jaw_joint
        joints[23] = eye_left
        joints[24] = eye_right
        return result.astype(np.float32), joints
