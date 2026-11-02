"""Skeleton-aware identity deformation and part labels.

The production path is :class:`core.deform.RestDeformer`: it deforms the
SMPL-X *rest* template, propagates every joint in topological order, shares a
single girth value at each joint, and then poses the edited rest mesh with the
model's own skin weights.  ``SMPLX_DEFORM=legacy`` selects the old post-LBS
warp in :mod:`core.geometry_legacy` for A/B comparison.
"""
from __future__ import annotations

import os

import numpy as np

from core.config import GENDERS, JOINT_NAMES
from core.deform import RestDeformer, pose_is_identity
from core.geometry_legacy import legacy_geometry
from core.head import HEAD_LABELS, HEAD_RANGES, HeadDeformer, is_default
from core.symmetry import SYMMETRY_JOINTS, SYMMETRY_SEGMENTS


def invalidate(cache, gender: str):
    cache.geometry_cache.pop(gender, None)


def _rest_state(cache, gender: str):
    """Edited rest mesh + joints, with no pose applied.

    Returns ``(model, vertices, joints, parents, deformer)``.  ``deformer`` is
    ``None`` when there is no identity edit, so callers can take an exact
    template shortcut.
    """
    model = cache._model(gender)
    parents = model.parents.cpu().numpy().astype(np.int32)
    segments = cache._segments(gender)
    head_params = cache.head_configs.get(gender)
    head_edit = bool(head_params) and not is_default(head_params)
    displacement = cache.displacements.get(gender)

    if not segments and not head_edit and displacement is None:
        vertices = model.v_template.detach().cpu().numpy().astype(np.float32)
        joints = (
            model.J_regressor @ model.v_template
        ).detach().cpu().numpy().astype(np.float32)
        return model, vertices, joints, parents, None

    deformer = RestDeformer(model, segments)
    vertices, joints = deformer.deform_rest()
    if head_edit:
        vertices, joints = HeadDeformer(model, joints, head_params).apply(vertices)
    if displacement is not None:
        vertices = (vertices + displacement).astype(np.float32)
    return model, vertices, joints, parents, deformer


def rest_geometry(cache, gender: str):
    """The exported identity: edited rest mesh, T-pose, no view pose."""
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    model, vertices, joints, parents, _deformer = _rest_state(cache, gender)
    return model, vertices, joints, parents


def geometry(cache, gender: str):
    if gender in cache.geometry_cache:
        return cache.geometry_cache[gender]
    if os.environ.get("SMPLX_DEFORM", "rest").strip().lower() == "legacy":
        return legacy_geometry(cache, gender)

    model = cache._model(gender)
    parents = model.parents.cpu().numpy().astype(np.int32)
    segments = cache._segments(gender)
    head_params = cache.head_configs.get(gender)
    head_edit = bool(head_params) and not is_default(head_params)
    displacement = cache.displacements.get(gender)
    body_pose = cache.poses.get(gender)
    global_orient = cache.global_orients.get(gender)

    # Keep the unedited, unposed output byte-for-byte identical to the old
    # viewer (SMPL-X's forward pass differs from ``v_template`` by ~1e-7).
    # The custom LBS matches the forward pass exactly once a pose is set, so
    # only this canonical T-pose shortcut is needed.
    if not segments and not head_edit and displacement is None and pose_is_identity(body_pose, global_orient):
        _model, output = cache._output(gender)
        vertices = output.vertices[0].cpu().numpy().astype(np.float32)
        joints = output.joints[0, :len(parents)].cpu().numpy().astype(np.float32)
        result = (model, vertices, joints, parents)
        cache.geometry_cache[gender] = result
        return result

    model, vertices, joints, parents, deformer = _rest_state(cache, gender)
    if not pose_is_identity(body_pose, global_orient):
        if deformer is None:
            deformer = RestDeformer(model, segments)
        vertices, joints = deformer.pose(vertices, joints, body_pose, global_orient)

    result = (model, vertices, joints, parents)
    cache.geometry_cache[gender] = result
    return result


def skeleton(cache, gender: str) -> dict:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    model, _vertices, joints, parents = cache._geometry(gender)
    # parents covers the 55 SMPL-X body and hand joints. The remaining
    # regressed landmarks are not a kinematic skeleton hierarchy.
    return {
        "joints": joints.tolist(),
        "parents": parents.tolist(),
        "names": JOINT_NAMES,
        "segments": cache._segments(gender),
        "head": cache.head(gender),
        "head_ranges": {key: list(value) for key, value in HEAD_RANGES.items()},
        "head_labels": HEAD_LABELS,
        "has_displacement": cache.displacement(gender) is not None,
        "symmetry": {
            "segments": {str(key): value for key, value in SYMMETRY_SEGMENTS.items()},
            "joints": {str(key): value for key, value in SYMMETRY_JOINTS.items()},
        },
    }


def owners(cache, gender: str) -> bytes:
    """One byte per vertex: the body segment (1..21) that owns it.

    This is the geometric partition used only for *highlighting* the segment
    under the cursor.  Deformation itself is weight-based (``core.deform``),
    so a highlighted region is an approximation of the surface a segment
    edit moves, not a hard skinning mask.
    """
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    model = cache._model(gender)
    vertices = model.v_template.detach().cpu().numpy().astype(np.float32)
    joints = (model.J_regressor @ model.v_template).detach().cpu().numpy()[:len(model.parents)]
    parents = model.parents.cpu().numpy().astype(np.int32)
    nearest_distance = np.full(len(vertices), np.inf, dtype=np.float32)
    nearest_segment = np.zeros(len(vertices), dtype=np.uint8)
    for candidate in range(1, 22):
        parent = int(parents[candidate])
        if parent < 0:
            continue
        start, end = joints[parent], joints[candidate]
        vector = end - start
        length_sq = float(vector @ vector)
        if length_sq < 1e-10:
            continue
        relative = vertices - start
        t = np.clip((relative @ vector) / length_sq, 0.0, 1.0)
        distance_sq = ((vertices - (start + t[:, None] * vector)) ** 2).sum(axis=1)
        replace = distance_sq < nearest_distance
        nearest_distance[replace] = distance_sq[replace]
        nearest_segment[replace] = candidate
    return nearest_segment.tobytes()
