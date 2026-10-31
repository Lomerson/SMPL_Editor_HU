"""Skeleton-aware geometry: the current post-LBS segment warp and part labels.

The smooth LBS deformation from the refactor plan is not implemented yet; this
module keeps the exact behaviour of the pre-split server.
"""
from __future__ import annotations

import numpy as np

from core.config import GENDERS, JOINT_NAMES

def invalidate(cache, gender: str):
    cache.geometry_cache.pop(gender, None)


def geometry(cache, gender: str):
    if gender in cache.geometry_cache:
        return cache.geometry_cache[gender]
    model, output = cache._output(gender)
    base_vertices = output.vertices[0].cpu().numpy().astype(np.float32)
    base_joints = output.joints[0, :len(model.parents)].cpu().numpy().astype(np.float32)
    parents = model.parents.cpu().numpy().astype(np.int32)
    joints = base_joints.copy()
    segments = cache._segments(gender)
    # No segment edits must mean the exact, unmodified SMPL-X output.
    # This avoids even harmless numerical drift in the canonical T-pose.
    if not segments:
        result = (model, base_vertices, joints, parents)
        cache.geometry_cache[gender] = result
        return result
    for child in range(1, len(parents)):
        parent = int(parents[child])
        if parent < 0:
            continue
        factor = float(segments.get(str(child), {}).get("length", 1.0))
        joints[child] = joints[parent] + (base_joints[child] - base_joints[parent]) * factor

    vertices = base_vertices.copy()
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    dominant_joint = np.argmax(weights[:, :len(joints)], axis=1)
    body_children = [index for index in range(1, 22) if parents[index] >= 0]
    nearest_distance = np.full(len(base_vertices), np.inf, dtype=np.float32)
    nearest_segment = np.full(len(base_vertices), -1, dtype=np.int32)
    for candidate in body_children:
        parent = int(parents[candidate])
        start, end = base_joints[parent], base_joints[candidate]
        vector = end - start
        length_sq = float(vector @ vector)
        if length_sq < 1e-10:
            continue
        relative = base_vertices - start
        t0 = np.clip((relative @ vector) / length_sq, 0.0, 1.0)
        distance_sq = ((base_vertices - (start + t0[:, None] * vector)) ** 2).sum(axis=1)
        replace = distance_sq < nearest_distance
        nearest_distance[replace] = distance_sq[replace]
        nearest_segment[replace] = candidate

    def descendants(root: int) -> set[int]:
        result = {root}
        changed = True
        while changed:
            changed = False
            for index, parent in enumerate(parents):
                if parent in result and index not in result:
                    result.add(index)
                    changed = True
        return result

    for child_text, config in segments.items():
        child = int(child_text)
        parent = int(parents[child])
        if child > 21 or parent < 0:
            continue
        start0, end0 = base_joints[parent], base_joints[child]
        axis0 = end0 - start0
        base_length = float(np.linalg.norm(axis0))
        if base_length < 1e-6:
            continue
        axis0 /= base_length
        relative0 = base_vertices - start0
        raw_t = relative0 @ axis0 / base_length
        feather = 0.08
        t = np.clip(raw_t, 0.0, 1.0)
        axial_mask = np.clip((raw_t + feather) / feather, 0.0, 1.0)
        axial_mask *= np.clip((1.0 + feather - raw_t) / feather, 0.0, 1.0)
        segment_owner = (nearest_segment == child).astype(np.float32) * axial_mask
        if child in (18, 19):
            # Upper-arm seams share vertices with torso and forearm. Gate
            # the continuous upper-arm deformation by this side's
            # shoulder+elbow weights so torso surface cannot be pulled
            # into the arm, and so the distal forearm remains a child
            # branch that translates with the elbow.
            if child == 18:
                arm_weight = weights[:, 16] + weights[:, 18]
                neighbor_weight = weights[:, 13] + weights[:, 20]
            else:
                arm_weight = weights[:, 17] + weights[:, 19]
                neighbor_weight = weights[:, 14] + weights[:, 21]
            upper_surface = (arm_weight > 0.03) & (arm_weight >= neighbor_weight * 0.55) & (raw_t <= 1.0)
            segment_owner = upper_surface.astype(np.float32) * axial_mask
        if child in (7, 8):
            # At the ankle seam, nearest-bone classification leaves some
            # genuine shin vertices assigned to thigh/foot. Use the local
            # SMPL-X knee+ankle weights, constrained against neighboring
            # same-side weights, so the opposite calf stays untouched.
            if child == 7:
                calf_weight = weights[:, 4] + weights[:, 7]
                neighbor_weight = weights[:, 1] + weights[:, 10]
            else:
                calf_weight = weights[:, 5] + weights[:, 8]
                neighbor_weight = weights[:, 2] + weights[:, 11]
            calf_surface = (calf_weight > 0.03) & (calf_weight >= neighbor_weight * 0.55) & (raw_t <= 1.0)
            segment_owner = calf_surface.astype(np.float32) * axial_mask
        # Use this link's local length change only. The absolute child
        # displacement also includes every ancestor's edit (for example a
        # shortened thigh moves the knee). Applying that absolute value to
        # a calf again double-counted the thigh deformation and separated
        # the mesh from its correctly-recursed skeleton.
        delta = (joints[child] - joints[parent]) - (base_joints[child] - base_joints[parent])
        desired_translation = t[:, None] * delta
        if child in (18, 19):
            # Keep a short shoulder-side attachment band fixed so the
            # torso-to-upper-arm seam does not cave in during resizing.
            shoulder_guard = 0.12
            upper_t = np.clip((t - shoulder_guard) / (1.0 - shoulder_guard), 0.0, 1.0)
            desired_translation = upper_t[:, None] * delta
        if child in (20, 21):
            # Weld the distal forearm to the rigidly translated hand. The
            # final wrist band transitions from axial scaling to the full
            # wrist displacement, preventing a fold across shared seam
            # triangles while preserving the forearm's main length edit.
            seam = np.clip((t - 0.78) / 0.22, 0.0, 1.0)
            seam = seam * seam * (3.0 - 2.0 * seam)
            desired_translation = (t + (1.0 - t) * seam)[:, None] * delta
        subtree = descendants(child)
        downstream_segments = list((subtree - {child}) & set(body_children))
        is_branch = np.isin(nearest_segment, downstream_segments)
        is_branch |= np.isin(dominant_joint, list(subtree)) & (raw_t > 1.0)
        rigid_hand = np.zeros(len(vertices), dtype=bool)
        rigid_forearm = np.zeros(len(vertices), dtype=bool)
        if child == 18:
            rigid_forearm = (weights[:, 18] + weights[:, 20] > 0.03) & (raw_t > 0.92)
        elif child == 19:
            rigid_forearm = (weights[:, 19] + weights[:, 21] > 0.03) & (raw_t > 0.92)
        if child in (20, 21):
            rigid_hand = np.isin(dominant_joint, list(subtree))
        downstream = (is_branch & (raw_t > 1.0)) | rigid_hand | rigid_forearm
        vertices[downstream] += delta
        segment_owner[rigid_hand] = 0.0
        segment_owner[rigid_forearm] = 0.0
        vertices += desired_translation * segment_owner[:, None]

        start, end = joints[parent], joints[child]
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            continue
        axis /= length
        relative = vertices - start
        radial = relative - np.outer(relative @ axis, axis)
        radii = np.asarray(config.get("radii", [1.0, 1.0, 1.0]), dtype=np.float32)
        scale = np.interp(t, np.linspace(0.0, 1.0, len(radii)), radii).astype(np.float32)
        vertices += radial * ((scale - 1.0) * segment_owner)[:, None]

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
    }


def owners(cache, gender: str) -> bytes:
    """One byte per vertex: the body segment (1..21) that owns it.

    This is the *same* geometric partition the segment editor deforms with
    (nearest bone segment in the rest pose), so the highlighted skin is
    exactly the surface a segment edit moves.  ``argmax(lbs_weights)`` would
    be wrong here: SMPL-X binds a bone's surface to its *proximal* joint, so
    a forearm segment would only light up ~200 hand vertices.
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
