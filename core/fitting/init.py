"""Reference-image alignment: bone-length least squares + pose refinement.

This is the current single-view initializer.  The multi-view profile/silhouette
modules from the refactor plan are intentionally not created because S1/S2 are
not implemented yet.
"""
from __future__ import annotations

import numpy as np

from core.config import GENDERS, REFERENCE_EYE_WEIGHT, REFERENCE_IK_FULL_BODY

def solve_reference_lengths(cache, gender: str, pixels_per_meter: float,
                            eye_midpoint, joints: dict,
                            z_weight: float = 3.0) -> dict:
    """Fit the 21 editable bone lengths so the model's joints land on the picture.

    The picture already sits on the model's eye midpoint, so every joint is
    compared *relative to the eyes*: that removes the ground-offset and the
    plane translation from the problem and makes the fit independent of how
    the picture is placed.

    With ``J_i = J_root + sum_{k in path(i)} v_k * L_k`` (``v_k`` = the posed
    bone vector, ``L_k`` the length factor -- exactly how ``_geometry`` builds
    the joints), the constraint ``J_i - J_eye = delta_i`` is **linear** in
    ``L``, so it is one least-squares solve with a light pull towards 1.
    """
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    mpp = 1.0 / float(pixels_per_meter)
    if not np.isfinite(mpp) or mpp <= 0 or not np.isfinite(float(pixels_per_meter)):
        raise ValueError("invalid pixels_per_meter")
    model, output = cache._output(gender)
    base = output.joints[0, :len(model.parents)].detach().cpu().numpy().astype(np.float64)
    parents = model.parents.cpu().numpy().astype(np.int64)
    bones = [k for k in range(1, 22) if parents[k] >= 0]
    slot = {k: i for i, k in enumerate(bones)}
    vectors = {k: base[k] - base[int(parents[k])] for k in bones}

    # J_eye = J_15 + eye_offset; the 15->23/24 bones are not editable.
    eye_offset = (base[23] + base[24]) / 2.0 - base[15]

    def path(joint: int) -> list[int]:
        chain = []
        while joint > 0:
            chain.append(joint)
            joint = int(parents[joint])
        return chain

    def row_for(joint: int) -> np.ndarray:
        row = np.zeros((3, len(bones)))
        for k in path(joint):
            if k in slot:
                row[0, slot[k]] += vectors[k][0]
                row[1, slot[k]] += vectors[k][1]
                row[2, slot[k]] += vectors[k][2]
        return row

    eye_row = row_for(15)
    rows, rhs = [], []
    used = 0
    for key, xy in joints.items():
        joint = int(key)
        if joint < 0 or joint > 22 or len(xy) != 2:
            continue
        target = np.array([(float(xy[0]) - eye_midpoint[0]) * mpp,
                           (eye_midpoint[1] - float(xy[1])) * mpp])
        full = row_for(joint)
        a = full[:2] - eye_row[:2]
        b = target + eye_offset[:2]
        rows.append(a[0]); rhs.append(b[0] * 1.0)
        rows.append(a[1]); rhs.append(b[1] * 1.0)
        # keep the joint's own depth: sum(v_kz * L_k) == sum(v_kz)
        rows.append(full[2] * z_weight)
        rhs.append(float((full[2]).sum()) * z_weight)
        used += 1
    if used < 4:
        raise ValueError("not enough 2D joints to fit the skeleton")

    design = np.asarray(rows)
    values = np.asarray(rhs)
    lam = 0.15
    design = np.vstack([design, lam * np.eye(len(bones))])
    values = np.concatenate([values, lam * np.ones(len(bones))])
    factors, *_ = np.linalg.lstsq(design, values, rcond=None)
    factors = np.clip(factors, 0.25, 4.0)

    with cache.lock:
        config = cache._segments(gender)
        for k, factor in zip(bones, factors):
            entry = dict(config.get(str(k), {}))
            entry["length"] = float(factor)
            entry.setdefault("radii", [1.0, 1.0, 1.0])
            config[str(k)] = entry
        cache._invalidate_geometry(gender)
        _model, vertices, new_joints, _parents = cache._geometry(gender)

    # residual measured on the geometry that will actually be rendered
    rms_px, max_px = cache._reference_residual(
        gender, pixels_per_meter, eye_midpoint, joints
    )
    return {
        "ok": True,
        "lengths": {str(k): float(f) for k, f in zip(bones, factors)},
        "num_joints": int(used),
        "rms_px": rms_px,
        "max_px": max_px,
    }


def reference_residual(cache, gender, pixels_per_meter, eye_midpoint, joints):
    """RMS / max pixel error of the joints on the geometry being rendered."""
    mpp = 1.0 / float(pixels_per_meter)
    _model, _vertices, new_joints, _parents = cache._geometry(gender)
    eye = (new_joints[23] + new_joints[24]) / 2.0
    errors = []
    for key, xy in joints.items():
        joint = int(key)
        if joint < 0 or joint > 22 or len(xy) != 2:
            continue
        predicted = np.array([
            eye_midpoint[0] + (new_joints[joint][0] - eye[0]) / mpp,
            eye_midpoint[1] - (new_joints[joint][1] - eye[1]) / mpp,
        ])
        errors.append(float(np.hypot(predicted[0] - xy[0], predicted[1] - xy[1])))
    errors = np.asarray(errors) if errors else np.zeros(1)
    return float(np.sqrt((errors ** 2).mean())), float(errors.max())


def scaled_joints(model, body_pose, lengths, global_orient=None):
    """Joints of the *rendered* skeleton: segment lengths applied on top.

    Mirrors ``_geometry``: ``S[k] = S[parent] + (J[k] - J[parent]) * L[k]``.
    """
    output = model(body_pose=body_pose, global_orient=global_orient)
    joints = output.joints[0][:len(model.parents)]
    scaled = joints.clone()
    parents = model.parents
    for index in range(1, len(parents)):
        parent = int(parents[index])
        if parent < 0:
            continue
        scaled[index] = scaled[parent] + (joints[index] - joints[parent]) * lengths[index]
    return scaled


def refine_pose_to_reference(cache, gender: str, pixels_per_meter: float,
                             eye_midpoint, joints: dict,
                             full_body: bool | None = None,
                             iterations: int = 160,
                             learning_rate: float = 0.02,
                             reference_z: dict | None = None,
                             z_weight: float = 20.0) -> dict:
    """Second stage: let the *pose* absorb what the bone lengths cannot.

    Note on "全身 IK": running the viewer's per-joint drag solver in a sweep
    (rotate only the direct parent, checkbox off) is unusable for the torso
    -- rotating a spine joint to fix the neck swings the whole upper body and
    the error explodes (measured: 17 px -> 251 px over one sweep).  The pose
    is therefore solved as a single least-squares problem over the 21 body
    joint rotations, which is stable and never ships a worse pose.

    Each joint is pulled to its picture ``(x, y)`` **inside its own plane**:
    the target depth stays the joint's original ``z`` (``reference_z``), so
    the alignment never tilts the body through the depth axis.

    ``REFERENCE_IK_FULL_BODY`` picks how much chain participates:
    ``False`` keeps the pelvis/root fixed (only the joints rotate locally);
    ``True`` also frees the root orientation, i.e. the whole body may turn.
    """
    if full_body is None:
        full_body = REFERENCE_IK_FULL_BODY
    import torch
    model = cache._model(gender)
    mpp = 1.0 / float(pixels_per_meter)
    deltas, indexes = [], []
    for key, xy in joints.items():
        index = int(key)
        # 23/24 are the eyes: including them pins the eye *line* (and keeps
        # the head from rolling), not just the eye midpoint.
        if 0 <= index <= 24 and len(xy) == 2:
            deltas.append([(float(xy[0]) - eye_midpoint[0]) * mpp,
                           (eye_midpoint[1] - float(xy[1])) * mpp])
            indexes.append(index)
    if not indexes:
        raise ValueError("no 2D joints to refine the pose with")
    target = torch.tensor(deltas, dtype=torch.float32, device=cache.device)
    joint_ids = torch.tensor(indexes, dtype=torch.long, device=cache.device)
    # the eyes are only two constraints against 21 body joints, so they need
    # extra weight to actually level the face
    joint_weight = torch.tensor(
        [REFERENCE_EYE_WEIGHT if index in (23, 24) else 1.0 for index in indexes],
        dtype=torch.float32, device=cache.device,
    )
    if reference_z is None:
        _model0, _v0, current0, _p0 = cache._geometry(gender)
        reference_z = {index: float(current0[index][2]) for index in indexes}
    target_z = torch.tensor(
        [float(reference_z[index]) for index in indexes],
        dtype=torch.float32, device=cache.device,
    )

    factors = np.ones(len(model.parents), dtype=np.float32)
    config = cache._segments(gender)
    for index in range(1, 22):
        factors[index] = float(config.get(str(index), {}).get("length", 1.0))
    lengths = torch.tensor(factors, dtype=torch.float32, device=cache.device)

    with cache.lock:
        pose0 = cache._pose(gender, model).clone()
        orient0 = cache._orient(gender, model).clone()
        before, _ = cache._reference_residual(
            gender, pixels_per_meter, eye_midpoint, joints
        )
        variables = torch.nn.Parameter(pose0.clone())
        parameters = [variables]
        root = None
        if full_body:
            root = torch.nn.Parameter(orient0.clone())
            parameters.append(root)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            scaled = cache._scaled_joints(
                model, variables, lengths,
                root if root is not None else None,
            )
            eye = (scaled[23] + scaled[24]) / 2.0
            points = scaled[joint_ids]
            residual = (points[:, :2] - eye[:2]) - target
            depth = points[:, 2] - target_z          # keep each joint in its own plane
            loss = (residual.square().sum(dim=1) * joint_weight).sum()
            loss = loss + z_weight * (depth.square() * joint_weight).sum()
            loss = loss + 0.02 * (variables - pose0).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            cache.poses[gender] = variables.detach().clone()
            if root is not None:
                cache.global_orients[gender] = root.detach().clone()
            cache.outputs[gender] = model(
                body_pose=cache.poses[gender],
                global_orient=cache.global_orients.get(gender),
            )
        cache._invalidate_geometry(gender)
        after, _ = cache._reference_residual(
            gender, pixels_per_meter, eye_midpoint, joints
        )
        if after > before:                      # never ship a worse pose
            cache.poses[gender] = pose0
            cache.global_orients[gender] = orient0
            cache.outputs.pop(gender, None)
            cache._invalidate_geometry(gender)
            after = before

    return {
        "moved": [int(v) for v in indexes],
        "skipped": [],
        "full_body": bool(full_body),
        "rms_before_px": float(before),
        "pose_rms_px": float(after),
    }


def align_reference(cache, gender: str, pixels_per_meter: float,
                    eye_midpoint, joints: dict,
                    full_body: bool | None = None) -> dict:
    """Whole reference alignment: bone lengths first, then the pose pass.

    Each joint keeps the depth it had before the alignment (``reference_z``),
    so the fit happens inside the joint's own xy plane and the body is never
    tilted through the z axis.
    """
    model, base_output = cache._output(gender)
    base_joints = base_output.joints[0].detach().cpu().numpy()
    reference_z = {index: float(base_joints[index][2]) for index in range(25)}
    fit = cache.solve_reference_lengths(gender, pixels_per_meter, eye_midpoint, joints)
    lengths_fit = {"rms_px": fit["rms_px"], "max_px": fit["max_px"]}
    pose = cache.refine_pose_to_reference(
        gender, pixels_per_meter, eye_midpoint, joints,
        full_body=full_body, reference_z=reference_z,
    )
    rms_px, max_px = cache._reference_residual(
        gender, pixels_per_meter, eye_midpoint, joints
    )
    _model, _vertices, current, _parents = cache._geometry(gender)
    z_drift = max(
        abs(float(current[index][2]) - reference_z[index]) for index in reference_z
    )
    eye_tilt = float(np.degrees(np.arctan2(
        float(current[24][1] - current[23][1]),
        float(current[24][0] - current[23][0]),
    )))
    if eye_tilt > 90.0:
        eye_tilt -= 180.0
    elif eye_tilt < -90.0:
        eye_tilt += 180.0
    return {
        "ok": True,
        "lengths": fit["lengths"],
        "num_joints": len(pose["moved"]),
        "rms_px": rms_px,
        "max_px": max_px,
        "lengths_rms_px": lengths_fit["rms_px"],
        "lengths_max_px": lengths_fit["max_px"],
        "z_drift_m": float(z_drift),
        "eye_tilt_deg": float(eye_tilt),
        "pose_moved": pose["moved"],
        "pose_skipped": pose["skipped"],
        "full_body": pose["full_body"],
    }
