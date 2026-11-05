"""Rest-pose identity deformation and custom LBS pose.

The editor no longer warps the already-posed SMPL-X mesh.  It deforms the
rest template once, then poses that edited rest mesh with SMPL-X's own skin
weights.  Keeping the two stages separate is what lets a spine edit move the
arms and head without a ``downstream`` mask: every joint, including the
fingers and the jaw/eyes, is moved by the same topological length pass and
then follows its normal skin weights.

Identity deformation has three parts:

``length``
    ``J[child] = J[parent] + (J0[child] - J0[parent]) * factor[child]`` for
    every joint in the 55-joint hierarchy.  The vertex displacement is the
    skin-weighted joint translation ``W @ (J - J0)``.  A branch (pelvis,
    chest) therefore moves all of its children; no vertex is left behind.

``girth``
    Each bone owns a 2..12 point radial profile.  The endpoints are shared with
    the adjacent bone (``update_segment`` keeps them equal), and the profile is
    applied with the *proximal* joint's SMPL-X weight -- the weight that
    actually owns that bone's surface.  The elbow therefore receives the same
    value from the upper arm's 100% and the forearm's 0%, with no hard
    ``segment_owner`` seam and no unscaled ring at joint 19.

The limbs also have a smooth left/right gate: a crotch vertex with equal hip
weights is not pulled sideways by a single thigh edit.  The thigh profile gets
an additional proximal guard because SMPL-X's hip weight also covers the
pelvis; the guard fades the edit out where the thigh weight is too weak to be
thigh surface.

The legacy post-LBS warp stays available through ``SMPLX_DEFORM=legacy`` for
comparison/rollback; see :mod:`core.geometry_legacy`.
"""
from __future__ import annotations

import numpy as np

# The UI exposes 0.5..1.8, but a reference-image fit may write 0.25..4.0.
# Clamp the kernel at the wider bounds so a stale/bad request cannot explode
# the mesh.
LENGTH_MIN = 0.25
LENGTH_MAX = 4.0
GIRTH_MIN = 0.3
GIRTH_MAX = 1.5


def _numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=dtype)


def _unit(vector, fallback=(0.0, 1.0, 0.0)):
    vector = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (vector / length).astype(np.float32)


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _clip_length(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, LENGTH_MIN, LENGTH_MAX))


def _clip_girth(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, GIRTH_MIN, GIRTH_MAX))


class RestDeformer:
    """Deform the SMPL-X rest template, then optionally pose it."""

    def __init__(self, model, segments):
        self.model = model
        self.vertices0 = _numpy(model.v_template)
        rest_joints = getattr(model, "rest_joints", None)
        self.joints0 = (
            _numpy(rest_joints)
            if rest_joints is not None
            else _numpy(model.J_regressor @ model.v_template)
        )
        self.parents = _numpy(model.parents, dtype=np.int64)
        self.weights = _numpy(model.lbs_weights)
        self.segments = {
            str(key): dict(value) for key, value in (segments or {}).items()
        }
        self.children = [[] for _ in range(len(self.parents))]
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                self.children[int(parent)].append(child)

        self.factors = self._length_factors()
        self.joints = self._joint_positions()
        self.gates = self._side_gates()
        self.profiles = self._bone_profiles()
        self.head_girth = self._head_girth()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _radii(self, key):
        entry = self.segments.get(str(key))
        if not entry:
            return None
        radii = entry.get("radii")
        if not isinstance(radii, (list, tuple)) or len(radii) < 2:
            return None
        return [_clip_girth(value) for value in radii]

    def _length_factors(self):
        factors = np.ones(len(self.parents), dtype=np.float32)
        for key, entry in self.segments.items():
            try:
                child = int(key)
            except (TypeError, ValueError):
                continue
            if 1 <= child < len(factors):
                factors[child] = _clip_length(entry.get("length", 1.0))
        return factors

    def _joint_positions(self):
        # Preserve the exact unedited template bit-for-bit: adding and
        # subtracting the same rest offset can lose low bits even when the
        # factor is 1.0.
        if not self.segments or bool(np.all(self.factors == 1.0)):
            return self.joints0.copy()
        joints = self.joints0.copy()
        # SMPL-X parents always precede their children, so one pass is enough.
        for child in range(1, len(self.parents)):
            parent = int(self.parents[child])
            if parent < 0:
                continue
            joints[child] = joints[parent] + (
                self.joints0[child] - self.joints0[parent]
            ) * self.factors[child]
        return joints.astype(np.float32)

    def _side_gates(self):
        """Smooth left/right gates for limb girth.

        A vertex in the crotch carries roughly equal left/right hip weights.
        Scaling it around only one hip would pull the midline sideways.  The
        gate fades to zero exactly on the midline, while the thigh surface
        (where one leg dominates) keeps the full edit.
        """
        weights = self.weights
        gates = np.ones((len(weights), 25), dtype=np.float32)

        left_leg = weights[:, 1] + weights[:, 4] + weights[:, 7] + weights[:, 10]
        right_leg = weights[:, 2] + weights[:, 5] + weights[:, 8] + weights[:, 11]
        leg_total = np.maximum(left_leg + right_leg, 1e-6)
        left_leg_gate = np.clip((left_leg - right_leg) / leg_total, 0.0, 1.0)
        right_leg_gate = np.clip((right_leg - left_leg) / leg_total, 0.0, 1.0)

        left_arm = weights[:, 13] + weights[:, 16] + weights[:, 18] + weights[:, 20]
        right_arm = weights[:, 14] + weights[:, 17] + weights[:, 19] + weights[:, 21]
        arm_total = np.maximum(left_arm + right_arm, 1e-6)
        left_arm_gate = np.clip((left_arm - right_arm) / arm_total, 0.0, 1.0)
        right_arm_gate = np.clip((right_arm - left_arm) / arm_total, 0.0, 1.0)

        for joint in (1, 4, 7, 10):
            gates[:, joint] = left_leg_gate
        for joint in (2, 5, 8, 11):
            gates[:, joint] = right_leg_gate
        for joint in (13, 16, 18, 20):
            gates[:, joint] = left_arm_gate
        for joint in (14, 17, 19, 21):
            gates[:, joint] = right_arm_gate
        return gates

    def _bone_profiles(self):
        profiles = []
        for child in range(1, 22):
            radii = self._radii(child)
            if radii is None:
                continue
            parent = int(self.parents[child])
            vector = self.joints0[child] - self.joints0[parent]
            length = float(np.linalg.norm(vector))
            if length < 1e-8:
                continue
            profiles.append(
                {
                    "parent": parent,
                    "child": child,
                    "axis": _unit(vector),
                    "length": length,
                    "radii": np.asarray(radii, dtype=np.float32),
                    # SMPL-X's hip weight also covers the pelvis; the thigh
                    # profile therefore needs a proximal weight guard.
                    "guard": child in (4, 5),
                }
            )
        return profiles

    def _head_girth(self):
        radii = self._radii(15)
        return float(radii[-1]) if radii is not None else 1.0

    # ------------------------------------------------------------------
    # identity deformation
    # ------------------------------------------------------------------
    def deform_rest(self):
        vertices = self.vertices0
        displacement = np.zeros_like(vertices)

        # Girth: one smooth radial profile per bone.  The weight is the
        # *proximal* joint's SMPL-X weight, which is the weight that actually
        # owns that bone's surface.  The elbow gets the same endpoint value
        # from the upper arm (100%) and the forearm (0%), so there is no
        # unscaled ring between them.
        for profile in self.profiles:
            parent = profile["parent"]
            child = profile["child"]
            axis = profile["axis"]
            length = profile["length"]
            radii = profile["radii"]
            relative = vertices - self.joints0[parent]
            t = np.clip((relative @ axis) / length, 0.0, 1.0)
            scale = np.interp(t, np.linspace(0.0, 1.0, len(radii)), radii)
            if float(np.abs(scale - 1.0).max()) < 1e-8:
                continue
            radial = relative - (relative @ axis)[:, None] * axis
            weight = self.weights[:, parent] * self.gates[:, child]
            if profile["guard"]:
                weight = weight * _smoothstep((self.weights[:, parent] - 0.5) / 0.3)
            displacement += (weight * (scale - 1.0))[:, None] * radial

        # The head segment needs its own width/depth scaling: its surface is
        # bound to joint 15, not to the neck's proximal joint.  This is a
        # stopgap until the dedicated head editor (which also scales the jaw
        # and eyes) lands.
        if abs(self.head_girth - 1.0) > 1e-8:
            relative = vertices - self.joints0[15]
            axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
            radial = relative - (relative @ axis)[:, None] * axis
            displacement += (
                self.weights[:, 15:16] * (self.head_girth - 1.0) * radial
            )

        # Length: one topological pass already moved every downstream joint,
        # so the skin-weighted translation follows branches automatically.
        displacement += self.weights @ (self.joints - self.joints0)
        return (vertices + displacement).astype(np.float32), self.joints.copy()

    # ------------------------------------------------------------------
    # pose
    # ------------------------------------------------------------------
    def pose(self, vertices, joints, body_pose=None, global_orient=None):
        """Apply SMPL-X LBS to the edited rest mesh.

        ``body_pose`` is the 63-vector used by the viewer (joints 1..21);
        ``global_orient`` is the root rotation.  Jaw/eye/hand poses are zero,
        matching the editor, but their rest joints still carry any identity
        edit and their vertices still follow the normal 55-joint weights.
        """
        import torch
        from smplx.lbs import batch_rigid_transform, batch_rodrigues

        device = self.model.v_template.device
        dtype = torch.float32
        vertices_t = torch.as_tensor(vertices, dtype=dtype, device=device)
        joints_t = torch.as_tensor(joints, dtype=dtype, device=device)

        if body_pose is None:
            body_pose = torch.zeros(1, 21 * 3, dtype=dtype, device=device)
        body_pose = torch.as_tensor(body_pose, dtype=dtype, device=device).reshape(
            1, 21, 3
        )
        if global_orient is None:
            global_orient = torch.zeros(1, 3, dtype=dtype, device=device)
        global_orient = torch.as_tensor(
            global_orient, dtype=dtype, device=device
        ).reshape(1, 1, 3)
        zeros = torch.zeros(1, 1, 3, dtype=dtype, device=device)
        hands = torch.zeros(1, 15, 3, dtype=dtype, device=device)
        full_pose = torch.cat(
            [
                global_orient,
                body_pose,
                zeros,  # jaw
                zeros,  # left eye
                zeros,  # right eye
                hands,  # left hand
                hands,  # right hand
            ],
            dim=1,
        ).reshape(1, -1)
        full_pose = full_pose + self.model.pose_mean.to(device=device, dtype=dtype)

        rotations = batch_rodrigues(full_pose.reshape(-1, 3)).view(1, -1, 3, 3)
        identity = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rotations[:, 1:] - identity).reshape(1, -1)
        offsets = (
            pose_feature
            @ self.model.posedirs.to(device=device, dtype=dtype)
        ).view(1, -1, 3)
        posed_input = vertices_t.unsqueeze(0) + offsets

        posed_joints, transforms = batch_rigid_transform(
            rotations,
            joints_t.unsqueeze(0),
            self.model.parents.to(device=device),
        )
        skin = torch.matmul(
            self.model.lbs_weights.to(device=device, dtype=dtype).unsqueeze(0),
            transforms.view(1, -1, 16),
        ).view(1, -1, 4, 4)
        homogeneous = torch.cat(
            [posed_input, torch.ones(1, posed_input.shape[1], 1, device=device)],
            dim=2,
        )
        posed = torch.matmul(skin, homogeneous.unsqueeze(-1))[:, :, :3, 0]
        return (
            posed[0].detach().cpu().numpy().astype(np.float32),
            posed_joints[0].detach().cpu().numpy().astype(np.float32),
        )


def pose_is_identity(body_pose, global_orient):
    """True when both pose inputs are exactly zero (the canonical T-pose)."""
    for value in (body_pose, global_orient):
        if value is None:
            continue
        if hasattr(value, "numel") and int(value.numel()) == 0:
            continue
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        array = np.asarray(value)
        if array.size and float(np.abs(array).max()) != 0.0:
            return False
    return True
