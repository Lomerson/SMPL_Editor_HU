"""Static SMPL-X-topology model uploaded by the viewer.

The fitted ``.npz`` files produced by the reference pipeline are not standard
SMPL-X model archives: they contain a T-pose template, joints and skinning
weights, but no ``shapedirs``/``posedirs`` of their own.  The viewer therefore
keeps a standard SMPL-X model for pose directions and parents, and wraps the
uploaded topology/weights/joints in this light adapter.
"""
from __future__ import annotations

import io
import numpy as np


class UploadedModel:
    """A SMPL-X-topology static/rest model backed by an uploaded NPZ."""

    is_custom = True

    def __init__(self, base, vertices, faces, joints, weights):
        import torch

        self.base = base
        device = base.v_template.device
        self.v_template = torch.as_tensor(
            np.asarray(vertices, dtype=np.float32), dtype=torch.float32, device=device
        )
        self.faces = torch.as_tensor(
            np.asarray(faces, dtype=np.int64), dtype=torch.long, device=device
        )
        self.lbs_weights = torch.as_tensor(
            np.asarray(weights, dtype=np.float32), dtype=torch.float32, device=device
        )
        self.rest_joints = torch.as_tensor(
            np.asarray(joints, dtype=np.float32), dtype=torch.float32, device=device
        )
        self.parents = base.parents
        self.pose_mean = base.pose_mean
        self.posedirs = base.posedirs
        self.J_regressor = base.J_regressor
        self.betas = base.betas
        self.NUM_BODY_JOINTS = int(base.NUM_BODY_JOINTS)
        self.NUM_JOINTS = int(base.NUM_JOINTS)
        self.gender = getattr(base, "gender", "")

    def __getattr__(self, name):
        # Only called for attributes not defined on the adapter.
        return getattr(self.base, name)

    def __call__(self, body_pose=None, global_orient=None, **kwargs):
        """Differentiable SMPL-X-style LBS for the uploaded rest surface.

        The pose solver needs gradients with respect to ``body_pose``; the
        regular geometry path still uses RestDeformer, but this keeps the
        standard editor controls working on uploaded models.
        """
        import types
        import torch
        from smplx.lbs import batch_rigid_transform, batch_rodrigues

        device = self.v_template.device
        dtype = torch.float32
        if body_pose is None:
            body_pose = torch.zeros(1, 21 * 3, dtype=dtype, device=device)
        if not torch.is_tensor(body_pose):
            body_pose = torch.as_tensor(body_pose, dtype=dtype, device=device)
        body_pose = body_pose.reshape(1, 21, 3)
        if global_orient is None:
            global_orient = torch.zeros(1, 3, dtype=dtype, device=device)
        if not torch.is_tensor(global_orient):
            global_orient = torch.as_tensor(global_orient, dtype=dtype, device=device)
        global_orient = global_orient.reshape(1, 1, 3)
        zeros = torch.zeros(1, 1, 3, dtype=dtype, device=device)
        hands = torch.zeros(1, 15, 3, dtype=dtype, device=device)
        full_pose = torch.cat(
            [global_orient, body_pose, zeros, zeros, zeros, hands, hands], dim=1
        ).reshape(1, -1)
        full_pose = full_pose + self.pose_mean.to(device=device, dtype=dtype)
        rotations = batch_rodrigues(full_pose.reshape(-1, 3)).view(1, -1, 3, 3)
        identity = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rotations[:, 1:] - identity).reshape(1, -1)
        offsets = (
            pose_feature @ self.posedirs.to(device=device, dtype=dtype)
        ).view(1, -1, 3)
        posed_input = self.v_template.unsqueeze(0) + offsets
        posed_joints, transforms = batch_rigid_transform(
            rotations, self.rest_joints.unsqueeze(0), self.parents.to(device=device)
        )
        skin = torch.matmul(
            self.lbs_weights.to(device=device, dtype=dtype).unsqueeze(0),
            transforms.view(1, -1, 16),
        ).view(1, -1, 4, 4)
        homogeneous = torch.cat(
            [posed_input, torch.ones(1, posed_input.shape[1], 1, device=device)],
            dim=2,
        )
        posed = torch.matmul(skin, homogeneous.unsqueeze(-1))[:, :, :3, 0]
        return types.SimpleNamespace(vertices=posed, joints=posed_joints)


def _transform(points, matrix):
    points = np.asarray(points, dtype=np.float32)
    if matrix is None:
        return points
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        return points
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ matrix.T)[:, :3].astype(np.float32)


def _final_vertices(archive, base):
    if "vertices" in archive.files:
        return np.asarray(archive["vertices"], dtype=np.float32)
    if "native_vertices" not in archive.files:
        raise ValueError("模型缺少 vertices/native_vertices")
    points = np.asarray(archive["native_vertices"], dtype=np.float32)
    if "inherited_reference_delta" in archive.files:
        points = points + np.asarray(
            archive["inherited_reference_delta"], dtype=np.float32
        )
    points = _transform(
        points,
        archive["native_to_world"] if "native_to_world" in archive.files else None,
    )
    if "two_view_delta" in archive.files:
        points = points + np.asarray(archive["two_view_delta"], dtype=np.float32)
    return points.astype(np.float32)


def _final_joints(archive):
    if "joints" in archive.files:
        joints = np.asarray(archive["joints"], dtype=np.float32)
    elif "native_joints" in archive.files:
        joints = _transform(
            archive["native_joints"],
            archive["native_to_world"] if "native_to_world" in archive.files else None,
        )
    else:
        raise ValueError("模型缺少 joints/native_joints")
    if joints.ndim == 2 and len(joints) >= 55:
        joints = joints[:55]
    return np.asarray(joints, dtype=np.float32)


def build_uploaded_model(base, payload: bytes) -> UploadedModel:
    """Parse and validate a fitted-model NPZ.

    Prefer the baked final surface (``vertices``/``joints``), which is exactly
    what the folder's OBJ/GLB contains.  Older files without those arrays are
    reconstructed from native vertices + inherited delta + native_to_world +
    two_view_delta.
    """
    try:
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ValueError("无法读取 .npz 文件（必须是有效的 NumPy NPZ）") from exc
    required = ("faces", "skinning_weights")
    missing = [name for name in required if name not in archive.files]
    if missing:
        raise ValueError("模型缺少字段: " + ", ".join(missing))

    vertices = _final_vertices(archive, base)
    joints = _final_joints(archive)
    faces = np.asarray(archive["faces"], dtype=np.int64)
    weights = np.asarray(archive["skinning_weights"], dtype=np.float32)
    expected_vertices = int(base.v_template.shape[0])
    expected_joints = int(base.parents.shape[0])
    if vertices.shape != (expected_vertices, 3):
        raise ValueError(
            f"顶点数必须是 {expected_vertices}（SMPL-X 拓扑），实际为 {vertices.shape}"
        )
    if faces.shape[1:] != (3,) or not len(faces):
        raise ValueError("faces 必须是 (N, 3)")
    if int(faces.max()) >= expected_vertices:
        raise ValueError("faces 索引超出顶点范围")
    if joints.shape != (expected_joints, 3):
        raise ValueError(f"关节数必须是 {expected_joints}，实际为 {joints.shape}")
    if weights.shape != (expected_vertices, expected_joints):
        raise ValueError(
            f"skinning_weights 必须是 ({expected_vertices}, {expected_joints})"
        )
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("模型包含非有限数值")
    sums = weights.sum(axis=1)
    if np.any(sums <= 1e-8):
        raise ValueError("skinning_weights 存在全零顶点")
    weights = weights / sums[:, None]
    return UploadedModel(base, vertices, faces, joints, weights)
