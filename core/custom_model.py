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
        """Minimal forward used by the pose solver.

        Geometry normally goes through RestDeformer; this callable keeps
        ``pose_joint`` working for uploaded models without touching the
        standard model's template.
        """
        import types
        import torch
        from core.deform import RestDeformer

        deformer = RestDeformer(self, {})
        vertices, joints = deformer.pose(
            self.v_template.detach().cpu().numpy(),
            self.rest_joints.detach().cpu().numpy(),
            body_pose,
            global_orient,
        )
        return types.SimpleNamespace(
            vertices=torch.as_tensor(vertices, dtype=torch.float32, device=self.v_template.device).unsqueeze(0),
            joints=torch.as_tensor(joints, dtype=torch.float32, device=self.v_template.device).unsqueeze(0),
        )


def build_uploaded_model(base, payload: bytes) -> UploadedModel:
    """Parse and validate a fitted-model NPZ, keeping its native T-pose arrays."""
    try:
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ValueError("无法读取 .npz 文件（必须是有效的 NumPy NPZ）") from exc
    required = ("faces", "skinning_weights")
    missing = [name for name in required if name not in archive.files]
    if missing:
        raise ValueError("模型缺少字段: " + ", ".join(missing))

    vertices_key = "native_vertices" if "native_vertices" in archive.files else "vertices"
    joints_key = "native_joints" if "native_joints" in archive.files else "joints"
    if vertices_key not in archive.files or joints_key not in archive.files:
        raise ValueError("模型缺少 vertices/native_vertices 或 joints/native_joints")

    vertices = np.asarray(archive[vertices_key], dtype=np.float32)
    joints = np.asarray(archive[joints_key], dtype=np.float32)
    faces = np.asarray(archive["faces"], dtype=np.int64)
    weights = np.asarray(archive["skinning_weights"], dtype=np.float32)
    if joints.ndim == 2 and len(joints) >= 55:
        joints = joints[:55]
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
