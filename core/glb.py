"""Minimal glTF 2.0 binary export with a real SMPL-X skin.

``trimesh.Trimesh.export('glb')`` writes a plain mesh; Blender then has no
armature, no vertex groups and no way to keep the rig while sculpting.  This
module writes the contract the rest of the pipeline needs:

* one mesh (10,475 vertices / 20,908 triangles) in rest T-pose;
* a 55-joint skin with ``JOINTS_0`` / ``WEIGHTS_0`` (top four, sum = 1);
* ``inverseBindMatrices`` matching the edited rest joints;
* a joint node hierarchy with joint translations, so Blender imports an
  Armature + Armature modifier + Vertex Groups;
* ``avatar_<gender>_params.json`` with the identity edits (segments, head,
  betas/pose placeholders, joint positions).

No third-party glTF dependency is required.
"""
from __future__ import annotations

import copy
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from core.config import GENDERS, JOINT_NAMES

SCHEMA = "smplx-avatar/1"
GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT = 5126
UNSIGNED_BYTE = 5121
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def _numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=dtype)


def _align4(data: bytearray):
    data.extend(b"\x00" * ((4 - len(data) % 4) % 4))


def _normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _top_four(weights):
    order = np.argsort(weights, axis=1)[:, -4:][:, ::-1]
    values = np.take_along_axis(weights, order, axis=1).astype(np.float32)
    total = values.sum(axis=1, keepdims=True)
    values = values / np.maximum(total, 1e-12)
    return order.astype(np.uint8), values


class _Builder:
    def __init__(self):
        self.data = bytearray()
        self.buffer_views = []
        self.accessors = []

    def add(self, array, component_type, type_name, target=None, minmax=False):
        array = np.ascontiguousarray(array)
        _align4(self.data)
        offset = len(self.data)
        self.data.extend(array.tobytes(order="C"))
        self.buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": int(array.nbytes),
                **({"target": target} if target is not None else {}),
            }
        )
        count = int(array.shape[0]) if array.ndim else 1
        accessor = {
            "bufferView": len(self.buffer_views) - 1,
            "byteOffset": 0,
            "componentType": component_type,
            "count": count,
            "type": type_name,
        }
        if minmax:
            accessor["min"] = array.min(axis=0).tolist()
            accessor["max"] = array.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _params(cache, gender):
    model, vertices, joints, parents = cache.rest_geometry(gender)
    return {
        "schema": SCHEMA,
        "gender": gender,
        "betas": [0.0] * 10,
        "body_pose": [0.0] * 63,
        "global_orient": [0.0] * 3,
        "segments": copy.deepcopy(cache._segments(gender)),
        "head": cache.head(gender),
        "skeleton": {
            "joints": np.asarray(joints, dtype=np.float32).tolist(),
            "parents": np.asarray(parents, dtype=np.int64).tolist(),
            "names": list(JOINT_NAMES),
        },
        "eye_centers": {
            "left": np.asarray(joints[23], dtype=np.float32).tolist(),
            "right": np.asarray(joints[24], dtype=np.float32).tolist(),
        },
        "displacement": (
            None
            if cache.displacement(gender) is None
            else {
                "file": f"avatar_{gender}_displacement.npy",
                "shape": [int(cache.displacement(gender).shape[0]), 3],
                "dtype": "float32",
                "max_abs_m": float(np.abs(cache.displacement(gender)).max()),
            }
        ),
        "coordinate_system": "glTF Y-up; rest T-pose",
        "tpose": True,
        "has_skin": True,
        "num_vertices": int(len(vertices)),
        "num_joints": int(len(parents)),
        "notes": "body_pose/global_orient are zero and are not baked into the glb; "
                 "the glb is the canonical rest identity.",
    }


def build_glb_bytes(cache, gender: str) -> bytes:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    model, vertices, joints, parents = cache.rest_geometry(gender)
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    faces = np.asarray(model.faces, dtype=np.int64)
    if len(joints) != 55:
        raise ValueError(f"expected 55 joints, got {len(joints)}")

    weights = _numpy(model.lbs_weights, np.float32)
    joints_u8, weights_top4 = _top_four(weights)
    normals = _normals(vertices, faces)
    indices = faces.reshape(-1)
    index_type = UNSIGNED_SHORT if int(indices.max()) < 65536 else UNSIGNED_INT
    indices = indices.astype(np.uint16 if index_type == UNSIGNED_SHORT else np.uint32)

    builder = _Builder()
    position_accessor = builder.add(
        vertices, FLOAT, "VEC3", target=ARRAY_BUFFER, minmax=True
    )
    normal_accessor = builder.add(
        normals, FLOAT, "VEC3", target=ARRAY_BUFFER
    )
    joints_accessor = builder.add(
        joints_u8, UNSIGNED_BYTE, "VEC4", target=ARRAY_BUFFER
    )
    weights_accessor = builder.add(
        weights_top4, FLOAT, "VEC4", target=ARRAY_BUFFER
    )
    index_accessor = builder.add(
        indices, index_type, "SCALAR", target=ELEMENT_ARRAY_BUFFER
    )

    inverse_bind = np.zeros((len(joints), 4, 4), dtype=np.float32)
    for index, joint in enumerate(joints):
        inverse_bind[index] = np.eye(4, dtype=np.float32)
        inverse_bind[index, :3, 3] = -joint
    # glTF matrices are column-major; transpose before flattening.
    ibm_accessor = builder.add(
        inverse_bind.transpose(0, 2, 1).reshape(len(joints), 16),
        FLOAT,
        "MAT4",
    )

    children = [[] for _ in range(len(parents))]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(int(child))
    nodes = []
    for index, joint in enumerate(joints):
        parent = int(parents[index])
        translation = joint if parent < 0 else joint - joints[parent]
        node = {
            "name": JOINT_NAMES[index] if index < len(JOINT_NAMES) else f"joint_{index}",
            "translation": np.asarray(translation, dtype=np.float32).tolist(),
        }
        if children[index]:
            node["children"] = children[index]
        nodes.append(node)

    armature_index = len(nodes)
    mesh_index = armature_index + 1
    nodes.append({"name": "SMPL-X_Armature", "children": [0]})
    nodes.append({"name": "SMPL-X_Body", "mesh": 0, "skin": 0})

    params = _params(cache, gender)
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "smplx_tpose_viewer_package/core/glb.py",
            "extras": {
                "schema": SCHEMA,
                "gender": gender,
                "head": params["head"],
                "segments": params["segments"],
            },
        },
        "scene": 0,
        "scenes": [{"nodes": [armature_index, mesh_index]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "SMPL-X_Body",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "JOINTS_0": joints_accessor,
                            "WEIGHTS_0": weights_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
                "extras": {
                    "gender": gender,
                    "num_vertices": int(len(vertices)),
                    "num_triangles": int(len(faces)),
                },
            }
        ],
        "skins": [
            {
                "name": "SMPL-X_Skin",
                "joints": list(range(len(joints))),
                "inverseBindMatrices": ibm_accessor,
                "skeleton": 0,
            }
        ],
        "materials": [
            {
                "name": "SMPL-X_Skin",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.78, 0.70, 0.62, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.72,
                },
                "doubleSided": True,
            }
        ],
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
    }

    _align4(builder.data)
    gltf["buffers"] = [{"byteLength": int(len(builder.data))}]
    json_bytes = json.dumps(
        gltf, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(builder.data)
    bin_bytes += b"\x00" * ((4 - len(bin_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
    json_header = struct.pack("<II", len(json_bytes), JSON_CHUNK)
    bin_header = struct.pack("<II", len(bin_bytes), BIN_CHUNK)
    return header + json_header + json_bytes + bin_header + bin_bytes


def export_glb(cache, gender: str) -> Path:
    with cache.lock:
        data = build_glb_bytes(cache, gender)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache.cache_dir / f"avatar_{gender}.glb"
    fd, temporary = tempfile.mkstemp(suffix=".glb", dir=cache.cache_dir)
    os.close(fd)
    try:
        Path(temporary).write_bytes(data)
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def export_params(cache, gender: str) -> Path:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache.cache_dir / f"avatar_{gender}_params.json"
    with cache.lock:
        payload = _params(cache, gender)
    fd, temporary = tempfile.mkstemp(suffix=".json", dir=cache.cache_dir)
    os.close(fd)
    try:
        Path(temporary).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    displacement = cache.displacement(gender)
    if displacement is not None:
        export_displacement(cache, gender)
    return target


def export_displacement(cache, gender: str) -> Path:
    displacement = cache.displacement(gender)
    if displacement is None:
        raise FileNotFoundError("no displacement imported")
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache.cache_dir / f"avatar_{gender}_displacement.npy"
    fd, temporary = tempfile.mkstemp(suffix=".npy", dir=cache.cache_dir)
    os.close(fd)
    try:
        np.save(temporary, np.asarray(displacement, dtype=np.float32))
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target
