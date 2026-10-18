#!/usr/bin/env python3
"""Headless preview renderer for the OBJ files produced by smplx_tpose_server.py.

Read-only utility: loads an OBJ, frames it with the same rule as the viewer's
`frame()` helper, and renders a shaded PNG with pytorch3d. It never modifies the
installed pytorch3d / torch stack.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, look_at_view_transform, TexturesVertex,
)

DEVICE = "cuda:0"
COLOR = torch.tensor([0.62, 0.72, 0.88], device=DEVICE)


def render(obj_path: Path, out_path: Path, azim=0.0, elev=6.0, size=900):
    meshes = load_objs_as_meshes([str(obj_path)], device=DEVICE)
    verts = meshes.verts_packed()
    lo, hi = verts.min(0).values, verts.max(0).values
    center = (lo + hi) / 2.0
    height = float(hi[1] - lo[1])

    meshes = meshes.clone()
    meshes.textures = TexturesVertex(
        verts_features=torch.ones_like(meshes.verts_packed())[None] * COLOR[None, None]
    )

    dist = height * 2.05
    rot, trans = look_at_view_transform(
        dist=dist, elev=elev, azim=azim,
        at=((float(center[0]), float(center[1]), float(center[2])),),
    )
    cams = FoVPerspectiveCameras(device=DEVICE, R=rot, T=trans, fov=38)
    rasterizer = MeshRasterizer(
        cameras=cams,
        raster_settings=RasterizationSettings(image_size=size, blur_radius=0, faces_per_pixel=1),
    )
    lights = PointLights(device=DEVICE, location=[[3.0, 4.0, -4.0]])
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(device=DEVICE, cameras=cams, lights=lights),
    )
    image = renderer(meshes)
    rgba = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # composite over a dark background like the web UI
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.array([17, 19, 24], dtype=np.float32)
    out = rgb * alpha + bg * (1 - alpha)
    Image.fromarray(out.astype(np.uint8)).save(out_path)
    print(f"{out_path}  height={height:.3f}m  bbox={lo.tolist()} .. {hi.tolist()}")


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]),
           azim=float(sys.argv[3]) if len(sys.argv) > 3 else 0.0,
           elev=float(sys.argv[4]) if len(sys.argv) > 4 else 6.0)
