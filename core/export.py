"""OBJ export for the current (warped) T-pose mesh."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from core.config import GENDERS

def export_obj(cache, gender: str) -> Path:
    if gender not in GENDERS or not cache.available(gender):
        raise FileNotFoundError(gender)
    target = cache.cache_dir / f"smplx_{gender}_tpose.obj"
    with cache.lock:
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        import trimesh
        model, vertices, _joints, _parents = cache._geometry(gender)
        mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        fd, temporary = tempfile.mkstemp(suffix=".obj", dir=cache.cache_dir)
        os.close(fd)
        try:
            mesh.export(temporary)
            Path(temporary).replace(target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target
