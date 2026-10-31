"""Export a Blender shape key as a SMPL-X per-vertex displacement.

Usage (from the project root):

    blender avatar.blend --background \
        --python blender/export_displacement.py -- /tmp/avatar_displacement.npy

The mesh must be the imported ``avatar_<gender>.glb`` body: 10,475 vertices,
unchanged topology, and a shape key (usually the last one) holding the sculpt.
The basis is the first shape key ("Basis").  Coordinates are exported in the
mesh's local/rest space, which is exactly the order the web viewer expects.
"""
import sys

import bpy
import numpy as np


def _output_path():
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if args:
            return args[0]
    raise SystemExit("pass an output path after --: ... -- /tmp/displacement.npy")


def main():
    path = _output_path()
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise SystemExit("select the imported SMPL-X mesh before running")
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
    if keys is None or len(keys) < 2:
        raise SystemExit("the mesh has no sculpt shape key (need Basis + sculpt)")
    basis, target = keys[0], keys[-1]
    if len(basis.data) != len(target.data):
        raise SystemExit("shape key vertex counts differ")

    count = len(basis.data)
    displacement = np.empty((count, 3), dtype=np.float32)
    for index in range(count):
        displacement[index] = target.data[index].co - basis.data[index].co
    np.save(path, displacement)
    print(
        f"wrote {path}: {count} vertices, "
        f"max displacement {float(np.abs(displacement).max()):.5f} m"
    )


if __name__ == "__main__":
    main()
