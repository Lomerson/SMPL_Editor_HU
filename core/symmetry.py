"""Left/right symmetry map for the body editor.

Head, neck and spine are deliberately absent: they have no mirrored partner.
The indices are SMPL-X body joint/segment child indices (1..21).
"""
from __future__ import annotations

import numpy as np

SYMMETRY_SEGMENTS = {
    1: 2,
    2: 1,
    4: 5,
    5: 4,
    7: 8,
    8: 7,
    10: 11,
    11: 10,
    13: 14,
    14: 13,
    16: 17,
    17: 16,
    18: 19,
    19: 18,
    20: 21,
    21: 20,
}

SYMMETRY_JOINTS = dict(SYMMETRY_SEGMENTS)

EXCLUDED_SEGMENTS = frozenset({3, 6, 9, 12, 15})
EXCLUDED_JOINTS = frozenset({3, 6, 9, 12, 15})


def mirror_segment(child: int) -> int | None:
    return SYMMETRY_SEGMENTS.get(int(child))


def mirror_joint(joint: int) -> int | None:
    return SYMMETRY_JOINTS.get(int(joint))


def mirror_target(target) -> list[float]:
    """Reflect a model-space point across the YZ plane (X -> -X)."""
    values = np.asarray(target, dtype=np.float64).reshape(3)
    return [-float(values[0]), float(values[1]), float(values[2])]


def segment_targets(child: int, symmetric: bool) -> list[int]:
    targets = [int(child)]
    mirror = mirror_segment(child) if symmetric else None
    if mirror is not None and mirror != int(child):
        targets.append(mirror)
    return targets


def joint_targets(joint: int, symmetric: bool) -> list[int]:
    targets = [int(joint)]
    mirror = mirror_joint(joint) if symmetric else None
    if mirror is not None and mirror != int(joint):
        targets.append(mirror)
    return targets
