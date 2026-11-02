"""SMPL-X model state: loading, device placement, pose/orient caches, segments."""
from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np

from core import detector, export, fourview, geometry, glb
from core.config import DEVICE_CHOICES, GENDERS, JOINT_NAMES
from core.deform import LENGTH_MAX, LENGTH_MIN
from core.fitting import init as fitting_init
from core.head import HEAD_DEFAULTS, normalize_head_params
from core.symmetry import joint_targets, mirror_target, segment_targets

def resolve_device(requested: str = "auto") -> tuple[str, str]:
    """Decide where SMPL-X inference runs, before any weights are loaded.

    ``auto`` keeps CUDA only when the driver, the CUDA-enabled PyTorch build and
    a real tensor operation all succeed; anything else falls back to the CPU,
    which is the behaviour of the original code.  The probe is one tiny
    allocation plus one elementwise op, so a broken driver surfaces here at
    startup instead of inside the first request.

    Returns ``(device, reason)``; ``reason`` is meant for the startup log.
    """
    requested = (requested or "auto").lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {requested!r}; expected one of {DEVICE_CHOICES}")
    if requested == "cpu":
        return "cpu", "requested on the command line"

    import torch

    if not torch.cuda.is_available():
        return "cpu", (
            f"torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, CUDA build {torch.version.cuda})"
        )

    try:
        probe = torch.zeros(8, dtype=torch.float32, device="cuda") + 1.0
        torch.cuda.synchronize()
        if not bool(torch.isfinite(probe).all()):
            raise RuntimeError("probe produced non-finite values")
    except Exception as exc:
        return "cpu", f"CUDA probe failed: {type(exc).__name__}: {exc}"

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return "cuda", f"{name} (sm_{major}{minor}, CUDA {torch.version.cuda})"
    except Exception:
        return "cuda", f"CUDA device (CUDA {torch.version.cuda})"


class ModelCache:
    def __init__(self, model_dir: Path, cache_dir: Path, device: str = "cpu",
                 detector_dir: str = ""):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.detector_dir = detector_dir
        self._detector = None
        self.device = device if device in ("cpu", "cuda") else "cpu"
        self.lock = threading.Lock()
        self.poses = {}
        self.global_orients = {}
        self.models = {}
        self.outputs = {}
        self.segment_configs = {}
        self.segment_sequences = {}
        self.head_configs = {}
        self.displacements = {}
        self.histories = {}
        self._sequence = 0
        self.geometry_cache = {}

    def source(self, gender: str) -> Path:
        return self.model_dir / f"SMPLX_{gender.upper()}.npz"

    def available(self, gender: str) -> bool:
        return self.source(gender).is_file()

    def _model(self, gender: str):
        if gender not in self.models:
            import smplx
            model = smplx.create(
                str(self.source(gender)), model_type="smplx", gender=gender,
                use_pca=False, flat_hand_mean=True, batch_size=1,
            )
            self.models[gender] = self._place_model(model)
        return self.models[gender]

    def _place_model(self, model):
        """Move a newly created model onto the selected device.

        A failed transfer (driver/architecture mismatch, out of memory) must not
        take the viewer down, so the session retreats to the CPU -- the original
        behaviour -- and reports why.
        """
        if self.device == "cpu":
            return model
        try:
            return model.to(self.device)
        except Exception as exc:
            print(
                f"[device] could not move SMPL-X to {self.device}: "
                f"{type(exc).__name__}: {exc} -> falling back to cpu",
                flush=True,
            )
            # Anything already cached has to come back with us, otherwise a CUDA
            # pose tensor would meet a CPU model on the next request.
            for cached in self.models.values():
                try:
                    cached.to("cpu")
                except Exception:
                    pass
            self.poses.clear()
            self.outputs.clear()
            self.geometry_cache.clear()
            self.device = "cpu"
            try:
                return model.to("cpu")
            except Exception:
                return model

    def _pose(self, gender: str, model):
        import torch
        if gender not in self.poses:
            self.poses[gender] = torch.zeros(
                (1, model.NUM_BODY_JOINTS * 3),
                dtype=torch.float32,
                device=self.device,
            )
        return self.poses[gender]

    def _orient(self, gender: str, model):
        import torch
        if gender not in self.global_orients:
            self.global_orients[gender] = torch.zeros(
                (1, 3), dtype=torch.float32, device=self.device
            )
        return self.global_orients[gender]

    def _output(self, gender: str):
        import torch
        if gender in self.outputs:
            return self._model(gender), self.outputs[gender]
        model = self._model(gender)
        with torch.no_grad():
            output = model(
                body_pose=self._pose(gender, model),
                global_orient=self._orient(gender, model),
            )
        self.outputs[gender] = output
        return model, output

    def _segments(self, gender: str):
        return self.segment_configs.setdefault(gender, {})

    # ------------------------------------------------------------------
    # undo timeline
    # ------------------------------------------------------------------
    def _snapshot_locked(self, gender: str):
        import torch
        pose = self.poses.get(gender)
        orient = self.global_orients.get(gender)
        displacement = self.displacements.get(gender)
        return {
            "segments": copy.deepcopy(self.segment_configs.get(gender, {})),
            "sequences": dict(self.segment_sequences.get(gender, {})),
            "head": copy.deepcopy(self.head_configs.get(gender, {})),
            "displacement": None if displacement is None else displacement.copy(),
            "pose": None if pose is None else pose.detach().clone(),
            "orient": None if orient is None else orient.detach().clone(),
        }

    def _restore_locked(self, gender: str, snapshot):
        if snapshot["segments"]:
            self.segment_configs[gender] = copy.deepcopy(snapshot["segments"])
        else:
            self.segment_configs.pop(gender, None)
        if snapshot["sequences"]:
            self.segment_sequences[gender] = dict(snapshot["sequences"])
        else:
            self.segment_sequences.pop(gender, None)
        if snapshot["head"]:
            self.head_configs[gender] = copy.deepcopy(snapshot["head"])
        else:
            self.head_configs.pop(gender, None)
        if snapshot["displacement"] is None:
            self.displacements.pop(gender, None)
        else:
            self.displacements[gender] = snapshot["displacement"].copy()
        if snapshot["pose"] is None:
            self.poses.pop(gender, None)
        else:
            self.poses[gender] = snapshot["pose"].detach().clone()
        if snapshot["orient"] is None:
            self.global_orients.pop(gender, None)
        else:
            self.global_orients[gender] = snapshot["orient"].detach().clone()
        self.outputs.pop(gender, None)
        self._invalidate_geometry(gender)

    def _same_snapshot(self, left, right) -> bool:
        import torch
        for key in ("segments", "sequences", "head"):
            if left[key] != right[key]:
                return False
        left_displacement = left["displacement"]
        right_displacement = right["displacement"]
        if (left_displacement is None) != (right_displacement is None):
            return False
        if left_displacement is not None and not np.array_equal(
            left_displacement, right_displacement
        ):
            return False
        for key in ("pose", "orient"):
            left_value = left[key]
            right_value = right[key]
            if (left_value is None) != (right_value is None):
                return False
            if left_value is not None and not bool(torch.equal(left_value, right_value)):
                return False
        return True

    def _ensure_history_locked(self, gender: str):
        if gender not in self.histories:
            self.histories[gender] = {
                "entries": [
                    {
                        "id": 0,
                        "label": "初始状态",
                        "snapshot": self._snapshot_locked(gender),
                    }
                ],
                "cursor": 0,
                "next_id": 1,
            }
        return self.histories[gender]

    def _history_index_locked(self, history):
        for index, entry in enumerate(history["entries"]):
            if entry["id"] == history["cursor"]:
                return index
        history["cursor"] = history["entries"][-1]["id"]
        return len(history["entries"]) - 1

    def _history_state_locked(self, history):
        index = self._history_index_locked(history)
        return {
            "entries": [
                {"id": entry["id"], "label": entry["label"]}
                for entry in history["entries"]
            ],
            "cursor": history["cursor"],
            "can_undo": index > 0,
            "can_redo": index < len(history["entries"]) - 1,
        }

    def _history_begin_locked(self, gender: str):
        self._ensure_history_locked(gender)
        return self._snapshot_locked(gender)

    def _history_commit_locked(self, gender: str, label: str, base):
        current = self._snapshot_locked(gender)
        if self._same_snapshot(base, current):
            return
        history = self._ensure_history_locked(gender)
        index = self._history_index_locked(history)
        del history["entries"][index + 1:]
        entry_id = history["next_id"]
        history["next_id"] += 1
        history["entries"].append(
            {"id": entry_id, "label": label, "snapshot": current}
        )
        history["cursor"] = entry_id
        limit = 60
        if len(history["entries"]) > limit:
            history["entries"] = history["entries"][-limit:]

    def begin_edit(self, gender: str, label: str):
        with self.lock:
            return self._history_begin_locked(gender)

    def commit_edit(self, gender: str, label: str, base):
        with self.lock:
            self._history_commit_locked(gender, label, base)

    def rollback_edit(self, gender: str, base):
        with self.lock:
            self._restore_locked(gender, base)

    def history_state(self, gender: str) -> dict:
        with self.lock:
            history = self._ensure_history_locked(gender)
            return self._history_state_locked(history)

    def undo(self, gender: str) -> dict:
        with self.lock:
            history = self._ensure_history_locked(gender)
            index = self._history_index_locked(history)
            if index <= 0:
                return {
                    "ok": False,
                    "label": None,
                    "history": self._history_state_locked(history),
                }
            current = history["entries"][index]
            entry = history["entries"][index - 1]
            history["cursor"] = entry["id"]
            self._restore_locked(gender, entry["snapshot"])
            return {
                "ok": True,
                "label": current["label"],
                "current_label": entry["label"],
                "history": self._history_state_locked(history),
            }

    def jump_history(self, gender: str, entry_id: int) -> dict:
        with self.lock:
            history = self._ensure_history_locked(gender)
            entry = next(
                (item for item in history["entries"] if item["id"] == int(entry_id)),
                None,
            )
            if entry is None:
                raise ValueError("history entry not found")
            history["cursor"] = entry["id"]
            self._restore_locked(gender, entry["snapshot"])
            return {
                "ok": True,
                "label": entry["label"],
                "history": self._history_state_locked(history),
            }

    def _segment_label(self, gender: str, child: int, symmetric: bool = False,
                       reset: bool = False) -> str:
        parents = self._model(gender).parents.cpu().numpy()
        parent = int(parents[child])
        name = f"{JOINT_NAMES[parent]}→{JOINT_NAMES[child]}"
        if symmetric and child in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21):
            name = f"对称 {name}"
        return ("重置骨段 " if reset else "骨段 ") + name

    def _pose_label(self, gender: str, joint: int, symmetric: bool) -> str:
        name = JOINT_NAMES[joint] if 0 <= joint < len(JOINT_NAMES) else str(joint)
        return ("对称姿态 " if symmetric else "姿态 ") + name

    def head(self, gender: str) -> dict:
        values = dict(HEAD_DEFAULTS)
        values.update(self.head_configs.get(gender, {}))
        return values

    def update_head(self, gender: str, params: dict, record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        values = normalize_head_params(params)
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            try:
                self.head_configs[gender] = values
                self._invalidate_geometry(gender)
            except Exception:
                if record:
                    self._restore_locked(gender, base)
                raise
            if record:
                self._history_commit_locked(gender, "头部编辑", base)

    def reset_head(self, gender: str, record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            self.head_configs.pop(gender, None)
            self._invalidate_geometry(gender)
            if record:
                self._history_commit_locked(gender, "重置头部", base)

    def displacement(self, gender: str):
        return self.displacements.get(gender)

    def update_displacement(self, gender: str, values, record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        array = np.asarray(values, dtype=np.float32)
        expected = int(self._model(gender).v_template.shape[0])
        if array.shape != (expected, 3):
            raise ValueError(
                f"displacement must have shape ({expected}, 3), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("displacement contains non-finite values")
        if float(np.abs(array).max()) > 0.5:
            raise ValueError("displacement magnitude is implausibly large (>0.5 m)")
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            self.displacements[gender] = array.copy()
            self._invalidate_geometry(gender)
            if record:
                self._history_commit_locked(gender, "导入位移", base)

    def reset_displacement(self, gender: str, record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            self.displacements.pop(gender, None)
            self._invalidate_geometry(gender)
            if record:
                self._history_commit_locked(gender, "清除位移", base)

    def _sync_joint_girth(self, gender: str, joint: int, value: float, parents):
        """Give every bone incident to ``joint`` the same endpoint girth.

        Girth lives on the joint, not on a half-open bone: the elbow value is
        shared by the upper arm's 100% and the forearm's 0% control point.
        Keeping the stored arrays in sync means the UI shows the same number
        from either side.
        """
        config = self._segments(gender)
        entry = config.get(str(joint))
        if entry is not None:
            radii = entry.get("radii")
            if isinstance(radii, list) and len(radii) >= 2:
                radii[-1] = float(value)
        for child in range(1, 22):
            if int(parents[child]) != joint:
                continue
            entry = config.get(str(child))
            if entry is None:
                continue
            radii = entry.get("radii")
            if isinstance(radii, list) and len(radii) >= 2:
                radii[0] = float(value)

    def _sync_all_girths(self, gender: str):
        """Re-share every joint's girth after a bulk write (reference fit)."""
        config = self._segments(gender)
        parents = self._model(gender).parents.cpu().numpy()
        sequences = self.segment_sequences.get(gender, {})
        for joint in range(0, 22):
            candidates = []
            entry = config.get(str(joint))
            if entry is not None:
                radii = entry.get("radii")
                if isinstance(radii, list) and len(radii) >= 2:
                    candidates.append((int(sequences.get(str(joint), 0)), 1, radii[-1]))
            for child in range(1, 22):
                if int(parents[child]) != joint:
                    continue
                entry = config.get(str(child))
                if entry is None:
                    continue
                radii = entry.get("radii")
                if isinstance(radii, list) and len(radii) >= 2:
                    candidates.append((int(sequences.get(str(child), 0)), 2, radii[0]))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1]))
            value = float(candidates[-1][2])
            self._sync_joint_girth(gender, joint, value, parents)

    def _apply_update_segment_locked(self, gender: str, child: int, length: float,
                                     radii: list[float]):
        if child == 15:
            # The head segment (neck->head) is owned by the head deformer.  A
            # girth edit must not reset a length written by the reference fit.
            stored_length = float(
                self._segments(gender).get(str(child), {}).get("length", 1.0)
            )
        else:
            stored_length = float(length)
        radii = [float(value) for value in radii]
        self._segments(gender)[str(child)] = {
            "length": stored_length,
            "radii": radii,
        }
        self._sequence += 1
        self.segment_sequences.setdefault(gender, {})[str(child)] = self._sequence
        parents = self._model(gender).parents.cpu().numpy()
        self._sync_joint_girth(gender, int(parents[child]), radii[0], parents)
        self._sync_joint_girth(gender, child, radii[-1], parents)
        self._invalidate_geometry(gender)

    def update_segment(self, gender: str, child: int, length: float,
                       radii: list[float], symmetric: bool = False,
                       record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if child < 1 or child > 21:
            raise ValueError("Invalid body segment")
        try:
            radii = [float(value) for value in radii]
            length = float(length)
        except (TypeError, ValueError) as exc:
            raise ValueError("length and radii must be numbers") from exc
        if not 2 <= len(radii) <= 12 or any(not 0.3 <= value <= 1.5 for value in radii):
            raise ValueError("Radius controls must contain 2-12 values between 0.3 and 1.5")
        if not np.isfinite(length) or not LENGTH_MIN <= length <= LENGTH_MAX:
            raise ValueError(
                f"length must be between {LENGTH_MIN:.2f} and {LENGTH_MAX:.2f}"
            )
        targets = segment_targets(child, symmetric)
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            try:
                for target in targets:
                    self._apply_update_segment_locked(gender, target, length, radii)
            except Exception:
                if record:
                    self._restore_locked(gender, base)
                raise
            if record:
                self._history_commit_locked(
                    gender, self._segment_label(gender, child, symmetric), base
                )

    def reset_segment(self, gender: str, child: int, keep_length: bool = False,
                      symmetric: bool = False, record: bool = True):
        """Reset a segment.

        ``keep_length=False`` restores both the bone length and the cross-section
        (the editor's normal "重置此段").  ``keep_length=True`` only clears the
        cross-section and leaves the length alone -- that is what the button does
        while a reference picture owns the lengths: the fit must survive, the
        girth may be reset.
        """
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if child < 1 or child > 21:
            raise ValueError("Invalid body segment")
        targets = segment_targets(child, symmetric)
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            try:
                parents = self._model(gender).parents.cpu().numpy()
                for target in targets:
                    entry = self._segments(gender).get(str(target))
                    if keep_length and entry is not None:
                        entry["radii"] = [1.0, 1.0, 1.0]
                        # A reset cross-section is a real edit of the shared
                        # joints, unlike deleting the entry (where an adjacent
                        # bone may still own the joint value).
                        self._sync_joint_girth(gender, int(parents[target]), 1.0, parents)
                        self._sync_joint_girth(gender, target, 1.0, parents)
                    else:
                        self._segments(gender).pop(str(target), None)
                        self.segment_sequences.get(gender, {}).pop(str(target), None)
                    self._invalidate_geometry(gender)
            except Exception:
                if record:
                    self._restore_locked(gender, base)
                raise
            if record:
                self._history_commit_locked(
                    gender, self._segment_label(gender, child, symmetric, reset=True), base
                )

    def reset(self, gender: str, record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        with self.lock:
            base = self._history_begin_locked(gender) if record else None
            model = self._model(gender)
            import torch
            self.poses[gender] = torch.zeros(
                (1, model.NUM_BODY_JOINTS * 3), dtype=torch.float32, device=self.device
            )
            self.global_orients.pop(gender, None)
            self.segment_configs.pop(gender, None)
            self.segment_sequences.pop(gender, None)
            self.head_configs.pop(gender, None)
            self.displacements.pop(gender, None)
            self.outputs.pop(gender, None)
            self._invalidate_geometry(gender)
            if record:
                self._history_commit_locked(gender, "重置", base)

    def pose_joint(self, gender: str, joint: int, target: list[float], full_body: bool,
                   iterations: int | None = None, symmetric: bool = False,
                   record: bool = True):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if joint < 1 or joint > 21:
            raise ValueError("Only body joints 1-21 are editable in this version")
        import torch
        with self.lock:
            history_base = self._history_begin_locked(gender) if record else None
            model = self._model(gender)
            base = self._pose(gender, model).clone()
            parents = model.parents.cpu().tolist()
            pairs = []
            for index, node in enumerate(joint_targets(joint, symmetric)):
                pairs.append((node, target if index == 0 else mirror_target(target)))
            chains = []
            chains_to_joint = []
            for node, _target in pairs:
                if full_body:
                    chain = []
                    cursor = node
                    while cursor > 0:
                        chain.append(cursor)
                        cursor = parents[cursor]
                else:
                    # A joint's origin is attached to its parent in SMPL-X. To
                    # move the selected point while keeping upstream points
                    # fixed, rotate its direct parent.
                    parent = parents[node]
                    if parent <= 0:
                        raise ValueError("This root-adjacent joint requires full-body IK")
                    chain = [parent]
                chain_to_joint = []
                cursor = node
                while cursor > 0:
                    chain_to_joint.append(cursor)
                    cursor = parents[cursor]
                chain_to_joint.reverse()
                chains.append(chain)
                chains_to_joint.append(chain_to_joint)

            variables = torch.nn.Parameter(base.clone())
            target_tensor = torch.tensor(
                [pair[1] for pair in pairs], dtype=torch.float32, device=self.device
            ).view(-1, 3)
            # The visible skeleton is the *length-scaled* one (segment editor),
            # so the solver compares against that same joint, not the raw one.
            factors = np.ones(len(parents), dtype=np.float32)
            segment_config = self._segments(gender)
            for index in range(1, min(22, len(parents))):
                factors[index] = float(
                    segment_config.get(str(index), {}).get("length", 1.0)
                )
            lengths = torch.tensor(factors, dtype=torch.float32, device=self.device)

            def scaled_position(output, chain_to_joint):
                joints = output.joints[0]
                position = joints[0]
                for node in chain_to_joint:
                    parent_index = parents[node]
                    position = position + (joints[node] - joints[parent_index]) * lengths[node]
                return position

            optimizer = torch.optim.Adam([variables], lr=0.035)
            editable = torch.zeros_like(variables)
            for chain in chains:
                for node in chain:
                    editable[:, (node - 1) * 3:node * 3] = 1
            global_orient = self._orient(gender, model)
            for _ in range(iterations if iterations is not None else (36 if full_body else 24)):
                optimizer.zero_grad(set_to_none=True)
                pose = variables * editable + base * (1 - editable)
                output = model(body_pose=pose, global_orient=global_orient)
                loss = None
                for index, chain_to_joint in enumerate(chains_to_joint):
                    squared = (
                        scaled_position(output, chain_to_joint) - target_tensor[index]
                    ).square().sum()
                    loss = squared if loss is None else loss + squared
                loss = loss + 0.003 * ((pose - base) * editable).square().mean()
                loss.backward()
                optimizer.step()
            self.poses[gender] = (variables.detach() * editable + base * (1 - editable))
            with torch.no_grad():
                self.outputs[gender] = model(
                    body_pose=self.poses[gender], global_orient=global_orient
                )
            self._invalidate_geometry(gender)
            if record:
                self._history_commit_locked(
                    gender, self._pose_label(gender, joint, symmetric), history_base
                )

    # ------------------------------------------------------------------
    # Delegates to core.geometry / core.export / core.detector / core.fitting
    # ------------------------------------------------------------------
    def _geometry(self, gender: str):
        return geometry.geometry(self, gender)

    def rest_geometry(self, gender: str):
        return geometry.rest_geometry(self, gender)

    def _invalidate_geometry(self, gender: str):
        geometry.invalidate(self, gender)

    def obj(self, gender: str) -> Path:
        return export.export_obj(self, gender)

    def glb(self, gender: str) -> Path:
        return glb.export_glb(self, gender)

    def params(self, gender: str) -> Path:
        return glb.export_params(self, gender)

    def displacement_export(self, gender: str) -> Path:
        return glb.export_displacement(self, gender)

    def fourview(self, gender: str) -> Path:
        return fourview.export_fourview(self, gender)

    def skeleton(self, gender: str) -> dict:
        return geometry.skeleton(self, gender)

    def owners(self, gender: str) -> bytes:
        return geometry.owners(self, gender)

    def _detector_module(self):
        return detector.detector_module(self)

    def reference_eyes(self, image_bytes: bytes) -> dict:
        return detector.reference_eyes(self, image_bytes)

    def solve_reference_lengths(self, gender: str, pixels_per_meter: float,
                                eye_midpoint, joints: dict,
                                z_weight: float = 3.0,
                                editable_bones=None) -> dict:
        return fitting_init.solve_reference_lengths(
            self, gender, pixels_per_meter, eye_midpoint, joints,
            z_weight=z_weight, editable_bones=editable_bones,
        )

    def _reference_residual(self, gender, pixels_per_meter, eye_midpoint, joints):
        return fitting_init.reference_residual(
            self, gender, pixels_per_meter, eye_midpoint, joints
        )

    def _scaled_joints(self, model, body_pose, lengths, global_orient=None):
        return fitting_init.scaled_joints(
            model, body_pose, lengths, global_orient=global_orient
        )

    def refine_pose_to_reference(self, gender: str, pixels_per_meter: float,
                                 eye_midpoint, joints: dict,
                                 full_body: bool | None = None,
                                 iterations: int = 160,
                                 learning_rate: float = 0.02,
                                 reference_z: dict | None = None,
                                 z_weight: float = 20.0) -> dict:
        return fitting_init.refine_pose_to_reference(
            self, gender, pixels_per_meter, eye_midpoint, joints,
            full_body=full_body, iterations=iterations,
            learning_rate=learning_rate, reference_z=reference_z,
            z_weight=z_weight,
        )

    def align_reference(self, gender: str, pixels_per_meter: float,
                        eye_midpoint, joints: dict,
                        full_body: bool | None = None) -> dict:
        return fitting_init.align_reference(
            self, gender, pixels_per_meter, eye_midpoint, joints,
            full_body=full_body,
        )

