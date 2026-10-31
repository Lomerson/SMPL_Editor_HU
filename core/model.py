"""SMPL-X model state: loading, device placement, pose/orient caches, segments."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from core import detector, export, geometry
from core.config import DEVICE_CHOICES, GENDERS
from core.fitting import init as fitting_init

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

    def update_segment(self, gender: str, child: int, length: float, radii: list[float]):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if child < 1 or child > 21:
            raise ValueError("Invalid body segment")
        if not 2 <= len(radii) <= 12 or any(not 0.3 <= value <= 1.5 for value in radii):
            raise ValueError("Radius controls must contain 2-12 values between 0.3 and 1.5")
        with self.lock:
            if child in (4, 5, 7, 8, 18, 19, 20, 21):
                stored_length = float(length)
            else:
                # Non-limb lengths are owned by the reference alignment: a girth
                # edit must neither reject the request nor silently reset them.
                stored_length = float(
                    self._segments(gender).get(str(child), {}).get("length", 1.0)
                )
            self._segments(gender)[str(child)] = {
                "length": stored_length,
                "radii": [float(value) for value in radii],
            }
            self._invalidate_geometry(gender)

    def reset_segment(self, gender: str, child: int, keep_length: bool = False):
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
        with self.lock:
            entry = self._segments(gender).get(str(child))
            if keep_length and entry is not None:
                entry["radii"] = [1.0, 1.0, 1.0]
            else:
                self._segments(gender).pop(str(child), None)
            self._invalidate_geometry(gender)

    def reset(self, gender: str):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        with self.lock:
            model = self._model(gender)
            import torch
            self.poses[gender] = torch.zeros(
                (1, model.NUM_BODY_JOINTS * 3), dtype=torch.float32, device=self.device
            )
            self.global_orients.pop(gender, None)
            self.segment_configs.pop(gender, None)
            self.outputs.pop(gender, None)
            self._invalidate_geometry(gender)

    def pose_joint(self, gender: str, joint: int, target: list[float], full_body: bool,
                   iterations: int | None = None):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if joint < 1 or joint > 21:
            raise ValueError("Only body joints 1-21 are editable in this version")
        import torch
        with self.lock:
            model = self._model(gender)
            base = self._pose(gender, model).clone()
            parents = model.parents.cpu().tolist()
            if full_body:
                chain = []
                cursor = joint
                while cursor > 0:
                    chain.append(cursor)
                    cursor = parents[cursor]
            else:
                # A joint's origin is attached to its parent in SMPL-X. To move
                # the selected point while keeping upstream points fixed, rotate
                # its direct parent; the selected point and its child subtree
                # then move together as one local branch.
                parent = parents[joint]
                if parent <= 0:
                    raise ValueError("This root-adjacent joint requires full-body IK")
                chain = [parent]
            variables = torch.nn.Parameter(base.clone())
            target_tensor = torch.tensor(
                target, dtype=torch.float32, device=self.device
            ).view(1, 3)
            # The visible skeleton is the *length-scaled* one (segment editor), so
            # the solver has to compare against that same joint, not the raw one.
            factors = np.ones(len(parents), dtype=np.float32)
            segment_config = self._segments(gender)
            for index in range(1, min(22, len(parents))):
                factors[index] = float(
                    segment_config.get(str(index), {}).get("length", 1.0)
                )
            lengths = torch.tensor(factors, dtype=torch.float32, device=self.device)
            chain_to_joint = []
            cursor = joint
            while cursor > 0:
                chain_to_joint.append(cursor)
                cursor = parents[cursor]
            chain_to_joint.reverse()

            def scaled_position(output):
                joints = output.joints[0]
                position = joints[0]
                for node in chain_to_joint:
                    parent_index = parents[node]
                    position = position + (joints[node] - joints[parent_index]) * lengths[node]
                return position

            optimizer = torch.optim.Adam([variables], lr=0.035)
            editable = torch.zeros_like(variables)
            for node in chain:
                editable[:, (node - 1) * 3:node * 3] = 1
            global_orient = self._orient(gender, model)
            for _ in range(iterations if iterations is not None else (36 if full_body else 24)):
                optimizer.zero_grad(set_to_none=True)
                pose = variables * editable + base * (1 - editable)
                output = model(body_pose=pose, global_orient=global_orient)
                loss = (scaled_position(output) - target_tensor[0]).square().sum()
                loss = loss + 0.003 * ((pose - base) * editable).square().mean()
                loss.backward()
                optimizer.step()
            self.poses[gender] = (variables.detach() * editable + base * (1 - editable))
            with torch.no_grad():
                self.outputs[gender] = model(
                    body_pose=self.poses[gender], global_orient=global_orient
                )
            self._invalidate_geometry(gender)

    # ------------------------------------------------------------------
    # Delegates to core.geometry / core.export / core.detector / core.fitting
    # ------------------------------------------------------------------
    def _geometry(self, gender: str):
        return geometry.geometry(self, gender)

    def _invalidate_geometry(self, gender: str):
        geometry.invalidate(self, gender)

    def obj(self, gender: str) -> Path:
        return export.export_obj(self, gender)

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
                                z_weight: float = 3.0) -> dict:
        return fitting_init.solve_reference_lengths(
            self, gender, pixels_per_meter, eye_midpoint, joints, z_weight=z_weight
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

