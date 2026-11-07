"""HTTP surface for the SMPL-X viewer; the implementation lives in core/."""
from __future__ import annotations

import argparse
import base64
import io
import zipfile
import json
import os
import sys
import tempfile
import threading

import numpy as np
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core import autofit, experiment_assets
from core.access import AccessController
from core.config import DEVICE_CHOICES, GENDERS
from core.model import ModelCache, resolve_device

WEB_ROOT = (Path(__file__).resolve().parent / "web").resolve()
HTML = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
EXPERIMENT_HTML = HTML.replace(
    "</head>",
    '<script>window.__SMPLX_EXPERIMENT__=true;</script></head>',
)
VENDOR_ROOT = (WEB_ROOT / "vendor").resolve()
PICS_PATH = (Path(__file__).resolve().parent / "pics.zip").resolve()
ADMIN_HTML = (WEB_ROOT / "admin.html").read_text(encoding="utf-8")

def handler_factory(cache: ModelCache, access: AccessController | None = None):
    access = access or AccessController.disabled()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, payload, status=HTTPStatus.OK, headers=None):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def send_html(self, body: str, status=HTTPStatus.OK):
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def send_login(self, error: str = ""):
            self.send_html(access.login_page(error))

        def admin_token(self, payload=None):
            header = str(self.headers.get("X-Admin-Token", "") or "").strip()
            if header:
                return header
            if isinstance(payload, dict):
                return str(payload.get("admin_token") or payload.get("token") or "").strip()
            return ""

        def require_admin(self, payload=None) -> bool:
            if access.admin_authorized(self.admin_token(payload)):
                return True
            self.send_json({"ok": False, "error": "admin token rejected"}, HTTPStatus.FORBIDDEN)
            return False

        def schedule_restart(self):
            def restart():
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.execv(
                        sys.executable,
                        [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]],
                    )
                except Exception:
                    os._exit(1)
            timer = threading.Timer(1.0, restart)
            timer.daemon = True
            timer.start()

        def cookie_secure(self):
            origin = str(self.headers.get("Origin", "")).lower()
            if origin.startswith("https://"):
                return True
            if origin.startswith("http://"):
                return False
            return None

        def current_user(self):
            return access.check(self.headers.get("Cookie"))

        def require_user(self) -> str | None:
            if not access.enabled:
                return "local"
            return self.current_user()

        def send_unauthorized(self):
            self.send_json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)

        def send_vendor(self, path: str):
            relative = path[len("/vendor/"):]
            if not relative:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            candidate = (VENDOR_ROOT / relative).resolve()
            if VENDOR_ROOT not in candidate.parents or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = {
                ".js": "text/javascript; charset=utf-8",
                ".mjs": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".png": "image/png",
                ".svg": "image/svg+xml",
                ".woff2": "font/woff2",
            }.get(candidate.suffix.lower(), "application/octet-stream")
            data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_pics(self):
            if not PICS_PATH.is_file():
                self.send_json({"ok": False, "error": "pics.zip is not available"}, HTTPStatus.NOT_FOUND)
                return
            data = PICS_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="pics.zip"')
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/login":
                self.send_login()
                return
            if path == "/healthz":
                self.send_json({"ok": True})
                return
            if path == "/admin":
                self.send_html(ADMIN_HTML)
                return
            if access.enabled and not self.current_user():
                if path in ("/", "/index.html"):
                    self.send_login()
                    return
                if path.startswith("/api/"):
                    self.send_unauthorized()
                    return
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            if path == "/api/session":
                self.send_json(
                    {
                        "ok": True,
                        "experiment": access.experiment_mode,
                        "user": access.active_user() if access.enabled else None,
                    }
                )
                return
            if path in ("/api/pics.zip", "/pics.zip"):
                self.send_pics()
                return
            if path.startswith("/vendor/"):
                self.send_vendor(path)
                return
            if path == "/" or path == "/index.html":
                self.send_html(EXPERIMENT_HTML if access.experiment_mode else HTML)
                return
            if path == "/api/models":
                self.send_json({"models": [{"gender": gender, "label": label, "available": cache.available(gender)} for gender, label in GENDERS.items()]})
                return
            if path.startswith("/api/model/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    model_path = cache.obj(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                data = model_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/export/glb/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    target = cache.glb(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{target.name}"',
                )
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/export/params/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    target = cache.params(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{target.name}"',
                )
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/export/displacement/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    target = cache.displacement_export(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "No imported displacement")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{target.name}"',
                )
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/export/fourview/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    target = cache.fourview(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{target.name}"',
                )
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/export/front/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    target = cache.fourview(gender)
                    with zipfile.ZipFile(target) as archive:
                        data = archive.read("front.png")
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="front_{gender}.png"',
                )
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/skeleton/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    self.send_json(cache.skeleton(gender))
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            if path.startswith("/api/weights/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    data = cache.owners(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_reference(self):
            """Decode an uploaded picture with AnimePoseDetector, return its eyes."""
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "empty upload")
                return
            if size > 32 * 1024 * 1024:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "picture larger than 32 MiB")
                return
            payload = self.rfile.read(size)
            try:
                result = cache.reference_eyes(payload)
            except Exception as exc:
                self.send_error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{type(exc).__name__}: {exc}")
                return
            self.send_json(result)

        def _handle_autofit_files(self, gender, data):
            """Auto-fit a folder uploaded as a JSON base64 file set."""
            entries = data.get("files")
            if not isinstance(entries, list) or not entries:
                self.send_json({"ok": False, "error": "folder contains no files"}, HTTPStatus.BAD_REQUEST)
                return
            if len(entries) > 500:
                self.send_json({"ok": False, "error": "folder has more than 500 files"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                with tempfile.TemporaryDirectory(prefix="autofit_folder_") as temporary:
                    root = Path(temporary)
                    total = 0
                    for entry in entries:
                        name = str((entry or {}).get("name", ""))
                        payload = (entry or {}).get("data", "")
                        if not name or not payload:
                            continue
                        if "," in payload:
                            payload = payload.split(",", 1)[1]
                        relative = Path(name.replace("\\", "/"))
                        if relative.is_absolute() or any(part in ("..", "") for part in relative.parts):
                            continue
                        raw = base64.b64decode(payload, validate=True)
                        total += len(raw)
                        if total > 64 * 1024 * 1024:
                            raise ValueError("folder larger than 64 MiB")
                        target = root / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(raw)
                    if access.experiment_mode:
                        shortcut = experiment_assets.try_apply(cache, gender, path=str(root))
                        if shortcut is not None:
                            self.send_json(shortcut)
                            return
                    result = autofit.run(cache, gender, path=str(root))
                    result["source"] = "文件夹（上传）"
                self.send_json(result)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
            except Exception as exc:
                self.send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _handle_autofit_upload(self, gender):
            """Run auto-fit on a raw image/zip/tar upload."""
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "empty auto-fit upload")
                return
            if size > 64 * 1024 * 1024:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "auto-fit input larger than 64 MiB")
                return
            payload = self.rfile.read(size)
            filename = self.headers.get("X-Filename", "upload")
            try:
                if access.experiment_mode:
                    shortcut = experiment_assets.try_apply(
                        cache, gender, data=payload, filename=filename
                    )
                    if shortcut is not None:
                        self.send_json(shortcut)
                        return
                result = autofit.run(cache, gender, data=payload, filename=filename)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                return
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
                return
            except Exception as exc:
                self.send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self.send_json(result)

        def _handle_displacement(self, gender):
            """Import a Blender shape-key displacement as a (N, 3) .npy file."""
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "empty displacement upload")
                return
            if size > 16 * 1024 * 1024:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "displacement larger than 16 MiB")
                return
            payload = self.rfile.read(size)
            try:
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                if not isinstance(array, np.ndarray):
                    raise ValueError("expected a single .npy array")
                cache.update_displacement(gender, array)
            except Exception as exc:
                self.send_error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{type(exc).__name__}: {exc}")
                return
            self.send_json({"ok": True, "has_displacement": True})

        def read_json(self):
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0:
                return {}
            return json.loads(self.rfile.read(size).decode("utf-8"))

        def _handle_admin_reset(self):
            try:
                data = self.read_json()
            except Exception:
                self.send_json({"ok": False, "error": "invalid admin payload"}, HTTPStatus.BAD_REQUEST)
                return
            if not self.require_admin(data):
                return
            access.reset_active()
            cache.reset_session_state()
            self.send_json({"ok": True})

        def _handle_admin_status(self):
            if not self.require_admin():
                return
            payload = {"ok": True}
            payload.update(access.admin_status())
            self.send_json(payload)

        def _handle_admin_generate(self):
            try:
                data = self.read_json()
            except Exception:
                self.send_json({"ok": False, "error": "invalid admin payload"}, HTTPStatus.BAD_REQUEST)
                return
            if not self.require_admin(data):
                return
            try:
                count = max(1, min(20, int(data.get("count", 1))))
            except Exception:
                count = 1
            label = str(data.get("label", "")).strip()
            users = []
            try:
                for index in range(count):
                    suffix = label if count == 1 or not label else f"{label}-{index + 1}"
                    users.append(access.add_participant(suffix))
            except Exception as exc:
                self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json({"ok": True, "users": users})

        def _handle_admin_restart(self):
            if not self.require_admin():
                return
            self.send_json({"ok": True, "restarting": True})
            self.schedule_restart()

        def _handle_login(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(size).decode("utf-8"))
            except Exception:
                self.send_json({"ok": False, "error": "invalid login payload"}, HTTPStatus.BAD_REQUEST)
                return
            status, token, message = access.login(
                str(data.get("username", "")), str(data.get("token", ""))
            )
            if token is None:
                self.send_json({"ok": False, "error": message}, status)
                return
            cache.reset_session_state()
            self.send_json(
                {"ok": True, "user": str(data.get("username", "")).strip()},
                status,
                headers={
                    "Set-Cookie": access.session_cookie(
                        token, secure=self.cookie_secure()
                    )
                },
            )

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/admin/status":
                self._handle_admin_status()
                return
            if path == "/api/admin/generate":
                self._handle_admin_generate()
                return
            if path == "/api/admin/restart":
                self._handle_admin_restart()
                return
            if path == "/api/admin/reset":
                self._handle_admin_reset()
                return
            if path == "/api/login":
                self._handle_login()
                return
            if path == "/api/logout":
                removed = access.logout(self.headers.get("Cookie"))
                if removed and access.experiment_mode:
                    cache.reset_session_state()
                self.send_json(
                    {"ok": True},
                    headers={
                        "Set-Cookie": access.clear_cookie(
                            secure=self.cookie_secure()
                        )
                    },
                )
                return
            if path == "/api/ping":
                if access.enabled and not self.current_user():
                    self.send_unauthorized()
                    return
                self.send_json({"ok": True})
                return
            if access.enabled and not self.current_user():
                self.send_unauthorized()
                return
            if path == "/api/autofit/path" and not access.allow_server_paths:
                self.send_json(
                    {"ok": False, "error": "server paths are disabled in experiment mode"},
                    HTTPStatus.FORBIDDEN,
                )
                return
            if path.startswith("/api/model/upload/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 32 * 1024 * 1024:
                        raise ValueError("模型文件大小必须在 1 B 到 32 MiB 之间")
                    payload = self.rfile.read(size)
                    result = cache.replace_custom_model(gender, payload)
                    self.send_json(result)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested base SMPL-X gender model is not installed")
                except ValueError as exc:
                    self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")
                return
            if path.startswith("/api/autofit/upload/"):
                self._handle_autofit_upload(path.rsplit("/", 1)[-1])
                return
            if path.startswith("/api/displacement/") and path != "/api/displacement/reset":
                self._handle_displacement(path.rsplit("/", 1)[-1])
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                gender = str(data["gender"])
                if path == "/api/reset":
                    cache.reset(gender)
                    self.send_json({"ok": True})
                    return
                if path == "/api/model/switch":
                    self.send_json(cache.switch_standard_model(gender))
                    return
                if path == "/api/undo":
                    self.send_json(cache.undo(gender))
                    return
                if path == "/api/history/jump":
                    self.send_json(cache.jump_history(gender, int(data["id"])))
                    return
                if path == "/api/pose":
                    target = data["target"]
                    if not isinstance(target, list) or len(target) != 3:
                        raise ValueError("target must be [x, y, z]")
                    cache.pose_joint(
                        gender,
                        int(data["joint"]),
                        [float(value) for value in target],
                        bool(data.get("full_body", False)),
                        symmetric=bool(data.get("symmetric", False)),
                    )
                    self.send_json({"ok": True})
                    return
                if path == "/api/head":
                    cache.update_head(
                        gender,
                        {
                            "head_length": float(data.get("head_length", 1.0)),
                            "head_width": float(data.get("head_width", 1.0)),
                            "head_depth": float(data.get("head_depth", 1.0)),
                            "jaw": float(data.get("jaw", 1.0)),
                            "eye": float(data.get("eye", 1.0)),
                        },
                    )
                    self.send_json({"ok": True, "head": cache.head(gender)})
                    return
                if path == "/api/head/reset":
                    cache.reset_head(gender)
                    self.send_json({"ok": True, "head": cache.head(gender)})
                    return
                if path == "/api/displacement/reset":
                    cache.reset_displacement(gender)
                    self.send_json({"ok": True, "has_displacement": False})
                    return
                if path == "/api/autofit/files":
                    self._handle_autofit_files(gender, data)
                    return
                if path == "/api/autofit/path":
                    result = autofit.run(
                        cache, gender, path=str(data["path"])
                    )
                    self.send_json(result)
                    return
                if path == "/api/segment":
                    radii = data["radii"]
                    if not isinstance(radii, list):
                        raise ValueError("radii must be a list")
                    cache.update_segment(
                        gender,
                        int(data["child"]),
                        float(data["length"]),
                        [float(value) for value in radii],
                        symmetric=bool(data.get("symmetric", False)),
                    )
                    self.send_json({"ok": True})
                    return
                if path == "/api/segment/reset":
                    cache.reset_segment(
                        gender, int(data["child"]),
                        keep_length=bool(data.get("keep_length", False)),
                        symmetric=bool(data.get("symmetric", False)),
                    )
                    self.send_json({"ok": True})
                    return
#                 if path == "/api/reference/solve":
#                     override = data.get("full_body")
#                     result = cache.align_reference(
#                         gender,
#                         float(data["pixels_per_meter"]),
#                         [float(v) for v in data["eye_midpoint"]],
#                         {str(k): [float(a), float(b)] for k, (a, b) in data["joints"].items()},
#                         full_body=None if override is None else bool(override),
#                     )
#                     self.send_json(result)
#                     return
                self.send_error(HTTPStatus.NOT_FOUND)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
            except (KeyError, TypeError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
    return Handler


def main():
    parser = argparse.ArgumentParser(description="SMPL-X T-pose viewer")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/hgc/models/smplx"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/hgc/project/AnimeAvatarGen/outputs/smplx_tpose_viewer"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=DEVICE_CHOICES,
        help="auto (CUDA when a working GPU is detected) | cuda | cpu",
    )
    parser.add_argument(
        "--detector-dir",
        default=os.environ.get("ANIME_POSE_DETECTOR_DIR", "/home/lomerson/MyProjects/AnimePoseDetector"),
        help="directory holding anime_pose_detector.py (reference-picture eye detection)",
    )
    args = parser.parse_args()
    device, reason = resolve_device(args.device)
    if args.device != "auto" and args.device != device:
        print(f"[device] {args.device} requested but unusable -> cpu", flush=True)
    print(f"device: {device} — {reason}", flush=True)
    print(f"detector: {args.detector_dir}", flush=True)
    try:
        access = AccessController.from_env()
    except RuntimeError as exc:
        raise SystemExit(f"[access] {exc}") from exc
    print(
        f"access: {'experiment gate' if access.enabled else 'local (no gate)'}"
        f" · server paths: {'allowed' if access.allow_server_paths else 'blocked'}",
        flush=True,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_factory(
            ModelCache(
                args.model_dir, args.cache_dir, device=device,
                detector_dir=args.detector_dir,
            ),
            access=access,
        ),
    )
    print(f"SMPL-X viewer: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
