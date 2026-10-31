"""HTTP surface for the SMPL-X viewer; the implementation lives in core/."""
from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core.config import DEVICE_CHOICES, GENDERS
from core.model import ModelCache, resolve_device

HTML = (Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")

def handler_factory(cache: ModelCache):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/reference":
                self._handle_reference()
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                gender = str(data["gender"])
                if path == "/api/reset":
                    cache.reset(gender)
                    self.send_json({"ok": True})
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
                    )
                    self.send_json({"ok": True})
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
                    )
                    self.send_json({"ok": True})
                    return
                if path == "/api/segment/reset":
                    cache.reset_segment(
                        gender, int(data["child"]),
                        keep_length=bool(data.get("keep_length", False)),
                    )
                    self.send_json({"ok": True})
                    return
                if path == "/api/reference/solve":
                    override = data.get("full_body")
                    result = cache.align_reference(
                        gender,
                        float(data["pixels_per_meter"]),
                        [float(v) for v in data["eye_midpoint"]],
                        {str(k): [float(a), float(b)] for k, (a, b) in data["joints"].items()},
                        full_body=None if override is None else bool(override),
                    )
                    self.send_json(result)
                    return
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
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_factory(
            ModelCache(
                args.model_dir, args.cache_dir, device=device,
                detector_dir=args.detector_dir,
            )
        ),
    )
    print(f"SMPL-X viewer: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
