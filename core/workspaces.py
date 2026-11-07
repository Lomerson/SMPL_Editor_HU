"""Per-account model workspaces.

The original viewer has one ``ModelCache`` shared by every request.  That is
correct for one experimenter but serializes five participants behind one lock
and lets their edit state overwrite each other.  This module gives every
username its own ``ModelCache`` with its own lock, state and export directory,
while keeping the HTTP handler's existing ``cache.<method>()`` call style via a
request-bound proxy.
"""
from __future__ import annotations

import re
import secrets
import shutil
import threading
from pathlib import Path

from core.model import ModelCache


def _safe_name(username: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(username or "local"))[:48]
    return cleaned or "local"


class WorkspacePool:
    def __init__(self, model_dir: Path, cache_root: Path, device: str = "cpu",
                 detector_dir: str = ""):
        self.model_dir = Path(model_dir)
        self.cache_root = Path(cache_root)
        self.sessions_root = self.cache_root / "sessions"
        self.device = device
        self.detector_dir = detector_dir
        self._lock = threading.RLock()
        self._caches: dict[str, ModelCache] = {}

    @property
    def cache(self) -> "BoundWorkspaceCache":
        return BoundWorkspaceCache(self)

    def get(self, username: str) -> ModelCache:
        username = str(username or "local")
        with self._lock:
            cache = self._caches.get(username)
            if cache is None:
                # Unique directory per login/workspace, so two generations of
                # the same username can never overwrite each other's exports.
                suffix = secrets.token_hex(4)
                cache_dir = self.sessions_root / f"{_safe_name(username)}-{suffix}"
                cache = ModelCache(
                    self.model_dir,
                    cache_dir,
                    device=self.device,
                    detector_dir=self.detector_dir,
                )
                cache.workspace_username = username
                self._caches[username] = cache
            return cache

    def drop(self, username: str):
        username = str(username or "")
        with self._lock:
            cache = self._caches.pop(username, None)
        if cache is None:
            return
        try:
            cache.reset_session_state()
        except Exception:
            pass
        # Best-effort cleanup.  Requests for this user have already been
        # invalidated by the access layer; a rare in-flight export may fail.
        try:
            shutil.rmtree(cache.cache_dir, ignore_errors=True)
        except Exception:
            pass

    def drop_all(self):
        with self._lock:
            usernames = list(self._caches)
        for username in usernames:
            self.drop(username)

    def prune(self, active_usernames):
        active = set(active_usernames)
        with self._lock:
            stale = [name for name in self._caches if name not in active]
        for username in stale:
            self.drop(username)

    def usernames(self):
        with self._lock:
            return sorted(self._caches)


class BoundWorkspaceCache:
    """Thread-local proxy: existing handler code calls ``cache.method()``."""

    def __init__(self, pool: WorkspacePool):
        self.pool = pool
        self._local = threading.local()

    def bind(self, username: str):
        self._local.username = str(username or "local")

    def current(self) -> ModelCache:
        username = getattr(self._local, "username", "local")
        return self.pool.get(username)

    def __getattr__(self, name):
        return getattr(self.current(), name)


class SingleWorkspacePool:
    """Compatibility wrapper for callers that still pass one ModelCache."""

    def __init__(self, cache: ModelCache):
        self._cache = cache
        self.cache_root = cache.cache_dir
        self.sessions_root = cache.cache_dir

    @property
    def cache(self) -> "SingleWorkspaceCache":
        return SingleWorkspaceCache(self)

    def get(self, _username: str = "local") -> ModelCache:
        return self._cache

    def drop(self, _username: str = "local"):
        self._cache.reset_session_state()

    def drop_all(self):
        self._cache.reset_session_state()

    def prune(self, _active_usernames):
        return

    def usernames(self):
        return ["local"]


class SingleWorkspaceCache(BoundWorkspaceCache):
    def __init__(self, pool: SingleWorkspacePool):
        super().__init__(pool)  # type: ignore[arg-type]
        self.bind("local")

    def bind(self, username: str):
        self._local.username = "local"
