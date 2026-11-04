"""Experiment access gate: random usernames, one active participant.

The viewer itself has no accounts and its ``ModelCache`` is intentionally a
single global object (one SMPL-X model, one edit state).  That is fine for a
scheduled experiment but not for a public Funnel URL: two participants would
otherwise share the same edits.  This module adds a deliberately small gate:

* each participant has a random username and a long random access token;
* a successful login sets ``smplx_session`` as an HttpOnly cookie;
* only one participant may hold the active lease at a time;
* every authenticated request refreshes a short TTL, so a crashed browser
  releases the experiment after a few minutes;
* the experimenter can reset the lease from localhost.

It is not a general user system and stores sessions in memory only.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path

COOKIE_NAME = "smplx_session"
DEFAULT_TTL = 300.0


@dataclass(frozen=True)
class Participant:
    username: str
    token: str
    label: str = ""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_participants(path: str) -> dict[str, Participant]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise RuntimeError(
            f"SMPLX_ACCESS_FILE 不存在: {source}；实验模式拒绝在没有访问清单时启动"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"无法读取参与者清单 {source}: {exc}") from exc
    users = payload.get("users") if isinstance(payload, dict) else payload
    if not isinstance(users, list) or not users:
        raise RuntimeError(f"参与者清单 {source} 必须包含非空 users 列表")
    result: dict[str, Participant] = {}
    for entry in users:
        if not isinstance(entry, dict):
            raise RuntimeError("参与者条目必须是对象")
        username = str(entry.get("username", "")).strip()
        token = str(entry.get("token", "")).strip()
        label = str(entry.get("label", "")).strip()
        if not username or not token:
            raise RuntimeError("参与者必须有 username 和 token")
        if len(token) < 16:
            raise RuntimeError(f"{username} 的 token 太短（至少 16 字符）")
        if username in result:
            raise RuntimeError(f"参与者用户名重复: {username}")
        result[username] = Participant(username, token, label)
    return result


class AccessController:
    def __init__(self, participants: dict[str, Participant] | None = None,
                 *, enabled: bool = False, ttl: float = DEFAULT_TTL,
                 cookie_secure: bool = True, admin_token: str = "",
                 participants_path: str | Path | None = None):
        self.participants = participants or {}
        self.participants_path = Path(participants_path).expanduser() if participants_path else None
        self.enabled = bool(enabled)
        self.ttl = max(30.0, float(ttl))
        self.cookie_secure = bool(cookie_secure)
        self.admin_token = str(admin_token or "")
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[str, float]] = {}
        self._active_user: str | None = None
        self._active_seen = 0.0
        self._last_error = ""

    @classmethod
    def disabled(cls) -> "AccessController":
        return cls(enabled=False)

    @classmethod
    def from_env(cls) -> "AccessController":
        if not _truthy(os.environ.get("SMPLX_EXPERIMENT")):
            return cls.disabled()
        path = os.environ.get(
            "SMPLX_ACCESS_FILE",
            str(Path.home() / ".config" / "smplx_experiment" / "participants.json"),
        )
        ttl = float(os.environ.get("SMPLX_SESSION_TTL", DEFAULT_TTL))
        secure = _truthy(os.environ.get("SMPLX_COOKIE_SECURE", "1"))
        return cls(
            _load_participants(path),
            enabled=True,
            ttl=ttl,
            cookie_secure=secure,
            admin_token=os.environ.get("SMPLX_ADMIN_TOKEN", ""),
            participants_path=path,
        )

    @property
    def experiment_mode(self) -> bool:
        return self.enabled

    @property
    def allow_server_paths(self) -> bool:
        # A public Funnel request must never be able to read an arbitrary
        # server path.  The local launcher can opt in for preparation work.
        return (not self.enabled) or _truthy(os.environ.get("SMPLX_ALLOW_PATHS"))

    def _expire_locked(self, now: float):
        for token, (user, seen) in list(self._sessions.items()):
            if now - seen > self.ttl:
                self._sessions.pop(token, None)
        if self._active_user and now - self._active_seen > self.ttl:
            self._active_user = None
            self._active_seen = 0.0

    def login(self, username: str, token: str):
        """Return (status, session_token, message)."""
        username = str(username or "").strip()
        token = str(token or "")
        with self._lock:
            participant = self.participants.get(username)
            if participant is None or not hmac.compare_digest(participant.token, token):
                return 401, None, "用户名或访问码错误"
            now = time.time()
            self._expire_locked(now)
            if self._active_user and self._active_user != username:
                return 423, None, "当前有受试者正在使用系统，请稍后再试"
            session_token = secrets.token_urlsafe(32)
            self._sessions[session_token] = (username, now)
            self._active_user = username
            self._active_seen = now
            return 200, session_token, "ok"

    def check(self, cookie_header: str | None) -> str | None:
        if not self.enabled:
            return None
        token = self._cookie_value(cookie_header)
        if not token:
            return None
        with self._lock:
            now = time.time()
            self._expire_locked(now)
            record = self._sessions.get(token)
            if record is None:
                return None
            username, _seen = record
            if self._active_user != username:
                self._sessions.pop(token, None)
                return None
            self._sessions[token] = (username, now)
            self._active_seen = now
            return username

    def ping(self, cookie_header: str | None) -> str | None:
        return self.check(cookie_header)

    def logout(self, cookie_header: str | None) -> bool:
        token = self._cookie_value(cookie_header)
        if not token:
            return False
        with self._lock:
            record = self._sessions.pop(token, None)
            if record and record[0] == self._active_user:
                self._active_user = None
                self._active_seen = 0.0
            return record is not None

    def reset_active(self):
        with self._lock:
            self._active_user = None
            self._active_seen = 0.0
            self._sessions.clear()

    def admin_authorized(self, token: str) -> bool:
        return bool(self.admin_token) and hmac.compare_digest(
            self.admin_token, str(token or "")
        )

    def admin_reset(self, token: str) -> bool:
        if not self.admin_authorized(token):
            return False
        self.reset_active()
        return True

    def _save_participants_locked(self):
        if self.participants_path is None:
            return
        payload = {
            "users": [
                {"username": item.username, "token": item.token, "label": item.label}
                for item in sorted(self.participants.values(), key=lambda item: item.label or item.username)
            ]
        }
        target = self.participants_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)

    def add_participant(self, label: str = "") -> dict:
        """Create and persist one random participant; returns its credentials."""
        label = str(label or "").strip()
        with self._lock:
            while True:
                username = "u_" + secrets.token_hex(4)
                if username not in self.participants:
                    break
            token = secrets.token_urlsafe(32)
            participant = Participant(username, token, label)
            self.participants[username] = participant
            self._save_participants_locked()
            return {
                "username": participant.username,
                "token": participant.token,
                "label": participant.label,
            }

    def admin_status(self) -> dict:
        with self._lock:
            self._expire_locked(time.time())
            return {
                "enabled": self.enabled,
                "active_user": self._active_user,
                "ttl_seconds": self.ttl,
                "participants": [
                    {"username": item.username, "label": item.label}
                    for item in sorted(self.participants.values(), key=lambda item: item.label or item.username)
                ],
            }

    def active_user(self) -> str | None:
        with self._lock:
            self._expire_locked(time.time())
            return self._active_user

    def session_cookie(self, token: str, secure: bool | None = None) -> str:
        max_age = int(max(self.ttl * 4, 600))
        bits = [
            f"{COOKIE_NAME}={token}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            f"Max-Age={max_age}",
        ]
        if (self.cookie_secure if secure is None else bool(secure)):
            bits.append("Secure")
        return "; ".join(bits)

    def clear_cookie(self, secure: bool | None = None) -> str:
        bits = [
            f"{COOKIE_NAME}=",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            "Max-Age=0",
        ]
        if (self.cookie_secure if secure is None else bool(secure)):
            bits.append("Secure")
        return "; ".join(bits)

    @staticmethod
    def _cookie_value(cookie_header: str | None) -> str | None:
        if not cookie_header:
            return None
        try:
            jar = SimpleCookie()
            jar.load(cookie_header)
            morsel = jar.get(COOKIE_NAME)
            return morsel.value if morsel else None
        except Exception:
            return None

    def login_page(self, error: str = "") -> str:
        error_html = f'<p class="error" id="login-error">{error}</p>'
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMPL-X 实验登录</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:#111318; color:#edf0f5; }}
.card {{ width:min(420px,92vw); padding:28px; border:1px solid #303642; border-radius:12px; background:#1a1e25; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
p {{ color:#aeb7c6; font-size:13px; line-height:1.6; }}
label {{ display:block; margin:16px 0 6px; color:#c7d0dd; font-size:13px; }}
input {{ width:100%; min-height:42px; border:1px solid #414957; border-radius:6px; background:#252b35; color:#eff2f7; padding:0 10px; font-size:14px; }}
button {{ width:100%; min-height:44px; margin-top:18px; border:1px solid #70a5f3; border-radius:6px; background:#24559a; color:#fff; font-size:14px; cursor:pointer; }}
button:disabled {{ opacity:.5; cursor:wait; }}
.error {{ color:#ffabb8; min-height:18px; }}
</style></head><body><main class="card">
<h1>SMPL-X 实验</h1>
<p>请输入主试发给你的随机用户名和访问码。系统同一时间只允许一位受试者使用。</p>
{error_html}
<form id="login-form">
<label for="username">用户名</label><input id="username" name="username" autocomplete="username" required>
<label for="token">访问码</label><input id="token" name="token" type="password" autocomplete="one-time-code" required>
<button id="submit" type="submit">进入实验</button>
</form>
<script>
try{{sessionStorage.removeItem('smplx_pics_downloaded');}}catch(_error){{}}
const form=document.querySelector('#login-form'), error=document.querySelector('#login-error'), button=document.querySelector('#submit');
form.addEventListener('submit',async (event)=>{{
  event.preventDefault(); button.disabled=true; error.textContent='';
  try {{
    const response=await fetch('/api/login',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{username:document.querySelector('#username').value,token:document.querySelector('#token').value}})}});
    const data=await response.json().catch(()=>({{}}));
    if(!response.ok) throw Error(data.error||('登录失败 '+response.status));
    window.location.href='/';
  }} catch (err) {{ error.textContent=err.message; button.disabled=false; }}
}});
</script></main></body></html>"""
