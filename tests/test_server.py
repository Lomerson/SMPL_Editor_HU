"""HTTP-level checks for multi-session routing, keep-alive and gzip."""
from __future__ import annotations

import gzip
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from core.access import AccessController, Participant
from core.workspaces import WorkspacePool
from server import handler_factory


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        users = {
            f"u_{index}": Participant(f"u_{index}", chr(96 + index) * 32, f"P{index:02d}")
            for index in range(1, 7)
        }
        self.access = AccessController(
            users, enabled=True, ttl=30.0, cookie_secure=False,
            admin_token="admin-test", max_active=5,
        )
        self.pool = WorkspacePool(root / "models", root / "cache", device="cpu")
        from http.server import ThreadingHTTPServer
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            handler_factory(self.pool.cache, access=self.access, workspaces=self.pool),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        for _ in range(50):
            try:
                probe = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                probe.close()
                break
            except OSError:
                time.sleep(0.02)
        else:
            self.fail("test HTTP server did not start listening")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.pool.drop_all()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None, cookie=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        headers = dict(headers or {})
        if cookie:
            headers["Cookie"] = cookie
        if body is not None and not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        status = response.status
        response_headers = dict(response.getheaders())
        connection.close()
        return status, payload, response_headers

    def login(self, index):
        status, payload, headers = self.request(
            "POST", "/api/login",
            {"username": f"u_{index}", "token": chr(96 + index) * 32},
        )
        self.assertEqual(status, 200, payload)
        cookie = headers.get("Set-Cookie", "").split(";", 1)[0]
        return cookie

    def test_five_sessions_and_sixth_rejected(self):
        cookies = [self.login(index) for index in range(1, 6)]
        self.assertEqual(len(set(cookies)), 5)
        status, payload, _ = self.request(
            "POST", "/api/login",
            {"username": "u_6", "token": "f" * 32},
        )
        self.assertEqual(status, 423, payload)
        status, payload, _ = self.request("GET", "/api/models", cookie=cookies[0])
        self.assertEqual(status, 200)
        status, payload, _ = self.request("POST", "/api/ping", cookie=cookies[0])
        self.assertEqual(status, 200)

    def test_admin_kick_only_targets_one_session(self):
        first = self.login(1)
        second = self.login(2)
        status, payload, _ = self.request(
            "POST", "/api/admin/kick",
            {"username": "u_1"},
            headers={"X-Admin-Token": "admin-test"},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(self.request("GET", "/api/models", cookie=first)[0], 401)
        self.assertEqual(self.request("GET", "/api/models", cookie=second)[0], 200)

    def test_gzip_html_and_keep_alive(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        connection.request("GET", "/admin", headers={"Accept-Encoding": "gzip"})
        response = connection.getresponse()
        body = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.version, 11)
        self.assertEqual(response.getheader("Content-Encoding"), "gzip")
        self.assertIn(b"SMPL-X", gzip.decompress(body))
        # Same connection is reused for the second request.
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b'{"ok": true}')
        connection.close()

    def test_admin_status_reports_capacity(self):
        status, payload, _ = self.request(
            "POST", "/api/admin/status", {},
            headers={"X-Admin-Token": "admin-test"},
        )
        self.assertEqual(status, 200)
        data = json.loads(payload)
        self.assertEqual(data["max_active"], 5)
        self.assertIn("active_users", data)


if __name__ == "__main__":
    unittest.main()
