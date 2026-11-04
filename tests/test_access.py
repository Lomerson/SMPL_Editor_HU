"""Checks for the experiment login/single-session gate."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from core.access import AccessController, COOKIE_NAME, Participant
from core.model import ModelCache


class AccessTest(unittest.TestCase):
    def setUp(self):
        self.users = {
            "u_alpha": Participant("u_alpha", "a" * 32, "P01"),
            "u_beta": Participant("u_beta", "b" * 32, "P02"),
        }
        self.access = AccessController(
            self.users, enabled=True, ttl=30.0, cookie_secure=False,
            admin_token="secret-admin",
        )

    def test_login_and_cookie(self):
        status, token, message = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        self.assertTrue(token)
        self.assertEqual(message, "ok")
        cookie = self.access.session_cookie(token)
        self.assertIn(COOKIE_NAME + "=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertNotIn("Secure", cookie)
        self.assertIn("Secure", self.access.session_cookie(token, secure=True))
        self.assertNotIn("Secure", self.access.session_cookie(token, secure=False))
        self.assertEqual(self.access.check(cookie), "u_alpha")

    def test_wrong_credentials(self):
        status, token, _ = self.access.login("u_alpha", "wrong")
        self.assertEqual(status, 401)
        self.assertIsNone(token)
        status, token, _ = self.access.login("missing", "a" * 32)
        self.assertEqual(status, 401)

    def test_only_one_active_participant(self):
        status, token, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        status, token2, _ = self.access.login("u_beta", "b" * 32)
        self.assertEqual(status, 423)
        self.assertIsNone(token2)
        self.assertEqual(self.access.active_user(), "u_alpha")

    def test_lease_expires_and_logout_releases(self):
        status, token, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        cookie = self.access.session_cookie(token)
        # Simulate an abandoned browser without waiting 30 seconds.
        self.access._active_seen = time.time() - 31
        self.access._sessions[token] = ("u_alpha", time.time() - 31)
        status, token2, _ = self.access.login("u_beta", "b" * 32)
        self.assertEqual(status, 200)
        self.assertEqual(self.access.active_user(), "u_beta")
        self.assertIsNone(self.access.check(cookie))

        self.access.logout(self.access.session_cookie(token2))
        self.assertIsNone(self.access.active_user())
        status, _token3, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)

    def test_generated_participant_persists_and_can_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "participants.json"
            access = AccessController(
                {}, enabled=True, cookie_secure=False,
                admin_token="secret-admin", participants_path=path,
            )
            created = access.add_participant("P03")
            self.assertEqual(created["label"], "P03")
            self.assertEqual(access.login(created["username"], created["token"])[0], 200)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["users"][0]["username"], created["username"])

    def test_logout_reports_whether_a_session_was_removed(self):
        self.assertFalse(self.access.logout("smplx_session=missing"))
        status, token, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        cookie = self.access.session_cookie(token)
        self.assertTrue(self.access.logout(cookie))
        self.assertFalse(self.access.logout(cookie))

    def test_admin_status_lists_participants_without_tokens(self):
        status = self.access.admin_status()
        self.assertTrue(status["enabled"])
        self.assertEqual({item["username"] for item in status["participants"]}, {"u_alpha", "u_beta"})
        self.assertNotIn("token", status["participants"][0])

    def test_admin_reset_requires_token(self):
        status, token, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        self.assertFalse(self.access.admin_reset("wrong"))
        self.assertIsNotNone(self.access.active_user())
        self.assertTrue(self.access.admin_reset("secret-admin"))
        self.assertIsNone(self.access.active_user())
        self.assertIsNone(self.access.check(self.access.session_cookie(token)))

    def test_new_session_reset_clears_history_and_identity(self):
        cache = ModelCache(Path("/tmp"), Path("/tmp/autofit_unused"))
        cache.segment_configs["female"] = {"3": {"length": 1.2, "radii": [1, 1]}}
        cache.head_configs["female"] = {"head_length": 1.1}
        cache.displacements["female"] = object()
        cache.histories["female"] = {"entries": [{"id": 0}], "cursor": 0, "next_id": 1}
        cache.reset_session_state()
        self.assertEqual(cache.segment_configs, {})
        self.assertEqual(cache.head_configs, {})
        self.assertEqual(cache.displacements, {})
        self.assertEqual(cache.histories, {})

    def test_server_paths_are_disabled_in_experiment_mode(self):
        self.assertFalse(self.access.allow_server_paths)
        with unittest.mock.patch.dict(os.environ, {"SMPLX_ALLOW_PATHS": "1"}):
            self.assertTrue(self.access.allow_server_paths)

    def test_from_env_loads_participants_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "participants.json"
            path.write_text(
                json.dumps({"users": [{"username": "u1", "token": "x" * 32}]}),
                encoding="utf-8",
            )
            with unittest.mock.patch.dict(
                os.environ,
                {"SMPLX_EXPERIMENT": "1", "SMPLX_ACCESS_FILE": str(path)},
            ):
                access = AccessController.from_env()
                self.assertTrue(access.enabled)
                self.assertEqual(access.login("u1", "x" * 32)[0], 200)

            with unittest.mock.patch.dict(
                os.environ,
                {"SMPLX_EXPERIMENT": "1", "SMPLX_ACCESS_FILE": str(Path(tmp) / "missing.json")},
            ):
                with self.assertRaises(RuntimeError):
                    AccessController.from_env()


if __name__ == "__main__":
    unittest.main()
