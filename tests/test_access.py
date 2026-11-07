"""Checks for the experiment login/multi-session gate."""
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

    def _six_user_access(self):
        users = {
            f"u_{index}": Participant(f"u_{index}", chr(96 + index) * 32, f"P{index:02d}")
            for index in range(1, 7)
        }
        return AccessController(
            users, enabled=True, ttl=30.0, cookie_secure=False,
            admin_token="secret-admin", max_active=5,
        )

    def test_five_active_participants_and_sixth_rejected(self):
        access = self._six_user_access()
        tokens = []
        for index in range(1, 6):
            status, token, _ = access.login(f"u_{index}", chr(96 + index) * 32)
            self.assertEqual(status, 200)
            tokens.append(token)
        self.assertEqual(len(access.active_users()), 5)
        status, token, message = access.login("u_6", "f" * 32)
        self.assertEqual(status, 423)
        self.assertIsNone(token)
        self.assertIn("5", message)
        self.assertEqual(access.active_users(), ["u_1", "u_2", "u_3", "u_4", "u_5"])

    def test_logout_frees_a_capacity_slot(self):
        access = self._six_user_access()
        tokens = {}
        for index in range(1, 6):
            status, token, _ = access.login(f"u_{index}", chr(96 + index) * 32)
            self.assertEqual(status, 200)
            tokens[index] = token
        self.assertEqual(access.login("u_6", "f" * 32)[0], 423)
        self.assertEqual(access.logout(access.session_cookie(tokens[1])), "u_1")
        status, token, _ = access.login("u_6", "f" * 32)
        self.assertEqual(status, 200)
        self.assertIn("u_6", access.active_users())

    def test_lease_expiry_frees_a_capacity_slot(self):
        access = self._six_user_access()
        for index in range(1, 6):
            status, token, _ = access.login(f"u_{index}", chr(96 + index) * 32)
            self.assertEqual(status, 200)
            # Simulate an abandoned browser without waiting 30 seconds.
            access._sessions[token] = (f"u_{index}", time.time() - 31)
        self.assertEqual(access.login("u_6", "f" * 32)[0], 200)

    def test_repeated_login_replaces_old_session(self):
        status, first, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        status, second, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        self.assertIsNone(self.access.check(self.access.session_cookie(first)))
        self.assertEqual(self.access.check(self.access.session_cookie(second)), "u_alpha")
        self.assertEqual(self.access.active_users(), ["u_alpha"])

    def test_generated_participant_persists_and_can_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "participants.json"
            access = AccessController(
                {}, enabled=True, cookie_secure=False,
                admin_token="secret-admin", participants_path=path,
            )
            created = access.add_participant("P03")
            self.assertEqual(created["label"], "P03")
            self.assertEqual(len(created["username"]), 6)
            self.assertEqual(len(created["token"]), 10)
            self.assertEqual(access.login(created["username"], created["token"])[0], 200)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["users"][0]["username"], created["username"])

    def test_logout_reports_the_username(self):
        self.assertIsNone(self.access.logout("smplx_session=missing"))
        status, token, _ = self.access.login("u_alpha", "a" * 32)
        self.assertEqual(status, 200)
        cookie = self.access.session_cookie(token)
        self.assertEqual(self.access.logout(cookie), "u_alpha")
        self.assertIsNone(self.access.logout(cookie))

    def test_admin_status_lists_participants_without_tokens(self):
        status = self.access.admin_status()
        self.assertTrue(status["enabled"])
        self.assertEqual({item["username"] for item in status["participants"]}, {"u_alpha", "u_beta"})
        self.assertNotIn("token", status["participants"][0])
        self.assertEqual(status["max_active"], 5)
        self.assertEqual(status["active_users"], [])

    def test_admin_kick_only_removes_target_user(self):
        self.access.login("u_alpha", "a" * 32)
        self.access.login("u_beta", "b" * 32)
        self.assertTrue(self.access.kick("u_alpha"))
        self.assertEqual(self.access.active_users(), ["u_beta"])
        self.assertFalse(self.access.kick("u_alpha"))

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
