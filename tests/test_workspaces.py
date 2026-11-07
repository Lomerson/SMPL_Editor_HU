"""Checks for per-account ModelCache isolation."""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from core.workspaces import BoundWorkspaceCache, SingleWorkspacePool, WorkspacePool


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.pool = WorkspacePool(
            root / "models", root / "cache", device="cpu"
        )

    def tearDown(self):
        self.pool.drop_all()
        self.tmp.cleanup()

    def test_two_users_get_distinct_cache_directories(self):
        a = self.pool.get("u_alpha")
        b = self.pool.get("u_beta")
        self.assertIsNot(a, b)
        self.assertNotEqual(a.cache_dir, b.cache_dir)
        self.assertTrue(str(a.cache_dir).startswith(str(self.pool.sessions_root)))
        a.head_configs["female"] = {"head_length": 1.2}
        self.assertNotIn("female", b.head_configs)

    def test_drop_only_removes_target(self):
        a = self.pool.get("u_alpha")
        b = self.pool.get("u_beta")
        b.head_configs["female"] = {"head_length": 0.8}
        self.pool.drop("u_alpha")
        self.assertEqual(self.pool.usernames(), ["u_beta"])
        self.assertEqual(self.pool.get("u_beta"), b)
        self.assertIn("female", self.pool.get("u_beta").head_configs)
        self.assertIsNot(self.pool.get("u_alpha"), a)

    def test_prune_drops_inactive_workspaces(self):
        self.pool.get("u_alpha")
        self.pool.get("u_beta")
        self.pool.prune(["u_beta"])
        self.assertEqual(self.pool.usernames(), ["u_beta"])

    def test_bound_proxy_is_thread_isolated(self):
        proxy = BoundWorkspaceCache(self.pool)
        self.pool.get("u_alpha").head_configs["female"] = {"head_length": 1.1}
        self.pool.get("u_beta").head_configs["female"] = {"head_length": 0.9}
        seen = {}

        def read(name):
            proxy.bind(name)
            seen[name] = proxy.head("female")["head_length"]

        threads = [
            threading.Thread(target=read, args=("u_alpha",)),
            threading.Thread(target=read, args=("u_beta",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(seen, {"u_alpha": 1.1, "u_beta": 0.9})

    def test_single_cache_compatibility_wrapper(self):
        cache = self.pool.get("u_alpha")
        single = SingleWorkspacePool(cache)
        proxy = single.cache
        proxy.bind("anything")
        self.assertIs(proxy.current(), cache)
        single.drop_all()
        self.assertEqual(cache.histories, {})


if __name__ == "__main__":
    unittest.main()
