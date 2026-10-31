#!/usr/bin/env python3
"""Compatibility entry point; the implementation lives in server.py + core/.

The pre-split module exposed ``ModelCache``, ``resolve_device`` and the HTML
string; re-export them so existing imports keep working.
"""
from core.config import GENDERS, JOINT_NAMES
from core.model import ModelCache, resolve_device
from server import HTML, handler_factory, main

__all__ = [
    "GENDERS",
    "JOINT_NAMES",
    "ModelCache",
    "resolve_device",
    "HTML",
    "handler_factory",
    "main",
]

if __name__ == "__main__":
    main()
