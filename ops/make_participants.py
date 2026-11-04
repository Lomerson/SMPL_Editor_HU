#!/usr/bin/env python3
"""Generate random experiment usernames/tokens and an admin token.

Writes two files, both mode 0600, outside the git checkout by default:

  ~/.config/smplx_experiment/participants.json
  ~/.config/smplx_experiment/admin_token

The viewer never invents credentials at login time; it only accepts exact
username/token pairs listed in participants.json.  This script is the
convenience generator for that file.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import string
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("count", nargs="?", type=int, default=1)
    parser.add_argument("--labels", default="")
    parser.add_argument(
        "--participants",
        default=str(Path.home() / ".config/smplx_experiment/participants.json"),
    )
    parser.add_argument(
        "--admin-token",
        default=str(Path.home() / ".config/smplx_experiment/admin_token"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be >= 1")
    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    users = []
    for index in range(args.count):
        label = labels[index] if index < len(labels) else f"P{index + 1:02d}"
        users.append(
            {
                "username": "u_" + secrets.token_hex(2),
                "token": "".join(
                    secrets.choice(string.ascii_lowercase + string.digits)
                    for _ in range(10)
                ),
                "label": label,
            }
        )
    participants = Path(args.participants).expanduser()
    admin_path = Path(args.admin_token).expanduser()
    if not args.force and (participants.exists() or admin_path.exists()):
        raise SystemExit(
            "refusing to overwrite existing credentials; pass --force to replace"
        )
    participants.parent.mkdir(parents=True, exist_ok=True)
    admin_path.parent.mkdir(parents=True, exist_ok=True)
    participants.write_text(
        json.dumps({"users": users}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    admin_token = secrets.token_urlsafe(32)
    admin_path.write_text(admin_token + "\n", encoding="utf-8")
    os.chmod(participants, 0o600)
    os.chmod(admin_path, 0o600)
    print(f"participants: {participants}")
    print(f"admin token : {admin_path}")
    for user in users:
        print(f"{user['label']}\n{user['username']}\n{user['token']}\n")


if __name__ == "__main__":
    main()
