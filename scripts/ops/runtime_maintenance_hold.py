#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_maintenance import (
    engage_maintenance_hold,
    maintenance_hold_snapshot,
    release_maintenance_hold,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the expiring runtime maintenance hold.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--engage", action="store_true")
    action.add_argument("--release", action="store_true")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--reason", default="runtime_maintenance")
    parser.add_argument("--owner", default=os.getenv("USER", "operator"))
    parser.add_argument("--ttl-seconds", type=int, default=8 * 60 * 60)
    parser.add_argument("--expected-token", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if args.engage:
        payload = engage_maintenance_hold(
            project_root,
            reason=str(args.reason or "runtime_maintenance"),
            owner=str(args.owner or "operator"),
            ttl_seconds=max(int(args.ttl_seconds), 60),
        )
        action_name = "engaged"
    elif args.release:
        payload = release_maintenance_hold(project_root, expected_token=str(args.expected_token or ""))
        action_name = "released" if bool(payload.get("released", False)) else "release_failed"
    else:
        payload = maintenance_hold_snapshot(project_root)
        action_name = "status"

    output = {"action": action_name, **payload}
    if args.json:
        print(json.dumps(output, ensure_ascii=True))
    else:
        print(
            "runtime_maintenance_hold "
            f"action={action_name} active={int(bool(output.get('active', False)))} "
            f"expired={int(bool(output.get('expired', False)))} "
            f"reason={output.get('reason', '')}"
        )
    return 1 if action_name == "release_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
