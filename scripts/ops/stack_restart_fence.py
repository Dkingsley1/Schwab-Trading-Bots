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

from core.stack_restart_coordination import (
    DEFAULT_TTL_SECONDS,
    engage_stack_restart_fence,
    release_stack_restart_fence,
    stack_restart_fence_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate an exclusive stack restart window.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--engage", action="store_true")
    action.add_argument("--release", action="store_true")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--owner", default="start_stack")
    parser.add_argument("--owner-pid", type=int, default=os.getppid())
    parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    parser.add_argument("--expected-token", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if args.engage:
        payload = engage_stack_restart_fence(
            project_root,
            owner_pid=int(args.owner_pid),
            owner=str(args.owner or "start_stack"),
            ttl_seconds=max(int(args.ttl_seconds), 60),
        )
        action_name = "engaged" if bool(payload.get("acquired", False)) else "engage_failed"
    elif args.release:
        payload = release_stack_restart_fence(project_root, expected_token=str(args.expected_token or ""))
        action_name = "released" if bool(payload.get("released", False)) else "release_failed"
    else:
        payload = stack_restart_fence_snapshot(project_root)
        action_name = "status"

    output = {"action": action_name, **payload}
    if args.json:
        print(json.dumps(output, ensure_ascii=True))
    else:
        print(
            "stack_restart_fence "
            f"action={action_name} active={int(bool(output.get('active', False)))} "
            f"owner_pid={int(output.get('owner_pid', 0) or 0)} reason={output.get('reason', '')}"
        )
    return 1 if action_name in {"engage_failed", "release_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
