#!/usr/bin/env python3
"""Explicit platform power controls; power-on is never a live-trading release."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.accountability import safe_write_json_atomic
from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import run_bounded_process_group

STATE = "governance/runtime/system_power_state.json"
OFF = "governance/health/SYSTEM_POWER_OFF.flag"


def local(root: Path, relative: str) -> Path:
    path = root / relative
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("system_power_requires_local_owned_paths")
    return path


def load(root: Path) -> dict:
    try:
        return json.loads(local(root, STATE).read_text())
    except FileNotFoundError:
        return {}


def write(root: Path, rel: str, payload: dict) -> None:
    if (
        safe_write_json_atomic(
            str(local(root, rel)),
            payload,
            project_root=str(root),
            source="system_power",
        )
        is False
    ):
        raise RuntimeError("system_power_state_not_persisted")


def command(root: Path, argv: list[str], timeout: int = 60) -> dict:
    env = dict(os.environ)
    env.update(
        ALLOW_ORDER_EXECUTION="0",
        TOP_BOT_ENABLE_LIVE_EXECUTION="0",
        EXECUTION_LANE_LIVE_ENABLED="0",
        MARKET_DATA_ONLY="1",
        PAPER_TRADE_LOCK="1",
    )
    try:
        result = run_bounded_process_group(
            argv, cwd=root, env=env, timeout_seconds=timeout
        )
        return {
            "rc": int(result.get("rc", 1)),
            "timed_out": bool(result.get("timed_out")),
        }
    except subprocess.TimeoutExpired:
        return {"rc": 124, "reason": "command_timeout_requires_status_check"}


def managed_agents(root: Path, agent_dir: Path) -> list[dict]:
    """Inspect only regular plist files, and own only this repository's agents."""
    rows = []
    if agent_dir.is_symlink():
        raise ValueError("launchagent_directory_symlink_not_supported")
    for path in sorted(agent_dir.glob("*.plist")):
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            continue
        try:
            with path.open("rb") as handle:
                plist = plistlib.load(handle)
        except (OSError, ValueError, plistlib.InvalidFileException):
            continue
        if not isinstance(plist, dict):
            continue
        args = plist.get("ProgramArguments", [])
        label = plist.get("Label", "")
        if not isinstance(args, list) or not isinstance(label, str):
            continue
        owned = any(
            isinstance(arg, str) and arg.startswith(str(root) + "/") for arg in args
        )
        if not owned or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            continue
        env = plist.get("EnvironmentVariables", {})
        if not isinstance(env, dict):
            continue
        live = any(
            str(env.get(key, "0")).lower() in {"1", "true", "yes", "on"}
            for key in (
                "ALLOW_ORDER_EXECUTION",
                "EXECUTION_LANE_LIVE_ENABLED",
                "TOP_BOT_ENABLE_LIVE_EXECUTION",
            )
        )
        live = live or any(
            args[i : i + 2] == ["--mode", "live"] for i in range(len(args))
        )
        live = live or any(
            args[i : i + 2] == ["--profile", "live"] for i in range(len(args))
        )
        live = live or any(
            arg in {"start-live", "--live"} for arg in args if isinstance(arg, str)
        )
        rows.append(
            {"label": label, "plist": str(path), "live_execution_requested": live}
        )
    return rows


def status(root: Path) -> dict:
    state = load(root)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_state": (
            "off"
            if local(root, OFF).exists()
            else state.get("requested_state", "not_set")
        ),
        "operator_stop": local(root, "governance/health/OPERATOR_STOP.flag").exists(),
        "global_halt": local(
            root, "governance/health/GLOBAL_TRADING_HALT.flag"
        ).exists(),
        "last_transition": state.get("transition", "none"),
        "last_transition_ok": state.get("ok"),
        "saved_agent_count": len(state.get("agents", [])),
        "running_processes_verified": False,
        "live_execution_authority": False,
        "broker_orders_canceled": False,
    }


def run(
    root: Path, action: str, *, agent_dir: Path | None = None, runner=command
) -> dict:
    if action == "status":
        return status(root)
    lock_path = local(root, "governance/locks/system_power.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ops = str(root / "scripts/ops/opsctl.sh")
        steps = []

        def step(name, argv, timeout=60):
            result = runner(root, argv, timeout)
            steps.append({"step": name, **result})
            return result["rc"] == 0

        if action == "clear-halts":
            if local(root, OFF).exists():
                return {
                    "ok": False,
                    "reason": "system_power_off_use_explicit_on",
                    **status(root),
                }
            released = step(
                "release_operator_stop", [ops, "operator-release", "--json"]
            )
            refreshed = released and step(
                "refresh_halt_evidence", [ops, "global-halt-refresh", "--json"], 300
            )
            cleared = refreshed and step(
                "safe_clear_global_halt", [ops, "global-halt-auto-clear", "--json"], 120
            )
            return {"ok": bool(cleared), "steps": steps, **status(root)}

        domain = f"gui/{os.getuid()}"
        state = load(root)
        agents = managed_agents(root, agent_dir or Path.home() / "Library/LaunchAgents")
        if action == "off":
            if not local(root, OFF).exists():
                active = [
                    row
                    for row in agents
                    if runner(
                        root, ["launchctl", "print", f"{domain}/{row['label']}"], 10
                    )["rc"]
                    == 0
                ]
                state = {
                    "agents": active,
                    "requested_state": "off",
                    "transition": "stopping",
                    "ok": False,
                }
                write(root, STATE, state)
                write(
                    root,
                    OFF,
                    {
                        "source": "system_power",
                        "reason": "explicit_platform_off",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            # Halt broker writers first; no broker cancellation or liquidation is implied.
            step(
                "engage_operator_stop",
                [
                    ops,
                    "operator-control",
                    "--engage",
                    "--reason",
                    "system_power_off",
                    "--json",
                ],
            )
            for row in agents:
                target = f"{domain}/{row['label']}"
                step(f"disable:{row['label']}", ["launchctl", "disable", target])
                if runner(root, ["launchctl", "print", target], 10)["rc"] == 0:
                    step(f"unload:{row['label']}", ["launchctl", "bootout", target])
            step("stop_runtime_loops", [ops, "stop"], 180)
            remaining = [
                row["label"]
                for row in agents
                if runner(root, ["launchctl", "print", f"{domain}/{row['label']}"], 10)[
                    "rc"
                ]
                == 0
            ]
            state.update(
                transition="off" if not remaining else "off_incomplete",
                requested_state="off",
                remaining_agents=remaining,
            )
        elif action == "on":
            # Only the explicit ON path can remove this persistent stop request.
            if (
                not step("release_operator_stop", [ops, "operator-release", "--json"])
                or not step(
                    "refresh_halt_evidence", [ops, "global-halt-refresh", "--json"], 300
                )
                or not step(
                    "safe_clear_global_halt",
                    [ops, "global-halt-auto-clear", "--json"],
                    120,
                )
            ):
                return {
                    "ok": False,
                    "reason": "active_safety_blockers_preserved",
                    "steps": steps,
                    **status(root),
                }
            local(root, OFF).unlink(missing_ok=True)
            if step("start_guarded_platform", [ops, "start", "--paper"], 600):
                by_label = {row["label"]: row for row in agents}
                for saved in state.get("agents", []):
                    row = by_label.get(saved["label"])
                    if (
                        not row
                        or row["plist"] != saved["plist"]
                        or row["live_execution_requested"]
                    ):
                        steps.append(
                            {
                                "step": f"restore:{saved['label']}",
                                "rc": 2,
                                "reason": "agent_changed_missing_or_live_execution_requested",
                            }
                        )
                        continue
                    target = f"{domain}/{row['label']}"
                    if (
                        step(f"enable:{row['label']}", ["launchctl", "enable", target])
                        and runner(root, ["launchctl", "print", target], 10)["rc"] != 0
                    ):
                        step(
                            f"restore:{row['label']}",
                            ["launchctl", "bootstrap", domain, row["plist"]],
                        )
            if all(row["rc"] == 0 for row in steps):
                state.update(requested_state="on", transition="on_requested")
            else:
                write(
                    root,
                    OFF,
                    {
                        "source": "system_power",
                        "reason": "start_incomplete",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                step(
                    "reengage_operator_stop",
                    [
                        ops,
                        "operator-control",
                        "--engage",
                        "--reason",
                        "system_power_start_incomplete",
                        "--json",
                    ],
                )
                state.update(
                    requested_state="off", transition="start_incomplete_requires_review"
                )
        else:
            raise ValueError("unsupported system power action")
        state.update(
            ok=all(row["rc"] == 0 for row in steps)
            and not (action == "off" and state.get("remaining_agents")),
            steps=steps,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        write(root, STATE, state)
        return {"ok": state["ok"], "steps": steps, **status(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("status", "on", "off", "clear-halts"),
        nargs="?",
        default="status",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run(ROOT, args.action)
    except Exception as exc:
        result = {
            "ok": False,
            "reason": type(exc).__name__,
            "live_execution_authority": False,
        }
    print(json.dumps(result, indent=None if args.json else 2))
    return 0 if result.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
