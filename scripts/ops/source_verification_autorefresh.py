#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
    from scripts.ops import source_verification_report as report_src
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload
    from . import source_verification_report as report_src


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "source_verification_autorefresh_latest.json"


def _command_key(command: list[str]) -> tuple[str, ...]:
    return tuple(str(part) for part in command)


def _run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_seconds), 1),
        )
        return {
            "command": [str(part) for part in command],
            "rc": int(proc.returncode),
            "ok": proc.returncode == 0,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-8:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "command": [str(part) for part in command],
            "rc": 124,
            "ok": False,
            "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "command": [str(part) for part in command],
            "rc": 1,
            "ok": False,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "timed_out": False,
        }


def _write_latest_source_report(project_root: Path, payload: dict[str, Any]) -> None:
    health = project_root / "governance" / "health"
    reports = project_root / "exports" / "reports"
    health.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    (health / "source_verification_latest.json").write_text(text, encoding="utf-8")
    (reports / "source_verification_latest.md").write_text(report_src._render_markdown(payload), encoding="utf-8")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    max_commands: int = 8,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    before = report_src.build_source_verification_payload(project_root)
    commands = before.get("recommended_refresh_commands") if isinstance(before.get("recommended_refresh_commands"), list) else []
    unique_commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in commands:
        if not isinstance(raw, list):
            continue
        command = [str(part) for part in raw if str(part).strip()]
        if not command:
            continue
        key = _command_key(command)
        if key in seen:
            continue
        seen.add(key)
        unique_commands.append(command)

    refresh_commands = [
        command
        for command in unique_commands
        if not any(str(part) == "source-verification" for part in command)
    ][: max(int(max_commands), 0)]
    results: list[dict[str, Any]] = []
    after = before
    if apply and refresh_commands:
        for command in refresh_commands:
            results.append(_run_command(command, cwd=project_root, timeout_seconds=int(timeout_seconds)))
        after = report_src.build_source_verification_payload(project_root)
        _write_latest_source_report(project_root, after)
    elif apply:
        _write_latest_source_report(project_root, after)

    failed = [row for row in results if not bool(row.get("ok", False))]
    status = "ready" if bool(after.get("ok", False)) else "needs_refresh"
    if apply and failed:
        status = "applied_with_failures"
    elif apply and results:
        status = "applied" if bool(after.get("ok", False)) else "applied_still_degraded"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status in {"ready", "applied"},
        "overall_status": status,
        "apply": bool(apply),
        "before": {
            "overall_status": str(before.get("overall_status") or ""),
            "unverified_sources": list(before.get("unverified_sources") or []),
            "stale_artifacts": list(before.get("stale_artifacts") or []),
            "degraded_artifacts": list(before.get("degraded_artifacts") or []),
        },
        "after": {
            "overall_status": str(after.get("overall_status") or ""),
            "unverified_sources": list(after.get("unverified_sources") or []),
            "stale_artifacts": list(after.get("stale_artifacts") or []),
            "degraded_artifacts": list(after.get("degraded_artifacts") or []),
        },
        "planned_commands": unique_commands,
        "applied_commands": refresh_commands if apply else [],
        "results": results,
        "recommended_actions": [
            "apply source-verification-refresh to refresh degraded artifacts"
            if not apply and unique_commands
            else "rerun source-verification after failed refresh commands"
            if failed
            else "source verification autorefresh completed",
        ],
        "source_verification": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh stale/degraded source-verification artifacts and rerun the verification report.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-commands", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        max_commands=int(args.max_commands),
        timeout_seconds=int(args.timeout_seconds),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "source_verification_autorefresh "
            f"status={payload.get('overall_status', '')} "
            f"planned={len(payload.get('planned_commands') or [])} "
            f"applied={len(payload.get('applied_commands') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_refresh", "applied", "applied_still_degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
