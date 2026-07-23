#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        write_payload,
    )
else:
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        write_payload,
    )


PYTHON_BIN = Path(sys.executable)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "stale_surface_autohealer_latest.json"
DEFAULT_LAUNCHD_LABEL_PATTERNS = (
    "com.dankingsley.schwab.codex.trainingdone",
)
SAFE_PYTHON_SCRIPTS = {
    "scripts/session_ready_check.py",
    "scripts/live_readiness_smoke.py",
    "scripts/ops/artifact_freshness_slo.py",
    "scripts/ops/process_watchdog.py",
    "scripts/build_runtime_training_snapshot.py",
}
SAFE_EXECUTABLE_SCRIPTS = {
    "scripts/ops/opsctl.sh",
}
COMPLETED_NOTICE_NAMES = (
    "codex_training_done_notice_latest.json",
    "codex_training_done_notice_latest.txt",
)
TEMP_MONITOR_PATTERNS = (
    "/private/tmp/schwab_training_done_monitor_*.sh",
    "/tmp/schwab_training_done_monitor_*.sh",
)
SHELL_META_CHARS = set("|;&<>`$")


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _path_age_minutes(path: Path) -> float | None:
    try:
        return max((time.time() - path.stat().st_mtime) / 60.0, 0.0)
    except Exception:
        return None


def _project_relative(path: str | Path, project_root: Path) -> str:
    text = str(path or "").strip()
    if text.startswith("./"):
        text = text[2:]
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(project_root.resolve()).as_posix()
        except Exception:
            return candidate.as_posix()
    return candidate.as_posix()


def _has_shell_meta(parts: list[str]) -> bool:
    return any(any(ch in str(part) for ch in SHELL_META_CHARS) for part in parts)


def _normalize_safe_command(command: Any, project_root: Path) -> tuple[list[str] | None, str]:
    if isinstance(command, str):
        try:
            parts = shlex.split(command)
        except Exception as exc:
            return None, f"shlex_failed:{exc}"
    elif isinstance(command, list):
        parts = [str(part) for part in command if str(part).strip()]
    else:
        return None, "command_not_string_or_list"

    if not parts:
        return None, "empty_command"
    if _has_shell_meta(parts):
        return None, "shell_metacharacters_rejected"

    first = parts[0]
    first_rel = _project_relative(first, project_root)
    if first_rel in SAFE_EXECUTABLE_SCRIPTS:
        return [str(project_root / first_rel), *parts[1:]], "safe_executable_script"
    if first_rel in SAFE_PYTHON_SCRIPTS:
        return [str(PYTHON_BIN), str(project_root / first_rel), *parts[1:]], "safe_python_script"

    if len(parts) >= 2:
        second_rel = _project_relative(parts[1], project_root)
        first_name = Path(first).name.lower()
        if second_rel in SAFE_PYTHON_SCRIPTS and (
            first_name.startswith("python") or first_name in {"python3", "python3.14"}
        ):
            return [first, str(project_root / second_rel), *parts[2:]], "safe_python_wrapper"

    return None, f"command_not_allowlisted:{first}"


def _run_command(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        timed_out = True
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": bool(timed_out),
        "duration_ms": duration_ms,
        "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
    }


def _run_text(cmd: list[str], *, cwd: Path, timeout_sec: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "timed_out": False,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-8:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "timed_out": True,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
        }


def _artifact_refresh_plan(
    project_root: Path,
    *,
    max_artifact_repairs: int,
) -> list[dict[str, Any]]:
    freshness = load_json(project_root / "governance" / "health" / "artifact_freshness_slo_latest.json")
    artifacts = freshness.get("artifacts") if isinstance(freshness.get("artifacts"), list) else []
    plan: list[dict[str, Any]] = []

    if not artifacts and str(freshness.get("overall_status") or "").strip().lower() in {"", "missing", "blocked"}:
        cmd = [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "artifact_freshness_slo.py"), "--json"]
        plan.append(
            {
                "name": "artifact_freshness_slo",
                "surface": "artifact_freshness",
                "reason": "freshness_artifact_missing_or_blocked",
                "action": "run_command",
                "cmd": cmd,
            }
        )
        return plan

    for row in artifacts:
        if not isinstance(row, dict) or not bool(row.get("stale", False)):
            continue
        normalized, policy = _normalize_safe_command(row.get("refresh_command"), project_root)
        entry = {
            "name": str(row.get("name") or "unknown_artifact"),
            "surface": "stale_artifact",
            "reason": "artifact_freshness_slo_stale",
            "required": bool(row.get("required", False)),
            "age_minutes": row.get("age_minutes"),
            "max_age_minutes": row.get("max_age_minutes"),
            "action": "run_command" if normalized else "manual_review",
            "cmd": normalized or [],
            "policy": policy,
        }
        plan.append(entry)
        if len([item for item in plan if item.get("action") == "run_command"]) >= max(int(max_artifact_repairs), 0):
            break
    return plan


def _watchdog_process_plan(project_root: Path, *, max_process_repairs: int) -> list[dict[str, Any]]:
    watchdog = load_json(project_root / "governance" / "health" / "process_watchdog_latest.json")
    intelligence = watchdog.get("watchdog_intelligence") if isinstance(watchdog.get("watchdog_intelligence"), dict) else {}
    needs = intelligence.get("exact_needs") if isinstance(intelligence.get("exact_needs"), list) else []
    plan: list[dict[str, Any]] = []

    for need in needs:
        if not isinstance(need, dict):
            continue
        if str(need.get("status") or "") == "intentional_hold":
            continue
        if str(need.get("severity") or "") == "info":
            continue
        normalized, policy = _normalize_safe_command(need.get("exact_command"), project_root)
        entry = {
            "name": str(need.get("target") or "unknown_target"),
            "surface": "stale_process",
            "reason": str(need.get("blocker") or need.get("reason") or "watchdog_needs_repair"),
            "required": True,
            "action": "run_command" if normalized else "manual_review",
            "cmd": normalized or [],
            "policy": policy,
            "risk_level": str(need.get("risk_level") or ""),
            "restart_storm_quarantinable": bool(need.get("restart_storm_quarantinable", False)),
        }
        plan.append(entry)
        if len([item for item in plan if item.get("action") == "run_command"]) >= max(int(max_process_repairs), 0):
            break
    return plan


def _label_patterns(raw_patterns: str | None = None) -> list[str]:
    values = [x.strip() for x in str(raw_patterns or "").split(",") if x.strip()]
    if not values:
        values = list(DEFAULT_LAUNCHD_LABEL_PATTERNS)
    return values


def _label_timestamp_age_minutes(label: str) -> float | None:
    match = re.search(r"(\d{8})_(\d{6})", str(label or ""))
    if not match:
        return None
    try:
        parsed = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 60.0, 0.0)


def _stale_launchd_plan(
    project_root: Path,
    *,
    max_one_shot_age_minutes: float,
    label_patterns: list[str],
) -> list[dict[str, Any]]:
    result = _run_text(["launchctl", "list"], cwd=project_root, timeout_sec=20)
    if int(result.get("rc", 1)) != 0:
        return [
            {
                "name": "launchctl_list",
                "surface": "stale_one_shot_launchd",
                "reason": "launchctl_list_failed",
                "action": "manual_review",
                "cmd": [],
                "error": str(result.get("stderr_tail") or result.get("stdout_tail") or ""),
            }
        ]
    stdout = str(result.get("stdout") or "")

    plan: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        parts = raw.strip().split(None, 2)
        if len(parts) != 3 or parts[0] == "PID":
            continue
        pid_raw, status_raw, label = parts
        if not any(pattern in label for pattern in label_patterns):
            continue
        label_age = _label_timestamp_age_minutes(label)
        completed_or_unloaded = pid_raw == "-"
        age_stale = label_age is not None and label_age >= float(max_one_shot_age_minutes)
        if not (completed_or_unloaded or age_stale):
            continue
        plan.append(
            {
                "name": label,
                "surface": "stale_one_shot_launchd",
                "reason": "completed_or_stale_one_shot_monitor",
                "required": True,
                "action": "launchctl_remove",
                "cmd": ["launchctl", "remove", label],
                "pid": None if pid_raw == "-" else _safe_int(pid_raw, 0),
                "status": status_raw,
                "label_age_minutes": round(float(label_age), 4) if label_age is not None else None,
                "max_age_minutes": float(max_one_shot_age_minutes),
            }
        )
    return plan


def _completed_notice_stale(path: Path, *, max_age_minutes: float) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".json":
        payload = load_json(path)
        final_state = str(payload.get("state") or payload.get("final_status") or payload.get("status") or "").strip().lower()
        ended = parse_iso_utc(payload.get("ended_utc"))
        if ended is not None and final_state in {"completed", "completed_successfully", "skipped_lock_busy"}:
            age = max((datetime.now(timezone.utc) - ended).total_seconds() / 60.0, 0.0)
            return age >= float(max_age_minutes)
        if final_state in {"completed", "completed_successfully", "skipped_lock_busy"}:
            age = _path_age_minutes(path)
            return age is not None and age >= float(max_age_minutes)
    age = _path_age_minutes(path)
    return age is not None and age >= float(max_age_minutes)


def _stale_file_cleanup_plan(
    project_root: Path,
    *,
    max_completed_notice_age_minutes: float,
    max_temp_monitor_age_minutes: float,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    health_root = project_root / "governance" / "health"
    for name in COMPLETED_NOTICE_NAMES:
        path = health_root / name
        if not _completed_notice_stale(path, max_age_minutes=max_completed_notice_age_minutes):
            continue
        plan.append(
            {
                "name": name,
                "surface": "stale_completion_notice",
                "reason": "completed_training_notice_stale",
                "required": True,
                "action": "unlink",
                "path": str(path),
                "age_minutes": round(float(_path_age_minutes(path) or 0.0), 4),
                "max_age_minutes": float(max_completed_notice_age_minutes),
            }
        )

    for pattern in TEMP_MONITOR_PATTERNS:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            age = _path_age_minutes(path)
            if age is None or age < float(max_temp_monitor_age_minutes):
                continue
            plan.append(
                {
                    "name": path.name,
                    "surface": "stale_temp_monitor_script",
                    "reason": "temporary_one_shot_monitor_script_stale",
                    "required": True,
                    "action": "unlink",
                    "path": str(path),
                    "age_minutes": round(float(age), 4),
                    "max_age_minutes": float(max_temp_monitor_age_minutes),
                }
            )
    return plan


def _apply_plan_row(row: dict[str, Any], *, project_root: Path, timeout_sec: int) -> dict[str, Any]:
    action = str(row.get("action") or "")
    if action in {"run_command", "launchctl_remove"}:
        cmd = [str(part) for part in row.get("cmd") or []]
        if not cmd:
            return {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": action,
                "required": bool(row.get("required", True)),
                "rc": 2,
                "error": "empty_command",
            }
        result = _run_command(cmd, cwd=project_root, timeout_sec=timeout_sec)
        result.update(
            {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": action,
                "required": bool(row.get("required", True)),
            }
        )
        return result
    if action == "unlink":
        path = Path(str(row.get("path") or ""))
        try:
            if path.exists():
                path.unlink()
            return {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": action,
                "required": bool(row.get("required", True)),
                "rc": 0,
                "deleted": True,
                "path": str(path),
            }
        except Exception as exc:
            return {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": action,
                "required": bool(row.get("required", True)),
                "rc": 1,
                "deleted": False,
                "path": str(path),
                "error": str(exc),
            }
    return {
        "name": row.get("name"),
        "surface": row.get("surface"),
        "action": action,
        "required": bool(row.get("required", True)),
        "rc": 2,
        "error": "manual_or_unknown_action",
    }


def _refresh_inputs(project_root: Path, *, timeout_sec: int) -> list[dict[str, Any]]:
    commands = [
        [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "artifact_freshness_slo.py"), "--json"],
        [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "process_watchdog.py"), "--json"],
    ]
    return [_run_command(cmd, cwd=project_root, timeout_sec=min(int(timeout_sec), 180)) for cmd in commands]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 180,
    max_artifact_repairs: int = 6,
    max_process_repairs: int = 5,
    max_one_shot_age_minutes: float = 180.0,
    max_completed_notice_age_minutes: float = 30.0,
    max_temp_monitor_age_minutes: float = 180.0,
    launchd_label_patterns: list[str] | None = None,
    refresh_inputs: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    input_refresh_attempts: list[dict[str, Any]] = []
    if apply and refresh_inputs:
        input_refresh_attempts = _refresh_inputs(project_root, timeout_sec=timeout_sec)

    plan: list[dict[str, Any]] = []
    plan.extend(_artifact_refresh_plan(project_root, max_artifact_repairs=max_artifact_repairs))
    plan.extend(_watchdog_process_plan(project_root, max_process_repairs=max_process_repairs))
    plan.extend(
        _stale_launchd_plan(
            project_root,
            max_one_shot_age_minutes=max_one_shot_age_minutes,
            label_patterns=launchd_label_patterns or _label_patterns(os.getenv("STALE_SURFACE_AUTOHEAL_LAUNCHD_LABEL_PATTERNS")),
        )
    )
    plan.extend(
        _stale_file_cleanup_plan(
            project_root,
            max_completed_notice_age_minutes=max_completed_notice_age_minutes,
            max_temp_monitor_age_minutes=max_temp_monitor_age_minutes,
        )
    )

    applyable = [row for row in plan if row.get("action") in {"run_command", "launchctl_remove", "unlink"}]
    manual_review = [row for row in plan if row.get("action") == "manual_review"]
    attempts: list[dict[str, Any]] = []
    if apply:
        for row in applyable:
            attempts.append(_apply_plan_row(row, project_root=project_root, timeout_sec=timeout_sec))
    post_refresh_attempts: list[dict[str, Any]] = []
    if apply:
        post_refresh_attempts = _refresh_inputs(project_root, timeout_sec=timeout_sec)

    failed_attempts = [
        row
        for row in attempts
        if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2}
    ]
    hard_failed_attempts = [
        row
        for row in failed_attempts
        if bool(row.get("required", True)) or str(row.get("surface") or "") != "stale_artifact"
    ]
    overall_status = "ready"
    if hard_failed_attempts:
        overall_status = "blocked"
    elif failed_attempts or manual_review or applyable:
        overall_status = "degraded"

    surfaces = ordered_unique([str(row.get("surface") or "") for row in plan])
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready" or (apply and not hard_failed_attempts and not manual_review),
        "overall_status": overall_status,
        "apply": bool(apply),
        "repair_plan": plan,
        "attempts": [
            {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": row.get("action"),
                "required": bool(row.get("required", True)),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in attempts
        ],
        "input_refresh_attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in input_refresh_attempts
        ],
        "post_refresh_attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in post_refresh_attempts
        ],
        "failed_attempts": [
            {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": row.get("action"),
                "required": bool(row.get("required", True)),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in failed_attempts
        ],
        "hard_failed_attempts": [
            {
                "name": row.get("name"),
                "surface": row.get("surface"),
                "action": row.get("action"),
                "required": bool(row.get("required", True)),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in hard_failed_attempts
        ],
        "metrics": {
            "planned_repair_count": len(plan),
            "applyable_repair_count": len(applyable),
            "manual_review_count": len(manual_review),
            "artifact_repair_count": sum(1 for row in plan if row.get("surface") == "stale_artifact"),
            "process_repair_count": sum(1 for row in plan if row.get("surface") == "stale_process"),
            "launchd_cleanup_count": sum(1 for row in plan if row.get("surface") == "stale_one_shot_launchd"),
            "file_cleanup_count": sum(1 for row in plan if row.get("action") == "unlink"),
        },
        "surfaces": surfaces,
        "recommended_actions": ordered_unique(
            [
                "keep stale-surface autohealing inside infrastructure-autofix so stale artifacts and stale watchdog targets repair during the soak",
                "treat manual_review rows as guardrails; they mean a stale surface advertised a non-allowlisted command",
                "keep one-shot Codex monitor cleanup enabled so old completion notices cannot keep reappearing",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely auto-heal stale artifacts, stale watchdog targets, and stale one-shot monitor leftovers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-artifact-repairs", type=int, default=6)
    parser.add_argument("--max-process-repairs", type=int, default=5)
    parser.add_argument("--one-shot-max-age-minutes", type=float, default=180.0)
    parser.add_argument("--completed-notice-max-age-minutes", type=float, default=30.0)
    parser.add_argument("--temp-monitor-max-age-minutes", type=float, default=180.0)
    parser.add_argument("--launchd-label-patterns", default="")
    parser.add_argument("--no-refresh-inputs", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        timeout_sec=int(args.timeout_sec),
        max_artifact_repairs=int(args.max_artifact_repairs),
        max_process_repairs=int(args.max_process_repairs),
        max_one_shot_age_minutes=float(args.one_shot_max_age_minutes),
        max_completed_notice_age_minutes=float(args.completed_notice_max_age_minutes),
        max_temp_monitor_age_minutes=float(args.temp_monitor_max_age_minutes),
        launchd_label_patterns=_label_patterns(args.launchd_label_patterns),
        refresh_inputs=not bool(args.no_refresh_inputs),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        print(
            "stale_surface_autohealer "
            f"overall_status={payload.get('overall_status', '')} "
            f"planned={int(metrics.get('planned_repair_count', 0) or 0)} "
            f"applyable={int(metrics.get('applyable_repair_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
