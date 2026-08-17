#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "chrome_headless_guard_latest.json"
DEFAULT_PROJECT_TIMELINE_LOCK = PROJECT_ROOT / "governance" / "locks" / "project_timeline_report.lock"
DEFAULT_MAX_HEADLESS_COUNT = 4
DEFAULT_STALE_HEADLESS_AGE_SECONDS = 900
DEFAULT_ORPHAN_GRACE_SECONDS = 120
DEFAULT_RUNAWAY_HEADLESS_AGE_SECONDS = 180
HEADLESS_PROFILE_MARKERS = (
    "project-timeline-pdf-",
    "report-bundle-pdf-open-",
    "system-summary-pdf-",
)

KillRunner = Callable[[int], bool]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _run_capture(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return completed.stdout or ""


def _parse_elapsed_seconds(raw: str) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = _safe_int(day_part, 0)
    parts = [segment for segment in text.split(":") if segment != ""]
    if not parts:
        return 0
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(parts[0]), int(parts[1]))
        else:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
    except Exception:
        return 0
    return max(((days * 24 + hours) * 60 + minutes) * 60 + seconds, 0)


def _parse_process_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid_text, ppid_text, elapsed, command = parts
        pid = _safe_int(pid_text, 0)
        ppid = _safe_int(ppid_text, 0)
        if pid <= 0 or not command:
            continue
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "elapsed": elapsed,
                "elapsed_seconds": _parse_elapsed_seconds(elapsed),
                "command": command,
            }
        )
    return rows


def _collect_process_rows() -> list[dict[str, Any]]:
    ps_text = _run_capture(["ps", "-axo", "pid,ppid,etime,command"])
    return _parse_process_rows(ps_text)


def _lock_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "age_seconds": None}
    try:
        mtime = path.stat().st_mtime
        age_seconds = max(int(round(time.time() - mtime)), 0)
    except Exception:
        age_seconds = None
    return {
        "present": True,
        "age_seconds": age_seconds,
        "path": str(path),
    }


def _is_headless_command(command: str) -> bool:
    lowered = str(command or "").lower()
    return (
        "--headless" in lowered
        or "google chrome helper --headless" in lowered
        or "chromium --headless" in lowered
        or _uses_temp_headless_profile(command)
    )


def _is_interactive_chrome_command(command: str) -> bool:
    lowered = str(command or "").lower()
    if "chrome_crashpad_handler" in lowered:
        return False
    if _uses_temp_headless_profile(command):
        return False
    if "google chrome helper --headless" in lowered or "--headless" in lowered:
        return False
    return "google chrome" in lowered


def _is_timeline_render_command(command: str) -> bool:
    lowered = str(command or "").lower()
    return (
        "project_timeline_report.py" in lowered
        or "project_timeline_autoupdate" in lowered
        or "report_pdf_bundle.py" in lowered
        or "report-bundle-pdf-open" in lowered
    )


def _uses_temp_headless_profile(command: str) -> bool:
    lowered = str(command or "").lower()
    return any(marker in lowered for marker in HEADLESS_PROFILE_MARKERS)


def _default_kill_runner(pid: int) -> bool:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        return False
    return True


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    process_rows: list[dict[str, Any]] | None = None,
    apply: bool = False,
    max_headless_count: int = DEFAULT_MAX_HEADLESS_COUNT,
    stale_headless_age_seconds: int = DEFAULT_STALE_HEADLESS_AGE_SECONDS,
    orphan_grace_seconds: int = DEFAULT_ORPHAN_GRACE_SECONDS,
    runaway_headless_age_seconds: int = DEFAULT_RUNAWAY_HEADLESS_AGE_SECONDS,
    kill_runner: KillRunner | None = None,
) -> dict[str, Any]:
    rows = list(process_rows or _collect_process_rows())
    pid_set = {int(row.get("pid", 0) or 0) for row in rows}
    headless_rows = [row for row in rows if _is_headless_command(str(row.get("command") or ""))]
    interactive_rows = [row for row in rows if _is_interactive_chrome_command(str(row.get("command") or ""))]
    timeline_rows = [row for row in rows if _is_timeline_render_command(str(row.get("command") or ""))]
    timeline_lock = _lock_snapshot(project_root / "governance" / "locks" / "project_timeline_report.lock")
    quiet_mode = bool(
        _env_flag("CHROME_HEADLESS_QUIET_MODE", "0")
        or not _env_flag("REPORT_HEADLESS_BROWSER_RENDER_ENABLED", "1")
    )

    stale_rows: list[dict[str, Any]] = []
    orphan_rows: list[dict[str, Any]] = []
    runaway_rows: list[dict[str, Any]] = []
    runaway_cleanup_rows: list[dict[str, Any]] = []
    timeline_cleanup_rows: list[dict[str, Any]] = []
    for row in headless_rows:
        elapsed_seconds = _safe_int(row.get("elapsed_seconds"), 0)
        ppid = _safe_int(row.get("ppid"), 0)
        orphan = bool(ppid <= 1 or ppid not in pid_set)
        if orphan and elapsed_seconds >= max(int(orphan_grace_seconds), 0):
            orphan_rows.append(row)
        if elapsed_seconds >= max(int(stale_headless_age_seconds), 0):
            stale_rows.append(row)
        if elapsed_seconds >= max(int(runaway_headless_age_seconds), 0):
            runaway_rows.append(row)
            if _uses_temp_headless_profile(str(row.get("command") or "")):
                runaway_cleanup_rows.append(row)
    for row in timeline_rows:
        elapsed_seconds = _safe_int(row.get("elapsed_seconds"), 0)
        command = str(row.get("command") or "")
        if elapsed_seconds >= max(int(runaway_headless_age_seconds), 0) and (
            "project_timeline_report.py" in command or "report_pdf_bundle.py" in command
        ):
            timeline_cleanup_rows.append(row)

    lock_recent = bool(
        timeline_lock.get("present")
        and _safe_int(timeline_lock.get("age_seconds"), stale_headless_age_seconds + 1) <= max(int(stale_headless_age_seconds), 1)
    )
    runaway_detected = bool(len(headless_rows) > max(int(max_headless_count), 0))
    runaway_without_lock = bool(runaway_detected and not lock_recent)
    timeline_pdf_policy = "allow"
    if interactive_rows or timeline_rows:
        timeline_pdf_policy = "headless_only"
    if stale_rows or orphan_rows or runaway_detected:
        timeline_pdf_policy = "suppress"
    if quiet_mode:
        timeline_pdf_policy = "suppress"

    cleanup_rows: list[dict[str, Any]] = []
    cleanup_rows.extend(orphan_rows)
    cleanup_rows.extend(stale_rows)
    if runaway_detected:
        cleanup_rows.extend(runaway_rows if not lock_recent else runaway_cleanup_rows)
        cleanup_rows.extend(timeline_cleanup_rows)
    if quiet_mode:
        cleanup_rows.extend(headless_rows)
        cleanup_rows.extend(timeline_rows)

    kill_candidates = ordered_unique([str(row.get("pid") or "") for row in cleanup_rows if row])
    kill_attempts: list[dict[str, Any]] = []
    applied_kill_count = 0
    if apply and kill_candidates:
        runner = kill_runner or _default_kill_runner
        for raw_pid in kill_candidates:
            pid = _safe_int(raw_pid, 0)
            if pid <= 0:
                continue
            ok = bool(runner(pid))
            if ok:
                applied_kill_count += 1
            kill_attempts.append({"pid": pid, "ok": ok})

    overall_status = "ready"
    if quiet_mode and not (stale_rows or orphan_rows or runaway_detected):
        overall_status = "ready"
    elif stale_rows or orphan_rows or runaway_detected:
        overall_status = "blocked"
    elif interactive_rows or headless_rows:
        overall_status = "degraded"

    policy_reason = "allow_normal_render"
    if quiet_mode:
        policy_reason = "quiet_mode_suppressed"
    elif timeline_pdf_policy == "headless_only":
        policy_reason = "interactive_chrome_protected"
    elif timeline_pdf_policy == "suppress":
        policy_reason = "stale_or_orphan_headless_suppressed"

    recommended_actions = ordered_unique(
        [
            "quiet mode is active, so automatic report PDF renders are suppressed until the override is removed" if quiet_mode else "",
            "keep project timeline auto-update on a headless-only path while Chrome is in interactive use" if interactive_rows else "",
            "suppress timeline PDF auto-render until stale or orphan headless helpers stop respawning" if timeline_pdf_policy == "suppress" else "",
            "keep report PDF rebuilds deferred while Chrome headless helpers are above the workstation budget" if runaway_detected else "",
            "reinstall the timeline autoupdate launcher so it respects the Chrome guard policy wrapper" if timeline_rows else "",
            "upgrade this Chrome guard alongside runtime-throttle and autonomy so foreground browser use stays protected from support jobs",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "headless_process_count": len(headless_rows),
        "interactive_chrome_process_count": len(interactive_rows),
        "timeline_runner_count": len(timeline_rows),
        "stale_headless_count": len(stale_rows),
        "orphan_headless_count": len(orphan_rows),
        "runaway_cleanup_count": len(runaway_cleanup_rows),
        "timeline_cleanup_count": len(timeline_cleanup_rows),
        "temp_headless_profile_count": sum(1 for row in headless_rows if _uses_temp_headless_profile(str(row.get("command") or ""))),
        "runaway_detected": runaway_detected,
        "runaway_without_lock": runaway_without_lock,
        "quiet_mode_active": quiet_mode,
        "timeline_lock": timeline_lock,
        "timeline_pdf_policy": timeline_pdf_policy,
        "policy_reason": policy_reason,
        "interactive_protection_active": bool(interactive_rows and timeline_pdf_policy in {"headless_only", "suppress"}),
        "timeline_autorender_suppressed": timeline_pdf_policy == "suppress",
        "killed_pid_count": applied_kill_count,
        "kill_candidates": [_safe_int(pid, 0) for pid in kill_candidates if _safe_int(pid, 0) > 0],
        "kill_attempts": kill_attempts,
        "headless_processes": headless_rows[:12],
        "interactive_chrome_processes": interactive_rows[:6],
        "timeline_processes": timeline_rows[:6],
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "chrome_headless_guard_v3",
            "co_managed_with": [
                "runtime_throttle_control",
                "autonomy_control_plane",
                "project_timeline_autoupdate",
            ],
            "future_upgrade_paths": [
                "launchd policy handoff that disables PDF autoupdate during interactive browser sessions and runaway headless budgets",
                "per-render-job parent tracking instead of process-name heuristics",
                "partner API exposure for workstation health and browser contention signals",
            ],
        },
        "recommended_actions": recommended_actions,
        "source_contract": {
            "process_source": "ps -axo pid,ppid,etime,command",
            "project_timeline_lock": str(project_root / "governance" / "locks" / "project_timeline_report.lock"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Protect foreground Chrome usage from stale or runaway headless render helpers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-headless-count", type=int, default=DEFAULT_MAX_HEADLESS_COUNT)
    parser.add_argument("--stale-headless-age-seconds", type=int, default=DEFAULT_STALE_HEADLESS_AGE_SECONDS)
    parser.add_argument("--orphan-grace-seconds", type=int, default=DEFAULT_ORPHAN_GRACE_SECONDS)
    parser.add_argument("--runaway-headless-age-seconds", type=int, default=DEFAULT_RUNAWAY_HEADLESS_AGE_SECONDS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        max_headless_count=int(args.max_headless_count),
        stale_headless_age_seconds=int(args.stale_headless_age_seconds),
        orphan_grace_seconds=int(args.orphan_grace_seconds),
        runaway_headless_age_seconds=int(args.runaway_headless_age_seconds),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "chrome_headless_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"headless_process_count={int(payload.get('headless_process_count', 0) or 0)} "
            f"timeline_pdf_policy={payload.get('timeline_pdf_policy', '')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
