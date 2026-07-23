#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import system_drift_guard
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from . import system_drift_guard
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_drift_autopilot_latest.json"
Runner = Callable[[list[str], Path, int], dict[str, Any]]
GuardBuilder = Callable[[Path], dict[str, Any]]
LEAK_CLEANUP_ROUTE_MARKERS = {
    "one-numbers-regression-guard": (
        "scripts/ops/one_numbers_regression_guard.py",
        "scripts/build_one_numbers_report.py",
    ),
    "one_numbers_regression_guard.py": (
        "scripts/ops/one_numbers_regression_guard.py",
        "scripts/build_one_numbers_report.py",
    ),
    "report-pdfs": (
        "report-bundle-pdf",
        "--headless",
        "--print-to-pdf=",
    ),
    "report_pdf": (
        "report-bundle-pdf",
        "--headless",
        "--print-to-pdf=",
    ),
}


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bounded_timeout(raw: Any, *, max_step_timeout_sec: int) -> int:
    timeout = _safe_int(raw, max_step_timeout_sec)
    return max(1, min(timeout, max(int(max_step_timeout_sec), 1)))


def _cleanup_markers(cmd: list[str]) -> list[str]:
    cmd_text = " ".join(str(part) for part in cmd)
    markers: list[str] = []
    for route_marker, child_markers in LEAK_CLEANUP_ROUTE_MARKERS.items():
        if route_marker in cmd_text:
            markers.extend(child_markers)
    return ordered_unique(markers)


def _project_processes(project_root: Path, markers: list[str]) -> dict[int, int]:
    if not markers:
        return {}
    try:
        proc = subprocess.run(
            ["ps", "-ax", "-o", "pid=,pgid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return {}
    root_text = str(project_root)
    current_pid = os.getpid()
    current_pgid = os.getpgrp()
    rows: dict[int, int] = {}
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or root_text not in line:
            continue
        if not any(marker in line for marker in markers):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except Exception:
            continue
        if pid == current_pid or pgid == current_pgid:
            continue
        rows[pid] = pgid
    return rows


def _terminate_leaked_processes(
    project_root: Path,
    markers: list[str],
    before_pids: set[int],
    *,
    sig: int = signal.SIGTERM,
) -> list[int]:
    leaked = {
        pid: pgid
        for pid, pgid in _project_processes(project_root, markers).items()
        if pid not in before_pids
    }
    if not leaked:
        return []
    terminated: list[int] = []
    pgids = ordered_unique([str(pgid) for pgid in leaked.values() if pgid > 0])
    for raw_pgid in pgids:
        pgid = int(raw_pgid)
        try:
            os.killpg(pgid, sig)
            terminated.append(-pgid)
        except ProcessLookupError:
            continue
        except Exception:
            continue
    remaining = {
        pid: pgid
        for pid, pgid in _project_processes(project_root, markers).items()
        if pid not in before_pids
    }
    if remaining:
        for pid in remaining:
            try:
                os.kill(pid, sig)
                terminated.append(pid)
            except ProcessLookupError:
                pass
            except Exception:
                pass
    return terminated


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    timeout_sec = max(int(timeout_sec), 1)
    proc: subprocess.Popen[str] | None = None
    cleanup_markers = _cleanup_markers(cmd)
    preexisting_cleanup_pids = set(_project_processes(project_root, cleanup_markers))
    cleanup_pids: list[int] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "payload": _parse_json_output(stdout or ""),
            "stdout_tail": "\n".join((stdout or "").splitlines()[-12:]),
            "stderr_tail": "\n".join((stderr or "").splitlines()[-12:]),
        }
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = ""
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                proc.terminate()
            cleanup_pids.extend(_terminate_leaked_processes(project_root, cleanup_markers, preexisting_cleanup_pids))
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    proc.kill()
                cleanup_pids.extend(
                    _terminate_leaked_processes(
                        project_root,
                        cleanup_markers,
                        preexisting_cleanup_pids,
                        sig=signal.SIGKILL,
                    )
                )
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    if proc.stdout is not None:
                        proc.stdout.close()
                    if proc.stderr is not None:
                        proc.stderr.close()
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass
        return {
            "cmd": list(cmd),
            "rc": 124,
            "payload": _parse_json_output(stdout),
            "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-12:]) or "timeout",
            "timeout_cleanup": {
                "markers": cleanup_markers,
                "terminated_processes": cleanup_pids,
            },
        }


def _repair_plan(guard_payload: dict[str, Any], *, max_steps: int) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    ranked_rows = sorted(
        [row for row in list(guard_payload.get("surfaces") or []) if isinstance(row, dict)],
        key=lambda row: (
            0 if str(row.get("status") or "") in {"blocked", "critical", "missing"} else 1,
            str(row.get("family") or ""),
            str(row.get("name") or ""),
        ),
    )

    for row in ranked_rows:
        status = str(row.get("status") or "")
        if status == "ready":
            continue
        commands = row.get("repair_commands") if isinstance(row.get("repair_commands"), list) else []
        for cmd in commands:
            if not isinstance(cmd, list) or not cmd:
                continue
            fingerprint = " ".join(str(part) for part in cmd)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            plan.append(
                {
                    "surface": str(row.get("name") or ""),
                    "family": str(row.get("family") or ""),
                    "reason": status,
                    "cmd": [str(part) for part in cmd],
                    "timeout_sec": 1200,
                    "recovery_deferred": bool(row.get("recovery_deferred", False)),
                    "recovery_deferred_reason": str(row.get("recovery_deferred_reason") or ""),
                }
            )
            if len(plan) >= max(int(max_steps), 1):
                return plan
    return plan


def _filter_plan_for_recovery_safety(
    repair_plan: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for step in repair_plan:
        if bool(step.get("recovery_deferred", False)):
            skipped.append(
                {
                    **step,
                    "skip_reason": "recovery_deferred",
                    "recovery_deferred_reason": str(step.get("recovery_deferred_reason") or ""),
                }
            )
            continue
        filtered.append(step)
    return filtered, skipped


def _surface_status(guard_payload: dict[str, Any], surface_name: str) -> str:
    for row in list(guard_payload.get("surfaces") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "") == surface_name:
            return str(row.get("status") or "")
    return "missing"


def _filter_plan_for_workstation_safety(
    repair_plan: list[dict[str, Any]],
    *,
    chrome_guard_status: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(chrome_guard_status or "") == "ready":
        return repair_plan, []
    filtered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for step in repair_plan:
        if str(step.get("family") or "") == "reporting_surface":
            skipped.append(
                {
                    **step,
                    "skip_reason": "chrome_guard_not_ready",
                    "chrome_guard_status": str(chrome_guard_status or "missing"),
                }
            )
            continue
        filtered.append(step)
    return filtered, skipped


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    max_steps: int = 12,
    max_step_timeout_sec: int = 300,
    runner: Runner | None = None,
    guard_builder: GuardBuilder | None = None,
) -> dict[str, Any]:
    run_step = runner or _run
    build_guard = guard_builder or system_drift_guard.build_payload

    initial_guard = build_guard(project_root)
    planned_repair_steps = _repair_plan(initial_guard, max_steps=max_steps)
    chrome_guard_status = _surface_status(initial_guard, "chrome_headless_guard")
    repair_plan, workstation_skipped_steps = _filter_plan_for_workstation_safety(
        planned_repair_steps,
        chrome_guard_status=chrome_guard_status,
    )
    skipped_steps = list(workstation_skipped_steps)
    repair_plan, recovery_skipped_steps = _filter_plan_for_recovery_safety(repair_plan)
    skipped_steps.extend(recovery_skipped_steps)
    attempts: list[dict[str, Any]] = []
    if apply:
        for step in repair_plan:
            timeout_sec = _bounded_timeout(step.get("timeout_sec", 1200), max_step_timeout_sec=max_step_timeout_sec)
            result = run_step(list(step["cmd"]), project_root, timeout_sec)
            payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            attempts.append(
                {
                    "surface": step["surface"],
                    "family": step["family"],
                    "reason": step["reason"],
                    "cmd": list(result.get("cmd") or []),
                    "rc": int(result.get("rc", 1)),
                    "timeout_sec": timeout_sec,
                    "payload_summary": {
                        key: payload.get(key)
                        for key in (
                            "overall_status",
                            "ok",
                            "repair_step_count",
                            "attempt_count",
                            "below_floor_count",
                        )
                        if key in payload
                    },
                    "stdout_tail": str(result.get("stdout_tail") or ""),
                    "stderr_tail": str(result.get("stderr_tail") or ""),
                    "timeout_cleanup": result.get("timeout_cleanup") if isinstance(result.get("timeout_cleanup"), dict) else {},
                }
            )

    final_guard = build_guard(project_root)
    operator_followups = [
        str(row.get("name") or "")
        for row in list(final_guard.get("surfaces") or [])
        if isinstance(row, dict) and str(row.get("status") or "") in {"blocked", "critical", "missing"} and not row.get("repair_commands")
    ]
    recommended_actions = ordered_unique(
        [
            "leave the system drift autopilot on a timer so safe repairs happen before contracts rot"
            if apply and repair_plan
            else "",
            "defer PDF/report repairs until the Chrome headless guard is ready so workstation recovery does not respawn the browser storm"
            if workstation_skipped_steps
            else "",
            "leave recovery-deferred drift surfaces to their owning guards instead of spending the drift repair budget on active bounded recovery"
            if recovery_skipped_steps
            else "",
            "review operator-gated or no-repair drift surfaces manually because the autopilot intentionally will not invent destructive fixes"
            if operator_followups
            else "",
        ]
        + [str(item or "") for item in (final_guard.get("recommended_actions") or [])]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": str(final_guard.get("overall_status") or "") == "ready",
        "overall_status": str(final_guard.get("overall_status") or ""),
        "apply": bool(apply),
        "max_steps": int(max_steps),
        "max_step_timeout_sec": int(max_step_timeout_sec),
        "repair_step_count": len(repair_plan),
        "planned_repair_step_count": len(planned_repair_steps),
        "skipped_step_count": len(skipped_steps),
        "attempt_count": len(attempts),
        "chrome_guard_status": chrome_guard_status,
        "initial_guard": {
            "overall_status": str(initial_guard.get("overall_status") or ""),
            "blocked_surface_count": int(((initial_guard.get("metrics") or {}).get("blocked_surface_count")) or 0),
            "degraded_surface_count": int(((initial_guard.get("metrics") or {}).get("degraded_surface_count")) or 0),
        },
        "final_guard": {
            "overall_status": str(final_guard.get("overall_status") or ""),
            "blocked_surface_count": int(((final_guard.get("metrics") or {}).get("blocked_surface_count")) or 0),
            "degraded_surface_count": int(((final_guard.get("metrics") or {}).get("degraded_surface_count")) or 0),
        },
        "repair_plan": repair_plan,
        "skipped_steps": skipped_steps,
        "attempts": attempts,
        "operator_followups": operator_followups,
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "system_drift_autopilot_v1",
            "co_managed_with": [
                "system_drift_guard",
                "infrastructure_autofix_bot",
                "command_validity_bot",
                "section_grade_autopilot",
            ],
            "future_upgrade_paths": [
                "cooldown budgets per surface family so noisy lanes retry less aggressively than brittle control surfaces",
                "window-aware repair sequencing that prefers off-hours PDF and reporting rebuilds",
                "tenant-facing drift alerts from the partner API once the command and reporting surfaces stabilize",
            ],
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair system-wide drift surfaces from the registry-backed drift guard.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--max-step-timeout-seconds",
        type=int,
        default=_safe_int(os.getenv("SYSTEM_DRIFT_AUTOPILOT_STEP_TIMEOUT_SECONDS"), 300),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        max_steps=int(args.max_steps),
        max_step_timeout_sec=int(args.max_step_timeout_seconds),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_drift_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_step_count={payload.get('repair_step_count', 0)} "
            f"attempt_count={payload.get('attempt_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
