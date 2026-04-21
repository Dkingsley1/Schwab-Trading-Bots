#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import backpressure_slo_bot as backpressure_src
    from scripts.ops import retention_debt_sheriff as sheriff_src
    from scripts.ops import writer_cycle_coordinator as coordinator_src
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from . import backpressure_slo_bot as backpressure_src
    from . import retention_debt_sheriff as sheriff_src
    from . import writer_cycle_coordinator as coordinator_src
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_backpressure_autopilot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_backpressure_autopilot.lock"
PYTHON_BIN = Path(sys.executable)
ACCEPTED_STEP_STATUSES = {
    "already_running",
    "applied",
    "applied_with_followups",
    "idle",
    "ready",
    "stable",
    "waiting_for_writer",
}
DEFAULT_MAX_CYCLES = 3
DEFAULT_TARGET_PENDING_LINES = 20000
DEFAULT_TARGET_RETENTION_DEBT_GB = 0.25
MIN_PENDING_PROGRESS_LINES = 250
MIN_RETENTION_PROGRESS_GB = 0.05


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def _status_rank(value: str) -> int:
    return {
        "blocked": 0,
        "needs_work": 1,
        "ready": 2,
    }.get(str(value or "").strip(), 0)


def _severity_rank(value: str) -> int:
    return {
        "critical": 0,
        "high": 1,
        "elevated": 2,
        "stable": 3,
    }.get(str(value or "").strip(), 0)


def _storage_snapshot(storage_control: dict[str, Any]) -> dict[str, Any]:
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage = storage_control.get("storage") if isinstance(storage_control.get("storage"), dict) else {}
    return {
        "overall_status": str(storage_control.get("overall_status") or ""),
        "severity": str(storage_control.get("severity") or ""),
        "total_pending_lines": _safe_int(backpressure.get("total_pending_lines"), 0),
        "retention_debt_gb": round(_safe_float(storage.get("retention_debt_gb"), 0.0), 3),
    }


def _clearance_state(
    storage_control: dict[str, Any],
    *,
    target_pending_lines: int,
    target_retention_debt_gb: float,
) -> dict[str, Any]:
    snapshot = _storage_snapshot(storage_control)
    cleared = bool(
        snapshot["total_pending_lines"] <= max(int(target_pending_lines), 0)
        and float(snapshot["retention_debt_gb"]) <= max(float(target_retention_debt_gb), 0.0)
        and str(snapshot["overall_status"] or "") != "blocked"
        and str(snapshot["severity"] or "") != "critical"
    )
    return {
        **snapshot,
        "target_pending_lines": max(int(target_pending_lines), 0),
        "target_retention_debt_gb": round(max(float(target_retention_debt_gb), 0.0), 3),
        "cleared": cleared,
    }


def _clearance_progress(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    pending_delta = max(_safe_int(before.get("total_pending_lines"), 0) - _safe_int(after.get("total_pending_lines"), 0), 0)
    retention_delta = max(
        round(_safe_float(before.get("retention_debt_gb"), 0.0) - _safe_float(after.get("retention_debt_gb"), 0.0), 3),
        0.0,
    )
    status_improved = _status_rank(str(after.get("overall_status") or "")) > _status_rank(str(before.get("overall_status") or ""))
    severity_improved = _severity_rank(str(after.get("severity") or "")) > _severity_rank(str(before.get("severity") or ""))
    progress_observed = bool(
        pending_delta >= MIN_PENDING_PROGRESS_LINES
        or retention_delta >= MIN_RETENTION_PROGRESS_GB
        or status_improved
        or severity_improved
    )
    return {
        "progress_observed": progress_observed,
        "pending_lines_reduced": int(pending_delta),
        "retention_debt_reduced_gb": round(retention_delta, 3),
        "status_improved": status_improved,
        "severity_improved": severity_improved,
        "before": before,
        "after": after,
    }


def _lane_cmd(
    lane: str,
    *,
    project_root: Path,
    poll_seconds: float,
    wait_timeout_seconds: float,
    command_timeout_seconds: int,
    backpressure_command_timeout_seconds: int,
    maintenance_force: bool = False,
    sheriff_force: bool = False,
) -> tuple[list[str], int]:
    if lane == "backpressure_slo_bot":
        return (
            [
                str(PYTHON_BIN),
                str(project_root / "scripts" / "ops" / "backpressure_slo_bot.py"),
                "--apply",
                "--command-timeout-seconds",
                str(max(int(backpressure_command_timeout_seconds), 1)),
                "--json",
            ],
            max(int(backpressure_command_timeout_seconds), 1) + 60,
        )
    if lane == "writer_cycle_coordinator":
        cmd = [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "ops" / "writer_cycle_coordinator.py"),
            "--apply",
            "--poll-seconds",
            str(max(float(poll_seconds), 0.1)),
            "--wait-timeout-seconds",
            str(max(float(wait_timeout_seconds), 0.0)),
            "--command-timeout-seconds",
            str(max(int(command_timeout_seconds), 1)),
        ]
        if maintenance_force:
            cmd.append("--maintenance-force")
        cmd.append("--json")
        timeout_sec = max(int(command_timeout_seconds), int(wait_timeout_seconds) + 240)
        return (cmd, timeout_sec)
    if lane == "retention_debt_sheriff":
        cmd = [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "ops" / "retention_debt_sheriff.py"),
            "--apply",
            "--poll-seconds",
            str(max(float(poll_seconds), 0.1)),
            "--wait-timeout-seconds",
            str(max(float(wait_timeout_seconds), 0.0)),
            "--command-timeout-seconds",
            str(max(int(command_timeout_seconds), 1)),
        ]
        if sheriff_force:
            cmd.append("--force")
        cmd.append("--json")
        timeout_sec = max(int(command_timeout_seconds), int(wait_timeout_seconds) + 240)
        return (cmd, timeout_sec)
    raise ValueError(f"unsupported lane: {lane}")


def _repair_plan(
    *,
    storage_control: dict[str, Any],
    backpressure_preview: dict[str, Any],
    coordinator_preview: dict[str, Any],
    sheriff_preview: dict[str, Any],
    project_root: Path,
    poll_seconds: float,
    wait_timeout_seconds: float,
    command_timeout_seconds: int,
    backpressure_command_timeout_seconds: int,
) -> list[dict[str, Any]]:
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage = storage_control.get("storage") if isinstance(storage_control.get("storage"), dict) else {}
    integrity = storage_control.get("data_integrity") if isinstance(storage_control.get("data_integrity"), dict) else {}
    overall_status = str(storage_control.get("overall_status") or "")
    severity = str(storage_control.get("severity") or "")
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    total_pending_lines = _safe_int(backpressure.get("total_pending_lines"), 0)
    total_drain_minutes = max(_safe_float(backpressure.get("estimated_total_drain_minutes"), 0.0), 0.0)
    retention_debt_gb = max(_safe_float(storage.get("retention_debt_gb"), 0.0), 0.0)
    backlog_drain_recommended = bool(storage.get("backlog_drain_recommended_now", False))
    backlog_quarantine_candidate_files = _safe_int(storage.get("backlog_quarantine_candidate_files"), 0)
    sql_invalid_lines = _safe_int(integrity.get("sql_invalid_lines"), 0)
    focus = sheriff_preview.get("focus") if isinstance(sheriff_preview.get("focus"), dict) else {}
    targeted_retention_debt_gb = max(_safe_float(focus.get("targeted_retention_debt_gb"), 0.0), 0.0)
    focus_shards = [str(row) for row in list(focus.get("focus_shards") or []) if str(row or "").strip()]
    severe_focus = bool(focus.get("severe_focus", False))
    coordinator_maintenance_ready = bool(coordinator_preview.get("maintenance_ready", False))
    maintenance_force = bool(severe_focus or targeted_retention_debt_gb >= 5.0)
    sheriff_force = bool(severe_focus or targeted_retention_debt_gb >= 5.0)

    plan: list[dict[str, Any]] = []

    if bool(backpressure_preview.get("actionable", False)) or overall_status in {"blocked", "needs_work"} or severity in {"high", "critical"}:
        cmd, timeout = _lane_cmd(
            "backpressure_slo_bot",
            project_root=project_root,
            poll_seconds=poll_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
            backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
        )
        plan.append(
            {
                "name": "backpressure_slo_bot",
                "reason": f"storage_status={overall_status or 'missing'} severity={severity or 'unknown'}",
                "cmd": cmd,
                "timeout_sec": timeout,
            }
        )

    coordinator_reasons = ordered_unique(
        [
            "coordinator_actionable" if bool(coordinator_preview.get("actionable", False)) else "",
            "drain_ready" if bool(coordinator_preview.get("drain_ready", False)) else "",
            "maintenance_ready" if bool(coordinator_preview.get("maintenance_ready", False)) else "",
            "backlog_drain_recommended" if backlog_drain_recommended else "",
            "quarantine_candidates_present" if backlog_quarantine_candidate_files > 0 else "",
            "sql_invalid_lines_present" if sql_invalid_lines > 0 else "",
            "backlog_above_threshold" if total_pending_lines >= max(pending_threshold * 3, 30000) else "",
            "drain_time_high" if total_drain_minutes >= 60.0 else "",
        ]
    )
    if coordinator_reasons:
        cmd, timeout = _lane_cmd(
            "writer_cycle_coordinator",
            project_root=project_root,
            poll_seconds=poll_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
            backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
            maintenance_force=maintenance_force,
        )
        plan.append(
            {
                "name": "writer_cycle_coordinator",
                "reason": ",".join(coordinator_reasons),
                "cmd": cmd,
                "timeout_sec": timeout,
            }
        )

    sheriff_needed = bool(sheriff_preview.get("actionable", False)) and (
        severe_focus or targeted_retention_debt_gb >= 1.0 or not coordinator_maintenance_ready
    )
    sheriff_reasons = ordered_unique(
        [
            "targeted_retention_debt" if targeted_retention_debt_gb > 0.0 else "",
            "severe_explanation_focus" if severe_focus else "",
            f"focus_shards={','.join(focus_shards[:3])}" if focus_shards else "",
            "coordinator_maintenance_not_ready" if not coordinator_maintenance_ready else "",
            "global_retention_debt_present" if retention_debt_gb > 0.0 else "",
        ]
    )
    if sheriff_needed:
        cmd, timeout = _lane_cmd(
            "retention_debt_sheriff",
            project_root=project_root,
            poll_seconds=poll_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
            backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
            sheriff_force=sheriff_force,
        )
        plan.append(
            {
                "name": "retention_debt_sheriff",
                "reason": ",".join(sheriff_reasons) or "targeted_retention_debt",
                "cmd": cmd,
                "timeout_sec": timeout,
            }
        )
    return plan


def _preview_bundle(
    *,
    project_root: Path,
    storage_control: dict[str, Any],
    poll_seconds: float,
    wait_timeout_seconds: float,
    command_timeout_seconds: int,
    backpressure_command_timeout_seconds: int,
) -> dict[str, Any]:
    backpressure_preview = backpressure_src.build_payload(
        project_root,
        apply=False,
        command_timeout_seconds=max(int(backpressure_command_timeout_seconds), 1),
    )
    coordinator_preview = coordinator_src.build_payload(
        project_root,
        apply=False,
        poll_seconds=max(float(poll_seconds), 0.1),
        wait_timeout_seconds=max(float(wait_timeout_seconds), 0.0),
        command_timeout_seconds=max(int(command_timeout_seconds), 1),
    )
    sheriff_preview = sheriff_src.build_payload(
        project_root,
        apply=False,
        poll_seconds=max(float(poll_seconds), 0.1),
        wait_timeout_seconds=max(float(wait_timeout_seconds), 0.0),
        command_timeout_seconds=max(int(command_timeout_seconds), 1),
    )

    repair_plan = _repair_plan(
        storage_control=storage_control,
        backpressure_preview=backpressure_preview,
        coordinator_preview=coordinator_preview,
        sheriff_preview=sheriff_preview,
        project_root=project_root,
        poll_seconds=poll_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
        backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
    )
    return {
        "backpressure_preview": backpressure_preview,
        "coordinator_preview": coordinator_preview,
        "sheriff_preview": sheriff_preview,
        "repair_plan": repair_plan,
    }


def _attempt_record(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    overall_status = str(payload.get("overall_status") or "")
    status = "ok"
    if bool(result.get("timed_out", False)):
        status = "timed_out"
    elif overall_status in {"applied_with_followups", "waiting_for_writer"}:
        status = "followup"
    elif overall_status and overall_status not in ACCEPTED_STEP_STATUSES:
        status = "error"
    elif int(result.get("rc", 0)) != 0 and overall_status not in ACCEPTED_STEP_STATUSES:
        status = "error"
    return {
        "name": str(payload.get("bot") or ""),
        "status": status,
        "rc": int(result.get("rc", 1)),
        "timed_out": bool(result.get("timed_out", False)),
        "overall_status": overall_status,
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    poll_seconds: float = 20.0,
    wait_timeout_seconds: float = 900.0,
    command_timeout_seconds: int = 2400,
    backpressure_command_timeout_seconds: int = 900,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    target_pending_lines: int = DEFAULT_TARGET_PENDING_LINES,
    target_retention_debt_gb: float = DEFAULT_TARGET_RETENTION_DEBT_GB,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")

    if not storage_control:
        return {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "blocked",
            "apply_requested": bool(apply),
            "repair_plan": [],
            "attempts": [],
            "storage_control": {},
            "operator_followups": [
                "refresh ingestion_storage_control first because the autopilot cannot make safe storage decisions without the latest health artifact"
            ],
            "recommended_actions": [
                "run the ingestion-storage control lane before the storage autopilot so drain and retention thresholds are current"
            ],
            "clearance_targets": {
                "target_pending_lines": max(int(target_pending_lines), 0),
                "target_retention_debt_gb": round(max(float(target_retention_debt_gb), 0.0), 3),
            },
            "metrics": {
                "repair_step_count": 0,
                "attempted_step_count": 0,
            },
        }

    preview_bundle = _preview_bundle(
        project_root=project_root,
        storage_control=storage_control,
        poll_seconds=poll_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
        backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
    )
    backpressure_preview = preview_bundle["backpressure_preview"]
    coordinator_preview = preview_bundle["coordinator_preview"]
    sheriff_preview = preview_bundle["sheriff_preview"]
    repair_plan = preview_bundle["repair_plan"]

    attempts: list[dict[str, Any]] = []
    attempt_payloads: dict[str, Any] = {}
    cycle_records: list[dict[str, Any]] = []
    clearance_targets = {
        "target_pending_lines": max(int(target_pending_lines), 0),
        "target_retention_debt_gb": round(max(float(target_retention_debt_gb), 0.0), 3),
    }
    clearance_before = _clearance_state(
        storage_control,
        target_pending_lines=clearance_targets["target_pending_lines"],
        target_retention_debt_gb=clearance_targets["target_retention_debt_gb"],
    )
    if apply:
        current_storage_control = storage_control
        current_repair_plan = repair_plan
        current_bundle = preview_bundle
        max_cycle_count = max(int(max_cycles), 1)
        for cycle_index in range(1, max_cycle_count + 1):
            cycle_names = [str(row.get("name") or "") for row in current_repair_plan if str(row.get("name") or "").strip()]
            if not current_repair_plan:
                cycle_records.append(
                    {
                        "cycle_index": cycle_index,
                        "repair_step_count": 0,
                        "repair_steps": [],
                        "attempts": [],
                        "progress": {"progress_observed": False},
                        "clearance_before": clearance_before,
                        "clearance_after": clearance_before,
                    }
                )
                break

            cycle_attempts: list[dict[str, Any]] = []
            for row in current_repair_plan:
                cmd = list(row.get("cmd") or [])
                if not cmd:
                    continue
                result = _run_json(cmd, cwd=project_root, timeout_sec=max(int(row.get("timeout_sec", 0) or 0), 1))
                payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
                payload.setdefault("bot", str(row.get("name") or ""))
                attempt_payloads[str(row.get("name") or "")] = payload
                record = _attempt_record(result)
                attempts.append(record)
                cycle_attempts.append(record)

            current_storage_control = load_json(health_root / "ingestion_storage_control_latest.json") or current_storage_control
            clearance_after = _clearance_state(
                current_storage_control,
                target_pending_lines=clearance_targets["target_pending_lines"],
                target_retention_debt_gb=clearance_targets["target_retention_debt_gb"],
            )
            progress = _clearance_progress(clearance_before, clearance_after)
            cycle_records.append(
                {
                    "cycle_index": cycle_index,
                    "repair_step_count": len(cycle_names),
                    "repair_steps": cycle_names,
                    "attempts": cycle_attempts,
                    "progress": progress,
                    "clearance_before": clearance_before,
                    "clearance_after": clearance_after,
                }
            )

            attempt_statuses = [str(row.get("status") or "") for row in cycle_attempts if isinstance(row, dict)]
            if "error" in attempt_statuses or "timed_out" in attempt_statuses:
                storage_control = current_storage_control
                break
            if clearance_after["cleared"]:
                storage_control = current_storage_control
                break
            if cycle_index >= max_cycle_count or not progress["progress_observed"]:
                storage_control = current_storage_control
                break

            storage_control = current_storage_control
            clearance_before = clearance_after
            current_bundle = _preview_bundle(
                project_root=project_root,
                storage_control=current_storage_control,
                poll_seconds=poll_seconds,
                wait_timeout_seconds=wait_timeout_seconds,
                command_timeout_seconds=command_timeout_seconds,
                backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
            )
            current_repair_plan = current_bundle["repair_plan"]

        final_bundle = _preview_bundle(
            project_root=project_root,
            storage_control=storage_control,
            poll_seconds=poll_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
            backpressure_command_timeout_seconds=backpressure_command_timeout_seconds,
        )
        backpressure_preview = final_bundle["backpressure_preview"]
        coordinator_preview = final_bundle["coordinator_preview"]
        sheriff_preview = final_bundle["sheriff_preview"]
        repair_plan = final_bundle["repair_plan"]

    attempt_statuses = [str(row.get("status") or "") for row in attempts if isinstance(row, dict)]
    clearance_after = _clearance_state(
        storage_control,
        target_pending_lines=clearance_targets["target_pending_lines"],
        target_retention_debt_gb=clearance_targets["target_retention_debt_gb"],
    )
    if not repair_plan:
        overall_status = "applied" if apply and clearance_after["cleared"] else "ready"
        ok = True
    elif not apply:
        overall_status = "ready"
        ok = True
    elif "error" in attempt_statuses or "timed_out" in attempt_statuses:
        overall_status = "apply_failed"
        ok = False
    elif clearance_after["cleared"]:
        overall_status = "applied"
        ok = True
    elif "followup" in attempt_statuses:
        overall_status = "applied_with_followups"
        ok = True
    else:
        overall_status = "applied_with_followups"
        ok = True

    operator_followups = ordered_unique(
        list(backpressure_preview.get("recommended_actions") or [])[:2]
        + list(coordinator_preview.get("recommended_actions") or [])[:2]
        + list(sheriff_preview.get("recommended_actions") or [])[:2]
        + [
            action
            for payload in attempt_payloads.values()
            if isinstance(payload, dict)
            for action in list(payload.get("recommended_actions") or [])[:2]
        ]
    )[:8]

    summary_focus = sheriff_preview.get("focus") if isinstance(sheriff_preview.get("focus"), dict) else {}
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "repair_plan": repair_plan,
        "attempts": attempts,
        "cycle_records": cycle_records,
        "storage_control": storage_control,
        "clearance_targets": clearance_targets,
        "clearance_state": {
            "before": clearance_before,
            "after": clearance_after,
            "cleared": bool(clearance_after.get("cleared", False)),
            "steady_state_ready": bool((((storage_control.get("steady_state") or {}).get("target_status") or {}).get("steady_state_ready", False))),
        },
        "previews": {
            "backpressure_slo_bot": {
                "overall_status": str(backpressure_preview.get("overall_status") or ""),
                "actionable": bool(backpressure_preview.get("actionable", False)),
                "recommended_profile": str(backpressure_preview.get("recommended_profile") or ""),
            },
            "writer_cycle_coordinator": {
                "overall_status": str(coordinator_preview.get("overall_status") or ""),
                "actionable": bool(coordinator_preview.get("actionable", False)),
                "drain_ready": bool(coordinator_preview.get("drain_ready", False)),
                "maintenance_ready": bool(coordinator_preview.get("maintenance_ready", False)),
            },
            "retention_debt_sheriff": {
                "overall_status": str(sheriff_preview.get("overall_status") or ""),
                "actionable": bool(sheriff_preview.get("actionable", False)),
                "focus_shards": list(summary_focus.get("focus_shards") or []),
                "targeted_retention_debt_gb": round(_safe_float(summary_focus.get("targeted_retention_debt_gb"), 0.0), 3),
                "severe_focus": bool(summary_focus.get("severe_focus", False)),
            },
        },
        "operator_followups": operator_followups,
        "recommended_actions": ordered_unique(
            [
                "keep the storage backpressure autopilot on a timer so drain, retention, and governor changes stay coordinated",
                "let the autopilot spend multiple repair cycles in one maintenance window so severe backlog and explanation debt are burned down instead of merely nudged",
                "leave the specialist storage bots available for manual use, but let the autopilot own the recurring lane",
                "treat explanation shard debt as a separate signal from broad backlog so retention work stays targeted",
            ]
            + operator_followups[:3]
        )[:8],
        "metrics": {
            "repair_step_count": len(repair_plan),
            "attempted_step_count": len(attempts),
            "cycle_count": len(cycle_records),
            "backpressure_actionable": bool(backpressure_preview.get("actionable", False)),
            "coordinator_actionable": bool(coordinator_preview.get("actionable", False)),
            "sheriff_actionable": bool(sheriff_preview.get("actionable", False)),
            "backpressure_quality_score": _safe_float(((storage_control.get("steady_state") or {}).get("quality_score")), 0.0),
            "pressure_index": _safe_float(storage_control.get("pressure_index"), 0.0),
            "core_pending_lines": _safe_int(((storage_control.get("backpressure") or {}).get("core_pending_lines")), 0),
            "steady_state_breach_count": _safe_int(((((storage_control.get("steady_state") or {}).get("target_status") or {}).get("target_breach_count"))), 0),
            "storage_total_pending_lines": _safe_int(((storage_control.get("backpressure") or {}).get("total_pending_lines")), 0),
            "storage_total_drain_minutes": _safe_float(((storage_control.get("backpressure") or {}).get("estimated_total_drain_minutes")), 0.0),
            "retention_debt_gb": _safe_float(((storage_control.get("storage") or {}).get("retention_debt_gb")), 0.0),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate storage governor, drain, and retention bots so backpressure remediation runs as one lane.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS", os.getenv("WRITER_CYCLE_COORDINATOR_POLL_SECONDS", "20"))),
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=float(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS", os.getenv("WRITER_CYCLE_COORDINATOR_WAIT_TIMEOUT_SECONDS", "900"))),
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=int(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS", os.getenv("WRITER_CYCLE_COORDINATOR_COMMAND_TIMEOUT_SECONDS", "2400"))),
    )
    parser.add_argument(
        "--backpressure-command-timeout-seconds",
        type=int,
        default=int(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS", os.getenv("BACKPRESSURE_SLO_BOT_COMMAND_TIMEOUT_SECONDS", "900"))),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=int(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES", str(DEFAULT_MAX_CYCLES))),
    )
    parser.add_argument(
        "--target-pending-lines",
        type=int,
        default=int(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES", str(DEFAULT_TARGET_PENDING_LINES))),
    )
    parser.add_argument(
        "--target-retention-debt-gb",
        type=float,
        default=float(os.getenv("STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_RETENTION_DEBT_GB", str(DEFAULT_TARGET_RETENTION_DEBT_GB))),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "pending",
    }
    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"overall_status": "already_running", "busy": True})
            write_payload(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("storage_backpressure_autopilot overall_status=already_running")
            return 0

        preview_payload = build_payload(
            project_root,
            apply=False,
            poll_seconds=float(args.poll_seconds),
            wait_timeout_seconds=float(args.wait_timeout_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
            backpressure_command_timeout_seconds=int(args.backpressure_command_timeout_seconds),
            max_cycles=int(args.max_cycles),
            target_pending_lines=int(args.target_pending_lines),
            target_retention_debt_gb=float(args.target_retention_debt_gb),
        )
        if not bool(args.apply):
            payload = preview_payload
            write_payload(out_file, payload)
        else:
            running_payload = dict(preview_payload)
            running_payload.update(
                {
                    "ok": True,
                    "overall_status": "running",
                    "busy": True,
                    "apply_requested": True,
                }
            )
            write_payload(out_file, running_payload)
            payload = build_payload(
                project_root,
                apply=True,
                poll_seconds=float(args.poll_seconds),
                wait_timeout_seconds=float(args.wait_timeout_seconds),
                command_timeout_seconds=int(args.command_timeout_seconds),
                backpressure_command_timeout_seconds=int(args.backpressure_command_timeout_seconds),
                max_cycles=int(args.max_cycles),
                target_pending_lines=int(args.target_pending_lines),
                target_retention_debt_gb=float(args.target_retention_debt_gb),
            )
            write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_backpressure_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_plan={int(payload.get('metrics', {}).get('repair_step_count', 0) or 0)}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"already_running", "ready", "applied", "applied_with_followups"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
