#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.ops.long_runtime_common import load_json, parse_iso_utc, write_payload

DEFAULT_INTERVAL_SECONDS = 300.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_from_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def interval_seconds(raw: float | int | str | None) -> float:
    try:
        value = float(raw if raw is not None else DEFAULT_INTERVAL_SECONDS)
    except (TypeError, ValueError):
        value = DEFAULT_INTERVAL_SECONDS
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_INTERVAL_SECONDS
    return value


def next_eligible_utc(
    completed_utc: str | datetime, cadence_seconds: float | int | str | None
) -> str:
    if isinstance(completed_utc, datetime):
        completed = completed_utc
    else:
        completed = parse_iso_utc(completed_utc) or utc_now()
    return iso_from_datetime(
        completed + timedelta(seconds=interval_seconds(cadence_seconds))
    )


def new_run_id(job_id: str) -> str:
    compact_job = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in job_id
    ).strip("_")
    return f"{compact_job or 'scheduled_job'}-{uuid.uuid4().hex}"


def infer_deferred_reason(*, stdout: str = "", stderr: str = "", rc: int = 0) -> str:
    text = f"{stdout}\n{stderr}".lower()
    markers = (
        ("local_storage_reserve_pressure", "local_storage_reserve_pressure"),
        ("runtime_maintenance_hold", "runtime_maintenance_hold"),
        ("quiet_window", "quiet_window"),
        ("refresh_already_running", "already_running"),
        ("one_numbers lock busy", "already_running"),
        ("already_running", "already_running"),
        ("writer_lock_busy", "writer_lock_busy"),
        ("sql_link_shard_manager busy", "writer_lock_busy"),
        (" busy owner=", "writer_lock_busy"),
        ("resource_guard_blocked", "resource_guard_blocked"),
        ("support_maintenance_frozen", "support_maintenance_frozen"),
        ("status=deferred", "deferred"),
        ("skip ", "skipped"),
    )
    if rc == 1 and "one_numbers lock busy" in text:
        return "already_running"
    if rc != 0:
        return ""
    for needle, reason in markers:
        if needle in text:
            return reason
    return ""


def command_tail(text: str, *, lines: int = 12, max_chars: int = 4000) -> str:
    tail = "\n".join(str(text or "").splitlines()[-max(int(lines), 1) :])
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def lifecycle_receipt(
    *,
    job_id: str,
    scheduled: bool,
    started_utc: str | datetime,
    completed_utc: str | datetime,
    schedule_interval_seconds: float | int | str | None,
    rc: int,
    terminal_status: str,
    ok: bool,
    deferred_reason: str = "",
    failure_reason: str = "",
    stdout_tail: str = "",
    stderr_tail: str = "",
    artifact_present_before: bool = False,
    artifact_present_after: bool = False,
    deadline_seconds: int = 0,
    command: list[str] | None = None,
    timed_out: bool = False,
    source: str = "scheduled_lifecycle_runner",
) -> dict[str, Any]:
    started = (
        iso_from_datetime(started_utc)
        if isinstance(started_utc, datetime)
        else str(started_utc or "")
    )
    completed = (
        iso_from_datetime(completed_utc)
        if isinstance(completed_utc, datetime)
        else str(completed_utc or "")
    )
    cadence = interval_seconds(schedule_interval_seconds)
    deferred = bool(deferred_reason) and not timed_out and int(rc) == 0
    non_failure_nonzero_statuses = {"completed_with_findings"}
    failed = bool(
        timed_out
        or (
            int(rc) != 0
            and str(terminal_status or "") not in non_failure_nonzero_statuses
        )
        or failure_reason
    )
    return {
        "schema_version": 1,
        "source": source,
        "job_id": str(job_id),
        "run_id": new_run_id(str(job_id)),
        "scheduled": bool(scheduled),
        "schedule_interval_seconds": cadence,
        "eligible": bool(scheduled) and not deferred,
        "deferred": deferred,
        "deferred_reason": str(deferred_reason or ""),
        "started": True,
        "started_utc": started,
        "completed": True,
        "completed_utc": completed,
        "failed": failed,
        "failure_reason": str(failure_reason or ""),
        "terminal_status": str(terminal_status or ""),
        "next_eligible_utc": next_eligible_utc(completed, cadence),
        "rc": int(rc),
        "timed_out": bool(timed_out),
        "deadline_seconds": max(int(deadline_seconds or 0), 0),
        "artifact_present_before": bool(artifact_present_before),
        "artifact_present_after": bool(artifact_present_after),
        "stdout_tail": command_tail(stdout_tail),
        "stderr_tail": command_tail(stderr_tail),
        "command": list(command or []),
        "authority": {
            "launchctl_mutation_authority": False,
            "live_execution_authority": False,
            "training_launch_authority": False,
            "competing_writer_authority": False,
            "storage_delete_authority": False,
            "credential_mutation_authority": False,
        },
    }


def attach_lifecycle(
    payload: dict[str, Any], lifecycle: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(payload)
    updated["job_lifecycle"] = dict(lifecycle)
    updated["scheduled_lifecycle"] = {
        "source": lifecycle.get("source", "scheduled_lifecycle_runner"),
        "run_id": lifecycle.get("run_id", ""),
        "job_id": lifecycle.get("job_id", ""),
        "completed_utc": lifecycle.get("completed_utc", ""),
        "terminal_status": lifecycle.get("terminal_status", ""),
        "failed": lifecycle.get("failed", False),
        "deferred": lifecycle.get("deferred", False),
    }
    return updated


def stamp_artifact(
    path: Path, lifecycle: dict[str, Any], *, base_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = base_payload if isinstance(base_payload, dict) else load_json(path)
    if not payload:
        payload = {
            "schema_version": 1,
            "timestamp_utc": lifecycle.get(
                "completed_utc", iso_from_datetime(utc_now())
            ),
            "source": "scheduled_lifecycle_stub",
            "ok": False,
            "overall_status": lifecycle.get("failure_reason")
            or "artifact_missing_after_run",
        }
    updated = attach_lifecycle(payload, lifecycle)
    write_payload(path, updated)
    return updated
