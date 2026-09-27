#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from scripts.ops import storage_maintenance_lane as maintenance_src
from scripts.ops import writer_cycle_coordinator as coordinator_src

PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "retention_debt_sheriff_latest.json"
)
DEFAULT_LOCK_PATH = (
    PROJECT_ROOT / "governance" / "locks" / "retention_debt_sheriff.lock"
)
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_WAIT_TIMEOUT_SECONDS = 900.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 2400
DEFAULT_SCHEDULE_INTERVAL_SECONDS = 300.0
SHERIFF_SHARDS = {
    "explanations",
    "crypto_explanations",
    "shadow_attribution",
    "crypto_shadow_attribution",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schedule_interval_seconds(raw: Any) -> float:
    try:
        interval = float(raw)
    except Exception:
        return 60.0
    if not math.isfinite(interval):
        return 60.0
    return max(interval, 60.0)


def _next_eligible_utc(completed_utc: str, interval_seconds: float) -> str:
    try:
        completed = datetime.fromisoformat(str(completed_utc).replace("Z", "+00:00"))
    except Exception:
        completed = datetime.now(timezone.utc)
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=timezone.utc)
    interval = _schedule_interval_seconds(interval_seconds)
    return (
        completed.astimezone(timezone.utc) + timedelta(seconds=interval)
    ).isoformat()


def _job_lifecycle(
    *,
    run_id: str,
    scheduled: bool,
    started_utc: str,
    completed_utc: str,
    schedule_interval_seconds: float,
    status: str,
    ok: bool,
    work_available: bool = False,
    apply_requested: bool = False,
    deferred: bool = False,
    deferred_reason: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": "retention_debt_sheriff",
        "launchd_label": "com.dankingsley.ops.retention_debt_sheriff",
        "run_id": run_id,
        "scheduled": bool(scheduled),
        "schedule_interval_seconds": _schedule_interval_seconds(
            schedule_interval_seconds
        ),
        "scheduled_utc": started_utc if scheduled else "",
        "work_available": bool(work_available),
        "apply_requested": bool(apply_requested),
        "eligible": not deferred,
        "deferred": bool(deferred),
        "deferred_reason": str(deferred_reason or ""),
        "started": not deferred,
        "started_utc": "" if deferred else started_utc,
        "completed": True,
        "completed_utc": completed_utc,
        "failed": bool(not ok and not deferred),
        "failure_reason": str(failure_reason or ""),
        "terminal_status": status,
        "next_eligible_utc": _next_eligible_utc(
            completed_utc, schedule_interval_seconds
        ),
        "authority": {
            "live_execution_authority": False,
            "training_launch_authority": False,
            "competing_sqlite_writer_authority": False,
            "source_delete_authority": False,
        },
    }


def _refresh_completion_metadata(
    payload: dict[str, Any], *, schedule_interval_seconds: float
) -> None:
    completed_utc = _utc_now()
    payload["timestamp_utc"] = completed_utc
    payload["completed_utc"] = completed_utc
    payload["next_eligible_utc"] = _next_eligible_utc(
        completed_utc, schedule_interval_seconds
    )
    lifecycle = payload.get("job_lifecycle")
    if isinstance(lifecycle, dict):
        lifecycle["completed_utc"] = completed_utc
        lifecycle["next_eligible_utc"] = payload["next_eligible_utc"]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed(
        [raw.strip() for raw in str(text or "").splitlines() if raw.strip()]
    ):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    payload_path: Path | None = None,
    timeout_sec: int,
) -> dict[str, Any]:
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
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="ignore")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="ignore")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
        rc = 124
        timed_out = True
    payload = _parse_json_output(stdout)
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    duration_ms = round(
        (datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3
    )
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
        "timed_out": timed_out,
    }


def _step_status(
    result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None
) -> str:
    if bool(result.get("timed_out", False)):
        return "timed_out"
    if int(result.get("rc", 1)) != 0:
        return "error"
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    reason = str(payload.get("reason") or "")
    accepted = nonfatal_reasons or set()
    if bool(payload.get("busy", False)) or reason in accepted:
        return "busy"
    if bool(payload.get("skipped", False)):
        return "skipped"
    if payload.get("ok") is False:
        return "error"
    return "ok"


def _step_record(
    result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None
) -> dict[str, Any]:
    return {
        "status": _step_status(result, nonfatal_reasons=nonfatal_reasons),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _refresh_surface_artifacts(project_root: Path) -> dict[str, Any]:
    refresh_steps: dict[str, Any] = {}
    for name, script_name in (
        ("ingestion_storage_control", "ingestion_storage_control.py"),
        ("runtime_gate_dashboard", "runtime_gate_dashboard.py"),
        ("operator_cockpit", "operator_cockpit.py"),
    ):
        refresh = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / script_name), "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / f"{name}_latest.json",
            timeout_sec=120,
        )
        refresh_steps[name] = _step_record(refresh)
    return refresh_steps


def _sheriff_focus(project_root: Path) -> dict[str, Any]:
    maintenance_focus = maintenance_src._priority_retention_focus(project_root, {})
    raw_rows = (
        maintenance_focus.get("priority_rows")
        if isinstance(maintenance_focus.get("priority_rows"), list)
        else []
    )
    focus_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        shard = str(raw.get("shard") or "").strip()
        if shard not in SHERIFF_SHARDS:
            continue
        focus_rows.append(
            {
                "shard": shard,
                "retention_debt_gb": round(
                    _safe_float(raw.get("retention_debt_gb"), 0.0), 3
                ),
                "latency_limit_multiplier": round(
                    _safe_float(raw.get("latency_limit_multiplier"), 0.0), 3
                ),
                "storage_breached": bool(raw.get("storage_breached", False)),
                "latency_breached": bool(raw.get("latency_breached", False)),
                "recommended_action": str(raw.get("recommended_action") or ""),
            }
        )
    focus_rows.sort(
        key=lambda row: (
            float(row.get("retention_debt_gb", 0.0) or 0.0),
            float(row.get("latency_limit_multiplier", 0.0) or 0.0),
        ),
        reverse=True,
    )
    focus_shards = [str(row.get("shard") or "") for row in focus_rows]
    targeted_retention_debt_gb = round(
        sum(float(row.get("retention_debt_gb", 0.0) or 0.0) for row in focus_rows), 3
    )
    severe_focus = bool(
        focus_rows
        and (
            targeted_retention_debt_gb >= 20.0
            or any(bool(row.get("storage_breached", False)) for row in focus_rows)
        )
    )
    top_actions: list[str] = []
    if focus_rows:
        top_actions.append(
            "keep explanation shard retention ahead of broad shard maintenance until their debt is near zero"
        )
    if any(bool(row.get("latency_breached", False)) for row in focus_rows):
        top_actions.append(
            "treat explanation shard latency breaches as a signal to drain debt before widening ingestion fan-in"
        )
    if severe_focus:
        top_actions.append(
            "keep the sheriff focused on explanations only until the oversized shard debt falls back under the storage limits"
        )
    return {
        "enabled": bool(focus_rows),
        "severe_focus": severe_focus,
        "focus_shards": focus_shards,
        "targeted_retention_debt_gb": targeted_retention_debt_gb,
        "priority_rows": focus_rows,
        "top_actions": top_actions[:5],
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    force: bool = False,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    scheduled: bool = False,
    schedule_interval_seconds: float = DEFAULT_SCHEDULE_INTERVAL_SECONDS,
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    started_utc = _utc_now()
    focus = _sheriff_focus(project_root)
    writer_state = coordinator_src.writer_state_snapshot(project_root)
    actionable = bool(focus.get("enabled", False))

    steps: dict[str, Any] = {}
    refresh_steps: dict[str, Any] = {}
    coordinator_payload: dict[str, Any] = {}
    maintenance_payload: dict[str, Any] = {}
    applied_ok = False

    if apply and actionable:
        if bool(writer_state.get("active", False)):
            coordinator_cmd = [
                str(PY),
                str(project_root / "scripts" / "ops" / "writer_cycle_coordinator.py"),
                "--apply",
                "--skip-drain",
                "--poll-seconds",
                str(float(poll_seconds)),
                "--wait-timeout-seconds",
                str(float(wait_timeout_seconds)),
                "--command-timeout-seconds",
                str(int(command_timeout_seconds)),
            ]
            if force:
                coordinator_cmd.append("--maintenance-force")
            coordinator_cmd.append("--json")
            coordinator = _run_json_command(
                coordinator_cmd,
                cwd=project_root,
                payload_path=project_root
                / "governance"
                / "health"
                / "writer_cycle_coordinator_latest.json",
                timeout_sec=max(
                    int(command_timeout_seconds), int(wait_timeout_seconds) + 240
                ),
            )
            steps["writer_cycle_coordinator"] = _step_record(
                coordinator, nonfatal_reasons={"already_running"}
            )
            coordinator_payload = (
                coordinator.get("payload")
                if isinstance(coordinator.get("payload"), dict)
                else {}
            )
            applied_ok = int(coordinator.get("rc", 1)) == 0 and bool(
                (
                    (coordinator_payload.get("summary") or {}).get(
                        "maintenance_applied", False
                    )
                )
            )
        else:
            maintenance_cmd = [
                str(PY),
                str(project_root / "scripts" / "ops" / "storage_maintenance_lane.py"),
            ]
            if force:
                maintenance_cmd.append("--force")
            maintenance_cmd.append("--json")
            maintenance = _run_json_command(
                maintenance_cmd,
                cwd=project_root,
                payload_path=project_root
                / "governance"
                / "health"
                / "storage_maintenance_latest.json",
                timeout_sec=max(int(command_timeout_seconds), 900),
            )
            steps["storage_maintenance_lane"] = _step_record(
                maintenance, nonfatal_reasons={"already_running"}
            )
            maintenance_payload = (
                maintenance.get("payload")
                if isinstance(maintenance.get("payload"), dict)
                else {}
            )
            applied_ok = (
                int(maintenance.get("rc", 1)) == 0
                and str(steps["storage_maintenance_lane"].get("status") or "") == "ok"
            )

        if steps:
            refresh_steps = _refresh_surface_artifacts(project_root)

    has_error = any(
        str((row or {}).get("status") or "") in {"error", "timed_out"}
        for row in steps.values()
        if isinstance(row, dict)
    )
    if not actionable:
        overall_status = "idle"
        ok = True
    elif bool(writer_state.get("active", False)) and not apply:
        overall_status = "waiting_for_writer"
        ok = True
    elif not apply:
        overall_status = "ready"
        ok = True
    elif has_error:
        overall_status = "apply_failed"
        ok = False
    elif applied_ok:
        overall_status = "applied"
        ok = True
    else:
        overall_status = "applied_with_followups"
        ok = False

    recommended_actions = _ordered_unique(
        list(focus.get("top_actions") or [])[:4]
        + (
            [
                "wait for the current SQL writer cycle to finish before forcing explanation shard maintenance"
            ]
            if bool(writer_state.get("active", False))
            else []
        )
    )[:6]
    if not recommended_actions:
        recommended_actions.append(
            "keep the sheriff idle until explanation shard debt reappears"
        )

    completed_utc = _utc_now()
    failure_reason = "" if ok else overall_status
    payload = {
        "timestamp_utc": completed_utc,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "next_eligible_utc": _next_eligible_utc(
            completed_utc, schedule_interval_seconds
        ),
        "run_id": run_id,
        "run_state": "completed",
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply": bool(apply),
        "force": bool(force),
        "actionable": actionable,
        "writer_state": writer_state,
        "focus": focus,
        "steps": steps,
        "refresh_steps": refresh_steps,
        "recommended_actions": recommended_actions,
        "job_lifecycle": _job_lifecycle(
            run_id=run_id,
            scheduled=scheduled,
            started_utc=started_utc,
            completed_utc=completed_utc,
            schedule_interval_seconds=schedule_interval_seconds,
            status=overall_status,
            ok=ok,
            work_available=actionable,
            apply_requested=apply,
            failure_reason=failure_reason,
        ),
        "summary": {
            "writer_active": bool(writer_state.get("active", False)),
            "focus_shards": list(focus.get("focus_shards") or []),
            "targeted_retention_debt_gb": round(
                _safe_float(focus.get("targeted_retention_debt_gb"), 0.0), 3
            ),
            "storage_maintenance_reason": str(
                ((maintenance_payload.get("reason")) or "")
            ),
            "coordinator_overall_status": str(
                coordinator_payload.get("overall_status") or ""
            ),
            "maintenance_applied": bool(applied_ok),
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Focus retention maintenance on the explanation shards that are dominating storage debt and latency pressure."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--wait-timeout-seconds", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--command-timeout-seconds", type=int, default=DEFAULT_COMMAND_TIMEOUT_SECONDS
    )
    parser.add_argument("--scheduled", action="store_true")
    parser.add_argument(
        "--schedule-interval-seconds",
        type=float,
        default=DEFAULT_SCHEDULE_INTERVAL_SECONDS,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    started_utc = _utc_now()
    payload: dict[str, Any] = {
        "timestamp_utc": started_utc,
        "started_utc": started_utc,
        "completed_utc": "",
        "next_eligible_utc": "",
        "run_id": run_id,
        "run_state": "pending",
        "schema_version": 1,
        "ok": True,
        "overall_status": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            completed_utc = _utc_now()
            payload.update(
                {
                    "timestamp_utc": completed_utc,
                    "completed_utc": completed_utc,
                    "next_eligible_utc": _next_eligible_utc(
                        completed_utc, args.schedule_interval_seconds
                    ),
                    "run_state": "deferred",
                    "overall_status": "already_running",
                    "busy": True,
                    "job_lifecycle": _job_lifecycle(
                        run_id=run_id,
                        scheduled=bool(args.scheduled),
                        started_utc=started_utc,
                        completed_utc=completed_utc,
                        schedule_interval_seconds=float(args.schedule_interval_seconds),
                        status="already_running",
                        ok=True,
                        work_available=False,
                        apply_requested=bool(args.apply),
                        deferred=True,
                        deferred_reason="single_flight_lock_held",
                    ),
                }
            )
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("retention_debt_sheriff overall_status=already_running")
            return 0

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            force=bool(args.force),
            poll_seconds=float(args.poll_seconds),
            wait_timeout_seconds=float(args.wait_timeout_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
            scheduled=bool(args.scheduled),
            schedule_interval_seconds=float(args.schedule_interval_seconds),
        )
        _write_json(out_file, payload)
        if bool(args.apply):
            payload["post_write_refresh_steps"] = _refresh_surface_artifacts(
                project_root
            )
            _refresh_completion_metadata(
                payload,
                schedule_interval_seconds=float(args.schedule_interval_seconds),
            )
            _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "retention_debt_sheriff "
            f"overall_status={payload.get('overall_status', '')} "
            f"actionable={int(bool(payload.get('actionable', False)))}"
        )
    return (
        0
        if bool(
            payload.get("ok", False)
            or str(payload.get("overall_status") or "")
            in {"already_running", "idle", "waiting_for_writer"}
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
