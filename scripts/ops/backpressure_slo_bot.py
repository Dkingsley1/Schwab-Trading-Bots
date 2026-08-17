#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_slo_bot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "backpressure_slo_bot.lock"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
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


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


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
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = _parse_json_output(stdout)
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
        "timed_out": timed_out,
    }


def _step_status(result: dict[str, Any]) -> str:
    if bool(result.get("timed_out", False)):
        return "timed_out"
    if int(result.get("rc", 1)) != 0:
        return "error"
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if payload.get("ok") is False:
        return "error"
    return "ok"


def _step_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _step_status(result),
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


def _latency_summary(health_gates: dict[str, Any]) -> dict[str, Any]:
    raw_rows = health_gates.get("priority_shards") if isinstance(health_gates.get("priority_shards"), list) else []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        shard = str(raw.get("shard") or "").strip()
        if not shard:
            continue
        latency_multiplier = max(_safe_float(raw.get("latency_limit_multiplier"), 0.0), 0.0)
        storage_breached = bool(raw.get("storage_breached", False))
        latency_breached = bool(raw.get("latency_breached", False))
        rows.append(
            {
                "shard": shard,
                "latency_limit_multiplier": round(latency_multiplier, 3),
                "storage_breached": storage_breached,
                "latency_breached": latency_breached,
            }
        )
    rows.sort(key=lambda row: float(row.get("latency_limit_multiplier", 0.0) or 0.0), reverse=True)
    breached_shards = [
        str(row.get("shard") or "")
        for row in rows
        if bool(row.get("storage_breached", False)) or bool(row.get("latency_breached", False))
    ]
    return {
        "priority_rows": rows[:10],
        "breached_shards": breached_shards[:8],
        "max_latency_limit_multiplier": round(max((float(row.get("latency_limit_multiplier", 0.0) or 0.0) for row in rows), default=0.0), 3),
    }


def _recommended_profile(
    *,
    hard_gate: bool,
    severe_overload: bool,
    core_drain_minutes: float,
    total_drain_minutes: float,
    retention_debt_gb: float,
    max_latency_limit_multiplier: float,
    steady_state_breach_count: int,
) -> str:
    if (
        hard_gate
        or severe_overload
        or core_drain_minutes >= 30.0
        or total_drain_minutes >= 180.0
        or retention_debt_gb >= 20.0
        or max_latency_limit_multiplier >= 1.25
    ):
        return "critical_backpressure"
    if (
        steady_state_breach_count > 0
        or
        core_drain_minutes >= 15.0
        or total_drain_minutes >= 60.0
        or retention_debt_gb > 0.0
        or max_latency_limit_multiplier >= 1.0
    ):
        return "elevated_backpressure"
    return "steady_state"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    storage_control = _load_json(health_root / "ingestion_storage_control_latest.json")
    governor = _load_json(health_root / "ingestion_storage_governor_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")

    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage = storage_control.get("storage") if isinstance(storage_control.get("storage"), dict) else {}
    latency = _latency_summary(health_gates)
    core_drain_minutes = max(_safe_float(backpressure.get("estimated_core_drain_minutes"), 0.0), 0.0)
    total_drain_minutes = max(_safe_float(backpressure.get("estimated_total_drain_minutes"), 0.0), 0.0)
    retention_debt_gb = max(
        _safe_float(storage.get("retention_debt_gb"), 0.0),
        _safe_float(((health_gates.get("storage_pressure") or {}).get("retention_debt_gb")), 0.0),
    )
    hard_gate_flags = health_gates.get("hard_gates") if isinstance(health_gates.get("hard_gates"), dict) else {}
    storage_hard_gate = any(
        bool(hard_gate_flags.get(key, False))
        for key in (
            "ingestion_pending_lines",
            "ingestion_oldest_age",
            "ingestion_invalid_lines",
            "ingestion_backpressure_overload",
            "priority_shard_storage",
            "sql_progress_stall",
            "sql_wal_pressure",
        )
    )
    hard_gate = bool(str(storage_control.get("overall_status") or "") == "blocked" or storage_hard_gate)
    severe_overload = bool(
        ((health_gates.get("storage_pressure") or {}).get("severe_backpressure_overload", False))
        or ((health_gates.get("ingestion_pressure") or {}).get("severe_backpressure_overload", False))
    )
    steady_state = storage_control.get("steady_state") if isinstance(storage_control.get("steady_state"), dict) else {}
    steady_state_targets = steady_state.get("targets") if isinstance(steady_state.get("targets"), dict) else {}
    steady_state_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    steady_state_breaches = [
        str(item)
        for item in list(steady_state_status.get("target_breaches") or [])
        if str(item or "").strip()
    ]
    backpressure_quality_score = round(_safe_float(steady_state.get("quality_score"), 0.0), 2)
    queue_health_actionable = bool(steady_state_breaches)
    priority_latency_actionable = bool(list(latency.get("breached_shards") or []))
    current_profile = str(governor.get("profile") or "")
    recommended_profile = _recommended_profile(
        hard_gate=hard_gate,
        severe_overload=severe_overload,
        core_drain_minutes=core_drain_minutes,
        total_drain_minutes=total_drain_minutes,
        retention_debt_gb=retention_debt_gb,
        max_latency_limit_multiplier=_safe_float(latency.get("max_latency_limit_multiplier"), 0.0),
        steady_state_breach_count=len(steady_state_breaches),
    )
    profile_drift = bool(current_profile and current_profile != recommended_profile)
    actionable = bool(
        profile_drift
        or hard_gate
        or severe_overload
        or retention_debt_gb > 0.0
        or core_drain_minutes >= 15.0
        or total_drain_minutes >= 60.0
        or queue_health_actionable
        or priority_latency_actionable
    )

    steps: dict[str, Any] = {}
    refresh_steps: dict[str, Any] = {}
    governor_payload: dict[str, Any] = {}
    if apply and actionable:
        governor_result = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "ingestion_storage_governor.py"), "apply", "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "ingestion_storage_governor_latest.json",
            timeout_sec=max(int(command_timeout_seconds), 120),
        )
        steps["ingestion_storage_governor"] = _step_record(governor_result)
        governor_payload = governor_result.get("payload") if isinstance(governor_result.get("payload"), dict) else {}
        refresh_steps = _refresh_surface_artifacts(project_root)

    has_error = any(str((row or {}).get("status") or "") in {"error", "timed_out"} for row in steps.values() if isinstance(row, dict))
    if not actionable:
        overall_status = "stable"
        ok = True
    elif not apply:
        overall_status = "ready"
        ok = True
    elif has_error:
        overall_status = "apply_failed"
        ok = False
    else:
        overall_status = "applied"
        ok = True

    recommended_actions = _ordered_unique(
        (
            ["apply the storage governor so the runtime profile matches the current drain and retention SLOs"]
            if profile_drift
            else []
        )
        + (
            ["keep the retention-debt sheriff enabled until explanation shard debt falls back near zero"]
            if retention_debt_gb > 0.0
            else []
        )
        + (
            [f"hold steady-state targets at pressure_index<={_safe_float(steady_state_targets.get('pressure_index'), 0.25):.2f}, core_pending_lines<={_safe_int(steady_state_targets.get('core_pending_lines'), 5000)}, total_drain_minutes<={_safe_float(steady_state_targets.get('estimated_total_drain_minutes'), 15.0):.1f}"]
            if queue_health_actionable
            else []
        )
        + (
            ["keep the writer-cycle coordinator installed so off-hours drain work can catch clean writer handoffs"]
            if total_drain_minutes >= 180.0 or severe_overload
            else []
        )
        + (
            ["treat priority shard latency breaches as a protection signal, not a reason to widen ingestion budgets"]
            if priority_latency_actionable
            else []
        )
    )[:6]
    if not recommended_actions:
        recommended_actions.append("keep the backpressure SLO bot in monitor mode until drain time or retention debt rises again")

    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply": bool(apply),
        "actionable": actionable,
        "current_profile": current_profile,
        "recommended_profile": recommended_profile,
        "profile_drift": profile_drift,
        "queue_health_actionable": queue_health_actionable,
        "priority_latency_actionable": priority_latency_actionable,
        "signals": {
            "hard_gate": hard_gate,
            "severe_backpressure_overload": severe_overload,
            "core_drain_minutes": round(core_drain_minutes, 3),
            "total_drain_minutes": round(total_drain_minutes, 3),
            "retention_debt_gb": round(retention_debt_gb, 3),
            "max_latency_limit_multiplier": round(_safe_float(latency.get("max_latency_limit_multiplier"), 0.0), 3),
            "breached_priority_shards": list(latency.get("breached_shards") or []),
            "storage_overall_status": str(storage_control.get("overall_status") or ""),
            "storage_severity": str(storage_control.get("severity") or ""),
            "backpressure_quality_score": backpressure_quality_score,
            "steady_state_ready": bool(steady_state_status.get("steady_state_ready", False)),
            "steady_state_target_breaches": steady_state_breaches,
        },
        "steady_state": {
            "quality_score": backpressure_quality_score,
            "quality_label": str(steady_state.get("quality_label") or ""),
            "targets": steady_state_targets,
            "target_status": steady_state_status,
        },
        "priority_latency": latency,
        "steps": steps,
        "refresh_steps": refresh_steps,
        "recommended_actions": recommended_actions,
        "summary": {
            "current_profile": current_profile,
            "recommended_profile": recommended_profile,
            "profile_drift": profile_drift,
            "governor_profile_after_apply": str(governor_payload.get("profile") or ""),
            "breached_priority_shard_count": _safe_int(len(list(latency.get("breached_shards") or [])), 0),
            "backpressure_quality_score": backpressure_quality_score,
            "steady_state_breach_count": len(steady_state_breaches),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Map measured drain time, retention debt, and priority shard latency onto the storage governor profile.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--command-timeout-seconds", type=int, default=DEFAULT_COMMAND_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"overall_status": "already_running", "busy": True})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("backpressure_slo_bot overall_status=already_running")
            return 0

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            command_timeout_seconds=int(args.command_timeout_seconds),
        )
        _write_json(out_file, payload)
        if bool(args.apply):
            payload["post_write_refresh_steps"] = _refresh_surface_artifacts(project_root)
            _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "backpressure_slo_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"recommended_profile={payload.get('recommended_profile', '')}"
        )
    return 0 if bool(payload.get("ok", False) or str(payload.get("overall_status") or "") in {"already_running", "stable"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
