#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "overnight_training_window_latest.json"
DEFAULT_EVENT_DIR = PROJECT_ROOT / "governance" / "events" / "training"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _out_path(project_root: Path) -> Path:
    return project_root / "governance" / "health" / DEFAULT_OUT_PATH.name


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = str(value or "").strip().split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"expected HH:MM wall clock time, got {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"invalid wall clock time: {value!r}")
    return hour, minute


def _next_wall_clock(now: datetime, hhmm: str) -> datetime:
    hour, minute = _parse_hhmm(hhmm)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _wall_clock_on_date(anchor: datetime, hhmm: str) -> datetime:
    hour, minute = _parse_hhmm(hhmm)
    return anchor.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _training_env_overrides() -> dict[str, str]:
    worker_cap = max(3, _safe_int(os.getenv("TRAINING_PCORE_MAX_WORKERS"), 3))
    training_nice = max(0, min(_safe_int(os.getenv("TRAINING_PCORE_NICE"), 0), 2))
    research_nice = max(0, min(_safe_int(os.getenv("RUNTIME_THROTTLE_RESEARCH_NICE"), 0), 2))
    return {
        "BOT_CPU_ALLOCATION_POLICY": "performance_core_primary",
        "BOT_CPU_QOS_POLICY": "performance_core_primary_no_background_writer",
        "BOT_CPU_SCHEDULER_INTENT": "performance_core_training",
        "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN": "1",
        "TRAINING_PCORE_MAX_WORKERS": str(worker_cap),
        "TRAINING_PCORE_NICE": str(training_nice),
        "RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND": "0",
        "RUNTIME_THROTTLE_RESEARCH_NICE": str(research_nice),
    }


def _run_json(
    command: list[str],
    *,
    project_root: Path,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_seconds), 1),
            env=env,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        parsed: Any = {}
        stripped = stdout.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = {}
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "ok": completed.returncode == 0,
            "started_local": started.isoformat(),
            "finished_local": datetime.now().astimezone().isoformat(),
            "json": parsed if isinstance(parsed, dict) else {},
            "env_overrides": env_overrides or {},
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "ok": False,
            "started_local": started.isoformat(),
            "finished_local": datetime.now().astimezone().isoformat(),
            "timed_out": True,
            "json": {},
            "env_overrides": env_overrides or {},
            "stdout_tail": str(exc.stdout or "")[-2000:],
            "stderr_tail": str(exc.stderr or "")[-2000:],
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "ok": False,
            "started_local": started.isoformat(),
            "finished_local": datetime.now().astimezone().isoformat(),
            "json": {},
            "env_overrides": env_overrides or {},
            "stdout_tail": "",
            "stderr_tail": str(exc)[:2000],
        }


def _run_ops(project_root: Path, args: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    return _run_json(["./scripts/ops/opsctl.sh", *args], project_root=project_root, timeout_seconds=timeout_seconds)


def _storage_green(storage: dict[str, Any]) -> bool:
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    status = str(storage.get("overall_status") or "").strip().lower()
    severity = str(storage.get("severity") or "").strip().lower()
    shedding = storage.get("writer_shedding") if isinstance(storage.get("writer_shedding"), dict) else {}
    hard_breaches = [str(item or "").strip() for item in shedding.get("hard_breaches") or [] if str(item or "").strip()]
    total = _safe_int(backpressure.get("total_pending_lines"), 0)
    core = _safe_int(backpressure.get("core_pending_lines"), 0)
    oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    status_clear = bool(
        status in {"ready", "advisory", "ok", "stable", ""}
        or (status == "needs_work" and severity in {"ready", "advisory", "ok", "stable", "watch", ""} and not hard_breaches)
    )
    return bool(status_clear and severity not in {"blocked", "critical"} and total <= 15000 and core <= 15000 and oldest <= 240.0)


def _storage_needs_super_drain(storage: dict[str, Any]) -> bool:
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    status = str(storage.get("overall_status") or "").strip().lower()
    severity = str(storage.get("severity") or "").strip().lower()
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    total = _safe_int(backpressure.get("total_pending_lines"), 0)
    core = _safe_int(backpressure.get("core_pending_lines"), 0)
    oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 0), 15000)
    oldest_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), 0.0), 240.0)
    return bool(
        status in {"blocked", "critical"}
        or severity in {"blocked", "critical", "high"}
        or pressure_index >= 1.0
        or total > pending_threshold
        or core > pending_threshold
        or oldest > oldest_threshold
    )


def _launch_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    contract = runtime.get("training_launch_contract")
    return contract if isinstance(contract, dict) else {}


def _host_gate(runtime: dict[str, Any]) -> dict[str, Any]:
    gate = runtime.get("host_training_headroom_gate")
    return gate if isinstance(gate, dict) else {}


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("json") if isinstance(result.get("json"), dict) else {}
    return {
        "ok": bool(result.get("ok", False)),
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out", False)),
        "overall_status": str(payload.get("overall_status") or ""),
        "stdout_tail": str(result.get("stdout_tail") or "")[-500:],
        "stderr_tail": str(result.get("stderr_tail") or "")[-500:],
    }


def _append_event(project_root: Path, record: dict[str, Any]) -> None:
    event_dir = project_root / "governance" / "events" / "training"
    event_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().astimezone().strftime("%Y%m%d")
    path = event_dir / f"overnight_training_window_{day}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _window_label(now: datetime, small_start: datetime, small_end: datetime) -> str:
    if small_start <= now < small_end:
        return "small_batch_window"
    return "overnight_batch_window"


def run_cycle(
    *,
    project_root: Path,
    apply: bool,
    end_local: datetime,
    small_start: datetime,
    small_end: datetime,
    large_limit: int,
    small_limit: int,
    command_timeout_seconds: int,
    window_remaining: int | None = None,
) -> dict[str, Any]:
    now = datetime.now().astimezone()
    label = _window_label(now, small_start, small_end)
    base_limit = max(int(small_limit if label == "small_batch_window" else large_limit), 1)
    requested_limit = base_limit
    if window_remaining is not None:
        requested_limit = max(min(base_limit, max(int(window_remaining), 0)), 0)
    cycle: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "now_local": now.isoformat(),
        "end_local": end_local.isoformat(),
        "window_label": label,
        "base_limit": base_limit,
        "requested_limit": requested_limit,
        "window_remaining_before": window_remaining,
        "apply": bool(apply),
        "protected_volumes": {
            "VIDEO": "never_touched",
        },
        "training_cpu_allocation_contract": _training_env_overrides(),
        "steps": {},
        "launch_attempted": False,
        "launch_result": {},
        "launched_batch_size": 0,
    }

    if requested_limit <= 0:
        cycle["launch_result"] = {"skipped": True, "reason": "window_target_already_reached"}
        write_payload(_out_path(project_root), cycle)
        _append_event(project_root, cycle)
        return cycle

    apply_arg = ["--apply"] if apply else []
    cycle["steps"]["runtime_throttle"] = _compact_result(
        _run_ops(project_root, ["runtime-throttle", *apply_arg, "--max-renice-processes", "30", "--json"], timeout_seconds=command_timeout_seconds)
    )
    memory = _run_ops(project_root, ["memory-pressure-intelligence", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds)
    cycle["steps"]["memory_pressure_intelligence"] = _compact_result(memory)

    storage_result = _run_ops(project_root, ["ingestion-storage-control", "--json"], timeout_seconds=command_timeout_seconds)
    storage_payload = storage_result.get("json") if isinstance(storage_result.get("json"), dict) else {}
    cycle["steps"]["ingestion_storage_control"] = _compact_result(storage_result)
    cycle["storage_green"] = _storage_green(storage_payload)

    if apply and not cycle["storage_green"]:
        if _storage_needs_super_drain(storage_payload):
            super_drain = _run_ops(
                project_root,
                ["backpressure-super-drainer", "--apply", "--max-waves", "5", "--target-pending-lines", "15000", "--json"],
                timeout_seconds=max(command_timeout_seconds, 1800),
            )
            cycle["steps"]["backpressure_super_drainer"] = _compact_result(super_drain)
        drain = _run_ops(project_root, ["writer-cycle-coordinator", "--apply", "--json"], timeout_seconds=max(command_timeout_seconds, 900))
        cycle["steps"]["writer_cycle_coordinator"] = _compact_result(drain)
        storage_result = _run_ops(project_root, ["ingestion-storage-control", "--json"], timeout_seconds=command_timeout_seconds)
        storage_payload = storage_result.get("json") if isinstance(storage_result.get("json"), dict) else {}
        cycle["steps"]["ingestion_storage_control_after_drain"] = _compact_result(storage_result)
        cycle["storage_green"] = _storage_green(storage_payload)

    governor = _run_ops(project_root, ["autonomic-governor", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds)
    cycle["steps"]["autonomic_governor"] = _compact_result(governor)
    cycle["steps"]["runtime_training_snapshot"] = _compact_result(
        _run_ops(project_root, ["runtime-training-snapshot", "--reuse-if-fresh-minutes", "360", "--json"], timeout_seconds=max(command_timeout_seconds, 900))
    )
    cycle["steps"]["training_quality"] = _compact_result(
        _run_ops(project_root, ["training-quality", "--json"], timeout_seconds=command_timeout_seconds)
    )

    runtime = _run_ops(project_root, ["training-runtime-control", "--limit", str(requested_limit), "--json"], timeout_seconds=command_timeout_seconds)
    runtime_payload = runtime.get("json") if isinstance(runtime.get("json"), dict) else {}
    contract = _launch_contract(runtime_payload)
    host_gate = _host_gate(runtime_payload)
    cycle["steps"]["training_runtime_control"] = _compact_result(runtime)
    cycle["training_gate"] = {
        "overall_status": str(runtime_payload.get("overall_status") or ""),
        "launch_allowed": bool(contract.get("launch_allowed", False)),
        "mode": str(contract.get("mode") or ""),
        "launch_blockers": list(contract.get("launch_blockers") or []),
        "recommended_batch_size": _safe_int(contract.get("recommended_batch_size"), 0),
        "recommended_command": list(contract.get("recommended_retrain_command") or contract.get("recommended_command") or []),
        "host_status": str(host_gate.get("status") or ""),
        "host_batch_cap": _safe_int(host_gate.get("batch_cap"), 0),
        "memory_status": str(host_gate.get("memory_status") or ""),
        "memory_decision": str(host_gate.get("memory_decision") or ""),
    }

    command = list(contract.get("recommended_retrain_command") or contract.get("recommended_command") or [])
    if apply and bool(contract.get("launch_allowed", False)) and command:
        cycle["launch_attempted"] = True
        launch = _run_json(
            command,
            project_root=project_root,
            timeout_seconds=max(command_timeout_seconds, 1800),
            env_overrides=_training_env_overrides(),
        )
        cycle["launch_result"] = _compact_result(launch)
        cycle["launch_result"]["env_overrides"] = _training_env_overrides()
        if cycle["launch_result"].get("ok"):
            cycle["launched_batch_size"] = _safe_int(contract.get("recommended_batch_size"), 0)
    elif not apply:
        cycle["launch_result"] = {"skipped": True, "reason": "dry_run"}
    else:
        cycle["launch_result"] = {"skipped": True, "reason": "gate_not_clear"}

    write_payload(_out_path(project_root), cycle)
    _append_event(project_root, cycle)
    return cycle


def run_window(
    *,
    project_root: Path,
    apply: bool,
    end_local_text: str,
    small_start_text: str,
    small_end_text: str,
    large_limit: int,
    small_limit: int,
    window_target: int,
    poll_seconds: int,
    command_timeout_seconds: int,
    max_cycles: int,
) -> dict[str, Any]:
    now = datetime.now().astimezone()
    end_local = _next_wall_clock(now, end_local_text)
    small_start = _wall_clock_on_date(end_local, small_start_text)
    small_end = _wall_clock_on_date(end_local, small_end_text)
    if small_end <= small_start:
        small_end += timedelta(days=1)

    records: list[dict[str, Any]] = []
    cycles_run = 0
    launched_batch_total = 0
    stop_reason = ""
    target = max(int(window_target), 0)
    while True:
        current = datetime.now().astimezone()
        if current >= end_local:
            stop_reason = "window_end_reached"
            break
        if target > 0 and launched_batch_total >= target:
            stop_reason = "window_target_reached"
            break
        cycles_run += 1
        remaining = max(target - launched_batch_total, 0) if target > 0 else None
        record = run_cycle(
            project_root=project_root,
            apply=apply,
            end_local=end_local,
            small_start=small_start,
            small_end=small_end,
            large_limit=large_limit,
            small_limit=small_limit,
            command_timeout_seconds=command_timeout_seconds,
            window_remaining=remaining,
        )
        records.append(record)
        launched_batch_total += _safe_int(record.get("launched_batch_size"), 0)
        if bool(record.get("launch_attempted", False)) and not bool((record.get("launch_result") or {}).get("ok", False)):
            stop_reason = "launch_failed"
            break
        if target > 0 and launched_batch_total >= target:
            stop_reason = "window_target_reached"
            break
        if max_cycles > 0 and cycles_run >= max_cycles:
            stop_reason = "max_cycles_reached"
            break
        sleep_for = min(max(int(poll_seconds), 5), max(int((end_local - datetime.now().astimezone()).total_seconds()), 0))
        if sleep_for <= 0:
            stop_reason = "window_end_reached"
            break
        time.sleep(sleep_for)

    latest = records[-1] if records else {}
    summary = {
        "timestamp_utc": iso_now(),
        "overall_status": "running" if datetime.now().astimezone() < end_local and (max_cycles <= 0 or cycles_run < max_cycles) else "complete",
        "apply": bool(apply),
        "cycles_run": cycles_run,
        "stop_reason": stop_reason or "complete",
        "end_local": end_local.isoformat(),
        "small_batch_window": {
            "start_local": small_start.isoformat(),
            "end_local": small_end.isoformat(),
            "limit": max(int(small_limit), 1),
        },
        "large_batch_limit": max(int(large_limit), 1),
        "window_target": target,
        "launched_batch_total": launched_batch_total,
        "remaining_target": max(target - launched_batch_total, 0) if target > 0 else None,
        "last_training_gate": latest.get("training_gate") if isinstance(latest, dict) else {},
        "last_launch_attempted": bool(latest.get("launch_attempted", False)) if isinstance(latest, dict) else False,
        "last_launch_result": latest.get("launch_result") if isinstance(latest, dict) else {},
        "protected_volumes": {
            "VIDEO": "never_touched",
        },
        "records_tail": records[-5:],
    }
    write_payload(_out_path(project_root), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a gate-aware overnight training window that waits for backlog and host headroom before launching.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--end-local", default="08:00")
    parser.add_argument("--small-window-start-local", default="06:00")
    parser.add_argument("--small-window-end-local", default="08:00")
    parser.add_argument("--large-limit", type=int, default=30)
    parser.add_argument("--small-limit", type=int, default=2)
    parser.add_argument("--window-target", type=int, default=0, help="Total successful bot trainings to launch before stopping; 0 means run until the window ends.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--command-timeout-seconds", type=int, default=420)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_window(
        project_root=Path(args.project_root).resolve(),
        apply=bool(args.apply),
        end_local_text=str(args.end_local),
        small_start_text=str(args.small_window_start_local),
        small_end_text=str(args.small_window_end_local),
        large_limit=int(args.large_limit),
        small_limit=int(args.small_limit),
        window_target=int(args.window_target),
        poll_seconds=int(args.poll_seconds),
        command_timeout_seconds=int(args.command_timeout_seconds),
        max_cycles=int(args.max_cycles),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        gate = payload.get("last_training_gate") if isinstance(payload.get("last_training_gate"), dict) else {}
        print(
            "overnight_training_window "
            f"overall_status={payload.get('overall_status', '')} "
            f"cycles_run={payload.get('cycles_run', 0)} "
            f"launch_allowed={int(bool(gate.get('launch_allowed', False)))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
