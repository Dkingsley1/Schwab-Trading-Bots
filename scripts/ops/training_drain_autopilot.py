#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_drain_autopilot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "training_drain_autopilot.lock"
DEFAULT_EVENT_DIR = PROJECT_ROOT / "governance" / "events" / "training"
PROTECTED_VOLUMES = ("/Volumes/VIDEO",)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _process_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _write_retrain_launch_payload(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for raw_path in [
        payload.get("artifact_path"),
        payload.get("latest_path"),
        *(payload.get("latest_alias_paths") if isinstance(payload.get("latest_alias_paths"), list) else []),
    ]:
        text = str(raw_path or "").strip()
        if not text or text in paths:
            continue
        write_payload(Path(text), payload)
        paths.append(text)
    return paths


def _finalize_timed_out_retrain_launch(project_root: Path, launch_result: dict[str, Any]) -> dict[str, Any]:
    if not bool(launch_result.get("timed_out", False)):
        return {}
    latest_path = project_root / "governance" / "health" / "retrain_launch_latest.json"
    payload = _load_json(latest_path)
    if not payload:
        return {"status": "missing_retrain_launch_latest", "latest_path": str(latest_path)}
    if str(payload.get("state") or "").strip().lower() != "running":
        return {
            "status": "latest_not_running",
            "latest_path": str(latest_path),
            "state": str(payload.get("state") or ""),
            "final_status": str(payload.get("final_status") or ""),
        }
    pid = _safe_int(payload.get("pid"), 0)
    alive = _process_alive(pid)
    if alive:
        return {
            "status": "process_still_alive",
            "latest_path": str(latest_path),
            "pid": pid,
            "phase": str(payload.get("phase") or ""),
            "progress": payload.get("progress") if isinstance(payload.get("progress"), dict) else {},
        }
    finalized = dict(payload)
    finalized["state"] = "completed"
    finalized["ended_utc"] = iso_now()
    finalized["final_status"] = "timed_out_by_training_drain_autopilot"
    finalized["exit_code"] = 124
    finalized["timeout_source"] = "training_drain_autopilot"
    finalized["timeout_phase"] = str(payload.get("phase") or "")
    finalized["timeout_progress"] = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    finalized["autopilot_timeout"] = {
        "returncode": launch_result.get("returncode"),
        "stdout_tail": str(launch_result.get("stdout_tail") or "")[-1200:],
        "stderr_tail": str(launch_result.get("stderr_tail") or "")[-1200:],
    }
    written = _write_retrain_launch_payload(finalized)
    return {
        "status": "finalized_timeout",
        "latest_path": str(latest_path),
        "written_paths": written,
        "pid": pid,
        "timeout_phase": finalized["timeout_phase"],
        "timeout_progress": finalized["timeout_progress"],
    }


def _acquire_lock(path: Path) -> tuple[Any | None, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            handle.seek(0)
            owner = handle.read().strip()
        except Exception:
            owner = ""
        handle.close()
        return None, owner
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={iso_now()}\n")
    handle.flush()
    return handle, ""


def _training_env_overrides() -> dict[str, str]:
    workers = max(1, min(_safe_int(os.getenv("TRAINING_PCORE_MAX_WORKERS"), 3), 7))
    nice = max(0, min(_safe_int(os.getenv("TRAINING_PCORE_NICE"), 2), 6))
    research_nice = max(0, min(_safe_int(os.getenv("RUNTIME_THROTTLE_RESEARCH_NICE"), 2), 6))
    return {
        "BOT_CPU_ALLOCATION_POLICY": "performance_core_primary",
        "BOT_CPU_QOS_POLICY": "performance_core_primary_no_background_writer",
        "BOT_CPU_SCHEDULER_INTENT": "performance_core_training",
        "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN": "1",
        "TRAINING_PCORE_MAX_WORKERS": str(workers),
        "TRAINING_PCORE_NICE": str(nice),
        "RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND": "0",
        "RUNTIME_THROTTLE_RESEARCH_NICE": str(research_nice),
        "PYTHONUNBUFFERED": "1",
        "RETRAIN_GREEN_MEMORY_SWAP_RELIEF": "1",
        "BOT_PROTECTED_VOLUME_DENYLIST": ",".join(PROTECTED_VOLUMES),
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
        returncode = int(completed.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        returncode = 124
        timed_out = True
    except Exception as exc:
        stdout = ""
        stderr = str(exc)
        returncode = 125
        timed_out = False

    parsed: Any = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            candidate = json.loads(raw)
        except Exception:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break

    return {
        "command": command,
        "returncode": returncode,
        "ok": returncode == 0,
        "timed_out": timed_out,
        "started_local": started.isoformat(),
        "finished_local": datetime.now().astimezone().isoformat(),
        "json": parsed if isinstance(parsed, dict) else {},
        "env_overrides": env_overrides or {},
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
    }


def _ops(project_root: Path, args: list[str], *, timeout_seconds: int, env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    return _run_json(["./scripts/ops/opsctl.sh", *args], project_root=project_root, timeout_seconds=timeout_seconds, env_overrides=env_overrides)


def _compact_step(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("json") if isinstance(result.get("json"), dict) else {}
    return {
        "ok": bool(result.get("ok", False)),
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out", False)),
        "overall_status": str(payload.get("overall_status") or payload.get("status") or ""),
        "stdout_tail": str(result.get("stdout_tail") or "")[-900:],
        "stderr_tail": str(result.get("stderr_tail") or "")[-900:],
    }


def _launch_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    contract = runtime.get("training_launch_contract")
    if isinstance(contract, dict):
        return contract
    reentry = runtime.get("reentry_gate")
    if isinstance(reentry, dict):
        return {
            "launch_allowed": bool(reentry.get("allowed", False)),
            "mode": reentry.get("mode"),
            "launch_blockers": reentry.get("blockers") or [],
            "recommended_batch_size": reentry.get("max_parallel_trainings") or 0,
            "recommended_retrain_command": reentry.get("recommended_command") or [],
        }
    return {}


def _host_gate(runtime: dict[str, Any]) -> dict[str, Any]:
    gate = runtime.get("host_training_headroom_gate")
    return gate if isinstance(gate, dict) else {}


def _command_from_contract(contract: dict[str, Any]) -> list[str]:
    for key in ("recommended_retrain_command", "recommended_command", "command"):
        raw = contract.get(key)
        if isinstance(raw, list) and raw:
            return [str(part) for part in raw]
    return []


def _training_gate_summary(runtime_payload: dict[str, Any]) -> dict[str, Any]:
    contract = _launch_contract(runtime_payload)
    host_gate = _host_gate(runtime_payload)
    blockers = [str(item) for item in list(contract.get("launch_blockers") or []) if str(item).strip()]
    command = _command_from_contract(contract)
    return {
        "overall_status": str(runtime_payload.get("overall_status") or ""),
        "launch_allowed": bool(contract.get("launch_allowed", False)),
        "mode": str(contract.get("mode") or ""),
        "launch_blockers": blockers,
        "recommended_batch_size": _safe_int(contract.get("recommended_batch_size"), 0),
        "recommended_command": command,
        "host_status": str(host_gate.get("status") or ""),
        "host_batch_cap": _safe_int(host_gate.get("batch_cap"), 0),
        "memory_status": str(host_gate.get("memory_status") or ""),
        "memory_decision": str(host_gate.get("memory_decision") or ""),
    }


def _storage_summary(storage_payload: dict[str, Any]) -> dict[str, Any]:
    backpressure = storage_payload.get("backpressure") if isinstance(storage_payload.get("backpressure"), dict) else {}
    storage = storage_payload.get("storage") if isinstance(storage_payload.get("storage"), dict) else {}
    return {
        "overall_status": str(storage_payload.get("overall_status") or ""),
        "severity": str(storage_payload.get("severity") or ""),
        "pressure_index": round(_safe_float(storage_payload.get("pressure_index"), 0.0), 3),
        "total_pending_lines": _safe_int(backpressure.get("total_pending_lines"), 0),
        "core_pending_lines": _safe_int(backpressure.get("core_pending_lines"), 0),
        "oldest_pending_age_seconds": round(_safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0), 3),
        "retention_debt_gb": round(_safe_float(storage.get("retention_debt_gb"), 0.0), 3),
    }


def _needs_storage_compaction(gate: dict[str, Any], quality_payload: dict[str, Any], storage_payload: dict[str, Any]) -> bool:
    text = json.dumps(
        {
            "gate": gate,
            "quality": quality_payload,
            "storage": storage_payload,
        },
        ensure_ascii=True,
        sort_keys=True,
    ).lower()
    return any(token in text for token in ("storage_quota_hard_breach", "governance_telemetry", "quota hard", "hard_breach"))


def _needs_backpressure_work(gate: dict[str, Any], storage_payload: dict[str, Any]) -> bool:
    storage = _storage_summary(storage_payload)
    blockers = " ".join(gate.get("launch_blockers") or []).lower()
    return bool(
        "backpressure" in blockers
        or storage["overall_status"] in {"blocked", "needs_work"}
        or storage["severity"] in {"blocked", "critical", "high"}
        or storage["pressure_index"] >= 1.0
        or storage["total_pending_lines"] > 15000
        or storage["core_pending_lines"] > 15000
        or storage["oldest_pending_age_seconds"] > 240.0
    )


def _append_event(project_root: Path, record: dict[str, Any]) -> None:
    event_dir = project_root / "governance" / "events" / "training"
    event_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().astimezone().strftime("%Y%m%d")
    path = event_dir / f"training_drain_autopilot_{day}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _bootstrap_steps(
    *,
    project_root: Path,
    apply: bool,
    command_timeout_seconds: int,
) -> dict[str, Any]:
    apply_arg = ["--apply"] if apply else []
    env = _training_env_overrides()
    steps: dict[str, Any] = {}
    steps["backlog_pcore_accelerator"] = _compact_step(
        _ops(project_root, ["backlog-pcore-accelerator", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds)
    )
    steps["backlog_pump_infrabots"] = _compact_step(
        _ops(project_root, ["backlog-pump-infrabots", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds)
    )
    steps["backlog_drain_uniform_process"] = _compact_step(
        _ops(project_root, ["backlog-drain-uniform-process", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds)
    )
    steps["runtime_throttle"] = _compact_step(
        _ops(project_root, ["runtime-throttle", *apply_arg, "--max-renice-processes", "30", "--json"], timeout_seconds=command_timeout_seconds)
    )
    steps["memory_pressure_intelligence"] = _compact_step(
        _ops(project_root, ["memory-pressure-intelligence", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds)
    )
    steps["autonomic_governor"] = _compact_step(
        _ops(project_root, ["autonomic-governor", *apply_arg, "--json"], timeout_seconds=command_timeout_seconds, env_overrides=env)
    )
    steps["runtime_training_snapshot"] = _compact_step(
        _ops(project_root, ["runtime-training-snapshot", "--reuse-if-fresh-minutes", "360", "--json"], timeout_seconds=max(command_timeout_seconds, 900))
    )
    return steps


def run_cycle(
    *,
    project_root: Path,
    apply: bool,
    prep_only: bool,
    limit: int,
    command_timeout_seconds: int,
    storage_autopilot_cycles: int,
    target_free_gb: float,
    min_telemetry_file_mb: float,
    max_telemetry_files: int,
    poll_seconds: int,
    wait_timeout_seconds: int,
) -> dict[str, Any]:
    cycle: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "apply": bool(apply),
        "requested_limit": max(int(limit), 1),
        "protected_volumes": {"VIDEO": "never_touched"},
        "training_cpu_allocation_contract": _training_env_overrides(),
        "steps": {},
        "launch_attempted": False,
        "prep_only": bool(prep_only),
        "launch_result": {},
    }

    cycle["steps"].update(_bootstrap_steps(project_root=project_root, apply=apply, command_timeout_seconds=command_timeout_seconds))

    storage_initial = _ops(project_root, ["ingestion-storage-control", "--json"], timeout_seconds=command_timeout_seconds)
    storage_payload = storage_initial.get("json") if isinstance(storage_initial.get("json"), dict) else {}
    cycle["steps"]["ingestion_storage_control_initial"] = _compact_step(storage_initial)
    cycle["storage_initial"] = _storage_summary(storage_payload)

    quality = _ops(project_root, ["training-quality", "--json"], timeout_seconds=command_timeout_seconds)
    quality_payload = quality.get("json") if isinstance(quality.get("json"), dict) else {}
    cycle["steps"]["training_quality_initial"] = _compact_step(quality)

    runtime = _ops(project_root, ["training-runtime-control", "--limit", str(max(int(limit), 1)), "--json"], timeout_seconds=command_timeout_seconds)
    runtime_payload = runtime.get("json") if isinstance(runtime.get("json"), dict) else {}
    gate = _training_gate_summary(runtime_payload)
    cycle["steps"]["training_runtime_control_initial"] = _compact_step(runtime)
    cycle["training_gate_initial"] = gate

    remediation: list[str] = []
    if apply and _needs_storage_compaction(gate, quality_payload, storage_payload):
        remediation.append("governance_telemetry_compactor")
        compactor = _ops(
            project_root,
            [
                "governance-telemetry-compactor",
                "--apply",
                "--target-free-gb",
                str(max(float(target_free_gb), 1.0)),
                "--min-file-mb",
                str(max(float(min_telemetry_file_mb), 1.0)),
                "--max-files",
                str(max(int(max_telemetry_files), 1)),
                "--json",
            ],
            timeout_seconds=max(command_timeout_seconds, 1800),
        )
        cycle["steps"]["governance_telemetry_compactor"] = _compact_step(compactor)

    if apply and _needs_backpressure_work(gate, storage_payload):
        remediation.append("storage_backpressure_autopilot")
        storage_auto = _ops(
            project_root,
            [
                "storage-backpressure-autopilot",
                "--apply",
                "--poll-seconds",
                str(max(int(poll_seconds), 5)),
                "--wait-timeout-seconds",
                str(max(int(wait_timeout_seconds), 60)),
                "--command-timeout-seconds",
                str(max(int(command_timeout_seconds), 420)),
                "--max-cycles",
                str(max(int(storage_autopilot_cycles), 1)),
                "--json",
            ],
            timeout_seconds=max(int(command_timeout_seconds) * max(int(storage_autopilot_cycles), 1), 1800),
        )
        cycle["steps"]["storage_backpressure_autopilot"] = _compact_step(storage_auto)

    cycle["remediation_applied"] = remediation

    storage_after = _ops(project_root, ["ingestion-storage-control", "--json"], timeout_seconds=command_timeout_seconds)
    storage_after_payload = storage_after.get("json") if isinstance(storage_after.get("json"), dict) else {}
    cycle["steps"]["ingestion_storage_control_after"] = _compact_step(storage_after)
    cycle["storage_after"] = _storage_summary(storage_after_payload)

    quality_after = _ops(project_root, ["training-quality", "--json"], timeout_seconds=command_timeout_seconds)
    cycle["steps"]["training_quality_after"] = _compact_step(quality_after)

    runtime_after = _ops(project_root, ["training-runtime-control", "--limit", str(max(int(limit), 1)), "--json"], timeout_seconds=command_timeout_seconds)
    runtime_after_payload = runtime_after.get("json") if isinstance(runtime_after.get("json"), dict) else {}
    final_gate = _training_gate_summary(runtime_after_payload)
    cycle["steps"]["training_runtime_control_after"] = _compact_step(runtime_after)
    cycle["training_gate_after"] = final_gate

    command = list(final_gate.get("recommended_command") or [])
    if apply and prep_only and final_gate.get("launch_allowed"):
        cycle["launch_result"] = {"skipped": True, "reason": "prep_only_mode"}
    elif apply and final_gate.get("launch_allowed") and command:
        cycle["launch_attempted"] = True
        launch = _run_json(
            command,
            project_root=project_root,
            timeout_seconds=max(int(command_timeout_seconds), 1800),
            env_overrides=_training_env_overrides(),
        )
        cycle["launch_result"] = _compact_step(launch)
        cycle["launch_result"]["env_overrides"] = _training_env_overrides()
        if bool(cycle["launch_result"].get("timed_out", False)):
            cycle["launch_result"]["retrain_launch_timeout_finalization"] = _finalize_timed_out_retrain_launch(
                project_root,
                cycle["launch_result"],
            )
    elif not apply:
        cycle["launch_result"] = {"skipped": True, "reason": "dry_run"}
    elif final_gate.get("launch_allowed") and not command:
        cycle["launch_result"] = {"skipped": True, "reason": "gate_clear_but_no_recommended_command"}
    else:
        cycle["launch_result"] = {"skipped": True, "reason": "gate_not_clear"}

    if cycle["launch_attempted"]:
        cycle["overall_status"] = "launched_training"
    elif prep_only and final_gate.get("launch_allowed"):
        cycle["overall_status"] = "prep_complete_training_ready"
    elif final_gate.get("launch_allowed"):
        cycle["overall_status"] = "ready_no_command"
    elif remediation:
        cycle["overall_status"] = "remediated_waiting_for_next_cycle"
    else:
        cycle["overall_status"] = "blocked_waiting"
    return cycle


def run_autopilot(
    *,
    project_root: Path,
    apply: bool,
    prep_only: bool,
    limit: int,
    max_cycles: int,
    poll_seconds: int,
    command_timeout_seconds: int,
    wait_timeout_seconds: int,
    storage_autopilot_cycles: int,
    target_free_gb: float,
    min_telemetry_file_mb: float,
    max_telemetry_files: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cycles = max(int(max_cycles), 1)
    for index in range(cycles):
        record = run_cycle(
            project_root=project_root,
            apply=apply,
            prep_only=prep_only,
            limit=limit,
            command_timeout_seconds=command_timeout_seconds,
            storage_autopilot_cycles=storage_autopilot_cycles,
            target_free_gb=target_free_gb,
            min_telemetry_file_mb=min_telemetry_file_mb,
            max_telemetry_files=max_telemetry_files,
            poll_seconds=poll_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        record["cycle_index"] = index + 1
        records.append(record)
        _append_event(project_root, record)
        if record.get("launch_attempted") or record.get("overall_status") in {"ready_no_command", "prep_complete_training_ready"}:
            break
        if index + 1 < cycles:
            time.sleep(max(int(poll_seconds), 5))

    latest = records[-1] if records else {}
    summary = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": str(latest.get("overall_status") or "blocked_waiting"),
        "apply": bool(apply),
        "prep_only": bool(prep_only),
        "cycles_run": len(records),
        "requested_limit": max(int(limit), 1),
        "last_training_gate": latest.get("training_gate_after") if isinstance(latest, dict) else {},
        "last_launch_attempted": bool(latest.get("launch_attempted", False)) if isinstance(latest, dict) else False,
        "last_launch_result": latest.get("launch_result") if isinstance(latest, dict) else {},
        "last_storage_before": latest.get("storage_initial") if isinstance(latest, dict) else {},
        "last_storage_after": latest.get("storage_after") if isinstance(latest, dict) else {},
        "remediation_applied": latest.get("remediation_applied") if isinstance(latest, dict) else [],
        "protected_volumes": {"VIDEO": "never_touched"},
        "integration_contract": {
            "drives_backlog_pcore_accelerator": True,
            "drives_backlog_pump_infrabots": True,
            "drives_backlog_drain_uniform_process": True,
            "drives_storage_backpressure_autopilot": True,
            "clears_governance_telemetry_quota_before_training": True,
            "launches_only_system_recommended_training_command": not bool(prep_only),
            "training_launches_disabled_by_prep_only": bool(prep_only),
            "uses_performance_core_training_contract": True,
            "never_touch_protected_volumes": list(PROTECTED_VOLUMES),
        },
        "records_tail": records[-3:],
    }
    write_payload(project_root / "governance" / "health" / DEFAULT_OUT_PATH.name, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Automatically clear drain/storage blockers, then launch recommended training when the gate is clear.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prep-only", action="store_true", help="Run remediation and gate refreshes without launching training even if the gate clears.")
    parser.add_argument("--limit", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_LIMIT", "4")))
    parser.add_argument("--max-cycles", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_MAX_CYCLES", "1")))
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_POLL_SECONDS", "120")))
    parser.add_argument("--command-timeout-seconds", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_COMMAND_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--wait-timeout-seconds", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_WAIT_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--storage-autopilot-cycles", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_STORAGE_CYCLES", "1")))
    parser.add_argument("--target-free-gb", type=float, default=float(os.getenv("TRAINING_DRAIN_AUTOPILOT_TELEMETRY_TARGET_FREE_GB", "32")))
    parser.add_argument("--min-telemetry-file-mb", type=float, default=float(os.getenv("TRAINING_DRAIN_AUTOPILOT_MIN_TELEMETRY_FILE_MB", "64")))
    parser.add_argument("--max-telemetry-files", type=int, default=int(os.getenv("TRAINING_DRAIN_AUTOPILOT_MAX_TELEMETRY_FILES", "64")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    lock_path = Path(args.lock_path).expanduser()
    if not lock_path.is_absolute():
        lock_path = project_root / lock_path

    lock_handle = None
    if args.apply:
        lock_handle, owner = _acquire_lock(lock_path)
        if lock_handle is None:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "overall_status": "already_running",
                "lock_owner": owner,
                "protected_volumes": {"VIDEO": "never_touched"},
            }
            write_payload(out_path, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("training_drain_autopilot overall_status=already_running")
            return 0

    try:
        payload = run_autopilot(
            project_root=project_root,
            apply=bool(args.apply),
            prep_only=bool(args.prep_only),
            limit=int(args.limit),
            max_cycles=int(args.max_cycles),
            poll_seconds=int(args.poll_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
            wait_timeout_seconds=int(args.wait_timeout_seconds),
            storage_autopilot_cycles=int(args.storage_autopilot_cycles),
            target_free_gb=float(args.target_free_gb),
            min_telemetry_file_mb=float(args.min_telemetry_file_mb),
            max_telemetry_files=int(args.max_telemetry_files),
        )
        write_payload(out_path, payload)
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_handle.close()

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        gate = payload.get("last_training_gate") if isinstance(payload.get("last_training_gate"), dict) else {}
        print(
            "training_drain_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"cycles_run={payload.get('cycles_run', 0)} "
            f"launch_allowed={int(bool(gate.get('launch_allowed', False)))} "
            f"launch_attempted={int(bool(payload.get('last_launch_attempted', False)))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
