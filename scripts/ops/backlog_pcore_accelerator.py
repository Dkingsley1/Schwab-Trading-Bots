#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_pcore_accelerator_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.backlog_pcore_accelerator_override"
BACKLOG_GREEN_AGE_SECONDS = 900.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    for key in ("overall_status", "status"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return default


def _storage_metrics(storage: dict[str, Any], governor: dict[str, Any]) -> dict[str, Any]:
    governor_storage = _as_dict(governor.get("storage_metrics"))
    backpressure = _as_dict(storage.get("backpressure"))
    stale = _as_dict(storage.get("stale_pending_locator"))
    oldest_sources = _as_list(governor_storage.get("oldest_sources")) or _as_list(stale.get("oldest_sources"))
    core = _safe_int(governor_storage.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    total = _safe_int(governor_storage.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    overlay = _safe_int(governor_storage.get("overlay_pending_lines"), 0)
    oldest_age = _safe_float(governor_storage.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    target = _safe_int(governor_storage.get("target_pending_lines"), _safe_int(backpressure.get("pending_lines_threshold"), 15000)) or 15000
    line_green = core <= target and total <= max(target, core)
    age_green = oldest_age <= BACKLOG_GREEN_AGE_SECONDS
    overlay_green = overlay <= target if overlay > 0 else True
    green = bool(line_green and age_green and overlay_green)
    return {
        "core_pending_lines": core,
        "total_pending_lines": total,
        "overlay_pending_lines": overlay,
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "target_pending_lines": target,
        "line_green": line_green,
        "age_green": age_green,
        "overlay_green": overlay_green,
        "green": green,
        "oldest_sources": oldest_sources[:8],
    }


def _writer_state(writer: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(writer.get("writer_state_before")) or _as_dict(writer.get("writer_state_after_wait"))


def _writer_active(writer: dict[str, Any], writer_intel: dict[str, Any]) -> bool:
    state = _writer_state(writer)
    health = _as_dict(writer_intel.get("writer_health"))
    return bool(
        state.get("active", False)
        or state.get("running", False)
        or health.get("active", False)
        or str(health.get("state") or "") in {"active_progressing", "stale_progress", "stalled"}
    )


def _process_topology(writer_intel: dict[str, Any]) -> dict[str, Any]:
    topology = _as_dict(writer_intel.get("process_topology"))
    return {
        "sql_link_writer_running_count": _safe_int(topology.get("sql_link_writer_running_count"), 1),
        "raw_sql_link_writer_running_count": _safe_int(topology.get("raw_sql_link_writer_running_count"), 1),
        "duplicate_sql_writer_processes": bool(topology.get("duplicate_sql_writer_processes", False)),
        "process_watchdog_status": str(topology.get("process_watchdog_status") or "unknown"),
        "process_fanout_status": str(topology.get("process_fanout_status") or "unknown"),
    }


def _host_lane_contract(governor: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    lanes = _as_dict(governor.get("host_lane_budget"))
    allocation = _as_dict(lanes.get("p_core_allocation_contract"))
    memory_class = _as_dict(memory.get("classification"))
    observer = _as_dict(memory.get("observer_overhead"))
    p_workers = _safe_int(lanes.get("selected_p_core_preprocess_workers"), _safe_int(memory_class.get("recommended_p_core_worker_cap"), 1))
    p_workers = max(p_workers, 1)
    return {
        "primary_compute_lanes": _safe_int(lanes.get("primary_compute_lanes"), 1),
        "selected_p_core_preprocess_workers": p_workers,
        "user_app_reserved_p_cores": _safe_int(allocation.get("user_app_reserved_p_cores"), 0),
        "efficiency_core_spillover": _safe_int(lanes.get("efficiency_core_spillover"), 0),
        "efficiency_core_total": _safe_int(lanes.get("efficiency_core_total"), 0),
        "memory_status": str(memory_class.get("status") or "unknown"),
        "memory_worker_cap": _safe_int(memory_class.get("recommended_p_core_worker_cap"), p_workers),
        "memory_safe_to_widen": bool(_as_dict(memory.get("reopen_gate")).get("safe_to_widen_p_core_workers", False)),
        "observer_overhead_active": bool(observer.get("active", False)),
        "policy": str(lanes.get("policy") or "performance_core_primary_single_writer_with_user_app_reserve"),
    }


def _accelerator_lanes(storage: dict[str, Any], host_lanes: dict[str, Any]) -> list[dict[str, Any]]:
    p_workers = max(_safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 1), 1)
    oldest_sources = _as_list(storage.get("oldest_sources"))
    hot_source_count = min(len(oldest_sources), max(p_workers, 1))
    density_workers = max(min(2, p_workers - 1), 1)
    stale_workers = max(min(2, p_workers), 1)
    return [
        {
            "lane": "stale_source_locator",
            "class": "p_core_preprocess",
            "workers": stale_workers,
            "writes_sqlite": False,
            "purpose": "identify oldest exact files and shards before a writer pass",
        },
        {
            "lane": "jsonl_density_sampler",
            "class": "p_core_preprocess",
            "workers": density_workers,
            "writes_sqlite": False,
            "purpose": "sample sparse or huge JSONL files without letting them monopolize the writer",
        },
        {
            "lane": "shard_priority_planner",
            "class": "p_core_preprocess",
            "workers": 1,
            "writes_sqlite": False,
            "purpose": "rank hot, warm, and cold shards before handoff to the single writer",
        },
        {
            "lane": "oldest_work_catchup_scheduler",
            "class": "p_core_preprocess",
            "workers": max(min(hot_source_count, p_workers), 1),
            "writes_sqlite": False,
            "purpose": "schedule bounded catch-up waves around the oldest pending work",
        },
        {
            "lane": "sqlite_single_writer",
            "class": "exclusive_sqlite_writer",
            "workers": 1,
            "writes_sqlite": True,
            "purpose": "perform all SQLite writes through the one lock-owning writer",
        },
    ]


def _wave_policy(storage: dict[str, Any], host_lanes: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    p_workers = max(_safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 1), 1)
    memory_status = str(host_lanes.get("memory_status") or "unknown")
    runtime_status = _status(runtime)
    if memory_status in {"hard_relief", "swap_relief", "compression_relief"}:
        max_seconds = 20
        waves = 2
        mode = "memory_relief_bounded"
    elif runtime_status in {"blocked", "critical", "degraded"}:
        max_seconds = 25
        waves = 3
        mode = "runtime_guarded"
    elif storage.get("green"):
        max_seconds = 15
        waves = 1
        mode = "maintenance"
    else:
        max_seconds = 35 if p_workers >= 4 else 25
        waves = 3
        mode = "p_core_catch_up"
    return {
        "mode": mode,
        "bounded_wave_limit": waves,
        "max_seconds_per_writer_cycle": max_seconds,
        "min_recheck_seconds": 45,
        "stop_conditions": [
            "one active SQL writer already owns the lock",
            "oldest pending age is below 15 minutes",
            "memory pressure moves from soft_guard to hard_relief/swap_relief",
            "backlog trend regresses after a writer pass",
        ],
    }


def _grade(storage: dict[str, Any], host_lanes: dict[str, Any], writer_active: bool, topology: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    if not bool(topology.get("duplicate_sql_writer_processes", False)):
        score += 20
    else:
        reasons.append("duplicate writer process risk")
    if bool(storage.get("line_green", False)):
        score += 20
    else:
        reasons.append("line backlog above target")
    if bool(storage.get("age_green", False)):
        score += 20
    else:
        reasons.append("oldest pending age not green")
    if str(host_lanes.get("memory_status") or "") in {"clear", "foreground_headroom", "soft_guard"}:
        score += 15
    else:
        reasons.append("memory still in relief mode")
    if _safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 0) >= 3:
        score += 15
    else:
        reasons.append("P-core accelerator width is too narrow")
    if not bool(_as_dict(memory.get("observer_overhead")).get("active", False)):
        score += 5
    else:
        reasons.append("observer overhead is distorting pressure")
    if writer_active or storage.get("green"):
        score += 5
    else:
        reasons.append("writer is idle while backlog is not green")
    if score >= 90:
        letter = "A"
    elif score >= 80:
        letter = "B"
    elif score >= 70:
        letter = "C"
    elif score >= 60:
        letter = "D"
    else:
        letter = "F"
    return {
        "score": score,
        "letter": letter,
        "reasons": reasons,
        "policy": "grades_backlog_drain_bulletproofing_not_market_or_strategy_quality",
    }


def _decision(storage: dict[str, Any], writer: dict[str, Any], writer_intel: dict[str, Any], host_lanes: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    state = _writer_state(writer)
    completed = _safe_int(state.get("completed_shard_count"), 0)
    planned = _safe_int(state.get("planned_shard_count"), 0)
    active = _writer_active(writer, writer_intel)
    duplicate = bool(topology.get("duplicate_sql_writer_processes", False))
    memory_status = str(host_lanes.get("memory_status") or "unknown")
    if duplicate:
        action = "enforce_single_writer_guard"
        command = ["./scripts/ops/opsctl.sh", "process-fanout-guard", "--apply", "--json"]
        reason = "duplicate SQL writer risk must be cleared before accelerating backlog"
        apply_safe = True
    elif active:
        action = "observe_active_writer"
        command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]
        reason = f"writer is active at {completed}/{planned} shards; do not launch a competing writer"
        apply_safe = False
    elif memory_status in {"hard_relief", "swap_relief"}:
        action = "hold_for_memory_relief"
        command = ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"]
        reason = "memory relief is too strong for new backlog waves"
        apply_safe = True
    elif not bool(storage.get("green", False)):
        action = "run_bounded_p_core_catch_up"
        command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"]
        reason = "backlog lines or age are not green; run bounded waves through the single writer"
        apply_safe = True
    else:
        action = "park_to_maintenance"
        command = ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"]
        reason = "backlog is green; keep accelerators in maintenance and allow other gates to re-open gradually"
        apply_safe = True
    return {
        "action": action,
        "next_command": command,
        "apply_safe": apply_safe,
        "reason": reason,
        "writer_shards": {"completed": completed, "planned": planned, "step": state.get("current_step", ""), "status": state.get("status", "")},
    }


def _needs(storage: dict[str, Any], decision: dict[str, Any], host_lanes: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    if not bool(storage.get("green", False)):
        oldest = _as_list(storage.get("oldest_sources"))
        exact = oldest[0] if oldest and isinstance(oldest[0], dict) else {}
        needs.append(
            {
                "blocker": "backlog_age_or_lines_not_green_for_p_core_acceleration",
                "exact_file": exact.get("source_rel") or "governance/health/ingestion_storage_control_latest.json",
                "exact_shard": exact.get("shard") or "",
                "command": decision.get("next_command", []),
                "expected_impact": "Uses P-core preprocess accelerators to prioritize stale work, then hands one bounded batch to the exclusive SQLite writer.",
                "risk_level": "low" if decision.get("apply_safe") else "observe",
                "stop_when": "oldest pending age is under 15 minutes and core/overlay pending are below target.",
            }
        )
    if str(host_lanes.get("memory_status") or "") not in {"clear", "foreground_headroom"}:
        needs.append(
            {
                "blocker": "memory_headroom_limits_backlog_p_core_width",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
                "expected_impact": "Refreshes the memory gate before widening P-core backlog accelerators.",
                "risk_level": "low",
                "stop_when": "memory is clear for two consecutive samples or the cap reaches the benchmark limit.",
            }
        )
    if bool(_as_dict(memory.get("observer_overhead")).get("active", False)):
        needs.append(
            {
                "blocker": "observer_overhead_distorts_backlog_pressure",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": [],
                "expected_impact": "Closing or reducing high-overhead monitors makes backlog pressure readings cleaner.",
                "risk_level": "operator_choice",
                "stop_when": "observer_overhead.active is false.",
            }
        )
    return needs


def _env_lines(payload: dict[str, Any]) -> list[str]:
    host_lanes = _as_dict(payload.get("host_lane_contract"))
    wave = _as_dict(payload.get("wave_policy"))
    decision = _as_dict(payload.get("decision_packet"))
    env = {
        "BACKLOG_PCORE_ACCELERATOR_ENABLED": "1",
        "BACKLOG_PCORE_ACCELERATOR_ACTION": str(decision.get("action") or "observe"),
        "BACKLOG_PCORE_ACCELERATOR_WORKERS": str(host_lanes.get("selected_p_core_preprocess_workers") or 1),
        "BACKLOG_PCORE_USER_APP_RESERVE": str(host_lanes.get("user_app_reserved_p_cores") or 0),
        "BACKLOG_ECORE_SPILLOVER_WORKERS": str(host_lanes.get("efficiency_core_spillover") or 0),
        "BACKLOG_SQLITE_WRITER_WORKERS": "1",
        "BACKLOG_ACCELERATOR_SQLITE_PARALLELISM": "1",
        "BACKLOG_CATCH_UP_WAVE_LIMIT": str(wave.get("bounded_wave_limit") or 1),
        "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": str(wave.get("max_seconds_per_writer_cycle") or 25),
        "BACKLOG_ACCELERATOR_SINGLE_WRITER_GUARD": "1",
        "BOT_CPU_ALLOCATION_POLICY": "performance_core_primary_single_writer_with_user_app_reserve",
    }
    return [f"{key}={shlex.quote(value)}" for key, value in env.items()]


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    apply: bool = False,
) -> dict[str, Any]:
    write_payload(out_path, payload)
    applied = False
    if apply:
        lines = [
            "# Managed by scripts/ops/backlog_pcore_accelerator.py",
            f"# updated_at_utc={payload.get('timestamp_utc')}",
            *_env_lines(payload),
            "",
        ]
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text("\n".join(lines), encoding="utf-8")
        applied = True
    return {"out_path": str(out_path), "override_path": str(override_path), "applied": applied}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    governor = load_json(health / "autonomic_resource_governor_latest.json")
    storage_payload = load_json(health / "ingestion_storage_control_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    writer_intel = load_json(health / "writer_process_intelligence_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    memory = load_json(health / "memory_pressure_intelligence_latest.json")
    drainer = load_json(health / "backpressure_drainer_fleet_latest.json")
    host_lanes = _host_lane_contract(governor, memory)
    storage = _storage_metrics(storage_payload, governor)
    topology = _process_topology(writer_intel)
    writer_is_active = _writer_active(writer, writer_intel)
    lanes = _accelerator_lanes(storage, host_lanes)
    wave = _wave_policy(storage, host_lanes, runtime)
    decision = _decision(storage, writer, writer_intel, host_lanes, topology)
    grade = _grade(storage, host_lanes, writer_is_active, topology, memory)
    needs = _needs(storage, decision, host_lanes, memory)
    overall = "ready" if grade["score"] >= 90 and not needs else "advisory"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall == "ready",
        "overall_status": overall,
        "mode": "backlog_pcore_accelerator",
        "input_contracts": {
            "autonomic_resource_governor": _status(governor),
            "ingestion_storage_control": _status(storage_payload),
            "writer_cycle_coordinator": _status(writer),
            "writer_process_intelligence": _status(writer_intel),
            "runtime_throttle_control": _status(runtime),
            "memory_pressure_intelligence": _status(memory),
            "backpressure_drainer_fleet": _status(drainer),
        },
        "host_lane_contract": host_lanes,
        "storage_contract": storage,
        "process_topology": topology,
        "accelerator_lanes": lanes,
        "wave_policy": wave,
        "decision_packet": decision,
        "bulletproof_score": grade,
        "what_do_you_need": {
            "status": "needs_action" if needs else "clear",
            "items": needs,
            "next_command": decision.get("next_command", []),
        },
        "integration_contract": {
            "single_sqlite_writer_only": True,
            "p_core_accelerators_preprocess_only": True,
            "sqlite_write_parallelism": 1,
            "uses_autonomic_resource_governor": True,
            "uses_memory_pressure_intelligence": True,
            "uses_writer_process_intelligence": True,
            "uses_ingestion_storage_control": True,
            "p_cores_are_primary": True,
            "e_cores_are_spillover_only": True,
            "never_touch_protected_volumes": ["/Volumes/VIDEO"],
            "policy": "accelerate_discovery_priority_and_batch_preparation_not_parallel_sqlite_writes",
        },
        "recommended_actions": ordered_unique(
            [
                "let active writer cycles finish before launching another writer",
                "use P-core workers for stale-source locating, density sampling, shard priority, and catch-up scheduling",
                "keep SQLite writes at one exclusive writer even when accelerators widen",
                "hold training and optional collectors until age, memory, and runtime gates clear",
                "close or reduce high-overhead observers if observer_overhead is active",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate P-core backlog accelerators around the single SQLite writer.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    result = write_outputs(payload, out_path=Path(args.out), override_path=Path(args.override), apply=args.apply)
    payload["write_result"] = result
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        decision = _as_dict(payload.get("decision_packet"))
        score = _as_dict(payload.get("bulletproof_score"))
        print(
            "backlog_pcore_accelerator "
            f"status={payload['overall_status']} "
            f"action={decision.get('action')} "
            f"grade={score.get('letter')} "
            f"score={score.get('score')} "
            f"applied={result['applied']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
