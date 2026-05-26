#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from scripts.ops import sql_link_shard_manager as shard_manager
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from . import sql_link_shard_manager as shard_manager


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "writer_process_intelligence_latest.json"
DEFAULT_CONTEXT_PATH = PROJECT_ROOT / "governance" / "health" / "writer_process_context_latest.json"

HOT_LANES = {
    "health_fast",
    "trading_fast",
    "crypto_trading_fast",
    "aggressive_trading",
    "trading",
    "crypto_trading",
    "runtime",
    "crypto_runtime",
    "crypto_api_ingress",
    "writer_progress",
}
WARM_LANES = {
    "governance",
    "crypto_governance",
    "risk_support",
    "support_watchdog",
    "schema_violations",
    "predictive_stability",
    "self_healing",
    "hot_path_storage",
}
COLD_LANES = {
    "data",
    "explanations",
    "crypto_explanations",
    "shadow_attribution",
    "crypto_shadow_attribution",
    "collector_utility",
    "admission_evidence",
    "reports",
}
UNOWNED_PROGRESS_GRACE_MINUTES = 2.0


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


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    explicit = payload.get("overall_status")
    if not isinstance(explicit, str):
        explicit = payload.get("status") if isinstance(payload.get("status"), str) else ""
    text = str(explicit or "").strip().lower()
    if text:
        return text
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


def _nested(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _age_minutes(payload: dict[str, Any], path: Path | None = None) -> float | None:
    try:
        return payload_age_minutes(payload, path)
    except Exception:
        return None


def _lock_snapshot(lock_path: Path) -> dict[str, Any]:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            handle.seek(0)
            owner = handle.read().strip()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"held": True, "owner": owner}
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            return {"held": False, "owner": owner}
    except Exception:
        return {"held": False, "owner": ""}


def _writer_progress_payload(writer_cycle: dict[str, Any], progress: dict[str, Any], lock_state: dict[str, Any]) -> dict[str, Any]:
    before = _nested(writer_cycle, "writer_state_before")
    after_wait = _nested(writer_cycle, "writer_state_after_wait")
    after_remediation = _nested(writer_cycle, "writer_state_after_remediation")
    current = after_wait or after_remediation or before or progress
    progress_age_candidates = [
        _safe_float(current.get("progress_age_minutes"), -1.0),
        _age_minutes(progress),
    ]
    progress_age = max([value for value in progress_age_candidates if value is not None and float(value) >= 0.0] or [0.0])
    if after_wait:
        progress_age = _safe_float(after_wait.get("progress_age_minutes"), progress_age)
    cycle_age = _safe_float(current.get("cycle_age_minutes"), 0.0)
    merged_rows = max(_safe_int(current.get("merged_rows_this_cycle"), 0), _safe_int(progress.get("merged_rows_this_cycle"), 0))
    completed_merges = max(_safe_int(current.get("completed_merge_count"), 0), _safe_int(progress.get("completed_merge_count"), 0))
    completed_shards = max(_safe_int(current.get("completed_shard_count"), 0), _safe_int(progress.get("completed_shard_count"), 0))
    planned_shards = max(_safe_int(current.get("planned_shard_count"), 0), _safe_int(progress.get("planned_shard_count"), 0))
    pending_shards = max(_safe_int(current.get("pending_shard_count"), 0), _safe_int(progress.get("pending_shard_count"), 0))
    timed_out_shards = max(_safe_int(current.get("timed_out_shard_count"), 0), _safe_int(progress.get("timed_out_shard_count"), 0))
    shard_link_plan = progress.get("shard_link_plan") if isinstance(progress.get("shard_link_plan"), dict) else {}
    lock_held = bool(lock_state.get("held", False) or current.get("writer_lock_held", False))
    current_step = str(current.get("effective_current_step") or current.get("current_step") or progress.get("current_step") or "")
    lock_owner = str(lock_state.get("owner") or "") or str(current.get("writer_lock_owner") or progress.get("writer_lock_owner") or "")
    progress_recent = progress_age <= 30.0
    running = bool(current.get("running", False) or progress.get("running", False) or str(progress.get("status") or "") == "running")
    grace_minutes = _safe_float(current.get("unowned_progress_grace_minutes"), UNOWNED_PROGRESS_GRACE_MINUTES)
    owner_pid_live_raw = current.get("writer_owner_pid_live")
    owner_pid_live_known = isinstance(owner_pid_live_raw, bool)
    owner_pid_live = bool(owner_pid_live_raw) if owner_pid_live_known else False
    owner_confirmed_dead = bool(owner_pid_live_known and not owner_pid_live)
    unowned_running_progress = bool(
        running
        and not lock_held
        and (
            not lock_owner
            or owner_confirmed_dead
        )
    )
    progress_orphaned = bool(
        current.get("progress_orphaned", False)
        or (unowned_running_progress and progress_age > max(float(grace_minutes), 0.0))
    )
    active = bool(lock_held or current.get("active", False) or (running and progress_recent and not progress_orphaned))
    if progress_orphaned:
        active = False

    child_writer_active = bool(current.get("child_writer_active", False))
    service_idle_holding_lock = bool(
        lock_held
        and not running
        and not child_writer_active
        and current_step == "complete"
        and str(current.get("active_source") or "") == "writer_lock"
    )

    if progress_orphaned:
        state = "orphaned_progress"
    elif service_idle_holding_lock:
        state = "service_idle_holding_lock"
    elif not active:
        state = "idle"
    elif progress_age >= 90.0 and merged_rows <= 0 and completed_merges <= 0:
        state = "stalled"
    elif progress_age >= 45.0:
        state = "stale_progress"
    else:
        state = "active_progressing"

    coordinator_summary = _nested(writer_cycle, "summary")
    return {
        "state": state,
        "active": bool(active),
        "current_step": current_step,
        "progress_age_minutes": round(float(progress_age), 3),
        "cycle_age_minutes": round(float(cycle_age), 3),
        "merged_rows_this_cycle": int(merged_rows),
        "completed_merge_count": int(completed_merges),
        "completed_shard_count": int(completed_shards),
        "planned_shard_count": int(planned_shards),
        "pending_shard_count": int(pending_shards),
        "timed_out_shard_count": int(timed_out_shards),
        "pending_shards": list(progress.get("pending_shards") or [])[:16],
        "timed_out_shards": list(progress.get("timed_out_shards") or [])[:16],
        "shard_link_plan_policy": str(shard_link_plan.get("policy") or ""),
        "shard_link_plan_order": list(shard_link_plan.get("planned_order") or [])[:32],
        "writer_lock_owner": lock_owner,
        "writer_lock_held": bool(lock_held),
        "progress_orphaned": bool(progress_orphaned),
        "child_writer_active": child_writer_active,
        "active_child_writer_count": _safe_int(current.get("active_child_writer_count"), 0),
        "active_child_writer_pids": list(current.get("active_child_writer_pids") or [])[:12],
        "active_source": str(current.get("active_source") or ("writer_lock" if lock_held else "recent_progress" if active else "idle")),
        "writer_progress_observed_by_coordinator": bool(coordinator_summary.get("writer_progress_observed", False)),
        "stale_writer_detected_by_coordinator": bool(coordinator_summary.get("stale_writer_detected", False)),
        "stale_writer_restart_attempted": bool(coordinator_summary.get("stale_writer_restart_attempted", False)),
    }


def _writer_running_count(process_watchdog: dict[str, Any]) -> tuple[int, int]:
    rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    normalized = 0
    raw = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name != "sql_link_writer":
            continue
        has_raw_count = "raw_running" in row
        row_raw = _safe_int(row.get("raw_running"), _safe_int(row.get("running"), 0))
        row_normalized = _safe_int(row.get("running"), 1 if bool(row.get("process_live", False)) else 0)
        if (
            not has_raw_count
            and row_normalized == 2
            and bool(row.get("heartbeat_ok", False))
            and bool(row.get("process_live", False))
        ):
            row_normalized = 1
        raw = max(raw, row_raw)
        normalized = max(normalized, row_normalized)
    return normalized, raw or normalized


def _process_watchdog_status(process_watchdog: dict[str, Any]) -> str:
    status = _status(process_watchdog)
    if status != "missing":
        return status
    rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    alerts = process_watchdog.get("alerts") if isinstance(process_watchdog.get("alerts"), list) else []
    if not rows:
        return "missing"
    return "degraded" if alerts else "ready"


def _process_topology(process_watchdog: dict[str, Any], process_fanout: dict[str, Any]) -> dict[str, Any]:
    normalized_writers, raw_writers = _writer_running_count(process_watchdog)
    fanout_summary = process_fanout.get("summary") if isinstance(process_fanout.get("summary"), dict) else {}
    fanout_status = _status(process_fanout)
    return {
        "process_watchdog_status": _process_watchdog_status(process_watchdog),
        "process_fanout_status": fanout_status,
        "sql_link_writer_running_count": int(normalized_writers),
        "raw_sql_link_writer_running_count": int(raw_writers),
        "duplicate_sql_writer_processes": bool(raw_writers > 1 and normalized_writers > 1),
        "fanout_guard_active": bool(
            fanout_status in {"active", "degraded"}
            or process_fanout.get("triggered", False)
            or fanout_summary.get("triggered", False)
        ),
        "fanout_targetable_count": _safe_int(
            process_fanout.get("targetable_process_count"),
            _safe_int(fanout_summary.get("targetable_process_count"), 0),
        ),
        "fanout_total_rss_mb": _safe_float(
            process_fanout.get("total_rss_mb"),
            _safe_float(fanout_summary.get("total_rss_mb"), 0.0),
        ),
    }


def _lane_family(name: str) -> str:
    text = str(name or "").lower()
    if text in {"health_fast", "writer_progress"}:
        return "writer_health"
    if "trading" in text or text in {"aggressive_trading"}:
        return "trading_hot_path"
    if "runtime" in text or "schema" in text:
        return "runtime_contracts"
    if text in {"predictive_stability", "self_healing", "hot_path_storage"}:
        return text
    if text in {"collector_utility", "admission_evidence", "reports"}:
        return text
    if "explanation" in text or "attribution" in text:
        return "cold_reporting"
    if text == "data":
        return "data_foundation"
    if "governance" in text or "support" in text:
        return "governance_support"
    return "other"


def _lane_tier(name: str) -> str:
    if name in HOT_LANES:
        return "hot"
    if name in WARM_LANES:
        return "warm"
    if name in COLD_LANES:
        return "cold"
    return "warm"


def _writer_lane_profiles() -> list[dict[str, Any]]:
    names = [name.strip() for name in str(shard_manager.CURRENT_DEFAULT_SHARDS).split(",") if name.strip()]
    profiles: list[dict[str, Any]] = []
    for order, name in enumerate(names, start=1):
        defaults = shard_manager.DEFAULT_SHARD_DEFS.get(name, {})
        merge_rows = max(
            _safe_int(defaults.get("merge_max_jsonl_rows"), 0),
            _safe_int(defaults.get("merge_max_json_file_rows"), 0),
        )
        profiles.append(
            {
                "name": name,
                "order": order,
                "family": _lane_family(name),
                "tier": _lane_tier(name),
                "merge_to_primary": bool(defaults.get("merge_to_primary", True)),
                "merge_priority": str(defaults.get("merge_priority", "normal") or "normal"),
                "json_files_enabled": not bool(defaults.get("skip_json_files", True)),
                "max_files": _safe_int(defaults.get("max_files"), 0),
                "max_lines_per_file": _safe_int(defaults.get("max_lines_per_file"), 0),
                "state_checkpoint_lines": _safe_int(defaults.get("state_checkpoint_lines"), 10_000),
                "merge_row_budget": int(merge_rows),
            }
        )
    return profiles


def _lane_family_summary(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in profiles:
        family = str(row.get("family") or "other")
        bucket = buckets.setdefault(
            family,
            {
                "family": family,
                "lane_count": 0,
                "hot_count": 0,
                "warm_count": 0,
                "cold_count": 0,
                "primary_merge_count": 0,
                "json_file_lane_count": 0,
            },
        )
        bucket["lane_count"] = _safe_int(bucket.get("lane_count"), 0) + 1
        tier = str(row.get("tier") or "")
        if tier == "hot":
            bucket["hot_count"] = _safe_int(bucket.get("hot_count"), 0) + 1
        elif tier == "cold":
            bucket["cold_count"] = _safe_int(bucket.get("cold_count"), 0) + 1
        else:
            bucket["warm_count"] = _safe_int(bucket.get("warm_count"), 0) + 1
        if bool(row.get("merge_to_primary", False)):
            bucket["primary_merge_count"] = _safe_int(bucket.get("primary_merge_count"), 0) + 1
        if bool(row.get("json_files_enabled", False)):
            bucket["json_file_lane_count"] = _safe_int(bucket.get("json_file_lane_count"), 0) + 1
    return sorted(buckets.values(), key=lambda row: (_safe_int(row.get("hot_count"), 0), _safe_int(row.get("lane_count"), 0)), reverse=True)


def _source_pressure(storage: dict[str, Any], runtime: dict[str, Any], memory_efficiency: dict[str, Any]) -> dict[str, Any]:
    memory_snapshot = memory_efficiency.get("memory_snapshot") if isinstance(memory_efficiency.get("memory_snapshot"), dict) else {}
    storage_pending = max(
        _safe_int(storage.get("total_pending_lines"), 0),
        _safe_int(storage.get("pending_lines_total"), 0),
        _safe_int(_nested(storage, "backpressure").get("total_pending_lines"), 0),
    )
    return {
        "storage_status": _status(storage),
        "storage_severity": str(storage.get("severity") or _nested(storage, "backpressure").get("severity") or ""),
        "pending_lines": int(storage_pending),
        "runtime_status": _status(runtime),
        "runtime_pressure_high": bool(
            _status(runtime) in {"blocked", "critical", "degraded"}
            or _safe_float(runtime.get("host_saturation_score"), 0.0) >= 80.0
            or str(runtime.get("memory_pressure_level") or "").lower() in {"high", "critical"}
        ),
        "memory_status": _status(memory_efficiency),
        "memory_pressure_high": bool(
            _status(memory_efficiency) in {"blocked", "critical", "degraded"}
            or str(memory_snapshot.get("memory_pressure_kind") or "").lower() not in {"", "none", "green", "normal"}
            or _safe_float(memory_snapshot.get("compressed_store_gb"), 0.0) >= 12.0
        ),
    }


def _risk_flags(
    *,
    writer_health: dict[str, Any],
    process_topology: dict[str, Any],
    pressure: dict[str, Any],
    drainer_intelligence: dict[str, Any],
    progress: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    state = str(writer_health.get("state") or "")
    if state == "service_idle_holding_lock":
        risks.append("writer_service_idle_lock")
    elif bool(writer_health.get("active", False)):
        risks.append("writer_active")
    if state == "orphaned_progress":
        risks.append("writer_progress_orphaned")
    if state == "stale_progress":
        risks.append("writer_progress_stale")
    if state == "stalled":
        risks.append("writer_progress_stalled")
    if _safe_int(writer_health.get("timed_out_shard_count"), 0) > 0:
        risks.append("shard_link_timeouts")
    if not progress:
        risks.append("missing_writer_progress_artifact")
    if bool(process_topology.get("duplicate_sql_writer_processes", False)):
        risks.append("duplicate_sql_writer_processes")
    if bool(process_topology.get("fanout_guard_active", False)) and (
        _safe_int(process_topology.get("fanout_targetable_count"), 0) > 0
        or _safe_float(process_topology.get("fanout_total_rss_mb"), 0.0) > 0.0
    ):
        risks.append("process_fanout_pressure")
    elif bool(process_topology.get("fanout_guard_active", False)):
        risks.append("process_fanout_hold")
    if str(pressure.get("storage_status") or "") in {"blocked", "critical"} or str(pressure.get("storage_severity") or "") == "critical":
        risks.append("storage_critical")
    if bool(pressure.get("runtime_pressure_high", False)):
        risks.append("runtime_pressure_high")
    if bool(pressure.get("memory_pressure_high", False)):
        risks.append("memory_pressure_high")
    decision = drainer_intelligence.get("decision_packet") if isinstance(drainer_intelligence.get("decision_packet"), dict) else {}
    if str(decision.get("action") or "") in {"verify_writer_progress_then_re_score", "run_writer_recovery_check_then_re_score"}:
        risks.append("drainer_waiting_on_writer_recovery")
    return ordered_unique(risks)


def _decision_action(writer_health: dict[str, Any], process_topology: dict[str, Any], pressure: dict[str, Any], risks: list[str]) -> str:
    if "duplicate_sql_writer_processes" in risks:
        return "enforce_single_writer_process_then_re_score"
    if "writer_progress_stalled" in risks:
        return "recover_stalled_writer_then_re_score"
    if "writer_progress_stale" in risks:
        return "verify_writer_progress_then_re_score"
    if "process_fanout_pressure" in risks:
        return "trim_fanout_before_writer_expansion"
    if "writer_service_idle_lock" in risks:
        if _safe_int(pressure.get("pending_lines"), 0) > 0:
            return "request_writer_service_handoff_then_re_score"
        return "observe_writer_service_idle_lock"
    if bool(writer_health.get("active", False)):
        return "wait_for_active_writer_progress"
    if _safe_int(pressure.get("pending_lines"), 0) > 0:
        return "run_focused_writer_cycle"
    return "park_writer_and_observe"


def _playbook(action: str) -> list[dict[str, Any]]:
    if action == "enforce_single_writer_process_then_re_score":
        return [
            {"step": "inspect_processes", "command": ["./scripts/ops/opsctl.sh", "process-fanout-guard", "--json"]},
            {"step": "refresh_writer", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    if action in {"recover_stalled_writer_then_re_score", "verify_writer_progress_then_re_score"}:
        return [
            {"step": "writer_coordinator_recovery", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--skip-maintenance", "--json"]},
            {"step": "refresh_storage", "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]},
            {"step": "refresh_writer_intelligence", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    if action == "trim_fanout_before_writer_expansion":
        return [
            {"step": "trim_process_fanout", "command": ["./scripts/ops/opsctl.sh", "process-fanout-guard", "--apply", "--json"]},
            {"step": "re_score_writer", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    if action == "run_focused_writer_cycle":
        return [
            {"step": "request_drainer_handoff", "command": ["./scripts/ops/opsctl.sh", "backpressure-drainer-fleet", "--apply", "--ttl-seconds", "900", "--json"]},
            {"step": "run_single_writer", "command": ["./scripts/ops/opsctl.sh", "sql-sync", "--json"]},
            {"step": "re_score_writer", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    if action == "request_writer_service_handoff_then_re_score":
        return [
            {"step": "request_drainer_handoff", "command": ["./scripts/ops/opsctl.sh", "backpressure-drainer-fleet", "--apply", "--ttl-seconds", "900", "--json"]},
            {"step": "observe_writer_service", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]},
            {"step": "re_score_writer", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    if action == "observe_writer_service_idle_lock":
        return [
            {"step": "observe_writer_service", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]},
            {"step": "re_score_writer", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    if action == "wait_for_active_writer_progress":
        return [
            {"step": "observe_writer", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]},
            {"step": "re_score_writer", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
        ]
    return [{"step": "observe", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]}]


def _confidence(risks: list[str], profiles: list[dict[str, Any]], writer_health: dict[str, Any]) -> float:
    score = 0.52
    if profiles:
        score += 0.16
    if str(writer_health.get("state") or "") in {"idle", "active_progressing", "service_idle_holding_lock"}:
        score += 0.1
    if "writer_progress_stale" in risks:
        score -= 0.08
    if "writer_progress_stalled" in risks:
        score -= 0.18
    if "shard_link_timeouts" in risks:
        score -= 0.04
    if "duplicate_sql_writer_processes" in risks:
        score -= 0.18
    if "process_fanout_pressure" in risks:
        score -= 0.08
    if "memory_pressure_high" in risks or "runtime_pressure_high" in risks:
        score -= 0.06
    return round(max(0.1, min(0.95, score)), 3)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    writer_cycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    health = Path(project_root) / "governance" / "health"
    progress_path = health / "sql_link_service_progress_latest.json"
    writer_cycle_payload = writer_cycle if isinstance(writer_cycle, dict) else load_json(health / "writer_cycle_coordinator_latest.json")
    progress = load_json(progress_path)
    lock_state = _lock_snapshot(Path(project_root) / "governance" / "locks" / "jsonl_sql_writer.lock")
    process_watchdog = load_json(health / "process_watchdog_latest.json")
    process_fanout = load_json(health / "process_fanout_guard_latest.json")
    drainer_intelligence = load_json(health / "drainer_intelligence_layer_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    memory_efficiency = load_json(health / "memory_efficiency_control_latest.json")

    writer_health = _writer_progress_payload(writer_cycle_payload, progress, lock_state)
    process_topology = _process_topology(process_watchdog, process_fanout)
    pressure = _source_pressure(storage, runtime, memory_efficiency)
    profiles = _writer_lane_profiles()
    family_summary = _lane_family_summary(profiles)
    risks = _risk_flags(
        writer_health=writer_health,
        process_topology=process_topology,
        pressure=pressure,
        drainer_intelligence=drainer_intelligence,
        progress=progress,
    )
    action = _decision_action(writer_health, process_topology, pressure, risks)
    confidence = _confidence(risks, profiles, writer_health)
    writer_recovery_required = action in {"recover_stalled_writer_then_re_score", "verify_writer_progress_then_re_score"}
    pressure_guarded = bool("memory_pressure_high" in risks or "runtime_pressure_high" in risks)
    status = "ready"
    if "duplicate_sql_writer_processes" in risks or "writer_progress_stalled" in risks:
        status = "degraded"
    elif writer_recovery_required or pressure_guarded:
        status = "advisory"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "mode": "writer_process_intelligence",
        "decision_packet": {
            "action": action,
            "confidence": confidence,
            "writer_state": str(writer_health.get("state") or ""),
            "pending_lines": _safe_int(pressure.get("pending_lines"), 0),
            "expanded_writer_lane_count": len(profiles),
            "hot_lane_count": sum(1 for row in profiles if row.get("tier") == "hot"),
            "warm_lane_count": sum(1 for row in profiles if row.get("tier") == "warm"),
            "cold_lane_count": sum(1 for row in profiles if row.get("tier") == "cold"),
            "risk_flags": risks,
            "reason_codes": ordered_unique(
                [
                    "single_writer_guard",
                    "writer_recovery_required" if writer_recovery_required else "",
                    "process_fanout_guard" if "process_fanout_pressure" in risks else "",
                    "storage_critical" if "storage_critical" in risks else "",
                    "pressure_guarded" if pressure_guarded else "",
                    "expanded_shard_lanes",
                ]
            ),
        },
        "writer_health": writer_health,
        "process_topology": process_topology,
        "source_pressure": pressure,
        "writer_lane_profiles": profiles,
        "lane_family_summary": family_summary,
        "process_playbook": _playbook(action),
        "safety_envelope": {
            "single_writer_only": True,
            "starts_parallel_sql_writers": False,
            "max_parallel_sql_writers": 1,
            "max_writer_cycles_now": 0 if bool(writer_health.get("active", False)) else 1,
            "writer_recovery_required": bool(writer_recovery_required),
            "shard_expansion_allowed": True,
            "process_trim_before_expansion": bool("process_fanout_pressure" in risks),
            "protected_process_markers": [
                "scripts/ops/sql_link_shard_manager.py",
                "scripts/ops/sql_link_writer_service.py",
                "scripts/ops/run_sql_link_writer_launchd.sh",
            ],
        },
        "writer_expansion_contract": {
            "expands_by": "more_shard_lanes_and_smarter_sequencing_not_parallel_sql_writers",
            "new_lane_families": [
                "writer_health",
                "predictive_stability",
                "self_healing",
                "collector_utility",
                "hot_path_storage",
                "admission_evidence",
                "reports",
            ],
            "single_writer_owner": "jsonl_sql_writer.lock",
            "handoff_sources": ["backpressure_drainer_fleet", "backpressure_super_drainer", "writer_cycle_coordinator"],
            "feeds": ["system_self_model", "drainer_intelligence_layer", "process_fanout_guard", "process_watchdog"],
        },
        "source_status": {
            "writer_cycle_coordinator": _status(writer_cycle_payload),
            "sql_link_service_progress": _status(progress),
            "process_watchdog": _process_watchdog_status(process_watchdog),
            "process_fanout_guard": _status(process_fanout),
            "drainer_intelligence_layer": _status(drainer_intelligence),
            "ingestion_storage_control": _status(storage),
            "runtime_throttle_control": _status(runtime),
            "memory_efficiency_control": _status(memory_efficiency),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the advisory intelligence layer for SQL writer processes and writer shard lanes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--context-file", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    out_file = Path(args.out_file).expanduser()
    if not out_file.is_absolute():
        out_file = project_root / out_file
    write_payload(out_file, payload)
    if args.apply:
        context_file = Path(args.context_file).expanduser()
        if not context_file.is_absolute():
            context_file = project_root / context_file
        write_payload(context_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        decision = payload.get("decision_packet") if isinstance(payload.get("decision_packet"), dict) else {}
        print(
            "writer_process_intelligence "
            f"status={payload.get('overall_status', '')} "
            f"action={decision.get('action', '')} "
            f"writer_state={decision.get('writer_state', '')}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
