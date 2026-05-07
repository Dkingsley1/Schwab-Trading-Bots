#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import eastern_off_hours_window, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, eastern_off_hours_window, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_drainer_fleet_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "backpressure_drainer_fleet.lock"
SERVICE_REQUEST_PATH = PROJECT_ROOT / "governance" / "health" / "sql_link_service_request_latest.json"
WRITER_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"
MIN_MATERIAL_PENDING_LINES = 100
MICRO_STALE_READY_AGE_SECONDS = 1_800.0
CORE_HARD_PENDING_LINES = 50_000
DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS = 240.0

DRAINER_OWNERS: dict[str, dict[str, Any]] = {
    "core_decision_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v187_data_collection_storage_budget_guard",
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        ],
        "pressure_lane": "core_decision_backpressure",
        "ops_infrabots": ["backpressure_slo_bot", "storage_backpressure_autopilot"],
    },
    "cold_stage_drainer": {
        "owner_bot_id": "brain_refinery_v187_data_collection_storage_budget_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "explanation_deferred_backpressure",
        "ops_infrabots": ["backpressure_drainer_fleet", "storage_pressure_clearance_bot"],
    },
    "governance_execution_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        ],
        "pressure_lane": "governance_execution_backpressure",
        "ops_infrabots": ["backpressure_drainer_fleet", "storage_backpressure_autopilot"],
    },
    "api_ingress_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        ],
        "pressure_lane": "api_ingress_backpressure",
        "ops_infrabots": ["backpressure_drainer_fleet", "data_plane_recovery_controller"],
    },
    "runtime_channel_drainer": {
        "owner_bot_id": "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "runtime_channel_backpressure",
        "ops_infrabots": ["runtime_throttle_control", "backpressure_drainer_fleet"],
    },
    "schema_violation_drainer": {
        "owner_bot_id": "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        "backup_bot_ids": [
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        ],
        "pressure_lane": "schema_violation_backpressure",
        "ops_infrabots": ["backpressure_drainer_fleet", "channel_schema_contract_guard"],
    },
    "support_watchdog_drainer": {
        "owner_bot_id": "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "support_watchdog_backpressure",
        "ops_infrabots": ["backpressure_slo_bot", "storage_pressure_clearance_bot"],
    },
    "fast_trade_bridge_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "paper_trade_bridge_backpressure",
        "ops_infrabots": ["paper_trade_lock_infrabot", "backpressure_drainer_fleet"],
    },
    "attribution_drainer": {
        "owner_bot_id": "brain_refinery_v187_data_collection_storage_budget_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "shadow_attribution_backpressure",
        "ops_infrabots": ["backpressure_drainer_fleet", "retention_debt_sheriff"],
    },
    "derivatives_surface_drainer": {
        "owner_bot_id": "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "derivatives_options_futures_backpressure",
        "ops_infrabots": ["options_risk_intelligence_v2", "backpressure_drainer_fleet"],
    },
    "market_data_provider_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "provider_market_data_tail_backpressure",
        "ops_infrabots": ["provider_adapter_verification", "data_source_confidence_engine"],
    },
    "macro_event_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "macro_event_backpressure",
        "ops_infrabots": ["macro_bulletin", "event_intelligence"],
    },
    "model_research_drainer": {
        "owner_bot_id": "brain_refinery_v187_data_collection_storage_budget_guard",
        "backup_bot_ids": [
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        ],
        "pressure_lane": "model_retrain_research_backpressure",
        "ops_infrabots": ["training_runtime_control", "quant_model_control"],
    },
    "feature_event_store_drainer": {
        "owner_bot_id": "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        "backup_bot_ids": [
            "brain_refinery_v187_data_collection_storage_budget_guard",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "feature_event_store_backpressure",
        "ops_infrabots": ["feature_store", "point_in_time_event_store"],
    },
    "report_cockpit_drainer": {
        "owner_bot_id": "brain_refinery_v187_data_collection_storage_budget_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "report_cockpit_backpressure",
        "ops_infrabots": ["reporter_infrabot", "operator_cockpit"],
    },
    "settlement_reconciliation_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        ],
        "pressure_lane": "settlement_reconciliation_backpressure",
        "ops_infrabots": ["portfolio_ledger_reconciliation", "broker_adapter_mesh", "paper_trade_lock_infrabot"],
    },
    "alert_notification_drainer": {
        "owner_bot_id": "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "alert_notification_backpressure",
        "ops_infrabots": ["remote_alert_control", "incident_timeline", "mac_notification_watch"],
    },
    "memory_runtime_artifact_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "memory_runtime_artifact_backpressure",
        "ops_infrabots": ["runtime_throttle_control", "memory_efficiency_control", "process_fanout_guard"],
    },
    "data_quality_contract_drainer": {
        "owner_bot_id": "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        "backup_bot_ids": [
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "data_quality_contract_backpressure",
        "ops_infrabots": ["data_quality_observatory", "provider_adapter_verification", "collector_contracts"],
    },
    "predictive_stability_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        ],
        "pressure_lane": "predictive_stability_backpressure",
        "ops_infrabots": ["predictive_stability", "runtime_throttle_control", "memory_efficiency_control"],
    },
    "self_healing_recovery_drainer": {
        "owner_bot_id": "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "self_healing_recovery_backpressure",
        "ops_infrabots": ["self_healing_router", "blackstart_recovery", "incident_timeline"],
    },
    "collector_utility_budget_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v187_data_collection_storage_budget_guard",
            "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        ],
        "pressure_lane": "collector_utility_budget_backpressure",
        "ops_infrabots": ["collector_utility_budget", "data_collection_observation_rollup", "collector_contracts"],
    },
    "hot_path_storage_budget_drainer": {
        "owner_bot_id": "brain_refinery_v187_data_collection_storage_budget_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        ],
        "pressure_lane": "hot_path_storage_budget_backpressure",
        "ops_infrabots": ["hot_path_storage_budget", "ingestion_storage_governor", "storage_tier_policy"],
    },
    "admission_evidence_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "admission_evidence_backpressure",
        "ops_infrabots": ["new_bot_admission_guard", "feature_store", "replay_hash_registry"],
    },
    "writer_progress_recovery_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "writer_progress_recovery_backpressure",
        "ops_infrabots": ["writer_cycle_coordinator", "backpressure_slo_bot", "storage_backpressure_autopilot"],
    },
    "training_lineage_drainer": {
        "owner_bot_id": "brain_refinery_v271_training_eligibility_graduation_gatekeeper",
        "backup_bot_ids": [
            "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        ],
        "pressure_lane": "training_lineage_backpressure",
        "ops_infrabots": ["training_lineage_manifest", "training_labeling_intelligence", "training_runtime_control"],
    },
    "label_contract_drainer": {
        "owner_bot_id": "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        "backup_bot_ids": [
            "brain_refinery_v271_training_eligibility_graduation_gatekeeper",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "label_contract_backpressure",
        "ops_infrabots": ["training_label_audit", "training_labeling_intelligence", "feature_store"],
    },
    "collector_telemetry_rollup_drainer": {
        "owner_bot_id": "brain_refinery_v273_data_collection_sampling_optimizer",
        "backup_bot_ids": [
            "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "collector_telemetry_rollup_backpressure",
        "ops_infrabots": ["data_collection_observation_rollup", "collector_contracts", "data_collection_storage_guard"],
    },
    "storage_route_reconcile_drainer": {
        "owner_bot_id": "brain_refinery_v187_data_collection_storage_budget_guard",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        ],
        "pressure_lane": "storage_route_reconcile_backpressure",
        "ops_infrabots": ["storage_resilience", "storage_transition_coordinator", "split_brain_reconcile"],
    },
    "ingestion_priority_drainer": {
        "owner_bot_id": "brain_refinery_v601_system_governor_collector_priority_ranker_bot",
        "backup_bot_ids": [
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
            "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        ],
        "pressure_lane": "ingestion_priority_backpressure",
        "ops_infrabots": ["ingestion_priority_queue", "ingestion_storage_governor", "backlog_quarantine"],
    },
}


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


def _acquire_nonblocking_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate(0)
    handle.write(f"pid={os.getpid()} timestamp_utc={iso_now()}\n")
    handle.flush()
    return handle


def _writer_owner(lock_path: Path = WRITER_LOCK_PATH) -> str:
    return str(_writer_lock_snapshot(lock_path).get("owner") or "")


def _writer_lock_snapshot(lock_path: Path = WRITER_LOCK_PATH) -> dict[str, Any]:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            handle.seek(0)
            owner = handle.read().strip()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"owner": owner, "held": True}
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            return {"owner": owner, "held": False}
    except Exception:
        return {"owner": "", "held": False}


def _rows_from(backpressure: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = backpressure.get(key)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _collect_sources(backpressure: dict[str, Any], prefixes: tuple[str, ...], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for key in keys:
        for row in _rows_from(backpressure, key):
            source_rel = str(row.get("source_rel") or "").strip()
            if not source_rel or not source_rel.startswith(prefixes):
                continue
            pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
            if pending_lines <= 0:
                continue
            age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
            current = by_source.get(source_rel)
            if current is None:
                by_source[source_rel] = {
                    "source_rel": source_rel,
                    "pending_lines": pending_lines,
                    "oldest_pending_age_seconds": round(age_seconds, 3),
                }
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(
                max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds),
                3,
            )
    return sorted(
        by_source.values(),
        key=lambda row: (_safe_int(row.get("pending_lines"), 0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0)),
        reverse=True,
    )


def _collect_sources_by_contains(
    backpressure: dict[str, Any],
    needles: tuple[str, ...],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    lowered_needles = tuple(str(needle or "").strip().lower() for needle in needles if str(needle or "").strip())
    if not lowered_needles:
        return []
    by_source: dict[str, dict[str, Any]] = {}
    for key in keys:
        for row in _rows_from(backpressure, key):
            source_rel = str(row.get("source_rel") or "").strip()
            if not source_rel:
                continue
            lowered = source_rel.lower()
            if not any(needle in lowered for needle in lowered_needles):
                continue
            pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
            if pending_lines <= 0:
                continue
            age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
            current = by_source.get(source_rel)
            if current is None:
                by_source[source_rel] = {
                    "source_rel": source_rel,
                    "pending_lines": pending_lines,
                    "oldest_pending_age_seconds": round(age_seconds, 3),
                }
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(
                max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds),
                3,
            )
    return sorted(
        by_source.values(),
        key=lambda row: (_safe_int(row.get("pending_lines"), 0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0)),
        reverse=True,
    )


def _concentration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(max(_safe_int(row.get("pending_lines"), 0), 0) for row in rows)
    top = sorted((max(_safe_int(row.get("pending_lines"), 0), 0) for row in rows), reverse=True)
    top1 = top[0] if top else 0
    top3 = sum(top[:3])
    top1_share = (top1 / float(total)) if total > 0 else 0.0
    top3_share = (top3 / float(total)) if total > 0 else 0.0
    concentrated = bool(total >= 5000 and (top1_share >= 0.45 or top3_share >= 0.75))
    return {
        "total_pending_lines": int(total),
        "top1_pending_lines": int(top1),
        "top3_pending_lines": int(top3),
        "top1_share": round(top1_share, 6),
        "top3_share": round(top3_share, 6),
        "concentrated": concentrated,
    }


def _base_env(*, critical: bool) -> dict[str, str]:
    return {
        "INGEST_MAX_DEFERRED_FILES": "6" if critical else "4",
        "JSONL_SQL_MAX_COLD_LANE_FILES": "2" if critical else "1",
        "LOG_DATA_INGRESS": "0",
        "LOG_API_CALLS": "0",
        "LOG_LOOP_STATE": "0",
        "LOG_SHADOW_PNL_ATTRIBUTION": "0",
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12" if critical else "15",
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "120" if critical else "150",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25" if critical else "45",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000" if critical else "200000",
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": "2400000" if critical else "1800000",
        "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
        "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
        "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.25" if critical else "0.5",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.25" if critical else "0.5",
    }


def _profile_self_accommodation(
    *,
    live_window_safe: bool,
    min_pending_lines: int,
    stale_ready_age_seconds: float,
    env: dict[str, str],
) -> dict[str, Any]:
    critical = str(env.get("SQL_LINK_SERVICE_INTERVAL_SECONDS") or "").strip() == "12"
    return {
        "self_accommodating": True,
        "coordination_model": "single_sql_writer_focused_handoff",
        "allowed_parallel_writers": 1,
        "starts_parallel_sql_writers": False,
        "max_waves_per_score": 1,
        "service_ttl_seconds": 600 if critical else 900,
        "cooldown_seconds": 45 if critical else 90,
        "min_pending_lines": int(max(min_pending_lines, 1)),
        "stale_ready_age_seconds": round(float(stale_ready_age_seconds), 3),
        "live_window_behavior": "live_window_safe" if live_window_safe else "protected_window_or_force_only",
        "backs_off_when": ordered_unique(
            [
                "jsonl_sql_writer_lock_held",
                "market_hours_guard" if not live_window_safe else "",
                "missing_backpressure_artifact",
                "runtime_or_memory_pressure_high",
                "storage_snapshot_stale",
                "progress_stall",
            ]
        ),
        "safe_expansion_rule": "sequence_bounded_handoffs_re_score_after_each_wave_do_not_add_parallel_sql_writers",
    }


def _profile(
    *,
    name: str,
    reason: str,
    rows: list[dict[str, Any]],
    shards: list[str],
    env: dict[str, str],
    priority_boost: int,
    live_window_safe: bool,
    min_pending_lines: int = MIN_MATERIAL_PENDING_LINES,
    stale_ready_age_seconds: float = 0.0,
) -> dict[str, Any]:
    concentration = _concentration_summary(rows)
    pending_lines = _safe_int(concentration.get("total_pending_lines"), 0)
    oldest_age = max([_safe_float(row.get("oldest_pending_age_seconds"), 0.0) for row in rows] or [0.0])
    path_focus = [str(row.get("source_rel") or "").strip() for row in rows[:8] if str(row.get("source_rel") or "").strip()]
    priority_score = int(priority_boost + pending_lines + (oldest_age / 60.0))
    owner_contract = DRAINER_OWNERS.get(name, {})
    material_ready = pending_lines >= max(int(min_pending_lines), 1)
    stale_tail_ready = bool(pending_lines > 0 and stale_ready_age_seconds > 0.0 and oldest_age >= stale_ready_age_seconds)
    readiness_reason = "material_pending" if material_ready else ("stale_tail" if stale_tail_ready else "below_threshold")
    return {
        "name": name,
        "reason": reason,
        "status": "ready" if material_ready or stale_tail_ready else "idle",
        "readiness_reason": readiness_reason,
        "min_pending_lines": int(max(min_pending_lines, 1)),
        "stale_ready_age_seconds": round(float(stale_ready_age_seconds), 3),
        "pending_lines": int(pending_lines),
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "priority_score": priority_score,
        "age_pressure_priority_bonus": 0,
        "effective_priority_score": priority_score,
        "owner_bot_id": str(owner_contract.get("owner_bot_id") or ""),
        "backup_bot_ids": list(owner_contract.get("backup_bot_ids") or []),
        "assigned_pressure_lane": str(owner_contract.get("pressure_lane") or ""),
        "ops_infrabots": list(owner_contract.get("ops_infrabots") or []),
        "shards": shards,
        "path_focus": path_focus,
        "concentration": concentration,
        "live_window_safe": bool(live_window_safe),
        "self_accommodation": _profile_self_accommodation(
            live_window_safe=bool(live_window_safe),
            min_pending_lines=int(max(min_pending_lines, 1)),
            stale_ready_age_seconds=float(stale_ready_age_seconds),
            env=env,
        ),
        "env_overrides": env,
    }


def _apply_age_pressure_priority(profiles: list[dict[str, Any]], backpressure: dict[str, Any]) -> list[dict[str, Any]]:
    threshold_seconds = max(
        _safe_float(backpressure.get("oldest_age_threshold_seconds"), DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS),
        1.0,
    )
    for row in profiles:
        pending_lines = _safe_int(row.get("pending_lines"), 0)
        min_pending_lines = _safe_int(row.get("min_pending_lines"), MIN_MATERIAL_PENDING_LINES)
        oldest_age = _safe_float(row.get("oldest_pending_age_seconds"), 0.0)
        readiness_reason = str(row.get("readiness_reason") or "")
        live_window_safe = bool(row.get("live_window_safe", False))
        eligible = bool(
            live_window_safe
            and oldest_age >= threshold_seconds
            and (pending_lines >= max(min_pending_lines, 1) or readiness_reason == "stale_tail")
        )
        bonus = 0
        if eligible:
            age_ratio = oldest_age / threshold_seconds
            bonus = int(min(max((age_ratio - 1.0) * 20_000.0, 0.0), 90_000.0))
        priority_score = _safe_int(row.get("priority_score"), 0)
        row["age_pressure_priority_bonus"] = bonus
        row["effective_priority_score"] = int(priority_score + bonus)
    return profiles


def _is_crypto_decision_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return any(
        part in rel
        for part in (
            "shadow_crypto/",
            "shadow_crypto_futures_crypto/",
            "default_crypto_coinbase",
            "crypto_futures_crypto_coinbase",
            "default_crypto_schwab",
            "crypto_futures_crypto_schwab",
        )
    )


def _is_aggressive_decision_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return any(
        part in rel
        for part in (
            "shadow_aggressive_",
            "shadow_intraday_aggressive_",
            "shadow_swing_aggressive_",
        )
    )


def _decision_drainer_env(base: dict[str, str], core_rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    concentration = _concentration_summary(core_rows)
    concentrated = bool(concentration.get("concentrated", False))
    regular_focus: list[str] = []
    aggressive_focus: list[str] = []
    crypto_focus: list[str] = []
    for row in core_rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
        elif _is_aggressive_decision_source(source_rel):
            aggressive_focus.append(source_rel)
        else:
            regular_focus.append(source_rel)

    shards: list[str] = []
    env: dict[str, str] = {**base}
    if concentrated:
        env.update(
            {
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1",
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_TOP1_SHARE": str(concentration.get("top1_share", 0.0)),
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_TOP3_SHARE": str(concentration.get("top3_share", 0.0)),
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "60",
                "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS": "32000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS": "24000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS": "32000",
            }
        )
    if regular_focus:
        shards.append("trading")
        env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES"] = "16"
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] = "24000" if concentrated else "64000"
        # Keep the companion aggressive shard in the handoff for mixed equity decision pressure.
        # Some aggressive sleeves write through regular decision-channel paths rather than
        # shadow_aggressive-prefixed files, so this preserves the broader hot-lane sweep.
        shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = "12000" if concentrated else "24000"
    if aggressive_focus:
        if "aggressive_trading" not in shards:
            shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"] = ",".join(aggressive_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = "12000" if concentrated else "24000"
    if crypto_focus:
        shards.append("crypto_trading")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE"] = "64000"

    if not shards:
        shards = ["trading", "aggressive_trading", "crypto_trading"]
    shards = ordered_unique([*shards, "health_fast", "support_watchdog"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _api_ingress_drainer_env(base: dict[str, str], rows: list[dict[str, Any]], *, critical: bool) -> tuple[list[str], dict[str, str]]:
    regular_focus: list[str] = []
    crypto_focus: list[str] = []
    for row in rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
        else:
            regular_focus.append(source_rel)

    shards: list[str] = []
    env: dict[str, str] = {**base}
    if crypto_focus:
        shards.append("crypto_api_ingress")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_API_INGRESS_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_API_INGRESS_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_API_INGRESS_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_API_INGRESS_STATE_CHECKPOINT_LINES"] = "1500"
    if regular_focus:
        shards.append("governance")
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_STATE_CHECKPOINT_LINES"] = "1500"

    if not shards:
        shards = ["governance", "crypto_api_ingress"]
    shards = ordered_unique([*shards, "health_fast"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _fast_trade_bridge_drainer_env(base: dict[str, str], rows: list[dict[str, Any]], *, critical: bool) -> tuple[list[str], dict[str, str]]:
    regular_focus: list[str] = []
    crypto_focus: list[str] = []
    for row in rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
        else:
            regular_focus.append(source_rel)

    shards: list[str] = []
    env: dict[str, str] = {**base}
    if regular_focus:
        shards.append("trading_fast")
        env["SQL_LINK_SERVICE_SHARD_TRADING_FAST_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_TRADING_FAST_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_TRADING_FAST_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
        env["SQL_LINK_SERVICE_SHARD_TRADING_FAST_STATE_CHECKPOINT_LINES"] = "1500"
    if crypto_focus:
        shards.append("crypto_trading_fast")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_FAST_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_FAST_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_FAST_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_FAST_STATE_CHECKPOINT_LINES"] = "1500"

    if not shards:
        shards = ["trading_fast", "crypto_trading_fast"]
    shards = ordered_unique([*shards, "health_fast"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _attribution_drainer_env(base: dict[str, str], rows: list[dict[str, Any]], *, critical: bool) -> tuple[list[str], dict[str, str]]:
    regular_focus: list[str] = []
    crypto_focus: list[str] = []
    for row in rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
        else:
            regular_focus.append(source_rel)

    shards: list[str] = []
    env: dict[str, str] = {**base}
    if regular_focus:
        shards.append("shadow_attribution")
        env["SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES"] = "8" if critical else "5"
        env["SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
        env["SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_STATE_CHECKPOINT_LINES"] = "2000"
    if crypto_focus:
        shards.append("crypto_shadow_attribution")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES"] = "8" if critical else "5"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_STATE_CHECKPOINT_LINES"] = "2000"

    if not shards:
        shards = ["shadow_attribution", "crypto_shadow_attribution"]
    shards = ordered_unique([*shards, "health_fast"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _shard_env_key(shard: str, suffix: str) -> str:
    return f"SQL_LINK_SERVICE_SHARD_{str(shard or '').strip().upper()}_{suffix}"


def _focused_shard_env(
    base: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    shards: list[str],
    critical: bool,
    max_files: int,
    max_lines_per_file: int,
    state_checkpoint_lines: int = 1500,
    include_health_fast: bool = True,
) -> tuple[list[str], dict[str, str]]:
    focus_paths = [str(row.get("source_rel") or "").strip() for row in rows[:10] if str(row.get("source_rel") or "").strip()]
    final_shards = ordered_unique([*shards, *( ["health_fast"] if include_health_fast else [] )])
    env: dict[str, str] = {**base, "SQL_LINK_SERVICE_SHARDS": ",".join(final_shards)}
    focused_max_files = str(max(max_files if critical else max_files - 2, 2))
    focused_max_lines = str(max(max_lines_per_file if critical else max_lines_per_file // 2, 4000))
    checkpoint = str(max(int(state_checkpoint_lines), 500))
    for shard in shards:
        env[_shard_env_key(shard, "PATH_CONTAINS")] = ",".join(focus_paths[:8])
        env[_shard_env_key(shard, "MAX_FILES")] = focused_max_files
        env[_shard_env_key(shard, "MAX_LINES_PER_FILE")] = focused_max_lines
        env[_shard_env_key(shard, "STATE_CHECKPOINT_LINES")] = checkpoint
    return final_shards, env


def _derivatives_drainer_env(base: dict[str, str], rows: list[dict[str, Any]], *, critical: bool) -> tuple[list[str], dict[str, str]]:
    regular_focus: list[str] = []
    aggressive_focus: list[str] = []
    crypto_focus: list[str] = []
    for row in rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
        elif _is_aggressive_decision_source(source_rel):
            aggressive_focus.append(source_rel)
        else:
            regular_focus.append(source_rel)

    shards: list[str] = []
    env: dict[str, str] = {**base}
    if regular_focus:
        shards.append("trading")
        env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
    if aggressive_focus:
        shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"] = ",".join(aggressive_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = "24000" if critical else "12000"
    if crypto_focus:
        shards.append("crypto_trading")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES"] = "10" if critical else "6"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE"] = "32000" if critical else "16000"
    if not shards:
        shards = ["trading", "aggressive_trading"]
    shards = ordered_unique([*shards, "health_fast"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _candidate_drainers(backpressure: dict[str, Any], *, critical: bool) -> list[dict[str, Any]]:
    base = _base_env(critical=critical)
    core_rows = _collect_sources(
        backpressure,
        ("governance/channels/decision/", "decisions/"),
        keys=("top_pending_files",),
    )
    governance_rows = _collect_sources(
        backpressure,
        ("governance/execution_lanes/",),
        keys=("top_pending_files",),
    )
    api_ingress_rows = _collect_sources(
        backpressure,
        ("governance/channels/api/", "governance/channels/ingress/"),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    runtime_rows = _collect_sources(
        backpressure,
        (
            "governance/channels/runtime/",
            "governance/channels/loop_state/",
        ),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    schema_rows = _collect_sources(
        backpressure,
        ("governance/events/channel_schema_violations_",),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    support_rows = _collect_sources(
        backpressure,
        ("governance/watchdog/",),
        keys=("top_support_telemetry_pending_files", "top_deferred_pending_files"),
    )
    bridge_rows = _collect_sources_by_contains(
        backpressure,
        ("paper_broker_bridge", "paper_trades_", "top_level_trade_links"),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    attribution_rows = _collect_sources_by_contains(
        backpressure,
        ("shadow_pnl_attribution_",),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    cold_rows = _collect_sources(
        backpressure,
        ("data/stale_stage/", "decision_explanations/"),
        keys=("top_cold_pending_files", "top_deferred_pending_files"),
    )
    derivatives_rows = _collect_sources_by_contains(
        backpressure,
        (
            "options",
            "option_",
            "greeks",
            "gamma",
            "vanna",
            "volga",
            "futures",
            "swaption",
            "variance",
            "volatility_swap",
            "rainbow",
            "barrier",
            "lookback",
            "structured_products",
            "synthetic",
            "cdo",
            "xva",
        ),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    derivatives_rows = [
        row for row in derivatives_rows
        if not _is_crypto_decision_source(str(row.get("source_rel") or ""))
    ]
    provider_rows = _collect_sources_by_contains(
        backpressure,
        (
            "tradingeconomics",
            "forex",
            "fx_",
            "market_context",
            "market_correlation",
            "data_source_divergence",
            "source_verification",
            "provider",
            "quote",
            "quotes",
            "options_chain",
            "tastytrade",
            "coinbase_api",
            "schwab_api",
            "external_feeds",
            "external_context",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    macro_rows = _collect_sources_by_contains(
        backpressure,
        (
            "macro",
            "fomc",
            "cspan",
            "powell",
            "federal_reserve",
            "fed_",
            "treasury",
            "calendar",
            "earnings",
            "sec_edgar",
            "news",
            "sentiment",
            "stress_scenario",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    model_rows = _collect_sources_by_contains(
        backpressure,
        (
            "retrain",
            "walk_forward",
            "champion_challenger",
            "promotion",
            "quant_models",
            "model_",
            "mlx",
            "training",
            "teacher_quality",
            "bot_quality",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    feature_rows = _collect_sources_by_contains(
        backpressure,
        (
            "feature_store",
            "event_store",
            "point_in_time",
            "replay_hash",
            "experiment",
            "label_contract",
            "collector_contract",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    report_rows = _collect_sources_by_contains(
        backpressure,
        (
            "reports",
            "report_",
            "operator_cockpit",
            "cockpit",
            "showcase",
            "documents",
            "pdf",
            "presentation",
            "timeline",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    settlement_rows = _collect_sources_by_contains(
        backpressure,
        (
            "settlement",
            "reconciliation",
            "position_reconcile",
            "positions",
            "fills",
            "allocation",
            "portfolio_exposure",
            "portfolio_ledger",
            "broker_adapter",
            "execution_gateway",
            "order_audit",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    alert_rows = _collect_sources_by_contains(
        backpressure,
        (
            "alerts",
            "notifications",
            "pager",
            "incident",
            "incident_timeline",
            "remote_alert",
            "mac_notification",
            "ops_alert",
            "operator_notify",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    memory_runtime_rows = _collect_sources_by_contains(
        backpressure,
        (
            "memory",
            "runtime_pressure",
            "resource_guard",
            "pressure_relief",
            "process_fanout",
            "runtime_gate",
            "throttle",
            "swap_pressure",
            "health_fast",
            "memory_efficiency",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    data_quality_rows = _collect_sources_by_contains(
        backpressure,
        (
            "collector_contract",
            "source_verification",
            "data_quality",
            "feature_quality",
            "provider_adapter",
            "schema_contract",
            "entitlement",
            "freshness",
            "replay_hash_contract",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    predictive_rows = _collect_sources_by_contains(
        backpressure,
        (
            "predictive_stability",
            "pressure_trajectory",
            "stability_forecast",
            "halt_forecast",
            "pressure_memory",
            "trajectory_memory",
            "runtime_forecast",
            "stability_oracle",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    self_healing_rows = _collect_sources_by_contains(
        backpressure,
        (
            "self_healing",
            "blocked_surface",
            "recovery_router",
            "blackstart",
            "safe_recovery",
            "autofix",
            "incident_closeout",
            "recovery_plan",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    collector_utility_rows = _collect_sources_by_contains(
        backpressure,
        (
            "collector_utility",
            "collector_budget",
            "collection_value",
            "collector_overlap",
            "observation_rollup",
            "collection_maturity",
            "freshness_value",
            "collector_thin",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    hot_path_budget_rows = _collect_sources_by_contains(
        backpressure,
        (
            "hot_path_storage",
            "storage_budget",
            "hot_lane_budget",
            "warm_lane_budget",
            "cold_lane_budget",
            "storage_tier_policy",
            "queue_watermark",
            "write_budget",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    admission_evidence_rows = _collect_sources_by_contains(
        backpressure,
        (
            "admission_evidence",
            "new_bot_admission",
            "sample_depth",
            "walk_forward_evidence",
            "promotion_evidence",
            "replay_hash_evidence",
            "feature_store_evidence",
            "teacher_lineage",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    writer_progress_rows = _collect_sources_by_contains(
        backpressure,
        (
            "writer_cycle",
            "writer_progress",
            "jsonl_sql_writer",
            "sql_link_service",
            "sql_link_shard_manager",
            "writer_lock",
            "merge_primary",
            "merge_progress",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    training_lineage_rows = _collect_sources_by_contains(
        backpressure,
        (
            "training_lineage",
            "lineage_manifest",
            "training_process_intelligence",
            "retrain_outcome",
            "training_readiness",
            "schema_lineage",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    label_contract_rows = _collect_sources_by_contains(
        backpressure,
        (
            "label_contract",
            "label_audit",
            "label_coverage",
            "universal_label_contract",
            "point_in_time_label",
            "label_family",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_cold_pending_files"),
    )
    collector_telemetry_rows = _collect_sources_by_contains(
        backpressure,
        (
            "collector_telemetry",
            "observation_rollup",
            "collector_sampling",
            "collector_storage_guard",
            "data_collection_storage_guard",
            "collection_observation",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    storage_route_rows = _collect_sources_by_contains(
        backpressure,
        (
            "storage_route",
            "storage_transition",
            "split_brain",
            "storage_resilience",
            "storage_reconnect",
            "storage_switch",
            "storage_quota",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    ingestion_priority_rows = _collect_sources_by_contains(
        backpressure,
        (
            "ingestion_priority",
            "queue_control",
            "backlog_quarantine",
            "ingestion_storage",
            "external_backlog",
            "backpressure_priority",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    decision_shards, decision_env = _decision_drainer_env(base, core_rows)
    api_ingress_shards, api_ingress_env = _api_ingress_drainer_env(base, api_ingress_rows, critical=critical)
    bridge_shards, bridge_env = _fast_trade_bridge_drainer_env(base, bridge_rows, critical=critical)
    attribution_shards, attribution_env = _attribution_drainer_env(base, attribution_rows, critical=critical)
    derivatives_shards, derivatives_env = _derivatives_drainer_env(base, derivatives_rows, critical=critical)
    provider_shards, provider_env = _focused_shard_env(base, provider_rows, shards=["data", "governance"], critical=critical, max_files=12, max_lines_per_file=32000, state_checkpoint_lines=1000)
    macro_shards, macro_env = _focused_shard_env(base, macro_rows, shards=["data", "governance"], critical=critical, max_files=10, max_lines_per_file=24000, state_checkpoint_lines=1000)
    model_shards, model_env = _focused_shard_env(base, model_rows, shards=["governance", "data"], critical=critical, max_files=8, max_lines_per_file=24000, state_checkpoint_lines=1200)
    feature_shards, feature_env = _focused_shard_env(base, feature_rows, shards=["data", "governance"], critical=critical, max_files=10, max_lines_per_file=32000, state_checkpoint_lines=1200)
    report_shards, report_env = _focused_shard_env(base, report_rows, shards=["governance", "data"], critical=critical, max_files=6, max_lines_per_file=16000, state_checkpoint_lines=1000)
    settlement_shards, settlement_env = _focused_shard_env(base, settlement_rows, shards=["governance", "trading_fast"], critical=critical, max_files=10, max_lines_per_file=32000, state_checkpoint_lines=1000)
    alert_shards, alert_env = _focused_shard_env(base, alert_rows, shards=["support_watchdog", "governance"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=1000)
    memory_runtime_shards, memory_runtime_env = _focused_shard_env(base, memory_runtime_rows, shards=["runtime", "support_watchdog"], critical=critical, max_files=10, max_lines_per_file=24000, state_checkpoint_lines=1000)
    data_quality_shards, data_quality_env = _focused_shard_env(base, data_quality_rows, shards=["data", "governance", "schema_violations"], critical=critical, max_files=10, max_lines_per_file=24000, state_checkpoint_lines=1000)
    predictive_shards, predictive_env = _focused_shard_env(base, predictive_rows, shards=["runtime", "governance", "support_watchdog"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    self_healing_shards, self_healing_env = _focused_shard_env(base, self_healing_rows, shards=["support_watchdog", "governance"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    collector_utility_shards, collector_utility_env = _focused_shard_env(base, collector_utility_rows, shards=["data", "governance"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    hot_path_budget_shards, hot_path_budget_env = _focused_shard_env(base, hot_path_budget_rows, shards=["governance", "data"], critical=critical, max_files=10, max_lines_per_file=24000, state_checkpoint_lines=1000)
    admission_evidence_shards, admission_evidence_env = _focused_shard_env(base, admission_evidence_rows, shards=["governance", "data"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    writer_progress_shards, writer_progress_env = _focused_shard_env(base, writer_progress_rows, shards=["support_watchdog", "health_fast", "governance"], critical=critical, max_files=6, max_lines_per_file=12000, state_checkpoint_lines=600, include_health_fast=False)
    training_lineage_shards, training_lineage_env = _focused_shard_env(base, training_lineage_rows, shards=["governance", "data"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    label_contract_shards, label_contract_env = _focused_shard_env(base, label_contract_rows, shards=["data", "governance", "schema_violations"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    collector_telemetry_shards, collector_telemetry_env = _focused_shard_env(base, collector_telemetry_rows, shards=["data", "governance"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    storage_route_shards, storage_route_env = _focused_shard_env(base, storage_route_rows, shards=["governance", "support_watchdog"], critical=critical, max_files=6, max_lines_per_file=12000, state_checkpoint_lines=700)
    ingestion_priority_shards, ingestion_priority_env = _focused_shard_env(base, ingestion_priority_rows, shards=["governance", "data", "support_watchdog"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=800)
    cold_focus_paths = [str(row["source_rel"]) for row in cold_rows[:8]]
    crypto_cold_focus_paths = [path for path in cold_focus_paths if _is_crypto_decision_source(path)]
    regular_cold_focus_paths = [path for path in cold_focus_paths if path not in crypto_cold_focus_paths]

    profiles = [
        _profile(
            name="derivatives_surface_drainer",
            reason="drain options, futures, Greeks, swaps, and exotic-derivative decision backlog through derivatives-focused trading shards",
            rows=derivatives_rows,
            shards=derivatives_shards,
            priority_boost=66_000,
            live_window_safe=True,
            env=derivatives_env,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="market_data_provider_drainer",
            reason="drain stale provider, quote, Trading Economics, FX, and source-verification tails before their age keeps storage pressure red",
            rows=provider_rows,
            shards=provider_shards,
            priority_boost=58_000,
            live_window_safe=True,
            env=provider_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="macro_event_drainer",
            reason="drain macro calendar, Fed/CSPAN, earnings, news, sentiment, and stress-scenario backlog without widening the main governance lane",
            rows=macro_rows,
            shards=macro_shards,
            priority_boost=54_000,
            live_window_safe=True,
            env=macro_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="model_research_drainer",
            reason="drain retrain, walk-forward, champion/challenger, MLX, and quant-model artifacts through research-safe shards",
            rows=model_rows,
            shards=model_shards,
            priority_boost=42_000,
            live_window_safe=False,
            env=model_env,
            min_pending_lines=50,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS * 2.0,
        ),
        _profile(
            name="feature_event_store_drainer",
            reason="drain feature-store, event-store, replay-hash, experiment, and label-contract backlog as a bounded data-plane lane",
            rows=feature_rows,
            shards=feature_shards,
            priority_boost=40_000,
            live_window_safe=True,
            env=feature_env,
            min_pending_lines=50,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="report_cockpit_drainer",
            reason="drain report, cockpit, showcase, document, PDF, and presentation artifacts behind live collection lanes",
            rows=report_rows,
            shards=report_shards,
            priority_boost=18_000,
            live_window_safe=False,
            env=report_env,
            min_pending_lines=50,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS * 4.0,
        ),
        _profile(
            name="settlement_reconciliation_drainer",
            reason="drain settlement, reconciliation, positions, fills, allocation, broker-adapter, and execution-audit backlog as a portfolio-ledger lane",
            rows=settlement_rows,
            shards=settlement_shards,
            priority_boost=52_000,
            live_window_safe=True,
            env=settlement_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="memory_runtime_artifact_drainer",
            reason="drain memory pressure, runtime-gate, fanout, throttle, and health-fast artifacts without widening core runtime loops",
            rows=memory_runtime_rows,
            shards=memory_runtime_shards,
            priority_boost=49_000,
            live_window_safe=True,
            env=memory_runtime_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="data_quality_contract_drainer",
            reason="drain data-quality, source-verification, provider-adapter, entitlement, freshness, and collector-contract backlog through data-plane guard shards",
            rows=data_quality_rows,
            shards=data_quality_shards,
            priority_boost=46_000,
            live_window_safe=True,
            env=data_quality_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="alert_notification_drainer",
            reason="drain alert, notification, pager, incident-timeline, remote-alert, and operator notification artifacts as a support lane",
            rows=alert_rows,
            shards=alert_shards,
            priority_boost=34_000,
            live_window_safe=True,
            env=alert_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="predictive_stability_drainer",
            reason="drain pressure trajectory, halt forecast, runtime forecast, and stability memory artifacts into an advisory predictive lane",
            rows=predictive_rows,
            shards=predictive_shards,
            priority_boost=63_000,
            live_window_safe=True,
            env=predictive_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="self_healing_recovery_drainer",
            reason="drain blocked-surface, recovery-router, blackstart, autofix, incident closeout, and safe recovery artifacts as a bounded self-healing lane",
            rows=self_healing_rows,
            shards=self_healing_shards,
            priority_boost=61_000,
            live_window_safe=True,
            env=self_healing_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="collector_utility_budget_drainer",
            reason="drain collector utility, collection value, overlap, observation rollup, and thinning-budget artifacts without widening live collectors",
            rows=collector_utility_rows,
            shards=collector_utility_shards,
            priority_boost=57_000,
            live_window_safe=True,
            env=collector_utility_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="hot_path_storage_budget_drainer",
            reason="drain hot, warm, cold, watermark, and write-budget storage artifacts so trading paths keep priority over reports and explainers",
            rows=hot_path_budget_rows,
            shards=hot_path_budget_shards,
            priority_boost=56_000,
            live_window_safe=True,
            env=hot_path_budget_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="admission_evidence_drainer",
            reason="drain admission evidence, sample-depth, walk-forward, replay-hash, feature-store, and teacher-lineage artifacts for blocked bot candidates",
            rows=admission_evidence_rows,
            shards=admission_evidence_shards,
            priority_boost=51_000,
            live_window_safe=True,
            env=admission_evidence_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="writer_progress_recovery_drainer",
            reason="drain writer progress, SQL-link service, shard manager, lock, merge-primary, and progress evidence artifacts without starting another writer",
            rows=writer_progress_rows,
            shards=writer_progress_shards,
            priority_boost=48_000,
            live_window_safe=True,
            env=writer_progress_env,
            min_pending_lines=10,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS / 2.0,
        ),
        _profile(
            name="training_lineage_drainer",
            reason="drain training lineage, manifest, process-intelligence, retrain-outcome, and schema-lineage artifacts before targeted retrain gates are evaluated",
            rows=training_lineage_rows,
            shards=training_lineage_shards,
            priority_boost=59_000,
            live_window_safe=True,
            env=training_lineage_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="label_contract_drainer",
            reason="drain label audit, label coverage, point-in-time label, and universal label-contract artifacts as an isolated training-data contract lane",
            rows=label_contract_rows,
            shards=label_contract_shards,
            priority_boost=55_000,
            live_window_safe=True,
            env=label_contract_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="collector_telemetry_rollup_drainer",
            reason="drain collector telemetry, observation rollup, sampling, and data-collection storage-guard artifacts so collector expansion does not outpace ingestion",
            rows=collector_telemetry_rows,
            shards=collector_telemetry_shards,
            priority_boost=53_000,
            live_window_safe=True,
            env=collector_telemetry_env,
            min_pending_lines=25,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS,
        ),
        _profile(
            name="storage_route_reconcile_drainer",
            reason="drain storage route, transition, split-brain, resilience, reconnect, switch, and quota artifacts as a storage-routing reconciliation lane",
            rows=storage_route_rows,
            shards=storage_route_shards,
            priority_boost=47_000,
            live_window_safe=True,
            env=storage_route_env,
            min_pending_lines=10,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS / 2.0,
        ),
        _profile(
            name="ingestion_priority_drainer",
            reason="drain ingestion priority, queue-control, backlog quarantine, ingestion-storage, and external backlog artifacts before broad backlog waves are launched",
            rows=ingestion_priority_rows,
            shards=ingestion_priority_shards,
            priority_boost=44_000,
            live_window_safe=True,
            env=ingestion_priority_env,
            min_pending_lines=10,
            stale_ready_age_seconds=MICRO_STALE_READY_AGE_SECONDS / 2.0,
        ),
        _profile(
            name="core_decision_drainer",
            reason="drain concentrated decision-channel backlog through the matching hot decision shards",
            rows=core_rows,
            shards=decision_shards,
            priority_boost=100_000 if _safe_int(backpressure.get("pending_lines"), 0) >= CORE_HARD_PENDING_LINES else 60_000,
            live_window_safe=True,
            env=decision_env,
        ),
        _profile(
            name="governance_execution_drainer",
            reason="drain stale execution-lane governance backlog before widening broad governance work",
            rows=governance_rows,
            shards=["governance", "health_fast", "support_watchdog"],
            priority_boost=80_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "governance,health_fast,support_watchdog",
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in governance_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": "14",
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE": "64000",
            },
        ),
        _profile(
            name="api_ingress_drainer",
            reason="drain API and ingress channel backlog separately from runtime heartbeats",
            rows=api_ingress_rows,
            shards=api_ingress_shards,
            priority_boost=70_000,
            live_window_safe=True,
            env=api_ingress_env,
        ),
        _profile(
            name="runtime_channel_drainer",
            reason="drain runtime channel files without pulling cold analytics work forward",
            rows=runtime_rows,
            shards=["runtime", "crypto_runtime", "health_fast"],
            priority_boost=50_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "runtime,crypto_runtime,health_fast",
                "SQL_LINK_SERVICE_SHARD_RUNTIME_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in runtime_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_FILES": "16",
                "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_FILES": "10",
            },
        ),
        _profile(
            name="schema_violation_drainer",
            reason="drain schema violation artifacts into the isolated low-priority shard",
            rows=schema_rows,
            shards=["schema_violations", "health_fast"],
            priority_boost=45_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "schema_violations,health_fast",
                "SQL_LINK_SERVICE_SHARD_SCHEMA_VIOLATIONS_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in schema_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_SCHEMA_VIOLATIONS_MAX_FILES": "12" if critical else "6",
                "SQL_LINK_SERVICE_SHARD_SCHEMA_VIOLATIONS_MAX_LINES_PER_FILE": "32000" if critical else "16000",
                "SQL_LINK_SERVICE_SHARD_SCHEMA_VIOLATIONS_STATE_CHECKPOINT_LINES": "1000",
            },
        ),
        _profile(
            name="support_watchdog_drainer",
            reason="drain failover, pager, and killswitch support telemetry off the main governance path",
            rows=support_rows,
            shards=["support_watchdog", "health_fast"],
            priority_boost=30_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "support_watchdog,health_fast",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in support_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_FILES": "20",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
            },
        ),
        _profile(
            name="fast_trade_bridge_drainer",
            reason="drain paper broker bridge and top-level trade-link backlog through the fast trading shards",
            rows=bridge_rows,
            shards=bridge_shards,
            priority_boost=25_000,
            live_window_safe=True,
            env=bridge_env,
        ),
        _profile(
            name="attribution_drainer",
            reason="drain shadow attribution backlog into non-primary attribution shards",
            rows=attribution_rows,
            shards=attribution_shards,
            priority_boost=22_000,
            live_window_safe=True,
            env=attribution_env,
        ),
        _profile(
            name="cold_stage_drainer",
            reason="drain stale-stage and explanation backlog only during protected drain windows",
            rows=cold_rows,
            shards=["data", "explanations", "crypto_explanations", "health_fast"],
            priority_boost=20_000,
            live_window_safe=False,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "data,explanations,crypto_explanations,health_fast",
                "JSONL_SQL_MAX_COLD_LANE_FILES": "4" if critical else "2",
                "SQL_LINK_SERVICE_SHARD_DATA_PATH_CONTAINS": ",".join(cold_focus_paths),
                "SQL_LINK_SERVICE_SHARD_DATA_MAX_FILES": "10",
                "SQL_LINK_SERVICE_SHARD_DATA_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": "8",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": "8",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS": ",".join(regular_cold_focus_paths),
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS": ",".join(crypto_cold_focus_paths),
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_STATE_CHECKPOINT_LINES": "2000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_STATE_CHECKPOINT_LINES": "2000",
            },
        ),
    ]
    profiles = _apply_age_pressure_priority(profiles, backpressure)
    ready = [row for row in profiles if str(row.get("status") or "") == "ready"]
    idle = [row for row in profiles if str(row.get("status") or "") != "ready"]
    ready.sort(
        key=lambda row: (
            _safe_int(row.get("effective_priority_score"), _safe_int(row.get("priority_score"), 0)),
            _safe_int(row.get("pending_lines"), 0),
        ),
        reverse=True,
    )
    return ready + idle


def _write_service_request(path: Path, *, active_drainer: dict[str, Any], now_utc: datetime, ttl_seconds: int) -> dict[str, Any]:
    expires_utc = now_utc.timestamp() + max(int(ttl_seconds), 300)
    env = active_drainer.get("env_overrides") if isinstance(active_drainer.get("env_overrides"), dict) else {}
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "active": True,
        "request_kind": "backpressure_drainer_fleet",
        "reason": f"backpressure_drainer_fleet:{active_drainer.get('name', '')}",
        "owner_bot_id": str(active_drainer.get("owner_bot_id") or ""),
        "backup_bot_ids": list(active_drainer.get("backup_bot_ids") or []),
        "assigned_pressure_lane": str(active_drainer.get("assigned_pressure_lane") or ""),
        "ops_infrabots": list(active_drainer.get("ops_infrabots") or []),
        "self_accommodation": active_drainer.get("self_accommodation") if isinstance(active_drainer.get("self_accommodation"), dict) else {},
        "requested_at": now_utc.isoformat(),
        "expires_utc": datetime.fromtimestamp(expires_utc, tz=timezone.utc).isoformat(),
        "env_overrides": {str(key): str(value) for key, value in env.items() if str(key).strip()},
    }
    write_payload(path, payload)
    return payload


def _fleet_self_accommodation(
    *,
    active_drainer: dict[str, Any],
    ready_drainers: list[dict[str, Any]],
    writer_lock_state: dict[str, Any],
    live_window_allowed: bool,
    blocked_reasons: list[str],
    service_request: dict[str, Any],
    apply_requested: bool,
    ttl_seconds: int,
) -> dict[str, Any]:
    writer_held = bool(writer_lock_state.get("held", False))
    if not active_drainer:
        next_safe_action = "refresh_backpressure_snapshot" if "missing_backpressure_artifact" in blocked_reasons else "idle_no_ready_drainer"
        mode = "idle"
    elif writer_held:
        next_safe_action = "wait_for_current_writer_then_re_score"
        mode = "writer_wait"
    elif not live_window_allowed:
        next_safe_action = "park_until_protected_window_or_force_live_window"
        mode = "market_hours_guard"
    elif service_request:
        next_safe_action = "single_writer_handoff_requested"
        mode = "focused_handoff"
    elif apply_requested:
        next_safe_action = "retry_after_blocker_clears"
        mode = "blocked" if blocked_reasons else "ready"
    else:
        next_safe_action = "run_backpressure_drainer_fleet_apply_or_bounded_super_drainer_wave"
        mode = "preview_ready"

    return {
        "self_accommodating": True,
        "mode": mode,
        "next_safe_action": next_safe_action,
        "active_drainer": str(active_drainer.get("name") or ""),
        "ready_drainer_count": len(ready_drainers),
        "allowed_parallel_writers": 1,
        "single_writer_only": True,
        "starts_parallel_sql_writers": False,
        "writer_lock_held": writer_held,
        "writer_lock_owner": str(writer_lock_state.get("owner") or ""),
        "service_ttl_seconds": int(max(ttl_seconds, 300)),
        "bounded_wave_size": 1,
        "backs_off_when": ordered_unique(
            [
                "jsonl_sql_writer_lock_held" if writer_held else "",
                *blocked_reasons,
                "progress_stall",
                "storage_snapshot_stale",
                "runtime_or_memory_pressure_high",
            ]
        ),
        "coordination_sequence": [
            "score_backpressure_lanes",
            "select_one_active_drainer",
            "write_single_writer_service_request",
            "let_writer_cycle_coordinator_run",
            "refresh_storage_snapshot",
            "re_score_before_next_wave",
        ],
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    force_live_window: bool = False,
    ttl_seconds: int = 900,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    backpressure = load_json(health_root / "ingestion_backpressure_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    governor = load_json(health_root / "ingestion_storage_governor_latest.json")
    critical = bool(
        str(governor.get("profile") or "").strip() == "critical_backpressure"
        or str(storage_control.get("severity") or "").strip() == "critical"
        or _safe_int(backpressure.get("pending_lines"), 0) >= CORE_HARD_PENDING_LINES
    )
    window = eastern_off_hours_window(now=now)
    drainers = _candidate_drainers(backpressure, critical=critical)
    ready_drainers = [row for row in drainers if str(row.get("status") or "") == "ready"]
    active_drainer = ready_drainers[0] if ready_drainers else {}
    live_window_allowed = bool(
        force_live_window
        or window.get("active", False)
        or bool(active_drainer.get("live_window_safe", False))
    )
    service_request: dict[str, Any] = {}
    blocked_reasons: list[str] = []
    if not backpressure:
        blocked_reasons.append("missing_backpressure_artifact")
    if active_drainer and not live_window_allowed:
        blocked_reasons.append("market_hours_guard")

    if apply and active_drainer and not blocked_reasons:
        service_request = _write_service_request(
            project_root / "governance" / "health" / "sql_link_service_request_latest.json",
            active_drainer=active_drainer,
            now_utc=now,
            ttl_seconds=ttl_seconds,
        )

    active_env = active_drainer.get("env_overrides") if isinstance(active_drainer.get("env_overrides"), dict) else {}
    writer_lock_state = _writer_lock_snapshot(project_root / "governance" / "locks" / "jsonl_sql_writer.lock")
    self_accommodation = _fleet_self_accommodation(
        active_drainer=active_drainer,
        ready_drainers=ready_drainers,
        writer_lock_state=writer_lock_state,
        live_window_allowed=live_window_allowed,
        blocked_reasons=blocked_reasons,
        service_request=service_request,
        apply_requested=bool(apply),
        ttl_seconds=int(ttl_seconds),
    )
    next_drainer_queue = [
        {
            "name": str(row.get("name") or ""),
            "pending_lines": _safe_int(row.get("pending_lines"), 0),
            "priority_score": _safe_int(row.get("priority_score"), 0),
            "live_window_safe": bool(row.get("live_window_safe", False)),
            "shards": list(row.get("shards") or []),
            "next_safe_action": str(
                (row.get("self_accommodation") or {}).get("safe_expansion_rule")
                if isinstance(row.get("self_accommodation"), dict)
                else "re_score_before_handoff"
            ),
        }
        for row in ready_drainers[1:5]
    ]
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(not blocked_reasons),
        "overall_status": "handoff_requested" if service_request else ("ready" if active_drainer and not blocked_reasons else ("idle" if not blocked_reasons else "blocked")),
        "apply_requested": bool(apply),
        "service_request": service_request,
        "service_request_path": str(project_root / "governance" / "health" / "sql_link_service_request_latest.json"),
        "writer_active": bool(writer_lock_state.get("held", False)),
        "writer_lock_owner": str(writer_lock_state.get("owner") or ""),
        "writer_lock_held": bool(writer_lock_state.get("held", False)),
        "self_accommodation": self_accommodation,
        "off_hours_window": window,
        "blocked_reasons": blocked_reasons,
        "active_drainer": {
            key: value
            for key, value in active_drainer.items()
            if key not in {"env_overrides"}
        },
        "ready_drainer_count": len(ready_drainers),
        "next_drainer_queue": next_drainer_queue,
        "candidate_drainers": [
            {key: value for key, value in row.items() if key not in {"env_overrides"}}
            for row in drainers
        ],
        "active_env_override_count": len(active_env),
        "active_env_overrides": active_env,
        "metrics": {
            "core_pending_lines": _safe_int(backpressure.get("pending_lines"), 0),
            "total_pending_lines": _safe_int(backpressure.get("pending_lines_total"), 0),
            "deferred_pending_lines": _safe_int(backpressure.get("pending_lines_deferred"), 0),
            "cold_pending_lines": _safe_int(backpressure.get("pending_lines_cold"), 0),
            "support_pending_lines": _safe_int(backpressure.get("pending_lines_support_telemetry"), 0),
            "ready_drainer_count": len(ready_drainers),
            "expanded_lane_count": len(drainers),
            "next_drainer_queue_count": len(next_drainer_queue),
            "stale_tail_ready_count": sum(1 for row in ready_drainers if str(row.get("readiness_reason") or "") == "stale_tail"),
            "live_window_safe_ready_count": sum(1 for row in ready_drainers if bool(row.get("live_window_safe", False))),
            "active_concentrated_backlog": bool(active_drainer.get("concentration", {}).get("concentrated", False)) if isinstance(active_drainer.get("concentration"), dict) else False,
            "self_accommodating_lane_count": sum(
                1
                for row in drainers
                if isinstance(row.get("self_accommodation"), dict)
                and bool(row["self_accommodation"].get("self_accommodating", False))
            ),
        },
        "recommended_actions": ordered_unique(
            [
                "keep one SQL writer active; use these drainers as focused handoffs instead of parallel SQLite writers",
                "run the highest-priority drainer first, then let the next storage-autopilot cycle re-score the backlog",
                "keep API, schema, attribution, derivatives, market-data, macro, feature-store, settlement, alert, memory-runtime, data-quality, predictive-stability, self-healing, collector-utility, hot-path-budget, admission-evidence, writer-progress, and paper-trade bridge drains isolated so they do not compete with hot decision files",
                "use stale-tail drainers for tiny old files whose age can keep storage pressure red even when line counts are low",
                "use live-window-safe drainers for core, runtime, provider, derivatives, feature, and support pressure; keep report/model/cold-stage drainers for protected windows",
                "let each drainer self-accommodate by parking on writer locks, market-hour guards, stale snapshots, and progress stalls",
                "keep the drainer fleet wired into storage-backpressure-autopilot so focused handoffs happen automatically",
            ]
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and hand off focused backpressure drainers to the single SQL writer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force-live-window", action="store_true")
    parser.add_argument("--ttl-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()

    lock = _acquire_nonblocking_lock(lock_file)
    if lock is None:
        payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": True,
            "overall_status": "already_running",
            "busy": True,
            "lock_file": str(lock_file),
        }
        write_payload(out_file, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print("backpressure_drainer_fleet overall_status=already_running")
        return 0

    with lock:
        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            force_live_window=bool(args.force_live_window),
            ttl_seconds=int(args.ttl_seconds),
        )
        payload["lock_file"] = str(lock_file)
        write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "backpressure_drainer_fleet "
            f"overall_status={payload.get('overall_status', '')} "
            f"ready_drainers={int(payload.get('ready_drainer_count', 0) or 0)}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"already_running", "ready", "idle", "handoff_requested"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
