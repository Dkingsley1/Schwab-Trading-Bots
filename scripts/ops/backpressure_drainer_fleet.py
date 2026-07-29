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
SPARSE_LARGE_LINE_BYTES_PER_LINE = 64 * 1024
SPARSE_LARGE_DECISION_MAX_BYTES_PER_FILE = 128 * 1024 * 1024
SPARSE_LARGE_DECISION_SQLITE_BATCH_MAX_BYTES = 32 * 1024 * 1024
STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS = 64_000
STALE_DECISION_CATCH_UP_MAX_BYTES_PER_FILE = 1024 * 1024 * 1024
STALE_DECISION_CATCH_UP_SQLITE_BATCH_MAX_BYTES = 256 * 1024 * 1024
SIGNAL_GENERATION_CATCH_UP_MAX_LINES = 512_000
SIGNAL_GENERATION_CATCH_UP_MERGE_MAX_JSONL_ROWS = 256_000
SIGNAL_GENERATION_CATCH_UP_MAX_BYTES_PER_FILE = 1024 * 1024 * 1024
SIGNAL_GENERATION_CATCH_UP_SQLITE_BATCH_MAX_BYTES = 256 * 1024 * 1024
SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS = 2_000
DEFAULT_STORAGE_EJECT_COOLDOWN_SECONDS = 60 * 60
RAW_LIVE_EXPANSION_HOT_DRAINERS = {
    "stale_decision_log_drainer",
    "core_decision_drainer",
    "operations_guard_drainer",
    "governance_execution_drainer",
    "api_ingress_drainer",
    "runtime_channel_drainer",
    "schema_violation_drainer",
    "fast_trade_bridge_drainer",
    "ingestion_priority_drainer",
    "hot_path_storage_budget_drainer",
    "writer_progress_recovery_drainer",
}
RAW_LIVE_EXPANSION_SUPPORT_DRAINERS = {"risk_support_drainer", "support_watchdog_drainer"}

DRAINER_OWNERS: dict[str, dict[str, Any]] = {
    "stale_decision_log_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v187_data_collection_storage_budget_guard",
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
        ],
        "pressure_lane": "stale_decision_log_backpressure",
        "ops_infrabots": ["backpressure_slo_bot", "storage_backpressure_autopilot", "writer_cycle_coordinator"],
    },
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
    "operations_guard_drainer": {
        "owner_bot_id": "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        ],
        "pressure_lane": "operations_guard_feedback_backpressure",
        "ops_infrabots": [
            "adaptive_regression_guard",
            "infrabot_adaptive_governor",
            "backpressure_drainer_fleet",
            "writer_cycle_coordinator",
        ],
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
    "risk_support_drainer": {
        "owner_bot_id": "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        "backup_bot_ids": [
            "brain_refinery_v316_collection_halt_pressure_preemption_guard",
            "brain_refinery_v187_data_collection_storage_budget_guard",
        ],
        "pressure_lane": "risk_support_backpressure",
        "ops_infrabots": ["backpressure_drainer_fleet", "storage_backpressure_autopilot", "training_runtime_control"],
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


def _performance_core_target() -> int:
    for name in (
        "BOT_PERFORMANCE_CORE_TARGET",
        "BOT_PERFORMANCE_CORE_PRIMARY_COUNT",
        "BACKLOG_PCORE_TARGET",
    ):
        value = _safe_int(os.getenv(name), 0)
        if value > 0:
            return value
    return min(max(os.cpu_count() or 1, 1), 8)


def _text_in(raw: str, tokens: tuple[str, ...]) -> bool:
    text = str(raw or "").strip().lower()
    return any(token in text for token in tokens)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _recent_storage_eject_signal(*, now: datetime | None = None) -> dict[str, Any]:
    if not _env_enabled("BACKLOG_RECENT_EJECT_DAMPING", True):
        return {"active": False, "enabled": False, "reason": "disabled_by_env"}
    if os.getenv("PYTEST_CURRENT_TEST") and "STORAGE_EJECT_GUARD_LOG" not in os.environ:
        return {"active": False, "enabled": True, "reason": "pytest_default_live_log_ignored"}
    log_path = Path(
        os.getenv(
            "STORAGE_EJECT_GUARD_LOG",
            str(Path.home() / "Library/Logs/schwab_trading_bot/storage_eject_guard.log"),
        )
    )
    cooldown_seconds = max(_safe_float(os.getenv("BACKLOG_RECENT_EJECT_COOLDOWN_SECONDS"), DEFAULT_STORAGE_EJECT_COOLDOWN_SECONDS), 0.0)
    if cooldown_seconds <= 0.0:
        return {"active": False, "enabled": True, "reason": "zero_cooldown", "cooldown_seconds": 0.0}
    if not log_path.exists():
        return {"active": False, "enabled": True, "reason": "log_missing", "log_path": str(log_path), "cooldown_seconds": cooldown_seconds}
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - 131_072, 0), os.SEEK_SET)
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {"active": False, "enabled": True, "reason": "log_read_failed", "error": str(exc), "log_path": str(log_path)}
    for line in reversed(lines):
        storage_event = "disk disappeared" in line or "handling unmount" in line or "handling eject" in line
        if not storage_event or "mountRoot=/Volumes/BOT_LOGS" not in line:
            continue
        marker_end = line.find("]")
        if not line.startswith("[") or marker_end <= 1:
            continue
        ts = _parse_iso_utc(line[1:marker_end])
        if ts is None:
            continue
        age_seconds = max((now_utc - ts).total_seconds(), 0.0)
        if age_seconds <= cooldown_seconds:
            cap = max(_safe_int(os.getenv("BACKLOG_RECENT_EJECT_MAX_WORKERS"), 3), 1)
            return {
                "active": True,
                "enabled": True,
                "age_seconds": round(float(age_seconds), 3),
                "cooldown_seconds": round(float(cooldown_seconds), 3),
                "max_workers": int(cap),
                "event_line": line[-240:],
                "log_path": str(log_path),
                "policy": "temporarily cap backlog preprocess burst after BOT_LOGS disappears so USB/APFS stability wins over raw drain speed",
            }
        return {
            "active": False,
            "enabled": True,
            "reason": "last_eject_outside_cooldown",
            "age_seconds": round(float(age_seconds), 3),
            "cooldown_seconds": round(float(cooldown_seconds), 3),
            "log_path": str(log_path),
        }
    return {"active": False, "enabled": True, "reason": "no_recent_eject_event", "cooldown_seconds": round(float(cooldown_seconds), 3), "log_path": str(log_path)}


def _p_core_foreground_reserve(*, p_cores: int, host_context: dict[str, Any] | None = None) -> int:
    explicit_reserve = _safe_int(os.getenv("BACKLOG_PCORE_FOREGROUND_RESERVE"), 0)
    if explicit_reserve > 0:
        return min(explicit_reserve, max(p_cores - 1, 1))
    context = host_context if isinstance(host_context, dict) else {}
    resource = context.get("resource_guard") if isinstance(context.get("resource_guard"), dict) else {}
    computer = context.get("computer_task") if isinstance(context.get("computer_task"), dict) else {}
    intent = str(os.getenv("COMPUTER_RESOURCE_INTENT") or "").strip().lower()
    primary_task = str(computer.get("primary_task") or os.getenv("COMPUTER_PRIMARY_TASK") or "").strip().lower()
    creative_kind = str(resource.get("creative_session_kind") or "").strip().lower()
    if any(
        _text_in(value, ("logic", "final", "video", "audio_production", "video_editing", "virtualization"))
        for value in (intent, primary_task, creative_kind)
    ):
        return 3
    if any(_text_in(value, ("yield", "foreground", "music", "developer", "browser")) for value in (intent, primary_task, creative_kind)):
        return 2
    return 1


def _p_core_burst_intelligence(
    *,
    p_core_count: int,
    foreground_reserve: int,
    writer_reserve: int,
    critical: bool,
    backlog_ratio: float,
    sparse_active: bool,
    host_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = host_context if isinstance(host_context, dict) else {}
    runtime = context.get("runtime_throttle") if isinstance(context.get("runtime_throttle"), dict) else {}
    resource = context.get("resource_guard") if isinstance(context.get("resource_guard"), dict) else {}
    computer = context.get("computer_task") if isinstance(context.get("computer_task"), dict) else {}
    throttle_profile = str(runtime.get("throttle_profile") or "").strip().lower()
    compute_pressure = str(runtime.get("compute_pressure_level") or "").strip().lower()
    memory_pressure = str(runtime.get("memory_pressure_level") or resource.get("memory_pressure_state") or "").strip().lower()
    memory_pressure_kind = str(runtime.get("memory_pressure_kind") or resource.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(runtime.get("swap_used_gb"), _safe_float(resource.get("swap_used_gb"), 0.0))
    compressed_store_gb = _safe_float(resource.get("compressed_store_gb"), 0.0)
    compressor_gb = _safe_float(resource.get("compressor_gb"), _safe_float(runtime.get("compressor_gb"), 0.0))
    compressed_pressure_gb = compressor_gb if compressor_gb > 0.0 else compressed_store_gb
    pages_throttled = _safe_float(resource.get("pages_throttled"), 0.0)
    host_saturation = _safe_float(runtime.get("host_saturation_score"), _safe_float(resource.get("load1_per_core"), 0.0) * 100.0)
    creative_level = str(resource.get("creative_session_level") or "").strip().lower()
    creative_kind = str(resource.get("creative_session_kind") or "").strip().lower()
    co_running_level = str(resource.get("co_running_session_level") or "").strip().lower()
    primary_task = str(computer.get("primary_task") or os.getenv("COMPUTER_PRIMARY_TASK") or "").strip().lower()
    off_hours_active = bool(context.get("off_hours_active", False))
    explicit = _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE"), 0)
    recent_eject = _recent_storage_eject_signal()
    user_reserve_target = max(
        _safe_int(
            os.getenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET"),
            _safe_int(os.getenv("AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET"), 0),
        ),
        0,
    )
    base_budget = max(min(int(p_core_count) - int(foreground_reserve) - int(writer_reserve), 6), 1)
    creative_heavy = bool(
        _text_in(primary_task, ("audio_production", "video_editing", "logic", "final", "virtualization"))
        or _text_in(creative_kind, ("audio_production", "video_editing", "logic", "final"))
        or creative_level in {"hot", "dual_pro"}
    )
    high_pressure = bool(
        throttle_profile in {"protect_live"}
        or compute_pressure in {"high", "blocked", "critical"}
        or memory_pressure in {"yellow", "red", "high", "critical", "blocked"}
        or host_saturation >= 70.0
        or (co_running_level in {"heavy_competition"} and host_saturation >= 55.0)
    )
    memory_critical = bool(
        memory_pressure in {"red", "critical", "blocked"}
        or memory_pressure_kind in {"throttled", "swap_exhaustion"}
        or pages_throttled > 0
        or swap_used_gb >= 18.0
    )
    memory_clear = bool(
        memory_pressure in {"", "normal", "green", "none", "clear"}
        and memory_pressure_kind in {"", "none", "normal", "clear"}
        and pages_throttled <= 0.0
    )
    memory_elevated = bool(
        memory_critical
        or memory_pressure in {"yellow", "high"}
        or memory_pressure_kind in {"swap_only", "swap_only_with_headroom"}
        or swap_used_gb >= 12.0
        or compressed_pressure_gb >= 16.0
    )
    background_task_clear = primary_task in {"", "idle", "none", "background", "backlog", "backlog_drain"}
    deep_memory_clear = bool(
        memory_pressure in {"", "normal", "green", "none", "clear"}
        and memory_pressure_kind in {"", "none", "normal", "clear"}
        and swap_used_gb < 2.0
        and compressed_pressure_gb < 8.0
        and pages_throttled <= 0.0
    )
    deep_host_cool = bool(
        host_saturation < 42.0
        and compute_pressure in {"", "normal", "green", "none", "clear", "watch"}
        and throttle_profile not in {"protect_live", "sustain"}
    )
    full_p_core_budget_requested = _env_enabled("BACKLOG_PCORE_USE_FULL_PERFORMANCE_CORE_BUDGET", False)
    seventh_core_allowed = bool(
        _env_enabled("BACKLOG_PCORE_ENABLE_SEVENTH", True)
        and critical
        and p_core_count >= 8
        and (off_hours_active or backlog_ratio >= 4.0 or sparse_active)
        and background_task_clear
        and not creative_heavy
        and not high_pressure
        and deep_memory_clear
        and deep_host_cool
    )
    if full_p_core_budget_requested and p_core_count >= 8 and memory_clear and not creative_heavy:
        max_budget = max(base_budget, min(int(p_core_count) - int(writer_reserve), 7))
    else:
        max_budget = max(base_budget, min(int(p_core_count) - int(writer_reserve), 7)) if seventh_core_allowed else base_budget
    user_reserve_worker_cap = 0
    elastic_reserve_loan_cap = 0
    elastic_reserve_loan_allowed = bool(
        user_reserve_target > 0
        and critical
        and throttle_profile == "protect_live"
        and (backlog_ratio >= 20.0 or sparse_active)
        and background_task_clear
        and not creative_heavy
        and memory_clear
        and swap_used_gb < 3.0
        and compressed_pressure_gb < 9.0
        and host_saturation < 76.0
        and compute_pressure in {"", "normal", "green", "none", "clear", "watch", "elevated"}
    )
    if user_reserve_target > 0:
        user_reserve_worker_cap = max(int(p_core_count) - int(user_reserve_target), 1)
        if elastic_reserve_loan_allowed:
            elastic_reserve_loan_cap = max(int(p_core_count) - max(int(user_reserve_target) - 1, 1), 1)
            max_budget = min(max_budget, max(user_reserve_worker_cap, elastic_reserve_loan_cap))
        else:
            max_budget = min(max_budget, user_reserve_worker_cap)
    burst_allowed = bool(
        critical
        and max_budget >= 6
        and (off_hours_active or backlog_ratio >= 2.0 or sparse_active)
        and not creative_heavy
        and not high_pressure
        and host_saturation < 50.0
        and memory_pressure in {"", "normal", "green", "none", "clear"}
    )
    protected_probe_compute_ok = bool(
        compute_pressure not in {"blocked", "critical"}
        and not (compute_pressure == "high" and host_saturation >= 66.0)
    )
    protected_backlog_probe_allowed = bool(
        critical
        and throttle_profile == "protect_live"
        and max_budget >= 3
        and (backlog_ratio >= 4.0 or sparse_active)
        and background_task_clear
        and not creative_heavy
        and not memory_elevated
        and protected_probe_compute_ok
        and host_saturation < 66.0
    )
    protected_backlog_probe_wide_allowed = bool(
        protected_backlog_probe_allowed
        and max_budget >= 4
        and (backlog_ratio >= 20.0 or sparse_active)
        and compute_pressure not in {"high", "blocked", "critical"}
        and host_saturation < 60.0
        and swap_used_gb < 3.0
        and compressed_pressure_gb < 11.0
        and pages_throttled <= 0.0
    )
    guarded_backlog_probe_allowed = bool(
        critical
        and throttle_profile == "protect_live"
        and max_budget >= 3
        and (backlog_ratio >= 20.0 or sparse_active)
        and background_task_clear
        and not creative_heavy
        and not memory_elevated
        and compute_pressure in {"", "normal", "green", "none", "clear", "watch", "elevated"}
        and host_saturation < 76.0
        and swap_used_gb < 4.0
        and compressed_pressure_gb < 14.0
        and pages_throttled <= 0.0
    )
    guarded_backlog_probe_wide_allowed = bool(
        guarded_backlog_probe_allowed
        and max_budget >= 4
        and elastic_reserve_loan_allowed
        and host_saturation < 76.0
        and swap_used_gb < 3.0
        and compressed_pressure_gb < 9.0
    )
    if explicit > 0:
        selected = max(1, min(explicit, max_budget))
        mode = "operator_override"
        reason = "explicit BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE applied"
    elif memory_critical:
        selected = min(max_budget, 2)
        mode = "memory_relief_2"
        reason = "memory pressure is critical, so backlog preprocessing is capped to preserve headroom"
    elif memory_elevated:
        selected = min(max_budget, 3)
        mode = "memory_relief_3"
        reason = "memory pressure is elevated, so backlog preprocessing is narrowed before swap gets worse"
    elif creative_heavy:
        selected = min(max_budget, 3)
        mode = "creative_foreground_protect_3"
        reason = "creative or foreground work is active, so backlog preprocessing leaves extra interactive headroom"
    elif high_pressure:
        if protected_backlog_probe_wide_allowed:
            selected = min(max_budget, 4)
            mode = "protect_live_backlog_probe_4"
            reason = "protect-live is active, but backlog pressure is extreme and memory is still normal, so a bounded fourth P-core backlog probe is allowed"
        elif protected_backlog_probe_allowed:
            selected = min(max_budget, 3)
            mode = "protect_live_backlog_probe_3"
            reason = "protect-live is active, but backlog pressure is severe and memory is normal, so a bounded third P-core backlog probe is allowed"
        elif guarded_backlog_probe_allowed:
            if guarded_backlog_probe_wide_allowed:
                selected = min(max_budget, 4)
                mode = "guarded_backlog_probe_4"
                reason = "protect-live host pressure is guarded, but backlog pressure is extreme and memory is normal, so one reserved P-core is loaned to the backlog pump"
            else:
                selected = min(max_budget, 3)
                mode = "guarded_backlog_probe_3"
                reason = "protect-live host pressure is guarded, but backlog pressure is extreme and memory is normal, so the third P-core pump stays active"
        elif host_saturation >= 85.0 or compute_pressure in {"blocked", "critical"} or throttle_profile == "protect_live":
            selected = min(max_budget, 2)
            mode = "host_pressure_relief_2"
            reason = "host pressure is high enough that backlog preprocessing must cool before widening again"
        else:
            selected = min(max_budget, 3)
            mode = "host_pressure_relief_3"
            reason = "host pressure is elevated, so backlog preprocessing is narrowed while the writer catches up"
    elif seventh_core_allowed or full_p_core_budget_requested:
        selected = min(max_budget, 7)
        mode = "full_p_core_budget_7_plus_primary_writer"
        reason = "full P-core backlog mode uses seven child/preprocess lanes plus the primary merge writer"
    elif burst_allowed:
        selected = min(max_budget, 6)
        mode = "burst_6"
        reason = "host is cool enough and backlog pressure can use a wider preprocessing burst"
    else:
        selected = min(max_budget, 5)
        mode = "daily_driver_5"
        reason = "normal daily-driver headroom with single-writer backlog acceleration"
    if bool(recent_eject.get("active")):
        cap = max(_safe_int(recent_eject.get("max_workers"), 3), 1)
        if selected > cap:
            recent_eject["previous_mode"] = mode
            recent_eject["previous_selected_workers"] = int(selected)
            selected = min(selected, cap)
            mode = f"storage_eject_cooldown_{int(cap)}"
            reason = "recent BOT_LOGS eject detected, so backlog preprocessing is temporarily capped while the external storage route proves stable"
    return {
        "mode": mode,
        "selected_workers": int(max(selected, 1)),
        "max_budget": int(max_budget),
        "reason": reason,
        "inputs": {
            "host_saturation_score": round(float(host_saturation), 3),
            "compute_pressure_level": compute_pressure,
            "memory_pressure_level": memory_pressure,
            "memory_pressure_kind": memory_pressure_kind,
            "swap_used_gb": round(float(swap_used_gb), 3),
            "compressed_store_gb": round(float(compressed_store_gb), 3),
            "compressor_gb": round(float(compressor_gb), 3),
            "compressed_pressure_gb": round(float(compressed_pressure_gb), 3),
            "pages_throttled": round(float(pages_throttled), 3),
            "throttle_profile": throttle_profile,
            "creative_session_level": creative_level,
            "creative_session_kind": creative_kind,
            "co_running_session_level": co_running_level,
            "primary_task": primary_task,
            "off_hours_active": bool(off_hours_active),
            "backlog_ratio": round(float(backlog_ratio), 3),
            "sparse_active": bool(sparse_active),
            "user_app_reserve_target_p_cores": int(user_reserve_target),
            "user_reserve_worker_cap": int(user_reserve_worker_cap),
            "elastic_reserve_loan_cap": int(elastic_reserve_loan_cap),
        },
        "storage_eject_cooldown": recent_eject,
        "user_app_reserve": {
            "target_p_cores": int(user_reserve_target),
            "worker_cap": int(user_reserve_worker_cap) if user_reserve_worker_cap else 0,
            "elastic_loan_allowed": bool(elastic_reserve_loan_allowed),
            "elastic_loan_worker_cap": int(elastic_reserve_loan_cap),
            "active": bool(user_reserve_target > 0),
            "policy": "operator_reserve_target_caps_backlog_preprocess_workers_before_burst_selection",
        },
        "seventh_core_burst": {
            "enabled": _env_enabled("BACKLOG_PCORE_ENABLE_SEVENTH", True),
            "allowed": bool(seventh_core_allowed),
            "base_budget": int(base_budget),
            "deep_host_cool": bool(deep_host_cool),
            "deep_memory_clear": bool(deep_memory_clear),
            "background_task_clear": bool(background_task_clear),
            "policy": "use_pcore_7_only_for_deep_green_background_backlog_bursts",
        },
        "protected_live_backlog_probe": {
            "allowed": bool(protected_backlog_probe_allowed),
            "wide_allowed": bool(protected_backlog_probe_wide_allowed),
            "max_workers": 4 if protected_backlog_probe_wide_allowed else 3,
            "standard_workers": 3,
            "wide_workers": 4,
            "host_saturation_ceiling": 66.0,
            "wide_host_saturation_ceiling": 60.0,
            "wide_swap_used_gb_ceiling": 3.0,
            "wide_compressed_store_gb_ceiling": 11.0,
            "requires_memory_below_elevated": True,
            "policy": "allow_one_or_two_extra_pcores_under_protect_live_only_for_severe_backlog_with_normal_memory",
        },
        "guarded_backlog_probe": {
            "allowed": bool(guarded_backlog_probe_allowed),
            "wide_allowed": bool(guarded_backlog_probe_wide_allowed),
            "max_workers": 4 if guarded_backlog_probe_wide_allowed else 3,
            "standard_workers": 3,
            "wide_workers": 4,
            "host_saturation_ceiling": 76.0,
            "compressed_pressure_gb_ceiling": 14.0,
            "wide_compressed_pressure_gb_ceiling": 9.0,
            "swap_used_gb_ceiling": 4.0,
            "policy": "keep_three_pcore_pump_active_under_guarded_protect_live; loan a fourth only when compressor, swap, and foreground pressure are clear",
        },
        "rules": {
            "memory_critical_relief": 2,
            "memory_elevated_relief": 3,
            "creative_or_foreground_pressure": 3,
            "protected_live_backlog_probe": "3-4",
            "host_pressure_relief": "2-3",
            "normal_daily_driver": 5,
            "cool_host_backlog_burst": 6,
            "deep_green_pcore7_burst": 7,
            "recent_storage_eject_cooldown": _safe_int(os.getenv("BACKLOG_RECENT_EJECT_MAX_WORKERS"), 3),
        },
    }


def _p_core_preprocess_workers(*, critical: bool, backpressure: dict[str, Any] | None = None, host_context: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    p_cores = max(_performance_core_target(), 1)
    foreground_reserve = _p_core_foreground_reserve(p_cores=p_cores, host_context=host_context)
    writer_reserve = 1
    pressure = backpressure if isinstance(backpressure, dict) else {}
    core = _safe_int(pressure.get("pending_lines"), 0)
    total = _safe_int(pressure.get("pending_lines_total"), core)
    oldest = _safe_float(pressure.get("oldest_pending_age_seconds"), 0.0)
    threshold = max(_safe_int(pressure.get("pending_lines_threshold"), 15000), 1)
    age_threshold = max(_safe_float(pressure.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    line_estimation = pressure.get("line_estimation") if isinstance(pressure.get("line_estimation"), dict) else {}
    backlog_ratio = max(core / max(threshold, 1), total / max(threshold, 1), oldest / age_threshold)
    sparse_active = bool(line_estimation.get("sparse_large_line_active", False))
    intelligence = _p_core_burst_intelligence(
        p_core_count=p_cores,
        foreground_reserve=foreground_reserve,
        writer_reserve=writer_reserve,
        critical=bool(critical),
        backlog_ratio=float(backlog_ratio),
        sparse_active=bool(sparse_active),
        host_context=host_context,
    )
    return _safe_int(intelligence.get("selected_workers"), 1), intelligence


def _p_core_backlog_allocation_contract(env: dict[str, str], sparse_pressure: dict[str, Any] | None = None) -> dict[str, Any]:
    sparse = sparse_pressure if isinstance(sparse_pressure, dict) else {}
    workers = _safe_int(env.get("BACKLOG_PCORE_PREPROCESS_WORKERS"), _safe_int(env.get("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 1))
    shard_writer_lanes = _safe_int(env.get("SQL_LINK_SERVICE_SHARD_WRITER_LANES"), workers)
    max_shard_writer_lanes = _safe_int(env.get("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), max(shard_writer_lanes, 1))
    return {
        "active": str(env.get("BACKLOG_PCORE_ALLOCATION_ACTIVE") or "") == "1",
        "policy": "p_core_preprocess_single_sql_writer",
        "sqlite_writer_count": 1,
        "primary_merge_writer_count": 1,
        "shard_link_writer_lanes": int(max(shard_writer_lanes, 1)),
        "max_shard_link_writer_lanes": int(max(max_shard_writer_lanes, 1)),
        "writer_lane_policy": "parallel_child_shard_writers_on_p_core_budget_single_serial_primary_merge",
        "preprocess_worker_budget": int(max(workers, 1)),
        "burst_worker_budget": int(max(workers, 1)),
        "reserve_policy": "adaptive_4_5_6_7_foreground_first",
        "p_core_burst_intelligence": {
            "mode": str(env.get("BACKLOG_PCORE_BURST_MODE") or ""),
            "selected_workers": int(max(workers, 1)),
            "reason": str(env.get("BACKLOG_PCORE_BURST_REASON") or ""),
            "rules": {
                "memory_critical_relief": 2,
                "memory_elevated_relief": 3,
                "creative_or_foreground_pressure": 3,
                "protected_live_backlog_probe": "3-4",
                "host_pressure_relief": "2-3",
                "normal_daily_driver": 5,
                "cool_host_backlog_burst": 6,
                "deep_green_pcore7_burst": 7,
            },
        },
        "single_writer_only": str(env.get("BACKLOG_DRAIN_SINGLE_WRITER_ONLY") or "") == "1",
        "avoid_background_taskpolicy": str(env.get("RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND") or "0") != "1",
        "performance_core_primary": str(env.get("BOT_CPU_ALLOCATION_POLICY") or "performance_core_primary").startswith("performance_core"),
        "training_pcore_gate": {
            "allowed_when_backlog_green": str(env.get("TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN") or "") == "1",
            "max_workers": _safe_int(env.get("TRAINING_PCORE_MAX_WORKERS"), 1),
            "nice_target": _safe_int(env.get("TRAINING_PCORE_NICE"), 8),
        },
        "sparse_huge_jsonl": {
            "active": bool(sparse.get("active", False)),
            "estimated_pending_bytes": _safe_int(sparse.get("estimated_pending_bytes"), 0),
            "max_bytes_per_file": _safe_int(env.get("INGEST_MAX_BYTES_PER_FILE"), 0),
            "sqlite_batch_max_bytes": _safe_int(env.get("SQLITE_BATCH_MAX_BYTES"), 0),
        },
        "control_env": {
            key: str(value)
            for key, value in env.items()
            if key
            in {
                "BACKLOG_PCORE_ALLOCATION_ACTIVE",
                "BACKLOG_DRAIN_SINGLE_WRITER_ONLY",
                "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY",
                "BACKLOG_PCORE_PREPROCESS_WORKERS",
                "BACKLOG_PCORE_BURST_MODE",
                "BACKLOG_PCORE_BURST_REASON",
                "BACKLOG_MEMORY_PRESSURE_CORE_OPTIMIZER",
                "BACKLOG_ACCELERATOR_ENABLED",
                "BACKLOG_ACCELERATOR_MODE",
                "BACKLOG_ACCELERATOR_PREPROCESS_WORKERS",
                "BACKLOG_CATCH_UP_WAVE_LIMIT",
                "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE",
                "SQL_LINK_SERVICE_PREPROCESS_WORKERS",
                "SQL_LINK_SERVICE_SHARD_WRITER_LANES",
                "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE",
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS",
                "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS",
                "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS",
                "SQL_LINK_SERVICE_SKIP_IDLE_SENTINELS",
                "SQL_LINK_WRITER_BACKGROUND_POLICY",
                "SQL_LINK_WRITER_NICE",
                "SQL_LINK_CHILD_WRITER_CPU_POLICY",
                "BOT_CPU_ALLOCATION_POLICY",
                "BOT_CPU_QOS_POLICY",
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO",
                "RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND",
                "RUNTIME_THROTTLE_RESEARCH_NICE",
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG",
                "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN",
                "TRAINING_PCORE_MAX_WORKERS",
                "TRAINING_PCORE_NICE",
                "WRITER_CYCLE_MAX_CATCH_UP_WAVES",
            }
        },
    }


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


def _pending_source_row(source_rel: str, row: dict[str, Any], *, pending_lines: int, age_seconds: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source_rel": source_rel,
        "pending_lines": int(pending_lines),
        "oldest_pending_age_seconds": round(age_seconds, 3),
    }
    for key in (
        "shard",
        "stream",
        "pressure_lane",
        "storage_temperature",
        "ingestion_lane",
        "file_size_bytes",
        "estimated_avg_bytes_per_line",
        "estimated_pending_bytes",
        "sample_bytes",
        "sample_newlines",
        "line_estimate_method",
        "sparse_large_line",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def _merge_pending_source_metadata(current: dict[str, Any], row: dict[str, Any]) -> None:
    for key in (
        "shard",
        "stream",
        "pressure_lane",
        "storage_temperature",
        "ingestion_lane",
        "file_size_bytes",
        "estimated_avg_bytes_per_line",
        "estimated_pending_bytes",
        "sample_bytes",
        "sample_newlines",
        "line_estimate_method",
        "sparse_large_line",
    ):
        if key in row and key not in current:
            current[key] = row.get(key)
    if bool(row.get("sparse_large_line", False)):
        current["sparse_large_line"] = True
    current["file_size_bytes"] = max(_safe_int(current.get("file_size_bytes"), 0), _safe_int(row.get("file_size_bytes"), 0))
    current["estimated_avg_bytes_per_line"] = max(
        _safe_float(current.get("estimated_avg_bytes_per_line"), 0.0),
        _safe_float(row.get("estimated_avg_bytes_per_line"), 0.0),
    )


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
                by_source[source_rel] = _pending_source_row(source_rel, row, pending_lines=pending_lines, age_seconds=age_seconds)
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(
                max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds),
                3,
            )
            _merge_pending_source_metadata(current, row)
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
                by_source[source_rel] = _pending_source_row(source_rel, row, pending_lines=pending_lines, age_seconds=age_seconds)
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(
                max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds),
                3,
            )
            _merge_pending_source_metadata(current, row)
    return sorted(
        by_source.values(),
        key=lambda row: (_safe_int(row.get("pending_lines"), 0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0)),
        reverse=True,
    )


def _stale_decision_rows_from_storage_control(storage_control: dict[str, Any]) -> list[dict[str, Any]]:
    locator = storage_control.get("stale_pending_locator") if isinstance(storage_control.get("stale_pending_locator"), dict) else {}
    rows = locator.get("oldest_sources") if isinstance(locator.get("oldest_sources"), list) else []
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not (source_rel.startswith("decisions/") or source_rel.startswith("governance/channels/decision/")):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
        if pending_lines <= 0 or age_seconds <= 0.0:
            continue
        current = by_source.get(source_rel)
        if current is None:
            by_source[source_rel] = _pending_source_row(source_rel, row, pending_lines=pending_lines, age_seconds=age_seconds)
        else:
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds), 3)
            _merge_pending_source_metadata(current, row)
    return sorted(
        by_source.values(),
        key=lambda row: (
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
            _safe_int(row.get("pending_lines"), 0),
        ),
        reverse=True,
    )


def _stale_decision_rows_from_backpressure(backpressure: dict[str, Any]) -> list[dict[str, Any]]:
    age_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS), 1.0)
    rows = backpressure.get("top_pending_files") if isinstance(backpressure.get("top_pending_files"), list) else []
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not (source_rel.startswith("decisions/") or source_rel.startswith("governance/channels/decision/")):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
        if pending_lines <= 0 or age_seconds < age_threshold:
            continue
        current = by_source.get(source_rel)
        if current is None:
            by_source[source_rel] = _pending_source_row(source_rel, row, pending_lines=pending_lines, age_seconds=age_seconds)
        else:
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds), 3)
            _merge_pending_source_metadata(current, row)
    return sorted(
        by_source.values(),
        key=lambda row: (
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
            _safe_int(row.get("pending_lines"), 0),
        ),
        reverse=True,
    )


def _overlay_rows_from_storage_control(storage_control: dict[str, Any], prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    overlay = storage_control.get("sql_ingestion_pending_overlay") if isinstance(storage_control.get("sql_ingestion_pending_overlay"), dict) else {}
    rows = overlay.get("top_pending_files") if isinstance(overlay.get("top_pending_files"), list) else []
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel or not source_rel.startswith(prefixes):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        if pending_lines <= 0:
            continue
        age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
        current = by_source.get(source_rel)
        if current is None:
            by_source[source_rel] = _pending_source_row(source_rel, row, pending_lines=pending_lines, age_seconds=age_seconds)
            continue
        current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
        current["oldest_pending_age_seconds"] = round(max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds), 3)
        _merge_pending_source_metadata(current, row)
    return sorted(
        by_source.values(),
        key=lambda row: (_safe_int(row.get("pending_lines"), 0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0)),
        reverse=True,
    )


def _storage_overlay_authoritative(storage_control: dict[str, Any]) -> bool:
    overlay = storage_control.get("sql_ingestion_pending_overlay") if isinstance(storage_control.get("sql_ingestion_pending_overlay"), dict) else {}
    if not bool(overlay.get("active", False)) or not bool(overlay.get("used_for_pressure", False)):
        return False
    if _safe_int(overlay.get("source_count"), 0) <= 0 or _safe_int(overlay.get("fresh_source_count"), 0) <= 0:
        return False
    if _safe_int(overlay.get("stale_source_count"), 0) > 0:
        return False
    truth = storage_control.get("backlog_truth") if isinstance(storage_control.get("backlog_truth"), dict) else {}
    mode = str(truth.get("authoritative_mode") or "").strip()
    if mode and not mode.startswith("overlay"):
        return False
    decay = truth.get("overlay_decay") if isinstance(truth.get("overlay_decay"), dict) else {}
    if bool(decay.get("should_decay", False)):
        return False
    attribution_ratio = _safe_float(decay.get("attribution_ratio"), 1.0)
    return attribution_ratio >= 0.95


def _storage_raw_live_backpressure(storage_control: dict[str, Any]) -> dict[str, Any]:
    storage_backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    for key in ("raw_live", "effective_raw_live"):
        candidate = storage_backpressure.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _storage_overlay_freshly_covers_prefixes(storage_control: dict[str, Any], prefixes: tuple[str, ...]) -> bool:
    if not _storage_overlay_authoritative(storage_control):
        return False
    overlay = storage_control.get("sql_ingestion_pending_overlay") if isinstance(storage_control.get("sql_ingestion_pending_overlay"), dict) else {}
    fresh_path_contains = overlay.get("fresh_path_contains") if isinstance(overlay.get("fresh_path_contains"), list) else []
    fresh_markers = [str(item or "").strip().lower() for item in fresh_path_contains if str(item or "").strip()]
    if not fresh_markers:
        return False
    clean_prefixes = [str(prefix or "").strip().lower() for prefix in prefixes if str(prefix or "").strip()]
    if not clean_prefixes:
        return False
    for prefix in clean_prefixes:
        if any(marker.startswith(prefix) or prefix.startswith(marker) for marker in fresh_markers):
            return True
    return False


def _preferred_source_rows(
    storage_control: dict[str, Any],
    backpressure: dict[str, Any],
    prefixes: tuple[str, ...],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    overlay_rows = _overlay_rows_from_storage_control(storage_control, prefixes)
    storage_raw_rows = _collect_sources(_storage_raw_live_backpressure(storage_control), prefixes, keys=keys)
    if not overlay_rows and _storage_overlay_freshly_covers_prefixes(storage_control, prefixes):
        return storage_raw_rows
    raw_rows = _merge_source_rows(_collect_sources(backpressure, prefixes, keys=keys), storage_raw_rows)
    if tuple(keys) == ("top_pending_files",) and overlay_rows:
        raw_core_pending = max(_safe_int(backpressure.get("pending_lines"), 0), 0)
        overlay_pending = sum(max(_safe_int(row.get("pending_lines"), 0), 0) for row in overlay_rows)
        if raw_core_pending < overlay_pending <= CORE_HARD_PENDING_LINES:
            return raw_rows
    if _storage_overlay_authoritative(storage_control) and overlay_rows:
        return overlay_rows
    return _merge_source_rows(overlay_rows, raw_rows)


def _merge_source_rows(*row_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            source_rel = str(row.get("source_rel") or "").strip()
            if not source_rel:
                continue
            pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
            age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
            current = by_source.get(source_rel)
            if current is None:
                by_source[source_rel] = _pending_source_row(source_rel, row, pending_lines=pending_lines, age_seconds=age_seconds)
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds), 3)
            _merge_pending_source_metadata(current, row)
    return sorted(
        by_source.values(),
        key=lambda row: (_safe_int(row.get("pending_lines"), 0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0)),
        reverse=True,
    )


def _stale_sticky_decision_rows(core_rows: list[dict[str, Any]], stale_decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stale_sources = {str(row.get("source_rel") or "").strip() for row in stale_decision_rows if str(row.get("source_rel") or "").strip()}
    if not stale_sources:
        return core_rows
    return [
        *[row for row in stale_decision_rows if str(row.get("source_rel") or "").strip()],
        *[row for row in core_rows if str(row.get("source_rel") or "").strip() not in stale_sources],
    ]


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


def _sparse_large_line_pressure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_rows: list[dict[str, Any]] = []
    for row in rows:
        avg_bytes = _safe_float(row.get("estimated_avg_bytes_per_line"), 0.0)
        if bool(row.get("sparse_large_line", False)) or avg_bytes >= SPARSE_LARGE_LINE_BYTES_PER_LINE:
            sparse_rows.append(row)
    pending_lines = sum(max(_safe_int(row.get("pending_lines"), 0), 0) for row in sparse_rows)
    file_bytes = sum(max(_safe_int(row.get("file_size_bytes"), 0), 0) for row in sparse_rows)
    pending_bytes = sum(max(_safe_int(row.get("estimated_pending_bytes"), 0), 0) for row in sparse_rows)
    top = sorted(
        sparse_rows,
        key=lambda row: (
            max(_safe_int(row.get("file_size_bytes"), 0), 0),
            max(_safe_int(row.get("pending_lines"), 0), 0),
        ),
        reverse=True,
    )
    return {
        "active": bool(sparse_rows),
        "file_count": len(sparse_rows),
        "pending_lines": int(pending_lines),
        "file_size_bytes": int(file_bytes),
        "estimated_pending_bytes": int(pending_bytes),
        "top_files": [
            {
                "source_rel": str(row.get("source_rel") or ""),
                "pending_lines": _safe_int(row.get("pending_lines"), 0),
                "file_size_bytes": _safe_int(row.get("file_size_bytes"), 0),
                "estimated_pending_bytes": _safe_int(row.get("estimated_pending_bytes"), 0),
                "estimated_avg_bytes_per_line": round(_safe_float(row.get("estimated_avg_bytes_per_line"), 0.0), 3),
            }
            for row in top[:5]
        ],
    }


def _base_env(*, critical: bool, backpressure: dict[str, Any] | None = None, host_context: dict[str, Any] | None = None) -> dict[str, str]:
    p_core_workers, burst_intelligence = _p_core_preprocess_workers(
        critical=critical,
        backpressure=backpressure,
        host_context=host_context,
    )
    max_shard_writer_lanes = max(1, min(_performance_core_target(), 8))
    training_workers = max(1, min(2, p_core_workers // 2 or 1))
    nice_target = str(_safe_int(os.getenv("SLEEVE_NICE_SPECIALIZED"), 8))
    user_reserve = (
        burst_intelligence.get("user_app_reserve")
        if isinstance(burst_intelligence.get("user_app_reserve"), dict)
        else {}
    )
    bp = backpressure if isinstance(backpressure, dict) else {}
    raw_live = bp.get("raw_live") if isinstance(bp.get("raw_live"), dict) else {}
    line_estimation = raw_live.get("line_estimation") if isinstance(raw_live.get("line_estimation"), dict) else bp.get("line_estimation") if isinstance(bp.get("line_estimation"), dict) else {}
    pending_threshold = max(_safe_int(bp.get("pending_lines_threshold"), 15000), 1)
    oldest_threshold = max(_safe_float(bp.get("oldest_age_threshold_seconds"), DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS), 1.0)
    core_pending = _safe_int(bp.get("core_pending_lines"), _safe_int(raw_live.get("core_pending_lines"), 0))
    total_pending = _safe_int(bp.get("total_pending_lines"), _safe_int(raw_live.get("total_pending_lines"), 0))
    oldest_age = _safe_float(bp.get("oldest_pending_age_seconds"), _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0))
    sparse_active = bool(line_estimation.get("sparse_large_line_active", False))
    sparse_pending_bytes = _safe_int(line_estimation.get("sparse_large_line_pending_bytes"), 0)
    backlog_ratio = max(
        core_pending / float(pending_threshold),
        total_pending / float(max(pending_threshold * 2, pending_threshold)),
        oldest_age / float(oldest_threshold),
    )
    accelerator_wave_limit = (
        6
        if critical and p_core_workers >= 4 and (sparse_active or sparse_pending_bytes >= 64 * 1024 * 1024 or backlog_ratio >= 20.0)
        else 5
        if critical and p_core_workers >= 4
        else 3
        if critical
        else 2
    )
    accelerator_merge_seconds = 150 if accelerator_wave_limit >= 6 else 120 if accelerator_wave_limit >= 5 else 90 if critical else 60
    accelerator_shard_timeout = 480 if accelerator_wave_limit >= 6 else 420 if critical else 150
    return {
        "INGEST_MAX_DEFERRED_FILES": "6" if critical else "4",
        "JSONL_SQL_MAX_COLD_LANE_FILES": "2" if critical else "1",
        "INGEST_TOP_PENDING_FILES": "24" if critical else "16",
        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
        "BACKLOG_DRAIN_SINGLE_WRITER_ONLY": "1",
        "BACKLOG_PCORE_PREPROCESS_WORKERS": str(p_core_workers),
        "BACKLOG_PCORE_USER_APP_RESERVE_TARGET": str(_safe_int(user_reserve.get("target_p_cores"), 0)),
        "BACKLOG_PCORE_BURST_MODE": str(burst_intelligence.get("mode") or ""),
        "BACKLOG_PCORE_BURST_REASON": str(burst_intelligence.get("reason") or ""),
        "BACKLOG_MEMORY_PRESSURE_CORE_OPTIMIZER": "1"
        if str(burst_intelligence.get("mode") or "").startswith("memory_relief")
        else "0",
        "BOT_CPU_ALLOCATION_POLICY": "performance_core_primary",
        "BOT_CPU_QOS_POLICY": "performance_core_primary_no_background_writer",
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.20" if critical else "0.30",
        "LOG_DATA_INGRESS": "0",
        "LOG_API_CALLS": "0",
        "LOG_LOOP_STATE": "0",
        "LOG_SHADOW_PNL_ATTRIBUTION": "0",
        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": str(p_core_workers),
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": str(p_core_workers),
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": str(max_shard_writer_lanes),
        "SQL_LINK_CHILD_WRITER_CPU_POLICY": "performance_core_primary",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": "0",
        "SQL_LINK_WRITER_NICE": "0",
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12" if critical else "15",
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": str(accelerator_shard_timeout),
        "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS": "1",
        "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS": "120" if critical else "90",
        "SQL_LINK_SERVICE_SKIP_IDLE_SENTINELS": "0",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": str(accelerator_merge_seconds),
        "SQL_LINK_SERVICE_CATCH_UP_WAVE": "1",
        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": str(accelerator_wave_limit),
        "BACKLOG_ACCELERATOR_ENABLED": "1",
        "BACKLOG_ACCELERATOR_MODE": "drainer_pcore_wave_6" if accelerator_wave_limit >= 6 else "drainer_pcore_wave_5" if accelerator_wave_limit >= 5 else "drainer_bounded_wave",
        "BACKLOG_ACCELERATOR_PREPROCESS_WORKERS": str(p_core_workers),
        "BACKLOG_CATCH_UP_WAVE_LIMIT": str(accelerator_wave_limit),
        "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": str(accelerator_merge_seconds),
        "SQL_LINK_SERVICE_SQLITE_TIMEOUT": "420" if critical else "300",
        "SQL_LINK_SERVICE_LOCK_RETRIES": "360" if critical else "240",
        "SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS": "0.35",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000" if critical else "200000",
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": "2400000" if critical else "1800000",
        "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
        "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
        "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.25" if critical else "0.5",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.25" if critical else "0.5",
        "SQLITE_CACHE_SIZE_KB": "32768" if critical else "24576",
        "SQLITE_MMAP_SIZE_MB": "0",
        "SQLITE_ALLOW_MMAP": "0",
        "SQLITE_WAL_AUTOCHECKPOINT_PAGES": "4000",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB": "8192" if critical else "6144",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB": "0",
        "BOT_OPS_SQLITE_ALLOW_MMAP": "0",
        "BOT_OPS_SQLITE_BUSY_TIMEOUT_MS": "420000" if critical else "300000",
        "RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND": "0",
        "RUNTIME_THROTTLE_RESEARCH_NICE": nice_target,
        "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG": "1",
        "HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG": "1" if critical else "0",
        "REPORT_REFRESH_PAUSED_FOR_BACKLOG": "1" if critical else "0",
        "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN": "1",
        "TRAINING_PCORE_MAX_WORKERS": str(training_workers),
        "TRAINING_PCORE_NICE": nice_target,
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
    sparse_pressure = _sparse_large_line_pressure(rows)
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
        "sparse_large_line_pressure": sparse_pressure,
        "live_window_safe": bool(live_window_safe),
        "p_core_backlog_allocation_contract": _p_core_backlog_allocation_contract(env, sparse_pressure),
        "self_accommodation": _profile_self_accommodation(
            live_window_safe=bool(live_window_safe),
            min_pending_lines=int(max(min_pending_lines, 1)),
            stale_ready_age_seconds=float(stale_ready_age_seconds),
            env=env,
        ),
        "env_overrides": env,
    }


def _raw_live_expansion_guard(
    backpressure: dict[str, Any],
    *,
    storage_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    storage = storage_control if isinstance(storage_control, dict) else {}
    existing = storage.get("raw_live_expansion_contract") if isinstance(storage.get("raw_live_expansion_contract"), dict) else {}
    target_core = max(_safe_int(os.getenv("BACKPRESSURE_TARGET_CORE_PENDING_LINES"), 5_000), 1)
    reserve_core = max(
        _safe_int(
            os.getenv("RAW_LIVE_EXPANSION_CORE_RESERVE_TARGET")
            or os.getenv("RAW_LIVE_CORE_RESERVE_TARGET"),
            _safe_int((existing.get("targets") or {}).get("core_reserve_lines"), int(target_core * 0.80))
            if isinstance(existing.get("targets"), dict)
            else int(target_core * 0.80),
        ),
        1,
    )
    reserve_total = max(
        _safe_int(
            os.getenv("RAW_LIVE_EXPANSION_TOTAL_RESERVE_TARGET")
            or os.getenv("RAW_LIVE_TOTAL_RESERVE_TARGET"),
            _safe_int((existing.get("targets") or {}).get("total_reserve_lines"), int(target_core * 1.10))
            if isinstance(existing.get("targets"), dict)
            else int(target_core * 1.10),
        ),
        reserve_core,
    )
    age_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS), 1.0)
    reserve_age = max(
        _safe_float(
            os.getenv("RAW_LIVE_EXPANSION_AGE_RESERVE_SECONDS")
            or os.getenv("RAW_LIVE_AGE_RESERVE_SECONDS"),
            _safe_float((existing.get("targets") or {}).get("oldest_age_reserve_seconds"), float(age_threshold) * 0.75)
            if isinstance(existing.get("targets"), dict)
            else float(age_threshold) * 0.75,
        ),
        30.0,
    )
    hot_source_markers = (
        "governance/channels/decision/",
        "decisions/",
        "governance/events/signal_generation_",
        "paper_trades",
        "exports/paper_broker_bridge/",
        "governance/channels/api/",
        "governance/channels/ingress/",
        "governance/channels/runtime/",
        "governance/channels/risk/",
        "governance/watchdog/",
        "governance/events/channel_schema_violations_",
        "governance/events/signal_generation_",
        "governance/events/auth_events_",
        "governance/events/execution_lane_stale_skips_",
        "governance/events/live_execution_guard_",
        "governance/events/premarket_token_guard_",
        "governance/events/write_failures_",
        "governance/events/paper_execution_guard_",
        "governance/distillation/teacher_student_events_",
        "governance/health/adaptive_regression_guard_feedback",
        "governance/health/infrabot_adaptive_feedback",
        "governance/training_diagnostics/requalification_queue",
        "live_orders",
    )
    hot_rows_by_source: dict[str, dict[str, Any]] = {}
    storage_raw_live = _storage_raw_live_backpressure(storage)
    for source_payload in (backpressure, storage_raw_live):
        for key in ("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"):
            rows = source_payload.get(key) if isinstance(source_payload.get(key), list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rel = str(row.get("source_rel") or "")
                if any(marker in rel for marker in hot_source_markers):
                    pending = _safe_int(row.get("pending_lines"), 0)
                    if pending > 0:
                        current = hot_rows_by_source.setdefault(rel, {"pending_lines": 0, "oldest_pending_age_seconds": 0.0})
                        current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending)
                        current["oldest_pending_age_seconds"] = max(
                            _safe_float(current.get("oldest_pending_age_seconds"), 0.0),
                            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
                        )
    source_core_pending = sum(_safe_int(row.get("pending_lines"), 0) for row in hot_rows_by_source.values())
    source_core_oldest = max([_safe_float(row.get("oldest_pending_age_seconds"), 0.0) for row in hot_rows_by_source.values()] or [0.0])
    canonical_raw_core = max(
        _safe_int(backpressure.get("pending_lines"), 0),
        _safe_int(storage_raw_live.get("core_pending_lines"), 0),
    )
    raw_core = max(canonical_raw_core, source_core_pending)
    raw_total = max(_safe_int(backpressure.get("pending_lines_total"), raw_core), _safe_int(storage_raw_live.get("total_pending_lines"), 0))
    raw_oldest = max(
        _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0),
        _safe_float(storage_raw_live.get("oldest_pending_age_seconds"), 0.0),
    )
    hot_guard_pending = max(raw_core, source_core_pending)
    raw_hot_material = bool(hot_guard_pending >= reserve_core)
    raw_hot_age_material = bool(
        existing.get("active", False)
        and source_core_pending >= MIN_MATERIAL_PENDING_LINES
        and source_core_oldest >= reserve_age
    )
    guard_total = max(source_core_pending, raw_core if raw_hot_material else 0)
    guard_oldest = max(source_core_oldest if raw_hot_age_material else 0.0, raw_oldest if raw_hot_material else 0.0)
    core_ratio = raw_core / max(float(reserve_core), 1.0)
    total_ratio = guard_total / max(float(reserve_total), 1.0) if guard_total > 0 else 0.0
    age_ratio = guard_oldest / max(float(reserve_age), 1.0) if raw_hot_material or raw_hot_age_material else 0.0
    pressure_ratio = max(core_ratio, total_ratio, age_ratio)
    active = bool(pressure_ratio > 1.0)
    live_priority_bonus = int(min(max(110_000.0 + (pressure_ratio - 1.0) * 180_000.0, 0.0), 450_000.0)) if active else 0
    cold_stage_penalty = int(min(max(40_000.0 + (pressure_ratio - 1.0) * 90_000.0, 0.0), 260_000.0)) if active else 0
    return {
        "active": active,
        "pressure_ratio": round(float(pressure_ratio), 3),
        "ratios": {
            "core": round(float(core_ratio), 3),
            "total": round(float(total_ratio), 3),
            "oldest_age": round(float(age_ratio), 3),
        },
        "raw_live": {
            "canonical_core_pending_lines": int(canonical_raw_core),
            "core_pending_lines": int(raw_core),
            "total_pending_lines": int(raw_total),
            "oldest_pending_age_seconds": round(float(raw_oldest), 3),
            "hot_source_pending_lines": int(source_core_pending),
            "hot_source_oldest_pending_age_seconds": round(float(source_core_oldest), 3),
            "guard_total_pending_lines": int(guard_total),
            "guard_oldest_pending_age_seconds": round(float(guard_oldest), 3),
            "excluded_deferred_or_cold_pending_lines": int(max(raw_total - guard_total, 0)),
        },
        "targets": {
            "core_reserve_lines": int(reserve_core),
            "total_reserve_lines": int(reserve_total),
            "oldest_age_reserve_seconds": round(float(reserve_age), 3),
        },
        "live_priority_bonus": int(live_priority_bonus),
        "cold_stage_penalty": int(cold_stage_penalty),
        "policy": "reserve_one_hot_raw_live_handoff_before_cold_overlay_tails_when_expansion_headroom_is_tight",
    }


def _apply_age_pressure_priority(
    profiles: list[dict[str, Any]],
    backpressure: dict[str, Any],
    *,
    raw_live_guard: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    threshold_seconds = max(
        _safe_float(backpressure.get("oldest_age_threshold_seconds"), DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS),
        1.0,
    )
    guard = raw_live_guard if isinstance(raw_live_guard, dict) else {}
    guard_active = bool(guard.get("active", False))
    guard_pressure_ratio = _safe_float(guard.get("pressure_ratio"), 0.0)
    preemption_active = bool(guard_active and guard_pressure_ratio >= 2.0)
    live_priority_bonus = _safe_int(guard.get("live_priority_bonus"), 0) if guard_active else 0
    cold_stage_penalty = _safe_int(guard.get("cold_stage_penalty"), 0) if guard_active else 0
    guard_raw_live = guard.get("raw_live") if isinstance(guard.get("raw_live"), dict) else {}
    canonical_core_pending = _safe_int(guard_raw_live.get("canonical_core_pending_lines"), _safe_int(guard_raw_live.get("core_pending_lines"), 0))
    for row in profiles:
        pending_lines = _safe_int(row.get("pending_lines"), 0)
        min_pending_lines = _safe_int(row.get("min_pending_lines"), MIN_MATERIAL_PENDING_LINES)
        oldest_age = _safe_float(row.get("oldest_pending_age_seconds"), 0.0)
        readiness_reason = str(row.get("readiness_reason") or "")
        live_window_safe = bool(row.get("live_window_safe", False))
        name = str(row.get("name") or "")
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
        dominant_support_pressure = bool(
            guard_active
            and name in RAW_LIVE_EXPANSION_SUPPORT_DRAINERS
            and pending_lines >= CORE_HARD_PENDING_LINES
            and canonical_core_pending < CORE_HARD_PENDING_LINES
        )
        raw_bonus = int(live_priority_bonus if guard_active and (name in RAW_LIVE_EXPANSION_HOT_DRAINERS or dominant_support_pressure) and pending_lines > 0 else 0)
        raw_size_bonus = (
            int(min(max(float(pending_lines) * 4.0, 0.0), 220_000.0))
            if guard_active and (name in RAW_LIVE_EXPANSION_HOT_DRAINERS or dominant_support_pressure) and pending_lines > 0
            else 0
        )
        cold_penalty = int(cold_stage_penalty if guard_active and name == "cold_stage_drainer" else 0)
        if preemption_active and (name in RAW_LIVE_EXPANSION_HOT_DRAINERS or dominant_support_pressure) and pending_lines > 0:
            preemption_tier = 3
        elif preemption_active and name in RAW_LIVE_EXPANSION_SUPPORT_DRAINERS and pending_lines > 0:
            preemption_tier = 1
        elif preemption_active and live_window_safe and pending_lines > 0:
            preemption_tier = 2
        else:
            preemption_tier = 0
        row["age_pressure_priority_bonus"] = bonus
        row["raw_live_expansion_guard_active"] = guard_active
        row["raw_live_expansion_preemption_active"] = preemption_active
        row["raw_live_expansion_preemption_tier"] = preemption_tier
        row["raw_live_expansion_priority_bonus"] = raw_bonus
        row["raw_live_expansion_size_priority_bonus"] = raw_size_bonus
        row["raw_live_expansion_cold_penalty"] = cold_penalty
        row["effective_priority_score"] = int(priority_score + bonus + raw_bonus + raw_size_bonus - cold_penalty)
        if guard_active and (raw_bonus or raw_size_bonus or cold_penalty):
            env = row.get("env_overrides") if isinstance(row.get("env_overrides"), dict) else {}
            env.update(
                {
                    "RAW_LIVE_EXPANSION_GUARD_ACTIVE": "1",
                    "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST": "1",
                    "SQL_LINK_SERVICE_RAW_LIVE_SIZE_PRIORITY_BOOST": str(raw_size_bonus),
                    "SQL_LINK_SERVICE_RAW_LIVE_RESERVE_WAVE": "1",
                    "SQL_LINK_SERVICE_COLD_STAGE_YIELDS_TO_RAW_LIVE": "1",
                }
            )
            row["env_overrides"] = env
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


def _is_explanation_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return bool(
        rel.startswith("decision_explanations/")
        or rel.startswith("data/stale_stage/decision_explanations/")
        or "/decision_explanations/" in rel
    )


def _is_runtime_or_loop_state_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return bool(
        rel.startswith("governance/channels/loop_state/")
        or rel.startswith("governance/channels/runtime/")
    )


def _is_core_signal_source(source_rel: str) -> bool:
    return str(source_rel or "").startswith("governance/events/signal_generation_")


def _decision_drainer_env(base: dict[str, str], core_rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    concentration = _concentration_summary(core_rows)
    concentrated = bool(concentration.get("concentrated", False))
    sparse_pressure = _sparse_large_line_pressure(core_rows)
    sparse_large_line_active = bool(sparse_pressure.get("active", False))
    regular_focus: list[str] = []
    regular_focus_shards: set[str] = set()
    aggressive_focus: list[str] = []
    crypto_focus: list[str] = []
    signal_focus: list[str] = []
    sparse_regular_focus = False
    sparse_aggressive_focus = False
    sparse_crypto_focus = False
    for row in core_rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        row_shard = str(row.get("shard") or "").strip()
        row_sparse = bool(row.get("sparse_large_line", False))
        if _is_core_signal_source(source_rel):
            signal_focus.append(source_rel)
        elif _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
            sparse_crypto_focus = sparse_crypto_focus or row_sparse
        elif _is_aggressive_decision_source(source_rel):
            aggressive_focus.append(source_rel)
            sparse_aggressive_focus = sparse_aggressive_focus or row_sparse
        else:
            regular_focus.append(source_rel)
            if row_shard:
                regular_focus_shards.add(row_shard)
            sparse_regular_focus = sparse_regular_focus or row_sparse

    shards: list[str] = []
    env: dict[str, str] = {**base}
    regular_focus_has_sparse_rows = bool(sparse_regular_focus)
    aggressive_focus_has_sparse_rows = bool(sparse_aggressive_focus)
    if concentrated or sparse_large_line_active:
        env.update(
            {
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1" if concentrated else "0",
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_TOP1_SHARE": str(concentration.get("top1_share", 0.0)),
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_TOP3_SHARE": str(concentration.get("top3_share", 0.0)),
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "90",
                "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS": "32000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS": "24000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS": "32000",
            }
        )
    if concentrated and not sparse_large_line_active:
        env.update(
            {
                "INGEST_MAX_BYTES_PER_FILE": "536870912",
                "SQLITE_BATCH_MAX_BYTES": "134217728",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "180",
                "SQL_LINK_SERVICE_STALE_DECISION_BYTE_WINDOW_BOOST": "1",
            }
        )
    if sparse_large_line_active:
        env.update(
            {
                "SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN": "1",
                "SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_FILE_COUNT": str(sparse_pressure.get("file_count", 0)),
                "INGEST_MAX_BYTES_PER_FILE": str(SPARSE_LARGE_DECISION_MAX_BYTES_PER_FILE),
                "SQLITE_BATCH_MAX_BYTES": str(SPARSE_LARGE_DECISION_SQLITE_BATCH_MAX_BYTES),
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "180",
            }
        )
        if sparse_regular_focus:
            env.update(
                {
                    "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": str(SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                    "SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS": str(SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                }
            )
        if sparse_aggressive_focus:
            env.update(
                {
                    "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": str(SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                    "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS": str(SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                }
            )
        if sparse_crypto_focus:
            env.update(
                {
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": str(SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS": str(SPARSE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                }
            )
    if regular_focus:
        shards.append("trading")
        env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES"] = "8" if regular_focus_has_sparse_rows else "16"
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] = (
            "12000"
            if regular_focus_has_sparse_rows
            else ("24000" if concentrated else "64000")
        )
        # Keep the companion aggressive shard in the handoff for mixed equity decision pressure.
        # Some aggressive sleeves write through regular decision-channel paths rather than
        # shadow_aggressive-prefixed files, so this preserves the broader hot-lane sweep.
        shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "8" if aggressive_focus_has_sparse_rows else "14"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = (
            "12000"
            if aggressive_focus_has_sparse_rows
            else ("12000" if concentrated else "24000")
        )
        if "crypto_trading" in regular_focus_shards and crypto_focus:
            shards.append("crypto_trading")
            env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
            env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES"] = "14"
            env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE"] = "64000"
    if aggressive_focus:
        if "aggressive_trading" not in shards:
            shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"] = ",".join(aggressive_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "8" if aggressive_focus_has_sparse_rows else "14"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = (
            "12000"
            if aggressive_focus_has_sparse_rows
            else ("12000" if concentrated else "24000")
        )
    if crypto_focus:
        shards.append("crypto_trading")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE"] = "64000"
    if signal_focus:
        shards.append("governance")
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"] = ",".join(signal_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"] = "8"
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE"] = str(SIGNAL_GENERATION_CATCH_UP_MAX_LINES)
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_STATE_CHECKPOINT_LINES"] = "8000"
        env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MERGE_MAX_JSONL_ROWS"] = str(SIGNAL_GENERATION_CATCH_UP_MERGE_MAX_JSONL_ROWS)
        env["INGEST_MAX_BYTES_PER_FILE"] = str(SIGNAL_GENERATION_CATCH_UP_MAX_BYTES_PER_FILE)
        env["SQLITE_BATCH_MAX_BYTES"] = str(SIGNAL_GENERATION_CATCH_UP_SQLITE_BATCH_MAX_BYTES)
        env["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] = "240"

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
    explanation_focus: list[str] = []
    crypto_explanation_focus: list[str] = []
    for row in rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_explanation_source(source_rel):
            if _is_crypto_decision_source(source_rel):
                crypto_explanation_focus.append(source_rel)
            else:
                explanation_focus.append(source_rel)
        elif _is_crypto_decision_source(source_rel):
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
    if explanation_focus:
        shards.append("explanations")
        env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS"] = ",".join(explanation_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_LINES_PER_FILE"] = "64000" if critical else "24000"
        env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_STATE_CHECKPOINT_LINES"] = "2000"
    if crypto_explanation_focus:
        shards.append("crypto_explanations")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS"] = ",".join(crypto_explanation_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES"] = "12" if critical else "8"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE"] = "64000" if critical else "24000"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_STATE_CHECKPOINT_LINES"] = "2000"
    if not shards:
        shards = ["trading", "aggressive_trading"]
    shards = ordered_unique([*shards, "health_fast"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _candidate_drainers(
    backpressure: dict[str, Any],
    *,
    critical: bool,
    host_context: dict[str, Any] | None = None,
    storage_control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base = _base_env(critical=critical, backpressure=backpressure, host_context=host_context)
    storage = storage_control if isinstance(storage_control, dict) else {}
    stale_decision_rows = _stale_decision_rows_from_storage_control(storage)
    overlay_authoritative = _storage_overlay_authoritative(storage)
    stale_decision_focus_rows = (
        stale_decision_rows
        if overlay_authoritative
        else _merge_source_rows(stale_decision_rows, _stale_decision_rows_from_backpressure(backpressure))
    )
    core_rows = _preferred_source_rows(
        storage,
        backpressure,
        ("governance/channels/decision/", "decisions/", "governance/events/signal_generation_"),
        keys=("top_pending_files",),
    )
    governance_rows = _preferred_source_rows(
        storage,
        backpressure,
        (
            "governance/execution_lanes/",
            "governance/events/auth_events_",
            "governance/events/execution_lane_stale_skips_",
            "governance/events/live_execution_guard_",
            "governance/events/premarket_token_guard_",
            "governance/events/write_failures_",
        ),
        keys=("top_pending_files",),
    )
    operations_guard_rows = _preferred_source_rows(
        storage,
        backpressure,
        (
            "governance/events/paper_execution_guard_",
            "governance/distillation/teacher_student_events_",
            "governance/health/adaptive_regression_guard_feedback",
            "governance/health/infrabot_adaptive_feedback",
            "governance/training_diagnostics/requalification_queue",
            "governance/health/bot_logs_cleanup_intelligence_history",
        ),
        keys=("top_pending_files", "top_deferred_pending_files", "top_support_telemetry_pending_files"),
    )
    api_ingress_rows = _preferred_source_rows(
        storage,
        backpressure,
        ("governance/channels/api/", "governance/channels/ingress/"),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    runtime_rows = _preferred_source_rows(
        storage,
        backpressure,
        (
            "governance/channels/runtime/",
            "governance/channels/loop_state/",
        ),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    schema_rows = _preferred_source_rows(
        storage,
        backpressure,
        ("governance/events/channel_schema_violations_",),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    support_rows = _preferred_source_rows(
        storage,
        backpressure,
        ("governance/watchdog/",),
        keys=("top_support_telemetry_pending_files", "top_deferred_pending_files"),
    )
    risk_rows = _preferred_source_rows(
        storage,
        backpressure,
        ("governance/channels/risk/",),
        keys=("top_pending_files", "top_support_telemetry_pending_files", "top_deferred_pending_files"),
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
    cold_rows = _preferred_source_rows(
        storage,
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
        and not _is_runtime_or_loop_state_source(str(row.get("source_rel") or ""))
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
    provider_rows = [
        row for row in provider_rows
        if not _is_runtime_or_loop_state_source(str(row.get("source_rel") or ""))
    ]
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
    macro_rows = [
        row for row in macro_rows
        if not _is_runtime_or_loop_state_source(str(row.get("source_rel") or ""))
    ]
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
    decision_focus_rows = _stale_sticky_decision_rows(core_rows, stale_decision_focus_rows)
    decision_shards, decision_env = _decision_drainer_env(base, decision_focus_rows)
    if stale_decision_focus_rows:
        decision_env["SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP"] = "1"
    if stale_decision_rows:
        decision_env.update(
            {
                "SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1500",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1500",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": "1500",
                "SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS": str(STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS": str(STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS": str(STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                "INGEST_MAX_BYTES_PER_FILE": str(STALE_DECISION_CATCH_UP_MAX_BYTES_PER_FILE),
                "SQLITE_BATCH_MAX_BYTES": str(STALE_DECISION_CATCH_UP_SQLITE_BATCH_MAX_BYTES),
            }
        )
    api_ingress_shards, api_ingress_env = _api_ingress_drainer_env(base, api_ingress_rows, critical=critical)
    operations_guard_shards, operations_guard_env = _focused_shard_env(
        base,
        operations_guard_rows,
        shards=["governance", "health_fast", "support_watchdog"],
        critical=critical,
        max_files=14,
        max_lines_per_file=64000,
        state_checkpoint_lines=1200,
    )
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
    risk_support_shards, risk_support_env = _focused_shard_env(
        base,
        risk_rows,
        shards=["risk_support"],
        critical=critical,
        max_files=6,
        max_lines_per_file=160000,
        state_checkpoint_lines=8000,
    )
    training_lineage_shards, training_lineage_env = _focused_shard_env(base, training_lineage_rows, shards=["governance", "data"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    label_contract_shards, label_contract_env = _focused_shard_env(base, label_contract_rows, shards=["data", "governance", "schema_violations"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    collector_telemetry_shards, collector_telemetry_env = _focused_shard_env(base, collector_telemetry_rows, shards=["data", "governance"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=900)
    storage_route_shards, storage_route_env = _focused_shard_env(base, storage_route_rows, shards=["governance", "support_watchdog"], critical=critical, max_files=6, max_lines_per_file=12000, state_checkpoint_lines=700)
    ingestion_priority_shards, ingestion_priority_env = _focused_shard_env(base, ingestion_priority_rows, shards=["governance", "data", "support_watchdog"], critical=critical, max_files=8, max_lines_per_file=16000, state_checkpoint_lines=800)
    cold_focus_paths = [str(row["source_rel"]) for row in cold_rows[:8]]
    crypto_cold_focus_paths = [
        str(row["source_rel"])
        for row in cold_rows[:8]
        if _is_crypto_decision_source(str(row.get("source_rel") or ""))
        or str(row.get("shard") or "").strip() == "crypto_explanations"
    ]
    regular_cold_focus_paths = [path for path in cold_focus_paths if path not in set(crypto_cold_focus_paths)]
    stale_decision_shards, stale_decision_env = _decision_drainer_env(base, stale_decision_rows)
    if stale_decision_rows:
        stale_wave_limit = max(_safe_int(stale_decision_env.get("WRITER_CYCLE_MAX_CATCH_UP_WAVES"), 3 if critical else 2), 5 if critical else 3)
        stale_merge_seconds = max(_safe_int(stale_decision_env.get("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"), 120 if critical else 90), 150 if stale_wave_limit >= 6 else 120 if critical else 90)
        stale_shard_timeout = max(_safe_int(stale_decision_env.get("SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"), 420), 480 if stale_wave_limit >= 6 else 420)
        stale_decision_env.update(
            {
                "SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP": "1",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": str(stale_merge_seconds),
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": str(stale_shard_timeout),
                "SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1500",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1500",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": "1500",
                "SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS": str(STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS": str(STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS": str(STALE_DECISION_CATCH_UP_MERGE_MAX_JSONL_ROWS),
                "INGEST_MAX_BYTES_PER_FILE": str(STALE_DECISION_CATCH_UP_MAX_BYTES_PER_FILE),
                "SQLITE_BATCH_MAX_BYTES": str(STALE_DECISION_CATCH_UP_SQLITE_BATCH_MAX_BYTES),
                "WRITER_CYCLE_MAX_CATCH_UP_WAVES": str(stale_wave_limit),
                "BACKLOG_CATCH_UP_WAVE_LIMIT": str(stale_wave_limit),
                "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": str(stale_merge_seconds),
            }
        )

    profiles = [
        _profile(
            name="stale_decision_log_drainer",
            reason="drain source-attributed old decision JSONL logs before tiny stale tails keep stealing writer focus",
            rows=stale_decision_rows,
            shards=stale_decision_shards,
            priority_boost=180_000,
            live_window_safe=True,
            env=stale_decision_env,
            min_pending_lines=25,
            stale_ready_age_seconds=DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS,
        ),
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
            name="risk_support_drainer",
            reason="drain risk-channel support backlog that can block training even when core decision backlog is green",
            rows=risk_rows,
            shards=risk_support_shards,
            priority_boost=175_000,
            live_window_safe=True,
            env=risk_support_env,
            min_pending_lines=25,
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
            rows=decision_focus_rows,
            shards=decision_shards,
            priority_boost=100_000 if _safe_int(backpressure.get("pending_lines"), 0) >= CORE_HARD_PENDING_LINES else 60_000,
            live_window_safe=True,
            env=decision_env,
            min_pending_lines=25,
            stale_ready_age_seconds=DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS,
        ),
        _profile(
            name="operations_guard_drainer",
            reason="drain paper execution, teacher/student, adaptive regression, infrabot feedback, requalification, and cleanup-intelligence tails before they hold the storage guard red",
            rows=operations_guard_rows,
            shards=operations_guard_shards,
            priority_boost=85_000,
            live_window_safe=True,
            env=operations_guard_env,
            min_pending_lines=10,
            stale_ready_age_seconds=DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS,
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
            min_pending_lines=25,
            stale_ready_age_seconds=DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS,
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
                "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_FILES": "8",
                "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_LINES_PER_FILE": "24000",
                "SQL_LINK_SERVICE_SHARD_RUNTIME_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_FILES": "6",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_LINES_PER_FILE": "16000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS": "0",
                "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS": "0",
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
            min_pending_lines=25,
            stale_ready_age_seconds=DEFAULT_AGE_PRESSURE_THRESHOLD_SECONDS,
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
    raw_live_guard = _raw_live_expansion_guard(backpressure, storage_control=storage)
    profiles = _apply_age_pressure_priority(profiles, backpressure, raw_live_guard=raw_live_guard)
    ready = [row for row in profiles if str(row.get("status") or "") == "ready"]
    idle = [row for row in profiles if str(row.get("status") or "") != "ready"]
    ready.sort(
        key=lambda row: (
            _safe_int(row.get("raw_live_expansion_preemption_tier"), 0),
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
        "p_core_backlog_allocation_contract": (
            active_drainer.get("p_core_backlog_allocation_contract")
            if isinstance(active_drainer.get("p_core_backlog_allocation_contract"), dict)
            else {}
        ),
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
    resource_guard = load_json(health_root / "resource_guard_latest.json")
    runtime_throttle = load_json(health_root / "runtime_throttle_control_latest.json")
    computer_task = load_json(health_root / "computer_task_intelligence_latest.json")
    storage_backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage_overlay = (
        storage_control.get("sql_ingestion_pending_overlay")
        if isinstance(storage_control.get("sql_ingestion_pending_overlay"), dict)
        else {}
    )
    critical = bool(
        str(governor.get("profile") or "").strip() == "critical_backpressure"
        or str(storage_control.get("severity") or "").strip() == "critical"
        or _safe_int(backpressure.get("pending_lines"), 0) >= CORE_HARD_PENDING_LINES
        or _safe_int(storage_backpressure.get("total_pending_lines"), 0) >= CORE_HARD_PENDING_LINES
        or _safe_int(storage_overlay.get("total_pending_lines"), 0) >= CORE_HARD_PENDING_LINES
    )
    window = eastern_off_hours_window(now=now)
    host_context = {
        "resource_guard": resource_guard,
        "runtime_throttle": runtime_throttle,
        "computer_task": computer_task,
        "off_hours_active": bool(window.get("active", False)),
    }
    raw_live_guard = _raw_live_expansion_guard(backpressure, storage_control=storage_control)
    drainers = _candidate_drainers(backpressure, critical=critical, host_context=host_context, storage_control=storage_control)
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
            "raw_live_expansion_guard": raw_live_guard,
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
                "reserve a hot raw/live handoff before cold overlay tails when expansion headroom is tight, then re-score before the next wave",
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
