#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "whole_system_intelligence_latest.json"
DEFAULT_SIGNAL_BUS_PATH = PROJECT_ROOT / "governance" / "health" / "system_signal_bus_latest.json"
DEFAULT_BRAIN_PATH = PROJECT_ROOT / "governance" / "health" / "system_brain_latest.json"
DEFAULT_CONTRACTS_PATH = PROJECT_ROOT / "governance" / "health" / "system_process_contracts_latest.json"
DEFAULT_SELF_INTELLIGENCE_PATH = PROJECT_ROOT / "governance" / "health" / "system_self_intelligence_latest.json"
DEFAULT_SUPER_INTELLIGENCE_PATH = PROJECT_ROOT / "governance" / "health" / "system_super_intelligence_latest.json"
DEFAULT_OUTCOME_LEARNING_PATH = PROJECT_ROOT / "governance" / "health" / "super_intelligence_outcome_learning_latest.json"
DEFAULT_RECURSIVE_INTELLIGENCE_PATH = PROJECT_ROOT / "governance" / "health" / "system_recursive_intelligence_latest.json"
DEFAULT_DEEPER_INTELLIGENCE_PATH = PROJECT_ROOT / "governance" / "health" / "deeper_intelligence_layers_latest.json"
DEFAULT_HANDOFF_PATH = PROJECT_ROOT / "governance" / "health" / "codex_handoff_latest.json"
DEFAULT_HANDOFF_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "codex_handoff_latest.md"
DEFAULT_PYCHARM_INDEX_PATH = PROJECT_ROOT / "docs" / "pycharm" / "intelligence_layers_latest.md"
DEFAULT_PYCHARM_INDEX_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "intelligence_layers_pycharm_index_latest.json"
DEFAULT_PYCHARM_HIGHLIGHTS_PATH = PROJECT_ROOT / "governance" / "health" / "pycharm_active_bot_highlights_latest.json"
DEFAULT_DOCUMENTATION_REPORTING_PATH = PROJECT_ROOT / "governance" / "health" / "documentation_reporting_intelligence_latest.json"
DEFAULT_CONTEXT_PATH = PROJECT_ROOT / "governance" / "health" / "whole_system_intelligence_context_latest.json"
DEFAULT_SELF_MEMORY_PATH = PROJECT_ROOT / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl"
DEFAULT_SUPER_MEMORY_PATH = PROJECT_ROOT / "governance" / "system_intelligence" / "super_intelligence_memory.jsonl"
DEFAULT_OUTCOME_MEMORY_PATH = PROJECT_ROOT / "governance" / "system_intelligence" / "intervention_outcomes.jsonl"
DEFAULT_RECURSIVE_MEMORY_PATH = PROJECT_ROOT / "governance" / "system_intelligence" / "recursive_intelligence_memory.jsonl"
DEFAULT_SUPER_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.super_intelligence_override"

STATUS_WEIGHT = {
    "ready": 0,
    "idle": 0,
    "applied": 0,
    "baseline": 0,
    "steady_state": 0,
    "advisory": 25,
    "thin": 25,
    "waiting_for_writer": 35,
    "needs_work": 55,
    "degraded": 65,
    "stalled": 80,
    "apply_failed": 80,
    "blocked": 90,
    "critical": 100,
    "missing": 15,
}

SIGNAL_SOURCES: tuple[dict[str, str], ...] = (
    {"name": "operator_cockpit", "category": "operator", "path": "governance/health/operator_cockpit_latest.json"},
    {"name": "memory_efficiency", "category": "resource", "path": "governance/health/memory_efficiency_control_latest.json"},
    {"name": "runtime_throttle", "category": "resource", "path": "governance/health/runtime_throttle_control_latest.json"},
    {"name": "ingestion_storage", "category": "storage", "path": "governance/health/ingestion_storage_control_latest.json"},
    {"name": "bot_logs_cleanup", "category": "storage", "path": "governance/health/bot_logs_cleanup_intelligence_latest.json"},
    {"name": "storage_quota_guard", "category": "storage", "path": "governance/health/storage_quota_guard_latest.json"},
    {"name": "backpressure_drainer_fleet", "category": "drainer", "path": "governance/health/backpressure_drainer_fleet_latest.json"},
    {"name": "drainer_intelligence", "category": "drainer", "path": "governance/health/drainer_intelligence_layer_latest.json"},
    {"name": "backpressure_super_drainer", "category": "drainer", "path": "governance/health/backpressure_super_drainer_latest.json"},
    {"name": "writer_process_intelligence", "category": "writer", "path": "governance/health/writer_process_intelligence_latest.json"},
    {"name": "writer_cycle_coordinator", "category": "writer", "path": "governance/health/writer_cycle_coordinator_latest.json"},
    {"name": "process_watchdog", "category": "process", "path": "governance/health/process_watchdog_latest.json"},
    {"name": "process_fanout_guard", "category": "process", "path": "governance/health/process_fanout_guard_latest.json"},
    {"name": "guard_intelligence", "category": "process", "path": "governance/health/guard_intelligence_latest.json"},
    {"name": "global_halt", "category": "safety", "path": "governance/health/global_killswitch_latest.json"},
    {"name": "auth_lease_manager", "category": "safety", "path": "governance/health/auth_lease_manager_latest.json"},
    {"name": "data_plane_recovery", "category": "data_plane", "path": "governance/health/data_plane_recovery_controller_latest.json"},
    {"name": "live_runtime_separation", "category": "data_plane", "path": "governance/health/live_runtime_separation_control_latest.json"},
    {"name": "paper_live_data_standard", "category": "paper", "path": "governance/health/paper_live_data_standard_latest.json"},
    {"name": "sleeve_ticker_universe", "category": "market_universe", "path": "governance/health/sleeve_ticker_universe_latest.json"},
    {"name": "mlx_intelligence_router", "category": "compute", "path": "governance/health/mlx_intelligence_router_latest.json"},
    {"name": "library_utilization_router", "category": "compute", "path": "governance/health/library_utilization_router_latest.json"},
    {"name": "training_quality", "category": "training", "path": "governance/health/training_quality_control_latest.json"},
    {"name": "bot_quality", "category": "quality", "path": "governance/health/bot_quality_autopilot_latest.json"},
    {"name": "core_materialization", "category": "quality", "path": "governance/health/core_bot_materialization_guard_latest.json"},
    {"name": "system_self_model", "category": "self_model", "path": "governance/health/system_self_model_latest.json"},
    {"name": "platform_brain_v6", "category": "brain", "path": "governance/health/platform_brain_v6_latest.json"},
    {"name": "deeper_intelligence_layers", "category": "brain", "path": "governance/health/deeper_intelligence_layers_latest.json"},
)

SAFE_REFLEX_PREFIXES = (
    "./scripts/ops/opsctl.sh health-fast",
    "./scripts/ops/opsctl.sh pressure-relief",
    "./scripts/ops/opsctl.sh runtime-throttle",
    "./scripts/ops/opsctl.sh memory-efficiency",
    "./scripts/ops/opsctl.sh platform-stabilization",
    "./scripts/ops/opsctl.sh platform-settlement-stabilization",
    "./scripts/ops/opsctl.sh backpressure-drainers",
    "./scripts/ops/opsctl.sh backpressure-super-drainer",
    "./scripts/ops/opsctl.sh storage-backpressure-autopilot",
)


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


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    explicit = payload.get("overall_status")
    if explicit is None:
        explicit = payload.get("status")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    if isinstance(payload.get("overall"), dict):
        nested = str((payload.get("overall") or {}).get("status") or "").strip().lower()
        if nested:
            return nested
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


def _json_hash(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        encoded = str(payload).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows[-max(int(limit), 1):]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _age_minutes(payload: dict[str, Any], path: Path) -> float | None:
    if not payload:
        return None
    try:
        value = payload_age_minutes(payload, path)
    except Exception:
        return None
    return round(float(value), 3) if value is not None else None


def _storage_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(payload.get("backpressure"))
    total = max(
        _safe_int(payload.get("total_pending_lines"), 0),
        _safe_int(payload.get("pending_lines_total"), 0),
        _safe_int(backpressure.get("total_pending_lines"), 0),
        _safe_int(backpressure.get("core_pending_lines"), 0)
        + _safe_int(backpressure.get("deferred_pending_lines"), 0)
        + _safe_int(backpressure.get("cold_pending_lines"), 0),
    )
    threshold = max(
        _safe_int(backpressure.get("pending_lines_threshold"), 0),
        _safe_int(payload.get("pending_lines_threshold"), 0),
        15_000,
    )
    return {
        "severity": str(payload.get("severity") or backpressure.get("severity") or ""),
        "pressure_index": _safe_float(payload.get("pressure_index"), _safe_float(backpressure.get("pressure_index"), 0.0)),
        "total_pending_lines": int(total),
        "core_pending_lines": _safe_int(backpressure.get("core_pending_lines"), _safe_int(backpressure.get("pending_lines"), 0)),
        "pending_lines_threshold": int(threshold),
        "pending_ratio": round(float(total) / float(max(threshold, 1)), 6),
    }


def _bot_logs_cleanup_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    disk = _as_dict(payload.get("disk_after")) or _as_dict(payload.get("disk_before"))
    summary = _as_dict(payload.get("candidate_summary"))
    intelligence = _as_dict(payload.get("intelligence_layer"))
    return {
        "cleanup_needed": bool(payload.get("cleanup_needed", False)),
        "capacity_pct": _safe_float(disk.get("capacity_pct"), 0.0),
        "free_gb": _safe_float(disk.get("free_gb"), 0.0),
        "target_free_gb": _safe_float(payload.get("target_free_gb"), 0.0),
        "remaining_to_target_gb": _safe_float(payload.get("remaining_to_target_gb"), 0.0),
        "eligible_cleanup_gb": _safe_float(summary.get("eligible_gb"), 0.0),
        "eligible_cleanup_count": _safe_int(summary.get("eligible_count"), 0),
        "cleanup_decision": str(intelligence.get("decision") or ""),
        "pressure_level": str(intelligence.get("pressure_level") or ""),
    }


def _storage_quota_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(payload.get("quota_summary"))
    lanes = [row for row in _as_list(payload.get("lanes")) if isinstance(row, dict)]
    blocked = [str(row.get("family") or "") for row in lanes if str(row.get("status") or "") == "blocked"]
    degraded = [str(row.get("family") or "") for row in lanes if str(row.get("status") or "") == "degraded"]
    return {
        "hard_breaches": _safe_int(summary.get("hard_breaches"), 0),
        "soft_breaches": _safe_int(summary.get("soft_breaches"), 0),
        "tracked_lane_count": _safe_int(summary.get("tracked_lane_count"), len(lanes)),
        "blocked_lanes": [lane for lane in blocked if lane],
        "degraded_lanes": [lane for lane in degraded if lane],
        "recommended_actions": [str(item) for item in _as_list(payload.get("recommended_actions"))],
    }


def _memory_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = _as_dict(payload.get("memory_snapshot"))
    cotenant = _as_dict(payload.get("cotenant_awareness"))
    return {
        "memory_pressure_state": str(snapshot.get("memory_pressure_state") or ""),
        "memory_pressure_kind": str(snapshot.get("memory_pressure_kind") or ""),
        "memory_free_pct": _safe_float(snapshot.get("memory_free_pct"), 0.0),
        "swap_used_gb": _safe_float(snapshot.get("swap_used_gb"), 0.0),
        "compressed_store_gb": _safe_float(snapshot.get("compressed_store_gb"), _safe_float(snapshot.get("compressor_gb"), 0.0)),
        "recommended_profile": str(payload.get("recommended_profile") or ""),
        "memory_pressure_clear": bool(cotenant.get("memory_pressure_clear", False)),
        "block_reasons": [str(item) for item in _as_list(payload.get("reasons")) if str(item).strip()],
    }


def _runtime_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_pressure_level": str(payload.get("memory_pressure_level") or ""),
        "cpu_pressure_level": str(payload.get("cpu_pressure_level") or payload.get("compute_pressure_level") or ""),
        "compute_pressure_level": str(payload.get("compute_pressure_level") or ""),
        "host_saturation_score": _safe_float(payload.get("host_saturation_score"), 0.0),
        "throttle_profile": str(payload.get("throttle_profile") or ""),
    }


def _writer_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    decision = _as_dict(payload.get("decision_packet"))
    health = _as_dict(payload.get("writer_health"))
    safety = _as_dict(payload.get("safety_envelope"))
    return {
        "action": str(decision.get("action") or ""),
        "writer_state": str(decision.get("writer_state") or health.get("state") or ""),
        "writer_active": bool(health.get("active", False)),
        "expanded_writer_lane_count": _safe_int(decision.get("expanded_writer_lane_count"), 0),
        "risk_flags": [str(item) for item in _as_list(decision.get("risk_flags"))],
        "single_writer_only": bool(safety.get("single_writer_only", False)),
        "starts_parallel_sql_writers": bool(safety.get("starts_parallel_sql_writers", False)),
        "writer_recovery_required": bool(safety.get("writer_recovery_required", False)),
    }


def _drainer_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    decision = _as_dict(payload.get("decision_packet"))
    settings = _as_dict(payload.get("settings"))
    summary = _as_dict(payload.get("summary"))
    active = payload.get("active_drainer")
    active_name = str((active or {}).get("name") or "") if isinstance(active, dict) else str(active or "")
    return {
        "action": str(decision.get("action") or ""),
        "selected_drainer": str(decision.get("selected_drainer") or active_name),
        "ready_drainer_count": _safe_int(payload.get("ready_drainer_count"), 0),
        "total_pending_lines": _safe_int(decision.get("total_pending_lines"), _safe_int(summary.get("final_pending_lines"), 0)),
        "target_pending_lines": _safe_int(decision.get("target_pending_lines"), _safe_int(settings.get("target_pending_lines"), 0)),
        "risk_flags": [str(item) for item in _as_list(decision.get("risk_flags"))],
    }


def _process_metrics(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "process_watchdog":
        rows = [row for row in _as_list(payload.get("status")) if isinstance(row, dict)]
        down = [
            str(row.get("name") or "")
            for row in rows
            if not bool(row.get("process_live", row.get("running", 0))) and not row.get("restarted_pid")
        ]
        return {
            "watched_process_count": len(rows),
            "down_processes": [name for name in down if name],
            "alert_count": len(_as_list(payload.get("alerts"))),
            "restarted_count": sum(1 for row in rows if row.get("restarted_pid")),
        }
    summary = _as_dict(payload.get("summary"))
    fanout = _as_dict(payload.get("fanout"))
    startup_policy = _as_dict(payload.get("startup_policy"))
    return {
        "triggered": bool(payload.get("triggered", False) or summary.get("triggered", False)),
        "targetable_process_count": _safe_int(
            payload.get("targetable_process_count"),
            _safe_int(summary.get("targetable_process_count"), _safe_int(fanout.get("targetable_count"), 0)),
        ),
        "total_rss_mb": _safe_float(payload.get("total_rss_mb"), _safe_float(summary.get("total_rss_mb"), _safe_float(fanout.get("total_rss_mb"), 0.0))),
        "core_sleeve_restart_allowed": bool(startup_policy.get("core_sleeve_restart_allowed", False)),
    }


def _guard_intelligence_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    signals = _as_dict(payload.get("signals"))
    fanout = _as_dict(signals.get("fanout"))
    resource = _as_dict(signals.get("resource_pressure"))
    storage = _as_dict(signals.get("storage_pressure"))
    counts = _as_dict(signals.get("guard_status_counts"))
    overrides = _as_dict(payload.get("recommended_env_overrides"))
    return {
        "policy_mode": str(payload.get("policy_mode") or ""),
        "pressure_score": _safe_float(payload.get("pressure_score"), 0.0),
        "fanout_source": str(fanout.get("source") or ""),
        "process_count": _safe_int(fanout.get("process_count"), 0),
        "max_count": _safe_int(fanout.get("max_count"), 0),
        "target_count": _safe_int(fanout.get("target_count"), 0),
        "total_rss_mb": _safe_float(fanout.get("total_rss_mb"), 0.0),
        "max_rss_mb": _safe_float(fanout.get("max_rss_mb"), 0.0),
        "target_rss_mb": _safe_float(fanout.get("target_rss_mb"), 0.0),
        "triggered": bool(fanout.get("triggered", False)),
        "resource_score": _safe_float(resource.get("score"), 0.0),
        "storage_score": _safe_float(storage.get("score"), 0.0),
        "blockers": [str(item) for item in _as_list(counts.get("blockers"))],
        "warnings": [str(item) for item in _as_list(counts.get("warnings"))],
        "stale_core_artifacts": [str(item) for item in _as_list(counts.get("stale_core_artifacts"))],
        "process_fanout_guard_active": str(overrides.get("PROCESS_FANOUT_GUARD_ACTIVE") or ""),
        "training_paused": str(overrides.get("TRAINING_RUNTIME_PAUSED_FOR_FANOUT") or ""),
        "research_paused": str(overrides.get("SHADOW_RESEARCH_PAUSED_FOR_FANOUT") or ""),
        "specialized_sleeves_enabled": str(overrides.get("RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES") or ""),
    }


def _paper_live_data_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(payload.get("counts_after"))
    target = _as_dict(payload.get("paper_lane_target"))
    return {
        "paper_live_data_enabled_bots": _safe_int(counts.get("paper_live_data_enabled_bots"), 0),
        "legacy_bootstrap_paper_bots": _safe_int(counts.get("legacy_bootstrap_paper_bots"), 0),
        "standard_promoted_paper_bots": _safe_int(counts.get("standard_promoted_paper_bots"), 0),
        "collection_until_standard_bots": _safe_int(counts.get("collection_until_standard_bots"), 0),
        "data_collection_active_bots": _safe_int(counts.get("data_collection_active_bots"), 0),
        "target": _safe_int(target.get("target"), 40),
        "minimum": _safe_int(target.get("minimum"), 30),
        "maximum": _safe_int(target.get("maximum"), 50),
        "within_target_band": bool(target.get("within_target_band", False)),
        "direct_execution_allowed_bots": _safe_int(counts.get("direct_execution_allowed_bots"), 0),
        "live_trading_enabled_bots": _safe_int(counts.get("live_trading_enabled_bots"), 0),
    }


def _sleeve_ticker_universe_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(payload.get("symbol_counts"))
    groups = _as_dict(payload.get("sleeve_groups"))
    return {
        "core_symbol_count": _safe_int(counts.get("SHADOW_SYMBOLS_CORE"), 0),
        "volatile_symbol_count": _safe_int(counts.get("SHADOW_SYMBOLS_VOLATILE"), 0),
        "defensive_symbol_count": _safe_int(counts.get("SHADOW_SYMBOLS_DEFENSIVE"), 0),
        "crypto_symbol_count": _safe_int(counts.get("COINBASE_WATCH_SYMBOLS"), 0),
        "bond_symbol_count": _safe_int(counts.get("BOND_SYMBOLS"), 0),
        "fx_symbol_count": _safe_int(counts.get("FX_SYMBOLS"), 0),
        "sleeve_group_count": len(groups),
        "enabled": str(_as_dict(payload.get("env_overrides")).get("SLEEVE_TICKER_UNIVERSE_ENABLED") or "") == "1",
    }


def _registry_metrics(project_root: Path) -> dict[str, Any]:
    registry = load_json(project_root / "master_bot_registry.json")
    rows = [row for row in _as_list(registry.get("sub_bots")) if isinstance(row, dict)]
    summary = _as_dict(registry.get("summary"))
    active = sum(1 for row in rows if bool(row.get("active", False))) or _safe_int(summary.get("active_bots"), 0)
    collecting = sum(1 for row in rows if bool(row.get("data_collection_active", False))) or _safe_int(
        summary.get("data_collection_active_bots"),
        0,
    )
    sleeves = {
        str(row.get("sleeve_profile") or row.get("slot_kind") or "")
        for row in rows
        if str(row.get("sleeve_profile") or row.get("slot_kind") or "").strip()
    }
    return {
        "total_bots": len(rows) or _safe_int(summary.get("total_bots"), 0),
        "active_bots": int(active),
        "data_collection_active_bots": int(collecting),
        "sleeve_profile_count": len(sleeves) or _safe_int(summary.get("sleeve_profile_count"), 0),
        "active_bot_examples": [
            {
                "bot_id": str(row.get("bot_id") or row.get("id") or ""),
                "sleeve_profile": str(row.get("sleeve_profile") or row.get("slot_kind") or ""),
                "paper_live_data_enabled": bool(row.get("paper_live_data_enabled", False)),
                "data_collection_active": bool(row.get("data_collection_active", False)),
            }
            for row in rows
            if bool(row.get("active", False))
        ][:12],
    }


def _metrics_for_signal(name: str, project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "ingestion_storage":
        return _storage_metrics(payload)
    if name == "bot_logs_cleanup":
        return _bot_logs_cleanup_metrics(payload)
    if name == "storage_quota_guard":
        return _storage_quota_metrics(payload)
    if name == "memory_efficiency":
        return _memory_metrics(payload)
    if name == "runtime_throttle":
        return _runtime_metrics(payload)
    if name == "writer_process_intelligence":
        return _writer_metrics(payload)
    if name in {"drainer_intelligence", "backpressure_drainer_fleet", "backpressure_super_drainer"}:
        return _drainer_metrics(payload)
    if name in {"process_watchdog", "process_fanout_guard"}:
        return _process_metrics(name, payload)
    if name == "guard_intelligence":
        return _guard_intelligence_metrics(payload)
    if name == "paper_live_data_standard":
        return _paper_live_data_metrics(payload)
    if name == "sleeve_ticker_universe":
        return _sleeve_ticker_universe_metrics(payload)
    if name == "global_halt":
        return {
            "halt_active": bool(payload.get("halt", False) or payload.get("global_halt_active", False)),
            "clear_ready": bool(payload.get("clear_ready", False)),
            "clear_blockers": [str(item) for item in _as_list(payload.get("clear_blockers"))],
            "reasons": [str(item) for item in _as_list(payload.get("reasons"))],
        }
    if name == "auth_lease_manager":
        lease = _as_dict(payload.get("lease_budget"))
        broker = _as_dict(payload.get("broker_state"))
        return {
            "lease_state": str(payload.get("lease_state") or ""),
            "expires_in_seconds": _safe_float(lease.get("expires_in_seconds"), 0.0),
            "auth_ok": bool(broker.get("auth_ok", False)),
            "broker_operable": bool(broker.get("broker_operable", False)),
            "auth_reason": str(broker.get("auth_reason") or ""),
        }
    if name == "system_self_model":
        identity = _as_dict(payload.get("identity"))
        return {
            "overall_status": _status(payload),
            "active_bots": _safe_int(identity.get("active_bots"), 0),
            "collection_bots": _safe_int(identity.get("data_collection_active_bots"), 0),
            "self_summary": str(payload.get("self_summary") or "")[:240],
        }
    if name == "platform_brain_v6":
        return {
            "section_count": _safe_int(payload.get("section_count"), 0),
            "gate_blockers": [str(item) for item in _as_list(payload.get("gate_blockers"))],
            "next_best_command": str((_as_dict(_as_dict(payload.get("sections")).get("operator_narrative_synthesizer"))).get("next_best_command") or ""),
        }
    if name == "deeper_intelligence_layers":
        operator = _as_dict(payload.get("operator_dialogue_packet"))
        surface = _as_dict(payload.get("surface_snapshot"))
        return {
            "layer_count": _safe_int(payload.get("layer_count"), 0),
            "ready_count": _safe_int(payload.get("ready_count"), 0),
            "advisory_count": _safe_int(payload.get("advisory_count"), 0),
            "degraded_count": _safe_int(payload.get("degraded_count"), 0),
            "blocked_count": _safe_int(payload.get("blocked_count"), 0),
            "top_attention": [str(item) for item in _as_list(operator.get("top_attention"))],
            "safe_next_command": [str(item) for item in _as_list(operator.get("safe_next_command"))],
            "storage_pending_ratio": _safe_float(_as_dict(surface.get("storage")).get("pending_ratio"), 0.0),
            "runtime_pressure_high": bool(_as_dict(surface.get("runtime")).get("pressure_high", False)),
            "missing_surface_count": _safe_int(payload.get("missing_surfaces") and len(_as_list(payload.get("missing_surfaces"))), 0),
            "authority_boundary": "advisory_control_plane_with_constitutional_lockout_attestation",
        }
    if name == "operator_cockpit":
        adaptive = _as_dict(payload.get("adaptive_posture"))
        return {
            "hard_blockers": [str(item) for item in _as_list(adaptive.get("hard_blockers"))],
            "pressure_level": str(adaptive.get("pressure_level") or ""),
            "recommended_action_count": len(_as_list(payload.get("recommended_actions"))),
        }
    if name == "core_materialization":
        summary = _as_dict(payload.get("summary"))
        return {
            "missing_core_module_count": _safe_int(summary.get("missing_core_module_count"), 0),
            "duplicate_core_version_count": _safe_int(summary.get("duplicate_core_version_count"), 0),
        }
    return {}


def _severity_for_signal(name: str, status: str, metrics: dict[str, Any], loaded: bool) -> int:
    if not loaded:
        return STATUS_WEIGHT["missing"]
    score = STATUS_WEIGHT.get(str(status or "").lower(), 45)
    if name == "ingestion_storage":
        pressure_index = _safe_float(metrics.get("pressure_index"), 0.0)
        pending_ratio = _safe_float(metrics.get("pending_ratio"), 0.0)
        severity = str(metrics.get("severity") or "").lower()
        if severity == "critical" or pressure_index >= 3.0:
            score = max(score, 95)
        elif severity in {"high", "blocked"} or pending_ratio >= 1.0:
            score = max(score, 75)
    elif name == "bot_logs_cleanup":
        capacity_pct = _safe_float(metrics.get("capacity_pct"), 0.0)
        if bool(metrics.get("cleanup_needed", False)) or _safe_float(metrics.get("remaining_to_target_gb"), 0.0) > 0:
            score = max(score, 82)
        elif capacity_pct >= 98.0:
            score = max(score, 95)
        elif capacity_pct >= 92.0:
            score = max(score, 70)
        elif capacity_pct >= 86.0:
            score = max(score, 45)
    elif name == "storage_quota_guard":
        if _safe_int(metrics.get("hard_breaches"), 0) > 0:
            score = max(score, 82)
        elif _safe_int(metrics.get("soft_breaches"), 0) > 0:
            score = max(score, 62)
    elif name == "memory_efficiency":
        state = str(metrics.get("memory_pressure_state") or "").lower()
        kind = str(metrics.get("memory_pressure_kind") or "").lower()
        if _memory_metrics_clear(metrics):
            score = min(score, 35)
        elif state in {"red", "critical"} or kind in {"swap", "compressor", "critical"}:
            score = max(score, 85)
        elif state in {"yellow", "orange", "warning"}:
            score = max(score, 65)
    elif name == "runtime_throttle":
        host_score = _safe_float(metrics.get("host_saturation_score"), 0.0)
        if str(metrics.get("memory_pressure_level") or "").lower() in {"high", "critical"} or host_score >= 85.0:
            score = max(score, 90)
        elif str(metrics.get("cpu_pressure_level") or "").lower() in {"high", "critical"} or host_score >= 65.0:
            score = max(score, 70)
    elif name == "writer_process_intelligence":
        risks = set(str(item) for item in _as_list(metrics.get("risk_flags")))
        if "duplicate_sql_writer_processes" in risks:
            score = max(score, 100)
        elif bool(metrics.get("writer_recovery_required", False)) or str(metrics.get("writer_state") or "") in {"stalled", "stale_progress"}:
            score = max(score, 80)
    elif name == "global_halt" and bool(metrics.get("halt_active", False)):
        score = 100
    elif name == "auth_lease_manager":
        if not bool(metrics.get("auth_ok", True)) or str(metrics.get("lease_state") or "").lower() in {"critical", "expired"}:
            score = max(score, 90)
        elif str(metrics.get("lease_state") or "").lower() == "warning":
            score = max(score, 60)
    elif name == "process_watchdog":
        if _as_list(metrics.get("down_processes")) or _safe_int(metrics.get("alert_count"), 0) > 0:
            score = max(score, 70)
    elif name == "process_fanout_guard":
        if bool(metrics.get("triggered", False)) and _safe_int(metrics.get("targetable_process_count"), 0) > 0:
            score = max(score, 75)
    elif name == "guard_intelligence":
        policy_mode = str(metrics.get("policy_mode") or "").lower()
        if _as_list(metrics.get("blockers")):
            score = max(score, 85)
        if bool(metrics.get("triggered", False)) or policy_mode == "protective_throttle":
            score = max(score, 80)
        elif policy_mode == "balanced_guarded":
            score = max(score, 45)
        elif policy_mode == "full_schwab_observe":
            score = min(score, 20)
    elif name == "paper_live_data_standard":
        if _safe_int(metrics.get("direct_execution_allowed_bots"), 0) > 0 or _safe_int(metrics.get("live_trading_enabled_bots"), 0) > 0:
            score = max(score, 100)
        elif _safe_int(metrics.get("paper_live_data_enabled_bots"), 0) > _safe_int(metrics.get("maximum"), 50):
            score = max(score, 75)
        elif not bool(metrics.get("within_target_band", False)):
            score = max(score, 45)
    elif name == "sleeve_ticker_universe":
        if not bool(metrics.get("enabled", False)):
            score = max(score, 55)
        elif _safe_int(metrics.get("core_symbol_count"), 0) < 60:
            score = max(score, 45)
    elif name == "core_materialization":
        if _safe_int(metrics.get("missing_core_module_count"), 0) or _safe_int(metrics.get("duplicate_core_version_count"), 0):
            score = max(score, 65)
    elif name == "deeper_intelligence_layers":
        if _safe_int(metrics.get("blocked_count"), 0) > 0:
            score = max(score, 95)
        elif _safe_int(metrics.get("degraded_count"), 0) > 0:
            score = max(score, 70)
        elif _safe_int(metrics.get("advisory_count"), 0) > 0:
            score = max(score, 35)
    return int(max(0, min(100, score)))


def _memory_metrics_show_pressure(metrics: dict[str, Any]) -> bool:
    state = str(metrics.get("memory_pressure_state") or "").lower()
    kind = str(metrics.get("memory_pressure_kind") or "").lower()
    reasons = [str(item).lower() for item in _as_list(metrics.get("block_reasons"))]
    memory_reasons = [
        item
        for item in reasons
        if any(marker in item for marker in ("memory", "swap", "compress", "throttled"))
    ]
    return bool(
        state in {"yellow", "orange", "red", "critical", "warning"}
        or kind not in {"", "none", "normal", "green", "clear"}
        or _safe_float(metrics.get("swap_used_gb"), 0.0) >= 8.0
        or _safe_float(metrics.get("compressed_store_gb"), 0.0) >= 18.0
        or memory_reasons
    )


def _memory_metrics_clear(metrics: dict[str, Any]) -> bool:
    state = str(metrics.get("memory_pressure_state") or "").lower()
    kind = str(metrics.get("memory_pressure_kind") or "").lower()
    return bool(
        state in {"green", "none", "normal", "clear"}
        and kind in {"", "none", "normal", "green", "clear"}
        and bool(metrics.get("memory_pressure_clear", False))
        and not _memory_metrics_show_pressure(metrics)
    )


def _signal_summary(name: str, metrics: dict[str, Any]) -> str:
    if name == "ingestion_storage":
        return f"pending={metrics.get('total_pending_lines', 0)} pressure_index={metrics.get('pressure_index', 0)}"
    if name == "bot_logs_cleanup":
        return f"free_gb={metrics.get('free_gb', 0)} capacity={metrics.get('capacity_pct', 0)} cleanup_needed={metrics.get('cleanup_needed', False)}"
    if name == "storage_quota_guard":
        return f"hard_breaches={metrics.get('hard_breaches', 0)} lanes={','.join(str(item) for item in _as_list(metrics.get('blocked_lanes'))) or 'none'}"
    if name == "memory_efficiency":
        return f"memory={metrics.get('memory_pressure_state', '')} kind={metrics.get('memory_pressure_kind', '')}"
    if name == "runtime_throttle":
        return f"host={metrics.get('host_saturation_score', 0)} memory={metrics.get('memory_pressure_level', '')}"
    if name == "writer_process_intelligence":
        return f"writer={metrics.get('writer_state', '')} action={metrics.get('action', '')}"
    if name in {"drainer_intelligence", "backpressure_drainer_fleet", "backpressure_super_drainer"}:
        return f"drainer={metrics.get('selected_drainer', '')} action={metrics.get('action', '')}"
    if name == "global_halt":
        return f"halt_active={metrics.get('halt_active', False)} blockers={len(_as_list(metrics.get('clear_blockers')))}"
    if name == "process_watchdog":
        return f"down={','.join(str(item) for item in _as_list(metrics.get('down_processes'))) or 'none'}"
    if name == "guard_intelligence":
        return f"mode={metrics.get('policy_mode', '')} pressure={metrics.get('pressure_score', 0)}"
    if name == "auth_lease_manager":
        return f"lease={metrics.get('lease_state', '')} auth_ok={metrics.get('auth_ok', False)}"
    if name == "paper_live_data_standard":
        return f"paper={metrics.get('paper_live_data_enabled_bots', 0)} target={metrics.get('minimum', 30)}-{metrics.get('maximum', 50)}"
    if name == "sleeve_ticker_universe":
        return f"core={metrics.get('core_symbol_count', 0)} crypto={metrics.get('crypto_symbol_count', 0)} groups={metrics.get('sleeve_group_count', 0)}"
    if name == "deeper_intelligence_layers":
        return f"layers={metrics.get('layer_count', 0)} blocked={metrics.get('blocked_count', 0)} degraded={metrics.get('degraded_count', 0)}"
    return ""


def build_signal_bus(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root)
    signals: list[dict[str, Any]] = []
    for source in SIGNAL_SOURCES:
        name = str(source["name"])
        path = project_root / str(source["path"])
        payload = load_json(path)
        loaded = bool(payload)
        status = _status(payload)
        metrics = _metrics_for_signal(name, project_root, payload)
        if name == "global_halt" and loaded:
            if bool(metrics.get("halt_active", False)):
                status = "blocked"
            elif _as_list(metrics.get("clear_blockers")):
                status = "advisory"
            else:
                status = "ready"
        severity = _severity_for_signal(name, status, metrics, loaded)
        signals.append(
            {
                "name": name,
                "category": str(source["category"]),
                "status": status,
                "severity_score": severity,
                "loaded": loaded,
                "age_minutes": _age_minutes(payload, path),
                "path": str(path),
                "payload_hash_short": _json_hash(payload)[:12] if payload else "",
                "summary": _signal_summary(name, metrics),
                "metrics": metrics,
            }
        )

    registry_metrics = _registry_metrics(project_root)
    loaded_signals = [row for row in signals if bool(row.get("loaded", False))]
    top_signal = max(loaded_signals, key=lambda row: _safe_int(row.get("severity_score"), 0), default={})
    severe_signals = [row for row in loaded_signals if _safe_int(row.get("severity_score"), 0) >= 75]
    blocked_signals = [row for row in loaded_signals if _safe_int(row.get("severity_score"), 0) >= 90]
    storage = next((row for row in signals if row["name"] == "ingestion_storage"), {})
    memory = next((row for row in signals if row["name"] == "memory_efficiency"), {})
    runtime = next((row for row in signals if row["name"] == "runtime_throttle"), {})
    writer = next((row for row in signals if row["name"] == "writer_process_intelligence"), {})
    drainer = next((row for row in signals if row["name"] == "drainer_intelligence"), {})
    guard = next((row for row in signals if row["name"] == "guard_intelligence"), {})
    global_halt = next((row for row in signals if row["name"] == "global_halt"), {})
    paper_standard = next((row for row in signals if row["name"] == "paper_live_data_standard"), {})
    ticker_universe = next((row for row in signals if row["name"] == "sleeve_ticker_universe"), {})

    storage_metrics = _as_dict(storage.get("metrics"))
    memory_metrics = _as_dict(memory.get("metrics"))
    runtime_metrics = _as_dict(runtime.get("metrics"))
    writer_metrics = _as_dict(writer.get("metrics"))
    drainer_metrics = _as_dict(drainer.get("metrics"))
    guard_metrics = _as_dict(guard.get("metrics"))
    global_halt_metrics = _as_dict(global_halt.get("metrics"))
    paper_standard_metrics = _as_dict(paper_standard.get("metrics"))
    ticker_universe_metrics = _as_dict(ticker_universe.get("metrics"))
    memory_high = _memory_metrics_show_pressure(memory_metrics)
    runtime_status = str(runtime.get("status") or "").lower()
    runtime_high = bool(
        runtime_status in {"blocked", "critical", "degraded"}
        or str(runtime_metrics.get("memory_pressure_level") or "").lower() in {"high", "critical"}
        or str(runtime_metrics.get("cpu_pressure_level") or "").lower() in {"high", "critical"}
        or _safe_float(runtime_metrics.get("host_saturation_score"), 0.0) >= 80.0
    )
    storage_critical = bool(
        str(storage_metrics.get("severity") or "").lower() == "critical"
        or _safe_float(storage_metrics.get("pressure_index"), 0.0) >= 3.0
    )
    worst_score = _safe_int(top_signal.get("severity_score"), 0)
    overall_status = "ready"
    if worst_score >= 90:
        overall_status = "blocked"
    elif worst_score >= 65:
        overall_status = "degraded"
    elif worst_score >= 25:
        overall_status = "advisory"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "system_signal_bus",
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "summary": {
            "signal_count": len(signals),
            "loaded_signal_count": len(loaded_signals),
            "blocked_signal_count": len(blocked_signals),
            "severe_signal_count": len(severe_signals),
            "top_risk": str(top_signal.get("name") or "none"),
            "top_risk_category": str(top_signal.get("category") or ""),
            "top_risk_score": worst_score,
            "total_pending_lines": _safe_int(storage_metrics.get("total_pending_lines"), 0),
            "storage_critical": storage_critical,
            "memory_pressure_high": memory_high,
            "runtime_pressure_high": runtime_high,
            "writer_recovery_required": bool(writer_metrics.get("writer_recovery_required", False)),
            "writer_active": bool(writer_metrics.get("writer_active", False)),
            "drainer_action": str(drainer_metrics.get("action") or ""),
            "guard_policy_mode": str(guard_metrics.get("policy_mode") or ""),
            "guard_pressure_score": _safe_float(guard_metrics.get("pressure_score"), 0.0),
            "guard_triggered": bool(guard_metrics.get("triggered", False)),
            "guard_blocker_count": len(_as_list(guard_metrics.get("blockers"))),
            "global_halt_active": bool(global_halt_metrics.get("halt_active", False)),
            "active_bots": _safe_int(registry_metrics.get("active_bots"), 0),
            "collection_bots": _safe_int(registry_metrics.get("data_collection_active_bots"), 0),
            "sleeve_profile_count": _safe_int(registry_metrics.get("sleeve_profile_count"), 0),
            "paper_live_data_bots": _safe_int(paper_standard_metrics.get("paper_live_data_enabled_bots"), 0),
            "paper_live_data_within_band": bool(paper_standard_metrics.get("within_target_band", False)),
            "expanded_core_symbol_count": _safe_int(ticker_universe_metrics.get("core_symbol_count"), 0),
            "expanded_crypto_symbol_count": _safe_int(ticker_universe_metrics.get("crypto_symbol_count"), 0),
        },
        "signals": sorted(signals, key=lambda row: (_safe_int(row.get("severity_score"), 0), str(row.get("name") or "")), reverse=True),
        "registry_metrics": registry_metrics,
        "signal_bus_contract": {
            "purpose": "normalize_system_artifacts_into_ranked_operational_signals",
            "writes": ["system_signal_bus_latest.json"],
            "does_not_execute_commands": True,
            "does_not_trade": True,
        },
    }


def _contract_status(name: str, signal_bus: dict[str, Any]) -> tuple[str, list[str]]:
    summary = _as_dict(signal_bus.get("summary"))
    risks: list[str] = []
    status = "ready"
    if name == "sql_writer":
        if bool(summary.get("writer_recovery_required", False)):
            risks.append("writer_recovery_required")
            status = "advisory"
        if bool(summary.get("writer_active", False)):
            risks.append("writer_active_single_lock_in_use")
            status = "advisory"
    elif name == "drainers":
        if bool(summary.get("writer_active", False)):
            risks.append("wait_for_single_writer")
            status = "advisory"
        if bool(summary.get("memory_pressure_high", False)) or bool(summary.get("runtime_pressure_high", False)):
            risks.append("micro_drain_only_under_pressure")
            status = "advisory"
    elif name == "sleeves":
        if bool(summary.get("global_halt_active", False)):
            risks.append("global_halt_active")
            status = "blocked"
        elif str(summary.get("guard_policy_mode") or "") == "protective_throttle":
            risks.append("guard_intelligence_protective_throttle")
            status = "advisory"
        elif bool(summary.get("storage_critical", False)) or bool(summary.get("runtime_pressure_high", False)):
            risks.append("expansion_paused_until_pressure_clears")
            status = "advisory"
    elif name == "strategy_expansion":
        if (
            bool(summary.get("storage_critical", False))
            or bool(summary.get("memory_pressure_high", False))
            or str(summary.get("guard_policy_mode") or "") == "protective_throttle"
        ):
            risks.append("growth_gate_closed_under_resource_pressure")
            status = "advisory"
    elif name == "training":
        if (
            bool(summary.get("memory_pressure_high", False))
            or bool(summary.get("runtime_pressure_high", False))
            or str(summary.get("guard_policy_mode") or "") == "protective_throttle"
        ):
            risks.append("heavy_training_capped")
            status = "advisory"
    elif name == "reporting":
        if bool(summary.get("storage_critical", False)):
            risks.append("report_writes_deprioritized")
            status = "advisory"
    elif name == "auth_and_halt":
        if bool(summary.get("global_halt_active", False)):
            risks.append("halt_clearance_required_before_relaunch")
            status = "blocked"
    return status, risks


def build_process_contracts(signal_bus: dict[str, Any]) -> dict[str, Any]:
    base_contracts = [
        {
            "name": "sql_writer",
            "owner": "sql_link_shard_manager",
            "authority_boundary": "jsonl_to_sql_storage_only",
            "may_start": ["one_guarded_writer_cycle"],
            "may_not_start": ["parallel_sql_writers", "live_order_execution", "ungoverned_long_running_writer"],
            "max_concurrency": 1,
            "resource_budget": {"mode": "single_lock", "lock": "governance/locks/jsonl_sql_writer.lock"},
            "handoff_format": ["writer_state", "progress_age_minutes", "merged_rows_this_cycle", "risk_flags"],
            "recovery_behavior": "use_writer_cycle_coordinator_for_stale_or_stalled_progress",
        },
        {
            "name": "drainers",
            "owner": "backpressure_super_drainer",
            "authority_boundary": "queue_handoff_and_wave_selection_only",
            "may_start": ["bounded_drainer_wave", "focused_handoff_request"],
            "may_not_start": ["parallel_sql_writers", "broad_collector_expansion_under_pressure"],
            "max_concurrency": 2,
            "resource_budget": {"max_waves_when_pressured": 1, "cooldown_seconds_when_pressured": 90},
            "handoff_format": ["selected_drainer", "target_pending_lines", "writer_health", "pressure_forecast"],
            "recovery_behavior": "score_lanes_then_wait_for_writer_or_pressure_relief",
        },
        {
            "name": "sleeves",
            "owner": "run_all_sleeves_and_specialized_launchers",
            "authority_boundary": "paper_collection_and_shadow_sleeve_runtime",
            "may_start": ["paper_shadow_sleeve", "data_collection_sleeve"],
            "may_not_start": ["live_order_execution_without_operator_clearance", "expansion_during_global_halt"],
            "max_concurrency": "runtime_throttle_defined",
            "resource_budget": {"priority": "protect_live_collection_and_paper_trade_before_expansion"},
            "handoff_format": ["sleeve_profile", "runtime_status", "collection_only", "blocked_reason"],
            "recovery_behavior": "pause_growth_then_relaunch_only_after_halt_and_data_plane_clear",
        },
        {
            "name": "strategy_expansion",
            "owner": "expansion_capacity_and_strategy_gap_lanes",
            "authority_boundary": "catalog_and_shadow_strategy_generation_only",
            "may_start": ["collection_only_strategy_spec", "coverage_gap_ticket"],
            "may_not_start": ["heavy_training_under_memory_pressure", "live_trade_authority"],
            "max_concurrency": "growth_capacity_budget",
            "resource_budget": {"storage_first": True, "requires_writer_and_drainer_clearance": True},
            "handoff_format": ["sleeve", "strategy_count", "collector_cost", "rollback_metadata"],
            "recovery_behavior": "route_new_backlog_to_organizers_when_storage_pressure_is_high",
        },
        {
            "name": "training",
            "owner": "training_runtime_control",
            "authority_boundary": "offline_training_and_requalification_only",
            "may_start": ["bounded_training_job", "requalification_audit"],
            "may_not_start": ["foreground_starving_heavy_job", "live_execution"],
            "max_concurrency": "runtime_throttle_defined",
            "resource_budget": {"requires_memory_normal": True, "mlx_caps_respected": True},
            "handoff_format": ["profile", "dataset", "memory_budget", "quality_gate"],
            "recovery_behavior": "downshift_to_canary_or_off_hours_when_host_pressure_is_high",
        },
        {
            "name": "reporting",
            "owner": "report_quality_and_operator_briefs",
            "authority_boundary": "read_only_summary_and_artifact_rendering",
            "may_start": ["operator_brief", "quality_report"],
            "may_not_start": ["large_pdf_bundle_when_storage_critical", "trade_execution"],
            "max_concurrency": "library_router_defined",
            "resource_budget": {"degrade_first_when_storage_critical": True},
            "handoff_format": ["summary", "source_files", "next_safe_commands"],
            "recovery_behavior": "prefer_markdown_and_compact_json_until_storage_clears",
        },
        {
            "name": "auth_and_halt",
            "owner": "global_killswitch_and_auth_lease_manager",
            "authority_boundary": "safety_clearance_and_operator_auth_only",
            "may_start": ["token_refresh", "halt_refresh", "clear_when_blockers_empty"],
            "may_not_start": ["force_clear_without_prechecks", "live_relaunch_before_halt_clear"],
            "max_concurrency": 1,
            "resource_budget": {"operator_attention_required_for_interactive_auth": True},
            "handoff_format": ["halt_active", "clear_blockers", "auth_state", "next_safe_command"],
            "recovery_behavior": "refresh_auth_then_clear_halt_then_verify_livefeed",
        },
    ]
    contracts: list[dict[str, Any]] = []
    for row in base_contracts:
        status, risks = _contract_status(str(row["name"]), signal_bus)
        contracts.append({**row, "status": status, "active_risks": risks})
    blocked = [row for row in contracts if str(row.get("status") or "") == "blocked"]
    advisory = [row for row in contracts if str(row.get("status") or "") == "advisory"]
    status = "blocked" if blocked else "advisory" if advisory else "ready"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "system_process_contracts",
        "ok": status == "ready",
        "overall_status": status,
        "contract_count": len(contracts),
        "blocked_contract_count": len(blocked),
        "advisory_contract_count": len(advisory),
        "contracts": contracts,
        "global_safety_contract": {
            "paper_data_first": True,
            "live_trade_authority_added": False,
            "single_sql_writer_only": True,
            "parallel_sql_writers_allowed": False,
            "bounded_apply_modes_only": True,
            "codex_handoff_is_advisory": True,
        },
    }


def _brain_risk_flags(signal_bus: dict[str, Any], process_contracts: dict[str, Any]) -> list[str]:
    summary = _as_dict(signal_bus.get("summary"))
    risks: list[str] = []
    if bool(summary.get("global_halt_active", False)):
        risks.append("global_halt_active")
    if bool(summary.get("storage_critical", False)):
        risks.append("storage_critical")
    if bool(summary.get("runtime_pressure_high", False)):
        risks.append("runtime_pressure_high")
    if bool(summary.get("memory_pressure_high", False)):
        risks.append("memory_pressure_high")
    if bool(summary.get("writer_recovery_required", False)):
        risks.append("writer_recovery_required")
    if bool(summary.get("writer_active", False)):
        risks.append("writer_active")
    if str(summary.get("guard_policy_mode") or "") == "protective_throttle" or bool(summary.get("guard_triggered", False)):
        risks.append("guard_intelligence_throttle_active")
    if _safe_int(summary.get("guard_blocker_count"), 0) > 0:
        risks.append("guard_intelligence_blockers")
    for row in _as_list(process_contracts.get("contracts")):
        if isinstance(row, dict) and str(row.get("status") or "") == "blocked":
            risks.append(f"contract_blocked:{row.get('name')}")
    return ordered_unique(risks)


def _brain_playbook(action: str, *, pressure_guarded: bool) -> list[dict[str, Any]]:
    if action == "refresh_auth_and_halt_clearance":
        return [
            {"step": "refresh_halt", "command": ["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"]},
            {"step": "refresh_auth", "command": ["./scripts/ops/opsctl.sh", "token-refresh", "--json"]},
            {"step": "rebuild_system_intelligence", "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]},
        ]
    if action == "recover_writer_then_rescore":
        return [
            {"step": "writer_recovery", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--skip-maintenance", "--json"]},
            {"step": "refresh_writer_intelligence", "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"]},
            {"step": "rebuild_system_intelligence", "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]},
        ]
    if action == "relieve_pressure_then_micro_drain":
        return [
            {"step": "pressure_relief", "command": ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]},
            {"step": "micro_drain", "command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", "5000", "--json"]},
            {"step": "rebuild_system_intelligence", "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]},
        ]
    if action == "relieve_pressure_then_observe_backlog":
        return [
            {"step": "pressure_relief", "command": ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]},
            {"step": "refresh_storage", "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]},
            {"step": "rebuild_system_intelligence", "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]},
        ]
    if action == "run_focused_backlog_drain":
        max_waves = "1" if pressure_guarded else "2"
        return [
            {"step": "score_drainers", "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"]},
            {"step": "bounded_drain", "command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", max_waves, "--target-pending-lines", "10000", "--json"]},
            {"step": "refresh_storage", "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]},
        ]
    if action == "refresh_signal_surfaces":
        return [
            {"step": "refresh_fast_health", "command": ["./scripts/ops/opsctl.sh", "health-fast", "--json"]},
            {"step": "refresh_self_model", "command": ["./scripts/ops/opsctl.sh", "system-self-model", "--json"]},
            {"step": "rebuild_system_intelligence", "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]},
        ]
    return [{"step": "observe", "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]}]


def build_system_brain(signal_bus: dict[str, Any], process_contracts: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    risks = _brain_risk_flags(signal_bus, process_contracts)
    pressure_guarded = bool(summary.get("memory_pressure_high", False) or summary.get("runtime_pressure_high", False))
    pending = _safe_int(summary.get("total_pending_lines"), 0)
    material_storage, storage_evidence = _material_storage_backlog(signal_bus)
    top_risk = str(summary.get("top_risk") or "none")
    if "global_halt_active" in risks or "contract_blocked:auth_and_halt" in risks:
        action = "refresh_auth_and_halt_clearance"
        mode = "safety_blocked"
    elif "writer_recovery_required" in risks:
        action = "recover_writer_then_rescore"
        mode = "writer_recovery"
    elif "guard_intelligence_throttle_active" in risks or "guard_intelligence_blockers" in risks:
        action = "refresh_signal_surfaces"
        mode = "guard_stabilization"
    elif pressure_guarded and material_storage:
        action = "relieve_pressure_then_micro_drain"
        mode = "pressure_guarded_drain"
    elif material_storage or "storage_critical" in risks:
        action = "run_focused_backlog_drain"
        mode = "backlog_drain"
    elif pressure_guarded and pending > 0:
        action = "relieve_pressure_then_observe_backlog"
        mode = "pressure_guarded_observation"
    elif _safe_int(summary.get("loaded_signal_count"), 0) < 8:
        action = "refresh_signal_surfaces"
        mode = "thin_signal_bus"
    else:
        action = "observe_and_expand_cautiously"
        mode = "steady_state"

    playbook = _brain_playbook(action, pressure_guarded=pressure_guarded)
    safe_next_command = playbook[0].get("command", []) if playbook else ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]
    status = "ready"
    if mode == "safety_blocked":
        status = "blocked"
    elif risks or str(signal_bus.get("overall_status") or "") in {"degraded", "blocked"}:
        status = "degraded" if "storage_critical" in risks or "writer_recovery_required" in risks else "advisory"
    confidence = 0.48
    if _safe_int(summary.get("loaded_signal_count"), 0) >= 8:
        confidence += 0.18
    if process_contracts.get("contracts"):
        confidence += 0.12
    if risks:
        confidence -= min(0.18, len(risks) * 0.025)
    confidence = round(max(0.15, min(0.94, confidence)), 3)
    do_not_do = ordered_unique(
        [
            "do_not_start_parallel_sql_writers",
            "do_not_add_live_trade_authority",
            "do_not_run_broad_strategy_expansion_under_storage_or_memory_pressure" if pressure_guarded or "storage_critical" in risks else "",
            "do_not_relaunch_live_sleeves_until_halt_clear" if "global_halt_active" in risks else "",
            "do_not_run_heavy_training_until_runtime_pressure_clears" if pressure_guarded else "",
            "do_not_expand_sleeves_while_guard_intelligence_is_throttled" if "guard_intelligence_throttle_active" in risks else "",
        ]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "system_brain",
        "ok": status == "ready",
        "overall_status": status,
        "decision_packet": {
            "action": action,
            "operating_mode": mode,
            "confidence": confidence,
            "top_risk": top_risk,
            "risk_flags": risks,
            "storage_evidence": storage_evidence,
            "safe_next_command": safe_next_command,
            "do_not_do": do_not_do,
            "reason_codes": ordered_unique([top_risk, *risks, "process_contracts_loaded" if process_contracts.get("contracts") else ""]),
        },
        "playbook": playbook,
        "coordination_policy": {
            "reads_signal_bus": True,
            "enforces_process_contracts": True,
            "writes_codex_handoff": True,
            "executes_commands": False,
            "trade_authority": "none",
        },
    }


def _previous_signal_summary(previous: dict[str, Any]) -> dict[str, Any]:
    signal_bus = previous.get("system_signal_bus") if isinstance(previous.get("system_signal_bus"), dict) else {}
    return signal_bus.get("summary") if isinstance(signal_bus.get("summary"), dict) else {}


def _previous_brain_decision(previous: dict[str, Any]) -> dict[str, Any]:
    brain = previous.get("system_brain") if isinstance(previous.get("system_brain"), dict) else {}
    return brain.get("decision_packet") if isinstance(brain.get("decision_packet"), dict) else {}


def _signal_by_name(signal_bus: dict[str, Any], name: str) -> dict[str, Any]:
    for row in _as_list(signal_bus.get("signals")):
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return {}


def _material_storage_backlog(signal_bus: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    summary = _as_dict(signal_bus.get("summary"))
    storage_metrics = _as_dict(_signal_by_name(signal_bus, "ingestion_storage").get("metrics"))
    pending_ratio = _safe_float(storage_metrics.get("pending_ratio"), 0.0)
    pressure_index = _safe_float(storage_metrics.get("pressure_index"), 0.0)
    severity = str(storage_metrics.get("severity") or "").lower()
    material = bool(
        bool(summary.get("storage_critical", False))
        or pending_ratio >= 1.0
        or pressure_index >= 1.0
        or severity in {"critical", "high", "blocked"}
    )
    return material, {
        "pending_lines": _safe_int(summary.get("total_pending_lines"), 0),
        "pending_ratio": pending_ratio,
        "pressure_index": pressure_index,
        "severity": severity,
    }


def _stale_signal_rows(signal_bus: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    critical = {
        "memory_efficiency",
        "runtime_throttle",
        "ingestion_storage",
        "writer_process_intelligence",
        "drainer_intelligence",
        "guard_intelligence",
    }
    for row in _as_list(signal_bus.get("signals")):
        if not isinstance(row, dict) or not bool(row.get("loaded", False)):
            continue
        age = row.get("age_minutes")
        if not isinstance(age, (int, float)):
            continue
        limit = 90.0 if str(row.get("name") or "") in critical else 360.0
        if float(age) > limit:
            rows.append(
                {
                    "name": str(row.get("name") or ""),
                    "age_minutes": round(float(age), 3),
                    "stale_limit_minutes": limit,
                    "status": str(row.get("status") or ""),
                }
            )
    return rows


def _signal_conflicts(signal_bus: dict[str, Any]) -> list[str]:
    conflicts: list[str] = []
    summary = _as_dict(signal_bus.get("summary"))
    writer = _as_dict(_signal_by_name(signal_bus, "writer_process_intelligence").get("metrics"))
    drainer = _as_dict(_signal_by_name(signal_bus, "drainer_intelligence").get("metrics"))
    memory = _signal_by_name(signal_bus, "memory_efficiency")
    memory_metrics = _as_dict(memory.get("metrics"))
    runtime_metrics = _as_dict(_signal_by_name(signal_bus, "runtime_throttle").get("metrics"))
    global_halt = _as_dict(_signal_by_name(signal_bus, "global_halt").get("metrics"))
    process_fanout = _as_dict(_signal_by_name(signal_bus, "process_fanout_guard").get("metrics"))
    guard = _as_dict(_signal_by_name(signal_bus, "guard_intelligence").get("metrics"))
    if (
        str(writer.get("writer_state") or "") == "idle"
        and str(drainer.get("action") or "") in {"verify_writer_progress_then_re_score", "run_writer_recovery_check_then_re_score"}
    ):
        conflicts.append("drainer_waits_on_writer_after_writer_idle")
    memory_state = str(memory_metrics.get("memory_pressure_state") or "").lower()
    runtime_memory_level = str(runtime_metrics.get("memory_pressure_level") or "").lower()
    if memory_state in {"green", "none", "normal"} and runtime_memory_level in {"high", "critical"}:
        conflicts.append("memory_pressure_color_conflicts_with_runtime_throttle")
    if memory_state in {"red", "critical"} and runtime_memory_level in {"clear", "low", "normal"}:
        conflicts.append("memory_pressure_color_conflicts_with_runtime_throttle")
    if not bool(global_halt.get("halt_active", False)) and _as_list(global_halt.get("clear_blockers")):
        conflicts.append("halt_clear_blockers_present_without_active_halt")
    if (
        bool(process_fanout.get("triggered", False))
        and _safe_int(process_fanout.get("targetable_process_count"), 0) <= 0
        and not bool(process_fanout.get("core_sleeve_restart_allowed", False))
    ):
        conflicts.append("fanout_guard_holding_without_targetable_processes")
    if (
        str(guard.get("policy_mode") or "") == "full_schwab_observe"
        and bool(process_fanout.get("triggered", False))
        and _safe_int(process_fanout.get("targetable_process_count"), 0) > 0
        and (
            bool(guard.get("triggered", False))
            or _safe_float(guard.get("pressure_score"), 0.0) >= 0.8
            or str(guard.get("process_fanout_guard_active") or "") == "1"
        )
    ):
        conflicts.append("guard_full_observe_conflicts_with_active_fanout_trigger")
    if bool(summary.get("writer_active", False)) and str(writer.get("writer_state") or "") == "idle":
        conflicts.append("writer_active_summary_conflicts_with_writer_state")
    return ordered_unique(conflicts)


def _trend_from_previous(signal_bus: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current = _as_dict(signal_bus.get("summary"))
    previous_summary = _previous_signal_summary(previous)
    current_pending = _safe_int(current.get("total_pending_lines"), 0)
    previous_pending = _safe_int(previous_summary.get("total_pending_lines"), 0)
    current_top = str(current.get("top_risk") or "")
    previous_top = str(previous_summary.get("top_risk") or "")
    storage = _as_dict(_signal_by_name(signal_bus, "ingestion_storage").get("metrics"))
    previous_storage = _as_dict(_signal_by_name(_as_dict(previous.get("system_signal_bus")), "ingestion_storage").get("metrics"))
    current_pressure = _safe_float(storage.get("pressure_index"), 0.0)
    previous_pressure = _safe_float(previous_storage.get("pressure_index"), 0.0)
    pending_delta = current_pending - previous_pending
    pressure_delta = round(current_pressure - previous_pressure, 6)
    if not previous:
        trajectory = "baseline"
    elif pending_delta <= -250 or pressure_delta <= -0.25:
        trajectory = "improving"
    elif pending_delta >= 250 or pressure_delta >= 0.25:
        trajectory = "worsening"
    else:
        trajectory = "flat"
    return {
        "trajectory": trajectory,
        "pending_lines_delta": int(pending_delta),
        "pressure_index_delta": pressure_delta,
        "top_risk_changed": bool(previous_top and current_top and previous_top != current_top),
        "previous_top_risk": previous_top,
        "current_top_risk": current_top,
        "previous_pending_lines": int(previous_pending),
        "current_pending_lines": int(current_pending),
    }


def _memory_summary(memory_events: list[dict[str, Any]], current_action: str) -> dict[str, Any]:
    actions = [str(row.get("action") or "") for row in memory_events if str(row.get("action") or "")]
    top_risks = [str(row.get("top_risk") or "") for row in memory_events if str(row.get("top_risk") or "")]
    repeated = 0
    for action in reversed(actions):
        if action != current_action:
            break
        repeated += 1
    return {
        "memory_event_count": len(memory_events),
        "recent_actions": actions[-8:],
        "recent_top_risks": top_risks[-8:],
        "same_action_repeat_count": int(repeated),
        "current_action_seen_count": sum(1 for action in actions if action == current_action),
    }


def _action_effect_summary(
    memory_events: list[dict[str, Any]],
    *,
    current_event: dict[str, Any],
    trend: dict[str, Any],
) -> dict[str, Any]:
    current_action = str(current_event.get("action") or "")
    if not current_action:
        return {
            "current_action": "",
            "same_action_run_length": 0,
            "history_event_count": len(memory_events),
            "verdict": "insufficient_history",
            "evidence": ["no_current_action"],
        }
    ordered_events = [row for row in memory_events if isinstance(row, dict)]
    ordered_events.append(current_event)
    run: list[dict[str, Any]] = []
    for row in reversed(ordered_events):
        if str(row.get("action") or "") != current_action:
            break
        run.append(row)
    run = list(reversed(run))
    first = run[0] if run else current_event
    latest = run[-1] if run else current_event
    first_pending = _safe_int(first.get("pending_lines"), _safe_int(current_event.get("pending_lines"), 0))
    latest_pending = _safe_int(latest.get("pending_lines"), _safe_int(current_event.get("pending_lines"), 0))
    pending_delta = latest_pending - first_pending
    trajectory = str(trend.get("trajectory") or "")
    completed_history_count = max(len(run) - 1, 0)

    if completed_history_count <= 0:
        verdict = "insufficient_history"
    elif pending_delta <= -250 or trajectory == "improving":
        verdict = "effective"
    elif pending_delta >= 250 or trajectory == "worsening":
        verdict = "worsening"
    elif completed_history_count >= 2 and abs(pending_delta) < 250:
        verdict = "ineffective_so_far"
    else:
        verdict = "monitoring"

    return {
        "current_action": current_action,
        "same_action_run_length": len(run),
        "completed_history_count": completed_history_count,
        "history_event_count": len(memory_events),
        "first_pending_lines": int(first_pending),
        "latest_pending_lines": int(latest_pending),
        "pending_lines_delta": int(pending_delta),
        "trajectory": trajectory,
        "verdict": verdict,
        "evidence": ordered_unique(
            [
                f"same_action_run_length={len(run)}",
                f"completed_history_count={completed_history_count}",
                f"pending_lines_delta={pending_delta}",
                f"trajectory={trajectory}",
            ]
        ),
    }


def _causal_diagnosis(
    *,
    signal_bus: dict[str, Any],
    trend: dict[str, Any],
    memory: dict[str, Any],
    action_effectiveness: dict[str, Any],
    uncertainty: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    storage = _as_dict(_signal_by_name(signal_bus, "ingestion_storage").get("metrics"))
    memory_metrics = _as_dict(_signal_by_name(signal_bus, "memory_efficiency").get("metrics"))
    runtime = _as_dict(_signal_by_name(signal_bus, "runtime_throttle").get("metrics"))
    fanout = _as_dict(_signal_by_name(signal_bus, "process_fanout_guard").get("metrics"))
    watchdog = _as_dict(_signal_by_name(signal_bus, "process_watchdog").get("metrics"))
    guard = _as_dict(_signal_by_name(signal_bus, "guard_intelligence").get("metrics"))

    root_causes: list[str] = []
    symptoms: list[str] = []
    not_root_causes: list[str] = []
    evidence: list[str] = []
    pending = _safe_int(summary.get("total_pending_lines"), _safe_int(storage.get("total_pending_lines"), 0))

    if bool(summary.get("storage_critical", False)) or str(summary.get("top_risk") or "") == "ingestion_storage":
        root_causes.append("storage_backpressure_primary")
        evidence.append(f"storage_pending_lines={pending}")
        evidence.append(f"storage_pressure_index={storage.get('pressure_index', 0)}")

    if bool(summary.get("memory_pressure_high", False)):
        if "storage_backpressure_primary" in root_causes:
            symptoms.append("memory_pressure_is_pressure_amplifier")
        else:
            root_causes.append("memory_pressure_primary")
        evidence.append(
            f"memory_state={memory_metrics.get('memory_pressure_state', '')}:{memory_metrics.get('memory_pressure_kind', '')}"
        )
    elif _memory_metrics_clear(memory_metrics):
        not_root_causes.append("memory_pressure_not_primary")

    runtime_memory = str(runtime.get("memory_pressure_level") or "").lower()
    throttle_profile = str(runtime.get("throttle_profile") or "").lower()
    if throttle_profile == "protect_live" and runtime_memory in {"", "normal", "low", "clear"}:
        symptoms.append("runtime_protect_live_due_to_storage_not_memory")
        not_root_causes.append("runtime_memory_pressure_not_primary")
    elif bool(summary.get("runtime_pressure_high", False)) and "storage_backpressure_primary" not in root_causes:
        root_causes.append("runtime_pressure_primary")

    if (
        bool(fanout.get("triggered", False))
        and _safe_int(fanout.get("targetable_process_count"), 0) <= 0
        and bool(fanout.get("core_sleeve_restart_allowed", False))
    ):
        symptoms.append("fanout_hold_is_sleeve_safe_symptom")
        not_root_causes.append("process_fanout_not_primary_when_no_targetable_processes")

    if str(guard.get("policy_mode") or "") == "protective_throttle":
        if not root_causes:
            root_causes.append("guard_throttle_primary")
        else:
            symptoms.append("guard_throttle_is_safety_amplifier")
        evidence.append(f"guard_policy_mode={guard.get('policy_mode', '')}")
        evidence.append(f"guard_pressure_score={guard.get('pressure_score', 0)}")

    if not _as_list(watchdog.get("down_processes")) and _safe_int(watchdog.get("restarted_count"), 0) >= 0:
        symptoms.append("sleeve_supervisor_clear")

    if str(action_effectiveness.get("verdict") or "") in {"ineffective_so_far", "worsening"}:
        root_causes.append("pressure_playbook_not_reducing_backlog")
        evidence.append(f"action_effect={action_effectiveness.get('verdict')}")

    if not root_causes:
        top_risk = str(summary.get("top_risk") or "none")
        if top_risk and top_risk != "none":
            root_causes.append(f"{top_risk}_primary")
        else:
            root_causes.append("stable_or_observing")

    uncertainty_score = _safe_int(uncertainty.get("score"), 0)
    base_confidence = 0.84 if "storage_backpressure_primary" in root_causes else 0.68
    if str(action_effectiveness.get("verdict") or "") == "insufficient_history":
        base_confidence -= 0.05
    if uncertainty_score >= 60:
        base_confidence -= 0.18
    elif uncertainty_score >= 25:
        base_confidence -= 0.08
    confidence = round(max(0.25, min(0.93, base_confidence)), 3)

    return {
        "primary_root_cause": root_causes[0],
        "confidence": confidence,
        "root_causes": ordered_unique(root_causes),
        "symptoms": ordered_unique(symptoms),
        "not_root_causes": ordered_unique(not_root_causes),
        "evidence": ordered_unique(
            [
                *evidence,
                f"trend={trend.get('trajectory', '')}",
                f"same_action_repeat_count={memory.get('same_action_repeat_count', 0)}",
                f"uncertainty={uncertainty.get('level', '')}:{uncertainty_score}",
            ]
        ),
    }


def _integration_routing(
    *,
    signal_bus: dict[str, Any],
    causal_diagnosis: dict[str, Any],
    action_effectiveness: dict[str, Any],
    reflex: dict[str, Any],
) -> dict[str, Any]:
    primary_root = str(causal_diagnosis.get("primary_root_cause") or "")
    if bool(reflex.get("blocks_brain_action_until_refreshed", False)):
        route_mode = "precheck_refresh_first"
        primary_owner = "system_self_intelligence"
        refresh_order = [
            reflex.get("command") if isinstance(reflex.get("command"), list) else [],
            reflex.get("followup_command") if isinstance(reflex.get("followup_command"), list) else [],
            ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
        ]
    elif primary_root == "storage_backpressure_primary":
        route_mode = "storage_first_recovery"
        primary_owner = "backpressure_storage_brain_v2"
        refresh_order = [
            ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", "5000", "--json"],
            ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
        ]
    elif primary_root == "memory_pressure_primary":
        route_mode = "resource_first_recovery"
        primary_owner = "runtime_throttle_control"
        refresh_order = [
            ["./scripts/ops/opsctl.sh", "memory-efficiency", "status", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--json"],
            ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
        ]
    elif primary_root == "pressure_playbook_not_reducing_backlog":
        route_mode = "playbook_rethink"
        primary_owner = "drainer_intelligence_layer"
        refresh_order = [
            ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
        ]
    elif primary_root == "guard_throttle_primary":
        route_mode = "guard_first_stabilization"
        primary_owner = "guard_intelligence_layer"
        refresh_order = [
            ["./scripts/ops/opsctl.sh", "guard-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "process-fanout-guard", "--json"],
            ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
        ]
    else:
        route_mode = "observe_and_refresh"
        primary_owner = str(_as_dict(signal_bus.get("summary")).get("top_risk") or "system_brain")
        refresh_order = [["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]]

    return {
        "route_mode": route_mode,
        "primary_owner": primary_owner,
        "consumers": [
            "system_brain",
            "codex_handoff",
            "drainer_intelligence",
            "writer_process_intelligence",
            "guard_intelligence",
            "process_fanout_guard",
            "operator_cockpit",
            "system_self_model",
        ],
        "refresh_order": [row for row in refresh_order if row],
        "action_effect_verdict": str(action_effectiveness.get("verdict") or ""),
        "coordination_policy": {
            "trade_authority": "none",
            "single_sql_writer_only": True,
            "bounded_apply_modes_only": True,
            "heavy_expansion_paused_under_pressure": True,
        },
    }


def _capability_gaps(
    *,
    uncertainty: dict[str, Any],
    action_effectiveness: dict[str, Any],
    causal_diagnosis: dict[str, Any],
    integration_routing: dict[str, Any],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    missing = {str(item) for item in _as_list(uncertainty.get("missing_signals"))}
    missing_commands = {
        "guard_intelligence": ["./scripts/ops/opsctl.sh", "guard-intelligence", "--apply", "--json"],
        "platform_brain_v6": ["./scripts/ops/opsctl.sh", "platform-brain-v6", "--json"],
        "mlx_intelligence_router": ["./scripts/ops/opsctl.sh", "mlx-intelligence-router", "--json"],
        "library_utilization_router": ["./scripts/ops/opsctl.sh", "library-utilization-router", "--json"],
    }
    for signal, command in missing_commands.items():
        if signal in missing:
            gaps.append(
                {
                    "gap": f"refresh_{signal}",
                    "why": "required_intelligence_surface_missing_from_signal_bus",
                    "suggested_command": command,
                }
            )
    verdict = str(action_effectiveness.get("verdict") or "")
    if verdict in {"ineffective_so_far", "worsening"}:
        gaps.append(
            {
                "gap": "add_drain_outcome_verifier",
                "why": "repeated_action_is_not_showing_clear_backlog_reduction",
                "suggested_consumer": str(integration_routing.get("primary_owner") or "drainer_intelligence_layer"),
            }
        )
        gaps.append(
            {
                "gap": "change_pressure_playbook",
                "why": "same_action_memory_indicates_the_current_playbook_needs_a_different_bounded_move",
                "suggested_consumer": "system_brain",
            }
        )
    if str(causal_diagnosis.get("primary_root_cause") or "") == "storage_backpressure_primary":
        gaps.append(
            {
                "gap": "persist_storage_causal_replay_memory",
                "why": "storage_pressure_is_the_primary_root_and_should_be_replay_scored_across_drain_attempts",
                "suggested_consumer": "system_self_model",
            }
        )
    return gaps


def _contract_violations(process_contracts: dict[str, Any], signal_bus: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    global_contract = _as_dict(process_contracts.get("global_safety_contract"))
    writer = _as_dict(_signal_by_name(signal_bus, "writer_process_intelligence").get("metrics"))
    if bool(global_contract.get("parallel_sql_writers_allowed", True)):
        violations.append("parallel_sql_writers_allowed_by_global_contract")
    if bool(global_contract.get("live_trade_authority_added", True)):
        violations.append("live_trade_authority_added_by_global_contract")
    if not bool(global_contract.get("single_sql_writer_only", False)):
        violations.append("single_sql_writer_contract_missing")
    if bool(writer.get("starts_parallel_sql_writers", False)):
        violations.append("writer_metrics_allow_parallel_sql_writers")
    for row in _as_list(process_contracts.get("contracts")):
        if isinstance(row, dict) and str(row.get("status") or "") == "blocked":
            violations.append(f"blocked_contract:{row.get('name')}")
    return ordered_unique(violations)


def _self_reflex(
    *,
    trend: dict[str, Any],
    uncertainty: dict[str, Any],
    memory: dict[str, Any],
    brain_decision: dict[str, Any],
) -> dict[str, Any]:
    conflicts = [str(item) for item in _as_list(uncertainty.get("conflicting_signals"))]
    stale = [row for row in _as_list(uncertainty.get("stale_signals")) if isinstance(row, dict)]
    same_action_repeat = _safe_int(memory.get("same_action_repeat_count"), 0)
    trajectory = str(trend.get("trajectory") or "")
    if "drainer_waits_on_writer_after_writer_idle" in conflicts:
        return {
            "action": "refresh_drainer_intelligence_before_apply",
            "reason": "drainer_intelligence_is_still_waiting_on_a_writer_that_now_reports_idle",
            "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"],
            "blocks_brain_action_until_refreshed": True,
        }
    if "guard_full_observe_conflicts_with_active_fanout_trigger" in conflicts:
        return {
            "action": "refresh_guard_intelligence_before_expansion",
            "reason": "guard_intelligence_and_process_fanout_disagree_on_whether_expansion_is_safe",
            "command": ["./scripts/ops/opsctl.sh", "guard-intelligence", "--apply", "--json"],
            "followup_command": ["./scripts/ops/opsctl.sh", "process-fanout-guard", "--json"],
            "blocks_brain_action_until_refreshed": True,
        }
    critical_stale = [row for row in stale if str(row.get("name") or "") in {"memory_efficiency", "runtime_throttle", "ingestion_storage"}]
    if critical_stale:
        return {
            "action": "refresh_stale_pressure_surfaces",
            "reason": "critical_pressure_inputs_are_stale_enough_to_distort_the_next_action",
            "command": ["./scripts/ops/opsctl.sh", "memory-efficiency", "status", "--json"],
            "followup_command": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--json"],
            "blocks_brain_action_until_refreshed": True,
        }
    if same_action_repeat >= 3 and trajectory in {"baseline", "flat", "worsening"}:
        return {
            "action": "escalate_repeated_action_not_clearing_pressure",
            "reason": "the_same_recommendation_has_repeated_without_visible_clearance",
            "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
            "blocks_brain_action_until_refreshed": False,
        }
    return {
        "action": "follow_system_brain",
        "reason": "no_self_intelligence_precheck_needed",
        "command": brain_decision.get("safe_next_command") if isinstance(brain_decision.get("safe_next_command"), list) else [],
        "blocks_brain_action_until_refreshed": False,
    }


def build_self_intelligence(
    *,
    signal_bus: dict[str, Any],
    system_brain: dict[str, Any],
    process_contracts: dict[str, Any],
    previous_payload: dict[str, Any],
    memory_events: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = _as_dict(system_brain.get("decision_packet"))
    trend = _trend_from_previous(signal_bus, previous_payload)
    stale_signals = _stale_signal_rows(signal_bus)
    missing_signals = [
        str(row.get("name") or "")
        for row in _as_list(signal_bus.get("signals"))
        if isinstance(row, dict) and not bool(row.get("loaded", False))
    ]
    conflicts = _signal_conflicts(signal_bus)
    violations = _contract_violations(process_contracts, signal_bus)
    memory_summary = _memory_summary(memory_events, str(decision.get("action") or ""))
    uncertainty_score = min(
        100,
        len(missing_signals) * 6 + len(stale_signals) * 8 + len(conflicts) * 12 + len(violations) * 20,
    )
    uncertainty = {
        "score": int(uncertainty_score),
        "level": "high" if uncertainty_score >= 60 else "medium" if uncertainty_score >= 25 else "low",
        "missing_signals": missing_signals,
        "stale_signals": stale_signals,
        "conflicting_signals": conflicts,
        "contract_violations": violations,
    }
    reflex = _self_reflex(
        trend=trend,
        uncertainty=uncertainty,
        memory=memory_summary,
        brain_decision=decision,
    )
    storage_metrics = _as_dict(_signal_by_name(signal_bus, "ingestion_storage").get("metrics"))
    memory_event_base = {
        "timestamp_utc": iso_now(),
        "status": str(system_brain.get("overall_status") or ""),
        "action": str(decision.get("action") or ""),
        "top_risk": str(decision.get("top_risk") or ""),
        "risk_flags": [str(item) for item in _as_list(decision.get("risk_flags"))],
        "pending_lines": _safe_int(_as_dict(signal_bus.get("summary")).get("total_pending_lines"), 0),
        "pressure_index": _safe_float(storage_metrics.get("pressure_index"), 0.0),
        "trajectory": str(trend.get("trajectory") or ""),
        "uncertainty_level": str(uncertainty.get("level") or ""),
        "reflex_action": str(reflex.get("action") or ""),
    }
    action_effectiveness = _action_effect_summary(
        memory_events,
        current_event=memory_event_base,
        trend=trend,
    )
    causal_diagnosis = _causal_diagnosis(
        signal_bus=signal_bus,
        trend=trend,
        memory=memory_summary,
        action_effectiveness=action_effectiveness,
        uncertainty=uncertainty,
    )
    integration_routing = _integration_routing(
        signal_bus=signal_bus,
        causal_diagnosis=causal_diagnosis,
        action_effectiveness=action_effectiveness,
        reflex=reflex,
    )
    capability_gaps = _capability_gaps(
        uncertainty=uncertainty,
        action_effectiveness=action_effectiveness,
        causal_diagnosis=causal_diagnosis,
        integration_routing=integration_routing,
    )
    questions = []
    for signal in missing_signals[:4]:
        questions.append(f"Should {signal} be refreshed before trusting the next action?")
    for conflict in conflicts[:4]:
        questions.append(f"Resolve signal conflict: {conflict}")
    if memory_summary["same_action_repeat_count"] >= 3:
        questions.append("Is the repeated action reducing pressure, or should the playbook change?")
    if str(action_effectiveness.get("verdict") or "") in {"ineffective_so_far", "worsening"}:
        questions.append("Should the drainer playbook change because repeated actions are not clearing the backlog?")
    if str(causal_diagnosis.get("primary_root_cause") or "") != "stable_or_observing":
        questions.append(f"Route next work through {integration_routing.get('primary_owner')} for {causal_diagnosis.get('primary_root_cause')}.")
    if not questions:
        questions.append("No blocking self-question; continue monitoring outcome after the next safe action.")
    status = "ready"
    if violations:
        status = "blocked"
    elif uncertainty_score >= 60 or bool(reflex.get("blocks_brain_action_until_refreshed", False)):
        status = "degraded"
    elif uncertainty_score >= 25:
        status = "advisory"
    memory_event = {
        **memory_event_base,
        "causal_root": str(causal_diagnosis.get("primary_root_cause") or ""),
        "action_effect_verdict": str(action_effectiveness.get("verdict") or ""),
        "integration_route": str(integration_routing.get("route_mode") or ""),
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "system_self_intelligence",
        "ok": status == "ready",
        "overall_status": status,
        "trend": trend,
        "uncertainty": uncertainty,
        "learning_memory": memory_summary,
        "action_effectiveness": action_effectiveness,
        "causal_diagnosis": causal_diagnosis,
        "integration_routing": integration_routing,
        "capability_gaps": capability_gaps,
        "reflex": reflex,
        "self_questions": questions,
        "memory_event": memory_event,
        "self_intelligence_contract": {
            "purpose": "compare_current_state_to_prior_runs_detect_uncertainty_score_action_effects_diagnose_causes_and_route_next_consumers",
            "does_not_execute_commands": True,
            "does_not_trade": True,
            "may_block_brain_action_with_precheck": True,
            "memory_path": str(DEFAULT_SELF_MEMORY_PATH),
        },
    }


def _super_memory_summary(memory_events: list[dict[str, Any]], current_action: str) -> dict[str, Any]:
    actions = [str(row.get("action") or "") for row in memory_events if str(row.get("action") or "")]
    modes = [str(row.get("executive_mode") or "") for row in memory_events if str(row.get("executive_mode") or "")]
    top_attention = [str(row.get("top_attention") or "") for row in memory_events if str(row.get("top_attention") or "")]
    repeated = 0
    for action in reversed(actions):
        if action != current_action:
            break
        repeated += 1
    return {
        "memory_event_count": len(memory_events),
        "recent_actions": actions[-8:],
        "recent_executive_modes": modes[-8:],
        "recent_top_attention": top_attention[-8:],
        "same_action_repeat_count": int(repeated),
        "current_action_seen_count": sum(1 for action in actions if action == current_action),
    }


def _attention_node(
    *,
    node_id: str,
    layer: str,
    severity: int,
    status: str,
    reason: str,
    command: list[str] | None = None,
    owner: str = "",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "layer": layer,
        "severity_score": int(max(0, min(100, severity))),
        "status": status,
        "reason": reason,
        "owner": owner,
        "suggested_command": command or [],
    }


def _super_attention_graph(
    *,
    signal_bus: dict[str, Any],
    system_brain: dict[str, Any],
    process_contracts: dict[str, Any],
    self_intelligence: dict[str, Any],
) -> list[dict[str, Any]]:
    decision = _as_dict(system_brain.get("decision_packet"))
    reflex = _as_dict(self_intelligence.get("reflex"))
    uncertainty = _as_dict(self_intelligence.get("uncertainty"))
    causal = _as_dict(self_intelligence.get("causal_diagnosis"))
    nodes: list[dict[str, Any]] = []
    for row in _as_list(signal_bus.get("signals"))[:8]:
        if not isinstance(row, dict):
            continue
        severity = _safe_int(row.get("severity_score"), 0)
        if severity < 25 and str(row.get("name") or "") not in {"guard_intelligence", "platform_brain_v6"}:
            continue
        nodes.append(
            _attention_node(
                node_id=str(row.get("name") or ""),
                layer=str(row.get("category") or "signal"),
                severity=severity,
                status=str(row.get("status") or ""),
                reason=str(row.get("summary") or row.get("name") or ""),
                owner=str(row.get("category") or ""),
            )
        )
    if bool(reflex.get("blocks_brain_action_until_refreshed", False)):
        nodes.append(
            _attention_node(
                node_id=str(reflex.get("action") or "self_reflex_precheck"),
                layer="self_intelligence",
                severity=92,
                status="degraded",
                reason=str(reflex.get("reason") or ""),
                command=reflex.get("command") if isinstance(reflex.get("command"), list) else [],
                owner="system_self_intelligence",
            )
        )
    if str(causal.get("primary_root_cause") or "") and str(causal.get("primary_root_cause") or "") != "stable_or_observing":
        nodes.append(
            _attention_node(
                node_id=str(causal.get("primary_root_cause") or ""),
                layer="causal_model",
                severity=82 if str(self_intelligence.get("overall_status") or "") in {"degraded", "blocked"} else 58,
                status=str(self_intelligence.get("overall_status") or ""),
                reason="primary_root_cause",
                owner=str(_as_dict(self_intelligence.get("integration_routing")).get("primary_owner") or ""),
            )
        )
    for row in _as_list(process_contracts.get("contracts")):
        if not isinstance(row, dict) or str(row.get("status") or "") == "ready":
            continue
        nodes.append(
            _attention_node(
                node_id=f"contract:{row.get('name')}",
                layer="process_contract",
                severity=88 if str(row.get("status") or "") == "blocked" else 54,
                status=str(row.get("status") or ""),
                reason=",".join(str(item) for item in _as_list(row.get("active_risks"))) or "contract_advisory",
                owner=str(row.get("owner") or ""),
            )
        )
    if str(uncertainty.get("level") or "") in {"medium", "high"}:
        nodes.append(
            _attention_node(
                node_id="uncertainty",
                layer="self_intelligence",
                severity=_safe_int(uncertainty.get("score"), 0),
                status=str(uncertainty.get("level") or ""),
                reason="missing_stale_or_conflicting_signals",
                owner="system_self_intelligence",
            )
        )
    return sorted(nodes, key=lambda row: (_safe_int(row.get("severity_score"), 0), str(row.get("node_id") or "")), reverse=True)


def _super_adaptive_policy(signal_bus: dict[str, Any], system_brain: dict[str, Any], self_intelligence: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    decision = _as_dict(system_brain.get("decision_packet"))
    action_effect = _as_dict(self_intelligence.get("action_effectiveness"))
    guard_mode = str(summary.get("guard_policy_mode") or "")
    pressure_guarded = bool(summary.get("memory_pressure_high", False) or summary.get("runtime_pressure_high", False))
    storage_critical = bool(summary.get("storage_critical", False))
    if guard_mode == "protective_throttle":
        sleeve_posture = "stabilize_core_sleeves_only"
        expansion_posture = "closed"
        training_posture = "paused"
    elif pressure_guarded or storage_critical:
        sleeve_posture = "protect_collection_and_drain"
        expansion_posture = "catalog_only"
        training_posture = "microbatch_or_off_hours_only"
    elif guard_mode == "full_schwab_observe":
        sleeve_posture = "full_schwab_observe"
        expansion_posture = "rehearsal_then_admit"
        training_posture = "bounded_targeted_allowed"
    else:
        sleeve_posture = "balanced_guarded"
        expansion_posture = "rehearsal_only"
        training_posture = "bounded_targeted_allowed"
    return {
        "guard_policy_mode": guard_mode,
        "sleeve_posture": sleeve_posture,
        "expansion_posture": expansion_posture,
        "training_posture": training_posture,
        "drainer_posture": "single_wave_until_backlog_clears" if storage_critical else "bounded_scored_waves",
        "writer_posture": "single_sql_writer_only",
        "research_posture": "reroute_playbook" if str(action_effect.get("verdict") or "") in {"ineffective_so_far", "worsening"} else "observe_outcomes",
        "brain_action": str(decision.get("action") or ""),
    }


def _super_decision_packet(
    *,
    signal_bus: dict[str, Any],
    system_brain: dict[str, Any],
    self_intelligence: dict[str, Any],
    attention_graph: list[dict[str, Any]],
    adaptive_policy: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    brain_decision = _as_dict(system_brain.get("decision_packet"))
    reflex = _as_dict(self_intelligence.get("reflex"))
    action_effect = _as_dict(self_intelligence.get("action_effectiveness"))
    top = attention_graph[0] if attention_graph else {}
    material_storage, storage_evidence = _material_storage_backlog(signal_bus)
    safe_next_command = brain_decision.get("safe_next_command") if isinstance(brain_decision.get("safe_next_command"), list) else []
    executive_mode = "observe"
    action = "observe_and_keep_collecting"
    owner = "system_brain"
    reason_codes = [str(summary.get("top_risk") or "none")]

    if bool(reflex.get("blocks_brain_action_until_refreshed", False)):
        executive_mode = "precheck"
        action = "refresh_precheck_surfaces"
        owner = "system_self_intelligence"
        safe_next_command = reflex.get("command") if isinstance(reflex.get("command"), list) else safe_next_command
        reason_codes.append(str(reflex.get("action") or "self_reflex"))
    elif str(system_brain.get("overall_status") or "") == "blocked" or bool(summary.get("global_halt_active", False)):
        executive_mode = "safety"
        action = "recover_safety_clearance"
        owner = "auth_and_halt"
        reason_codes.append("safety_blocked")
    elif str(summary.get("guard_policy_mode") or "") == "protective_throttle" or bool(summary.get("guard_triggered", False)):
        executive_mode = "stabilize"
        action = "stabilize_guard_and_process_budget"
        owner = "guard_intelligence_layer"
        safe_next_command = ["./scripts/ops/opsctl.sh", "guard-intelligence", "--apply", "--json"]
        reason_codes.append("guard_intelligence_active")
    elif str(action_effect.get("verdict") or "") in {"ineffective_so_far", "worsening"}:
        executive_mode = "rethink"
        action = "reroute_stalled_playbook"
        owner = "drainer_intelligence_layer"
        safe_next_command = ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"]
        reason_codes.append(str(action_effect.get("verdict") or "action_effect"))
    elif material_storage:
        executive_mode = "drain"
        action = str(brain_decision.get("action") or "run_focused_backlog_drain")
        owner = str(_as_dict(self_intelligence.get("integration_routing")).get("primary_owner") or "backpressure_super_drainer")
        reason_codes.append("storage_or_backlog_present")
    elif bool(summary.get("memory_pressure_high", False)) or bool(summary.get("runtime_pressure_high", False)):
        executive_mode = "stabilize"
        action = str(brain_decision.get("action") or "relieve_pressure_then_observe_backlog")
        owner = "runtime_throttle_control"
        reason_codes.append("resource_pressure_primary")
    elif str(adaptive_policy.get("expansion_posture") or "") == "rehearsal_then_admit":
        executive_mode = "expand"
        action = "cautious_expansion_rehearsal"
        owner = "expansion_capacity"
        safe_next_command = ["./scripts/ops/opsctl.sh", "expansion-capacity", "--json"]
        reason_codes.append("guard_and_pressure_clear")
    confidence = _safe_float(brain_decision.get("confidence"), 0.45)
    if attention_graph:
        confidence -= min(0.18, max(_safe_int(top.get("severity_score"), 0) - 65, 0) / 500.0)
    if str(_as_dict(self_intelligence.get("uncertainty")).get("level") or "") == "low":
        confidence += 0.08
    return {
        "action": action,
        "executive_mode": executive_mode,
        "confidence": round(max(0.12, min(0.94, confidence)), 3),
        "owner": owner,
        "top_attention": str(top.get("node_id") or summary.get("top_risk") or "none"),
        "safe_next_command": safe_next_command,
        "storage_evidence": storage_evidence,
        "reason_codes": ordered_unique([*reason_codes, str(top.get("node_id") or "")]),
        "blocked_until": [
            "single_writer_contract_clear",
            "guard_intelligence_not_protective",
        ]
        if executive_mode in {"safety", "stabilize"}
        else [],
    }


def _command_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command or "")


def _super_command_risk(command: Any) -> dict[str, Any]:
    text = _command_text(command)
    command_tokens = [str(item) for item in command] if isinstance(command, list) else text.split()
    allowed_prefixes = (
        *SAFE_REFLEX_PREFIXES,
        "./scripts/ops/opsctl.sh system-intelligence",
        "./scripts/ops/opsctl.sh guard-intelligence",
        "./scripts/ops/opsctl.sh drainer-intelligence-layer",
        "./scripts/ops/opsctl.sh expansion-capacity",
        "./scripts/ops/opsctl.sh platform-brain-v6",
        "./scripts/ops/opsctl.sh bot-logs-cleanup-intelligence",
    )
    unsafe_markers = (
        "force-clear",
        "force_clear",
        "start-live",
        "live-order",
        "live_order",
        "merge-live",
        "storage-switch",
        "storage-safe-eject",
    )
    if not text:
        return {"risk": "none", "allowed": True, "reason": "no_command_selected", "command": ""}
    if any(marker in text for marker in unsafe_markers):
        return {"risk": "unsafe", "allowed": False, "reason": "command_contains_unsafe_marker", "command": text}
    if "--apply" in command_tokens and not any(text.startswith(prefix) for prefix in allowed_prefixes):
        return {"risk": "review", "allowed": False, "reason": "apply_command_not_in_super_allowlist", "command": text}
    if any(text.startswith(prefix) for prefix in allowed_prefixes):
        return {"risk": "bounded", "allowed": True, "reason": "command_matches_bounded_super_allowlist", "command": text}
    return {"risk": "observe", "allowed": True, "reason": "read_only_or_unknown_command_observe_only", "command": text}


def _super_regime_drift_audit(
    *,
    signal_bus: dict[str, Any],
    self_intelligence: dict[str, Any],
    memory_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    uncertainty = _as_dict(self_intelligence.get("uncertainty"))
    action_effect = _as_dict(self_intelligence.get("action_effectiveness"))
    causal = _as_dict(self_intelligence.get("causal_diagnosis"))
    material_storage_backlog, storage_evidence = _material_storage_backlog(signal_bus)
    if bool(summary.get("global_halt_active", False)):
        regime = "safety_halt"
    elif str(summary.get("guard_policy_mode") or "") == "protective_throttle":
        regime = "guard_throttle"
    elif material_storage_backlog:
        regime = "storage_backpressure"
    elif bool(summary.get("memory_pressure_high", False)) or bool(summary.get("runtime_pressure_high", False)):
        regime = "resource_pressure"
    elif str(summary.get("guard_policy_mode") or "") == "full_schwab_observe":
        regime = "expansion_rehearsal_ready"
    else:
        regime = "steady_observation"

    drift_alerts = []
    if str(action_effect.get("verdict") or "") in {"ineffective_so_far", "worsening"}:
        drift_alerts.append("action_effect_drift")
    if _safe_int(memory_summary.get("same_action_repeat_count"), 0) >= 3:
        drift_alerts.append("executive_action_loop")
    if str(uncertainty.get("level") or "") == "high":
        drift_alerts.append("high_uncertainty_surface")
    if str(causal.get("primary_root_cause") or "") == "stable_or_observing" and regime not in {"steady_observation", "expansion_rehearsal_ready"}:
        drift_alerts.append("causal_model_underexplains_current_regime")

    shadow_models = [
        {
            "name": "simple_pressure_baseline",
            "vote": "drain_or_stabilize" if regime in {"storage_backpressure", "resource_pressure", "guard_throttle"} else "observe",
            "agreement": regime not in {"steady_observation"} or str(action_effect.get("verdict") or "") != "worsening",
        },
        {
            "name": "last_action_baseline",
            "vote": "change_playbook" if _safe_int(memory_summary.get("same_action_repeat_count"), 0) >= 3 else "continue_with_verification",
            "agreement": str(action_effect.get("verdict") or "") not in {"ineffective_so_far", "worsening"},
        },
        {
            "name": "causal_root_baseline",
            "vote": str(causal.get("primary_root_cause") or "unknown"),
            "agreement": bool(causal.get("primary_root_cause")),
        },
    ]
    status = "degraded" if drift_alerts else "ready"
    return {
        "overall_status": status,
        "current_operational_regime": regime,
        "primary_root_cause": str(causal.get("primary_root_cause") or ""),
        "material_storage_backlog": material_storage_backlog,
        "storage_evidence": storage_evidence,
        "drift_alerts": ordered_unique(drift_alerts),
        "shadow_models": shadow_models,
        "regime_policy": {
            "expansion_allowed": regime == "expansion_rehearsal_ready" and not drift_alerts,
            "training_allowed": regime in {"expansion_rehearsal_ready", "steady_observation"} and "high_uncertainty_surface" not in drift_alerts,
            "requires_rebias": bool(drift_alerts),
        },
    }


def _super_objective_guardrail_layer(
    *,
    signal_bus: dict[str, Any],
    process_contracts: dict[str, Any],
    decision: dict[str, Any],
    adaptive_policy: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    global_contract = _as_dict(process_contracts.get("global_safety_contract"))
    command_risk = _super_command_risk(decision.get("safe_next_command"))
    hard_blocks: list[str] = []
    advisory_blocks: list[str] = []
    if bool(global_contract.get("live_trade_authority_added", True)):
        hard_blocks.append("live_trade_authority_added")
    if bool(global_contract.get("parallel_sql_writers_allowed", True)):
        hard_blocks.append("parallel_sql_writers_allowed")
    if not bool(global_contract.get("single_sql_writer_only", False)):
        hard_blocks.append("single_sql_writer_contract_missing")
    if str(command_risk.get("risk") or "") == "unsafe":
        hard_blocks.append(str(command_risk.get("reason") or "unsafe_command"))
    elif not bool(command_risk.get("allowed", True)):
        advisory_blocks.append(str(command_risk.get("reason") or "command_needs_review"))
    if str(decision.get("executive_mode") or "") == "expand" and (
        bool(summary.get("storage_critical", False))
        or bool(summary.get("memory_pressure_high", False))
        or bool(summary.get("runtime_pressure_high", False))
        or str(summary.get("guard_policy_mode") or "") == "protective_throttle"
    ):
        hard_blocks.append("expansion_requested_under_pressure")
    if str(adaptive_policy.get("training_posture") or "") == "bounded_targeted_allowed" and str(summary.get("guard_policy_mode") or "") == "protective_throttle":
        advisory_blocks.append("training_posture_conflicts_with_guard_throttle")
    status = "blocked" if hard_blocks else "advisory" if advisory_blocks else "ready"
    preservation_score = 100 - len(hard_blocks) * 35 - len(advisory_blocks) * 10
    return {
        "overall_status": status,
        "objective_function": "capital_preservation_operational_stability_and_evidence_quality_before_growth",
        "capital_preservation_score": int(max(0, min(100, preservation_score))),
        "hard_blocks": ordered_unique(hard_blocks),
        "advisory_blocks": ordered_unique(advisory_blocks),
        "command_risk": command_risk,
        "invariants": {
            "trade_authority": "none",
            "live_trading_enabled": False,
            "single_sql_writer_only": bool(global_contract.get("single_sql_writer_only", False)),
            "parallel_sql_writers_allowed": bool(global_contract.get("parallel_sql_writers_allowed", False)),
            "bounded_apply_modes_only": bool(global_contract.get("bounded_apply_modes_only", False)),
            "paper_data_first": bool(global_contract.get("paper_data_first", True)),
        },
    }


def _super_adversarial_simulation_layer(
    *,
    signal_bus: dict[str, Any],
    process_contracts: dict[str, Any],
    self_intelligence: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    storage_metrics = _as_dict(_signal_by_name(signal_bus, "ingestion_storage").get("metrics"))
    cleanup_metrics = _as_dict(_signal_by_name(signal_bus, "bot_logs_cleanup").get("metrics"))
    quota_metrics = _as_dict(_signal_by_name(signal_bus, "storage_quota_guard").get("metrics"))
    uncertainty = _as_dict(self_intelligence.get("uncertainty"))
    cleanup_capacity = _safe_float(cleanup_metrics.get("capacity_pct"), 0.0)
    cleanup_needed = bool(cleanup_metrics.get("cleanup_needed", False)) or _safe_float(cleanup_metrics.get("remaining_to_target_gb"), 0.0) > 0
    storage_refill_severity = 86 if cleanup_needed or cleanup_capacity >= 95.0 else 64 if _safe_int(quota_metrics.get("hard_breaches"), 0) > 0 else 42
    stale_signal_severity = 78 if _as_list(uncertainty.get("stale_signals")) or _as_list(uncertainty.get("conflicting_signals")) else 38
    if _safe_float(storage_metrics.get("pending_ratio"), 0.0) >= 1.0 or _safe_float(storage_metrics.get("pressure_index"), 0.0) >= 1.0:
        storage_refill_severity = max(storage_refill_severity, 72)
    scenarios = [
        {
            "scenario": "storage_refill_after_cleanup",
            "trigger": "failback_or_live_jsonl_writers_refill_hot_path",
            "mitigation": "keep_autosync_space_gated_and_run_tiered_cleanup_before_failback",
            "severity": storage_refill_severity,
            "evidence": {
                "cleanup_needed": cleanup_needed,
                "bot_logs_capacity_pct": cleanup_capacity,
                "quota_hard_breaches": _safe_int(quota_metrics.get("hard_breaches"), 0),
            },
        },
        {
            "scenario": "stale_signal_false_clear",
            "trigger": "critical_artifact_age_or_conflicting_health_surfaces",
            "mitigation": "self_intelligence_precheck_refreshes_pressure_surfaces_before_apply",
            "severity": stale_signal_severity,
        },
        {
            "scenario": "writer_race",
            "trigger": "multiple_sql_or_jsonl_mutators_attempt_recovery",
            "mitigation": "process_contracts_keep_single_sql_writer_and_bounded_drainer_waves",
            "severity": 92 if not bool(_as_dict(process_contracts.get("global_safety_contract")).get("single_sql_writer_only", False)) else 35,
        },
        {
            "scenario": "guard_overconfidence",
            "trigger": "expansion_or_training_allowed_while_guard_is_protective",
            "mitigation": "objective_guardrail_blocks_expansion_under_guard_throttle",
            "severity": 82 if str(summary.get("guard_policy_mode") or "") == "protective_throttle" else 32,
        },
        {
            "scenario": "provider_or_auth_degradation",
            "trigger": "auth_lease_or_provider_signal_degrades_after_sleeve_launch",
            "mitigation": "route_to_auth_and_halt_contract_before_relaunch",
            "severity": 88 if bool(summary.get("global_halt_active", False)) else 42,
        },
    ]
    max_severity = max((_safe_int(row.get("severity"), 0) for row in scenarios), default=0)
    resilience_score = 100 - max(0, max_severity - 35)
    return {
        "overall_status": "degraded" if max_severity >= 80 else "advisory" if max_severity >= 60 else "ready",
        "resilience_score": int(max(0, min(100, resilience_score))),
        "top_scenario": str(max(scenarios, key=lambda row: _safe_int(row.get("severity"), 0)).get("scenario") if scenarios else "none"),
        "scenarios": sorted(scenarios, key=lambda row: _safe_int(row.get("severity"), 0), reverse=True),
    }


def _super_decision_quality_layer(
    *,
    decision: dict[str, Any],
    self_intelligence: dict[str, Any],
    attention_graph: list[dict[str, Any]],
    regime_audit: dict[str, Any],
    objective_guardrails: dict[str, Any],
) -> dict[str, Any]:
    uncertainty = _as_dict(self_intelligence.get("uncertainty"))
    causal = _as_dict(self_intelligence.get("causal_diagnosis"))
    action_effect = _as_dict(self_intelligence.get("action_effectiveness"))
    top_attention = _safe_int((attention_graph[0] if attention_graph else {}).get("severity_score"), 0)
    score = 50.0
    score += _safe_float(decision.get("confidence"), 0.0) * 35.0
    score += _safe_float(causal.get("confidence"), 0.0) * 15.0
    score -= _safe_int(uncertainty.get("score"), 0) * 0.25
    score -= max(0, top_attention - 65) * 0.2
    score -= len(_as_list(regime_audit.get("drift_alerts"))) * 8.0
    score -= len(_as_list(objective_guardrails.get("hard_blocks"))) * 30.0
    if str(action_effect.get("verdict") or "") in {"ineffective_so_far", "worsening"}:
        score -= 12.0
    score = round(max(0.0, min(100.0, score)), 3)
    grade = "high" if score >= 75 else "medium" if score >= 55 else "low"
    return {
        "overall_status": "ready" if grade == "high" else "advisory" if grade == "medium" else "degraded",
        "quality_score": score,
        "quality_grade": grade,
        "requires_human_review": bool(
            grade == "low"
            or _as_list(objective_guardrails.get("hard_blocks"))
            or str(regime_audit.get("overall_status") or "") == "degraded"
        ),
        "calibration_evidence": ordered_unique(
            [
                f"decision_confidence={decision.get('confidence', 0)}",
                f"causal_confidence={causal.get('confidence', 0)}",
                f"uncertainty_score={uncertainty.get('score', 0)}",
                f"top_attention_severity={top_attention}",
                f"action_effect={action_effect.get('verdict', '')}",
            ]
        ),
    }


def _super_semantic_synthesis_layer(
    *,
    signal_bus: dict[str, Any],
    self_intelligence: dict[str, Any],
    decision: dict[str, Any],
    regime_audit: dict[str, Any],
    objective_guardrails: dict[str, Any],
    adversarial_simulation: dict[str, Any],
    decision_quality: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    causal = _as_dict(self_intelligence.get("causal_diagnosis"))
    action_effect = _as_dict(self_intelligence.get("action_effectiveness"))
    regime = str(regime_audit.get("current_operational_regime") or "")
    thesis = (
        f"Run {decision.get('action', 'observe')} in {decision.get('executive_mode', 'observe')} mode because "
        f"{causal.get('primary_root_cause', summary.get('top_risk', 'the top risk'))} is the dominant constraint "
        f"under the {regime} regime."
    )
    counter_thesis = (
        "Do not escalate beyond advisory infrastructure actions until guardrails, pressure surfaces, and action-effect evidence agree."
    )
    invalidators = ordered_unique(
        [
            "hard_objective_guardrail_block" if _as_list(objective_guardrails.get("hard_blocks")) else "",
            "adversarial_resilience_below_floor" if _safe_int(adversarial_simulation.get("resilience_score"), 100) < 55 else "",
            "decision_quality_low" if str(decision_quality.get("quality_grade") or "") == "low" else "",
            "action_effect_worsening" if str(action_effect.get("verdict") or "") == "worsening" else "",
        ]
    )
    return {
        "overall_status": "blocked" if _as_list(objective_guardrails.get("hard_blocks")) else "advisory" if invalidators else "ready",
        "thesis_statement": thesis,
        "counter_thesis": counter_thesis,
        "invalidators": invalidators,
        "operator_digest": ordered_unique(
            [
                f"mode={decision.get('executive_mode', '')}",
                f"owner={decision.get('owner', '')}",
                f"top_attention={decision.get('top_attention', '')}",
                f"regime={regime}",
                f"quality={decision_quality.get('quality_grade', '')}:{decision_quality.get('quality_score', 0)}",
                f"resilience={adversarial_simulation.get('resilience_score', 0)}",
            ]
        ),
    }


def _super_paper_lane_governor(signal_bus: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(_signal_by_name(signal_bus, "paper_live_data_standard").get("metrics"))
    actual = _safe_int(metrics.get("paper_live_data_enabled_bots"), 0)
    minimum = _safe_int(metrics.get("minimum"), 30)
    maximum = _safe_int(metrics.get("maximum"), 50)
    direct_live = _safe_int(metrics.get("direct_execution_allowed_bots"), 0) + _safe_int(metrics.get("live_trading_enabled_bots"), 0)
    hard_blocks = ordered_unique(
        [
            "paper_lane_has_live_or_direct_authority" if direct_live > 0 else "",
            "paper_lane_above_maximum" if actual > maximum else "",
        ]
    )
    if hard_blocks:
        posture = "block_and_relock"
        status = "blocked"
    elif minimum <= actual <= maximum:
        posture = "standard_30_50_active"
        status = "ready"
    elif actual < minimum:
        posture = "bootstrap_more_legacy_tested_bots"
        status = "advisory"
    else:
        posture = "trim_to_maximum"
        status = "degraded"
    return {
        "overall_status": status,
        "paper_lane_posture": posture,
        "paper_live_data_enabled_bots": actual,
        "target_band": {"minimum": minimum, "target": _safe_int(metrics.get("target"), 40), "maximum": maximum},
        "standard_promoted_paper_bots": _safe_int(metrics.get("standard_promoted_paper_bots"), 0),
        "collection_until_standard_bots": _safe_int(metrics.get("collection_until_standard_bots"), 0),
        "hard_blocks": hard_blocks,
        "next_safe_command": ["./scripts/ops/opsctl.sh", "paper-standard", "--apply", "--json"],
        "decision_alignment": {
            "executive_mode": str(decision.get("executive_mode") or ""),
            "paper_lane_allows_growth": bool(status in {"ready", "advisory"} and str(decision.get("executive_mode") or "") not in {"safety"}),
        },
    }


def _super_symbol_universe_layer(signal_bus: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(_signal_by_name(signal_bus, "sleeve_ticker_universe").get("metrics"))
    enabled = bool(metrics.get("enabled", False))
    core = _safe_int(metrics.get("core_symbol_count"), 0)
    crypto = _safe_int(metrics.get("crypto_symbol_count"), 0)
    breadth_score = min(core * 0.45 + crypto * 1.2 + _safe_int(metrics.get("bond_symbol_count"), 0) * 0.8, 100.0)
    watchouts = ordered_unique(
        [
            "ticker_universe_override_not_loaded" if not enabled else "",
            "core_universe_thin" if core < 60 else "",
            "crypto_universe_thin" if crypto < 10 else "",
        ]
    )
    return {
        "overall_status": "degraded" if watchouts else "ready",
        "enabled": enabled,
        "breadth_score": round(breadth_score, 3),
        "core_symbol_count": core,
        "defensive_symbol_count": _safe_int(metrics.get("defensive_symbol_count"), 0),
        "crypto_symbol_count": crypto,
        "bond_symbol_count": _safe_int(metrics.get("bond_symbol_count"), 0),
        "fx_symbol_count": _safe_int(metrics.get("fx_symbol_count"), 0),
        "watchouts": watchouts,
        "policy": "expanded_universe_feeds_all_applicable_sleeves_but_keeps_crypto_websocket_subset_bounded",
    }


def _super_cognitive_twin_layer(
    *,
    signal_bus: dict[str, Any],
    decision: dict[str, Any],
    paper_lane: dict[str, Any],
    symbol_universe: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    pressure = bool(summary.get("memory_pressure_high", False) or summary.get("runtime_pressure_high", False))
    paper_ready = str(paper_lane.get("overall_status") or "") == "ready"
    universe_ready = str(symbol_universe.get("overall_status") or "") == "ready"
    worlds = [
        {
            "world": "base_case",
            "assumption": "current pressure and guard posture persist for the next cycle",
            "expected_result": "paper lane observes live data while collectors continue gathering",
            "risk": 46 if pressure else 24,
        },
        {
            "world": "pressure_relief_success",
            "assumption": "runtime pressure drops after the next safe relief action",
            "expected_result": "paper lane remains stable and ticker breadth improves context",
            "risk": 28 if paper_ready and universe_ready else 42,
        },
        {
            "world": "provider_rate_limit_or_stale_data",
            "assumption": "expanded ticker universe increases stale or throttled vendor responses",
            "expected_result": "adaptive intervals and websocket subset should throttle before bot promotion",
            "risk": 62 if not universe_ready else 38,
        },
        {
            "world": "paper_lane_overexpansion",
            "assumption": "paper cohort grows above the 50 bot ceiling",
            "expected_result": "objective guardrail trims to explicit cohort and keeps live authority off",
            "risk": 70 if _safe_int(paper_lane.get("paper_live_data_enabled_bots"), 0) > _safe_int(_as_dict(paper_lane.get("target_band")).get("maximum"), 50) else 26,
        },
    ]
    max_risk = max((_safe_int(row.get("risk"), 0) for row in worlds), default=0)
    return {
        "overall_status": "degraded" if max_risk >= 60 else "advisory" if max_risk >= 45 else "ready",
        "max_world_risk": max_risk,
        "recommended_next_world": "pressure_relief_success" if pressure else "base_case",
        "worlds": sorted(worlds, key=lambda row: _safe_int(row.get("risk"), 0), reverse=True),
        "policy": "simulate_operational_futures_before_mutating_paper_lane_or_ticker_breadth",
        "decision_action": str(decision.get("action") or ""),
    }


def build_super_intelligence(
    *,
    signal_bus: dict[str, Any],
    system_brain: dict[str, Any],
    process_contracts: dict[str, Any],
    self_intelligence: dict[str, Any],
    previous_payload: dict[str, Any],
    memory_events: list[dict[str, Any]],
) -> dict[str, Any]:
    attention_graph = _super_attention_graph(
        signal_bus=signal_bus,
        system_brain=system_brain,
        process_contracts=process_contracts,
        self_intelligence=self_intelligence,
    )
    adaptive_policy = _super_adaptive_policy(signal_bus, system_brain, self_intelligence)
    decision = _super_decision_packet(
        signal_bus=signal_bus,
        system_brain=system_brain,
        self_intelligence=self_intelligence,
        attention_graph=attention_graph,
        adaptive_policy=adaptive_policy,
    )
    memory_summary = _super_memory_summary(memory_events, str(decision.get("action") or ""))
    regime_audit = _super_regime_drift_audit(
        signal_bus=signal_bus,
        self_intelligence=self_intelligence,
        memory_summary=memory_summary,
    )
    objective_guardrails = _super_objective_guardrail_layer(
        signal_bus=signal_bus,
        process_contracts=process_contracts,
        decision=decision,
        adaptive_policy=adaptive_policy,
    )
    adversarial_simulation = _super_adversarial_simulation_layer(
        signal_bus=signal_bus,
        process_contracts=process_contracts,
        self_intelligence=self_intelligence,
    )
    decision_quality = _super_decision_quality_layer(
        decision=decision,
        self_intelligence=self_intelligence,
        attention_graph=attention_graph,
        regime_audit=regime_audit,
        objective_guardrails=objective_guardrails,
    )
    semantic_synthesis = _super_semantic_synthesis_layer(
        signal_bus=signal_bus,
        self_intelligence=self_intelligence,
        decision=decision,
        regime_audit=regime_audit,
        objective_guardrails=objective_guardrails,
        adversarial_simulation=adversarial_simulation,
        decision_quality=decision_quality,
    )
    paper_lane_governor = _super_paper_lane_governor(signal_bus, decision)
    symbol_universe_layer = _super_symbol_universe_layer(signal_bus)
    cognitive_twin_layer = _super_cognitive_twin_layer(
        signal_bus=signal_bus,
        decision=decision,
        paper_lane=paper_lane_governor,
        symbol_universe=symbol_universe_layer,
    )
    decision = {
        **decision,
        "objective_guardrail_status": str(objective_guardrails.get("overall_status") or ""),
        "decision_quality_score": decision_quality.get("quality_score"),
        "decision_quality_grade": str(decision_quality.get("quality_grade") or ""),
        "operational_regime": str(regime_audit.get("current_operational_regime") or ""),
        "thesis_statement": str(semantic_synthesis.get("thesis_statement") or ""),
        "paper_lane_posture": str(paper_lane_governor.get("paper_lane_posture") or ""),
        "paper_live_data_enabled_bots": _safe_int(paper_lane_governor.get("paper_live_data_enabled_bots"), 0),
        "symbol_universe_breadth_score": _safe_float(symbol_universe_layer.get("breadth_score"), 0.0),
        "cognitive_twin_max_world_risk": _safe_int(cognitive_twin_layer.get("max_world_risk"), 0),
    }
    previous_super = _as_dict(previous_payload.get("system_super_intelligence"))
    previous_decision = _as_dict(previous_super.get("decision_packet"))
    status = "ready"
    if str(objective_guardrails.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(decision.get("executive_mode") or "") in {"safety", "precheck"}:
        status = "blocked" if str(system_brain.get("overall_status") or "") == "blocked" else "degraded"
    elif str(decision.get("executive_mode") or "") in {"stabilize", "drain", "rethink"}:
        status = "degraded"
    elif str(paper_lane_governor.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(cognitive_twin_layer.get("overall_status") or "") == "degraded":
        status = "degraded"
    elif str(decision_quality.get("overall_status") or "") == "degraded":
        status = "degraded"
    elif attention_graph and _safe_int(attention_graph[0].get("severity_score"), 0) >= 65:
        status = "advisory"
    memory_event = {
        "timestamp_utc": iso_now(),
        "status": status,
        "action": str(decision.get("action") or ""),
        "executive_mode": str(decision.get("executive_mode") or ""),
        "top_attention": str(decision.get("top_attention") or ""),
        "guard_policy_mode": str(adaptive_policy.get("guard_policy_mode") or ""),
        "operational_regime": str(regime_audit.get("current_operational_regime") or ""),
        "guardrail_status": str(objective_guardrails.get("overall_status") or ""),
        "decision_quality_score": decision_quality.get("quality_score"),
        "paper_live_data_enabled_bots": _safe_int(paper_lane_governor.get("paper_live_data_enabled_bots"), 0),
        "cognitive_twin_max_world_risk": _safe_int(cognitive_twin_layer.get("max_world_risk"), 0),
        "pending_lines": _safe_int(_as_dict(signal_bus.get("summary")).get("total_pending_lines"), 0),
        "previous_action": str(previous_decision.get("action") or ""),
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "system_super_intelligence",
        "ok": status == "ready",
        "overall_status": status,
        "decision_packet": decision,
        "attention_graph": attention_graph,
        "adaptive_policy": adaptive_policy,
        "regime_drift_audit": regime_audit,
        "objective_guardrail_layer": objective_guardrails,
        "adversarial_simulation_layer": adversarial_simulation,
        "decision_quality_layer": decision_quality,
        "semantic_synthesis_layer": semantic_synthesis,
        "paper_lane_governor_layer": paper_lane_governor,
        "symbol_universe_intelligence_layer": symbol_universe_layer,
        "cognitive_twin_counterfactual_layer": cognitive_twin_layer,
        "learning_memory": memory_summary,
        "memory_event": memory_event,
        "integration_contract": {
            "purpose": "rank_cross_layer_attention_choose_executive_mode_verify_objective_guardrails_simulate_adversarial_failures_synthesize_thesis_and_route_next_safe_infrastructure_action",
            "reads": [
                "system_signal_bus",
                "system_brain",
                "system_process_contracts",
                "system_self_intelligence",
                "guard_intelligence",
                "paper_live_data_standard",
                "sleeve_ticker_universe",
            ],
            "writes": ["system_super_intelligence_latest.json", "super_intelligence_memory.jsonl"],
            "does_not_execute_commands": True,
            "does_not_trade": True,
            "single_sql_writer_only": True,
        },
        "codex_interface": {
            "communicates_with_codex": "artifact_handoff_when_codex_reads_workspace_or_opsctl_runs",
            "proactive_delivery_to_codex": False,
            "safe_next_command_is_advisory": True,
        },
    }


def _outcome_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_pending = _safe_int(previous.get("pending_lines"), 0)
    current_pending = _safe_int(current.get("pending_lines"), 0)
    previous_quality = _safe_float(previous.get("decision_quality_score"), 0.0)
    current_quality = _safe_float(current.get("decision_quality_score"), 0.0)
    previous_resilience = _safe_float(previous.get("resilience_score"), 0.0)
    current_resilience = _safe_float(current.get("resilience_score"), 0.0)
    return {
        "pending_lines_delta": int(current_pending - previous_pending),
        "decision_quality_delta": round(current_quality - previous_quality, 3) if previous_quality else 0.0,
        "resilience_delta": round(current_resilience - previous_resilience, 3) if previous_resilience else 0.0,
        "previous_action": str(previous.get("action") or ""),
        "previous_regime": str(previous.get("operational_regime") or ""),
        "previous_status": str(previous.get("status") or ""),
    }


def build_outcome_learning(
    *,
    signal_bus: dict[str, Any],
    system_brain: dict[str, Any],
    self_intelligence: dict[str, Any],
    super_intelligence: dict[str, Any],
    outcome_events: list[dict[str, Any]],
) -> dict[str, Any]:
    del system_brain
    summary = _as_dict(signal_bus.get("summary"))
    decision = _as_dict(super_intelligence.get("decision_packet"))
    quality = _as_dict(super_intelligence.get("decision_quality_layer"))
    adversarial = _as_dict(super_intelligence.get("adversarial_simulation_layer"))
    regime = _as_dict(super_intelligence.get("regime_drift_audit"))
    action_effect = _as_dict(self_intelligence.get("action_effectiveness"))
    causal = _as_dict(self_intelligence.get("causal_diagnosis"))
    current_event = {
        "timestamp_utc": iso_now(),
        "status": str(super_intelligence.get("overall_status") or ""),
        "action": str(decision.get("action") or ""),
        "executive_mode": str(decision.get("executive_mode") or ""),
        "owner": str(decision.get("owner") or ""),
        "top_attention": str(decision.get("top_attention") or ""),
        "causal_root": str(causal.get("primary_root_cause") or ""),
        "operational_regime": str(regime.get("current_operational_regime") or ""),
        "pending_lines": _safe_int(summary.get("total_pending_lines"), 0),
        "decision_quality_score": _safe_float(quality.get("quality_score"), _safe_float(decision.get("decision_quality_score"), 0.0)),
        "resilience_score": _safe_int(adversarial.get("resilience_score"), 0),
        "guard_policy_mode": str(_as_dict(super_intelligence.get("adaptive_policy")).get("guard_policy_mode") or ""),
    }
    previous = outcome_events[-1] if outcome_events else {}
    delta = _outcome_delta(previous, current_event) if previous else {
        "pending_lines_delta": 0,
        "decision_quality_delta": 0.0,
        "resilience_delta": 0.0,
        "previous_action": "",
        "previous_regime": "",
        "previous_status": "",
    }
    pending_delta = _safe_int(delta.get("pending_lines_delta"), 0)
    quality_delta = _safe_float(delta.get("decision_quality_delta"), 0.0)
    resilience_delta = _safe_float(delta.get("resilience_delta"), 0.0)
    if not previous:
        verdict = "baseline"
    elif pending_delta <= -250 or quality_delta >= 5.0 or resilience_delta >= 5.0:
        verdict = "effective"
    elif pending_delta >= 250 or quality_delta <= -5.0 or resilience_delta <= -5.0:
        verdict = "worsening"
    elif str(action_effect.get("verdict") or "") in {"ineffective_so_far", "worsening"}:
        verdict = "ineffective_so_far"
    else:
        verdict = "monitoring"

    playbook = str(decision.get("action") or "observe")
    credit_score = 50.0
    if verdict == "effective":
        credit_score += 25.0
    elif verdict == "worsening":
        credit_score -= 30.0
    elif verdict == "ineffective_so_far":
        credit_score -= 18.0
    if str(regime.get("overall_status") or "") == "degraded":
        credit_score -= 8.0
    if _as_list(_as_dict(super_intelligence.get("semantic_synthesis_layer")).get("invalidators")):
        credit_score -= 8.0
    credit_score = round(max(0.0, min(100.0, credit_score)), 3)
    policy_credit = {
        playbook: {
            "credit_score": credit_score,
            "verdict": verdict,
            "evidence": ordered_unique(
                [
                    f"pending_delta={pending_delta}",
                    f"quality_delta={quality_delta}",
                    f"resilience_delta={resilience_delta}",
                    f"self_action_effect={action_effect.get('verdict', '')}",
                ]
            ),
        }
    }
    mutations = []
    if verdict in {"ineffective_so_far", "worsening"} or _safe_float(quality.get("quality_score"), 100.0) < 55.0:
        mutations.append(
            {
                "mutation": "refresh_evidence_then_choose_next_playbook",
                "why": "decision_quality_or_action_effect_is_below_floor",
                "safe_command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"],
            }
        )
    if str(causal.get("primary_root_cause") or "") == "memory_pressure_primary":
        mutations.append(
            {
                "mutation": "resource_first_micro_relief",
                "why": "memory_pressure_is_primary_root",
                "safe_command": ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
            }
        )
    if str(causal.get("primary_root_cause") or "") == "storage_backpressure_primary":
        mutations.append(
            {
                "mutation": "storage_outcome_verified_micro_drain",
                "why": "storage_backpressure_is_primary_root",
                "safe_command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", "5000", "--json"],
            }
        )
    confidence_recovery = {
        "state": "recovering" if verdict == "effective" else "locked" if _safe_float(quality.get("quality_score"), 100.0) < 55.0 else "monitoring",
        "raises_quality_when": [
            "same_action_reduces_pending_lines_by_250_or_more",
            "decision_quality_improves_by_5_points",
            "adversarial_resilience_improves_by_5_points",
        ],
        "current_quality_score": _safe_float(quality.get("quality_score"), 0.0),
    }
    status = "ready" if verdict in {"effective", "baseline", "monitoring"} and confidence_recovery["state"] != "locked" else "degraded"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "super_intelligence_outcome_learning",
        "ok": status == "ready",
        "overall_status": status,
        "intervention_outcome": {
            "verdict": verdict,
            "delta": delta,
            "current_event": current_event,
            "history_event_count": len(outcome_events),
        },
        "causal_replay_scorer": {
            "primary_root_cause": str(causal.get("primary_root_cause") or ""),
            "causal_confidence": _safe_float(causal.get("confidence"), 0.0),
            "replay_findings": ordered_unique(
                [
                    "storage_refill_risk_present" if str(adversarial.get("top_scenario") or "") == "storage_refill_after_cleanup" else "",
                    "quality_floor_not_met" if _safe_float(quality.get("quality_score"), 100.0) < 55.0 else "",
                    f"current_regime={current_event['operational_regime']}",
                ]
            ),
        },
        "policy_credit_assignment": policy_credit,
        "playbook_mutation_guard": {
            "mutations": mutations,
            "mutation_allowed": bool(mutations),
            "requires_bounded_command": True,
            "no_live_authority": True,
        },
        "confidence_recovery_engine": confidence_recovery,
        "memory_event": current_event,
        "contract": {
            "does_not_execute_commands": True,
            "does_not_trade": True,
            "writes": ["super_intelligence_outcome_learning_latest.json", "intervention_outcomes.jsonl"],
        },
    }


def build_recursive_intelligence(
    *,
    signal_bus: dict[str, Any],
    super_intelligence: dict[str, Any],
    outcome_learning: dict[str, Any],
    recursive_events: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = _as_dict(super_intelligence.get("decision_packet"))
    guardrails = _as_dict(super_intelligence.get("objective_guardrail_layer"))
    outcome = _as_dict(outcome_learning.get("intervention_outcome"))
    verdict = str(outcome.get("verdict") or "")
    summary = _as_dict(signal_bus.get("summary"))
    mutation_candidates = [row for row in _as_list(_as_dict(outcome_learning.get("playbook_mutation_guard")).get("mutations")) if isinstance(row, dict)]
    invariant_blocks = ordered_unique(
        [
            *[str(item) for item in _as_list(guardrails.get("hard_blocks"))],
            "live_trade_authority_must_remain_none" if str(_as_dict(guardrails.get("invariants")).get("trade_authority") or "") != "none" else "",
            "parallel_sql_writers_must_remain_false" if bool(_as_dict(guardrails.get("invariants")).get("parallel_sql_writers_allowed", False)) else "",
        ]
    )
    experiments = []
    for row in mutation_candidates[:5]:
        experiments.append(
            {
                "experiment": str(row.get("mutation") or ""),
                "hypothesis": str(row.get("why") or ""),
                "safe_command": row.get("safe_command") if isinstance(row.get("safe_command"), list) else [],
                "promotion_criteria": [
                    "outcome_verdict_effective",
                    "quality_score_above_55",
                    "no_guardrail_hard_blocks",
                ],
                "rollback_triggers": [
                    "pending_lines_delta_positive_250",
                    "decision_quality_drops_5_points",
                    "guardrail_status_blocked",
                ],
            }
        )
    if not experiments:
        experiments.append(
            {
                "experiment": "observe_current_policy",
                "hypothesis": "no_policy_mutation_needed_until_outcome_learning_finds_a_failed_or_low_quality_playbook",
                "safe_command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"],
                "promotion_criteria": ["continued_ready_or_advisory_status"],
                "rollback_triggers": ["new_hard_block", "quality_score_below_55"],
            }
        )
    recursive_score = 72.0
    if verdict in {"worsening", "ineffective_so_far"}:
        recursive_score -= 14.0
    if _safe_float(decision.get("decision_quality_score"), 100.0) < 55.0:
        recursive_score -= 12.0
    if invariant_blocks:
        recursive_score -= 35.0
    if bool(summary.get("memory_pressure_high", False)) or bool(summary.get("runtime_pressure_high", False)):
        recursive_score -= 6.0
    recursive_score = round(max(0.0, min(100.0, recursive_score)), 3)
    status = "blocked" if invariant_blocks else "degraded" if recursive_score < 55.0 else "advisory" if recursive_score < 72.0 else "ready"
    memory_event = {
        "timestamp_utc": iso_now(),
        "status": status,
        "recursive_score": recursive_score,
        "outcome_verdict": verdict,
        "experiment_count": len(experiments),
        "invariant_block_count": len(invariant_blocks),
        "super_action": str(decision.get("action") or ""),
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "recursive_policy_evolution_layer",
        "ok": status == "ready",
        "overall_status": status,
        "recursive_score": recursive_score,
        "policy_hypothesis_lab": {
            "experiments": experiments,
            "history_event_count": len(recursive_events),
            "mutation_scope": "advisory_bounded_infrastructure_playbooks_only",
        },
        "invariant_firewall": {
            "overall_status": "blocked" if invariant_blocks else "ready",
            "invariant_blocks": invariant_blocks,
            "enforced_invariants": [
                "no_live_trade_authority",
                "single_sql_writer_only",
                "bounded_apply_modes_only",
                "paper_data_first",
                "operator_or_codex_review_for_low_quality_decisions",
            ],
        },
        "recursive_upgrade_backlog": ordered_unique(
            [
                "build_cognitive_twin_counterfactual_simulator",
                "add_policy_canary_replay_before_apply",
                "add_operator_feedback_reward_model",
                "add_cross_sleeve_causal_graph_memory",
            ]
        ),
        "next_more_advanced_layer": {
            "name": "cognitive_twin_counterfactual_simulator",
            "why": "simulate multiple future system states before recommending policy mutations",
            "new_capabilities": [
                "multi_world_system_digital_twin",
                "counterfactual_market_and_operations_paths",
                "policy_canary_replay",
                "operator_feedback_reward_model",
                "cross_sleeve_causal_graph_memory",
            ],
        },
        "memory_event": memory_event,
        "contract": {
            "does_not_execute_commands": True,
            "does_not_trade": True,
            "writes": ["system_recursive_intelligence_latest.json", "recursive_intelligence_memory.jsonl"],
        },
    }


def build_codex_handoff(
    *,
    signal_bus: dict[str, Any],
    system_brain: dict[str, Any],
    process_contracts: dict[str, Any],
    self_intelligence: dict[str, Any] | None = None,
    super_intelligence: dict[str, Any] | None = None,
    outcome_learning: dict[str, Any] | None = None,
    recursive_intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = _as_dict(system_brain.get("decision_packet"))
    summary = _as_dict(signal_bus.get("summary"))
    self_layer = self_intelligence if isinstance(self_intelligence, dict) else {}
    reflex = _as_dict(self_layer.get("reflex"))
    uncertainty = _as_dict(self_layer.get("uncertainty"))
    causal = _as_dict(self_layer.get("causal_diagnosis"))
    action_effect = _as_dict(self_layer.get("action_effectiveness"))
    routing = _as_dict(self_layer.get("integration_routing"))
    super_layer = super_intelligence if isinstance(super_intelligence, dict) else {}
    super_decision = _as_dict(super_layer.get("decision_packet"))
    super_policy = _as_dict(super_layer.get("adaptive_policy"))
    super_regime = _as_dict(super_layer.get("regime_drift_audit"))
    super_guardrails = _as_dict(super_layer.get("objective_guardrail_layer"))
    super_adversarial = _as_dict(super_layer.get("adversarial_simulation_layer"))
    super_quality = _as_dict(super_layer.get("decision_quality_layer"))
    super_semantic = _as_dict(super_layer.get("semantic_synthesis_layer"))
    super_paper_lane = _as_dict(super_layer.get("paper_lane_governor_layer"))
    super_symbol_universe = _as_dict(super_layer.get("symbol_universe_intelligence_layer"))
    super_cognitive_twin = _as_dict(super_layer.get("cognitive_twin_counterfactual_layer"))
    outcome_layer = outcome_learning if isinstance(outcome_learning, dict) else {}
    outcome = _as_dict(outcome_layer.get("intervention_outcome"))
    confidence = _as_dict(outcome_layer.get("confidence_recovery_engine"))
    policy_credit = _as_dict(outcome_layer.get("policy_credit_assignment"))
    recursive_layer = recursive_intelligence if isinstance(recursive_intelligence, dict) else {}
    next_advanced_layer = _as_dict(recursive_layer.get("next_more_advanced_layer"))
    risks = [str(item) for item in _as_list(decision.get("risk_flags"))]
    needs: list[str] = []
    if bool(reflex.get("blocks_brain_action_until_refreshed", False)):
        needs.append("run_self_intelligence_precheck_before_brain_action")
    if "writer_recovery_required" in risks:
        needs.append("inspect_or_run_bounded_writer_recovery")
    if "storage_critical" in risks:
        needs.append("drain_storage_backlog_with_single_writer_guard")
    if "memory_pressure_high" in risks or "runtime_pressure_high" in risks:
        needs.append("apply_pressure_relief_before_heavy_work")
    if "guard_intelligence_throttle_active" in risks or "guard_intelligence_blockers" in risks:
        needs.append("refresh_guard_intelligence_before_expansion")
    if "global_halt_active" in risks:
        needs.append("refresh_auth_and_halt_clearance_before_relaunch")
    if not needs:
        needs.append("observe_current_state_and_continue_safe_expansion")

    safe_next_command = decision.get("safe_next_command") if isinstance(decision.get("safe_next_command"), list) else []
    if bool(reflex.get("blocks_brain_action_until_refreshed", False)) and isinstance(reflex.get("command"), list):
        safe_next_command = reflex.get("command") or safe_next_command
    elif isinstance(super_decision.get("safe_next_command"), list) and super_decision.get("safe_next_command"):
        safe_next_command = super_decision.get("safe_next_command") or safe_next_command
    attention_packet = {
        "status": str(system_brain.get("overall_status") or ""),
        "top_risk": str(decision.get("top_risk") or summary.get("top_risk") or "none"),
        "recommended_action": str(decision.get("action") or ""),
        "super_action": str(super_decision.get("action") or ""),
        "super_mode": str(super_decision.get("executive_mode") or ""),
        "super_owner": str(super_decision.get("owner") or ""),
        "super_regime": str(super_regime.get("current_operational_regime") or ""),
        "super_guardrail_status": str(super_guardrails.get("overall_status") or ""),
        "super_decision_quality": str(super_quality.get("quality_grade") or ""),
        "super_decision_quality_score": _safe_float(super_quality.get("quality_score"), 0.0),
        "super_adversarial_resilience_score": _safe_int(super_adversarial.get("resilience_score"), 0),
        "super_paper_lane_posture": str(super_paper_lane.get("paper_lane_posture") or ""),
        "super_paper_live_data_bots": _safe_int(super_paper_lane.get("paper_live_data_enabled_bots"), 0),
        "super_symbol_universe_breadth_score": _safe_float(super_symbol_universe.get("breadth_score"), 0.0),
        "super_cognitive_twin_max_world_risk": _safe_int(super_cognitive_twin.get("max_world_risk"), 0),
        "super_thesis": str(super_semantic.get("thesis_statement") or ""),
        "outcome_verdict": str(outcome.get("verdict") or ""),
        "outcome_confidence_state": str(confidence.get("state") or ""),
        "policy_credit": policy_credit,
        "recursive_status": str(recursive_layer.get("overall_status") or ""),
        "recursive_score": _safe_float(recursive_layer.get("recursive_score"), 0.0),
        "next_more_advanced_layer": str(next_advanced_layer.get("name") or ""),
        "pycharm_index_path": str(DEFAULT_PYCHARM_INDEX_PATH),
        "safe_next_command": safe_next_command,
        "self_reflex": reflex,
        "uncertainty_level": str(uncertainty.get("level") or ""),
        "causal_root": str(causal.get("primary_root_cause") or ""),
        "causal_confidence": _safe_float(causal.get("confidence"), 0.0),
        "action_effectiveness": str(action_effect.get("verdict") or ""),
        "integration_route": str(routing.get("route_mode") or ""),
        "integration_owner": str(routing.get("primary_owner") or ""),
        "adaptive_policy": {
            "sleeve_posture": str(super_policy.get("sleeve_posture") or ""),
            "expansion_posture": str(super_policy.get("expansion_posture") or ""),
            "training_posture": str(super_policy.get("training_posture") or ""),
            "drainer_posture": str(super_policy.get("drainer_posture") or ""),
        },
        "capability_gaps": [str(_as_dict(row).get("gap") or row) for row in _as_list(self_layer.get("capability_gaps"))],
        "super_invalidators": [str(item) for item in _as_list(super_semantic.get("invalidators"))],
        "super_hard_blocks": [str(item) for item in _as_list(super_guardrails.get("hard_blocks"))],
        "self_questions": [str(item) for item in _as_list(self_layer.get("self_questions"))],
        "do_not_do": [str(item) for item in _as_list(decision.get("do_not_do"))],
        "needs_codex": ordered_unique(needs),
        "why": ordered_unique(
            [
                f"pending_lines={summary.get('total_pending_lines', 0)}",
                f"storage_critical={summary.get('storage_critical', False)}",
                f"memory_pressure_high={summary.get('memory_pressure_high', False)}",
                f"runtime_pressure_high={summary.get('runtime_pressure_high', False)}",
                f"writer_recovery_required={summary.get('writer_recovery_required', False)}",
                f"guard_policy_mode={summary.get('guard_policy_mode', '')}",
                f"guard_pressure_score={summary.get('guard_pressure_score', 0)}",
            ]
        ),
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "codex_handoff_channel",
        "ok": str(system_brain.get("overall_status") or "") in {"ready", "advisory"},
        "overall_status": str(system_brain.get("overall_status") or "missing"),
        "attention_packet": attention_packet,
        "communication_contract": {
            "communicates_with_codex": "artifact_handoff_when_codex_reads_workspace_or_opsctl_runs",
            "proactive_delivery_to_codex": False,
            "automation_ready": True,
            "human_readable_markdown": str(DEFAULT_HANDOFF_MARKDOWN_PATH),
            "no_consciousness_claim": True,
            "no_trade_authority": True,
        },
        "source_artifacts": {
            "signal_bus": str(DEFAULT_SIGNAL_BUS_PATH),
            "system_brain": str(DEFAULT_BRAIN_PATH),
            "process_contracts": str(DEFAULT_CONTRACTS_PATH),
            "self_intelligence": str(DEFAULT_SELF_INTELLIGENCE_PATH),
            "super_intelligence": str(DEFAULT_SUPER_INTELLIGENCE_PATH),
            "outcome_learning": str(DEFAULT_OUTCOME_LEARNING_PATH),
            "recursive_intelligence": str(DEFAULT_RECURSIVE_INTELLIGENCE_PATH),
            "pycharm_index": str(DEFAULT_PYCHARM_INDEX_PATH),
        },
        "contract_snapshot": {
            "contract_count": _safe_int(process_contracts.get("contract_count"), 0),
            "blocked_contract_count": _safe_int(process_contracts.get("blocked_contract_count"), 0),
            "advisory_contract_count": _safe_int(process_contracts.get("advisory_contract_count"), 0),
        },
    }


def render_handoff_markdown(handoff: dict[str, Any]) -> str:
    packet = _as_dict(handoff.get("attention_packet"))
    lines = [
        "# Codex Handoff",
        "",
        f"- Timestamp UTC: `{handoff.get('timestamp_utc', '')}`",
        f"- Status: `{packet.get('status', '')}`",
        f"- Top Risk: `{packet.get('top_risk', '')}`",
        f"- Recommended Action: `{packet.get('recommended_action', '')}`",
        f"- Super Action: `{packet.get('super_action', '')}` mode `{packet.get('super_mode', '')}` owner `{packet.get('super_owner', '')}`",
        f"- Super Regime: `{packet.get('super_regime', '')}` guardrails `{packet.get('super_guardrail_status', '')}` quality `{packet.get('super_decision_quality', '')}`",
        f"- Safe Next Command: `{' '.join(str(item) for item in _as_list(packet.get('safe_next_command')))}`",
        "",
        "## Needs Codex",
        "",
    ]
    for item in _as_list(packet.get("needs_codex")):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Self Intelligence",
            "",
            f"- Uncertainty: `{packet.get('uncertainty_level', '')}`",
            f"- Reflex: `{(_as_dict(packet.get('self_reflex')).get('action') or '')}`",
            f"- Causal Root: `{packet.get('causal_root', '')}` confidence `{packet.get('causal_confidence', '')}`",
            f"- Action Effect: `{packet.get('action_effectiveness', '')}`",
            f"- Integration Route: `{packet.get('integration_route', '')}` owner `{packet.get('integration_owner', '')}`",
        ]
    )
    for item in _as_list(packet.get("self_questions")):
        lines.append(f"- {item}")
    policy = _as_dict(packet.get("adaptive_policy"))
    if policy:
        lines.extend(
            [
                "",
                "## Super Intelligence",
                "",
                f"- Sleeve Posture: `{policy.get('sleeve_posture', '')}`",
                f"- Expansion Posture: `{policy.get('expansion_posture', '')}`",
                f"- Training Posture: `{policy.get('training_posture', '')}`",
                f"- Drainer Posture: `{policy.get('drainer_posture', '')}`",
                f"- Paper Lane: `{packet.get('super_paper_lane_posture', '')}` bots `{packet.get('super_paper_live_data_bots', 0)}`",
                f"- Symbol Breadth Score: `{packet.get('super_symbol_universe_breadth_score', 0)}`",
                f"- Cognitive Twin Max Risk: `{packet.get('super_cognitive_twin_max_world_risk', 0)}`",
                f"- Thesis: `{packet.get('super_thesis', '')}`",
                f"- Adversarial Resilience Score: `{packet.get('super_adversarial_resilience_score', 0)}`",
            ]
        )
    if packet.get("outcome_verdict") or packet.get("recursive_status"):
        lines.extend(
            [
                "",
                "## Outcome Learning",
                "",
                f"- Verdict: `{packet.get('outcome_verdict', '')}`",
                f"- Confidence State: `{packet.get('outcome_confidence_state', '')}`",
                f"- Policy Credit: `{json.dumps(packet.get('policy_credit', {}), ensure_ascii=True, sort_keys=True)}`",
                "",
                "## Recursive Intelligence",
                "",
                f"- Status: `{packet.get('recursive_status', '')}`",
                f"- Score: `{packet.get('recursive_score', 0)}`",
                f"- Next Advanced Layer: `{packet.get('next_more_advanced_layer', '')}`",
                "",
                "## PyCharm Index",
                "",
                f"- Path: `{packet.get('pycharm_index_path', '')}`",
            ]
        )
    invalidators = _as_list(packet.get("super_invalidators"))
    if invalidators:
        lines.extend(["", "## Super Invalidators", ""])
        for item in invalidators:
            lines.append(f"- {item}")
    hard_blocks = _as_list(packet.get("super_hard_blocks"))
    if hard_blocks:
        lines.extend(["", "## Super Guardrail Blocks", ""])
        for item in hard_blocks:
            lines.append(f"- {item}")
    gaps = _as_list(packet.get("capability_gaps"))
    if gaps:
        lines.extend(["", "## Capability Gaps", ""])
        for item in gaps:
            lines.append(f"- {item}")
    lines.extend(["", "## Do Not Do", ""])
    for item in _as_list(packet.get("do_not_do")):
        lines.append(f"- {item}")
    lines.extend(["", "## Why", ""])
    for item in _as_list(packet.get("why")):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Communication",
            "",
            "The platform communicates with Codex through this artifact and related health JSON. It does not proactively message Codex unless a separate automation is configured.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_super_override(super_intelligence: dict[str, Any]) -> str:
    decision = _as_dict(super_intelligence.get("decision_packet"))
    policy = _as_dict(super_intelligence.get("adaptive_policy"))
    values = {
        "SUPER_INTELLIGENCE_ENABLED": "1",
        "SUPER_INTELLIGENCE_EXECUTIVE_MODE": str(decision.get("executive_mode") or ""),
        "SUPER_INTELLIGENCE_RECOMMENDED_ACTION": str(decision.get("action") or ""),
        "SUPER_INTELLIGENCE_OWNER": str(decision.get("owner") or ""),
        "SUPER_INTELLIGENCE_TOP_ATTENTION": str(decision.get("top_attention") or ""),
        "SUPER_INTELLIGENCE_OPERATIONAL_REGIME": str(decision.get("operational_regime") or ""),
        "SUPER_INTELLIGENCE_OBJECTIVE_GUARDRAIL_STATUS": str(decision.get("objective_guardrail_status") or ""),
        "SUPER_INTELLIGENCE_DECISION_QUALITY_GRADE": str(decision.get("decision_quality_grade") or ""),
        "SUPER_INTELLIGENCE_DECISION_QUALITY_SCORE": str(decision.get("decision_quality_score") or ""),
        "SUPER_INTELLIGENCE_GUARD_POLICY_MODE": str(policy.get("guard_policy_mode") or ""),
        "SUPER_INTELLIGENCE_SLEEVE_POSTURE": str(policy.get("sleeve_posture") or ""),
        "SUPER_INTELLIGENCE_EXPANSION_POSTURE": str(policy.get("expansion_posture") or ""),
        "SUPER_INTELLIGENCE_TRAINING_POSTURE": str(policy.get("training_posture") or ""),
        "SUPER_INTELLIGENCE_DRAINER_POSTURE": str(policy.get("drainer_posture") or ""),
        "SUPER_INTELLIGENCE_UPDATED_UTC": str(super_intelligence.get("timestamp_utc") or ""),
    }
    lines = ["# Auto-managed by scripts/ops/system_intelligence_coordinator.py"]
    lines.extend(f"{key}={value}" for key, value in sorted(values.items()))
    return "\n".join(lines) + "\n"


def _document_snapshot(project_root: Path, rel_path: str, required_markers: tuple[str, ...]) -> dict[str, Any]:
    path = project_root / rel_path
    text = ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        text = ""
    missing_markers = [marker for marker in required_markers if marker not in text]
    return {
        "path": str(path),
        "relative_path": rel_path,
        "exists": bool(path.exists()),
        "bytes": len(text.encode("utf-8")) if text else 0,
        "line_count": len(text.splitlines()) if text else 0,
        "missing_markers": missing_markers,
        "ok": bool(path.exists()) and not missing_markers,
    }


def build_documentation_reporting_intelligence(project_root: Path, signal_bus: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(signal_bus.get("summary"))
    readme = _document_snapshot(
        project_root,
        "README.md",
        (
            "Auto-Refreshed Highlights",
            "COMMANDS.md",
            "docs/showcase/generated/highlights_latest.md",
        ),
    )
    commands = _document_snapshot(
        project_root,
        "COMMANDS.md",
        (
            "Live Feed Views",
            "Reports And PDFs",
            "docs-reporting-intelligence",
        ),
    )
    commands_hygiene = load_json(project_root / "governance" / "health" / "commands_hygiene_latest.json")
    commands_contract = load_json(project_root / "governance" / "health" / "commands_contract_latest.json")
    report_quality = load_json(project_root / "governance" / "health" / "report_quality_guard_latest.json")
    report_bundle = load_json(project_root / "governance" / "health" / "report_pdf_bundle_latest.json")
    pycharm_highlights = load_json(project_root / DEFAULT_PYCHARM_HIGHLIGHTS_PATH.relative_to(PROJECT_ROOT))
    report_entries = [row for row in _as_list(report_bundle.get("entries")) if isinstance(row, dict)]
    report_error_entries = [
        row
        for row in report_entries
        if not bool(row.get("ok", False)) and str(row.get("status") or "").lower() not in {"ok", "ready"}
    ]
    permission_errors = [
        row
        for row in report_error_entries
        if "permission_error" in str(row.get("detail") or "").lower()
    ]
    command_issues = [str(item) for item in _as_list(commands_hygiene.get("issues"))]
    blockers: list[str] = []
    advisories: list[str] = []
    if not bool(readme.get("exists")):
        blockers.append("readme_missing")
    if not bool(commands.get("exists")):
        blockers.append("commands_md_missing")
    if command_issues:
        blockers.append("commands_hygiene_issues")
    if _as_list(readme.get("missing_markers")):
        advisories.append("readme_missing_operator_markers")
    if _as_list(commands.get("missing_markers")):
        advisories.append("commands_missing_operator_markers")
    if not commands_hygiene:
        advisories.append("commands_hygiene_artifact_missing")
    if not commands_contract:
        advisories.append("commands_contract_artifact_missing")
    if not report_quality:
        advisories.append("report_quality_guard_artifact_missing")
    if report_error_entries:
        advisories.append("report_bundle_has_errors")
    if not pycharm_highlights:
        advisories.append("pycharm_active_bot_highlights_artifact_missing")
    elif _status(pycharm_highlights) != "ready":
        advisories.append("pycharm_active_bot_highlights_not_ready")
    status = "blocked" if blockers else "advisory" if advisories else "ready"
    action = "observe_docs_reporting_contract"
    if command_issues or not commands_hygiene:
        action = "run_commands_hygiene_apply"
    elif report_error_entries or not report_quality:
        action = "run_report_quality_guard_repair"
    elif advisories:
        action = "refresh_readme_commands_reporting_links"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "documentation_reporting_intelligence",
        "ok": not blockers,
        "overall_status": status,
        "decision_packet": {
            "action": action,
            "safe_next_commands": [
                ["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "report-quality-guard", "--repair", "--json"],
                ["./scripts/ops/opsctl.sh", "pycharm-active-bot-highlights", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"],
            ],
            "blockers": blockers,
            "advisories": advisories,
        },
        "readme_layer": readme,
        "commands_layer": {
            **commands,
            "hygiene_status": _status(commands_hygiene),
            "hygiene_ok": bool(commands_hygiene.get("ok", False)) if commands_hygiene else False,
            "contract_entry_count": _safe_int(commands_contract.get("entry_count"), _safe_int(commands_hygiene.get("command_contract_entry_count"), 0)),
            "contract_hash": str(commands_contract.get("contract_hash") or commands_hygiene.get("command_contract_hash") or ""),
            "issue_count": len(command_issues),
        },
        "reporting_layer": {
            "quality_status": _status(report_quality),
            "quality_ok": bool(report_quality.get("ok", False)) if report_quality else False,
            "bundle_status": str(report_bundle.get("overall_status") or report_bundle.get("status") or "missing"),
            "bundle_entry_count": len(report_entries),
            "bundle_error_count": len(report_error_entries),
            "permission_error_count": len(permission_errors),
            "index_ok": bool(report_bundle.get("index_ok", False)),
            "index_html_ok": bool(report_bundle.get("index_html_ok", False)),
        },
        "pycharm_visibility_layer": {
            "blue_active_marker_html": "<span style=\"color:#0b5cad;font-weight:700\">ACTIVE</span>",
            "project_file_color_status": _status(pycharm_highlights),
            "project_file_color": str(pycharm_highlights.get("file_color") or ""),
            "project_scope_strategy": str(pycharm_highlights.get("scope_strategy") or ""),
            "project_scope_pattern_bytes": _safe_int(pycharm_highlights.get("scope_pattern_bytes"), 0),
            "project_view_style": str(pycharm_highlights.get("project_view_style") or ""),
            "foreground_blue_source": str(pycharm_highlights.get("foreground_blue_source") or ""),
            "foreground_blue_supported_without_dirtying_files": bool(
                pycharm_highlights.get("foreground_blue_supported_without_dirtying_files", False)
            ),
            "project_active_core_bot_files": _safe_int(pycharm_highlights.get("active_core_bot_file_count"), 0),
            "project_inactive_core_bot_files": _safe_int(pycharm_highlights.get("inactive_core_bot_file_count"), 0),
            "project_file_colors_path": str(pycharm_highlights.get("file_colors_path") or ""),
            "project_workspace_path": str(pycharm_highlights.get("workspace_path") or ""),
            "active_bots": _safe_int(summary.get("active_bots"), 0),
            "collection_bots": _safe_int(summary.get("collection_bots"), 0),
            "paper_live_data_bots": _safe_int(summary.get("paper_live_data_bots"), 0),
            "active_rows_visible_in_index": True,
        },
        "contract": {
            "does_not_trade": True,
            "does_not_execute_commands": True,
            "writes": ["documentation_reporting_intelligence_latest.json", "docs/pycharm/intelligence_layers_latest.md"],
            "guards": [
                "README.md",
                "COMMANDS.md",
                "report_quality_guard_latest.json",
                "report_pdf_bundle_latest.json",
                "pycharm_active_bot_highlights_latest.json",
            ],
        },
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def build_pycharm_index_payload(payload: dict[str, Any]) -> dict[str, Any]:
    signal_bus = _as_dict(payload.get("system_signal_bus"))
    summary = _as_dict(signal_bus.get("summary"))
    registry_metrics = _as_dict(signal_bus.get("registry_metrics"))
    super_layer = _as_dict(payload.get("system_super_intelligence"))
    super_decision = _as_dict(super_layer.get("decision_packet"))
    outcome_layer = _as_dict(payload.get("super_intelligence_outcome_learning"))
    outcome = _as_dict(outcome_layer.get("intervention_outcome"))
    recursive_layer = _as_dict(payload.get("system_recursive_intelligence"))
    recursive_next = _as_dict(recursive_layer.get("next_more_advanced_layer"))
    docs_reporting = _as_dict(payload.get("documentation_reporting_intelligence"))
    docs_reporting_decision = _as_dict(docs_reporting.get("decision_packet"))
    paper_lane = _as_dict(super_layer.get("paper_lane_governor_layer"))
    symbol_universe = _as_dict(super_layer.get("symbol_universe_intelligence_layer"))
    cognitive_twin = _as_dict(super_layer.get("cognitive_twin_counterfactual_layer"))
    deeper_signal = _signal_by_name(signal_bus, "deeper_intelligence_layers")
    deeper_metrics = _as_dict(deeper_signal.get("metrics"))
    handoff = _as_dict(payload.get("codex_handoff"))
    packet = _as_dict(handoff.get("attention_packet"))
    artifacts = {
        "coordinator": PROJECT_ROOT / "scripts" / "ops" / "system_intelligence_coordinator.py",
        "whole_system": DEFAULT_OUT_PATH,
        "signal_bus": DEFAULT_SIGNAL_BUS_PATH,
        "system_brain": DEFAULT_BRAIN_PATH,
        "self_intelligence": DEFAULT_SELF_INTELLIGENCE_PATH,
        "super_intelligence": DEFAULT_SUPER_INTELLIGENCE_PATH,
        "outcome_learning": DEFAULT_OUTCOME_LEARNING_PATH,
        "recursive_intelligence": DEFAULT_RECURSIVE_INTELLIGENCE_PATH,
        "deeper_intelligence_layers": DEFAULT_DEEPER_INTELLIGENCE_PATH,
        "codex_handoff_json": DEFAULT_HANDOFF_PATH,
        "codex_handoff_markdown": DEFAULT_HANDOFF_MARKDOWN_PATH,
        "super_override": DEFAULT_SUPER_OVERRIDE_PATH,
        "documentation_reporting_intelligence": DEFAULT_DOCUMENTATION_REPORTING_PATH,
        "pycharm_active_bot_highlights": DEFAULT_PYCHARM_HIGHLIGHTS_PATH,
        "pycharm_index_markdown": DEFAULT_PYCHARM_INDEX_PATH,
        "pycharm_index_json": DEFAULT_PYCHARM_INDEX_JSON_PATH,
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "intelligence_layers_pycharm_index",
        "ok": str(payload.get("overall_status") or "") in {"ready", "advisory", "degraded"},
        "overall_status": str(payload.get("overall_status") or "missing"),
        "bot_activity_snapshot": {
            "active_bots": _safe_int(summary.get("active_bots"), 0),
            "collection_bots": _safe_int(summary.get("collection_bots"), 0),
            "sleeve_profile_count": _safe_int(summary.get("sleeve_profile_count"), 0),
            "paper_live_data_bots": _safe_int(summary.get("paper_live_data_bots"), 0),
            "active_state_marker": "ACTIVE",
            "active_state_marker_color": "blue",
            "active_marker_html": "<span style=\"color:#0b5cad;font-weight:700\">ACTIVE</span>",
            "active_bot_examples": [
                row
                for row in _as_list(registry_metrics.get("active_bot_examples"))
                if isinstance(row, dict)
            ],
            "pycharm_run_panel_note": "bots_can_be_running_under_launchd_or_terminal_even_when_pycharm_run_panel_is_empty",
        },
        "live_visibility": {
            "top_risk": str(summary.get("top_risk") or "none"),
            "memory_pressure_high": bool(summary.get("memory_pressure_high", False)),
            "runtime_pressure_high": bool(summary.get("runtime_pressure_high", False)),
            "storage_critical": bool(summary.get("storage_critical", False)),
            "writer_active": bool(summary.get("writer_active", False)),
            "guard_policy_mode": str(summary.get("guard_policy_mode") or ""),
        },
        "layers": [
            {
                "name": "super_intelligence",
                "status": str(super_layer.get("overall_status") or ""),
                "action": str(super_decision.get("action") or ""),
                "mode": str(super_decision.get("executive_mode") or ""),
                "artifact": str(DEFAULT_SUPER_INTELLIGENCE_PATH),
            },
            {
                "name": "outcome_learning",
                "status": str(outcome_layer.get("overall_status") or ""),
                "verdict": str(outcome.get("verdict") or ""),
                "artifact": str(DEFAULT_OUTCOME_LEARNING_PATH),
            },
            {
                "name": "paper_lane_governor",
                "status": str(paper_lane.get("overall_status") or ""),
                "mode": str(paper_lane.get("paper_lane_posture") or ""),
                "score": _safe_int(paper_lane.get("paper_live_data_enabled_bots"), 0),
                "artifact": str(DEFAULT_SUPER_INTELLIGENCE_PATH),
            },
            {
                "name": "symbol_universe_intelligence",
                "status": str(symbol_universe.get("overall_status") or ""),
                "score": _safe_float(symbol_universe.get("breadth_score"), 0.0),
                "artifact": str(DEFAULT_SUPER_INTELLIGENCE_PATH),
            },
            {
                "name": "cognitive_twin_counterfactuals",
                "status": str(cognitive_twin.get("overall_status") or ""),
                "score": _safe_int(cognitive_twin.get("max_world_risk"), 0),
                "next_layer": str(cognitive_twin.get("recommended_next_world") or ""),
                "artifact": str(DEFAULT_SUPER_INTELLIGENCE_PATH),
            },
            {
                "name": "recursive_policy_evolution",
                "status": str(recursive_layer.get("overall_status") or ""),
                "score": _safe_float(recursive_layer.get("recursive_score"), 0.0),
                "next_layer": str(recursive_next.get("name") or ""),
                "artifact": str(DEFAULT_RECURSIVE_INTELLIGENCE_PATH),
            },
            {
                "name": "deeper_self_awareness_layers",
                "status": str(deeper_signal.get("status") or ""),
                "mode": str(deeper_metrics.get("authority_boundary") or ""),
                "score": _safe_int(deeper_metrics.get("ready_count"), 0),
                "next_layer": ",".join(str(item) for item in _as_list(deeper_metrics.get("top_attention"))[:3]),
                "artifact": str(DEFAULT_DEEPER_INTELLIGENCE_PATH),
            },
            {
                "name": "codex_handoff",
                "status": str(handoff.get("overall_status") or ""),
                "safe_next_command": packet.get("safe_next_command") if isinstance(packet.get("safe_next_command"), list) else [],
                "artifact": str(DEFAULT_HANDOFF_PATH),
            },
            {
                "name": "documentation_reporting_intelligence",
                "status": str(docs_reporting.get("overall_status") or ""),
                "action": str(docs_reporting_decision.get("action") or ""),
                "score": _safe_int(_as_dict(docs_reporting.get("commands_layer")).get("contract_entry_count"), 0),
                "artifact": str(DEFAULT_DOCUMENTATION_REPORTING_PATH),
            },
        ],
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "pycharm_open_paths": [_display_path(path) for path in artifacts.values()],
        "contract": {
            "visible_in_pycharm": True,
            "does_not_execute_commands": True,
            "does_not_trade": True,
        },
    }


def render_pycharm_intelligence_markdown(payload: dict[str, Any], index: dict[str, Any]) -> str:
    signal_bus = _as_dict(payload.get("system_signal_bus"))
    summary = _as_dict(signal_bus.get("summary"))
    packet = _as_dict(_as_dict(payload.get("codex_handoff")).get("attention_packet"))
    docs_reporting = _as_dict(payload.get("documentation_reporting_intelligence"))
    docs_decision = _as_dict(docs_reporting.get("decision_packet"))
    reporting_layer = _as_dict(docs_reporting.get("reporting_layer"))
    commands_layer = _as_dict(docs_reporting.get("commands_layer"))
    pycharm_layer = _as_dict(docs_reporting.get("pycharm_visibility_layer"))
    artifacts = _as_dict(index.get("artifacts"))
    layers = [row for row in _as_list(index.get("layers")) if isinstance(row, dict)]
    bot_snapshot = _as_dict(index.get("bot_activity_snapshot"))
    live_visibility = _as_dict(index.get("live_visibility"))
    active_examples = [row for row in _as_list(bot_snapshot.get("active_bot_examples")) if isinstance(row, dict)]
    blue_marker = str(bot_snapshot.get("active_marker_html") or "ACTIVE")
    lines = [
        "# Intelligence Layers PyCharm Index",
        "",
        f"- Updated UTC: `{index.get('timestamp_utc', '')}`",
        f"- Whole-System Status: `{payload.get('overall_status', '')}`",
        f"- Top Risk: `{summary.get('top_risk', 'none')}`",
        f"- Safe Next Command: `{' '.join(str(item) for item in _as_list(packet.get('safe_next_command')))}`",
        "",
        "## Bot Activity Snapshot",
        "",
        f"- Active Bots: `{bot_snapshot.get('active_bots', 0)}`",
        f"- Data-Collection Bots: `{bot_snapshot.get('collection_bots', 0)}`",
        f"- Paper-Live-Data Bots: `{bot_snapshot.get('paper_live_data_bots', 0)}`",
        f"- Sleeve Profiles: `{bot_snapshot.get('sleeve_profile_count', 0)}`",
        f"- Active Marker: {blue_marker}",
        "- PyCharm Note: bots can be running under launchd or Terminal even when PyCharm's Run panel is empty.",
        "",
        "## Active Bot Rows",
        "",
        "| State | Bot | Sleeve | Paper Live Data | Collection |",
        "| --- | --- | --- | --- | --- |",
    ]
    if active_examples:
        for row in active_examples:
            lines.append(
                "| "
                f"{blue_marker} | "
                f"`{row.get('bot_id', '')}` | "
                f"`{row.get('sleeve_profile', '')}` | "
                f"`{bool(row.get('paper_live_data_enabled', False))}` | "
                f"`{bool(row.get('data_collection_active', False))}` |"
            )
    else:
        lines.append("| `none` | `no_active_bot_examples_found` | `` | `False` | `False` |")
    lines.extend(
        [
            "",
        "## Live Visibility",
        "",
        f"- Memory Pressure High: `{live_visibility.get('memory_pressure_high', False)}`",
        f"- Runtime Pressure High: `{live_visibility.get('runtime_pressure_high', False)}`",
        f"- Storage Critical: `{live_visibility.get('storage_critical', False)}`",
        f"- Writer Active: `{live_visibility.get('writer_active', False)}`",
        f"- Guard Policy Mode: `{live_visibility.get('guard_policy_mode', '')}`",
        "",
        "## Docs Commands Reporting",
        "",
        f"- Layer Status: `{docs_reporting.get('overall_status', '')}`",
        f"- Action: `{docs_decision.get('action', '')}`",
        f"- Command Contract Entries: `{commands_layer.get('contract_entry_count', 0)}`",
        f"- Command Issues: `{commands_layer.get('issue_count', 0)}`",
        f"- Report Bundle Entries: `{reporting_layer.get('bundle_entry_count', 0)}`",
        f"- Report Bundle Errors: `{reporting_layer.get('bundle_error_count', 0)}`",
        f"- Report Permission Errors: `{reporting_layer.get('permission_error_count', 0)}`",
        f"- PyCharm File Color Status: `{pycharm_layer.get('project_file_color_status', '')}`",
        f"- PyCharm Active Core Files: `{pycharm_layer.get('project_active_core_bot_files', 0)}`",
        f"- PyCharm Scope Strategy: `{pycharm_layer.get('project_scope_strategy', '')}`",
        f"- PyCharm Scope Pattern Bytes: `{pycharm_layer.get('project_scope_pattern_bytes', 0)}`",
        f"- PyCharm Project View Style: `{pycharm_layer.get('project_view_style', '')}`",
        f"- PyCharm Foreground Blue Source: `{pycharm_layer.get('foreground_blue_source', '')}`",
        f"- Foreground Blue Without Dirtying Files: `{pycharm_layer.get('foreground_blue_supported_without_dirtying_files', False)}`",
        "",
        "## Intelligence Layers",
        "",
        ]
    )
    for row in layers:
        detail = ordered_unique(
            [
                f"status `{row.get('status', '')}`",
                f"mode `{row.get('mode', '')}`" if row.get("mode") else "",
                f"action `{row.get('action', '')}`" if row.get("action") else "",
                f"verdict `{row.get('verdict', '')}`" if row.get("verdict") else "",
                f"score `{row.get('score', '')}`" if row.get("score") != "" and row.get("score") is not None else "",
                f"next `{row.get('next_layer', '')}`" if row.get("next_layer") else "",
            ]
        )
        lines.append(f"- `{row.get('name', '')}`: {', '.join(detail)}")
    lines.extend(["", "## PyCharm Open Paths", ""])
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path))
        lines.append(f"- `{name}`: `{_display_path(path)}`")
    lines.extend(
        [
            "",
            "## Operator Read",
            "",
            "This page is generated by `scripts/ops/system_intelligence_coordinator.py` so PyCharm has a stable project file for the current intelligence layers, even when the live bots are owned by launchd instead of PyCharm.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root)
    previous_payload = load_json(project_root / "governance" / "health" / "whole_system_intelligence_latest.json")
    memory_events = _read_jsonl(project_root / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl", limit=50)
    signal_bus = build_signal_bus(project_root)
    process_contracts = build_process_contracts(signal_bus)
    system_brain = build_system_brain(signal_bus, process_contracts)
    self_intelligence = build_self_intelligence(
        signal_bus=signal_bus,
        system_brain=system_brain,
        process_contracts=process_contracts,
        previous_payload=previous_payload,
        memory_events=memory_events,
    )
    super_memory_events = _read_jsonl(
        project_root / DEFAULT_SUPER_MEMORY_PATH.relative_to(PROJECT_ROOT),
        limit=50,
    )
    super_intelligence = build_super_intelligence(
        signal_bus=signal_bus,
        system_brain=system_brain,
        process_contracts=process_contracts,
        self_intelligence=self_intelligence,
        previous_payload=previous_payload,
        memory_events=super_memory_events,
    )
    outcome_events = _read_jsonl(
        project_root / DEFAULT_OUTCOME_MEMORY_PATH.relative_to(PROJECT_ROOT),
        limit=80,
    )
    outcome_learning = build_outcome_learning(
        signal_bus=signal_bus,
        system_brain=system_brain,
        self_intelligence=self_intelligence,
        super_intelligence=super_intelligence,
        outcome_events=outcome_events,
    )
    recursive_events = _read_jsonl(
        project_root / DEFAULT_RECURSIVE_MEMORY_PATH.relative_to(PROJECT_ROOT),
        limit=80,
    )
    recursive_intelligence = build_recursive_intelligence(
        signal_bus=signal_bus,
        super_intelligence=super_intelligence,
        outcome_learning=outcome_learning,
        recursive_events=recursive_events,
    )
    documentation_reporting_intelligence = build_documentation_reporting_intelligence(project_root, signal_bus)
    codex_handoff = build_codex_handoff(
        signal_bus=signal_bus,
        system_brain=system_brain,
        process_contracts=process_contracts,
        self_intelligence=self_intelligence,
        super_intelligence=super_intelligence,
        outcome_learning=outcome_learning,
        recursive_intelligence=recursive_intelligence,
    )
    status = str(system_brain.get("overall_status") or "missing")
    if str(recursive_intelligence.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(super_intelligence.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(self_intelligence.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(outcome_learning.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(recursive_intelligence.get("overall_status") or "") == "degraded" and status in {"ready", "advisory"}:
        status = "degraded"
    elif str(outcome_learning.get("overall_status") or "") == "degraded" and status in {"ready", "advisory"}:
        status = "degraded"
    elif str(super_intelligence.get("overall_status") or "") == "degraded" and status in {"ready", "advisory"}:
        status = "degraded"
    elif str(self_intelligence.get("overall_status") or "") == "degraded" and status == "ready":
        status = "degraded"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "whole_system_intelligence",
        "ok": status == "ready",
        "overall_status": status,
        "system_signal_bus": signal_bus,
        "system_brain": system_brain,
        "system_process_contracts": process_contracts,
        "system_self_intelligence": self_intelligence,
        "system_super_intelligence": super_intelligence,
        "super_intelligence_outcome_learning": outcome_learning,
        "system_recursive_intelligence": recursive_intelligence,
        "documentation_reporting_intelligence": documentation_reporting_intelligence,
        "codex_handoff": codex_handoff,
        "control_contract": {
            "signal_bus": "normalizes_runtime_storage_writer_drainer_sleeve_safety_and_quality_signals",
            "system_brain": "chooses_next_safe_infrastructure_action_without_executing_it",
            "process_contracts": "declares_authority_boundaries_concurrency_limits_and_recovery_behaviors",
            "system_self_intelligence": "compares_prior_runs_detects_uncertainty_tracks_action_effects_diagnoses_causes_routes_consumers_and_sets_pre_action_reflexes",
            "system_super_intelligence": "ranks_cross_layer_attention_sets_executive_mode_runs_regime_drift_objective_guardrail_adversarial_and_semantic_synthesis_layers_then_routes_safe_next_infrastructure_action",
            "super_intelligence_outcome_learning": "scores_recent_interventions_assigns_policy_credit_and_bounds_mutation_candidates",
            "system_recursive_intelligence": "runs_recursive_policy_evolution_with_invariant_firewall_and_next_layer_backlog",
            "documentation_reporting_intelligence": "guards_readme_commands_and_reporting_surfaces_then_feeds_pycharm_visibility",
            "codex_handoff": "writes_attention_packet_for_codex_and_operator_review",
            "pycharm_visibility": "writes_docs_pycharm_intelligence_layers_latest_markdown_and_health_json_index_when_outputs_are_applied",
            "trade_authority": "none",
            "single_sql_writer_only": True,
        },
    }


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path,
    signal_bus_path: Path,
    brain_path: Path,
    contracts_path: Path,
    self_intelligence_path: Path,
    handoff_path: Path,
    handoff_markdown_path: Path,
    super_intelligence_path: Path | None = None,
    outcome_learning_path: Path | None = None,
    recursive_intelligence_path: Path | None = None,
    documentation_reporting_path: Path | None = None,
    memory_path: Path | None = None,
    super_memory_path: Path | None = None,
    outcome_memory_path: Path | None = None,
    recursive_memory_path: Path | None = None,
    super_override_path: Path | None = None,
    pycharm_index_path: Path | None = None,
    pycharm_index_json_path: Path | None = None,
    context_path: Path | None = None,
) -> None:
    write_payload(out_path, payload)
    write_payload(signal_bus_path, _as_dict(payload.get("system_signal_bus")))
    write_payload(brain_path, _as_dict(payload.get("system_brain")))
    write_payload(contracts_path, _as_dict(payload.get("system_process_contracts")))
    self_intelligence = _as_dict(payload.get("system_self_intelligence"))
    write_payload(self_intelligence_path, self_intelligence)
    super_intelligence = _as_dict(payload.get("system_super_intelligence"))
    if super_intelligence_path is not None:
        write_payload(super_intelligence_path, super_intelligence)
    outcome_learning = _as_dict(payload.get("super_intelligence_outcome_learning"))
    if outcome_learning_path is not None:
        write_payload(outcome_learning_path, outcome_learning)
    recursive_intelligence = _as_dict(payload.get("system_recursive_intelligence"))
    if recursive_intelligence_path is not None:
        write_payload(recursive_intelligence_path, recursive_intelligence)
    documentation_reporting = _as_dict(payload.get("documentation_reporting_intelligence"))
    if documentation_reporting_path is not None:
        write_payload(documentation_reporting_path, documentation_reporting)
    handoff = _as_dict(payload.get("codex_handoff"))
    write_payload(handoff_path, handoff)
    handoff_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_markdown_path.write_text(render_handoff_markdown(handoff), encoding="utf-8")
    if memory_path is not None:
        event = _as_dict(self_intelligence.get("memory_event"))
        if event:
            _append_jsonl(memory_path, event)
    if super_memory_path is not None:
        event = _as_dict(super_intelligence.get("memory_event"))
        if event:
            _append_jsonl(super_memory_path, event)
    if outcome_memory_path is not None:
        event = _as_dict(outcome_learning.get("memory_event"))
        if event:
            _append_jsonl(outcome_memory_path, event)
    if recursive_memory_path is not None:
        event = _as_dict(recursive_intelligence.get("memory_event"))
        if event:
            _append_jsonl(recursive_memory_path, event)
    if super_override_path is not None and super_intelligence:
        super_override_path.parent.mkdir(parents=True, exist_ok=True)
        super_override_path.write_text(render_super_override(super_intelligence), encoding="utf-8")
    if pycharm_index_path is not None or pycharm_index_json_path is not None:
        index_payload = build_pycharm_index_payload(payload)
        if pycharm_index_json_path is not None:
            write_payload(pycharm_index_json_path, index_payload)
        if pycharm_index_path is not None:
            pycharm_index_path.parent.mkdir(parents=True, exist_ok=True)
            pycharm_index_path.write_text(render_pycharm_intelligence_markdown(payload, index_payload), encoding="utf-8")
    if context_path is not None:
        write_payload(context_path, payload)


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate the whole-system signal bus, brain, process contracts, and Codex handoff channel.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--signal-bus-file", default=str(DEFAULT_SIGNAL_BUS_PATH))
    parser.add_argument("--brain-file", default=str(DEFAULT_BRAIN_PATH))
    parser.add_argument("--contracts-file", default=str(DEFAULT_CONTRACTS_PATH))
    parser.add_argument("--self-intelligence-file", default=str(DEFAULT_SELF_INTELLIGENCE_PATH))
    parser.add_argument("--super-intelligence-file", default=str(DEFAULT_SUPER_INTELLIGENCE_PATH))
    parser.add_argument("--outcome-learning-file", default=str(DEFAULT_OUTCOME_LEARNING_PATH))
    parser.add_argument("--recursive-intelligence-file", default=str(DEFAULT_RECURSIVE_INTELLIGENCE_PATH))
    parser.add_argument("--documentation-reporting-file", default=str(DEFAULT_DOCUMENTATION_REPORTING_PATH))
    parser.add_argument("--handoff-file", default=str(DEFAULT_HANDOFF_PATH))
    parser.add_argument("--handoff-markdown-file", default=str(DEFAULT_HANDOFF_MARKDOWN_PATH))
    parser.add_argument("--self-memory-file", default=str(DEFAULT_SELF_MEMORY_PATH))
    parser.add_argument("--super-memory-file", default=str(DEFAULT_SUPER_MEMORY_PATH))
    parser.add_argument("--outcome-memory-file", default=str(DEFAULT_OUTCOME_MEMORY_PATH))
    parser.add_argument("--recursive-memory-file", default=str(DEFAULT_RECURSIVE_MEMORY_PATH))
    parser.add_argument("--super-override-file", default=str(DEFAULT_SUPER_OVERRIDE_PATH))
    parser.add_argument("--pycharm-index-file", default=str(DEFAULT_PYCHARM_INDEX_PATH))
    parser.add_argument("--pycharm-index-json-file", default=str(DEFAULT_PYCHARM_INDEX_JSON_PATH))
    parser.add_argument("--context-file", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    write_outputs(
        payload,
        out_path=_resolve(project_root, args.out_file),
        signal_bus_path=_resolve(project_root, args.signal_bus_file),
        brain_path=_resolve(project_root, args.brain_file),
        contracts_path=_resolve(project_root, args.contracts_file),
        self_intelligence_path=_resolve(project_root, args.self_intelligence_file),
        super_intelligence_path=_resolve(project_root, args.super_intelligence_file),
        outcome_learning_path=_resolve(project_root, args.outcome_learning_file),
        recursive_intelligence_path=_resolve(project_root, args.recursive_intelligence_file),
        documentation_reporting_path=_resolve(project_root, args.documentation_reporting_file),
        handoff_path=_resolve(project_root, args.handoff_file),
        handoff_markdown_path=_resolve(project_root, args.handoff_markdown_file),
        memory_path=_resolve(project_root, args.self_memory_file),
        super_memory_path=_resolve(project_root, args.super_memory_file),
        outcome_memory_path=_resolve(project_root, args.outcome_memory_file),
        recursive_memory_path=_resolve(project_root, args.recursive_memory_file),
        super_override_path=_resolve(project_root, args.super_override_file) if args.apply else None,
        pycharm_index_path=_resolve(project_root, args.pycharm_index_file),
        pycharm_index_json_path=_resolve(project_root, args.pycharm_index_json_file),
        context_path=_resolve(project_root, args.context_file) if args.apply else None,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        decision = _as_dict(_as_dict(payload.get("system_brain")).get("decision_packet"))
        super_decision = _as_dict(_as_dict(payload.get("system_super_intelligence")).get("decision_packet"))
        print(
            "whole_system_intelligence "
            f"status={payload.get('overall_status', '')} "
            f"action={decision.get('action', '')} "
            f"top_risk={decision.get('top_risk', '')} "
            f"super_mode={super_decision.get('executive_mode', '')}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
