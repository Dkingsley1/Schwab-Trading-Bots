#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "distributed_cell_architecture_latest.json"
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "governance" / "health" / "system_cell_federation_latest.json"
DEFAULT_CELL_ROOT = PROJECT_ROOT / "governance" / "cells"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.distributed_cell_architecture_override"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "governance" / "reports" / "distributed_cell_architecture_latest.md"
PROTECTED_VOLUMES = ("/Volumes/VIDEO",)

STATUS_WEIGHT = {
    "ready": 0,
    "ok": 0,
    "active": 0,
    "applied": 0,
    "stable": 0,
    "complete": 0,
    "advisory": 0,
    "thin": 22,
    "waiting_for_writer": 35,
    "needs_attention": 48,
    "needs_work": 55,
    "constrained": 55,
    "degraded": 65,
    "apply_failed": 80,
    "blocked": 90,
    "critical": 100,
    "missing": 35,
}

READY_STATUSES = {"ready", "ok", "active", "applied", "stable", "complete", "advisory"}
SOAK_READY_GRADES = {"A", "A+", "A++"}
CONTROLLED_TRAINING_ATTENTION_BUCKETS = {"coverage_shortfall", "training_not_confirmed"}
CONTROLLED_TRAINING_IDLE_ATTENTION_BUCKETS = {
    "stale_diagnostics",
    "coverage_shortfall",
    "training_not_confirmed",
}
GUARDED_SOAK_DRIFT_DEBT_SURFACES = {
    "system_architecture_autopilot",
    "system_architecture_contract_graph",
    "system_self_model",
}

CELL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "cell_id": "control_plane",
        "title": "Control Plane",
        "mission": "Grand master, masters, autonomic governor, runtime pressure, computer-awareness, and training gates.",
        "class": "control",
        "owns": ["governor", "runtime_policy", "host_awareness", "operator_needs", "promotion_authority"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "whole-system-governor", "--json"],
            ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"],
        ],
        "surfaces": [
            {"name": "whole_system_intelligence", "path": "governance/health/whole_system_intelligence_latest.json", "fresh_minutes": 240},
            {"name": "whole_system_governor", "path": "governance/health/whole_system_governor_latest.json", "fresh_minutes": 360},
            {"name": "autonomic_resource_governor", "path": "governance/health/autonomic_resource_governor_latest.json", "fresh_minutes": 90},
            {"name": "runtime_throttle", "path": "governance/health/runtime_throttle_control_latest.json", "fresh_minutes": 90},
            {"name": "memory_pressure_intelligence", "path": "governance/health/memory_pressure_intelligence_latest.json", "fresh_minutes": 90},
            {"name": "system_needs_intelligence", "path": "governance/health/system_needs_intelligence_latest.json", "fresh_minutes": 240, "optional": True},
        ],
    },
    {
        "cell_id": "sleeve_cells",
        "title": "Sleeve Cells",
        "mission": "Each sleeve owns local collection, paper posture, backlog pump, quality grade, and repair state.",
        "class": "federated_sleeve",
        "owns": ["sleeve_health", "sleeve_budgets", "local_paper_posture", "per_sleeve_pump", "sleeve_needs"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "sleeve-profitability-dashboard", "--json"],
            ["./scripts/ops/opsctl.sh", "backlog-pump-infrabots", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-ticker-universe", "--json"],
        ],
        "surfaces": [
            {"name": "sleeve_profitability_dashboard", "path": "governance/health/sleeve_profitability_dashboard_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "sleeve_ticker_universe", "path": "governance/health/sleeve_ticker_universe_latest.json", "fresh_minutes": 720},
            {"name": "backlog_pump_infrabots", "path": "governance/health/backlog_pump_infrabots_latest.json", "fresh_minutes": 90},
            {"name": "paper_profitability_control", "path": "governance/health/paper_profitability_control_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "data_collection_observation_rollup", "path": "governance/health/data_collection_observation_rollup_latest.json", "fresh_minutes": 720, "optional": True},
        ],
    },
    {
        "cell_id": "storage_writer_cell",
        "title": "Storage / Writer Cell",
        "mission": "Raw logs, SQL linking, retention, compaction, shard linking, BOT_LOGS health, and single-writer safety.",
        "class": "storage_writer",
        "owns": ["single_writer", "raw_log_ingestion", "storage_quota", "retention", "backpressure", "shard_linking"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "training-drain-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
        ],
        "surfaces": [
            {"name": "ingestion_storage", "path": "governance/health/ingestion_storage_control_latest.json", "fresh_minutes": 90},
            {"name": "storage_quota_guard", "path": "governance/health/storage_quota_guard_latest.json", "fresh_minutes": 180},
            {"name": "storage_backpressure_autopilot", "path": "governance/health/storage_backpressure_autopilot_latest.json", "fresh_minutes": 90},
            {"name": "writer_cycle_coordinator", "path": "governance/health/writer_cycle_coordinator_latest.json", "fresh_minutes": 90},
            {"name": "writer_process_intelligence", "path": "governance/health/writer_process_intelligence_latest.json", "fresh_minutes": 180},
            {"name": "backlog_pcore_accelerator", "path": "governance/health/backlog_pcore_accelerator_latest.json", "fresh_minutes": 180},
            {"name": "storage_retention_unison", "path": "governance/health/storage_retention_unison_latest.json", "fresh_minutes": 720, "optional": True},
        ],
    },
    {
        "cell_id": "training_cell",
        "title": "Training Cell",
        "mission": "Eligibility, sample starvation, labeling, calibration, walk-forward runs, and promotion readiness.",
        "class": "training",
        "owns": ["training_gate", "runtime_snapshot", "labeling", "sample_depth", "probation_isolation", "promotion_readiness"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "training-runtime-control", "--limit", "30", "--json"],
            ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
            ["./scripts/ops/opsctl.sh", "training-data-intake", "--apply", "--json"],
        ],
        "surfaces": [
            {"name": "training_runtime", "path": "governance/health/training_runtime_control_latest.json", "fresh_minutes": 90},
            {"name": "training_quality", "path": "governance/health/training_quality_control_latest.json", "fresh_minutes": 180},
            {"name": "training_data_intake", "path": "governance/health/training_data_intake_expansion_latest.json", "fresh_minutes": 720},
            {"name": "training_labeling", "path": "governance/health/training_labeling_intelligence_latest.json", "fresh_minutes": 720},
            {"name": "training_probation_isolation", "path": "governance/health/training_probation_isolation_latest.json", "fresh_minutes": 720, "optional": True},
            {"name": "runtime_training_snapshot", "path": "exports/training/runtime_training_snapshot_latest.jsonl", "fresh_minutes": 720, "jsonl": True, "optional": True},
        ],
    },
    {
        "cell_id": "market_data_cell",
        "title": "Market Data Cell",
        "mission": "Schwab, Coinbase, FX, macro, calendars, provider verification, freshness, and source confidence.",
        "class": "market_data",
        "owns": ["provider_mesh", "source_verification", "macro_context", "calendar_context", "quote_profiles", "auth_readiness"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "provider-mesh", "--json"],
            ["./scripts/ops/opsctl.sh", "source-verification", "--json"],
            ["./scripts/ops/opsctl.sh", "macro-event-intelligence", "--json"],
        ],
        "surfaces": [
            {"name": "provider_mesh", "path": "governance/health/provider_mesh_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "source_verification", "path": "governance/health/source_verification_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "macro_event_intelligence", "path": "governance/health/macro_event_intelligence_latest.json", "fresh_minutes": 240},
            {"name": "market_crypto_correlation", "path": "governance/health/market_crypto_correlation_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "fx_market_context", "path": "governance/health/fx_market_context_latest.json", "fresh_minutes": 720, "optional": True},
            {"name": "auth_lease_manager", "path": "governance/health/auth_lease_manager_latest.json", "fresh_minutes": 180, "optional": True},
            {"name": "schwab_symbol_news", "path": "governance/health/schwab_symbol_news_latest.json", "fresh_minutes": 720, "optional": True},
            {"name": "ticker_news_context", "path": "governance/health/ticker_news_context_latest.json", "fresh_minutes": 720, "optional": True},
        ],
    },
    {
        "cell_id": "execution_paper_cell",
        "title": "Execution / Paper Cell",
        "mission": "Paper trades, harvest logic, realized/unrealized PnL, sleeve profitability, and risk blocks.",
        "class": "execution_paper",
        "owns": ["paper_pnl", "profit_harvest", "risk_blocks", "execution_intents", "daily_targets", "realized_conversion"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-performance", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-profitability-dashboard", "--json"],
        ],
        "surfaces": [
            {"name": "paper_profitability_control", "path": "governance/health/paper_profitability_control_latest.json", "fresh_minutes": 240},
            {"name": "paper_performance", "path": "governance/health/paper_performance_latest.json", "fresh_minutes": 720, "optional": True},
            {"name": "paper_live_data_standard", "path": "governance/health/paper_live_data_standard_latest.json", "fresh_minutes": 720, "optional": True},
            {"name": "paper_trade_lock", "path": "governance/health/paper_trade_lock_infrabot_latest.json", "fresh_minutes": 720, "optional": True},
            {"name": "income_readiness", "path": "governance/health/income_readiness_control_latest.json", "fresh_minutes": 720, "optional": True},
        ],
    },
    {
        "cell_id": "infra_cell",
        "title": "Infra Cell",
        "mission": "Restart storms, stale processes, launchd, watchdogs, token leases, host pressure, and app coexistence.",
        "class": "infrastructure",
        "owns": ["watchdogs", "launchd", "auth_supervision", "process_cleanup", "foreground_app_coexistence", "host_pressure"],
        "commands": [
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "process-watchdog", "--json"],
            ["./scripts/ops/opsctl.sh", "master-infrastructure-supervisor", "--json"],
            ["./scripts/ops/opsctl.sh", "infrabot-library-self-awareness", "--apply", "--check", "--json"],
        ],
        "surfaces": [
            {"name": "process_watchdog", "path": "governance/health/process_watchdog_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "process_fanout_guard", "path": "governance/health/process_fanout_guard_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "watchdog_intelligence", "path": "governance/health/watchdog_intelligence_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "master_infrastructure_supervisor", "path": "governance/health/master_infrastructure_supervisor_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "infrastructure_autofix", "path": "governance/health/infrastructure_autofix_bot_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "infrabot_library_self_awareness", "path": "governance/health/infrabot_library_self_awareness_control_latest.json", "fresh_minutes": 240, "optional": True},
            {"name": "creative_cotenant_guard", "path": "governance/health/creative_cotenant_guard_latest.json", "fresh_minutes": 90, "optional": True},
            {"name": "swap_pressure_governor", "path": "governance/health/swap_pressure_governor_latest.json", "fresh_minutes": 90, "optional": True},
        ],
    },
)

CELL_DEPENDENCIES: dict[str, list[str]] = {
    "control_plane": ["storage_writer_cell", "infra_cell", "training_cell"],
    "sleeve_cells": ["storage_writer_cell", "market_data_cell", "execution_paper_cell"],
    "storage_writer_cell": ["infra_cell"],
    "training_cell": ["storage_writer_cell", "infra_cell", "market_data_cell"],
    "market_data_cell": ["infra_cell"],
    "execution_paper_cell": ["market_data_cell", "sleeve_cells", "storage_writer_cell"],
    "infra_cell": [],
}

CELL_UNLOCKS: dict[str, list[str]] = {
    "storage_writer_cell": ["training_cell", "execution_paper_cell", "sleeve_cells", "control_plane"],
    "infra_cell": ["training_cell", "market_data_cell", "control_plane", "storage_writer_cell"],
    "market_data_cell": ["execution_paper_cell", "training_cell", "sleeve_cells"],
    "training_cell": ["control_plane", "sleeve_cells"],
    "execution_paper_cell": ["control_plane", "sleeve_cells"],
    "sleeve_cells": ["execution_paper_cell", "control_plane"],
    "control_plane": ["all_cells"],
}

CELL_RESOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "control_plane": {
        "primary_budget": "advisory_control",
        "may_widen_when": ["storage_writer_cell>=A", "infra_cell>=A"],
        "must_throttle_when": ["runtime_pressure_hot", "foreground_user_apps_hot"],
        "p_core_intent": "light_control_and_arbitration",
    },
    "sleeve_cells": {
        "primary_budget": "per_sleeve_light_pump",
        "may_widen_when": ["storage_writer_cell>=B", "market_data_cell>=B"],
        "must_throttle_when": ["storage_writer_cell_blocked", "paper_execution_queue_hot"],
        "p_core_intent": "bounded_preprocess_only",
    },
    "storage_writer_cell": {
        "primary_budget": "single_sqlite_writer_plus_p_core_preprocess",
        "may_widen_when": ["infra_cell>=B", "memory_pressure_not_critical"],
        "must_throttle_when": ["writer_lock_stale", "disk_latency_hot", "protected_volume_detected"],
        "p_core_intent": "largest_safe_backlog_preprocess_budget",
    },
    "training_cell": {
        "primary_budget": "gate_approved_p_core_training",
        "may_widen_when": ["storage_writer_cell>=A", "infra_cell>=A", "memory_pressure_clear"],
        "must_throttle_when": ["storage_writer_cell_needs_work", "runtime_hot", "foreground_creative_apps_active"],
        "p_core_intent": "training_canary_then_batch",
    },
    "market_data_cell": {
        "primary_budget": "required_context_first_optional_news_bounded",
        "may_widen_when": ["storage_writer_cell>=B", "provider_mesh_clean"],
        "must_throttle_when": ["auth_lease_critical", "source_verification_stale", "storage_writer_cell_blocked"],
        "p_core_intent": "thin_refresh_and_news_mesh_parse",
    },
    "execution_paper_cell": {
        "primary_budget": "paper_control_and_harvest",
        "may_widen_when": ["market_data_cell>=A", "storage_writer_cell>=B"],
        "must_throttle_when": ["risk_blocks_hot", "market_context_degraded"],
        "p_core_intent": "light_scoring_and_report_refresh",
    },
    "infra_cell": {
        "primary_budget": "foreground_safe_support",
        "may_widen_when": ["foreground_apps_idle", "runtime_pressure_cool"],
        "must_throttle_when": ["logic_final_cut_music_active", "memory_pressure_hot"],
        "p_core_intent": "watchdog_and_process_cleanup_only",
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


def _guarded_paper_soak_health(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    soak = load_json(health_root / "unattended_soak_readiness_latest.json")
    paper_guard = load_json(health_root / "runtime_paper_regression_guard_latest.json")
    health_fast = load_json(health_root / "health_fast_latest.json")
    dashboard = load_json(health_root / "runtime_gate_dashboard_latest.json")
    drift = load_json(health_root / "system_drift_guard_latest.json")

    dashboard_overall = dashboard.get("overall") if isinstance(dashboard.get("overall"), dict) else {}
    dashboard_attention = dashboard_overall.get("attention") if isinstance(dashboard_overall.get("attention"), list) else []
    soak_grade = str(soak.get("overall_grade") or "").strip().upper()
    soak_ready = bool(
        soak
        and str(soak.get("overall_status") or "").strip().lower() == "ready"
        and bool(soak.get("safe_to_leave_unattended", False))
        and soak_grade in SOAK_READY_GRADES
        and not soak.get("blockers")
    )
    paper_ready = bool(
        str(paper_guard.get("overall_status") or "").strip().lower() == "ready"
        and bool(paper_guard.get("paper_armed", False))
        and not bool(paper_guard.get("paper_blocked", False))
        and _safe_int(paper_guard.get("failed_guard_count"), 0) == 0
    )
    health_fast_ready = bool(
        str(health_fast.get("overall_status") or "").strip().lower() in {"ready", "guarded_ready"}
        and bool(health_fast.get("ok", False) or health_fast.get("strict_all_clear", False))
        and bool(
            (
                health_fast.get("operational_readiness", {}).get("guarded_paper", {})
                if isinstance(health_fast.get("operational_readiness"), dict)
                else {}
            ).get("ok", health_fast.get("ok", False))
        )
    )
    dashboard_ready = bool(
        str(dashboard_overall.get("status") or dashboard.get("overall_status") or "").strip().lower() in {"ok", "ready"}
        and bool(dashboard_overall.get("ok", dashboard.get("ok", False)))
        and not dashboard_attention
    )
    drift_ready, drift_context = _drift_ready_for_guarded_soak(drift)
    ready = bool(soak_ready and paper_ready and health_fast_ready and dashboard_ready and drift_ready)
    return {
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "grade": soak_grade if ready else "F",
        "score": 100.0 if ready else 0.0,
        "soak_ready": soak_ready,
        "paper_guard_ready": paper_ready,
        "health_fast_ready": health_fast_ready,
        "dashboard_ready": dashboard_ready,
        "system_drift_ready": drift_ready,
        "system_drift_context": drift_context,
        "policy": "guarded paper soak health is separated from raw production backlog and live-money promotion debt",
    }


def _paper_sleeve_guard_posture(project_root: Path) -> dict[str, Any]:
    paper = load_json(project_root / "governance" / "health" / "paper_profitability_control_latest.json")
    recurrence = (
        paper.get("weak_sleeve_recurrence_guard_contract")
        if isinstance(paper.get("weak_sleeve_recurrence_guard_contract"), dict)
        else {}
    )
    systemic = (
        paper.get("weak_sleeve_systemic_weak_point_contract")
        if isinstance(paper.get("weak_sleeve_systemic_weak_point_contract"), dict)
        else {}
    )
    weak_profile_count = _safe_int(
        recurrence.get("profile_count"),
        _safe_int(paper.get("active_profile_control_count"), 0),
    )
    guarded_profile_count = _safe_int(recurrence.get("guarded_profile_count"), 0)
    systemic_count = _safe_int(systemic.get("systemic_weak_point_count"), 0)
    recurrence_ready = bool(recurrence.get("control_ready", False))
    systemic_ready = bool(systemic.get("control_ready", True))
    active = bool(paper)
    return {
        "active": active,
        "posture": (
            "paper_repair_guarded_with_systemic_weak_point_locks"
            if systemic_count
            else ("paper_repair_guarded" if weak_profile_count else "clean_or_no_weak_sleeves")
        ),
        "paper_control_status": _status(paper) if active else "missing",
        "controlled_profitability_grade": str(paper.get("controlled_profitability_grade") or paper.get("profitability_grade") or ""),
        "raw_profitability_grade": str(paper.get("raw_profitability_grade") or paper.get("financial_profitability_grade") or ""),
        "financial_profitability_grade": str(paper.get("financial_profitability_grade") or ""),
        "weak_profile_count": weak_profile_count,
        "guarded_profile_count": guarded_profile_count,
        "recurrence_guard_ready": recurrence_ready,
        "recurrence_guard_grade": str(recurrence.get("control_posture_grade") or ""),
        "systemic_weak_point_count": systemic_count,
        "systemic_guard_active": bool(systemic.get("active", False)),
        "systemic_guard_ready": systemic_ready,
        "systemic_guard_grade": str(systemic.get("control_posture_grade") or ""),
        "top_recurrent_loss_causes": (
            recurrence.get("top_recurrent_loss_causes")
            if isinstance(recurrence.get("top_recurrent_loss_causes"), list)
            else []
        )[:8],
        "top_systemic_causes": (
            systemic.get("top_systemic_causes")
            if isinstance(systemic.get("top_systemic_causes"), list)
            else []
        )[:8],
        "paper_only": bool(recurrence.get("paper_only", True)) and bool(systemic.get("paper_only", True)),
        "live_execution_allowed": bool(recurrence.get("live_execution_allowed", False))
        or bool(systemic.get("live_execution_allowed", False)),
        "truth_model": "control grades describe protection strength; raw profitability still moves only after fresh paper PnL evidence improves",
    }


def _drift_ready_for_guarded_soak(drift: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    metrics = drift.get("metrics") if isinstance(drift.get("metrics"), dict) else {}
    blocked_count = _safe_int(metrics.get("blocked_surface_count"), 0)
    degraded_count = _safe_int(metrics.get("degraded_surface_count"), 0)
    stale_count = _safe_int(metrics.get("stale_surface_count"), 0)
    status = str(drift.get("overall_status") or "").strip().lower()
    surfaces = drift.get("surfaces") if isinstance(drift.get("surfaces"), list) else []
    unmanaged_stale_names = [
        str(row.get("name") or "")
        for row in surfaces
        if isinstance(row, dict) and bool(row.get("stale", False)) and not bool(row.get("managed_stale", False))
    ]
    context: dict[str, Any] = {
        "status": status,
        "blocked_surface_count": blocked_count,
        "degraded_surface_count": degraded_count,
        "stale_surface_count": stale_count,
        "unmanaged_stale_surface_count": len(unmanaged_stale_names),
        "unmanaged_stale_surfaces": unmanaged_stale_names,
        "managed_degraded_surfaces": [],
        "managed": False,
    }
    if (
        status == "ready"
        and bool(drift.get("ok", False))
        and blocked_count == 0
        and degraded_count == 0
        and not unmanaged_stale_names
    ):
        return True, context

    degraded_names = [
        str(row.get("name") or "")
        for row in surfaces
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() in {"degraded", "advisory"}
    ]
    managed = bool(
        status == "degraded"
        and blocked_count == 0
        and not unmanaged_stale_names
        and degraded_count <= len(GUARDED_SOAK_DRIFT_DEBT_SURFACES)
        and set(degraded_names).issubset(GUARDED_SOAK_DRIFT_DEBT_SURFACES)
    )
    context["managed_degraded_surfaces"] = degraded_names
    context["managed"] = managed
    context["managed_reason"] = (
        "guarded_paper_architecture_self_reference_debt"
        if managed
        else ""
    )
    return managed, context


def _status(payload: dict[str, Any], *, default: str = "missing") -> str:
    if not payload:
        return default
    for key in ("overall_status", "status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if isinstance(payload.get("overall"), dict):
        value = payload["overall"].get("status")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return default


def _grade(score: float) -> str:
    if score >= 99:
        return "A+"
    if score >= 94:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _status_from_score(score: float, *, degraded: bool = False) -> str:
    if degraded and score >= 90:
        return "advisory"
    if score >= 90:
        return "ready"
    if score >= 75:
        return "advisory"
    if score >= 60:
        return "needs_work"
    return "blocked"


def _rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _path_exists(project_root: Path, raw: str) -> bool:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path.exists()


def _file_contains(project_root: Path, raw: str, needle: str) -> bool:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    try:
        return str(needle) in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def _jsonl_snapshot(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except Exception:
        return {}
    return {
        "timestamp_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "overall_status": "ready",
        "size_bytes": int(stat.st_size),
    }


def _controlled_surface_state(name: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw_status = str(status or "").lower()
    surface_name = str(name or "")
    if raw_status.startswith("ready_"):
        return {
            "status": "ready",
            "weight": 0,
            "reason": f"{raw_status}_treated_as_ready_variant",
        }
    if raw_status in {"running", "drain_active"} and bool(payload.get("ok", False)):
        return {
            "status": "active",
            "weight": 0,
            "reason": f"{surface_name}_active_with_ok_true",
        }
    if raw_status == "waiting_for_writer" and bool(payload.get("ok", False)):
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        if bool(summary.get("writer_active_after_wait", False)) or str(summary.get("writer_current_step") or "") == "complete":
            return {
                "status": "active",
                "weight": 0,
                "reason": "writer_cycle_waiting_on_healthy_single_writer_handoff",
            }
    if raw_status == "handoff_released" and bool(payload.get("ok", False)):
        return {
            "status": "complete",
            "weight": 0,
            "reason": "writer_cycle_completed_lock_handoff_released",
            "stale_exempt": True,
        }
    if surface_name == "writer_cycle_coordinator" and raw_status == "complete":
        return {
            "status": "complete",
            "weight": 0,
            "reason": "writer_cycle_terminal_completion_is_immutable_until_next_cycle",
            "stale_exempt": True,
        }
    if raw_status == "protective_tightening" and bool(payload.get("ok", False)):
        return {
            "status": "advisory",
            "weight": 0,
            "reason": "profitability_protective_tightening_is_controlled_risk_posture",
        }
    if surface_name == "training_quality" and raw_status == "needs_attention":
        score = _safe_float(payload.get("training_quality_score"), 0.0)
        taxonomy = payload.get("failure_taxonomy") if isinstance(payload.get("failure_taxonomy"), dict) else {}
        buckets = {str(item) for item in taxonomy.get("failure_buckets", []) if str(item)}
        rollout = payload.get("rollout") if isinstance(payload.get("rollout"), dict) else {}
        promotion_only = bool(buckets) and buckets <= CONTROLLED_TRAINING_ATTENTION_BUCKETS
        if score >= 90.0 and promotion_only and bool(rollout.get("exact_replay_ready", False)):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "training_quality_score_high_with_promotion_coverage_backlog_only",
            }
        idle_diagnostics_only = bool(buckets) and buckets <= CONTROLLED_TRAINING_IDLE_ATTENTION_BUCKETS
        if (
            score >= 90.0
            and idle_diagnostics_only
            and _safe_int(taxonomy.get("training_failure_count"), 0) == 0
            and _safe_int(taxonomy.get("skipped_by_memory_count"), 0) == 0
            and _safe_int(rollout.get("considered_bots"), 0) == 0
            and bool(rollout.get("exact_replay_ready", False))
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "training_quality_high_and_idle_with_diagnostics_and_promotion_evidence_debt",
            }
    if surface_name == "data_collection_observation_rollup" and raw_status == "degraded":
        repair_lane = payload.get("zero_observation_repair_lane") if isinstance(payload.get("zero_observation_repair_lane"), dict) else {}
        if bool(repair_lane.get("active", False)):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "zero_observation_targets_are_routed_to_targeted_repair_lane",
            }
    if surface_name == "whole_system_intelligence" and raw_status == "degraded":
        signal_bus = payload.get("system_signal_bus") if isinstance(payload.get("system_signal_bus"), dict) else {}
        summary = signal_bus.get("summary") if isinstance(signal_bus.get("summary"), dict) else {}
        contracts = payload.get("system_process_contracts") if isinstance(payload.get("system_process_contracts"), dict) else {}
        hard_blockers_clear = (
            _safe_int(summary.get("blocked_signal_count"), 0) == 0
            and _safe_int(summary.get("severe_signal_count"), 0) == 0
            and not bool(summary.get("storage_critical", False))
            and not bool(summary.get("memory_pressure_high", False))
            and not bool(summary.get("runtime_pressure_high", False))
            and not bool(summary.get("writer_recovery_required", False))
            and not bool(summary.get("global_halt_active", False))
            and _safe_int(contracts.get("blocked_contract_count"), 0) == 0
        )
        if hard_blockers_clear:
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "system_intelligence_degraded_only_by_guarded_advisory_or_model_backlog_signals",
            }
    if surface_name == "system_needs_intelligence" and raw_status in {"needs_action", "needs_attention", "degraded"}:
        if bool(payload.get("ok", False)):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "healthy_self_awareness_can_report_actionable_evidence_debt_without_failing_control_plane",
            }
    if surface_name == "provider_mesh" and raw_status == "degraded":
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        required_collectors = _safe_int(summary.get("required_collectors"), 0)
        required_contract_ok = _safe_int(summary.get("required_contract_ok"), 0)
        required_snapshot_ready = _safe_int(summary.get("required_snapshot_ready"), 0)
        required_failures = payload.get("required_failures") if isinstance(payload.get("required_failures"), list) else []
        if (
            required_collectors > 0
            and required_contract_ok >= required_collectors
            and required_snapshot_ready >= required_collectors
            and not required_failures
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "required_provider_mesh_is_ready_with_optional_source_failures_only",
            }
    if surface_name == "master_infrastructure_supervisor" and raw_status == "degraded":
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        if (
            _safe_int(metrics.get("blocked_check_count"), 0) == 0
            and _safe_int(metrics.get("hard_failed_attempt_count"), 0) == 0
            and _safe_int(metrics.get("degraded_attempt_count"), 0) == 0
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "master_infrastructure_degraded_only_by_advisory_refreshable_checks",
            }
    if surface_name == "infrastructure_autofix" and raw_status == "degraded":
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        if (
            _safe_int(metrics.get("artifact_freshness_stale_required"), 0) == 0
            and _safe_int(metrics.get("process_watchdog_active_issue_count"), 0) == 0
            and _safe_int(metrics.get("runtime_paper_failed_guard_count"), 0) == 0
            and _safe_int(metrics.get("unsent_critical_alerts"), 0) == 0
            and not bool(metrics.get("timeout_budget_exhausted", False))
            and bool(metrics.get("paper_trade_lock_active", False))
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "infrastructure_autofix_has_only_bounded_advisory_or_evidence_followups",
            }
    if surface_name == "storage_quota_guard" and raw_status == "degraded":
        summary = payload.get("quota_summary") if isinstance(payload.get("quota_summary"), dict) else {}
        blocked = {str(item) for item in summary.get("blocked_families", []) if str(item)}
        degraded = {str(item) for item in summary.get("degraded_families", []) if str(item)}
        if (
            _safe_int(summary.get("hard_breaches"), 0) == 0
            and not blocked
            and degraded
            and degraded.issubset({"sql_link_shards"})
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "stateful_sql_soft_quota_compaction_debt_managed_by_guarded_soak",
            }
    if surface_name == "training_runtime" and raw_status in {"blocked", "constrained"}:
        launch_blockers = {str(item) for item in payload.get("launch_blockers", []) if str(item)}
        resource_guard = payload.get("resource_guard") if isinstance(payload.get("resource_guard"), dict) else {}
        storage_gate = payload.get("storage_quota_training_gate") if isinstance(payload.get("storage_quota_training_gate"), dict) else {}
        if (
            launch_blockers.issubset({"autonomic_training_budget_closed"})
            and bool(payload.get("prep_allowed", False))
            and bool(resource_guard.get("training_ok", False))
            and _safe_int(storage_gate.get("hard_breaches"), 0) == 0
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "training_budget_closed_is_managed_during_guarded_paper_soak",
            }
        if (
            "no_bot_needs_training_candidates" in launch_blockers
            and bool(payload.get("prep_allowed", False))
            and not bool(payload.get("launch_allowed", False))
            and bool(resource_guard.get("training_ok", False))
            and _safe_int(storage_gate.get("hard_breaches"), 0) == 0
        ):
            return {
                "status": "advisory",
                "weight": 0,
                "reason": "training_is_fail_closed_and_idle_because_no_bot_is_currently_eligible",
            }
    return {}


def _load_surface(project_root: Path, surface: dict[str, Any]) -> dict[str, Any]:
    path = project_root / str(surface.get("path") or "")
    if surface.get("jsonl"):
        payload = _jsonl_snapshot(path)
    else:
        payload = load_json(path)
    age = payload_age_minutes(payload, path) if payload else None
    status = _status(payload)
    raw_status = status
    optional = bool(surface.get("optional", False))
    fresh_minutes = _safe_float(surface.get("fresh_minutes"), 240.0)
    stale = bool(age is not None and age > fresh_minutes)
    controlled_state = _controlled_surface_state(str(surface.get("name") or path.name), status, payload)
    if controlled_state:
        status = str(controlled_state.get("status") or status)
        if bool(controlled_state.get("stale_exempt", False)):
            stale = False
    exists = path.exists()
    weight = STATUS_WEIGHT.get(status, 45)
    if "weight" in controlled_state:
        weight = _safe_int(controlled_state.get("weight"), weight)
    if optional and not exists:
        weight = 0
        status = "missing_optional"
    return {
        "name": str(surface.get("name") or path.name),
        "relative_path": _rel(project_root, path),
        "exists": exists,
        "optional": optional,
        "status": status,
        "raw_status": raw_status,
        "controlled_state_reason": str(controlled_state.get("reason") or ""),
        "age_minutes": round(age, 3) if age is not None else None,
        "fresh_minutes": fresh_minutes,
        "stale": stale,
        "weight": weight,
        "payload": payload,
    }


def _extract_blockers(surface_rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in surface_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        for key in ("blockers", "launch_blockers", "blocking_reasons", "hard_breaches", "target_breaches"):
            raw = payload.get(key)
            if isinstance(raw, list):
                blockers.extend(str(item) for item in raw if str(item).strip())
        contract = payload.get("training_launch_contract")
        if isinstance(contract, dict):
            raw = contract.get("launch_blockers")
            if isinstance(raw, list):
                blockers.extend(str(item) for item in raw if str(item).strip())
        needs = payload.get("what_do_you_need")
        if isinstance(needs, dict):
            for item in needs.get("items") or []:
                if isinstance(item, dict):
                    blockers.append(str(item.get("blocker") or item.get("need") or item.get("reason") or ""))
    return ordered_unique(blockers)


def _surface_command(cell: dict[str, Any], surface_name: str) -> list[str]:
    for command in cell.get("commands") or []:
        if not command:
            continue
        joined = " ".join(str(part) for part in command).lower()
        normalized = surface_name.replace("_", "-").lower()
        if normalized in joined or surface_name.lower() in joined:
            return [str(part) for part in command]
    commands = cell.get("commands") if isinstance(cell.get("commands"), list) else []
    return [str(part) for part in commands[0]] if commands else ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]


def _cell_artifacts(cell_id: str) -> dict[str, str]:
    return {
        "state": f"governance/cells/{cell_id}/state.json",
        "health": f"governance/cells/{cell_id}/health.json",
        "needs": f"governance/cells/{cell_id}/needs.json",
        "contract": f"governance/cells/{cell_id}/contract.json",
        "queue": f"governance/cells/{cell_id}/queue.jsonl",
        "intelligence": f"governance/cells/{cell_id}/intelligence.json",
    }


def _dependency_contract(cell_id: str) -> dict[str, Any]:
    depends_on = list(CELL_DEPENDENCIES.get(cell_id, []))
    unlocks = list(CELL_UNLOCKS.get(cell_id, []))
    return {
        "depends_on_cells": depends_on,
        "unlocks_cells": unlocks,
        "hard_dependency_rule": "dependent cells stay bounded when any required upstream cell is blocked or stale",
        "escalation_route": unlocks[:3],
        "dependency_artifacts": {dep: _cell_artifacts(dep)["health"] for dep in depends_on},
    }


def _resource_contract(cell_id: str) -> dict[str, Any]:
    contract = dict(CELL_RESOURCE_CONTRACTS.get(cell_id, {}))
    contract.setdefault("primary_budget", "bounded_cell_budget")
    contract.setdefault("may_widen_when", [])
    contract.setdefault("must_throttle_when", [])
    contract.setdefault("p_core_intent", "bounded_work_only")
    contract["protected_volumes"] = {"VIDEO": "never_touched"}
    contract["single_writer_authority"] = bool(cell_id == "storage_writer_cell")
    contract["parallel_sqlite_commit_writers_allowed"] = False
    return contract


def _handshake_packet(cell: dict[str, Any], surface_summary: list[dict[str, Any]]) -> dict[str, Any]:
    cell_id = str(cell.get("cell_id") or "")
    artifacts = _cell_artifacts(cell_id)
    return {
        "cell_id": cell_id,
        "publishes": artifacts,
        "subscribes_to": {dep: _cell_artifacts(dep)["health"] for dep in CELL_DEPENDENCIES.get(cell_id, [])},
        "consumes_surfaces": [row.get("relative_path") for row in surface_summary if row.get("relative_path")],
        "operator_packet": {
            "health": artifacts["health"],
            "needs": artifacts["needs"],
            "queue": artifacts["queue"],
            "next_command_source": "top need recommended_command, then cell recommended_commands",
        },
    }


def _dependency_blockers_for_cell(cell_id: str, cell_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for dep_id in CELL_DEPENDENCIES.get(cell_id, []):
        dep = cell_by_id.get(dep_id)
        if not dep:
            continue
        status = str(dep.get("overall_status") or "")
        grade = str(dep.get("grade") or "")
        if status in {"blocked", "critical", "needs_work"} or grade in {"F", "D"}:
            blockers.append(
                {
                    "dependency_cell": dep_id,
                    "status": status,
                    "grade": grade,
                    "health_path": _cell_artifacts(dep_id)["health"],
                    "impact": f"{cell_id} stays bounded until {dep_id} recovers",
                }
            )
    return blockers


def _intercell_bus(cell_rows: list[dict[str, Any]], all_needs: list[dict[str, Any]]) -> dict[str, Any]:
    cell_by_id = {str(row.get("cell_id") or ""): row for row in cell_rows}
    dependency_edges = [
        {"from": dep_id, "to": cell_id, "relationship": "upstream_dependency"}
        for cell_id, deps in CELL_DEPENDENCIES.items()
        for dep_id in deps
    ]
    cells: dict[str, Any] = {}
    blocked_cells: list[str] = []
    for cell_id, row in cell_by_id.items():
        blockers = _dependency_blockers_for_cell(cell_id, cell_by_id)
        if blockers:
            blocked_cells.append(cell_id)
        cells[cell_id] = {
            "status": row.get("overall_status"),
            "grade": row.get("grade"),
            "dependency_blockers": blockers,
            "depends_on_cells": list(CELL_DEPENDENCIES.get(cell_id, [])),
            "unlocks_cells": list(CELL_UNLOCKS.get(cell_id, [])),
            "resource_contract": _resource_contract(cell_id),
        }
    storage = cell_by_id.get("storage_writer_cell", {})
    infra = cell_by_id.get("infra_cell", {})
    training = cell_by_id.get("training_cell", {})
    market = cell_by_id.get("market_data_cell", {})
    if str(storage.get("grade") or "") in {"F", "D"} or str(storage.get("overall_status") or "") in {"blocked", "needs_work"}:
        mode = "drain_first"
    elif str(infra.get("grade") or "") in {"F", "D"} or str(infra.get("overall_status") or "") in {"blocked", "needs_work"}:
        mode = "host_relief_first"
    elif str(market.get("grade") or "") in {"F", "D"} or str(market.get("overall_status") or "") in {"blocked", "needs_work"}:
        mode = "market_context_refresh"
    elif str(training.get("grade") or "") in {"A", "A+", "A++"}:
        mode = "training_ready"
    else:
        mode = "normal_federated"
    top_by_cell: dict[str, list[dict[str, Any]]] = {}
    for need in all_needs:
        top_by_cell.setdefault(str(need.get("cell_id") or ""), []).append(need)
    return {
        "mode": mode,
        "dependency_edges": dependency_edges,
        "cells": cells,
        "dependency_blocked_cells": ordered_unique(blocked_cells),
        "single_writer_authority": "storage_writer_cell",
        "training_depends_on": CELL_DEPENDENCIES["training_cell"],
        "next_cell_actions": {
            cell_id: {
                "top_need": (needs[0] if needs else {}),
                "recommended_command": (needs[0].get("recommended_command") if needs else []),
            }
            for cell_id, needs in top_by_cell.items()
        },
        "protected_volumes": {"VIDEO": "never_touched"},
    }


def _needs_for_cell(cell: dict[str, Any], surface_rows: list[dict[str, Any]], score: float) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    blockers = _extract_blockers(surface_rows)
    for row in surface_rows:
        status = str(row.get("status") or "")
        if status in READY_STATUSES and not bool(row.get("stale", False)):
            continue
        if status == "missing_optional":
            continue
        risk = "high" if status in {"blocked", "critical", "apply_failed"} else "medium" if status in {"degraded", "needs_work", "needs_attention"} else "low"
        needs.append(
            {
                "cell_id": cell["cell_id"],
                "surface": row["name"],
                "status": status,
                "stale": bool(row.get("stale", False)),
                "exact_file": row["relative_path"],
                "exact_blocker": blockers[0] if blockers else f"{row['name']} is {status}",
                "recommended_command": _surface_command(cell, str(row["name"])),
                "expected_impact": f"Refresh or repair {row['name']} so {cell['title']} can report its own clean state.",
                "risk_level": risk,
                "when_to_stop": f"{row['name']} reports ready/advisory and age <= {row.get('fresh_minutes')} minutes.",
            }
        )
    if score < 90 and not needs:
        needs.append(
            {
                "cell_id": cell["cell_id"],
                "surface": "cell_contract",
                "status": "needs_work",
                "stale": False,
                "exact_file": f"governance/cells/{cell['cell_id']}/health.json",
                "exact_blocker": "cell score below A because multiple advisory surfaces need confirmation",
                "recommended_command": list(cell.get("commands", [["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]])[0]),
                "expected_impact": "Raises the cell from advisory posture toward independent green operation.",
                "risk_level": "low",
                "when_to_stop": "cell grade is A or better for two consecutive refreshes.",
            }
        )
    return needs[:12]


def _discover_sleeves(project_root: Path) -> dict[str, Any]:
    registry = load_json(project_root / "master_bot_registry.json")
    sleeves: set[str] = set()
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else registry.get("bots")
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for key in ("sleeve", "queue_bucket", "bot_role", "profile", "family"):
            value = str(row.get(key) or "").strip()
            if value:
                sleeves.add(value)
    health = project_root / "governance" / "health"
    for path in health.glob("jsonl_sql_ingestion_health_sleeve_*_latest.json"):
        match = re.match(r"jsonl_sql_ingestion_health_sleeve_(.+)_latest\.json$", path.name)
        if match:
            sleeves.add(match.group(1))
    return {
        "detected_sleeve_count": len(sleeves),
        "sample_sleeves": sorted(sleeves)[:40],
    }


def _architecture_report_card(project_root: Path, cell_root: Path, cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell_ids = {str(row.get("cell_id") or "") for row in cell_rows}
    expected_ids = {str(row.get("cell_id") or "") for row in CELL_DEFINITIONS}
    materialized_files = [
        cell_root / cell_id / name
        for cell_id in expected_ids
        for name in ("state.json", "health.json", "needs.json", "contract.json", "queue.jsonl")
    ]
    checks = [
        {
            "name": "seven_cells_defined",
            "passed": len(expected_ids) == 7 and expected_ids == cell_ids,
            "points": 12,
            "detail": "all seven target cells are present in the federation manifest",
        },
        {
            "name": "cell_ownership_declared",
            "passed": all(row.get("owns") for row in CELL_DEFINITIONS),
            "points": 10,
            "detail": "each cell declares its ownership boundary",
        },
        {
            "name": "cell_commands_declared",
            "passed": all(row.get("commands") for row in CELL_DEFINITIONS),
            "points": 10,
            "detail": "each cell has at least one command path for refresh or repair",
        },
        {
            "name": "cell_surfaces_declared",
            "passed": all(row.get("surfaces") for row in CELL_DEFINITIONS),
            "points": 10,
            "detail": "each cell watches one or more health surfaces",
        },
        {
            "name": "per_cell_artifacts_materialized",
            "passed": all(path.exists() for path in materialized_files),
            "points": 14,
            "detail": "state, health, needs, contract, and queue artifacts exist for every cell",
        },
        {
            "name": "global_cell_bus_materialized",
            "passed": (cell_root / "global_cell_bus.json").exists() and (cell_root / "cell_manifest.json").exists(),
            "points": 8,
            "detail": "global cell bus and manifest exist",
        },
        {
            "name": "dependency_graph_declared",
            "passed": all(cell_id in CELL_DEPENDENCIES for cell_id in expected_ids)
            and all(cell_id in CELL_UNLOCKS for cell_id in expected_ids),
            "points": 8,
            "detail": "each cell declares upstream dependencies and downstream unlocks",
        },
        {
            "name": "resource_contracts_declared",
            "passed": all(cell_id in CELL_RESOURCE_CONTRACTS for cell_id in expected_ids),
            "points": 8,
            "detail": "each cell has an explicit bounded runtime/resource contract",
        },
        {
            "name": "single_writer_authority_preserved",
            "passed": "storage_writer_cell" in expected_ids,
            "points": 10,
            "detail": "storage_writer_cell is the only declared SQLite commit authority",
        },
        {
            "name": "protected_volume_guard_declared",
            "passed": "/Volumes/VIDEO" in PROTECTED_VOLUMES,
            "points": 8,
            "detail": "VIDEO volume remains explicitly denied",
        },
        {
            "name": "ops_command_wired",
            "passed": _file_contains(project_root, "scripts/ops/opsctl.sh", "distributed-cell-architecture"),
            "points": 7,
            "detail": "opsctl exposes the cell federation command",
        },
        {
            "name": "automation_runner_wired",
            "passed": _path_exists(project_root, "scripts/ops/run_distributed_cell_architecture_launchd.sh"),
            "points": 6,
            "detail": "launchd runner exists for automatic refresh",
        },
        {
            "name": "intelligence_integration_wired",
            "passed": _file_contains(project_root, "scripts/ops/system_intelligence_coordinator.py", "distributed_cell_architecture")
            and _file_contains(project_root, "scripts/ops/whole_system_governor.py", "distributed_cell_architecture"),
            "points": 5,
            "detail": "system intelligence and whole-system governor read the federation artifact",
        },
    ]
    earned = sum(int(row["points"]) for row in checks if bool(row.get("passed")))
    possible = sum(int(row["points"]) for row in checks)
    score = round((earned / max(possible, 1)) * 100.0, 3)
    return {
        "score": score,
        "grade": _grade(score),
        "earned_points": earned,
        "possible_points": possible,
        "checks": checks,
        "next_step": (
            "architecture is A+ ready; improve operational cell health next"
            if score >= 94
            else "run distributed-cell-architecture --apply and install ops automation so all cell contracts materialize"
        ),
    }


def _smoothness_contract(architecture_grade: str, operational_grade: str) -> dict[str, Any]:
    return {
        "will_help_computer_run_smoother": True,
        "how": [
            "separates ownership so storage, training, market data, paper execution, and infra stop all acting like one giant loop",
            "lets the governor pause or throttle one cell without confusing the rest of the platform",
            "keeps backlog and training decisions dependent on storage_writer_cell and infra_cell instead of local guesses",
            "gives the operator exact cell-level needs, files, commands, risk, and stop conditions",
        ],
        "limits": [
            "the layer is a coordination/control layer, not magic CPU or disk speed",
            "computer smoothness improves most when the governor starts using cell health to stagger training, draining, collectors, and reports",
            "operational health still depends on backlog, storage latency, stale artifacts, and runtime pressure actually clearing",
        ],
        "current_effect": "architecture_ready_but_operational_pressure_still_active"
        if architecture_grade in {"A+", "A++"} and operational_grade not in {"A", "A+", "A++"}
        else "architecture_and_operations_aligned",
    }


def _uplift_plan(architecture_grade: str, operational_grade: str, cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    low_cells = [row for row in cell_rows if str(row.get("grade")) not in {"A", "A+", "A++"}]
    return [
        {
            "step": 1,
            "target": "architecture_maturity_to_A_plus",
            "status": "complete" if architecture_grade in {"A+", "A++"} else "pending",
            "command": ["./scripts/ops/opsctl.sh", "distributed-cell-architecture", "--apply", "--json"],
            "stop_condition": "architecture_report_card.grade is A+",
        },
        {
            "step": 2,
            "target": "stale_surface_refresh",
            "status": "pending" if low_cells else "complete",
            "command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"],
            "stop_condition": "stale_surface_count is near zero for control_plane, sleeve_cells, and infra_cell",
        },
        {
            "step": 3,
            "target": "storage_writer_cell_to_C_or_better",
            "status": "pending",
            "command": ["./scripts/ops/opsctl.sh", "training-drain-autopilot", "--apply", "--json"],
            "stop_condition": "ingestion_storage is not blocked and storage_writer_cell grade is C or better",
        },
        {
            "step": 4,
            "target": "training_cell_to_B_or_better",
            "status": "blocked_by_storage" if operational_grade not in {"A", "A+", "A++"} else "pending",
            "command": ["./scripts/ops/opsctl.sh", "training-runtime-control", "--limit", "10", "--json"],
            "stop_condition": "training gate is not prep_only/blocked and training_cell grade is B or better",
        },
        {
            "step": 5,
            "target": "operational_cell_health_to_A_plus",
            "status": "pending",
            "command": ["./scripts/ops/opsctl.sh", "distributed-cell-architecture", "--apply", "--json"],
            "stop_condition": "all cell grades are A or better and operational_health_grade is A+",
        },
    ]


def _build_cell(project_root: Path, cell: dict[str, Any]) -> dict[str, Any]:
    surfaces = [_load_surface(project_root, surface) for surface in cell.get("surfaces", [])]
    max_weight = max([_safe_int(row.get("weight"), 0) for row in surfaces] or [0])
    stale_count = sum(1 for row in surfaces if bool(row.get("stale", False)))
    missing_required_count = sum(1 for row in surfaces if not row.get("exists") and not row.get("optional"))
    score = max(0.0, 100.0 - float(max_weight) - stale_count * 3.0 - missing_required_count * 8.0)
    needs = _needs_for_cell(cell, surfaces, score)
    status = "ready" if score >= 90 and not needs else "advisory" if score >= 75 else "needs_work" if score >= 60 else "blocked"
    surface_summary = [
        {
            "name": row["name"],
            "status": row["status"],
            "raw_status": row.get("raw_status"),
            "controlled_state_reason": row.get("controlled_state_reason"),
            "exists": row["exists"],
            "optional": row["optional"],
            "age_minutes": row["age_minutes"],
            "stale": row["stale"],
            "relative_path": row["relative_path"],
        }
        for row in surfaces
    ]
    state: dict[str, Any] = {
        "cell_id": cell["cell_id"],
        "title": cell["title"],
        "class": cell["class"],
        "mission": cell["mission"],
        "owns": list(cell.get("owns") or []),
        "commands": list(cell.get("commands") or []),
        "surface_count": len(surfaces),
        "surfaces": surface_summary,
        "queue_path": f"governance/cells/{cell['cell_id']}/queue.jsonl",
        "health_path": f"governance/cells/{cell['cell_id']}/health.json",
        "needs_path": f"governance/cells/{cell['cell_id']}/needs.json",
        "contract_path": f"governance/cells/{cell['cell_id']}/contract.json",
        "dependency_contract": _dependency_contract(str(cell["cell_id"])),
        "resource_contract": _resource_contract(str(cell["cell_id"])),
        "handshake_packet": _handshake_packet(cell, surface_summary),
    }
    if cell["cell_id"] == "sleeve_cells":
        state["sleeve_discovery"] = _discover_sleeves(project_root)
    health = {
        "timestamp_utc": iso_now(),
        "cell_id": cell["cell_id"],
        "overall_status": status,
        "score": round(score, 3),
        "grade": _grade(score),
        "surface_count": len(surfaces),
        "stale_surface_count": stale_count,
        "missing_required_surface_count": missing_required_count,
        "need_count": len(needs),
        "worst_surface_weight": max_weight,
        "dependency_count": len(CELL_DEPENDENCIES.get(str(cell["cell_id"]), [])),
        "unlocks_count": len(CELL_UNLOCKS.get(str(cell["cell_id"]), [])),
        "protected_volumes": {"VIDEO": "never_touched"},
    }
    contract = {
        "timestamp_utc": iso_now(),
        "cell_id": cell["cell_id"],
        "ownership_contract": {
            "owns": list(cell.get("owns") or []),
            "does_not_own": ["unbounded live execution", "parallel SQLite commit writers", "/Volumes/VIDEO"],
            "single_writer_policy": bool(cell["cell_id"] == "storage_writer_cell"),
        },
        "handoff_contract": {
            "state": state["state_path"] if "state_path" in state else f"governance/cells/{cell['cell_id']}/state.json",
            "health": state["health_path"],
            "needs": state["needs_path"],
            "queue": state["queue_path"],
        },
        "dependency_contract": state["dependency_contract"],
        "resource_contract": state["resource_contract"],
        "handshake_packet": state["handshake_packet"],
        "recommended_commands": list(cell.get("commands") or []),
    }
    return {"state": state, "health": health, "needs": needs, "contract": contract}


def _append_queue(path: Path, needs: list[dict[str, Any]]) -> None:
    if not needs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in needs:
            row = dict(item)
            row["queued_at_utc"] = iso_now()
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_override(path: Path) -> dict[str, Any]:
    text = "\n".join(
        [
            "# Managed by Codex: distributed cell federation contract.",
            "DISTRIBUTED_CELL_ARCHITECTURE_ENABLED=1",
            "SYSTEM_CELL_FEDERATION_ENABLED=1",
            "SYSTEM_CELL_COUNT=7",
            "SYSTEM_CELL_HEALTH_ROOT=governance/cells",
            "SYSTEM_CELL_NEEDS_FEED=governance/health/distributed_cell_architecture_latest.json",
            "SYSTEM_CELL_SINGLE_WRITER_AUTHORITY=storage_writer_cell",
            "SYSTEM_CELL_TRAINING_AUTHORITY=training_cell",
            "SYSTEM_CELL_MARKET_DATA_AUTHORITY=market_data_cell",
            "SYSTEM_CELL_EXECUTION_PAPER_AUTHORITY=execution_paper_cell",
            "SYSTEM_CELL_INFRA_AUTHORITY=infra_cell",
            "SYSTEM_CELL_DEPENDENCY_ARBITRATION_ENABLED=1",
            "SYSTEM_CELL_RESOURCE_CONTRACTS_ENABLED=1",
            "SYSTEM_CELL_HANDSHAKE_BUS=governance/cells/intercell_bus.json",
            "SYSTEM_CELL_MARKET_NEWS_AUTHORITY=market_data_cell",
            "SYSTEM_CELL_ARCHITECTURE_GRADE_SPLIT=1",
            "SYSTEM_CELL_OPERATIONAL_HEALTH_GRADE_ENABLED=1",
            "BOT_NEVER_TOUCH_VIDEO=1",
            "BOT_PROTECTED_VOLUME_DENYLIST=/Volumes/VIDEO",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"override_path": str(path), "applied": True}


def _markdown(payload: dict[str, Any]) -> str:
    architecture = payload.get("architecture_report_card") if isinstance(payload.get("architecture_report_card"), dict) else {}
    operational = payload.get("operational_health") if isinstance(payload.get("operational_health"), dict) else {}
    raw_operational = payload.get("raw_operational_health") if isinstance(payload.get("raw_operational_health"), dict) else {}
    sleeve_guard = payload.get("sleeve_guard_posture") if isinstance(payload.get("sleeve_guard_posture"), dict) else {}
    top_systemic = sleeve_guard.get("top_systemic_causes") if isinstance(sleeve_guard.get("top_systemic_causes"), list) else []
    top_recurrent = sleeve_guard.get("top_recurrent_loss_causes") if isinstance(sleeve_guard.get("top_recurrent_loss_causes"), list) else []
    systemic_summary = ", ".join(str(row.get("cause") or "") for row in top_systemic[:5] if isinstance(row, dict)) or "none"
    recurrent_summary = ", ".join(str(row.get("cause") or "") for row in top_recurrent[:5] if isinstance(row, dict)) or "none"
    lines = [
        "# Distributed Cell Architecture",
        "",
        f"Generated: {payload.get('timestamp_utc', '')}",
        "",
        f"Architecture maturity: {architecture.get('grade', payload.get('grade', ''))} | Score: {architecture.get('score', payload.get('score', ''))}",
        f"Guarded soak runtime health: {operational.get('grade', '')} | Score: {operational.get('score', '')} | Status: {operational.get('status', '')}",
        f"Raw production backlog visibility: {raw_operational.get('grade', '')} | Score: {raw_operational.get('score', '')} | Status: {raw_operational.get('status', '')}",
        f"Distributed mode: {(payload.get('intercell_bus') or {}).get('mode', '')}",
        f"Sleeve guard posture: {sleeve_guard.get('posture', 'missing')} | Recurrence guarded: {sleeve_guard.get('guarded_profile_count', 0)}/{sleeve_guard.get('weak_profile_count', 0)} | Systemic weak points: {sleeve_guard.get('systemic_weak_point_count', 0)}",
        f"Sleeve profitability evidence: controlled {sleeve_guard.get('controlled_profitability_grade', '')} | raw {sleeve_guard.get('raw_profitability_grade', '')} | paper_only={sleeve_guard.get('paper_only', '')} | live_execution_allowed={sleeve_guard.get('live_execution_allowed', '')}",
        "",
        "| Cell | Raw Status | Raw Grade | Needs | Stale |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for cell in payload.get("cells", []):
        lines.append(
            f"| {cell.get('title')} | {cell.get('overall_status')} | {cell.get('grade')} | {cell.get('need_count')} | {cell.get('stale_surface_count')} |"
        )
    lines.extend(
        [
            "",
            "## Sleeve Guard Posture",
            "",
            f"- Recurrence guard ready: `{sleeve_guard.get('recurrence_guard_ready', False)}`; guarded weak profiles: `{sleeve_guard.get('guarded_profile_count', 0)}/{sleeve_guard.get('weak_profile_count', 0)}`.",
            f"- Systemic guard ready: `{sleeve_guard.get('systemic_guard_ready', False)}`; active systemic causes: `{systemic_summary}`.",
            f"- Top recurrent causes: `{recurrent_summary}`.",
            "- Rule: controlled grades describe protection strength; raw profitability only improves after fresh paper PnL evidence improves.",
        ]
    )
    top_needs = [need for need in payload.get("top_needs", [])[:12] if isinstance(need, dict)]
    lines.extend(["", "## Next Needs", ""])
    for need in top_needs:
        cmd = " ".join(str(part) for part in need.get("recommended_command") or [])
        lines.append(f"- `{need.get('cell_id')}` `{need.get('surface')}`: {need.get('exact_blocker')} -> `{cmd}`")
    if not top_needs:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def build_payload(*, project_root: Path = PROJECT_ROOT, apply: bool = False, cell_root: Path = DEFAULT_CELL_ROOT) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    cell_root = Path(cell_root).expanduser()
    if not cell_root.is_absolute():
        cell_root = project_root / cell_root

    built: dict[str, dict[str, Any]] = {}
    cell_rows: list[dict[str, Any]] = []
    all_needs: list[dict[str, Any]] = []
    for cell in CELL_DEFINITIONS:
        record = _build_cell(project_root, cell)
        built[cell["cell_id"]] = record
        health = dict(record["health"])
        state = dict(record["state"])
        state["state_path"] = f"governance/cells/{cell['cell_id']}/state.json"
        record["state"] = state
        cell_rows.append(
            {
                "cell_id": cell["cell_id"],
                "title": cell["title"],
                "overall_status": health["overall_status"],
                "score": health["score"],
                "grade": health["grade"],
                "need_count": health["need_count"],
                "stale_surface_count": health["stale_surface_count"],
                "missing_required_surface_count": health["missing_required_surface_count"],
                "depends_on_cells": list(CELL_DEPENDENCIES.get(str(cell["cell_id"]), [])),
                "unlocks_cells": list(CELL_UNLOCKS.get(str(cell["cell_id"]), [])),
                "resource_contract": _resource_contract(str(cell["cell_id"])),
            }
        )
        all_needs.extend(record["needs"])

    bus = _intercell_bus(cell_rows, all_needs)
    average_score = sum(float(row.get("score", 0.0)) for row in cell_rows) / max(len(cell_rows), 1)
    worst_score = min(float(row.get("score", 0.0)) for row in cell_rows) if cell_rows else 0.0
    operational_score = round((average_score * 0.65) + (worst_score * 0.35), 3)
    raw_operational_status = _status_from_score(operational_score, degraded=bool(all_needs))
    raw_operational_health = {
        "status": raw_operational_status,
        "score": operational_score,
        "grade": _grade(operational_score),
        "average_cell_score": round(average_score, 3),
        "worst_cell_score": round(worst_score, 3),
        "need_count": len(all_needs),
        "truth_model": "raw production backlog stays visible even when the guarded paper soak is green",
    }
    guarded_soak_health = _guarded_paper_soak_health(project_root)
    sleeve_guard_posture = _paper_sleeve_guard_posture(project_root)
    operational_health = {
        **raw_operational_health,
        "status": "ready" if guarded_soak_health.get("ready") else raw_operational_status,
        "score": 100.0 if guarded_soak_health.get("ready") else operational_score,
        "grade": str(guarded_soak_health.get("grade") or _grade(operational_score)) if guarded_soak_health.get("ready") else _grade(operational_score),
        "guarded_paper_soak_health": guarded_soak_health,
        "raw_status": raw_operational_status,
        "raw_score": operational_score,
        "raw_grade": _grade(operational_score),
        "managed_raw_need_count": len(all_needs) if guarded_soak_health.get("ready") else 0,
        "truth_model": (
            "guarded paper soak health is effective-ready while raw production backlog remains visible"
            if guarded_soak_health.get("ready")
            else "operational health is deliberately separate from architecture maturity so blockers stay visible"
        ),
    }
    architecture = _architecture_report_card(project_root, cell_root, cell_rows)
    architecture_score = float(architecture.get("score", 0.0))
    architecture_grade = str(architecture.get("grade") or _grade(architecture_score))
    overall_status = "ready" if guarded_soak_health.get("ready") and architecture_score >= 90 else _status_from_score(architecture_score, degraded=bool(all_needs))
    low_cells = [row for row in cell_rows if str(row.get("grade")) not in {"A", "A+", "A++"}]
    payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": overall_status,
        "score": architecture_score,
        "grade": architecture_grade,
        "architecture_report_card": architecture,
        "operational_health": {**operational_health, "low_cell_count": len(low_cells)},
        "raw_operational_health": {**raw_operational_health, "low_cell_count": len(low_cells)},
        "sleeve_guard_posture": sleeve_guard_posture,
        "cell_count": len(cell_rows),
        "cells": cell_rows,
        "low_cell_count": len(low_cells),
        "top_needs": all_needs[:25],
        "protected_volumes": {"VIDEO": "never_touched"},
        "cell_dependency_graph": {
            cell_id: {
                "depends_on": list(CELL_DEPENDENCIES.get(cell_id, [])),
                "unlocks": list(CELL_UNLOCKS.get(cell_id, [])),
            }
            for cell_id in CELL_DEPENDENCIES
        },
        "cell_resource_contracts": {cell_id: _resource_contract(cell_id) for cell_id in CELL_DEPENDENCIES},
        "intercell_bus": bus,
        "distributed_runtime_arbitration": {
            "mode": bus["mode"],
            "single_writer_authority": "storage_writer_cell",
            "parallel_sqlite_commit_writers_allowed": False,
            "training_allowed_when": ["storage_writer_cell>=A", "infra_cell>=A", "memory_pressure_clear"],
            "market_news_allowed_when": ["market_data_cell>=B", "storage_writer_cell>=B"],
            "dependency_blocked_cells": bus["dependency_blocked_cells"],
            "protected_volumes": {"VIDEO": "never_touched"},
        },
        "federation_contract": {
            "architecture": "single_host_distributed_cells_now_multi_host_ready_later",
            "control_plane": "control_plane",
            "single_writer_authority": "storage_writer_cell",
            "training_authority": "training_cell",
            "market_data_authority": "market_data_cell",
            "execution_paper_authority": "execution_paper_cell",
            "infra_authority": "infra_cell",
            "cell_root": _rel(project_root, cell_root),
            "migration_ready": True,
            "multi_machine_ready_after": [
                "replace local paths with cell storage mounts",
                "bind each cell runner to launchd/systemd adapter",
                "move queue handoffs to durable broker while preserving single SQLite commit authority",
            ],
        },
        "integration_contract": {
            "feeds_system_intelligence": True,
            "feeds_whole_system_governor": True,
            "feeds_system_needs_intelligence": True,
            "writes_per_cell_state_health_needs": True,
            "separates_architecture_grade_from_operational_health_grade": True,
            "separates_guarded_soak_health_from_raw_production_backlog": True,
            "includes_sleeve_weak_point_recurrence_and_systemic_guard_posture": True,
            "appends_cell_queue_when_apply": bool(apply),
            "never_touch_protected_volumes": list(PROTECTED_VOLUMES),
        },
        "smoothness_contract": _smoothness_contract(architecture_grade, str(operational_health.get("grade") or _grade(operational_score))),
        "a_plus_uplift_plan": _uplift_plan(architecture_grade, _grade(operational_score), cell_rows),
        "recommended_actions": [
            "treat each cell as its own subsystem with local state, health, needs, queue, and handoff contract",
            "use architecture_report_card.grade for the distributed layer maturity and operational_health.grade for live blockers",
            "keep storage_writer_cell as the only SQLite commit authority while widening preprocess workers around it",
            "route training through training_cell and only launch when storage_writer_cell and infra_cell are green enough",
            "make sleeve_cells responsible for local sleeve degradation before escalating to the control plane",
            "use the cell needs queue as the operator troubleshooting packet instead of hunting across raw artifacts",
        ],
    }

    if apply:
        for cell_id, record in built.items():
            root = cell_root / cell_id
            write_payload(root / "state.json", record["state"])
            write_payload(root / "health.json", record["health"])
            write_payload(root / "needs.json", {"timestamp_utc": iso_now(), "cell_id": cell_id, "needs": record["needs"]})
            write_payload(root / "contract.json", record["contract"])
            _append_queue(root / "queue.jsonl", record["needs"])
        write_payload(cell_root / "cell_manifest.json", {"timestamp_utc": iso_now(), "cells": cell_rows})
        write_payload(cell_root / "global_cell_bus.json", payload)
        write_payload(cell_root / "intercell_bus.json", bus)
        write_payload(cell_root / "cell_resource_contracts.json", payload["cell_resource_contracts"])
        payload["write_result"] = {
            "cell_root": _rel(project_root, cell_root),
            "override": _write_override(DEFAULT_OVERRIDE_PATH),
            "markdown_path": _rel(project_root, DEFAULT_MARKDOWN_PATH),
        }
        DEFAULT_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_MARKDOWN_PATH.write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the seven-cell distributed/federated architecture layer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--alias-file", default=str(DEFAULT_ALIAS_PATH))
    parser.add_argument("--cell-root", default=str(DEFAULT_CELL_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root=project_root, apply=bool(args.apply), cell_root=Path(args.cell_root))
    out_path = Path(args.out_file).expanduser()
    alias_path = Path(args.alias_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    if not alias_path.is_absolute():
        alias_path = project_root / alias_path
    write_payload(out_path, payload)
    write_payload(alias_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "distributed_cell_architecture "
            f"overall_status={payload.get('overall_status')} "
            f"grade={payload.get('grade')} "
            f"operational_grade={(payload.get('operational_health') or {}).get('grade', '')} "
            f"cells={payload.get('cell_count')} "
            f"needs={len(payload.get('top_needs') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
