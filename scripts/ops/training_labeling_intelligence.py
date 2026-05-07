#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_VERSION = 1614
TARGET_PLATFORM_TOTAL_BOTS = 1628
PACK_VERSION = "training_labeling_intelligence_v1"
PACK_SLUG = "training_labeling_intelligence"
PACK_DISPLAY_NAME = "Training And Labeling Intelligence Pack"
SLEEVE_FAMILY = "training_labeling_intelligence"
UNIVERSAL_LABEL_CONTRACT_VERSION = "universal_training_label_contract_v1"
MINIMUM_TRAINING_OBSERVATIONS = 70000
MINIMUM_COLLECTION_DAYS = 180
SAMPLE_RATE = 0.01
MAX_DAILY_MB_PER_BOT = 1


INTELLIGENCE_SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "label_contract_normalizer",
        "layer": "labeling",
        "display_name": "Label Contract Normalizer",
        "objective": "Give every bot a point-in-time label contract without erasing specialized existing contracts.",
        "outputs": ["label_contract_diff", "missing_label_repair_packet", "label_family_map"],
    },
    {
        "slug": "point_in_time_label_guard",
        "layer": "labeling",
        "display_name": "Point-In-Time Label Guard",
        "objective": "Block future leakage, lookahead joins, unbounded raw-feed joins, and unlabeled promotion packets.",
        "outputs": ["join_contract_verdict", "leakage_risk_score", "label_context_gap_list"],
    },
    {
        "slug": "lane_balance_scheduler",
        "layer": "training",
        "display_name": "Lane Balance Scheduler",
        "objective": "Turn lane dominance, symbol concentration, and coverage shortfall into narrow retrain plans.",
        "outputs": ["lane_balanced_retrain_plan", "lookback_guidance", "dominance_cap_vote"],
    },
    {
        "slug": "coverage_repair_orchestrator",
        "layer": "training",
        "display_name": "Coverage Repair Orchestrator",
        "objective": "Prefer coverage repair candidates when the normal targeted retrain shortlist is empty.",
        "outputs": ["coverage_repair_queue", "runtime_input_repair_plan", "walk_forward_cycle_budget"],
    },
    {
        "slug": "schema_lineage_gatekeeper",
        "layer": "lineage",
        "display_name": "Schema Lineage Gatekeeper",
        "objective": "Keep schema, feature-store, replay, experiment, and promotion lineage gates synchronized before retrain.",
        "outputs": ["schema_lineage_gate_status", "missing_contract_repair_order", "promotion_packet_readiness"],
    },
    {
        "slug": "retrain_outcome_memory",
        "layer": "learning",
        "display_name": "Retrain Outcome Memory",
        "objective": "Record which targeted retrains reduced coverage gaps, label errors, gate blockers, and runtime failures.",
        "outputs": ["retrain_effect_delta", "retry_or_rotate_vote", "training_playbook_reward"],
    },
]


ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "telemetry_collector", "label": "Telemetry Collector", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "quality_scorer", "label": "Quality Scorer", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "policy_guard", "label": "Policy Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "training_bridge", "label": "Training Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]


BASE_DATA_INTAKES = [
    "training_quality_trace",
    "training_runtime_trace",
    "training_label_audit_trace",
    "runtime_training_snapshot_trace",
    "coverage_gap_closer_trace",
    "feature_store_lineage_trace",
    "promotion_gate_trace",
    "whole_system_governor_trace",
    "codex_handoff_trace",
]


REQUIRED_LABELS = [
    "forward_return_bucket",
    "risk_adjusted_return_bucket",
    "action_effect_bucket",
    "label_quality_bucket",
    "lane_balance_bucket",
    "coverage_gap_status",
    "lineage_gate_status",
    "promotion_gate_status",
]


STORAGE_TARGETS = [
    "governance/training_labeling_intelligence",
    *[f"governance/training_labeling_intelligence/{system['slug']}" for system in INTELLIGENCE_SYSTEMS],
    "governance/health/training_labeling_intelligence_latest.json",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_first_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        payload = _load_json(path)
        if payload:
            return payload
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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


def _ordered_unique(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else registry.get("bots")
    return [row for row in rows or [] if isinstance(row, dict)]


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", bot_id)
    return int(match.group("version")) if match else None


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for system in INTELLIGENCE_SYSTEMS:
        for role in ROLE_TEMPLATES:
            role_slug = f"{system['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"training_labeling_{role_slug}_bot",
                    "label": f"{system['display_name']} {role['label']}",
                    "system": system["slug"],
                    "layer": system["layer"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {system['objective']}",
                    "target_functions": list(system["outputs"]),
                }
            )
    return specs


BOTS = _bot_specs()


def _slot_kind(bot: dict[str, Any]) -> str:
    return f"{PACK_SLUG}_{bot['role_slug']}"


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    existing_by_slot = {
        str(row.get("slot_kind") or ""): str(row.get("bot_id") or "")
        for row in rows
        if str(row.get("slot_kind") or "") and str(row.get("bot_id") or "")
    }
    used_versions = {
        version
        for row in rows
        for version in [_version_from_bot_id(str(row.get("bot_id") or ""))]
        if version is not None
    }
    assigned: dict[str, str] = {}
    for index, bot in enumerate(BOTS):
        slot = _slot_kind(bot)
        if slot in existing_by_slot:
            assigned[slot] = existing_by_slot[slot]
            continue
        desired = BASE_VERSION + index
        if desired not in used_versions:
            version = desired
            used_versions.add(version)
        else:
            version = _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _system(bot: dict[str, Any]) -> dict[str, Any]:
    for system in INTELLIGENCE_SYSTEMS:
        if system["slug"] == bot["system"]:
            return system
    return {"slug": bot["system"], "layer": bot.get("layer", ""), "display_name": bot["system"], "outputs": []}


def _existing_contract(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("label_contract") or row.get("training_label_contract")
    return contract if isinstance(contract, dict) else {}


def _contract_complete(row: dict[str, Any]) -> bool:
    contract = _existing_contract(row)
    primary = str(contract.get("primary_horizon") or contract.get("primary_label_horizon") or "").strip()
    required_context = contract.get("required_context") or contract.get("required_label_context")
    required_labels = contract.get("required_labels")
    return bool(primary and (isinstance(required_context, list) or isinstance(required_labels, list) or row.get("data_label_contract_version")))


def _infer_label_family(row: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "bot_id",
            "slot_kind",
            "slot_label",
            "sleeve_profile",
            "sleeve_family",
            "strategy_family",
            "bot_role",
            "intelligence_system",
            "governance_layer",
        )
    )
    rules: list[tuple[tuple[str, ...], str, str, list[str], list[str]]] = [
        (("label", "training", "retrain", "coverage"), "training_process_quality", "retrain_cycle_improves_gate_status", ["coverage_gap_delta", "label_quality_delta", "runtime_failure_delta"], ["training_quality_trace", "coverage_gap_trace", "runtime_snapshot_trace"]),
        (("governor", "backlog", "storage", "memory", "auth", "operator", "lineage", "guard"), "operational_guard_effect", "guard_prevents_bad_runtime_action", ["false_positive_guard", "incident_prevention", "pressure_delta"], ["runtime_health", "incident_log", "operator_context"]),
        (("option", "gamma", "iv", "0dte"), "options_surface", "iv_realized_1d_5d", ["gamma", "skew", "spread_quality", "event_vol_reset"], ["options_chain", "iv_surface", "open_interest", "bid_ask_spread"]),
        (("future", "basis", "curve"), "futures_event_session", "session_event_followthrough", ["basis", "curve", "macro_event_window"], ["futures_bars", "session_calendar", "basis_context", "macro_calendar"]),
        (("crypto",), "crypto_microstructure", "crypto_session_followthrough", ["liquidity_sweep", "basis", "funding_stress"], ["crypto_bars", "order_book_proxy", "funding_context"]),
        (("dividend", "income", "drip", "payout"), "income_total_return", "20d_total_return_income", ["payout_safety", "dividend_cut_risk", "ex_dividend_window"], ["ex_dividend_calendar", "payout_metrics", "rate_context"]),
        (("conservative", "capital_preservation", "cash_parking"), "risk_adjusted_preservation", "drawdown_avoidance_5d", ["vol_adjusted_return", "max_drawdown", "cash_parking"], ["volatility_budget", "credit_stress", "liquidity_state"]),
        (("intraday", "scalp", "vwap", "opening_range", "same_session"), "intraday_fast", "5m_30m_forward_return", ["1m", "5m", "15m", "60m"], ["one_minute_bars", "vwap", "spread_quality", "relative_volume"]),
        (("swing", "position", "multi_day"), "multi_day", "2d_5d_forward_return", ["1d", "5d", "10d"], ["daily_bars", "sector_context", "macro_context", "overnight_gap"]),
        (("quant", "alpha", "factor", "model", "research"), "alpha_research", "walk_forward_alpha_after_cost", ["regime_edge", "slippage_adjusted_edge", "overfit_gap"], ["feature_store_lineage", "walk_forward_trace", "execution_cost_context"]),
    ]
    for tokens, family, primary, aux, context in rules:
        if any(token in text for token in tokens):
            return family, primary, aux, context
    return "generic_directional", "1d_forward_return", ["5d_forward_return", "risk_adjusted_return"], ["price_bars", "volume", "market_context"]


def _training_lane_for_family(label_family: str) -> str:
    if label_family in {"intraday_fast", "options_surface", "futures_event_session", "crypto_microstructure"}:
        return "lane_specific_fast"
    if label_family in {"training_process_quality", "operational_guard_effect"}:
        return "governance_effect"
    if label_family in {"alpha_research"}:
        return "research_walk_forward"
    if label_family in {"income_total_return", "risk_adjusted_preservation", "multi_day"}:
        return "slow_lane_balanced"
    return "general_balanced"


def _universal_contract(row: dict[str, Any]) -> dict[str, Any]:
    existing = _existing_contract(row)
    family, primary, aux, context = _infer_label_family(row)
    existing_family = str(existing.get("label_family") or existing.get("family") or "").strip()
    existing_primary = str(existing.get("primary_horizon") or existing.get("primary_label_horizon") or "").strip()
    existing_aux = existing.get("aux_horizons") or existing.get("aux_label_horizons")
    existing_context = existing.get("required_context") or existing.get("required_label_context")
    required_labels = existing.get("required_labels") if isinstance(existing.get("required_labels"), list) else REQUIRED_LABELS
    label_family = existing_family or family
    return {
        "version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "label_family": label_family,
        "primary_horizon": existing_primary or primary,
        "aux_horizons": list(existing_aux) if isinstance(existing_aux, list) and existing_aux else aux,
        "required_context": list(existing_context) if isinstance(existing_context, list) and existing_context else context,
        "required_labels": list(required_labels),
        "required_join_mode": str(existing.get("required_join_mode") or "point_in_time_only"),
        "forbidden_join_modes": list(existing.get("forbidden_join_modes") or ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"]),
        "quality_floor": _safe_float(existing.get("quality_floor"), 0.84),
        "training_lane": _training_lane_for_family(label_family),
        "source": "preserved_existing_contract" if existing else "inferred_from_registry_identity",
    }


def _apply_universal_label_contracts(rows: list[dict[str, Any]], now: str) -> dict[str, Any]:
    missing_before = 0
    normalized_missing = 0
    normalized_incomplete = 0
    preserved_explicit = 0
    family_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    updated_bot_ids: list[str] = []

    for row in rows:
        had_any = bool(_existing_contract(row) or row.get("data_label_contract_version"))
        complete = _contract_complete(row)
        if not had_any:
            missing_before += 1
        contract = _universal_contract(row)
        family_counts[str(contract["label_family"])] += 1
        lane_counts[str(contract["training_lane"])] += 1
        status = "preserved_explicit" if complete else "normalized_incomplete" if had_any else "normalized_missing"
        if status == "preserved_explicit":
            preserved_explicit += 1
        elif status == "normalized_incomplete":
            normalized_incomplete += 1
        else:
            normalized_missing += 1
        row["universal_label_contract"] = contract
        row["universal_label_contract_version"] = UNIVERSAL_LABEL_CONTRACT_VERSION
        row["training_labeling_intelligence_version"] = PACK_VERSION
        row["training_label_contract_status"] = status
        row["training_lane"] = contract["training_lane"]
        row["label_contract_last_reviewed_utc"] = now
        if not complete:
            row["label_contract"] = contract
            row["data_label_contract_version"] = UNIVERSAL_LABEL_CONTRACT_VERSION
            updated_bot_ids.append(str(row.get("bot_id") or ""))
        existing_tags = row.get("labeling_tags") if isinstance(row.get("labeling_tags"), list) else []
        row["labeling_tags"] = _ordered_unique(
            [
                *existing_tags,
                "universal_label_contract",
                "point_in_time_only",
                f"label_family:{contract['label_family']}",
                f"training_lane:{contract['training_lane']}",
                f"label_contract_version:{UNIVERSAL_LABEL_CONTRACT_VERSION}",
            ]
        )
    return {
        "total_rows": len(rows),
        "missing_contracts_before": missing_before,
        "normalized_missing_contracts": normalized_missing,
        "normalized_incomplete_contracts": normalized_incomplete,
        "preserved_explicit_contracts": preserved_explicit,
        "updated_label_contract_bot_count": len(updated_bot_ids),
        "updated_label_contract_bot_ids": updated_bot_ids[:250],
        "label_family_counts": dict(sorted(family_counts.items())),
        "training_lane_counts": dict(sorted(lane_counts.items())),
        "coverage_ratio_after": 1.0 if rows else 0.0,
    }


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "sleeve_family": SLEEVE_FAMILY,
        "display_name": PACK_DISPLAY_NAME,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "6_training_labeling_systems_4_bots_each_24_bot_intelligence_layer",
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "training_labeling_hot_3d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": MAX_DAILY_MB_PER_BOT,
            "capture_mode": "thin_digest_and_event_delta_only",
            "sample_rate": SAMPLE_RATE,
            "dedupe_required": True,
            "self_accommodation": "heartbeat_when_whole_system_governor_is_protective",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "anchor_bot_ids": {
            bot["system"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("telemetry_collector")
        },
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "authority_boundary": "advisory_labeling_and_training_process_intelligence_no_execution_no_allocation",
    }


def _threshold_progress() -> dict[str, Any]:
    return {
        "observations": 0,
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "observations_ready": False,
        "collection_age_days": 0.0,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "days_ready": False,
        "training_ready": False,
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    system_slug = str(system["slug"])
    layer = str(system["layer"])
    label_contract = {
        "version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "label_family": "training_process_quality",
        "primary_horizon": f"{system_slug}_improves_training_or_labeling_gate_status",
        "aux_horizons": ["coverage_gap_delta", "label_quality_delta", "runtime_failure_delta"],
        "required_context": [*BASE_DATA_INTAKES, f"{system_slug}_effect_trace"],
        "required_labels": list(REQUIRED_LABELS),
        "required_join_mode": "point_in_time_only",
        "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
        "quality_floor": 0.89,
        "training_lane": "governance_effect",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "training_labeling_intelligence_expansion_slot",
        "weight": 0.0,
        "preference_score": 0.0,
        "quality_score": 0.0,
        "test_accuracy": None,
        "candidate_test_accuracy": None,
        "candidate_quality_score": 0.0,
        "previous_best_accuracy": None,
        "no_improvement_streak": 0,
        "deleted_from_rotation": False,
        "delete_reason": "",
        "promoted": False,
        "promotion_reason": "training_labeling_intelligence_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": ["protective_pressure", "coverage_repair", "label_audit", "schema_gate_repair", "off_hours_targeted_retrain"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v1522_quant_operational_backlog_outcome_verifier_telemetry_collector_bot",
            "brain_refinery_v1554_quant_operational_operator_decision_packet_builder_telemetry_collector_bot",
            "brain_refinery_v1562_autonomic_governance_sleeve_budget_market_telemetry_collector_bot",
        ],
        "data_intake_collections": [*BASE_DATA_INTAKES, f"{system_slug}_effect_trace", f"{system_slug}_label_quality_trace"],
        "storage_targets": ["governance/training_labeling_intelligence", f"governance/training_labeling_intelligence/{system_slug}", "governance/health/training_labeling_intelligence_latest.json"],
        "freshness_slo_seconds": 1800,
        "retention_profile": "training_labeling_hot_3d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "training_labeling_intelligence_collect_only_until_label_and_training_effect_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "training_labeling_intelligence_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_training_labeling_effect_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_label_contract_clearance": True,
            "requires_runtime_pressure_clearance": True,
            "requires_backpressure_clearance": True,
            "requires_schema_lineage_clearance": True,
            "requires_paper_live_separation_clearance": True,
            "requires_global_halt_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_digest_with_heartbeat_fallback",
        "data_collection_sample_rate": SAMPLE_RATE,
        "data_collection_max_daily_storage_mb": MAX_DAILY_MB_PER_BOT,
        "data_collection_max_daily_mb": float(MAX_DAILY_MB_PER_BOT),
        "data_collection_compute_guard_mode": "pressure_self_accommodating",
        "self_accommodating_policy": {
            "steady": "thin_digest",
            "protective": "heartbeat",
            "critical": "parked_until_operator_review",
            "raw_trace_allowed": False,
        },
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": system_slug,
        "sleeve_family": SLEEVE_FAMILY,
        "training_labeling_layer": layer,
        "intelligence_system": system_slug,
        "strategy_family": "training_and_labeling_governance",
        "correlation_peer_sleeves": ["whole_system_governor", "autonomic_governance_mesh", "quant_operational_intelligence", "system_self_model", "codex_handoff"],
        "correlation_dependencies": ["training_quality_control", "training_runtime_control", "coverage_gap_closer", "schema_migration_guard", "feature_store_manifest", "promotion_quality_gate"],
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": ["master_bot_registry", "governance_health", "training_quality_control", "coverage_gap_closer", "feature_store_manifest", "codex_handoff"],
        "label_contract": label_contract,
        "universal_label_contract": label_contract,
        "data_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "training_labeling_intelligence_version": PACK_VERSION,
        "training_lane": "governance_effect",
        "training_label_contract_status": "pack_native",
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "universal_label_contract",
            "point_in_time_only",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"training_labeling_layer:{layer}",
            "label_family:training_process_quality",
            "training_lane:governance_effect",
        ],
        "execution_policy_label": "collection_only_training_labeling_intelligence_no_execution",
        "eligible_for_master_vote": False,
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_traits": ["paper_first_safety", "global_halt_awareness", "resource_throttle_awareness", "decision_explanation_contract", "registry_auditable_identity", "point_in_time_labeling"],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_trade_lock_required": True,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "capability_pack_contract": contract,
        "training_labeling_intelligence_contract": {
            "contract_version": "training_labeling_intelligence_layers_v1",
            "capability_pack": PACK_SLUG,
            "training_labeling_layer": layer,
            "intelligence_system": system_slug,
            "system_display_name": system["display_name"],
            "system_outputs": list(system["outputs"]),
            "authority_boundary": "collection_only_advisory_no_execution_no_allocation_no_halt_clearance",
        },
    }


def _ensure_storage_targets(project_root: Path) -> list[str]:
    ready: list[str] = []
    for target in STORAGE_TARGETS:
        path = project_root / target
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            ready.append(str(path.parent.relative_to(project_root)))
        else:
            path.mkdir(parents=True, exist_ok=True)
            ready.append(str(path.relative_to(project_root)))
    return sorted(set(ready))


def _pack_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _pack_contract(assigned_ids)
    return {
        "slug": PACK_SLUG,
        "display_name": PACK_DISPLAY_NAME,
        "sleeve_family": SLEEVE_FAMILY,
        "objective": "Add a collect-only intelligence layer for label contracts, point-in-time labeling, lane-balanced retrain planning, coverage repair, schema lineage, and retrain outcome memory.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "intelligence_systems": list(INTELLIGENCE_SYSTEMS),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "anchor_bot_ids": contract["anchor_bot_ids"],
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
    }


def _training_process_intelligence(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_forward_root = project_root / "governance" / "walk_forward"
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    runtime_control = _load_json(health_root / "training_runtime_control_latest.json")
    coverage_seed = _load_first_json(
        [
            walk_forward_root / "coverage_seed_latest.json",
            health_root / "walk_forward_coverage_seed_latest.json",
        ]
    )
    coverage_gap = _load_first_json(
        [
            walk_forward_root / "coverage_gap_closer_latest.json",
            health_root / "coverage_gap_closer_latest.json",
        ]
    )
    schema_compat = _load_json(health_root / "retrain_schema_compatibility_latest.json")
    schema_migration = _load_json(health_root / "schema_migration_guard_latest.json")
    feature_store = _load_json(project_root / "governance" / "feature_store" / "latest.json")
    lineage = _load_json(health_root / "training_lineage_manifest_latest.json")
    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    normal_targets = _ordered_unique(targeted_actions.get("targeted_retrain_bot_ids") if isinstance(targeted_actions.get("targeted_retrain_bot_ids"), list) else [])
    stage_candidates = coverage_gap.get("active_stage_candidates") if isinstance(coverage_gap.get("active_stage_candidates"), list) else []
    seed_rows = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []
    coverage_targets = _ordered_unique(
        [str(row.get("bot_id") or "") for row in stage_candidates if isinstance(row, dict)]
        or [
            str(row.get("bot_id") or "")
            for row in seed_rows
            if isinstance(row, dict)
            and (
                bool(row.get("needs_runtime_input_repair", False))
                or bool(row.get("needs_diagnostic_refresh", False))
                or "targeted_retrain" in [str(action or "") for action in row.get("actions") or []]
            )
        ]
    )
    precompute_targets = runtime_control.get("precompute_targets") if isinstance(runtime_control.get("precompute_targets"), list) else []
    precompute_ids = _ordered_unique([str(row.get("bot_id") or "") for row in precompute_targets if isinstance(row, dict)])
    launch_contract = (coverage_gap.get("autopilot_contract") or {}).get("launch_contract") if isinstance(coverage_gap.get("autopilot_contract"), dict) else {}
    autopilot_blocking_reasons = []
    if isinstance(coverage_gap.get("autopilot_contract"), dict):
        raw_blocking_reasons = (coverage_gap.get("autopilot_contract") or {}).get("blocking_reasons")
        autopilot_blocking_reasons = raw_blocking_reasons if isinstance(raw_blocking_reasons, list) else []
    blocked_reasons = _ordered_unique(
        [
            *_ordered_unique(autopilot_blocking_reasons),
            "training_quality_blocked" if str(training_quality.get("overall_status") or "") == "blocked" else "",
            "schema_migration_guard_blocked" if schema_migration and not bool(schema_migration.get("ok", False)) else "",
            "lineage_manifest_not_ready" if lineage and str(lineage.get("overall_status") or "") in {"blocked", "needs_attention"} else "",
        ]
    )
    selected_targets = coverage_targets or precompute_ids or normal_targets
    return {
        "process_version": "training_process_intelligence_v1",
        "normal_targeted_retrain_bot_ids": normal_targets,
        "coverage_repair_bot_ids": coverage_targets,
        "precompute_target_bot_ids": precompute_ids[:12],
        "selected_targeted_retrain_bot_ids": selected_targets[:12],
        "selected_target_source": "coverage_repair" if coverage_targets else "runtime_precompute" if precompute_ids else "normal_targeted_shortlist",
        "recommended_retrain_profile": "coverage_canary" if coverage_targets else "lane_specific",
        "blocked_reasons": blocked_reasons,
        "launch_contract": launch_contract if isinstance(launch_contract, dict) else {},
        "quality_snapshot": {
            "overall_status": training_quality.get("overall_status"),
            "training_quality_score": training_quality.get("training_quality_score") or training_quality.get("training_quality_index"),
            "top_priorities": training_quality.get("top_priorities") if isinstance(training_quality.get("top_priorities"), list) else [],
        },
        "runtime_snapshot": {
            "overall_status": runtime_control.get("overall_status"),
            "snapshot_ready": runtime_control.get("snapshot_ready"),
            "resource_guard": runtime_control.get("resource_guard") if isinstance(runtime_control.get("resource_guard"), dict) else {},
        },
        "schema_and_lineage": {
            "schema_compatibility_status": schema_compat.get("overall_status") or schema_compat.get("status"),
            "schema_migration_status": schema_migration.get("overall_status") or schema_migration.get("status"),
            "feature_store_ok": feature_store.get("ok"),
            "lineage_status": lineage.get("overall_status"),
        },
        "safe_preflight_order": [
            "./scripts/ops/opsctl.sh schema-migration --json",
            "./scripts/ops/opsctl.sh feature-store --json",
            "./scripts/ops/opsctl.sh training-label-audit --json",
            "./scripts/ops/opsctl.sh runtime-training-snapshot --json",
            "./scripts/ops/opsctl.sh coverage-gap-closer --apply-stage --json",
            "./scripts/ops/opsctl.sh training-runtime-control --json",
        ],
        "safe_targeted_retrain_template": [
            "./scripts/ops/opsctl.sh",
            "retrain-force-targeted",
            "--include-bot-ids",
            ",".join(selected_targets[:4]),
            "--retrain-profile",
            "coverage_canary",
            "--skip-master-update",
            "--runtime-train-use-snapshot",
            "--thread-cap",
            "1",
            "--memory-guard",
        ],
    }


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    existing_slot_kinds = {str(row.get("slot_kind") or "") for row in rows}
    assigned_ids = _assign_bot_ids(rows)
    now = _utc_now()
    planned_rows: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    for bot in BOTS:
        slot = _slot_kind(bot)
        if slot in existing_slot_kinds:
            skipped_existing.append(slot)
            continue
        planned_rows.append(_row_for_bot(bot, assigned_ids[slot], assigned_ids, now))
    return {
        "generated_at_utc": now,
        "training_labeling_intelligence_version": PACK_VERSION,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_total_after_apply": len(rows) + len(planned_rows),
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_reaches_target_total": len(rows) + len(planned_rows) >= TARGET_PLATFORM_TOTAL_BOTS,
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "pack": _pack_summary(assigned_ids),
    }


def _refresh_summary(registry: dict[str, Any], label_summary: dict[str, Any]) -> None:
    rows = _registry_rows(registry)
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("training_labeling_intelligence_version") or "") == PACK_VERSION and str(row.get("capability_pack_slug") or "") == PACK_SLUG]
    universal_rows = [row for row in rows if str(row.get("universal_label_contract_version") or "") == UNIVERSAL_LABEL_CONTRACT_VERSION]
    contract_rows = [row for row in rows if isinstance(row.get("label_contract"), dict) or str(row.get("data_label_contract_version") or "")]
    versions = [
        int(match.group(1))
        for row in rows
        for match in [re.match(r"^brain_refinery_v(\d+)", str(row.get("bot_id") or ""))]
        if match
    ]
    summary = dict(registry.get("summary") or {})
    summary.update(
        {
            "total_bots": len(rows),
            "active_bots": len(active),
            "inactive_bots": len(inactive),
            "active_signal_sub_bots": len(signal_active),
            "active_infrastructure_sub_bots": len(infra_active),
            "data_collection_active_bots": sum(1 for row in rows if bool(row.get("data_collection_active"))),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))),
            "structured_capability_pack_bot_count": len(structured),
            "training_labeling_intelligence_bot_count": len(pack_rows),
            "latest_training_labeling_intelligence": PACK_VERSION,
            "training_label_contract_bot_count": len(contract_rows),
            "universal_label_contract_bot_count": len(universal_rows),
            "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
            "training_label_contract_coverage_ratio": round(len(contract_rows) / max(len(rows), 1), 4),
            "training_label_contracts_normalized_latest": label_summary.get("updated_label_contract_bot_count", 0),
            "max_bot_version": max(versions) if versions else None,
            "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
        }
    )
    registry["summary"] = summary


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = _registry_rows(registry)
    plan = plan_registry_expansion(registry)
    missing_contracts = sum(1 for row in rows if not (_existing_contract(row) or row.get("data_label_contract_version")))
    incomplete_contracts = sum(1 for row in rows if (_existing_contract(row) or row.get("data_label_contract_version")) and not _contract_complete(row))
    return {
        "ok": True,
        "generated_at_utc": plan["generated_at_utc"],
        "mode": "dry_run",
        "registry_path": str((project_root / "master_bot_registry.json").resolve()),
        "current_total_bots": len(rows),
        "current_active_bots": sum(1 for row in rows if bool(row.get("active"))),
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_total_after_apply": plan["planned_total_after_apply"],
        "planned_reaches_target_total": plan["planned_reaches_target_total"],
        "training_labeling_intelligence_version": PACK_VERSION,
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "missing_label_contract_count": missing_contracts,
        "incomplete_label_contract_count": incomplete_contracts,
        "pack": plan["pack"],
        "training_process_intelligence": _training_process_intelligence(project_root),
        "recommended_apply_command": "./scripts/ops/opsctl.sh training-labeling-intelligence --apply --json",
    }


def apply_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = _registry_rows(registry)
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    storage_targets_ready = _ensure_storage_targets(project_root)
    backup_dir = project_root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"master_bot_registry_before_training_labeling_intelligence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(registry_path, backup)
    now = _utc_now()
    if added_rows:
        rows.extend(added_rows)
    label_summary = _apply_universal_label_contracts(rows, now)
    registry["sub_bots"] = rows
    registry["updated_at_utc"] = now
    _refresh_summary(registry, label_summary)
    _write_json(registry_path, registry)

    process = _training_process_intelligence(project_root)
    payload = build_payload(project_root)
    payload.update(
        {
            "mode": "applied",
            "added_bot_count": len(added_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in added_rows],
            "backup_path": str(backup),
            "new_total_bots": len(rows),
            "new_active_bots": sum(1 for row in rows if bool(row.get("active"))),
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
            "storage_targets_ready": storage_targets_ready,
            "label_contract_summary": label_summary,
            "training_process_intelligence": process,
        }
    )
    config_payload = {
        "generated_at_utc": _utc_now(),
        "training_labeling_intelligence_version": PACK_VERSION,
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "pack": payload["pack"],
    }
    _write_json(project_root / "config" / "training_labeling_intelligence_v1.json", config_payload)
    _write_json(project_root / "governance" / "health" / "training_labeling_intelligence_latest.json", payload)
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "label_coverage_latest.json", label_summary)
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "training_process_intelligence_latest.json", process)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Improve training process intelligence and normalize universal label contracts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = apply_registry(project_root) if args.apply else build_payload(project_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "training_labeling_intelligence "
            f"mode={payload['mode']} systems={payload['system_count']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)} "
            f"missing_labels={payload.get('missing_label_contract_count', 0)} "
            f"selected_targets={len(payload['training_process_intelligence']['selected_targeted_retrain_bot_ids'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
