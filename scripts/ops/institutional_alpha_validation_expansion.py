#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_VERSION = 1086
TARGET_PLATFORM_TOTAL_BOTS = 1196
PACK_VERSION = "institutional_alpha_validation_v1"
PACK_SLUG = "institutional_alpha_validation"
PACK_DISPLAY_NAME = "Institutional Alpha Validation Pack"
SLEEVE_FAMILY = "institutional_validation_control_plane"
SLEEVE_PROFILE = "institutional_alpha_validation"
LABEL_CONTRACT_VERSION = "institutional_alpha_validation_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 60000
MINIMUM_COLLECTION_DAYS = 210
PAPER_RUNTIME_CAPACITY_FLOOR = 1000

SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "alpha_evidence_court",
        "display_name": "Alpha Evidence Court",
        "objective": "Prove which alpha theses deserve resources through walk-forward, multiple-testing, leakage, and retirement evidence.",
        "outputs": ["alpha_case_file", "walk_forward_verdict", "retirement_vote"],
    },
    {
        "slug": "execution_quality_lab_v2",
        "display_name": "Execution Quality Lab v2",
        "objective": "Price paper fills with queue position, partial fills, slippage surfaces, spread, and latency realism.",
        "outputs": ["fill_quality_surface", "slippage_forecast", "paper_realism_verdict"],
    },
    {
        "slug": "portfolio_intelligence_layer",
        "display_name": "Portfolio Intelligence Layer",
        "objective": "Coordinate CVaR, drawdown recovery, capital budget auctions, overlap, and correlation breakdown across sleeves.",
        "outputs": ["portfolio_budget_vote", "cvar_drawdown_packet", "overlap_referee_score"],
    },
    {
        "slug": "regime_transition_engine",
        "display_name": "Regime Transition Engine",
        "objective": "Detect regime shifts across volatility, liquidity, macro, shock persistence, and sleeve routing.",
        "outputs": ["regime_shift_warning", "vol_liquidity_state", "shock_persistence_score"],
    },
    {
        "slug": "options_risk_intelligence_v2",
        "display_name": "Options Risk Intelligence v2",
        "objective": "Net Greeks, crowding, assignment risk, IV-surface anomalies, and margin-aware structure quality.",
        "outputs": ["greeks_netting_packet", "iv_surface_alert", "assignment_margin_vote"],
    },
    {
        "slug": "futures_cross_asset_basis_lab",
        "display_name": "Futures And Cross-Asset Basis Lab",
        "objective": "Map futures basis, curve roll pressure, cash/futures divergence, hedge quality, and macro futures stress.",
        "outputs": ["basis_dislocation_packet", "curve_roll_pressure", "hedge_quality_score"],
    },
    {
        "slug": "data_quality_observatory",
        "display_name": "Data Quality Observatory",
        "objective": "Track feature drift, provider confidence, missing label gaps, point-in-time joins, and data value ranking.",
        "outputs": ["feature_drift_heatmap", "provider_confidence_score", "label_gap_rank"],
    },
    {
        "slug": "replay_crisis_simulation_factory",
        "display_name": "Replay And Crisis Simulation Factory",
        "objective": "Replay COVID, 2008, flash crash, Fed stress scenarios, and synthetic crises against strategy behavior.",
        "outputs": ["crisis_replay_score", "scenario_loss_surface", "synthetic_crisis_packet"],
    },
    {
        "slug": "model_governance_board",
        "display_name": "Model Governance Board",
        "objective": "Write model cards, audit promotion evidence, track champion/challenger decay, and gate retraining.",
        "outputs": ["model_card", "promotion_evidence_audit", "champion_challenger_state"],
    },
    {
        "slug": "operator_copilot_v2",
        "display_name": "Operator Copilot v2",
        "objective": "Explain halts, suggest next commands, produce morning briefs, reserve creative-app headroom, and narrate reports.",
        "outputs": ["operator_next_command", "halt_explanation", "morning_brief_packet"],
    },
    {
        "slug": "backpressure_storage_brain_v2",
        "display_name": "Backpressure And Storage Brain v2",
        "objective": "Forecast SQL shard load, queue drain ETA, hot/warm/cold routing, stale artifact value, and external-drive reconnect safety.",
        "outputs": ["queue_drain_eta", "shard_load_forecast", "storage_route_vote"],
    },
    {
        "slug": "grandmaster_decision_quality_layer",
        "display_name": "Grandmaster Decision Quality Layer",
        "objective": "Audit cross-sleeve vote quality, master disagreement, provenance graphs, consensus errors, and confidence calibration.",
        "outputs": ["grandmaster_confidence_score", "consensus_error_flag", "vote_provenance_graph"],
    },
    {
        "slug": "liquidity_stress_market_impact_lab",
        "display_name": "Liquidity Stress And Market Impact Lab",
        "objective": "Stress-test liquidity holes, market impact, spread blowouts, queue fade, and capacity cliffs before scaling.",
        "outputs": ["impact_curve", "liquidity_cliff_alert", "capacity_cliff_score"],
    },
    {
        "slug": "alternative_data_entitlement_router",
        "display_name": "Alternative Data Entitlement Router",
        "objective": "Track alternative data availability, entitlement limits, source quality, cache fallback, and compliance-ready provenance.",
        "outputs": ["alt_data_route_packet", "entitlement_status", "source_provenance_card"],
    },
    {
        "slug": "cross_asset_risk_transfer_lab",
        "display_name": "Cross-Asset Risk Transfer Lab",
        "objective": "Map risk transfer between equities, rates, FX, commodities, crypto, credit, vol, and options sleeves.",
        "outputs": ["risk_transfer_graph", "hedge_leakage_score", "cross_asset_basis_vote"],
    },
    {
        "slug": "tax_corporate_actions_intelligence",
        "display_name": "Tax And Corporate Actions Intelligence",
        "objective": "Track dividends, splits, spin-offs, buybacks, wash-sale windows, holding periods, and tax-aware evidence.",
        "outputs": ["corporate_action_packet", "tax_window_vote", "dividend_adjustment_score"],
    },
    {
        "slug": "funding_collateral_margin_intelligence",
        "display_name": "Funding Collateral Margin Intelligence",
        "objective": "Analyze margin usage, collateral stress, borrow costs, financing rates, and forced deleveraging risk.",
        "outputs": ["margin_stress_packet", "collateral_pressure_score", "borrow_cost_alert"],
    },
    {
        "slug": "broker_venue_reliability_lab",
        "display_name": "Broker Venue Reliability Lab",
        "objective": "Score broker, API, venue, auth, order-routing, and data-plane reliability under load and provider denials.",
        "outputs": ["broker_reliability_score", "venue_route_health", "auth_data_plane_vote"],
    },
    {
        "slug": "feature_store_ontology_governance",
        "display_name": "Feature Store Ontology Governance",
        "objective": "Keep feature names, labels, joins, schemas, sleeve ontology, and point-in-time contracts consistent.",
        "outputs": ["feature_ontology_diff", "schema_contract_alert", "join_integrity_score"],
    },
    {
        "slug": "research_paper_assimilation_foundry",
        "display_name": "Research Paper Assimilation Foundry",
        "objective": "Turn research papers into testable hypotheses, proof obligations, implementation plans, and rejection reasons.",
        "outputs": ["research_thesis_card", "proof_obligation", "implementation_risk_vote"],
    },
    {
        "slug": "adversarial_market_abuse_defense",
        "display_name": "Adversarial Market Abuse Defense",
        "objective": "Detect spoof-like data, toxic flow, manipulation-prone signals, adversarial examples, and brittle model behavior.",
        "outputs": ["adversarial_signal_alert", "market_abuse_risk_score", "robustness_guard_vote"],
    },
    {
        "slug": "scenario_generation_synthetic_markets_v2",
        "display_name": "Scenario Generation And Synthetic Markets v2",
        "objective": "Generate synthetic market paths, crisis regimes, correlation convergence, slippage freezes, and replay labels.",
        "outputs": ["synthetic_market_path", "scenario_label_packet", "stress_generation_score"],
    },
    {
        "slug": "master_sleeve_curriculum_council",
        "display_name": "Master Sleeve Curriculum Council",
        "objective": "Schedule bot curricula by sleeve, sample debt, novelty, resource cost, and promotion readiness.",
        "outputs": ["sleeve_curriculum_plan", "training_sequence_vote", "sample_debt_priority"],
    },
    {
        "slug": "institutional_reporting_evidence_pack",
        "display_name": "Institutional Reporting Evidence Pack",
        "objective": "Produce report-ready proof surfaces, evidence packets, audit trails, caveats, and program-head presentation narratives.",
        "outputs": ["report_evidence_packet", "audit_trail_summary", "presentation_readiness_score"],
    },
]

ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "evidence_collector", "label": "Evidence Collector", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "signal_scorer", "label": "Signal Scorer", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "policy_guard", "label": "Policy Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "replay_auditor", "label": "Replay Auditor", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "master_bridge", "label": "Master Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]

DATA_INTAKES = [
    "institutional_alpha_case_trace",
    "institutional_execution_quality_trace",
    "institutional_portfolio_risk_trace",
    "institutional_regime_transition_trace",
    "institutional_options_risk_trace",
    "institutional_futures_basis_trace",
    "institutional_data_quality_trace",
    "institutional_crisis_replay_trace",
    "institutional_model_governance_trace",
    "institutional_operator_copilot_trace",
    "institutional_backpressure_storage_trace",
    "institutional_grandmaster_quality_trace",
    "institutional_liquidity_impact_trace",
    "institutional_alt_data_entitlement_trace",
    "institutional_cross_asset_risk_transfer_trace",
    "institutional_tax_corporate_actions_trace",
    "institutional_margin_collateral_trace",
    "institutional_broker_venue_reliability_trace",
    "institutional_feature_ontology_trace",
    "institutional_research_assimilation_trace",
    "institutional_adversarial_market_defense_trace",
    "institutional_synthetic_market_trace",
    "institutional_sleeve_curriculum_trace",
    "institutional_reporting_evidence_trace",
]

STORAGE_TARGETS = [
    "governance/institutional_alpha_validation",
    *[f"governance/institutional_alpha_validation/{system['slug']}" for system in SYSTEMS],
    "governance/health/institutional_alpha_validation_latest.json",
]

REQUIRED_LABELS = [
    "alpha_evidence_verdict_bucket",
    "walk_forward_quality_bucket",
    "execution_realism_bucket",
    "portfolio_cvar_pressure_bucket",
    "regime_transition_status",
    "options_greek_margin_bucket",
    "futures_basis_dislocation_bucket",
    "data_quality_join_status",
    "crisis_replay_loss_bucket",
    "model_governance_verdict",
    "operator_action_safety_status",
    "backpressure_storage_eta_bucket",
    "grandmaster_vote_quality_bucket",
    "liquidity_impact_bucket",
    "alt_data_entitlement_status",
    "cross_asset_risk_transfer_bucket",
    "corporate_action_adjustment_status",
    "margin_collateral_pressure_bucket",
    "broker_venue_reliability_bucket",
    "feature_ontology_integrity_bucket",
    "research_assimilation_status",
    "adversarial_market_abuse_bucket",
    "synthetic_scenario_quality_bucket",
    "sleeve_curriculum_priority_bucket",
    "report_evidence_readiness_bucket",
]


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for role in ROLE_TEMPLATES:
            role_slug = f"{system['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"institutional_{role_slug}_bot",
                    "label": f"{system['display_name']} {role['label']}",
                    "system": system["slug"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {system['objective']}",
                    "target_functions": list(system.get("outputs", [])),
                }
            )
    return specs


BOTS = _bot_specs()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", bot_id)
    return int(match.group("version")) if match else None


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


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
        version = desired if desired not in used_versions else _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        used_versions.add(version)
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _system(bot: dict[str, Any]) -> dict[str, Any]:
    for system in SYSTEMS:
        if system["slug"] == bot["system"]:
            return system
    return {"slug": bot["system"], "display_name": bot["system"], "objective": bot["objective"], "outputs": []}


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


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": SLEEVE_FAMILY,
            "sleeve_profile": SLEEVE_PROFILE,
            "display_name": PACK_DISPLAY_NAME,
        },
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "24_systems_5_bots_each_120_bot_institutional_validation_layer",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "institutional_validation_hot_7d_warm_180d_cold_900d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 6,
            "capture_mode": "thin_digest_first_institutional_trace",
            "sample_rate": 0.04,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_verdict_and_evidence_digests_stage_raw_institutional_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "anchor_bot_ids": {
            bot["system"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("evidence_collector")
        },
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "institutional_systems": [system["slug"] for system in SYSTEMS],
        "global_halt_contract": "institutional_validation_can_explain_and_reduce_false_halt_paths_but_never_force_clear",
        "paper_lock_contract": "no_execution_no_allocation_no_training_until_210_days_60000_observations_and_institutional_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    institutional_contract = {
        "contract_version": "institutional_alpha_validation_layers_v1",
        "capability_pack": PACK_SLUG,
        "system": bot["system"],
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "platform_brain_v6_dependency": "platform_brain_v6_foresight_cortex",
        "institutional_boundary": "validation_governance_and_evidence_collection_only_no_execution_authority",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "institutional_alpha_validation_expansion_slot",
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
        "promotion_reason": "institutional_alpha_validation_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": [
            "institutional_review",
            "paper_realism_review",
            "portfolio_risk_review",
            "regime_transition",
            "global_halt_recovery",
            "backpressure_spike",
            "program_head_reporting",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v934_intelligence_metacognitive_reasoning_budget_allocator_bot",
            "brain_refinery_v964_apex_self_model_state_vector_builder_bot",
            "brain_refinery_v1038_frontier_counterfactual_causal_lab_state_builder_bot",
            "brain_refinery_v1074_frontier_formal_safety_verification_state_builder_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 1200,
        "retention_profile": "institutional_validation_hot_7d_warm_180d_cold_900d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "institutional_alpha_validation_collect_only_until_evidence_execution_portfolio_regime_and_safety_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "institutional_validation_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_institutional_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_platform_brain_v6_clearance": True,
            "requires_alpha_evidence_court_clearance": True,
            "requires_execution_quality_clearance": True,
            "requires_portfolio_risk_clearance": True,
            "requires_model_governance_clearance": True,
            "requires_formal_safety_clearance": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_sampled",
        "data_collection_sample_rate": 0.04,
        "data_collection_max_daily_storage_mb": 6,
        "data_collection_max_daily_mb": 6.0,
        "data_collection_compute_guard_mode": "thin_digest",
        "data_collection_resource_guard_reason": "institutional_validation_uses_evidence_digests_for_pressure_safe_growth",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_control_refresh_seconds": 300,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "frontier_intelligence",
            "deep_recursive_awareness",
            "apex_self_awareness_intelligence",
            "alpha_intelligence_evolution",
            "coordination_intelligence",
            "model_lifecycle",
            "stress_lab",
        ],
        "correlation_dependencies": [
            "platform_brain_v6",
            "platform_stabilization_quality",
            "expansion_capacity_planner",
            "paper_trade_lock_guard",
            "model_governance_board",
            "execution_lab",
            "report_quality_guard",
        ],
        "provider_capability_profile": "internal_institutional_validation_control_plane_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "governance_health",
            "platform_brain_v6",
            "frontier_intelligence",
            "deep_recursive_awareness",
            "decision_provenance",
            "reports",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "institutional_validation_has_no_direct_broker_dependency_or_execution_authority",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.86,
            "freshness_slo_seconds": 1200,
            "regression_guard_bot_id": contract["anchor_bot_ids"].get("model_governance_board", ""),
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"sleeve_profile:{SLEEVE_PROFILE}",
            f"capability_pack:{PACK_SLUG}",
            f"institutional_system:{bot['system']}",
            "institutional_alpha_validation",
            "platform_brain_v6_aware",
            "evidence_first",
            "point_in_time_only",
            "training_after_threshold",
            "global_halt_aware",
            "operator_context_aware",
            "1000_plus_bot_platform",
        ],
        "execution_policy_label": "collection_only_institutional_validation_no_execution",
        "eligible_for_master_vote": False,
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "decision_explanation_contract",
            "registry_auditable_identity",
            "system_self_model_awareness",
            "frontier_counterfactual_reasoning",
            "institutional_evidence_governance",
            "formal_safety_boundary",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "thin_digest_institutional_validation",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "institutional_alpha_validation_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "institutional_alpha_validation_contract": institutional_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("institutional_alpha_validation_version") or "") == PACK_VERSION]
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
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded"))),
            "structured_capability_pack_bot_count": len(structured),
            "institutional_alpha_validation_bot_count": len(pack_rows),
            "latest_institutional_alpha_validation": PACK_VERSION,
            "max_bot_version": max(versions) if versions else None,
            "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
        }
    )
    registry["summary"] = summary


def _pack_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _pack_contract(assigned_ids)
    return {
        "slug": PACK_SLUG,
        "display_name": PACK_DISPLAY_NAME,
        "sleeve_family": SLEEVE_FAMILY,
        "sleeve_profile": SLEEVE_PROFILE,
        "objective": "Add 24 institutional validation systems covering alpha evidence, execution realism, portfolio risk, regime transitions, options/futures risk, data quality, replay, governance, operator copilots, storage, Grandmaster quality, liquidity, alternatives data, risk transfer, tax, margin, broker reliability, feature ontology, research assimilation, adversarial defense, synthetic markets, curriculum, and reporting evidence.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "systems": list(SYSTEMS),
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "institutional_systems": list(contract["institutional_systems"]),
        "anchor_bot_ids": contract["anchor_bot_ids"],
    }


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
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
        "institutional_alpha_validation_version": PACK_VERSION,
        "system_count": len(SYSTEMS),
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


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    plan = plan_registry_expansion(registry)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
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
        "institutional_alpha_validation_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh institutional-alpha-validation --apply --json",
    }


def apply_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    storage_targets_ready = _ensure_storage_targets(project_root)
    backup_path = ""
    if added_rows:
        backup_dir = project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"master_bot_registry_before_institutional_alpha_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(registry_path, backup)
        backup_path = str(backup)
        rows.extend(added_rows)
        registry["sub_bots"] = rows
        registry["updated_at_utc"] = _utc_now()
        _refresh_summary(registry)
        _write_json(registry_path, registry)

    payload = build_payload(project_root)
    payload.update(
        {
            "mode": "applied",
            "added_bot_count": len(added_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in added_rows],
            "backup_path": backup_path,
            "new_total_bots": len(rows),
            "new_active_bots": sum(1 for row in rows if bool(row.get("active"))),
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
            "storage_targets_ready": storage_targets_ready,
        }
    )
    _write_json(
        project_root / "config" / "institutional_alpha_validation_v1.json",
        {"generated_at_utc": _utc_now(), "institutional_alpha_validation_version": PACK_VERSION, "pack": payload["pack"]},
    )
    _write_json(project_root / "governance" / "health" / "institutional_alpha_validation_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 120-bot institutional alpha validation collect-only pack.")
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
            "institutional_alpha_validation "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
