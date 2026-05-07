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
BASE_VERSION = 1396
TARGET_PLATFORM_TOTAL_BOTS = 1456
PACK_VERSION = "trading_muscle_systems_v1"
PACK_SLUG = "trading_muscle_systems"
PACK_DISPLAY_NAME = "Trading Muscle Systems Pack"
SLEEVE_FAMILY = "trading_muscles"
LABEL_CONTRACT_VERSION = "trading_muscle_systems_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 75000
MINIMUM_COLLECTION_DAYS = 180
PAPER_RUNTIME_CAPACITY_FLOOR = 1000
SAMPLE_RATE = 0.025
MAX_DAILY_MB_PER_BOT = 5


MUSCLES: list[dict[str, Any]] = [
    {
        "slug": "intraday_momentum_muscle",
        "display_name": "Intraday Momentum Muscle",
        "objective": "Learn opening-range, VWAP reclaim, relative-volume, tape-confirmed continuation, and power-hour candidate behavior without execution authority.",
        "outputs": ["intraday_momentum_candidate", "momentum_entry_quality_score", "momentum_exhaustion_flag"],
        "peer_sleeves": ["intraday_aggressive", "day_trading", "order_flow_market_microstructure"],
        "data_intakes": ["rvol_trace", "vwap_reclaim_trace", "opening_range_trace", "tape_momentum_trace"],
    },
    {
        "slug": "intraday_mean_reversion_muscle",
        "display_name": "Intraday Mean Reversion Muscle",
        "objective": "Learn failed-break, VWAP snapback, liquidity-refill, overextension fade, and auction-reversion candidates with strict volatility and liquidity filters.",
        "outputs": ["mean_reversion_candidate", "snapback_quality_score", "fade_invalidation_level"],
        "peer_sleeves": ["intraday_aggressive", "market_microstructure", "auction_imbalance"],
        "data_intakes": ["failed_break_trace", "vwap_deviation_trace", "liquidity_refill_trace"],
    },
    {
        "slug": "swing_trend_muscle",
        "display_name": "Swing Trend Muscle",
        "objective": "Learn multi-day trend, pullback-to-support, relative-strength, volatility-contraction, and earnings-drift continuation candidates.",
        "outputs": ["swing_trend_candidate", "trend_quality_score", "multi_day_exit_plan"],
        "peer_sleeves": ["swing_aggressive", "earnings_drift_quality", "sector_rotation"],
        "data_intakes": ["swing_setup_trace", "relative_strength_trace", "volatility_contraction_trace"],
    },
    {
        "slug": "options_convexity_muscle",
        "display_name": "Options Convexity Muscle",
        "objective": "Learn gamma, vega, skew, event-vol, and spread-structured convexity candidates while tracking greeks, assignment, and liquidity risk.",
        "outputs": ["options_convexity_candidate", "greeks_risk_packet", "convexity_payoff_map"],
        "peer_sleeves": ["options_risk_intelligence_v2", "gamma_scalping", "second_third_order_greeks"],
        "data_intakes": ["options_chain_trace", "greeks_surface_trace", "event_vol_trace", "opra_nbbo_trace"],
    },
    {
        "slug": "options_income_muscle",
        "display_name": "Options Income Muscle",
        "objective": "Learn conservative premium, covered-call, cash-secured-put, vertical-spread, and theta candidate behavior with assignment and drawdown guardrails.",
        "outputs": ["options_income_candidate", "theta_risk_reward_score", "assignment_risk_packet"],
        "peer_sleeves": ["dividend_income", "covered_call_income", "options_risk_intelligence_v2"],
        "data_intakes": ["theta_decay_trace", "assignment_risk_trace", "covered_call_trace"],
    },
    {
        "slug": "futures_macro_muscle",
        "display_name": "Futures Macro Muscle",
        "objective": "Learn ES, NQ, RTY, rates, energy, metals, and macro-event futures candidates using regime-aware risk and overnight-session context.",
        "outputs": ["futures_macro_candidate", "macro_contract_risk_score", "overnight_session_plan"],
        "peer_sleeves": ["schwab_futures", "futures_cross_asset_basis_lab", "sovereign_debt_macro"],
        "data_intakes": ["futures_curve_trace", "macro_event_trace", "overnight_liquidity_trace"],
    },
    {
        "slug": "crypto_basis_muscle",
        "display_name": "Crypto Basis Muscle",
        "objective": "Learn crypto spot, futures, perp-basis, funding, liquidation-ladder, cross-exchange divergence, and weekend-liquidity candidates.",
        "outputs": ["crypto_basis_candidate", "funding_basis_score", "liquidation_risk_packet"],
        "peer_sleeves": ["crypto", "crypto_futures", "crypto_funding_basis_rv_v2"],
        "data_intakes": ["crypto_funding_trace", "liquidation_ladder_trace", "cross_exchange_trace"],
    },
    {
        "slug": "volatility_arbitrage_muscle",
        "display_name": "Volatility Arbitrage Muscle",
        "objective": "Learn IV/RV, term-structure, skew, dispersion, variance, and volatility-risk-premium candidates with hedge-cost realism.",
        "outputs": ["vol_arb_candidate", "iv_rv_edge_score", "hedge_cost_packet"],
        "peer_sleeves": ["volatility_arbitrage", "volatility_risk_premium_harvesting", "dispersion_trading"],
        "data_intakes": ["vol_surface_trace", "realized_vol_trace", "dispersion_trace"],
    },
    {
        "slug": "event_driven_muscle",
        "display_name": "Event-Driven Muscle",
        "objective": "Learn earnings, FOMC, CPI, corporate-action, deal-spread, and regulatory-event candidates with pre/post-event decay maps.",
        "outputs": ["event_driven_candidate", "event_probability_packet", "post_event_decay_score"],
        "peer_sleeves": ["event_intelligence", "merger_event_arbitrage", "macro_context"],
        "data_intakes": ["event_calendar_trace", "earnings_reaction_trace", "macro_release_trace"],
    },
    {
        "slug": "relative_value_pairs_muscle",
        "display_name": "Relative Value And Pairs Muscle",
        "objective": "Learn cointegration, residualized pairs, ETF/NAV, ADR parity, sector spread, and cross-listing relative-value candidates.",
        "outputs": ["relative_value_candidate", "spread_reversion_score", "hedge_ratio_packet"],
        "peer_sleeves": ["statistical_arbitrage", "cointegration_ou_pairs", "etf_basket_nav_arbitrage"],
        "data_intakes": ["pairs_spread_trace", "hedge_ratio_trace", "nav_parity_trace"],
    },
    {
        "slug": "portfolio_hedging_muscle",
        "display_name": "Portfolio Hedging Muscle",
        "objective": "Learn hedge candidates for drawdown control, beta offsets, collar structures, tail protection, correlation spikes, and sleeve crowding.",
        "outputs": ["portfolio_hedge_candidate", "tail_hedge_efficiency_score", "exposure_offset_packet"],
        "peer_sleeves": ["portfolio_brain", "tail_risk_parity", "black_swan_hedging"],
        "data_intakes": ["portfolio_exposure_trace", "tail_risk_trace", "correlation_spike_trace"],
    },
    {
        "slug": "execution_timing_muscle",
        "display_name": "Execution Timing Muscle",
        "objective": "Learn order timing, order type, queue, spread, auction, venue, and broker-state plans that reduce slippage in paper rehearsals.",
        "outputs": ["execution_timing_plan", "slippage_reduction_score", "order_type_vote"],
        "peer_sleeves": ["execution_realism_layer", "execution_quality_lab_v2", "broker_venue_reliability_lab"],
        "data_intakes": ["paper_fill_trace", "spread_cost_trace", "queue_position_trace", "venue_health_trace"],
    },
    {
        "slug": "position_sizing_muscle",
        "display_name": "Position Sizing Muscle",
        "objective": "Learn confidence, volatility, liquidity, correlation, margin, Kelly-capped, and drawdown-aware sizing plans before any paper allocation can occur.",
        "outputs": ["position_size_plan", "risk_budget_score", "margin_safe_size_cap"],
        "peer_sleeves": ["portfolio_intelligence_layer", "funding_collateral_margin_intelligence", "capital_simulator"],
        "data_intakes": ["risk_budget_trace", "margin_guard_trace", "liquidity_capacity_trace"],
    },
    {
        "slug": "exit_rebalance_muscle",
        "display_name": "Exit And Rebalance Muscle",
        "objective": "Learn stop, trim, profit-taking, thesis-decay, time-stop, rebalance, and de-risk plans for every candidate family.",
        "outputs": ["exit_plan", "rebalance_trigger_score", "thesis_decay_alert"],
        "peer_sleeves": ["position_lifecycle", "alpha_decay_tracker", "portfolio_brain"],
        "data_intakes": ["exit_outcome_trace", "rebalance_trace", "thesis_decay_trace"],
    },
]


ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "signal_collector", "label": "Signal Collector", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "candidate_modeler", "label": "Candidate Modeler", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "risk_sizer", "label": "Risk Sizer", "bot_role": "signal_sub_bot", "priority": "critical"},
    {"suffix": "execution_rehearsal_guard", "label": "Execution Rehearsal Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "master_bridge", "label": "Master Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]


BASE_DATA_INTAKES = [
    "trading_muscle_signal_trace",
    "trade_candidate_lifecycle_trace",
    "paper_trade_lock_trace",
    "execution_realism_trace",
    "position_sizing_trace",
    "portfolio_exposure_trace",
    "regime_router_trace",
    "alpha_decay_trace",
    "global_halt_status_trace",
    "data_quality_contract_trace",
]

REQUIRED_LABELS = [
    "candidate_edge_bucket",
    "candidate_execution_quality_bucket",
    "candidate_risk_budget_bucket",
    "candidate_regime_fit_bucket",
    "candidate_hedge_need_bucket",
    "candidate_exit_quality_bucket",
    "paper_trade_readiness_state",
    "promotion_gate_status",
]

STORAGE_TARGETS = [
    "governance/trading_muscle_systems",
    *[f"governance/trading_muscle_systems/{muscle['slug']}" for muscle in MUSCLES],
    "governance/health/trading_muscle_systems_latest.json",
]


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for muscle in MUSCLES:
        for role in ROLE_TEMPLATES:
            role_slug = f"{muscle['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"trading_muscle_{role_slug}_bot",
                    "label": f"{muscle['display_name']} {role['label']}",
                    "muscle": muscle["slug"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {muscle['objective']}",
                    "target_functions": list(muscle.get("outputs", [])),
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
        if desired not in used_versions:
            version = desired
            used_versions.add(version)
        else:
            version = _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _muscle(bot: dict[str, Any]) -> dict[str, Any]:
    for muscle in MUSCLES:
        if muscle["slug"] == bot["muscle"]:
            return muscle
    return {"slug": bot["muscle"], "display_name": bot["muscle"], "objective": bot["objective"], "outputs": []}


def _threshold_progress() -> dict[str, Any]:
    return {
        "observations": 0,
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "observations_ready": False,
        "collection_age_days": 0.0,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "days_ready": False,
        "training_ready": False,
        "paper_readiness_ready": False,
    }


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "sleeve_family": SLEEVE_FAMILY,
        "display_name": PACK_DISPLAY_NAME,
        "muscle_count": len(MUSCLES),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "14_trading_muscles_5_bots_each_70_bot_action_candidate_layer",
        "muscle_systems": [muscle["slug"] for muscle in MUSCLES],
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "trading_muscles_hot_5d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": MAX_DAILY_MB_PER_BOT,
            "capture_mode": "thin_digest_first_trade_candidate_trace",
            "sample_rate": SAMPLE_RATE,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_candidate_scores_and_execution_digests_stage_raw_trade_traces",
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
            bot["muscle"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("signal_collector")
        },
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "global_halt_contract": "trading_muscle_pack_can_generate_candidates_only_when_halt_gates_are_clear_and_never_force_clear_halts",
        "paper_lock_contract": "no_execution_no_allocation_no_paper_no_training_until_180_days_75000_observations_and_candidate_execution_risk_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    muscle = _muscle(bot)
    contract = _pack_contract(assigned_ids)
    muscle_slug = str(muscle["slug"])
    data_intakes = list(BASE_DATA_INTAKES) + list(muscle.get("data_intakes", [])) + [
        f"{muscle_slug}_candidate_trace",
        f"{muscle_slug}_outcome_label_trace",
    ]
    peer_sleeves = [
        "platform_organ_systems",
        "execution_realism_layer",
        "portfolio_brain",
        "bot_promotion_court",
        "quant_strategy_gap",
        *list(muscle.get("peer_sleeves", [])),
    ]
    muscle_contract = {
        "contract_version": "trading_muscle_systems_layers_v1",
        "capability_pack": PACK_SLUG,
        "trading_muscle": muscle_slug,
        "muscle_display_name": muscle["display_name"],
        "muscle_outputs": list(muscle.get("outputs", [])),
        "operational_boundary": "candidate_generation_and_rehearsal_only_no_execution_authority_until_training_paper_and_global_halt_gates_clear",
        "pressure_boundary": "thin_digest_storage_low_compute_collect_only",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "trading_muscle_systems_expansion_slot",
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
        "promotion_reason": "trading_muscle_systems_expansion_slot",
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
            "risk_on_trend",
            "risk_off_trend",
            "rangebound_transition",
            "event_window",
            "liquidity_dislocation",
            "volatility_expansion",
            "overnight_drain",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v1086_institutional_alpha_evidence_court_evidence_collector_bot",
            "brain_refinery_v1091_institutional_execution_quality_lab_v2_evidence_collector_bot",
            "brain_refinery_v1206_strategy_gap_convertible_bond_arbitrage_evidence_collector_bot",
            "brain_refinery_v1371_platform_organ_operator_cockpit_v2_telemetry_collector_bot",
        ],
        "data_intake_collections": data_intakes,
        "storage_targets": [
            "governance/trading_muscle_systems",
            f"governance/trading_muscle_systems/{muscle_slug}",
            "governance/health/trading_muscle_systems_latest.json",
        ],
        "freshness_slo_seconds": 900,
        "retention_profile": "trading_muscles_hot_5d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_trade_candidate_observer",
        "data_collection_reason": "trading_muscle_systems_collect_candidate_evidence_until_data_quality_execution_risk_and_promotion_gates_clear",
        "trade_candidate_collection_active": True,
        "paper_trade_readiness_gated": True,
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "trading_muscle_systems_collection_only_zero_weight_no_trade_authority",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_trade_candidate_and_execution_rehearsal_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_data_quality_clearance": True,
            "requires_execution_realism_clearance": True,
            "requires_portfolio_risk_clearance": True,
            "requires_regime_fit_clearance": True,
            "requires_alpha_decay_clearance": True,
            "requires_paper_trade_lock_clearance": True,
            "requires_global_halt_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_sampled",
        "data_collection_sample_rate": SAMPLE_RATE,
        "data_collection_max_daily_storage_mb": MAX_DAILY_MB_PER_BOT,
        "data_collection_max_daily_mb": float(MAX_DAILY_MB_PER_BOT),
        "data_collection_compute_guard_mode": "thin_digest",
        "data_collection_resource_guard_reason": "trading_muscle_pack_uses_digest_only_candidate_capture_to_protect_cpu_memory_storage",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "blocked_until_candidate_evidence_execution_risk_and_promotion_thresholds_clear",
        "paper_runtime_control_refresh_seconds": 300,
        "sleeve_profile": muscle_slug,
        "sleeve_family": SLEEVE_FAMILY,
        "trading_muscle": muscle_slug,
        "strategy_family": "trading_action_candidate_muscles",
        "correlation_peer_sleeves": sorted(set(peer_sleeves)),
        "correlation_dependencies": [
            "platform_organ_systems",
            "execution_realism_layer",
            "portfolio_brain",
            "bot_promotion_court",
            "alpha_decay_tracker",
            "regime_router",
            "paper_trade_lock_guard",
            "global_halt_guard",
        ],
        "provider_capability_profile": "market_data_and_internal_candidate_telemetry_collect_only",
        "direct_market_data_available": True,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "governance_health",
            "decision_provenance",
            "paper_fill_rehearsals",
            "execution_quality_context",
            "portfolio_risk_context",
        ],
        "schwab_direct_inputs": ["quotes", "chains", "market_hours", "fundamentals", "corporate_actions"],
        "proxy_only_reason": "trading_muscle_pack_collects_candidate_and_rehearsal_labels_only_until_training_and_paper_gates_clear",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "primary_horizon": f"{muscle_slug}_paper_candidate_quality_after_costs_and_risk",
            "required_context": data_intakes,
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.88,
            "freshness_slo_seconds": 900,
            "regression_guard_bot_id": contract["anchor_bot_ids"].get(muscle_slug, ""),
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"sleeve_profile:{muscle_slug}",
            f"trading_muscle:{muscle_slug}",
            f"capability_pack:{PACK_SLUG}",
            "trading_muscle_systems",
            "trade_candidate_only",
            "point_in_time_only",
            "training_after_threshold",
            "global_halt_aware",
            "pressure_safe",
            "mlx_default",
        ],
        "execution_policy_label": "collection_only_trading_muscle_systems_no_execution",
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
            "trade_candidate_rehearsal",
            "point_in_time_labeling",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "thin_digest_trading_muscle_systems",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "trading_muscle_systems_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "trading_muscle_systems_contract": muscle_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("trading_muscle_systems_version") or "") == PACK_VERSION]
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
            "trading_muscle_systems_bot_count": len(pack_rows),
            "latest_trading_muscle_systems": PACK_VERSION,
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
        "objective": "Add 14 trading muscle systems that turn platform intelligence into safe trade candidates, sizing plans, hedge plans, exits, and execution rehearsals without live or paper authority until gates clear.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "muscle_count": len(MUSCLES),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "muscles": list(MUSCLES),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "muscle_systems": list(contract["muscle_systems"]),
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
        "trading_muscle_systems_version": PACK_VERSION,
        "muscle_count": len(MUSCLES),
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
        "trading_muscle_systems_version": PACK_VERSION,
        "muscle_count": plan["muscle_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh trading-muscles --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_trading_muscle_systems_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "trading_muscle_systems_v1.json",
        {"generated_at_utc": _utc_now(), "trading_muscle_systems_version": PACK_VERSION, "pack": payload["pack"]},
    )
    _write_json(project_root / "governance" / "health" / "trading_muscle_systems_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 70-bot trading muscle systems collect-only pack.")
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
            "trading_muscle_systems "
            f"mode={payload['mode']} muscles={payload['muscle_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
