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
BASE_VERSION = 1206
TARGET_PLATFORM_TOTAL_BOTS = 1316
PACK_VERSION = "quant_strategy_gap_v1"
PACK_SLUG = "quant_strategy_gap"
PACK_DISPLAY_NAME = "Quant Strategy Gap Pack"
SLEEVE_FAMILY = "tradable_alpha_gap"
LABEL_CONTRACT_VERSION = "quant_strategy_gap_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 45000
MINIMUM_COLLECTION_DAYS = 120
PAPER_RUNTIME_CAPACITY_FLOOR = 1000
SAMPLE_RATE = 0.03
MAX_DAILY_MB_PER_BOT = 4


STRATEGIES: list[dict[str, Any]] = [
    {
        "slug": "convertible_bond_arbitrage",
        "display_name": "Convertible Bond Arbitrage",
        "objective": "Track convertible bond relative value across equity delta, credit spread, rates, implied volatility, and borrow pressure.",
        "outputs": ["convertible_rv_packet", "delta_credit_vol_score", "convertible_hedge_feasibility"],
        "peer_sleeves": ["credit_derivatives_cdx_cds", "volatility_arbitrage", "second_third_order_greeks"],
        "proxy_sources": ["sec_edgar_context", "rates_curve_context", "options_flow_context", "credit_proxy_context"],
    },
    {
        "slug": "capital_structure_arbitrage",
        "display_name": "Capital Structure Arbitrage",
        "objective": "Compare equity, credit, volatility, borrow, and balance-sheet signals to locate capital structure dislocations.",
        "outputs": ["capital_structure_rv_score", "equity_credit_gap_packet", "balance_sheet_stress_flag"],
        "peer_sleeves": ["credit_derivatives_cdx_cds", "repo_securities_lending", "xva_counterparty_margin"],
        "proxy_sources": ["sec_edgar_context", "credit_proxy_context", "options_flow_context", "borrow_fee_context"],
    },
    {
        "slug": "merger_event_arbitrage",
        "display_name": "Merger And Event Arbitrage",
        "objective": "Monitor deal spreads, event probability, regulatory risk, financing stress, break-risk, and post-announcement behavior.",
        "outputs": ["deal_spread_packet", "event_probability_score", "break_risk_alert"],
        "peer_sleeves": ["event_intelligence", "tax_corporate_actions_intelligence", "model_risk_validation"],
        "proxy_sources": ["sec_edgar_context", "news_context", "corporate_actions_context", "market_reaction_context"],
    },
    {
        "slug": "etf_basket_nav_arbitrage",
        "display_name": "ETF Basket And NAV Arbitrage",
        "objective": "Track ETF premium/discount, creation-redemption pressure, basket liquidity, and NAV dislocation persistence.",
        "outputs": ["etf_nav_gap_packet", "basket_liquidity_score", "creation_redemption_pressure"],
        "peer_sleeves": ["market_structure", "liquidity_regime", "portfolio_construction"],
        "proxy_sources": ["etf_holdings_context", "market_micro_context", "volume_liquidity_context"],
    },
    {
        "slug": "index_rebalance_arbitrage",
        "display_name": "Index Rebalance Arbitrage",
        "objective": "Estimate forced-flow pressure around index additions, deletions, weight changes, and rebalance windows.",
        "outputs": ["rebalance_flow_forecast", "index_event_pressure_score", "forced_flow_decay_packet"],
        "peer_sleeves": ["event_intelligence", "sector_rotation", "liquidity_regime"],
        "proxy_sources": ["index_constituent_context", "corporate_actions_context", "volume_liquidity_context"],
    },
    {
        "slug": "volatility_risk_premium_harvesting",
        "display_name": "Volatility Risk Premium Harvesting",
        "objective": "Measure implied-vs-realized volatility carry by tenor, skew, regime, event calendar, and hedge cost.",
        "outputs": ["vrp_term_packet", "implied_realized_spread_score", "hedged_vol_carry_vote"],
        "peer_sleeves": ["volatility_arbitrage", "variance_volatility_swaps", "options_risk_intelligence_v2"],
        "proxy_sources": ["options_flow_context", "vol_surface_context", "realized_vol_context"],
    },
    {
        "slug": "cross_asset_carry",
        "display_name": "Cross-Asset Carry",
        "objective": "Unify futures carry, FX carry, rates carry, dividend carry, crypto basis, and volatility carry into one risk-adjusted view.",
        "outputs": ["cross_asset_carry_board", "carry_drawdown_stress_score", "carry_crowding_flag"],
        "peer_sleeves": ["futures_cross_asset_basis_lab", "fx", "dividend_income", "crypto_futures"],
        "proxy_sources": ["futures_curve_context", "fx_market_context", "rates_curve_context", "crypto_market_context"],
    },
    {
        "slug": "residual_equity_stat_arb",
        "display_name": "Residualized Equity Stat-Arb",
        "objective": "Residualize equity returns against beta, sector, style factors, volatility, liquidity, and macro before ranking reversion.",
        "outputs": ["residual_alpha_rank", "factor_neutrality_packet", "residual_half_life_score"],
        "peer_sleeves": ["statistical_arbitrage", "portfolio_construction", "feature_quality_data_confidence"],
        "proxy_sources": ["factor_model_context", "market_micro_context", "extended_quant_context"],
    },
    {
        "slug": "cointegration_ou_pairs",
        "display_name": "Cointegration And OU Pairs Trading",
        "objective": "Track cointegrated pairs with dynamic hedge ratios, OU half-life, spread z-score, liquidity, and borrow feasibility.",
        "outputs": ["cointegration_pair_rank", "ou_spread_state", "dynamic_hedge_ratio_packet"],
        "peer_sleeves": ["statistical_arbitrage", "state_space_models", "portfolio_construction"],
        "proxy_sources": ["extended_quant_context", "market_micro_context", "borrow_fee_context"],
    },
    {
        "slug": "auction_imbalance",
        "display_name": "Auction Imbalance Strategies",
        "objective": "Track opening and closing auction imbalance, MOC/LOC flow, liquidity absorption, and post-auction drift.",
        "outputs": ["auction_imbalance_packet", "moc_loc_pressure_score", "auction_drift_followthrough"],
        "peer_sleeves": ["market_structure", "order_flow_market_microstructure", "intraday_aggressive"],
        "proxy_sources": ["market_micro_context", "auction_imbalance_context", "volume_liquidity_context"],
    },
    {
        "slug": "short_borrow_squeeze",
        "display_name": "Short Interest And Borrow Squeeze",
        "objective": "Track short interest, borrow cost, fail-to-deliver stress, float constraints, options gamma, and squeeze risk.",
        "outputs": ["borrow_squeeze_packet", "short_interest_pressure_score", "float_constraint_alert"],
        "peer_sleeves": ["repo_securities_lending", "dealer_opex_pinning_v2", "options_risk_intelligence_v2"],
        "proxy_sources": ["borrow_fee_context", "short_interest_context", "options_flow_context"],
    },
    {
        "slug": "dealer_opex_pinning_v2",
        "display_name": "Dealer Expiry And OPEX Pinning v2",
        "objective": "Estimate dealer gamma, charm, vanna, max-pain, open-interest walls, expiry-week pinning, and unwind risk.",
        "outputs": ["dealer_expiry_map", "pinning_pressure_score", "expiry_unwind_risk_packet"],
        "peer_sleeves": ["gamma_scalping", "second_third_order_greeks", "order_flow_toxicity"],
        "proxy_sources": ["options_flow_context", "opra_nbbo_context", "vol_surface_context"],
    },
    {
        "slug": "commodity_seasonal_curve",
        "display_name": "Commodity Seasonal Curve Strategies",
        "objective": "Track commodity term-structure carry, inventory seasonality, roll yield, inflation sensitivity, and spread dislocations.",
        "outputs": ["commodity_curve_carry_packet", "seasonal_inventory_score", "roll_yield_stress_flag"],
        "peer_sleeves": ["futures_cross_asset_basis_lab", "sovereign_debt_macro", "macro_crisis_scenario_lab"],
        "proxy_sources": ["futures_curve_context", "official_macro_context", "energy_inventory_context"],
    },
    {
        "slug": "energy_weather_demand_shock",
        "display_name": "Energy Weather And Demand Shock",
        "objective": "Monitor weather, inventory, production, refinery utilization, and demand shocks for energy-sensitive assets.",
        "outputs": ["energy_weather_shock_packet", "demand_supply_gap_score", "inventory_surprise_alert"],
        "peer_sleeves": ["commodity_seasonal_curve", "macro_crisis_scenario_lab", "futures_cross_asset_basis_lab"],
        "proxy_sources": ["weather_context", "eia_inventory_context", "official_macro_context"],
    },
    {
        "slug": "insider_buyback_signal",
        "display_name": "Insider Buying And Buyback Signal",
        "objective": "Track insider transactions, buyback authorizations, blackout windows, dilution, and follow-through quality.",
        "outputs": ["insider_buyback_packet", "corporate_bid_score", "blackout_window_flag"],
        "peer_sleeves": ["tax_corporate_actions_intelligence", "event_intelligence", "dividend_income"],
        "proxy_sources": ["sec_edgar_context", "corporate_actions_context", "news_context"],
    },
    {
        "slug": "earnings_drift_quality",
        "display_name": "Earnings Drift Quality",
        "objective": "Score post-earnings announcement drift using surprise quality, guidance tone, revisions, liquidity, and event crowding.",
        "outputs": ["earnings_drift_quality_score", "guidance_revision_packet", "post_event_decay_flag"],
        "peer_sleeves": ["event_intelligence", "swing_aggressive", "feature_quality_data_confidence"],
        "proxy_sources": ["sec_edgar_context", "earnings_context", "news_context", "market_reaction_context"],
    },
    {
        "slug": "crypto_funding_basis_rv_v2",
        "display_name": "Crypto Funding And Basis Relative Value v2",
        "objective": "Track spot-perp-futures basis, funding crowding, liquidation ladders, cross-exchange spreads, and collateral stress.",
        "outputs": ["crypto_basis_rv_packet", "funding_crowding_score", "liquidation_basis_alert"],
        "peer_sleeves": ["crypto_futures", "cross_asset_basis_training", "order_flow_toxicity"],
        "proxy_sources": ["crypto_market_context", "coinbase_futures_context", "derivatives_exchange_context"],
    },
    {
        "slug": "rates_curve_relative_value",
        "display_name": "Rates Curve Relative Value",
        "objective": "Monitor curve butterflies, steepeners, flatteners, SOFR pressure, auction liquidity, inflation breakevens, and duration hedges.",
        "outputs": ["rates_curve_rv_packet", "curve_butterfly_score", "duration_hedge_quality"],
        "peer_sleeves": ["sovereign_debt_macro", "futures_cross_asset_basis_lab", "bond"],
        "proxy_sources": ["rates_curve_context", "official_macro_context", "treasury_auction_context"],
    },
    {
        "slug": "credit_equity_vol_relative_value",
        "display_name": "Credit Equity Vol Relative Value",
        "objective": "Compare credit spread movement, equity downside, implied skew, realized vol, and balance-sheet stress.",
        "outputs": ["credit_equity_vol_gap_packet", "skew_credit_stress_score", "downside_hedge_rv_vote"],
        "peer_sleeves": ["capital_structure_arbitrage", "credit_derivatives_cdx_cds", "volatility_arbitrage"],
        "proxy_sources": ["credit_proxy_context", "options_flow_context", "sec_edgar_context"],
    },
    {
        "slug": "corporate_bond_etf_discount_arb",
        "display_name": "Corporate Bond ETF Discount Arbitrage",
        "objective": "Track bond ETF premium/discount, credit liquidity, NAV stale pricing, creation-redemption pressure, and spread shock risk.",
        "outputs": ["bond_etf_discount_packet", "credit_liquidity_stale_nav_score", "bond_etf_rebalance_alert"],
        "peer_sleeves": ["etf_basket_nav_arbitrage", "credit_derivatives_cdx_cds", "liquidity_stress_market_impact_lab"],
        "proxy_sources": ["etf_holdings_context", "credit_proxy_context", "rates_curve_context"],
    },
    {
        "slug": "adr_cross_listing_parity",
        "display_name": "ADR And Cross-Listing Parity",
        "objective": "Track ADR parity gaps, FX translation, home-market lead-lag, liquidity windows, and corporate action adjustments.",
        "outputs": ["adr_parity_packet", "cross_listing_lead_lag_score", "fx_translation_gap_alert"],
        "peer_sleeves": ["fx", "cross_asset_risk_transfer_lab", "market_structure"],
        "proxy_sources": ["fx_market_context", "global_equity_context", "corporate_actions_context"],
    },
    {
        "slug": "tax_loss_rebalance_flow",
        "display_name": "Tax-Loss And Rebalance Flow",
        "objective": "Estimate tax-loss selling, seasonal rebalance flows, wash-sale windows, January effect behavior, and forced-flow recovery.",
        "outputs": ["tax_loss_flow_packet", "rebalance_recovery_score", "wash_sale_window_flag"],
        "peer_sleeves": ["tax_corporate_actions_intelligence", "index_rebalance_arbitrage", "dividend_income"],
        "proxy_sources": ["corporate_actions_context", "calendar_context", "market_reaction_context"],
    },
    {
        "slug": "passive_liquidity_provision_sim",
        "display_name": "Passive Liquidity Provision Simulator",
        "objective": "Simulate passive quoting, queue position, adverse selection, spread capture, inventory drift, and cancellation discipline.",
        "outputs": ["passive_quote_sim_packet", "adverse_selection_score", "inventory_drift_guard"],
        "peer_sleeves": ["high_frequency_market_making", "execution_quality_lab_v2", "order_flow_market_microstructure"],
        "proxy_sources": ["market_micro_context", "opra_nbbo_context", "execution_quality_context"],
    },
    {
        "slug": "sector_pair_rotation_spread_arb",
        "display_name": "Sector Pair Rotation And Spread Arb",
        "objective": "Track sector pair spreads, factor-neutral rotation, macro catalyst sensitivity, dispersion, and crowding pressure.",
        "outputs": ["sector_pair_spread_packet", "rotation_factor_neutral_score", "sector_crowding_alert"],
        "peer_sleeves": ["sector_rotation", "dispersion_trading", "portfolio_construction"],
        "proxy_sources": ["factor_model_context", "sector_etf_context", "market_micro_context"],
    },
]


ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "evidence_collector", "label": "Evidence Collector", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "signal_modeler", "label": "Signal Modeler", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "risk_guard", "label": "Risk Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "execution_simulator", "label": "Execution Simulator", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "master_bridge", "label": "Master Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]


BASE_DATA_INTAKES = [
    "strategy_gap_registry_trace",
    "cross_asset_market_context_trace",
    "point_in_time_feature_join_trace",
    "execution_realism_trace",
    "portfolio_overlap_trace",
    "liquidity_capacity_trace",
    "backpressure_safe_digest_trace",
    "model_governance_training_gate_trace",
]

REQUIRED_LABELS = [
    "strategy_gap_alpha_bucket",
    "tradability_quality_bucket",
    "execution_cost_bucket",
    "liquidity_capacity_bucket",
    "source_confidence_bucket",
    "cross_sleeve_overlap_bucket",
    "regime_dependency_bucket",
    "paper_trade_readiness_bucket",
    "training_evidence_status",
]

STORAGE_TARGETS = [
    "governance/quant_strategy_gap",
    *[f"governance/quant_strategy_gap/{strategy['slug']}" for strategy in STRATEGIES],
    "governance/health/quant_strategy_gap_latest.json",
]


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for role in ROLE_TEMPLATES:
            role_slug = f"{strategy['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"strategy_gap_{role_slug}_bot",
                    "label": f"{strategy['display_name']} {role['label']}",
                    "strategy": strategy["slug"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {strategy['objective']}",
                    "target_functions": list(strategy.get("outputs", [])),
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


def _strategy(bot: dict[str, Any]) -> dict[str, Any]:
    for strategy in STRATEGIES:
        if strategy["slug"] == bot["strategy"]:
            return strategy
    return {"slug": bot["strategy"], "display_name": bot["strategy"], "objective": bot["objective"], "outputs": []}


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
        "sleeve_family": SLEEVE_FAMILY,
        "display_name": PACK_DISPLAY_NAME,
        "strategy_count": len(STRATEGIES),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "24_strategies_5_bots_each_120_bot_tradable_alpha_gap_layer",
        "strategy_sleeves": [strategy["slug"] for strategy in STRATEGIES],
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "strategy_gap_hot_5d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": MAX_DAILY_MB_PER_BOT,
            "capture_mode": "thin_digest_first_strategy_trace",
            "sample_rate": SAMPLE_RATE,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_strategy_scores_and_evidence_digests_stage_raw_gap_traces",
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
            bot["strategy"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("evidence_collector")
        },
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "global_halt_contract": "strategy_gap_pack_can_reduce_bad_alpha_and_pressure_sources_but_never_force_clear_halts",
        "paper_lock_contract": "no_execution_no_allocation_no_training_until_120_days_45000_observations_and_strategy_evidence_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    strategy = _strategy(bot)
    contract = _pack_contract(assigned_ids)
    strategy_slug = str(strategy["slug"])
    data_intakes = list(BASE_DATA_INTAKES) + [f"{strategy_slug}_feature_trace", f"{strategy_slug}_label_quality_trace"]
    peer_sleeves = [
        "institutional_alpha_validation",
        "execution_quality_lab_v2",
        "portfolio_construction",
        "model_risk_validation",
        "feature_quality_data_confidence",
        *list(strategy.get("peer_sleeves", [])),
    ]
    proxy_sources = [
        "master_bot_registry",
        "governance_health",
        "decision_provenance",
        *list(strategy.get("proxy_sources", [])),
    ]
    strategy_contract = {
        "contract_version": "quant_strategy_gap_layers_v1",
        "capability_pack": PACK_SLUG,
        "strategy_sleeve": strategy_slug,
        "strategy_display_name": strategy["display_name"],
        "strategy_outputs": list(strategy.get("outputs", [])),
        "tradability_boundary": "collection_only_no_execution_authority_until_evidence_and_paper_gates_clear",
        "pressure_boundary": "thin_digest_storage_low_compute_collect_only",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "quant_strategy_gap_expansion_slot",
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
        "promotion_reason": "quant_strategy_gap_expansion_slot",
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
            "rangebound_transition",
            "fragile_transition",
            "risk_off_trend",
            "risk_on_trend",
            "event_window",
            "liquidity_dislocation",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v408_statistical_arbitrage_factor_residual_reversion_bot",
            "brain_refinery_v433_volatility_arbitrage_iv_rv_spread_bot",
            "brain_refinery_v704_etf_creation_redemption_flow_bot",
            "brain_refinery_v1086_institutional_alpha_evidence_court_evidence_collector_bot",
        ],
        "data_intake_collections": data_intakes,
        "storage_targets": [
            "governance/quant_strategy_gap",
            f"governance/quant_strategy_gap/{strategy_slug}",
            "governance/health/quant_strategy_gap_latest.json",
        ],
        "freshness_slo_seconds": 1800,
        "retention_profile": "strategy_gap_hot_5d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "quant_strategy_gap_collect_only_until_tradability_data_quality_execution_and_portfolio_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "quant_strategy_gap_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_strategy_gap_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_alpha_evidence_court_clearance": True,
            "requires_execution_quality_clearance": True,
            "requires_portfolio_overlap_clearance": True,
            "requires_data_quality_clearance": True,
            "requires_duplicate_alpha_clearance": True,
            "requires_global_halt_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_sampled",
        "data_collection_sample_rate": SAMPLE_RATE,
        "data_collection_max_daily_storage_mb": MAX_DAILY_MB_PER_BOT,
        "data_collection_max_daily_mb": float(MAX_DAILY_MB_PER_BOT),
        "data_collection_compute_guard_mode": "thin_digest",
        "data_collection_resource_guard_reason": "quant_strategy_gap_pack_uses_digest_only_capture_to_protect_cpu_memory_storage",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "buffered_jsonl_batching_after_threshold_only",
        "paper_runtime_control_refresh_seconds": 300,
        "sleeve_profile": strategy_slug,
        "sleeve_family": SLEEVE_FAMILY,
        "strategy_sleeve": strategy_slug,
        "strategy_family": "tradable_alpha_gap",
        "correlation_peer_sleeves": sorted(set(peer_sleeves)),
        "correlation_dependencies": [
            "platform_brain_v6",
            "institutional_alpha_validation",
            "expansion_capacity_planner",
            "paper_trade_lock_guard",
            "execution_lab",
            "portfolio_exposure_brain",
            "data_source_confidence_engine",
        ],
        "provider_capability_profile": "mixed_direct_and_proxy_market_data_collect_only",
        "direct_market_data_available": True,
        "direct_execution_allowed": False,
        "proxy_data_sources": sorted(set(proxy_sources)),
        "schwab_direct_inputs": ["quotes", "chains", "market_hours", "fundamentals", "corporate_actions"],
        "proxy_only_reason": "strategy_gap_pack_collects_features_and_labels_only_until_training_and_paper_gates_clear",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "primary_horizon": f"{strategy_slug}_risk_adjusted_alpha_after_costs",
            "required_context": data_intakes,
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.84,
            "freshness_slo_seconds": 1800,
            "regression_guard_bot_id": contract["anchor_bot_ids"].get(strategy_slug, ""),
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"sleeve_profile:{strategy_slug}",
            f"strategy_sleeve:{strategy_slug}",
            f"capability_pack:{PACK_SLUG}",
            "tradable_alpha_gap",
            "point_in_time_only",
            "training_after_threshold",
            "global_halt_aware",
            "pressure_safe",
            "mlx_default",
        ],
        "execution_policy_label": "collection_only_quant_strategy_gap_no_execution",
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
            "tradable_alpha_evidence",
            "point_in_time_labeling",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "thin_digest_quant_strategy_gap",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "quant_strategy_gap_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "quant_strategy_gap_contract": strategy_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("quant_strategy_gap_version") or "") == PACK_VERSION]
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
            "quant_strategy_gap_bot_count": len(pack_rows),
            "latest_quant_strategy_gap": PACK_VERSION,
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
        "objective": "Add 24 practical tradable-alpha sleeves covering event arb, relative value, carry, auction flow, dealer expiry, rates/credit/ETF dislocations, crypto basis, sector spreads, and execution-safe liquidity simulation.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "strategy_count": len(STRATEGIES),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "strategies": list(STRATEGIES),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "strategy_sleeves": list(contract["strategy_sleeves"]),
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
        "quant_strategy_gap_version": PACK_VERSION,
        "strategy_count": len(STRATEGIES),
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
        "quant_strategy_gap_version": PACK_VERSION,
        "strategy_count": plan["strategy_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh quant-strategy-gap --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_quant_strategy_gap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "quant_strategy_gap_v1.json",
        {"generated_at_utc": _utc_now(), "quant_strategy_gap_version": PACK_VERSION, "pack": payload["pack"]},
    )
    _write_json(project_root / "governance" / "health" / "quant_strategy_gap_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 120-bot quant strategy gap collect-only pack.")
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
            "quant_strategy_gap "
            f"mode={payload['mode']} strategies={payload['strategy_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
