#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "deep_quant_layer_upgrade_latest.json"
REPORT_PATH = PROJECT_ROOT / "governance" / "deep_quant_layer_upgrade" / "deep_quant_layer_upgrade_latest.md"


DEEP_QUANT_LAYERS: list[dict[str, Any]] = [
    {
        "layer": 1,
        "slug": "universal_residual_alpha_layer",
        "display_name": "Universal Residual Alpha Layer",
        "objective": "Residualize every candidate against market beta, sector, style, macro, volatility, liquidity, and crowding so rankings prefer unexplained edge.",
        "signals": [
            "rolling_factor_exposures",
            "pca_residual_returns",
            "sector_style_neutrality",
            "residual_half_life",
            "alpha_novelty",
        ],
        "outputs": [
            "residual_alpha_score",
            "factor_neutrality_packet",
            "residual_decay_alert",
            "sleeve_overlap_adjustment",
        ],
        "peer_sleeves": [
            "statistical_arbitrage",
            "stat_arb_market_neutral",
            "portfolio_construction",
            "feature_quality_data_confidence",
            "alpha_research_os",
        ],
    },
    {
        "layer": 2,
        "slug": "meta_labeling_trade_filter",
        "display_name": "Meta-Labeling Trade Filter",
        "objective": "Add a second-pass trade filter that decides whether a primary signal should be traded, skipped, downsized, delayed, or watched only.",
        "signals": [
            "primary_signal_confidence",
            "post_signal_mfe_mae",
            "paper_live_fill_gap",
            "regime_state",
            "liquidity_and_event_risk",
        ],
        "outputs": [
            "meta_label_vote",
            "trade_skip_reason",
            "size_multiplier_hint",
            "follow_through_quality",
        ],
        "peer_sleeves": [
            "signal_governance_integrity",
            "model_risk_validation",
            "transaction_cost_slippage_intelligence",
            "feature_quality_data_confidence",
        ],
    },
    {
        "layer": 3,
        "slug": "conformal_uncertainty_abstention_layer",
        "display_name": "Conformal Uncertainty And Abstention Layer",
        "objective": "Attach calibrated prediction intervals to model outputs and abstain when uncertainty, drift, or missing-data risk is too high.",
        "signals": [
            "conformal_residuals",
            "model_disagreement",
            "calibration_drift",
            "tail_state",
            "source_confidence",
        ],
        "outputs": [
            "prediction_interval",
            "abstention_gate",
            "uncertainty_bucket",
            "calibration_drift_alert",
        ],
        "peer_sleeves": [
            "uncertainty_robust_control",
            "model_risk_validation",
            "feature_quality_data_confidence",
            "tail_dependency_risk",
        ],
    },
    {
        "layer": 4,
        "slug": "execution_cost_alpha_decay_model",
        "display_name": "Execution-Cost Alpha Decay Model",
        "objective": "Convert theoretical alpha into expected net edge after spread, slippage, impact, latency, queue position, fill probability, and signal decay.",
        "signals": [
            "bid_ask_spread",
            "depth_and_queue_position",
            "realized_slippage",
            "latency_bucket",
            "signal_half_life",
        ],
        "outputs": [
            "net_alpha_after_cost",
            "max_trade_size_hint",
            "decay_clock",
            "venue_route_quality",
        ],
        "peer_sleeves": [
            "transaction_cost_slippage_intelligence",
            "execution_quality_lab_v2",
            "market_making_liquidity",
            "order_flow_market_microstructure",
            "liquidity_regime",
        ],
    },
    {
        "layer": 5,
        "slug": "cross_impact_crowded_trade_layer",
        "display_name": "Cross-Impact And Crowded Trade Layer",
        "objective": "Detect when sleeves, accounts, symbols, or assets are piling into the same exposure, liquidity pocket, or crowded institutional flow.",
        "signals": [
            "cross_sleeve_order_overlap",
            "factor_exposure_overlap",
            "options_gamma_open_interest",
            "etf_flow_pressure",
            "borrow_and_volume_stress",
        ],
        "outputs": [
            "crowding_score",
            "cross_impact_map",
            "concentration_alert",
            "deconflict_size_hint",
        ],
        "peer_sleeves": [
            "portfolio_construction",
            "liquidity_regime",
            "etf_flow_creation_redemption",
            "dealer_positioning_gamma_inventory",
            "repo_securities_lending",
        ],
    },
    {
        "layer": 6,
        "slug": "online_regime_changepoint_engine",
        "display_name": "Online Regime Change And Changepoint Engine",
        "objective": "Detect live regime breaks and feed model-weight, sleeve-pause, cadence, and retraining-priority decisions.",
        "signals": [
            "bayesian_changepoints",
            "volatility_shift",
            "liquidity_shift",
            "correlation_break",
            "macro_event_state",
        ],
        "outputs": [
            "regime_break_probability",
            "model_weight_shift",
            "sleeve_pause_hint",
            "retrain_priority",
        ],
        "peer_sleeves": [
            "causal_regime_discovery",
            "state_space_models",
            "liquidity_regime",
            "macro_crisis_scenario_lab",
        ],
    },
    {
        "layer": 7,
        "slug": "dealer_vol_control_systematic_flow_layer",
        "display_name": "Dealer, Vol-Control, And Systematic Flow Layer",
        "objective": "Estimate non-fundamental flows from dealer hedging, vol-control funds, CTA/risk-parity behavior, ETF flows, buybacks, and expiry mechanics.",
        "signals": [
            "dealer_gamma_charm_vanna",
            "vol_targeting_pressure",
            "trend_following_flow",
            "etf_creation_redemption",
            "opex_pinning_pressure",
        ],
        "outputs": [
            "systematic_flow_map",
            "dealer_hedge_pressure",
            "vol_control_flow_risk",
            "opex_pin_alert",
        ],
        "peer_sleeves": [
            "dealer_positioning_gamma_inventory",
            "etf_flow_creation_redemption",
            "gamma_scalping",
            "second_third_order_greeks",
            "event_intelligence",
        ],
    },
    {
        "layer": 8,
        "slug": "robust_portfolio_optimizer",
        "display_name": "Robust Portfolio Optimizer",
        "objective": "Promote candidate ideas through robust portfolio math that respects uncertainty, impact, turnover, drawdown, tail dependence, account rules, and tax constraints.",
        "signals": [
            "expected_return_uncertainty",
            "covariance_shrinkage",
            "tail_dependence",
            "drawdown_state",
            "account_policy_constraints",
        ],
        "outputs": [
            "robust_weight_target",
            "turnover_budget",
            "risk_contribution_map",
            "allocation_veto",
        ],
        "peer_sleeves": [
            "portfolio_construction",
            "tail_dependency_risk",
            "uncertainty_robust_control",
            "collateral_margin_liquidity",
            "model_risk_validation",
        ],
    },
    {
        "layer": 9,
        "slug": "corporate_action_special_situations_arb_layer",
        "display_name": "Corporate Action And Special Situations Arb Layer",
        "objective": "Track M&A, tenders, buybacks, spin-offs, dividends, index changes, earnings drift, and financing stress as event-relative-value inputs.",
        "signals": [
            "sec_filing_event",
            "deal_spread",
            "break_risk",
            "corporate_action_calendar",
            "index_and_borrow_pressure",
        ],
        "outputs": [
            "special_situation_packet",
            "event_probability_score",
            "deal_spread_decay",
            "corporate_action_alert",
        ],
        "peer_sleeves": [
            "event_intelligence",
            "tax_corporate_actions_intelligence",
            "dividend_income",
            "earnings_event",
            "sector_rotation",
        ],
    },
    {
        "layer": 10,
        "slug": "alpha_research_governance_layer",
        "display_name": "Alpha Research Governance Layer",
        "objective": "Force every new alpha through provenance, point-in-time safety, leakage tests, capacity, walk-forward evidence, source confidence, and retirement rules.",
        "signals": [
            "data_lineage",
            "leakage_test_result",
            "walk_forward_coverage",
            "source_confidence",
            "paper_live_gap",
        ],
        "outputs": [
            "research_thesis_card",
            "novelty_score",
            "promotion_evidence_gap",
            "retirement_candidate",
        ],
        "peer_sleeves": [
            "alpha_research_os",
            "research_meta_governance",
            "model_risk_validation",
            "signal_governance_integrity",
            "expansion_quality_governance",
        ],
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _inventory_sleeves(inventory: dict[str, Any]) -> set[str]:
    sleeves: set[str] = set()
    advanced = inventory.get("advanced_collection_sleeves")
    if isinstance(advanced, list):
        sleeves.update(_norm(item) for item in advanced if str(item or "").strip())
    detailed = inventory.get("sleeves")
    if isinstance(detailed, list):
        for sleeve in detailed:
            if isinstance(sleeve, dict):
                sleeves.add(_norm(sleeve.get("name")))
    return {item for item in sleeves if item}


def _activation_blockers(
    lane_upgrade: dict[str, Any],
    retrain_launch: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    gate_state = lane_upgrade.get("gate_state") if isinstance(lane_upgrade.get("gate_state"), dict) else {}
    gate_names = [
        "global_halt_clear",
        "storage_green",
        "runtime_green",
        "paper_400_ready",
        "promotion_quality_ready",
        "quality_gate_ok",
        "promotion_readiness_ok",
        "promotion_packet_ok",
    ]
    for gate_name in gate_names:
        if gate_name in gate_state and not bool(gate_state.get(gate_name)):
            blockers.append(f"{gate_name}=false")
    for item in gate_state.get("promotion_quality_failed_checks") or []:
        blockers.append(f"promotion_quality:{item}")
    for item in gate_state.get("promotion_readiness_blockers") or []:
        blockers.append(f"promotion_readiness:{item}")
    if str(retrain_launch.get("state") or "").strip().lower() == "running":
        blockers.append("large_training_batch_running_control_plane_only")
    return list(dict.fromkeys(str(item) for item in blockers if str(item).strip()))


def _layer_payload(layer: dict[str, Any], available_sleeves: set[str], paper_ready: bool) -> dict[str, Any]:
    peers = [str(peer) for peer in layer.get("peer_sleeves") or []]
    existing_peers = [peer for peer in peers if _norm(peer) in available_sleeves]
    missing_peers = [peer for peer in peers if _norm(peer) not in available_sleeves]
    return {
        **layer,
        "activation_state": "collection_only_advisory",
        "collection_enabled": True,
        "advisory_enabled": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "execution_enabled": False,
        "allocation_enabled": False,
        "training_intake_enabled": False,
        "paper_activation_ready": bool(paper_ready),
        "coverage_status": "mapped_to_existing_sleeves" if existing_peers else "manifest_ready_no_peer_match",
        "existing_peer_sleeves": existing_peers,
        "missing_peer_sleeves": missing_peers,
        "applies_to": [
            "equities",
            "options",
            "futures",
            "crypto",
            "income_positions",
            "multi_account_portfolio_context",
        ],
        "safety_gates": [
            "global_halt_clear",
            "storage_green",
            "runtime_green",
            "paper_400_ready",
            "promotion_quality_ready",
            "walk_forward_coverage_ready",
            "paper_live_execution_calibration_ready",
            "account_position_policy_clearance",
        ],
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    inventory = _load_json(project_root / "governance" / "health" / "strategy_inventory_latest.json")
    lane_upgrade = _load_json(project_root / "governance" / "health" / "quant_strategy_lane_upgrades_latest.json")
    retrain_launch = _load_json(project_root / "governance" / "health" / "retrain_launch_latest.json")
    available_sleeves = _inventory_sleeves(inventory)
    paper_ready = bool(lane_upgrade.get("paper_activation_ready"))
    layers = [_layer_payload(layer, available_sleeves, paper_ready) for layer in DEEP_QUANT_LAYERS]
    covered_layers = [
        layer["slug"]
        for layer in layers
        if str(layer.get("coverage_status") or "") == "mapped_to_existing_sleeves"
    ]
    blockers = _activation_blockers(lane_upgrade, retrain_launch)
    forbidden_enabled = [
        field
        for layer in layers
        for field in [
            "paper_trading_enabled",
            "live_trading_enabled",
            "execution_enabled",
            "allocation_enabled",
            "training_intake_enabled",
        ]
        if bool(layer.get(field))
    ]
    ok = len(layers) == 10 and not forbidden_enabled
    status = (
        "invalid_deep_quant_layer_contract"
        if not ok
        else "paper_activation_ready"
        if paper_ready and not blockers
        else "deep_quant_layers_installed_collection_only_activation_blocked"
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": status,
        "layer_count": len(layers),
        "covered_layer_count": len(covered_layers),
        "coverage_ratio": round(len(covered_layers) / max(len(layers), 1), 4),
        "collection_enabled_layer_count": sum(1 for layer in layers if bool(layer.get("collection_enabled"))),
        "advisory_enabled_layer_count": sum(1 for layer in layers if bool(layer.get("advisory_enabled"))),
        "paper_enabled_layer_count": sum(1 for layer in layers if bool(layer.get("paper_trading_enabled"))),
        "live_enabled_layer_count": sum(1 for layer in layers if bool(layer.get("live_trading_enabled"))),
        "execution_enabled_layer_count": sum(1 for layer in layers if bool(layer.get("execution_enabled"))),
        "training_intake_enabled_layer_count": sum(1 for layer in layers if bool(layer.get("training_intake_enabled"))),
        "available_sleeve_count": len(available_sleeves),
        "strategy_inventory_counts": {
            "bot_count": int(inventory.get("bot_count") or 0),
            "sleeve_count": int(inventory.get("sleeve_count") or 0),
            "strategy_count": int(inventory.get("strategy_count") or 0),
        },
        "quant_lane_upgrade_status": lane_upgrade.get("overall_status"),
        "quant_lane_paper_activation_ready": paper_ready,
        "training_batch_active": str(retrain_launch.get("state") or "").strip().lower() == "running",
        "training_batch_pid": retrain_launch.get("pid"),
        "activation_blockers": blockers,
        "forbidden_enabled": forbidden_enabled,
        "layers": layers,
        "recommended_actions": [
            "consume layer outputs as advisory evidence only",
            "do not enable paper, live, allocation, execution, or training intake for these layers until promotion gates clear",
            "use covered peer sleeves to backfill labels and point-in-time evidence for each layer",
            "rerun quant-strategy-lane-upgrades after current training finishes to reassess promotion readiness",
        ],
        "artifacts": {
            "json": str(OUT_PATH),
            "report": str(REPORT_PATH),
        },
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Deep Quant Layer Upgrade",
        "",
        f"- timestamp_utc: `{payload.get('timestamp_utc')}`",
        f"- status: `{payload.get('overall_status')}`",
        f"- layers: `{payload.get('layer_count')}`",
        f"- covered_layers: `{payload.get('covered_layer_count')}`",
        f"- paper_enabled_layers: `{payload.get('paper_enabled_layer_count')}`",
        f"- live_enabled_layers: `{payload.get('live_enabled_layer_count')}`",
        f"- training_batch_active: `{payload.get('training_batch_active')}`",
        "",
        "## Activation Blockers",
        "",
    ]
    blockers = payload.get("activation_blockers") if isinstance(payload.get("activation_blockers"), list) else []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Layers", ""])
    for layer in payload.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        peers = ", ".join(str(peer) for peer in layer.get("existing_peer_sleeves") or []) or "none"
        lines.extend(
            [
                f"### {layer.get('layer')}. {layer.get('display_name')}",
                "",
                f"- slug: `{layer.get('slug')}`",
                f"- state: `{layer.get('activation_state')}`",
                f"- existing_peer_sleeves: {peers}",
                f"- primary_outputs: {', '.join(str(item) for item in layer.get('outputs') or [])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the 10-layer deep quant advisory manifest.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload.")
    parser.add_argument("--no-write", action="store_true", help="Build the payload without writing artifacts.")
    args = parser.parse_args()

    payload = build_payload()
    if not args.no_write:
        _write_json(OUT_PATH, payload)
        _write_text(REPORT_PATH, render_report(payload))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "deep_quant_layer_upgrade "
            f"status={payload.get('overall_status')} "
            f"layers={payload.get('layer_count')} "
            f"covered={payload.get('covered_layer_count')} "
            f"paper_enabled={payload.get('paper_enabled_layer_count')} "
            f"live_enabled={payload.get('live_enabled_layer_count')}"
        )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
