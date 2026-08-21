from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.base_trader import BaseTrader
from core.profitability_hardening import (
    coalesce_paper_intents,
    evaluate_paper_execution_authority,
    evaluate_profitability_entry,
    evaluate_retirement_evidence,
    position_valuation_compatible,
    post_cost_adjusted_forward_return,
    resolve_contract_valuation,
)
from scripts.ops.profitability_hardening_control import build_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_contract_valuation_resolves_known_derivatives_and_rejects_unknown_future() -> None:
    future = resolve_contract_valuation("/ES")
    option = resolve_contract_valuation("AAPL260918C00200000")
    unknown = resolve_contract_valuation("/UNKNOWN")

    assert future["valuation_ready"] is True
    assert future["contract_multiplier"] == 50.0
    assert option["asset_type"] == "OPTION"
    assert option["contract_multiplier"] == 100.0
    assert unknown["valuation_ready"] is False
    assert unknown["contract_multiplier"] == 0.0


def test_legacy_derivative_position_must_reconcile_before_new_exposure() -> None:
    valuation = resolve_contract_valuation("/NQ")

    ok, reason = position_valuation_compatible({"qty": 1.0, "avg_price": 22000.0}, valuation)

    assert ok is False
    assert reason == "legacy_derivative_position_requires_reconciliation"


def test_paper_book_values_futures_pnl_in_contract_dollars() -> None:
    trader = object.__new__(BaseTrader)
    positions: dict = {}

    opened = trader._update_paper_book(
        positions=positions,
        realized_total=0.0,
        symbol_key="/ES",
        signed_qty=1.0,
        fill_price=5000.0,
        mark_price=5001.0,
        contract_multiplier=50.0,
    )
    closed = trader._update_paper_book(
        positions=positions,
        realized_total=0.0,
        symbol_key="/ES",
        signed_qty=-1.0,
        fill_price=5002.0,
        mark_price=5002.0,
        contract_multiplier=50.0,
    )

    assert opened["unrealized_pnl_symbol"] == 50.0
    assert closed["realized_pnl_delta"] == 100.0
    assert closed["contract_multiplier"] == 50.0


def test_entry_policy_blocks_explicit_overlap_and_selects_passive_execution() -> None:
    policy = evaluate_profitability_entry(
        profile="swing_aggressive",
        features={
            "market_micro_tradeability_score_norm": 0.72,
            "execution_fitness_norm": 0.62,
            "core_cross_asset_confirmation_norm": 0.70,
            "day_regime_trend_norm": 0.72,
            "day_regime_chop_norm": 0.20,
            "core_portfolio_overlap_pressure_norm": 0.86,
            "spread_bps": 18.0,
        },
    )

    assert policy["allowed"] is False
    assert any("portfolio_overlap" in reason for reason in policy["blockers"])
    assert policy["execution_plan"]["style"] == "passive_limit"
    assert policy["execution_plan"]["market_orders_allowed"] is False


def test_consensus_coalesces_bot_votes_and_abstains_on_conflict() -> None:
    aligned = coalesce_paper_intents(
        [
            {"bot_id": "a", "action": "BUY", "score": 0.70, "threshold": 0.55, "weight": 0.4, "test_accuracy": 0.7, "features": {}},
            {"bot_id": "b", "action": "BUY", "score": 0.66, "threshold": 0.55, "weight": 0.3, "test_accuracy": 0.65, "features": {}},
            {"bot_id": "c", "action": "SELL", "score": 0.35, "threshold": 0.55, "weight": 0.05, "test_accuracy": 0.55, "features": {}},
        ]
    )
    conflict = coalesce_paper_intents(
        [
            {"bot_id": "a", "action": "BUY", "score": 0.70, "threshold": 0.55, "weight": 0.3, "test_accuracy": 0.6, "features": {}},
            {"bot_id": "b", "action": "SELL", "score": 0.30, "threshold": 0.55, "weight": 0.3, "test_accuracy": 0.6, "features": {}},
        ]
    )

    assert aligned["action"] == "BUY"
    assert aligned["constituent_count"] == 3
    assert len(aligned["constituent_attribution"]) == 2
    assert 0.0 < aligned["quantity_multiplier"] <= 1.0
    assert conflict["action"] == "HOLD"
    assert conflict["reason"] == "portfolio_consensus_abstention"


def test_paper_execution_authority_fails_closed_and_rejects_control_identity() -> None:
    legacy = evaluate_paper_execution_authority(
        {
            "bot_id": "signal_legacy",
            "bot_role": "signal_sub_bot",
            "active": True,
            "test_accuracy": 0.70,
            "quality_score": 0.80,
            "paper_live_data_enabled": True,
        }
    )
    control = evaluate_paper_execution_authority(
        {
            "bot_id": "risk_allocator_signal",
            "bot_role": "signal_sub_bot",
            "active": True,
            "test_accuracy": 0.70,
            "quality_score": 0.80,
            "paper_execution_authority": True,
        }
    )

    assert legacy["allowed"] is False
    assert "explicit_paper_execution_authority_missing" in legacy["reasons"]
    assert control["allowed"] is False
    assert any(reason.startswith("control_identity_token:allocator") for reason in control["reasons"])


def test_paper_probation_requalification_is_explicit_and_paper_only() -> None:
    verdict = evaluate_paper_execution_authority(
        {
            "bot_id": "candidate_signal",
            "bot_role": "signal_sub_bot",
            "active": True,
            "lifecycle_state": "paper_live_data",
            "training_excluded": True,
            "test_accuracy": 0.61,
            "quality_score": 0.72,
            "paper_probation_authority": True,
            "paper_probation_requalification_allowed": True,
        }
    )

    assert verdict["allowed"] is True
    assert verdict["tier"] == "probation"
    assert verdict["live_execution_authority"] is False


def test_consensus_deduplicates_and_caps_correlated_signals() -> None:
    result = coalesce_paper_intents(
        [
            {
                "bot_id": "a",
                "action": "BUY",
                "score": 0.70,
                "threshold": 0.55,
                "weight": 0.8,
                "test_accuracy": 0.7,
                "correlation_cluster_id": "trend_cluster",
                "sub_sleeve_id": "trend_fast",
                "signal_fingerprint": "same_signal",
                "features": {},
            },
            {
                "bot_id": "b",
                "action": "BUY",
                "score": 0.69,
                "threshold": 0.55,
                "weight": 0.7,
                "test_accuracy": 0.7,
                "correlation_cluster_id": "trend_cluster",
                "sub_sleeve_id": "trend_fast",
                "signal_fingerprint": "same_signal",
                "features": {},
            },
            {
                "bot_id": "c",
                "action": "BUY",
                "score": 0.68,
                "threshold": 0.55,
                "weight": 0.4,
                "test_accuracy": 0.7,
                "correlation_cluster_id": "breadth_cluster",
                "sub_sleeve_id": "breadth",
                "features": {},
            },
        ],
        max_bot_weight=1.0,
        minimum_distinct_clusters=2,
    )

    assert result["action"] == "BUY"
    assert result["duplicate_signal_count"] == 1
    assert result["correlation_weight_capped"] is True
    assert result["distinct_correlation_clusters"] == 2


def test_runtime_consensus_fails_closed_without_hierarchy_identity() -> None:
    result = coalesce_paper_intents(
        [
            {
                "bot_id": "unmapped",
                "action": "BUY",
                "score": 0.75,
                "threshold": 0.55,
                "weight": 1.0,
                "test_accuracy": 0.70,
                "features": {},
            }
        ],
        require_hierarchy_identity=True,
    )

    assert result["action"] == "HOLD"
    assert result["skipped_reasons"]["hierarchy_identity_missing"] == 1
    assert result["hierarchy_identity_required"] is True


def test_consensus_allows_capped_candidate_bound_buy_scale() -> None:
    result = coalesce_paper_intents(
        [
            {
                "bot_id": bot_id,
                "action": "BUY",
                "score": 0.75,
                "threshold": 0.55,
                "weight": 0.5,
                "test_accuracy": 0.80,
                "correlation_cluster_id": cluster,
                "sub_sleeve_id": cluster,
                "sleeve_id": "default",
                "features": {
                    "paper_profitability_strategy_size_multiplier_norm": 1.10,
                    "profitability_regime_fit_norm": 1.0,
                    "execution_fitness_norm": 1.0,
                },
            }
            for bot_id, cluster in (("winner_a", "trend"), ("winner_b", "breadth"))
        ],
        max_bot_weight=1.0,
        minimum_distinct_clusters=2,
        require_hierarchy_identity=True,
    )

    assert result["action"] == "BUY"
    assert result["quantity_multiplier"] == 1.10


def test_consensus_never_shrinks_sell_exit_from_entry_scaling() -> None:
    result = coalesce_paper_intents(
        [
            {
                "bot_id": bot_id,
                "action": "SELL",
                "score": 0.25,
                "threshold": 0.55,
                "weight": 0.5,
                "test_accuracy": 0.80,
                "correlation_cluster_id": cluster,
                "sub_sleeve_id": cluster,
                "sleeve_id": "default",
                "features": {
                    "paper_profitability_strategy_size_multiplier_norm": 0.0,
                    "profitability_regime_fit_norm": 1.0,
                    "execution_fitness_norm": 1.0,
                },
            }
            for bot_id, cluster in (("exit_a", "trend"), ("exit_b", "breadth"))
        ],
        max_bot_weight=1.0,
        minimum_distinct_clusters=2,
        require_hierarchy_identity=True,
    )

    assert result["action"] == "SELL"
    assert result["quantity_multiplier"] == 1.0


def test_strict_entry_economics_block_stale_quotes_and_edge_below_costs() -> None:
    verdict = evaluate_profitability_entry(
        profile="intraday_aggressive",
        features={
            "profitability_strict_evidence_required": True,
            "market_micro_tradeability_score_norm": 0.80,
            "execution_fitness_norm": 0.80,
            "liquidity_quality_norm": 0.80,
            "session_quality_norm": 0.80,
            "spread_bps": 5.0,
            "quote_age_ms": 9000.0,
            "predicted_edge_lower_confidence_bound_bps": 12.0,
            "expected_round_trip_cost_bps": 7.0,
            "core_cross_asset_confirmation_norm": 0.70,
            "day_regime_trend_norm": 0.75,
            "day_regime_chop_norm": 0.10,
        },
    )

    assert verdict["allowed"] is False
    assert any(reason.startswith("quote_age_ms=") for reason in verdict["blockers"])
    assert any(reason.startswith("edge_cost_margin=") for reason in verdict["blockers"])


def test_paper_turnover_guard_blocks_churn_but_not_reductions(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MAX_NEW_ENTRIES_PER_SYMBOL_DAY", "2")
    monkeypatch.setenv("PAPER_NEW_ENTRY_COOLDOWN_SECONDS", "300")
    trader = object.__new__(BaseTrader)
    trader._paper_trade_activity = {
        "default|spy": {
            "day_utc": "2026-08-16",
            "entries_today": 2,
            "last_entry_utc": "2026-08-16T14:59:00+00:00",
            "last_entry_action": "BUY",
        }
    }
    blocked, reason, _ = trader._paper_turnover_new_entry_blocked(
        exposure={"increases_exposure": True, "crosses_through_flat": False},
        profile="default",
        symbol="SPY",
        action="BUY",
        now_utc=datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc),
    )
    reduction_blocked, reduction_reason, _ = trader._paper_turnover_new_entry_blocked(
        exposure={"increases_exposure": False, "crosses_through_flat": False},
        profile="default",
        symbol="SPY",
        action="SELL",
        now_utc=datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc),
    )

    assert blocked is True
    assert reason == "paper_symbol_daily_entry_cap_block"
    assert reduction_blocked is False
    assert reduction_reason == "reduce_close_or_hold"


def test_post_cost_labels_penalize_both_trade_directions() -> None:
    buy = post_cost_adjusted_forward_return(
        action="BUY", forward_return=0.0010, entry_cost_bps=2.0, exit_cost_bps=3.0
    )
    sell = post_cost_adjusted_forward_return(
        action="SELL", forward_return=-0.0010, entry_cost_bps=2.0, exit_cost_bps=3.0
    )

    assert buy["post_cost_forward_return"] == 0.0005
    assert sell["post_cost_forward_return"] == -0.0005
    assert buy["round_trip_cost_bps"] == 5.0


def test_retirement_requires_repeated_negative_post_cost_evidence() -> None:
    weak = evaluate_retirement_evidence(
        {
            "post_cost_samples": 140,
            "observed_days": 14,
            "failed_retests": 4,
            "post_cost_expectancy": -0.02,
            "post_cost_lower_confidence_bound": -0.05,
        }
    )
    thin = evaluate_retirement_evidence(
        {
            "post_cost_samples": 20,
            "observed_days": 3,
            "failed_retests": 4,
            "post_cost_expectancy": -0.02,
            "post_cost_lower_confidence_bound": -0.05,
        }
    )

    assert weak["retire"] is True
    assert thin["retire"] is False


def test_hardening_report_materializes_consensus_and_post_cost_evidence(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "config" / "trade_learning_policy.json",
        {"behavior_forward_labels": {"post_cost_labels": {"enabled": True}}},
    )
    _write_json(
        project_root / "data" / "trade_history" / "trade_learning_dataset.json",
        {"label_contract": {"post_cost_labels_enabled": True}},
    )
    timestamp = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    paper_path = project_root / "exports" / "trade_logs" / "paper" / "paper_trades_test.jsonl"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "decision_id": "decision-1",
                "symbol": "/ES",
                "action": "BUY",
                "quantity": 1,
                "strategy": "paper_portfolio_consensus::default::futures",
                "post_cost_pnl_delta": 4.0,
                "contract_multiplier": 50.0,
                "paper_valuation_asset_type": "FUTURE",
                "paper_valuation_ready": True,
                "paper_valuation_multiplier_source": "curated_contract_spec",
                "metadata": {
                    "source_profile": "default",
                    "layer": "paper_portfolio_consensus",
                    "execution_style": "passive_limit",
                    "risk_multiplier_norm": 0.7,
                    "entry_policy": {"overlap_pressure_norm": 0.2},
                    "constituent_attribution": [
                        {"bot_id": "bot_a", "weight_share": 1.0, "action": "BUY"}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_payload(project_root, lookback_days=2)

    assert payload["overall_status"] == "ready"
    assert payload["portfolio_consensus"]["consensus_execution_rows"] == 1
    assert payload["derivative_valuation"]["unknown_multiplier_rows"] == 0
    assert payload["post_cost_training"]["dataset_materialized"] is True
    assert payload["retirement_court"]["bot_evidence"][0]["bot_id"] == "bot_a"
