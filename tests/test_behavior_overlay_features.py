from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.build_behavior_dataset_from_decisions as behavior_ds
import scripts.run_shadow_training_loop as loop
import scripts.train_trade_behavior_bot as trainer


def test_behavior_feature_schema_appends_lane_overlay_features() -> None:
    assert behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES == loop._BEHAVIOR_LANE_FEATURE_NAMES
    assert behavior_ds.PAPER_CONTEXT_FEATURE_NAMES == loop._BEHAVIOR_PAPER_FEATURE_NAMES
    capital_tail = behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES
    paper_tail = behavior_ds.PAPER_CONTEXT_FEATURE_NAMES
    capital_tail_start = len(behavior_ds.FEATURE_NAMES) - len(capital_tail)
    paper_tail_start = capital_tail_start - len(paper_tail)
    lane_tail_start = paper_tail_start - len(behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES)
    assert behavior_ds.FEATURE_NAMES[lane_tail_start:paper_tail_start] == behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[lane_tail_start:paper_tail_start] == loop._BEHAVIOR_LANE_FEATURE_NAMES
    assert behavior_ds.FEATURE_NAMES[paper_tail_start:capital_tail_start] == paper_tail
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[paper_tail_start:capital_tail_start] == paper_tail
    assert behavior_ds.FEATURE_NAMES[-len(capital_tail) :] == capital_tail
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[-len(loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES) :] == loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES
    assert behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES == loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES


def test_load_paper_trade_context_joins_snapshot_and_symbol_history() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=2)
    ts = datetime.now(timezone.utc)
    by_snapshot, by_symbol = behavior_ds._load_paper_trade_context(
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "SPY",
                "action": "BUY",
                "fill_price": 100.10,
                "reference_price": 100.00,
                "mark_price": 100.40,
                "metadata": {"snapshot_id": "snap-1"},
            }
        ],
        since_utc=since_utc,
    )

    assert by_snapshot["snap-1"]["count"] == 1.0
    assert by_snapshot["snap-1"]["mean_slippage_bps"] > 0.0
    assert by_snapshot["snap-1"]["mean_return_proxy_bps"] > 0.0
    assert "SPY" in by_symbol


def test_behavior_feature_vector_v2_accepts_paper_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "BUY",
        {
            "pct_from_close": 0.003,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "paper_snapshot_trade_count_norm": 0.50,
            "paper_snapshot_slippage_bps_norm": 0.20,
            "paper_snapshot_return_proxy_signed_scaled": 0.35,
            "paper_recent_trade_count_norm": 0.75,
            "paper_recent_slippage_bps_norm": 0.10,
            "paper_recent_return_proxy_signed_scaled": -0.25,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("paper_snapshot_trade_count_norm")] == 0.50
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("paper_snapshot_return_proxy_signed_scaled")] == 0.35
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("paper_recent_return_proxy_signed_scaled")] == -0.25


def test_behavior_feature_vector_v2_accepts_tastytrade_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "BUY",
        {
            "pct_from_close": 0.002,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "tasty_iv_rank_norm": 0.61,
            "tasty_implied_volatility_index_norm": 0.57,
            "tasty_liquidity_rating_norm": 0.83,
            "tasty_expected_move_norm": 0.29,
            "tasty_beta_norm": 0.54,
            "tasty_watchlist_presence_norm": 1.0,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("tasty_iv_rank_norm")] == 0.61
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("tasty_liquidity_rating_norm")] == 0.83
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("tasty_watchlist_presence_norm")] == 1.0


def test_behavior_feature_vector_v2_accepts_crypto_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "BTC-USD",
        "BUY",
        {
            "pct_from_close": 0.004,
            "mom_5m": 0.002,
            "vol_30m": 0.009,
            "crypto_deribit_mark_iv_norm": 0.71,
            "crypto_hyperliquid_funding_norm": 0.57,
            "crypto_coinmetrics_tx_count_norm": 0.63,
            "crypto_coingecko_momentum_norm": 0.69,
            "crypto_cross_provider_price_agreement_norm": 0.93,
            "crypto_defillama_stablecoin_growth_norm": 0.59,
            "crypto_etherscan_gas_norm": 0.04,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("crypto_deribit_mark_iv_norm")] == 0.71
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("crypto_hyperliquid_funding_norm")] == 0.57
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("crypto_cross_provider_price_agreement_norm")] == 0.93


def test_behavior_feature_vector_v2_accepts_market_crypto_correlation_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "BTC-USD",
        "BUY",
        {
            "pct_from_close": 0.004,
            "mom_5m": 0.002,
            "vol_30m": 0.009,
            "market_crypto_risk_corr_norm": 0.58,
            "market_crypto_spy_corr_norm": 0.61,
            "market_crypto_qqq_corr_norm": 0.55,
            "market_crypto_tlt_corr_norm": 0.50,
            "market_crypto_uup_inverse_corr_norm": 0.54,
            "market_crypto_gold_corr_norm": 0.46,
            "market_crypto_current_alignment_norm": 0.24,
            "market_crypto_divergence_norm": 0.69,
            "market_crypto_corr_confidence_norm": 1.0,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_risk_corr_norm")] == 0.58
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_current_alignment_norm")] == 0.24
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_corr_confidence_norm")] == 1.0


def test_behavior_feature_schema_includes_market_crypto_correlation_features() -> None:
    keys = [
        "market_crypto_risk_corr_norm",
        "market_crypto_spy_corr_norm",
        "market_crypto_qqq_corr_norm",
        "market_crypto_tlt_corr_norm",
        "market_crypto_uup_inverse_corr_norm",
        "market_crypto_gold_corr_norm",
        "market_crypto_current_alignment_norm",
        "market_crypto_divergence_norm",
        "market_crypto_corr_confidence_norm",
    ]
    for key in keys:
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_behavior_feature_schema_includes_dividend_drip_features() -> None:
    keys = [
        "dividend_drip_active_norm",
        "dividend_drip_recent_reinvest_norm",
        "dividend_drip_cash_only_norm",
        "dividend_drip_share_credit_norm",
        "dividend_drip_event_recency_norm",
        "dividend_drip_confidence_norm",
    ]
    for key in keys:
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_behavior_regime_index_marks_dividend_defensive_context_mean_revert() -> None:
    _, regime = behavior_ds._regime_index(
        "SCHD",
        {
            "pct_from_close": 0.0004,
            "mom_5m": 0.0002,
            "vol_30m": 0.003,
            "dividend_yield_norm": 0.72,
            "dividend_quality_score_norm": 0.81,
            "dividend_drip_active_norm": 0.84,
        },
    )

    assert regime == "mean_revert"


def test_behavior_regime_index_marks_futures_event_risk_context_shock() -> None:
    _, regime = behavior_ds._regime_index(
        "ES=F",
        {
            "pct_from_close": 0.001,
            "mom_5m": 0.0005,
            "vol_30m": 0.004,
            "calendar_event_proximity_norm": 0.68,
            "futures_order_book_imbalance_norm": 0.77,
            "futures_term_structure_norm": 0.61,
        },
    )

    assert regime == "shock"


def test_behavior_feature_vector_v2_accepts_dividend_drip_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SCHD",
        "BUY",
        {
            "pct_from_close": 0.002,
            "mom_5m": 0.001,
            "vol_30m": 0.003,
            "dividend_drip_active_norm": 0.84,
            "dividend_drip_recent_reinvest_norm": 0.65,
            "dividend_drip_cash_only_norm": 0.18,
            "dividend_drip_share_credit_norm": 0.57,
            "dividend_drip_event_recency_norm": 0.92,
            "dividend_drip_confidence_norm": 0.88,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("dividend_drip_active_norm")] == 0.84
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("dividend_drip_recent_reinvest_norm")] == 0.65
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("dividend_drip_confidence_norm")] == 0.88


def test_load_governance_index_captures_lane_strategy_features() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "snap-1",
            "lane_strategy_features": {
                "day_regime_trend_norm": 0.81,
                "swing_regime_chop_norm": 0.24,
                "bond_curve_steepener_norm": 0.67,
            },
        }
    ]

    out = behavior_ds._load_governance_index(rows, since_utc=since_utc)

    assert out["snap-1"]["day_regime_trend_norm"] == 0.81
    assert out["snap-1"]["swing_regime_chop_norm"] == 0.24
    assert out["snap-1"]["bond_curve_steepener_norm"] == 0.67
    assert out["snap-1"]["day_execution_cost_risk_norm"] == 0.0


def test_load_governance_index_captures_capital_flow_features() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "snap-2",
            "capital_flow": {
                "capital_flow_signed_scaled": -0.72,
                "capital_flow_inflow_norm": 0.0,
                "capital_flow_outflow_norm": 0.61,
            },
        }
    ]

    out = behavior_ds._load_governance_index(rows, since_utc=since_utc)

    assert out["snap-2"]["capital_flow_signed_scaled"] == -0.72
    assert out["snap-2"]["capital_flow_inflow_norm"] == 0.0
    assert out["snap-2"]["capital_flow_outflow_norm"] == 0.61


def test_behavior_feature_vector_v2_accepts_appended_lane_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "NVDA",
        "BUY",
        {
            "pct_from_close": 0.01,
            "mom_5m": 0.004,
            "vol_30m": 0.008,
            "range_pos": 0.8,
            "spread_bps": 3.0,
            "day_regime_trend_norm": 0.83,
            "swing_regime_alignment_norm": 0.66,
            "bond_carry_roll_norm": 0.41,
        },
        {},
    )

    assert vec is not None
    assert vec.shape[1] == len(loop._BEHAVIOR_FEATURE_NAMES_V2)
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("day_regime_trend_norm")] == 0.83
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("swing_regime_alignment_norm")] == 0.66
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("bond_carry_roll_norm")] == 0.41


def test_behavior_feature_vector_v2_accepts_capital_flow_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "SELL",
        {
            "pct_from_close": -0.004,
            "mom_5m": -0.002,
            "vol_30m": 0.005,
            "capital_flow_signed_scaled": -0.88,
            "capital_flow_inflow_norm": 0.0,
            "capital_flow_outflow_norm": 0.74,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_signed_scaled")] == -0.88
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_inflow_norm")] == 0.0
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_outflow_norm")] == 0.74


def test_effective_account_equity_proxy_prefers_fresh_broker_truth() -> None:
    effective, meta = loop._effective_account_equity_proxy(
        {
            "status": "mismatch",
            "age_iters": 1,
            "account_metrics": {"equity": 152500.0, "cash_balance": 48000.0},
        },
        fallback_equity_proxy=100000.0,
    )

    assert effective == 152500.0
    assert meta["source"] == "broker_truth_account_metrics"


def test_estimate_capital_flow_state_detects_large_outflow() -> None:
    flow = loop._estimate_capital_flow_state(
        {"equity": 87000.0, "cash_balance": 22000.0},
        {"equity": 120000.0, "cash_balance": 55000.0},
    )

    assert flow["detected"] is True
    assert flow["estimated_amount"] < 0.0
    assert flow["capital_flow_signed_scaled"] < 0.0
    assert flow["capital_flow_outflow_norm"] > 0.0
    assert flow["capital_flow_inflow_norm"] == 0.0


def test_rollback_schema_compatible_allows_prefix_compatible_extension() -> None:
    ok, reason = trainer._rollback_schema_compatible(
        {
            "load_ok": True,
            "effective_dim": 3,
            "feature_names": ["a", "b", "c"],
        },
        dataset_feature_dim=5,
        dataset_feature_names=["a", "b", "c", "d", "e"],
        require_feature_names=True,
    )

    assert ok is True
    assert reason.startswith("prefix_compatible")


def test_curated_dataset_guard_accepts_curated_behavior_dataset() -> None:
    ok, reason, summary = trainer._curated_dataset_guard(
        {
            "dataset_kind": "curated_decision_governance",
            "source": {
                "decision_files": 3,
                "decision_sql_files": 2,
                "governance_files": 2,
                "governance_sql_files": 1,
                "pnl_attribution_files": 1,
                "pnl_sql_files": 1,
            },
        }
    )

    assert ok is True
    assert reason == "ok"
    assert summary["decision_sources"] == 5
    assert summary["governance_sources"] == 3


def test_curated_dataset_guard_rejects_legacy_dataset_kind() -> None:
    ok, reason, summary = trainer._curated_dataset_guard(
        {
            "dataset_kind": "legacy_trade_history",
            "source": {
                "decision_files": 3,
                "governance_files": 2,
            },
        }
    )

    assert ok is False
    assert reason == "dataset_kind_not_curated"
    assert summary["dataset_kind"] == "legacy_trade_history"
