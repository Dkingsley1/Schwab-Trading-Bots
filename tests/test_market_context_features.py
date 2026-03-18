from datetime import datetime, timedelta, timezone

from core.market_context_features import (
    summarize_bond_reference_context,
    summarize_breadth_context,
    summarize_credit_context,
    summarize_data_quality_context,
    summarize_structured_news_items,
)


def test_summarize_structured_news_items_emits_source_topics_and_session_flags() -> None:
    ts_now = datetime(2026, 3, 13, 13, 0, tzinfo=timezone.utc).timestamp()
    items = [
        {
            "headline": "Reuters: NVDA earnings beat and raises guidance",
            "publisher": "Reuters",
            "symbols": ["NVDA"],
            "publishedDate": datetime(2026, 3, 13, 12, 5, tzinfo=timezone.utc).isoformat(),
        },
        {
            "headline": "Reuters: NVDA earnings beat and raises guidance",
            "publisher": "Reuters",
            "symbols": ["NVDA"],
            "publishedDate": datetime(2026, 3, 13, 12, 15, tzinfo=timezone.utc).isoformat(),
        },
        {
            "headline": "Bloomberg: NVDA faces SEC investigation after after-hours filing",
            "publisher": "Bloomberg",
            "symbols": ["NVDA"],
            "publishedDate": datetime(2026, 3, 13, 21, 5, tzinfo=timezone.utc).isoformat(),
        },
    ]

    out = summarize_structured_news_items(items, symbol="NVDA", now_ts=ts_now, max_items=10)

    assert out["news_source_quality_norm"] > 0.7
    assert out["news_entity_relevance_norm"] > 0.8
    assert out["news_topic_earnings_norm"] > 0.0
    assert out["news_topic_guidance_norm"] > 0.0
    assert out["news_duplicate_cluster_norm"] > 0.0
    assert out["news_premarket_norm"] > 0.0


def test_summarize_breadth_context_proxy_emits_norms() -> None:
    out = summarize_breadth_context(
        symbol="SPY",
        market_snapshot={"pct_from_close": 0.009, "queue_depth": 1200.0},
        context_market={
            "QQQ": {"pct_from_close": 0.012, "queue_depth": 1400.0},
            "IWM": {"pct_from_close": -0.004, "queue_depth": 900.0},
            "XLF": {"pct_from_close": 0.006, "queue_depth": 800.0},
            "XLK": {"pct_from_close": 0.014, "queue_depth": 1500.0},
        },
        external_snapshot={},
    )

    assert 0.0 <= out["breadth_advance_decline_norm"] <= 1.0
    assert 0.0 <= out["breadth_up_down_volume_norm"] <= 1.0
    assert out["breadth_thrust_norm"] > 0.0


def test_summarize_bond_and_credit_context_use_snapshot_and_market_fallbacks() -> None:
    market_snapshot = {
        "bond_duration_years_norm": 0.65,
        "bond_convexity_norm": 0.44,
        "bond_nav_discount_norm": 0.53,
        "bond_etf_flow_5d_norm": 0.21,
        "bond_ytm_norm": 0.47,
    }
    context_market = {
        "TLT": {"pct_from_close": 0.011, "mom_5m": 0.002},
        "IEF": {"pct_from_close": 0.004, "mom_5m": 0.001},
        "SHY": {"pct_from_close": -0.001, "mom_5m": -0.0002},
        "TIP": {"pct_from_close": 0.005, "mom_5m": 0.0004},
        "HYG": {"pct_from_close": 0.009, "mom_5m": 0.0015},
        "LQD": {"pct_from_close": 0.003, "mom_5m": 0.0002},
    }
    external_snapshot = {
        "treasury_yields": {"2y": 4.1, "5y": 4.0, "10y": 4.2, "30y": 4.4, "real_10y": 1.9},
        "auction_tail_bps": 0.8,
        "symbols": {
            "TLT": {
                "duration_years_norm": 0.72,
                "convexity_norm": 0.61,
                "nav_discount_norm": 0.49,
                "flow_5d_norm": 0.18,
                "ytm_norm": 0.51,
            },
            "HYG": {
                "credit_spread_bps": 340.0,
                "credit_spread_change_bps": 24.0,
                "hy_ig_flow_ratio": 0.35,
                "nav_discount_pct": 0.014,
            },
        },
    }
    calendar_features = {"calendar_treasury_auction_norm": 0.6}

    bond_out = summarize_bond_reference_context(
        symbol="TLT",
        market_snapshot=market_snapshot,
        context_market=context_market,
        calendar_features=calendar_features,
        external_snapshot=external_snapshot,
    )
    credit_out = summarize_credit_context(
        symbol="HYG",
        market_snapshot={"bond_nav_discount_norm": 0.48},
        context_market=context_market,
        external_snapshot=external_snapshot,
    )

    assert bond_out["bond_yield_10y_norm"] > 0.0
    assert bond_out["bond_curve_2s10s_norm"] > 0.0
    assert bond_out["bond_duration_years_norm"] > 0.0
    assert bond_out["bond_auction_window_norm"] > 0.0
    assert credit_out["bond_credit_spread_level_norm"] > 0.0
    assert credit_out["bond_credit_spread_change_norm"] != 0.5
    assert credit_out["bond_hy_ig_flow_norm"] != 0.5


def test_summarize_data_quality_context_uses_quote_and_streaks() -> None:
    out = summarize_data_quality_context(
        market_snapshot={
            "quote_history_agreement_norm": 0.82,
            "quote_history_relative_deviation": 0.011,
            "bid_size": 900.0,
            "ask_size": 1500.0,
            "market_data_latency_ms": 180.0,
        },
        freshness_ok=False,
        freshness_age_seconds=240.0,
        symbol_fail_count=2,
        symbol_stale_count=3,
        symbol_circuit_hits=1,
        quarantine_seconds=120.0,
        missing_feature_count=1,
        required_feature_count=5,
    )

    assert out["data_quality_quote_agreement_norm"] > 0.0
    assert out["data_quality_quote_deviation_norm"] > 0.0
    assert out["data_quality_stale_streak_norm"] > 0.0
    assert out["data_quality_fail_streak_norm"] > 0.0
    assert out["data_quality_missing_feature_ratio_norm"] > 0.0
