import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v12_news_shocks as v12


def _obs(symbol="TSLA", last_price=100.0, **overrides):
    base_features = {
        "pct_from_close": 0.0042,
        "mom_5m": 0.0020,
        "vol_30m": 0.010,
        "range_pos": 0.74,
        "spread_bps": 11.0,
        "market_data_latency_ms": 36.0,
        "news_available": 1.0,
        "news_items_30m": 0.72,
        "news_items_2h": 0.66,
        "news_items_24h": 0.38,
        "news_sentiment": 0.58,
        "news_negative_share": 0.12,
        "news_positive_share": 0.70,
        "news_shock_rate": 0.64,
        "news_recent_impact": 0.60,
        "news_source_quality_norm": 0.88,
        "news_entity_relevance_norm": 0.84,
        "news_topic_earnings_norm": 0.62,
        "news_topic_guidance_norm": 0.54,
        "news_topic_mna_norm": 0.18,
        "news_topic_regulatory_norm": 0.12,
        "news_novelty_norm": 0.72,
        "news_duplicate_cluster_norm": 0.22,
        "news_premarket_norm": 0.10,
        "news_intraday_norm": 0.76,
        "news_after_hours_norm": 0.08,
        "calendar_event_proximity_norm": 0.52,
        "calendar_high_impact_24h_norm": 0.60,
        "calendar_macro_event_norm": 0.40,
        "calendar_macro_surprise_norm": 0.10,
        "calendar_macro_abs_surprise_norm": 0.18,
        "calendar_macro_revision_norm": 0.08,
        "calendar_fomc_event_norm": 0.14,
        "calendar_cpi_event_norm": 0.10,
        "calendar_labor_event_norm": 0.08,
        "calendar_treasury_auction_norm": 0.06,
        "ctx_VIX_X_pct_from_close": 0.010,
        "ctx_UUP_pct_from_close": -0.002,
        "breadth_advance_decline_norm": 0.18,
        "breadth_risk_off_norm": 0.24,
        "options_iv_atm_norm": 0.54,
        "options_iv_skew_norm": 0.42,
        "options_vol_expectation_norm": 0.60,
        "options_unusual_flow_norm": 0.56,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.06,
        "data_quality_stale_streak_norm": 0.12,
        "data_quality_market_data_latency_norm": 0.10,
        "market_micro_opening_auction_norm": 0.20,
        "market_micro_closing_auction_norm": 0.18,
        "market_micro_relative_volume_norm": 0.72,
        "market_micro_order_flow_imbalance_norm": 0.78,
        "market_micro_options_flow_norm": 0.62,
        "market_micro_short_pressure_norm": 0.12,
        "market_micro_credit_flow_norm": 0.10,
        "market_micro_block_trade_norm": 0.30,
        "options_specialist_vote": 0.56,
        "futures_specialist_vote": 0.34,
        "behavior_prior": 0.28,
        "active_sub_bots": 4.0,
        "active_futures_sub_bots": 1.0,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_news_shocks_sample_filter_accepts_high_quality_event() -> None:
    sequence = [_obs()]
    assert v12._runtime_sample_filter(sequence, 0, 6) is True


def test_news_shocks_label_returns_positive_for_supported_upside_event() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.28),
        _obs(last_price=100.56),
        _obs(last_price=100.88),
        _obs(last_price=101.22),
        _obs(last_price=101.54),
        _obs(last_price=101.88),
    ]
    assert v12._runtime_shock_label(sequence, 0, 6) == 1.0


def test_news_shocks_label_returns_negative_for_supported_downside_event() -> None:
    sequence = [
        _obs(
            symbol="SOXS",
            last_price=100.0,
            pct_from_close=-0.0042,
            mom_5m=-0.0022,
            range_pos=0.26,
            news_sentiment=-0.62,
            news_negative_share=0.72,
            news_positive_share=0.10,
            news_topic_guidance_norm=0.22,
            news_topic_regulatory_norm=0.46,
            ctx_UUP_pct_from_close=0.003,
            breadth_advance_decline_norm=-0.18,
            breadth_risk_off_norm=0.76,
            market_micro_order_flow_imbalance_norm=0.18,
            market_micro_short_pressure_norm=0.52,
            market_micro_credit_flow_norm=0.48,
            options_specialist_vote=-0.48,
            futures_specialist_vote=-0.32,
            behavior_prior=-0.30,
        ),
        _obs(symbol="SOXS", last_price=99.72),
        _obs(symbol="SOXS", last_price=99.44),
        _obs(symbol="SOXS", last_price=99.16),
        _obs(symbol="SOXS", last_price=98.88),
        _obs(symbol="SOXS", last_price=98.60),
        _obs(symbol="SOXS", last_price=98.34),
    ]
    assert v12._runtime_shock_label(sequence, 0, 6) == 0.0


def test_train_brain_uses_scoped_runtime_path_without_fallback(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v12, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    result = v12.train_brain()

    assert result == "ok"
    assert captured["run_tag"] == "brain_refinery_v12_news_shocks"
    assert captured["allow_fallback_on_insufficient_data"] is False
    assert captured["mode_allowlist"] == v12._NEWS_RUNTIME_MODES
    assert "TSLA" in captured["symbol_allowlist"]
    assert "TLT" in captured["symbol_allowlist"]
    assert captured["runtime_label_builder"] is v12._runtime_shock_label
    assert captured["sample_filter"] is v12._runtime_sample_filter
    assert captured["confidence_builder"] is v12._runtime_confidence
    assert captured["require_both_sides_precision"] is True
    assert captured["min_accuracy_lift_over_majority"] == 0.01
