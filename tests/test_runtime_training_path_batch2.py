import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v23_atr_adx_regime as v23
from core import brain_refinery_v39_event_risk_proximity as v39
from core import brain_refinery_v44_intraday_scalp_1m_5m as v44
from core import brain_refinery_v46_swing_2d_5d as v46
from core import brain_refinery_v51_vol_regime_transition as v51
from core import brain_refinery_v54_event_shock_decay as v54


def _obs(symbol="SPY", last_price=100.0, **overrides):
    base_features = {
        "pct_from_close": 0.0022,
        "mom_5m": 0.0012,
        "mom_15m": 0.0018,
        "vol_30m": 0.008,
        "range_pos": 0.70,
        "spread_bps": 8.0,
        "queue_depth": 3.0,
        "market_data_latency_ms": 35.0,
        "lag_expected_fill_delta_bps": 0.10,
        "lag_latency_ms": 20.0,
        "lag_slippage_bps": 0.08,
        "day_execution_cost_risk_norm": 0.18,
        "day_opening_auction_signal_norm": 0.38,
        "market_micro_opening_auction_signal_norm": 0.34,
        "market_micro_opening_auction_norm": 0.34,
        "day_session_open_norm": 0.36,
        "day_session_midday_norm": 0.18,
        "day_session_power_hour_norm": 0.32,
        "market_micro_relative_volume_norm": 0.64,
        "market_micro_trend_persistence_norm": 0.68,
        "market_micro_order_flow_imbalance_norm": 0.72,
        "market_micro_block_trade_norm": 0.42,
        "day_regime_trend_norm": 0.74,
        "day_regime_alignment_norm": 0.70,
        "lead_lag_alignment_norm": 0.64,
        "breadth_advance_decline_norm": 0.28,
        "breadth_sector_dispersion_norm": 0.20,
        "breadth_risk_off_norm": 0.18,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.06,
        "behavior_prior": 0.24,
        "futures_specialist_vote": 0.72,
        "swing_sector_relative_strength_norm": 0.74,
        "swing_weekly_trend_confirm_norm": 0.70,
        "swing_regime_trend_norm": 0.69,
        "swing_regime_alignment_norm": 0.66,
        "capital_flow_signed_scaled": 0.20,
        "flow_direction_signed": 0.22,
        "flow_risk_on_norm": 0.64,
        "ctx_VIX_X_pct_from_close": 0.002,
        "ctx_UUP_pct_from_close": -0.001,
        "options_chain_available": 1.0,
        "options_iv_atm_norm": 0.44,
        "options_iv_skew_norm": 0.52,
        "options_iv_term_structure_norm": 0.55,
        "options_iv_realized_spread_norm": 0.50,
        "options_vol_expectation_norm": 0.36,
        "options_negative_bias_norm": 0.18,
        "options_unusual_flow_norm": 0.22,
        "options_gamma_expiry_skew_norm": 0.54,
        "news_available": 1.0,
        "news_items_30m": 0.68,
        "news_items_2h": 0.62,
        "news_items_24h": 0.58,
        "news_sentiment": 0.62,
        "news_negative_share": 0.22,
        "news_positive_share": 0.64,
        "news_shock_rate": 0.66,
        "news_recent_impact": 0.70,
        "news_topic_earnings_norm": 0.40,
        "news_topic_guidance_norm": 0.34,
        "news_topic_mna_norm": 0.22,
        "news_topic_regulatory_norm": 0.18,
        "news_novelty_norm": 0.56,
        "news_duplicate_cluster_norm": 0.20,
        "calendar_event_proximity_norm": 0.72,
        "calendar_high_impact_24h_norm": 0.76,
        "calendar_macro_event_norm": 0.64,
        "calendar_macro_abs_surprise_norm": 0.46,
        "calendar_macro_revision_norm": 0.26,
        "calendar_fomc_event_norm": 0.38,
        "calendar_cpi_event_norm": 0.30,
        "calendar_labor_event_norm": 0.24,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_runtime_feature_vector_shapes_batch2() -> None:
    sequence = [_obs()]
    assert v23._runtime_feature_vector(sequence, 0).shape == (27,)
    assert v44._runtime_feature_vector(sequence, 0).shape == (28,)
    assert v46._runtime_feature_vector(sequence, 0).shape == (29,)
    assert v51._runtime_feature_vector(sequence, 0).shape == (26,)
    assert v39._runtime_feature_vector(sequence, 0).shape == (37,)
    assert v54._runtime_feature_vector(sequence, 0).shape == (31,)


def test_v23_runtime_label_returns_positive_for_supported_breakout() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.15),
        _obs(last_price=100.29),
        _obs(last_price=100.46),
        _obs(last_price=100.62),
        _obs(last_price=100.78),
        _obs(last_price=100.95),
    ]
    assert v23._runtime_atr_regime_label(sequence, 0, 6) == 1.0


def test_v44_runtime_label_returns_positive_for_supported_scalp() -> None:
    sequence = [
        _obs(symbol="NVDA", last_price=100.0),
        _obs(symbol="NVDA", last_price=100.10),
        _obs(symbol="NVDA", last_price=100.24),
        _obs(symbol="NVDA", last_price=100.39),
    ]
    assert v44._runtime_scalp_label(sequence, 0, 3) == 1.0


def test_v46_runtime_label_returns_positive_for_supported_swing() -> None:
    sequence = [
        _obs(symbol="SCHD", last_price=100.0),
        _obs(symbol="SCHD", last_price=100.12),
        _obs(symbol="SCHD", last_price=100.25),
        _obs(symbol="SCHD", last_price=100.40),
        _obs(symbol="SCHD", last_price=100.55),
        _obs(symbol="SCHD", last_price=100.71),
        _obs(symbol="SCHD", last_price=100.88),
        _obs(symbol="SCHD", last_price=101.04),
        _obs(symbol="SCHD", last_price=101.21),
    ]
    assert v46._runtime_swing_label(sequence, 0, 8) == 1.0


def test_v39_runtime_label_returns_positive_for_event_follow_through() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.16),
        _obs(last_price=100.30),
        _obs(last_price=100.48),
        _obs(last_price=100.64),
        _obs(last_price=100.81),
        _obs(last_price=100.98),
    ]
    assert v39._runtime_event_label(sequence, 0, 6) == 1.0


def test_v51_runtime_label_returns_positive_for_vol_easing() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.14),
        _obs(last_price=100.29),
        _obs(last_price=100.45),
        _obs(last_price=100.60),
        _obs(last_price=100.76),
        _obs(last_price=100.92),
    ]
    assert v51._runtime_vol_transition_label(sequence, 0, 6) == 1.0


def test_v54_runtime_label_returns_positive_for_shock_decay_rebound() -> None:
    sequence = [
        _obs(
            last_price=100.0,
            pct_from_close=-0.0028,
            mom_5m=-0.0018,
            news_sentiment=0.24,
            news_positive_share=0.58,
            news_negative_share=0.34,
        ),
        _obs(last_price=100.12),
        _obs(last_price=100.27),
        _obs(last_price=100.43),
        _obs(last_price=100.60),
    ]
    assert v54._runtime_decay_label(sequence, 0, 4) == 1.0


def test_train_brain_uses_runtime_path_without_silent_fallback_batch2(monkeypatch) -> None:
    modules = [
        (v23, "brain_refinery_v23_atr_adx_regime", "SPY", 0.02),
        (v44, "brain_refinery_v44_intraday_scalp_1m_5m", "NVDA", 0.015),
        (v46, "brain_refinery_v46_swing_2d_5d", "SCHD", 0.02),
        (v51, "brain_refinery_v51_vol_regime_transition", "SPY", 0.02),
        (v39, "brain_refinery_v39_event_risk_proximity", "SPY", 0.02),
        (v54, "brain_refinery_v54_event_shock_decay", "SPY", 0.02),
    ]
    for module, run_tag, expected_symbol, expected_lift in modules:
        captured = {}

        def _fake_train_runtime_indicator_bot(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(module, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

        result = module.train_brain()

        assert result == "ok"
        assert captured["run_tag"] == run_tag
        assert captured["allow_fallback_on_insufficient_data"] is False
        assert callable(captured["runtime_label_builder"])
        assert callable(captured["sample_filter"])
        assert captured["mode_allowlist"]
        assert expected_symbol in captured["symbol_allowlist"]
        if module is v44:
            assert "shadow_aggressive_equities" in captured["mode_allowlist"]
            assert "shadow_intraday_aggressive_equities" in captured["mode_allowlist"]
            assert captured["lookback_days"] == 45
            assert captured["min_confidence"] == 0.46
            assert "TQQQ" not in captured["symbol_allowlist"]
        assert captured["require_both_sides_precision"] is True
        assert captured["min_accuracy_lift_over_majority"] == expected_lift
