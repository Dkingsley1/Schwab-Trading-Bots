import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

import indicator_bot_common as ibc
from core import brain_refinery_v15_liquidity_droughts as v15
from core import brain_refinery_v13_choppy as v13
from core import brain_refinery_v4_simple as v4
from core import brain_refinery_v35_dmi_state_machine as v35
from core import brain_refinery_v36_volume_profile_proxy as v36
from core import brain_refinery_v43_intraday_ultrafast_proxy as v43
from core import brain_refinery_v48_position_1m_3m as v48
from core import brain_refinery_v51_vol_regime_transition as v51
from core import brain_refinery_v56_meta_ranker as v56
from core import brain_refinery_v75_model_drift_guard as v75
from core import brain_refinery_v98_crypto_execution_throttle_reentry as v98
from core import brain_refinery_v100_stock_crypto_overlap_context as v100


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
        "data_quality_market_data_latency_norm": 0.12,
        "lag_expected_fill_delta_bps": 0.08,
        "lag_latency_ms": 18.0,
        "lag_slippage_bps": 0.07,
        "market_micro_relative_volume_norm": 0.66,
        "market_micro_trend_persistence_norm": 0.68,
        "market_micro_order_flow_imbalance_norm": 0.72,
        "market_micro_options_flow_norm": 0.34,
        "market_micro_short_pressure_norm": 0.18,
        "market_micro_credit_flow_norm": 0.20,
        "market_micro_block_trade_norm": 0.22,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.05,
        "data_quality_missing_feature_ratio_norm": 0.10,
        "behavior_prior": 0.26,
        "futures_specialist_vote": 0.72,
        "master_vote": 0.30,
        "grand_master_vote": 0.34,
        "options_master_vote": 0.18,
        "futures_master_vote": 0.16,
        "options_specialist_vote": 0.24,
        "day_regime_trend_norm": 0.74,
        "day_regime_chop_norm": 0.24,
        "day_regime_alignment_norm": 0.70,
        "day_session_open_norm": 0.36,
        "day_session_midday_norm": 0.20,
        "day_session_power_hour_norm": 0.34,
        "breadth_advance_decline_norm": 0.28,
        "breadth_risk_off_norm": 0.18,
        "breadth_sector_dispersion_norm": 0.20,
        "ctx_VIX_X_pct_from_close": 0.002,
        "options_chain_available": 1.0,
        "options_iv_skew_norm": 0.52,
        "options_iv_term_structure_norm": 0.55,
        "options_vol_expectation_norm": 0.36,
        "options_negative_bias_norm": 0.18,
        "options_unusual_flow_norm": 0.22,
        "swing_sector_relative_strength_norm": 0.74,
        "swing_weekly_trend_confirm_norm": 0.70,
        "swing_regime_trend_norm": 0.69,
        "swing_regime_alignment_norm": 0.66,
        "capital_flow_signed_scaled": 0.20,
        "flow_direction_signed": 0.22,
        "flow_risk_on_norm": 0.64,
        "lead_lag_alignment_norm": 0.64,
        "news_recent_impact": 0.34,
        "calendar_macro_abs_surprise_norm": 0.26,
        "calendar_treasury_auction_norm": 0.12,
        "active_sub_bots": 12.0,
        "active_options_sub_bots": 4.0,
        "active_futures_sub_bots": 3.0,
        "news_sentiment": 0.18,
        "news_source_quality_norm": 0.72,
        "news_entity_relevance_norm": 0.70,
        "calendar_event_proximity_norm": 0.28,
        "calendar_high_impact_24h_norm": 0.24,
        "fx_usd_strength_norm": 0.42,
        "fx_eurusd_momentum_norm": 0.46,
        "fx_usdjpy_momentum_norm": 0.54,
        "fx_proxy_agreement_norm": 0.52,
        "fx_risk_on_alignment_norm": 0.58,
        "fx_crypto_alignment_norm": 0.58,
        "fx_corr_confidence_norm": 0.44,
        "bond_curve_2s10s_norm": 0.46,
        "bond_real_yield_10y_norm": 0.48,
        "bond_credit_risk_on_norm": 0.52,
        "bond_credit_risk_off_norm": 0.28,
        "bond_hy_ig_flow_norm": 0.54,
        "bond_auction_window_norm": 0.20,
        "bond_auction_tail_norm": 0.10,
        "infra_risk_throttle_norm": 0.26,
        "infra_veto_active": 0.0,
        "market_micro_opening_auction_norm": 0.34,
        "market_micro_closing_auction_norm": 0.28,
        "market_micro_options_flow_norm": 0.34,
        "market_micro_short_pressure_norm": 0.18,
        "market_micro_credit_flow_norm": 0.20,
        "market_micro_block_trade_norm": 0.22,
        "lag_adjusted_return_1m": 0.0012,
        "crypto_cross_provider_price_agreement_norm": 0.76,
        "crypto_deribit_mark_iv_norm": 0.50,
        "crypto_deribit_basis_norm": 0.58,
        "crypto_hyperliquid_open_interest_norm": 0.62,
        "crypto_hyperliquid_funding_norm": 0.54,
        "crypto_hyperliquid_basis_norm": 0.57,
        "crypto_coingecko_momentum_norm": 0.60,
        "crypto_defillama_dex_volume_growth_norm": 0.48,
        "crypto_defillama_stablecoin_growth_norm": 0.44,
        "crypto_etherscan_gas_norm": 0.46,
        "futures_order_book_imbalance_norm": 0.78,
        "futures_taker_imbalance_norm": 0.74,
        "futures_basis_bps_norm": 0.61,
        "futures_basis_divergence_norm": 0.38,
        "futures_term_structure_norm": 0.58,
        "futures_session_volume_profile_norm": 0.76,
        "futures_negative_bias_norm": 0.24,
        "futures_liquidation_risk_norm": 0.16,
        "calendar_macro_event_norm": 0.62,
        "calendar_fomc_event_norm": 0.36,
        "calendar_cpi_event_norm": 0.30,
        "calendar_labor_event_norm": 0.22,
        "market_crypto_risk_corr_norm": 0.60,
        "market_crypto_spy_corr_norm": 0.62,
        "market_crypto_qqq_corr_norm": 0.61,
        "market_crypto_tlt_corr_norm": 0.44,
        "market_crypto_uup_inverse_corr_norm": 0.38,
        "market_crypto_gold_corr_norm": 0.32,
        "market_crypto_current_alignment_norm": 0.60,
        "market_crypto_divergence_norm": 0.24,
        "market_crypto_corr_confidence_norm": 0.42,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_calibrated_threshold_prefers_balanced_sides() -> None:
    threshold, meta = ibc._select_calibrated_acted_threshold(
        pred_probs_np=[0.62, 0.61, 0.60, 0.58, 0.42, 0.40, 0.39, 0.38],
        y_true_np=[1, 1, 1, 1, 0, 0, 0, 0],
        default_threshold=0.68,
    )

    assert threshold < 0.68
    assert meta["validation_metrics"]["long_acted_count"] >= 2
    assert meta["validation_metrics"]["short_acted_count"] >= 2


def test_action_threshold_calibration_can_choose_asymmetric_cutoffs() -> None:
    long_threshold, short_threshold, meta = ibc._select_calibrated_action_thresholds(
        pred_probs_np=[0.57, 0.56, 0.55, 0.54, 0.46, 0.45, 0.44, 0.43],
        y_true_np=[1, 1, 1, 1, 0, 0, 0, 0],
        default_threshold=0.68,
        min_long_acted_count=2,
        min_short_acted_count=2,
    )

    assert long_threshold < 0.68
    assert short_threshold <= 0.46
    assert meta["validation_metrics"]["long_acted_count"] >= 2
    assert meta["validation_metrics"]["short_acted_count"] >= 2


def test_action_threshold_calibration_prefers_coverage_bound_when_available() -> None:
    long_threshold, short_threshold, meta = ibc._select_calibrated_action_thresholds(
        pred_probs_np=[0.99, 0.97, 0.95, 0.93, 0.88, 0.12, 0.10, 0.08, 0.06, 0.04],
        y_true_np=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        default_threshold=0.60,
        min_long_acted_count=2,
        min_short_acted_count=2,
        max_acted_coverage=0.60,
        min_long_precision=0.50,
        min_short_precision=0.50,
    )

    assert long_threshold >= 0.88
    assert short_threshold <= 0.12
    assert meta["validation_metrics"]["acted_coverage"] <= 0.60


def test_v15_train_brain_exposes_runtime_path(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v15, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v15.train_brain() == "ok"
    assert captured["min_confidence"] == 0.32
    assert captured["acted_prob_threshold"] == 0.60
    assert captured["allow_fallback_on_insufficient_data"] is False


def test_v36_train_brain_exposes_runtime_path(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v36, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v36.train_brain() == "ok"
    assert captured["min_confidence"] == 0.36
    assert captured["acted_prob_threshold"] == 0.60
    assert captured["allow_fallback_on_insufficient_data"] is False


def test_v13_train_brain_uses_reduced_runtime_floor(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v13, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v13.train_brain() == "ok"
    assert captured["lookback_days"] == 90
    assert captured["min_samples"] == 56
    assert captured["min_positive_samples"] == 12
    assert captured["min_negative_samples"] == 12
    assert captured["min_confidence"] == 0.36
    assert captured["sample_stride"] == 1
    assert "SPY" in captured["symbol_allowlist"]
    assert "META" not in captured["symbol_allowlist"]
    assert "AAPL" not in captured["symbol_allowlist"]
    assert "shadow_dividend_equities" in captured["mode_allowlist"]
    assert "shadow_intraday_aggressive_equities" in captured["mode_allowlist"]
    assert captured["acted_prob_threshold"] == 0.64
    assert captured["max_acted_coverage"] == 0.28


def test_v4_train_brain_uses_cleaner_runtime_floor(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v4, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v4.train_brain() == "ok"
    assert "SCHD" not in captured["symbol_allowlist"]
    assert "NVDA" in captured["symbol_allowlist"]
    assert "AMZN" in captured["symbol_allowlist"]
    assert "shadow_dividend_equities" in captured["mode_allowlist"]
    assert "shadow_swing_aggressive_equities" in captured["mode_allowlist"]
    assert captured["lookback_days"] == 60
    assert captured["min_samples"] == 192
    assert captured["min_positive_samples"] == 32
    assert captured["min_negative_samples"] == 32
    assert captured["min_confidence"] == 0.40
    assert captured["sample_stride"] == 2
    assert captured["acted_prob_threshold"] == 0.68
    assert captured["min_long_acted_count"] == 5
    assert captured["min_short_acted_count"] == 5
    assert captured["max_acted_coverage"] == 0.34


def test_v100_train_brain_uses_broader_recovery_floor(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v100, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v100.train_brain() == "ok"
    assert captured["lookback_days"] == 90
    assert captured["sample_stride"] == 1
    assert captured["min_confidence"] == 0.30
    assert captured["min_samples"] == 96
    assert captured["min_positive_samples"] == 16
    assert captured["min_negative_samples"] == 16
    assert captured["acted_prob_threshold"] == 0.66


def test_v43_train_brain_exposes_runtime_path(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v43, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v43.train_brain() == "ok"
    assert captured["lookback_days"] == 60
    assert captured["min_confidence"] == 0.60
    assert captured["sample_stride"] == 4
    assert captured["symbol_allowlist"] == ["BTC-USD", "ETH-USD"]
    assert captured["min_long_acted_count"] == 0
    assert captured["min_short_acted_count"] == 4
    assert captured["acted_prob_threshold"] == 0.68
    assert captured["require_both_sides_precision"] is False
    assert captured["min_long_precision"] == 0.0
    assert captured["max_acted_coverage"] == 0.26
    assert captured["allow_fallback_on_insufficient_data"] is False


def test_v56_train_brain_exposes_runtime_path(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v56, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v56.train_brain() == "ok"
    assert captured["lookback_days"] == 60
    assert captured["min_confidence"] == 0.44
    assert captured["sample_stride"] == 2
    assert "shadow_crypto" not in captured["mode_allowlist"]
    assert "SPY" in captured["symbol_allowlist"]
    assert captured["min_samples"] == 1200
    assert captured["min_positive_samples"] == 96
    assert captured["min_negative_samples"] == 96
    assert captured["acted_prob_threshold"] == 0.62
    assert captured["min_long_acted_count"] == 6
    assert captured["min_short_acted_count"] == 6
    assert "role_crypto" in captured["feature_names"]
    assert captured["max_acted_coverage"] == 0.22
    assert captured["allow_fallback_on_insufficient_data"] is False


def test_v35_train_brain_uses_broader_runtime_floor(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v35, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v35.train_brain() == "ok"
    assert "shadow_dividend_equities" not in captured["mode_allowlist"]
    assert "shadow_crypto" not in captured["mode_allowlist"]
    assert "shadow_crypto_futures_crypto" not in captured["mode_allowlist"]
    assert "NVDA" in captured["symbol_allowlist"]
    assert "BTC-USD" not in captured["symbol_allowlist"]
    assert "AMZN" in captured["symbol_allowlist"]
    assert "DOGE-USD" not in captured["symbol_allowlist"]
    assert "XLY" not in captured["symbol_allowlist"]
    assert captured["min_confidence"] == 0.46
    assert captured["sample_stride"] == 4
    assert captured["batch_size"] == 16
    assert captured["min_samples"] == 80
    assert captured["min_positive_samples"] == 20
    assert captured["min_negative_samples"] == 20
    assert captured["acted_prob_threshold"] == 0.70
    assert captured["max_acted_coverage"] == 0.30


def test_v48_train_brain_exposes_balanced_runtime_path(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v48, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v48.train_brain() == "ok"
    assert captured["lookback_days"] == 30
    assert captured["min_confidence"] == 0.34
    assert captured["min_samples"] == 192
    assert captured["min_positive_samples"] == 32
    assert captured["min_negative_samples"] == 32
    assert captured["acted_prob_threshold"] == 0.64
    assert captured["allow_fallback_on_insufficient_data"] is False
    assert captured["require_both_sides_precision"] is True


def test_v75_train_brain_exposes_balanced_runtime_guards(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v75, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    assert v75.train_brain() == "ok"
    assert captured["sample_stride"] == 4
    assert captured["lookback_days"] == 60
    assert captured["min_confidence"] == 0.50
    assert "shadow_crypto" not in captured["mode_allowlist"]
    assert "shadow_crypto_futures_crypto" not in captured["mode_allowlist"]
    assert captured["min_samples"] == 224
    assert captured["acted_prob_threshold"] == 0.66
    assert captured["min_long_precision"] == 0.54
    assert captured["min_short_precision"] == 0.54
    assert captured["require_both_sides_precision"] is True
    assert captured["min_long_acted_count"] == 5
    assert captured["min_short_acted_count"] == 5
    assert captured["max_acted_coverage"] == 0.22


def test_v75_runtime_label_skips_unconfirmed_short_regime() -> None:
    sequence = [
        _obs(
            last_price=100.0,
            behavior_prior=0.26,
            master_vote=0.72,
            grand_master_vote=0.70,
            options_master_vote=0.66,
            futures_master_vote=0.64,
            options_specialist_vote=0.68,
            futures_specialist_vote=0.70,
            market_micro_order_flow_imbalance_norm=0.68,
            news_recent_impact=0.56,
            breadth_risk_off_norm=0.30,
        ),
        _obs(last_price=99.93),
        _obs(last_price=99.86),
        _obs(last_price=99.79),
        _obs(last_price=99.72),
        _obs(last_price=99.65),
        _obs(last_price=99.58),
    ]

    assert v75._runtime_sample_filter(sequence, 0, 6) is True
    assert v75._runtime_drift_label(sequence, 0, 6) is None


def test_v75_runtime_sample_filter_rejects_high_friction_drift_setup() -> None:
    sequence = [
        _obs(
            behavior_prior=-0.28,
            master_vote=0.18,
            grand_master_vote=0.20,
            options_master_vote=0.22,
            futures_master_vote=0.18,
            options_specialist_vote=0.24,
            futures_specialist_vote=0.20,
            data_quality_quote_agreement_norm=0.82,
            data_quality_quote_deviation_norm=0.20,
            market_micro_relative_volume_norm=0.42,
            lag_expected_fill_delta_bps=9.4,
            lag_slippage_bps=8.8,
            news_recent_impact=0.64,
            breadth_risk_off_norm=0.56,
        )
    ]

    assert v75._drift_signal(sequence[0]) >= 0.16
    assert v75._directional_drift_conviction(sequence[0]) >= 0.24
    assert v75._drift_tradeability(sequence[0]) < 0.44
    assert v75._runtime_sample_filter(sequence, 0, 6) is False


def test_v75_runtime_label_emits_short_for_confirmed_downside() -> None:
    sequence = [
        _obs(
            last_price=100.0,
            behavior_prior=-0.32,
            master_vote=0.18,
            grand_master_vote=0.20,
            options_master_vote=0.24,
            futures_master_vote=0.18,
            options_specialist_vote=0.26,
            futures_specialist_vote=0.22,
            market_micro_order_flow_imbalance_norm=0.18,
            news_recent_impact=0.62,
            breadth_risk_off_norm=0.58,
        ),
        _obs(last_price=99.89),
        _obs(last_price=99.78),
        _obs(last_price=99.66),
        _obs(last_price=99.54),
        _obs(last_price=99.43),
        _obs(last_price=99.31),
    ]

    assert v75._runtime_sample_filter(sequence, 0, 6) is True
    assert v75._runtime_drift_label(sequence, 0, 6) == 0.0


def test_v35_runtime_sample_filter_rejects_conflicted_trend_setup() -> None:
    sequence = [
        _obs(
            symbol="SPY",
            spread_bps=12.0,
            queue_depth=3.0,
            data_quality_quote_agreement_norm=0.95,
            data_quality_quote_deviation_norm=0.06,
            day_regime_trend_norm=0.76,
            day_regime_alignment_norm=0.74,
            market_micro_trend_persistence_norm=0.72,
            market_micro_relative_volume_norm=0.70,
            behavior_prior=0.70,
            market_micro_order_flow_imbalance_norm=0.22,
            futures_specialist_vote=0.22,
            breadth_advance_decline_norm=-0.25,
            breadth_sector_dispersion_norm=0.30,
            breadth_risk_off_norm=0.24,
            infra_risk_throttle_norm=0.24,
            mom_15m=0.0050,
            mom_5m=0.0050,
            pct_from_close=0.0065,
        )
    ]

    assert abs(v35._direction_bias(sequence[0])) >= 0.18
    assert v35._trend_directional_agreement(sequence[0]) < 0.56
    assert v35._runtime_sample_filter(sequence, 0, 6) is False


def test_v35_runtime_label_abstains_on_unstable_trend_state() -> None:
    sequence = [
        _obs(
            symbol="SPY",
            last_price=100.0,
            spread_bps=22.0,
            queue_depth=3.0,
            data_quality_quote_agreement_norm=0.95,
            data_quality_quote_deviation_norm=0.12,
            day_regime_trend_norm=0.78,
            day_regime_alignment_norm=0.76,
            market_micro_trend_persistence_norm=0.74,
            market_micro_relative_volume_norm=0.70,
            breadth_sector_dispersion_norm=0.72,
            breadth_risk_off_norm=0.58,
            infra_risk_throttle_norm=0.92,
            day_session_midday_norm=1.0,
        ),
        _obs(last_price=100.16),
        _obs(last_price=100.32),
        _obs(last_price=100.48),
        _obs(last_price=100.65),
        _obs(last_price=100.82),
        _obs(last_price=100.99),
    ]

    assert v35._trend_directional_agreement(sequence[0], v35._direction_bias(sequence[0])) >= 0.56
    assert v35._trend_instability(sequence[0]) > 0.66
    assert v35._runtime_trend_label(sequence, 0, 6) is None


def test_v35_runtime_label_emits_short_for_bearish_headwind_setup() -> None:
    sequence = [
        _obs(
            symbol="SPY",
            last_price=100.0,
            spread_bps=14.0,
            queue_depth=3.0,
            data_quality_quote_agreement_norm=0.94,
            data_quality_quote_deviation_norm=0.08,
            day_regime_trend_norm=0.78,
            day_regime_alignment_norm=0.38,
            market_micro_trend_persistence_norm=0.76,
            market_micro_relative_volume_norm=0.74,
            behavior_prior=-0.30,
            market_micro_order_flow_imbalance_norm=0.14,
            futures_specialist_vote=0.16,
            breadth_advance_decline_norm=-0.56,
            breadth_sector_dispersion_norm=0.52,
            breadth_risk_off_norm=0.72,
            infra_risk_throttle_norm=0.60,
            day_session_open_norm=0.62,
            day_session_power_hour_norm=0.46,
            market_crypto_current_alignment_norm=0.34,
            market_crypto_divergence_norm=0.64,
            crypto_coingecko_momentum_norm=0.32,
            fx_crypto_alignment_norm=0.36,
            mom_15m=-0.0032,
            mom_5m=-0.0026,
            pct_from_close=-0.0042,
        ),
        _obs(last_price=99.86),
        _obs(last_price=99.72),
        _obs(last_price=99.56),
        _obs(last_price=99.40),
        _obs(last_price=99.22),
        _obs(last_price=99.04),
    ]

    assert v35._runtime_sample_filter(sequence, 0, 6) is True
    assert v35._runtime_trend_label(sequence, 0, 6) == 0.0


def test_v35_crypto_sample_filter_accepts_moderate_crypto_setup() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            spread_bps=36.0,
            queue_depth=0.0,
            data_quality_quote_agreement_norm=0.82,
            data_quality_quote_deviation_norm=0.18,
            market_crypto_current_alignment_norm=0.72,
            market_crypto_divergence_norm=0.20,
            crypto_coingecko_momentum_norm=0.68,
            fx_crypto_alignment_norm=0.66,
            behavior_prior=0.20,
            futures_specialist_vote=0.18,
            market_micro_relative_volume_norm=0.74,
            market_micro_trend_persistence_norm=0.80,
        )
    ]

    assert v35._runtime_sample_filter(sequence, 0, 6) is True


def test_v35_runtime_label_returns_positive_for_crypto_trend_follow_through() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            last_price=100.0,
            spread_bps=36.0,
            queue_depth=0.0,
            data_quality_quote_agreement_norm=0.79,
            data_quality_quote_deviation_norm=0.18,
            behavior_prior=0.22,
            futures_specialist_vote=0.20,
            market_crypto_current_alignment_norm=0.76,
            market_crypto_divergence_norm=0.14,
            crypto_coingecko_momentum_norm=0.72,
            fx_crypto_alignment_norm=0.68,
            market_micro_relative_volume_norm=0.74,
            market_micro_trend_persistence_norm=0.78,
        ),
        _obs(symbol="BTC-USD", last_price=100.20),
        _obs(symbol="BTC-USD", last_price=100.42),
        _obs(symbol="BTC-USD", last_price=100.60),
        _obs(symbol="BTC-USD", last_price=100.88),
        _obs(symbol="BTC-USD", last_price=101.05),
        _obs(symbol="BTC-USD", last_price=101.22),
    ]

    assert v35._runtime_trend_label(sequence, 0, 6) == 1.0


def test_v15_runtime_label_can_emit_long_and_short() -> None:
    long_sequence = [
        _obs(
            last_price=100.0,
            spread_bps=26.0,
            queue_depth=0.5,
            behavior_prior=0.22,
            market_micro_order_flow_imbalance_norm=0.80,
            market_micro_gap_continuation_norm=0.64,
            market_micro_reversal_risk_norm=0.20,
            futures_specialist_vote=0.22,
            options_specialist_vote=0.12,
        ),
        _obs(last_price=100.24),
    ]
    short_sequence = [
        _obs(
            last_price=100.0,
            spread_bps=29.0,
            queue_depth=0.4,
            behavior_prior=-0.24,
            market_micro_order_flow_imbalance_norm=0.18,
            market_micro_gap_continuation_norm=0.58,
            market_micro_reversal_risk_norm=0.14,
            futures_specialist_vote=-0.18,
            options_specialist_vote=-0.10,
        ),
        _obs(last_price=99.70),
    ]

    assert v15._runtime_liquidity_label(long_sequence, 0, 1) == 1.0
    assert v15._runtime_liquidity_label(short_sequence, 0, 1) == 0.0


def test_v36_runtime_label_can_emit_long_and_short() -> None:
    long_sequence = [
        _obs(
            last_price=100.0,
            range_pos=0.82,
            market_micro_order_flow_imbalance_norm=0.84,
            market_micro_gap_continuation_norm=0.70,
            market_micro_reversal_risk_norm=0.16,
            futures_specialist_vote=0.20,
            options_specialist_vote=0.10,
            market_micro_trend_persistence_norm=0.72,
        ),
        _obs(last_price=100.20),
    ]
    short_sequence = [
        _obs(
            last_price=100.0,
            range_pos=0.18,
            market_micro_order_flow_imbalance_norm=0.16,
            market_micro_gap_continuation_norm=0.66,
            market_micro_reversal_risk_norm=0.12,
            futures_specialist_vote=-0.18,
            options_specialist_vote=-0.12,
            market_micro_trend_persistence_norm=0.68,
        ),
        _obs(last_price=99.76),
    ]

    assert v36._runtime_profile_label(long_sequence, 0, 1) == 1.0
    assert v36._runtime_profile_label(short_sequence, 0, 1) == 0.0


def test_v56_runtime_label_returns_negative_for_crypto_throttle_setup() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            last_price=100.0,
            behavior_prior=-0.34,
            master_vote=-0.32,
            grand_master_vote=-0.29,
            market_crypto_current_alignment_norm=0.72,
            market_crypto_divergence_norm=0.58,
            market_crypto_corr_confidence_norm=0.66,
            crypto_coingecko_momentum_norm=0.62,
            crypto_hyperliquid_funding_norm=0.60,
            infra_risk_throttle_norm=0.82,
            infra_veto_active=1.0,
            lag_adjusted_return_1m=-0.0006,
        ),
        _obs(symbol="BTC-USD", last_price=99.98),
        _obs(symbol="BTC-USD", last_price=99.97),
        _obs(symbol="BTC-USD", last_price=99.95),
        _obs(symbol="BTC-USD", last_price=99.92),
        _obs(symbol="BTC-USD", last_price=99.90),
        _obs(symbol="BTC-USD", last_price=99.88),
    ]

    assert v56._runtime_meta_label(sequence, 0, 6) == 0.0


def test_v56_runtime_label_can_return_positive_for_crypto_follow_through() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            last_price=100.0,
            behavior_prior=0.36,
            master_vote=0.34,
            grand_master_vote=0.32,
            options_master_vote=0.18,
            futures_master_vote=0.16,
            market_crypto_current_alignment_norm=0.82,
            market_crypto_divergence_norm=0.10,
            market_crypto_corr_confidence_norm=0.78,
            crypto_coingecko_momentum_norm=0.76,
            crypto_hyperliquid_funding_norm=0.68,
            infra_risk_throttle_norm=0.24,
            infra_veto_active=0.0,
            lag_adjusted_return_1m=0.0016,
            breadth_risk_off_norm=0.14,
        ),
        _obs(symbol="BTC-USD", last_price=100.18),
        _obs(symbol="BTC-USD", last_price=100.38),
        _obs(symbol="BTC-USD", last_price=100.56),
        _obs(symbol="BTC-USD", last_price=100.80),
        _obs(symbol="BTC-USD", last_price=101.02),
        _obs(symbol="BTC-USD", last_price=101.24),
    ]

    assert v56._runtime_meta_label(sequence, 0, 6) == 1.0


def test_v56_runtime_label_returns_short_for_negative_vote_follow_through() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            last_price=100.0,
            behavior_prior=-0.34,
            master_vote=-0.30,
            grand_master_vote=-0.28,
            options_master_vote=-0.16,
            futures_master_vote=-0.18,
            market_crypto_current_alignment_norm=0.78,
            market_crypto_divergence_norm=0.14,
            market_crypto_corr_confidence_norm=0.72,
            crypto_coingecko_momentum_norm=0.70,
            crypto_hyperliquid_funding_norm=0.66,
            infra_risk_throttle_norm=0.22,
            infra_veto_active=0.0,
            lag_adjusted_return_1m=-0.0015,
            breadth_risk_off_norm=0.18,
        ),
        _obs(symbol="BTC-USD", last_price=99.82),
        _obs(symbol="BTC-USD", last_price=99.60),
        _obs(symbol="BTC-USD", last_price=99.44),
        _obs(symbol="BTC-USD", last_price=99.20),
        _obs(symbol="BTC-USD", last_price=99.04),
        _obs(symbol="BTC-USD", last_price=98.90),
    ]

    assert v56._runtime_meta_label(sequence, 0, 6) == 0.0


def test_v56_runtime_label_skips_misaligned_vote_follow_through() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            last_price=100.0,
            behavior_prior=0.34,
            master_vote=0.30,
            grand_master_vote=0.28,
            options_master_vote=0.16,
            futures_master_vote=0.18,
            market_crypto_current_alignment_norm=0.78,
            market_crypto_divergence_norm=0.12,
            market_crypto_corr_confidence_norm=0.74,
            crypto_coingecko_momentum_norm=0.70,
            crypto_hyperliquid_funding_norm=0.66,
            infra_risk_throttle_norm=0.20,
            infra_veto_active=0.0,
            lag_adjusted_return_1m=-0.0014,
            breadth_risk_off_norm=0.16,
        ),
        _obs(symbol="BTC-USD", last_price=99.82),
        _obs(symbol="BTC-USD", last_price=99.64),
        _obs(symbol="BTC-USD", last_price=99.42),
        _obs(symbol="BTC-USD", last_price=99.20),
        _obs(symbol="BTC-USD", last_price=99.02),
        _obs(symbol="BTC-USD", last_price=98.86),
    ]

    assert v56._runtime_meta_label(sequence, 0, 6) is None


def test_v51_runtime_label_returns_negative_for_supported_stress() -> None:
    sequence = [
        _obs(
            last_price=100.0,
            behavior_prior=-0.28,
            mom_5m=-0.0014,
            pct_from_close=-0.0022,
            breadth_advance_decline_norm=0.14,
            breadth_risk_off_norm=0.72,
            ctx_VIX_X_pct_from_close=0.024,
            options_negative_bias_norm=0.76,
            options_vol_expectation_norm=0.78,
            options_iv_term_structure_norm=0.86,
            options_iv_skew_norm=0.82,
        ),
        _obs(last_price=99.82),
        _obs(last_price=99.60),
        _obs(last_price=99.36),
        _obs(last_price=99.14),
        _obs(last_price=98.94),
        _obs(last_price=98.74),
    ]
    assert v51._runtime_vol_transition_label(sequence, 0, 6) == 0.0


def test_v98_sample_filter_accepts_moderate_reentry_setup() -> None:
    sequence = [
        _obs(
            symbol="BTC-USD",
            crypto_cross_provider_price_agreement_norm=0.73,
            crypto_deribit_mark_iv_norm=0.42,
            crypto_hyperliquid_open_interest_norm=0.50,
            infra_risk_throttle_norm=0.34,
            behavior_prior=0.18,
            mom_5m=0.0007,
        )
    ]
    assert v98._runtime_sample_filter(sequence, 0, 3) is True


def test_v100_sample_filter_rejects_weak_overlap_gap() -> None:
    sequence = [
        _obs(
            market_crypto_spy_corr_norm=0.51,
            market_crypto_qqq_corr_norm=0.51,
            market_crypto_risk_corr_norm=0.50,
            fx_crypto_alignment_norm=0.50,
            market_crypto_tlt_corr_norm=0.56,
            market_crypto_gold_corr_norm=0.55,
            market_crypto_uup_inverse_corr_norm=0.54,
            market_crypto_corr_confidence_norm=0.30,
            fx_corr_confidence_norm=0.30,
        )
    ]
    assert v100._runtime_sample_filter(sequence, 0, 4) is False
