import pytest

import scripts.run_shadow_training_loop as loop


@pytest.fixture(autouse=True)
def _isolate_live_profitability_controls(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_profile_profitability_control", lambda profile: {})
    monkeypatch.setattr(loop, "_profile_profit_harvest_control", lambda profile: {})
    monkeypatch.setattr(loop, "_strategy_profit_harvest_control", lambda profile, strategy: {})
    monkeypatch.setattr(loop, "_profit_harvest_report_card_snapshot", lambda: {})
    monkeypatch.setattr(loop, "_profit_harvest_aplus_campaign_snapshot", lambda: {})
    monkeypatch.setattr(loop, "_profit_realization_contract_snapshot", lambda: {})
    monkeypatch.setattr(loop, "_profit_rotation_contract_snapshot", lambda: {})
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {})
    monkeypatch.setattr(loop, "_raw_profitability_a_recovery_contract", lambda: {})
    monkeypatch.setattr(loop, "_paper_debt_recovery_contract", lambda: {})
    monkeypatch.setattr(loop, "_profile_profitability_scaling_control", lambda profile: {})
    monkeypatch.setattr(loop, "_strategy_profitability_scaling_control", lambda profile, strategy: {})
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})


def test_coinbase_paper_mirror_env_prefers_coinbase_specific_allowlist(monkeypatch) -> None:
    sentinel = "__paper_profile_disabled_by_profitability_quarantine__"
    monkeypatch.setenv("TOP_BOT_PAPER_TRADING_PROFILES", sentinel)
    monkeypatch.setenv("TOP_BOT_PAPER_TRADING_TOP_N", "700")
    monkeypatch.setenv("COINBASE_TOP_BOT_PAPER_TRADING_PROFILES", "default")
    monkeypatch.setenv("COINBASE_TOP_BOT_PAPER_TRADING_TOP_N", "50")
    monkeypatch.setenv("COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES", "crypto_futures")
    monkeypatch.setenv("COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N", "30")

    assert loop._paper_mirror_env_value_for_broker("coinbase", "default", "PROFILES") == "default"
    assert loop._paper_mirror_env_value_for_broker("coinbase", "default", "TOP_N") == "50"
    assert loop._paper_mirror_env_value_for_broker("coinbase", "crypto_futures", "PROFILES") == "crypto_futures"
    assert loop._paper_mirror_env_value_for_broker("coinbase", "crypto_futures", "TOP_N") == "30"
    assert loop._paper_mirror_env_value_for_broker("schwab", "default", "PROFILES") == sentinel


def test_advanced_sleeve_logic_enabled_defaults_true_and_can_disable(monkeypatch) -> None:
    assert loop._advanced_sleeve_logic_enabled("default") is True
    assert loop._advanced_sleeve_logic_enabled("intraday_aggressive") is True

    monkeypatch.setenv("ENABLE_INTRADAY_AGGRESSIVE_ADVANCED_OVERLAYS", "0")
    assert loop._advanced_sleeve_logic_enabled("intraday_aggressive") is False
    assert loop._advanced_sleeve_logic_enabled("swing_aggressive") is True


def test_trend_chop_regime_metrics_distinguish_trend_from_chop() -> None:
    trend, chop, alignment = loop._trend_chop_regime_metrics(
        {
            "pct_from_close": 0.012,
            "mom_5m": 0.006,
            "vol_30m": 0.010,
            "range_pos": 0.92,
            "spread_bps": 2.0,
            "ctx_SPY_pct_from_close": 0.008,
            "ctx_QQQ_pct_from_close": 0.009,
            "ctx_IWM_pct_from_close": 0.007,
        }
    )

    assert trend > 0.75
    assert chop < 0.35
    assert alignment >= 0.99

    trend2, chop2, alignment2 = loop._trend_chop_regime_metrics(
        {
            "pct_from_close": 0.0003,
            "mom_5m": 0.0001,
            "vol_30m": 0.004,
            "range_pos": 0.50,
            "spread_bps": 14.0,
            "ctx_SPY_pct_from_close": 0.0002,
            "ctx_QQQ_pct_from_close": -0.0002,
            "ctx_IWM_pct_from_close": 0.0001,
        }
    )

    assert trend2 < 0.45
    assert chop2 > 0.60
    assert alignment2 == 0.5


def test_day_overlay_promotes_hold_when_regime_trend_is_strong(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "intraday_aggressive")
    monkeypatch.setattr(loop, "_session_phase_norms", lambda: (0.0, 0.0, 0.0))

    action, score, reasons, out_features = loop._apply_day_strategy_overlay(
        symbol="NVDA",
        action="HOLD",
        score=0.51,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "pct_from_close": 0.011,
            "mom_5m": 0.0055,
            "vol_30m": 0.008,
            "range_pos": 0.91,
            "spread_bps": 2.0,
            "bid_size": 1400.0,
            "ask_size": 700.0,
            "ctx_SPY_pct_from_close": 0.007,
            "ctx_QQQ_pct_from_close": 0.008,
            "ctx_IWM_pct_from_close": 0.006,
            "execution_fitness_norm": 0.72,
            "market_micro_tradeability_score_norm": 0.79,
            "market_micro_relative_volume_norm": 0.76,
            "market_micro_trend_persistence_norm": 0.74,
        },
        state={},
    )

    assert action == "BUY"
    assert score >= 0.55
    assert any("day_regime_trend_bias" in reason for reason in reasons)
    assert out_features["day_regime_trend_norm"] >= 0.72
    assert out_features["day_regime_chop_norm"] < 0.50


def test_day_overlay_blocks_directional_trade_in_chop(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "intraday_aggressive")
    monkeypatch.setattr(loop, "_session_phase_norms", lambda: (0.0, 0.8, 0.0))
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, out_features = loop._apply_day_strategy_overlay(
        symbol="AAPL",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "pct_from_close": 0.0003,
            "mom_5m": 0.0001,
            "vol_30m": 0.004,
            "range_pos": 0.50,
            "spread_bps": 12.0,
            "bid_size": 600.0,
            "ask_size": 610.0,
            "ctx_SPY_pct_from_close": 0.0002,
            "ctx_QQQ_pct_from_close": -0.0002,
            "ctx_IWM_pct_from_close": 0.0001,
            "execution_fitness_norm": 0.64,
            "market_micro_tradeability_score_norm": 0.61,
            "market_micro_lunch_chop_norm": 0.72,
        },
        state={},
    )

    assert action == "HOLD"
    assert score < 0.60
    assert any(
        marker in reason
        for reason in reasons
        for marker in ("day_regime_chop_guard", "intraday_allowlist_score")
    )
    assert out_features["day_regime_chop_norm"] >= 0.68


def test_day_overlay_blocks_failed_breakout_risk_near_open(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "intraday_aggressive")
    monkeypatch.setattr(loop, "_session_phase_norms", lambda: (0.85, 0.0, 0.0))
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, out_features = loop._apply_day_strategy_overlay(
        symbol="AMD",
        action="BUY",
        score=0.67,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "pct_from_close": 0.010,
            "mom_5m": 0.003,
            "vol_30m": 0.010,
            "range_pos": 0.88,
            "spread_bps": 3.0,
            "bid_size": 900.0,
            "ask_size": 500.0,
            "ctx_SPY_pct_from_close": 0.006,
            "ctx_QQQ_pct_from_close": 0.007,
            "ctx_IWM_pct_from_close": 0.005,
            "execution_fitness_norm": 0.74,
            "market_micro_tradeability_score_norm": 0.77,
            "market_micro_relative_volume_norm": 0.80,
            "market_micro_trend_persistence_norm": 0.72,
            "market_micro_opening_drive_pressure_norm": 0.78,
            "market_micro_quote_fade_rate_norm": 1.00,
            "market_micro_queue_depth_decay_norm": 1.00,
            "market_micro_reversal_risk_norm": 1.00,
        },
        state={},
    )

    assert action == "HOLD"
    assert score < 0.60
    assert any("day_failed_breakout_risk" in reason for reason in reasons)
    assert out_features["day_failed_breakout_risk_norm"] >= 0.70


def test_swing_overlay_promotes_hold_when_regime_and_weekly_trend_align(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "swing_aggressive")

    action, score, reasons, out_features = loop._apply_swing_strategy_overlay(
        symbol="MSFT",
        action="HOLD",
        score=0.50,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "pct_from_close": 0.018,
            "mom_5m": 0.006,
            "vol_30m": 0.007,
            "range_pos": 0.89,
            "spread_bps": 3.0,
            "ctx_SPY_pct_from_close": 0.006,
            "ctx_QQQ_pct_from_close": 0.007,
            "ctx_IWM_pct_from_close": 0.005,
            "news_sentiment": 0.18,
            "news_shock_rate": 0.20,
            "calendar_event_proximity_norm": 0.10,
            "calendar_high_impact_24h_norm": 0.15,
        },
        state={"weekly_trend_ema_by_symbol": {"MSFT": 0.025}},
    )

    assert action == "BUY"
    assert score >= 0.55
    assert any("swing_regime_trend_bias" in reason for reason in reasons)
    assert out_features["swing_regime_trend_norm"] >= 0.72


def test_swing_overlay_blocks_directional_trade_in_chop(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "swing_aggressive")

    action, score, reasons, out_features = loop._apply_swing_strategy_overlay(
        symbol="IWM",
        action="BUY",
        score=0.63,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "pct_from_close": 0.0004,
            "mom_5m": 0.0002,
            "vol_30m": 0.004,
            "range_pos": 0.50,
            "spread_bps": 11.0,
            "ctx_SPY_pct_from_close": 0.0001,
            "ctx_QQQ_pct_from_close": -0.0002,
            "ctx_IWM_pct_from_close": 0.0001,
            "news_sentiment": 0.02,
            "news_shock_rate": 0.10,
            "calendar_event_proximity_norm": 0.05,
            "calendar_high_impact_24h_norm": 0.05,
        },
        state={"weekly_trend_ema_by_symbol": {"IWM": 0.0}},
    )

    assert action == "HOLD"
    assert score < 0.60
    assert any("swing_regime_chop_guard" in reason for reason in reasons)
    assert out_features["swing_regime_chop_norm"] >= 0.70


def test_swing_overlay_promotes_weekly_pullback_quality(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "swing_aggressive")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, out_features = loop._apply_swing_strategy_overlay(
        symbol="AMZN",
        action="HOLD",
        score=0.51,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "pct_from_close": -0.018,
            "mom_5m": 0.003,
            "vol_30m": 0.006,
            "range_pos": 0.42,
            "spread_bps": 4.0,
            "ctx_SPY_pct_from_close": -0.030,
            "ctx_QQQ_pct_from_close": -0.028,
            "ctx_IWM_pct_from_close": -0.029,
            "news_sentiment": 0.16,
            "news_shock_rate": 0.14,
            "calendar_event_proximity_norm": 0.08,
            "calendar_high_impact_24h_norm": 0.10,
        },
        state={"weekly_trend_ema_by_symbol": {"AMZN": 0.030}},
    )

    assert action == "BUY"
    assert score >= 0.55
    assert any("swing_weekly_pullback_quality" in reason for reason in reasons)
    assert out_features["swing_weekly_pullback_quality_norm"] >= 0.68


def test_core_default_overlay_blocks_bot_dependency_concentration(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="SPY",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.58,
            "execution_fitness_norm": 0.56,
        },
        rows=[
            {"bot_id": "brain_refinery_v13_choppy", "action": "BUY", "weight": 0.50, "direction": 1.0},
            {"bot_id": "brain_refinery_v35_dmi_state_machine", "action": "BUY", "weight": 0.35, "direction": 1.0},
            {"bot_id": "brain_refinery_v4_simple", "action": "HOLD", "weight": 0.15, "direction": 0.0},
        ],
        profile="default",
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any("default_dependency_guard" in reason for reason in reasons)
    assert out_features["core_default_dependency_norm"] >= 0.68


def test_core_aggressive_overlay_promotes_high_conviction_breakout(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "aggressive")

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="NVDA",
        action="HOLD",
        score=0.51,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "market_micro_tradeability_score_norm": 0.76,
            "execution_fitness_norm": 0.74,
            "market_micro_range_expansion_norm": 0.92,
            "market_micro_trend_persistence_norm": 0.88,
            "market_micro_relative_volume_norm": 0.90,
            "cross_bot_conflict_norm": 0.10,
            "lead_lag_signal_signed": 0.62,
            "flow_direction_signed": 0.58,
            "market_micro_order_flow_imbalance_norm": 0.80,
            "pct_from_close": 0.014,
            "mom_5m": 0.007,
            "vol_30m": 0.009,
            "range_pos": 0.93,
            "spread_bps": 2.0,
            "ctx_SPY_pct_from_close": 0.009,
            "ctx_QQQ_pct_from_close": 0.010,
            "ctx_IWM_pct_from_close": 0.007,
        },
        rows=[],
        profile="aggressive",
    )

    assert action == "BUY"
    assert score >= 0.55
    assert any("aggressive_breakout_conviction" in reason for reason in reasons)
    assert out_features["core_aggressive_breakout_conviction_norm"] >= 0.74
    assert "core_options_structure_edge_norm" in out_features
    assert "core_cross_sectional_rank_norm" in out_features
    assert "aggressive_relative_strength_burst_norm" in out_features


def test_core_overlay_blocks_bot_concentration_cap(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="SPY",
        action="BUY",
        score=0.63,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.60,
            "execution_fitness_norm": 0.55,
        },
        rows=[
            {"bot_id": "brain_refinery_v13_choppy", "action": "BUY", "weight": 0.88, "direction": 1.0},
            {"bot_id": "brain_refinery_v35_dmi_state_machine", "action": "BUY", "weight": 0.12, "direction": 1.0},
        ],
        profile="default",
    )

    assert action == "HOLD"
    assert any("bot_concentration_cap" in reason for reason in reasons)
    assert out_features["core_bot_concentration_norm"] >= 0.72


def test_core_overlay_applies_profitability_upgrade_contracts(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})
    monkeypatch.setattr(
        loop,
        "_profile_profitability_control",
        lambda profile: {
            "active": True,
            "action": "tighten_entry_quality",
            "drag_score": 0.52,
            "profit_score": 0.34,
            "position_size_multiplier": 0.41,
            "thresholds": {
                "min_source_quality_norm": 0.10,
                "min_tradeability_norm": 0.10,
                "min_execution_fitness_norm": 0.10,
                "min_cross_asset_confirmation_norm": 0.10,
            },
            "outcome_weighted_training": {"sample_weight_multiplier": 2.0},
            "exit_intelligence": {
                "active": True,
                "prefer_reduce_over_add": True,
                "tighten_exit_bias_norm": 0.80,
            },
            "execution_aware_alpha": {
                "active": True,
                "unknown_fill_score_discount_norm": 0.34,
            },
            "portfolio_conflict_control": {
                "active": True,
                "max_overlap_pressure_norm": 0.90,
                "block_when_confirmation_below_norm": 0.58,
            },
        },
    )

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="MSFT",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.90,
            "execution_fitness_norm": 0.90,
            "news_source_quality_norm": 0.90,
            "calendar_event_proximity_norm": 0.90,
        },
        rows=[],
        profile="default",
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any("exit_drag_bias" in reason for reason in reasons)
    assert out_features["paper_profitability_size_multiplier_norm"] == 0.41
    assert out_features["paper_profitability_profit_score_norm"] == 0.34
    assert out_features["paper_profitability_exit_pressure_norm"] == 0.80
    assert out_features["paper_profitability_execution_discount_norm"] == 0.34


def test_core_overlay_promotes_profit_harvest_trim(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_profit_realization": True})
    monkeypatch.setattr(
        loop,
        "_profile_profit_harvest_control",
        lambda profile: {
            "active": True,
            "harvest_pressure_norm": 0.82,
            "unrealized_profit_share_norm": 0.88,
            "recommended_trim_fraction_norm": 0.40,
            "block_new_adds_when_unrealized_share_above_norm": 0.70,
            "promote_trim_when_exit_quality_above_norm": 0.55,
            "promote_trim_when_harvest_pressure_above_norm": 0.52,
        },
    )

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="MSFT",
        action="HOLD",
        score=0.51,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "market_micro_tradeability_score_norm": 0.82,
            "execution_fitness_norm": 0.84,
            "news_source_quality_norm": 0.80,
            "calendar_event_proximity_norm": 0.70,
            "core_cross_asset_confirmation_norm": 0.72,
        },
        rows=[],
        profile="default",
    )

    assert action == "SELL"
    assert score <= 0.45
    assert any("paper_profit_harvest" in reason for reason in reasons)
    assert out_features["paper_profit_harvest_active_norm"] == 1.0
    assert out_features["paper_profit_harvest_trim_fraction_norm"] == 0.40


def test_core_overlay_raw_d_recovery_raises_harvest_urgency(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})
    monkeypatch.setattr(
        loop,
        "_profitability_global_policy",
        lambda: {
            "apply_profit_realization": True,
            "apply_raw_d_recovery_ladder": True,
            "force_profit_harvest_on_raw_d": False,
            "do_not_force_trades_for_raw_recovery": True,
            "block_widening_while_raw_d": True,
            "raw_d_recovery_pressure_norm": 1.0,
            "raw_d_recovery_trim_boost_norm": 0.12,
        },
    )
    monkeypatch.setattr(
        loop,
        "_profile_profit_harvest_control",
        lambda profile: {
            "active": True,
            "harvest_pressure_norm": 0.55,
            "unrealized_profit_share_norm": 0.80,
            "recommended_trim_fraction_norm": 0.30,
            "block_new_adds_when_unrealized_share_above_norm": 0.95,
            "promote_trim_when_exit_quality_above_norm": 0.58,
            "promote_trim_when_harvest_pressure_above_norm": 0.60,
            "force_trim_when_harvest_pressure_above_norm": 0.74,
            "force_trim_when_unrealized_share_above_norm": 0.94,
        },
    )

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="MSFT",
        action="HOLD",
        score=0.51,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "market_micro_tradeability_score_norm": 0.86,
            "execution_fitness_norm": 0.86,
            "news_source_quality_norm": 0.86,
            "calendar_event_proximity_norm": 0.72,
            "core_cross_asset_confirmation_norm": 0.76,
        },
        rows=[],
        profile="default",
    )

    assert action == "SELL"
    assert any("raw_d_recovery_pressure" in reason for reason in reasons)
    assert not any("paper_profit_harvest_force_trim" in reason for reason in reasons)
    assert out_features["paper_raw_d_recovery_active_norm"] == 1.0
    assert out_features["paper_raw_d_recovery_no_force_norm"] == 1.0
    assert out_features["paper_raw_d_recovery_pressure_norm"] == 1.0
    assert out_features["paper_profit_harvest_trim_fraction_norm"] > 0.30


def test_core_overlay_uses_daily_sleeve_harvest_goal_to_block_adds(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "crypto_futures")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})
    monkeypatch.setattr(
        loop,
        "_profitability_global_policy",
        lambda: {
            "apply_profit_realization": True,
            "apply_daily_sleeve_harvest_targets": True,
            "block_new_adds_until_daily_realization_goal": True,
        },
    )
    monkeypatch.setattr(
        loop,
        "_profile_profit_harvest_control",
        lambda profile: {
            "active": True,
            "harvest_pressure_norm": 0.72,
            "unrealized_profit_share_norm": 0.91,
            "recommended_trim_fraction_norm": 0.30,
            "block_new_adds_when_unrealized_share_above_norm": 0.76,
            "promote_trim_when_exit_quality_above_norm": 0.58,
            "promote_trim_when_harvest_pressure_above_norm": 0.60,
            "daily_harvest_goal": {
                "active": True,
                "daily_goal_progress_norm": 0.05,
                "daily_harvest_pressure_norm": 0.82,
                "daily_harvest_pnl_target_total": 3500.0,
                "daily_trim_boost_norm": 0.08,
                "block_new_adds_until_daily_goal": True,
            },
        },
    )

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="BTC/USD",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.88,
            "execution_fitness_norm": 0.88,
            "news_source_quality_norm": 0.82,
            "calendar_event_proximity_norm": 0.70,
            "core_cross_asset_confirmation_norm": 0.78,
        },
        rows=[],
        profile="crypto_futures",
    )

    assert action == "HOLD"
    assert any("paper_daily_harvest_goal" in reason for reason in reasons)
    assert out_features["paper_daily_harvest_goal_active_norm"] == 1.0
    assert out_features["paper_daily_harvest_block_adds_norm"] == 1.0
    assert out_features["paper_daily_previous_target_met_norm"] == 0.0
    assert out_features["paper_daily_target_raise_active_norm"] == 0.0
    assert out_features["paper_profit_harvest_trim_fraction_norm"] > 0.30


def test_core_overlay_defers_harvest_when_intelligence_sees_continuation(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "crypto_futures")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})
    monkeypatch.setattr(
        loop,
        "_profitability_global_policy",
        lambda: {
            "apply_profit_realization": True,
            "apply_profit_harvest_intelligence": True,
            "apply_trend_continuation_holdback": True,
        },
    )
    monkeypatch.setattr(
        loop,
        "_profile_profit_harvest_control",
        lambda profile: {
            "active": True,
            "harvest_pressure_norm": 0.70,
            "unrealized_profit_share_norm": 0.82,
            "recommended_trim_fraction_norm": 0.36,
            "block_new_adds_when_unrealized_share_above_norm": 0.76,
            "promote_trim_when_exit_quality_above_norm": 0.58,
            "promote_trim_when_harvest_pressure_above_norm": 0.52,
            "force_trim_when_harvest_pressure_above_norm": 0.86,
            "force_trim_when_unrealized_share_above_norm": 0.92,
            "harvest_intelligence": {
                "active": True,
                "trend_continuation_score_norm": 0.94,
                "harvest_regret_risk_norm": 0.95,
                "realized_conversion_skill_norm": 0.10,
                "trim_aggressiveness_multiplier_norm": 0.55,
                "dynamic_exit_quality_floor_norm": 0.74,
                "hold_winner_when_trend_continuation_above_norm": 0.70,
                "force_trim_only_when_harvest_pressure_above_norm": 0.86,
            },
        },
    )

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="BTC/USD",
        action="HOLD",
        score=0.54,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "pct_from_close": 0.02,
            "mom_5m": 0.01,
            "range_pos": 0.95,
            "spread_bps": 1.0,
            "vol_30m": 0.005,
            "lead_lag_signal_signed": 0.90,
            "flow_direction_signed": 0.86,
            "market_micro_order_flow_imbalance_norm": 0.93,
            "futures_order_book_imbalance_norm": 0.90,
            "crypto_hyperliquid_basis_norm": 0.88,
            "market_micro_tradeability_score_norm": 0.88,
            "execution_fitness_norm": 0.90,
            "news_source_quality_norm": 0.82,
            "calendar_event_proximity_norm": 0.70,
            "core_cross_asset_confirmation_norm": 0.88,
        },
        rows=[],
        profile="crypto_futures",
    )

    assert action == "HOLD"
    assert score == 0.54
    assert any("paper_profit_harvest_hold_winner" in reason for reason in reasons)
    assert out_features["paper_profit_harvest_intelligence_active_norm"] == 1.0
    assert out_features["paper_profit_harvest_holdback_active_norm"] == 1.0
    assert out_features["paper_profit_harvest_regret_risk_norm"] >= 0.80


def test_paper_mirror_profitability_quarantines_losing_strategy(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_loser_quarantine": True})
    monkeypatch.setattr(
        loop,
        "_strategy_profitability_control",
        lambda profile, strategy: {
            "mode": "paper_quarantine",
            "score_penalty_norm": 0.82,
            "position_size_multiplier": 0.08,
        },
    )

    action, score, reasons, features = loop._apply_paper_mirror_profitability_control(
        profile="aggressive",
        strategy="paper_mirror::brain_refinery_v48_position_1m_3m",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={"market_micro_tradeability_score_norm": 0.7},
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any("paper_strategy_quarantine" in reason for reason in reasons)
    assert features["paper_profitability_strategy_control_active_norm"] == 1.0
    assert features["paper_profitability_strategy_size_multiplier_norm"] == 0.08


def test_paper_mirror_profitability_quarantine_keeps_sell_exit_open(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_loser_quarantine": True})
    monkeypatch.setattr(
        loop,
        "_strategy_profitability_control",
        lambda profile, strategy: {
            "mode": "paper_quarantine",
            "score_penalty_norm": 0.82,
            "position_size_multiplier": 0.0,
        },
    )

    action, _score, reasons, features = loop._apply_paper_mirror_profitability_control(
        profile="aggressive",
        strategy="paper_mirror::brain_refinery_v48_position_1m_3m",
        action="SELL",
        score=0.36,
        threshold=0.55,
        reasons=["base_sell"],
        features={"market_micro_tradeability_score_norm": 0.7},
    )

    assert action == "SELL"
    assert any("paper_strategy_reduce_only_quarantine_exit_open" in reason for reason in reasons)
    assert features["paper_profitability_strategy_size_multiplier_norm"] == 1.0


def test_candidate_bound_strategy_scaling_can_expand_only_after_contract_allows(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_strategy_profitability_control", lambda profile, strategy: {})
    monkeypatch.setattr(
        loop,
        "_strategy_profitability_scaling_control",
        lambda profile, strategy: {
            "tier": "scale_tier_1",
            "entry_size_multiplier_norm": 1.05,
            "above_baseline_scale_ready": True,
            "block_new_entries": False,
        },
    )

    action, _score, reasons, features = loop._apply_paper_mirror_profitability_control(
        profile="default",
        strategy="paper_mirror::candidate_bound_winner",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={"market_micro_tradeability_score_norm": 0.8},
    )

    assert action == "BUY"
    assert features["paper_profitability_strategy_size_multiplier_norm"] == 1.05
    assert features["paper_profitability_strategy_above_baseline_scale_norm"] == 1.0
    assert any("candidate_bound_strategy_scaling tier=scale_tier_1" in reason for reason in reasons)


def test_paper_consensus_recomputes_execution_realism_before_entry_policy(monkeypatch) -> None:
    captured = {}

    def evaluate(*, profile, features):
        captured.update(features)
        return {
            "allowed": True,
            "regime_fit_norm": 0.8,
            "evidence_quality_norm": 0.8,
            "execution_plan": {"style": "passive_limit"},
        }

    class PaperTrader:
        def execute_decision(self, **kwargs):
            captured["executed_features"] = kwargs["features"]
            return {"status": "PAPER_EXECUTED"}

    monkeypatch.setattr(loop, "evaluate_profitability_entry", evaluate)
    result = loop._execute_paper_mirror_consensus(
        broker="schwab",
        symbol="AAPL",
        profile="default",
        segment="core",
        snapshot_id="snapshot-1",
        candidates=[
            {
                "bot_id": "alpha",
                "action": "BUY",
                "score": 0.7,
                "threshold": 0.55,
                "weight": 1.0,
                "test_accuracy": 0.8,
                "quality_score": 0.8,
                "paper_execution_authority": True,
                "bot_role": "signal_sub_bot",
                "sleeve_id": "equity_core",
                "sub_sleeve_id": "trend",
                "correlation_cluster_id": "cluster_a",
                "features": {"execution_fitness_norm": 1.0},
                "eligible": True,
            },
            {
                "bot_id": "beta",
                "action": "BUY",
                "score": 0.69,
                "threshold": 0.55,
                "weight": 0.8,
                "test_accuracy": 0.78,
                "quality_score": 0.79,
                "paper_execution_authority": True,
                "bot_role": "signal_sub_bot",
                "sleeve_id": "equity_core",
                "sub_sleeve_id": "breadth",
                "correlation_cluster_id": "cluster_b",
                "features": {"execution_fitness_norm": 1.0},
                "eligible": True,
            },
        ],
        shared_features={
            "last_price": 100.0,
            "market_micro_tradeability_score_norm": 0.8,
            "execution_fitness_norm": 0.0,
            "data_quality_quote_agreement_norm": 1.0,
            "spread_bps": 4.0,
        },
        gates={"market_data_ok": True},
        execution_lane_enabled=False,
        paper_trader=PaperTrader(),
        selection_reason="unit_test",
    )

    assert result["action"] == "BUY"
    assert captured["execution_fitness_norm"] > 0.2
    assert captured["paper_entry_execution_realism_recomputed_norm"] == 1.0
    assert captured["executed_features"]["execution_fitness_norm"] > 0.2


def test_paper_mirror_strategy_harvest_blocks_adds(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_strategy_profit_harvest": True})
    monkeypatch.setattr(loop, "_strategy_profitability_control", lambda profile, strategy: {})
    monkeypatch.setattr(
        loop,
        "_strategy_profit_harvest_control",
        lambda profile, strategy: {
            "active": True,
            "tier": "tier_3_protect_runner",
            "profile_harvest_pressure_norm": 0.88,
            "recommended_trim_fraction_norm": 0.34,
            "block_new_adds": True,
            "promote_partial_trim": True,
            "protect_runner_when_trend_continuation_above_norm": 0.80,
            "force_trim_when_harvest_pressure_above_norm": 0.90,
        },
    )

    action, score, reasons, features = loop._apply_paper_mirror_profitability_control(
        profile="default",
        strategy="paper_mirror::brain_refinery_v21_flash_crash",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.80,
            "execution_fitness_norm": 0.82,
            "news_source_quality_norm": 0.78,
            "core_cross_asset_confirmation_norm": 0.76,
        },
    )

    assert action == "HOLD"
    assert any("paper_strategy_harvest_block_add" in reason for reason in reasons)
    assert features["paper_strategy_profit_harvest_active_norm"] == 1.0
    assert features["paper_strategy_profit_harvest_trim_fraction_norm"] == 0.34


def test_paper_mirror_strategy_harvest_promotes_trim(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_strategy_profit_harvest": True})
    monkeypatch.setattr(loop, "_strategy_profitability_control", lambda profile, strategy: {})
    monkeypatch.setattr(
        loop,
        "_strategy_profit_harvest_control",
        lambda profile, strategy: {
            "active": True,
            "tier": "tier_2_pay_the_system",
            "profile_harvest_pressure_norm": 0.74,
            "recommended_trim_fraction_norm": 0.28,
            "block_new_adds": False,
            "promote_partial_trim": True,
            "protect_runner_when_trend_continuation_above_norm": 0.82,
            "force_trim_when_harvest_pressure_above_norm": 0.90,
        },
    )

    action, score, reasons, features = loop._apply_paper_mirror_profitability_control(
        profile="fx",
        strategy="paper_mirror::brain_refinery_v10_seasonal",
        action="HOLD",
        score=0.52,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "market_micro_tradeability_score_norm": 0.86,
            "execution_fitness_norm": 0.88,
            "news_source_quality_norm": 0.80,
            "calendar_event_proximity_norm": 0.72,
            "core_cross_asset_confirmation_norm": 0.62,
        },
    )

    assert action == "SELL"
    assert any("paper_strategy_harvest_trim" in reason for reason in reasons)
    assert features["paper_strategy_profit_harvest_active_norm"] == 1.0
    assert features["paper_strategy_profit_harvest_runner_protected_norm"] == 0.0


def test_paper_mirror_confirmation_bias_blocks_low_evidence_strategy(monkeypatch) -> None:
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_loser_quarantine": True})
    monkeypatch.setattr(
        loop,
        "_strategy_profitability_control",
        lambda profile, strategy: {
            "mode": "deweight",
            "score_penalty_norm": 0.44,
            "position_size_multiplier": 0.62,
            "confirmation_bias_control": {
                "active": True,
                "confirmation_bias_score_norm": 0.68,
                "min_independent_evidence_channels": 4,
                "block_when_quality_gate_below_norm": 0.62,
                "score_dampen_when_quality_below_norm": 0.70,
            },
        },
    )

    action, score, reasons, features = loop._apply_paper_mirror_profitability_control(
        profile="swing_aggressive",
        strategy="paper_mirror::brain_refinery_v48_position_1m_3m",
        action="BUY",
        score=0.66,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.48,
            "execution_fitness_norm": 0.46,
            "news_source_quality_norm": 0.42,
            "calendar_event_proximity_norm": 0.20,
            "core_cross_asset_confirmation_norm": 0.36,
            "cross_bot_conflict_norm": 0.52,
        },
    )

    assert action == "HOLD"
    assert score < 0.60
    assert any("confirmation_bias_guard" in reason for reason in reasons)
    assert features["paper_profitability_confirmation_bias_active_norm"] == 1.0
    assert features["paper_profitability_strategy_confirmation_quality_norm"] < 0.62


def test_day_overlay_intraday_allowlist_blocks_weak_symbol(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "intraday_aggressive")
    monkeypatch.setattr(loop, "_session_phase_norms", lambda: (0.3, 0.0, 0.0))
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.70)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, out_features = loop._apply_day_strategy_overlay(
        symbol="AAPL",
        action="BUY",
        score=0.62,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "pct_from_close": 0.004,
            "mom_5m": 0.002,
            "vol_30m": 0.012,
            "range_pos": 0.62,
            "spread_bps": 8.0,
            "bid_size": 500.0,
            "ask_size": 520.0,
            "execution_fitness_norm": 0.60,
            "market_micro_tradeability_score_norm": 0.58,
            "market_micro_relative_volume_norm": 0.40,
            "market_micro_trend_persistence_norm": 0.42,
            "market_micro_lunch_chop_norm": 0.46,
            "ctx_SPY_pct_from_close": 0.002,
            "ctx_QQQ_pct_from_close": 0.002,
            "ctx_IWM_pct_from_close": 0.001,
        },
        state={},
    )

    assert action == "HOLD"
    assert any("intraday_allowlist_score" in reason for reason in reasons)
    assert out_features["intraday_allowlist_score_norm"] < 0.60
    assert "day_open_drive_conviction_norm" in out_features


def test_core_fx_overlay_requires_macro_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "fx")

    action, score, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="EURUSD",
        action="BUY",
        score=0.62,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.63,
            "execution_fitness_norm": 0.61,
            "fx_dxy_yield_confirmation_norm": 0.34,
            "fx_proxy_agreement_norm": 0.42,
            "fx_session_london_norm": 1.0,
            "fx_session_ny_norm": 0.0,
            "fx_rollover_risk_norm": 0.10,
        },
        rows=[],
        profile="fx",
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any("fx_macro_confirmation_guard" in reason for reason in reasons)
    assert out_features["core_fx_macro_confirmation_norm"] < 0.58


def test_live_macro_sleeve_gate_features_separate_headwinds_from_tailwinds() -> None:
    snapshot = {
        "active": True,
        "broad_market": True,
        "published": "2026-04-02T17:03:38+00:00",
        "market_actionable_score": 3.7125,
        "source_priority_norm": 0.97,
        "official_source_norm": 1.0,
        "transcript_quality_norm": 0.5178,
        "market_high_conviction": True,
        "market_confirmation": {"distinct_segments": 5, "confirmed": True},
        "items": [
            {
                "published": "2026-04-02T17:03:38+00:00",
                "broad_market": True,
                "symbols": ["SPY", "QQQ", "USO", "XLE", "GLD", "TLT", "UUP", "DAL"],
                "sentiment_hint": -0.95,
                "shock_hint": 1.0,
                "signal_types": ["military_escalation", "oil_shipping_supply"],
            }
        ],
    }

    qqq_features = loop._derive_live_macro_sleeve_gate_features(
        snapshot=snapshot,
        symbol="QQQ",
        now_ts=1_775_218_700.0,
    )
    uso_features = loop._derive_live_macro_sleeve_gate_features(
        snapshot=snapshot,
        symbol="USO",
        now_ts=1_775_218_700.0,
    )

    assert qqq_features["live_macro_gate_active_norm"] == 1.0
    assert qqq_features["live_macro_headwind_norm"] >= 0.74
    assert qqq_features["live_macro_tailwind_norm"] < 0.56
    assert uso_features["live_macro_tailwind_norm"] >= 0.78
    assert uso_features["live_macro_oil_shock_norm"] >= 0.90


def test_core_overlay_blocks_broad_risk_buy_on_live_macro_headwind(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")

    action, score, reasons, _ = loop._apply_core_sleeve_strategy_overlay(
        symbol="QQQ",
        action="BUY",
        score=0.64,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.68,
            "execution_fitness_norm": 0.66,
            "live_macro_gate_active_norm": 1.0,
            "live_macro_gate_confidence_norm": 0.88,
            "live_macro_headwind_norm": 0.86,
            "live_macro_tailwind_norm": 0.12,
            "live_macro_event_alignment_norm": 0.18,
        },
        rows=[],
        profile="default",
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any("core_live_macro_headwind" in reason for reason in reasons)


def test_core_overlay_allows_coinbase_paper_probation_to_retest_weak_profile(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")
    monkeypatch.setattr(
        loop,
        "_profile_profitability_control",
        lambda profile: {
            "active": True,
            "action": "quarantine_new_entries",
            "block_new_entries": True,
            "new_entry_cap": 0,
            "thresholds": {
                "min_source_quality_norm": 0.70,
                "min_tradeability_norm": 0.70,
                "min_execution_fitness_norm": 0.70,
                "min_cross_asset_confirmation_norm": 0.56,
                "min_event_proximity_norm": 0.70,
            },
        },
    )
    monkeypatch.setattr(
        loop,
        "_raw_profitability_a_recovery_contract",
        lambda: {
            "active": True,
            "runtime_enforcement": {
                "block_new_entries_on_weak_profiles": True,
                "min_quality_gate_norm": 0.72,
                "min_tradeability_norm": 0.58,
                "min_execution_fitness_norm": 0.58,
                "min_cross_asset_confirmation_norm": 0.56,
                "max_overlap_pressure_norm": 0.58,
            },
        },
    )
    features = {
        "market_micro_tradeability_score_norm": 0.82,
        "execution_fitness_norm": 0.83,
        "news_source_quality_norm": 0.86,
        "calendar_event_proximity_norm": 0.80,
        "cross_asset_confirmation_norm": 0.84,
        "core_cross_asset_confirmation_norm": 0.84,
        "lead_lag_confirmation_norm": 0.90,
        "lead_lag_signal_signed": 0.90,
        "flow_direction_signed": 0.90,
        "flow_conviction_norm": 0.90,
        "ctx_SPY_pct_from_close": 0.012,
        "ctx_QQQ_pct_from_close": 0.012,
        "ctx_IWM_pct_from_close": 0.010,
        "cross_bot_conflict_norm": 0.02,
        "core_portfolio_overlap_pressure_norm": 0.12,
    }

    blocked_action, _, blocked_reasons, _ = loop._apply_core_sleeve_strategy_overlay(
        symbol="ETH-USD",
        action="BUY",
        score=0.76,
        threshold=0.55,
        reasons=["base_buy"],
        features=features,
        rows=[],
        profile="default",
    )

    assert blocked_action == "HOLD"
    assert any("weak_profile=default" in reason for reason in blocked_reasons)

    monkeypatch.setenv("SHADOW_BROKER", "coinbase")
    monkeypatch.setenv("TOP_BOT_PAPER_TRADING_ENABLED", "1")
    monkeypatch.setenv("COINBASE_PAPER_PROBATION_ENABLED", "1")
    monkeypatch.setenv("COINBASE_PAPER_PROBATIONARY_PROFILES", "default,crypto_futures")

    action, score, reasons, _ = loop._apply_core_sleeve_strategy_overlay(
        symbol="ETH-USD",
        action="BUY",
        score=0.76,
        threshold=0.55,
        reasons=["base_buy"],
        features=features,
        rows=[],
        profile="default",
    )

    assert action == "BUY"
    assert score == pytest.approx(0.76)
    assert not any("weak_profile=default" in reason for reason in reasons)
    assert not any("paper_profitability_quarantine" in reason for reason in reasons)


def test_day_overlay_blocks_buy_on_live_macro_headwind(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "intraday_aggressive")
    monkeypatch.setattr(loop, "_session_phase_norms", lambda: (0.25, 0.0, 0.0))
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, _ = loop._apply_day_strategy_overlay(
        symbol="SPY",
        action="BUY",
        score=0.66,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "pct_from_close": 0.007,
            "mom_5m": 0.003,
            "vol_30m": 0.008,
            "spread_bps": 3.0,
            "bid_size": 1200.0,
            "ask_size": 900.0,
            "execution_fitness_norm": 0.76,
            "market_micro_tradeability_score_norm": 0.80,
            "market_micro_relative_volume_norm": 0.84,
            "market_micro_trend_persistence_norm": 0.76,
            "live_macro_gate_active_norm": 1.0,
            "live_macro_gate_confidence_norm": 0.90,
            "live_macro_headwind_norm": 0.88,
            "live_macro_tailwind_norm": 0.14,
            "ctx_SPY_pct_from_close": -0.010,
            "ctx_QQQ_pct_from_close": -0.014,
            "ctx_IWM_pct_from_close": -0.012,
        },
        state={},
    )

    assert action == "HOLD"
    assert any("day_live_macro_headwind" in reason for reason in reasons)


def test_swing_overlay_blocks_buy_on_live_macro_headwind(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "swing_aggressive")
    monkeypatch.setattr(loop, "_profile_symbol_drag_penalty_norm", lambda profile, symbol: 0.0)
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, _ = loop._apply_swing_strategy_overlay(
        symbol="IWM",
        action="BUY",
        score=0.62,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "pct_from_close": 0.009,
            "mom_5m": 0.003,
            "vol_30m": 0.008,
            "spread_bps": 4.0,
            "ctx_SPY_pct_from_close": -0.011,
            "ctx_QQQ_pct_from_close": -0.015,
            "ctx_IWM_pct_from_close": -0.012,
            "news_sentiment": -0.30,
            "news_shock_rate": 0.65,
            "calendar_event_proximity_norm": 0.55,
            "calendar_high_impact_24h_norm": 0.64,
            "live_macro_gate_active_norm": 1.0,
            "live_macro_gate_confidence_norm": 0.88,
            "live_macro_headwind_norm": 0.84,
            "live_macro_tailwind_norm": 0.18,
        },
        state={"weekly_trend_ema_by_symbol": {"IWM": 0.01}},
    )

    assert action == "HOLD"
    assert any("swing_live_macro_headwind" in reason for reason in reasons)


def test_bond_overlay_holds_sell_when_safe_haven_tailwind_is_active(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "bond")
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, _ = loop._apply_bond_strategy_overlay(
        symbol="TLT",
        action="SELL",
        score=0.61,
        threshold=0.55,
        reasons=["base_sell"],
        features={
            "market_micro_tradeability_score_norm": 0.70,
            "ctx_TLT_pct_from_close": 0.008,
            "ctx_SHY_pct_from_close": -0.001,
            "dividend_yield_norm": 0.30,
            "vol_30m": 0.010,
            "range_pos": 0.55,
            "live_macro_gate_active_norm": 1.0,
            "live_macro_headwind_norm": 0.18,
            "live_macro_tailwind_norm": 0.86,
            "live_macro_oil_shock_norm": 0.82,
        },
        rows=[
            {"bot_id": "brain_refinery_v95_rates_regime_bond_bot", "action": "SELL", "weight": 1.0, "direction": -1.0},
        ],
    )

    assert action == "HOLD"
    assert any("bond_live_macro_safe_haven_tailwind" in reason for reason in reasons)


def test_long_term_allocation_blocks_buy_on_live_macro_headwind(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "long_term_core_etf")

    action, score, reasons, _ = loop._apply_long_term_allocation_policy(
        symbol="SPY",
        action="BUY",
        score=0.61,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "long_term_accumulation_discipline_norm": 0.72,
            "long_term_factor_quality_mix_norm": 0.74,
            "long_term_rebalance_overlap_penalty_norm": 0.18,
            "long_term_overlap_crowding_norm": 0.22,
            "long_term_factor_exposure_control_norm": 0.70,
            "long_term_overlap_rebalance_norm": 0.20,
            "long_term_valuation_reserve_norm": 0.66,
            "long_term_compounder_conviction_norm": 0.70,
            "long_term_downside_preservation_norm": 0.60,
            "pct_from_close": 0.035,
            "live_macro_gate_active_norm": 1.0,
            "live_macro_headwind_norm": 0.86,
            "live_macro_tailwind_norm": 0.18,
        },
        iter_count=64,
        state={
            "last_buy_iter_by_symbol": {"SPY": 40.0},
            "position_state_by_symbol": {
                "SPY": {
                    "position_open": 1.0,
                    "first_buy_iter": 40.0,
                    "last_buy_iter": 40.0,
                    "hold_iters": 24.0,
                    "avg_cost": 100.0,
                }
            },
        },
    )

    assert action == "HOLD"
    assert any("long_term_live_macro_headwind" in reason for reason in reasons)


def test_bond_overlay_blocks_equity_style_contamination(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "bond")

    action, score, reasons, out_features = loop._apply_bond_strategy_overlay(
        symbol="TLT",
        action="BUY",
        score=0.62,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.62,
            "ctx_TLT_pct_from_close": 0.001,
            "ctx_SHY_pct_from_close": 0.001,
            "dividend_yield_norm": 0.20,
            "vol_30m": 0.018,
            "range_pos": 0.48,
        },
        rows=[
            {"bot_id": "brain_refinery_v13_choppy", "action": "BUY", "weight": 0.45, "direction": 1.0},
            {"bot_id": "brain_refinery_v21_flash_crash", "action": "BUY", "weight": 0.30, "direction": 1.0},
            {"bot_id": "brain_refinery_v4_simple", "action": "BUY", "weight": 0.20, "direction": 1.0},
            {"bot_id": "brain_refinery_v95_rates_regime_bond_bot", "action": "HOLD", "weight": 0.05, "direction": 0.0},
        ],
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any(
        marker in reason
        for reason in reasons
        for marker in ("bond_equity_style_contamination", "bond_hard_roster_guard")
    )
    assert out_features["bond_duration_regime_norm"] < 0.70
    assert out_features["bond_equity_contamination_norm"] >= 0.60


def test_bond_overlay_requires_hard_roster_alignment(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "bond")
    monkeypatch.setattr(loop, "_profile_drag_snapshot", lambda profile: {"active": False, "drag_norm": 0.0})

    action, score, reasons, out_features = loop._apply_bond_strategy_overlay(
        symbol="IEF",
        action="BUY",
        score=0.62,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.62,
            "ctx_TLT_pct_from_close": -0.002,
            "ctx_SHY_pct_from_close": 0.001,
            "dividend_yield_norm": 0.40,
            "vol_30m": 0.010,
            "range_pos": 0.50,
        },
        rows=[
            {"bot_id": "brain_refinery_v13_choppy", "action": "BUY", "weight": 0.50, "direction": 1.0},
            {"bot_id": "brain_refinery_v4_simple", "action": "BUY", "weight": 0.50, "direction": 1.0},
        ],
    )

    assert action == "HOLD"
    assert any("bond_hard_roster_guard" in reason for reason in reasons)
    assert out_features["bond_bot_roster_alignment_norm"] < 0.52
    assert "bond_equity_contamination_norm" in out_features


def test_dividend_overlay_prefers_reinvest_bias_when_capture_edge_is_low(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "dividend")
    monkeypatch.setenv("DIVIDEND_STRATEGY_MODE", "compound")

    action, score, reasons, _ = loop._apply_dividend_safety_tax_overlay(
        symbol="SCHD",
        action="HOLD",
        score=0.51,
        threshold=0.55,
        reasons=["base_hold"],
        features={
            "dividend_quality_score_norm": 0.82,
            "dividend_payout_ratio_norm": 0.42,
            "dividend_fcf_coverage_norm": 0.80,
            "dividend_structure_aware_quality_norm": 0.78,
            "dividend_income_quality_norm": 0.77,
            "dividend_cut_freeze_risk_norm": 0.18,
            "dividend_capture_vs_hold_edge_norm": 0.24,
            "dividend_reinvest_cadence_norm": 0.77,
            "dividend_payout_stress_forward_norm": 0.18,
            "dividend_forward_hazard_norm": 0.20,
            "dividend_ex_date_proximity_norm": 0.10,
            "dividend_pay_date_proximity_norm": 0.12,
            "dividend_tax_qualified_hold_norm": 0.84,
            "dividend_drip_active_norm": 0.8,
            "dividend_drip_confidence_norm": 0.9,
            "dividend_drip_recent_reinvest_norm": 0.8,
        },
        iter_count=20,
        state={},
    )

    assert action == "BUY"
    assert score >= 0.55
    assert any("dividend_reinvest_underwriting_bias" in reason for reason in reasons)


def test_long_term_allocation_policy_uses_staggered_accumulation(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "long_term_core_etf")

    action, score, reasons, _ = loop._apply_long_term_allocation_policy(
        symbol="VOO",
        action="HOLD",
        score=0.52,
        threshold=0.55,
        reasons=["base_hold"],
            features={
                "long_term_accumulation_discipline_norm": 0.82,
                "long_term_factor_quality_mix_norm": 0.79,
                "long_term_rebalance_overlap_penalty_norm": 0.18,
                "long_term_overlap_crowding_norm": 0.20,
                "long_term_factor_exposure_control_norm": 0.78,
                "long_term_overlap_rebalance_norm": 0.22,
                "long_term_valuation_reserve_norm": 0.74,
                "long_term_compounder_conviction_norm": 0.76,
                "long_term_recent_pullback_norm": 0.32,
                "pct_from_close": 0.03,
            },
        iter_count=24,
        state={
            "last_buy_iter_by_symbol": {"VOO": 12.0},
            "position_state_by_symbol": {
                "VOO": {
                    "position_open": 1.0,
                    "first_buy_iter": 12.0,
                    "last_buy_iter": 12.0,
                    "hold_iters": 12.0,
                    "avg_cost": 100.0,
                }
            },
        },
    )

    assert action == "BUY"
    assert score >= 0.55
    assert any("long_term_staggered_accumulation" in reason for reason in reasons)


def test_market_stress_quorum_ignores_non_actionable_quotes_and_requires_distinct_symbols() -> None:
    state = {}

    hold = loop._update_actionable_market_stress_quorum(
        state=state,
        symbol="ILLIQUID",
        action="HOLD",
        metric_value=90.0,
        threshold=35.0,
        now_ts=100.0,
        window_seconds=120.0,
        minimum_distinct_symbols=3,
    )
    first = loop._update_actionable_market_stress_quorum(
        state=state,
        symbol="AAA",
        action="BUY",
        metric_value=40.0,
        threshold=35.0,
        now_ts=101.0,
        window_seconds=120.0,
        minimum_distinct_symbols=3,
    )
    repeated = loop._update_actionable_market_stress_quorum(
        state=state,
        symbol="AAA",
        action="SELL",
        metric_value=45.0,
        threshold=35.0,
        now_ts=102.0,
        window_seconds=120.0,
        minimum_distinct_symbols=3,
    )
    loop._update_actionable_market_stress_quorum(
        state=state,
        symbol="BBB",
        action="BUY",
        metric_value=50.0,
        threshold=35.0,
        now_ts=103.0,
        window_seconds=120.0,
        minimum_distinct_symbols=3,
    )
    quorum = loop._update_actionable_market_stress_quorum(
        state=state,
        symbol="CCC",
        action="BUY",
        metric_value=55.0,
        threshold=35.0,
        now_ts=104.0,
        window_seconds=120.0,
        minimum_distinct_symbols=3,
    )

    assert hold["observed"] is False
    assert first["triggered"] is False
    assert repeated["distinct_symbol_count"] == 1
    assert quorum["triggered"] is True
    assert quorum["symbols"] == ["AAA", "BBB", "CCC"]


def test_market_stress_quorum_expires_old_symbols() -> None:
    state = {"OLD": 10.0, "FRESH": 95.0}

    result = loop._update_actionable_market_stress_quorum(
        state=state,
        symbol="NEW",
        action="BUY",
        metric_value=50.0,
        threshold=35.0,
        now_ts=100.0,
        window_seconds=20.0,
        minimum_distinct_symbols=2,
    )

    assert result["triggered"] is True
    assert result["symbols"] == ["FRESH", "NEW"]
    assert "OLD" not in state


def test_decision_disposition_distinguishes_no_edge_guarded_hold_and_trade() -> None:
    no_edge = loop._decision_disposition(
        intent_action="HOLD",
        final_action="HOLD",
        reasons=["grand_master_deadband"],
    )
    protected = loop._decision_disposition(
        intent_action="BUY",
        final_action="HOLD",
        reasons=[
            "lane_kill_switch_pause lane=equities",
            "execution_guard_block spread_ok=0",
        ],
    )
    trade = loop._decision_disposition(
        intent_action="BUY",
        final_action="BUY",
        reasons=[],
    )

    assert no_edge == {
        "disposition": "no_edge_hold",
        "blocking_stage": "signal_selection",
        "guard_categories": [],
        "guard_reasons": [],
    }
    assert protected["disposition"] == "protected_hold"
    assert protected["blocking_stage"] == "execution"
    assert protected["guard_categories"] == ["execution", "circuit_breaker"]
    assert trade["disposition"] == "paper_trade"


def test_symbol_circuit_strikes_only_track_symbol_local_failures() -> None:
    lane_pause = loop._symbol_scoped_guard_failure(
        intent_action="BUY",
        final_action="HOLD",
        execution_guard_ok=True,
        feature_freshness_ok=True,
    )
    bad_execution = loop._symbol_scoped_guard_failure(
        intent_action="BUY",
        final_action="HOLD",
        execution_guard_ok=False,
        feature_freshness_ok=True,
    )
    stale_features = loop._symbol_scoped_guard_failure(
        intent_action="SELL",
        final_action="HOLD",
        execution_guard_ok=True,
        feature_freshness_ok=False,
    )

    assert lane_pause is False
    assert bad_execution is True
    assert stale_features is True


def test_paper_debt_recovery_pauses_buys_without_closing_reduce_path(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_paper_debt_recovery": True})
    monkeypatch.setattr(
        loop,
        "_paper_debt_recovery_contract",
        lambda: {
            "active": True,
            "state": "paused_drawdown",
            "baseline_debt_amount": 20_000.0,
            "remaining_debt_amount": 20_250.0,
            "recovery_progress_norm": 0.0,
            "risk_budget": {"new_entries_paused": True},
            "runtime_enforcement": {
                "block_new_entries_on_weak_profiles": True,
                "do_not_force_trades": True,
                "prohibit_martingale": True,
                "prohibit_averaging_down_for_recovery": True,
                "prohibit_loss_based_size_increase": True,
                "recovery_entry_size_multiplier_norm": 0.0,
                "min_quality_gate_norm": 0.72,
                "min_tradeability_norm": 0.58,
                "min_execution_fitness_norm": 0.58,
                "min_cross_asset_confirmation_norm": 0.56,
                "max_overlap_pressure_norm": 0.58,
            },
        },
    )
    features = {
        "market_micro_tradeability_score_norm": 0.90,
        "execution_fitness_norm": 0.90,
        "news_source_quality_norm": 0.90,
        "cross_asset_confirmation_norm": 0.90,
        "core_cross_asset_confirmation_norm": 0.90,
        "core_portfolio_overlap_pressure_norm": 0.05,
    }

    buy_action, _, buy_reasons, buy_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="SPY",
        action="BUY",
        score=0.72,
        threshold=0.55,
        reasons=["base_buy"],
        features=features,
        rows=[],
        profile="default",
    )
    sell_action, _, sell_reasons, _ = loop._apply_core_sleeve_strategy_overlay(
        symbol="SPY",
        action="SELL",
        score=0.28,
        threshold=0.55,
        reasons=["base_sell"],
        features=features,
        rows=[],
        profile="default",
    )

    assert buy_action == "HOLD"
    assert any("paper_debt_recovery_gate paused=paused_drawdown" in reason for reason in buy_reasons)
    assert buy_features["paper_debt_recovery_entry_size_multiplier_norm"] == 0.0
    assert buy_features["paper_debt_recovery_no_loss_chasing_norm"] == 1.0
    assert sell_action == "SELL"
    assert not any("paper_debt_recovery_gate" in reason for reason in sell_reasons)


def test_paper_debt_recovery_caps_clean_buy_size_while_evidence_accumulates(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "default")
    monkeypatch.setattr(loop, "_profitability_global_policy", lambda: {"apply_paper_debt_recovery": True})
    monkeypatch.setattr(
        loop,
        "_paper_debt_recovery_contract",
        lambda: {
            "active": True,
            "state": "collecting_recovery_evidence",
            "baseline_debt_amount": 20_000.0,
            "remaining_debt_amount": 20_000.0,
            "recovery_progress_norm": 0.0,
            "risk_budget": {"new_entries_paused": False},
            "runtime_enforcement": {
                "do_not_force_trades": True,
                "prohibit_martingale": True,
                "prohibit_averaging_down_for_recovery": True,
                "prohibit_loss_based_size_increase": True,
                "recovery_entry_size_multiplier_norm": 0.25,
                "min_quality_gate_norm": 0.60,
                "min_tradeability_norm": 0.58,
                "min_execution_fitness_norm": 0.58,
                "min_cross_asset_confirmation_norm": 0.56,
                "max_overlap_pressure_norm": 0.58,
            },
        },
    )

    action, _, reasons, out_features = loop._apply_core_sleeve_strategy_overlay(
        symbol="SPY",
        action="BUY",
        score=0.72,
        threshold=0.55,
        reasons=["base_buy"],
        features={
            "market_micro_tradeability_score_norm": 0.90,
            "execution_fitness_norm": 0.90,
            "news_source_quality_norm": 0.90,
            "cross_asset_confirmation_norm": 0.90,
            "core_cross_asset_confirmation_norm": 0.90,
            "core_portfolio_overlap_pressure_norm": 0.05,
            "lead_lag_confirmation_norm": 0.90,
            "lead_lag_signal_signed": 0.90,
            "flow_direction_signed": 0.90,
            "flow_conviction_norm": 0.90,
            "ctx_SPY_pct_from_close": 0.012,
            "ctx_QQQ_pct_from_close": 0.012,
            "ctx_IWM_pct_from_close": 0.010,
        },
        rows=[],
        profile="default",
    )

    assert action == "BUY"
    assert not any("paper_debt_recovery_gate" in reason for reason in reasons)
    assert out_features["paper_debt_recovery_active_norm"] == 1.0
    assert out_features["paper_debt_recovery_entry_size_multiplier_norm"] == 0.25
