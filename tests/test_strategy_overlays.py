import scripts.run_shadow_training_loop as loop


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
    assert score <= 0.55
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
    assert score <= 0.55
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
