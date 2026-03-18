import scripts.run_shadow_training_loop as loop


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
        },
        state={},
    )

    assert action == "HOLD"
    assert score <= 0.55
    assert any("day_regime_chop_guard" in reason for reason in reasons)
    assert out_features["day_regime_chop_norm"] >= 0.68


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
