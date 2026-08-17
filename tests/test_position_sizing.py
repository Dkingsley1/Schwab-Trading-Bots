from core.position_sizing import size_from_action


def test_position_sizing_scales_down_under_pressure() -> None:
    baseline = size_from_action(
        action="BUY",
        score=0.72,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.06,
    )
    pressured = size_from_action(
        action="BUY",
        score=0.72,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.06,
        confidence_norm=0.42,
        liquidity_score=0.35,
        drawdown_pressure=0.70,
        overlap_pressure=0.64,
        correlation_pressure=0.58,
        profile_multiplier=0.82,
    )

    assert baseline > 0.0
    assert pressured > 0.0
    assert pressured < baseline


def test_position_sizing_uses_quant_kelly_signal_as_capped_governor() -> None:
    neutral = size_from_action(
        action="BUY",
        score=0.58,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.20,
        kelly_signal_norm=0.5,
    )
    favorable = size_from_action(
        action="BUY",
        score=0.58,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.20,
        kelly_signal_norm=0.85,
    )
    unfavorable = size_from_action(
        action="BUY",
        score=0.58,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.20,
        kelly_signal_norm=0.15,
    )

    assert favorable > neutral
    assert unfavorable < neutral
    assert favorable <= 20000.0


def test_position_sizing_uses_strategy_conviction_as_bounded_structural_scale() -> None:
    neutral = size_from_action(
        action="BUY",
        score=0.64,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.20,
        strategy_conviction_norm=0.5,
    )
    high_conviction = size_from_action(
        action="BUY",
        score=0.64,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.20,
        strategy_conviction_norm=0.90,
    )
    low_conviction = size_from_action(
        action="BUY",
        score=0.64,
        threshold=0.55,
        volatility_1m=0.008,
        equity_proxy=100000.0,
        max_notional_pct=0.20,
        strategy_conviction_norm=0.10,
    )

    assert high_conviction > neutral
    assert low_conviction < neutral
    assert high_conviction <= 20000.0
