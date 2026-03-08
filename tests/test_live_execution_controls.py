from core.live_execution_controls import LiveExecutionGuard, LiveRiskConfig


def _cfg(**overrides):
    base = LiveRiskConfig(
        max_position_qty_per_symbol=10.0,
        max_order_notional=5000.0,
        max_open_orders_total=5,
        max_open_orders_per_symbol=2,
        daily_loss_cap=100.0,
        api_fail_limit=2,
        api_cooldown_seconds=60,
        trade_min_interval_seconds=8.0,
        trade_min_interval_global_seconds=1.0,
        max_slippage_bps=35.0,
        max_fill_deviation_bps=45.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_position_limit_blocks_projected_qty():
    guard = LiveExecutionGuard(_cfg(max_position_qty_per_symbol=5.0))
    guard.record_fill(symbol="AAPL", action="BUY", quantity=5.0, fill_price=100.0, now_ts=1000.0)

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "position_limit"


def test_daily_loss_cap_blocks_new_trade():
    guard = LiveExecutionGuard(_cfg(daily_loss_cap=50.0))
    guard.record_realized_pnl(-60.0, now_ts=1000.0)

    decision = guard.pre_trade_check(
        symbol="MSFT",
        action="BUY",
        quantity=1.0,
        reference_price=200.0,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "daily_loss_cap"


def test_trade_throttle_symbol_blocks_fast_reentry():
    guard = LiveExecutionGuard(_cfg(trade_min_interval_seconds=10.0, trade_min_interval_global_seconds=0.0))

    first = guard.pre_trade_check(
        symbol="NVDA",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1000.0,
    )
    assert first.ok is True
    guard.mark_trade_submitted(symbol="NVDA", now_ts=1000.0)

    second = guard.pre_trade_check(
        symbol="NVDA",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1005.0,
    )
    assert second.ok is False
    assert second.gate == "trade_throttle_symbol"


def test_api_failure_guard_trips_circuit_breaker():
    guard = LiveExecutionGuard(_cfg(api_fail_limit=2, api_cooldown_seconds=30))

    assert guard.allow_api_call("broker_api") is True
    tripped_1 = guard.record_api_failure("broker_api")
    tripped_2 = guard.record_api_failure("broker_api")

    assert tripped_1 is False
    assert tripped_2 is True
    assert guard.allow_api_call("broker_api") is False


def test_open_order_limits_enforced():
    guard = LiveExecutionGuard(_cfg(max_open_orders_total=2, max_open_orders_per_symbol=1))

    guard.register_open_order(order_id="1", symbol="AAPL", action="BUY", quantity=1.0)

    by_symbol = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1000.0,
    )
    assert by_symbol.ok is False
    assert by_symbol.gate == "open_order_limit_symbol"

    guard.register_open_order(order_id="2", symbol="MSFT", action="BUY", quantity=1.0)
    total = guard.pre_trade_check(
        symbol="NVDA",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1001.0,
    )
    assert total.ok is False
    assert total.gate == "open_order_limit_total"


def test_set_local_position_is_used_by_pre_trade_check():
    guard = LiveExecutionGuard(_cfg(max_position_qty_per_symbol=10.0))
    guard.set_local_position(symbol="AAPL", quantity=9.0, avg_price=100.0)

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=2.0,
        reference_price=100.0,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "position_limit"


def test_slippage_limit_blocks_adverse_buy_price():
    guard = LiveExecutionGuard(_cfg(max_slippage_bps=20.0, trade_min_interval_seconds=0.0, trade_min_interval_global_seconds=0.0))

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        intended_price=100.30,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "slippage_limit"
    assert float(decision.details.get("adverse_slippage_bps", 0.0)) > 20.0


def test_slippage_limit_allows_favorable_sell_price():
    guard = LiveExecutionGuard(_cfg(max_slippage_bps=20.0, trade_min_interval_seconds=0.0, trade_min_interval_global_seconds=0.0))

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="SELL",
        quantity=1.0,
        reference_price=100.0,
        intended_price=100.20,
        now_ts=1010.0,
    )

    assert decision.ok is True


def test_record_fill_includes_modeling_metrics():
    guard = LiveExecutionGuard(_cfg())
    result = guard.record_fill(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        fill_price=100.08,
        reference_price=100.00,
        now_ts=1010.0,
    )

    assert float(result["expected_fill_price"]) > 0.0
    assert float(result["realized_slippage_bps"]) > 0.0
    assert "fill_quality" in result
    assert "fill_deviation_bps" in result["fill_quality"]

    snap = guard.snapshot()
    assert snap["fill_modeling"]["fill_count"] == 1


def test_fill_quality_can_fail_deviation_threshold():
    guard = LiveExecutionGuard(_cfg(max_fill_deviation_bps=5.0))
    quality = guard.evaluate_fill_quality(
        action="BUY",
        actual_fill_price=100.20,
        expected_fill_price=100.00,
    )
    assert quality["ok"] is False
    assert quality["reason"] == "fill_deviation_limit"
    assert float(quality["fill_deviation_bps"]) > 5.0


def test_reconcile_order_lifecycle_detects_mismatch_and_position_break():
    guard = LiveExecutionGuard(_cfg())
    guard.register_open_order(order_id="o1", symbol="AAPL", action="BUY", quantity=1.0)
    guard.set_local_position(symbol="AAPL", quantity=2.0, avg_price=100.0)
    guard.reconcile_broker_position(symbol="AAPL", broker_qty=1.0)

    out = guard.reconcile_order_lifecycle(
        broker_open_orders=[{"order_id": "o2", "symbol": "AAPL"}],
        position_tolerance=0.0001,
    )

    assert out["ok"] is False
    assert out["missing_on_broker"] == ["o1"]
    assert out["missing_local"] == ["o2"]
    assert out["position_checks"]
    assert out["position_checks"][0]["ok"] is False



def test_reconcile_broker_position_marks_manual_adjustment_window():
    guard = LiveExecutionGuard(_cfg())
    guard.set_local_position(symbol="AAPL", quantity=0.0, avg_price=0.0)

    out = guard.reconcile_broker_position(
        symbol="AAPL",
        broker_qty=1.0,
        tolerance=0.0001,
        manual_adjustment_tolerance=2.0,
    )

    assert out["ok"] is False
    assert out["manual_adjustment_detected"] is True
    assert out["status"] == "manual_adjustment_detected"
