from dataclasses import dataclass


@dataclass
class ExecutionSimResult:
    action: str
    expected_fill_price: float
    slippage_bps: float
    latency_ms: float
    adjusted_return_1m: float
    impact_bps: float


def simulate_execution(
    *,
    action: str,
    last_price: float,
    return_1m: float,
    spread_bps: float,
    volatility_1m: float,
    latency_ms: float = 120.0,
    bid_size: float = 0.0,
    ask_size: float = 0.0,
    order_size: float = 1.0,
) -> ExecutionSimResult:
    action = (action or "HOLD").upper()
    price = float(last_price or 0.0)
    ret = float(return_1m or 0.0)
    spread = max(float(spread_bps or 0.0), 0.0)
    vol = max(float(volatility_1m or 0.0), 0.0)

    if action not in {"BUY", "SELL"}:
        return ExecutionSimResult(
            action=action,
            expected_fill_price=price,
            slippage_bps=0.0,
            latency_ms=latency_ms,
            adjusted_return_1m=0.0,
            impact_bps=0.0,
        )

    half_spread = spread * 0.5
    vol_impact = min(vol * 10000.0 * 0.12, 25.0)
    latency_impact = min((latency_ms / 1000.0) * max(vol * 10000.0, 0.0) * 0.05, 12.0)

    depth_same_side = max(float(ask_size if action == "BUY" else bid_size), 0.0)
    size = max(float(order_size or 1.0), 0.0)
    if depth_same_side <= 0.0:
        depth_impact = 8.0
    else:
        participation = min(size / depth_same_side, 5.0)
        depth_impact = min(participation * 6.0, 18.0)

    total_bps = half_spread + vol_impact + latency_impact + depth_impact

    fill_mult = 1.0 + (total_bps / 10000.0) if action == "BUY" else 1.0 - (total_bps / 10000.0)
    fill_price = price * fill_mult if price > 0 else price

    drag = total_bps / 10000.0
    adjusted_ret = (ret - drag) if action == "BUY" else ((-ret) - drag)

    return ExecutionSimResult(
        action=action,
        expected_fill_price=fill_price,
        slippage_bps=total_bps,
        latency_ms=latency_ms,
        adjusted_return_1m=adjusted_ret,
        impact_bps=depth_impact,
    )
