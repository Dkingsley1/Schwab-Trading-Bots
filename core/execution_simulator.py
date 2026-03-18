import os
import re
from dataclasses import dataclass


@dataclass
class ExecutionSimResult:
    action: str
    expected_fill_price: float
    slippage_bps: float
    latency_ms: float
    adjusted_return_1m: float
    impact_bps: float
    fee_bps: float


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _sim_env_value(base: str, default: float, *, broker: str = "", market_kind: str = "", symbol: str = "") -> float:
    symbol_key = re.sub(r"[^A-Z0-9]+", "_", str(symbol or "").upper()).strip("_")
    market_key = str(market_kind or "").strip().upper()
    broker_key = str(broker or "").strip().upper()
    names = []
    if symbol_key:
        names.append(f"{base}_{symbol_key}")
    if market_key:
        names.append(f"{base}_{market_key}")
    if broker_key:
        names.append(f"{base}_{broker_key}")
    names.append(base)
    value = float(default)
    for name in names:
        value = _env_float(name, value)
    return value


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
    broker: str = "",
    market_kind: str = "",
    symbol: str = "",
) -> ExecutionSimResult:
    action = (action or "HOLD").upper()
    price = float(last_price or 0.0)
    ret = float(return_1m or 0.0)
    spread = max(float(spread_bps or 0.0), 0.0)
    vol = max(float(volatility_1m or 0.0), 0.0)
    resolved_market_kind = str(market_kind or "").strip().lower()
    if not resolved_market_kind:
        resolved_market_kind = "crypto" if str(broker or "").strip().lower() == "coinbase" else "equities"

    slippage_scale = max(
        _sim_env_value(
            "EXEC_SIM_SLIPPAGE_SCALE",
            1.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.1,
    )
    vol_mult = max(
        _sim_env_value(
            "EXEC_SIM_VOL_IMPACT_MULTIPLIER",
            1.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )
    latency_mult = max(
        _sim_env_value(
            "EXEC_SIM_LATENCY_IMPACT_MULTIPLIER",
            1.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )
    depth_mult = max(
        _sim_env_value(
            "EXEC_SIM_DEPTH_IMPACT_MULTIPLIER",
            1.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )
    fee_bps = max(
        _sim_env_value(
            "EXEC_SIM_BASE_FEE_BPS",
            1.2 if resolved_market_kind == "crypto" else 0.15,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )

    if action not in {"BUY", "SELL"}:
        return ExecutionSimResult(
            action=action,
            expected_fill_price=price,
            slippage_bps=0.0,
            latency_ms=latency_ms,
            adjusted_return_1m=0.0,
            impact_bps=0.0,
            fee_bps=0.0,
        )

    half_spread = spread * 0.5
    vol_impact = min(vol * 10000.0 * 0.12 * vol_mult * slippage_scale, 25.0 * max(vol_mult * slippage_scale, 1.0))
    latency_impact = min(
        (latency_ms / 1000.0) * max(vol * 10000.0, 0.0) * 0.05 * latency_mult * slippage_scale,
        12.0 * max(latency_mult * slippage_scale, 1.0),
    )

    depth_same_side = max(float(ask_size if action == "BUY" else bid_size), 0.0)
    size = max(float(order_size or 1.0), 0.0)
    if depth_same_side <= 0.0:
        depth_impact = 8.0 * depth_mult * slippage_scale
    else:
        participation = min(size / depth_same_side, 5.0)
        depth_impact = min(
            participation * 6.0 * depth_mult * slippage_scale,
            18.0 * max(depth_mult * slippage_scale, 1.0),
        )

    total_bps = half_spread + fee_bps + vol_impact + latency_impact + depth_impact

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
        fee_bps=fee_bps,
    )
