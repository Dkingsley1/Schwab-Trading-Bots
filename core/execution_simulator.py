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
    partial_fill_ratio: float
    spread_jump_penalty_bps: float
    symbol_curve_multiplier: float
    fill_quality_bucket: str
    queue_position_ratio: float = 0.0
    cancel_probability: float = 0.0
    borrow_fee_bps: float = 0.0
    venue_rule_penalty_bps: float = 0.0
    venue: str = ""
    queue_priority_score: float = 0.0
    requote_probability: float = 0.0
    session_penalty_bps: float = 0.0
    crowding_penalty_bps: float = 0.0
    spread_regime: str = ""
    latency_bucket: str = ""
    session: str = ""
    order_type: str = ""


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


def _resolve_venue(broker: str, market_kind: str) -> str:
    broker_key = str(broker or "").strip().lower()
    market_key = str(market_kind or "").strip().lower()
    if broker_key and market_key:
        return f"{broker_key}_{market_key}"
    if broker_key:
        return broker_key
    if market_key:
        return market_key
    return "default"


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
    session: str = "regular",
    order_type: str = "market",
    live_fill_slippage_bps: float = 0.0,
) -> ExecutionSimResult:
    action = (action or "HOLD").upper()
    buy_like_actions = {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}
    sell_like_actions = {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}
    price = float(last_price or 0.0)
    ret = float(return_1m or 0.0)
    spread = max(float(spread_bps or 0.0), 0.0)
    vol = max(float(volatility_1m or 0.0), 0.0)
    resolved_market_kind = str(market_kind or "").strip().lower()
    if not resolved_market_kind:
        resolved_market_kind = "crypto" if str(broker or "").strip().lower() == "coinbase" else "equities"
    venue = _resolve_venue(broker, resolved_market_kind)
    short_side = action in {"SELL_SHORT", "SELL_TO_OPEN"}
    session_key = str(session or "regular").strip().lower()
    order_type_key = str(order_type or "market").strip().lower()

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
    symbol_curve_multiplier = max(
        _sim_env_value(
            "EXEC_SIM_SYMBOL_CURVE_MULTIPLIER",
            1.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.25,
    )
    borrow_fee_bps = max(
        _sim_env_value(
            "EXEC_SIM_BASE_BORROW_FEE_BPS",
            6.0 if short_side and resolved_market_kind == "equities" else 0.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )
    venue_rule_penalty_bps = max(
        _sim_env_value(
            "EXEC_SIM_VENUE_RULE_PENALTY_BPS",
            2.5 if short_side and resolved_market_kind == "equities" else 0.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )
    session_penalty_bps = max(
        _sim_env_value(
            "EXEC_SIM_SESSION_PENALTY_BPS",
            3.0 if session_key in {"open", "close", "overnight_gap"} else 0.0,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )
    order_type_penalty_bps = max(
        _sim_env_value(
            "EXEC_SIM_ORDER_TYPE_PENALTY_BPS",
            1.5 if order_type_key == "market" else 0.6,
            broker=broker,
            market_kind=resolved_market_kind,
            symbol=symbol,
        ),
        0.0,
    )

    if action not in buy_like_actions.union(sell_like_actions):
        return ExecutionSimResult(
            action=action,
            expected_fill_price=price,
            slippage_bps=0.0,
            latency_ms=latency_ms,
            adjusted_return_1m=0.0,
            impact_bps=0.0,
            fee_bps=0.0,
            partial_fill_ratio=1.0,
            spread_jump_penalty_bps=0.0,
            symbol_curve_multiplier=float(symbol_curve_multiplier),
            fill_quality_bucket="none",
            queue_position_ratio=0.0,
            cancel_probability=0.0,
            borrow_fee_bps=float(borrow_fee_bps if short_side else 0.0),
            venue_rule_penalty_bps=float(venue_rule_penalty_bps if short_side else 0.0),
            venue=venue,
            queue_priority_score=0.0,
            requote_probability=0.0,
            session_penalty_bps=float(session_penalty_bps),
            crowding_penalty_bps=0.0,
            spread_regime="none",
            latency_bucket="none",
            session=session_key,
            order_type=order_type_key,
        )

    is_buy_like = action in buy_like_actions
    half_spread = spread * 0.5
    vol_impact = min(vol * 10000.0 * 0.12 * vol_mult * slippage_scale, 25.0 * max(vol_mult * slippage_scale, 1.0))
    latency_impact = min(
        (latency_ms / 1000.0) * max(vol * 10000.0, 0.0) * 0.05 * latency_mult * slippage_scale,
        12.0 * max(latency_mult * slippage_scale, 1.0),
    )

    depth_same_side = max(float(ask_size if is_buy_like else bid_size), 0.0)
    size = max(float(order_size or 1.0), 0.0)
    if depth_same_side <= 0.0:
        depth_impact = 8.0 * depth_mult * slippage_scale
        queue_position_ratio = 1.0
    else:
        participation = min(size / depth_same_side, 5.0)
        depth_impact = min(
            participation * 6.0 * depth_mult * slippage_scale,
            18.0 * max(depth_mult * slippage_scale, 1.0),
        )
        queue_position_ratio = max(0.0, min(participation / (1.0 + participation), 1.0))

    spread_jump_penalty_bps = min(
        max(spread - 12.0, 0.0) * 0.35 * slippage_scale * symbol_curve_multiplier,
        16.0 * max(symbol_curve_multiplier, 1.0),
    )
    crowding_penalty_bps = min(
        max(depth_impact - 4.0, 0.0) * 0.45 + max(live_fill_slippage_bps, 0.0) * 0.25,
        18.0,
    )
    queue_priority_score = max(0.0, min((1.0 - queue_position_ratio) * max(1.0 - min(spread / 40.0, 1.0), 0.0), 1.0))
    cancel_probability = max(
        0.0,
        min(
            0.02
            + (0.18 * queue_position_ratio)
            + (0.025 * min(latency_ms / 500.0, 8.0))
            + (0.40 * min(vol, 1.0))
            + (0.01 * max(spread - 8.0, 0.0) / 4.0),
            0.95,
        ),
    )
    requote_probability = max(
        0.0,
        min(
            0.01
            + (0.22 * min(spread / 30.0, 1.0))
            + (0.12 * min(latency_ms / 750.0, 1.0))
            + (0.10 * min(vol * 8.0, 1.0)),
            0.85,
        ),
    )
    total_bps = (
        half_spread
        + fee_bps
        + borrow_fee_bps
        + venue_rule_penalty_bps
        + session_penalty_bps
        + order_type_penalty_bps
        + vol_impact
        + latency_impact
        + depth_impact
        + crowding_penalty_bps
        + spread_jump_penalty_bps
        + max(live_fill_slippage_bps, 0.0)
    ) * symbol_curve_multiplier
    if depth_same_side <= 0.0:
        partial_fill_ratio = 0.45
    else:
        partial_fill_ratio = max(0.35, min(1.0, depth_same_side / max(size, 1e-6)))
    if total_bps <= 8.0:
        fill_quality_bucket = "excellent"
    elif total_bps <= 18.0:
        fill_quality_bucket = "good"
    elif total_bps <= 35.0:
        fill_quality_bucket = "fair"
    else:
        fill_quality_bucket = "poor"
    spread_regime = "wide" if spread >= 20.0 else ("normal" if spread >= 6.0 else "tight")
    latency_bucket = "slow" if latency_ms >= 400.0 else ("watch" if latency_ms >= 180.0 else "fast")

    fill_mult = 1.0 + (total_bps / 10000.0) if is_buy_like else 1.0 - (total_bps / 10000.0)
    fill_price = price * fill_mult if price > 0 else price

    drag = total_bps / 10000.0
    adjusted_ret = (ret - drag) if is_buy_like else ((-ret) - drag)

    return ExecutionSimResult(
        action=action,
        expected_fill_price=fill_price,
        slippage_bps=total_bps,
        latency_ms=latency_ms,
        adjusted_return_1m=adjusted_ret,
        impact_bps=depth_impact,
        fee_bps=fee_bps,
        partial_fill_ratio=float(partial_fill_ratio),
        spread_jump_penalty_bps=float(spread_jump_penalty_bps),
        symbol_curve_multiplier=float(symbol_curve_multiplier),
        fill_quality_bucket=fill_quality_bucket,
        queue_position_ratio=float(queue_position_ratio),
        cancel_probability=float(cancel_probability),
        borrow_fee_bps=float(borrow_fee_bps),
        venue_rule_penalty_bps=float(venue_rule_penalty_bps),
        venue=venue,
        queue_priority_score=float(queue_priority_score),
        requote_probability=float(requote_probability),
        session_penalty_bps=float(session_penalty_bps + order_type_penalty_bps),
        crowding_penalty_bps=float(crowding_penalty_bps),
        spread_regime=spread_regime,
        latency_bucket=latency_bucket,
        session=session_key,
        order_type=order_type_key,
    )
