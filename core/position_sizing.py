import math
import os


def kelly_lite_fraction(*, win_rate: float, payoff_ratio: float, floor: float = 0.0, cap: float = 0.35) -> float:
    p = min(max(float(win_rate), 0.0), 1.0)
    b = max(float(payoff_ratio), 1e-6)
    q = 1.0 - p
    raw = (b * p - q) / b
    return min(max(raw, floor), cap)


def volatility_target_scale(*, volatility_1m: float, target_vol_1m: float, min_scale: float = 0.25, max_scale: float = 2.0) -> float:
    vol = max(float(volatility_1m), 1e-6)
    target = max(float(target_vol_1m), 1e-6)
    scale = target / vol
    return min(max(scale, min_scale), max_scale)


def compute_position_size(
    *,
    score: float,
    threshold: float,
    volatility_1m: float,
    equity_proxy: float,
    max_notional_pct: float,
) -> float:
    edge = max(float(score) - float(threshold), 0.0)
    win_rate = min(max(0.5 + edge, 0.5), 0.9)
    payoff = float(os.getenv("SIZING_PAYOFF_RATIO", "1.3"))
    kf = kelly_lite_fraction(win_rate=win_rate, payoff_ratio=payoff, cap=float(os.getenv("SIZING_KELLY_CAP", "0.18")))

    vol_scale = volatility_target_scale(
        volatility_1m=float(volatility_1m),
        target_vol_1m=float(os.getenv("SIZING_TARGET_VOL_1M", "0.012")),
        min_scale=float(os.getenv("SIZING_MIN_SCALE", "0.25")),
        max_scale=float(os.getenv("SIZING_MAX_SCALE", "1.75")),
    )

    notional_pct = min(max(kf * vol_scale, 0.0), float(max_notional_pct))
    qty = max((float(equity_proxy) * notional_pct), 0.0)
    return round(qty, 6)


def size_from_action(
    *,
    action: str,
    score: float,
    threshold: float,
    volatility_1m: float,
    equity_proxy: float,
    max_notional_pct: float,
) -> float:
    if (action or "HOLD").upper() not in {"BUY", "SELL"}:
        return 0.0
    return compute_position_size(
        score=score,
        threshold=threshold,
        volatility_1m=volatility_1m,
        equity_proxy=equity_proxy,
        max_notional_pct=max_notional_pct,
    )
