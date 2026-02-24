import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def rolling_drawdown(close, window=220):
    out = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        peak = np.max(close[start : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def downside_vol(x, window=120):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        seg = x[start : i + 1]
        neg = np.minimum(seg, 0.0)
        out[i] = np.sqrt(np.mean(neg * neg))
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]

    dd = rolling_drawdown(c, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)

    vol = rolling_std(r, 120)
    dvol = downside_vol(r, window=120)

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    q_alpha = q_ret - q_bench

    # Trap proxy: high downside stress + weak relative trend.
    trap_score = np.maximum(-dd_fast, 0.0) + np.maximum(-q_alpha, 0.0) + np.maximum(dvol - vol, 0.0)
    recovery = ema(r, 13)

    return np.stack([r, dd, dd_fast, dd_slow, q_alpha, vol, dvol, trap_score, recovery], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v94_dividend_yield_trap_avoidance",
        feature_names=[
            "ret",
            "drawdown",
            "dd_fast",
            "dd_slow",
            "q_alpha",
            "vol",
            "downside_vol",
            "trap_score",
            "recovery",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )
