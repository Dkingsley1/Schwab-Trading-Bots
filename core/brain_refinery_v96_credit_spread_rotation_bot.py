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


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    rel = q_ret - q_bench

    rel_fast = ema(rel, 6)
    rel_slow = ema(rel, 16)
    spread_mom = rel_fast - rel_slow

    dd = rolling_drawdown(c, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)

    vol = rolling_std(r, 120)
    tail_risk = np.maximum(-dd_fast, 0.0) + np.maximum(vol - ema(vol, 20), 0.0)

    # Credit sleeve proxy: prefer widening relative strength, avoid tail-risk spikes.
    credit_rotation = spread_mom - 0.45 * tail_risk

    return np.stack([r, rel, rel_fast, rel_slow, spread_mom, dd, dd_fast, dd_slow, vol, tail_risk, credit_rotation], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v96_credit_spread_rotation_bot",
        feature_names=[
            "ret",
            "rel",
            "rel_fast",
            "rel_slow",
            "spread_mom",
            "drawdown",
            "dd_fast",
            "dd_slow",
            "vol",
            "tail_risk",
            "credit_rotation",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )
