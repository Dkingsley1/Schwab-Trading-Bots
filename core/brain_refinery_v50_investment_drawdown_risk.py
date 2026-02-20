import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_drawdown(close, window=200):
    dd = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        peak = np.max(close[start : i + 1])
        dd[i] = (close[i] - peak) / (peak + 1e-8)
    return dd


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]

    dd = rolling_drawdown(c, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)
    crash_prob = np.abs(dd_fast - dd_slow)

    qret = hold_sample(r, 1170)
    qbeta = hold_sample(b, 1170)
    qalpha = qret - qbeta
    vol = rolling_std(r, 100)

    return np.stack([r, dd, dd_fast, dd_slow, crash_prob, qalpha, vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v50_investment_drawdown_risk",
        feature_names=["ret", "drawdown", "dd_fast", "dd_slow", "crash_prob", "qalpha", "vol"],
        feature_builder=build_features,
        window=72,
        horizon=20,
    )
