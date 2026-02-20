import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        j = (i // step) * step
        out[i] = x[j]
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    # Longer interval context proxies (e.g., 15m/30m/60m style states).
    r_15 = hold_sample(r, 3)
    r_30 = hold_sample(r, 6)
    r_60 = hold_sample(r, 12)

    t_15 = ema(r_15, 10)
    t_30 = ema(r_30, 10)
    t_60 = ema(r_60, 10)

    long_drift = ema(np.diff(c, prepend=c[0]) / (c + 1e-8), 30)
    regime_strength = (np.abs(t_30) + np.abs(t_60)) / (rolling_std(r, 30) + 1e-8)

    align_15_60 = np.sign(t_15) * np.sign(t_60)
    align_30_60 = np.sign(t_30) * np.sign(t_60)

    return np.stack(
        [
            r,
            t_15,
            t_30,
            t_60,
            long_drift,
            regime_strength,
            align_15_60,
            align_30_60,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v41_long_interval_trend",
        feature_names=[
            "ret",
            "trend_15",
            "trend_30",
            "trend_60",
            "long_drift",
            "regime_strength",
            "align_15_60",
            "align_30_60",
        ],
        feature_builder=build_features,
        window=36,
        horizon=4,
    )
