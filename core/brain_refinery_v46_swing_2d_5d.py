import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    c = panel["close"]

    d2 = hold_sample(r, 16)
    d5 = hold_sample(r, 40)
    m2 = ema(d2, 8)
    m5 = ema(d5, 8)
    drift = ema(np.diff(c, prepend=c[0]) / (c + 1e-8), 30)
    vol = rolling_std(r, 30)
    align = np.sign(m2) * np.sign(m5)

    return np.stack([r, m2, m5, drift, vol, align], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v46_swing_2d_5d",
        feature_names=["ret", "mom_2d", "mom_5d", "drift", "vol", "align"],
        feature_builder=build_features,
        window=48,
        horizon=6,
    )
