import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]

    w1 = hold_sample(r, 78)
    w3 = hold_sample(r, 234)
    m1 = ema(w1, 8)
    m3 = ema(w3, 8)
    spread = m1 - m3
    vol = rolling_std(r, 50)
    conviction = np.abs(spread) / (vol + 1e-8)

    return np.stack([r, m1, m3, spread, vol, conviction], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v47_swing_1w_3w",
        feature_names=["ret", "mom_1w", "mom_3w", "spread", "vol", "conviction"],
        feature_builder=build_features,
        window=56,
        horizon=8,
    )
