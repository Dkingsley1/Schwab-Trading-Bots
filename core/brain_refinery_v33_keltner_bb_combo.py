import numpy as np

from indicator_bot_common import atr, bollinger, ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    lower, mid, upper = bollinger(c, window=20, k=2.0)
    bb_width = (upper - lower) / (mid + 1e-8)

    atr20 = atr(h, l, c, period=20)
    kel_upper = ema(c, 20) + 2.0 * atr20
    kel_lower = ema(c, 20) - 2.0 * atr20
    kel_width = (kel_upper - kel_lower) / (ema(c, 20) + 1e-8)

    squeeze_ratio = bb_width / (kel_width + 1e-8)
    percent_b = (c - lower) / ((upper - lower) + 1e-8)
    breakout_energy = np.abs(np.diff(percent_b, prepend=percent_b[0])) / (rolling_std(r, 20) + 1e-8)

    return np.stack([r, bb_width, kel_width, squeeze_ratio, percent_b, breakout_energy], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v33_keltner_bb_combo",
        feature_names=["ret", "bb_width", "keltner_width", "squeeze_ratio", "percent_b", "breakout_energy"],
        feature_builder=build_features,
    )
