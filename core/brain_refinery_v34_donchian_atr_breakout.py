import numpy as np

from indicator_bot_common import atr, rolling_std, train_indicator_bot


def donchian(high, low, window=20):
    up = np.zeros_like(high)
    dn = np.zeros_like(low)
    for i in range(len(high)):
        start = max(0, i - window + 1)
        up[i] = np.max(high[start : i + 1])
        dn[i] = np.min(low[start : i + 1])
    return up, dn


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    up, dn = donchian(h, l, window=20)
    width = (up - dn) / (c + 1e-8)
    breakout_up = (c - up) / (up + 1e-8)
    breakout_dn = (c - dn) / (dn + 1e-8)

    atr14 = atr(h, l, c, period=14) / (c + 1e-8)
    vol20 = rolling_std(r, 20)
    breakout_quality = (np.abs(breakout_up) + np.abs(breakout_dn)) / (atr14 + 1e-8)

    return np.stack([r, width, breakout_up, breakout_dn, atr14, vol20, breakout_quality], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v34_donchian_atr_breakout",
        feature_names=["ret", "donchian_width", "breakout_up", "breakout_dn", "atr14_pct", "vol20", "breakout_quality"],
        feature_builder=build_features,
    )
