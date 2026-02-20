import numpy as np

from indicator_bot_common import atr, ema, rolling_std, train_indicator_bot, vwap


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]

    atr_pct = atr(h, l, c, period=14) / (c + 1e-8)
    ret_ema = ema(r, 8)
    ret_vol = rolling_std(r, 20)
    downside = np.where(r < 0, r * r, 0.0)
    downside_risk = np.sqrt(rolling_std(downside, 20) + 1e-8)

    vw = vwap(c, v, session=60)
    vw_dev = (c - vw) / (vw + 1e-8)

    return np.stack([r, ret_ema, atr_pct, ret_vol, downside_risk, vw_dev], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v30_low_vol_drift",
        feature_names=["ret", "ret_ema8", "atr_pct", "ret_vol20", "downside_risk", "vwap_dev"],
        feature_builder=build_features,
    )
