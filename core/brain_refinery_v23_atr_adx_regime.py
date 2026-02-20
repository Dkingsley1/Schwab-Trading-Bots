import numpy as np

from indicator_bot_common import adx, atr, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    atr14 = atr(h, l, c, period=14)
    atr_pct = atr14 / (c + 1e-8)
    adx14 = adx(h, l, c, period=14)
    vol20 = rolling_std(r, 20)
    trend_proxy = np.diff(c, prepend=c[0]) / (c + 1e-8)

    return np.stack([r, atr_pct, adx14, vol20, trend_proxy], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v23_atr_adx_regime",
        feature_names=["ret", "atr_pct", "adx14", "vol20", "trend_proxy"],
        feature_builder=build_features,
    )
