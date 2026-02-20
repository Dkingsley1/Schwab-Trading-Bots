import numpy as np

from indicator_bot_common import bollinger, ema, rolling_std, train_indicator_bot, vwap


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    ret1 = r
    ret5 = ema(r, 5)
    ret15 = ema(r, 15)
    _, mid, _ = bollinger(c, window=20, k=2.0)
    dist_mid = (c - mid) / (mid + 1e-8)
    vw = vwap(c, v, session=45)
    vdev = (c - vw) / (vw + 1e-8)
    noise = rolling_std(ret1, 20)

    return np.stack([ret1, ret5, ret15, dist_mid, vdev, noise], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v44_intraday_scalp_1m_5m",
        feature_names=["ret1", "ret5", "ret15", "dist_mid", "vwap_dev", "noise"],
        feature_builder=build_features,
        window=36,
        horizon=2,
    )
