import numpy as np

from indicator_bot_common import rolling_std, train_indicator_bot, vwap


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    vwap60 = vwap(c, v, session=60)
    dev = (c - vwap60) / (vwap60 + 1e-8)
    dev_z = dev / (rolling_std(dev, 30) + 1e-8)
    vol_z = (v - np.mean(v)) / (np.std(v) + 1e-8)
    ret_vol = rolling_std(r, 15)

    return np.stack([r, dev, dev_z, vol_z, ret_vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v24_vwap_deviation",
        feature_names=["ret", "vwap_dev", "vwap_dev_z", "volume_z", "ret_vol_15"],
        feature_builder=build_features,
    )
