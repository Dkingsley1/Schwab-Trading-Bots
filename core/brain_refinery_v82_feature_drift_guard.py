import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def drift_z(x, fast=20, slow=120):
    mu_f = rolling_mean(x, fast)
    mu_s = rolling_mean(x, slow)
    sig_s = rolling_std(x, slow) + 1e-8
    return (mu_f - mu_s) / sig_s


def build_features(panel):
    r = panel["ret"]
    v = panel["volume"]
    vix = panel["vix"]
    c = panel["close"]

    vol = rolling_std(r, 20)
    rv = v / (rolling_mean(v, 40) + 1e-8)
    price_z = (c - rolling_mean(c, 60)) / (rolling_std(c, 60) + 1e-8)

    drift_ret = drift_z(r)
    drift_vol = drift_z(vol)
    drift_flow = drift_z(rv)
    drift_vix = drift_z(vix)
    drift_price = drift_z(price_z)

    drift_pressure = ema(
        np.abs(drift_ret) + np.abs(drift_vol) + np.abs(drift_flow) + np.abs(drift_vix) + np.abs(drift_price),
        8,
    )
    drift_guard_score = 1.0 - np.clip(0.20 * drift_pressure, 0.0, 1.0)

    return np.stack(
        [r, vol, rv, price_z, drift_ret, drift_vol, drift_flow, drift_vix, drift_price, drift_pressure, drift_guard_score],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v82_feature_drift_guard",
        feature_names=[
            "ret",
            "vol",
            "rel_vol",
            "price_z",
            "drift_ret",
            "drift_vol",
            "drift_flow",
            "drift_vix",
            "drift_price",
            "drift_pressure",
            "drift_guard_score",
        ],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )
