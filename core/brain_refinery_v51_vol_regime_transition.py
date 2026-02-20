import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]

    vol_fast = rolling_std(r, 10)
    vol_slow = rolling_std(r, 40)
    vol_ratio = vol_fast / (vol_slow + 1e-8)

    vol_accel = np.diff(vol_fast, prepend=vol_fast[0])
    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    regime = ema(vol_ratio + np.maximum(vix_chg, 0.0), 8)

    return np.stack([r, vol_fast, vol_slow, vol_ratio, vol_accel, vix_chg, regime], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v51_vol_regime_transition",
        feature_names=["ret", "vol_fast", "vol_slow", "vol_ratio", "vol_accel", "vix_chg", "regime"],
        feature_builder=build_features,
        window=42,
        horizon=3,
    )
