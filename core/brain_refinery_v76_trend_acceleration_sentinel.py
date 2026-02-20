import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]

    t_fast = ema(r, 6)
    t_mid = ema(r, 14)
    t_slow = ema(r, 30)

    accel = np.diff(t_fast, prepend=t_fast[0])
    jerk = np.diff(accel, prepend=accel[0])
    trend_spread = t_fast - t_slow
    trend_power = np.abs(trend_spread) / (rolling_std(r, 20) + 1e-8)

    return np.stack([r, t_fast, t_mid, t_slow, accel, jerk, trend_spread, trend_power], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v76_trend_acceleration_sentinel",
        feature_names=["ret", "t_fast", "t_mid", "t_slow", "accel", "jerk", "trend_spread", "trend_power"],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )
