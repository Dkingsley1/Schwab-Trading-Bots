import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    alpha = r - rb
    rs_fast = ema(alpha, 8)
    rs_slow = ema(alpha, 30)
    rs_spread = rs_fast - rs_slow

    week_alpha = hold_sample(alpha, 78)
    month_alpha = hold_sample(alpha, 390)
    rot_pressure = ema(week_alpha + month_alpha, 10)
    rs_noise = rolling_std(alpha, 30)

    return np.stack([r, rb, alpha, rs_fast, rs_slow, rs_spread, rot_pressure, rs_noise], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v52_sector_rotation_rs",
        feature_names=["ret", "bench_ret", "alpha", "rs_fast", "rs_slow", "rs_spread", "rot_pressure", "rs_noise"],
        feature_builder=build_features,
        window=48,
        horizon=5,
    )
