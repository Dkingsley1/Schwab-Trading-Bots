import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]

    bad_range = ((h < l) | (h <= 0) | (l <= 0) | (c <= 0)).astype(float)
    range_proxy = (h - l) / (c + 1e-8)

    jump = np.abs(np.diff(c, prepend=c[0])) / (c + 1e-8)
    jump_z = jump / (rolling_std(jump, 40) + 1e-8)
    extreme_jump = (jump_z > 4.0).astype(float)

    rel_vol = v / (rolling_mean(v, 40) + 1e-8)
    volume_void = (rel_vol < 0.20).astype(float)

    stale = (np.abs(np.diff(c, prepend=c[0])) < 1e-8).astype(float)
    stale_rate = ema(stale, 10)

    integrity_pressure = ema(
        0.35 * bad_range + 0.25 * extreme_jump + 0.20 * volume_void + 0.20 * stale_rate,
        8,
    )
    integrity_score = 1.0 - np.clip(integrity_pressure, 0.0, 1.0)

    return np.stack(
        [r, bad_range, range_proxy, jump, jump_z, extreme_jump, rel_vol, volume_void, stale_rate, integrity_pressure, integrity_score],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v87_data_integrity_sentinel_v2",
        feature_names=[
            "ret",
            "bad_range",
            "range_proxy",
            "jump",
            "jump_z",
            "extreme_jump",
            "rel_vol",
            "volume_void",
            "stale_rate",
            "integrity_pressure",
            "integrity_score",
        ],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
