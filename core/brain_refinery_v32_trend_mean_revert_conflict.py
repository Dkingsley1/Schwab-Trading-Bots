import numpy as np

from indicator_bot_common import bollinger, ema, macd_line, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    macd = macd_line(c, fast=12, slow=26)
    trend = ema(r, 12)

    lower, mid, upper = bollinger(c, window=20, k=2.0)
    dist_mid = (c - mid) / (mid + 1e-8)
    z_band = (c - mid) / ((upper - lower) + 1e-8)

    trend_strength = np.abs(trend) / (rolling_std(r, 20) + 1e-8)
    conflict = np.abs(macd) * np.abs(z_band)

    return np.stack([r, macd, trend, dist_mid, z_band, trend_strength, conflict], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v32_trend_mean_revert_conflict",
        feature_names=["ret", "macd", "trend_ema12", "dist_mid", "z_band", "trend_strength", "conflict"],
        feature_builder=build_features,
    )
