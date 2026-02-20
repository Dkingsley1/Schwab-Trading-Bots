import numpy as np

from indicator_bot_common import bollinger, ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    _, mid, up = bollinger(c, window=20, k=2.0)
    dist_mid = (c - mid) / (mid + 1e-8)
    band_span = (up - mid) / (mid + 1e-8)

    mom = ema(r, 10)
    mom_fade = np.maximum(np.abs(ema(r, 4)) - np.abs(mom), 0.0)
    blowoff = np.abs(dist_mid) / (band_span + 1e-8)
    exhaustion = ema(mom_fade + blowoff, 8) / (rolling_std(r, 20) + 1e-8)

    return np.stack([r, mom, dist_mid, band_span, mom_fade, blowoff, exhaustion], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v77_trend_exhaustion_sentinel",
        feature_names=["ret", "mom", "dist_mid", "band_span", "mom_fade", "blowoff", "exhaustion"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
