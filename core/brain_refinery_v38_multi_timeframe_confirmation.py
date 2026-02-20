import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def downsample(x, step):
    y = np.zeros_like(x)
    for i in range(len(x)):
        j = (i // step) * step
        y[i] = x[j]
    return y


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    # proxy 5m, 30m, 60m states from synthetic stream
    ret_5 = r
    ret_30 = downsample(r, 6)
    ret_60 = downsample(r, 12)

    trend_5 = ema(ret_5, 8)
    trend_30 = ema(ret_30, 8)
    trend_60 = ema(ret_60, 8)

    confirm_30 = np.sign(trend_5) * np.sign(trend_30)
    confirm_60 = np.sign(trend_5) * np.sign(trend_60)
    disagreement = (confirm_30 < 0).astype(float) + (confirm_60 < 0).astype(float)
    noise = rolling_std(ret_5, 20)

    return np.stack([ret_5, trend_5, trend_30, trend_60, confirm_30, confirm_60, disagreement, noise], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v38_multi_timeframe_confirmation",
        feature_names=["ret_5", "trend_5", "trend_30", "trend_60", "confirm_30", "confirm_60", "disagreement", "noise"],
        feature_builder=build_features,
    )
