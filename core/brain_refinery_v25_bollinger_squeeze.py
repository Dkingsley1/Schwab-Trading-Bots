import numpy as np

from indicator_bot_common import bollinger, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    lower, mid, upper = bollinger(c, window=20, k=2.0)
    bandwidth = (upper - lower) / (mid + 1e-8)
    percent_b = (c - lower) / ((upper - lower) + 1e-8)
    squeeze = bandwidth / (rolling_std(bandwidth, 40) + 1e-8)
    breakout_proxy = np.diff(percent_b, prepend=percent_b[0])

    return np.stack([r, bandwidth, percent_b, squeeze, breakout_proxy], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v25_bollinger_squeeze",
        feature_names=["ret", "bb_bandwidth", "percent_b", "squeeze_ratio", "breakout_proxy"],
        feature_builder=build_features,
    )
