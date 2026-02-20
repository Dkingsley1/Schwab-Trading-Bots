import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    # Consensus proxy between close path and range/derived estimates.
    mid = 0.5 * (h + l)
    range_proxy = (h - l) / (c + 1e-8)
    close_vs_mid = (c - mid) / (mid + 1e-8)

    expected = ema(r, 12)
    observed = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    disagreement = np.abs(observed - expected)

    consensus_error = close_vs_mid * close_vs_mid + disagreement + 0.3 * range_proxy
    consensus_rate = ema(consensus_error, 10)
    anomaly = consensus_rate / (rolling_std(consensus_rate, 30) + 1e-8)

    return np.stack([r, close_vs_mid, range_proxy, disagreement, consensus_rate, anomaly], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v73_feed_consensus_validator",
        feature_names=["ret", "close_vs_mid", "range_proxy", "disagreement", "consensus_rate", "anomaly"],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
