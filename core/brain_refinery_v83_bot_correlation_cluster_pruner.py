import numpy as np

from indicator_bot_common import ema, macd_line, rolling_std, stochastic_momentum_index, train_indicator_bot, tsi


def rolling_corr(a, b, window=40):
    out = np.zeros_like(a)
    for i in range(len(a)):
        s = max(0, i - window + 1)
        xa = a[s : i + 1]
        xb = b[s : i + 1]
        if len(xa) < 3:
            out[i] = 0.0
            continue
        xa = xa - np.mean(xa)
        xb = xb - np.mean(xb)
        den = np.sqrt(np.sum(xa * xa) * np.sum(xb * xb)) + 1e-8
        out[i] = np.sum(xa * xb) / den
    return out


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    s1 = macd_line(c, 12, 26)
    s2 = tsi(c, 13, 25)
    s3 = stochastic_momentum_index(c, h, l, period=14, smooth=3)
    s4 = ema(r, 20) / (rolling_std(r, 20) + 1e-8)

    c12 = np.abs(rolling_corr(s1, s2, 40))
    c13 = np.abs(rolling_corr(s1, s3, 40))
    c14 = np.abs(rolling_corr(s1, s4, 40))
    c23 = np.abs(rolling_corr(s2, s3, 40))
    c24 = np.abs(rolling_corr(s2, s4, 40))
    c34 = np.abs(rolling_corr(s3, s4, 40))

    cluster_pressure = np.maximum.reduce([c12, c13, c14, c23, c24, c34])
    diversity_score = 1.0 - np.clip(cluster_pressure, 0.0, 1.0)

    return np.stack([r, s1, s2, s3, s4, c12, c13, c14, c23, c24, c34, cluster_pressure, diversity_score], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v83_bot_correlation_cluster_pruner",
        feature_names=[
            "ret",
            "sig1_macd",
            "sig2_tsi",
            "sig3_smi",
            "sig4_trend",
            "corr_12",
            "corr_13",
            "corr_14",
            "corr_23",
            "corr_24",
            "corr_34",
            "cluster_pressure",
            "diversity_score",
        ],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
