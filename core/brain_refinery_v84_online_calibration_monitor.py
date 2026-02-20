import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    signal_score = ema(r - rb, 12) / (rolling_std(r - rb, 20) + 1e-8)
    p_up = np.clip(sigmoid(signal_score), 1e-4, 1.0 - 1e-4)

    realized = (r > 0).astype(float)
    brier = (p_up - realized) ** 2
    brier_trend = ema(brier, 12)

    conf = np.abs(p_up - 0.5) * 2.0
    overconf_penalty = np.maximum(conf - (1.0 - np.clip(2.0 * np.sqrt(brier_trend), 0.0, 1.0)), 0.0)
    calibration_score = 1.0 - np.clip(brier_trend + 0.5 * overconf_penalty, 0.0, 1.0)

    return np.stack([r, rb, signal_score, p_up, realized, brier, brier_trend, conf, overconf_penalty, calibration_score], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v84_online_calibration_monitor",
        feature_names=[
            "ret",
            "bench_ret",
            "signal_score",
            "p_up",
            "realized_up",
            "brier",
            "brier_trend",
            "confidence",
            "overconf_penalty",
            "calibration_score",
        ],
        feature_builder=build_features,
        window=42,
        horizon=3,
    )
