import numpy as np

from indicator_bot_common import bollinger, ema, rolling_std, train_indicator_bot


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_features(panel):
    r = panel["ret"]
    c = panel["close"]

    # Proxy raw confidence from momentum/vol mix, then learn reliability corrections.
    raw_conf = sigmoid(18.0 * ema(r, 6))
    conf_vel = np.diff(raw_conf, prepend=raw_conf[0])

    _, mid, up = bollinger(c, window=20, k=2.0)
    bb_span = (up - mid) / (mid + 1e-8)
    noise = rolling_std(r, 20)

    calibration_gap = np.abs(raw_conf - sigmoid(12.0 * ema(r, 20)))
    overconf = np.maximum(raw_conf - (1.0 - noise / (np.max(noise) + 1e-8)), 0.0)
    underconf = np.maximum((0.5 - raw_conf), 0.0) * np.maximum(ema(r, 12), 0.0)

    return np.stack(
        [r, raw_conf, conf_vel, bb_span, noise, calibration_gap, overconf, underconf],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v57_confidence_calibrator",
        feature_names=[
            "ret",
            "raw_conf",
            "conf_vel",
            "bb_span",
            "noise",
            "calibration_gap",
            "overconf",
            "underconf",
        ],
        feature_builder=build_features,
        window=46,
        horizon=4,
    )
