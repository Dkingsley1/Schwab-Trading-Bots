import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_mean(x, w=50):
    out = np.zeros_like(x)
    for i in range(len(x)):
        s = max(0, i-w+1)
        out[i] = np.mean(x[s:i+1])
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    v = panel["volume"]

    feat = np.stack([r, rb, v / (np.mean(v) + 1e-8)], axis=1)
    m = np.column_stack([rolling_mean(feat[:,i], 60) for i in range(feat.shape[1])])
    s = np.column_stack([rolling_std(feat[:,i], 60) + 1e-8 for i in range(feat.shape[1])])

    z = (feat - m) / s
    drift_mag = np.sqrt(np.sum(z*z, axis=1))
    drift_fast = ema(drift_mag, 8)
    drift_slow = ema(drift_mag, 30)
    drift_alert = (drift_fast > drift_slow + np.std(drift_mag)).astype(float)

    return np.stack([r, rb, drift_mag, drift_fast, drift_slow, drift_alert], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v70_drift_detection_layer",
        feature_names=["ret", "bench_ret", "drift_mag", "drift_fast", "drift_slow", "drift_alert"],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )
