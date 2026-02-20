import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_mean(x, w=60):
    out = np.zeros_like(x)
    for i in range(len(x)):
        s = max(0, i - w + 1)
        out[i] = np.mean(x[s:i+1])
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    v = panel["volume"]

    # Drift proxy on feature manifold (returns + bench + volume state).
    f1 = r
    f2 = rb
    f3 = np.log(v + 1.0)

    m1, m2, m3 = rolling_mean(f1, 80), rolling_mean(f2, 80), rolling_mean(f3, 80)
    s1, s2, s3 = rolling_std(f1, 80) + 1e-8, rolling_std(f2, 80) + 1e-8, rolling_std(f3, 80) + 1e-8

    z1 = (f1 - m1) / s1
    z2 = (f2 - m2) / s2
    z3 = (f3 - m3) / s3

    drift_mag = np.sqrt(z1 * z1 + z2 * z2 + z3 * z3)
    drift_fast = ema(drift_mag, 8)
    drift_slow = ema(drift_mag, 30)
    drift_guard = np.maximum(drift_fast - drift_slow, 0.0)

    return np.stack([r, rb, z1, z2, z3, drift_mag, drift_fast, drift_slow, drift_guard], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v75_model_drift_guard",
        feature_names=["ret", "bench_ret", "z1", "z2", "z3", "drift_mag", "drift_fast", "drift_slow", "drift_guard"],
        feature_builder=build_features,
        window=52,
        horizon=4,
    )
