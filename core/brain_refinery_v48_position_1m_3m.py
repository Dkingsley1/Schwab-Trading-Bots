import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    b = panel["bench_ret"]
    c = panel["close"]

    m1 = ema(hold_sample(r, 390), 6)
    m3 = ema(hold_sample(r, 1170), 6)
    beta_adj = r - 0.8 * b
    beta_m = ema(beta_adj, 20)
    vol = rolling_std(r, 80)
    long_drift = ema(np.diff(c, prepend=c[0]) / (c + 1e-8), 80)

    return np.stack([r, b, m1, m3, beta_m, vol, long_drift], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v48_position_1m_3m",
        feature_names=["ret", "bench_ret", "mom_1m", "mom_3m", "beta_m", "vol", "long_drift"],
        feature_builder=build_features,
        window=64,
        horizon=12,
    )
