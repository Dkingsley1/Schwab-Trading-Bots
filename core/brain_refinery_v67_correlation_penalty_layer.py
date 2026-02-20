import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def corr(a, b, w=30):
    out = np.zeros_like(a)
    for i in range(len(a)):
        s = max(0, i - w + 1)
        xa = a[s:i+1]
        xb = b[s:i+1]
        if len(xa) < 3:
            out[i] = 0.0
            continue
        xa = xa - np.mean(xa)
        xb = xb - np.mean(xb)
        den = np.sqrt(np.sum(xa*xa)*np.sum(xb*xb)) + 1e-8
        out[i] = np.sum(xa*xb)/den
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    ch1 = ema(r, 8)
    ch2 = ema(r, 20)
    ch3 = ema(r - rb, 12)

    c12 = corr(ch1, ch2, 35)
    c13 = corr(ch1, ch3, 35)
    c23 = corr(ch2, ch3, 35)

    corr_load = (np.abs(c12) + np.abs(c13) + np.abs(c23)) / 3.0
    penalty = np.maximum(corr_load - 0.6, 0.0)
    diversity = 1.0 - corr_load

    return np.stack([r, rb, c12, c13, c23, corr_load, penalty, diversity], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v67_correlation_penalty_layer",
        feature_names=["ret", "bench_ret", "c12", "c13", "c23", "corr_load", "penalty", "diversity"],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )
