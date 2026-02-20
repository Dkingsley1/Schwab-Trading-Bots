import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_corr(a, b, window=40):
    out = np.zeros_like(a)
    for i in range(len(a)):
        s = max(0, i - window + 1)
        xa = a[s : i + 1]
        xb = b[s : i + 1]
        if len(xa) < 3:
            out[i] = 0.0
            continue
        ca = xa - np.mean(xa)
        cb = xb - np.mean(xb)
        den = np.sqrt(np.sum(ca * ca) * np.sum(cb * cb)) + 1e-8
        out[i] = np.sum(ca * cb) / den
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    corr = rolling_corr(r, rb, window=40)
    corr_fast = ema(corr, 6)
    corr_slow = ema(corr, 20)
    corr_break = corr_fast - corr_slow

    alpha = r - rb
    alpha_ema = ema(alpha, 10)
    alpha_vol = rolling_std(alpha, 25)

    return np.stack([r, rb, corr, corr_fast, corr_slow, corr_break, alpha_ema, alpha_vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v55_correlation_breakdown",
        feature_names=["ret", "bench_ret", "corr", "corr_fast", "corr_slow", "corr_break", "alpha_ema", "alpha_vol"],
        feature_builder=build_features,
        window=48,
        horizon=5,
    )
