import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def rolling_corr(a, b, window=30):
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
    r = panel["ret"]
    rb = panel["bench_ret"]

    # Proxy ensemble members: fast trend, mean-revert, and shock-style channels.
    trend_ch = ema(r, 8)
    meanrev_ch = -ema(r, 20)
    shock_ch = np.sign(np.diff(r, prepend=r[0])) * rolling_std(r, 12)

    c_tm = rolling_corr(trend_ch, meanrev_ch, window=35)
    c_ts = rolling_corr(trend_ch, shock_ch, window=35)
    c_ms = rolling_corr(meanrev_ch, shock_ch, window=35)

    corr_load = (np.abs(c_tm) + np.abs(c_ts) + np.abs(c_ms)) / 3.0
    diversity = 1.0 - corr_load

    bench_align = np.sign(ema(r, 12)) * np.sign(ema(rb, 12))
    conflict = np.abs(trend_ch - meanrev_ch) / (rolling_std(r, 20) + 1e-8)

    slow_regime = ema(hold_sample(r, 12), 12)

    return np.stack(
        [r, trend_ch, meanrev_ch, shock_ch, c_tm, c_ts, c_ms, corr_load, diversity, bench_align, conflict, slow_regime],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v58_ensemble_diversity_controller",
        feature_names=[
            "ret",
            "trend_ch",
            "meanrev_ch",
            "shock_ch",
            "corr_tm",
            "corr_ts",
            "corr_ms",
            "corr_load",
            "diversity",
            "bench_align",
            "conflict",
            "slow_regime",
        ],
        feature_builder=build_features,
        window=52,
        horizon=5,
    )
