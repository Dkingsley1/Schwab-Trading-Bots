import numpy as np

from indicator_bot_common import ema, rolling_std, simulate_market_panel, train_price_indicator_bot

WINDOW = 52
HORIZON = 5
FEATURE_NAMES = [
    "ret",
    "bench_ret",
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
    "vol_ratio",
    "diversity_momentum",
    "ensemble_health",
]


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
    vol_ratio = rolling_std(r, 12) / (rolling_std(r, 48) + 1e-8)
    diversity_momentum = ema(np.diff(diversity, prepend=diversity[0]), 8)
    ensemble_health = ema(
        0.42 * diversity
        + 0.24 * (1.0 - corr_load)
        + 0.20 * (1.0 / (1.0 + np.maximum(conflict, 0.0)))
        + 0.14 * ((bench_align + 1.0) / 2.0),
        6,
    )

    return np.stack(
        [
            r,
            rb,
            trend_ch,
            meanrev_ch,
            shock_ch,
            c_tm,
            c_ts,
            c_ms,
            corr_load,
            diversity,
            bench_align,
            conflict,
            slow_regime,
            vol_ratio,
            diversity_momentum,
            ensemble_health,
        ],
        axis=1,
    )


def _build_ensemble_diversity_dataset(prices):
    n = int(len(prices))
    panel = simulate_market_panel(n)
    features = build_features(panel)
    model_features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)

    diversity = features[:, FEATURE_NAMES.index("diversity")]
    corr_load = features[:, FEATURE_NAMES.index("corr_load")]
    bench_align = features[:, FEATURE_NAMES.index("bench_align")]
    conflict = features[:, FEATURE_NAMES.index("conflict")]
    ensemble_health = features[:, FEATURE_NAMES.index("ensemble_health")]
    vol_ratio = features[:, FEATURE_NAMES.index("vol_ratio")]

    samples = []
    scores = []
    anchors = []
    for idx in range(WINDOW - 1, n - HORIZON):
        future_slice = slice(idx + 1, idx + HORIZON + 1)
        future_score = (
            0.36 * float(np.mean(ensemble_health[future_slice]))
            + 0.22 * float(np.mean(diversity[future_slice]))
            + 0.18 * float(np.mean(1.0 - corr_load[future_slice]))
            + 0.14 * float(np.mean(1.0 / (1.0 + np.maximum(conflict[future_slice], 0.0))))
            + 0.06 * float(np.mean((bench_align[future_slice] + 1.0) / 2.0))
            + 0.04 * float(np.mean(1.0 / (1.0 + np.maximum(vol_ratio[future_slice] - 1.0, 0.0))))
        )
        samples.append(model_features[idx - WINDOW + 1 : idx + 1].reshape(-1))
        scores.append(future_score)
        anchors.append(idx)

    score_array = np.asarray(scores, dtype=np.float32)
    threshold = float(np.median(score_array)) if score_array.size else 0.5
    labels = (score_array >= threshold).astype(np.float32).reshape(-1, 1)
    return (
        np.asarray(samples, dtype=np.float32),
        labels,
        np.asarray(anchors, dtype=np.int64),
    )


def _price_axis(n):
    return np.arange(max(int(n), 1), dtype=np.float64)


def train_brain():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v58_ensemble_diversity_controller",
        feature_names=FEATURE_NAMES,
        feature_builder=lambda prices: np.zeros((len(prices), 1), dtype=np.float32),
        price_simulator=_price_axis,
        dataset_builder=_build_ensemble_diversity_dataset,
        num_points=6000,
        window=WINDOW,
        horizon=HORIZON,
        learning_rate=0.00035,
        epochs=160,
        batch_size=128,
        patience=12,
        acted_prob_threshold=0.70,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.58,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.48,
        max_acted_coverage=0.32,
        defer_on_quality_failure=True,
    )


if __name__ == "__main__":
    train_brain()
