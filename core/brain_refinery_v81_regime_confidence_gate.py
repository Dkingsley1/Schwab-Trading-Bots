import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    vix = panel["vix"]

    trend_score = ema(r, 20) / (rolling_std(r, 20) + 1e-8)
    bench_score = ema(rb, 20) / (rolling_std(rb, 20) + 1e-8)
    trend_gap = trend_score - bench_score

    chop_ratio = rolling_std(r, 8) / (rolling_std(r, 40) + 1e-8)
    vix_regime = (vix - rolling_mean(vix, 35)) / (rolling_std(vix, 35) + 1e-8)

    agreement = 1.0 / (1.0 + np.abs(np.diff(trend_gap, prepend=trend_gap[0])))
    raw_conf = np.abs(trend_gap) / (1.0 + chop_ratio + np.maximum(vix_regime, 0.0))
    confidence = np.clip(ema(raw_conf, 6), 0.0, 1.0)

    return np.stack([r, rb, trend_score, bench_score, trend_gap, chop_ratio, vix_regime, agreement, confidence], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v81_regime_confidence_gate",
        feature_names=[
            "ret",
            "bench_ret",
            "trend_score",
            "bench_score",
            "trend_gap",
            "chop_ratio",
            "vix_regime",
            "agreement",
            "confidence",
        ],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
