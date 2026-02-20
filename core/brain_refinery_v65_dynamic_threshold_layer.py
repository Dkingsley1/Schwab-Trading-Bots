import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    score_proxy = sigmoid(20.0 * ema(r - rb, 10))
    vol = rolling_std(r, 20)
    uncertainty = vol / (np.max(vol) + 1e-8)

    threshold_up = 0.52 + 0.10 * uncertainty
    threshold_dn = 0.48 - 0.10 * uncertainty
    margin = np.abs(score_proxy - 0.5)
    confidence_gap = margin - np.abs(threshold_up - 0.5)

    return np.stack([r, rb, score_proxy, uncertainty, threshold_up, threshold_dn, confidence_gap], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v65_dynamic_threshold_layer",
        feature_names=["ret", "bench_ret", "score_proxy", "uncertainty", "threshold_up", "threshold_dn", "confidence_gap"],
        feature_builder=build_features,
        window=46,
        horizon=3,
    )
