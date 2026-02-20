import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_hit(x, w=10):
    out = np.zeros_like(x)
    sgn = np.sign(x)
    for i in range(len(x)):
        a = max(1, i - w + 1)
        seg = sgn[a:i+1]
        if len(seg) < 2:
            out[i] = 0.5
        else:
            out[i] = np.mean(seg[1:] == seg[:-1])
    return out


def build_features(panel):
    r = panel["ret"]

    hit = rolling_hit(r, w=12)
    hit_fast = ema(hit, 6)
    hit_slow = ema(hit, 20)
    drift = hit_fast - hit_slow

    # Brier-like proxy: penalize large moves following low persistence.
    risk = np.abs(r) / (rolling_std(r, 20) + 1e-8)
    brier_proxy = (1.0 - hit_fast) * risk
    reliability = np.clip(hit_fast - 0.3 * brier_proxy, 0.0, 1.0)

    return np.stack([r, hit_fast, hit_slow, drift, risk, brier_proxy, reliability], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v66_reliability_score_layer",
        feature_names=["ret", "hit_fast", "hit_slow", "drift", "risk", "brier_proxy", "reliability"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
