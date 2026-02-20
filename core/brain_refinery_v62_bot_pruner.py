import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_hit_proxy(r, horizon=3):
    # Approximate short-term directional hit-rate from own returns stream.
    out = np.zeros_like(r)
    for i in range(len(r)):
        s = max(0, i - horizon + 1)
        seg = r[s : i + 1]
        hits = np.sum(np.sign(seg[1:]) == np.sign(seg[:-1])) if len(seg) > 1 else 0.0
        out[i] = hits / max(len(seg) - 1, 1)
    return out


def build_features(panel):
    r = panel["ret"]
    c = panel["close"]

    hit = rolling_hit_proxy(r, horizon=5)
    hit_fast = ema(hit, 6)
    hit_slow = ema(hit, 20)
    degradation = np.maximum(hit_slow - hit_fast, 0.0)

    draw = np.minimum(np.diff(c, prepend=c[0]) / (c + 1e-8), 0.0)
    draw_stress = np.sqrt(rolling_std(draw, 30) + 1e-8)
    noise = rolling_std(r, 20)

    prune_pressure = degradation + 0.5 * draw_stress + 0.3 * noise

    return np.stack([r, hit_fast, hit_slow, degradation, draw_stress, noise, prune_pressure], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v62_bot_pruner",
        feature_names=["ret", "hit_fast", "hit_slow", "degradation", "draw_stress", "noise", "prune_pressure"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
