import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    v = panel["volume"]

    # Proxy latency stress from burstiness + stale-like micro-moves.
    micro = np.abs(np.diff(r, prepend=r[0]))
    burst = micro / (rolling_std(micro, 20) + 1e-8)
    stale_like = (np.abs(r) < 1e-6).astype(float)
    stale_rate = ema(stale_like, 12)
    rel_vol = v / (ema(v, 25) + 1e-8)

    latency_stress = ema(burst + stale_rate + np.maximum(0.0, 1.0 - rel_vol), 8)

    return np.stack([r, micro, burst, stale_rate, rel_vol, latency_stress], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v72_latency_sentinel",
        feature_names=["ret", "micro", "burst", "stale_rate", "rel_vol", "latency_stress"],
        feature_builder=build_features,
        window=46,
        horizon=3,
    )
