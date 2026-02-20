import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    # Proxy allocator signals for fast/medium/slow specialist groups.
    fast = ema(r - rb, 6)
    medium = ema(r - rb, 20)
    slow = ema(hold_sample(r - rb, 12), 12)

    vol_fast = rolling_std(r, 12)
    vol_slow = rolling_std(r, 40)
    regime = vol_fast / (vol_slow + 1e-8)

    agreement = np.sign(fast) * np.sign(medium) + np.sign(medium) * np.sign(slow)
    edge = (0.5 * fast + 0.3 * medium + 0.2 * slow) / (vol_fast + 1e-8)

    return np.stack([r, rb, fast, medium, slow, regime, agreement, edge], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v61_bot_allocator",
        feature_names=["ret", "bench_ret", "fast", "medium", "slow", "regime", "agreement", "edge"],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )
