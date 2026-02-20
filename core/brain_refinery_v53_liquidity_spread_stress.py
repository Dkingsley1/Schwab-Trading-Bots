import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]

    spread_proxy = (h - l) / (c + 1e-8)
    spread_fast = ema(spread_proxy, 6)
    spread_slow = ema(spread_proxy, 20)
    spread_stress = spread_fast / (spread_slow + 1e-8)

    rel_vol = v / (ema(v, 20) + 1e-8)
    illiquidity = np.abs(r) / (rel_vol + 1e-8)
    stress = spread_stress * (1.0 + illiquidity)
    stress_vol = rolling_std(stress, 20)

    return np.stack([r, spread_proxy, spread_stress, rel_vol, illiquidity, stress, stress_vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v53_liquidity_spread_stress",
        feature_names=["ret", "spread_proxy", "spread_stress", "rel_vol", "illiquidity", "stress", "stress_vol"],
        feature_builder=build_features,
        window=40,
        horizon=3,
    )
