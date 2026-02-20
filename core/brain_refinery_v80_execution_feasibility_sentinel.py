import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]

    spread_proxy = (h - l) / (c + 1e-8)
    spread_baseline = rolling_mean(spread_proxy, 40)
    spread_z = (spread_proxy - spread_baseline) / (rolling_std(spread_proxy, 40) + 1e-8)

    depth_proxy = v / (rolling_mean(v, 30) + 1e-8)
    impact_proxy = np.abs(r) / (depth_proxy + 1e-8)
    impact_trend = ema(impact_proxy, 8)

    liquidity_stress = np.maximum(1.0 - depth_proxy, 0.0)
    slippage_risk = ema(np.maximum(spread_z, 0.0) + impact_trend + liquidity_stress, 6)
    feasibility_score = 1.0 - np.clip(0.50 * np.maximum(spread_z, 0.0) + 0.30 * impact_trend + 0.20 * liquidity_stress, 0.0, 1.0)

    return np.stack(
        [r, spread_proxy, spread_z, depth_proxy, impact_proxy, impact_trend, liquidity_stress, slippage_risk, feasibility_score],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v80_execution_feasibility_sentinel",
        feature_names=[
            "ret",
            "spread_proxy",
            "spread_z",
            "depth_proxy",
            "impact_proxy",
            "impact_trend",
            "liquidity_stress",
            "slippage_risk",
            "feasibility_score",
        ],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
