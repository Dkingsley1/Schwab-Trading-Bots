import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, simulate_market_panel, train_price_indicator_bot

WINDOW = 44
HORIZON = 3


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


def _price_axis(n):
    return np.arange(max(int(n), 1), dtype=np.float64)


def _build_feasibility_dataset(prices):
    panel = simulate_market_panel(int(len(prices)))
    features = build_features(panel)
    model_features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)
    feasibility = features[:, -1]
    slippage_risk = features[:, -2]
    liquidity_stress = features[:, -3]

    samples = []
    scores = []
    anchors = []
    for idx in range(WINDOW - 1, len(features) - HORIZON):
        future_slice = slice(idx + 1, idx + HORIZON + 1)
        score = (
            0.62 * float(np.mean(feasibility[future_slice]))
            + 0.22 * float(1.0 - np.mean(slippage_risk[future_slice]))
            + 0.16 * float(1.0 - np.mean(liquidity_stress[future_slice]))
        )
        samples.append(model_features[idx - WINDOW + 1 : idx + 1].reshape(-1))
        scores.append(score)
        anchors.append(idx)

    score_array = np.asarray(scores, dtype=np.float32)
    threshold = float(np.median(score_array)) if score_array.size else 0.5
    labels = (score_array >= threshold).astype(np.float32).reshape(-1, 1)
    return (
        np.asarray(samples, dtype=np.float32),
        labels,
        np.asarray(anchors, dtype=np.int64),
    )


def train_brain():
    return train_price_indicator_bot(
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
        feature_builder=lambda prices: np.zeros((len(prices), 1), dtype=np.float32),
        price_simulator=_price_axis,
        dataset_builder=_build_feasibility_dataset,
        window=WINDOW,
        horizon=HORIZON,
    )


if __name__ == "__main__":
    train_brain()
