import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]

    delta = np.diff(c, prepend=c[0])
    stale_price = (np.abs(delta) < 1e-7).astype(float)
    stale_price_rate = ema(stale_price, 12)

    range_proxy = (h - l) / (c + 1e-8)
    freeze_range = (range_proxy < np.percentile(range_proxy, 15)).astype(float)
    freeze_rate = ema(freeze_range, 10)

    rel_vol = v / (rolling_mean(v, 30) + 1e-8)
    low_flow = (rel_vol < 0.30).astype(float)
    low_flow_rate = ema(low_flow, 10)

    jitter = np.abs(np.diff(r, prepend=r[0])) / (rolling_std(r, 20) + 1e-8)
    staleness_risk = ema(0.45 * stale_price_rate + 0.30 * freeze_rate + 0.25 * low_flow_rate + 0.10 * np.minimum(jitter, 3.0), 6)
    freshness_score = 1.0 - np.clip(staleness_risk, 0.0, 1.0)

    return np.stack([r, stale_price_rate, range_proxy, freeze_rate, rel_vol, low_flow_rate, jitter, staleness_risk, freshness_score], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v85_latency_and_staleness_guard",
        feature_names=[
            "ret",
            "stale_price_rate",
            "range_proxy",
            "freeze_rate",
            "rel_vol",
            "low_flow_rate",
            "jitter",
            "staleness_risk",
            "freshness_score",
        ],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
