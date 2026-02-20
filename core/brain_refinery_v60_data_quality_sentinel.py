import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]

    # Data quality proxies: stale-like moves, sudden range explosions, zero-ish flow anomalies.
    delta = np.diff(c, prepend=c[0])
    stale_like = (np.abs(delta) < 1e-6).astype(float)
    stale_rate = ema(stale_like, 12)

    range_proxy = (h - l) / (c + 1e-8)
    range_jump = np.abs(np.diff(range_proxy, prepend=range_proxy[0]))

    rel_vol = v / (ema(v, 20) + 1e-8)
    low_flow = (rel_vol < 0.25).astype(float)
    low_flow_rate = ema(low_flow, 10)

    return_spike = np.abs(r) / (rolling_std(r, 30) + 1e-8)
    bad_tick_proxy = (range_jump > np.percentile(range_jump, 95)).astype(float)
    bad_tick_rate = ema(bad_tick_proxy, 10)

    quality_score = 1.0 - np.clip(0.45 * stale_rate + 0.25 * low_flow_rate + 0.30 * bad_tick_rate, 0.0, 1.0)

    return np.stack(
        [r, stale_rate, range_proxy, range_jump, rel_vol, low_flow_rate, return_spike, bad_tick_rate, quality_score],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v60_data_quality_sentinel",
        feature_names=[
            "ret",
            "stale_rate",
            "range_proxy",
            "range_jump",
            "rel_vol",
            "low_flow_rate",
            "return_spike",
            "bad_tick_rate",
            "quality_score",
        ],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
