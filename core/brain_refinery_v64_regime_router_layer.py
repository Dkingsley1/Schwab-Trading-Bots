import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    c = panel["close"]

    trend = ema(r, 12)
    chop = rolling_std(r, 20)
    shock = np.maximum(np.diff(vix, prepend=vix[0]) / (vix + 1e-8), 0.0)
    event_like = np.abs(np.diff(c, prepend=c[0]) / (c + 1e-8)) / (rolling_std(r, 30) + 1e-8)

    regime_trend = (np.abs(trend) > chop).astype(float)
    regime_chop = (chop > np.abs(trend)).astype(float)
    regime_shock = (shock > np.percentile(shock, 85)).astype(float)

    return np.stack([r, trend, chop, shock, event_like, regime_trend, regime_chop, regime_shock], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v64_regime_router_layer",
        feature_names=["ret", "trend", "chop", "shock", "event_like", "regime_trend", "regime_chop", "regime_shock"],
        feature_builder=build_features,
        window=52,
        horizon=4,
    )
