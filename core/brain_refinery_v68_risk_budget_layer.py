import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    c = panel["close"]

    edge = np.abs(ema(r, 10)) / (rolling_std(r, 20) + 1e-8)
    vol = rolling_std(r, 20)
    draw = np.minimum(np.diff(c, prepend=c[0]) / (c + 1e-8), 0.0)
    draw_stress = np.sqrt(rolling_std(draw, 30) + 1e-8)
    vix_shock = np.maximum(np.diff(vix, prepend=vix[0]) / (vix + 1e-8), 0.0)

    risk_budget = edge / (vol + draw_stress + vix_shock + 1e-8)
    low = (risk_budget < np.percentile(risk_budget, 33)).astype(float)
    high = (risk_budget > np.percentile(risk_budget, 66)).astype(float)

    return np.stack([r, edge, vol, draw_stress, vix_shock, risk_budget, low, high], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v68_risk_budget_layer",
        feature_names=["ret", "edge", "vol", "draw_stress", "vix_shock", "risk_budget", "low_budget", "high_budget"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
