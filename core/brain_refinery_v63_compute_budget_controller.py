import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    v = panel["volume"]
    vix = panel["vix"]

    # Proxy when extra compute (complex bots) is likely worth it.
    edge_proxy = np.abs(ema(r, 8)) / (rolling_std(r, 20) + 1e-8)
    uncertainty = rolling_std(r, 15)
    rel_vol = v / (ema(v, 25) + 1e-8)
    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)

    # High budget when edge is high and uncertainty not extreme.
    compute_value = edge_proxy * rel_vol
    compute_risk = uncertainty + np.maximum(vix_chg, 0.0)
    budget_signal = compute_value / (compute_risk + 1e-8)

    return np.stack([r, edge_proxy, uncertainty, rel_vol, vix_chg, compute_value, compute_risk, budget_signal], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v63_compute_budget_controller",
        feature_names=["ret", "edge_proxy", "uncertainty", "rel_vol", "vix_chg", "compute_value", "compute_risk", "budget_signal"],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
