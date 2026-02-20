import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    h = panel["high"]
    l = panel["low"]
    c = panel["close"]
    v = panel["volume"]

    edge_proxy = np.abs(ema(r, 8))
    spread_proxy = (h - l) / (c + 1e-8)
    slippage_proxy = spread_proxy / (v / (ema(v, 20) + 1e-8) + 1e-8)
    tx_cost = spread_proxy + slippage_proxy

    net_edge = edge_proxy - tx_cost
    net_edge_z = net_edge / (rolling_std(net_edge, 20) + 1e-8)
    pass_filter = (net_edge > 0.0).astype(float)

    return np.stack([r, edge_proxy, spread_proxy, slippage_proxy, tx_cost, net_edge, net_edge_z, pass_filter], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v69_cost_aware_execution_filter",
        feature_names=["ret", "edge_proxy", "spread_proxy", "slippage_proxy", "tx_cost", "net_edge", "net_edge_z", "pass_filter"],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )
