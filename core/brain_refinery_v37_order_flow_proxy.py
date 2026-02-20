import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    tick_dir = np.sign(np.diff(c, prepend=c[0]))
    signed_vol = tick_dir * v
    flow_ema = ema(signed_vol, 10)
    flow_impulse = np.diff(flow_ema, prepend=flow_ema[0])
    rel_vol = v / (np.convolve(v, np.ones(20) / 20.0, mode="same") + 1e-8)
    micro_imbalance = (flow_ema / (np.abs(flow_ema) + 1e-8)) * rel_vol
    shock = np.abs(r) / (rolling_std(r, 20) + 1e-8)

    return np.stack([r, tick_dir, rel_vol, flow_ema, flow_impulse, micro_imbalance, shock], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v37_order_flow_proxy",
        feature_names=["ret", "tick_dir", "rel_vol", "flow_ema", "flow_impulse", "micro_imbalance", "shock"],
        feature_builder=build_features,
    )
