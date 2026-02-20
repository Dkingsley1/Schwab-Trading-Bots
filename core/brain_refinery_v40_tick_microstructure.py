import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    # Tick-style proxies from bar data: direction flips, micro returns, signed flow.
    tick_ret = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    tick_dir = np.sign(tick_ret)
    tick_flip = np.abs(np.diff(tick_dir, prepend=tick_dir[0]))

    signed_vol = tick_dir * v
    signed_flow = ema(signed_vol, 6)
    flow_impulse = np.diff(signed_flow, prepend=signed_flow[0])

    micro_noise = rolling_std(tick_ret, 8)
    trade_imbalance = signed_flow / (ema(np.abs(signed_vol), 10) + 1e-8)

    return np.stack(
        [
            r,
            tick_ret,
            tick_dir,
            tick_flip,
            signed_flow,
            flow_impulse,
            micro_noise,
            trade_imbalance,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v40_tick_microstructure",
        feature_names=[
            "ret",
            "tick_ret",
            "tick_dir",
            "tick_flip",
            "signed_flow",
            "flow_impulse",
            "micro_noise",
            "trade_imbalance",
        ],
        feature_builder=build_features,
        window=40,
        horizon=2,
    )
