import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]
    bench_r = panel["bench_ret"]
    adv = panel["adv"]
    dec = panel["dec"]
    up_vol = panel["up_vol"]
    down_vol = panel["down_vol"]
    vix = panel["vix"]

    # Tick-style proxies from bar data: direction flips, micro returns, signed flow.
    tick_ret = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    tick_dir = np.sign(tick_ret)
    tick_flip = np.abs(np.diff(tick_dir, prepend=tick_dir[0]))

    signed_vol = tick_dir * v
    signed_flow = ema(signed_vol, 6)
    flow_impulse = np.diff(signed_flow, prepend=signed_flow[0])
    flow_z = signed_flow / (ema(np.abs(signed_vol), 20) + 1e-8)

    micro_noise = rolling_std(tick_ret, 8)
    trade_imbalance = signed_flow / (ema(np.abs(signed_vol), 10) + 1e-8)
    rel_vol = v / (ema(v, 30) + 1e-8)
    bench_mom = rolling_mean(bench_r, 12)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_change = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    micro_trend = ema(tick_ret, 5) - ema(tick_ret, 20)

    return np.stack(
        [
            r,
            bench_r,
            bench_mom,
            tick_ret,
            tick_dir,
            tick_flip,
            signed_flow,
            flow_impulse,
            flow_z,
            micro_noise,
            trade_imbalance,
            rel_vol,
            breadth,
            volume_breadth,
            vix_change,
            micro_trend,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v40_tick_microstructure",
        feature_names=[
            "ret",
            "bench_ret",
            "bench_mom_12",
            "tick_ret",
            "tick_dir",
            "tick_flip",
            "signed_flow",
            "flow_impulse",
            "flow_z",
            "micro_noise",
            "trade_imbalance",
            "relative_volume_30",
            "breadth_adv_dec",
            "volume_breadth",
            "vix_change",
            "micro_trend_5_20",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=48,
        horizon=2,
        learning_rate=0.0007,
        epochs=260,
        patience=24,
    )
