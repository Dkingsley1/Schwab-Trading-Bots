import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot, vwap


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
    gap = panel["gap"]

    vwap60 = vwap(c, v, session=60)
    dev = (c - vwap60) / (vwap60 + 1e-8)
    dev_z = dev / (rolling_std(dev, 30) + 1e-8)
    dev_ema = ema(dev, 8)
    dev_slope = np.diff(dev_ema, prepend=dev_ema[0])
    vol_z = (v - np.mean(v)) / (np.std(v) + 1e-8)
    rel_vol = v / (ema(v, 30) + 1e-8)
    ret_vol = rolling_std(r, 15)
    bench_mom = rolling_mean(bench_r, 12)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_change = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    mean_reversion = (c - rolling_mean(c, 30)) / (rolling_std(c, 30) + 1e-8)

    return np.stack(
        [
            r,
            bench_r,
            bench_mom,
            dev,
            dev_z,
            dev_ema,
            dev_slope,
            vol_z,
            rel_vol,
            ret_vol,
            breadth,
            volume_breadth,
            vix_change,
            gap,
            mean_reversion,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v24_vwap_deviation",
        feature_names=[
            "ret",
            "bench_ret",
            "bench_mom_12",
            "vwap_dev",
            "vwap_dev_z",
            "vwap_dev_ema_8",
            "vwap_dev_slope",
            "volume_z",
            "relative_volume_30",
            "ret_vol_15",
            "breadth_adv_dec",
            "volume_breadth",
            "vix_change",
            "gap",
            "mean_reversion_z_30",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=48,
        horizon=3,
        learning_rate=0.0007,
        epochs=260,
        patience=24,
    )
