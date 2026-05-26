import numpy as np

from indicator_bot_common import atr, ema, rolling_mean, rolling_std, train_indicator_bot


def donchian(high, low, window=20):
    up = np.zeros_like(high)
    dn = np.zeros_like(low)
    for i in range(len(high)):
        start = max(0, i - window + 1)
        up[i] = np.max(high[start : i + 1])
        dn[i] = np.min(low[start : i + 1])
    return up, dn


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]
    bench_r = panel["bench_ret"]
    adv = panel["adv"]
    dec = panel["dec"]
    up_vol = panel["up_vol"]
    down_vol = panel["down_vol"]
    vix = panel["vix"]

    up, dn = donchian(h, l, window=20)
    width = (up - dn) / (c + 1e-8)
    width_slope = np.diff(ema(width, 8), prepend=width[0])
    breakout_up = (c - up) / (up + 1e-8)
    breakout_dn = (c - dn) / (dn + 1e-8)
    channel_pos = (c - dn) / ((up - dn) + 1e-8)

    atr14 = atr(h, l, c, period=14) / (c + 1e-8)
    vol20 = rolling_std(r, 20)
    breakout_quality = (np.abs(breakout_up) + np.abs(breakout_dn)) / (atr14 + 1e-8)
    trend = (ema(c, 8) - ema(c, 34)) / (c + 1e-8)
    bench_mom = rolling_mean(bench_r, 12)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_change = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)

    return np.stack(
        [
            r,
            bench_r,
            bench_mom,
            width,
            width_slope,
            breakout_up,
            breakout_dn,
            channel_pos,
            atr14,
            vol20,
            breakout_quality,
            trend,
            breadth,
            volume_breadth,
            vix_change,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v34_donchian_atr_breakout",
        feature_names=[
            "ret",
            "bench_ret",
            "bench_mom_12",
            "donchian_width",
            "donchian_width_slope",
            "breakout_up",
            "breakout_dn",
            "channel_position",
            "atr14_pct",
            "vol20",
            "breakout_quality",
            "trend_ema8_34",
            "breadth_adv_dec",
            "volume_breadth",
            "vix_change",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=48,
        horizon=4,
        learning_rate=0.0007,
        epochs=260,
        patience=24,
    )
