import numpy as np

from indicator_bot_common import atr, bollinger, ema, rolling_mean, rolling_std, train_indicator_bot


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

    lower, mid, upper = bollinger(c, window=20, k=2.0)
    bb_width = (upper - lower) / (mid + 1e-8)

    atr20 = atr(h, l, c, period=20)
    ema20 = ema(c, 20)
    kel_upper = ema20 + 2.0 * atr20
    kel_lower = ema20 - 2.0 * atr20
    kel_width = (kel_upper - kel_lower) / (ema20 + 1e-8)

    squeeze_ratio = bb_width / (kel_width + 1e-8)
    squeeze_slope = np.diff(ema(squeeze_ratio, 6), prepend=squeeze_ratio[0])
    percent_b = (c - lower) / ((upper - lower) + 1e-8)
    percent_b_ema = ema(percent_b, 6)
    breakout_energy = np.abs(np.diff(percent_b, prepend=percent_b[0])) / (rolling_std(r, 20) + 1e-8)
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
            bb_width,
            kel_width,
            squeeze_ratio,
            squeeze_slope,
            percent_b,
            percent_b_ema,
            breakout_energy,
            trend,
            breadth,
            volume_breadth,
            vix_change,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v33_keltner_bb_combo",
        feature_names=[
            "ret",
            "bench_ret",
            "bench_mom_12",
            "bb_width",
            "keltner_width",
            "squeeze_ratio",
            "squeeze_slope_ema6",
            "percent_b",
            "percent_b_ema6",
            "breakout_energy",
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
