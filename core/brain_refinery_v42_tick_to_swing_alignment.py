import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        j = (i // step) * step
        out[i] = x[j]
    return out


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]
    rb = panel["bench_ret"]
    adv = panel["adv"]
    dec = panel["dec"]
    up_vol = panel["up_vol"]
    down_vol = panel["down_vol"]
    vix = panel["vix"]

    # Fast tick-like layer
    tick_ret = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    tick_mom = ema(tick_ret, 5)
    tick_noise = rolling_std(tick_ret, 8)

    # Slow swing layer
    swing_ret = hold_sample(r, 10)
    swing_mom = ema(swing_ret, 10)
    swing_noise = rolling_std(swing_ret, 20)
    bench_mom = ema(rb, 8)

    # Alignment and conflict metrics
    align = np.sign(tick_mom) * np.sign(swing_mom)
    conflict_mag = np.abs(tick_mom - swing_mom)

    rel_vol = v / (ema(v, 20) + 1e-8)
    rel_vol_z = rel_vol / (rolling_mean(rel_vol, 30) + 1e-8)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_change = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    trend_spread = tick_mom + swing_mom + bench_mom
    noise_ratio = tick_noise / (swing_noise + 1e-8)

    return np.stack(
        [
            r,
            tick_mom,
            tick_noise,
            swing_mom,
            swing_noise,
            bench_mom,
            align,
            conflict_mag,
            rel_vol,
            rel_vol_z,
            breadth,
            volume_breadth,
            vix_change,
            trend_spread,
            noise_ratio,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v42_tick_to_swing_alignment",
        feature_names=[
            "ret",
            "tick_mom",
            "tick_noise",
            "swing_mom",
            "swing_noise",
            "bench_mom",
            "align",
            "conflict_mag",
            "rel_vol",
            "rel_vol_z",
            "breadth",
            "volume_breadth",
            "vix_change",
            "trend_spread",
            "noise_ratio",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=48,
        horizon=4,
        patience=24,
    )
