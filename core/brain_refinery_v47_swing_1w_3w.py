import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    rb = panel["bench_ret"]
    adv = panel["adv"]
    dec = panel["dec"]
    up_vol = panel["up_vol"]
    down_vol = panel["down_vol"]
    vix = panel["vix"]

    w1 = hold_sample(r, 78)
    w3 = hold_sample(r, 234)
    m1 = ema(w1, 8)
    m3 = ema(w3, 8)
    bench_mom = ema(hold_sample(rb, 78), 8)
    spread = m1 - m3
    vol = rolling_std(r, 50)
    conviction = np.abs(spread) / (vol + 1e-8)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_trend = ema(np.diff(vix, prepend=vix[0]) / (vix + 1e-8), 8)
    prev_close = np.concatenate([[c[0]], c[:-1]])
    price_trend = ema(np.diff(c, prepend=c[0]) / (prev_close + 1e-8), 34)
    trend_quality = (m1 + m3 + bench_mom + price_trend) / (vol + 1e-8)
    breadth_smooth = ema(breadth, 20)
    conviction_z = conviction / (rolling_mean(conviction, 80) + 1e-8)
    return np.stack(
        [
            r,
            m1,
            m3,
            bench_mom,
            spread,
            vol,
            conviction,
            breadth,
            breadth_smooth,
            volume_breadth,
            vix_trend,
            price_trend,
            trend_quality,
            conviction_z,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v47_swing_1w_3w",
        feature_names=[
            "ret",
            "mom_1w",
            "mom_3w",
            "bench_mom",
            "spread",
            "vol",
            "conviction",
            "breadth",
            "breadth_smooth",
            "volume_breadth",
            "vix_trend",
            "price_trend",
            "trend_quality",
            "conviction_z",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=64,
        horizon=10,
        patience=24,
    )
