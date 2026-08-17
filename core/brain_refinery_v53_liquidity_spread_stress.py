import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    v = panel["volume"]
    r = panel["ret"]
    bench_r = panel["bench_ret"]
    adv = panel["adv"]
    dec = panel["dec"]
    up_vol = panel["up_vol"]
    down_vol = panel["down_vol"]
    vix = panel["vix"]

    spread_proxy = (h - l) / (c + 1e-8)
    spread_fast = ema(spread_proxy, 6)
    spread_slow = ema(spread_proxy, 20)
    spread_stress = spread_fast / (spread_slow + 1e-8)
    spread_slope = np.diff(spread_fast, prepend=spread_fast[0])

    rel_vol = v / (ema(v, 20) + 1e-8)
    illiquidity = np.abs(r) / (rel_vol + 1e-8)
    stress = spread_stress * (1.0 + illiquidity)
    stress_vol = rolling_std(stress, 20)
    stress_z = (stress - rolling_mean(stress, 40)) / (rolling_std(stress, 40) + 1e-8)
    bench_mom = rolling_mean(bench_r, 12)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_change = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    volatility_regime = rolling_std(r, 20) / (rolling_std(r, 60) + 1e-8)

    return np.stack(
        [
            r,
            bench_r,
            bench_mom,
            spread_proxy,
            spread_stress,
            spread_slope,
            rel_vol,
            illiquidity,
            stress,
            stress_vol,
            stress_z,
            breadth,
            volume_breadth,
            vix_change,
            volatility_regime,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v53_liquidity_spread_stress",
        feature_names=[
            "ret",
            "bench_ret",
            "bench_mom_12",
            "spread_proxy",
            "spread_stress",
            "spread_slope",
            "relative_volume_20",
            "illiquidity",
            "stress",
            "stress_vol",
            "stress_z_40",
            "breadth_adv_dec",
            "volume_breadth",
            "vix_change",
            "volatility_regime_20_60",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=48,
        horizon=3,
        learning_rate=0.0007,
        epochs=260,
        patience=24,
    )
