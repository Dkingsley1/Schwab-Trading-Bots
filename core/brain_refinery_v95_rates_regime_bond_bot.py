import numpy as np

from indicator_bot_common import atr, ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]
    b = panel["bench_ret"]
    vix = panel.get("vix", np.zeros_like(c))

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    q_alpha = q_ret - q_bench

    trend_fast = ema(q_ret, 6)
    trend_slow = ema(q_ret, 16)
    trend_spread = trend_fast - trend_slow

    atr_l = atr(h, l, c, period=28) / (c + 1e-8)
    vol = rolling_std(r, 120)

    # Proxy for rates risk regime pressure.
    vix_z = (vix - np.mean(vix)) / (np.std(vix) + 1e-8)
    rates_regime = trend_spread - 0.35 * vix_z - 0.25 * vol

    return np.stack([r, q_alpha, trend_fast, trend_slow, trend_spread, atr_l, vol, vix_z, rates_regime], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v95_rates_regime_bond_bot",
        feature_names=[
            "ret",
            "q_alpha",
            "trend_fast",
            "trend_slow",
            "trend_spread",
            "atr_long",
            "vol",
            "vix_z",
            "rates_regime",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )
