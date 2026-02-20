import numpy as np

from indicator_bot_common import adx, atr, ema, rolling_std, train_indicator_bot


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

    q_ret = hold_sample(r, 1170)
    y_ret = hold_sample(r, 4680)
    q_mom = ema(q_ret, 6)
    y_mom = ema(y_ret, 6)

    adx_l = adx(h, l, c, period=28)
    atr_l = atr(h, l, c, period=28) / (c + 1e-8)
    quality = np.abs(q_mom + y_mom) / (rolling_std(r, 120) + 1e-8)

    return np.stack([r, q_mom, y_mom, adx_l, atr_l, quality], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v49_investment_trend_quality",
        feature_names=["ret", "q_mom", "y_mom", "adx_long", "atr_long", "quality"],
        feature_builder=build_features,
        window=72,
        horizon=20,
    )
