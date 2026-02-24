import numpy as np

from indicator_bot_common import adx, atr, ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def downside_vol(x, window=120):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        seg = x[start : i + 1]
        neg = np.minimum(seg, 0.0)
        out[i] = np.sqrt(np.mean(neg * neg))
    return out


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]
    b = panel["bench_ret"]

    # Slow-sampled momentum proxies for quality compounding behavior.
    q_ret = hold_sample(r, 1170)
    y_ret = hold_sample(r, 4680)
    q_mom = ema(q_ret, 8)
    y_mom = ema(y_ret, 8)

    rel_alpha = q_ret - hold_sample(b, 1170)
    adx_l = adx(h, l, c, period=28)
    atr_l = atr(h, l, c, period=28) / (c + 1e-8)

    dvol = downside_vol(r, window=150)
    vol = rolling_std(r, 150)

    # Proxy "dividend quality": persistent relative strength with controlled downside.
    quality_compound = (np.maximum(q_mom + y_mom + 0.5 * rel_alpha, 0.0)) / (dvol + 0.5 * vol + 1e-8)

    return np.stack([r, q_mom, y_mom, rel_alpha, adx_l, atr_l, dvol, quality_compound], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v93_dividend_quality_compounder",
        feature_names=[
            "ret",
            "q_mom",
            "y_mom",
            "rel_alpha",
            "adx_long",
            "atr_long",
            "downside_vol",
            "quality_compound",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )
