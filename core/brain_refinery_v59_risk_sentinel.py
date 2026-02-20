import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def rolling_drawdown(close, window=200):
    out = np.zeros_like(close)
    for i in range(len(close)):
        s = max(0, i - window + 1)
        peak = np.max(close[s : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def rolling_corr(a, b, window=40):
    out = np.zeros_like(a)
    for i in range(len(a)):
        s = max(0, i - window + 1)
        xa = a[s : i + 1]
        xb = b[s : i + 1]
        if len(xa) < 3:
            out[i] = 0.0
            continue
        xa = xa - np.mean(xa)
        xb = xb - np.mean(xb)
        den = np.sqrt(np.sum(xa * xa) * np.sum(xb * xb)) + 1e-8
        out[i] = np.sum(xa * xb) / den
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    rb = panel["bench_ret"]
    vix = panel["vix"]

    vol_fast = rolling_std(r, 12)
    vol_slow = rolling_std(r, 40)
    vol_ratio = vol_fast / (vol_slow + 1e-8)

    dd = rolling_drawdown(c, window=220)
    dd_fast = ema(dd, 10)
    dd_pressure = np.abs(dd_fast)

    corr = rolling_corr(r, rb, window=40)
    corr_stress = np.maximum(np.abs(corr) - 0.7, 0.0)

    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    risk_pressure = ema(vol_ratio + dd_pressure + corr_stress + np.maximum(vix_chg, 0.0), 8)

    return np.stack(
        [r, rb, vol_fast, vol_slow, vol_ratio, dd, dd_pressure, corr, corr_stress, vix_chg, risk_pressure],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v59_risk_sentinel",
        feature_names=[
            "ret",
            "bench_ret",
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "drawdown",
            "dd_pressure",
            "corr",
            "corr_stress",
            "vix_chg",
            "risk_pressure",
        ],
        feature_builder=build_features,
        window=52,
        horizon=4,
    )
