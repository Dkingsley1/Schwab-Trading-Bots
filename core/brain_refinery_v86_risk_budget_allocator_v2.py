import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def rolling_drawdown(close, window=180):
    out = np.zeros_like(close)
    for i in range(len(close)):
        s = max(0, i - window + 1)
        peak = np.max(close[s : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    rb = panel["bench_ret"]
    vix = panel["vix"]
    vix9d = panel["vix9d"]
    vix3m = panel["vix3m"]

    vol_fast = rolling_std(r, 10)
    vol_slow = rolling_std(r, 40)
    vol_ratio = vol_fast / (vol_slow + 1e-8)

    dd = rolling_drawdown(c, 220)
    dd_pressure = np.abs(ema(dd, 10))

    corr_proxy = ema(r * rb, 20) / (np.sqrt(ema(r * r, 20) * ema(rb * rb, 20)) + 1e-8)
    corr_stress = np.maximum(np.abs(corr_proxy) - 0.65, 0.0)

    term_slope = (vix9d - vix3m) / (vix + 1e-8)
    vol_of_vol = rolling_std(vix, 20) / (rolling_mean(vix, 20) + 1e-8)

    budget_risk = ema(vol_ratio + dd_pressure + corr_stress + np.maximum(term_slope, 0.0) + vol_of_vol, 8)
    risk_budget_score = 1.0 - np.clip(0.35 * vol_ratio + 0.30 * dd_pressure + 0.20 * corr_stress + 0.15 * np.maximum(term_slope, 0.0), 0.0, 1.0)

    return np.stack(
        [r, rb, vol_fast, vol_slow, vol_ratio, dd, dd_pressure, corr_proxy, corr_stress, term_slope, vol_of_vol, budget_risk, risk_budget_score],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v86_risk_budget_allocator_v2",
        feature_names=[
            "ret",
            "bench_ret",
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "drawdown",
            "dd_pressure",
            "corr_proxy",
            "corr_stress",
            "term_slope",
            "vol_of_vol",
            "budget_risk",
            "risk_budget_score",
        ],
        feature_builder=build_features,
        window=52,
        horizon=4,
    )
