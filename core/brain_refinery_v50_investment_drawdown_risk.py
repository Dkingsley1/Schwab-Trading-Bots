import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def rolling_drawdown(close, window=200):
    dd = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        peak = np.max(close[start : i + 1])
        dd[i] = (close[i] - peak) / (peak + 1e-8)
    return dd


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def rolling_zscore(x, window):
    mean = rolling_mean(x, window)
    std = rolling_std(x, window) + 1e-8
    return (x - mean) / std


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]
    bench = panel["bench_close"]

    dd = rolling_drawdown(c, window=260)
    bench_dd = rolling_drawdown(bench, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)
    dd_slope = dd_fast - dd_slow
    dd_accel = ema(dd_slope, 5) - ema(dd_slope, 21)
    drawdown_z = rolling_zscore(dd, 120)
    relative_dd = dd - bench_dd
    crash_prob = np.maximum(-dd_slope, 0.0) + np.maximum(-dd_accel, 0.0)
    drawdown_repair = np.maximum(dd - dd_fast, 0.0)
    distance_to_high = -dd

    qret = hold_sample(r, 1170)
    qbeta = hold_sample(b, 1170)
    qalpha = qret - qbeta
    mom_20 = rolling_mean(r, 20)
    mom_60 = rolling_mean(r, 60)
    mom_120 = rolling_mean(r, 120)
    bench_mom_60 = rolling_mean(b, 60)
    rel_strength = ema(r - b, 21)
    beta_proxy = rolling_mean(r * b, 80) / (rolling_std(b, 80) ** 2 + 1e-8)
    vol_fast = rolling_std(r, 40)
    vol_slow = rolling_std(r, 120) + 1e-8
    vol_ratio = vol_fast / vol_slow
    downside_vol = rolling_std(np.minimum(r, 0.0), 80)
    rebound_pressure = np.maximum(ema(r, 13), 0.0) - np.maximum(-dd_slope, 0.0)
    breadth_pressure = (panel["adv"] - panel["dec"]) / (panel["adv"] + panel["dec"] + 1e-8)
    volume_pressure = (panel["up_vol"] - panel["down_vol"]) / (panel["up_vol"] + panel["down_vol"] + 1e-8)
    vix_term = (panel["vix9d"] - panel["vix3m"]) / (panel["vix"] + 1e-8)
    investment_cycle = np.arange(len(c), dtype=np.float64) * (2.0 * np.pi / 1100.0)
    cycle_sin = np.sin(investment_cycle)
    cycle_cos = np.cos(investment_cycle)

    return np.stack(
        [
            r,
            dd,
            dd_fast,
            dd_slow,
            dd_slope,
            dd_accel,
            drawdown_z,
            bench_dd,
            relative_dd,
            crash_prob,
            drawdown_repair,
            distance_to_high,
            qalpha,
            mom_20,
            mom_60,
            mom_120,
            bench_mom_60,
            rel_strength,
            beta_proxy,
            vol_fast,
            vol_slow,
            vol_ratio,
            downside_vol,
            rebound_pressure,
            breadth_pressure,
            volume_pressure,
            vix_term,
            cycle_sin,
            cycle_cos,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v50_investment_drawdown_risk",
        feature_names=[
            "ret",
            "drawdown",
            "dd_fast",
            "dd_slow",
            "dd_slope",
            "dd_accel",
            "drawdown_z",
            "bench_drawdown",
            "relative_drawdown",
            "crash_prob",
            "drawdown_repair",
            "distance_to_high",
            "qalpha",
            "mom_20",
            "mom_60",
            "mom_120",
            "bench_mom_60",
            "rel_strength",
            "beta_proxy",
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "downside_vol",
            "rebound_pressure",
            "breadth_pressure",
            "volume_pressure",
            "vix_term",
            "cycle_sin",
            "cycle_cos",
        ],
        feature_builder=build_features,
        window=96,
        horizon=26,
        learning_rate=0.0006,
        epochs=260,
        batch_size=96,
        patience=24,
    )
