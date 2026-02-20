import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    bench = panel["bench_ret"]
    vix = panel["vix"]
    vix3m = panel["vix3m"]

    vol_trend = ema(rolling_std(r, 25), 20)
    demand_proxy = ema(bench, 35)

    # Inflation pressure proxy: volatility term structure + persistent upside drift.
    cpi_proxy = 2.0 + 55.0 * vol_trend + 260.0 * np.maximum(demand_proxy, -0.01)
    pce_proxy = 1.8 + 45.0 * ema(np.abs(r), 30) + 220.0 * np.maximum(ema(bench, 55), -0.01)

    inflation_surprise = (cpi_proxy - rolling_mean(cpi_proxy, 45)) + 0.5 * (pce_proxy - rolling_mean(pce_proxy, 45))
    disinflation_impulse = -ema(inflation_surprise, 18)

    vol_term = (vix3m - vix) / (vix + 1e-8)

    return np.stack(
        [
            r,
            demand_proxy,
            vol_trend,
            cpi_proxy / 10.0,
            pce_proxy / 10.0,
            inflation_surprise / 10.0,
            disinflation_impulse / 10.0,
            vol_term,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v89_macro_inflation_pce_cpi",
        feature_names=[
            "ret",
            "demand_proxy",
            "vol_trend",
            "cpi_proxy",
            "pce_proxy",
            "inflation_surprise",
            "disinflation_impulse",
            "vol_term",
        ],
        feature_builder=build_features,
    )
