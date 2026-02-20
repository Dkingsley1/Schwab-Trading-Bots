import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    bench = panel["bench_ret"]
    vix = panel["vix"]
    adv = panel["adv"]
    dec = panel["dec"]

    breadth = (adv - dec) / (adv + dec + 1e-8)
    growth = ema(bench, 28)
    vol = ema(rolling_std(r, 22), 10)

    # Labor shock proxy: growth deceleration + volatility expansion + weak breadth.
    labor_strength = np.tanh(4.0 * growth - 1.8 * vol)
    labor_weakness = np.tanh(1.6 * vol - 3.6 * growth - 1.1 * breadth)

    nfp_surprise_proxy = ema(np.diff(breadth, prepend=breadth[0]), 6) - ema(np.diff(vix, prepend=vix[0]) / (vix + 1e-8), 6)
    unemployment_drift_proxy = rolling_mean(labor_weakness, 18) - rolling_mean(labor_strength, 18)

    return np.stack(
        [
            r,
            breadth,
            growth,
            vol,
            labor_strength,
            labor_weakness,
            nfp_surprise_proxy,
            unemployment_drift_proxy,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v91_macro_labor_shock",
        feature_names=[
            "ret",
            "breadth",
            "growth",
            "vol",
            "labor_strength",
            "labor_weakness",
            "nfp_surprise_proxy",
            "unemployment_drift_proxy",
        ],
        feature_builder=build_features,
    )
