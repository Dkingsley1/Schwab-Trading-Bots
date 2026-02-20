import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    bench = panel["bench_ret"]
    vix = panel["vix"]

    growth_trend = ema(bench, 40)
    growth_fast = ema(bench, 12)
    growth_slow = ema(bench, 80)

    # Proxy PMI/ISM diffusion behavior from broad return persistence and volatility drag.
    pmi_proxy = 50.0 + 900.0 * growth_trend - 2.2 * (vix - rolling_mean(vix, 30))
    ism_proxy = 50.0 + 850.0 * growth_fast - 700.0 * rolling_std(r, 20)
    pmi_ism_spread = pmi_proxy - ism_proxy

    soft_landing_score = np.tanh((growth_fast - growth_slow) * 180.0)
    recession_risk = np.tanh((rolling_std(r, 30) * 120.0) - 0.6 * (growth_trend * 1200.0))

    return np.stack(
        [
            r,
            growth_trend,
            growth_fast,
            growth_slow,
            pmi_proxy / 100.0,
            ism_proxy / 100.0,
            pmi_ism_spread / 100.0,
            soft_landing_score,
            recession_risk,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v88_macro_pmi_ism_regime",
        feature_names=[
            "ret",
            "growth_trend",
            "growth_fast",
            "growth_slow",
            "pmi_proxy",
            "ism_proxy",
            "pmi_ism_spread",
            "soft_landing_score",
            "recession_risk",
        ],
        feature_builder=build_features,
    )
