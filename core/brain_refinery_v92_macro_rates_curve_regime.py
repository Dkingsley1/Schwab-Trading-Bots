import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    bench = panel["bench_ret"]
    vix = panel["vix"]
    vix3m = panel["vix3m"]

    short_rate_proxy = 2.0 + 180.0 * ema(rolling_std(r, 20), 12)
    long_rate_proxy = 2.2 + 140.0 * ema(rolling_std(bench, 45), 20) + 90.0 * np.maximum(ema(bench, 70), -0.01)

    curve_slope_proxy = long_rate_proxy - short_rate_proxy
    real_rate_proxy = short_rate_proxy - (2.0 + 40.0 * ema(np.abs(r), 35))

    term_vol = (vix3m - vix) / (vix + 1e-8)
    curve_inversion_risk = np.tanh(-curve_slope_proxy / 1.8)
    liquidity_fracture = np.tanh(1.3 * term_vol + 0.9 * (rolling_std(r, 30) * 130.0))

    return np.stack(
        [
            r,
            short_rate_proxy / 10.0,
            long_rate_proxy / 10.0,
            curve_slope_proxy / 10.0,
            real_rate_proxy / 10.0,
            term_vol,
            curve_inversion_risk,
            liquidity_fracture,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v92_macro_rates_curve_regime",
        feature_names=[
            "ret",
            "short_rate_proxy",
            "long_rate_proxy",
            "curve_slope_proxy",
            "real_rate_proxy",
            "term_vol",
            "curve_inversion_risk",
            "liquidity_fracture",
        ],
        feature_builder=build_features,
    )
