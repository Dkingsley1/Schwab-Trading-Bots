import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    bench = panel["bench_ret"]
    vix = panel["vix"]
    upv = panel["up_vol"]
    dnv = panel["down_vol"]

    # FOMC tone proxy from risk-on breadth + volatility response.
    breadth_impulse = (upv - dnv) / (upv + dnv + 1e-8)
    risk_trend = ema(bench, 20)
    vol_stress = ema(rolling_std(r, 18), 14)

    fomc_hawkish_proxy = np.tanh(2.2 * (vol_stress * 120.0) - 3.0 * risk_trend * 120.0)
    fomc_dovish_proxy = np.tanh(3.2 * risk_trend * 120.0 - 1.6 * (vix - rolling_mean(vix, 25)))
    liquidity_tightness = np.tanh(1.4 * (vix - rolling_mean(vix, 35)) + 1.1 * (rolling_std(r, 30) * 130.0))

    policy_shock = ema(np.diff(vix, prepend=vix[0]) / (vix + 1e-8), 8)

    return np.stack(
        [
            r,
            breadth_impulse,
            risk_trend,
            vol_stress,
            fomc_hawkish_proxy,
            fomc_dovish_proxy,
            liquidity_tightness,
            policy_shock,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v90_macro_fomc_tone_liquidity",
        feature_names=[
            "ret",
            "breadth_impulse",
            "risk_trend",
            "vol_stress",
            "fomc_hawkish_proxy",
            "fomc_dovish_proxy",
            "liquidity_tightness",
            "policy_shock",
        ],
        feature_builder=build_features,
    )
