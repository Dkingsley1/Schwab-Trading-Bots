import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    vix9d = panel["vix9d"]
    vix3m = panel["vix3m"]

    front_ratio = vix9d / (vix + 1e-8)
    back_ratio = vix / (vix3m + 1e-8)
    slope = (vix3m - vix9d) / (vix + 1e-8)
    term_shock = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    regime = ema(front_ratio - back_ratio, 10)
    realized_vol = rolling_std(r, 20)

    return np.stack([r, front_ratio, back_ratio, slope, term_shock, regime, realized_vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v27_term_structure_vol",
        feature_names=["ret", "vix9d_over_vix", "vix_over_vix3m", "term_slope", "term_shock", "regime", "realized_vol"],
        feature_builder=build_features,
    )
