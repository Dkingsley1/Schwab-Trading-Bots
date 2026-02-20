import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    b = panel["bench_close"]
    r = panel["ret"]
    rb = panel["bench_ret"]

    rs = c / (b + 1e-8)
    rs_mom = np.diff(rs, prepend=rs[0]) / (rs + 1e-8)
    alpha = r - rb
    alpha_ema = ema(alpha, 15)
    alpha_vol = rolling_std(alpha, 20)

    return np.stack([r, rb, rs_mom, alpha, alpha_ema, alpha_vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v26_relative_strength_cross_section",
        feature_names=["ret", "bench_ret", "rs_mom", "alpha", "alpha_ema", "alpha_vol"],
        feature_builder=build_features,
    )
