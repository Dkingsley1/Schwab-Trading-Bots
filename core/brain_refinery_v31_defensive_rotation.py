import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    c = panel["close"]
    b = panel["bench_close"]
    vix = panel["vix"]

    rs = c / (b + 1e-8)
    rs_fast = ema(rs, 8)
    rs_slow = ema(rs, 34)
    rs_spread = (rs_fast - rs_slow) / (rs_slow + 1e-8)

    alpha = r - rb
    alpha_smooth = ema(alpha, 10)
    alpha_vol = rolling_std(alpha, 20)

    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)

    return np.stack([r, rb, alpha, alpha_smooth, alpha_vol, rs_spread, vix_chg], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v31_defensive_rotation",
        feature_names=["ret", "bench_ret", "alpha", "alpha_ema10", "alpha_vol20", "rs_spread", "vix_chg"],
        feature_builder=build_features,
    )
