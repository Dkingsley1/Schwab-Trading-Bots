import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    adv = panel["adv"]
    dec = panel["dec"]
    upv = panel["up_vol"]
    dnv = panel["down_vol"]

    ad_line = (adv - dec) / (adv + dec + 1e-8)
    uv_ratio = (upv - dnv) / (upv + dnv + 1e-8)
    ad_ema = ema(ad_line, 10)
    uv_ema = ema(uv_ratio, 10)
    breadth_vol = rolling_std(ad_line, 20)

    return np.stack([r, ad_line, uv_ratio, ad_ema, uv_ema, breadth_vol], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v28_breadth_internals",
        feature_names=["ret", "ad_line", "up_down_vol_ratio", "ad_ema10", "uv_ema10", "breadth_vol"],
        feature_builder=build_features,
    )
