import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]

    t_fast = ema(r, 5)
    t_slow = ema(r, 20)
    cross = np.sign(t_fast) - np.sign(t_slow)

    cross_flip = np.abs(np.diff(cross, prepend=cross[0]))
    momentum_flip = np.abs(np.diff(np.sign(t_fast), prepend=np.sign(t_fast[0])))
    whipsaw = (cross_flip + momentum_flip) / 2.0

    reversal_pressure = ema(whipsaw + np.abs(t_fast - t_slow), 8) / (rolling_std(r, 20) + 1e-8)

    return np.stack([r, t_fast, t_slow, cross, cross_flip, momentum_flip, whipsaw, reversal_pressure], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v78_trend_reversal_sentinel",
        feature_names=["ret", "t_fast", "t_slow", "cross", "cross_flip", "momentum_flip", "whipsaw", "reversal_pressure"],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )
