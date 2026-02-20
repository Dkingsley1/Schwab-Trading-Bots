import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    t = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    t1 = ema(t, 3)
    t2 = ema(t, 6)
    jerk = np.diff(t1, prepend=t1[0])
    flip = np.abs(np.diff(np.sign(t), prepend=np.sign(t[0])))
    relv = v / (ema(v, 12) + 1e-8)
    burst = np.abs(t) / (rolling_std(t, 12) + 1e-8)

    return np.stack([r, t, t1, t2, jerk, flip, relv, burst], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v43_intraday_ultrafast_proxy",
        feature_names=["ret", "tick_ret", "t_ema3", "t_ema6", "jerk", "flip", "relv", "burst"],
        feature_builder=build_features,
        window=48,
        horizon=1,
    )
