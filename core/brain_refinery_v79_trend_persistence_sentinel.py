import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def streak(signals):
    out = np.zeros_like(signals)
    run = 0
    prev = 0
    for i, s in enumerate(signals):
        sgn = int(np.sign(s))
        if sgn != 0 and sgn == prev:
            run += 1
        elif sgn != 0:
            run = 1
        else:
            run = 0
        prev = sgn if sgn != 0 else prev
        out[i] = run
    return out


def build_features(panel):
    r = panel["ret"]

    t_fast = ema(r, 6)
    t_mid = ema(r, 16)
    t_slow = ema(r, 32)

    s_fast = streak(t_fast)
    s_mid = streak(t_mid)
    s_slow = streak(t_slow)

    align = (np.sign(t_fast) == np.sign(t_mid)).astype(float) + (np.sign(t_mid) == np.sign(t_slow)).astype(float)
    persistence = (0.5 * s_fast + 0.3 * s_mid + 0.2 * s_slow) / (rolling_std(r, 25) + 1e-8)

    return np.stack([r, t_fast, t_mid, t_slow, s_fast, s_mid, s_slow, align, persistence], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v79_trend_persistence_sentinel",
        feature_names=["ret", "t_fast", "t_mid", "t_slow", "s_fast", "s_mid", "s_slow", "align", "persistence"],
        feature_builder=build_features,
        window=52,
        horizon=5,
    )
