import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def event_curve(n, period=390, anchors=(30, 120, 240, 330)):
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t = i % period
        d = min(abs(t - a) for a in anchors)
        out[i] = np.exp(-(d ** 2) / (2.0 * (20.0 ** 2)))
    return out


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    n = len(r)

    evt = event_curve(n)
    shock = np.abs(r) / (rolling_std(r, 20) + 1e-8)
    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)

    decay_fast = ema(shock * evt, 5)
    decay_slow = ema(shock * evt, 20)
    decay_spread = decay_fast - decay_slow
    risk_after_event = np.maximum(vix_chg, 0.0) * evt

    return np.stack([r, evt, shock, vix_chg, decay_fast, decay_slow, decay_spread, risk_after_event], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v54_event_shock_decay",
        feature_names=["ret", "event_proximity", "shock", "vix_chg", "decay_fast", "decay_slow", "decay_spread", "risk_after_event"],
        feature_builder=build_features,
        window=44,
        horizon=4,
    )
