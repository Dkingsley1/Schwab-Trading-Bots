import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def event_proximity(n, cycle=390, anchors=(60, 180, 300)):
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t = i % cycle
        d = min(abs(t - a) for a in anchors)
        out[i] = np.exp(-(d ** 2) / (2.0 * (25.0 ** 2)))
    return out


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    n = len(r)

    eprox = event_proximity(n)
    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    risk_boost = eprox * np.maximum(vix_chg, 0.0)
    ret_noise = rolling_std(r, 20)
    ret_ema = ema(r, 10)
    jump = np.abs(r) / (ret_noise + 1e-8)

    return np.stack([r, ret_ema, eprox, vix_chg, risk_boost, ret_noise, jump], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v39_event_risk_proximity",
        feature_names=["ret", "ret_ema10", "event_proximity", "vix_chg", "risk_boost", "ret_noise", "jump"],
        feature_builder=build_features,
    )
