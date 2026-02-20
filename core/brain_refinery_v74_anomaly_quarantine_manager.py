import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    c = panel["close"]
    v = panel["volume"]

    jump = np.abs(r) / (rolling_std(r, 25) + 1e-8)
    gap_like = np.abs(np.diff(c, prepend=c[0]) / (c + 1e-8))
    gap_jump = gap_like / (rolling_std(gap_like, 25) + 1e-8)

    rel_vol = v / (ema(v, 30) + 1e-8)
    low_liq = np.maximum(0.0, 1.0 - rel_vol)

    anomaly_score = ema(jump + 0.7 * gap_jump + 0.6 * low_liq, 8)
    quarantine_pressure = np.maximum(anomaly_score - np.percentile(anomaly_score, 70), 0.0)

    return np.stack([r, jump, gap_jump, rel_vol, low_liq, anomaly_score, quarantine_pressure], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v74_anomaly_quarantine_manager",
        feature_names=["ret", "jump", "gap_jump", "rel_vol", "low_liq", "anomaly_score", "quarantine_pressure"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )
