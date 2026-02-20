import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def session_clock(n, period=390):
    t = np.arange(n) % period
    open_phase = np.exp(-((t - 20) ** 2) / (2 * 18.0 ** 2))
    close_phase = np.exp(-((t - (period - 20)) ** 2) / (2 * 18.0 ** 2))
    mid_phase = np.exp(-((t - period / 2) ** 2) / (2 * 40.0 ** 2))
    return open_phase, mid_phase, close_phase


def build_features(panel):
    r = panel["ret"]
    v = panel["volume"]
    n = len(r)
    op, mp, cp = session_clock(n)

    r_fast = ema(r, 6)
    r_slow = ema(r, 20)
    vol = rolling_std(r, 20)
    relv = v / (ema(v, 30) + 1e-8)
    phase_pressure = op * np.abs(r_fast) + cp * np.abs(r_fast)

    return np.stack([r, r_fast, r_slow, vol, relv, op, mp, cp, phase_pressure], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v45_intraday_open_close_regimes",
        feature_names=["ret", "r_fast", "r_slow", "vol", "relv", "open_phase", "mid_phase", "close_phase", "phase_pressure"],
        feature_builder=build_features,
        window=42,
        horizon=3,
    )
