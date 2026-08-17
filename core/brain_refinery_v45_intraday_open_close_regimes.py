import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot


def session_clock(n, period=390):
    t = np.arange(n) % period
    open_phase = np.exp(-((t - 20) ** 2) / (2 * 18.0 ** 2))
    close_phase = np.exp(-((t - (period - 20)) ** 2) / (2 * 18.0 ** 2))
    mid_phase = np.exp(-((t - period / 2) ** 2) / (2 * 40.0 ** 2))
    return open_phase, mid_phase, close_phase


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    v = panel["volume"]
    adv = panel["adv"]
    dec = panel["dec"]
    up_vol = panel["up_vol"]
    down_vol = panel["down_vol"]
    gap = panel["gap"]
    vix = panel["vix"]
    n = len(r)
    op, mp, cp = session_clock(n)

    r_fast = ema(r, 6)
    r_slow = ema(r, 20)
    bench_fast = ema(rb, 6)
    bench_slow = ema(rb, 20)
    vol = rolling_std(r, 20)
    relv = v / (ema(v, 30) + 1e-8)
    relv_z = relv / (rolling_mean(relv, 30) + 1e-8)
    breadth = (adv - dec) / (adv + dec + 1e-8)
    volume_breadth = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
    vix_change = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    phase_pressure = op * np.abs(r_fast) + cp * np.abs(r_fast)
    opening_drive = op * (r_fast + gap + bench_fast)
    closing_drive = cp * (r_fast + bench_fast)
    phase_trend = (op + cp) * (r_fast - r_slow)

    return np.stack(
        [
            r,
            r_fast,
            r_slow,
            bench_fast,
            bench_slow,
            vol,
            relv,
            relv_z,
            op,
            mp,
            cp,
            breadth,
            volume_breadth,
            vix_change,
            phase_pressure,
            opening_drive,
            closing_drive,
            phase_trend,
        ],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v45_intraday_open_close_regimes",
        feature_names=[
            "ret",
            "r_fast",
            "r_slow",
            "bench_fast",
            "bench_slow",
            "vol",
            "relv",
            "relv_z",
            "open_phase",
            "mid_phase",
            "close_phase",
            "breadth",
            "volume_breadth",
            "vix_change",
            "phase_pressure",
            "opening_drive",
            "closing_drive",
            "phase_trend",
        ],
        feature_builder=build_features,
        num_points=9000,
        window=48,
        horizon=3,
        patience=24,
    )
