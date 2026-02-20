import numpy as np

from indicator_bot_common import adx, rolling_std, train_indicator_bot, true_range


def dmi(high, low, close, period=14):
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close) + 1e-8

    # light smoothing
    plus_di = 100.0 * np.convolve(plus_dm, np.ones(period) / period, mode="same") / (np.convolve(tr, np.ones(period) / period, mode="same") + 1e-8)
    minus_di = 100.0 * np.convolve(minus_dm, np.ones(period) / period, mode="same") / (np.convolve(tr, np.ones(period) / period, mode="same") + 1e-8)
    return plus_di, minus_di


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    plus_di, minus_di = dmi(h, l, c, period=14)
    adx14 = adx(h, l, c, period=14)

    trend_state = np.where(adx14 > 25.0, 1.0, 0.0)
    dir_state = np.sign(plus_di - minus_di)
    state_flip = np.abs(np.diff(dir_state, prepend=dir_state[0]))
    chop = rolling_std(r, 14)

    return np.stack([r, plus_di, minus_di, adx14, trend_state, dir_state, state_flip, chop], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v35_dmi_state_machine",
        feature_names=["ret", "plus_di", "minus_di", "adx14", "trend_state", "dir_state", "state_flip", "chop"],
        feature_builder=build_features,
    )
