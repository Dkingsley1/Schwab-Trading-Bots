import numpy as np

from indicator_bot_common import rolling_std, train_indicator_bot


def build_features(panel):
    c = panel["close"]
    o = panel["open"]
    r = panel["ret"]
    g = panel["gap"]

    open_drive = (c - o) / (o + 1e-8)
    gap_abs = np.abs(g)
    gap_vol = rolling_std(g, 20)
    continuation = g * np.sign(open_drive)
    first_move = np.diff(open_drive, prepend=open_drive[0])

    return np.stack([r, g, gap_abs, gap_vol, open_drive, continuation, first_move], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v29_gap_open_auction",
        feature_names=["ret", "gap", "gap_abs", "gap_vol", "open_drive", "gap_continuation", "first_move"],
        feature_builder=build_features,
    )
