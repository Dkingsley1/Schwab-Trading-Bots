import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    champion = ema(r, 12)
    challenger = ema(r - rb, 8)
    long_ref = ema(hold_sample(r, 12), 16)

    champ_score = champion / (rolling_std(r, 20) + 1e-8)
    chall_score = challenger / (rolling_std(r - rb, 20) + 1e-8)

    delta = chall_score - champ_score
    switch_pressure = ema(delta, 6)
    stability = 1.0 / (1.0 + np.abs(np.diff(champ_score, prepend=champ_score[0])))

    return np.stack([r, rb, champ_score, chall_score, delta, switch_pressure, stability, long_ref], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v71_champion_challenger_layer",
        feature_names=["ret", "bench_ret", "champ_score", "chall_score", "delta", "switch_pressure", "stability", "long_ref"],
        feature_builder=build_features,
        window=52,
        horizon=5,
    )
