import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    c = panel["close"]

    # Proxy for bot-level regime fitness using horizon-aligned alpha windows.
    alpha = r - rb
    alpha_fast = ema(alpha, 8)
    alpha_mid = ema(alpha, 24)
    alpha_slow = ema(alpha, 60)

    regime_fit = np.sign(alpha_fast) * np.sign(alpha_mid)
    persistence = np.sign(alpha_mid) * np.sign(alpha_slow)
    alpha_vol = rolling_std(alpha, 30)

    draw_proxy = np.minimum(np.diff(c, prepend=c[0]) / (c + 1e-8), 0.0)
    draw_stress = np.sqrt(rolling_std(draw_proxy, 40) + 1e-8)

    rank_signal = (0.5 * alpha_fast + 0.3 * alpha_mid + 0.2 * alpha_slow) / (alpha_vol + 1e-8)

    return np.stack(
        [r, rb, alpha_fast, alpha_mid, alpha_slow, regime_fit, persistence, alpha_vol, draw_stress, rank_signal],
        axis=1,
    )


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v56_meta_ranker",
        feature_names=[
            "ret",
            "bench_ret",
            "alpha_fast",
            "alpha_mid",
            "alpha_slow",
            "regime_fit",
            "persistence",
            "alpha_vol",
            "draw_stress",
            "rank_signal",
        ],
        feature_builder=build_features,
        window=54,
        horizon=6,
    )
