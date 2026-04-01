import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_requested_bot_common import clip01
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    risk_support_label_builder,
)

_DRIFT_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
]


def rolling_mean(x, w=60):
    out = np.zeros_like(x)
    for i in range(len(x)):
        s = max(0, i - w + 1)
        out[i] = np.mean(x[s : i + 1])
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    v = panel["volume"]

    f1 = r
    f2 = rb
    f3 = np.log(v + 1.0)

    m1, m2, m3 = rolling_mean(f1, 80), rolling_mean(f2, 80), rolling_mean(f3, 80)
    s1, s2, s3 = rolling_std(f1, 80) + 1e-8, rolling_std(f2, 80) + 1e-8, rolling_std(f3, 80) + 1e-8

    z1 = (f1 - m1) / s1
    z2 = (f2 - m2) / s2
    z3 = (f3 - m3) / s3

    drift_mag = np.sqrt(z1 * z1 + z2 * z2 + z3 * z3)
    drift_fast = ema(drift_mag, 8)
    drift_slow = ema(drift_mag, 30)
    drift_guard = np.maximum(drift_fast - drift_slow, 0.0)

    return np.stack([r, rb, z1, z2, z3, drift_mag, drift_fast, drift_slow, drift_guard], axis=1)


def _drift_signal(obs):
    vote_dispersion = max(
        abs(observation_feature(obs, "master_vote") - observation_feature(obs, "grand_master_vote")),
        abs(observation_feature(obs, "options_master_vote") - observation_feature(obs, "futures_master_vote")),
        abs(observation_feature(obs, "options_specialist_vote") - observation_feature(obs, "futures_specialist_vote")),
    )
    friction = clip01(
        abs(observation_feature(obs, "lag_expected_fill_delta_bps")) / 12.0
        + abs(observation_feature(obs, "lag_slippage_bps")) / 12.0
    )
    return clip01(
        (0.38 * vote_dispersion)
        + (0.18 * abs(observation_feature(obs, "news_recent_impact")))
        + (0.16 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.16 * friction)
        + (0.12 * observation_feature(obs, "market_micro_order_flow_imbalance_norm"))
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "master_vote"),
            observation_feature(obs, "grand_master_vote"),
            observation_feature(obs, "options_master_vote"),
            observation_feature(obs, "futures_master_vote"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "active_sub_bots"),
            observation_feature(obs, "active_options_sub_bots"),
            observation_feature(obs, "active_futures_sub_bots"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "master_vote", 6),
            feature_std(sequence, idx, "grand_master_vote", 6),
            feature_ema(sequence, idx, "master_vote", 4),
            feature_ema(sequence, idx, "grand_master_vote", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "active_sub_bots") >= 8.0
        and observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.74
        and _drift_signal(obs) >= 0.12
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.40 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    return (
        (0.38 * _drift_signal(obs))
        + (0.18 * quote_quality)
        + (0.16 * observation_feature(obs, "market_micro_order_flow_imbalance_norm"))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.14 * clip01(abs(observation_feature(obs, "behavior_prior"))))
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v75_model_drift_guard",
        feature_names=["ret", "bench_ret", "z1", "z2", "z3", "drift_mag", "drift_fast", "drift_slow", "drift_guard"],
        feature_builder=build_features,
        window=52,
        horizon=4,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v75_model_drift_guard",
        feature_names=[
            "behavior_prior",
            "master_vote",
            "grand_master_vote",
            "options_master_vote",
            "futures_master_vote",
            "options_specialist_vote",
            "futures_specialist_vote",
            "active_sub_bots",
            "active_options_sub_bots",
            "active_futures_sub_bots",
            "news_recent_impact",
            "breadth_risk_off_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "lag_expected_fill_delta_bps",
            "lag_slippage_bps",
            "lag_latency_ms",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_relative_volume_norm",
            "ret_3",
            "master_vote_std_6",
            "grand_master_vote_std_6",
            "master_vote_ema_4",
            "grand_master_vote_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=risk_support_label_builder(
            min_return=-0.0002,
            max_drawdown=0.018,
            max_realized_vol=0.022,
            vol_multiplier=2.5,
        ),
        mode_allowlist=_DRIFT_MODES,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.30,
        lookback_days=45,
        window=18,
        horizon=6,
        min_samples=256,
        min_sequences=10,
        acted_prob_threshold=0.56,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.56,
        min_acted_accuracy=0.56,
        min_accuracy_lift_over_majority=0.015,
    )


if __name__ == "__main__":
    train_brain()
