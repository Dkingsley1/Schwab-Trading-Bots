import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_return,
    observation_feature,
    price_change,
)


def corr(a, b, w=30):
    out = np.zeros_like(a)
    for i in range(len(a)):
        s = max(0, i - w + 1)
        xa = a[s : i + 1]
        xb = b[s : i + 1]
        if len(xa) < 3:
            out[i] = 0.0
            continue
        xa = xa - np.mean(xa)
        xb = xb - np.mean(xb)
        den = np.sqrt(np.sum(xa * xa) * np.sum(xb * xb)) + 1e-8
        out[i] = np.sum(xa * xb) / den
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    ch1 = ema(r, 8)
    ch2 = ema(r, 20)
    ch3 = ema(r - rb, 12)

    c12 = corr(ch1, ch2, 35)
    c13 = corr(ch1, ch3, 35)
    c23 = corr(ch2, ch3, 35)

    corr_load = (np.abs(c12) + np.abs(c13) + np.abs(c23)) / 3.0
    penalty = np.maximum(corr_load - 0.6, 0.0)
    diversity = 1.0 - corr_load

    return np.stack([r, rb, c12, c13, c23, corr_load, penalty, diversity], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _vote_conviction(raw_vote):
    return _clip01(abs(float(raw_vote) - 0.5) * 2.0)


def _diversity_signal(obs):
    active = max(observation_feature(obs, "active_sub_bots"), 1.0)
    options_active = observation_feature(obs, "active_options_sub_bots")
    futures_active = observation_feature(obs, "active_futures_sub_bots")
    mode_mix = _clip01((options_active + futures_active) / max(active, 1.0))
    vote_gap = _clip01(
        max(
            abs(observation_feature(obs, "master_vote") - observation_feature(obs, "grand_master_vote")),
            abs(observation_feature(obs, "options_master_vote") - observation_feature(obs, "futures_master_vote")),
            abs(observation_feature(obs, "options_specialist_vote") - observation_feature(obs, "futures_specialist_vote")),
        )
    )
    return _clip01((0.55 * vote_gap) + (0.45 * mode_mix))


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
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "master_vote", 6),
            feature_std(sequence, idx, "grand_master_vote", 6),
            feature_ema(sequence, idx, "behavior_prior", 4),
            feature_ema(sequence, idx, "master_vote", 4),
            feature_ema(sequence, idx, "grand_master_vote", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.74
        and observation_feature(obs, "active_sub_bots") >= 3.0
        and max(
            _diversity_signal(obs),
            abs(observation_feature(obs, "master_vote") - observation_feature(obs, "grand_master_vote")),
            abs(observation_feature(obs, "options_specialist_vote") - observation_feature(obs, "futures_specialist_vote")),
        )
        >= 0.08
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(
        0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    specialist_dispersion = _clip01(
        abs(observation_feature(obs, "options_specialist_vote") - observation_feature(obs, "futures_specialist_vote"))
    )
    master_dispersion = _clip01(
        abs(observation_feature(obs, "master_vote") - observation_feature(obs, "grand_master_vote"))
    )
    return (
        (0.36 * _diversity_signal(obs))
        + (0.20 * specialist_dispersion)
        + (0.16 * master_dispersion)
        + (0.14 * observation_feature(obs, "market_micro_order_flow_imbalance_norm"))
        + (0.14 * quote_quality)
    )


def _runtime_diversity_label(sequence, idx, horizon):
    obs = sequence[idx]
    diversity = _diversity_signal(obs)
    if diversity < 0.08:
        return None
    future_ret = future_return(sequence, idx, horizon)
    stack_conviction = max(
        abs(observation_feature(obs, "behavior_prior")),
        abs(observation_feature(obs, "master_vote")),
        abs(observation_feature(obs, "grand_master_vote")),
    )
    if abs(future_ret) < max(0.0005, 0.0012 - (0.0005 * stack_conviction)):
        return None
    return 1.0 if future_ret > 0.0 else 0.0


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v67_correlation_penalty_layer",
        feature_names=["ret", "bench_ret", "c12", "c13", "c23", "corr_load", "penalty", "diversity"],
        feature_builder=build_features,
        window=50,
        horizon=4,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v67_correlation_penalty_layer",
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
            "breadth_risk_off_norm",
            "news_recent_impact",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "ret_3",
            "master_vote_std_6",
            "grand_master_vote_std_6",
            "behavior_prior_ema_4",
            "master_vote_ema_4",
            "grand_master_vote_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_diversity_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        lookback_days=30,
        window=18,
        horizon=6,
        min_samples=224,
        min_sequences=8,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )
