import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_OVERLAP_MODES = [
    "shadow_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
]
_OVERLAP_SYMBOLS = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "SOL-USD"]


def build_features(panel):
    r = panel["ret"]
    b = panel["bench_ret"]
    corr_proxy = ema(r * b, 12)
    alpha = r - b
    alpha_fast = ema(alpha, 6)
    alpha_slow = ema(alpha, 18)
    overlap = alpha_fast - alpha_slow
    stress = rolling_std(r, 20)

    return np.stack([r, b, corr_proxy, alpha_fast, alpha_slow, overlap, stress], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _overlap_context_signal(obs):
    corr_conf = observation_feature(obs, "market_crypto_corr_confidence_norm")
    fx_conf = observation_feature(obs, "fx_corr_confidence_norm")
    corr_strength = max(
        abs(_centered01(observation_feature(obs, "market_crypto_spy_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "market_crypto_risk_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "fx_crypto_alignment_norm", 0.5))),
    )
    extreme_penalty = max(
        _clip01((abs(_centered01(observation_feature(obs, "market_crypto_spy_corr_norm", 0.5))) - 0.85) / 0.15),
        _clip01((abs(_centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5))) - 0.85) / 0.15),
    )
    return _clip01(
        (0.32 * corr_strength)
        + (0.24 * max(corr_conf, fx_conf))
        + (0.18 * observation_feature(obs, "fx_crypto_alignment_norm"))
        + (0.14 * observation_feature(obs, "behavior_prior"))
        + (0.12 * _quote_quality(obs))
        - (0.14 * extreme_penalty)
    )


def _overlap_regime_gap(obs):
    risk_overlap = max(
        abs(_centered01(observation_feature(obs, "market_crypto_spy_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "market_crypto_risk_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "fx_crypto_alignment_norm", 0.5))),
    )
    defensive_overlap = max(
        abs(_centered01(observation_feature(obs, "market_crypto_tlt_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "market_crypto_gold_corr_norm", 0.5))),
        abs(_centered01(observation_feature(obs, "market_crypto_uup_inverse_corr_norm", 0.5))),
    )
    return float(risk_overlap - (0.55 * defensive_overlap))


def _overlap_bias(obs):
    return float(
        np.clip(
            (0.24 * observation_feature(obs, "behavior_prior"))
            + (0.20 * _centered01(observation_feature(obs, "market_crypto_spy_corr_norm", 0.5)))
            + (0.16 * _centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5)))
            + (0.14 * _centered01(observation_feature(obs, "fx_crypto_alignment_norm", 0.5)))
            + (0.10 * observation_feature(obs, "mom_5m") * 120.0)
            + (0.08 * observation_feature(obs, "pct_from_close") * 100.0)
            - (0.08 * observation_feature(obs, "breadth_risk_off_norm")),
            -1.0,
            1.0,
        )
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "market_crypto_risk_corr_norm"),
            observation_feature(obs, "market_crypto_spy_corr_norm"),
            observation_feature(obs, "market_crypto_qqq_corr_norm"),
            observation_feature(obs, "market_crypto_tlt_corr_norm"),
            observation_feature(obs, "market_crypto_uup_inverse_corr_norm"),
            observation_feature(obs, "market_crypto_gold_corr_norm"),
            observation_feature(obs, "market_crypto_corr_confidence_norm"),
            observation_feature(obs, "fx_crypto_alignment_norm"),
            observation_feature(obs, "fx_corr_confidence_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "market_crypto_spy_corr_norm", 4),
            feature_ema(sequence, idx, "market_crypto_qqq_corr_norm", 4),
            feature_ema(sequence, idx, "fx_crypto_alignment_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    corr_conf = max(
        observation_feature(obs, "market_crypto_corr_confidence_norm"),
        observation_feature(obs, "fx_corr_confidence_norm"),
    )
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.30
        and corr_conf >= 0.22
        and _overlap_context_signal(obs) >= 0.24
        and _overlap_regime_gap(obs) >= 0.04
        and abs(_overlap_bias(obs)) >= 0.10
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.30 * _overlap_context_signal(obs))
        + (0.10 * _clip01(_overlap_regime_gap(obs) / 0.25))
        + (0.22 * _clip01(abs(_overlap_bias(obs)) / 0.9))
        + (0.18 * max(observation_feature(obs, "market_crypto_corr_confidence_norm"), observation_feature(obs, "fx_corr_confidence_norm")))
        + (0.16 * observation_feature(obs, "fx_crypto_alignment_norm"))
        + (0.14 * _quote_quality(obs))
    )


def _runtime_overlap_label(sequence, idx, horizon):
    obs = sequence[idx]
    signal = _overlap_context_signal(obs)
    bias = _overlap_bias(obs)
    regime_gap = _overlap_regime_gap(obs)
    if signal < 0.24 or regime_gap < 0.04 or abs(bias) < 0.10:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00055, 0.00110 - (0.00030 * signal))
    if abs(fwd_ret) < move_threshold and realized < 0.020 and drawdown < 0.012:
        return None

    success_score = (
        signed_ret
        + (0.00075 * signal)
        + (0.00020 * regime_gap)
        + (0.00020 * _quote_quality(obs))
        - (0.18 * realized)
        - (0.24 * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + (0.00065 * signal)
        + (0.16 * realized)
        + (0.20 * drawdown)
    )
    if success_score >= 0.00055:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00060:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v100_stock_crypto_overlap_context",
        feature_names=["ret", "bench_ret", "corr_proxy", "alpha_fast", "alpha_slow", "overlap", "stress"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v100_stock_crypto_overlap_context",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "breadth_risk_off_norm",
            "ctx_VIX_X_pct_from_close",
            "futures_specialist_vote",
            "market_crypto_risk_corr_norm",
            "market_crypto_spy_corr_norm",
            "market_crypto_qqq_corr_norm",
            "market_crypto_tlt_corr_norm",
            "market_crypto_uup_inverse_corr_norm",
            "market_crypto_gold_corr_norm",
            "market_crypto_corr_confidence_norm",
            "fx_crypto_alignment_norm",
            "fx_corr_confidence_norm",
            "ret_3",
            "ret_6",
            "market_crypto_spy_corr_ema_4",
            "market_crypto_qqq_corr_ema_4",
            "fx_crypto_alignment_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_overlap_label,
        mode_allowlist=_OVERLAP_MODES,
        symbol_allowlist=_OVERLAP_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.38,
        lookback_days=45,
        window=18,
        horizon=4,
        min_samples=224,
        min_sequences=4,
        min_positive_samples=32,
        min_negative_samples=32,
        acted_prob_threshold=0.66,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.28,
    )


if __name__ == "__main__":
    train_brain()
