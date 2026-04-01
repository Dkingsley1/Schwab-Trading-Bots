import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    selective_direction_label_builder,
)


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    vix9d = panel["vix9d"]
    vix3m = panel["vix3m"]

    front_ratio = vix9d / (vix + 1e-8)
    back_ratio = vix / (vix3m + 1e-8)
    slope = (vix3m - vix9d) / (vix + 1e-8)
    term_shock = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    regime = ema(front_ratio - back_ratio, 10)
    realized_vol = rolling_std(r, 20)

    return np.stack([r, front_ratio, back_ratio, slope, term_shock, regime, realized_vol], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered_feature(obs, name):
    return abs(observation_feature(obs, name, 0.5) - 0.5) * 2.0


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "options_chain_available"),
            observation_feature(obs, "options_iv_atm_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_iv_term_structure_norm"),
            observation_feature(obs, "options_put_call_oi_ratio_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "options_0dte_share_norm"),
            observation_feature(obs, "options_net_call_premium_bias_norm"),
            observation_feature(obs, "options_iv_percentile_norm"),
            observation_feature(obs, "options_iv_realized_spread_norm"),
            observation_feature(obs, "options_gamma_front_share_norm"),
            observation_feature(obs, "options_gamma_expiry_skew_norm"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "options_master_vote"),
            observation_feature(obs, "tasty_iv_rank_norm"),
            observation_feature(obs, "tasty_implied_volatility_index_norm"),
            observation_feature(obs, "tasty_liquidity_rating_norm"),
            observation_feature(obs, "tasty_expected_move_norm"),
            observation_feature(obs, "tasty_beta_norm"),
            observation_feature(obs, "tasty_watchlist_presence_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "options_iv_skew_norm", 4),
            feature_ema(sequence, idx, "options_iv_term_structure_norm", 4),
        ],
        dtype=np.float32,
    )


def _options_regime_signal(obs):
    return _clip01(
        max(
            _centered_feature(obs, "options_iv_term_structure_norm"),
            _centered_feature(obs, "options_iv_skew_norm"),
            _centered_feature(obs, "options_iv_realized_spread_norm"),
            _centered_feature(obs, "options_net_call_premium_bias_norm"),
            _centered_feature(obs, "options_gamma_expiry_skew_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "options_0dte_share_norm"),
            observation_feature(obs, "options_gamma_front_share_norm"),
            abs(observation_feature(obs, "options_specialist_vote")),
            abs(observation_feature(obs, "options_master_vote")),
            observation_feature(obs, "tasty_iv_rank_norm"),
            observation_feature(obs, "tasty_implied_volatility_index_norm"),
        )
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "options_chain_available") >= 0.5
        and observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.75
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.35
        and _options_regime_signal(obs) >= 0.18
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.40 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    tasty_signal = _clip01(
        0.40 * observation_feature(obs, "tasty_iv_rank_norm")
        + 0.20 * observation_feature(obs, "tasty_implied_volatility_index_norm")
        + 0.20 * observation_feature(obs, "tasty_liquidity_rating_norm")
        + 0.20 * observation_feature(obs, "tasty_watchlist_presence_norm")
    )
    options_signal = _clip01(
        0.35 * _options_regime_signal(obs)
        + 0.20 * observation_feature(obs, "options_iv_atm_norm")
        + 0.15 * observation_feature(obs, "options_vol_expectation_norm")
        + 0.15 * observation_feature(obs, "options_unusual_flow_norm")
        + 0.15 * observation_feature(obs, "options_0dte_share_norm")
    )
    return (0.44 * options_signal) + (0.24 * tasty_signal) + (0.20 * quote_quality) + (0.12 * _clip01(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03))


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v27_term_structure_vol",
        feature_names=["ret", "vix9d_over_vix", "vix_over_vix3m", "term_slope", "term_shock", "regime", "realized_vol"],
        feature_builder=build_features,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v27_term_structure_vol",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "ctx_VIX_X_pct_from_close",
            "options_chain_available",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_iv_term_structure_norm",
            "options_put_call_oi_ratio_norm",
            "options_vol_expectation_norm",
            "options_unusual_flow_norm",
            "options_0dte_share_norm",
            "options_net_call_premium_bias_norm",
            "options_iv_percentile_norm",
            "options_iv_realized_spread_norm",
            "options_gamma_front_share_norm",
            "options_gamma_expiry_skew_norm",
            "options_specialist_vote",
            "options_master_vote",
            "tasty_iv_rank_norm",
            "tasty_implied_volatility_index_norm",
            "tasty_liquidity_rating_norm",
            "tasty_expected_move_norm",
            "tasty_beta_norm",
            "tasty_watchlist_presence_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "options_iv_skew_ema_4",
            "options_iv_term_structure_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=selective_direction_label_builder(min_abs_return=0.0008),
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.40,
        lookback_days=30,
        window=24,
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
