import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    observation_feature,
    price_change,
    symbol_role_features,
)


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


_META_ROLE_MAP = {
    "shock": ["UVXY", "VIXY", "SOXL", "SOXS", "MSTR", "COIN", "TSLA", "NVDA"],
    "bond": ["TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "AGG", "BND"],
    "core_index": ["SPY", "QQQ", "IWM", "DIA"],
    "crypto": ["AVAX-USD", "BTC-USD", "DOGE-USD", "ETH-USD", "LINK-USD", "LTC-USD", "SOL-USD"],
}


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _meta_vote(obs):
    return float(
        (0.22 * observation_feature(obs, "behavior_prior"))
        + (0.18 * observation_feature(obs, "master_vote"))
        + (0.22 * observation_feature(obs, "grand_master_vote"))
        + (0.12 * observation_feature(obs, "options_master_vote"))
        + (0.10 * observation_feature(obs, "futures_master_vote"))
        + (0.08 * observation_feature(obs, "options_specialist_vote"))
        + (0.08 * observation_feature(obs, "futures_specialist_vote"))
    )


def _quote_quality(obs):
    quote_deviation = observation_feature(obs, "data_quality_quote_deviation_norm")
    return _clip01(
        0.55 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.25 * (1.0 - quote_deviation)
        + 0.20 * (1.0 - observation_feature(obs, "data_quality_missing_feature_ratio_norm"))
    )


def _regime_intensity(obs):
    return max(
        observation_feature(obs, "news_recent_impact"),
        observation_feature(obs, "calendar_macro_abs_surprise_norm"),
        observation_feature(obs, "calendar_treasury_auction_norm"),
        observation_feature(obs, "bond_auction_window_norm"),
        observation_feature(obs, "market_micro_options_flow_norm"),
        observation_feature(obs, "market_micro_short_pressure_norm"),
        observation_feature(obs, "market_micro_credit_flow_norm"),
        observation_feature(obs, "market_micro_block_trade_norm"),
        observation_feature(obs, "market_micro_relative_volume_norm"),
        observation_feature(obs, "day_regime_trend_norm"),
        observation_feature(obs, "day_regime_chop_norm"),
    )


def _crypto_context(obs):
    return max(
        observation_feature(obs, "crypto_coingecko_momentum_norm"),
        observation_feature(obs, "crypto_hyperliquid_funding_norm"),
        observation_feature(obs, "market_crypto_current_alignment_norm"),
        observation_feature(obs, "market_crypto_corr_confidence_norm"),
        1.0 - observation_feature(obs, "market_crypto_divergence_norm"),
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _META_ROLE_MAP)
    meta_vote = _meta_vote(obs)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "active_sub_bots"),
            observation_feature(obs, "active_options_sub_bots"),
            observation_feature(obs, "active_futures_sub_bots"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "master_vote"),
            observation_feature(obs, "grand_master_vote"),
            observation_feature(obs, "options_master_vote"),
            observation_feature(obs, "futures_master_vote"),
            meta_vote,
            abs(meta_vote),
            observation_feature(obs, "news_sentiment"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "news_source_quality_norm"),
            observation_feature(obs, "news_entity_relevance_norm"),
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_treasury_auction_norm"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "fx_usd_strength_norm"),
            observation_feature(obs, "fx_eurusd_momentum_norm"),
            observation_feature(obs, "fx_usdjpy_momentum_norm"),
            observation_feature(obs, "fx_proxy_agreement_norm"),
            observation_feature(obs, "fx_risk_on_alignment_norm"),
            observation_feature(obs, "fx_crypto_alignment_norm"),
            observation_feature(obs, "fx_corr_confidence_norm"),
            observation_feature(obs, "bond_curve_2s10s_norm"),
            observation_feature(obs, "bond_real_yield_10y_norm"),
            observation_feature(obs, "bond_credit_risk_on_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_hy_ig_flow_norm"),
            observation_feature(obs, "bond_auction_window_norm"),
            observation_feature(obs, "bond_auction_tail_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "data_quality_missing_feature_ratio_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "infra_risk_throttle_norm"),
            observation_feature(obs, "infra_veto_active"),
            observation_feature(obs, "market_micro_opening_auction_norm"),
            observation_feature(obs, "market_micro_closing_auction_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_options_flow_norm"),
            observation_feature(obs, "market_micro_short_pressure_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "crypto_coingecko_momentum_norm"),
            observation_feature(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "market_crypto_current_alignment_norm"),
            observation_feature(obs, "market_crypto_divergence_norm"),
            observation_feature(obs, "market_crypto_corr_confidence_norm"),
            _regime_intensity(obs),
            _crypto_context(obs),
            observation_feature(obs, "lag_adjusted_return_1m"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_chop_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "behavior_prior", 4),
            feature_ema(sequence, idx, "master_vote", 4),
            feature_ema(sequence, idx, "grand_master_vote", 4),
            feature_ema(sequence, idx, "news_sentiment", 4),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "master_vote", 6),
            roles["role_shock"],
            roles["role_bond"],
            roles["role_core_index"],
            roles["role_crypto"],
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    regime_signal = max(
        observation_feature(obs, "news_recent_impact"),
        observation_feature(obs, "calendar_macro_abs_surprise_norm"),
        observation_feature(obs, "calendar_treasury_auction_norm"),
        observation_feature(obs, "bond_auction_window_norm"),
        observation_feature(obs, "market_micro_options_flow_norm"),
        observation_feature(obs, "market_micro_short_pressure_norm"),
        observation_feature(obs, "market_micro_credit_flow_norm"),
        observation_feature(obs, "market_micro_relative_volume_norm"),
        observation_feature(obs, "market_micro_block_trade_norm"),
        observation_feature(obs, "day_regime_trend_norm"),
        observation_feature(obs, "day_regime_chop_norm"),
        observation_feature(obs, "crypto_coingecko_momentum_norm"),
        observation_feature(obs, "market_crypto_current_alignment_norm"),
        observation_feature(obs, "market_crypto_corr_confidence_norm"),
        abs(observation_feature(obs, "behavior_prior")),
        abs(observation_feature(obs, "master_vote")),
        abs(observation_feature(obs, "grand_master_vote")),
    )
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.72
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.36
        and observation_feature(obs, "data_quality_missing_feature_ratio_norm", 0.0) <= 0.32
        and regime_signal >= 0.10
        and max(
            abs(observation_feature(obs, "behavior_prior")),
            abs(observation_feature(obs, "master_vote")),
            abs(observation_feature(obs, "grand_master_vote")),
        ) >= 0.06
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    prior_signal = abs(observation_feature(obs, "behavior_prior"))
    specialist_signal = max(
        abs(observation_feature(obs, "options_specialist_vote")),
        abs(observation_feature(obs, "futures_specialist_vote")),
    )
    macro_signal = max(
        abs(observation_feature(obs, "news_recent_impact")),
        abs(observation_feature(obs, "calendar_macro_abs_surprise_norm")),
        abs(observation_feature(obs, "breadth_risk_off_norm") - 0.5) * 2.0,
    )
    meta_signal = max(
        abs(observation_feature(obs, "master_vote")),
        abs(observation_feature(obs, "grand_master_vote")),
        abs(observation_feature(obs, "options_master_vote")),
        abs(observation_feature(obs, "futures_master_vote")),
    )
    regime_signal = max(
        observation_feature(obs, "calendar_treasury_auction_norm"),
        observation_feature(obs, "bond_auction_window_norm"),
        observation_feature(obs, "market_micro_relative_volume_norm"),
        observation_feature(obs, "market_micro_options_flow_norm"),
        observation_feature(obs, "market_micro_short_pressure_norm"),
        observation_feature(obs, "market_micro_credit_flow_norm"),
        observation_feature(obs, "market_micro_block_trade_norm"),
        observation_feature(obs, "day_regime_trend_norm"),
        observation_feature(obs, "day_regime_chop_norm"),
        observation_feature(obs, "crypto_coingecko_momentum_norm"),
        observation_feature(obs, "market_crypto_current_alignment_norm"),
        observation_feature(obs, "market_crypto_corr_confidence_norm"),
    )
    realized_feedback = min(abs(observation_feature(obs, "lag_adjusted_return_1m")) / 0.0035, 1.0)
    quote_signal = max(
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
        1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0),
    )
    return float(
        np.clip(
            (0.22 * prior_signal)
            + (0.16 * specialist_signal)
            + (0.18 * meta_signal)
            + (0.16 * macro_signal)
            + (0.14 * regime_signal)
            + (0.06 * realized_feedback)
            + (0.08 * quote_signal),
            0.0,
            1.0,
        )
    )


def _runtime_meta_label(sequence, idx, horizon):
    obs = sequence[idx]
    if (idx + horizon) >= len(sequence):
        return None
    curr_price = observation_feature(obs, "last_price", 0.0)
    fut_price = observation_feature(sequence[idx + horizon], "last_price", 0.0) if (idx + horizon) < len(sequence) else 0.0
    if curr_price <= 0.0 or fut_price <= 0.0:
        return None
    future_ret = (fut_price / max(curr_price, 1e-8)) - 1.0
    dd = abs(future_max_drawdown(sequence, idx, horizon))
    realized_vol = future_realized_vol(sequence, idx, horizon)
    realized_feedback = observation_feature(obs, "lag_adjusted_return_1m")
    roles = symbol_role_features(str(obs.get("symbol") or ""), _META_ROLE_MAP)

    meta_vote = _meta_vote(obs)
    vote_conviction = abs(meta_vote)
    if vote_conviction < 0.08:
        return None
    vote_direction = 1.0 if meta_vote >= 0.0 else 0.0

    regime_intensity = _regime_intensity(obs)
    crypto_context = _crypto_context(obs)
    quote_deviation = observation_feature(obs, "data_quality_quote_deviation_norm")
    quote_quality = _quote_quality(obs)
    infra_risk = observation_feature(obs, "infra_risk_throttle_norm")
    risk_off = max(
        observation_feature(obs, "breadth_risk_off_norm"),
        observation_feature(obs, "bond_credit_risk_off_norm"),
        _clip01(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03),
    )
    stable_regime = (0.42 * regime_intensity) + (0.24 * quote_quality) + (0.18 * (1.0 - infra_risk)) + (0.16 * crypto_context)
    if stable_regime < 0.18 or quote_quality < 0.70:
        return None
    realized_edge = (0.74 * future_ret) + (0.26 * realized_feedback)
    directional_edge = abs(realized_edge)
    direction_label = 1.0 if realized_edge >= 0.0 else 0.0
    vote_alignment = 1.0 if abs(realized_edge) > 0.0 and np.sign(meta_vote) == np.sign(realized_edge) else 0.0
    move_threshold = max(0.00052, 0.00145 - (0.00082 * stable_regime))
    if directional_edge < move_threshold:
        if (
            roles["role_crypto"] > 0.5
            and max(infra_risk, observation_feature(obs, "infra_veto_active")) >= 0.60
            and vote_conviction >= 0.12
            and realized_edge <= -0.00020
            and meta_vote <= 0.0
        ):
            return 0.0
        return None

    if vote_alignment <= 0.0:
        return None

    directional_score = (
        directional_edge
        + (0.16 * regime_intensity)
        + (0.10 * quote_quality)
        + (0.05 * vote_conviction)
        + (0.05 * crypto_context)
        + (0.03 * vote_alignment)
        + (0.06 * roles["role_crypto"] * max(0.0, crypto_context - 0.45))
        - (0.62 * dd)
        - (0.24 * realized_vol)
        - (0.12 * infra_risk)
        - (0.12 * quote_deviation)
    )
    direction_gate = (
        0.00098
        + (0.00040 * max(risk_off - 0.45, 0.0))
        + (0.00022 * max(infra_risk - 0.55, 0.0))
        + (0.00008 * roles["role_crypto"])
        - (0.00018 * max(0.0, crypto_context - 0.60))
        - (0.00008 * vote_alignment)
    )
    if directional_score >= direction_gate:
        return vote_direction
    return None


def _train_synthetic():
    return train_indicator_bot(
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


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v56_meta_ranker",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "behavior_prior",
            "active_sub_bots",
            "active_options_sub_bots",
            "active_futures_sub_bots",
            "options_specialist_vote",
            "futures_specialist_vote",
            "master_vote",
            "grand_master_vote",
            "options_master_vote",
            "futures_master_vote",
            "meta_vote_signed",
            "meta_vote_abs",
            "news_sentiment",
            "news_recent_impact",
            "news_source_quality_norm",
            "news_entity_relevance_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "calendar_macro_abs_surprise_norm",
            "calendar_treasury_auction_norm",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "fx_usd_strength_norm",
            "fx_eurusd_momentum_norm",
            "fx_usdjpy_momentum_norm",
            "fx_proxy_agreement_norm",
            "fx_risk_on_alignment_norm",
            "fx_crypto_alignment_norm",
            "fx_corr_confidence_norm",
            "bond_curve_2s10s_norm",
            "bond_real_yield_10y_norm",
            "bond_credit_risk_on_norm",
            "bond_credit_risk_off_norm",
            "bond_hy_ig_flow_norm",
            "bond_auction_window_norm",
            "bond_auction_tail_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "data_quality_missing_feature_ratio_norm",
            "data_quality_market_data_latency_norm",
            "infra_risk_throttle_norm",
            "infra_veto_active",
            "market_micro_opening_auction_norm",
            "market_micro_closing_auction_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_options_flow_norm",
            "market_micro_short_pressure_norm",
            "market_micro_credit_flow_norm",
            "market_micro_block_trade_norm",
            "options_unusual_flow_norm",
            "crypto_coingecko_momentum_norm",
            "crypto_hyperliquid_funding_norm",
            "market_crypto_current_alignment_norm",
            "market_crypto_divergence_norm",
            "market_crypto_corr_confidence_norm",
            "regime_intensity",
            "crypto_context",
            "lag_adjusted_return_1m",
            "lag_slippage_bps",
            "day_regime_trend_norm",
            "day_regime_chop_norm",
            "day_regime_alignment_norm",
            "ret_3",
            "ret_6",
            "behavior_prior_ema_4",
            "master_vote_ema_4",
            "grand_master_vote_ema_4",
            "news_sentiment_ema_4",
            "pct_from_close_std_6",
            "master_vote_std_6",
            "role_shock",
            "role_bond",
            "role_core_index",
            "role_crypto",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_meta_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.32,
        lookback_days=45,
        window=18,
        horizon=6,
        min_samples=1400,
        min_sequences=10,
        min_positive_samples=120,
        min_negative_samples=120,
        acted_prob_threshold=0.54,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6800,
        max_final_val_loss=0.7150,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.54,
        min_long_acted_count=8,
        min_short_acted_count=8,
        min_accuracy_lift_over_majority=0.02,
        min_label_balance_score=0.18,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
