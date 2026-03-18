import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    risk_support_label_builder,
    rolling_drawdown,
)


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    c = panel["close"]

    edge = np.abs(ema(r, 10)) / (rolling_std(r, 20) + 1e-8)
    vol = rolling_std(r, 20)
    draw = np.minimum(np.diff(c, prepend=c[0]) / (c + 1e-8), 0.0)
    draw_stress = np.sqrt(rolling_std(draw, 30) + 1e-8)
    vix_shock = np.maximum(np.diff(vix, prepend=vix[0]) / (vix + 1e-8), 0.0)

    risk_budget = edge / (vol + draw_stress + vix_shock + 1e-8)
    low = (risk_budget < np.percentile(risk_budget, 33)).astype(float)
    high = (risk_budget > np.percentile(risk_budget, 66)).astype(float)

    return np.stack([r, edge, vol, draw_stress, vix_shock, risk_budget, low, high], axis=1)


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    pct = observation_feature(obs, "pct_from_close")
    return np.asarray(
        [
            pct,
            abs(pct),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "options_chain_available"),
            observation_feature(obs, "options_iv_atm_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_spread_bps_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "news_shock_rate"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_credit_spread_level_norm"),
            observation_feature(obs, "dividend_quality_score_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "lag_adjusted_return_1m"),
            observation_feature(obs, "lag_slippage_bps"),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "vol_30m", 6),
            rolling_drawdown(sequence, idx, 10),
            feature_ema(sequence, idx, "options_iv_skew_norm", 4),
        ],
        dtype=np.float32,
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v68_risk_budget_layer",
        feature_names=["ret", "edge", "vol", "draw_stress", "vix_shock", "risk_budget", "low_budget", "high_budget"],
        feature_builder=build_features,
        window=48,
        horizon=4,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v68_risk_budget_layer",
        feature_names=[
            "pct_from_close",
            "abs_pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "market_data_latency_ms",
            "queue_depth",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "options_chain_available",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_spread_bps_norm",
            "options_vol_expectation_norm",
            "news_shock_rate",
            "news_recent_impact",
            "bond_duration_regime_norm",
            "bond_credit_risk_off_norm",
            "bond_credit_spread_level_norm",
            "dividend_quality_score_norm",
            "breadth_risk_off_norm",
            "calendar_macro_abs_surprise_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_market_data_latency_norm",
            "lag_adjusted_return_1m",
            "lag_slippage_bps",
            "ret_3",
            "pct_from_close_std_6",
            "vol_30m_std_6",
            "drawdown_10",
            "options_iv_skew_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=risk_support_label_builder(
            min_return=-0.0005,
            max_drawdown=0.012,
            max_realized_vol=0.018,
            vol_multiplier=3.5,
        ),
        lookback_days=21,
        window=24,
        horizon=8,
        min_samples=224,
        min_sequences=3,
        fallback_trainer=_train_synthetic,
    )
