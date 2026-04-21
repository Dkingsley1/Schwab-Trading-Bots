import numpy as np

from indicator_bot_common import train_runtime_indicator_bot
from runtime_requested_bot_common import base_runtime_gate, centered01, clip01, quote_quality
from runtime_training_common import (
    feature_ema,
    observation_feature,
    price_change,
    risk_support_label_builder,
)

_MASTER_CANDIDATE_MODES = [
    "shadow_default_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_bond_equities",
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
    "shadow_schwab_futures_equities",
    "shadow_fx_equities",
]
_MASTER_CANDIDATE_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "NVDA",
    "SCHD",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "/ES",
    "/NQ",
    "/CL",
    "/GC",
    "EURUSD",
    "USDJPY",
]


def _signed01(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _cross_asset_support(obs) -> float:
    risk_on = observation_feature(obs, "flow_risk_on_norm")
    risk_off = observation_feature(obs, "flow_risk_off_norm")
    breadth_risk_off = observation_feature(obs, "breadth_risk_off_norm")
    capital_signed = _signed01(observation_feature(obs, "capital_flow_signed_scaled"))
    lead_lag_signal = _signed01(observation_feature(obs, "lead_lag_signal_signed"))
    futures_basis_stability = clip01(
        1.0 - abs(centered01(observation_feature(obs, "futures_basis_divergence_norm", 0.5)))
    )
    crypto_basis_stability = clip01(
        1.0 - abs(centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5)))
    )
    bond_balance = clip01(
        1.0
        - abs(
            observation_feature(obs, "bond_credit_risk_on_norm")
            - observation_feature(obs, "bond_credit_risk_off_norm")
        )
    )
    fx_alignment = max(
        observation_feature(obs, "fx_proxy_agreement_norm"),
        observation_feature(obs, "fx_crypto_alignment_norm"),
    )
    return clip01(
        (0.20 * quote_quality(obs))
        + (0.15 * clip01(1.0 - abs(risk_on - risk_off)))
        + (0.12 * clip01(1.0 - breadth_risk_off))
        + (0.12 * clip01(0.5 + (0.5 * capital_signed)))
        + (0.10 * futures_basis_stability)
        + (0.10 * crypto_basis_stability)
        + (0.09 * bond_balance)
        + (0.07 * fx_alignment)
        + (0.05 * clip01(0.5 + (0.5 * lead_lag_signal)))
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
            observation_feature(obs, "flow_risk_on_norm"),
            observation_feature(obs, "flow_risk_off_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "capital_flow_signed_scaled"),
            observation_feature(obs, "bond_credit_risk_on_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "futures_basis_divergence_norm"),
            observation_feature(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "fx_proxy_agreement_norm"),
            observation_feature(obs, "fx_crypto_alignment_norm"),
            observation_feature(obs, "lead_lag_signal_signed"),
            observation_feature(obs, "lead_lag_confidence_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "flow_risk_on_norm", 4),
            feature_ema(sequence, idx, "breadth_risk_off_norm", 4),
            feature_ema(sequence, idx, "capital_flow_signed_scaled", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(
            obs,
            min_quote_agreement=0.80,
            max_quote_deviation=0.26,
            max_spread_bps=32.0,
            min_queue_depth=0.0,
            max_latency_ms=3400.0,
        )
        and _cross_asset_support(obs) >= 0.24
        and abs(_signed01(observation_feature(obs, "capital_flow_signed_scaled"))) >= 0.06
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.42 * _cross_asset_support(obs))
        + (0.18 * quote_quality(obs))
        + (0.14 * observation_feature(obs, "lead_lag_confidence_norm"))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.12 * clip01(abs(_signed01(observation_feature(obs, "capital_flow_signed_scaled")))))
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v107_cross_asset_master_candidate",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "flow_risk_on_norm",
            "flow_risk_off_norm",
            "breadth_risk_off_norm",
            "capital_flow_signed_scaled",
            "bond_credit_risk_on_norm",
            "bond_credit_risk_off_norm",
            "bond_duration_regime_norm",
            "futures_basis_divergence_norm",
            "crypto_hyperliquid_basis_norm",
            "fx_proxy_agreement_norm",
            "fx_crypto_alignment_norm",
            "lead_lag_signal_signed",
            "lead_lag_confidence_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_credit_flow_norm",
            "behavior_prior",
            "ret_3",
            "ret_6",
            "flow_risk_on_ema_4",
            "breadth_risk_off_ema_4",
            "capital_flow_signed_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=risk_support_label_builder(
            min_return=-0.0008,
            max_drawdown=0.020,
            max_realized_vol=0.030,
            vol_multiplier=3.25,
        ),
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.30,
        lookback_days=60,
        mode_allowlist=_MASTER_CANDIDATE_MODES,
        symbol_allowlist=_MASTER_CANDIDATE_SYMBOLS,
        window=18,
        horizon=6,
        min_samples=480,
        min_sequences=6,
        min_positive_samples=120,
        max_best_val_loss=0.695,
        max_final_val_loss=0.708,
        min_acted_accuracy=0.57,
        min_accuracy_lift_over_majority=0.020,
        allow_fallback_on_insufficient_data=False,
        require_both_sides_precision=False,
    )


if __name__ == "__main__":
    train_brain()
