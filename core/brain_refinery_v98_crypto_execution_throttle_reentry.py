import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_CRYPTO_REENTRY_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "DOGE-USD"]
_CRYPTO_REENTRY_MODES = ["shadow_crypto", "shadow_crypto_futures_crypto"]


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    ret_fast = ema(r, 4)
    ret_slow = ema(r, 10)
    volume_burst = v / (ema(v, 16) + 1e-8)
    vol_fast = rolling_std(r, 10)
    jerk = np.diff(ret_fast, prepend=ret_fast[0])
    impulse = np.abs(np.diff(c, prepend=c[0]) / (c + 1e-8))

    return np.stack([r, ret_fast, ret_slow, volume_burst, vol_fast, jerk, impulse], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.20 * observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0))
        + (0.20 * (1.0 - observation_feature(obs, "data_quality_market_data_latency_norm", 0.0)))
    )


def _crypto_setup_signal(obs):
    return _clip01(
        max(
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            abs(_centered01(observation_feature(obs, "crypto_deribit_basis_norm", 0.5))),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            abs(_centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5))),
            abs(_centered01(observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5))),
            abs(_centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5))),
            observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"),
            observation_feature(obs, "crypto_etherscan_gas_norm"),
        )
    )


def _throttle_relief_signal(obs):
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps", 0.0)) / 24.0))
    latency_ok = _clip01(1.0 - (observation_feature(obs, "market_data_latency_ms", 0.0) / 2400.0))
    queue_ok = _clip01(observation_feature(obs, "queue_depth", 0.0) / 4.0)
    return _clip01(
        (0.34 * _clip01(1.0 - observation_feature(obs, "infra_risk_throttle_norm")))
        + (0.18 * _quote_quality(obs))
        + (0.14 * spread_ok)
        + (0.12 * latency_ok)
        + (0.12 * queue_ok)
        + (0.10 * observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0))
    )


def _reentry_bias(obs):
    return float(
        np.clip(
            (0.30 * observation_feature(obs, "behavior_prior"))
            + (0.18 * _centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)))
            + (0.14 * _centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5)))
            + (0.10 * _centered01(observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5)))
            + (0.10 * _centered01(observation_feature(obs, "futures_specialist_vote", 0.5)))
            + (0.10 * observation_feature(obs, "mom_5m") * 160.0)
            + (0.08 * observation_feature(obs, "pct_from_close") * 120.0)
            - (0.10 * observation_feature(obs, "infra_risk_throttle_norm")),
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
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "infra_risk_throttle_norm"),
            observation_feature(obs, "crypto_cross_provider_price_agreement_norm"),
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            observation_feature(obs, "crypto_deribit_basis_norm"),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            observation_feature(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "crypto_coingecko_momentum_norm"),
            observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"),
            observation_feature(obs, "crypto_defillama_stablecoin_growth_norm"),
            observation_feature(obs, "crypto_etherscan_gas_norm"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "infra_risk_throttle_norm", 4),
            feature_ema(sequence, idx, "crypto_hyperliquid_open_interest_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        abs(observation_feature(obs, "spread_bps", 0.0)) <= 28.0
        and observation_feature(obs, "queue_depth", 0.0) >= 0.8
        and observation_feature(obs, "market_data_latency_ms", 0.0) <= 2800.0
        and observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.80
        and observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0) >= 0.72
        and _crypto_setup_signal(obs) >= 0.20
        and _throttle_relief_signal(obs) >= 0.18
        and abs(_reentry_bias(obs)) >= 0.07
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.26 * _crypto_setup_signal(obs))
        + (0.24 * _throttle_relief_signal(obs))
        + (0.20 * _clip01(abs(_reentry_bias(obs)) / 0.9))
        + (0.16 * _quote_quality(obs))
        + (0.14 * observation_feature(obs, "crypto_hyperliquid_open_interest_norm"))
    )


def _runtime_crypto_reentry_label(sequence, idx, horizon):
    obs = sequence[idx]
    setup = _crypto_setup_signal(obs)
    relief = _throttle_relief_signal(obs)
    bias = _reentry_bias(obs)
    if setup < 0.20 or relief < 0.18 or abs(bias) < 0.07:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00035, 0.00095 - (0.00035 * max(setup, relief)))
    if abs(fwd_ret) < move_threshold and realized < 0.020 and drawdown < 0.012:
        return None

    success_score = (
        signed_ret
        + (0.00090 * setup)
        + (0.00055 * relief)
        - (0.00070 * observation_feature(obs, "infra_risk_throttle_norm"))
        - (0.16 * realized)
        - (0.22 * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + (0.00070 * observation_feature(obs, "infra_risk_throttle_norm"))
        + (0.14 * realized)
        + (0.18 * drawdown)
    )
    if success_score >= 0.00030:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00055:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v98_crypto_execution_throttle_reentry",
        feature_names=["ret", "ret_fast", "ret_slow", "volume_burst", "vol_fast", "jerk", "impulse"],
        feature_builder=build_features,
        window=42,
        horizon=3,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v98_crypto_execution_throttle_reentry",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "lag_expected_fill_delta_bps",
            "lag_latency_ms",
            "lag_slippage_bps",
            "data_quality_quote_agreement_norm",
            "data_quality_market_data_latency_norm",
            "behavior_prior",
            "futures_specialist_vote",
            "infra_risk_throttle_norm",
            "crypto_cross_provider_price_agreement_norm",
            "crypto_deribit_mark_iv_norm",
            "crypto_deribit_basis_norm",
            "crypto_hyperliquid_open_interest_norm",
            "crypto_hyperliquid_funding_norm",
            "crypto_hyperliquid_basis_norm",
            "crypto_coingecko_momentum_norm",
            "crypto_defillama_dex_volume_growth_norm",
            "crypto_defillama_stablecoin_growth_norm",
            "crypto_etherscan_gas_norm",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "infra_risk_throttle_ema_4",
            "crypto_hyperliquid_open_interest_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_crypto_reentry_label,
        mode_allowlist=_CRYPTO_REENTRY_MODES,
        symbol_allowlist=_CRYPTO_REENTRY_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.36,
        lookback_days=45,
        window=16,
        horizon=3,
        min_samples=160,
        min_sequences=4,
        min_positive_samples=20,
        min_negative_samples=20,
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
