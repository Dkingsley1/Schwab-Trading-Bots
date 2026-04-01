import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_TOP_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "DOGE-USD", "LTC-USD", "XRP-USD"]


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    t = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    t1 = ema(t, 3)
    t2 = ema(t, 6)
    jerk = np.diff(t1, prepend=t1[0])
    flip = np.abs(np.diff(np.sign(t), prepend=np.sign(t[0])))
    relv = v / (ema(v, 12) + 1e-8)
    burst = np.abs(t) / (rolling_std(t, 12) + 1e-8)

    return np.stack([r, t, t1, t2, jerk, flip, relv, burst], axis=1)


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
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "crypto_cross_provider_price_agreement_norm"),
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            observation_feature(obs, "crypto_deribit_basis_norm"),
            observation_feature(obs, "crypto_kraken_range_norm"),
            observation_feature(obs, "crypto_kraken_volume_norm"),
            observation_feature(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            observation_feature(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "crypto_coingecko_momentum_norm"),
            observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"),
            observation_feature(obs, "crypto_defillama_stablecoin_growth_norm"),
            observation_feature(obs, "crypto_etherscan_gas_norm"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "pct_from_close", 3),
            feature_ema(sequence, idx, "spread_bps", 3),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _vote_conviction(raw_vote):
    return _clip01(abs(float(raw_vote) - 0.5) * 2.0)


def _centered_signal(obs, name):
    return abs(observation_feature(obs, name, 0.5) - 0.5) * 2.0


def _crypto_burst_signal(obs):
    return _clip01(
        max(
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            _centered_signal(obs, "crypto_deribit_basis_norm"),
            observation_feature(obs, "crypto_kraken_range_norm"),
            observation_feature(obs, "crypto_kraken_volume_norm"),
            _centered_signal(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            _centered_signal(obs, "crypto_hyperliquid_basis_norm"),
            _centered_signal(obs, "crypto_coingecko_momentum_norm"),
            _centered_signal(obs, "crypto_defillama_dex_volume_growth_norm"),
            _centered_signal(obs, "crypto_defillama_stablecoin_growth_norm"),
            observation_feature(obs, "crypto_etherscan_gas_norm"),
        )
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    spread = abs(observation_feature(obs, "spread_bps"))
    queue_depth = observation_feature(obs, "queue_depth")
    latency_ms = observation_feature(obs, "market_data_latency_ms")
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    provider_agreement = observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0)
    burst = max(
        abs(observation_feature(obs, "mom_5m")),
        feature_std(sequence, idx, "pct_from_close", 4),
        _crypto_burst_signal(obs) * 0.002,
    )
    return (
        spread <= 14.0
        and queue_depth >= 1.5
        and latency_ms <= 1400.0
        and quote_agreement >= 0.90
        and max(provider_agreement, _crypto_burst_signal(obs)) >= 0.50
        and abs(_directional_burst_bias(obs)) >= 0.16
        and burst >= 0.0010
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    burst = _clip01(
        max(
            abs(observation_feature(obs, "mom_5m")) * 180.0,
            feature_std(sequence, idx, "pct_from_close", 4) * 180.0,
        )
    )
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps")) / 18.0))
    latency_ok = _clip01(1.0 - (observation_feature(obs, "market_data_latency_ms") / 2000.0))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    provider_ok = _clip01(observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0))
    specialist = _vote_conviction(observation_feature(obs, "futures_specialist_vote"))
    behavior = _clip01(abs(observation_feature(obs, "behavior_prior")) * 3.0)
    regime = _crypto_burst_signal(obs)
    return (
        (0.22 * burst)
        + (0.20 * regime)
        + (0.16 * max(provider_ok, regime * 0.85))
        + (0.16 * spread_ok)
        + (0.12 * latency_ok)
        + (0.08 * quote_ok)
        + (0.04 * specialist)
        + (0.02 * behavior)
    )


def _directional_burst_bias(obs):
    centered_vote = (observation_feature(obs, "futures_specialist_vote", 0.5) - 0.5) * 2.0
    centered_momentum = (observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5) - 0.5) * 2.0
    centered_basis = (observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5) - 0.5) * 2.0
    centered_funding = (observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5) - 0.5) * 2.0
    centered_deribit_basis = (observation_feature(obs, "crypto_deribit_basis_norm", 0.5) - 0.5) * 2.0
    return float(
        (120.0 * observation_feature(obs, "mom_5m"))
        + (0.40 * observation_feature(obs, "behavior_prior"))
        + (0.18 * centered_vote)
        + (0.14 * centered_momentum)
        + (0.14 * centered_basis)
        + (0.08 * centered_funding)
        + (0.06 * centered_deribit_basis)
    )


def _runtime_burst_label(sequence, idx, horizon):
    obs = sequence[idx]
    burst = _crypto_burst_signal(obs)
    bias = _directional_burst_bias(obs)
    if burst < 0.36 or abs(bias) < 0.16:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00070, 0.00125 - (0.00045 * burst))
    if abs(fwd_ret) < move_threshold and realized < 0.026:
        return None

    success_score = (
        signed_ret
        + (0.00105 * burst)
        + (0.00030 * _clip01(observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0)))
        - (0.18 * realized)
    )
    failure_score = (-signed_ret) + (0.17 * realized)
    if success_score >= 0.00060:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00075:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v43_intraday_ultrafast_proxy",
        feature_names=["ret", "tick_ret", "t_ema3", "t_ema6", "jerk", "flip", "relv", "burst"],
        feature_builder=build_features,
        window=48,
        horizon=1,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v43_intraday_ultrafast_proxy",
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
            "futures_specialist_vote",
            "behavior_prior",
            "crypto_cross_provider_price_agreement_norm",
            "crypto_deribit_mark_iv_norm",
            "crypto_deribit_basis_norm",
            "crypto_kraken_range_norm",
            "crypto_kraken_volume_norm",
            "crypto_hyperliquid_funding_norm",
            "crypto_hyperliquid_open_interest_norm",
            "crypto_hyperliquid_basis_norm",
            "crypto_coingecko_momentum_norm",
            "crypto_defillama_dex_volume_growth_norm",
            "crypto_defillama_stablecoin_growth_norm",
            "crypto_etherscan_gas_norm",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "pct_from_close_ema_3",
            "spread_bps_ema_3",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_burst_label,
        mode_allowlist=["shadow_crypto", "shadow_crypto_futures_crypto"],
        symbol_allowlist=_TOP_CRYPTO_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.56,
        sample_stride=4,
        lookback_days=45,
        window=16,
        horizon=3,
        min_samples=224,
        min_sequences=4,
        min_positive_samples=40,
        min_negative_samples=40,
        acted_prob_threshold=0.62,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=6,
        min_short_acted_count=6,
        min_accuracy_lift_over_majority=0.015,
        min_precision_balance_score=0.25,
    )


if __name__ == "__main__":
    train_brain()
