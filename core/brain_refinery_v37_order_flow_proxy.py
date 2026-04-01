import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    selective_direction_label_builder,
)

_TOP_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "DOGE-USD", "LTC-USD", "XRP-USD"]


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    tick_dir = np.sign(np.diff(c, prepend=c[0]))
    signed_vol = tick_dir * v
    flow_ema = ema(signed_vol, 10)
    flow_impulse = np.diff(flow_ema, prepend=flow_ema[0])
    rel_vol = v / (np.convolve(v, np.ones(20) / 20.0, mode="same") + 1e-8)
    micro_imbalance = (flow_ema / (np.abs(flow_ema) + 1e-8)) * rel_vol
    shock = np.abs(r) / (rolling_std(r, 20) + 1e-8)

    return np.stack([r, tick_dir, rel_vol, flow_ema, flow_impulse, micro_imbalance, shock], axis=1)


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    bid = observation_feature(obs, "bid_size")
    ask = observation_feature(obs, "ask_size")
    imbalance = (bid - ask) / (abs(bid) + abs(ask) + 1e-8)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            bid,
            ask,
            imbalance,
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "data_quality_bid_ask_imbalance_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "crypto_cross_provider_price_agreement_norm"),
            observation_feature(obs, "crypto_deribit_futures_oi_norm"),
            observation_feature(obs, "crypto_deribit_options_oi_norm"),
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            observation_feature(obs, "crypto_deribit_basis_norm"),
            observation_feature(obs, "crypto_kraken_volume_norm"),
            observation_feature(obs, "crypto_kraken_range_norm"),
            observation_feature(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            observation_feature(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "crypto_coinmetrics_tx_count_norm"),
            observation_feature(obs, "crypto_coinmetrics_active_addr_norm"),
            observation_feature(obs, "crypto_coingecko_volume_norm"),
            observation_feature(obs, "crypto_coingecko_momentum_norm"),
            observation_feature(obs, "crypto_defillama_stablecoin_growth_norm"),
            observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"),
            observation_feature(obs, "crypto_etherscan_gas_norm"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "spread_bps", 4),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _vote_conviction(raw_vote):
    return _clip01(abs(float(raw_vote) - 0.5) * 2.0)


def _centered_signal(obs, name):
    return abs(observation_feature(obs, name, 0.5) - 0.5) * 2.0


def _crypto_regime_signal(obs):
    return _clip01(
        max(
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            observation_feature(obs, "crypto_deribit_futures_oi_norm"),
            observation_feature(obs, "crypto_deribit_options_oi_norm"),
            _centered_signal(obs, "crypto_deribit_basis_norm"),
            observation_feature(obs, "crypto_kraken_volume_norm"),
            observation_feature(obs, "crypto_kraken_range_norm"),
            _centered_signal(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            _centered_signal(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "crypto_coinmetrics_tx_count_norm"),
            observation_feature(obs, "crypto_coinmetrics_active_addr_norm"),
            observation_feature(obs, "crypto_coingecko_volume_norm"),
            _centered_signal(obs, "crypto_coingecko_momentum_norm"),
            _centered_signal(obs, "crypto_defillama_stablecoin_growth_norm"),
            _centered_signal(obs, "crypto_defillama_dex_volume_growth_norm"),
            observation_feature(obs, "crypto_etherscan_gas_norm"),
        )
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    bid = observation_feature(obs, "bid_size")
    ask = observation_feature(obs, "ask_size")
    imbalance = (bid - ask) / (abs(bid) + abs(ask) + 1e-8)
    spread = abs(observation_feature(obs, "spread_bps"))
    queue_depth = observation_feature(obs, "queue_depth")
    latency_ms = observation_feature(obs, "market_data_latency_ms")
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    provider_agreement = observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0)
    regime_signal = _crypto_regime_signal(obs)
    return (
        bid > 0.0
        and ask > 0.0
        and queue_depth >= 1.0
        and spread <= 24.0
        and latency_ms <= 2500.0
        and quote_agreement >= 0.82
        and provider_agreement >= 0.45
        and max(abs(imbalance), regime_signal, _vote_conviction(observation_feature(obs, "futures_specialist_vote"))) >= 0.12
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    bid = observation_feature(obs, "bid_size")
    ask = observation_feature(obs, "ask_size")
    imbalance = abs((bid - ask) / (abs(bid) + abs(ask) + 1e-8))
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps")) / 24.0))
    latency_ok = _clip01(1.0 - (observation_feature(obs, "market_data_latency_ms") / 2500.0))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    provider_ok = _clip01(observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0))
    specialist = max(
        _vote_conviction(observation_feature(obs, "options_specialist_vote")),
        _vote_conviction(observation_feature(obs, "futures_specialist_vote")),
    )
    regime = _crypto_regime_signal(obs)
    return (
        (0.24 * _clip01(imbalance))
        + (0.18 * regime)
        + (0.16 * provider_ok)
        + (0.14 * spread_ok)
        + (0.12 * latency_ok)
        + (0.08 * quote_ok)
        + (0.08 * specialist)
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v37_order_flow_proxy",
        feature_names=["ret", "tick_dir", "rel_vol", "flow_ema", "flow_impulse", "micro_imbalance", "shock"],
        feature_builder=build_features,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v37_order_flow_proxy",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "spread_bps",
            "queue_depth",
            "bid_size",
            "ask_size",
            "book_imbalance",
            "market_data_latency_ms",
            "data_quality_bid_ask_imbalance_norm",
            "data_quality_quote_agreement_norm",
            "lag_expected_fill_delta_bps",
            "lag_slippage_bps",
            "lag_latency_ms",
            "options_specialist_vote",
            "futures_specialist_vote",
            "crypto_cross_provider_price_agreement_norm",
            "crypto_deribit_futures_oi_norm",
            "crypto_deribit_options_oi_norm",
            "crypto_deribit_mark_iv_norm",
            "crypto_deribit_basis_norm",
            "crypto_kraken_volume_norm",
            "crypto_kraken_range_norm",
            "crypto_hyperliquid_funding_norm",
            "crypto_hyperliquid_open_interest_norm",
            "crypto_hyperliquid_basis_norm",
            "crypto_coinmetrics_tx_count_norm",
            "crypto_coinmetrics_active_addr_norm",
            "crypto_coingecko_volume_norm",
            "crypto_coingecko_momentum_norm",
            "crypto_defillama_stablecoin_growth_norm",
            "crypto_defillama_dex_volume_growth_norm",
            "crypto_etherscan_gas_norm",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "spread_bps_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=selective_direction_label_builder(min_abs_return=0.0007),
        mode_allowlist=["shadow_crypto", "shadow_crypto_futures_crypto"],
        symbol_allowlist=_TOP_CRYPTO_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.54,
        lookback_days=30,
        window=18,
        horizon=4,
        min_samples=224,
        min_sequences=5,
        acted_prob_threshold=0.72,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.53,
        min_short_precision=0.53,
        require_both_sides_precision=True,
        min_acted_accuracy=0.54,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )
