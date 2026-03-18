import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    selective_direction_label_builder,
)

_TOP_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


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
    bid = observation_feature(sequence[idx], "bid_size")
    ask = observation_feature(sequence[idx], "ask_size")
    imbalance = (bid - ask) / (abs(bid) + abs(ask) + 1e-8)
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "queue_depth"),
            bid,
            ask,
            imbalance,
            observation_feature(sequence[idx], "market_data_latency_ms"),
            observation_feature(sequence[idx], "data_quality_bid_ask_imbalance_norm"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            observation_feature(sequence[idx], "lag_expected_fill_delta_bps"),
            observation_feature(sequence[idx], "lag_slippage_bps"),
            observation_feature(sequence[idx], "lag_latency_ms"),
            observation_feature(sequence[idx], "options_specialist_vote"),
            observation_feature(sequence[idx], "futures_specialist_vote"),
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


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    bid = observation_feature(obs, "bid_size")
    ask = observation_feature(obs, "ask_size")
    spread = abs(observation_feature(obs, "spread_bps"))
    queue_depth = observation_feature(obs, "queue_depth")
    latency_ms = observation_feature(obs, "market_data_latency_ms")
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    return (
        bid > 0.0
        and ask > 0.0
        and queue_depth >= 1.0
        and spread <= 22.0
        and latency_ms <= 2500.0
        and quote_agreement >= 0.85
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    bid = observation_feature(obs, "bid_size")
    ask = observation_feature(obs, "ask_size")
    imbalance = abs((bid - ask) / (abs(bid) + abs(ask) + 1e-8))
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps")) / 22.0))
    latency_ok = _clip01(1.0 - (observation_feature(obs, "market_data_latency_ms") / 2500.0))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    specialist = max(
        _vote_conviction(observation_feature(obs, "options_specialist_vote")),
        _vote_conviction(observation_feature(obs, "futures_specialist_vote")),
    )
    return (0.30 * _clip01(imbalance)) + (0.20 * spread_ok) + (0.20 * latency_ok) + (0.15 * quote_ok) + (0.15 * specialist)


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
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "spread_bps_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=selective_direction_label_builder(min_abs_return=0.0005),
        mode_allowlist=["shadow_crypto", "shadow_crypto_futures_crypto"],
        symbol_allowlist=_TOP_CRYPTO_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.50,
        lookback_days=21,
        window=18,
        horizon=4,
        min_samples=192,
        min_sequences=3,
        acted_prob_threshold=0.72,
        fallback_trainer=_train_synthetic,
    )
