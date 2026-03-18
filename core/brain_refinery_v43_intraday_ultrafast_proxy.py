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

    t = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-8)
    t1 = ema(t, 3)
    t2 = ema(t, 6)
    jerk = np.diff(t1, prepend=t1[0])
    flip = np.abs(np.diff(np.sign(t), prepend=np.sign(t[0])))
    relv = v / (ema(v, 12) + 1e-8)
    burst = np.abs(t) / (rolling_std(t, 12) + 1e-8)

    return np.stack([r, t, t1, t2, jerk, flip, relv, burst], axis=1)


def _runtime_feature_vector(sequence, idx):
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "queue_depth"),
            observation_feature(sequence[idx], "market_data_latency_ms"),
            observation_feature(sequence[idx], "lag_expected_fill_delta_bps"),
            observation_feature(sequence[idx], "lag_latency_ms"),
            observation_feature(sequence[idx], "lag_slippage_bps"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            observation_feature(sequence[idx], "data_quality_market_data_latency_norm"),
            observation_feature(sequence[idx], "futures_specialist_vote"),
            observation_feature(sequence[idx], "behavior_prior"),
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


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    spread = abs(observation_feature(obs, "spread_bps"))
    queue_depth = observation_feature(obs, "queue_depth")
    latency_ms = observation_feature(obs, "market_data_latency_ms")
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    burst = max(
        abs(observation_feature(obs, "mom_5m")),
        feature_std(sequence, idx, "pct_from_close", 4),
    )
    return (
        spread <= 18.0
        and queue_depth >= 1.0
        and latency_ms <= 2000.0
        and quote_agreement >= 0.85
        and burst >= 0.0008
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
    specialist = _vote_conviction(observation_feature(obs, "futures_specialist_vote"))
    behavior = _clip01(abs(observation_feature(obs, "behavior_prior")) * 3.0)
    return (0.28 * burst) + (0.20 * spread_ok) + (0.18 * latency_ok) + (0.16 * quote_ok) + (0.10 * specialist) + (0.08 * behavior)


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v43_intraday_ultrafast_proxy",
        feature_names=["ret", "tick_ret", "t_ema3", "t_ema6", "jerk", "flip", "relv", "burst"],
        feature_builder=build_features,
        window=48,
        horizon=1,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
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
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "pct_from_close_ema_3",
            "spread_bps_ema_3",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=selective_direction_label_builder(min_abs_return=0.00045),
        mode_allowlist=["shadow_crypto_futures_crypto"],
        symbol_allowlist=_TOP_CRYPTO_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.52,
        lookback_days=21,
        window=16,
        horizon=3,
        min_samples=192,
        min_sequences=3,
        acted_prob_threshold=0.72,
        fallback_trainer=_train_synthetic,
    )
