import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    multi_horizon_direction_label_builder,
    observation_feature,
    price_change,
    rolling_drawdown,
)

_TOP_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    b = panel["bench_ret"]
    c = panel["close"]

    m1 = ema(hold_sample(r, 390), 6)
    m3 = ema(hold_sample(r, 1170), 6)
    beta_adj = r - 0.8 * b
    beta_m = ema(beta_adj, 20)
    vol = rolling_std(r, 80)
    long_drift = ema(np.diff(c, prepend=c[0]) / (c + 1e-8), 80)

    return np.stack([r, b, m1, m3, beta_m, vol, long_drift], axis=1)


def _runtime_feature_vector(sequence, idx):
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "mom_15m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "range_pos"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "queue_depth"),
            observation_feature(sequence[idx], "market_data_latency_ms"),
            observation_feature(sequence[idx], "ctx_VIX_X_pct_from_close"),
            observation_feature(sequence[idx], "ctx_UUP_pct_from_close"),
            observation_feature(sequence[idx], "breadth_advance_decline_norm"),
            observation_feature(sequence[idx], "breadth_risk_off_norm"),
            observation_feature(sequence[idx], "options_vol_expectation_norm"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            observation_feature(sequence[idx], "lag_adjusted_return_1m"),
            observation_feature(sequence[idx], "lag_expected_fill_delta_bps"),
            observation_feature(sequence[idx], "behavior_prior"),
            observation_feature(sequence[idx], "futures_specialist_vote"),
            observation_feature(sequence[idx], "day_session_open_norm"),
            observation_feature(sequence[idx], "day_session_power_hour_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "vol_30m", 8),
            feature_ema(sequence, idx, "behavior_prior", 4),
            rolling_drawdown(sequence, idx, 12),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _vote_conviction(raw_vote):
    return _clip01(abs(float(raw_vote) - 0.5) * 2.0)


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    session_focus = max(
        observation_feature(obs, "day_session_open_norm"),
        observation_feature(obs, "day_session_power_hour_norm"),
    )
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    spread = abs(observation_feature(obs, "spread_bps"))
    queue_depth = observation_feature(obs, "queue_depth")
    conviction = max(
        abs(observation_feature(obs, "behavior_prior")) * 3.0,
        _vote_conviction(observation_feature(obs, "futures_specialist_vote")),
    )
    return (
        (session_focus >= 0.05 or conviction >= 0.35)
        and quote_agreement >= 0.75
        and spread <= 40.0
        and queue_depth >= 1.0
        and conviction >= 0.08
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    session_focus = max(
        observation_feature(obs, "day_session_open_norm"),
        observation_feature(obs, "day_session_power_hour_norm"),
    )
    behavior_conviction = _clip01(abs(observation_feature(obs, "behavior_prior")) * 3.0)
    futures_conviction = _vote_conviction(observation_feature(obs, "futures_specialist_vote"))
    vol_context = _clip01(abs(observation_feature(obs, "options_vol_expectation_norm")))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps")) / 35.0))
    return (
        (0.18 * session_focus)
        + (0.30 * max(behavior_conviction, futures_conviction))
        + (0.18 * vol_context)
        + (0.15 * quote_ok)
        + (0.15 * spread_ok)
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v48_position_1m_3m",
        feature_names=["ret", "bench_ret", "mom_1m", "mom_3m", "beta_m", "vol", "long_drift"],
        feature_builder=build_features,
        window=64,
        horizon=12,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v48_position_1m_3m",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
            "lag_adjusted_return_1m",
            "lag_expected_fill_delta_bps",
            "behavior_prior",
            "futures_specialist_vote",
            "day_session_open_norm",
            "day_session_power_hour_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "vol_30m_std_8",
            "behavior_prior_ema_4",
            "drawdown_12",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=multi_horizon_direction_label_builder(horizons=[3, 6], min_return=0.00025),
        mode_allowlist=["shadow_crypto", "shadow_crypto_futures_crypto"],
        symbol_allowlist=_TOP_CRYPTO_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.30,
        lookback_days=21,
        window=24,
        horizon=6,
        min_samples=224,
        min_sequences=3,
        acted_prob_threshold=0.68,
        fallback_trainer=_train_synthetic,
    )
