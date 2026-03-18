import numpy as np

from indicator_bot_common import rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_std,
    observation_feature,
    price_change,
    selective_direction_label_builder,
)


def build_features(panel):
    c = panel["close"]
    o = panel["open"]
    r = panel["ret"]
    g = panel["gap"]

    open_drive = (c - o) / (o + 1e-8)
    gap_abs = np.abs(g)
    gap_vol = rolling_std(g, 20)
    continuation = g * np.sign(open_drive)
    first_move = np.diff(open_drive, prepend=open_drive[0])

    return np.stack([r, g, gap_abs, gap_vol, open_drive, continuation, first_move], axis=1)


def _runtime_feature_vector(sequence, idx):
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "range_pos"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "queue_depth"),
            observation_feature(sequence[idx], "market_data_latency_ms"),
            observation_feature(sequence[idx], "day_opening_auction_signal_norm"),
            observation_feature(sequence[idx], "day_session_open_norm"),
            observation_feature(sequence[idx], "day_execution_cost_risk_norm"),
            observation_feature(sequence[idx], "calendar_event_proximity_norm"),
            observation_feature(sequence[idx], "calendar_high_impact_24h_norm"),
            observation_feature(sequence[idx], "news_shock_rate"),
            observation_feature(sequence[idx], "news_recent_impact"),
            observation_feature(sequence[idx], "breadth_advance_decline_norm"),
            observation_feature(sequence[idx], "ctx_VIX_X_pct_from_close"),
            observation_feature(sequence[idx], "ctx_UUP_pct_from_close"),
            observation_feature(sequence[idx], "lag_expected_fill_delta_bps"),
            observation_feature(sequence[idx], "lag_slippage_bps"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_std(sequence, idx, "vol_30m", 4),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    open_signal = max(
        observation_feature(obs, "day_opening_auction_signal_norm"),
        observation_feature(obs, "day_session_open_norm"),
    )
    event_impulse = max(
        observation_feature(obs, "calendar_high_impact_24h_norm"),
        abs(observation_feature(obs, "news_recent_impact")),
        abs(observation_feature(obs, "news_shock_rate")),
    )
    spread = abs(observation_feature(obs, "spread_bps"))
    queue_depth = observation_feature(obs, "queue_depth")
    latency_ms = observation_feature(obs, "market_data_latency_ms")
    exec_risk = observation_feature(obs, "day_execution_cost_risk_norm")
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    move_mag = max(
        abs(observation_feature(obs, "pct_from_close")),
        abs(observation_feature(obs, "mom_5m")),
    )
    return (
        (open_signal >= 0.15 or event_impulse >= 0.35)
        and move_mag >= 0.0006
        and spread <= 35.0
        and queue_depth >= 1.0
        and latency_ms <= 3500.0
        and exec_risk <= 0.80
        and quote_agreement >= 0.80
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    open_signal = max(
        observation_feature(obs, "day_opening_auction_signal_norm"),
        observation_feature(obs, "day_session_open_norm"),
    )
    event_impulse = _clip01(
        max(
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            abs(observation_feature(obs, "news_recent_impact")),
            abs(observation_feature(obs, "news_shock_rate")),
        )
    )
    move_mag = _clip01(
        max(
            abs(observation_feature(obs, "pct_from_close")) * 160.0,
            abs(observation_feature(obs, "mom_5m")) * 120.0,
        )
    )
    news_impulse = _clip01(
        abs(observation_feature(obs, "news_recent_impact")) * 2.0
        + abs(observation_feature(obs, "news_shock_rate")) * 0.5
    )
    exec_ok = _clip01(1.0 - observation_feature(obs, "day_execution_cost_risk_norm"))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps")) / 35.0))
    return (0.20 * open_signal) + (0.20 * event_impulse) + (0.25 * move_mag) + (0.20 * exec_ok) + (0.15 * min(quote_ok, spread_ok))


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v29_gap_open_auction",
        feature_names=["ret", "gap", "gap_abs", "gap_vol", "open_drive", "gap_continuation", "first_move"],
        feature_builder=build_features,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v29_gap_open_auction",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "day_opening_auction_signal_norm",
            "day_session_open_norm",
            "day_execution_cost_risk_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "news_shock_rate",
            "news_recent_impact",
            "breadth_advance_decline_norm",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "lag_expected_fill_delta_bps",
            "lag_slippage_bps",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "vol_30m_std_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=selective_direction_label_builder(min_abs_return=0.0008),
        mode_allowlist=["shadow_crypto"],
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.30,
        lookback_days=21,
        window=18,
        horizon=4,
        min_samples=192,
        min_sequences=3,
        acted_prob_threshold=0.70,
        fallback_trainer=_train_synthetic,
    )
