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

_EVENT_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_conservative_equities",
]
_EVENT_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "TLT",
    "GLD",
    "SMH",
    "XLF",
    "XLK",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
]


def event_proximity(n, cycle=390, anchors=(60, 180, 300)):
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t = i % cycle
        d = min(abs(t - a) for a in anchors)
        out[i] = np.exp(-(d**2) / (2.0 * (25.0**2)))
    return out


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]
    n = len(r)

    eprox = event_proximity(n)
    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    risk_boost = eprox * np.maximum(vix_chg, 0.0)
    ret_noise = rolling_std(r, 20)
    ret_ema = ema(r, 10)
    jump = np.abs(r) / (ret_noise + 1e-8)

    return np.stack([r, ret_ema, eprox, vix_chg, risk_boost, ret_noise, jump], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _event_signal(obs):
    event_signal = max(
        observation_feature(obs, "calendar_event_proximity_norm"),
        observation_feature(obs, "calendar_high_impact_24h_norm"),
        observation_feature(obs, "calendar_macro_event_norm"),
        observation_feature(obs, "calendar_fomc_event_norm"),
        observation_feature(obs, "calendar_cpi_event_norm"),
        observation_feature(obs, "calendar_labor_event_norm"),
    )
    news_signal = max(
        observation_feature(obs, "news_shock_rate"),
        observation_feature(obs, "news_recent_impact"),
        observation_feature(obs, "news_items_30m"),
        observation_feature(obs, "news_items_2h") * 0.9,
        observation_feature(obs, "news_topic_earnings_norm"),
        observation_feature(obs, "news_topic_guidance_norm"),
        observation_feature(obs, "news_topic_mna_norm"),
        observation_feature(obs, "news_topic_regulatory_norm"),
    )
    market_signal = max(
        _clip01(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03),
        observation_feature(obs, "options_vol_expectation_norm"),
        observation_feature(obs, "options_negative_bias_norm"),
        observation_feature(obs, "options_unusual_flow_norm"),
    )
    return _clip01(max(event_signal, news_signal, market_signal))


def _event_bias(obs):
    polarity = observation_feature(obs, "news_positive_share") - observation_feature(obs, "news_negative_share")
    order_flow = (observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5) - 0.5) * 2.0
    return float(
        np.clip(
            (0.30 * observation_feature(obs, "news_sentiment"))
            + (0.18 * polarity)
            + (0.16 * order_flow)
            + (0.14 * observation_feature(obs, "behavior_prior"))
            + (0.10 * observation_feature(obs, "pct_from_close") * 80.0)
            + (0.08 * observation_feature(obs, "mom_5m") * 90.0)
            - (0.04 * observation_feature(obs, "breadth_risk_off_norm")),
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
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "news_available"),
            observation_feature(obs, "news_items_30m"),
            observation_feature(obs, "news_items_2h"),
            observation_feature(obs, "news_sentiment"),
            observation_feature(obs, "news_negative_share"),
            observation_feature(obs, "news_positive_share"),
            observation_feature(obs, "news_shock_rate"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "news_topic_earnings_norm"),
            observation_feature(obs, "news_topic_guidance_norm"),
            observation_feature(obs, "news_topic_mna_norm"),
            observation_feature(obs, "news_topic_regulatory_norm"),
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_macro_revision_norm"),
            observation_feature(obs, "calendar_fomc_event_norm"),
            observation_feature(obs, "calendar_cpi_event_norm"),
            observation_feature(obs, "calendar_labor_event_norm"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "news_sentiment", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.30
        and _event_signal(obs) >= 0.24
        and abs(_event_bias(obs)) >= 0.08
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.34 * _event_signal(obs))
        + (0.22 * _clip01(abs(_event_bias(obs)) / 0.9))
        + (0.16 * _quote_quality(obs))
        + (0.14 * observation_feature(obs, "options_vol_expectation_norm"))
        + (0.14 * observation_feature(obs, "news_items_30m"))
    )


def _runtime_event_label(sequence, idx, horizon):
    obs = sequence[idx]
    signal = _event_signal(obs)
    bias = _event_bias(obs)
    if signal < 0.24 or abs(bias) < 0.08:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00055, 0.00125 - (0.00050 * signal))
    if abs(fwd_ret) < move_threshold and realized < 0.020 and drawdown < 0.012:
        return None

    success_score = (
        signed_ret
        + (0.0010 * signal)
        + (0.00020 * _quote_quality(obs))
        - (0.00020 * observation_feature(obs, "breadth_risk_off_norm"))
        - (0.20 * realized)
        - (0.24 * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + (0.00085 * signal)
        + (0.18 * realized)
        + (0.22 * drawdown)
    )
    if success_score >= 0.00050:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00075:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v39_event_risk_proximity",
        feature_names=["ret", "ret_ema10", "event_proximity", "vix_chg", "risk_boost", "ret_noise", "jump"],
        feature_builder=build_features,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v39_event_risk_proximity",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "news_available",
            "news_items_30m",
            "news_items_2h",
            "news_sentiment",
            "news_negative_share",
            "news_positive_share",
            "news_shock_rate",
            "news_recent_impact",
            "news_topic_earnings_norm",
            "news_topic_guidance_norm",
            "news_topic_mna_norm",
            "news_topic_regulatory_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "calendar_macro_event_norm",
            "calendar_macro_abs_surprise_norm",
            "calendar_macro_revision_norm",
            "calendar_fomc_event_norm",
            "calendar_cpi_event_norm",
            "calendar_labor_event_norm",
            "ctx_VIX_X_pct_from_close",
            "breadth_risk_off_norm",
            "options_vol_expectation_norm",
            "options_negative_bias_norm",
            "options_unusual_flow_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "market_micro_order_flow_imbalance_norm",
            "behavior_prior",
            "ret_3",
            "pct_from_close_std_6",
            "news_sentiment_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_event_label,
        mode_allowlist=_EVENT_RUNTIME_MODES,
        symbol_allowlist=_EVENT_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.36,
        lookback_days=30,
        window=24,
        horizon=6,
        min_samples=256,
        min_sequences=6,
        acted_prob_threshold=0.68,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
