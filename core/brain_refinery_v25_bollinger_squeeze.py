import numpy as np

from indicator_bot_common import bollinger, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    direction_label_builder,
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
)


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    lower, mid, upper = bollinger(c, window=20, k=2.0)
    bandwidth = (upper - lower) / (mid + 1e-8)
    percent_b = (c - lower) / ((upper - lower) + 1e-8)
    squeeze = bandwidth / (rolling_std(bandwidth, 40) + 1e-8)
    breakout_proxy = np.diff(percent_b, prepend=percent_b[0])

    return np.stack([r, bandwidth, percent_b, squeeze, breakout_proxy], axis=1)


def _runtime_feature_vector(sequence, idx):
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "range_pos"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "queue_depth"),
            observation_feature(sequence[idx], "ctx_VIX_X_pct_from_close"),
            observation_feature(sequence[idx], "options_iv_atm_norm"),
            observation_feature(sequence[idx], "options_iv_skew_norm"),
            observation_feature(sequence[idx], "options_iv_term_structure_norm"),
            observation_feature(sequence[idx], "options_vol_expectation_norm"),
            observation_feature(sequence[idx], "breadth_risk_off_norm"),
            observation_feature(sequence[idx], "calendar_event_proximity_norm"),
            observation_feature(sequence[idx], "news_recent_impact"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            feature_std(sequence, idx, "vol_30m", 8),
            feature_ema(sequence, idx, "options_iv_skew_norm", 4),
        ],
        dtype=np.float32,
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v25_bollinger_squeeze",
        feature_names=["ret", "bb_bandwidth", "percent_b", "squeeze_ratio", "breakout_proxy"],
        feature_builder=build_features,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v25_bollinger_squeeze",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "ctx_VIX_X_pct_from_close",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_iv_term_structure_norm",
            "options_vol_expectation_norm",
            "breadth_risk_off_norm",
            "calendar_event_proximity_norm",
            "news_recent_impact",
            "data_quality_quote_agreement_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_8",
            "vol_30m_std_8",
            "options_iv_skew_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=direction_label_builder(min_return=0.0007),
        lookback_days=21,
        window=24,
        horizon=6,
        min_samples=224,
        min_sequences=3,
        fallback_trainer=_train_synthetic,
    )
