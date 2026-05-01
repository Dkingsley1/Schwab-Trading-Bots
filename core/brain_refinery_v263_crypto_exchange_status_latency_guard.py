import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, train_crypto_runtime_bot
from runtime_requested_bot_common import clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v263_crypto_exchange_status_latency_guard"

FEATURE_FIELDS = [
    "pct_from_close",
    "vol_30m",
    "spread_bps",
    "queue_depth",
    "market_data_latency_ms",
    "lag_latency_ms",
    "lag_slippage_bps",
    "crypto_cross_provider_price_agreement_norm",
    "crypto_exchange_status_degraded_norm",
    "infra_risk_throttle_norm",
    "data_quality_quote_agreement_norm",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _exchange_health_signal(obs):
    latency_risk = clip01(observation_feature(obs, "market_data_latency_ms") / 3200.0)
    spread_risk = clip01(abs(observation_feature(obs, "spread_bps")) / 40.0)
    return clip01(
        (0.28 * observation_feature(obs, "crypto_exchange_status_degraded_norm"))
        + (0.20 * latency_risk)
        + (0.18 * spread_risk)
        + (0.16 * observation_feature(obs, "lag_slippage_bps") / 24.0)
        + (0.10 * observation_feature(obs, "infra_risk_throttle_norm"))
        + (0.08 * (1.0 - crypto_quality(obs)))
    )


def _exchange_health_bias(obs):
    return float(
        np.clip(
            -(
                (0.32 * _exchange_health_signal(obs))
                + (0.20 * observation_feature(obs, "infra_risk_throttle_norm"))
                + (0.14 * clip01(observation_feature(obs, "market_data_latency_ms") / 3200.0))
            )
            + (0.12 * observation_feature(obs, "behavior_prior")),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_exchange_health_signal,
    bias_builder=_exchange_health_bias,
    min_signal=0.18,
    min_abs_bias=0.05,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
