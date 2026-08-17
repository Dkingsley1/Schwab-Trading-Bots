import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v264_crypto_cross_exchange_divergence_arbitrage_bot"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "range_pos",
    "spread_bps",
    "queue_depth",
    "crypto_cross_provider_price_agreement_norm",
    "crypto_exchange_volume_dispersion_norm",
    "crypto_exchange_price_divergence_norm",
    "market_data_latency_ms",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _divergence_signal(obs):
    return clip01(
        (0.28 * observation_feature(obs, "crypto_exchange_price_divergence_norm"))
        + (0.22 * observation_feature(obs, "crypto_exchange_volume_dispersion_norm"))
        + (0.18 * (1.0 - observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0)))
        + (0.14 * observation_feature(obs, "range_pos"))
        + (0.10 * observation_feature(obs, "vol_30m"))
        + (0.08 * crypto_quality(obs))
    )


def _divergence_bias(obs):
    return float(
        np.clip(
            (0.24 * centered01(observation_feature(obs, "range_pos", 0.5)))
            + (0.20 * observation_feature(obs, "mom_5m") * 150.0)
            + (0.18 * centered01(observation_feature(obs, "crypto_exchange_volume_dispersion_norm", 0.5)))
            + (0.16 * observation_feature(obs, "behavior_prior"))
            - (0.12 * clip01(observation_feature(obs, "market_data_latency_ms") / 3200.0))
            - (0.10 * observation_feature(obs, "infra_risk_throttle_norm")),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_divergence_signal,
    bias_builder=_divergence_bias,
    min_signal=0.18,
    defer_on_quality_failure=True,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
