import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, liquidity_impulse, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v266_crypto_weekend_gap_liquidity_bot"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "crypto_weekend_session_norm",
    "crypto_weekend_gap_norm",
    "crypto_exchange_liquidity_norm",
    "crypto_defillama_stablecoin_growth_norm",
    "market_crypto_qqq_corr_norm",
    "macro_event_proximity_norm",
    "spread_bps",
    "queue_depth",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _weekend_gap_signal(obs):
    return clip01(
        (0.24 * observation_feature(obs, "crypto_weekend_session_norm"))
        + (0.22 * observation_feature(obs, "crypto_weekend_gap_norm"))
        + (0.18 * (1.0 - observation_feature(obs, "crypto_exchange_liquidity_norm", 1.0)))
        + (0.14 * liquidity_impulse(obs))
        + (0.12 * crypto_quality(obs))
        + (0.10 * observation_feature(obs, "macro_event_proximity_norm"))
    )


def _weekend_gap_bias(obs):
    return float(
        np.clip(
            (0.22 * centered01(observation_feature(obs, "crypto_weekend_gap_norm", 0.5)))
            + (0.18 * observation_feature(obs, "mom_5m") * 145.0)
            + (0.16 * centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5)))
            + (0.14 * centered01(observation_feature(obs, "crypto_defillama_stablecoin_growth_norm", 0.5)))
            + (0.12 * observation_feature(obs, "behavior_prior"))
            - (0.10 * observation_feature(obs, "macro_event_proximity_norm")),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_weekend_gap_signal,
    bias_builder=_weekend_gap_bias,
    min_signal=0.18,
    defer_on_quality_failure=True,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
