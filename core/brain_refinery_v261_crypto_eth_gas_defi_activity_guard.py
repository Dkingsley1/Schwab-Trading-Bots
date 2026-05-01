import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v261_crypto_eth_gas_defi_activity_guard"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "crypto_etherscan_gas_norm",
    "crypto_defillama_dex_volume_growth_norm",
    "crypto_eth_btc_relative_strength_norm",
    "crypto_cross_provider_price_agreement_norm",
    "data_quality_quote_agreement_norm",
    "spread_bps",
    "queue_depth",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _eth_activity_signal(obs):
    return clip01(
        (0.26 * observation_feature(obs, "crypto_etherscan_gas_norm"))
        + (0.24 * observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"))
        + (0.20 * observation_feature(obs, "crypto_eth_btc_relative_strength_norm"))
        + (0.16 * crypto_quality(obs))
        + (0.14 * observation_feature(obs, "crypto_coingecko_momentum_norm"))
    )


def _eth_activity_bias(obs):
    return float(
        np.clip(
            (0.26 * centered01(observation_feature(obs, "crypto_eth_btc_relative_strength_norm", 0.5)))
            + (0.18 * centered01(observation_feature(obs, "crypto_defillama_dex_volume_growth_norm", 0.5)))
            + (0.16 * observation_feature(obs, "behavior_prior"))
            + (0.14 * observation_feature(obs, "mom_5m") * 135.0)
            + (0.12 * observation_feature(obs, "pct_from_close") * 90.0)
            - (0.14 * max(observation_feature(obs, "crypto_etherscan_gas_norm") - 0.72, 0.0)),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_eth_activity_signal,
    bias_builder=_eth_activity_bias,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
