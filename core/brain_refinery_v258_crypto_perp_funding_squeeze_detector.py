import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, funding_pressure, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v258_crypto_perp_funding_squeeze_detector"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "spread_bps",
    "queue_depth",
    "crypto_hyperliquid_funding_norm",
    "crypto_hyperliquid_basis_norm",
    "crypto_hyperliquid_open_interest_norm",
    "crypto_deribit_mark_iv_norm",
    "crypto_liquidation_pressure_norm",
    "infra_risk_throttle_norm",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _squeeze_signal(obs):
    return clip01((0.62 * funding_pressure(obs)) + (0.20 * crypto_quality(obs)) + (0.18 * observation_feature(obs, "vol_30m")))


def _squeeze_bias(obs):
    return float(
        np.clip(
            (0.28 * centered01(observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5)))
            + (0.24 * centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5)))
            + (0.16 * observation_feature(obs, "mom_5m") * 160.0)
            + (0.14 * observation_feature(obs, "behavior_prior"))
            - (0.18 * observation_feature(obs, "crypto_liquidation_pressure_norm")),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_squeeze_signal,
    bias_builder=_squeeze_bias,
    min_signal=0.24,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
