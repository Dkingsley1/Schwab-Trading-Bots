import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, risk_off_pressure, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v262_crypto_solana_high_beta_rotation_bot"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "mom_15m",
    "vol_30m",
    "crypto_solana_relative_strength_norm",
    "crypto_eth_btc_relative_strength_norm",
    "crypto_coingecko_momentum_norm",
    "market_crypto_qqq_corr_norm",
    "flow_risk_on_norm",
    "breadth_risk_off_norm",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _sol_rotation_signal(obs):
    return clip01(
        (0.30 * observation_feature(obs, "crypto_solana_relative_strength_norm"))
        + (0.20 * observation_feature(obs, "crypto_coingecko_momentum_norm"))
        + (0.16 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.14 * abs(centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5))))
        + (0.12 * crypto_quality(obs))
        - (0.10 * risk_off_pressure(obs))
    )


def _sol_rotation_bias(obs):
    return float(
        np.clip(
            (0.28 * centered01(observation_feature(obs, "crypto_solana_relative_strength_norm", 0.5)))
            + (0.18 * centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)))
            + (0.16 * observation_feature(obs, "mom_5m") * 150.0)
            + (0.14 * observation_feature(obs, "flow_risk_on_norm"))
            + (0.10 * observation_feature(obs, "pct_from_close") * 90.0)
            - (0.14 * risk_off_pressure(obs)),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_sol_rotation_signal,
    bias_builder=_sol_rotation_bias,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
