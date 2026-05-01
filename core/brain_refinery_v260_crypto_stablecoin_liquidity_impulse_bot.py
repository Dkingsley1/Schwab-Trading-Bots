import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, liquidity_impulse, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v260_crypto_stablecoin_liquidity_impulse_bot"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "crypto_defillama_stablecoin_growth_norm",
    "crypto_defillama_dex_volume_growth_norm",
    "crypto_exchange_liquidity_norm",
    "crypto_cross_provider_price_agreement_norm",
    "fx_dollar_funding_stress_norm",
    "flow_risk_on_norm",
    "spread_bps",
    "queue_depth",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _liquidity_signal(obs):
    return clip01((0.72 * liquidity_impulse(obs)) + (0.16 * observation_feature(obs, "crypto_coingecko_momentum_norm")) + (0.12 * observation_feature(obs, "flow_risk_on_norm")))


def _liquidity_bias(obs):
    return float(
        np.clip(
            (0.24 * centered01(observation_feature(obs, "crypto_defillama_stablecoin_growth_norm", 0.5)))
            + (0.20 * centered01(observation_feature(obs, "crypto_defillama_dex_volume_growth_norm", 0.5)))
            + (0.18 * observation_feature(obs, "flow_risk_on_norm"))
            + (0.14 * observation_feature(obs, "behavior_prior"))
            + (0.12 * observation_feature(obs, "pct_from_close") * 85.0)
            - (0.12 * observation_feature(obs, "fx_dollar_funding_stress_norm")),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_liquidity_signal,
    bias_builder=_liquidity_bias,
    min_signal=0.22,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
