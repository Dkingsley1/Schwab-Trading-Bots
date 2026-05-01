import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v257_crypto_spot_momentum_regime_bot"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "mom_15m",
    "vol_30m",
    "range_pos",
    "spread_bps",
    "queue_depth",
    "behavior_prior",
    "crypto_cross_provider_price_agreement_norm",
    "crypto_coingecko_momentum_norm",
    "market_crypto_spy_corr_norm",
    "market_crypto_qqq_corr_norm",
    "flow_risk_on_norm",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _spot_momentum_signal(obs):
    return clip01(
        (0.24 * observation_feature(obs, "crypto_coingecko_momentum_norm"))
        + (0.18 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.16 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.16 * crypto_quality(obs))
        + (0.14 * abs(centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5))))
        + (0.12 * observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"))
    )


def _spot_momentum_bias(obs):
    return float(
        np.clip(
            (0.26 * observation_feature(obs, "behavior_prior"))
            + (0.22 * centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)))
            + (0.16 * observation_feature(obs, "mom_5m") * 150.0)
            + (0.14 * observation_feature(obs, "mom_15m") * 110.0)
            + (0.12 * centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5)))
            + (0.10 * observation_feature(obs, "pct_from_close") * 90.0),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_spot_momentum_signal,
    bias_builder=_spot_momentum_bias,
    min_signal=0.22,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
