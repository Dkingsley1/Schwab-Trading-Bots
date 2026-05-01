import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, crypto_quality, train_crypto_runtime_bot
from runtime_requested_bot_common import centered01, clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v259_crypto_etf_tradfi_flow_bridge"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "ctx_QQQ_pct_from_close",
    "ctx_SPY_pct_from_close",
    "ctx_UUP_pct_from_close",
    "ctx_GLD_pct_from_close",
    "market_crypto_spy_corr_norm",
    "market_crypto_qqq_corr_norm",
    "market_crypto_gold_corr_norm",
    "fx_crypto_alignment_norm",
    "flow_risk_on_norm",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _tradfi_bridge_signal(obs):
    corr_conf = max(observation_feature(obs, "market_crypto_corr_confidence_norm"), observation_feature(obs, "fx_corr_confidence_norm"))
    return clip01(
        (0.24 * corr_conf)
        + (0.20 * abs(centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5))))
        + (0.18 * observation_feature(obs, "fx_crypto_alignment_norm"))
        + (0.16 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.12 * crypto_quality(obs))
        + (0.10 * abs(observation_feature(obs, "ctx_QQQ_pct_from_close") * 80.0))
    )


def _tradfi_bridge_bias(obs):
    return float(
        np.clip(
            (0.22 * observation_feature(obs, "behavior_prior"))
            + (0.18 * centered01(observation_feature(obs, "market_crypto_qqq_corr_norm", 0.5)))
            + (0.16 * centered01(observation_feature(obs, "market_crypto_spy_corr_norm", 0.5)))
            + (0.14 * centered01(observation_feature(obs, "fx_crypto_alignment_norm", 0.5)))
            + (0.12 * observation_feature(obs, "ctx_QQQ_pct_from_close") * 90.0)
            - (0.10 * observation_feature(obs, "ctx_UUP_pct_from_close") * 80.0)
            + (0.08 * observation_feature(obs, "pct_from_close") * 75.0),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_tradfi_bridge_signal,
    bias_builder=_tradfi_bridge_bias,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
