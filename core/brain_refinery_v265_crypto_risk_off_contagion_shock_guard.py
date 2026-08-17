import numpy as np

from crypto_runtime_bot_common import CryptoRuntimeSpec, risk_off_pressure, train_crypto_runtime_bot
from runtime_requested_bot_common import clip01
from runtime_training_common import observation_feature

BOT_ID = "brain_refinery_v265_crypto_risk_off_contagion_shock_guard"

FEATURE_FIELDS = [
    "pct_from_close",
    "mom_5m",
    "vol_30m",
    "breadth_risk_off_norm",
    "ctx_VIX_X_pct_from_close",
    "ctx_QQQ_pct_from_close",
    "fx_dollar_funding_stress_norm",
    "crypto_liquidation_pressure_norm",
    "crypto_hyperliquid_funding_norm",
    "market_crypto_spy_corr_norm",
    "infra_risk_throttle_norm",
]
FEATURE_NAMES = [*FEATURE_FIELDS, "ret_1", "ret_3", "pct_from_close_std_4", "open_interest_ema_4", "quote_agreement_ema_4"]


def _contagion_signal(obs):
    return clip01(
        (0.56 * risk_off_pressure(obs))
        + (0.18 * observation_feature(obs, "crypto_liquidation_pressure_norm"))
        + (0.14 * max(-observation_feature(obs, "ctx_QQQ_pct_from_close") * 95.0, 0.0))
        + (0.12 * observation_feature(obs, "vol_30m"))
    )


def _contagion_bias(obs):
    return float(
        np.clip(
            -(
                (0.34 * _contagion_signal(obs))
                + (0.18 * observation_feature(obs, "breadth_risk_off_norm"))
                + (0.16 * observation_feature(obs, "crypto_liquidation_pressure_norm"))
                + (0.12 * observation_feature(obs, "fx_dollar_funding_stress_norm"))
            )
            + (0.10 * observation_feature(obs, "behavior_prior")),
            -1.0,
            1.0,
        )
    )


SPEC = CryptoRuntimeSpec(
    bot_id=BOT_ID,
    feature_names=FEATURE_NAMES,
    feature_fields=FEATURE_FIELDS,
    signal_builder=_contagion_signal,
    bias_builder=_contagion_bias,
    min_signal=0.08,
    min_abs_bias=0.01,
    min_confidence=0.20,
    window=4,
    horizon=1,
    lookback_days=60,
    min_samples=48,
    min_sequences=8,
    min_positive_samples=16,
    min_negative_samples=16,
    batch_size=24,
    defer_on_quality_failure=True,
)


def train_brain():
    return train_crypto_runtime_bot(SPEC)


if __name__ == "__main__":
    train_brain()
