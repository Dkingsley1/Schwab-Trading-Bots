import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def test_crypto_runtime_spec_default_modes_are_python312_safe() -> None:
    from crypto_runtime_bot_common import CRYPTO_MODES, CryptoRuntimeSpec, crypto_quality

    spec = CryptoRuntimeSpec(
        bot_id="test_crypto_runtime_bot",
        feature_names=("quality",),
        feature_fields=("queue_depth",),
        signal_builder=crypto_quality,
        bias_builder=crypto_quality,
    )

    assert CRYPTO_MODES == ("shadow_crypto", "shadow_crypto_futures_crypto")
    assert spec.mode_allowlist == CRYPTO_MODES


def test_v265_uses_sparse_crypto_collection_contract() -> None:
    import brain_refinery_v265_crypto_risk_off_contagion_shock_guard as v265

    assert v265.SPEC.window == 4
    assert v265.SPEC.horizon == 1
    assert v265.SPEC.min_signal == 0.08
    assert v265.SPEC.min_abs_bias == 0.01
    assert v265.SPEC.min_samples == 48
    assert v265.SPEC.min_positive_samples == 16
    assert v265.SPEC.min_negative_samples == 16
    assert v265.SPEC.batch_size == 24
    assert v265.SPEC.defer_on_quality_failure is True


def test_crypto_expansion_collection_bots_defer_quality_guard() -> None:
    modules = [
        "brain_refinery_v257_crypto_spot_momentum_regime_bot",
        "brain_refinery_v258_crypto_perp_funding_squeeze_detector",
        "brain_refinery_v259_crypto_etf_tradfi_flow_bridge",
        "brain_refinery_v260_crypto_stablecoin_liquidity_impulse_bot",
        "brain_refinery_v261_crypto_eth_gas_defi_activity_guard",
        "brain_refinery_v262_crypto_solana_high_beta_rotation_bot",
        "brain_refinery_v264_crypto_cross_exchange_divergence_arbitrage_bot",
        "brain_refinery_v266_crypto_weekend_gap_liquidity_bot",
    ]

    for module_name in modules:
        module = __import__(module_name)
        assert module.SPEC.defer_on_quality_failure is True
