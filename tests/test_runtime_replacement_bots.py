import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v101_guard_heavy_regime_memory as v101
from core import brain_refinery_v102_open_drive_liquidity_pressure as v102
from core import brain_refinery_v103_crypto_throttle_relief_momentum as v103
from core import brain_refinery_v104_futures_event_followthrough as v104
from core import brain_refinery_v105_feed_consensus_execution_guard as v105
from core import brain_refinery_v106_cross_asset_regime_stability_guard as v106


def _obs(symbol="SPY", last_price=100.0, **overrides):
    base_features = {
        "pct_from_close": 0.0025,
        "mom_5m": 0.0014,
        "vol_30m": 0.008,
        "range_pos": 0.74,
        "spread_bps": 8.0,
        "queue_depth": 3.2,
        "market_data_latency_ms": 45.0,
        "lag_latency_ms": 18.0,
        "lag_expected_fill_delta_bps": 0.10,
        "lag_slippage_bps": 0.08,
        "flow_risk_on_norm": 0.66,
        "capital_flow_outflow_norm": 0.14,
        "breadth_risk_off_norm": 0.18,
        "bond_credit_risk_off_norm": 0.20,
        "options_negative_bias_norm": 0.22,
        "infra_risk_throttle_norm": 0.18,
        "market_micro_relative_volume_norm": 0.72,
        "market_micro_order_flow_imbalance_norm": 0.70,
        "market_micro_range_expansion_norm": 0.58,
        "behavior_prior": 0.22,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.04,
        "data_quality_market_data_latency_norm": 0.10,
        "crypto_cross_provider_price_agreement_norm": 0.90,
        "crypto_deribit_mark_iv_norm": 0.44,
        "crypto_hyperliquid_open_interest_norm": 0.56,
        "crypto_hyperliquid_funding_norm": 0.60,
        "crypto_hyperliquid_basis_norm": 0.61,
        "crypto_coingecko_momentum_norm": 0.64,
        "crypto_defillama_dex_volume_growth_norm": 0.54,
        "futures_specialist_vote": 0.72,
        "futures_order_book_imbalance_norm": 0.68,
        "futures_taker_imbalance_norm": 0.66,
        "futures_basis_divergence_norm": 0.56,
        "futures_session_volume_profile_norm": 0.64,
        "calendar_event_proximity_norm": 0.58,
        "calendar_high_impact_24h_norm": 0.60,
        "calendar_macro_event_norm": 0.56,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_replacement_runtime_feature_vector_shapes() -> None:
    sequence = [_obs()]
    assert v101._runtime_feature_vector(sequence, 0).shape == (18,)
    assert v102._runtime_feature_vector(sequence, 0).shape == (22,)
    assert v103._runtime_feature_vector(sequence, 0).shape == (22,)
    assert v104._runtime_feature_vector(sequence, 0).shape == (24,)
    assert v105._runtime_feature_vector(sequence, 0).shape == (17,)
    assert v106._runtime_feature_vector(sequence, 0).shape == (19,)


def test_replacement_signal_labels_can_emit_positive() -> None:
    seq = [
        _obs(last_price=100.0),
        _obs(last_price=100.16),
        _obs(last_price=100.33),
        _obs(last_price=100.51),
        _obs(last_price=100.70),
        _obs(last_price=100.88),
        _obs(last_price=101.03),
    ]
    assert v101._runtime_guard_memory_label(seq, 0, 6) == 1.0
    assert v102._runtime_open_drive_label(seq, 0, 4) == 1.0
    assert v103._runtime_crypto_relief_label(seq, 0, 4) == 1.0
    assert v104._runtime_event_followthrough_label(seq, 0, 4) == 1.0


def test_replacement_train_brain_uses_runtime_path(monkeypatch) -> None:
    modules = [
        (v101, "brain_refinery_v101_guard_heavy_regime_memory"),
        (v102, "brain_refinery_v102_open_drive_liquidity_pressure"),
        (v103, "brain_refinery_v103_crypto_throttle_relief_momentum"),
        (v104, "brain_refinery_v104_futures_event_followthrough"),
        (v105, "brain_refinery_v105_feed_consensus_execution_guard"),
        (v106, "brain_refinery_v106_cross_asset_regime_stability_guard"),
    ]
    for module, run_tag in modules:
        captured = {}

        def _fake_train_runtime_indicator_bot(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(module, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)
        assert module.train_brain() == "ok"
        assert captured["run_tag"] == run_tag
        assert captured["allow_fallback_on_insufficient_data"] is False
        assert callable(captured["runtime_label_builder"])
        assert callable(captured["sample_filter"])
        if module in {v105, v106}:
            assert captured.get("require_both_sides_precision", False) is False
        else:
            assert captured["require_both_sides_precision"] is True
