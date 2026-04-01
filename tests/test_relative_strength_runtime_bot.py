import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v26_relative_strength_cross_section as v26


def _obs(**overrides):
    base_features = {
        "pct_from_close": 0.0025,
        "mom_5m": 0.0012,
        "mom_15m": 0.0018,
        "vol_30m": 0.008,
        "range_pos": 0.72,
        "spread_bps": 8.0,
        "queue_depth": 3.0,
        "ctx_VIX_X_pct_from_close": 0.002,
        "ctx_UUP_pct_from_close": -0.001,
        "breadth_advance_decline_norm": 0.28,
        "breadth_risk_off_norm": 0.18,
        "market_micro_relative_volume_norm": 0.62,
        "market_micro_trend_persistence_norm": 0.66,
        "market_micro_order_flow_imbalance_norm": 0.24,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.06,
        "behavior_prior": 0.26,
        "day_regime_trend_norm": 0.74,
        "day_regime_alignment_norm": 0.69,
        "swing_sector_relative_strength_norm": 0.72,
        "swing_weekly_trend_confirm_norm": 0.70,
        "swing_regime_trend_norm": 0.68,
        "swing_regime_alignment_norm": 0.66,
        "capital_flow_signed_scaled": 0.20,
        "flow_direction_signed": 0.24,
        "flow_risk_on_norm": 0.62,
        "lead_lag_alignment_norm": 0.60,
        "options_vol_expectation_norm": 0.42,
        "options_negative_bias_norm": 0.18,
    }
    price = float(overrides.pop("last_price", 100.0))
    base_features.update(overrides)
    return {"price": price, "features": base_features}


def test_relative_strength_feature_vector_shape() -> None:
    sequence = [_obs()]
    features = v26._runtime_feature_vector(sequence, 0)
    assert features.shape == (35,)


def test_relative_strength_label_returns_positive_for_supported_uptrend() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.18),
        _obs(last_price=100.34),
        _obs(last_price=100.52),
        _obs(last_price=100.66),
        _obs(last_price=100.78),
        _obs(last_price=100.94),
    ]
    label = v26._runtime_relative_strength_label(sequence, 0, 6)
    assert label == 1.0


def test_relative_strength_label_returns_negative_for_supported_downtrend() -> None:
    sequence = [
        _obs(
            last_price=100.0,
            pct_from_close=-0.0022,
            mom_5m=-0.0012,
            mom_15m=-0.0017,
            range_pos=0.28,
            breadth_advance_decline_norm=-0.24,
            behavior_prior=-0.22,
            swing_sector_relative_strength_norm=0.30,
            capital_flow_signed_scaled=-0.18,
            flow_direction_signed=-0.20,
            flow_risk_on_norm=0.32,
        ),
        _obs(last_price=99.86),
        _obs(last_price=99.72),
        _obs(last_price=99.58),
        _obs(last_price=99.45),
        _obs(last_price=99.34),
        _obs(last_price=99.20),
    ]
    label = v26._runtime_relative_strength_label(sequence, 0, 6)
    assert label == 0.0


def test_train_brain_uses_runtime_trainer_without_synthetic_fallback(monkeypatch) -> None:
    captured = {}

    def _fake_train_runtime_indicator_bot(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(v26, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

    result = v26.train_brain()

    assert result == "ok"
    assert captured["run_tag"] == "brain_refinery_v26_relative_strength_cross_section"
    assert captured["allow_fallback_on_insufficient_data"] is False
    assert captured["min_accuracy_lift_over_majority"] == 0.01
    assert "shadow_equities" in captured["mode_allowlist"]
    assert "SPY" in captured["symbol_allowlist"]
    assert captured["min_confidence"] == 0.36
    assert captured["min_samples"] == 320
    assert "swing_sector_relative_strength_norm" in captured["feature_names"]
    assert captured["runtime_label_builder"] is v26._runtime_relative_strength_label
