import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v35_dmi_state_machine as v35
from core import brain_refinery_v38_multi_timeframe_confirmation as v38
from core import brain_refinery_v41_long_interval_trend as v41
from core import brain_refinery_v52_sector_rotation_rs as v52


def _obs(symbol="SPY", last_price=100.0, **overrides):
    base_features = {
        "pct_from_close": 0.0022,
        "mom_5m": 0.0012,
        "mom_15m": 0.0018,
        "vol_30m": 0.008,
        "range_pos": 0.70,
        "spread_bps": 9.0,
        "queue_depth": 3.0,
        "market_data_latency_ms": 35.0,
        "market_micro_relative_volume_norm": 0.64,
        "market_micro_trend_persistence_norm": 0.68,
        "market_micro_order_flow_imbalance_norm": 0.72,
        "day_regime_trend_norm": 0.74,
        "day_regime_alignment_norm": 0.70,
        "lead_lag_alignment_norm": 0.64,
        "breadth_advance_decline_norm": 0.28,
        "breadth_sector_dispersion_norm": 0.20,
        "breadth_risk_off_norm": 0.18,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.06,
        "behavior_prior": 0.24,
        "futures_specialist_vote": 0.72,
        "day_session_open_norm": 0.36,
        "day_session_midday_norm": 0.20,
        "day_session_power_hour_norm": 0.34,
        "swing_sector_relative_strength_norm": 0.74,
        "swing_weekly_trend_confirm_norm": 0.70,
        "swing_regime_trend_norm": 0.69,
        "swing_regime_alignment_norm": 0.66,
        "options_vol_expectation_norm": 0.36,
        "options_negative_bias_norm": 0.18,
        "capital_flow_signed_scaled": 0.20,
        "flow_direction_signed": 0.22,
        "flow_risk_on_norm": 0.64,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_runtime_feature_vector_shapes() -> None:
    sequence = [_obs()]
    assert v35._runtime_feature_vector(sequence, 0).shape == (27,)
    assert v38._runtime_feature_vector(sequence, 0).shape == (31,)
    assert v41._runtime_feature_vector(sequence, 0).shape == (29,)
    assert v52._runtime_feature_vector(sequence, 0).shape == (33,)


def test_v35_runtime_label_returns_positive_for_supported_trend() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.15),
        _obs(last_price=100.29),
        _obs(last_price=100.46),
        _obs(last_price=100.61),
        _obs(last_price=100.76),
        _obs(last_price=100.94),
    ]
    assert v35._runtime_trend_label(sequence, 0, 6) == 1.0


def test_v52_runtime_label_returns_positive_for_supported_sector_rotation() -> None:
    sequence = [
        _obs(symbol="XLK", last_price=100.0),
        _obs(symbol="XLK", last_price=100.18),
        _obs(symbol="XLK", last_price=100.33),
        _obs(symbol="XLK", last_price=100.50),
        _obs(symbol="XLK", last_price=100.66),
        _obs(symbol="XLK", last_price=100.80),
        _obs(symbol="XLK", last_price=100.96),
        _obs(symbol="XLK", last_price=101.12),
        _obs(symbol="XLK", last_price=101.28),
    ]
    assert v52._runtime_sector_rotation_label(sequence, 0, 8) == 1.0


def test_train_brain_uses_runtime_path_without_silent_fallback(monkeypatch) -> None:
    modules = [
        (v35, "brain_refinery_v35_dmi_state_machine", "SPY"),
        (v38, "brain_refinery_v38_multi_timeframe_confirmation", "QQQ"),
        (v41, "brain_refinery_v41_long_interval_trend", "SCHD"),
        (v52, "brain_refinery_v52_sector_rotation_rs", "XLK"),
    ]
    for module, run_tag, expected_symbol in modules:
        captured = {}

        def _fake_train_runtime_indicator_bot(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(module, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

        result = module.train_brain()

        assert result == "ok"
        assert captured["run_tag"] == run_tag
        assert captured["allow_fallback_on_insufficient_data"] is False
        assert callable(captured["runtime_label_builder"])
        assert callable(captured["sample_filter"])
        assert captured["mode_allowlist"]
        assert expected_symbol in captured["symbol_allowlist"]
        assert captured["require_both_sides_precision"] is True
        assert captured["min_accuracy_lift_over_majority"] == 0.02
