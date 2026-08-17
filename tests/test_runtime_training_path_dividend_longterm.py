import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v93_dividend_quality_compounder as v93
from core import brain_refinery_v94_dividend_yield_trap_avoidance as v94


def _obs(symbol="SCHD", last_price=100.0, **overrides):
    base_features = {
        "pct_from_close": 0.0020,
        "mom_5m": 0.0011,
        "vol_30m": 0.006,
        "range_pos": 0.61,
        "ctx_VIX_X_pct_from_close": 0.008,
        "bond_duration_regime_norm": 0.52,
        "bond_credit_risk_off_norm": 0.44,
        "dividend_yield_norm": 0.39,
        "dividend_payout_ratio_norm": 0.46,
        "dividend_quality_score_norm": 0.82,
        "dividend_capture_entry_signal_norm": 0.55,
        "dividend_capture_exit_signal_norm": 0.21,
        "dividend_compound_bias_norm": 0.68,
        "dividend_compound_growth_norm": 0.70,
        "dividend_compound_drawdown_norm": 0.10,
        "dividend_drip_active_norm": 0.84,
        "dividend_drip_recent_reinvest_norm": 0.63,
        "dividend_drip_cash_only_norm": 0.08,
        "dividend_drip_share_credit_norm": 0.57,
        "dividend_drip_confidence_norm": 0.88,
        "dividend_safety_composite_norm": 0.80,
        "dividend_growth_momentum_norm": 0.71,
        "dividend_streak_quality_norm": 0.74,
        "dividend_fcf_coverage_norm": 0.68,
        "dividend_interest_coverage_quality_norm": 0.66,
        "dividend_structure_aware_quality_norm": 0.72,
        "dividend_income_quality_norm": 0.70,
        "dividend_total_return_income_norm": 0.73,
        "dividend_position_age_norm": 0.56,
        "dividend_tax_friction_norm": 0.24,
        "dividend_cut_freeze_risk_norm": 0.18,
        "dividend_debt_funded_risk_norm": 0.14,
        "dividend_forward_hazard_norm": 0.22,
        "dividend_trap_internal_risk_norm": 0.16,
        "dividend_corporate_action_hazard_norm": 0.12,
        "long_term_quality_dividend_norm": 0.76,
        "long_term_total_return_income_norm": 0.67,
        "capital_flow_outflow_norm": 0.12,
        "options_negative_bias_norm": 0.16,
        "data_quality_quote_agreement_norm": 0.96,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_runtime_feature_vector_shapes_dividend_slow_sleeves() -> None:
    sequence = [_obs()]
    assert v93._runtime_feature_vector(sequence, 0).shape == (38,)
    assert v94._runtime_feature_vector(sequence, 0).shape == (39,)


def test_v93_runtime_label_returns_positive_for_supported_compound_income_path() -> None:
    sequence = [
        _obs(last_price=100.0),
        _obs(last_price=100.12),
        _obs(last_price=100.26),
        _obs(last_price=100.42),
        _obs(last_price=100.58),
        _obs(last_price=100.74),
        _obs(last_price=100.92),
        _obs(last_price=101.10),
        _obs(last_price=101.28),
        _obs(last_price=101.46),
        _obs(last_price=101.64),
        _obs(last_price=101.82),
        _obs(last_price=102.00),
    ]
    assert v93._runtime_quality_compound_label(sequence, 0, 12) == 1.0


def test_train_brain_uses_extended_runtime_features_for_dividend_sleeves(monkeypatch) -> None:
    modules = [
        (v93, "brain_refinery_v93_dividend_quality_compounder"),
        (v94, "brain_refinery_v94_dividend_yield_trap_avoidance"),
    ]
    for module, run_tag in modules:
        captured = {}

        def _fake_train_runtime_indicator_bot(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(module, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)
        assert module.train_brain() == "ok"
        assert captured["run_tag"] == run_tag
        assert "dividend_total_return_income_norm" in captured["feature_names"]
        assert "dividend_fcf_coverage_norm" in captured["feature_names"]
