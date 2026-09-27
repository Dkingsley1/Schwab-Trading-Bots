from __future__ import annotations

from core.sleeve_model_boundary import (
    evaluate_bot_scope,
    load_policy,
    project_model_features,
)
from core.institutional_decision_flow import load_policy as load_decision_policy


def test_conservative_projection_excludes_known_derivative_domains() -> None:
    features = {
        "mom_5m": 0.2,
        "quality_score_norm": 0.8,
        "risk_on_off_state_norm": 0.4,
        "futures_basis_norm": 0.7,
        "option_gamma_norm": 0.5,
        "crypto_funding_norm": 0.6,
        "custom_legacy_feature": 1.0,
    }

    projected, receipt = project_model_features(
        features,
        profile="conservative",
        family_id="balanced_directional",
    )

    assert projected["mom_5m"] == 0.2
    assert projected["quality_score_norm"] == 0.8
    assert projected["risk_on_off_state_norm"] == 0.4
    assert projected["custom_legacy_feature"] == 1.0
    assert "futures_basis_norm" not in projected
    assert "option_gamma_norm" not in projected
    assert "crypto_funding_norm" not in projected
    assert receipt["excluded_feature_count"] == 3
    assert receipt["projection_receipt_sha256"]
    assert receipt["full_observability_preserved"] is True


def test_futures_profile_addition_preserves_futures_inputs() -> None:
    features = {
        "mom_5m": 0.2,
        "quote_age_norm": 0.1,
        "futures_basis_norm": 0.7,
        "open_interest_change_norm": 0.3,
        "option_gamma_norm": 0.5,
    }

    projected, receipt = project_model_features(
        features,
        profile="futures_index_intraday",
        family_id="intraday_momentum",
    )

    assert "futures_basis_norm" in projected
    assert "open_interest_change_norm" in projected
    assert "option_gamma_norm" not in projected
    assert "futures" in receipt["allowed_feature_groups"]


def test_projection_receipt_is_deterministic_for_same_schema() -> None:
    features = {"mom_5m": 0.2, "option_gamma_norm": 0.5}
    _, first = project_model_features(
        features,
        profile="conservative",
        family_id="balanced_directional",
    )
    _, second = project_model_features(
        {"mom_5m": 9.0, "option_gamma_norm": 8.0},
        profile="conservative",
        family_id="balanced_directional",
    )

    assert first["projection_receipt_sha256"] == second["projection_receipt_sha256"]


def test_cross_cutting_research_and_safety_context_is_preserved() -> None:
    projected, receipt = project_model_features(
        {
            "quant_strategy_selection_confidence_norm": 0.8,
            "infra_veto_active": 1.0,
            "runtime_memory_pressure_norm": 0.3,
            "option_gamma_norm": 0.5,
        },
        profile="conservative",
        family_id="balanced_directional",
    )

    assert projected["quant_strategy_selection_confidence_norm"] == 0.8
    assert projected["infra_veto_active"] == 1.0
    assert projected["runtime_memory_pressure_norm"] == 0.3
    assert "option_gamma_norm" not in projected
    assert receipt["excluded_feature_count"] == 1


def test_explicit_cross_family_bot_is_blocked() -> None:
    decision = evaluate_bot_scope(
        {
            "bot_id": "rates_specialist",
            "bot_role": "signal_sub_bot",
            "sleeve_profile": "futures_rates_curve",
        },
        runtime_profile="conservative",
        runtime_family_id="balanced_directional",
        decision_policy=load_decision_policy(),
    )

    assert decision["allowed"] is False
    assert decision["status"] == "blocked_cross_sleeve_scope"
    assert decision["scope_receipt_sha256"]


def test_same_family_bot_and_legacy_unscoped_bot_remain_eligible() -> None:
    decision_policy = load_decision_policy()
    same_family = evaluate_bot_scope(
        {
            "bot_id": "day_specialist",
            "bot_role": "signal_sub_bot",
            "sleeve_profile": "day_trading",
        },
        runtime_profile="futures_index_intraday",
        runtime_family_id="intraday_momentum",
        decision_policy=decision_policy,
    )
    legacy = evaluate_bot_scope(
        {"bot_id": "legacy", "bot_role": "signal_sub_bot"},
        runtime_profile="conservative",
        runtime_family_id="balanced_directional",
        decision_policy=decision_policy,
    )

    assert same_family["allowed"] is True
    assert same_family["status"] == "allowed_explicit_scope"
    assert legacy["allowed"] is True
    assert legacy["status"] == "allowed_legacy_unscoped"


def test_sub_sleeve_identifier_is_audited_but_not_treated_as_family_scope() -> None:
    decision = evaluate_bot_scope(
        {
            "bot_id": "legacy_scoped_by_execution_contract",
            "bot_role": "signal_sub_bot",
            "paper_sub_sleeve_id": "futures_rates_curve_child_7",
        },
        runtime_profile="conservative",
        runtime_family_id="balanced_directional",
        decision_policy=load_decision_policy(),
    )

    assert decision["allowed"] is True
    assert decision["status"] == "allowed_legacy_unscoped"
    assert decision["scope_claims"] == []
    assert decision["paper_sub_sleeve_id"] == "futures_rates_curve_child_7"


def test_derivative_role_is_rejected_outside_compatible_family() -> None:
    decision_policy = load_decision_policy()
    blocked = evaluate_bot_scope(
        {"bot_id": "options_specialist", "bot_role": "options_sub_bot"},
        runtime_profile="conservative",
        runtime_family_id="balanced_directional",
        decision_policy=decision_policy,
    )
    allowed = evaluate_bot_scope(
        {"bot_id": "futures_specialist", "bot_role": "futures_sub_bot"},
        runtime_profile="futures_index_intraday",
        runtime_family_id="intraday_momentum",
        decision_policy=decision_policy,
    )

    assert blocked["status"] == "blocked_role_family_mismatch"
    assert blocked["allowed"] is False
    assert allowed["allowed"] is True


def test_default_policy_validates() -> None:
    policy = load_policy()
    assert policy["policy_id"] == "sleeve_model_boundary_v1"
