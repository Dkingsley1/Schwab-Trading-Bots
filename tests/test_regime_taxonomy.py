from __future__ import annotations

import copy
import json
from pathlib import Path

from core.regime_taxonomy import (
    REQUIRED_REGIME_AXES,
    build_regime_metadata_access,
    build_regime_metadata_view,
    classify_regime_profile,
    evaluate_regime_compatibility,
    validate_regime_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model() -> dict:
    policy = json.loads(
        (PROJECT_ROOT / "config" / "bot_organization_v1.json").read_text(encoding="utf-8")
    )
    return policy["regime_model"]


def _profile(
    preferred_regimes: list[str],
    *,
    role: str = "signal_sub_bot",
    family: str = "trend_and_momentum",
    regime_axes: dict | None = None,
) -> dict:
    row = {
        "bot_id": "test_bot",
        "bot_role": role,
        "preferred_regimes": preferred_regimes,
    }
    if regime_axes is not None:
        row["regime_axes"] = regime_axes
    return classify_regime_profile(
        row=row,
        module_spec={},
        classification_text="test bot",
        raw_role=role,
        role_id="signal" if role == "signal_sub_bot" else "shared_service",
        sub_sleeve_id=family,
        horizon_id="intraday",
        model=_model(),
    )


def test_policy_declares_exact_versioned_axis_contract() -> None:
    model = _model()

    assert validate_regime_model(model) == []
    assert tuple(row["axis_id"] for row in model["axes"]) == REQUIRED_REGIME_AXES
    assert model["mode"] == "multi_axis_shadow_only"
    assert model["safety_contract"]["paper_execution_authority"] is False
    assert model["safety_contract"]["live_execution_authority"] is False
    assert model["scenario_partition_contract"]["version"] == "regime_scenario_partition_v1"
    assert model["scenario_partition_contract"]["mode"] == "bounded_best_match_shadow_only"
    assert model["scenario_partition_contract"]["maximum_scenarios_per_profile"] == 12
    assert model["scenario_partition_contract"]["paper_execution_authority"] is False
    assert model["scenario_partition_contract"]["live_execution_authority"] is False
    assert model["metadata_access_contract"]["version"] == "regime_metadata_access_v1"
    assert model["metadata_access_contract"]["mode"] == "read_only_runtime_context"
    assert model["metadata_access_contract"]["infer_missing_profile_preferences"] is False
    assert model["metadata_access_contract"]["paper_execution_authority"] is False
    assert model["metadata_access_contract"]["live_execution_authority"] is False


def test_legacy_profile_gets_read_only_metadata_without_fabricated_preferences() -> None:
    profile = _profile([])

    access = build_regime_metadata_access(profile, _model())
    view = build_regime_metadata_view(
        profile,
        {
            "axes": {
                "market_direction": ["bull_trend"],
                "volatility_state": ["normal"],
                "liquidity_state": ["deep"],
            }
        },
        _model(),
    )

    assert access["access_ready"] is True
    assert access["profile_preferences_complete"] is False
    assert set(access["runtime_context_required_axis_ids"]) == set(profile["unknown_axes"])
    assert access["preference_inference_allowed"] is False
    assert view["context_ready"] is True
    assert view["context_axes"]["market_direction"] == ["bull_trend"]
    assert view["authority"]["training_observation_allowed"] is True
    assert view["authority"]["paper_execution_authority"] is False
    assert view["authority"]["live_execution_authority"] is False


def test_metadata_access_contract_fails_closed_on_execution_authority() -> None:
    unsafe = copy.deepcopy(_model())
    unsafe["metadata_access_contract"]["live_execution_authority"] = True

    assert (
        "regime_metadata_access_live_execution_authority_must_be_false"
        in validate_regime_model(unsafe)
    )


def test_market_labels_are_classified_independently_across_axes() -> None:
    profile = _profile(["risk_off_shock", "event_volatility", "thin_liquidity"])

    assert profile["scope"] == "market_signal"
    assert profile["axes"]["market_direction"]["values"] == ["bear_trend"]
    assert profile["axes"]["volatility_state"]["values"] == ["crisis", "elevated"]
    assert profile["axes"]["liquidity_state"]["values"] == ["thin"]
    assert profile["axes"]["event_phase"]["values"] == ["event_window"]
    assert profile["axes"]["operational_state"]["values"] == ["not_applicable"]


def test_operational_labels_do_not_masquerade_as_market_regimes() -> None:
    profile = _profile(
        ["schema_drift", "backpressure_spike"],
        role="infrastructure_sub_bot",
        family="data_and_model_governance",
    )

    assert profile["scope"] == "operational_control"
    assert profile["axes"]["operational_state"]["values"] == [
        "backlog_pressure",
        "evidence_review",
    ]
    for axis_id in REQUIRED_REGIME_AXES[:-1]:
        assert profile["axes"][axis_id]["values"] == ["not_applicable"]


def test_all_weather_is_an_explicit_wildcard_not_an_unknown() -> None:
    profile = _profile(["all_weather"])

    assert profile["unknown_axes"] == []
    assert set(profile["wildcard_axes"]) == set(profile["quality_axes"])
    assert profile["axis_coverage_ratio"] == 1.0
    assert profile["axis_specificity_ratio"] == 0.0


def test_explicit_axis_metadata_precedes_legacy_labels() -> None:
    profile = _profile(
        ["risk_off_shock"],
        regime_axes={
            "market_direction": ["bull_trend"],
            "volatility_state": ["normal"],
            "liquidity_state": ["deep"],
        },
    )

    assert profile["axes"]["market_direction"]["values"] == ["bull_trend"]
    assert profile["axes"]["market_direction"]["source"] == "registry_explicit"
    assert profile["axes"]["volatility_state"]["values"] == ["normal"]
    assert profile["axes"]["liquidity_state"]["values"] == ["deep"]


def test_compatibility_is_weighted_auditable_and_execution_free() -> None:
    profile = _profile(["risk_off_shock", "thin_liquidity"])
    matching = evaluate_regime_compatibility(
        profile,
        {
            "axes": {
                "market_direction": ["bear_trend"],
                "volatility_state": ["crisis"],
                "liquidity_state": ["thin"],
            }
        },
        _model(),
    )
    mismatch = evaluate_regime_compatibility(
        profile,
        {
            "axes": {
                "market_direction": ["bull_trend"],
                "volatility_state": ["normal"],
                "liquidity_state": ["deep"],
            }
        },
        _model(),
    )

    assert matching["compatible"] is True
    assert matching["score"] == 1.0
    assert mismatch["compatible"] is False
    assert mismatch["reason"] == "critical_regime_axis_mismatch"
    assert mismatch["hard_mismatch_axis_ids"] == [
        "market_direction",
        "volatility_state",
        "liquidity_state",
    ]
    assert matching["authority"]["order_payload_created"] is False
    assert matching["authority"]["paper_execution_authority"] is False
    assert matching["authority"]["live_execution_authority"] is False


def test_invalid_context_value_fails_closed() -> None:
    result = evaluate_regime_compatibility(
        _profile(["all_weather"]),
        {
            "axes": {
                "market_direction": ["invented_state"],
                "volatility_state": ["normal"],
                "liquidity_state": ["normal"],
            }
        },
        _model(),
    )

    assert result["compatible"] is False
    assert result["reason"] == "invalid_regime_context"
    assert result["invalid_context_values"] == {
        "market_direction": ["invented_state"]
    }


def test_broad_router_is_partitioned_into_bounded_operational_scenarios() -> None:
    profile = classify_regime_profile(
        row={
            "bot_id": "router",
            "bot_role": "signal_sub_bot",
            "regime_scope": "operational_control",
            "preferred_regimes": ["normal_collection", "market_hours_pressure", "global_halt_review"],
            "regime_scenarios": [
                {
                    "scenario_id": "normal_collection",
                    "preferred_regimes": ["normal_collection"],
                    "regime_axes": {"operational_state": ["normal_operations"]},
                },
                {
                    "scenario_id": "market_hours_pressure",
                    "preferred_regimes": ["market_hours_pressure"],
                    "regime_axes": {"operational_state": ["resource_pressure"]},
                },
                {
                    "scenario_id": "global_halt_review",
                    "preferred_regimes": ["global_halt_review"],
                    "regime_axes": {"operational_state": ["halted"]},
                },
            ],
        },
        module_spec={},
        classification_text="operational regime router",
        raw_role="signal_sub_bot",
        role_id="signal",
        sub_sleeve_id="regime_and_forecasting",
        horizon_id="medium_term",
        model=_model(),
    )

    assert profile["scope"] == "operational_control"
    assert profile["scope_source"] == "registry_explicit"
    assert profile["scenario_partitioned"] is True
    assert profile["scenario_count"] == 3
    assert profile["scenario_contract_errors"] == []
    assert profile["scenario_review_reasons"] == []
    assert profile["multi_value_axes"] == []
    assert profile["axis_coverage_ratio"] == 1.0
    assert profile["axis_specificity_ratio"] == 1.0
    assert profile["requires_review"] is False

    compatibility = evaluate_regime_compatibility(
        profile,
        {"axes": {"operational_state": ["resource_pressure"]}},
        _model(),
    )
    assert compatibility["compatible"] is True
    assert compatibility["reason"] == "regime_scenario_compatible"
    assert compatibility["selected_scenario_id"] == "market_hours_pressure"
    assert compatibility["evaluated_scenario_count"] == 3
    assert compatibility["authority"]["paper_execution_authority"] is False
    assert compatibility["authority"]["live_execution_authority"] is False


def test_invalid_scenario_partition_fails_closed() -> None:
    profile = classify_regime_profile(
        row={
            "bot_id": "broken_router",
            "bot_role": "infrastructure_sub_bot",
            "regime_scope": "operational_control",
            "regime_scenarios": [
                {
                    "scenario_id": "duplicate",
                    "regime_axes": {"operational_state": ["normal_operations"]},
                },
                {
                    "scenario_id": "duplicate",
                    "regime_axes": {"operational_state": ["halted"]},
                },
            ],
        },
        module_spec={},
        classification_text="broken operational router",
        raw_role="infrastructure_sub_bot",
        role_id="shared_service",
        sub_sleeve_id="control_and_evidence",
        horizon_id="continuous",
        model=_model(),
    )

    assert profile["requires_review"] is True
    assert "regime_scenario_duplicate_id_duplicate" in profile["scenario_contract_errors"]
    result = evaluate_regime_compatibility(
        profile,
        {"axes": {"operational_state": ["normal_operations"]}},
        _model(),
    )
    assert result["compatible"] is False
    assert result["reason"] == "invalid_regime_scenario_contract"
    assert result["authority"]["order_payload_created"] is False
