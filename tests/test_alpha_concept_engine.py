from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.alpha_concept_engine import (
    active_learning_value_of_information,
    capacity_impact_surface,
    cost_stress_survival,
    cross_fitted_causal_transportability,
    economic_alpha_decomposition,
    effective_breadth_transfer_coefficient,
    execution_alpha_attribution,
    factor_neutral_residualization,
    hierarchical_bayesian_skill,
    information_coefficient_term_structure,
    point_in_time_security_master_audit,
    regime_conditional_robustness,
    residual_redundancy_graph,
    sequential_change_point_stability,
    split_conformal_residual_calibration,
    subsample_stability_selection,
)
from scripts.ops.alpha_concept_report import build_payload, render_markdown
from scripts.ops.runtime_artifact_refresh import (
    PAPER_SOAK_MANAGED_STEPS,
    REFRESH_SCOPE_ROOTS,
    _step_specs,
)
from scripts.ops.system_self_model import (
    _alpha_concept_awareness,
    _generation_attribution_awareness,
    _sleeve_alpha_toolbox_awareness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_information_coefficient_term_structure_measures_rank_skill() -> None:
    observations = []
    for period in range(6):
        for asset in range(5):
            forecast = asset - 2.0 + period * 0.01
            observations.append(
                {
                    "horizon": "1d",
                    "period": f"p{period}",
                    "regime": "risk_on" if period < 3 else "risk_off",
                    "forecast": forecast,
                    "realized": 1.5 * forecast + ((period + asset) % 2) * 0.01,
                }
            )

    result = information_coefficient_term_structure(observations)

    assert result["available"] is True
    assert result["passes"] is True
    assert result["supported_horizon_count"] == 1
    assert result["horizons"][0]["rank_ic"] > 0.99
    assert result["horizons"][0]["period_ic_count"] == 6
    assert not any(result["authority"].values())


def test_effective_breadth_and_transfer_coefficient_are_bounded() -> None:
    forecasts = []
    realized = []
    for period in range(30):
        row = [
            math.sin(period * 0.17),
            math.cos(period * 0.11),
            math.sin(period * 0.07 + 1.2),
        ]
        forecasts.append(row)
        realized.append(
            [value + 0.01 * math.sin(period + index) for index, value in enumerate(row)]
        )

    result = effective_breadth_transfer_coefficient(
        forecasts,
        realized,
        implemented_weight_matrix=forecasts,
    )

    assert result["available"] is True
    assert result["passes"] is True
    assert 1.0 <= result["effective_bets_per_period"] <= 3.0
    assert result["effective_total_breadth"] <= 90.0
    assert result["transfer_coefficient"] == pytest.approx(1.0)


def test_hierarchical_bayesian_skill_shrinks_group_estimates() -> None:
    groups = {
        "alpha": [8.0, 10.0, 7.0, 11.0] * 8,
        "beta": [3.0, 5.0, 2.0, 4.0] * 8,
        "flat": [-1.0, 1.0, -0.5, 0.5] * 8,
    }

    result = hierarchical_bayesian_skill(groups)

    assert result["available"] is True
    assert result["supported_group_count"] >= 2
    alpha = next(row for row in result["groups"] if row["group"] == "alpha")
    assert result["grand_mean"] < alpha["sample_mean"]
    assert result["grand_mean"] < alpha["posterior_mean"] < alpha["sample_mean"]


def test_stability_selection_finds_repeated_feature_without_knockoff_claim() -> None:
    strong = [float(index) for index in range(80)]
    weak = [float((index * 17) % 13) for index in range(80)]
    noise = [float((index * 29) % 19) for index in range(80)]
    outcomes = [3.0 * value + 0.1 * weak[index] for index, value in enumerate(strong)]

    result = subsample_stability_selection(
        {"strong": strong, "weak": weak, "noise": noise},
        outcomes,
        replications=100,
        selected_feature_count=1,
        seed=5,
    )

    assert result["available"] is True
    assert result["stable_features"] == ["strong"]
    assert result["model_x_knockoff_guarantee_claimed"] is False


def test_factor_residualization_and_economic_decomposition_retain_true_alpha() -> None:
    factor = [float(index - 40) / 20.0 for index in range(80)]
    residual = [0.1 if index % 2 else -0.1 for index in range(80)]
    gross = [2.0 + 1.5 * value + residual[index] for index, value in enumerate(factor)]
    costs = [0.5] * len(gross)

    residualized = factor_neutral_residualization(gross, {"market": factor})
    decomposed = economic_alpha_decomposition(
        gross,
        {"market": factor},
        costs,
    )

    assert residualized["available"] is True
    assert residualized["passes"] is True
    assert residualized["intercept_alpha"] == pytest.approx(2.0, abs=0.02)
    assert abs(residualized["residual_factor_correlations"]["market"]) < 1e-6
    assert decomposed["available"] is True
    assert decomposed["passes"] is True
    assert decomposed["execution_cost_drag"] == pytest.approx(-0.5)
    assert decomposed["net_active_return_mean"] == pytest.approx(
        decomposed["gross_active_return_mean"] - 0.5
    )


def test_cross_fitted_dml_recovers_stable_environment_effect() -> None:
    rng = np.random.default_rng(42)
    count = 180
    x = rng.normal(size=count)
    z = rng.normal(size=count)
    treatment = 0.8 * x + 0.3 * z + rng.normal(scale=0.5, size=count)
    outcome = 2.0 * treatment + 0.5 * x - 0.2 * z + rng.normal(scale=0.3, size=count)
    environments = ["calm", "volatile", "event"] * 60

    result = cross_fitted_causal_transportability(
        outcome.tolist(),
        treatment.tolist(),
        {"x": x.tolist(), "z": z.tolist()},
        environments,
    )

    assert result["available"] is True
    assert result["passes"] is True
    assert result["cross_fitted_effect"] == pytest.approx(2.0, abs=0.2)
    assert result["causal_claim_proven"] is False
    assert result["cross_fit_basis"] == "leave_one_environment_out_nuisance_estimation"


def test_capacity_surface_requires_explicit_costs_and_finds_cliff() -> None:
    missing = capacity_impact_surface(
        expected_gross_alpha_bps=None,
        half_spread_bps=None,
        fees_bps=None,
        baseline_slippage_bps=None,
        daily_dollar_volume=None,
        volatility_bps=None,
        impact_coefficient=None,
        notionals=[],
    )
    result = capacity_impact_surface(
        expected_gross_alpha_bps=20.0,
        half_spread_bps=1.0,
        fees_bps=0.2,
        baseline_slippage_bps=0.8,
        daily_dollar_volume=1_000_000.0,
        volatility_bps=200.0,
        impact_coefficient=0.2,
        notionals=[1_000.0, 10_000.0, 100_000.0, 500_000.0],
    )

    assert missing["available"] is False
    assert missing["unknown_cost_defaults_used"] is False
    assert result["available"] is True
    assert result["passes"] is True
    assert (
        result["surface"][0]["net_alpha_bps"] > result["surface"][-1]["net_alpha_bps"]
    )


def test_execution_attribution_uses_decision_arrival_fill_and_markout() -> None:
    result = execution_alpha_attribution(
        [
            {
                "side": "BUY",
                "quantity": 10,
                "decision_price": 100.0,
                "arrival_price": 100.1,
                "fill_price": 100.2,
                "fill_mid_price": 100.15,
                "markout_price": 101.0,
                "fee_bps": 0.1,
            },
            {
                "side": "SELL",
                "quantity": 5,
                "decision_price": 50.0,
                "arrival_price": 49.95,
                "fill_price": 49.9,
                "fill_mid_price": 49.925,
                "markout_price": 49.2,
                "fee_bps": 0.1,
            },
        ]
    )

    assert result["available"] is True
    assert result["passes"] is True
    assert result["markout_coverage_ratio"] == 1.0
    assert result["weighted_execution_alpha_after_markout_bps"] > 0.0


def test_point_in_time_security_master_resolves_symbol_reuse() -> None:
    records = [
        {
            "security_id": "old-abc",
            "symbol": "ABC",
            "valid_from": "2020-01-01T00:00:00Z",
            "valid_to": "2024-01-01T00:00:00Z",
            "identifiers": {"figi": "FIGI-OLD"},
            "status": "delisted",
            "delisted_at": "2023-12-31T21:00:00Z",
            "corporate_actions": [
                {
                    "effective_at": "2022-06-01T00:00:00Z",
                    "adjustment_factor": 0.5,
                }
            ],
        },
        {
            "security_id": "new-abc",
            "symbol": "ABC",
            "valid_from": "2024-01-01T00:00:00Z",
            "identifiers": {"figi": "FIGI-NEW"},
            "status": "active",
            "corporate_actions": [],
        },
    ]
    observations = [
        {"symbol": "ABC", "timestamp_utc": "2023-06-01T00:00:00Z"},
        {"symbol": "ABC", "timestamp_utc": "2025-06-01T00:00:00Z"},
    ]

    result = point_in_time_security_master_audit(records, observations)

    assert result["available"] is True
    assert result["passes"] is True
    assert result["resolved_observation_count"] == 2
    assert result["symbol_interval_overlap_count"] == 0


def test_split_conformal_residual_calibration_preserves_time_order() -> None:
    predictions = [float(index) / 10.0 for index in range(80)]
    outcomes = [
        prediction + (0.1 if index % 2 else -0.1)
        for index, prediction in enumerate(predictions)
    ]

    result = split_conformal_residual_calibration(predictions, outcomes)

    assert result["available"] is True
    assert result["passes"] is True
    assert result["time_order_preserved"] is True
    assert result["random_split_used"] is False
    assert result["empirical_coverage"] >= result["target_coverage"]
    assert not any(result["authority"].values())


def test_page_cusum_flags_recent_candidate_drift() -> None:
    stable = [0.1 * math.sin(index) for index in range(50)]
    shifted = stable[:40] + [4.0] * 10

    result = sequential_change_point_stability(
        {"stable": stable, "shifted": shifted},
        alarm_threshold_standard_deviations=5.0,
        recent_window_observations=12,
    )

    shifted_row = next(row for row in result["groups"] if row["group"] == "shifted")
    assert result["available"] is True
    assert shifted_row["recent_change_point"] is True
    assert result["passes"] is False


def test_residual_redundancy_graph_preserves_independent_components() -> None:
    first = [math.sin(index * 0.2) for index in range(80)]
    duplicate = [value + 0.001 * math.cos(index) for index, value in enumerate(first)]
    independent = [math.cos(index * 0.13) for index in range(80)]

    result = residual_redundancy_graph(
        {"first": first, "duplicate": duplicate, "independent": independent},
        absolute_correlation_threshold=0.8,
    )

    assert result["available"] is True
    assert result["passes"] is True
    assert result["redundant_pair_count"] >= 1
    assert result["independent_component_count"] == 2


def test_regime_robustness_requires_positive_lower_bounds_across_regimes() -> None:
    outcomes = [1.0 + 0.05 * math.sin(index) for index in range(60)]
    regimes = ["calm"] * 20 + ["volatile"] * 20 + ["event"] * 20

    result = regime_conditional_robustness(outcomes, regimes)

    assert result["available"] is True
    assert result["passes"] is True
    assert result["supported_regime_count"] == 3


def test_cost_stress_survival_is_monotonic_and_fail_closed() -> None:
    resilient = cost_stress_survival(20.0, 5.0, uncertainty_buffer_bps=1.0)
    fragile = cost_stress_survival(6.0, 5.0, uncertainty_buffer_bps=0.0)

    assert resilient["available"] is True
    assert resilient["passes"] is True
    assert (
        resilient["scenarios"][0]["conservative_net_edge_bps"]
        > resilient["scenarios"][-1]["conservative_net_edge_bps"]
    )
    assert fragile["passes"] is False
    assert not any(resilient["authority"].values())


def test_active_learning_ranks_economic_information_per_cost_without_label_authority() -> (
    None
):
    result = active_learning_value_of_information(
        [
            {
                "gap_id": "high_value",
                "uncertainty": 1.0,
                "economic_relevance": 1.0,
                "expected_uncertainty_reduction": 0.9,
                "novelty": 1.0,
                "coverage_deficit": 1.0,
                "observation_cost": 1.0,
                "resource_pressure": 0.0,
                "minimum_observations": 20,
                "observed_observations": 2,
            },
            {
                "gap_id": "low_value",
                "uncertainty": 0.2,
                "economic_relevance": 0.2,
                "expected_uncertainty_reduction": 0.2,
                "novelty": 0.2,
                "coverage_deficit": 0.2,
                "observation_cost": 2.0,
                "resource_pressure": 1.0,
            },
        ]
    )

    assert result["available"] is True
    assert result["top_gap_id"] == "high_value"
    assert result["ranked_collection_gaps"][0]["remaining_observations"] == 18
    assert result["labels_or_samples_created"] is False
    assert not any(result["authority"].values())


def _write_report_project(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    project = tmp_path / "project"
    policy = json.loads(
        (PROJECT_ROOT / "config" / "alpha_concept_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = project / "config" / "alpha_concept_registry_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(policy), encoding="utf-8")
    owner_paths = {
        str(row.get("owner") or "") for row in policy["family_defaults"].values()
    }
    owner_paths.update(
        str(row.get("owner") or "") for row in policy["operational_overrides"].values()
    )
    for relative in owner_paths:
        if not relative:
            continue
        owner = project / relative
        owner.parent.mkdir(parents=True, exist_ok=True)
        owner.touch()
    candidate_path = (
        project / "governance" / "runtime" / "production_candidate_state.json"
    )
    performance_path = (
        project / "governance" / "health" / "paper_performance_latest.json"
    )
    candidate_path.parent.mkdir(parents=True)
    performance_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps({"candidate_id": "pc-test-g1", "generation": 1}),
        encoding="utf-8",
    )
    performance_path.write_text(
        json.dumps(
            {
                "profitability_evidence_window": {
                    "candidate_id": "pc-test-g1",
                    "candidate_generation": 1,
                    "candidate_cutoff_utc": "2026-01-01T00:00:00Z",
                    "evidence_through_utc": "2026-02-01T00:00:00Z",
                    "candidate_filter_active": True,
                    "candidate_binding_required": True,
                    "candidate_binding_mismatch_rows_excluded": 0,
                },
                "candidate_post_cost_daily_series": {},
            }
        ),
        encoding="utf-8",
    )
    return project, policy


def test_report_separates_implementation_catalog_evidence_and_economics(
    tmp_path: Path,
) -> None:
    project, _policy = _write_report_project(tmp_path)

    payload = build_payload(
        project,
        generated_at_utc="2026-02-01T00:01:00Z",
    )
    markdown = render_markdown(payload)

    assert payload["ok"] is True
    assert payload["overall_status"] == "collecting_candidate_evidence"
    assert payload["grades"]["implementation_grade"] == "A+"
    assert payload["grades"]["catalog_routing_grade"] == "A+"
    assert payload["grades"]["candidate_evidence_grade"] == "F"
    assert payload["grades"]["economic_support_grade"] == "F"
    assert payload["catalog_summary"]["family_count"] == 16
    assert payload["catalog_summary"]["concept_count"] == 128
    assert payload["implemented_measurement_engine_count"] == 16
    assert payload["evidence_ready_measurement_engine_count"] == 0
    assert payload["advisory_measurement_engine_ready_count"] == 1
    assert not any(payload["authority_contract"].values())
    assert "Profitability guaranteed: **no**" in markdown


def test_report_accepts_only_candidate_bound_measurement_inputs(tmp_path: Path) -> None:
    project, _policy = _write_report_project(tmp_path)
    inputs_path = (
        project / "governance" / "research" / "alpha_concept_inputs_latest.json"
    )
    inputs_path.parent.mkdir(parents=True)
    inputs_path.write_text(
        json.dumps(
            {
                "candidate_id": "pc-test-g1",
                "timestamp_utc": "2026-02-01T00:00:00Z",
                "measurements": {
                    "execution_alpha_attribution": {
                        "inputs": {
                            "fills": [
                                {
                                    "side": "BUY",
                                    "quantity": 1,
                                    "decision_price": 100.0,
                                    "arrival_price": 100.1,
                                    "fill_price": 100.2,
                                    "markout_price": 101.0,
                                    "fee_bps": 0.1,
                                }
                            ]
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    accepted = build_payload(project)
    assert accepted["measurement_input_binding"]["bound"] is True
    assert accepted["measurements"]["execution_alpha_attribution"]["available"] is True
    assert accepted["evidence_ready_measurement_engine_count"] == 1
    assert accepted["economically_supported_measurement_engine_count"] == 0

    raw = json.loads(inputs_path.read_text(encoding="utf-8"))
    raw["measurements"]["execution_alpha_attribution"]["economic_grade_eligible"] = True
    raw["measurements"]["execution_alpha_attribution"][
        "evidence_class"
    ] = "candidate_bound_schema_v2_fill"
    inputs_path.write_text(json.dumps(raw), encoding="utf-8")
    economically_eligible = build_payload(project)
    assert economically_eligible["economically_supported_measurement_engine_count"] == 1

    raw["candidate_id"] = "wrong-candidate"
    inputs_path.write_text(json.dumps(raw), encoding="utf-8")
    rejected = build_payload(project)
    assert rejected["ok"] is False
    assert rejected["overall_status"] == "blocked"
    assert "measurement_input_candidate_mismatch" in rejected["blockers"]
    assert rejected["measurement_input_binding"]["bound"] is False
    assert rejected["measurements"]["execution_alpha_attribution"]["available"] is False
    assert rejected["evidence_ready_measurement_engine_count"] == 0


def test_report_rejects_authority_and_universal_coverage_claim(tmp_path: Path) -> None:
    project, policy = _write_report_project(tmp_path)
    config_path = project / "config" / "alpha_concept_registry_v1.json"
    unsafe = deepcopy(policy)
    unsafe["authority"]["changes_position_size"] = True
    config_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden authority"):
        build_payload(project)

    universal = deepcopy(policy)
    universal["scope_contract"][
        "claims_to_enumerate_every_possible_alpha_concept"
    ] = True
    config_path.write_text(json.dumps(universal), encoding="utf-8")
    with pytest.raises(ValueError, match="universal enumeration"):
        build_payload(project)


def test_alpha_report_is_managed_by_profitability_refresh_scopes() -> None:
    specs = {row["name"]: row for row in _step_specs(PROJECT_ROOT)}
    spec = specs["alpha_concept_report_verified"]

    assert "alpha_concept_report_verified" in PAPER_SOAK_MANAGED_STEPS
    assert "alpha_concept_report_verified" in REFRESH_SCOPE_ROOTS["profitability"]
    assert (
        "alpha_concept_report_verified" in REFRESH_SCOPE_ROOTS["training-profitability"]
    )
    assert spec["depends_on"] == [
        "paper_performance_verified",
        "quantitative_challenger_verified",
        "alpha_generation_control",
    ]
    assert spec["cmd"][-1] == "--json"
    toolbox = specs["sleeve_alpha_toolbox_verified"]
    assert "sleeve_alpha_toolbox_verified" in PAPER_SOAK_MANAGED_STEPS
    assert "sleeve_alpha_toolbox_verified" in REFRESH_SCOPE_ROOTS["profitability"]
    assert toolbox["depends_on"] == ["alpha_concept_report_verified"]
    assert toolbox["cmd"][-1] == "--json"


def test_self_model_keeps_collecting_alpha_evidence_advisory() -> None:
    awareness = _alpha_concept_awareness(
        {
            "ok": True,
            "overall_status": "collecting_candidate_evidence",
            "grades": {
                "implementation_grade": "A+",
                "catalog_routing_grade": "A+",
                "candidate_evidence_grade": "F",
                "economic_support_grade": "F",
            },
            "catalog_summary": {"concept_count": 128, "family_count": 16},
            "candidate_evidence_measurement_engine_count": 15,
            "evidence_ready_measurement_engine_count": 0,
            "authority_contract": {"submits_live_orders": False},
        }
    )

    assert awareness["status"] == "advisory"
    assert awareness["candidate_evidence_ready_count"] == 0
    assert awareness["candidate_evidence_measurement_count"] == 15
    assert awareness["economic_support_grade"] == "F"
    assert awareness["authority_contract"]["submits_live_orders"] is False
    assert awareness["authority_clear"] is True


def test_self_model_exposes_toolbox_and_generation_limits() -> None:
    toolbox = _sleeve_alpha_toolbox_awareness(
        {
            "ok": True,
            "overall_status": "structurally_ready_collecting_candidate_evidence",
            "candidate_binding": {"candidate_id": "candidate-g101"},
            "coverage": {
                "declared_sleeve_count": 111,
                "routed_sleeve_count": 111,
                "missing_sleeve_count": 0,
                "candidate_evidence_ready_sleeve_count": 0,
                "policy_family_counts": {"research": 111},
                "policy_match_source_counts": {"exact_profile": 111},
            },
            "authority": {"submits_live_order": False},
        }
    )
    generation = _generation_attribution_awareness(
        {
            "candidate_event_chain": {"valid": True},
            "comparison": {
                "from_generation": 65,
                "to_generation": 99,
                "behavior_comparison_ready": True,
                "identity_bound_behavior_comparison_ready": False,
                "legacy_window_association_involved": True,
                "economic_comparison_ready": False,
            },
            "cumulative_soak_context": {
                "main_soak_elapsed_hours": 458.5,
                "main_soak_active_runtime_evidence_hours": 454.0,
            },
        }
    )

    assert toolbox["status"] == "advisory"
    assert toolbox["routed_sleeve_count"] == 111
    assert toolbox["default_fallback_route_count"] == 0
    assert toolbox["live_execution_authority"] is False
    assert generation["status"] == "advisory"
    assert generation["legacy_window_association_involved"] is True
    assert generation["association_is_causal_proof"] is False
