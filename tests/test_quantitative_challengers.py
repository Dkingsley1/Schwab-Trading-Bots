from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.quantitative_challengers import (
    block_bootstrap_model_selection,
    combinatorial_purged_splits,
    cost_aware_expert_aggregation,
    cpcv_triple_barrier_diagnostic,
    drawdown_constrained_kelly,
    entropy_pooling_downside_view,
    least_squares_optimal_stopping,
    probabilistic_sharpe_bayesian_utility,
    sequential_sign_sprt,
    triple_barrier_events,
)
from scripts.quantitative_challenger_report import METHOD_IDS, build_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _edge_returns(count: int = 120) -> list[float]:
    pattern = [18.0, 12.0, 9.0, -5.0, 14.0, 7.0, -3.0, 11.0, 16.0, -4.0]
    return [pattern[index % len(pattern)] for index in range(count)]


def _performance_payload(candidate_id: str = "pc-test-g1") -> dict[str, object]:
    profiles: dict[str, list[dict[str, object]]] = {}
    for profile, offset in (("alpha", 0.0), ("beta", -2.0), ("gamma", -5.0)):
        profiles[profile] = [
            {
                "day_utc": f"202601{index + 1:02d}",
                "sample_count": 2,
                "post_cost_pnl_delta_total": value / 10.0,
                "post_cost_return_bps_total": value + offset,
            }
            for index, value in enumerate(_edge_returns(30))
        ]
    return {
        "timestamp_utc": "2026-02-01T00:00:00+00:00",
        "profitability_evidence_window": {
            "candidate_id": candidate_id,
            "candidate_generation": 1,
            "candidate_cutoff_utc": "2026-01-01T00:00:00+00:00",
            "evidence_through_utc": "2026-02-01T00:00:00+00:00",
            "candidate_filter_active": True,
            "candidate_binding_required": True,
            "candidate_binding_mismatch_rows_excluded": 0,
        },
        "candidate_post_cost_daily_series": profiles,
        "sleeve_latest": [{"profile": profile} for profile in profiles],
    }


def test_sequential_inference_is_always_valid_and_supports_known_edge() -> None:
    result = sequential_sign_sprt(_edge_returns(), minimum_observations=20)

    assert result["available"] is True
    assert result["passes"] is True
    assert result["decision"] == "support_positive_edge"
    assert 0.0 <= result["always_valid_p_value"] <= 0.05


def test_spa_reality_check_is_seeded_and_model_family_aware() -> None:
    series = {
        "edge": _edge_returns(60),
        "flat": [4.0, -4.0] * 30,
        "weak": [2.0, -5.0] * 30,
    }
    first = block_bootstrap_model_selection(series, replications=200, seed=42)
    second = block_bootstrap_model_selection(series, replications=200, seed=42)

    assert first == second
    assert first["available"] is True
    assert first["best_profile"] == "edge"
    assert first["profile_count"] == 3


def test_probabilistic_sharpe_and_bayesian_utility_are_reproducible() -> None:
    first = probabilistic_sharpe_bayesian_utility(
        _edge_returns(), posterior_draws=500, seed=7
    )
    second = probabilistic_sharpe_bayesian_utility(
        _edge_returns(), posterior_draws=500, seed=7
    )

    assert first == second
    assert first["available"] is True
    assert first["probabilistic_sharpe_probability"] > 0.95
    assert first["posterior_positive_mean_probability"] > 0.95


def test_drawdown_kelly_is_bounded_and_has_no_sizing_authority() -> None:
    result = drawdown_constrained_kelly(
        _edge_returns(), max_fraction=0.20, drawdown_limit=0.02
    )

    assert result["available"] is True
    assert 0.0 <= result["challenger_fraction"] <= 0.20
    assert result["challenger_max_drawdown"] <= 0.02
    assert result["authority"] == "diagnostic_only_no_sizing_authority"


def test_entropy_pooling_increases_downside_mass_without_rewriting_samples() -> None:
    result = entropy_pooling_downside_view(
        _edge_returns(), target_tail_probability=0.40
    )

    assert result["available"] is True
    assert result["target_tail_probability"] >= result["base_tail_probability"]
    assert result["effective_scenario_count"] <= result["observation_count"]
    assert result["view"] == "increase_probability_mass_on_empirical_downside_tail"


def test_optimal_stopping_uses_disjoint_paths_and_holdout() -> None:
    returns = [10.0, 8.0, -15.0, -10.0, -8.0] * 24
    result = least_squares_optimal_stopping(
        returns,
        horizon=5,
        minimum_paths=12,
    )

    assert result["available"] is True
    assert result["overlapping_paths_used"] is False
    assert result["training_path_count"] + result["holdout_path_count"] == result["independent_path_count"]
    assert result["authority"] == "diagnostic_only_no_entry_or_exit_authority"


def test_cpcv_purges_overlapping_labels_and_embargoes_neighbors() -> None:
    events = triple_barrier_events(
        [12.0, -4.0, 8.0, -3.0, 11.0, -5.0] * 10,
        horizon=4,
    )
    splits = combinatorial_purged_splits(
        events,
        group_count=6,
        test_group_count=2,
        embargo_observations=1,
    )
    diagnostic = cpcv_triple_barrier_diagnostic(
        [12.0, -4.0, 8.0, -3.0, 11.0, -5.0] * 10,
        minimum_observations=30,
    )

    assert splits["split_count"] == 15
    assert splits["leakage_violation_count"] == 0
    assert any(row["purged_count"] > 0 for row in splits["splits"])
    assert diagnostic["training_authority"] is False


def test_online_expert_aggregation_respects_cost_and_weight_caps() -> None:
    result = cost_aware_expert_aggregation(
        {
            "alpha": _edge_returns(40),
            "beta": [2.0, -3.0] * 20,
            "gamma": [-1.0, 1.0] * 20,
        },
        maximum_weight=0.50,
        transaction_cost_bps=3.0,
    )

    assert result["available"] is True
    assert max(result["final_weights"].values()) <= 0.50 + 1e-8
    assert pytest.approx(sum(result["final_weights"].values()), abs=1e-7) == 1.0
    assert result["authority"] == "paper_counterfactual_only_no_allocator_authority"


def test_report_implements_all_eight_methods_and_is_candidate_bound(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_dir = project / "config"
    health_dir = project / "governance" / "health"
    config_dir.mkdir(parents=True)
    health_dir.mkdir(parents=True)
    policy = json.loads(
        (PROJECT_ROOT / "config" / "quantitative_challengers_v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy_path = config_dir / "quantitative_challengers_v1.json"
    performance_path = health_dir / "paper_performance_latest.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    performance_path.write_text(json.dumps(_performance_payload()), encoding="utf-8")

    payload = build_payload(
        project,
        policy_path=policy_path,
        performance_path=performance_path,
        generated_at_utc="2026-02-01T00:01:00+00:00",
    )

    assert payload["ok"] is True
    assert payload["candidate_binding"]["bound"] is True
    assert payload["concept_count"] == payload["implemented_concept_count"] == 8
    assert set(payload["method_availability"]) == set(METHOD_IDS)
    assert not any(payload["authority_contract"].values())
    metadata = payload["decision_metadata_by_profile"]["alpha"]
    assert metadata["authority"] == "read_only_metadata_no_decision_authority"
    assert metadata["method_count"] == 8
    assert set(metadata["method_statuses"]) == set(METHOD_IDS)


def test_report_rejects_candidate_mismatch_and_policy_authority(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_dir = project / "config"
    health_dir = project / "governance" / "health"
    config_dir.mkdir(parents=True)
    health_dir.mkdir(parents=True)
    policy = json.loads(
        (PROJECT_ROOT / "config" / "quantitative_challengers_v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy_path = config_dir / "quantitative_challengers_v1.json"
    performance_path = health_dir / "paper_performance_latest.json"
    performance = _performance_payload()
    performance["profitability_evidence_window"]["candidate_binding_mismatch_rows_excluded"] = 1
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    performance_path.write_text(json.dumps(performance), encoding="utf-8")

    payload = build_payload(
        project,
        policy_path=policy_path,
        performance_path=performance_path,
    )
    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert payload["evidence_ready_concept_count"] == 0

    unsafe = deepcopy(policy)
    unsafe["authority"]["changes_position_size"] = True
    policy_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden authority"):
        build_payload(
            project,
            policy_path=policy_path,
            performance_path=performance_path,
        )
