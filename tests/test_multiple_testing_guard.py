import gzip
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.multiple_testing_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_multiple_testing_guard_builds_hypothesis_family(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "replay_feature_ablation_latest.json",
        {
            "ok": True,
            "ablation": {
                "baseline": {"macro_f1": 0.55},
                "without_macro_context": {"macro_f1": 0.53},
                "without_paper_replay": {"macro_f1": 0.50},
            },
            "strict_checks": {"require_full_dim_match": True},
            "delta": {"macro_f1_no_paper_minus_base": -0.05},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "counterfactual_replay_latest.json",
        {
            "ok": True,
            "candidate_count": 11,
            "profiles_reviewed": ["default", "intraday_aggressive"],
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 4},
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["family_size"] == 17
    assert payload["correction_method"] == "benjamini_hochberg_fdr"
    assert sorted(payload["regime_segments"]) == ["default", "intraday_aggressive"]


def test_multiple_testing_guard_uses_partial_contract_when_ablation_artifact_is_missing(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "counterfactual_replay_latest.json",
        {
            "ok": True,
            "candidate_count": 11,
            "profiles_reviewed": ["default", "intraday_aggressive"],
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 0, "coverage_shortfall_bots": 4},
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is False
    assert payload["overall_status"] == "needs_work"
    assert payload["contract_present"] is True
    assert payload["counterfactual_contract_ready"] is True


def test_multiple_testing_guard_never_uses_less_than_registered_strategy_family(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "replay_feature_ablation_latest.json",
        {
            "ok": True,
            "ablation": {"baseline": {"macro_f1": 0.5}, "without_feature": {"macro_f1": 0.4}},
            "strict_checks": {"require_full_dim_match": True},
            "failed_checks": [],
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "counterfactual_replay_latest.json",
        {"ok": True, "candidate_count": 2, "profiles_reviewed": ["default"]},
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 1},
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"strategy-{index}", "bot_role": "signal_sub_bot"}
                for index in range(20)
            ]
            + [{"bot_id": "infra", "bot_role": "infrastructure_bot"}]
            + [
                {
                    "bot_id": "deleted",
                    "bot_role": "signal_sub_bot",
                    "lifecycle_state": "deleted",
                    "training_excluded": True,
                }
            ],
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["family_size"] == 21
    assert payload["family_size_components"]["derived_research_family_size"] == 4
    assert payload["family_size_components"]["registry_strategy_hypothesis_count"] == 21
    assert payload["experiment_lineage"]["complete"] is True


def test_multiple_testing_guard_uses_aligned_candidate_forward_periods(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "replay_feature_ablation_latest.json",
        {
            "ok": True,
            "ablation": {
                "baseline": {"macro_f1": 0.55},
                "without_macro_context": {"macro_f1": 0.53},
            },
            "strict_checks": {"require_full_dim_match": True},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "counterfactual_replay_latest.json",
        {
            "ok": True,
            "candidate_count": 11,
            "profiles_reviewed": ["default", "dividend"],
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 2},
    )
    days = [f"202608{day:02d}" for day in range(1, 9)]
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_id": "pc-test-g3",
                "candidate_generation": 3,
                "candidate_cutoff_utc": "2026-08-01T00:00:00+00:00",
                "evidence_through_utc": "2026-08-09T00:00:00+00:00",
                "candidate_filter_active": True,
                "candidate_binding_required": True,
                "candidate_binding_mismatch_rows_excluded": 0,
            },
            "sleeve_latest": [
                {
                    "profile": "default",
                    "post_cost_expectancy": {
                        "robust_statistics": {
                            "one_sided_positive_expectancy_p_value": 0.01,
                            "deflated_sharpe": {
                                "available": True,
                                "probability": 0.97,
                            },
                        }
                    },
                },
                {
                    "profile": "dividend",
                    "post_cost_expectancy": {
                        "robust_statistics": {
                            "one_sided_positive_expectancy_p_value": 0.02,
                            "deflated_sharpe": {
                                "available": True,
                                "probability": 0.96,
                            },
                        }
                    },
                },
            ],
            "candidate_post_cost_daily_series": {
                "default": [
                    {
                        "day_utc": day,
                        "post_cost_return_bps_total": 2.0 + index,
                    }
                    for index, day in enumerate(days)
                ],
                "dividend": [
                    {
                        "day_utc": day,
                        "post_cost_return_bps_total": 1.0 + index,
                    }
                    for index, day in enumerate(days)
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["candidate_binding"]["bound"] is True
    assert payload["candidate_binding"]["candidate_id"] == "pc-test-g3"
    assert payload["candidate_binding"]["pbo_series_scope"] == (
        "candidate_forward_profile_daily_post_cost_returns"
    )
    assert payload["candidate_binding"]["pbo_common_period_days"] == days
    assert payload["probability_of_backtest_overfitting"]["available"] is True
    assert payload["probability_of_backtest_overfitting"]["period_count"] == 8


def test_multiple_testing_guard_counts_compressed_immutable_experiment_history(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "replay_feature_ablation_latest.json",
        {
            "ok": True,
            "ablation": {"baseline": {}, "without_feature": {}},
            "strict_checks": {"require_full_dim_match": True},
            "failed_checks": [],
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "counterfactual_replay_latest.json",
        {"ok": True, "candidate_count": 1, "profiles_reviewed": ["default"]},
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 1},
    )
    ledger_path = (
        tmp_path
        / "governance"
        / "experiments"
        / "immutable_experiment_ledger.jsonl.gz"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ledger_path, "wt", encoding="utf-8") as handle:
        for index in range(12):
            handle.write(json.dumps({"experiment_id": f"exp-{index}"}) + "\n")

    payload = src.build_payload(tmp_path)

    assert payload["family_size"] == 12
    assert payload["family_size_components"]["experiment_ledger_hypothesis_count"] == 12
    assert payload["experiment_lineage"]["compressed_immutable_ledger_supported"] is True
    assert str(ledger_path) in payload["experiment_lineage"]["experiment_ledger_paths"]
