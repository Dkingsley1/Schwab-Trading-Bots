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
