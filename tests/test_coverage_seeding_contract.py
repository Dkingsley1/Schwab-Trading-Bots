import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import coverage_gap_closer, walk_forward_coverage_seed


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_walk_forward_coverage_seed_marks_strong_candidates(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 0, "coverage_shortfall_bots": 4, "thresholds": {"min_considered_bots": 4, "min_runs_per_bot": 12}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "training_requalification_latest.json",
        {
            "top_candidates": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "quality_score": 0.86,
                    "test_accuracy": 0.81,
                    "walk_forward_runs": 0,
                    "priority": 97.0,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs"],
                },
                {
                    "bot_id": "brain_refinery_v101_guard_heavy_regime_memory",
                    "bot_role": "signal_sub_bot",
                    "quality_score": 0.0,
                    "test_accuracy": 0.0,
                    "walk_forward_runs": 0,
                    "priority": 10.0,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs"],
                },
            ]
        },
    )
    _write_json(tmp_path / "governance" / "walk_forward" / "walk_forward_latest.json", {"bots": {}})

    payload = walk_forward_coverage_seed.build_payload(tmp_path, limit=4)
    rows = payload["seed_queue"]

    assert rows[0]["bot_id"] == "brain_refinery_v35_dmi_state_machine"
    assert rows[0]["strong_seed_candidate"] is True
    assert rows[0]["priority"] > rows[0]["base_priority"]
    assert rows[1]["strong_seed_candidate"] is False


def test_walk_forward_coverage_seed_separates_diagnostic_refresh_from_runtime_repair(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"considered_bots": 3, "coverage_shortfall_bots": 1, "thresholds": {"min_considered_bots": 4, "min_runs_per_bot": 12}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "training_requalification_latest.json",
        {
            "top_candidates": [
                {
                    "bot_id": "brain_refinery_v1",
                    "bot_role": "signal_sub_bot",
                    "quality_score": 0.0,
                    "test_accuracy": 0.0,
                    "walk_forward_runs": 0,
                    "priority": 30.0,
                    "actions": ["refresh_training_diagnostics"],
                }
            ]
        },
    )
    _write_json(tmp_path / "governance" / "walk_forward" / "walk_forward_latest.json", {"bots": {}})

    payload = walk_forward_coverage_seed.build_payload(tmp_path, limit=4)
    row = payload["seed_queue"][0]

    assert row["needs_runtime_input_repair"] is False
    assert row["needs_diagnostic_refresh"] is True
    assert payload["standing_queue"]["repair_before_seed_count"] == 0
    assert payload["standing_queue"]["diagnostic_refresh_before_seed_count"] == 1


def test_coverage_gap_closer_stages_strong_infra_before_weak_bootstrap(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "queue_bucket": "signal",
                    "priority": 109.0,
                    "current_runs": 0,
                    "needs_runtime_input_repair": True,
                    "strong_seed_candidate": True,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs"],
                },
                {
                    "bot_id": "brain_refinery_v56_meta_ranker",
                    "bot_role": "infrastructure_sub_bot",
                    "queue_bucket": "infrastructure",
                    "priority": 74.0,
                    "current_runs": 0,
                    "needs_runtime_input_repair": True,
                    "strong_seed_candidate": True,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs"],
                },
                {
                    "bot_id": "brain_refinery_v101_guard_heavy_regime_memory",
                    "bot_role": "signal_sub_bot",
                    "queue_bucket": "signal",
                    "priority": 10.0,
                    "current_runs": 0,
                    "needs_runtime_input_repair": True,
                    "strong_seed_candidate": False,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs"],
                },
            ]
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"thresholds": {"min_considered_bots": 4}},
    )
    _write_json(tmp_path / "governance" / "walk_forward" / "walk_forward_latest.json", {"bots": {}})

    payload = coverage_gap_closer._candidate_pool(tmp_path, candidate_limit=4, stage_count=2)

    staged_ids = [row["bot_id"] for row in payload["active_stage"]]
    backup_ids = [row["bot_id"] for row in payload["backup_candidates"]]

    assert staged_ids == ["brain_refinery_v35_dmi_state_machine", "brain_refinery_v56_meta_ranker"]
    assert "brain_refinery_v101_guard_heavy_regime_memory" in backup_ids
