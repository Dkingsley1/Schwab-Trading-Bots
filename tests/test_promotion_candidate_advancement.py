import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import promotion_candidate_advancement as advancement


NOW = datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)


def _seed_candidates(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, samples in enumerate((746, 57, 312, 1122, 1), start=1):
        model = project_root / "models" / f"bot_{index}.npz"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("model", encoding="utf-8")
        rows.append(
            {
                "bot_id": f"bot_{index}",
                "bot_role": "signal_sub_bot",
                "lifecycle_state": "active",
                "quality_score": 0.9 - index / 100.0,
                "test_accuracy": 0.8,
                "diagnostic_age_hours": 12,
                "sample_count": samples,
                "walk_forward_runs": 0,
                "model_path": str(model),
            }
        )
    (health / "training_requalification_latest.json").write_text(
        json.dumps({"top_reactivation_ready": rows}),
        encoding="utf-8",
    )


def test_top_candidates_are_split_between_training_and_data_first(tmp_path: Path, monkeypatch) -> None:
    _seed_candidates(tmp_path)
    monkeypatch.setattr(
        advancement.training_runtime_control,
        "build_payload",
        lambda *_args, **_kwargs: {
            "overall_status": "ready",
            "training_launch_contract": {
                "launch_allowed": True,
                "launch_blockers": [],
                "recommended_batch_size": 3,
                "recommended_retrain_profile": "coverage_small_canary",
                "canary_batch": [{"bot_id": "bot_1"}, {"bot_id": "bot_3"}, {"bot_id": "bot_4"}],
            },
        },
    )

    payload = advancement.build_payload(tmp_path, limit=5, now=NOW)

    assert payload["staged_candidate_count"] == 5
    assert payload["training_ready_bot_ids"] == ["bot_1", "bot_3", "bot_4"]
    assert payload["data_or_repair_first_bot_ids"] == ["bot_2", "bot_5"]
    assert payload["runtime_approved_bot_ids"] == ["bot_1", "bot_3", "bot_4"]
    assert "--skip-master-update" in payload["recommended_command"]
    assert payload["execution"]["status"] == "publish_only"


def test_runtime_gate_prevents_execution(tmp_path: Path, monkeypatch) -> None:
    _seed_candidates(tmp_path)
    monkeypatch.setattr(
        advancement.training_runtime_control,
        "build_payload",
        lambda *_args, **_kwargs: {
            "overall_status": "blocked",
            "training_launch_contract": {
                "launch_allowed": False,
                "launch_blockers": ["resource_guard_not_green"],
                "recommended_batch_size": 0,
                "canary_batch": [],
            },
        },
    )

    payload = advancement.build_payload(tmp_path, limit=5, execute=True, now=NOW)

    assert payload["runtime_approved_bot_ids"] == []
    assert payload["execution"]["attempted"] is False
    assert payload["execution"]["status"] == "not_runtime_approved"
