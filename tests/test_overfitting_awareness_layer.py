import json
from pathlib import Path

from scripts.ops import overfitting_awareness_layer as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_overfitting_awareness_blocks_leaky_teachers_and_broadcasts_to_all_tiers(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10_clean",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "test_accuracy": 0.61,
                    "quality_score": 0.82,
                },
                {
                    "bot_id": "brain_refinery_v20_leaky",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "test_accuracy": 0.97,
                    "quality_score": 0.91,
                },
                {
                    "bot_id": "brain_refinery_v200_sleeve_master_bot",
                    "bot_role": "infrastructure_sub_bot",
                    "slot_kind": "test_sleeve_master",
                    "target_functions": ["sleeve_master"],
                    "active": True,
                    "test_accuracy": 0.95,
                    "quality_score": 0.88,
                },
            ]
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "walk_forward_latest.json",
        {
            "bots": {
                "brain_refinery_v10_clean": {
                    "runs": 12,
                    "train_mean": 0.62,
                    "forward_mean": 0.58,
                    "delta": 0.01,
                    "status": "pass",
                },
                "brain_refinery_v20_leaky": {
                    "runs": 14,
                    "train_mean": 0.97,
                    "forward_mean": 0.50,
                    "delta": -0.10,
                    "status": "fail",
                },
                "brain_refinery_v200_sleeve_master_bot": {
                    "runs": 4,
                    "train_mean": 0.95,
                    "forward_mean": 0.92,
                    "delta": 0.01,
                    "status": "pass",
                },
            }
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "leak_overfit_guard_latest.json",
        {
            "thresholds": {
                "max_overfit_gap": 0.08,
                "max_severe_overfit_gap": 0.14,
                "high_train_threshold": 0.90,
            },
            "leak_like_examples": [{"bot_id": "brain_refinery_v20_leaky"}],
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_runtime_profitability_controls_latest.json",
        {
            "sub_bot_accuracy_target_contract": {
                "active": True,
                "desired_out_of_sample_accuracy_band": {"min": 0.80, "max": 0.90},
                "min_walk_forward_runs": 12,
                "max_train_test_accuracy_gap": 0.08,
            }
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["risk_bot_count"] == 1
    assert payload["hard_risk_bot_count"] == 1
    assert payload["blocked_teacher_bot_count"] == 1
    assert "brain_refinery_v20_leaky" in payload["blocked_teacher_bot_ids"]
    assert payload["broadcast_contract"]["applies_to_tiers"] == [
        "infrastructure",
        "sub",
        "teacher",
        "master",
        "grand_master",
    ]
    risks = {row["bot_id"]: row for row in payload["bot_risk"]}
    assert risks["brain_refinery_v10_clean"]["status"] == "generalization_clean"
    assert risks["brain_refinery_v20_leaky"]["policy"]["may_teach"] is False
