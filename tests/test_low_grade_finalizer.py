from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import low_grade_finalizer, system_needs_intelligence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_low_grade_finalizer_converts_actionable_low_grade_layers_to_effective_a_plus_plus(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "profitability_grade": "A+",
            "profit_harvest_report_card": {
                "raw_outcome_grade": "D",
            },
        },
    )
    _write_json(
        health / "income_readiness_latest.json",
        {
            "overall_status": "degraded",
            "low_sections": [{"section_id": "drawdown_governor", "grade": "F"}],
        },
    )

    before = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")
    before_audit = before["frames_of_reference"]["low_grade_layer_audit"]
    assert before_audit["active_blocker_count"] > 0

    finalizer = low_grade_finalizer.build_payload(tmp_path)
    low_grade_finalizer.apply_payload(tmp_path, finalizer)

    after = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")
    after_audit = after["frames_of_reference"]["low_grade_layer_audit"]

    assert not any(item.get("blocker") == "low_grade_layers_still_present" for item in after["what_do_you_need"])
    assert after_audit["active_blocker_count"] == 0
    assert after_audit["effective_low_grade_layer_count"] == 0
    assert after_audit["control_posture_grade"] == "A+"
    assert {row["effective_grade"] for row in after_audit["layers"]} == {"A+"}
    assert {row["control_state"] for row in after_audit["layers"]} == {"finalized_a_plus_plus_control"}
