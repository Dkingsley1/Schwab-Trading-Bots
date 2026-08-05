from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import low_grade_finalizer, system_needs_intelligence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_low_grade_finalizer_preserves_actionable_low_grade_evidence(tmp_path: Path) -> None:
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

    assert any(item.get("blocker") == "low_grade_layers_still_present" for item in after["what_do_you_need"])
    assert after_audit["active_blocker_count"] == before_audit["active_blocker_count"]
    assert after_audit["effective_low_grade_layer_count"] > 0
    assert after_audit["control_posture_grade"] != "A+"
    assert {row["effective_grade"] for row in after_audit["layers"]} <= {"D", "F"}
    assert "finalized_a_plus_plus_control" not in {row["control_state"] for row in after_audit["layers"]}

    written = json.loads((health / "low_grade_finalizer_latest.json").read_text(encoding="utf-8"))
    assert written["finalization_contract"]["mode"] == "truthful_low_grade_classification_v2"
    assert written["finalization_contract"]["cosmetic_grade_uplift_allowed"] is False
    assert written["active_blocker_count_after_finalization"] > 0


def test_low_grade_audit_separates_live_promotion_debt_from_soak_blockers(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "production_excellence_control_latest.json",
        {"overall_grade": "F", "pillars": [{"grade": "D"}]},
    )

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    assert audit["active_blocker_count"] == 0
    assert audit["promotion_evidence_layer_count"] == 2
    assert {row["control_state"] for row in audit["layers"]} == {"live_promotion_evidence_debt"}
    assert {row["effective_grade"] for row in audit["layers"]} == {"D", "F"}
