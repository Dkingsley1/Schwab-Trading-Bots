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


def test_low_grade_audit_treats_live_transition_c_grade_as_promotion_evidence(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "live_transition_integrity_control_latest.json", {"transition_readiness_grade": "C"})

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    assert audit["active_blocker_count"] == 0
    assert audit["promotion_evidence_layer_count"] == 1
    assert audit["layers"][0]["control_state"] == "live_promotion_evidence_debt"


def test_low_grade_audit_does_not_turn_operationally_controlled_raw_pnl_into_runtime_blockers(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "ok": True,
            "low_grade_layer_summary": {"control_posture_grade": "B", "active_blocker_count": 1},
            "profit_harvest_report_card": {
                "raw_outcome_grade": "D",
                "control_grade": "D",
                "base_raw_outcome_grade": "D",
            },
            "weak_sleeve_a_plus_plus_strengthening_contract": {
                "profile_controls": [{"profile": "aggressive", "raw_profit_grade": "F"}],
            },
        },
    )

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    assert audit["active_blocker_count"] == 0
    assert {row["scope"] for row in audit["layers"]} == {"paper_outcome_evidence"}
    assert {row["control_state"] for row in audit["layers"]} == {
        "raw_paper_outcome_under_operational_control"
    }


def test_low_grade_audit_treats_aggregate_and_elapsed_evidence_as_non_operational(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "readiness_blocker_rollup_latest.json",
        {"root_causes": [{"evidence": [{"pillar_grade": "F"}]}]},
    )
    _write_json(
        health / "uniform_hardening_contract_latest.json",
        {"all_domain_evidence_grade": "F", "critical_runtime_grade": "A+"},
    )
    _write_json(
        health / "continuous_soak_integrity_control_latest.json",
        {"elapsed_evidence_grade": "F", "operational_capacity_grade": "A+"},
    )

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    assert audit["active_blocker_count"] == 0
    assert audit["contained_or_controlled_count"] == 3


def test_low_grade_audit_includes_c_grades(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "runtime_component_latest.json", {"operational_grade": "C"})

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    assert audit["active_blocker_count"] == 1
    assert audit["layers"][0]["current_grade"] == "C"


def test_low_grade_audit_skips_owned_alias_even_when_alias_metadata_differs(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "distributed_cell_architecture_latest.json",
        {"timestamp_utc": "new", "operational_grade": "F"},
    )
    _write_json(
        health / "system_cell_federation_latest.json",
        {"timestamp_utc": "old", "operational_grade": "F", "alias": True},
    )

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    assert audit["duplicate_alias_file_count"] == 1
    assert audit["unique_low_grade_layer_count"] == 1
    assert audit["layers"][0]["exact_file"].endswith("distributed_cell_architecture_latest.json")


def test_low_grade_audit_classifies_sleeve_dashboard_as_controlled_paper_outcome(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_profitability_control_latest.json",
        {"ok": True, "low_grade_layer_summary": {"control_posture_grade": "A+", "active_blocker_count": 0}},
    )
    _write_json(
        health / "sleeve_profitability_dashboard_latest.json",
        {"overall_status": "ready", "profiles": [{"profile": "example", "raw_profit_grade": "D"}]},
    )

    audit = system_needs_intelligence._low_grade_layer_audit(tmp_path)

    sleeve = next(row for row in audit["layers"] if "sleeve_profitability_dashboard" in row["exact_file"])
    assert sleeve["scope"] == "paper_outcome_evidence"
    assert sleeve["control_state"] == "raw_paper_outcome_under_operational_control"
    assert sleeve["active_blocker"] is False
