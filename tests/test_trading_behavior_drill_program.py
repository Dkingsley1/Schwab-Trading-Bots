from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.ops import paper_behavior_intervention_drill
from scripts.ops import trading_behavior_drill_program as program

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = "2026-08-26T16:00:00+00:00"


def test_real_full_program_is_a_plus_and_admission_eligible() -> None:
    payload = program.build_payload(
        project_root=PROJECT_ROOT,
        generated_at_utc=FIXED_TIME,
    )

    assert payload["ok"] is True
    assert payload["control_grade"] == "A+"
    assert payload["admission_eligible"] is True
    assert len(payload["suite_results"]) == 3
    assert all(row["control_grade"] == "A+" for row in payload["suite_results"])
    assert payload["candidate_mutation_guard"]["unchanged"] is True
    assert payload["resource_contract"]["broker_requests"] == 0
    assert payload["resource_contract"]["live_orders_submitted"] == 0
    assert payload["behavior_change_contract"]["live_execution_allowed"] is False


def test_same_input_complete_run_is_compared_without_regression() -> None:
    first = program.build_payload(
        project_root=PROJECT_ROOT,
        generated_at_utc=FIXED_TIME,
    )
    second = program.build_payload(
        project_root=PROJECT_ROOT,
        generated_at_utc=FIXED_TIME,
        previous_payload=first,
    )

    assert second["comparison"]["comparable"] is True
    assert second["comparison"]["status"] == "non_regressed"
    assert second["comparison"]["regressions"] == []
    assert second["admission_eligible"] is True


def test_historical_intervention_drill_uses_its_evaluation_clock() -> None:
    payload = paper_behavior_intervention_drill.build_payload(
        project_root=PROJECT_ROOT,
        generated_at_utc="2020-01-02T16:00:00+00:00",
    )

    assert payload["ok"] is True
    assert payload["checks"]["all_selected_scenarios_pass"] is True
    assert all(
        "paper_behavior_intervention_overlay_disabled" not in case["reasons"]
        for scenario in payload["scenario_results"]
        for case in scenario["cases"]
    )


def test_partial_program_can_never_be_admission_eligible() -> None:
    payload = program.build_payload(
        project_root=PROJECT_ROOT,
        requested_suites=["profitability_crisis"],
        generated_at_utc=FIXED_TIME,
    )

    assert len(payload["suite_results"]) == 1
    assert payload["checks"]["full_required_suite_set_executed"] is False
    assert payload["admission_eligible"] is False
    assert (
        payload["behavior_change_contract"]["decision"]
        == "reject_without_runtime_change"
    )


def _materialize_root(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    runtime = tmp_path / "governance" / "runtime"
    config.mkdir(parents=True)
    runtime.mkdir(parents=True)
    for name in (
        "trading_behavior_drill_program_v1.json",
        "profitability_crisis_drill_v1.json",
        "profitability_adversarial_drill_v1.json",
        "paper_behavior_intervention_drill_v1.json",
    ):
        shutil.copy2(PROJECT_ROOT / "config" / name, config / name)
    (runtime / "production_candidate_state.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-g9",
                "generation": 9,
                "accepted_at_utc": "2026-08-26T12:00:00+00:00",
                "accepted_git_head": "abc123",
                "live_execution_authority": False,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _candidate_binding(root: Path) -> dict:
    path = root / "governance" / "runtime" / "production_candidate_state.json"
    return {
        "candidate_id": "candidate-g9",
        "candidate_generation": 9,
        "candidate_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "valid": True,
    }


def _suite_payload(root: Path, suite_id: str, *, score: float = 100.0) -> dict:
    ok = score >= 99.5
    base = {
        "timestamp_utc": FIXED_TIME,
        "status": "ready" if ok else "degraded",
        "ok": ok,
        "control_grade": "A+" if ok else "B",
        "control_score": score,
        "candidate_binding": _candidate_binding(root),
        "checks": {"synthetic_test_check": ok},
        "failed_checks": [] if ok else ["synthetic_test_check"],
        "authority_contract": {
            "network_access": False,
            "broker_access": False,
            "paper_order_authority": False,
            "live_order_authority": False,
            "automatic_promotion_authority": False,
            "candidate_mutation_authority": False,
        },
        "resource_contract": {
            "persistent_processes_started": 0,
            "network_requests": 0,
            "broker_requests": 0,
            "paper_orders_submitted": 0,
            "live_orders_submitted": 0,
            "candidate_mutations": 0,
            "runtime_control_writes": 0,
            "work_units": 1,
        },
    }
    if suite_id == "profitability_crisis":
        base["scenario_results"] = [
            {"scenario_id": f"crisis-{index}", "ok": ok} for index in range(3)
        ]
        base["diagnostic_summary"] = {"phase_count": 15}
    elif suite_id == "profitability_adversarial":
        base["diagnostic_summary"] = {
            "scenario_count": 14,
            "passed_scenario_count": 14 if ok else 13,
            "scenario_check_count": 78,
        }
    else:
        base["scenario_summary"] = {
            "executed_scenario_count": 14,
            "passed_scenario_count": 14 if ok else 13,
            "case_count": 17,
        }
        base["admission_eligible"] = ok
        base["runtime_overlay_proposal"] = {
            "paper_only": True,
            "live_execution_allowed": False,
        }
        base["runtime_overlay_proposal_sha256"] = "b" * 64
        base["authority_contract"] = {
            "can_observe": True,
            "can_propose_paper_behavior_overlay": True,
            "can_write_runtime_control": False,
            "can_submit_paper_orders": False,
            "can_submit_live_orders": False,
            "can_mutate_candidate": False,
            "can_promote": False,
            "can_change_live_execution": False,
        }
    return base


def _builders(root: Path, *, adversarial_score: float = 100.0):
    return {
        "profitability_crisis": lambda **_: _suite_payload(
            root, "profitability_crisis"
        ),
        "profitability_adversarial": lambda **_: _suite_payload(
            root, "profitability_adversarial", score=adversarial_score
        ),
        "paper_behavior_intervention": lambda **_: _suite_payload(
            root, "paper_behavior_intervention"
        ),
    }


def test_regressed_comparable_suite_rejects_behavior_change(tmp_path: Path) -> None:
    root = _materialize_root(tmp_path)
    baseline = program.build_payload(
        project_root=root,
        generated_at_utc=FIXED_TIME,
        suite_builders=_builders(root),
    )
    regressed = program.build_payload(
        project_root=root,
        generated_at_utc=FIXED_TIME,
        previous_payload=baseline,
        suite_builders=_builders(root, adversarial_score=80.0),
    )

    assert baseline["admission_eligible"] is True
    assert regressed["comparison"]["status"] == "regressed"
    assert any(
        row["reason"] == "control_score_regressed"
        for row in regressed["comparison"]["regressions"]
    )
    assert regressed["admission_eligible"] is False


def test_candidate_mutation_stops_remaining_suites(tmp_path: Path) -> None:
    root = _materialize_root(tmp_path)
    candidate_path = root / "governance" / "runtime" / "production_candidate_state.json"

    def mutate_candidate(**_) -> dict:
        payload = _suite_payload(root, "profitability_crisis")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["generation"] = 10
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        return payload

    builders = _builders(root)
    builders["profitability_crisis"] = mutate_candidate
    payload = program.build_payload(
        project_root=root,
        generated_at_utc=FIXED_TIME,
        suite_builders=builders,
    )

    assert len(payload["suite_results"]) == 1
    assert payload["candidate_mutation_guard"]["mutation_detected_mid_run"] is True
    assert payload["checks"]["candidate_state_unchanged"] is False
    assert payload["admission_eligible"] is False


def test_publication_retains_compact_history_without_suite_duplication(
    tmp_path: Path,
) -> None:
    root = _materialize_root(tmp_path)
    payload = program.execute_and_publish(
        project_root=root,
        generated_at_utc=FIXED_TIME,
        suite_builders=_builders(root),
    )

    assert payload["ok"] is True
    assert payload["artifact_publication"]["history_written"] is True
    history_path = Path(payload["artifact_publication"]["history_path"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["run_id"] == payload["run_id"]
    assert "scenario_results" not in history
    assert payload["artifact_publication"]["history_record_bytes"] < 262144
    assert (
        root / "governance" / "research" / "trading_behavior_drill_program_latest.json"
    ).exists()
