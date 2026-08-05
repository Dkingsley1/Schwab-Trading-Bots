import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import section_grade_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_section_grade_guard_makes_training_debt_advisory_when_guarded_paper_is_ready(tmp_path: Path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready"},
                "live_execution": {
                    "ok": False,
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
            },
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "guarded_live_read_only"}},
    )

    def section(state: str, score: float = 96.0) -> dict:
        return {
            "floor_state": state,
            "letter_grade": "A+",
            "raw_letter_grade": "A+" if state != "below_floor" else "B+",
            "score": score,
            "raw_score": score if state != "below_floor" else 84.0,
            "target_floor_letter_grade": "A",
            "target_floor_score": 92.0,
            "floor_contract_active": state != "below_floor",
            "floor_reason": "",
            "signals": {},
        }

    snapshot = {
        "overall_score": 97.74,
        "overall_letter_grade": "A+",
        "raw_overall_score": 95.18,
        "raw_overall_letter_grade": "A",
        "section_grades": {
            slug: section("below_floor" if slug == "training_and_model_quality" else "at_floor")
            for slug in src.SECTION_COMMANDS
        },
    }

    class DummyConnector:
        exposed_endpoints = [object()] * 16

    monkeypatch.setattr(src, "DefaultLicensingAPIConnector", lambda: DummyConnector())
    monkeypatch.setattr(src, "build_grade_snapshot", lambda **_: snapshot)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["below_floor_sections"] == ["training_and_model_quality"]
    assert payload["blocking_below_floor_sections"] == []
    assert payload["advisory_below_floor_sections"] == ["training_and_model_quality"]
    assert payload["paper_soak_advisory_below_floor"] is True


def test_section_grade_guard_makes_current_guarded_paper_floor_debt_advisory(tmp_path: Path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {
                    "ok": False,
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
            },
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}},
    )

    below_floor_sections = {"live_trading_readiness", "training_and_model_quality", "ops_and_autonomy"}

    def section(slug: str) -> dict:
        below = slug in below_floor_sections
        return {
            "floor_state": "below_floor" if below else "at_floor",
            "letter_grade": "B+" if below else "A+",
            "raw_letter_grade": "B+" if below else "A+",
            "score": 86.0 if below else 96.0,
            "raw_score": 86.0 if below else 96.0,
            "target_floor_letter_grade": "A",
            "target_floor_score": 92.0,
            "floor_contract_active": False,
            "floor_reason": "",
            "signals": {},
        }

    snapshot = {
        "overall_score": 94.05,
        "overall_letter_grade": "A",
        "raw_overall_score": 94.05,
        "raw_overall_letter_grade": "A",
        "section_grades": {slug: section(slug) for slug in src.SECTION_COMMANDS},
    }

    class DummyConnector:
        exposed_endpoints = [object()] * 16

    monkeypatch.setattr(src, "DefaultLicensingAPIConnector", lambda: DummyConnector())
    monkeypatch.setattr(src, "build_grade_snapshot", lambda **_: snapshot)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["blocking_below_floor_sections"] == []
    assert set(payload["advisory_below_floor_sections"]) == below_floor_sections
    assert payload["guarded_paper_strict_clear"] is True


def test_section_grade_guard_accepts_managed_coverage_deferred_as_live_locked(tmp_path: Path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "pending_release", "blockers": []},
            },
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "managed_coverage_stage_deferred"}},
    )

    def section(slug: str) -> dict:
        below = slug in {"live_trading_readiness", "ops_and_autonomy"}
        return {
            "floor_state": "below_floor" if below else "at_floor",
            "letter_grade": "B+" if below else "A+",
            "raw_letter_grade": "B+" if below else "A+",
            "score": 86.0 if below else 96.0,
            "raw_score": 86.0 if below else 96.0,
            "target_floor_letter_grade": "A",
            "target_floor_score": 92.0,
            "floor_contract_active": False,
            "floor_reason": "",
            "signals": {},
        }

    class DummyConnector:
        exposed_endpoints = [object()] * 16

    monkeypatch.setattr(src, "DefaultLicensingAPIConnector", lambda: DummyConnector())
    monkeypatch.setattr(
        src,
        "build_grade_snapshot",
        lambda **_: {
            "overall_score": 96.5,
            "overall_letter_grade": "A+",
            "raw_overall_score": 95.0,
            "raw_overall_letter_grade": "A",
            "section_grades": {slug: section(slug) for slug in src.SECTION_COMMANDS},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["live_execution_locked"] is True
    assert set(payload["advisory_below_floor_sections"]) == {"live_trading_readiness", "ops_and_autonomy"}
    assert payload["blocking_below_floor_sections"] == []


def test_section_grade_guard_makes_bounded_storage_floor_debt_advisory_for_paper_soak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {
                    "ok": False,
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
            },
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "guarded_live_read_only"}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.293,
            "continuous_run_soak_contract": {"status": "watch", "soak_ready": True, "blockers": []},
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 4400,
                    "total_pending_lines": 11669,
                    "oldest_pending_age_seconds": 0.0,
                }
            },
        },
    )

    def section(slug: str) -> dict:
        below = slug == "data_ingestion_and_storage"
        return {
            "floor_state": "below_floor" if below else "at_floor",
            "letter_grade": "B" if below else "A+",
            "raw_letter_grade": "B" if below else "A+",
            "score": 83.5 if below else 96.0,
            "raw_score": 83.5 if below else 96.0,
            "target_floor_letter_grade": "A",
            "target_floor_score": 92.0,
            "floor_contract_active": False,
            "floor_reason": "",
            "signals": {},
        }

    class DummyConnector:
        exposed_endpoints = [object()] * 16

    monkeypatch.setattr(src, "DefaultLicensingAPIConnector", lambda: DummyConnector())
    monkeypatch.setattr(
        src,
        "build_grade_snapshot",
        lambda **_: {
            "overall_score": 94.5,
            "overall_letter_grade": "A",
            "raw_overall_score": 94.5,
            "raw_overall_letter_grade": "A",
            "section_grades": {slug: section(slug) for slug in src.SECTION_COMMANDS},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["below_floor_sections"] == ["data_ingestion_and_storage"]
    assert payload["blocking_below_floor_sections"] == []
    assert payload["advisory_below_floor_sections"] == ["data_ingestion_and_storage"]


def test_section_grade_guard_accepts_safe_transient_storage_drain_during_paper_soak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.697,
            "continuous_run_soak_contract": {
                "status": "blocked",
                "soak_ready": False,
                "blockers": ["steady_state_targets_not_clear"],
            },
            "bounded_recovery_contract": {
                "route_verified": True,
                "active_drain_progress": True,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
            "data_integrity": {},
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 2056,
                    "total_pending_lines": 3479,
                    "oldest_pending_age_seconds": 167.287,
                }
            },
        },
    )

    def section(slug: str) -> dict:
        below = slug == "data_ingestion_and_storage"
        return {
            "floor_state": "below_floor" if below else "at_floor",
            "letter_grade": "A-" if below else "A+",
            "raw_letter_grade": "A-" if below else "A+",
            "score": 89.43 if below else 96.0,
            "raw_score": 89.43 if below else 96.0,
            "target_floor_letter_grade": "A",
            "target_floor_score": 92.0,
            "floor_contract_active": False,
            "floor_reason": "",
            "signals": {},
        }

    class DummyConnector:
        exposed_endpoints = [object()] * 16

    monkeypatch.setattr(src, "DefaultLicensingAPIConnector", lambda: DummyConnector())
    monkeypatch.setattr(
        src,
        "build_grade_snapshot",
        lambda **_: {
            "overall_score": 97.0,
            "overall_letter_grade": "A+",
            "raw_overall_score": 97.0,
            "raw_overall_letter_grade": "A+",
            "section_grades": {slug: section(slug) for slug in src.SECTION_COMMANDS},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["blocking_below_floor_sections"] == []
    assert payload["advisory_below_floor_sections"] == ["data_ingestion_and_storage"]


def test_section_grade_guard_reports_degraded_when_sections_are_floor_protected(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "live_readiness_smoke_latest.json", {"overall_status": "blocked", "readiness_score": 68.0})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "degraded"})
    _write_json(health / "portable_brain_contract_latest.json", {"overall_status": "ready", "portability_score": 100.0, "host_contract": {"memory_architecture": "unified"}})
    _write_json(
        health / "training_quality_control_latest.json",
        {"overall_status": "blocked", "training_quality_score": 68.88, "rollout": {"considered_gap": 4}},
    )
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "blocked", "lineage_score": 87.5})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "degraded", "committee_packet_seed_ready": True})
    _write_json(health / "retrain_launch_latest.json", {"state": "completed", "final_status": "ok"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recovery_quality_score": 91.0,
            "backpressure_quality_score": 84.0,
            "pressure_index": 5.0,
            "recovery_state": "blocked_backpressure",
        },
    )
    _write_json(
        health / "cost_telemetry_latest.json",
        {
            "overall_status": "ready",
            "storage_cost_proxy": {"tracked_sqlite_gb": 220.1},
            "portable_backend_cost_proxy": {"proof_present_count": 3},
        },
    )
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "summary": {"passed_checks": 20, "failed_checks": 0}})
    _write_json(health / "security_evidence_autofix_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "incident_closeout_autopilot_latest.json",
        {"overall_status": "blocked", "closeout_score": 54.0, "open_incident_count": 1},
    )
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "blocked",
            "open_incident_count": 1,
            "open_surfaces": [{"surface": "live_readiness"}],
        },
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "broker_state": {"auth_ok": True, "configured_for_refresh": True},
        },
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {"overall_status": "ready", "critical_backlog": {"unacked_count": 0, "unsent_count": 0}},
    )
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "blocked", "autonomy_score": 70.83})
    _write_json(health / "runtime_artifact_refresh_latest.json", {"overall_status": "ready", "required_missing_after": []})
    _write_json(health / "chrome_headless_guard_latest.json", {"overall_status": "degraded"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {
            "overall_status": "applied",
            "metrics": {
                "attempted_step_count": 2,
                "cycle_count": 1,
                "backpressure_actionable": True,
                "coordinator_actionable": True,
            },
        },
    )
    _write_json(
        health / "platform_control_plane_latest.json",
        {
            "institutional_readiness": {"overall_score": 91.85},
            "institutional_domains_by_slug": {
                "developer_process": {"score": 100.0},
                "formal_model_governance": {"score": 89.0},
                "high_fidelity_simulator": {"score": 100.0},
                "immutable_experiment_tracking": {"score": 68.0},
                "independent_risk_services": {"score": 88.0},
                "observability_and_slo": {"score": 94.0},
                "point_in_time_data_lineage": {"score": 100.0},
                "portfolio_construction": {"score": 84.0},
                "reliability_engineering": {"score": 100.0},
                "security_and_compliance": {"score": 100.0},
                "statistical_research_discipline": {"score": 95.25},
                "transaction_cost_and_capacity": {"score": 84.0},
            },
        },
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "preclearance_score": 95.0,
            "staged_preclearance_ready": True,
            "preapproved_supervised_ready": True,
            "supervised_canary_ready": False,
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["below_floor_count"] == 0
    assert payload["protected_by_floor_count"] >= 1
    training_row = next(row for row in payload["sections"] if row["section"] == "training_and_model_quality")
    assert training_row["state"] == "protected_by_floor"
    assert training_row["letter_grade"].startswith("A")
    assert training_row["raw_letter_grade"].startswith("B")


def test_section_grade_guard_uses_bounded_training_and_ops_contracts(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    walk = tmp_path / "governance" / "walk_forward"
    _write_json(health / "live_readiness_smoke_latest.json", {"overall_status": "blocked", "readiness_score": 13.0})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "degraded"})
    _write_json(
        health / "portable_brain_contract_latest.json",
        {"overall_status": "ready", "portability_score": 100.0, "host_contract": {"memory_architecture": "unified"}},
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 66.38,
            "rollout": {"considered_gap": 4},
            "immutable_lineage": {
                "provisional_lineage_ready": True,
                "stronger_provisional_lineage_ready": False,
                "replay_hash_guard_ok": True,
            },
        },
    )
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "blocked", "lineage_score": 82.5})
    _write_json(
        walk / "coverage_gap_closer_latest.json",
        {
            "overall_status": "waiting_for_idle",
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "launch_state": "waiting_for_idle",
                "stage_candidate_count": 4,
                "can_apply_stage": True,
            },
        },
    )
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "degraded", "committee_packet_seed_ready": True})
    _write_json(health / "retrain_launch_latest.json", {"state": "running", "final_status": "running"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recovery_quality_score": 91.0,
            "backpressure_quality_score": 72.0,
            "pressure_index": 5.57,
            "recovery_state": "recovering_under_guard",
        },
    )
    _write_json(
        health / "cost_telemetry_latest.json",
        {
            "overall_status": "ready",
            "storage_cost_proxy": {"tracked_sqlite_gb": 220.1},
            "portable_backend_cost_proxy": {"proof_present_count": 3},
        },
    )
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "summary": {"passed_checks": 20, "failed_checks": 0}})
    _write_json(health / "security_evidence_autofix_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "incident_closeout_autopilot_latest.json",
        {"overall_status": "blocked", "closeout_score": 54.0, "open_incident_count": 3},
    )
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "blocked",
            "open_incident_count": 1,
            "open_surfaces": [{"surface": "live_readiness"}],
        },
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "broker_state": {"auth_ok": True, "configured_for_refresh": True},
        },
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {"overall_status": "ready", "critical_backlog": {"unacked_count": 0, "unsent_count": 0}},
    )
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "blocked", "autonomy_score": 60.9})
    _write_json(health / "runtime_artifact_refresh_latest.json", {"overall_status": "ready", "required_missing_after": []})
    _write_json(
        health / "chrome_headless_guard_latest.json",
        {
            "overall_status": "blocked",
            "timeline_pdf_policy": "suppress",
            "interactive_protection_active": True,
            "timeline_autorender_suppressed": True,
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})
    _write_json(
        health / "platform_control_plane_latest.json",
        {
            "institutional_readiness": {"overall_score": 91.85},
            "institutional_domains_by_slug": {
                "developer_process": {"score": 100.0},
                "formal_model_governance": {"score": 89.0},
                "high_fidelity_simulator": {"score": 100.0},
                "immutable_experiment_tracking": {"score": 68.0},
                "independent_risk_services": {"score": 88.0},
                "observability_and_slo": {"score": 94.0},
                "point_in_time_data_lineage": {"score": 100.0},
                "portfolio_construction": {"score": 84.0},
                "reliability_engineering": {"score": 100.0},
                "security_and_compliance": {"score": 100.0},
                "statistical_research_discipline": {"score": 94.95},
                "transaction_cost_and_capacity": {"score": 84.0},
            },
        },
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "preclearance_score": 95.0,
            "staged_preclearance_ready": True,
            "preapproved_supervised_ready": True,
            "supervised_canary_ready": False,
        },
    )

    payload = src.build_payload(tmp_path)

    training_row = next(row for row in payload["sections"] if row["section"] == "training_and_model_quality")
    ops_row = next(row for row in payload["sections"] if row["section"] == "ops_and_autonomy")

    assert training_row["state"] == "protected_by_floor"
    assert training_row["letter_grade"] == "A"
    assert training_row["signals"]["bounded_coverage_contract_ready"] is True
    assert ops_row["state"] == "protected_by_floor"
    assert ops_row["letter_grade"] == "A"
    assert ops_row["signals"]["live_readiness_only_open"] is True
    assert ops_row["signals"]["grouped_alert_backlog_clear"] is True


def test_section_grade_guard_treats_storage_bound_watchdog_as_recoverable(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    walk = tmp_path / "governance" / "walk_forward"
    _write_json(health / "live_readiness_smoke_latest.json", {"overall_status": "blocked", "readiness_score": 68.0})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "degraded"})
    _write_json(health / "portable_brain_contract_latest.json", {"overall_status": "ready", "portability_score": 100.0, "host_contract": {"memory_architecture": "unified"}})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 66.38,
            "rollout": {"considered_gap": 4},
            "immutable_lineage": {
                "provisional_lineage_ready": True,
                "stronger_provisional_lineage_ready": False,
                "replay_hash_guard_ok": True,
            },
        },
    )
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "blocked", "lineage_score": 82.5})
    _write_json(
        walk / "coverage_gap_closer_latest.json",
        {
            "overall_status": "waiting_for_idle",
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "launch_state": "waiting_for_idle",
                "stage_candidate_count": 4,
                "can_apply_stage": True,
            },
        },
    )
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "degraded", "committee_packet_seed_ready": True})
    _write_json(health / "retrain_launch_latest.json", {"state": "running", "final_status": "running"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recovery_quality_score": 71.0,
            "backpressure_quality_score": 15.0,
            "pressure_index": 5.22,
            "recovery_state": "blocked_backpressure",
        },
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {
            "overall_status": "applied_with_followups",
            "metrics": {
                "attempted_step_count": 2,
                "cycle_count": 1,
                "backpressure_actionable": True,
                "coordinator_actionable": True,
            },
        },
    )
    _write_json(
        health / "cost_telemetry_latest.json",
        {
            "overall_status": "ready",
            "storage_cost_proxy": {"tracked_sqlite_gb": 220.1},
            "portable_backend_cost_proxy": {"proof_present_count": 3},
        },
    )
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "summary": {"passed_checks": 20, "failed_checks": 0}})
    _write_json(health / "security_evidence_autofix_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "incident_closeout_autopilot_latest.json",
        {"overall_status": "blocked", "closeout_score": 54.0, "open_incident_count": 2},
    )
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "blocked",
            "open_incident_count": 2,
            "open_surfaces": [{"surface": "live_readiness"}, {"surface": "process_watchdog"}],
        },
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "broker_state": {"auth_ok": True, "configured_for_refresh": True},
        },
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {"overall_status": "ready", "critical_backlog": {"unacked_count": 0, "unsent_count": 0}},
    )
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "blocked", "autonomy_score": 61.53})
    _write_json(health / "runtime_artifact_refresh_latest.json", {"overall_status": "ready", "required_missing_after": []})
    _write_json(
        health / "chrome_headless_guard_latest.json",
        {
            "overall_status": "degraded",
            "timeline_pdf_policy": "headless_only",
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [{"name": "execution_lane_paper"}],
            "alerts": [{"name": "execution_lane_paper"}],
        },
    )
    _write_json(
        health / "platform_control_plane_latest.json",
        {
            "institutional_readiness": {"overall_score": 91.85},
            "institutional_domains_by_slug": {
                "developer_process": {"score": 100.0},
                "formal_model_governance": {"score": 89.0},
                "high_fidelity_simulator": {"score": 100.0},
                "immutable_experiment_tracking": {"score": 68.0},
                "independent_risk_services": {"score": 88.0},
                "observability_and_slo": {"score": 94.0},
                "point_in_time_data_lineage": {"score": 100.0},
                "portfolio_construction": {"score": 84.0},
                "reliability_engineering": {"score": 100.0},
                "security_and_compliance": {"score": 100.0},
                "statistical_research_discipline": {"score": 94.95},
                "transaction_cost_and_capacity": {"score": 84.0},
            },
        },
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "preclearance_score": 95.0,
            "staged_preclearance_ready": True,
            "preapproved_supervised_ready": True,
            "supervised_canary_ready": False,
        },
    )

    payload = src.build_payload(tmp_path)

    data_row = next(row for row in payload["sections"] if row["section"] == "data_ingestion_and_storage")
    ops_row = next(row for row in payload["sections"] if row["section"] == "ops_and_autonomy")

    assert data_row["state"] == "protected_by_floor"
    assert data_row["letter_grade"] == "A"
    assert data_row["signals"]["storage_follow_through_ready"] is True
    assert ops_row["state"] == "protected_by_floor"
    assert ops_row["letter_grade"] == "A"
    assert ops_row["signals"]["derived_paper_lane_watchdog"] is True
