import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import grade_regression_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_grade_regression_guard_reports_degraded_when_surfaces_are_recovering(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "needs_attention", "training_quality_score": 82.62})
    _write_json(
        health / "training_lineage_manifest_latest.json",
        {
            "overall_status": "degraded",
            "lineage_score": 72.5,
            "promotion_packet_seed_ready": True,
            "repairable_lineage_contract": {"lineage_recovery_ready": True},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "degraded", "pressure_index": 3.2, "recovery_state": "recovering_under_guard"},
    )
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "passed_checks": 11, "failed_checks": 0})
    _write_json(
        health / "incident_closeout_autopilot_latest.json",
        {"overall_status": "degraded", "open_incident_count": 3, "bounded_data_plane_recovery": True},
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {"overall_status": "degraded", "recommended_mode": "staged_preclearance", "staged_preclearance_ready": True},
    )
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "degraded", "autonomy_score": 50.0})
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {"overall_status": "degraded", "packet_completeness_score": 45.0, "promotion_ready": False},
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["blocked_surface_count"] == 0
    assert payload["degraded_surface_count"] >= 1
    assert any(row["surface"] == "training_quality" and row["state"] == "degraded" for row in payload["surfaces"])
    assert payload["regression_guardrail_contract"]["per_surface_retry_budgets"] is True
    assert all(row["retry_budget"]["max_attempts_per_run"] >= 1 for row in payload["surfaces"])
    assert all(row["notification_contract"]["dedupe_key"].startswith("grade_regression:") for row in payload["surfaces"])


def test_grade_regression_guard_blocks_on_hard_regression(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "training_quality_score": 40.0})
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "blocked", "lineage_score": 20.0})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked", "pressure_index": 9.0, "recovery_state": "stalled"})
    _write_json(health / "security_audit_latest.json", {"overall_status": "needs_work", "passed_checks": 6, "failed_checks": 12})
    _write_json(health / "incident_closeout_autopilot_latest.json", {"overall_status": "blocked", "open_incident_count": 8, "bounded_data_plane_recovery": False})
    _write_json(health / "live_canary_control_latest.json", {"overall_status": "blocked", "recommended_mode": "hold"})
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "blocked", "autonomy_score": 20.0})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "blocked", "packet_completeness_score": 0.0, "promotion_ready": False})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["blocked_surface_count"] >= 4


def test_grade_regression_guard_softens_training_quality_for_guarded_paper_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "training_quality_score": 56.0})
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "ready", "lineage_score": 92.5, "promotion_bundle_ready": True})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "pressure_index": 0.01, "recovery_state": "steady_state"})
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "passed_checks": 17, "failed_checks": 0})
    _write_json(health / "incident_closeout_autopilot_latest.json", {"overall_status": "ready", "open_incident_count": 0})
    _write_json(health / "live_canary_control_latest.json", {"overall_status": "ready", "recommended_mode": "supervised_canary", "supervised_canary_ready": True})
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "ready", "autonomy_score": 96.0})
    _write_json(
        health / "section_grade_guard_latest.json",
        {
            "overall_status": "degraded",
            "ok": True,
            "paper_soak_advisory_below_floor": True,
            "guarded_paper_ready": True,
            "live_execution_locked": True,
            "advisory_below_floor_sections": ["training_and_model_quality"],
        },
    )
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "packet_completeness_score": 100.0, "promotion_ready": True})

    payload = src.build_payload(tmp_path)
    training_row = next(row for row in payload["surfaces"] if row["surface"] == "training_quality")

    assert payload["overall_status"] == "degraded"
    assert payload["blocked_surface_count"] == 0
    assert training_row["state"] == "degraded"
    assert training_row["severity"] == "warning"
    assert training_row["metrics"]["paper_soak_advisory"] is True
    assert training_row["notification_contract"]["tenant_visible"] is False


def test_grade_regression_guard_treats_current_guarded_paper_debt_as_advisory(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "training_quality_score": 38.88})
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "ready", "lineage_score": 92.5, "promotion_bundle_ready": True})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "pressure_index": 0.006, "recovery_state": "steady_state"})
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "passed_checks": 17, "failed_checks": 0})
    _write_json(health / "incident_closeout_autopilot_latest.json", {"overall_status": "blocked", "open_incident_count": 2})
    _write_json(health / "live_canary_control_latest.json", {"overall_status": "blocked", "recommended_mode": "validate_only"})
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "ready", "autonomy_score": 96.0})
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
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
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "packet_completeness_score": 100.0, "promotion_ready": True})

    payload = src.build_payload(tmp_path)
    rows = {row["surface"]: row for row in payload["surfaces"]}

    assert payload["overall_status"] == "degraded"
    assert payload["blocked_surface_count"] == 0
    assert rows["training_quality"]["state"] == "degraded"
    assert rows["training_quality"]["metrics"]["paper_soak_advisory"] is True
    assert rows["incident_closeout"]["state"] == "degraded"
    assert rows["incident_closeout"]["metrics"]["health_fast_strict_clear"] is True
    assert rows["live_canary"]["state"] == "degraded"
    assert rows["live_canary"]["metrics"]["guarded_paper_soak_advisory"] is True


def test_grade_regression_guard_accepts_paper_soak_lineage_without_signed_promotion_packet(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "training_quality_score": 63.0})
    _write_json(
        health / "training_lineage_manifest_latest.json",
        {
            "overall_status": "blocked",
            "lineage_score": 92.5,
            "feature_store_lineage_ok": True,
            "exact_replay_ready": True,
            "replay_hash_registry_ok": True,
            "hash_bundle_complete": True,
            "lineage_contract_ready": True,
            "promotion_bundle_ready": False,
            "promotion_packet_seed_ready": True,
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "pressure_index": 0.01, "recovery_state": "steady_state"})
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "passed_checks": 17, "failed_checks": 0})
    _write_json(health / "incident_closeout_autopilot_latest.json", {"overall_status": "ready", "open_incident_count": 0})
    _write_json(health / "live_canary_control_latest.json", {"overall_status": "degraded", "recommended_mode": "validate_only"})
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "ready", "autonomy_score": 96.0})
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["live_locked"]},
            },
        },
    )
    _write_json(
        health / "section_grade_guard_latest.json",
        {"overall_status": "degraded", "ok": True, "guarded_paper_ready": True, "live_execution_locked": True},
    )
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "degraded", "packet_completeness_score": 40.0, "promotion_ready": False})

    payload = src.build_payload(tmp_path)
    lineage = next(row for row in payload["surfaces"] if row["surface"] == "training_lineage")

    assert lineage["state"] == "ready"
    assert lineage["metrics"]["paper_soak_lineage_ready"] is True
    assert lineage["metrics"]["promotion_bundle_ready"] is False
    assert payload["blocked_surface_count"] == 0


def test_grade_regression_guard_accepts_zero_open_incidents_with_stale_blocked_status(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 93.0})
    _write_json(health / "training_lineage_manifest_latest.json", {"overall_status": "ready", "lineage_score": 92.5, "promotion_bundle_ready": True})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "pressure_index": 0.01, "recovery_state": "steady_state"})
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "passed_checks": 17, "failed_checks": 0})
    _write_json(health / "incident_closeout_autopilot_latest.json", {"overall_status": "blocked", "open_incident_count": 0})
    _write_json(health / "live_canary_control_latest.json", {"overall_status": "ready", "recommended_mode": "supervised_canary", "supervised_canary_ready": True})
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "ready", "autonomy_score": 96.0})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "packet_completeness_score": 100.0, "promotion_ready": True})

    payload = src.build_payload(tmp_path)
    incident = next(row for row in payload["surfaces"] if row["surface"] == "incident_closeout")

    assert payload["blocked_surface_count"] == 0
    assert incident["state"] == "ready"
    assert incident["metrics"]["open_incident_count"] == 0
    assert incident["metrics"]["stale_status_overridden"] is True


def test_grade_regression_guard_respects_live_bounded_storage_recovery_even_with_high_pressure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "needs_attention", "training_quality_score": 80.62})
    _write_json(
        health / "training_lineage_manifest_latest.json",
        {
            "overall_status": "degraded",
            "lineage_score": 72.5,
            "promotion_packet_seed_ready": True,
            "repairable_lineage_contract": {"lineage_recovery_ready": True},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "pressure_index": 114.189,
            "recovery_state": "recovering_under_guard",
            "bounded_recovery_contract": {
                "active": True,
                "active_drain_progress": True,
                "drain_delta_signal_observed": True,
                "guarded_blocked_queue": True,
            },
        },
    )
    _write_json(health / "security_audit_latest.json", {"overall_status": "ready", "passed_checks": 17, "failed_checks": 0})
    _write_json(
        health / "incident_closeout_autopilot_latest.json",
        {"overall_status": "degraded", "open_incident_count": 3, "bounded_data_plane_recovery": True},
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {"overall_status": "degraded", "recommended_mode": "staged_preclearance", "staged_preclearance_ready": True},
    )
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "degraded", "autonomy_score": 59.01})
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {"overall_status": "degraded", "packet_completeness_score": 25.0, "promotion_ready": False},
    )

    payload = src.build_payload(tmp_path)
    storage_row = next(row for row in payload["surfaces"] if row["surface"] == "storage_control")

    assert payload["blocked_surface_count"] == 0
    assert storage_row["state"] == "degraded"
    assert storage_row["quiet_hours_preferred"] is True
