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
