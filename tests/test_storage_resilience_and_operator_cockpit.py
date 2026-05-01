import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import daily_verify_auto_remediation_bot as remediation_src
from scripts.ops import operator_cockpit as cockpit_src
from scripts.ops import storage_resilience_control as resilience_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_storage_resilience_control_scores_warm_failover_and_checksums(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "local_fallback_storage").mkdir(parents=True, exist_ok=True)
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": True})
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"mode": "external"})
    _write_json(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "ok": True})
    _write_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json", {"ok": True})

    payload = resilience_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["restore_drill_fresh"] is True
    assert payload["checksum_scrub"]["targets"]


def test_storage_resilience_control_fast_mode_skips_large_db_quick_check(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "local_fallback_storage" / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "governance").mkdir(parents=True, exist_ok=True)
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": True})
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"mode": "external"})
    _write_json(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "ok": True})
    _write_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json", {"ok": True})
    (project_root / "data" / "jsonl_link.sqlite3").write_bytes(b"0" * 2048)

    payload = resilience_src.build_payload(project_root, fast=True, max_quick_check_db_gb=0.000001)

    assert payload["integrity_mode"] == "fast"
    assert payload["database_integrity_checks"][0]["quick_check"] == "skipped_fast_mode_large_db"


def test_operator_cockpit_aggregates_upgrade_surfaces(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json", {"overall": {"status": "degraded", "ok": False, "attention": ["storage_resilience_control_needs_work"]}})
    _write_json(project_root / "governance" / "health" / "platform_control_plane_latest.json", {"institutional_readiness": {"overall_status": "advancing"}})
    _write_json(project_root / "governance" / "health" / "training_report_latest.json", {"overall_status": "blocked"})
    _write_json(project_root / "governance" / "health" / "training_quality_control_latest.json", {"overall_status": "blocked"})
    _write_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json", {"overall_status": "blocked", "top_actions": ["drain core lane"]})
    _write_json(project_root / "governance" / "health" / "ingestion_storage_governor_latest.json", {"profile": "critical_backpressure", "top_actions": ["normalize SQL route"], "sql_primary_db": {"route_drift": True}})
    _write_json(project_root / "governance" / "health" / "storage_tier_policy_latest.json", {"overall_status": "degraded", "pressure": {"hot_path_over_budget_bytes": 2048}, "upgrade_plan": {"recommended_actions": ["split hot and cold storage"]}})
    _write_json(project_root / "governance" / "health" / "training_runtime_control_latest.json", {"overall_status": "blocked", "snapshot_ready": False, "precompute_targets": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy"}], "recommended_actions": ["refresh runtime snapshot"]})
    _write_json(project_root / "governance" / "health" / "external_backlog_drain_latest.json", {"overall_status": "ready", "top_actions": ["run external backlog drain"], "recommended_now": True})
    _write_json(project_root / "governance" / "health" / "ingestion_priority_queue_latest.json", {"top_actions": ["drain queue"]})
    _write_json(project_root / "governance" / "health" / "storage_resilience_control_latest.json", {"overall_status": "needs_work", "top_actions": ["refresh restore drill"]})
    _write_json(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 1}})
    _write_json(project_root / "governance" / "health" / "training_requalification_latest.json", {"recommended_actions": ["build requalification lane"]})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"overall_status": "needs_coverage", "coverage_shortfall_bots": 4, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal"}], "recommended_actions": ["seed coverage"]})
    _write_json(project_root / "governance" / "health" / "regime_control_plane_latest.json", {"overall_status": "thin", "regime_state": "mixed_transition", "stance_label": "neutral", "recommended_actions": ["backfill regime memory"]})
    _write_json(project_root / "governance" / "health" / "supportability_control_latest.json", {"overall_status": "blocked", "supportability": {"active_supportability_score": 0.0}, "teacher_student": {"students_without_teachers": 3}, "recommended_actions": ["assign teachers"]})
    _write_json(project_root / "governance" / "health" / "calibration_abstention_control_latest.json", {"top_actions": ["tighten thresholds"], "overall_status": "needs_tuning"})
    _write_json(project_root / "governance" / "health" / "paper_execution_calibration_latest.json", {"overall_status": "needs_tuning", "metrics": {"mae_bps": 18.5}, "top_actions": ["prioritize profile-level recalibration"]})
    _write_json(project_root / "governance" / "health" / "roster_expansion_slots_latest.json", {"overall_status": "degraded", "summary": {"registered_slot_count": 6, "missing_slot_count": 4}, "recommended_actions": ["register missing roster slots"]})
    _write_json(project_root / "governance" / "health" / "daily_verify_auto_remediation_bot_latest.json", {"recommended_actions": ["remediate"], "overall_status": "pending"})

    payload = cockpit_src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["upgrade_lanes"]["storage_split"]["status"] == "degraded"
    assert payload["upgrade_lanes"]["training_runtime"]["status"] == "blocked"
    assert payload["upgrade_lanes"]["coverage_seeding"]["status"] == "needs_coverage"
    assert payload["upgrade_lanes"]["lifecycle_teaching"]["status"] == "blocked"
    assert payload["upgrade_lanes"]["roster_expansion"]["status"] == "degraded"
    assert "drain core lane" in payload["recommended_actions"]
    assert "normalize SQL route" in payload["recommended_actions"]
    assert "run external backlog drain" in payload["recommended_actions"]
    assert "split hot and cold storage" in payload["recommended_actions"]
    assert "refresh runtime snapshot" in payload["recommended_actions"]
    assert "register missing roster slots" in payload["recommended_actions"]
    assert payload["surfaces"]["ingestion_storage_governor"]["status"] == "critical_backpressure"
    assert payload["surfaces"]["training_runtime_control"]["status"] == "blocked"
    assert payload["surfaces"]["roster_expansion_slots"]["status"] == "degraded"
    assert payload["surfaces"]["regime_control_plane"]["status"] == "thin"
    assert payload["surfaces"]["external_backlog_drain"]["status"] == "ready"
    assert payload["surfaces"]["daily_verify_auto_remediation_bot"]["status"] == "pending"


def test_daily_verify_auto_remediation_bot_builds_actionable_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json", {"failed_checks": ["replay_hash_registry_guard", "db_integrity"]})

    payload = remediation_src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "pending"
    assert len(payload["attempts"]) == 2
    assert all(row["actionable"] for row in payload["attempts"])
