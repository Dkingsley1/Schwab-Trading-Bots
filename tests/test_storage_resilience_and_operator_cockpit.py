import json
import sqlite3
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


def test_storage_resilience_control_fast_zero_threshold_skips_db_quick_check(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "local_fallback_storage").mkdir(parents=True, exist_ok=True)
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": True})
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"mode": "external"})
    _write_json(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "ok": True})
    _write_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json", {"ok": True})
    (project_root / "data" / "jsonl_link.sqlite3").write_bytes(b"tiny")

    payload = resilience_src.build_payload(project_root, fast=True, max_quick_check_db_gb=0)

    assert payload["integrity_mode"] == "fast"
    assert payload["database_integrity_checks"][0]["quick_check"] == "skipped_fast_mode_large_db"


def test_storage_resilience_control_uses_local_fallback_for_broken_routed_sqlite(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    routed_db = project_root / "data" / "jsonl_link.sqlite3"
    missing_external_db = tmp_path / "missing_bot_logs" / "data" / "jsonl_link.sqlite3"
    fallback_db = project_root / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    routed_db.parent.mkdir(parents=True, exist_ok=True)
    fallback_db.parent.mkdir(parents=True, exist_ok=True)
    routed_db.symlink_to(missing_external_db)
    with sqlite3.connect(fallback_db) as conn:
        conn.execute("CREATE TABLE rows(id INTEGER PRIMARY KEY)")
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": False})
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"mode": "local_fallback"})
    _write_json(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json", {"ok": True})

    payload = resilience_src.build_payload(project_root)

    primary_check = payload["database_integrity_checks"][0]
    assert primary_check["db_path"] == str(fallback_db)
    assert primary_check["present"] is True
    assert primary_check["ok"] is True


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


def test_operator_cockpit_keeps_expanded_collection_green_with_adaptive_followups(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "overall": {
                "status": "degraded",
                "ok": False,
                "attention": [
                    "external_backlog_drain_recommended",
                    "external_backlog_retry_bot_followups",
                    "memory_efficiency_control_needs_work",
                    "live_runtime_separation_control_needs_work",
                    "auth_lease_manager_needs_work",
                    "runtime_snapshot_cache_control_needs_work",
                    "promotion_not_ready",
                    "training_quality_control_blocked",
                ],
            }
        },
    )
    _write_json(health / "training_report_latest.json", {"overall_status": "blocked"})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.01,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "queue_watermarks": {"breaches": {"hard": [], "elevated": []}},
            "backpressure": {
                "total_pending_lines": 69,
                "core_pending_lines": 34,
                "estimated_total_drain_minutes": 15.0,
            },
            "storage": {"backlog_drain_recommended_now": False},
            "writer_shedding": {"active": False},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "material_drain_recommended": False})
    _write_json(health / "external_backlog_retry_bot_latest.json", {"overall_status": "applied_with_followups", "recommended_actions": ["retry again"]})
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "needs_work",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 0.4},
            "expansion_session": {
                "total_bots": 869,
                "active_bots": 814,
                "data_collection_active_bots": 784,
                "sleeve_profile_count": 206,
                "pressure_level": "massive",
            },
        },
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "summary": {
                "total_bots": 884,
                "active_bots": 829,
                "data_collection_active_bots": 799,
                "sleeve_profile_count": 207,
            }
        },
    )
    _write_json(health / "global_killswitch_latest.json", {"halt": False, "action": "none", "reasons": []})
    _write_json(
        health / "storage_tier_policy_latest.json",
        {
            "overall_status": "blocked",
            "pressure": {"hot_path_over_budget_bytes": 2048},
            "upgrade_plan": {"top_hot_path_families": [{"family": "sql_link_shards", "bytes": 4096}]},
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "shared_host_pressure": {"contention_score": 2, "signals": {"swap_pressure_elevated": False, "restart_storm_present": False}},
            "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"},
        },
    )
    _write_json(health / "runtime_snapshot_cache_control_latest.json", {"overall_status": "degraded", "cache_health": {"snapshot_ready": True}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "degraded", "lease_state": "warning", "lease_budget": {"expires_in_seconds": 1800, "critical_lease_seconds": 600}})
    _write_json(
        health / "rolling_restart_controller_latest.json",
        {
            "overall_status": "blocked",
            "restart_due": True,
            "recommended_scope": "none",
            "due_signals": {
                "checkpoint_missing_or_stale": True,
                "session_stale": False,
                "shadow_heartbeat_stale": False,
                "swap_pressure_high": False,
                "restart_storm_present": False,
            },
        },
    )
    _write_json(
        health / "artifact_freshness_slo_latest.json",
        {
            "overall_status": "blocked",
            "sla_summary": {"stale_required": 1},
            "artifacts": [{"name": "process_watchdog", "required": True, "stale": True}],
        },
    )
    _write_json(
        health / "service_control_plane_latest.json",
        {
            "overall_status": "blocked",
            "upgrade_lanes": {
                "runtime_separation": {"status": "blocked", "summary": "contention_score=4"},
                "operator_cockpit_contract": {"status": "degraded", "summary": "recommended_actions=14"},
            },
        },
    )
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "overall_status": "blocked",
            "operator_followups": [],
            "hardening_scorecard": {
                "truth_layer_ready": True,
                "storage_route_certified": True,
                "process_ownership_canonical": True,
                "command_surface_clean": True,
                "launchd_jobs_installed": True,
            },
            "checks": [{"name": "process_lane_ownership", "status": "ready"}],
        },
    )

    payload = cockpit_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["adaptive_posture"]["overall_status"] == "stable_expansion"
    assert payload["adaptive_posture"]["active_bots"] == 829
    assert payload["adaptive_posture"]["data_collection_active_bots"] == 799
    assert payload["adaptive_posture"]["sleeve_profile_count"] == 207
    assert payload["readiness_domains"]["live_collection"]["status"] == "ready"
    assert payload["readiness_domains"]["training_and_promotion"]["status"] == "blocked"
    assert payload["upgrade_lanes"]["storage_split"]["status"] == "advisory"
    assert payload["upgrade_lanes"]["runtime_separation"]["status"] == "advisory"
    assert payload["surfaces"]["external_backlog_retry_bot"]["status"] == "advisory"
    assert payload["surfaces"]["auth_lease_manager"]["status"] == "advisory"
    assert payload["surfaces"]["rolling_restart_controller"]["status"] == "advisory"
    assert payload["surfaces"]["artifact_freshness_slo"]["status"] == "advisory"
    assert "external_backlog_drain_recommended" not in payload["recommended_actions"]
    assert "memory_efficiency_control_needs_work" not in payload["recommended_actions"]


def test_operator_cockpit_keeps_storage_steady_when_sql_overlay_clears_raw_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {"overall": {"status": "ready", "ok": True, "attention": ["external_backlog_drain_recommended"]}},
    )
    _write_json(health / "training_report_latest.json", {"overall_status": "blocked"})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "queue_watermarks": {"breaches": {"hard": [], "elevated": []}},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "estimated_total_drain_minutes": 15.0,
                "overlay_adjusted": True,
                "overlay_pressure_clear": True,
                "effective_raw_live_source": "fresh_empty_sql_ingestion_overlay",
                "effective_raw_live": {
                    "source": "fresh_empty_sql_ingestion_overlay",
                    "core_pending_lines": 0,
                    "total_pending_lines": 0,
                    "raw_live_estimate": {
                        "core_pending_lines": 20318,
                        "total_pending_lines": 1413269,
                    },
                },
            },
            "storage": {"backlog_drain_recommended_now": True},
            "writer_shedding": {"active": False},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {"overall_status": "drain_active", "recommended_now": True, "material_drain_recommended": True},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "normal", "swap_used_gb": 0.4},
            "expansion_session": {
                "total_bots": 1771,
                "active_bots": 1732,
                "data_collection_active_bots": 1732,
                "sleeve_profile_count": 6,
                "pressure_level": "massive",
            },
        },
    )
    _write_json(project_root / "master_bot_registry.json", {"summary": {"total_bots": 1771, "active_bots": 1732, "data_collection_active_bots": 1732, "sleeve_profile_count": 6}})
    _write_json(health / "global_killswitch_latest.json", {"halt": False, "action": "none", "reasons": []})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "runtime_snapshot_cache_control_latest.json", {"overall_status": "ready", "cache_health": {"snapshot_ready": True}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready"})
    _write_json(health / "rolling_restart_controller_latest.json", {"overall_status": "ready"})
    _write_json(health / "artifact_freshness_slo_latest.json", {"overall_status": "ready", "sla_summary": {"stale_required": 0}})
    _write_json(health / "blackstart_recovery_latest.json", {"overall_status": "ready"})
    _write_json(health / "sleeve_isolation_guard_latest.json", {"overall_status": "ready"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_quota_guard_latest.json", {"overall_status": "ready"})
    _write_json(health / "chaos_drill_coordinator_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "overall_status": "ready",
            "operator_followups": [],
            "hardening_scorecard": {
                "truth_layer_ready": True,
                "storage_route_certified": True,
                "process_ownership_canonical": True,
                "command_surface_clean": True,
                "launchd_jobs_installed": True,
            },
            "checks": [{"name": "process_lane_ownership", "status": "ready"}],
        },
    )

    payload = cockpit_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["adaptive_posture"]["storage_steady"] is True
    assert payload["readiness_domains"]["storage_backpressure"]["status"] == "ready"


def test_daily_verify_auto_remediation_bot_builds_actionable_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json", {"failed_checks": ["replay_hash_registry_guard", "db_integrity"]})

    payload = remediation_src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "pending"
    assert len(payload["attempts"]) == 2
    assert all(row["actionable"] for row in payload["attempts"])
