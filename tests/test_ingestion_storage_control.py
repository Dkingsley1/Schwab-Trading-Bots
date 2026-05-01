import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ingestion_storage_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_ingestion_storage_control_estimates_drain_time_and_retention_pressure(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 18000,
            "pending_lines_total": 900000,
            "pending_lines_deferred": 882000,
            "pending_lines_cold": 600000,
            "pending_lines_support_telemetry": 220000,
            "pending_lines_stale_stage": 580000,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 600.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 120000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 19.4})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 4.2, "severe_backpressure_overload": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "resource_guard_blocked"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": True},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
            "throttle_controls": {
                "deferred_files_budget": 0,
                "cold_files_budget": 0,
                "log_api_calls": "0",
                "log_loop_state": "0",
                "log_data_ingress": "0",
                "log_shadow_pnl_attribution": "0",
            },
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "ready",
            "recommended_now": True,
            "aged_candidate_files": 4,
            "off_hours_window": {"active": True},
            "drain_overrides": {"deferred_files_budget": 6, "cold_files_budget": 2},
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "warning", "ready_count": 1, "tracked_count": 3, "coverage_ratio": 0.333333, "mismatches": ["data/jsonl_link.sqlite3"]}},
    )
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 12, "staged_files": 9}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 3}})
    _write_json(health / "data_retention_latest.json", {"deleted": 22})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc))

    assert payload["overall_status"] == "blocked"
    assert payload["severity"] == "critical"
    assert payload["backpressure"]["estimated_core_drain_minutes"] is not None
    assert payload["storage"]["retention_debt_gb"] == 4.2
    assert payload["storage"]["governor_profile"] == "critical_backpressure"
    assert payload["storage"]["sql_primary_route_drift"] is True
    assert payload["storage"]["backlog_drain_recommended_now"] is True
    assert payload["storage"]["aged_backlog_candidate_files"] == 4
    assert payload["throttling"]["deferred_files_budget"] == 0
    assert payload["throttling"]["backlog_drain_deferred_budget"] == 6
    assert payload["backpressure"]["support_pending_lines"] == 220000
    assert payload["backpressure"]["stale_stage_pending_lines"] == 580000
    assert payload["backpressure_quality_score"] < 50.0
    assert payload["steady_state"]["target_status"]["target_breach_count"] >= 4
    assert payload["recommended_operating_mode"] == "maintenance_drain_window"
    assert payload["top_actions"][0].startswith("normalize the SQL linker")
    assert any("support shard" in action for action in payload["top_actions"])
    assert any("stale-stage" in action for action in payload["top_actions"])
    assert payload["queue_watermarks"]["overall_status"] == "blocked"
    assert payload["writer_shedding"]["level"] == "protect_core"
    assert payload["external_route_verification"]["verification_state"] == "warning"


def test_ingestion_storage_control_uses_market_hours_protection_when_only_quarantine_is_available(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 9000,
            "pending_lines_total": 450000,
            "pending_lines_deferred": 220000,
            "pending_lines_cold": 180000,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 1800.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=8)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 2.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 3.5, "severe_backpressure_overload": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "throttle_controls": {"deferred_files_budget": 0, "cold_files_budget": 0},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "blocked",
            "recommended_now": True,
            "aged_candidate_files": 2,
            "off_hours_window": {"active": False},
            "drain_overrides": {"deferred_files_budget": 6, "cold_files_budget": 2},
        },
    )
    _write_json(
        health / "backlog_quarantine_bot_latest.json",
        {
            "overall_status": "ready",
            "candidate_files": 2,
            "moved_files": 0,
            "moved_pending_lines": 0,
        },
    )
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 15, 0, tzinfo=timezone.utc))

    assert payload["recommended_operating_mode"] == "market_hours_backlog_protection"
    assert payload["storage"]["backlog_quarantine_candidate_files"] == 2


def test_ingestion_storage_control_reports_green_steady_state_targets(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 300,
            "pending_lines_total": 1200,
            "pending_lines_deferred": 900,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 15.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.1})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
            "throttle_controls": {"deferred_files_budget": 2, "cold_files_budget": 1},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc))

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["backpressure_quality_score"] >= 95.0
    assert payload["steady_state"]["target_status"]["steady_state_ready"] is True
    assert payload["steady_state"]["target_status"]["target_breaches"] == []
    assert payload["queue_watermarks"]["overall_status"] == "ready"
    assert payload["writer_shedding"]["active"] is False


def test_ingestion_storage_control_prefers_live_backpressure_over_stale_governor_watermarks(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 315,
            "pending_lines_total": 389,
            "pending_lines_deferred": 74,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 1,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 13.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 1000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {
                "overall_status": "blocked",
                "lanes": {"core": {"pending_lines": 64458}},
                "breaches": {"hard": ["core"]},
            },
            "writer_shedding": {"active": True, "level": "protect_core"},
            "throttle_controls": {"deferred_files_budget": 0, "cold_files_budget": 0},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 5, 1, 2, 15, tzinfo=timezone.utc))

    assert payload["queue_watermarks_source"] == "live_backpressure"
    assert payload["queue_watermarks"]["overall_status"] == "ready"
    assert payload["queue_watermarks"]["lanes"]["core"]["pending_lines"] == 315
    assert payload["bounded_recovery_contract"]["stale_hard_gate_suppressed"] == ["ingestion_backpressure_overload"]
    assert payload["bounded_recovery_contract"]["hard_gate_keys"] == []
    assert payload["overall_status"] == "ready"


def test_ingestion_storage_control_degrades_ready_state_when_storage_resilience_needs_work(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 300,
            "pending_lines_total": 1200,
            "pending_lines_deferred": 900,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 15.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.1})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
            "throttle_controls": {"deferred_files_budget": 2, "cold_files_budget": 1},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "needs_work",
            "resilience_score": 62,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 1,
        },
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc))

    assert payload["overall_status"] == "needs_work"
    assert payload["ok"] is False
    assert payload["storage_resilience"]["overall_status"] == "needs_work"
    assert payload["storage_resilience"]["unresolved_split_brain_conflicts"] == 1
    assert any("restore drill" in action for action in payload["top_actions"])


def test_ingestion_storage_control_ignores_non_storage_hard_gates(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 5000,
            "pending_lines_total": 30000,
            "pending_lines_deferred": 12000,
            "pending_lines_cold": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 30.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=3)).isoformat(),
            "merged_rows_this_cycle": 60000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.2})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {
                "collector_contracts": True,
                "ingestion_backpressure_overload": False,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "throttle_controls": {"deferred_files_budget": 4, "cold_files_budget": 1},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "backlog_quarantine_bot_latest.json", {"overall_status": "idle", "candidate_files": 0, "moved_files": 0, "moved_pending_lines": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 6, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] in {"stable", "elevated"}
    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True


def test_ingestion_storage_control_marks_bounded_critical_pressure_as_recovering(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 24000,
            "pending_lines_total": 600000,
            "pending_lines_deferred": 480000,
            "pending_lines_cold": 120000,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 5000,
            "oldest_pending_age_seconds": 60.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 400000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 1.1})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": True, "aged_candidate_files": 0})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 82,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 21, 0, tzinfo=timezone.utc))

    assert payload["severity"] == "critical"
    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "recovering_under_guard"
    assert payload["bounded_recovery_contract"]["active"] is True


def test_ingestion_storage_control_accepts_recoverable_hard_gates_when_drain_is_active(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 20995,
            "pending_lines_total": 21311,
            "pending_lines_deferred": 316,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 291.366,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=2)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "collector_contracts": True,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "degraded"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True, "throttle_deferred_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "apply_requested": True,
            "aged_candidate_files": 1,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": 4791, "total_pending_lines": 4781},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 80,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] == "critical"
    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "recovering_under_guard"
    assert payload["bounded_recovery_contract"]["active"] is True
    assert payload["bounded_recovery_contract"]["recoverable_hard_gate_only"] is True
    assert payload["bounded_recovery_contract"]["drain_delta_core_lines"] == 4791


def test_ingestion_storage_control_accepts_guarded_blocked_queue_with_negative_drain_deltas(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 90197,
            "pending_lines_total": 1708861,
            "pending_lines_deferred": 1618666,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 312.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "collector_contracts": True,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True, "throttle_deferred_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "apply_requested": True,
            "aged_candidate_files": 1,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": -78, "total_pending_lines": -102},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 80,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] == "critical"
    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "stabilized_recovery"
    assert payload["bounded_recovery_contract"]["active"] is True
    assert payload["bounded_recovery_contract"]["guarded_blocked_queue"] is True
    assert payload["bounded_recovery_contract"]["quality_ready"] is True
    assert payload["bounded_recovery_contract"]["drain_delta_signal_observed"] is True
    assert payload["bounded_recovery_contract"]["drain_delta_core_lines"] == -78
    assert payload["backpressure_quality_score"] >= 96.0
    assert payload["recovery_quality_score"] >= 96.0


def test_ingestion_storage_control_accepts_sql_progress_stall_as_recoverable_under_active_drain(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 107061,
            "pending_lines_total": 2100000,
            "pending_lines_deferred": 1992939,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 330.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "sql_progress_stall": True,
                "sql_wal_pressure": True,
            },
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True, "throttle_deferred_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "apply_requested": True,
            "aged_candidate_files": 1,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": -7743, "total_pending_lines": -8930},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "stabilized_recovery"
    assert payload["bounded_recovery_contract"]["active"] is True
    assert payload["bounded_recovery_contract"]["recoverable_hard_gate_only"] is True
    assert payload["bounded_recovery_contract"]["quality_ready"] is True
    assert payload["backpressure_quality_score"] >= 96.0
    assert payload["recovery_quality_score"] >= 96.0
