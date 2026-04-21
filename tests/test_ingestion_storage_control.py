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
