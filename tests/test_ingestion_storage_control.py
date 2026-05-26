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
                "queue_prune_orphans": "1",
                "queue_orphan_days": 7,
                "queue_max_db_gb": 8,
                "stale_purge_low_value_days": 3,
                "stale_purge_medium_value_days": 14,
                "stale_purge_high_value_days": 30,
                "stale_purge_critical_value_days": 90,
                "stale_purge_max_gb": 20,
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
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 12, "candidate_bytes": 4096, "staged_files": 9, "staged_bytes": 3072}})
    _write_json(
        health / "stale_artifact_reaper_bot_latest.json",
        {
            "summary": {
                "candidate_files": 7,
                "candidate_bytes": 2048,
                "candidate_files_raw": 9,
                "candidate_bytes_raw": 4096,
                "deleted_files": 3,
                "deleted_bytes": 1024,
                "delete_errors": 1,
                "budget_limited": True,
                "skipped_by_budget_files": 2,
                "skipped_by_tier_files": 4,
                "manifest_lines_after": 12,
                "purge_policy": {"low_value_days": 3, "medium_value_days": 14},
            }
        },
    )
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
    assert payload["throttling"]["queue_prune_orphans"] == "1"
    assert payload["throttling"]["stale_purge_low_value_days"] == 3
    assert payload["backpressure"]["support_pending_lines"] == 220000
    assert payload["backpressure"]["stale_stage_pending_lines"] == 580000
    assert payload["storage"]["stale_stage_deleted_bytes"] == 1024
    assert payload["storage"]["stale_stage_budget_limited"] is True
    assert payload["storage"]["stale_stage_delete_errors"] == 1
    assert payload["storage"]["stale_stage_purge_policy"]["low_value_days"] == 3
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


def test_ingestion_storage_control_credits_a_plus_relief_when_queue_is_small_but_not_freshest(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 342,
            "pending_lines_total": 422,
            "pending_lines_deferred": 80,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 77,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 128.98,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
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
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backlog_relief_contract"]["overall_grade"] == "A++"
    assert payload["backlog_relief_contract"]["active_issue_count"] == 0
    assert payload["backpressure_quality_score"] >= 97.0
    assert payload["steady_state"]["target_status"]["backlog_relief_a_plus_ready"] is True


def test_ingestion_storage_control_does_not_penalize_tiny_idle_queue_for_missing_drain_estimate(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 125,
            "pending_lines_total": 352,
            "pending_lines_deferred": 227,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 6,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 78,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["backpressure_quality_score"] >= 95.0
    assert payload["recovery_quality_score"] >= 88.0
    assert payload["recovery_contract"]["steady_state_recovery_ready"] is True
    assert "estimated_total_drain_minutes" not in payload["steady_state"]["target_status"]["target_breaches"]


def test_ingestion_storage_control_bounds_isolated_support_overlay_pressure(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 125,
            "pending_lines_total": 352,
            "pending_lines_deferred": 227,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 6,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "status": "running",
            "cycle_started_utc": (now - timedelta(minutes=3)).isoformat(),
            "merged_rows_this_cycle": 1200,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_support_watchdog_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 55000,
                "files_with_pending": 1,
                "invalid": 0,
                "oldest_uningested_age_seconds": 600.0,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/risk/conservative_equities_schwab/risk_20260521.jsonl",
                        "pending_lines": 55000,
                        "oldest_pending_age_seconds": 600.0,
                    }
                ],
            },
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
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

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["raw_live"]["total_pending_lines"] == 352
    assert payload["backpressure"]["support_pending_lines"] == 55000
    assert payload["steady_state"]["support_overlay_isolated"] is True
    assert payload["backpressure_quality_score"] >= 96.0
    assert payload["recovery_quality_score"] >= 88.0


def test_ingestion_storage_control_decays_fresh_overlay_when_raw_backpressure_cleared(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 516,
            "pending_lines_total": 1149,
            "pending_lines_deferred": 633,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 10,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.898,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 140000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": True, "level": "protect_core"},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {"overall_status": "drain_active", "recommended_now": True, "aged_candidate_files": 0},
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 122221,
                "files_with_pending": 1,
                "oldest_uningested_age_seconds": 4.279,
                "top_pending_files": [
                    {
                        "source_rel": "governance/events/signal_generation_20260524.jsonl",
                        "stream": "governance_events",
                        "storage_temperature": "warm",
                        "ingestion_lane": "nearline_lane",
                        "pending_lines": 122221,
                        "oldest_pending_age_seconds": 4.279,
                    }
                ],
            },
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_crypto_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 106398,
                "files_with_pending": 2,
                "oldest_uningested_age_seconds": 17.308,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/shadow_crypto/trade_decisions_20260524.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 60441,
                        "oldest_pending_age_seconds": 13.844,
                    },
                    {
                        "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260524.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 45957,
                        "oldest_pending_age_seconds": 17.308,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is False
    assert payload["backpressure"]["core_pending_lines"] == 516
    assert payload["backpressure"]["estimated_total_drain_minutes"] <= 15.0
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is True
    assert payload["backlog_truth"]["authoritative_mode"] == "raw_live_overlay_decayed"
    assert payload["overlay_decay"]["should_decay"] is True
    assert payload["overlay_decay"]["reason"] == "raw_live_clear_overlay_fresh_overstates_after_drain"


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


def test_ingestion_storage_control_stabilizes_tiny_hot_queue_under_active_drain(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 586,
            "pending_lines_total": 586,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 19.4,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(seconds=25)).isoformat(),
            "merged_rows_this_cycle": 1,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "maintenance_only",
            "hard_gates": {"ingestion_backpressure_overload": True},
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
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
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
        {"timestamp_utc": now.isoformat(), "files_discovered": 2, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is True
    assert payload["stabilization_contract"]["drain_minutes_total_bounded"] is True
    assert payload["bounded_recovery_contract"]["stale_hard_gate_suppressed"] == ["ingestion_backpressure_overload"]
    assert payload["bounded_recovery_contract"]["stale_severe_backpressure_suppressed"] == ["severe_backpressure_overload"]
    assert payload["backpressure"]["estimated_total_drain_minutes"] == 15.0


def test_ingestion_storage_control_tolerates_incidental_side_lane_trickle(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1058,
            "pending_lines_total": 1061,
            "pending_lines_deferred": 3,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 3,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 2.9,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(seconds=18)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
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
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
        },
    )
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
        {"timestamp_utc": now.isoformat(), "files_discovered": 3, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is True
    assert payload["backpressure"]["estimated_total_drain_minutes"] == 15.0


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


def test_ingestion_storage_control_uses_fresh_sql_ingestion_overlay_when_summary_undercounts(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 18, 20, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 5805,
            "pending_lines_total": 6806,
            "pending_lines_deferred": 1001,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 45.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 60000,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "files_discovered": 2,
            "sqlite": {
                "inserted": 8810,
                "invalid": 0,
                "oversize_payloads": 0,
                "ops_write_failures": 0,
                "pending_lines": 134199,
                "oldest_uningested_age_seconds": 558.97,
                "files_with_pending": 2,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/paper/trade_decisions_20260519.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 92520,
                        "oldest_pending_age_seconds": 535.049,
                        "total_lines": 154027,
                        "last_line": 61507,
                    },
                    {
                        "source_rel": "decisions/shadow_conservative_equities/trade_decisions_20260519.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 41679,
                        "oldest_pending_age_seconds": 452.991,
                        "total_lines": 50000,
                        "last_line": 8321,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is True
    assert payload["backpressure"]["raw_live"]["total_pending_lines"] == 6806
    assert payload["backpressure"]["core_pending_lines"] == 134199
    assert payload["backpressure"]["total_pending_lines"] >= 134199
    assert payload["queue_watermarks_source"] == "live_backpressure+sql_ingestion_overlay"
    assert payload["sql_ingestion_pending_overlay"]["used_for_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["fresh_source_count"] == 1
    assert payload["sql_ingestion_pending_overlay"]["top_pending_files"][0]["source_rel"] == "decisions/paper/trade_decisions_20260519.jsonl"
    assert payload["backlog_truth"]["authoritative_mode"] == "overlay_source_attributed"
    assert payload["backlog_truth"]["raw_live"]["total_pending_lines"] == 6806
    assert payload["backlog_truth"]["sql_overlay"]["used_for_pressure"] is True
    assert payload["stale_pending_locator"]["status"] == "attributed"
    assert payload["stale_pending_locator"]["oldest_sources"][0]["source_rel"] == "decisions/paper/trade_decisions_20260519.jsonl"
    assert payload["data_integrity"]["sql_overlay_pending_lines"] == 134199
    assert any("SQL ingestion overlay" in action for action in payload["top_actions"])


def test_ingestion_storage_control_ignores_stale_sql_ingestion_overlay(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 18, 20, tzinfo=timezone.utc)
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
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=2)).isoformat(),
            "files_discovered": 1,
            "sqlite": {
                "invalid": 0,
                "pending_lines": 200000,
                "oldest_uningested_age_seconds": 7200.0,
                "files_with_pending": 1,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/paper/trade_decisions_20260519.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 200000,
                    }
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is False
    assert payload["backpressure"]["core_pending_lines"] == 300
    assert payload["backpressure"]["total_pending_lines"] == 1200
    assert payload["queue_watermarks_source"] == "live_backpressure"
    assert payload["sql_ingestion_pending_overlay"]["active"] is False
    assert payload["sql_ingestion_pending_overlay"]["fresh_source_count"] == 0
    assert payload["sql_ingestion_pending_overlay"]["stale_source_count"] == 1
    assert payload["backlog_truth"]["authoritative_mode"] == "raw_live"
    assert payload["backlog_truth"]["sql_overlay"]["used_for_pressure"] is False


def test_ingestion_storage_control_uses_fresh_overlay_to_reconcile_stale_raw_pressure_downward(tmp_path: Path) -> None:
    now = datetime(2026, 5, 22, 22, 40, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 3863,
            "pending_lines_total": 251486,
            "pending_lines_deferred": 247623,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 6,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.013,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 1290,
                "oldest_uningested_age_seconds": 73.346,
                "files_with_pending": 2,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/decision/intraday_aggressive_equities_schwab/decision_20260522.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "nearline_lane",
                        "pending_lines": 716,
                        "oldest_pending_age_seconds": 73.267,
                    },
                    {
                        "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260522.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "nearline_lane",
                        "pending_lines": 574,
                        "oldest_pending_age_seconds": 38.329,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is True
    assert payload["backpressure"]["raw_live"]["total_pending_lines"] == 251486
    assert payload["backpressure"]["total_pending_lines"] == 1290
    assert payload["backpressure"]["deferred_pending_lines"] == 0
    assert payload["sql_ingestion_pending_overlay"]["used_for_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["reconciled_downward_for_pressure"] is True
    assert payload["backlog_truth"]["authoritative_mode"] == "overlay_fresh_shard_level"


def test_ingestion_storage_control_decays_unattributed_overlay_pressure(tmp_path: Path) -> None:
    now = datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 200,
            "pending_lines_total": 400,
            "pending_lines_deferred": 200,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 60.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"cycle_started_utc": (now - timedelta(minutes=2)).isoformat(), "merged_rows_this_cycle": 1000})
    _write_json(health / "ingestion_storage_governor_latest.json", {"writer_shedding": {"active": False}, "sql_primary_db": {"route_drift": False}})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "external_backlog_drain_latest.json", {"recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 120000,
                "oldest_uningested_age_seconds": 900.0,
                "files_with_pending": 4,
                "top_pending_files": [],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overlay_decay"]["should_decay"] is True
    assert payload["backpressure"]["overlay_adjusted"] is False
    assert payload["backlog_truth"]["authoritative_mode"] == "raw_live_overlay_decayed"
    assert payload["backpressure"]["total_pending_lines"] == 400


def test_ingestion_storage_control_surfaces_sparse_large_line_action(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 21, 45, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 42000,
            "pending_lines_total": 42000,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 900.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_files": 3,
                "sparse_large_line_pending_lines": 12000,
                "sparse_large_line_bytes": 5_000_000_000,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 10000,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False})

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["raw_live"]["line_estimation"]["sparse_large_line_active"] is True
    assert any("sparse-large-line decision drainer profile" in action for action in payload["top_actions"])


def test_ingestion_storage_control_builds_five_part_backlog_relief_contract(tmp_path: Path) -> None:
    now = datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 42000,
            "pending_lines_total": 53000,
            "pending_lines_deferred": 9000,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 2000,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 3600.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_files": 2,
                "sparse_large_line_pending_lines": 8000,
                "sparse_large_line_pending_bytes": 1_200_000_000,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=30)).isoformat(),
            "merged_rows_this_cycle": 12000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.75})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core"},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "ready",
            "recommended_now": True,
            "aged_candidate_files": 3,
            "off_hours_window": {"active": True},
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    contract = payload["backlog_relief_contract"]
    assert contract["active"] is True
    assert set(contract["active_issue_ids"]) == {
        "single_writer_merge_speed",
        "storage_write_latency",
        "sparse_huge_jsonl_files",
        "intake_outpaces_drain",
        "stale_old_pending_work",
    }
    assert contract["control_env_recommendations"]["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "90"
    assert contract["control_env_recommendations"]["INGEST_MAX_BYTES_PER_FILE"] == str(128 * 1024 * 1024)
    assert contract["control_env_recommendations"]["BACKLOG_PCORE_ALLOCATION_ACTIVE"] == "1"
    assert contract["control_env_recommendations"]["BACKLOG_DRAIN_SINGLE_WRITER_ONLY"] == "1"
    assert contract["control_env_recommendations"]["TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN"] == "1"
    assert float(contract["control_env_recommendations"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"]) <= 0.30
    p_core = contract["p_core_backlog_allocation_contract"]
    assert p_core["policy"] == "p_core_preprocess_single_sql_writer"
    assert p_core["sqlite_writer_count"] == 1
    assert p_core["training_pcore_gate"]["small_targeted_training_allowed_now"] is False
    assert p_core["catch_up_wave_controller"]["max_waves"] == 5
    assert any("bounded catch-up waves" in action for action in payload["top_actions"])


def test_backlog_relief_ignores_tiny_sparse_tail_for_training_gate() -> None:
    contract = src._backlog_relief_contract(
        core_pending_lines=125,
        total_pending_lines=352,
        deferred_pending_lines=227,
        cold_pending_lines=0,
        support_pending_lines=6,
        stale_stage_pending_lines=0,
        oldest_age_seconds=0.0,
        age_threshold_seconds=240.0,
        pending_threshold=15000,
        drain_minutes_total=0.12,
        target_total_drain_minutes=30.0,
        throughput_rows_per_second=50.0,
        merged_rows_this_cycle=14027,
        line_estimation={
            "sparse_large_line_active": True,
            "sparse_large_line_files": 1,
            "sparse_large_line_pending_lines": 125,
            "sparse_large_line_pending_bytes": 1_856_299,
        },
        sql_pending_overlay={},
        sql_service={},
        route_drift=False,
        writer_shedding_active=False,
        aged_candidate_files=0,
    )

    sparse_issue = next(row for row in contract["issues"] if row["id"] == "sparse_huge_jsonl_files")
    assert sparse_issue["active"] is False
    assert sparse_issue["evidence"]["sparse_large_line_detected"] is True
    assert contract["active"] is False
    assert contract["overall_grade"] == "A++"
    assert contract["p_core_backlog_allocation_contract"]["training_pcore_gate"]["small_targeted_training_allowed_now"] is True


def test_p_core_burst_intelligence_uses_seven_when_host_is_deep_green(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "throttle_profile": "soft_cap",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    assert contract["preprocess_worker_budget"] == 7
    assert contract["p_core_burst_intelligence"]["mode"] == "burst_7"
    assert contract["p_core_burst_intelligence"]["seventh_core_burst"]["allowed"] is True
    assert contract["control_env"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "7"


def test_p_core_burst_intelligence_protects_creative_work(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 30.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
            },
            "resource_guard": {"creative_session_level": "hot", "creative_session_kind": "video_editing"},
            "computer_task": {"primary_task": "video_editing"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "creative_foreground_protect_3"


def test_p_core_burst_intelligence_narrows_host_pressure_before_runtime_degrades(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 79.0,
                "compute_pressure_level": "high",
                "memory_pressure_level": "normal",
                "throttle_profile": "sustain",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "host_pressure_relief_3"


def test_p_core_burst_intelligence_keeps_daily_driver_at_five(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed"],
        core_pending_lines=11000,
        total_pending_lines=17000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=300.0,
        age_threshold_seconds=240.0,
        sparse_active=False,
        sparse_pending_bytes=0,
        host_context={
            "off_hours_active": False,
            "runtime_throttle": {
                "host_saturation_score": 45.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
            },
            "resource_guard": {"creative_session_level": "active", "creative_session_kind": "music_playback"},
            "computer_task": {"primary_task": "music_playback"},
        },
    )

    assert contract["preprocess_worker_budget"] == 5
    assert contract["p_core_burst_intelligence"]["mode"] == "daily_driver_5"


def test_p_core_burst_intelligence_narrows_for_memory_pressure(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "yellow",
                "throttle_profile": "sustain",
            },
            "resource_guard": {"memory_pressure_kind": "swap_only", "swap_used_gb": 13.0},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "memory_relief_3"
    assert contract["control_env"]["BACKLOG_MEMORY_PRESSURE_CORE_OPTIMIZER"] == "1"
