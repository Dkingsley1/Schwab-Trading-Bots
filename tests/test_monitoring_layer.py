import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.daily_auto_verify as daily_auto_verify
import scripts.health_gates as health_gates
import scripts.ops.runtime_gate_dashboard as runtime_gate_dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_health_gates_prefers_freshest_ingestion_payload(tmp_path: Path) -> None:
    older = tmp_path / "jsonl_sql_ingestion_health_latest.json"
    newer = tmp_path / "jsonl_sql_ingestion_health_trading_latest.json"
    now = datetime.now(timezone.utc)

    _write_json(older, {"timestamp_utc": (now - timedelta(hours=3)).isoformat(), "sqlite": {"pending_lines": 90}})
    _write_json(newer, {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 5}})

    payload, source = health_gates._freshest_non_empty_json([older, newer])

    assert source == str(newer)
    assert payload["sqlite"]["pending_lines"] == 5


def test_health_gates_does_not_hard_gate_tiny_old_ingestion_tail(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"
    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.010000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(sql_root / "daily_runtime_summary_latest.json", {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}})
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 999, "oldest_uningested_age_seconds": 3131.7, "invalid": 0},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 7602,
            "pending_files": 8,
            "oldest_pending_age_seconds": 73392.5,
            "overload": False,
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["hard_gate_triggered"] is False
    assert payload["hard_gates"]["ingestion_oldest_age"] is False
    assert payload["inputs"]["ingest_oldest_age_material_lines"] == 2000


def test_daily_auto_verify_resolves_best_freshness_artifact(tmp_path: Path) -> None:
    original_root = daily_auto_verify.PROJECT_ROOT
    original_groups = daily_auto_verify.DEFAULT_FRESHNESS_FILE_GROUPS
    try:
        daily_auto_verify.PROJECT_ROOT = tmp_path
        old_file = tmp_path / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json"
        new_file = tmp_path / "governance" / "health" / "jsonl_sql_ingestion_health_trading_latest.json"
        now = datetime.now(timezone.utc)
        _write_json(old_file, {"timestamp_utc": (now - timedelta(hours=2)).isoformat()})
        _write_json(new_file, {"timestamp_utc": now.isoformat()})
        daily_auto_verify.DEFAULT_FRESHNESS_FILE_GROUPS = [[old_file, new_file]]

        resolved = daily_auto_verify._resolve_freshness_files("")

        assert resolved == [new_file]
    finally:
        daily_auto_verify.PROJECT_ROOT = original_root
        daily_auto_verify.DEFAULT_FRESHNESS_FILE_GROUPS = original_groups


def test_runtime_gate_dashboard_ignores_stale_daily_verify_failures_when_fresh_gates_are_green(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["new_bot_graduation_gate", "replay_hash_registry_guard"],
            "completed_checks": 39,
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(
        walk_root / "new_bot_graduation_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "maturity": {"mature_bots": 8}, "immature_active_count": 0},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "daily_auto_verify_not_ok" not in payload["overall"]["attention"]
    assert payload["artifacts"]["daily_auto_verify"]["summary"]["effective_failed_checks"] == []
    assert sorted(payload["artifacts"]["daily_auto_verify"]["summary"]["resolved_failed_checks"]) == [
        "new_bot_graduation_gate",
        "replay_hash_registry_guard",
    ]


def test_runtime_gate_dashboard_dedupes_daily_verify_when_only_promotion_gate_remains(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["promotion_quality_gate"],
            "completed_checks": 39,
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"timestamp_utc": now.isoformat(), "ok": False, "failed_checks": ["promotion_gate_blocked"]})
    _write_json(
        walk_root / "new_bot_graduation_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "maturity": {"mature_bots": 8}, "immature_active_count": 0},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": False, "considered_bots": 5, "failed_bots": 3, "fail_share": 0.6},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "promotion_not_ready" in payload["overall"]["attention"]
    assert "daily_auto_verify_not_ok" not in payload["overall"]["attention"]
    assert payload["artifacts"]["daily_auto_verify"]["summary"]["effective_failed_checks"] == ["promotion_quality_gate"]


def test_runtime_gate_dashboard_resolves_signed_seed_ready_promotion_packet_daily_failure(tmp_path: Path) -> None:
    packet_path = tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json"
    _write_json(
        packet_path,
        {
            "ok": False,
            "committee_packet_seed_ready": True,
            "signing_material_ready": True,
            "trained_models_complete": True,
            "signature": {"status": "verified", "verified": True},
            "replayability_contract": {"hash_bundle_complete": True, "exact_replay_ready": True},
            "gate_results": {
                "training_success_confirmed": False,
                "feature_store_manifest_strict_ok": True,
                "new_bot_admission_ok": True,
            },
        },
    )

    unresolved, resolved = runtime_gate_dashboard._resolved_daily_auto_verify_failures(
        {"failed_checks": ["promotion_packet_builder"]},
        {"promotion_packet": {"ok": False, "path": str(packet_path)}},
    )

    assert unresolved == []
    assert resolved == ["promotion_packet_builder"]


def test_runtime_gate_dashboard_manages_paper_soak_cold_lane_degradations(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(health_root / "session_ready_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "checks": []})
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 91.5, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}, "files_discovered": 1},
    )
    _write_json(
        health_root / "unattended_soak_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "overall_status": "ready", "overall_grade": "A+", "safe_to_leave_unattended": True},
    )
    _write_json(
        health_root / "runtime_paper_regression_guard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
            "failed_guards": [],
            "paper_stage": "armed",
        },
    )
    _write_json(health_root / "health_fast_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "overall_status": "ready"})
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "completed_checks": 50,
            "failed_checks": [
                "snapshot_coverage_sentinel",
                "feature_store_manifest",
                "retrain_schema_compatibility_guard",
                "promotion_packet_builder",
                "promotion_quality_gate",
                "db_integrity",
            ],
        },
    )
    _write_json(
        health_root / "sqlite_maintenance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "timed_out": False,
            "checkpoint_only": True,
            "running": False,
            "current_step": "complete",
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.405,
            "continuous_run_soak_contract": {"status": "watch", "soak_ready": True, "blockers": []},
        },
    )
    _write_json(
        health_root / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "drain_active",
            "recommended_now": True,
            "material_drain_recommended": True,
            "writer_busy": True,
            "aged_candidate_files": 0,
            "candidate_files": 0,
            "follow_through": {"status": "handoff_requested", "progress_state": "requested_live_writer"},
        },
    )
    _write_json(
        health_root / "external_backlog_retry_bot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "applied_with_followups",
            "actionable": True,
            "backlog_needed": True,
        },
    )
    _write_json(
        health_root / "training_quality_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "blocked",
            "training_quality_score": 57.0,
            "top_priorities": [
                "active_probation_isolation",
                "experiment_replayability",
                "feature_store_lineage",
                "multiple_testing_control",
                "promotion_coverage",
            ],
            "supportability": {"active_supportability_score": 100.0},
        },
    )
    _write_json(
        health_root / "infrastructure_autofix_bot_latest.json",
        {"timestamp_utc": now.isoformat(), "overall_status": "blocked", "applyable_repair_count": 7, "operator_followups": []},
    )
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "degraded",
            "live_plane": {"ready": True},
            "release_contract": {"live_lane_should_be_read_only": True, "promotions_should_wait_for_cold_lane": True},
            "shared_host_pressure": {
                "restart_storms": 0,
                "restart_storm_contention_count": 0,
                "storage_overlay_relief": {"raw_live_clear": True},
            },
        },
    )
    _write_json(
        health_root / "coordination_state_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "blocked",
            "policies": {
                "live_orders": {
                    "allowed": False,
                    "blockers": ["paper_trade_lock_active", "runtime_release_live_read_only", "live_runtime_release_read_only"],
                },
                "paper_execution": {"allowed": True, "paper_trade_lock_active": True},
                "terminal_restart": {"safe": True},
                "light_livefeed": {"allowed": True},
            },
        },
    )
    _write_json(
        health_root / "storage_quota_guard_latest.json",
        {"timestamp_utc": now.isoformat(), "overall_status": "degraded", "quota_summary": {"hard_breaches": 0, "soft_breaches": 1, "blocked_families": []}},
    )
    _write_json(
        health_root / "runtime_snapshot_cache_control_latest.json",
        {"timestamp_utc": now.isoformat(), "overall_status": "degraded", "cache_health": {"snapshot_ready": True}},
    )
    _write_json(walk_root / "promotion_readiness_latest.json", {"timestamp_utc": now.isoformat(), "promote_ok": False})
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)
    managed = {row["attention"] for row in payload["overall"]["managed_controls"]}

    assert payload["overall"]["ok"] is True
    assert payload["overall"]["attention"] == []
    assert {
        "daily_auto_verify_not_ok",
        "training_quality_control_blocked",
        "infrastructure_autofix_bot_blocked",
        "live_runtime_separation_control_needs_work",
        "coordination_state_control_blocked",
        "storage_quota_guard_needs_work",
        "external_backlog_retry_bot_followups",
    }.issubset(managed)
    assert payload["overall"]["raw_attention"] == []
    assert "daily_auto_verify_not_ok" in payload["overall"]["forensic_attention"]
    assert "daily_auto_verify_not_ok" in payload["overall"]["managed_attention"]


def test_runtime_gate_dashboard_resolves_recovered_nightly_resilience_and_artifact_freshness(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    nightly_path = health_root / "nightly_resilience_latest.json"
    _write_json(
        nightly_path,
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "failed_checks": [],
            "metrics": {"watchdog_process_count": 1, "shadow_loop_process_count": 4, "watchdog_log_age_minutes": 1.2},
        },
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["nightly_resilience_check", "artifact_freshness"],
            "completed_checks": 39,
            "checks": {
                "artifact_freshness": {
                    "ok": False,
                    "max_age_minutes": 20.0,
                    "fresh_if_newer_than_utc": "",
                    "stale_files": [str(nightly_path)],
                    "missing_files": [],
                    "rows": [
                        {
                            "path": str(nightly_path),
                            "exists": True,
                            "age_minutes": 180.0,
                            "ok": False,
                        }
                    ],
                }
            },
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "daily_auto_verify_not_ok" not in payload["overall"]["attention"]
    assert payload["artifacts"]["daily_auto_verify"]["summary"]["effective_failed_checks"] == []
    assert sorted(payload["artifacts"]["daily_auto_verify"]["summary"]["resolved_failed_checks"]) == [
        "artifact_freshness",
        "nightly_resilience_check",
    ]


def test_runtime_gate_dashboard_surfaces_storage_governor_route_drift(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(health_root / "session_ready_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []})
    _write_json(health_root / "health_gates_latest.json", {"timestamp_utc": now.isoformat(), "data_quality_score": 99.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.0}})
    _write_json(health_root / "jsonl_sql_ingestion_health_trading_latest.json", {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}})
    _write_json(health_root / "sql_link_service_progress_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 4.2,
            "recommended_operating_mode": "maintenance_only",
            "backpressure": {"estimated_core_drain_minutes": 95.0, "estimated_total_drain_minutes": 420.0},
            "storage": {"retention_debt_gb": 11.0},
        },
    )
    _write_json(
        health_root / "ingestion_storage_governor_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": True},
            "throttle_controls": {"deferred_files_budget": 0, "cold_files_budget": 0},
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_primary_route_drift" in payload["overall"]["attention"]
    assert payload["storage"]["pressure_profile"] == "critical_backpressure"
    assert payload["storage"]["sql_primary_route_drift"] is True


def test_runtime_gate_dashboard_surfaces_external_backlog_drain_signal(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(health_root / "session_ready_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []})
    _write_json(health_root / "health_gates_latest.json", {"timestamp_utc": now.isoformat(), "data_quality_score": 98.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.0}})
    _write_json(health_root / "jsonl_sql_ingestion_health_trading_latest.json", {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}})
    _write_json(health_root / "sql_link_service_progress_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"})
    _write_json(
        health_root / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "recommended_now": True,
            "writer_busy": False,
            "aged_candidate_files": 3,
            "follow_through": {"status": "completed"},
            "off_hours_window": {"active": True},
            "drain_overrides": {"deferred_files_budget": 6, "cold_files_budget": 2},
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "external_backlog_drain_recommended" in payload["overall"]["attention"]
    assert payload["storage"]["backlog_drain_recommended"] is True
    assert payload["storage"]["backlog_drain_aged_candidate_files"] == 3
    assert payload["storage"]["backlog_drain_follow_through_status"] == "completed"


def test_runtime_gate_dashboard_treats_progressing_follow_through_as_non_stalled(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(health_root / "session_ready_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []})
    _write_json(health_root / "health_gates_latest.json", {"timestamp_utc": now.isoformat(), "data_quality_score": 98.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.0}})
    _write_json(health_root / "jsonl_sql_ingestion_health_trading_latest.json", {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}})
    _write_json(health_root / "sql_link_service_progress_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"})
    _write_json(
        health_root / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "drain_active",
            "recommended_now": True,
            "writer_busy": True,
            "aged_candidate_files": 3,
            "follow_through": {"status": "timed_out", "progress_state": "progressing", "progress_observed": True},
            "off_hours_window": {"active": True},
            "drain_overrides": {"deferred_files_budget": 6, "cold_files_budget": 2},
        },
    )
    _write_json(
        health_root / "external_backlog_retry_bot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "applied_progressing",
            "actionable": True,
            "backlog_needed": True,
            "drain_result": {"follow_through_status": "timed_out", "follow_through_progress_state": "progressing", "follow_through_attempts": 4},
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "external_backlog_drain_follow_through_stalled" not in payload["overall"]["attention"]
    assert "external_backlog_retry_bot_followups" not in payload["overall"]["attention"]
    assert payload["storage"]["backlog_drain_follow_through_progress_state"] == "progressing"
    assert payload["storage"]["backlog_drain_follow_through_progress_observed"] is True


def test_runtime_gate_dashboard_ignores_idle_writer_busy_without_material_backlog(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(health_root / "session_ready_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []})
    _write_json(health_root / "health_gates_latest.json", {"timestamp_utc": now.isoformat(), "data_quality_score": 98.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.0}})
    _write_json(health_root / "jsonl_sql_ingestion_health_trading_latest.json", {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}})
    _write_json(health_root / "sql_link_service_progress_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"})
    _write_json(
        health_root / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "drain_active",
            "recommended_now": False,
            "material_drain_recommended": False,
            "writer_busy": True,
            "aged_candidate_files": 0,
            "follow_through": {"status": "handoff_requested", "progress_state": "requested_live_writer"},
            "off_hours_window": {"active": True},
            "drain_overrides": {"deferred_files_budget": 4, "cold_files_budget": 1},
        },
    )
    _write_json(
        health_root / "external_backlog_retry_bot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "applied_with_followups",
            "actionable": True,
            "backlog_needed": True,
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "external_backlog_drain_writer_busy" not in payload["overall"]["attention"]
    assert "external_backlog_retry_bot_followups" not in payload["overall"]["attention"]
    assert payload["storage"]["backlog_drain_recommended"] is False


def test_daily_auto_verify_artifact_freshness_accepts_artifacts_written_during_run(tmp_path: Path) -> None:
    artifact = tmp_path / "governance" / "health" / "session_ready_latest.json"
    now = datetime.now(timezone.utc)
    _write_json(artifact, {"timestamp_utc": (now - timedelta(minutes=90)).isoformat()})

    status = daily_auto_verify._artifact_freshness_status(
        [artifact],
        max_age_minutes=20.0,
        fresh_if_newer_than=now - timedelta(hours=2),
    )

    assert status["ok"] is True
    assert status["rows"][0]["refreshed_in_run"] is True


def test_daily_auto_verify_infers_clear_sql_ingestion_freshness_from_service_heartbeat(tmp_path: Path) -> None:
    original_root = daily_auto_verify.PROJECT_ROOT
    try:
        daily_auto_verify.PROJECT_ROOT = tmp_path
        health_root = tmp_path / "governance" / "health"
        now = datetime.now(timezone.utc)
        artifact = health_root / "jsonl_sql_ingestion_health_trading_latest.json"
        _write_json(
            artifact,
            {
                "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
                "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0, "files_with_pending": 0},
            },
        )
        _write_json(
            health_root / "sql_link_service_progress_latest.json",
            {
                "timestamp_utc": now.isoformat(),
                "ok": True,
                "status": "ok",
                "current_step": "complete",
                "merged_rows_this_cycle": 7,
            },
        )

        status = daily_auto_verify._artifact_freshness_status([artifact], max_age_minutes=20.0, fresh_if_newer_than=now)

        assert status["ok"] is True
        assert status["rows"][0]["freshness_inferred_from_sql_service"] is True
    finally:
        daily_auto_verify.PROJECT_ROOT = original_root


def test_runtime_gate_dashboard_uses_current_registry_and_trading_ingestion(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default", "fx"], "checks": []},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": [], "completed_checks": 5},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "data_quality_score": 88.2,
            "hard_gate_triggered": False,
            "inputs": {"blocked_rate": 0.22},
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=4)).isoformat(),
            "sqlite": {"pending_lines": 777, "oldest_uningested_age_seconds": 999.0, "invalid": 3},
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "files_discovered": 12,
            "sqlite": {"pending_lines": 5, "oldest_uningested_age_seconds": 12.0, "invalid": 0},
        },
    )
    _write_json(
        health_root / "sql_link_service_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=3)).isoformat(),
            "ok": True,
            "current_step": "complete",
            "merged_rows_this_cycle": 99,
        },
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "running": True,
            "status": "running",
            "current_step": "merge_primary",
            "completed_shard_count": 2,
            "completed_merge_count": 1,
            "merged_rows_this_cycle": 42,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "needs_work",
            "severity": "high",
            "pressure_index": 1.8,
            "recommended_operating_mode": "maintenance_only",
            "backpressure": {"estimated_core_drain_minutes": 22.0, "estimated_total_drain_minutes": 95.0},
            "storage": {"retention_debt_gb": 2.4},
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "needs_work",
            "recommended_profile": "air_safe",
            "memory_snapshot": {"memory_pressure_state": "yellow", "memory_pressure_kind": "swap_only", "swap_used_gb": 18.5},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "promote_ok": False,
            "considered_bots": 7,
            "failed_bots": 2,
            "fail_share": 0.285714,
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "updated_at_utc": now.isoformat(),
            "summary": {
                "total_bots": 96,
                "active_bots": 15,
                "deleted_from_rotation": 69,
                "deletion_guard_ok": False,
                "deletion_guard_reason": "training_success_not_confirmed",
                "top_active": [{"bot_id": "brain_refinery_v4_simple"}],
            },
            "sub_bots": [{"bot_id": "brain_refinery_v4_simple", "active": True}],
        },
    )

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["artifacts"]["sql_ingestion"]["path"].endswith("jsonl_sql_ingestion_health_trading_latest.json")
    assert payload["artifacts"]["sql_ingestion"]["summary"]["pending_lines"] == 5
    assert payload["artifacts"]["sql_link_service"]["path"].endswith("sql_link_service_progress_latest.json")
    assert payload["artifacts"]["sql_link_service"]["summary"]["current_step"] == "merge_primary"
    assert payload["registry"]["active_bots"] == 15
    assert payload["registry"]["total_bots"] == 96
    assert payload["overall"]["status"] == "warn"
    assert "promotion_not_ready" in payload["overall"]["attention"]
    assert payload["storage"]["severity"] == "high"
    assert payload["memory"]["recommended_profile"] == "air_safe"


def test_runtime_gate_dashboard_uses_service_heartbeat_for_sql_ingestion_freshness(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": (now - timedelta(minutes=5)).isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
            "files_discovered": 12,
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["artifacts"]["sql_ingestion"]["stale"] is False
    assert payload["artifacts"]["sql_ingestion"]["summary"]["freshness_via_service_heartbeat"] is True
    assert "sql_ingestion_stale" not in payload["overall"]["attention"]


def test_runtime_gate_dashboard_suppresses_sql_service_stale_when_ingestion_is_fresh(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": (now - timedelta(minutes=45)).isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_link_service_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["sql_link_service"]["summary"]["freshness_inferred_from_sql_ingestion"] is True


def test_runtime_gate_dashboard_uses_storage_control_overlay_for_idle_sql_plane_freshness(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=35)).isoformat(),
            "ok": True,
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "merged_rows_this_cycle": 7,
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=65)).isoformat(),
            "files_discovered": 12,
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.05,
            "sql_ingestion_pending_overlay": {
                "active": True,
                "fresh_source_count": 2,
                "fresh_pending_unknown_source_count": 0,
                "total_pending_lines": 0,
                "files_with_pending": 0,
                "invalid_lines": 0,
                "stale_pending_lines": 0,
                "ops_write_failures": 0,
            },
            "continuous_run_soak_contract": {"ready": True, "soak_ready": True, "grade": "A+"},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_link_service_stale" not in payload["overall"]["attention"]
    assert "sql_ingestion_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["sql_link_service"]["stale"] is False
    assert payload["artifacts"]["sql_ingestion"]["stale"] is False
    assert payload["artifacts"]["sql_link_service"]["summary"]["freshness_inferred_from_ingestion_storage_control"] is True
    assert payload["artifacts"]["sql_ingestion"]["summary"]["freshness_inferred_from_ingestion_storage_control"] is True


def test_runtime_gate_dashboard_uses_storage_control_backpressure_for_idle_sql_plane_freshness(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "data_quality_score": 99.9,
            "hard_gate_triggered": False,
            "inputs": {"blocked_rate": 0.01},
        },
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=35)).isoformat(),
            "ok": True,
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "merged_rows_this_cycle": 7,
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=65)).isoformat(),
            "files_discovered": 12,
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.058,
            "backpressure": {
                "effective_raw_live_source": "fresh_empty_sql_ingestion_overlay",
                "raw_live": {
                    "core_pending_lines": 866,
                    "deferred_pending_lines": 1769,
                    "support_pending_lines": 38,
                    "total_pending_lines": 2673,
                    "oldest_pending_age_seconds": 0.0,
                },
            },
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {
                "hard_breaches": [],
                "elevated_breaches": [],
            },
            "continuous_run_soak_contract": {
                "ready": False,
                "soak_ready": False,
                "grade": "D",
                "blockers": ["steady_state_targets_not_clear", "drain_time_above_target"],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_link_service_stale" not in payload["overall"]["attention"]
    assert "sql_ingestion_stale" not in payload["overall"]["attention"]
    inference = payload["artifacts"]["sql_ingestion"]["summary"]["freshness_inference"]
    assert inference["source"] == "ingestion_storage_control_reconciled_backpressure"
    assert inference["raw_live_total_pending_lines"] == 2673
    assert inference["continuous_soak_blockers_managed_for_sql_freshness"] is True
    assert inference["managed_continuous_soak_blockers"] == [
        "drain_time_above_target",
        "steady_state_targets_not_clear",
    ]


def test_runtime_gate_dashboard_suppresses_sql_stale_when_live_writer_lock_exists(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    lock_path = tmp_path / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"pid={os.getpid()} started={now.isoformat()} cmd=sql_link_shard_manager", encoding="utf-8")

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
            "ok": True,
            "running": True,
            "status": "running",
            "current_step": "merge_primary",
            "lock_path": str(lock_path),
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
            "files_discovered": 20,
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_link_service_stale" not in payload["overall"]["attention"]
    assert "sql_ingestion_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["sql_link_service"]["summary"]["freshness_inferred_from_live_lock"] is True
    assert payload["artifacts"]["sql_link_service"]["summary"]["lock_owner_pid"] == os.getpid()
    assert payload["artifacts"]["sql_ingestion"]["summary"]["freshness_via_service_heartbeat"] is True


def test_runtime_gate_dashboard_suppresses_session_ready_stale_when_shadow_loop_is_fresh(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": (now - timedelta(minutes=20)).isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "shadow_loop_default_equities_schwab_1234.json",
        {"timestamp_utc": now.isoformat(), "ok": True},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "session_ready_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["session_ready"]["summary"]["freshness_inferred_from_shadow_loop"] is True


def test_runtime_gate_dashboard_uses_day_based_units_for_optional_artifacts(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "official_macro_context_sync_latest.json",
        {"timestamp_utc": (now - timedelta(hours=10)).isoformat(), "ok": True, "sources": {"fed": {"ok": True}}},
    )
    _write_json(
        health_root / "live_macro_media_status.json",
        {"timestamp_utc": (now - timedelta(hours=12)).isoformat(), "ok": True, "learning_ready": True, "training_feature_count": 10},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["artifacts"]["official_macro_context_sync"]["stale"] is False
    assert payload["artifacts"]["live_macro_media"]["stale"] is False


def test_runtime_gate_dashboard_exposes_normalized_runtime_and_apple_fields(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(health_root / "session_ready_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "checks": []})
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 96.2, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "runtime_access_mode_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "mode": "portable",
            "ml_backend": "portable_auto",
            "portable_enabled": True,
            "backend_contract": {"effective_backend": "pytorch", "observation_only": True},
            "detected_backends": {"mlx": True, "pytorch": True},
        },
    )
    _write_json(
        health_root / "apple_silicon_profile_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "detected_tier": "max_throughput",
            "applied_tier": "max_throughput",
            "hardware": {"chip": "Apple M4 Max", "memory_gb": 64.0},
            "override_exists": True,
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "needs_attention",
            "blocking_reasons": ["promotion_not_ready"],
            "summary": {"confirmed_training_success": True, "target_count": 5, "trained_count": 5},
        },
    )
    _write_json(
        health_root / "training_quality_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "blocked",
            "training_quality_score": 61.5,
            "top_priorities": ["runtime_input_coverage", "stale_active_diagnostics"],
            "supportability": {"active_supportability_score": 40.0},
            "implemented_improvement_count": 17,
        },
    )
    _write_json(
        health_root / "platform_control_plane_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "institutional_readiness": {
                "overall_status": "upgrade_required",
                "overall_score": 63.4,
                "top_priorities": ["Publish a canonical feature-store manifest."],
                "weakest_domains": [{"slug": "security_and_compliance"}, {"slug": "developer_process"}],
                "domain_count": 12,
            },
        },
    )
    _write_json(
        health_root / "stale_artifact_sweeper_bot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "summary": {"candidate_files": 10, "staged_files": 8, "staged_bytes": 4096, "delete_errors": 0},
        },
    )
    _write_json(
        health_root / "stale_artifact_reaper_bot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "summary": {"candidate_files": 3, "deleted_files": 2, "deleted_bytes": 1024, "delete_errors": 0},
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["data_quality_score"] == 96.2
    assert payload["runtime"]["backend_contract"]["effective_backend"] == "pytorch"
    assert payload["apple_silicon"]["applied_tier"] == "max_throughput"
    assert payload["training"]["overall_status"] == "needs_attention"
    assert payload["training"]["quality_score"] == 61.5
    assert payload["platform"]["overall_status"] == "upgrade_required"
    assert payload["platform"]["overall_score"] == 63.4
    assert payload["artifacts"]["stale_artifact_sweeper_bot"]["summary"]["staged_files"] == 8
    assert payload["artifacts"]["stale_artifact_reaper_bot"]["summary"]["deleted_files"] == 2
    assert payload["artifacts"]["platform_control_plane"]["summary"]["weakest_domains"] == [
        "security_and_compliance",
        "developer_process",
    ]


def test_runtime_gate_dashboard_keeps_advisory_controls_from_degrading_operational_status(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running"},
    )
    _write_json(health_root / "daily_auto_verify_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"timestamp_utc": now.isoformat(), "ok": False, "overall_status": "needs_work"})
    _write_json(health_root / "retrain_artifact_freshness_latest.json", {"timestamp_utc": now.isoformat(), "ok": False, "failed_checks": ["paper_replay"]})
    _write_json(health_root / "infrastructure_autofix_bot_latest.json", {"timestamp_utc": now.isoformat(), "ok": False, "overall_status": "degraded"})
    _write_json(health_root / "rolling_restart_controller_latest.json", {"timestamp_utc": now.isoformat(), "ok": False, "overall_status": "blocked"})
    _write_json(
        health_root / "artifact_freshness_slo_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": False, "overall_status": "blocked", "sla_summary": {"stale_required": 1}},
    )
    _write_json(
        health_root / "runtime_snapshot_cache_control_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": False, "overall_status": "degraded", "cache_health": {"snapshot_ready": True}},
    )
    _write_json(
        health_root / "chaos_drill_coordinator_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": False, "overall_status": "degraded", "overdue_drills": [{"name": "restore"}]},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["overall"]["status"] == "ok"
    assert payload["overall"]["ok"] is True
    assert "memory_efficiency_control_needs_work" in payload["overall"]["attention_tiers"]["advisory"]
    assert "retrain_artifact_freshness_not_ok" in payload["overall"]["attention_tiers"]["advisory"]
    assert "infrastructure_autofix_bot_needs_work" in payload["overall"]["attention_tiers"]["advisory"]
    assert "artifact_freshness_slo_blocked" in payload["overall"]["attention_tiers"]["advisory"]
    assert "runtime_snapshot_cache_control_needs_work" in payload["overall"]["attention_tiers"]["advisory"]
    assert "chaos_drill_coordinator_needs_work" in payload["overall"]["attention_tiers"]["advisory"]
    assert payload["overall"]["attention_tiers"]["critical"] == []
    assert payload["overall"]["attention_tiers"]["degraded"] == []
    owners = {row["attention"]: row["owner"] for row in payload["overall"]["remediation_actions"]}
    assert owners["retrain_artifact_freshness_not_ok"] == "retrain_artifact_freshness_guard"
    assert owners["rolling_restart_controller_blocked"] == "rolling_restart_controller"
    assert owners["artifact_freshness_slo_blocked"] == "artifact_freshness_slo"
    assert owners["runtime_snapshot_cache_control_needs_work"] == "runtime_snapshot_cache_control"
    assert owners["chaos_drill_coordinator_needs_work"] == "chaos_drill_coordinator"


def test_runtime_gate_dashboard_manages_soak_deferred_controls_when_paper_guard_is_green(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running"},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "completed_checks": 50,
            "failed_checks": ["promotion_packet_builder", "artifact_freshness"],
        },
    )
    _write_json(
        health_root / "artifact_freshness_slo_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "overall_status": "ready", "sla_summary": {"stale_required": 0}},
    )
    _write_json(
        health_root / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        health_root / "runtime_paper_regression_guard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "paper_stage": "armed",
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )
    _write_json(health_root / "health_fast_latest.json", {"timestamp_utc": now.isoformat(), "overall_status": "ready", "ok": True})
    _write_json(walk_root / "promotion_readiness_latest.json", {"timestamp_utc": now.isoformat(), "promote_ok": False})
    _write_json(
        health_root / "bot_quality_autopilot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "blocked",
            "quality_blockers": {"quality_probation_bot_ids": ["bot_a"], "coverage_shortfall_bots": 4},
        },
    )
    _write_json(
        health_root / "runtime_snapshot_cache_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "degraded",
            "cache_health": {"snapshot_exists": True, "snapshot_ready": True},
            "upstream_inputs": {"retrain_artifact_freshness_ok": False, "coverage_shortfall_bots": 4},
        },
    )
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "degraded",
            "bench": {"active_supportable_bots": 1584, "bench_depth": 1592},
            "a_plus_contract": {
                "active_supportable_target": 3,
                "bench_depth_target": 5,
                "active_supportable_bots": 1584,
                "bench_depth": 1592,
                "coverage_shortfall_bots": 4,
            },
        },
    )
    _write_json(
        health_root / "chaos_drill_coordinator_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "blocked",
            "overdue_drills": [{"drill": "snapshot_restore"}, {"drill": "reboot_blackstart"}],
            "schedule_contract": {"discipline_ready": True},
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)
    managed = {row["attention"]: row["managed_control_state"] for row in payload["overall"]["managed_controls"]}

    assert payload["overall"]["status"] == "ok"
    assert payload["overall"]["ok"] is True
    assert payload["overall"]["attention"] == []
    assert payload["overall"]["raw_attention"] == []
    assert set(payload["overall"]["forensic_attention"]) >= {
        "promotion_not_ready",
        "daily_auto_verify_not_ok",
        "bot_quality_autopilot_blocked",
        "runtime_snapshot_cache_control_needs_work",
        "roster_resilience_planner_needs_work",
        "chaos_drill_coordinator_blocked",
    }
    assert managed["daily_auto_verify_not_ok"] == "daily_verify_training_promotion_checks_deferred_while_paper_soak_is_green"
    assert managed["promotion_not_ready"] == "promotion_deferred_while_paper_soak_is_green"
    assert managed["bot_quality_autopilot_blocked"] == "bot_quality_retrain_queue_deferred_while_training_budget_is_closed"
    assert managed["runtime_snapshot_cache_control_needs_work"] == "snapshot_cache_upstream_training_freshness_deferred_while_snapshot_is_ready"
    assert managed["roster_resilience_planner_needs_work"] == "roster_coverage_topoff_deferred_while_paper_soak_is_green"
    assert managed["chaos_drill_coordinator_blocked"] == "disruptive_recovery_drills_deferred_while_paper_soak_is_green"


def test_runtime_gate_dashboard_manages_stateful_sql_soft_quota_when_soak_storage_ready(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.0, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running"},
    )
    _write_json(health_root / "daily_auto_verify_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(
        health_root / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        health_root / "runtime_paper_regression_guard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "paper_stage": "armed",
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )
    _write_json(health_root / "health_fast_latest.json", {"timestamp_utc": now.isoformat(), "overall_status": "ready", "ok": True})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.667,
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 598,
                    "total_pending_lines": 598,
                    "oldest_pending_age_seconds": 160.133,
                }
            },
        },
    )
    _write_json(
        health_root / "storage_quota_guard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "overall_status": "degraded",
            "quota_summary": {
                "hard_breaches": 0,
                "soft_breaches": 1,
                "blocked_families": [],
                "degraded_families": ["sql_link_shards"],
            },
            "lanes": [
                {
                    "family": "sql_link_shards",
                    "status": "degraded",
                    "over_hard_gb": 0.0,
                    "hard_ratio": 0.879,
                }
            ],
        },
    )
    _write_json(
        health_root / "storage_retention_unison_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "continuous_run_contract": {
                "ready": True,
                "status": "ready",
                "storage_controls": {"quota_ready": True, "quota_status": "degraded"},
            },
            "storage_growth_forecast": {"status": "stable_or_improving", "days_until_pressure_free": None},
            "integration_contract": {"stateful_sql_compaction_only": True},
        },
    )
    _write_json(
        health_root / "storage_tier_policy_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "degraded",
            "manifest_backed_offload_contract": {
                "stateful_sql_policy": "checkpoint, vacuum, incremental vacuum, or verified mirror only; never source-delete from this policy"
            },
        },
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)
    managed = {row["attention"]: row["managed_control_state"] for row in payload["overall"]["managed_controls"]}

    assert payload["overall"]["status"] == "ok"
    assert payload["overall"]["ok"] is True
    assert payload["overall"]["attention"] == []
    assert payload["overall"]["raw_attention"] == []
    assert "storage_quota_guard_needs_work" in payload["overall"]["forensic_attention"]
    assert "storage_quota_guard_needs_work" in payload["overall"]["managed_attention"]
    assert managed["storage_quota_guard_needs_work"] == "soft_storage_quota_pressure_managed_by_ingestion_soak_contract"


def test_daily_auto_verify_uses_slow_timeout_for_heavy_checks() -> None:
    slow_timeout = 300

    assert daily_auto_verify._timeout_for_check("daily_runtime_summary", slow_timeout) == slow_timeout
    assert daily_auto_verify._timeout_for_check("data_source_divergence_bot", slow_timeout) == slow_timeout
    assert daily_auto_verify._timeout_for_check("replay_preopen_sanity", slow_timeout) == 45
    assert daily_auto_verify._timeout_for_check("resource_guard", slow_timeout) == daily_auto_verify.DEFAULT_CMD_TIMEOUT_SEC


def test_daily_auto_verify_active_progress_pid_requires_recent_live_pid(tmp_path: Path) -> None:
    progress_path = tmp_path / "daily_auto_verify_progress_latest.json"
    now = datetime.now(timezone.utc)
    _write_json(
        progress_path,
        {
            "timestamp_utc": now.isoformat(),
            "running": True,
            "pid": os.getpid(),
        },
    )

    active_pid = daily_auto_verify._active_progress_pid(progress_path, max_age_seconds=300)

    assert active_pid == os.getpid()


def test_daily_auto_verify_active_progress_pid_ignores_stale_progress(tmp_path: Path) -> None:
    progress_path = tmp_path / "daily_auto_verify_progress_latest.json"
    _write_json(
        progress_path,
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            "running": True,
            "pid": os.getpid(),
        },
    )

    active_pid = daily_auto_verify._active_progress_pid(progress_path, max_age_seconds=300)

    assert active_pid is None


def test_daily_auto_verify_main_skips_when_recent_progress_pid_is_alive(tmp_path: Path, monkeypatch, capsys) -> None:
    original_progress = daily_auto_verify.PROGRESS_PATH
    original_lock = daily_auto_verify.LOCK_PATH
    try:
        progress_path = tmp_path / "governance" / "health" / "daily_auto_verify_progress_latest.json"
        lock_path = tmp_path / "governance" / "locks" / "daily_auto_verify.lock"
        _write_json(
            progress_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "running": True,
                "pid": os.getpid(),
            },
        )
        daily_auto_verify.PROGRESS_PATH = progress_path
        daily_auto_verify.LOCK_PATH = lock_path
        monkeypatch.setattr(sys, "argv", ["daily_auto_verify.py", "--day", "20260327", "--json"])

        rc = daily_auto_verify.main()
        payload = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert payload["note"] == f"already_running_progress pid={os.getpid()}"
        assert payload["lock_path"] == str(lock_path)
        assert not lock_path.exists()
    finally:
        daily_auto_verify.PROGRESS_PATH = original_progress
        daily_auto_verify.LOCK_PATH = original_lock


def test_daily_auto_verify_recovers_stale_progress_to_latest(tmp_path: Path) -> None:
    progress_path = tmp_path / "governance" / "health" / "daily_auto_verify_progress_latest.json"
    latest_path = tmp_path / "governance" / "health" / "daily_auto_verify_latest.json"
    _write_json(
        progress_path,
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            "running": True,
            "pid": 999999,
            "current_check": "health_gates",
            "completed_checks": 36,
            "ok": True,
            "failed_checks": [],
            "checks": {"health_gates": {"ok": True}},
        },
    )
    _write_json(
        latest_path,
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(),
            "running": False,
            "ok": True,
            "failed_checks": [],
            "checks": {},
        },
    )

    note = daily_auto_verify._recover_stale_progress(progress_path, latest_path, max_age_seconds=300)
    recovered = json.loads(latest_path.read_text(encoding="utf-8"))

    assert "recovered_stale_progress" in note
    assert progress_path.exists() is False
    assert recovered["running"] is False
    assert recovered["ok"] is False
    assert "incomplete_run_recovered" in recovered["failed_checks"]
    assert recovered["current_check"] == "health_gates"


def test_health_gates_weights_data_blocked_more_than_risk_blocked(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.560000",
            "data_blocked_rate": "0.100000",
            "risk_blocked_rate": "0.460000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(
        sql_root / "daily_runtime_summary_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "watchdog": {"restarts": 0},
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 5.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 0,
            "pending_files": 0,
            "oldest_pending_age_seconds": 0.0,
            "overload": False,
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["hard_gate_triggered"] is False
    assert payload["inputs"]["combined_blocked_rate"] == 0.56
    assert abs(float(payload["inputs"]["blocked_rate"]) - 0.215) < 1e-9


def test_health_gates_falls_back_to_legacy_combined_blocked_rate(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.410000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(health_root / "ingestion_backpressure_latest.json", {"timestamp_utc": now.isoformat(), "overload": False})
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(sql_root / "daily_runtime_summary_latest.json", {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}})

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert payload["inputs"]["data_blocked_rate"] == 0.41
    assert payload["inputs"]["risk_blocked_rate"] == 0.0
    assert payload["inputs"]["blocked_rate"] == 0.41


def test_health_gates_fail_on_priority_shard_latency_and_storage(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.020000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(
        sql_root / "daily_runtime_summary_latest.json",
        {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 4.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 0,
            "pending_files": 0,
            "oldest_pending_age_seconds": 0.0,
            "overload": False,
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_crypto_explanations_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 334.0, "slo_breach_ratio_gt_300s": 0.14}}},
        },
    )
    _write_json(
        health_root / "sql_link_service_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "shard_hot_retention": [
                {
                    "shard": "crypto_explanations",
                    "db_size_gb_before": 26.184,
                    "db_size_gb_after": 26.184,
                    "max_db_gb": 8.0,
                    "trigger_reasons": [],
                    "skipped_reason": "below_data_trigger",
                }
            ],
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))
    stability = json.loads((health_root / "system_stability_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["hard_gate_triggered"] is False
    assert payload["hard_gates"]["priority_shard_latency"] is False
    assert payload["hard_gates"]["priority_shard_storage"] is False
    assert payload["priority_shards"][0]["shard"] == "crypto_explanations"
    assert payload["priority_shards"][0]["tier"] == "supporting"
    assert payload["priority_shards"][0]["size_over_max"] is True
    assert payload["priority_shards"][0]["recommended_action"] == "force_retention_and_throttle"
    assert payload["recommended_operating_mode"] == "live_full"
    assert payload["storage_pressure"]["retention_debt_gb"] == 18.184
    assert "force_priority_shard_retention" in payload["recommendations"]
    assert stability["safe_operating_envelope"] is True


def test_health_gates_prefers_live_shard_size_over_stale_retention_snapshot(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"
    shard_db = tmp_path / "data" / "sql_link_shards" / "jsonl_link_explanations.sqlite3"

    shard_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(shard_db))
    conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, payload_json TEXT)")
    conn.execute("INSERT INTO jsonl_records (payload_json) VALUES (?)", ("{}",))
    conn.commit()
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    conn.close()
    with shard_db.open("ab") as fh:
        fh.truncate((page_size * page_count) + (32 * 1024 * 1024))

    _write_json(
        health_root / "one_numbers_latest.json",
        {"timestamp_utc": now.isoformat(), "blocked_rate": 0.0, "window_seconds": 3600, "windows_total": 1, "windows_passed": 1},
    )
    _write_json(
        sql_root / "daily_runtime_summary_20260415.json",
        {"timestamp_utc": now.isoformat(), "watchdog_restarts": 0},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_data_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {"timestamp_utc": now.isoformat(), "pending_lines": 0, "pending_files": 0, "oldest_pending_age_seconds": 0.0, "overload": False},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_explanations_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 25.0, "slo_breach_ratio_gt_300s": 0.0}}},
        },
    )
    _write_json(
        health_root / "sql_link_service_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "shard_hot_retention": [
                {
                    "shard": "explanations",
                    "db_size_gb_before": 13.811,
                    "db_size_gb_after": 13.811,
                    "max_db_gb": 4.0,
                    "trigger_reasons": ["bootstrap_db_size_gb>=4"],
                    "skipped_reason": "",
                }
            ],
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))
    explanations = next(row for row in payload["priority_shards"] if row["shard"] == "explanations")

    assert rc == 0
    assert explanations["db_size_gb"] < 4.0
    assert explanations["storage_breached"] is False
    assert payload["storage_pressure"]["retention_debt_gb"] == 0.0


def test_health_gates_fail_on_sql_progress_stall_and_wal_pressure(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"
    primary_db = tmp_path / "data" / "jsonl_link.sqlite3"
    wal_path = tmp_path / "data" / "jsonl_link.sqlite3-wal"

    primary_db.parent.mkdir(parents=True, exist_ok=True)
    primary_db.write_bytes(b"db")
    wal_path.write_bytes(b"x" * 2_500_000)

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.010000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(
        sql_root / "daily_runtime_summary_latest.json",
        {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 2.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 0,
            "pending_files": 0,
            "oldest_pending_age_seconds": 0.0,
            "overload": False,
        },
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=3)).isoformat(),
            "running": True,
            "status": "running",
            "current_step": "merge_primary",
            "primary_db": str(primary_db),
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "health_gates.py",
            "--project-root",
            str(tmp_path),
            "--sql-progress-idle-seconds-limit",
            "60",
            "--sql-wal-size-gb-limit",
            "0.00000001",
        ],
    )
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))
    stability = json.loads((health_root / "system_stability_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["hard_gates"]["sql_progress_stall"] is True
    assert payload["hard_gates"]["sql_wal_pressure"] is True
    assert payload["inputs"]["sql_progress_age_seconds"] > 60
    assert payload["inputs"]["sql_wal_size_gb_live"] > 0.0
    assert stability["sql_pressure"]["progress_stalled"] is True
    assert stability["sql_pressure"]["wal_pressure"] is True


def test_health_gates_fail_on_required_collector_contracts(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.010000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(
        sql_root / "daily_runtime_summary_latest.json",
        {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 2.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 0,
            "pending_files": 0,
            "oldest_pending_age_seconds": 0.0,
            "overload": False,
        },
    )
    _write_json(
        health_root / "collector_contracts_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "required_failures": ["official_macro_context"],
            "soft_failures": ["options_flow_context"],
            "rows": [],
        },
    )
    _write_json(
        health_root / "storage_tier_policy_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pressure": {"hot_bytes": 10, "warm_bytes": 20, "cold_lane_bytes": 0},
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))
    stability = json.loads((health_root / "system_stability_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["hard_gates"]["collector_contracts"] is True
    assert payload["inputs"]["collector_required_failures"] == ["official_macro_context"]
    assert stability["collector_contracts"]["required_failures"] == ["official_macro_context"]


def test_health_gates_does_not_gate_on_non_severe_backpressure_overload(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.010000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(sql_root / "daily_runtime_summary_latest.json", {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}})
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 4.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 50,
            "pending_files": 1,
            "oldest_pending_age_seconds": 30.0,
            "overload": True,
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["hard_gates"]["ingestion_backpressure_overload"] is False
    assert payload["inputs"]["backpressure_overload"] is True
    assert payload["inputs"]["backpressure_overload_severe"] is False


def test_health_gates_uses_fresh_storage_control_queue_truth_over_stale_backpressure(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.010000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(sql_root / "daily_runtime_summary_latest.json", {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}})
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 4.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=40)).isoformat(),
            "pending_lines": 50000,
            "pending_files": 12,
            "oldest_pending_age_seconds": 1800.0,
            "overload": True,
            "line_pressure": True,
            "age_pressure": True,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "effective_raw_live_source": "raw_live_backpressure",
                "total_pending_lines": 5111,
                "core_pending_lines": 406,
                "oldest_pending_age_seconds": 0.0,
                "effective_raw_live": {
                    "total_pending_lines": 5111,
                    "core_pending_lines": 406,
                    "oldest_pending_age_seconds": 0.0,
                    "age_reconciled_from_stale_locator": True,
                    "age_reconciliation_source": "fresh_empty_sql_overlay",
                },
            },
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "steady_state": {
                "targets": {
                    "total_pending_lines": 15000,
                    "core_pending_lines": 5000,
                    "oldest_pending_age_seconds": 600,
                }
            },
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["hard_gates"]["ingestion_backpressure_overload"] is False
    assert payload["inputs"]["backpressure_overload"] is False
    assert payload["inputs"]["backpressure_pending_lines"] == 406
    assert payload["inputs"]["backpressure_oldest_pending_age_seconds"] == 0.0
    assert payload["inputs"]["backpressure_storage_control_override"]["active"] is True
    assert payload["inputs"]["backpressure_storage_control_override"]["reason"] == "fresh_storage_control_queue_clear"


def test_health_gates_accepts_guarded_age_clean_sql_overlay_storage_control() -> None:
    now = datetime.now(timezone.utc)

    override = health_gates._storage_control_backpressure_override(
        {
            "timestamp_utc": (now - timedelta(minutes=25)).isoformat(),
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {
                "overlay_adjusted": True,
                "overlay_pressure_clear": True,
                "effective_raw_live_source": "fresh_empty_sql_ingestion_overlay",
                "effective_raw_live": {
                    "total_pending_lines": 0,
                    "core_pending_lines": 0,
                    "oldest_pending_age_seconds": 0.0,
                    "source": "fresh_empty_sql_ingestion_overlay",
                },
            },
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
        }
    )

    assert override["active"] is True
    assert override["pending_lines_total"] == 0
    assert override["reason"] == "fresh_sql_overlay_clear"
