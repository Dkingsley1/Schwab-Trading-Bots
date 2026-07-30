import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import unattended_soak_readiness as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _ready_artifacts(project_root: Path, external_root: Path) -> None:
    health = project_root / "governance" / "health"
    now = src.iso_now()
    _write_json(
        health / "storage_mount_guard_latest.json",
        {"timestamp_utc": now, "external_root": str(external_root), "external_low_space": False},
    )
    _write_json(
        health / "storage_retention_unison_latest.json",
        {
            "timestamp_utc": now,
            "continuous_run_contract": {
                "status": "ready",
                "ready": True,
                "pressure_free_gb": 64.0,
                "safety_buffer_gb": 32.0,
                "effective_daily_growth_gb": 0.5,
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "continuous_run_soak_contract": {"status": "ready", "ready": True},
        },
    )
    _write_json(
        health / "bot_logs_cleanup_intelligence_latest.json",
        {"timestamp_utc": now, "projected_free_gb": 160.0, "selected_reclaimable_gb": 0.0},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {"timestamp_utc": now, "ok": True, "overall_status": "ready", "database_integrity_checks": []},
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "restart_storms": [], "alerts": []},
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {"timestamp_utc": now, "ok": True, "overall_status": "ready"},
    )
    _write_json(
        health / "broker_readiness_latest.json",
        {"timestamp_utc": now, "ready_for_open": True},
    )
    _write_json(
        health / "notification_escalation_ladder_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "attended_runtime_ready": True,
            "unattended_runtime_ready": True,
            "remote_pager_ready": True,
            "phone_bridge_ready": True,
            "critical_backlog": {"grouped_unsent_count": 0, "grouped_unacked_count": 0},
        },
    )
    _write_json(
        health / "livefeed_refresh_guard_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "health": {"ok": True},
            "blockers": [],
            "warnings": [],
        },
    )


def test_unattended_soak_readiness_ready_when_storage_power_runtime_and_alerts_clear(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n autopoweroff 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert payload["overall_score"] == 100.0
    assert payload["sections"]["artifact_freshness"]["score"] == 100.0
    assert payload["safe_to_leave_unattended"] is True
    assert payload["blockers"] == []
    assert payload["control_env"]["BOT_UNATTENDED_SOAK_READY"] == "1"


def test_unattended_soak_readiness_blocks_storage_margin_and_host_sleep(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 1\n disksleep 10\n standby 1\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 80.0, "used_pct": 92.0},
    )

    assert payload["overall_status"] == "blocked"
    assert "storage_margin_not_30_day_ready" in payload["blockers"]
    assert "host_sleep_not_disabled_on_ac" in payload["blockers"]
    assert "disk_sleep_not_disabled_on_ac" in payload["blockers"]
    assert payload["safe_to_leave_unattended"] is False


def test_unattended_soak_readiness_accepts_approved_cold_archive_spillover(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_retention_unison_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "continuous_run_contract": {
                "status": "watch",
                "ready": True,
                "pressure_free_gb": 64.0,
                "safety_buffer_gb": 32.0,
                "effective_daily_growth_gb": 0.5,
                "cold_archive_spillover_ready": True,
                "cold_archive_spillover_capacity_gb": 64.0,
                "cold_archive_adjusted_margin_gb": 58.0,
                "cold_archive_primary_pressure_buffer_gb": 16.0,
            },
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n autopoweroff 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 105.0, "used_pct": 88.0},
    )

    assert payload["overall_status"] == "ready"
    assert payload["sections"]["storage"]["available_margin_gb"] == -6.0
    assert payload["sections"]["storage"]["cold_archive_spillover_ready"] is True
    assert payload["sections"]["storage"]["cold_archive_adjusted_margin_gb"] == 58.0
    assert "approved_cold_archive_spillover" in payload["managed_controls"]
    assert "storage_margin_not_30_day_ready" not in payload["blockers"]


def test_unattended_soak_readiness_manages_bounded_ingestion_steady_state_watch(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    health = project_root / "governance" / "health"
    now = src.iso_now()
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.395,
            "continuous_run_soak_contract": {
                "status": "blocked",
                "ready": False,
                "soak_ready": False,
                "blockers": ["steady_state_targets_not_clear"],
                "warnings": ["drain_time_unknown", "storage_growth_baseline_watch"],
            },
            "steady_state": {
                "target_status": {
                    "steady_state_ready": False,
                    "target_breaches": ["pressure_index", "estimated_total_drain_minutes"],
                    "backlog_relief_a_plus_ready": True,
                    "backlog_relief_a_plus_plus_ready": True,
                }
            },
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 266,
                    "total_pending_lines": 1382,
                    "oldest_pending_age_seconds": 94.9,
                }
            },
            "stale_pending_locator": {"status": "clear"},
            "external_route_verification": {"verification_state": "ready"},
            "storage_resilience": {"overall_status": "ready"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n autopoweroff 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert payload["safe_to_leave_unattended"] is True
    assert "ingestion_soak_contract_not_ready" not in payload["blockers"]
    assert payload["sections"]["storage"]["ingestion_soak_ready"] is True
    assert payload["sections"]["storage"]["ingestion_managed_watch"] is True
    assert "ingestion_soak_contract_managed_by_bounded_backlog_watch" in payload["sections"]["storage"]["warnings"]
    assert "ingestion_soak_contract_managed_by_bounded_backlog_watch" in payload["sections"]["storage"]["managed_controls"]


def test_unattended_soak_readiness_manages_elevated_ingestion_when_bounded_drain_is_safe(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    health = project_root / "governance" / "health"
    now = src.iso_now()
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.979,
            "continuous_run_soak_contract": {
                "status": "blocked",
                "ready": False,
                "soak_ready": False,
                "blockers": ["steady_state_targets_not_clear", "backlog_relief_contract_active", "drain_time_above_target"],
                "inputs": {
                    "route_verified": True,
                    "resilience_status": "ready",
                    "collector_intake_status": "enforced",
                    "collector_intake_soak_safe": False,
                    "storage_efficiency_status": "ready",
                    "storage_efficiency_grade": "A+",
                    "a_plus_drain_time_horizon_ok": True,
                },
            },
            "steady_state": {
                "target_status": {
                    "steady_state_ready": False,
                    "target_breaches": ["pressure_index", "estimated_total_drain_minutes"],
                    "backlog_relief_a_plus_ready": False,
                    "backlog_relief_a_plus_plus_ready": False,
                }
            },
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 4938,
                    "total_pending_lines": 7751,
                    "oldest_pending_age_seconds": 234.997,
                }
            },
            "bounded_recovery_contract": {
                "active_drain_progress": True,
                "drain_delta_signal_observed": False,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "storage": {"efficiency_grade": "A+", "storage_plane_phase": "deep_cold_managed_steady_state"},
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
            "stale_pending_locator": {"status": "clear"},
            "external_route_verification": {"verification_state": "ready"},
            "storage_resilience": {"overall_status": "ready", "unresolved_split_brain_conflicts": 0},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n autopoweroff 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert payload["safe_to_leave_unattended"] is True
    assert payload["sections"]["storage"]["ingestion_managed_watch"] is True
    assert "ingestion_soak_contract_not_ready" not in payload["blockers"]


def test_unattended_soak_readiness_blocks_elevated_ingestion_without_bounded_drain_progress(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    health = project_root / "governance" / "health"
    now = src.iso_now()
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.979,
            "continuous_run_soak_contract": {
                "status": "blocked",
                "ready": False,
                "soak_ready": False,
                "blockers": ["steady_state_targets_not_clear", "backlog_relief_contract_active"],
            },
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 4938,
                    "total_pending_lines": 7751,
                    "oldest_pending_age_seconds": 234.997,
                }
            },
            "bounded_recovery_contract": {
                "active_drain_progress": False,
                "drain_delta_signal_observed": False,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
            "stale_pending_locator": {"status": "clear"},
            "external_route_verification": {"verification_state": "ready"},
            "storage_resilience": {"overall_status": "ready"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n autopoweroff 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "blocked"
    assert payload["sections"]["storage"]["ingestion_managed_watch"] is False
    assert "ingestion_soak_contract_not_ready" in payload["blockers"]


def test_unattended_soak_readiness_tracks_caffeinate_guard_as_managed_control(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 1\n disksleep 10\n standby 1\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="/usr/bin/caffeinate -dimsu\n",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert payload["sections"]["host_power"]["warnings"] == []
    assert "host_sleep_not_disabled_on_ac_overridden_by_caffeinate_guard" in payload["managed_controls"]
    assert payload["managed_warnings"] == []


def test_unattended_soak_readiness_allows_operator_approved_battery_runtime(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("BOT_UNATTENDED_SOAK_ALLOW_BATTERY", "1")
    monkeypatch.setenv("BOT_UNATTENDED_SOAK_BATTERY_REASON", "operator_approved_mobile_runtime_test")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n autopoweroff 0\n",
        pmset_batt_text="Now drawing from 'Battery Power'\n -InternalBattery-0; discharging; present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert "host_not_on_ac_power" not in payload["blockers"]
    assert "host_not_on_ac_power_operator_approved_battery_override" in payload["managed_warnings"]
    assert "host_not_on_ac_power_operator_approved_battery_override" in payload["managed_controls"]
    assert payload["sections"]["host_power"]["ac_attached"] is False
    assert payload["sections"]["host_power"]["battery_override_allowed"] is True
    assert payload["sections"]["host_power"]["battery_override_reason"] == "operator_approved_mobile_runtime_test"


def test_unattended_soak_readiness_treats_live_plane_ready_cold_lane_defer_as_managed_control(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {
            "timestamp_utc": "2026-07-02T12:00:00+00:00",
            "overall_status": "degraded",
            "live_plane": {
                "ready": True,
                "broker_ready": True,
                "session_ready": True,
                "live_lane_running": True,
            },
            "clearance_plan": {"clearance_state": "awaiting_cold_lane"},
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert "live_runtime_separation_degraded" not in payload["warnings"]
    assert "live_plane_ready_cold_lane_refresh_deferred" in payload["managed_controls"]


def test_unattended_soak_readiness_treats_paper_soak_live_read_only_cold_lane_defer_as_managed_control(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {
            "timestamp_utc": "2026-07-23T12:00:00+00:00",
            "overall_status": "degraded",
            "live_plane": {
                "ready": False,
                "broker_ready": True,
                "session_ready": True,
                "live_lane_running": False,
            },
            "release_contract": {"live_lane_should_be_read_only": True},
            "clearance_plan": {"clearance_state": "awaiting_cold_lane"},
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert "live_runtime_separation_degraded" not in payload["warnings"]
    assert "paper_soak_live_money_locked_cold_lane_deferred" in payload["managed_controls"]


def test_unattended_soak_readiness_treats_isolated_read_only_restart_storms_as_managed(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    _write_json(
        project_root / "governance" / "health" / "process_watchdog_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "restart_storms": [
                {
                    "name": "coinbase_loop",
                    "impact": "read_only_collection",
                    "quarantinable": True,
                    "blocks_execution_clear": False,
                }
            ],
            "restart_storm_isolation": {
                "isolated_count": 1,
                "execution_blocking_count": 0,
                "isolated_targets": ["coinbase_loop"],
                "execution_blocking_targets": [],
                "all_active_storms_isolated": True,
            },
            "alerts": [{"name": "coinbase_loop", "type": "restart_storm"}],
        },
    )
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"timestamp_utc": src.iso_now(), "overall_status": "degraded", "clearance_plan": {"clearance_state": "protect_live"}},
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    runtime = payload["sections"]["runtime_loops"]
    assert payload["overall_status"] == "ready"
    assert "process_watchdog_not_ready" not in payload["blockers"]
    assert "restart_storms_present" not in payload["blockers"]
    assert "process_watchdog_alerts_present" not in payload["warnings"]
    assert runtime["isolated_read_only_restart_storms"] is True
    assert "read_only_collection_restart_storms_isolated" in payload["managed_controls"]


def test_unattended_soak_readiness_allows_paper_soak_auth_grace(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    health = project_root / "governance" / "health"
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ok": False,
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 1365, "critical_lease_seconds": 600},
            "broker_state": {
                "broker_operable": True,
                "network_ok": True,
                "configured_for_refresh": True,
            },
        },
    )
    _write_json(
        health / "schwab_auth_supervisor_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "token": {"ready": True, "expires_in_seconds": 1365, "readiness_refresh_needed": False},
            "min_ready_expires_seconds": 900,
        },
    )
    _write_json(
        health / "broker_readiness_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ready_for_open": True,
            "auth_ok": False,
            "network_ok": True,
            "token_expires_in_seconds": 1365,
            "preflight_checks": {
                "token_exists": True,
                "token_ready_for_open": True,
                "readiness_refresh_needed_after": False,
            },
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert "auth_lease_not_ready" not in payload["blockers"]
    assert payload["sections"]["runtime_loops"]["strict_auth_ready"] is False
    assert payload["sections"]["runtime_loops"]["paper_soak_auth_ready"] is True


def test_unattended_soak_readiness_blocks_missing_unattended_pager(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    _write_json(
        project_root / "governance" / "health" / "notification_escalation_ladder_latest.json",
        {
            "timestamp_utc": "2026-07-02T12:00:00+00:00",
            "overall_status": "ready",
            "attended_runtime_ready": True,
            "unattended_runtime_ready": False,
            "remote_pager_ready": False,
            "phone_bridge_ready": True,
            "critical_backlog": {"grouped_unsent_count": 0, "grouped_unacked_count": 0},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "livefeed_refresh_guard_latest.json",
        {
            "timestamp_utc": "2026-07-02T12:00:00+00:00",
            "ok": False,
            "overall_status": "blocked",
            "health": {"ok": False},
            "blockers": ["livefeed_remote_viewer_not_ready"],
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "blocked"
    assert "unattended_remote_pager_not_ready" in payload["blockers"]
    assert "phone_bridge_ready_but_remote_pager_missing" in payload["warnings"]


def test_unattended_soak_readiness_accepts_mobile_operator_coverage_without_remote_pager(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    _ready_artifacts(project_root, external_root)
    _write_json(
        project_root / "governance" / "health" / "notification_escalation_ladder_latest.json",
        {
            "timestamp_utc": "2026-07-02T12:00:00+00:00",
            "overall_status": "ready",
            "attended_runtime_ready": True,
            "unattended_runtime_ready": False,
            "remote_pager_ready": False,
            "phone_bridge_ready": True,
            "critical_backlog": {"grouped_unsent_count": 0, "grouped_unacked_count": 0},
        },
    )
    monkeypatch.setattr(src.platform, "system", lambda: "Darwin")

    payload = src.build_payload(
        project_root,
        pmset_custom_text="AC Power:\n sleep 0\n disksleep 0\n standby 0\n",
        pmset_batt_text="Now drawing from 'AC Power'\n -InternalBattery-0; AC attached; not charging present: true",
        process_text="",
        disk_snapshot_fn=lambda path: {"path": str(path), "exists": True, "free_gb": 160.0, "used_pct": 60.0},
    )

    assert payload["overall_status"] == "ready"
    assert "unattended_remote_pager_not_ready" not in payload["blockers"]
    assert payload["sections"]["alerting"]["mobile_operator_coverage_ready"] is True
    assert payload["sections"]["alerting"]["zero_touch_unattended_ready"] is False
    assert payload["sections"]["alerting"]["operator_coverage_model"] == "daily_supervised_mobile_operator"
    assert "zero_touch_remote_pager_missing_mobile_operator_coverage_active" not in payload["warnings"]
    assert payload["managed_warnings"] == []
    assert "daily_mobile_operator_coverage_active_without_zero_touch_remote_pager" in payload["managed_controls"]
