import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import artifact_freshness_slo as freshness_src
from scripts.ops import auth_lease_manager as auth_src
from scripts.ops import blackstart_recovery as blackstart_src
from scripts.ops import chaos_drill_coordinator as chaos_src
from scripts.ops import live_runtime_separation_control as separation_src
from scripts.ops import operator_cockpit as cockpit_src
from scripts.ops import release_freeze_guard as freeze_src
from scripts.ops import remote_alert_control as alert_src
from scripts.ops import rolling_restart_controller as restart_src
from scripts.ops import roster_resilience_planner as roster_src
from scripts.ops import runtime_gate_dashboard as dashboard_src
from scripts.ops import runtime_snapshot_cache_control as snapshot_cache_src
from scripts.ops import sleeve_isolation_guard as isolation_src
from scripts.ops import storage_quota_guard as quota_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_long_runtime_runtime_controls_surface_pressure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {"overall_status": "blocked", "snapshot_ready": False, "precompute_targets": [{"bot_id": "bot_a"}]},
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 3, "seed_queue": [{"bot_id": "bot_a"}]},
    )
    _write_json(
        health / "storage_tier_policy_latest.json",
        {"overall_status": "blocked", "pressure": {"hot_path_over_budget_bytes": 25}},
    )
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 9.5, "memory_pressure_state": "yellow"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [{"service": "sql"}]})
    _write_json(health / "session_ready_latest.json", {"timestamp_utc": "2026-04-09T14:00:00+00:00", "ok": True})
    _write_json(
        project_root / "exports" / "state_snapshot_drills" / "latest.json",
        {"timestamp_utc": "2026-04-09T15:45:00+00:00", "files_checked": 4, "missing_files": []},
    )
    _write_json(
        health / "premarket_token_guard_latest.json",
        {
            "timestamp_utc": auth_src.iso_now(),
            "ok": True,
            "network": {"ok": True},
            "auth": {"ok": True},
            "token_before": {"exists": True, "expires_in_seconds": 500},
            "token_after": {"exists": True, "expires_in_seconds": 500},
        },
    )
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": False})
    _write_json(health / "reboot_resilience_latest.json", {"ok": False, "recovered": []})
    _write_json(health / "storage_resilience_control_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})

    separation = separation_src.build_payload(project_root)
    restart = restart_src.build_payload(project_root)
    auth = auth_src.build_payload(project_root)
    blackstart = blackstart_src.build_payload(project_root)

    assert separation["overall_status"] == "blocked"
    assert separation["clearance_plan"]["clearance_state"] == "awaiting_cold_lane"
    assert separation["clearance_plan"]["cold_lane_refresh"]["refresh_required"] is True
    assert separation["clearance_plan"]["coverage_gap_closer"]["shortfall_bots"] == 3
    assert separation["release_contract"]["shared_host_training_resume_allowed"] is False
    assert restart["recommended_scope"] in {"worker_only", "full_stack"}
    assert auth["overall_status"] == "blocked"
    assert blackstart["overall_status"] == "blocked"
    assert blackstart["blocked_stage_count"] >= 1
    assert blackstart["recovery_contract"]["restart_sanity_ready"] is False


def test_live_runtime_separation_uses_storage_control_steady_state_to_bound_hot_path_pressure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready", "snapshot_ready": True, "precompute_targets": []})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0})
    _write_json(
        health / "storage_tier_policy_latest.json",
        {"overall_status": "blocked", "pressure": {"hot_path_over_budget_bytes": 25}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure_quality_score": 100.0,
            "recovery_quality_score": 96.0,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "external_route_verification": {"verification_state": "ready"},
        },
    )
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0, "memory_pressure_state": "green"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["shared_host_pressure"]["contention_score"] == 0
    assert payload["shared_host_pressure"]["signals"]["storage_hot_path_blocked"] is False
    assert payload["shared_host_pressure"]["signals"]["storage_hot_path_bounded_by_control"] is True


def test_live_runtime_separation_ignores_isolated_read_only_restart_storm_contention(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready", "snapshot_ready": True, "precompute_targets": []})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready", "pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure_quality_score": 100.0,
            "recovery_quality_score": 96.0,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "external_route_verification": {"verification_state": "ready"},
        },
    )
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0, "memory_pressure_state": "green"})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [{"name": "coinbase_loop", "quarantinable": True, "blocks_execution_clear": False}],
            "restart_storm_isolation": {
                "isolated_count": 1,
                "execution_blocking_count": 0,
                "isolated_targets": ["coinbase_loop"],
                "execution_blocking_targets": [],
                "all_active_storms_isolated": True,
            },
        },
    )

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["shared_host_pressure"]["contention_score"] == 0
    assert payload["shared_host_pressure"]["restart_storms"] == 1
    assert payload["shared_host_pressure"]["restart_storm_contention_count"] == 0
    assert payload["shared_host_pressure"]["signals"]["restart_storm_present"] is False
    assert payload["shared_host_pressure"]["signals"]["isolated_read_only_restart_storms"] is True


def test_live_runtime_separation_bounds_near_steady_state_storage_when_raw_live_is_clear(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready", "snapshot_ready": True, "precompute_targets": []})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0})
    _write_json(
        health / "storage_tier_policy_latest.json",
        {"overall_status": "blocked", "pressure": {"hot_path_over_budget_bytes": 38_954_508_830}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.297,
            "backpressure_quality_score": 93.42,
            "recovery_quality_score": 96.0,
            "steady_state": {
                "target_status": {
                    "steady_state_ready": False,
                    "target_breach_count": 1,
                    "target_breaches": ["pressure_index"],
                }
            },
            "external_route_verification": {"verification_state": "ready"},
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 4462,
                    "total_pending_lines": 5018,
                    "oldest_pending_age_seconds": 0.021,
                }
            },
        },
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"runtime_snapshot": {"storage_pressure": {"overlay_capacity_relief": False}}})
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0, "memory_pressure_state": "green"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["clearance_plan"]["clearance_state"] == "ready"
    assert payload["shared_host_pressure"]["storage_steady_state_strict_ready"] is False
    assert payload["shared_host_pressure"]["storage_near_steady_state_ready"] is True
    assert payload["shared_host_pressure"]["signals"]["storage_hot_path_blocked"] is False
    assert payload["shared_host_pressure"]["signals"]["storage_hot_path_bounded_by_control"] is True


def test_live_runtime_separation_treats_cool_raw_live_sql_overlay_as_guarded_read_only(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "blocked", "snapshot_ready": False, "precompute_targets": []})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0})
    _write_json(
        health / "storage_tier_policy_latest.json",
        {"overall_status": "blocked", "pressure": {"hot_path_over_budget_bytes": 25}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 374,
                    "total_pending_lines": 374,
                    "oldest_pending_age_seconds": 0.0,
                },
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "throttle_profile": "sustain",
            "runtime_snapshot": {"storage_pressure": {"overlay_capacity_relief": True}},
        },
    )
    _write_json(health / "cold_lane_refresh_latest.json", {"ok": True, "reason": "fresh_strategy_research_reused"})
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0, "memory_pressure_state": "green"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["clearance_plan"]["clearance_state"] == "guarded_live_read_only"
    assert payload["shared_host_pressure"]["signals"]["storage_hot_path_blocked"] is False
    assert payload["shared_host_pressure"]["signals"]["storage_hot_path_bounded_by_control"] is True
    assert payload["shared_host_pressure"]["storage_overlay_relief"]["active"] is True
    assert payload["release_contract"]["shared_host_training_resume_allowed"] is False


def test_live_runtime_separation_treats_staged_cold_lane_defer_as_managed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "degraded", "snapshot_ready": True, "precompute_targets": []})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready", "pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.9, "memory_pressure_state": "green"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "cold_lane_refresh_latest.json", {"ok": True, "reason": "resource_guard_blocked", "skipped": True})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4})
    _write_json(
        walk / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "overall_status": "degraded",
                "launch_state": "stage_only_training_blocked",
                "can_launch_now": False,
                "can_auto_launch_off_hours": False,
                "next_action": "keep candidates staged until the cold lane opens",
            },
        },
    )

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["clearance_plan"]["clearance_state"] == "managed_cold_lane_deferred"
    assert payload["shared_host_pressure"]["managed_cold_lane_deferred"] is True
    assert payload["release_contract"]["live_lane_should_be_read_only"] is True
    assert payload["release_contract"]["shared_host_training_resume_allowed"] is False


def test_live_runtime_separation_treats_staged_coverage_budget_wait_as_managed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "degraded", "snapshot_ready": True, "precompute_targets": []})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready", "pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0, "memory_pressure_state": "green"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "cold_lane_refresh_latest.json", {"ok": True, "reason": "fresh_strategy_research_reused", "skipped": True})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4})
    _write_json(
        walk / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "overall_status": "degraded",
                "launch_state": "stage_only_training_blocked",
                "can_launch_now": False,
                "can_auto_launch_off_hours": False,
                "off_hours_preferred": True,
                "launch_contract": {
                    "training_launch_blocked": True,
                    "launch_guard": "off_hours_only",
                },
                "next_action": "keep candidates staged until the cold lane opens",
            },
        },
    )

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["clearance_plan"]["clearance_state"] == "managed_coverage_stage_deferred"
    assert payload["shared_host_pressure"]["managed_coverage_stage_deferred"] is True
    assert payload["release_contract"]["live_lane_should_be_read_only"] is True
    assert payload["release_contract"]["shared_host_training_resume_allowed"] is False


def test_live_runtime_separation_treats_ok_cold_lane_refresh_as_managed_stage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": True,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": True,
        },
    )
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "degraded", "snapshot_ready": True})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready", "pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "cold_lane_refresh_latest.json", {"ok": True, "reason": "ok", "ran": True})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4})
    _write_json(
        walk / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "overall_status": "degraded",
                "launch_state": "stage_only_training_blocked",
                "off_hours_preferred": True,
                "launch_contract": {"training_launch_blocked": True, "launch_guard": "off_hours_only"},
            },
        },
    )

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["clearance_plan"]["clearance_state"] == "managed_coverage_stage_deferred"


def test_live_runtime_separation_accepts_guarded_paper_soak_when_live_lane_is_intentionally_off(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "timestamp_utc": separation_src.iso_now(),
            "ok": False,
            "broker_ready": True,
            "session_ready": True,
            "live_lane_running": False,
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "overall_status": "ready",
            "paper_armed": True,
            "paper_blocked": False,
        },
    )
    _write_json(health / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "blockers": []})
    _write_json(health / "health_fast_latest.json", {"overall_status": "ready", "strict_all_clear": True})
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready", "snapshot_ready": True})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready", "pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(health / "resource_guard_latest.json", {"swap_used_gb": 0.0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "cold_lane_refresh_latest.json", {"ok": True, "reason": "ok", "ran": True})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4})
    _write_json(
        walk / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "overall_status": "degraded",
                "launch_state": "stage_only_training_blocked",
                "off_hours_preferred": True,
                "launch_contract": {"training_launch_blocked": True, "launch_guard": "off_hours_only"},
            },
        },
    )

    payload = separation_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["live_plane"]["ready"] is False
    assert payload["live_plane"]["guarded_paper_soak_ready"] is True
    assert payload["clearance_plan"]["clearance_state"] == "managed_coverage_stage_deferred"
    assert payload["shared_host_pressure"]["managed_coverage_stage_deferred"] is True


def test_blackstart_treats_warning_lease_as_operable_when_broker_is_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "reboot_resilience_latest.json", {"ok": True, "recovered": []})
    _write_json(health / "session_ready_latest.json", {"timestamp_utc": "2026-04-09T14:00:00+00:00", "ok": True})
    _write_json(health / "live_readiness_smoke_latest.json", {"timestamp_utc": "2026-04-09T14:05:00+00:00", "ok": True})
    _write_json(health / "storage_resilience_control_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "ok": False,
            "overall_status": "degraded",
            "lease_state": "warning",
            "broker_state": {
                "broker_ready": True,
                "network_ok": True,
                "auth_ok": True,
            },
        },
    )

    blackstart = blackstart_src.build_payload(project_root)

    auth_stage = next(row for row in blackstart["stages"] if row["name"] == "auth_lease")
    assert auth_stage["ok"] is True
    assert blackstart["overall_status"] in {"ready", "degraded"}


def test_auth_lease_default_accepts_fresh_schwab_half_hour_token(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "premarket_token_guard_latest.json",
        {
            "timestamp_utc": auth_src.iso_now(),
            "ok": True,
            "network": {"ok": True},
            "auth": {"ok": True},
            "token_before": {"exists": True, "expires_in_seconds": 1790},
            "token_after": {"exists": True, "expires_in_seconds": 1790},
        },
    )
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    auth = auth_src.build_payload(project_root)

    assert auth["overall_status"] == "ready"
    assert auth["lease_state"] == "healthy"
    assert auth["lease_budget"]["min_lease_seconds"] == 1200


def test_auth_lease_treats_probe_denied_but_token_operable_as_warning(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "premarket_token_guard_latest.json",
        {
            "timestamp_utc": auth_src.iso_now(),
            "ok": True,
            "network": {"ok": True},
            "auth": {"ok": False, "reason": "account_probe_failed:403"},
            "token_before": {"exists": True, "expires_in_seconds": 1365},
            "token_after": {"exists": True, "expires_in_seconds": 1365},
        },
    )
    _write_json(
        health / "broker_readiness_latest.json",
        {
            "ready_for_open": True,
            "auth_ok": False,
            "network_ok": True,
            "account_probe_status_code": 403,
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    auth = auth_src.build_payload(project_root)

    assert auth["overall_status"] == "degraded"
    assert auth["lease_state"] == "warning"
    assert auth["lease_budget"]["token_lease_grace"] is True
    assert auth["broker_state"]["auth_ok"] is False


def test_auth_lease_uses_off_hours_probe_grace_for_short_schwab_lease(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    monkeypatch.setattr(
        auth_src,
        "_market_window",
        lambda: {
            "timezone": "America/New_York",
            "local_time": "2026-05-02T09:15:00-04:00",
            "is_weekend": True,
            "regular_session_open": False,
            "off_hours": True,
        },
    )
    _write_json(
        health / "premarket_token_guard_latest.json",
        {
            "timestamp_utc": auth_src.iso_now(),
            "token_before": {"exists": True},
            "token_after": {"exists": True, "expires_in_seconds": 120},
            "network": {"ok": True},
            "auth": {"ok": False, "reason": "auth_succeeded_but_token_not_ready:token_expiring_soon:120.0"},
        },
    )
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": False, "auth_ok": False, "network_ok": True, "account_probe_status_code": 200})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    auth = auth_src.build_payload(project_root)

    assert auth["overall_status"] == "degraded"
    assert auth["lease_state"] == "warning"
    assert auth["lease_budget"]["off_hours_probe_grace"] is True
    assert auth["broker_state"]["auth_probe_ok"] is True


def test_auth_lease_treats_off_hours_probe_backed_comfort_floor_as_ready(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    monkeypatch.setattr(
        auth_src,
        "_market_window",
        lambda: {
            "timezone": "America/New_York",
            "local_time": "2026-05-02T18:15:00-04:00",
            "is_weekend": True,
            "regular_session_open": False,
            "off_hours": True,
        },
    )
    _write_json(
        health / "premarket_token_guard_latest.json",
        {
            "timestamp_utc": auth_src.iso_now(),
            "token_before": {"exists": True},
            "token_after": {"exists": True, "expires_in_seconds": 900},
            "network": {"ok": True},
            "auth": {"ok": True, "reason": "auth_success"},
        },
    )
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True, "account_probe_status_code": 200})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    auth = auth_src.build_payload(project_root)

    assert auth["overall_status"] == "ready"
    assert auth["lease_state"] == "healthy"
    assert auth["lease_budget"]["off_hours_probe_grace"] is True


def test_long_runtime_storage_and_freshness_controls(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "quarantine_pressure_latest.json", {"quarantine_events": 200, "top_symbols": [{"symbol": "SPY", "count": 7}]})
    _write_json(health / "daily_auto_verify_latest.json", {"failed_checks": ["promotion_quality_gate"]})
    _write_json(
        health / "lane_thaw_controller_latest.json",
        {
            "overall_status": "blocked",
            "systemic_guardrails": {
                "global_killswitch_active": True,
                "risk_halt_events": 1,
                "account_snapshot_failure_count": 1,
                "write_failure_count": 0,
                "queue_depth": 15000,
            },
            "lanes": [
                {
                    "lane": "aggressive_equities_schwab",
                    "thaw_state": "candidate",
                    "thaw_contract": {"stage": "supervised_canary", "release_ready": True},
                }
            ],
            "blocked": [{"lane": "aggressive_equities_schwab", "decision": "hold"}],
        },
    )
    _write_json(
        health / "data_ingress_latest_aggressive_equities_schwab.json",
        {"profile": "aggressive", "domain": "equities", "broker": "schwab", "loop_state": "paused_anomaly_killswitch", "pause_reason": "anomaly"},
    )
    _write_json(
        health / "data_ingress_latest_crypto_futures_crypto_coinbase.json",
        {"profile": "crypto_futures", "domain": "crypto", "broker": "coinbase", "loop_state": "running"},
    )
    _write_json(health / "session_ready_latest.json", {"timestamp_utc": "2026-04-06T00:00:00+00:00", "ok": True})
    _write_json(health / "process_watchdog_latest.json", {"timestamp_utc": "2026-04-06T00:00:00+00:00", "status": []})
    _write_json(health / "live_readiness_smoke_latest.json", {"timestamp_utc": "2026-04-01T00:00:00+00:00", "ok": True})
    _write_json(
        health / "runtime_training_snapshot_latest.json",
        {"timestamp_utc": "2026-04-01T00:00:00+00:00", "row_count": 10, "sequence_count": 2, "coverage": {"top_sequences": []}},
    )
    _write_json(health / "training_runtime_control_latest.json", {"snapshot_ready": False, "precompute_targets": []})
    _write_json(health / "retrain_artifact_freshness_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4})
    _write_json(
        health / "storage_tier_policy_latest.json",
        {
            "by_family": {
                "sql_link_shards": {"bytes": 390 * 1024**3},
                "decision_explanations": {"bytes": 50 * 1024**3},
                "decisions": {"bytes": 10 * 1024**3},
                "content_store": {"bytes": 1 * 1024**3},
            },
            "by_service_role": {"governance_telemetry": {"bytes": 9 * 1024**3}},
        },
    )

    isolation = isolation_src.build_payload(project_root)
    freshness = freshness_src.build_payload(project_root)
    snapshot_cache = snapshot_cache_src.build_payload(project_root)
    quota = quota_src.build_payload(project_root)

    assert isolation["overall_status"] == "blocked"
    assert isolation["blast_radius_score"] < 100.0
    assert isolation["repeatable_thaw_contract"]["ready"] is False
    assert isolation["repeatable_thaw_contract"]["supervised_candidate_count"] == 1
    assert isolation["systemic_guardrails"]["global_killswitch_active"] is True
    assert isolation["gates"]["unresolved_daily_verify_checks"] == []
    assert freshness["overall_status"] == "blocked"
    assert snapshot_cache["overall_status"] == "blocked"
    assert quota["overall_status"] == "blocked"


def test_sleeve_isolation_excludes_session_pauses_and_resolves_repaired_daily_checks(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "quarantine_pressure_latest.json", {"quarantine_events": 0})
    _write_json(
        health / "daily_auto_verify_latest.json",
        {
            "ok": False,
            "running": False,
            "completed_checks": 6,
            "note": "recovered_stale_progress",
            "failed_checks": ["bot_support_owner_guard", "data_source_divergence_bot", "incomplete_run_recovered"],
        },
    )
    _write_json(health / "bot_support_owner_guard_latest.json", {"ok": True})
    _write_json(health / "data_source_divergence_latest.json", {"ok": True})
    _write_json(
        health / "data_ingress_latest_aggressive_equities_schwab.json",
        {"profile": "aggressive", "domain": "equities", "broker": "schwab", "loop_state": "paused_session_gate", "pause_reason": "weekend"},
    )
    _write_json(
        health / "data_ingress_latest_dividend_equities_schwab.json",
        {"profile": "dividend", "domain": "equities", "broker": "schwab", "loop_state": "paused_session_gate", "pause_reason": "post_window"},
    )
    _write_json(
        health / "data_ingress_latest_intraday_equities_schwab.json",
        {"profile": "intraday", "domain": "equities", "broker": "schwab", "loop_state": "paused_anomaly_killswitch", "pause_reason": "data_anomaly"},
    )
    _write_json(
        health / "data_ingress_latest_crypto_coinbase.json",
        {"profile": "default", "domain": "crypto", "broker": "coinbase", "loop_state": "running"},
    )

    isolation = isolation_src.build_payload(project_root)

    assert isolation["overall_status"] == "degraded"
    assert isolation["sleeve_matrix"]["isolated_lane_count"] == 1
    assert isolation["sleeve_matrix"]["session_paused_lane_count"] == 2
    assert isolation["sleeve_matrix"]["running_lane_count"] == 1
    assert isolation["gates"]["unresolved_daily_verify_checks"] == []


def test_storage_quota_guard_recommends_actions_for_decision_and_governance_breaches(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_tier_policy_latest.json",
        {
            "by_family": {
                "decisions": {"bytes": 64 * 1024**3},
                "decision_explanations": {"bytes": 3 * 1024**3},
                "sql_link_shards": {"bytes": 8 * 1024**3},
                "content_store": {"bytes": 1 * 1024**3},
            },
            "by_service_role": {"governance_telemetry": {"bytes": 193 * 1024**3}},
        },
    )

    quota = quota_src.build_payload(project_root)
    summary = quota["quota_summary"]
    actions = quota["recommended_actions"]
    decisions = next(row for row in quota["lanes"] if row["family"] == "decisions")
    governance = next(row for row in quota["lanes"] if row["family"] == "governance_telemetry")

    assert quota["overall_status"] == "blocked"
    assert summary["hard_breaches"] == 2
    assert summary["blocked_families"] == ["decisions", "governance_telemetry"]
    assert summary["worst_over_hard_gb"] > 100
    assert decisions["over_hard_gb"] == 28.0
    assert governance["over_hard_gb"] == 181.0
    assert any("core decision drainer" in action for action in actions)
    assert any("verbose governance telemetry" in action for action in actions)
    assert any("heavy training gated" in action for action in actions)


def test_alert_freeze_roster_and_chaos_controls(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    watchdog = project_root / "governance" / "watchdog"
    health = project_root / "governance" / "health"
    ack_state = project_root / "governance" / "watchdog" / "remote_alert_ack_state.json"
    window_path = project_root / "governance" / "runtime" / "release_freeze_window.json"
    chaos_state = project_root / "governance" / "runtime" / "chaos_drill_state.json"

    _write_jsonl(
        watchdog / "pager_alerts.jsonl",
        [
            {"timestamp_utc": "2026-04-09T16:00:00+00:00", "severity": "critical", "event": "token_expiry", "message": "token low", "sent": True, "suppressed": False},
            {"timestamp_utc": "2026-04-09T16:10:00+00:00", "severity": "critical", "event": "storage_pressure", "message": "disk high", "sent": False, "suppressed": False},
        ],
    )
    _write_json(ack_state, {"events": {"token_expiry": {"acknowledged_at_utc": "2026-04-09T16:05:00+00:00"}}})
    _write_json(
        window_path,
        {"active": True, "started_at_utc": "2026-04-09T00:00:00+00:00", "ends_at_utc": "2026-12-30T00:00:00+00:00", "reason": "runtime_window"},
    )
    _write_json(
        health / "supportability_control_latest.json",
        {
            "overall_status": "blocked",
            "supportability": {"active_bots": 1, "active_supportable_bots": 0},
            "teacher_student": {"teacher_count": 0, "students_without_teachers": 5},
        },
    )
    _write_json(health / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4, "standing_queue": {"seed_queue_size": 7}})
    _write_json(project_root / "governance" / "walk_forward" / "new_bot_graduation_latest.json", {"maturity": {"mature_bots": 0}})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json", {"promote_ok": False})
    _write_json(chaos_state, {"drills": {"snapshot_restore": {"completed_at_utc": "2026-03-20T00:00:00+00:00"}}})
    _write_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", {"timestamp_utc": "2026-03-20T00:00:00+00:00"})
    _write_json(health / "reboot_resilience_latest.json", {"timestamp_utc": "2026-03-20T00:00:00+00:00"})
    _write_json(health / "storage_resilience_control_latest.json", {"timestamp_utc": "2026-03-20T00:00:00+00:00"})
    _write_json(health / "premarket_token_guard_latest.json", {"timestamp_utc": "2026-03-20T00:00:00+00:00"})
    _write_json(health / "process_watchdog_latest.json", {"timestamp_utc": "2026-03-20T00:00:00+00:00"})
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "install_weekly_dr_drill_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (scripts_dir / "daily_state_snapshot_drill.py").write_text("print('ok')\n", encoding="utf-8")
    (scripts_dir / "backup_restore_verify.py").write_text("print('ok')\n", encoding="utf-8")

    alerts = alert_src.build_payload(project_root, ack_state_path=ack_state)
    freeze = freeze_src.build_payload(project_root, window_path=window_path)
    roster = roster_src.build_payload(project_root)
    chaos = chaos_src.build_payload(project_root, state_path=chaos_state)

    assert alerts["overall_status"] == "blocked"
    assert freeze["window"]["active"] is True
    assert roster["overall_status"] == "blocked"
    assert chaos["overall_status"] == "blocked"
    assert chaos["restore_discipline"]["snapshot_restore_present"] is True
    assert chaos["schedule_contract"]["weekly_drill_installer_present"] is True


def test_chaos_drill_coordinator_degrades_overdue_drills_when_discipline_is_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    state_path = project_root / "governance" / "runtime" / "chaos_drill_state.json"
    old_stamp = "2026-03-20T00:00:00+00:00"

    _write_json(state_path, {"drills": {}})
    _write_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", {"timestamp_utc": old_stamp})
    _write_json(health / "reboot_resilience_latest.json", {"timestamp_utc": old_stamp})
    _write_json(health / "storage_resilience_control_latest.json", {"timestamp_utc": old_stamp})
    _write_json(health / "premarket_token_guard_latest.json", {"timestamp_utc": old_stamp})
    _write_json(health / "process_watchdog_latest.json", {"timestamp_utc": old_stamp})
    backup_events = project_root / "governance" / "watchdog" / "backup_restore_events.jsonl"
    backup_events.parent.mkdir(parents=True, exist_ok=True)
    backup_events.write_text("{}\n", encoding="utf-8")

    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "install_weekly_dr_drill_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (scripts_dir / "daily_state_snapshot_drill.py").write_text("print('ok')\n", encoding="utf-8")
    (scripts_dir / "backup_restore_verify.py").write_text("print('ok')\n", encoding="utf-8")

    payload = chaos_src.build_payload(project_root, state_path=state_path, overdue_days=7.0)

    assert payload["overall_status"] == "degraded"
    assert len(payload["overdue_drills"]) >= 2
    assert payload["restore_discipline"]["restore_proof_ready"] is True
    assert payload["schedule_contract"]["discipline_ready"] is True


def test_operator_cockpit_and_dashboard_include_long_runtime_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"

    _write_json(health / "session_ready_latest.json", {"timestamp_utc": "2026-04-09T16:00:00+00:00", "ok": True, "checks": []})
    _write_json(health / "health_gates_latest.json", {"timestamp_utc": "2026-04-09T16:00:00+00:00", "hard_gate_triggered": False, "data_quality_score": 91.0, "inputs": {}})
    _write_json(health / "sql_link_service_latest.json", {"timestamp_utc": "2026-04-09T16:00:00+00:00", "running": True})
    _write_json(health / "jsonl_sql_ingestion_health_latest.json", {"timestamp_utc": "2026-04-09T16:00:00+00:00", "sqlite": {"pending_lines": 0, "invalid": 0}, "files_discovered": 1})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"overall": {"ok": True, "status": "ok", "attention": []}})
    _write_json(health / "platform_control_plane_latest.json", {"institutional_readiness": {"overall_status": "ready", "overall_score": 88.0, "top_priorities": [], "weakest_domains": [], "domain_count": 5}})
    _write_json(health / "training_report_latest.json", {"overall_status": "ready", "summary": {"confirmed_training_success": True, "target_count": 2, "trained_count": 2}})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 80.0, "top_priorities": [], "supportability": {"active_supportability_score": 0.8}})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "low", "pressure_index": 0.1})
    _write_json(health / "ingestion_storage_governor_latest.json", {"profile": "normal", "sql_primary_db": {"route_drift": False}, "throttle_controls": {"deferred_files_budget": 1, "cold_files_budget": 1}})
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "writer_busy": False})
    _write_json(health / "external_backlog_retry_bot_latest.json", {"overall_status": "ready", "actionable": False})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 0, "items_synced": 0, "lane_counts": {"core": {"pending_lines": 0}}, "event_count": 0})
    _write_json(health / "storage_resilience_control_latest.json", {"overall_status": "ready", "resilience_score": 90, "restore_drill_fresh": True, "unresolved_split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0, "conflict_files": 0, "force_failback_eligible": False}})
    _write_json(health / "training_requalification_latest.json", {"reactivation_ready_count": 0, "candidate_count": 1})
    _write_json(walk / "coverage_seed_latest.json", {"overall_status": "needs_coverage", "coverage_shortfall_bots": 1, "seed_queue": [{"bot_id": "bot_a"}], "recommended_actions": ["seed coverage"]})
    _write_json(health / "calibration_abstention_control_latest.json", {"overall_status": "ready", "top_actions": []})
    _write_json(health / "paper_execution_calibration_latest.json", {"overall_status": "ready", "metrics": {"mae_bps": 3.0}, "top_actions": []})
    _write_json(health / "daily_verify_auto_remediation_bot_latest.json", {"overall_status": "ready", "resolved_checks": [], "unresolved_checks": []})
    _write_json(health / "storage_tier_policy_latest.json", {"overall_status": "ready", "pressure": {"hot_path_over_budget_bytes": 0}, "upgrade_plan": {"recommended_actions": []}})
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready", "snapshot_ready": True, "precompute_targets": [], "recommended_actions": []})
    _write_json(health / "regime_control_plane_latest.json", {"overall_status": "ready", "regime_state": "balanced", "stance_label": "neutral", "recommended_actions": []})
    _write_json(health / "supportability_control_latest.json", {"overall_status": "ready", "supportability": {"active_supportability_score": 0.9}, "teacher_student": {"students_without_teachers": 0}, "recommended_actions": []})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "ready", "recommended_actions": []})
    _write_json(
        health / "service_control_plane_latest.json",
        {
            "overall_status": "degraded",
            "recommended_actions": ["finish control plane"],
            "upgrade_lanes": {
                "control_plane": {"status": "ready", "summary": "ops_ok=1"},
                "provider_mesh": {"status": "ready", "summary": "required_contract_ok=2/2"},
                "execution_gateway": {"status": "degraded", "summary": "pre_trade_orders=0"},
            },
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "blocked", "shared_host_pressure": {"contention_score": 3}, "recommended_actions": ["separate lanes"]})
    _write_json(health / "rolling_restart_controller_latest.json", {"overall_status": "degraded", "restart_due": True, "recommended_scope": "worker_only", "recommended_actions": ["restart workers"]})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "lease_budget": {"expires_in_seconds": 3000}})
    _write_json(health / "blackstart_recovery_latest.json", {"overall_status": "ready", "stages": [{"name": "launchd_recovery", "ok": True}]})
    _write_json(health / "sleeve_isolation_guard_latest.json", {"overall_status": "ready", "sleeve_matrix": {"isolated_lane_count": 0}})
    _write_json(health / "artifact_freshness_slo_latest.json", {"overall_status": "ready", "sla_summary": {"stale_required": 0}})
    _write_json(health / "runtime_snapshot_cache_control_latest.json", {"overall_status": "ready", "cache_health": {"snapshot_ready": True}})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "ready", "critical_backlog": {"unacked_count": 0}})
    _write_json(health / "storage_quota_guard_latest.json", {"overall_status": "ready", "quota_summary": {"hard_breaches": 0}})
    _write_json(health / "release_freeze_guard_latest.json", {"overall_status": "ready", "window": {"active": True}})
    _write_json(health / "roster_resilience_planner_latest.json", {"overall_status": "ready", "bench": {"bench_depth": 3}})
    _write_json(health / "chaos_drill_coordinator_latest.json", {"overall_status": "ready", "overdue_drills": []})
    _write_json(project_root / "master_bot_registry.json", {"summary": {"total_bots": 3, "active_bots": 2, "deleted_from_rotation": 0, "deletion_guard_ok": True}, "sub_bots": []})

    cockpit = cockpit_src.build_payload(project_root)
    dashboard = dashboard_src.build_dashboard(project_root)

    assert cockpit["overall_status"] == "degraded"
    assert "live_runtime_separation" in cockpit["long_run_lanes"]
    assert "control_plane" in cockpit["upgrade_lanes"]
    assert dashboard["long_runtime"]["live_runtime_separation_status"] == "blocked"
    assert "live_runtime_separation_control_blocked" in dashboard["overall"]["attention"]
