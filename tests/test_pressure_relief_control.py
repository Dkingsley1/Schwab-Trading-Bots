import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import health_fast, pressure_relief_control


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_pressure_relief_enables_twenty_eight_pressure_controls(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "host_saturation_score": 64.0,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "elevated",
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {"co_running_session": {"level": "heavy_competition"}},
    )
    _write_json(
        health_root / "swap_pressure_governor_latest.json",
        {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.5}},
    )
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.02})

    payload = pressure_relief_control.build_payload(project_root)
    env = payload["recommended_env_overrides"]

    assert payload["tier"] == "guarded_relief"
    assert len(payload["pressure_relief_items"]) == 28
    assert env["OPS_HEALTH_FAST_ENABLED"] == "1"
    assert env["LIVE_FEED_HEAVY_TTL_ENABLED"] == "1"
    assert env["MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW"] == "1"
    assert env["SQL_LINK_SERVICE_ADAPTIVE_WRITER_ENABLED"] == "1"
    assert env["MLX_LAZY_IMPORTS"] == "1"
    assert env["HEALTH_ARTIFACT_COALESCE_ENABLED"] == "1"
    assert env["BOT_COLLECTION_DUTY_CYCLE_ENABLED"] == "1"
    assert env["PAPER_TRADE_EVENT_QUEUE_JITTER_ENABLED"] == "1"
    assert env["TRAINING_RESEARCH_PAUSE_ON_PRESSURE"] == "1"


def test_pressure_relief_freezes_support_maintenance_when_system_cotenant_is_hot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "host_saturation_score": 61.0,
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "host_pressure_attribution": {
                "system_cotenant_hot": True,
                "external_pressure_dominant": True,
                "support_jobs_hot": True,
            },
        },
    )
    _write_json(health_root / "memory_efficiency_control_latest.json", {"co_running_session": {"level": "idle"}})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.5}})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.02})

    payload = pressure_relief_control.build_payload(project_root)
    env = payload["recommended_env_overrides"]

    assert payload["support_maintenance_stabilization"]["active"] is True
    assert env["OPS_SUPPORT_MAINTENANCE_STABILIZER_ACTIVE"] == "1"
    assert env["OPS_SUPPORT_MAINTENANCE_FREEZE"] == "1"
    assert env["OPS_SUPPORT_JOB_NICE"] == "16"
    assert float(env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"]) <= 0.35


def test_pressure_relief_keeps_concentrated_sql_drain_fast_under_deep_relief(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "host_saturation_score": 91.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "elevated",
        },
    )
    _write_json(health_root / "memory_efficiency_control_latest.json", {"co_running_session": {"level": "heavy_competition"}})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.5}})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "high", "pressure_index": 2.2})
    _write_json(
        health_root / "backpressure_drainer_fleet_latest.json",
        {
            "active_drainer": {
                "name": "core_decision_drainer",
                "concentration": {
                    "total_pending_lines": 33623,
                    "top1_share": 0.544806,
                    "top3_share": 0.92871,
                    "concentrated": True,
                },
            },
            "service_request": {"env_overrides": {"SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1"}},
        },
    )

    payload = pressure_relief_control.build_payload(project_root)
    env = payload["recommended_env_overrides"]

    assert payload["tier"] == "deep_relief"
    assert payload["storage_pressure"]["sql_writer_coordination"]["concentrated_core_drain"] is True
    assert env["SQL_LINK_SERVICE_INTERVAL_SECONDS"] == "12"
    assert env["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert env["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "90"
    assert env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] == "12000"


def test_health_fast_is_read_only_and_ready_from_latest_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(health_root / "process_watchdog_latest.json", {"alerts": [], "safety_pause": {"active": False}, "status": []})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 2, "bots_with_observations": 2})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})

    payload = health_fast.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["strict_all_clear"] is True
    assert payload["read_only"] is True
    assert payload["started_heavy_reports"] is False
    assert payload["operational_readiness"]["guarded_paper"]["ok"] is True
    assert payload["collection"]["bots_with_observations"] == 2


def test_health_fast_surfaces_all_sleeves_effective_runtime_reconciliation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "process_watchdog_latest.json",
        {
            "alerts": [],
            "safety_pause": {"active": False},
            "status": [
                {
                    "name": "all_sleeves",
                    "running": 0,
                    "heartbeat_ok": True,
                    "process_live": True,
                    "effective_process_live": True,
                    "launcher_live": False,
                    "child_process_live": True,
                    "child_fanout_ok": True,
                    "child_fanout": {"ok": True, "child_process_count": 100},
                    "launcher_artifact_certified_fanout": True,
                    "launcher_artifact_health": {
                        "ok": True,
                        "reason": "fresh_launcher_artifact_certifies_full_fanout",
                    },
                    "process_live_reason": "fresh_launcher_artifact_certifies_full_fanout",
                }
            ],
        },
    )
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})

    payload = health_fast.build_payload(project_root)

    runtime = payload["process_watchdog"]["all_sleeves_effective_runtime"]
    assert runtime["ok"] is True
    assert runtime["launcher_live"] is False
    assert runtime["effective_live"] is True
    assert runtime["child_process_count"] == 100
    assert runtime["launcher_artifact_certified_fanout"] is True


def test_health_fast_reports_guarded_ready_with_isolated_collector_repairs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "process_watchdog_latest.json",
        {
            "alerts": [
                {
                    "name": "coinbase_loop",
                    "type": "budget_exhausted",
                    "alert": {
                        "stdout": json.dumps(
                            {
                                "severity": "warn",
                                "event": "watchdog_restart_budget_exhausted_isolated",
                            }
                        )
                    },
                }
            ],
            "restart_storm_isolation": {
                "isolated_count": 1,
                "execution_blocking_count": 0,
                "isolated_targets": ["coinbase_loop"],
            },
            "safety_pause": {"active": False},
            "status": [],
        },
    )
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 2, "bots_with_observations": 2})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "blockers": []})
    _write_json(health_root / "platform_stabilization_quality_latest.json", {"overall_status": "blocked", "next_best_command": "./scripts/ops/opsctl.sh pressure-relief --apply --json"})

    payload = health_fast.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded_ready"
    assert payload["strict_all_clear"] is False
    assert payload["operational_readiness"]["guarded_paper"]["ok"] is True
    assert payload["operational_readiness"]["collector_repair"]["status"] == "managed_isolated"
    assert payload["operational_readiness"]["collector_repair"]["managed_isolated"] is True
    assert payload["operational_readiness"]["platform_repair"]["status"] == "needs_work"
    assert payload["process_watchdog"]["alert_summary"]["critical_count"] == 0


def test_health_fast_strict_clear_allows_managed_isolated_read_only_collectors(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "process_watchdog_latest.json",
        {
            "alerts": [
                {
                    "name": "coinbase_loop",
                    "type": "budget_exhausted",
                    "alert": {
                        "stdout": json.dumps(
                            {
                                "severity": "warn",
                                "event": "watchdog_restart_budget_exhausted_isolated",
                            }
                        )
                    },
                },
                {
                    "name": "coinbase_futures_loop",
                    "type": "restart_storm",
                    "alert": {
                        "stdout": json.dumps(
                            {
                                "severity": "warn",
                                "event": "watchdog_restart_storm_isolated",
                            }
                        )
                    },
                },
            ],
            "restart_storm_isolation": {
                "isolated_count": 2,
                "execution_blocking_count": 0,
                "isolated_targets": ["coinbase_futures_loop", "coinbase_loop"],
            },
            "safety_pause": {"active": False},
            "status": [],
        },
    )
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 2, "bots_with_observations": 2})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "blockers": []})
    _write_json(health_root / "system_architecture_hardening_latest.json", {"overall_status": "ready"})

    payload = health_fast.build_payload(project_root)
    collector = payload["operational_readiness"]["collector_repair"]

    assert payload["overall_status"] == "ready"
    assert payload["strict_all_clear"] is True
    assert payload["repair_backlog_active"] is False
    assert collector["status"] == "managed_isolated"
    assert collector["blocks_strict_clear"] is False


def test_health_fast_allows_guarded_paper_when_paper_ramp_relieves_write_path_blocker(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(health_root / "process_watchdog_latest.json", {"alerts": [], "safety_pause": {"active": False}, "status": []})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 2, "bots_with_observations": 2})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_blocked", "clear_blockers": ["write_path_recovery_pending"]})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})
    _write_json(
        health_root / "paper_400_ramp_latest.json",
        {
            "stage": "armed",
            "blockers": [],
            "gates": {
                "global_halt": {
                    "ok": True,
                    "status": "write_path_recovery_advisory",
                    "clear_blocker_relief": {
                        "active": True,
                        "clear_blockers": ["write_path_recovery_pending"],
                    },
                }
            },
        },
    )

    payload = health_fast.build_payload(project_root)

    guarded = payload["operational_readiness"]["guarded_paper"]
    assert guarded["ok"] is True
    assert guarded["advisory_clear_blockers"] == ["write_path_recovery_pending"]
    assert "global_clear_blocker=write_path_recovery_pending" not in guarded["blockers"]


def test_health_fast_ignores_stale_paper_ramp_global_halt_blocker_when_current_halt_is_clear(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(health_root / "process_watchdog_latest.json", {"alerts": [], "safety_pause": {"active": False}, "status": []})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 2, "bots_with_observations": 2})
    _write_json(health_root / "global_killswitch_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})
    _write_json(
        health_root / "paper_400_ramp_latest.json",
        {
            "stage": "blocked",
            "blockers": ["global_halt_or_clear_blocker_active"],
        },
    )

    payload = health_fast.build_payload(project_root)

    guarded = payload["operational_readiness"]["guarded_paper"]
    assert guarded["ok"] is True
    assert guarded["paper_ramp_stale_global_blocker_ignored"] is True
    assert "paper_ramp_not_armed" not in guarded["blockers"]


def test_health_fast_blocks_guarded_paper_on_critical_process_alert(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "process_watchdog_latest.json",
        {
            "alerts": [
                {
                    "name": "execution_lane_live",
                    "type": "restart_storm",
                    "alert": {"stdout": json.dumps({"severity": "critical", "event": "watchdog_restart_storm"})},
                }
            ],
            "safety_pause": {"active": False},
            "status": [],
        },
    )
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})

    payload = health_fast.build_payload(project_root)

    assert payload["ok"] is False
    assert payload["overall_status"] == "degraded"
    assert payload["operational_readiness"]["guarded_paper"]["blockers"] == ["critical_process_alerts_active"]


def test_health_fast_surfaces_system_architecture_hardening_as_platform_repair(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(health_root / "process_watchdog_latest.json", {"alerts": [], "safety_pause": {"active": False}, "status": []})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "blockers": []})
    _write_json(
        health_root / "system_architecture_hardening_latest.json",
        {"overall_status": "needs_work", "next_best_command": "./scripts/ops/opsctl.sh system-architecture-hardening --apply --json"},
    )

    payload = health_fast.build_payload(project_root)
    platform_repair = payload["operational_readiness"]["platform_repair"]

    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded_ready"
    assert platform_repair["status"] == "needs_work"
    assert platform_repair["issues"][0]["source"] == "system_architecture_hardening"
