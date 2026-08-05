import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import system_architecture_hardening as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_opsctl(project_root: Path, *, missing: list[str] | None = None) -> None:
    missing_set = set(missing or [])
    text = "\n".join(cmd for cmd in src.REQUIRED_OPSCTL_COMMANDS if cmd not in missing_set)
    path = project_root / "scripts" / "ops" / "opsctl.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _seed_ready_project(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "read_only": True,
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )
    _write_json(health / "paper_400_ramp_latest.json", {"stage": "armed", "blockers": []})
    _write_json(health / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "risk_flags": [],
            "writer_health": {
                "writer_lock_held": True,
                "shard_writer_lane_contract": {
                    "primary_merge_writer_count": 1,
                    "sqlite_primary_writer_count": 1,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(health / "backpressure_drainer_fleet_latest.json", {"writer_lock_held": True})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"severity": "stable", "pressure_index": 0.01, "backpressure": {"total_pending_lines": 12, "pending_lines_threshold": 1000}},
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 20, "compute_pressure_level": "normal", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.1}})
    _write_json(health / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "status": [{"name": "sql_link_writer", "running": 1, "heartbeat_ok": True}],
            "alert_summary": {"critical_count": 0, "warning_count": 0},
            "restart_storm_isolation": {"isolated_count": 0, "execution_blocking_count": 0, "isolated_targets": []},
            "safety_pause": {"active": False},
        },
    )
    _write_json(health / "platform_intelligence_expansion_latest.json", {"overall_status": "ready"})
    _write_json(health / "platform_brain_v5_latest.json", {"overall_status": "ready"})
    _write_json(health / "platform_stabilization_quality_latest.json", {"overall_status": "ready"})
    _write_json(health / "platform_settlement_stabilization_latest.json", {"overall_status": "ready"})
    _write_json(health / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 10, "bots_with_observations": 10, "zero_observation_count": 0, "total_observations": 1000})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 88.0})
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready", "launch_allowed": False, "launch_blockers": []})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "ready", "summary": {"required_contract_ok": 2, "required_collectors": 2}, "cooldowns": []})
    _write_json(health / "source_verification_latest.json", {"overall_status": "ready"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "read_only": True})
    _write_opsctl(project_root)


def test_architecture_hardening_ready_when_all_contracts_align(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["hard_section_count"] == 0
    assert payload["sections"]["safety_execution_boundary"]["overall_status"] == "ready"
    assert payload["sections"]["storage_writer_data_plane"]["overall_status"] == "ready"
    assert payload["anatomy_status"] == "ready"
    assert payload["anatomy_strength_score"] == 100.0
    assert payload["anatomy_layers"]["body"]["strength_label"] == "reinforced"
    assert payload["anatomy_layers"]["skeleton"]["overall_status"] == "ready"
    assert payload["anatomy_layers"]["mind"]["overall_status"] == "ready"
    assert payload["recommended_env_overrides"]["ALLOW_ORDER_EXECUTION"] == "0"


def test_architecture_hardening_accepts_idle_on_demand_sql_writer(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "risk_flags": [],
            "writer_health": {
                "state": "idle",
                "current_step": "complete",
                "writer_lock_held": False,
                "shard_writer_lane_contract": {
                    "primary_merge_writer_count": 1,
                    "sqlite_primary_writer_count": 1,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(health / "backpressure_drainer_fleet_latest.json", {"writer_lock_held": False})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "status": [{"name": "sql_link_writer", "running": 0, "heartbeat_ok": True}],
            "alert_summary": {"critical_count": 0, "warning_count": 0},
            "restart_storm_isolation": {"isolated_count": 0, "execution_blocking_count": 0, "isolated_targets": []},
            "safety_pause": {"active": False},
        },
    )
    _write_json(health / "system_plumbing_control_latest.json", {"overall_status": "ready", "plumbing_score": 100, "blockers": [], "warnings": []})

    payload = src.build_payload(tmp_path)
    writer = payload["sections"]["storage_writer_data_plane"]

    assert payload["overall_status"] == "ready"
    assert writer["overall_status"] == "ready"
    assert writer["watch_items"] == []
    assert writer["evidence"]["writer_idle_complete"] is True


def test_architecture_hardening_manages_closed_training_budget_during_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "launch_allowed": False,
            "launch_blockers": ["autonomic_training_budget_closed"],
        },
    )

    payload = src.build_payload(tmp_path)
    training = payload["sections"]["training_evidence_contract"]
    contract = training["evidence"]["managed_training_evidence_contract"]

    assert payload["overall_status"] == "ready"
    assert training["overall_status"] == "ready"
    assert training["watch_items"] == []
    assert contract["active"] is True
    assert contract["training_budget_closed_managed"] is True
    assert contract["reason"] == "training_budget_closed_is_managed_during_guarded_paper_soak"


def test_architecture_hardening_treats_no_training_candidates_as_healthy_idle(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "prep_allowed": True,
            "launch_allowed": False,
            "launch_blockers": ["no_bot_needs_training_candidates"],
        },
    )

    payload = src.build_payload(tmp_path)
    training = payload["sections"]["training_evidence_contract"]
    contract = training["evidence"]["managed_training_evidence_contract"]

    assert payload["overall_status"] == "ready"
    assert training["overall_status"] == "ready"
    assert training["watch_items"] == []
    assert contract["active"] is True
    assert contract["training_idle_no_candidates_managed"] is True
    assert contract["reason"] == "no_training_candidates_is_healthy_idle_during_guarded_paper_soak"


def test_architecture_hardening_manages_no_candidates_with_closed_budget(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "prep_allowed": True,
            "launch_allowed": False,
            "launch_blockers": ["no_bot_needs_training_candidates", "autonomic_training_budget_closed"],
        },
    )

    payload = src.build_payload(tmp_path)
    training = payload["sections"]["training_evidence_contract"]
    contract = training["evidence"]["managed_training_evidence_contract"]

    assert payload["overall_status"] == "ready"
    assert training["overall_status"] == "ready"
    assert training["watch_items"] == []
    assert contract["active"] is True
    assert contract["training_budget_closed_managed"] is True
    assert contract["training_idle_no_candidates_managed"] is True
    assert contract["reason"] == "no_training_candidates_is_healthy_idle_during_guarded_paper_soak"


def test_architecture_hardening_uses_actionable_collection_zero_contract(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "overall_status": "ready",
            "collector_count": 10,
            "bots_with_observations": 7,
            "effective_bots_with_observations": 10,
            "zero_observation_count": 0,
            "unmanaged_zero_observation_count": 0,
            "managed_zero_observation_count": 3,
            "raw_zero_observation_count": 3,
            "total_observations": 1000,
            "training_ready_count": 0,
        },
    )

    payload = src.build_payload(tmp_path)
    training = payload["sections"]["training_evidence_contract"]

    assert payload["overall_status"] == "ready"
    assert training["overall_status"] == "ready"
    assert training["findings"] == []
    assert training["watch_items"] == []
    assert training["evidence"]["managed_zero_observation_count"] == 3
    assert training["evidence"]["zero_observation_count"] == 0


def test_architecture_hardening_treats_plumbed_sql_overlay_cleanup_as_watch(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 24.535,
            "backpressure": {
                "core_pending_lines": 2591,
                "total_pending_lines": 2731,
                "pending_lines_threshold": 15000,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 2059,
                    "total_pending_lines": 2172,
                    "oldest_pending_age_seconds": 23.484,
                },
            },
        },
    )
    _write_json(
        health / "system_plumbing_control_latest.json",
        {
            "overall_status": "ready",
            "plumbing_score": 85,
            "blockers": [],
            "warnings": ["sql_overlay_cleanup_advisory"],
            "sections": {
                "queue_backpressure": {
                    "overlay_relief": {
                        "active": True,
                        "overlay_total_pending_lines": 2731,
                        "raw_total_pending_lines": 2172,
                    }
                }
            },
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["hard_section_count"] == 0
    assert payload["sections"]["storage_writer_data_plane"]["overall_status"] == "watch"
    assert payload["sections"]["storage_writer_data_plane"]["blocks_guarded_paper"] is False
    assert payload["sections"]["storage_writer_data_plane"]["evidence"]["storage_overlay_relief"]["active"] is True


def test_architecture_hardening_manages_bounded_transient_writer_pressure(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.504,
            "backpressure": {
                "total_pending_lines": 2076,
                "pending_lines_threshold": 15000,
                "raw_live": {
                    "core_pending_lines": 950,
                    "total_pending_lines": 2076,
                    "oldest_pending_age_seconds": 45.0,
                },
            },
            "bounded_recovery_contract": {
                "route_verified": True,
                "active_drain_progress": True,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
        },
    )
    _write_json(
        health / "system_plumbing_control_latest.json",
        {"overall_status": "ready", "plumbing_score": 100, "blockers": [], "warnings": []},
    )

    payload = src.build_payload(tmp_path)
    writer = payload["sections"]["storage_writer_data_plane"]

    assert payload["ok"] is True
    assert writer["overall_status"] == "ready"
    assert writer["blocks_guarded_paper"] is False
    assert writer["evidence"]["bounded_writer_pressure_managed"] is True


def test_architecture_hardening_allows_isolated_read_only_collector_watch(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "process_watchdog_latest.json",
        {
            "status": [{"name": "sql_link_writer", "running": 1, "heartbeat_ok": True}],
            "alert_summary": {"critical_count": 0, "warning_count": 1},
            "restart_storm_isolation": {"isolated_count": 1, "execution_blocking_count": 0, "isolated_targets": ["coinbase_loop"]},
            "safety_pause": {"active": False},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["sections"]["collector_process_quarantine"]["overall_status"] == "watch"
    assert payload["sections"]["collector_process_quarantine"]["blocks_guarded_paper"] is False
    assert payload["anatomy_layers"]["heart"]["overall_status"] == "watch"
    assert payload["anatomy_layers"]["organs"]["overall_status"] == "watch"
    assert payload["anatomy_layers"]["skeleton"]["overall_status"] == "ready"


def test_architecture_hardening_manages_explicitly_isolated_read_only_collector_storms(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health_fast = json.loads((tmp_path / "governance" / "health" / "health_fast_latest.json").read_text(encoding="utf-8"))
    health_fast["process_watchdog"] = {
        "alert_summary": {
            "critical_count": 0,
            "warning_count": 1,
            "rows": [
                {
                    "target": "coinbase_loop",
                    "type": "restart_storm",
                    "severity": "warn",
                    "blocks_guarded_paper": False,
                }
            ],
        }
    }
    _write_json(tmp_path / "governance" / "health" / "health_fast_latest.json", health_fast)
    _write_json(
        tmp_path / "governance" / "health" / "process_watchdog_latest.json",
        {
            "status": [{"name": "sql_link_writer", "running": 1, "heartbeat_ok": True}],
            "alert_summary": {"critical_count": 0, "warning_count": 0},
            "restart_storm_isolation": {"isolated_count": 1, "execution_blocking_count": 0, "isolated_targets": ["coinbase_loop"]},
            "safety_pause": {"active": False},
        },
    )

    payload = src.build_payload(tmp_path)
    collector = payload["sections"]["collector_process_quarantine"]

    assert payload["overall_status"] == "ready"
    assert collector["overall_status"] == "ready"
    assert collector["watch_items"] == []
    assert collector["evidence"]["managed_quarantine_contract"]["active"] is True
    assert payload["anatomy_layers"]["heart"]["overall_status"] == "ready"


def test_architecture_hardening_prefers_health_fast_restart_isolation_for_collector_rows(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health_fast = json.loads((tmp_path / "governance" / "health" / "health_fast_latest.json").read_text(encoding="utf-8"))
    health_fast["process_watchdog"] = {
        "alert_summary": {
            "critical_count": 0,
            "warning_count": 2,
            "rows": [
                {
                    "target": "all_sleeves",
                    "type": "restart_storm",
                    "severity": "warn",
                    "blocks_guarded_paper": False,
                },
                {
                    "target": "coinbase_loop",
                    "type": "budget_exhausted",
                    "severity": "warn",
                    "blocks_guarded_paper": False,
                },
            ],
        },
        "restart_storm_isolation": {
            "isolated_count": 2,
            "execution_blocking_count": 0,
            "isolated_targets": ["all_sleeves", "coinbase_loop"],
        },
        "safety_pause": {"active": False},
    }
    _write_json(tmp_path / "governance" / "health" / "health_fast_latest.json", health_fast)
    _write_json(
        tmp_path / "governance" / "health" / "process_watchdog_latest.json",
        {
            "status": [{"name": "sql_link_writer", "running": 1, "heartbeat_ok": True}],
            "alert_summary": {"critical_count": 0, "warning_count": 0},
            "restart_storm_isolation": {"isolated_count": 1, "execution_blocking_count": 0, "isolated_targets": ["coinbase_loop"]},
            "safety_pause": {"active": False},
        },
    )

    payload = src.build_payload(tmp_path)
    collector = payload["sections"]["collector_process_quarantine"]

    assert payload["overall_status"] == "ready"
    assert collector["overall_status"] == "ready"
    assert collector["evidence"]["isolated_targets"] == ["all_sleeves", "coinbase_loop"]
    assert collector["evidence"]["managed_quarantine_contract"]["active"] is True


def test_architecture_hardening_blocks_truthy_live_execution_flags(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        {"stage": "armed", "blockers": [], "recommended_env_overrides": {"ALLOW_ORDER_EXECUTION": "1"}},
    )

    payload = src.build_payload(tmp_path)
    safety = payload["sections"]["safety_execution_boundary"]

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert safety["overall_status"] == "blocked"
    assert safety["evidence"]["truthy_live_enable_flags"][0]["path"].endswith("ALLOW_ORDER_EXECUTION")


def test_architecture_hardening_blocks_duplicate_sql_writer(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "risk_flags": ["duplicate_sql_writer_processes"],
            "writer_health": {
                "writer_lock_held": True,
                "shard_writer_lane_contract": {
                    "primary_merge_writer_count": 2,
                    "sqlite_primary_writer_count": 2,
                    "single_primary_merge_writer": False,
                },
            },
        },
    )

    payload = src.build_payload(tmp_path)
    writer = payload["sections"]["storage_writer_data_plane"]

    assert payload["ok"] is False
    assert writer["overall_status"] == "blocked"
    assert "duplicate_sql_writer_processes" in writer["findings"]
    assert payload["anatomy_layers"]["skeleton"]["overall_status"] == "blocked"
    assert payload["anatomy_layers"]["heart"]["overall_status"] == "blocked"


def test_architecture_hardening_marks_stale_paper_ramp_global_blocker_as_watch(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        {"stage": "blocked", "blockers": ["global_halt_or_clear_blocker_active"]},
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["sections"]["safety_execution_boundary"]["overall_status"] == "watch"
    assert payload["sections"]["truth_source_consistency"]["overall_status"] == "watch"


def test_architecture_hardening_treats_protect_live_as_safety_boundary_not_runtime_debt(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "release_contract": {"live_lane_should_be_read_only": True},
            "clearance_plan": {"clearance_state": "protect_live"},
        },
    )

    payload = src.build_payload(tmp_path)
    runtime = payload["sections"]["runtime_capacity_partition"]

    assert payload["ok"] is True
    assert runtime["overall_status"] == "ready"
    assert runtime["evidence"]["live_runtime_separation_read_only_policy"] is True
    assert payload["anatomy_layers"]["skin"]["overall_status"] == "ready"


def test_architecture_hardening_surfaces_mac_fluidity_debt_as_runtime_watch(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "host_saturation_score": 48,
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "mac_fluidity_contract": {
                "overall_status": "watch",
                "fluidity_band": "strained",
                "fluidity_score": 79.0,
            },
        },
    )

    payload = src.build_payload(tmp_path)
    runtime = payload["sections"]["runtime_capacity_partition"]

    assert payload["ok"] is True
    assert runtime["overall_status"] == "watch"
    assert "mac_fluidity_status=watch" in runtime["watch_items"]
    assert runtime["evidence"]["mac_fluidity_band"] == "strained"


def test_architecture_hardening_manages_guarded_smooth_runtime_capacity(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "throttle_profile": "soft_cap",
            "host_saturation_score": 51.36,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "mac_fluidity_contract": {
                "overall_status": "watch",
                "fluidity_band": "guarded_smooth",
                "fluidity_score": 88.27,
            },
        },
    )

    payload = src.build_payload(tmp_path)
    runtime = payload["sections"]["runtime_capacity_partition"]

    assert payload["overall_status"] == "ready"
    assert runtime["overall_status"] == "ready"
    assert runtime["watch_items"] == []
    assert runtime["evidence"]["managed_capacity_contract"]["active"] is True
    assert payload["anatomy_strength_score"] == 100.0


def test_architecture_hardening_manages_runtime_ready_bounded_writer_contract(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "throttle_profile": "sustain",
            "host_saturation_score": 69.8,
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "soft_cap_advisory_reclassification": {
                "active": True,
                "to_status": "ready",
                "reason": "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready",
                "measurements": {
                    "runtime_ready_guarded": True,
                    "storage_writer_cooling_guarded_ready": True,
                },
            },
            "mac_fluidity_contract": {
                "overall_status": "needs_work",
                "fluidity_band": "strained",
                "fluidity_score": 69.8,
            },
        },
    )

    payload = src.build_payload(tmp_path)
    runtime = payload["sections"]["runtime_capacity_partition"]

    assert payload["overall_status"] == "ready"
    assert runtime["overall_status"] == "ready"
    assert runtime["findings"] == []
    assert runtime["evidence"]["managed_runtime_ready_contract"]["active"] is True
    assert "mac_fluidity_status=needs_work" in runtime["evidence"]["managed_runtime_ready_contract"]["managed_findings"]


def test_architecture_hardening_consumes_plumbing_runtime_memory_relief(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "host_saturation_score": 48.63,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "elevated",
            "mac_fluidity_contract": {
                "overall_status": "watch",
                "fluidity_band": "guarded_smooth",
                "fluidity_score": 81.23,
            },
        },
    )
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "needs_work"})
    _write_json(
        health / "system_plumbing_control_latest.json",
        {
            "overall_status": "ready",
            "sections": {
                "runtime_memory": {
                    "ok": True,
                    "status": "advisory",
                    "paper_only_runtime_memory_relief": True,
                    "memory_pressure_level": "elevated",
                }
            },
        },
    )

    payload = src.build_payload(tmp_path)
    runtime = payload["sections"]["runtime_capacity_partition"]

    assert payload["overall_status"] == "ready"
    assert runtime["overall_status"] == "ready"
    assert runtime["findings"] == []
    assert runtime["evidence"]["managed_plumbing_runtime_contract"]["active"] is True
    assert "memory_status=needs_work" in runtime["evidence"]["managed_plumbing_runtime_contract"]["managed_findings"]


def test_architecture_hardening_treats_platform_watch_as_managed_under_strict_clear(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    for name in (
        "platform_intelligence_expansion_latest.json",
        "platform_brain_v5_latest.json",
        "platform_stabilization_quality_latest.json",
        "platform_settlement_stabilization_latest.json",
    ):
        _write_json(health / name, {"overall_status": "watch"})

    payload = src.build_payload(tmp_path)
    platform = payload["sections"]["platform_watch_semantics"]

    assert payload["overall_status"] == "ready"
    assert platform["overall_status"] == "ready"
    assert platform["watch_items"] == []
    assert platform["evidence"]["managed_watch_contract"]["active"] is True
    assert payload["anatomy_layers"]["brain"]["overall_status"] == "ready"


def test_architecture_hardening_treats_platform_watch_as_managed_under_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    health_fast = json.loads((health / "health_fast_latest.json").read_text(encoding="utf-8"))
    health_fast["strict_all_clear"] = False
    health_fast["global_halt"] = {"halt": False, "clear_blockers": []}
    _write_json(health / "health_fast_latest.json", health_fast)
    for name in (
        "platform_intelligence_expansion_latest.json",
        "platform_brain_v5_latest.json",
        "platform_stabilization_quality_latest.json",
        "platform_settlement_stabilization_latest.json",
    ):
        _write_json(health / name, {"overall_status": "watch"})

    payload = src.build_payload(tmp_path)
    platform = payload["sections"]["platform_watch_semantics"]

    assert payload["overall_status"] == "ready"
    assert platform["overall_status"] == "ready"
    assert platform["watch_items"] == []
    assert platform["evidence"]["managed_watch_contract"]["active"] is True
    assert platform["evidence"]["managed_watch_contract"]["reason"] == "platform_watch_states_are_nonblocking_under_guarded_paper_ready"


def test_architecture_hardening_isolates_optional_provider_source_debt(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "provider_mesh_latest.json",
        {
            "overall_status": "degraded",
            "summary": {"required_contract_ok": 4, "required_collectors": 4},
            "required_failures": [],
            "soft_failures": ["tradingeconomics_guest", "sec_edgar_context"],
            "cooldowns": [],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall_status": "degraded",
            "unverified_sources": ["macro_crossstack", "sec_edgar_context"],
            "stale_artifacts": ["sec_edgar_context", "extended_quant_context"],
            "degraded_artifacts": ["macro_crossstack", "sec_edgar_context", "extended_quant_context"],
            "autorefresh_contract": {"enabled": True},
        },
    )

    payload = src.build_payload(tmp_path)
    provider = payload["sections"]["provider_source_mesh"]
    contract = provider["evidence"]["source_mesh_debt_contract"]

    assert payload["overall_status"] == "ready"
    assert provider["overall_status"] == "ready"
    assert provider["watch_items"] == []
    assert contract["active"] is True
    assert contract["required_provider_ready"] is True
    assert contract["critical_source_debt"] == []
    assert "sec_edgar_context" in contract["optional_source_debt"]
    assert "macro_crossstack" in contract["managed_verification_debt"]


def test_architecture_hardening_treats_collection_maturity_as_watch_under_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "overall_status": "degraded",
            "collector_count": 173,
            "bots_with_observations": 28,
            "zero_observation_count": 145,
            "total_observations": 3887,
            "training_ready_count": 0,
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "training_quality_score": 93.5})

    payload = src.build_payload(tmp_path)
    training = payload["sections"]["training_evidence_contract"]

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["hard_section_count"] == 0
    assert training["overall_status"] == "watch"
    assert training["evidence"]["managed_training_evidence_contract"]["active"] is True


def test_architecture_hardening_treats_refreshing_core_source_debt_as_watch_under_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall_status": "degraded",
            "unverified_sources": ["public_macro_feeds"],
            "degraded_artifacts": ["public_macro_feeds"],
            "stale_artifacts": ["public_macro_feeds"],
            "autorefresh_contract": {"enabled": True},
        },
    )

    payload = src.build_payload(tmp_path)
    provider = payload["sections"]["provider_source_mesh"]
    contract = provider["evidence"]["source_mesh_debt_contract"]

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["hard_section_count"] == 0
    assert provider["overall_status"] == "watch"
    assert "core_source_verification_debt_managed_by_guarded_paper_autorefresh" in provider["watch_items"]
    assert contract["guarded_paper_source_debt_advisory"] is True


def test_architecture_hardening_writes_section_config_and_override_artifacts(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    payload = src.build_payload(tmp_path)

    written = src.write_outputs(tmp_path, tmp_path / "governance" / "health" / "system_architecture_hardening_latest.json", payload)

    assert Path(written["latest"]).exists()
    assert Path(written["config"]).exists()
    assert Path(written["env_override"]).exists()
    assert len(written["section_artifacts"]) == payload["section_count"]
    assert len(written["anatomy_artifacts"]) == payload["anatomy_layer_count"]
    assert Path(written["anatomy_artifacts"]["body"]).exists()
    latest = json.loads(Path(written["latest"]).read_text(encoding="utf-8"))
    assert latest["written_artifacts"]["anatomy_artifacts"]["body"] == written["anatomy_artifacts"]["body"]
    assert "ALLOW_ORDER_EXECUTION=0" in Path(written["env_override"]).read_text(encoding="utf-8")


def test_architecture_hardening_guards_opsctl_command_spine(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_opsctl(tmp_path, missing=["provider-mesh"])

    payload = src.build_payload(tmp_path)
    spine = payload["sections"]["opsctl_command_spine"]

    assert payload["ok"] is False
    assert spine["overall_status"] == "needs_work"
    assert "provider-mesh" in spine["evidence"]["missing_commands"]
