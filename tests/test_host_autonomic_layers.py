from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import autonomic_resource_governor as governor
from scripts.ops import host_capability_contract as host_contract
from scripts.ops import host_self_benchmark
from scripts.ops import memory_pressure_intelligence
from scripts.ops import migration_readiness_report
from scripts.ops import os_adapter_layer
from scripts.ops import runtime_throttle_control
from scripts.ops import system_needs_intelligence
from scripts.ops import workload_class_registry


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _host_payload() -> dict:
    return {
        "timestamp_utc": "2026-05-20T00:00:00+00:00",
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "body_map": {
            "system": {"os": "darwin", "system": "Darwin", "machine": "arm64"},
            "cpu_topology": {
                "topology": "apple_silicon_p_e",
                "performance_core_count": 8,
                "efficiency_core_count": 2,
                "recommended_primary_compute_lanes": 8,
                "hard_affinity_supported": False,
                "core_allocator": "darwin_qos_nice_taskpolicy",
            },
            "memory": {"memory_gb": 64, "swap_used_gb": 1.2, "pressure_level": "normal"},
            "gpu_stack": {
                "primary_gpu_stack": "MLX",
                "mlx_available": True,
                "metal_available": True,
                "cuda_available": False,
                "rocm_available": False,
            },
            "storage_layout": {
                "protected_volumes": ["/Volumes/VIDEO"],
                "denylist_rules": [{"path": "/Volumes/VIDEO", "policy": "never_write_or_prune"}],
                "bot_logs_external_mount": "/Volumes/BOT_LOGS",
            },
            "launch_system": {"primary": "launchd"},
            "foreground_apps_and_user_activity": {"open_apps": [], "creative_level": "none", "co_running_level": "none"},
        },
    }


def test_host_capability_contract_publishes_body_map_and_video_denylist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(host_contract, "_system_profile", lambda: {"os": "darwin", "system": "Darwin", "machine": "arm64"})
    monkeypatch.setattr(host_contract, "_cpu_topology", lambda system, apple: _host_payload()["body_map"]["cpu_topology"])
    monkeypatch.setattr(host_contract, "_memory_profile", lambda system, runtime, memory: {"memory_gb": 64, "pressure_level": "normal"})
    monkeypatch.setattr(host_contract, "_gpu_stack", lambda system, mlx: _host_payload()["body_map"]["gpu_stack"])
    monkeypatch.setattr(host_contract, "_storage_layout", lambda: _host_payload()["body_map"]["storage_layout"])
    monkeypatch.setattr(host_contract, "_launch_system", lambda system: {"primary": "launchd"})
    monkeypatch.setattr(host_contract, "_foreground_context", lambda computer, resource: {"open_apps": [], "user_coexistent_required": False})

    payload = host_contract.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["body_map"]["cpu_topology"]["performance_core_count"] == 8
    assert payload["body_map"]["protected_volume_policy"]["never_touch_video_volume"] is True
    assert "/Volumes/VIDEO" in payload["body_map"]["storage_layout"]["protected_volumes"]


def test_os_adapter_layer_maps_macos_to_qos_and_launchd() -> None:
    payload = os_adapter_layer.build_payload(host=_host_payload())

    assert payload["overall_status"] == "ready"
    assert payload["adapter_id"] == "macos_apple_silicon_mlx_launchd"
    assert payload["adapters"]["process_priority"]["adapter"] == "renice_taskpolicy_qos"
    assert payload["adapters"]["process_priority"]["hard_affinity_supported"] is False
    assert payload["adapters"]["service_startup"]["primary"] == "launchd"
    assert payload["adapters"]["protected_storage"]["denylist"] == ["/Volumes/VIDEO"]


def test_workload_class_registry_classifies_writer_and_user_apps() -> None:
    writer = workload_class_registry.classify_command("python scripts/ops/sql_link_shard_manager.py")
    music = workload_class_registry.classify_command("/Applications/Music.app/Contents/MacOS/Music")

    assert writer["class_id"] == "backlog_drain"
    assert music["class_id"] == "user_coexistent"
    payload = workload_class_registry.build_payload()
    assert "live_critical" in payload["class_order"]
    assert payload["class_contract"]["single_writer_remains_exclusive_for_sqlite_writes"] is True


def test_autonomic_governor_prioritizes_backlog_and_writes_override(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 30000,
                "support_pending_lines": 2000,
                "deferred_pending_lines": 1000,
                "total_pending_lines": 33000,
                "oldest_pending_age_seconds": 120000,
                "pending_lines_threshold": 5000,
            },
            "stale_pending_locator": {
                "oldest_sources": [
                    {"source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260520.jsonl", "shard": "trading"}
                ]
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "advisory", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 6}},
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": ["PyCharm"], "co_running_level": "active"}})

    payload = governor.build_payload(tmp_path)
    result = governor.write_outputs(
        payload,
        out_path=health / "autonomic_resource_governor_latest.json",
        override_path=tmp_path / "config" / ".env.autonomic_resource_governor_override",
        apply=True,
    )

    assert payload["unified_decision"] == "backlog_recovery"
    assert payload["budgets"]["backlog_writer"]["mode"] == "catch_up_waves"
    assert payload["budgets"]["backlog_writer"]["p_core_preprocess_workers"] == 6
    assert payload["budgets"]["collectors"]["max_active_ratio"] <= 0.2
    assert payload["budgets"]["training"]["allowed"] is False
    assert payload["backlog_green_gate"]["status"] == "not_green"
    assert payload["adaptive_controls"]["p_core_widening"]["mode"] == "hold_until_backlog_age_green"
    assert payload["adaptive_controls"]["collector_reopening"]["stage"] == "protect_core"
    assert "backlog_age_not_green" in payload["adaptive_controls"]["training_reentry"]["blockers"]
    assert payload["what_do_you_need"]["items"][0]["exact_shard"] == "trading"
    assert result["applied"] is True
    override = (tmp_path / "config" / ".env.autonomic_resource_governor_override").read_text(encoding="utf-8")
    assert "AUTONOMIC_PCORE_PREPROCESS_WORKERS=6" in override
    assert "AUTONOMIC_BACKLOG_GREEN=0" in override
    assert "AUTONOMIC_COLLECTOR_REOPEN_STAGE=protect_core" in override
    assert "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.12" in override


def test_autonomic_governor_honors_six_p_core_user_reserve_target(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET", "6")
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 30_000,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 30_000,
                "oldest_pending_age_seconds": 1200,
                "pending_lines_threshold": 5000,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "advisory", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 6}},
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})

    payload = governor.build_payload(tmp_path)
    result = governor.write_outputs(
        payload,
        out_path=health / "autonomic_resource_governor_latest.json",
        override_path=tmp_path / "config" / ".env.autonomic_resource_governor_override",
        apply=True,
    )
    allocation = payload["host_lane_budget"]["p_core_allocation_contract"]

    assert payload["host_lane_budget"]["selected_p_core_preprocess_workers"] == 2
    assert allocation["user_app_reserved_p_cores"] == 6
    assert allocation["user_app_reserve_target_p_cores"] == 6
    assert payload["host_lane_budget"]["p_core_widening_controller"]["user_reserve_worker_cap"] == 2
    override = (tmp_path / "config" / ".env.autonomic_resource_governor_override").read_text(encoding="utf-8")
    assert result["applied"] is True
    assert "AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET=6" in override
    assert "BACKLOG_PCORE_USER_APP_RESERVE_TARGET=6" in override


def test_autonomic_governor_steps_up_collectors_and_training_when_backlog_is_green(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 800,
                "support_pending_lines": 10,
                "deferred_pending_lines": 100,
                "total_pending_lines": 910,
                "oldest_pending_age_seconds": 120,
                "pending_lines_threshold": 5000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 800}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 3}},
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"drain_effectiveness": {"status": "strong_progress", "merged_rows": 2500}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 2, "compile_mode": "direct_stable"}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "timestamp_utc": "2026-05-20T00:00:00+00:00",
            "storage_metrics": {
                "core_pending_lines": 1200,
                "total_pending_lines": 1400,
                "overlay_pending_lines": 1200,
                "oldest_pending_age_seconds": 300,
            },
            "stability_state": {
                "consecutive_green_samples": 2,
                "consecutive_runtime_clear_samples": 2,
                "consecutive_writer_idle_samples": 2,
                "consecutive_improving_samples": 2,
            },
        },
    )

    payload = governor.build_payload(tmp_path)

    assert payload["backlog_green_gate"]["status"] == "green"
    assert payload["stability_state"]["consecutive_green_samples"] == 3
    assert payload["host_lane_budget"]["selected_p_core_preprocess_workers"] == 4
    assert payload["adaptive_controls"]["p_core_widening"]["mode"] == "step_up_one_worker"
    assert payload["budgets"]["collectors"]["max_active_ratio"] == 0.55
    assert payload["adaptive_controls"]["collector_reopening"]["stage"] == "normal_reopen"
    assert payload["budgets"]["training"]["allowed"] is True
    assert payload["budgets"]["training"]["profile"] == "coverage_canary"


def test_autonomic_governor_reads_watchdog_intelligence_before_training(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 100,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 100,
                "oldest_pending_age_seconds": 10,
                "pending_lines_threshold": 15000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 0}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 3}},
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"drain_effectiveness": {"status": "strong_progress", "merged_rows": 1000}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 2, "compile_mode": "direct_stable"}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "timestamp_utc": "2026-05-20T00:00:00+00:00",
            "storage_metrics": {
                "core_pending_lines": 200,
                "total_pending_lines": 200,
                "overlay_pending_lines": 0,
                "oldest_pending_age_seconds": 20,
            },
            "stability_state": {
                "consecutive_green_samples": 3,
                "consecutive_runtime_clear_samples": 3,
                "consecutive_runtime_pressure_clear_samples": 3,
                "consecutive_writer_idle_samples": 3,
                "consecutive_improving_samples": 3,
            },
        },
    )
    _write_json(
        health / "watchdog_intelligence_latest.json",
        {
            "overall_status": "degraded",
            "grade": "C",
            "score": 76.0,
            "active_issue_count": 1,
            "restart_storm_count": 0,
            "exact_needs": [
                {
                    "target": "all_sleeves",
                    "status": "needs_repair",
                    "blocker": "heartbeat_stale",
                    "exact_file": "governance/health/process_watchdog_latest.json",
                    "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
                    "expected_impact": "repair watchdog before adding work",
                }
            ],
        },
    )

    payload = governor.build_payload(tmp_path)

    assert payload["statuses"]["watchdog_intelligence"] == "degraded"
    assert payload["budgets"]["watchdogs"]["healthy"] is False
    assert payload["budgets"]["training"]["allowed"] is False
    assert payload["what_do_you_need"]["items"][0]["blocker"] == "watchdog_heartbeat_stale"
    assert payload["integration_contract"]["reads_watchdog_intelligence"] is True


def test_autonomic_governor_keeps_micro_canary_open_for_nonblocking_watchdog_advisory(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 0,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 0,
                "oldest_pending_age_seconds": 0,
                "pending_lines_threshold": 15000,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "advisory",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "high",
            "host_saturation_score": 50.0,
            "p_core_runtime_feedback": {"preprocess_worker_budget": 4},
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"writer_state_before": {"active": False, "running": False}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 4}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom", "recommended_p_core_worker_cap": 4},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": False,
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "small_canary_max_parallel_trainings": 1,
                "training_batch_cap": 1,
                "training_profile": "coverage_micro_canary",
            },
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "watchdog_intelligence_latest.json",
        {
            "overall_status": "degraded",
            "active_issue_count": 1,
            "restart_storm_count": 0,
            "alert_count": 0,
            "exact_needs": [{"severity": "advisory", "risk_level": "low", "blocker": "observer_noise"}],
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 2, "consecutive_runtime_clear_samples": 1, "consecutive_writer_idle_samples": 2}},
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]

    assert payload["budgets"]["watchdogs"]["healthy"] is False
    assert training["allowed"] is True
    assert training["watchdog_training_blocked"] is False
    assert training["profile"] == "coverage_micro_canary"


def test_autonomic_governor_treats_intentionally_held_sleeves_as_nonblocking_for_training(tmp_path: Path) -> None:
    payload = {
        "overall_status": "degraded",
        "active_issue_count": 0,
        "restart_storm_count": 0,
        "alert_count": 0,
        "exact_needs": [
            {"target": "all_sleeves", "status": "intentional_hold", "blocker": "training_window", "risk_level": "low"},
            {
                "target": "baseline_parallel",
                "source": "all_sleeves_launcher_readiness",
                "blocker": "exited",
                "risk_level": "medium",
            },
            {
                "target": "fx",
                "source": "all_sleeves_launcher_readiness",
                "blocker": "exited",
                "risk_level": "low",
            },
        ],
    }

    assert governor._watchdog_blocking_needs(payload) == []
    assert governor._watchdog_blocks_training(payload) is False


def test_autonomic_storage_metrics_trust_explicit_empty_sql_overlay_over_stale_raw_live() -> None:
    metrics = governor._storage_metrics(
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 259.871,
            "backpressure": {
                "total_pending_lines": 12585,
                "core_pending_lines": 7686,
                "deferred_pending_lines": 4899,
                "oldest_pending_age_seconds": 62368.944,
                "pending_lines_threshold": 15000,
                "effective_raw_live": {
                    "total_pending_lines": 12585,
                    "core_pending_lines": 7686,
                    "deferred_pending_lines": 4899,
                    "oldest_pending_age_seconds": 62368.944,
                },
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "fresh_source_count": 18,
                "explicit_empty_source_count": 18,
                "stale_pending_lines": 0,
                "total_pending_lines": 0,
                "files_with_pending": 0,
                "oldest_pending_age_seconds": 0.0,
                "top_pending_files": [],
            },
        }
    )

    assert metrics["total_pending_lines"] == 0
    assert metrics["core_pending_lines"] == 0
    assert metrics["oldest_pending_age_seconds"] == 0.0
    assert metrics["green"] is True
    assert metrics["green_gate"]["direct_sql_overlay_clear"] is True


def test_autonomic_governor_allows_batch20_guarded_waves_under_soft_runtime_pressure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 325,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 325,
                "oldest_pending_age_seconds": 30,
                "pending_lines_threshold": 15000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 0}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 38.0,
            "p_core_runtime_feedback": {"preprocess_worker_budget": 5},
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "running": True,
                "status": "running",
                "current_step": "merge_primary",
                "progress_age_minutes": 2.0,
            },
        },
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "recommended_p_core_worker_cap": 6},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": True,
                "small_canary_training_safe": True,
                "small_batch_training_safe": True,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "batch20_requires_between_target_memory_recheck": True,
                "training_batch_cap": 20,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "timestamp_utc": "2026-05-20T00:00:00+00:00",
            "storage_metrics": {
                "core_pending_lines": 500,
                "total_pending_lines": 500,
                "overlay_pending_lines": 0,
                "oldest_pending_age_seconds": 45,
            },
            "stability_state": {
                "consecutive_green_samples": 8,
                "consecutive_runtime_clear_samples": 0,
                "consecutive_writer_idle_samples": 0,
                "consecutive_improving_samples": 8,
            },
        },
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]
    reentry = training["reentry_gate"]

    assert training["allowed"] is True
    assert training["profile"] == "coverage_batch20_canary"
    assert reentry["max_parallel_trainings"] == 20
    assert reentry["writer_active_green_safe"] is True
    assert reentry["batch20_runtime_wave_clear"] is True
    assert reentry["batch20_execution_mode"] == "sequential_memory_guarded_waves"


def test_autonomic_governor_uses_host_pressure_attribution_as_control_signal(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 120,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 120,
                "oldest_pending_age_seconds": 10,
                "pending_lines_threshold": 15000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 0}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "high",
            "host_saturation_score": 58.0,
            "p_core_runtime_feedback": {"preprocess_worker_budget": 5},
            "host_pressure_attribution": {
                "bot_owned_cpu_percent": 42.0,
                "external_cpu_percent": 120.0,
                "macos_system_cpu_percent": 92.0,
                "foreground_app_cpu_percent": 0.0,
                "dominant_bucket": "macos_system",
                "external_pressure_dominant": True,
                "system_cotenant_hot": True,
                "support_jobs_hot": False,
                "protected_work_hot": False,
                "hot_external_processes": [
                    {"pid": 707, "cpu_percent": 91.0, "category": "system_cotenant", "command": "spotlightknowledged"}
                ],
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"writer_state_before": {"active": False, "running": False}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 2}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "recommended_p_core_worker_cap": 6},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": True,
                "safe_for_training": True,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "training_batch_cap": 20,
                "consecutive_memory_clear_samples": 4,
            },
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 4, "consecutive_runtime_clear_samples": 0, "consecutive_writer_idle_samples": 4}},
    )

    payload = governor.build_payload(tmp_path)
    source = payload["runtime_pressure_source"]

    assert source["mode"] == "macos_system_cooldown"
    assert source["training_allowed"] is False
    assert payload["adaptive_controls"]["p_core_widening"]["mode"] == "hold_pressure_attribution"
    assert payload["adaptive_controls"]["collector_reopening"]["stage"] == "runtime_pressure_attribution_cooldown"
    assert payload["budgets"]["training"]["allowed"] is False
    assert payload["budgets"]["training"]["reentry_gate"]["host_pressure_attribution_gate"]["system_cotenant_hot"] is True
    assert "host_pressure_attribution_not_clear" in payload["budgets"]["training"]["reentry_gate"]["blockers"]
    assert payload["budgets"]["mlx_gpu_jobs"]["compile_mode"] == "off"
    assert payload["what_do_you_need"]["items"][0]["blocker"] == "runtime_pressure_macos_system_cooldown"
    assert payload["integration_contract"]["uses_runtime_pressure_attribution"] is True


def test_autonomic_governor_allows_micro_canary_from_current_green_storage_when_full_headroom_is_soaking(
    tmp_path: Path,
) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "severity": "stable",
            "storage_plane_contract": {
                "allowed_work": {"training": True},
                "blocked_work": ["storage_resilience_refresh"],
            },
            "backpressure": {
                "core_pending_lines": 0,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 0,
                "oldest_pending_age_seconds": 0.0,
                "pending_lines_threshold": 15000,
                "effective_raw_live": {
                    "core_pending_lines": 0,
                    "total_pending_lines": 0,
                    "oldest_pending_age_seconds": 0.0,
                    "source": "fresh_empty_sql_ingestion_overlay",
                },
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 0}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "high",
            "host_saturation_score": 46.0,
            "p_core_runtime_feedback": {"preprocess_worker_budget": 5},
            "host_pressure_attribution": {
                "bot_owned_cpu_percent": 18.0,
                "paper_execution_cpu_percent": 0.0,
                "research_training_cpu_percent": 0.0,
                "throttle_candidate_support_cpu_percent": 0.0,
                "operator_observability_cpu_percent": 18.0,
                "external_cpu_percent": 80.0,
                "macos_system_cpu_percent": 36.0,
                "foreground_app_cpu_percent": 42.0,
                "dominant_bucket": "foreground_apps",
                "external_pressure_dominant": True,
                "support_jobs_hot": False,
                "protected_work_hot": False,
            },
            "runtime_saturation_governor_v2": {
                "training_policy": {
                    "mode": "micro_canary_only",
                    "training_paused": False,
                    "max_parallel_trainings": 1,
                    "micro_canary_allowed": True,
                }
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"writer_state_before": {"active": False, "running": False}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom", "recommended_p_core_worker_cap": 5},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": True,
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "small_batch_training_safe": True,
                "training_batch_cap": 2,
                "training_profile": "coverage_small_canary",
                "consecutive_memory_clear_samples": 1,
            },
            "trend": {"status": "cooling"},
            "multitasking_headroom": {
                "active": False,
                "level": "interactive_developer",
                "collector_ratio_cap": 0.28,
                "training_allowed_by_multitasking": True,
                "training_max_parallel_trainings": 2,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 0, "consecutive_runtime_clear_samples": 0, "consecutive_writer_idle_samples": 0}},
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]
    reentry = training["reentry_gate"]

    assert payload["backlog_green_gate"]["status"] == "green"
    assert reentry["micro_green_samples_ok"] is True
    assert reentry["micro_host_pressure_allows_training"] is True
    assert training["allowed"] is True
    assert training["profile"] == "coverage_micro_canary"
    assert reentry["max_parallel_trainings"] == 1


def test_runtime_pressure_attribution_treats_low_pressure_operator_activity_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 40.0,
            "throttle_profile": "balanced",
            "host_pressure_attribution": {
                "external_pressure_dominant": True,
                "system_cotenant_hot": False,
                "support_jobs_hot": False,
                "protected_work_hot": False,
                "dominant_bucket": "foreground_operator",
                "foreground_app_cpu_percent": 82.0,
            },
        }
    )

    assert source["mode"] == "operator_foreground_advisory"
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is True
    assert source["collector_ratio_cap"] == 0.35


def test_runtime_pressure_attribution_treats_low_pressure_system_activity_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "ready",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "normal",
            "host_saturation_score": 27.0,
            "throttle_profile": "observe",
            "host_pressure_attribution": {
                "external_pressure_dominant": False,
                "system_cotenant_hot": True,
                "support_jobs_hot": False,
                "protected_work_hot": False,
                "operator_observability_hot": True,
                "dominant_bucket": "bot_owned",
                "foreground_app_cpu_percent": 25.0,
            },
        }
    )

    assert source["mode"] == "macos_system_advisory"
    assert source["low_pressure_system_advisory"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is True
    assert source["collector_ratio_cap"] == 0.35


def test_runtime_pressure_attribution_treats_guarded_foreground_activity_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "advisory",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "high",
            "host_saturation_score": 66.0,
            "throttle_profile": "sustain",
            "host_pressure_attribution": {
                "external_pressure_dominant": True,
                "system_cotenant_hot": False,
                "support_jobs_hot": False,
                "protected_work_hot": False,
                "dominant_bucket": "foreground_apps",
                "foreground_app_cpu_percent": 118.0,
            },
        }
    )

    assert source["mode"] == "operator_foreground_guarded_advisory"
    assert source["guarded_foreground_advisory"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is False
    assert source["collector_ratio_cap"] == 0.28


def test_runtime_pressure_attribution_treats_niced_support_work_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "advisory",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 64.0,
            "throttle_profile": "sustain",
            "host_pressure_attribution": {
                "external_pressure_dominant": False,
                "system_cotenant_hot": False,
                "support_jobs_hot": True,
                "support_hot_low_priority": True,
                "protected_work_hot": False,
                "dominant_bucket": "bot_owned",
                "foreground_app_cpu_percent": 18.0,
            },
        }
    )

    assert source["mode"] == "support_maintenance_niced_advisory"
    assert source["guarded_support_advisory"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is False
    assert source["collector_ratio_cap"] == 0.28


def test_runtime_pressure_attribution_treats_niced_support_with_secondary_system_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 74.0,
            "throttle_profile": "sustain",
            "host_pressure_attribution": {
                "external_pressure_dominant": False,
                "bot_owned_pressure_dominant": True,
                "support_pressure_dominant": True,
                "system_secondary_to_bot_owned": True,
                "support_trim_required": True,
                "system_cotenant_hot": True,
                "support_jobs_hot": True,
                "support_hot_low_priority": True,
                "protected_work_hot": True,
                "protected_pressure_dominant": False,
                "dominant_bucket": "bot_owned",
                "foreground_app_cpu_percent": 40.0,
            },
        }
    )

    assert source["mode"] == "support_maintenance_niced_advisory"
    assert source["guarded_niced_support_advisory"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is False


def test_runtime_pressure_attribution_trims_support_before_macos_when_bot_owned_dominates() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 62.0,
            "throttle_profile": "sustain",
            "host_pressure_attribution": {
                "bot_owned_cpu_percent": 360.0,
                "external_cpu_percent": 220.0,
                "macos_system_cpu_percent": 150.0,
                "throttle_candidate_support_cpu_percent": 225.0,
                "external_pressure_dominant": False,
                "bot_owned_pressure_dominant": True,
                "system_secondary_to_bot_owned": True,
                "support_trim_required": True,
                "system_cotenant_hot": True,
                "support_jobs_hot": True,
                "support_hot_low_priority": False,
                "protected_work_hot": True,
                "dominant_bucket": "bot_owned",
                "foreground_app_cpu_percent": 60.0,
            },
        }
    )

    assert source["mode"] == "trim_support_maintenance"
    assert source["bot_owned_pressure_dominant"] is True
    assert source["system_secondary_to_bot_owned"] is True
    assert source["support_trim_required"] is True
    assert source["training_allowed"] is False
    assert source["collector_reopen_allowed"] is False


def test_runtime_pressure_attribution_treats_foreground_system_mix_as_guarded_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 47.0,
            "throttle_profile": "soft_cap",
            "host_pressure_attribution": {
                "external_pressure_dominant": True,
                "bot_owned_pressure_dominant": False,
                "system_cotenant_hot": True,
                "support_jobs_hot": False,
                "operator_observability_hot": True,
                "protected_work_hot": False,
                "dominant_bucket": "foreground_apps",
                "foreground_app_cpu_percent": 120.0,
                "macos_system_cpu_percent": 54.0,
            },
        }
    )

    assert source["mode"] == "operator_foreground_guarded_advisory"
    assert source["foreground_system_guarded"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is True


def test_runtime_pressure_attribution_treats_operator_observability_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "advisory",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 50.0,
            "throttle_profile": "soft_cap",
            "host_pressure_attribution": {
                "external_pressure_dominant": False,
                "system_cotenant_hot": False,
                "support_jobs_hot": False,
                "operator_observability_hot": True,
                "protected_work_hot": False,
                "dominant_bucket": "operator_observability",
                "foreground_app_cpu_percent": 18.0,
            },
        }
    )

    assert source["mode"] == "operator_observability_guarded_advisory"
    assert source["guarded_operator_observability_advisory"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is True
    assert source["collector_ratio_cap"] == 0.28


def test_runtime_pressure_attribution_treats_bounded_protected_work_as_advisory() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "advisory",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "normal",
            "host_saturation_score": 38.0,
            "throttle_profile": "soft_cap",
            "host_pressure_attribution": {
                "external_pressure_dominant": True,
                "system_cotenant_hot": False,
                "support_jobs_hot": False,
                "operator_observability_hot": True,
                "protected_work_hot": True,
                "dominant_bucket": "bot_owned",
                "foreground_app_cpu_percent": 80.0,
            },
        }
    )

    assert source["mode"] == "protected_work_guarded_advisory"
    assert source["guarded_protected_work_advisory"] is True
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is True
    assert source["collector_ratio_cap"] == 0.28


def test_runtime_pressure_attribution_allows_micro_training_when_only_support_jobs_are_warm() -> None:
    source = governor._runtime_pressure_attribution_policy(
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "normal",
            "host_saturation_score": 43.0,
            "throttle_profile": "soft_cap",
            "host_pressure_attribution": {
                "external_pressure_dominant": False,
                "system_cotenant_hot": False,
                "support_jobs_hot": True,
                "protected_work_hot": False,
                "dominant_bucket": "bot_owned",
                "foreground_app_cpu_percent": 38.0,
            },
        }
    )

    assert source["mode"] == "support_maintenance_advisory"
    assert source["training_allowed"] is True
    assert source["collector_reopen_allowed"] is True
    assert source["p_core_widen_allowed"] is False
    assert source["collector_ratio_cap"] == 0.28


def test_runtime_throttle_treats_ytdlp_metadata_probe_as_throttleable_support() -> None:
    row = runtime_throttle_control._classify_process(
        "/opt/homebrew/bin/yt-dlp --dump-single-json --no-playlist https://www.youtube.com/watch?v=test"
    )

    assert row["category"] == "support_maintenance"
    assert row["priority_tier"] == "throttle_first"
    assert row["throttle_candidate"] is True


def test_runtime_throttle_does_not_call_allocation_only_compression_high_pressure() -> None:
    resource_guard = {
        "memory_pressure_state": "green",
        "memory_pressure_kind": "none",
        "swap_used_gb": 1.8,
    }
    memory_efficiency = {
        "overall_status": "blocked",
        "reasons": ["storage_pressure_critical", "compressed_memory_high"],
        "memory_snapshot": {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 52.0,
            "swap_used_gb": 1.8,
            "compressed_store_gb": 24.0,
            "compressor_gb": 10.7,
        },
        "cotenant_awareness": {"memory_pressure_clear": True},
    }

    assert runtime_throttle_control._memory_pressure_level(resource_guard, memory_efficiency) == "normal"


def test_autonomic_governor_caps_p_core_workers_when_memory_pressure_rises(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 5.0,
        "pressure_level": "normal",
        "memory_snapshot": {"compressed_store_gb": 15.0, "swap_used_gb": 5.0, "memory_pressure_kind": "none"},
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=host))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 700,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 700,
                "oldest_pending_age_seconds": 100,
                "pending_lines_threshold": 5000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 700}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 6}},
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"drain_effectiveness": {"status": "strong_progress", "merged_rows": 2000}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(health / "autonomic_resource_governor_latest.json", {"stability_state": {"consecutive_green_samples": 2, "consecutive_runtime_clear_samples": 2, "consecutive_writer_idle_samples": 2}})

    payload = governor.build_payload(tmp_path)
    pcore = payload["adaptive_controls"]["p_core_widening"]

    assert pcore["mode"] == "memory_pressure_cap"
    assert pcore["selected_workers"] == 3
    assert pcore["memory_pressure_controller"]["status"] == "compression_relief"
    assert pcore["memory_pressure_controller"]["max_memory_safe_workers"] == 3


def test_memory_pressure_intelligence_reserves_headroom_for_creative_apps(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 1.0,
        "pressure_level": "normal",
        "memory_snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 1.0, "memory_pressure_kind": "none"},
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_latest.json", {"overall_status": "ready", "memory_snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 1.0}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": ["Logic Pro", "Final Cut Pro"], "creative_level": "active"}})

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "foreground_headroom"
    assert payload["classification"]["recommended_p_core_worker_cap"] == 3
    assert payload["multitasking_headroom"]["level"] == "realtime_creative"
    assert payload["multitasking_headroom"]["training_allowed_by_multitasking"] is False
    assert payload["reopen_gate"]["safe_for_training"] is False


def test_memory_pressure_intelligence_treats_safe_media_headroom_as_managed_control(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 0.2,
        "pressure_level": "normal",
        "memory_snapshot": {
            "compressed_store_gb": 2.4,
            "compressor_gb": 2.4,
            "swap_used_gb": 0.2,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(
        health / "memory_efficiency_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "compressed_store_gb": 2.4,
                "compressor_gb": 2.4,
                "swap_used_gb": 0.2,
                "memory_pressure_kind": "none",
            },
        },
    )
    _write_json(
        health / "computer_task_intelligence_latest.json",
        {"timestamp_utc": "2099-01-01T00:00:00+00:00", "session_context": {"open_apps": ["Music"], "creative_level": "active"}},
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "reopen_gate": {"consecutive_memory_clear_samples": 1, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_pressure_gb": 2.4, "swap_used_gb": 0.2, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "foreground_headroom"
    assert payload["reopen_gate"]["safe_to_widen_p_core_workers"] is True
    assert payload["what_do_you_need"]["status"] == "clear"
    assert payload["what_do_you_need"]["items"] == []
    assert payload["managed_controls"][0]["id"] == "foreground_app_headroom_reserved"
    assert payload["managed_controls"][0]["status"] == "managed"
    assert payload["managed_controls"][0]["app_classes"] == ["media_playback"]


def test_memory_pressure_intelligence_allows_micro_canary_under_warm_background_memory(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 1.0,
        "pressure_level": "normal",
        "memory_snapshot": {"compressed_store_gb": 10.8, "swap_used_gb": 1.0, "memory_pressure_kind": "none"},
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "degraded", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_latest.json", {"overall_status": "ready", "memory_snapshot": {"compressed_store_gb": 10.8, "swap_used_gb": 1.0}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "reopen_gate": {"consecutive_memory_clear_samples": 0, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 10.8, "swap_used_gb": 1.0, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "soft_guard"
    assert payload["reopen_gate"]["safe_for_training"] is False
    assert payload["reopen_gate"]["small_canary_training_safe"] is True
    assert payload["workload_guidance"]["small_canary_training_allowed_by_memory"] is True


def test_memory_pressure_intelligence_uses_compressor_pressure_for_high_free_memory(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 1.8,
        "pressure_level": "normal",
        "memory_snapshot": {
            "compressed_store_gb": 6.0,
            "compressor_gb": 7.0,
            "swap_used_gb": 1.8,
            "memory_free_pct": 85.0,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "memory_pressure_level": "high",
            "p_core_runtime_feedback": {
                "p_core_burst_intelligence": {
                    "inputs": {
                        "compressed_store_gb": 24.0,
                        "swap_used_gb": 1.8,
                        "pages_throttled": 0,
                        "memory_pressure_kind": "none",
                    }
                }
            },
        },
    )
    _write_json(
        health / "memory_efficiency_latest.json",
        {
            "overall_status": "blocked",
            "memory_snapshot": {
                "compressed_store_gb": 24.0,
                "compressor_gb": 10.7,
                "swap_used_gb": 1.8,
                "memory_free_pct": 85.0,
                "memory_pressure_kind": "none",
            },
        },
    )
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "reopen_gate": {"consecutive_memory_clear_samples": 0, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_pressure_gb": 10.8, "swap_used_gb": 1.8, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["snapshot"]["compressed_store_gb"] == 24.0
    assert payload["snapshot"]["compressed_pressure_gb"] == 10.7
    assert payload["classification"]["status"] == "soft_guard"
    assert payload["reopen_gate"]["small_batch_training_safe"] is True
    assert payload["reopen_gate"]["training_batch_cap"] == 2


def test_memory_pressure_intelligence_allows_one_bot_canary_under_bounded_compression_relief(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 2.6,
        "pressure_level": "elevated",
        "memory_snapshot": {"compressed_store_gb": 18.5, "swap_used_gb": 2.6, "memory_pressure_kind": "none"},
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "degraded", "memory_pressure_level": "elevated"})
    _write_json(health / "memory_efficiency_latest.json", {"overall_status": "needs_work", "memory_snapshot": {"compressed_store_gb": 18.5, "swap_used_gb": 2.6}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "reopen_gate": {"consecutive_memory_clear_samples": 0, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 18.5, "swap_used_gb": 2.6, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "compression_relief"
    assert payload["reopen_gate"]["safe_for_training"] is False
    assert payload["reopen_gate"]["compression_relief_micro_canary_safe"] is True
    assert payload["reopen_gate"]["small_canary_training_safe"] is True
    assert payload["reopen_gate"]["training_batch_cap"] == 1
    assert payload["reopen_gate"]["training_profile"] == "coverage_micro_canary"


def test_memory_pressure_intelligence_allows_batch20_waves_under_high_free_compression_relief(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 1.8,
        "pressure_level": "elevated",
        "memory_snapshot": {
            "compressed_store_gb": 18.8,
            "swap_used_gb": 1.8,
            "memory_free_pct": 85.0,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "degraded", "memory_pressure_level": "elevated"})
    _write_json(
        health / "memory_efficiency_latest.json",
        {"overall_status": "needs_work", "memory_snapshot": {"compressed_store_gb": 18.8, "swap_used_gb": 1.8, "memory_free_pct": 85.0}},
    )
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "reopen_gate": {"consecutive_memory_clear_samples": 0, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 18.8, "swap_used_gb": 1.8, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "compression_relief"
    assert payload["classification"]["recommended_p_core_worker_cap"] == 3
    assert payload["reopen_gate"]["safe_for_training"] is False
    assert payload["reopen_gate"]["compression_relief_micro_canary_safe"] is True
    assert payload["reopen_gate"]["compression_relief_batch20_wave_training_safe"] is True
    assert payload["reopen_gate"]["batch20_training_safe"] is True
    assert payload["reopen_gate"]["batch20_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["reopen_gate"]["batch20_wave_size"] == 3
    assert payload["reopen_gate"]["batch20_requires_between_target_memory_recheck"] is True
    assert payload["reopen_gate"]["batch30_training_safe"] is True
    assert payload["reopen_gate"]["batch30_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["reopen_gate"]["batch30_wave_size"] == 3
    assert payload["reopen_gate"]["batch30_requires_between_target_memory_recheck"] is True
    assert payload["reopen_gate"]["training_batch_cap"] == 30
    assert payload["reopen_gate"]["training_profile"] == "coverage_batch30_canary"


def test_memory_pressure_intelligence_ignores_stale_host_foreground_when_computer_task_is_fresh(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["timestamp_utc"] = "2026-05-20T00:00:00+00:00"
    host["body_map"]["foreground_apps_and_user_activity"] = {
        "source": "computer_task_intelligence",
        "creative_kind": "none",
        "creative_level": "none",
        "co_running_level": "none",
        "open_apps": ["Music"],
        "user_coexistent_required": False,
    }
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 1.0,
        "pressure_level": "normal",
        "memory_snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 1.0, "memory_pressure_kind": "none"},
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_latest.json", {"overall_status": "ready", "memory_snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 1.0}})
    _write_json(
        health / "computer_task_intelligence_latest.json",
        {
            "timestamp_utc": "2026-05-21T00:00:00+00:00",
            "session_context": {
                "open_apps": [],
                "creative_level": "none",
                "co_running_level": "none",
                "process_context_infrabot": {"ignored_memory_efficiency_app_context": False},
            },
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["snapshot"]["open_apps"] == []
    assert payload["snapshot"]["user_active"] is False
    assert payload["snapshot"]["app_context_quality"]["ignored_stale_host_foreground"] is True
    assert payload["multitasking_headroom"]["level"] == "background_available"


def test_memory_pressure_intelligence_opens_batch20_after_clear_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 0.2,
        "pressure_level": "normal",
        "memory_snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 0.2, "memory_pressure_kind": "none"},
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_latest.json", {"overall_status": "ready", "memory_snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 0.2}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": "2026-05-21T00:00:00+00:00",
            "reopen_gate": {"consecutive_memory_clear_samples": 4, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 6.0, "swap_used_gb": 0.2, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "clear"
    assert payload["reopen_gate"]["batch10_training_safe"] is True
    assert payload["reopen_gate"]["batch20_training_safe"] is True
    assert payload["reopen_gate"]["batch30_training_safe"] is True
    assert payload["reopen_gate"]["training_batch_cap"] == 30
    assert payload["workload_guidance"]["training_profile"] == "coverage_batch30_canary"


def test_memory_pressure_intelligence_allows_batch20_as_memory_guarded_waves_when_headroom_is_high(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 1.4,
        "pressure_level": "normal",
        "memory_snapshot": {
            "compressed_store_gb": 8.8,
            "swap_used_gb": 1.4,
            "memory_free_pct": 86.0,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(
        health / "memory_efficiency_latest.json",
        {"overall_status": "ready", "memory_snapshot": {"compressed_store_gb": 8.8, "swap_used_gb": 1.4, "memory_free_pct": 86.0}},
    )
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": "2026-05-21T00:00:00+00:00",
            "reopen_gate": {"consecutive_memory_clear_samples": 4, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 4.8, "swap_used_gb": 0.8, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "heating_guard"
    assert payload["reopen_gate"]["safe_to_widen_p_core_workers"] is False
    assert payload["reopen_gate"]["batch20_training_safe"] is True
    assert payload["reopen_gate"]["batch20_wave_training_safe"] is True
    assert payload["reopen_gate"]["batch20_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["reopen_gate"]["batch20_requires_between_target_memory_recheck"] is True
    assert payload["reopen_gate"]["training_batch_cap"] == 20


def test_memory_pressure_intelligence_reopens_pcores_for_stale_allocation_high_water(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 7.645,
        "pressure_level": "elevated",
        "compressor_gb": 0.386,
        "memory_snapshot": {
            "compressed_store_gb": 19.552,
            "compressor_gb": 0.386,
            "swap_used_gb": 7.645,
            "memory_free_pct": 91.0,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "degraded", "memory_pressure_level": "elevated"})
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "memory_free_pct": 91.0,
                "swap_used_gb": 0.185,
                "compressed_store_gb": 8.0,
                "compressor_gb": 0.386,
                "allocation_relief_active": True,
            },
        },
    )
    _write_json(
        health / "swap_pressure_governor_latest.json",
        {
            "swap_pressure": {
                "tier": "normal",
                "swap_used_gb": 0.185,
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
            }
        },
    )
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": "2026-05-21T00:00:00+00:00",
            "reopen_gate": {"consecutive_memory_clear_samples": 1, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 19.552, "swap_used_gb": 7.645, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["snapshot"]["memory_truth_reconciliation"]["active"] is True
    assert payload["snapshot"]["raw_allocation_snapshot"]["swap_used_gb"] == 7.645
    assert payload["snapshot"]["swap_used_gb"] == 0.185
    assert payload["snapshot"]["compressed_store_gb"] == 8.0
    assert payload["classification"]["status"] == "clear"
    assert payload["classification"]["recommended_p_core_worker_cap"] == 7
    assert payload["reopen_gate"]["safe_to_widen_p_core_workers"] is True


def test_memory_pressure_intelligence_allows_batch20_on_single_deep_green_sample(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 1.2,
        "pressure_level": "normal",
        "memory_snapshot": {
            "compressed_store_gb": 8.5,
            "swap_used_gb": 1.2,
            "memory_free_pct": 88.0,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(
        health / "memory_efficiency_latest.json",
        {"overall_status": "ready", "memory_snapshot": {"compressed_store_gb": 8.5, "swap_used_gb": 1.2, "memory_free_pct": 88.0}},
    )
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": "2026-05-21T00:00:00+00:00",
            "reopen_gate": {"consecutive_memory_clear_samples": 0, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_store_gb": 8.5, "swap_used_gb": 1.2, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "clear"
    assert payload["reopen_gate"]["single_sample_deep_green_batch_widening"] is True
    assert payload["reopen_gate"]["batch20_training_safe"] is True
    assert payload["reopen_gate"]["batch20_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["reopen_gate"]["batch30_training_safe"] is True
    assert payload["reopen_gate"]["batch30_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["reopen_gate"]["training_batch_cap"] == 30


def test_memory_pressure_intelligence_opens_weekend_media_soft_guard_batch_waves(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAINING_WEEKEND_BATCH_WINDOW", "1")
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 32,
        "swap_used_gb": 2.58,
        "pressure_level": "normal",
        "memory_snapshot": {
            "compressed_store_gb": 9.7,
            "compressor_gb": 1.1,
            "swap_used_gb": 2.58,
            "memory_free_pct": 92.0,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal"},
    )
    _write_json(
        health / "memory_efficiency_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "compressed_store_gb": 9.7,
                "compressor_gb": 1.1,
                "swap_used_gb": 2.58,
                "memory_free_pct": 92.0,
                "memory_pressure_kind": "none",
            },
        },
    )
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": ["Music"], "creative_level": "none"}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "reopen_gate": {"consecutive_memory_clear_samples": 0, "consecutive_cooling_samples": 0},
            "snapshot": {"compressed_pressure_gb": 1.1, "swap_used_gb": 2.58, "pages_throttled": 0},
        },
    )

    payload = memory_pressure_intelligence.build_payload(tmp_path)

    assert payload["classification"]["status"] == "soft_guard"
    assert payload["multitasking_headroom"]["weekend_media_training_window"] is True
    assert payload["multitasking_headroom"]["training_max_parallel_trainings"] == 30
    assert payload["reopen_gate"]["weekend_large_batch_window"] is True
    assert payload["reopen_gate"]["batch20_training_safe"] is True
    assert payload["reopen_gate"]["batch30_training_safe"] is True
    assert payload["reopen_gate"]["batch30_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["reopen_gate"]["training_batch_cap"] == 30


def test_autonomic_governor_allows_weekend_media_large_batch_while_user_active() -> None:
    gate = governor._training_reentry_gate(
        {
            "green": True,
            "total_pending_lines": 1200,
            "target_pending_lines": 15000,
            "oldest_pending_age_seconds": 8,
            "severity": "ready",
            "green_gate": {"line_green": True, "age_green": True},
        },
        {
            "overall_status": "ready",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 47.0,
        },
        {"user_active": True},
        {"overall_status": "ready"},
        {"writer_state_before": {"current_step": "complete", "status": "ok", "running": False}},
        {"status": "improving"},
        {"consecutive_green_samples": 3, "trend_regressing": False},
        {
            "safe_for_training": False,
            "batch10_training_safe": True,
            "batch20_training_safe": True,
            "batch20_execution_mode": "sequential_memory_guarded_waves",
            "batch30_training_safe": True,
            "batch30_execution_mode": "sequential_memory_guarded_waves",
            "multitasking_training_cap": 30,
            "multitasking_headroom_level": "media_playback",
            "weekend_large_batch_window": True,
        },
        {"training_allowed": True, "mode": "clear", "attribution": {}},
    )

    assert gate["allowed"] is True
    assert gate["mode"] == "batch30_canary"
    assert gate["profile"] == "coverage_batch30_canary"
    assert gate["max_parallel_trainings"] == 30
    assert gate["user_active_large_batch_allowed"] is True
    assert "foreground_user_apps_active" not in gate["blockers"]


def test_autonomic_governor_allows_weekend_batch_when_control_plane_support_is_hot() -> None:
    gate = governor._training_reentry_gate(
        {
            "green": True,
            "total_pending_lines": 1200,
            "target_pending_lines": 15000,
            "oldest_pending_age_seconds": 8,
            "severity": "ready",
            "green_gate": {"line_green": True, "age_green": True},
        },
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "compute_pressure_level": "elevated",
            "host_saturation_score": 49.0,
        },
        {"user_active": True},
        {"overall_status": "ready"},
        {"writer_state_before": {"current_step": "complete", "status": "ok", "running": False}},
        {"status": "improving"},
        {"consecutive_green_samples": 3, "trend_regressing": False},
        {
            "safe_for_training": False,
            "batch10_training_safe": True,
            "batch20_training_safe": True,
            "batch20_execution_mode": "sequential_memory_guarded_waves",
            "batch30_training_safe": True,
            "batch30_execution_mode": "sequential_memory_guarded_waves",
            "multitasking_training_cap": 30,
            "multitasking_headroom_level": "media_playback",
            "weekend_large_batch_window": True,
        },
        {
            "training_allowed": False,
            "mode": "trim_support_maintenance",
            "support_jobs_hot": True,
            "support_hot_low_priority": True,
            "protected_work_hot": False,
            "attribution": {
                "hot_support_processes": [
                    {"command_excerpt": "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/creative_cotenant_guard.py"}
                ]
            },
        },
    )

    assert gate["allowed"] is True
    assert gate["mode"] == "batch30_canary"
    assert gate["weekend_control_plane_pressure_allowed"] is True
    assert gate["micro_host_pressure_allows_training"] is True
    assert "host_pressure_attribution_not_clear" not in gate["blockers"]


def test_autonomic_governor_consumes_memory_intelligence_and_limits_e_core_spillover(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 700,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 700,
                "oldest_pending_age_seconds": 100,
                "pending_lines_threshold": 5000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 700}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 6}},
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"drain_effectiveness": {"status": "strong_progress", "merged_rows": 2000}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": ["Logic Pro"], "creative_level": "active"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom", "recommended_p_core_worker_cap": 3, "reason": "creative reserve"},
            "reopen_gate": {"safe_to_widen_p_core_workers": False, "safe_for_training": False, "consecutive_memory_clear_samples": 0},
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "realtime_creative", "collector_ratio_cap": 0.12},
        },
    )
    _write_json(health / "autonomic_resource_governor_latest.json", {"stability_state": {"consecutive_green_samples": 2, "consecutive_runtime_clear_samples": 2, "consecutive_writer_idle_samples": 2}})

    payload = governor.build_payload(tmp_path)

    assert payload["host_lane_budget"]["selected_p_core_preprocess_workers"] == 3
    assert payload["host_lane_budget"]["p_core_allocation_contract"]["user_app_reserved_p_cores"] == 5
    assert payload["host_lane_budget"]["efficiency_core_spillover"] == 1
    assert payload["adaptive_controls"]["efficiency_core_pressure_guard"]["mode"] == "p_core_primary_foreground_reserve"
    assert payload["budgets"]["training"]["allowed"] is False


def test_autonomic_governor_allows_micro_canary_when_backlog_green_and_writer_active_fresh(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 70,
                "support_pending_lines": 0,
                "deferred_pending_lines": 5,
                "total_pending_lines": 75,
                "oldest_pending_age_seconds": 0,
                "pending_lines_threshold": 15000,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "degraded", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 4}},
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "shard_linking",
                "completed_shard_count": 12,
                "planned_shard_count": 14,
                "progress_age_minutes": 0.5,
            }
        },
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "advisory", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 4}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard", "recommended_p_core_worker_cap": 4},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": False,
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "small_canary_max_parallel_trainings": 1,
                "consecutive_memory_clear_samples": 0,
            },
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 2, "consecutive_runtime_clear_samples": 0, "consecutive_writer_idle_samples": 0}},
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]

    assert training["allowed"] is True
    assert training["mode"] == "micro_canary"
    assert training["profile"] == "coverage_micro_canary"
    assert training["reentry_gate"]["writer_active_green_safe"] is True
    assert training["reentry_gate"]["writer_idle_required"] is False


def test_autonomic_governor_allows_micro_canary_when_live_backlog_green_but_overlay_lags_under_system_pressure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(health / "workload_class_registry_latest.json", workload_class_registry.build_payload())
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "severity": "stable",
            "backpressure": {
                "core_pending_lines": 1323,
                "support_pending_lines": 15,
                "deferred_pending_lines": 734,
                "total_pending_lines": 2057,
                "oldest_pending_age_seconds": 65.0,
                "pending_lines_threshold": 15000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 29141}},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "memory_pressure_level": "elevated",
            "compute_pressure_level": "high",
            "host_saturation_score": 92.5,
            "p_core_runtime_feedback": {"preprocess_worker_budget": 3},
            "host_pressure_attribution": {
                "bot_owned_cpu_percent": 80.2,
                "external_cpu_percent": 175.6,
                "macos_system_cpu_percent": 20.5,
                "foreground_app_cpu_percent": 66.4,
                "dominant_bucket": "bot_owned",
                "external_pressure_dominant": False,
                "system_cotenant_hot": False,
                "support_jobs_hot": True,
                "hot_support_processes": [
                    {
                        "pid": 101,
                        "nice": 0,
                        "cpu_percent": 89.1,
                        "command_excerpt": "/opt/homebrew/bin/python scripts/ops/creative_cotenant_guard.py",
                    }
                ],
                "protected_work_hot": False,
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"writer_state_before": {"active": False, "running": False}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 3}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "compression_relief", "recommended_p_core_worker_cap": 3},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": False,
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "small_canary_max_parallel_trainings": 1,
                "compression_relief_micro_canary_safe": True,
                "training_batch_cap": 1,
                "training_profile": "coverage_micro_canary",
                "consecutive_memory_clear_samples": 0,
            },
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 0, "consecutive_runtime_clear_samples": 0, "consecutive_writer_idle_samples": 18}},
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]

    assert training["allowed"] is True
    assert training["mode"] == "micro_canary"
    assert training["profile"] == "coverage_micro_canary"
    assert training["reentry_gate"]["micro_backlog_green"] is True
    assert training["reentry_gate"]["micro_host_pressure_allows_training"] is True
    assert training["reentry_gate"]["runtime_micro_clear"] is True
    assert training["reentry_gate"]["support_hot_is_control_plane"] is True
    assert training["reentry_gate"]["max_parallel_trainings"] == 1


def test_autonomic_governor_prearms_storage_requested_pcore_pump_while_writer_active(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 2_100_000,
                "support_pending_lines": 0,
                "deferred_pending_lines": 1_900_000,
                "total_pending_lines": 4_400_000,
                "oldest_pending_age_seconds": 29_000,
                "pending_lines_threshold": 15000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 2_200_000}},
            "backlog_relief_contract": {
                "p_core_backlog_allocation_contract": {
                    "preprocess_worker_budget": 3,
                    "shard_link_writer_lanes": 3,
                    "p_core_burst_intelligence": {
                        "mode": "guarded_backlog_probe_3",
                        "reason": "severe backlog with normal memory",
                    },
                }
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "host_saturation_score": 59.5,
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "p_core_runtime_feedback": {"preprocess_worker_budget": 2},
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "shard_linking",
                "completed_shard_count": 15,
                "planned_shard_count": 26,
                "progress_age_minutes": 0.1,
            },
            "summary": {"writer_progress_observed": True},
            "drain_effectiveness": {"status": "progress", "merged_rows": 5000},
        },
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "advisory", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 4}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard", "recommended_p_core_worker_cap": 4},
            "reopen_gate": {"safe_to_widen_p_core_workers": False, "safe_for_training": False, "consecutive_memory_clear_samples": 0},
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 0, "consecutive_runtime_clear_samples": 0, "consecutive_writer_idle_samples": 0}},
    )

    payload = governor.build_payload(tmp_path)

    assert payload["host_lane_budget"]["selected_p_core_preprocess_workers"] == 3
    assert payload["host_lane_budget"]["p_core_widening_controller"]["mode"] == "hold_active_writer_prearmed_storage_target"
    assert payload["host_lane_budget"]["p_core_widening_controller"]["storage_requested_workers"] == 3
    assert payload["budgets"]["backlog_writer"]["p_core_preprocess_workers"] == 3


def test_autonomic_governor_keeps_fourth_backlog_worker_when_compression_is_allocation_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    host["body_map"]["memory"] = {
        "memory_gb": 64,
        "swap_used_gb": 1.4,
        "pressure_level": "normal",
        "memory_snapshot": {
            "compressed_store_gb": 16.9,
            "compressor_gb": 5.3,
            "compressed_pressure_gb": 5.3,
            "swap_used_gb": 1.4,
            "memory_pressure_kind": "none",
        },
    }
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=host))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 440_000,
                "support_pending_lines": 0,
                "deferred_pending_lines": 200,
                "total_pending_lines": 440_200,
                "oldest_pending_age_seconds": 120_000,
                "pending_lines_threshold": 5000,
            },
            "backlog_truth": {"sql_overlay": {"total_pending_lines": 440_200}},
            "backlog_relief_contract": {
                "p_core_backlog_allocation_contract": {
                    "preprocess_worker_budget": 4,
                    "shard_link_writer_lanes": 4,
                    "p_core_burst_intelligence": {
                        "mode": "guarded_backlog_probe_4",
                        "reason": "borrow one reserved P-core for extreme backlog when memory is allocation-only",
                    },
                }
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "host_saturation_score": 42.0,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "p_core_runtime_feedback": {
                "preprocess_worker_budget": 4,
                "p_core_burst_intelligence": {
                    "mode": "guarded_backlog_probe_4",
                    "user_app_reserve": {"elastic_loan_allowed": True, "elastic_loan_worker_cap": 4},
                    "inputs": {
                        "compressed_store_gb": 16.9,
                        "compressor_gb": 5.3,
                        "compressed_pressure_gb": 5.3,
                        "swap_used_gb": 1.4,
                        "pages_throttled": 0,
                        "memory_pressure_kind": "none",
                    },
                },
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "writer_state_before": {"active": False, "running": False, "current_step": "complete", "status": "ok"},
            "drain_effectiveness": {"status": "strong_progress", "merged_rows": 6000},
        },
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "advisory", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {
                "status": "compression_relief",
                "recommended_p_core_worker_cap": 3,
                "reason": "compressed memory is elevated enough to hold P-core width",
            },
            "reopen_gate": {"safe_to_widen_p_core_workers": False, "safe_for_training": False, "consecutive_memory_clear_samples": 0},
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(health / "autonomic_resource_governor_latest.json", {"stability_state": {"consecutive_green_samples": 0, "consecutive_runtime_clear_samples": 0, "consecutive_writer_idle_samples": 1}})

    payload = governor.build_payload(tmp_path)
    widening = payload["host_lane_budget"]["p_core_widening_controller"]

    assert payload["host_lane_budget"]["selected_p_core_preprocess_workers"] == 4
    assert widening["storage_requested_workers"] == 4
    assert widening["max_safe_workers"] == 4
    assert widening["elastic_reserve_loan_allowed"] is True
    assert widening["memory_pressure_controller"]["allocation_only_compression"] is True
    assert widening["memory_pressure_controller"]["max_memory_safe_workers"] == 4


def test_autonomic_governor_allows_batch10_when_backlog_and_memory_are_clear(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 50,
                "support_pending_lines": 0,
                "deferred_pending_lines": 0,
                "total_pending_lines": 50,
                "oldest_pending_age_seconds": 0,
                "pending_lines_threshold": 15000,
            },
        },
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 6}})
    _write_json(health / "writer_cycle_coordinator_latest.json", {"writer_state_before": {"active": False, "running": False}})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 6}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "recommended_p_core_worker_cap": 6},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": True,
                "safe_for_training": True,
                "batch10_training_safe": True,
                "batch10_max_parallel_trainings": 10,
                "batch20_training_safe": False,
                "training_batch_cap": 10,
                "training_profile": "coverage_batch10_canary",
                "consecutive_memory_clear_samples": 4,
            },
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 3, "consecutive_runtime_clear_samples": 3, "consecutive_writer_idle_samples": 3}},
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]

    assert training["allowed"] is True
    assert training["mode"] == "batch10_canary"
    assert training["profile"] == "coverage_batch10_canary"
    assert training["reentry_gate"]["max_parallel_trainings"] == 10
    assert training["reentry_gate"]["recommended_command"][-2:] == ["coverage_batch10_canary", "--json"]


def test_autonomic_governor_allows_four_bot_canary_during_green_safe_writer_pass(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=_host_payload()))
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 80,
                "support_pending_lines": 0,
                "deferred_pending_lines": 20,
                "total_pending_lines": 100,
                "oldest_pending_age_seconds": 0,
                "pending_lines_threshold": 15000,
            },
        },
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "degraded", "memory_pressure_level": "normal", "p_core_runtime_feedback": {"preprocess_worker_budget": 5}})
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "shard_linking",
                "completed_shard_count": 25,
                "planned_shard_count": 26,
                "progress_age_minutes": 0.4,
            }
        },
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "runtime_caps": {"max_concurrent_mlx_jobs": 1}})
    _write_json(health / "computer_task_intelligence_latest.json", {"session_context": {"open_apps": [], "co_running_level": "none"}})
    _write_json(health / "host_self_benchmark_latest.json", {"self_tuned_limits": {"recommended_p_core_preprocess_workers": 5}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "recommended_p_core_worker_cap": 6},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": True,
                "safe_for_training": True,
                "small_batch_training_safe": True,
                "small_batch_max_parallel_trainings": 2,
                "batch10_training_safe": True,
                "batch10_max_parallel_trainings": 10,
                "batch20_training_safe": True,
                "batch20_max_parallel_trainings": 20,
                "training_batch_cap": 20,
                "training_profile": "coverage_batch20_canary",
                "consecutive_memory_clear_samples": 6,
            },
            "trend": {"status": "flat"},
            "multitasking_headroom": {"level": "background_available", "collector_ratio_cap": 0.55},
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {"stability_state": {"consecutive_green_samples": 4, "consecutive_runtime_clear_samples": 4, "consecutive_writer_idle_samples": 0}},
    )

    payload = governor.build_payload(tmp_path)
    training = payload["budgets"]["training"]

    assert training["allowed"] is True
    assert training["mode"] == "small_targeted"
    assert training["profile"] == "coverage_canary"
    assert training["reentry_gate"]["max_parallel_trainings"] == 4
    assert training["reentry_gate"]["writer_active_green_safe"] is True
    assert training["reentry_gate"]["writer_active_small_batch_safe"] is True


def test_migration_readiness_flags_linux_gpu_and_preserves_video_denylist(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    host = _host_payload()
    _write_json(health / "host_capability_contract_latest.json", host)
    _write_json(health / "os_adapter_layer_latest.json", os_adapter_layer.build_payload(host=host))

    payload = migration_readiness_report.build_payload(tmp_path, target_os="linux")

    areas = {item["area"]: item for item in payload["migration_items"]}
    assert payload["overall_status"] in {"advisory", "needs_work"}
    assert areas["gpu_backend"]["status"] == "needs_backend_switch"
    assert areas["protected_volumes"]["status"] == "ready"
    assert "/Volumes/VIDEO" in areas["protected_volumes"]["current"]
    assert payload["migration_binder"]["protected_volume_rule"]["path"] == "/Volumes/VIDEO"
    assert any(step["step"] == "switch_gpu_backend_contract" for step in payload["migration_binder"]["operator_rebind_steps"])


def test_host_self_benchmark_uses_safe_limits(monkeypatch, tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "host_capability_contract_latest.json", _host_payload())
    monkeypatch.setattr(host_self_benchmark, "_jsonl_parse_benchmark", lambda rows=5000: {"rows": rows, "rows_per_second": 100000.0})
    monkeypatch.setattr(host_self_benchmark, "_sqlite_latency_benchmark", lambda directory, rows=1500: {"rows": rows, "rows_per_second": 30000.0, "benchmark_dir": str(directory), "protected_volume_safe": True})
    monkeypatch.setattr(host_self_benchmark, "_storage_write_latency", lambda directory, bytes_to_write=1024 * 1024: {"mb_per_second": 200.0, "benchmark_dir": str(directory), "protected_volume_safe": True})

    payload = host_self_benchmark.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["protected_volume_policy"]["protected_volume_safe"] is True
    assert payload["self_tuned_limits"]["recommended_p_core_preprocess_workers"] == 7


def test_system_needs_intelligence_preserves_fix_frames(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "what_do_you_need": {
                "items": [
                    {
                        "blocker": "backlog_above_target_or_old_pending_work",
                        "exact_file": "decision.jsonl",
                        "exact_shard": "trading",
                        "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
                        "expected_impact": "drain",
                        "risk_level": "low",
                        "stop_when": "green",
                    }
                ]
            }
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"drain_effectiveness": {"status": "strong_progress"}})
    fix_log = tmp_path / "governance" / "health" / "system_needs_fix_log.jsonl"
    fix_log.write_text(json.dumps({"timestamp_utc": "2026-05-20T00:00:00+00:00", "fix": "writer pass", "result": "strong"}) + "\n", encoding="utf-8")

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=fix_log)

    assert payload["overall_status"] == "needs_action"
    assert payload["what_do_you_need"][0]["exact_shard"] == "trading"
    assert payload["frames_of_reference"]["recent_fix_log"][0]["fix"] == "writer pass"


def test_system_needs_includes_green_gate_and_migration_binder_frames(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "backlog_green_gate": {"status": "green"},
            "adaptive_controls": {"collector_reopening": {"stage": "normal_reopen"}},
            "what_do_you_need": {"items": []},
        },
    )
    _write_json(health / "migration_readiness_report_latest.json", {"migration_binder": {"enabled": True}})

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "ready"
    assert payload["frames_of_reference"]["backlog_green_gate"]["status"] == "green"
    assert payload["frames_of_reference"]["adaptive_controls"]["collector_reopening"]["stage"] == "normal_reopen"
    assert payload["frames_of_reference"]["migration_binder"]["enabled"] is True


def test_system_needs_includes_runtime_pressure_attribution_frames(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "runtime_pressure_source": {"mode": "macos_system_cooldown", "dominant_bucket": "macos_system"},
            "what_do_you_need": {"items": []},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "host_pressure_attribution": {
                "dominant_bucket": "macos_system",
                "system_cotenant_hot": True,
            }
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["frames_of_reference"]["runtime_pressure_source"]["mode"] == "macos_system_cooldown"
    assert payload["frames_of_reference"]["host_pressure_attribution"]["system_cotenant_hot"] is True


def test_system_needs_falls_back_to_health_surface_repairs(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "degraded",
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "status": "needs_repair",
                    "launcher_live": False,
                    "child_fanout_ok": False,
                    "heartbeat_fresh": False,
                }
            },
        },
    )
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "required_failures": ["official_macro_context", "market_micro_context"],
            "soft_failures": [],
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gates": {"ingestion_backpressure_overload": True, "collector_contracts": True},
            "inputs": {
                "collector_required_failures": ["official_macro_context", "market_micro_context"],
                "backpressure_storage_control_override": {"active": False},
            },
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")
    blockers = {row["blocker"] for row in payload["what_do_you_need"]}

    assert payload["overall_status"] == "needs_action"
    assert payload["needs"] == payload["what_do_you_need"]
    assert "all_sleeves_launcher_fanout_needs_repair" in blockers
    assert "collector_contracts_required_failures" in blockers
    assert "ingestion_backpressure_health_gate_needs_reconciliation" in blockers
    assert payload["next_command"][1] == "start"


def test_system_needs_surfaces_ready_batch20_training_action(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "backlog_green_gate": {"status": "green"},
            "what_do_you_need": {"items": []},
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "launch_allowed": True,
                "recommended_batch_size": 20,
                "training_quality_recovery_canary": True,
                "recommended_retrain_command": [
                    "./scripts/ops/opsctl.sh",
                    "retrain-force-targeted",
                    "--include-bot-ids",
                    "brain_refinery_v10,brain_refinery_v17",
                    "--retrain-profile",
                    "coverage_batch20_canary",
                    "--skip-master-update",
                ],
                "host_training_headroom_gate": {
                    "selected_training_profile": "coverage_batch20_canary",
                    "batch20_execution_mode": "sequential_memory_guarded_waves",
                    "batch20_wave_size": 4,
                },
            },
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "ready"
    assert payload["what_do_you_need"] == []
    assert payload["ready_actions"][0]["action"] == "run_guarded_training_batch"
    assert payload["ready_actions"][0]["batch20_execution_mode"] == "sequential_memory_guarded_waves"
    assert payload["next_ready_command"][1] == "retrain-force-targeted"
    assert payload["frames_of_reference"]["training_runtime_control"]["launch_contract"]["recommended_batch_size"] == 20


def test_system_needs_turns_training_runtime_blockers_into_exact_needs(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "launch_allowed": False,
                "requested_batch_size": 20,
                "launch_blockers": ["writer_progress_stale_before_training"],
                "recommended_prep_commands": [
                    ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"]
                ],
            },
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "needs_action"
    assert payload["what_do_you_need"][0]["blocker"] == "training_runtime_writer_progress_stale_before_training"
    assert payload["what_do_you_need"][0]["command"][1] == "writer-cycle-coordinator"


def test_system_needs_manages_training_expansion_blockers_during_green_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "paper_stage": "blocked",
            "paper_armed": False,
            "paper_blocked": True,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "launch_allowed": False,
                "requested_batch_size": 8,
                "launch_blockers": [
                    "pretraining_drain_buffer_active",
                    "autonomic_training_budget_closed",
                    "training_quality_blocked",
                ],
                "recommended_prep_commands": [
                    ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
                    ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
                ],
            },
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "ready"
    assert payload["what_do_you_need"] == []
    assert {row["blocker"] for row in payload["managed_controls"]} == {
        "training_runtime_pretraining_drain_buffer_active",
        "training_runtime_autonomic_training_budget_closed",
        "training_runtime_training_quality_blocked",
    }
    assert {row["managed_control_state"] for row in payload["managed_controls"]} == {
        "training_expansion_parked_for_unattended_soak",
    }
    assert payload["frames_of_reference"]["soak_management_context"]["enabled"] is True


def test_system_needs_manages_optional_mlx_cap_during_green_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "what_do_you_need": {
                "items": [
                    {
                        "blocker": "mlx_or_gpu_lane_capped",
                        "exact_file": "governance/health/mlx_intelligence_router_latest.json",
                        "exact_shard": "",
                        "command": ["./scripts/ops/opsctl.sh", "mlx-intelligence-router", "--apply", "--json"],
                        "expected_impact": "defer optional MLX capacity",
                        "risk_level": "low",
                        "stop_when": "MLX route is ready",
                    }
                ]
            }
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "paper_stage": "blocked",
            "paper_armed": False,
            "paper_blocked": True,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "ready"
    assert payload["what_do_you_need"] == []
    assert payload["managed_controls"][0]["blocker"] == "mlx_or_gpu_lane_capped"
    assert payload["managed_controls"][0]["managed_control_state"] == "optional_mlx_capacity_deferred_during_unattended_soak"


def test_system_needs_manages_foreground_memory_headroom_during_green_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom"},
            "what_do_you_need": {
                "items": [
                    {
                        "blocker": "foreground_app_headroom_reserved",
                        "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                        "exact_shard": "classification.foreground_headroom",
                        "command": ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
                        "expected_impact": "keep foreground app headroom while paper soak remains green",
                        "risk_level": "low",
                        "stop_when": "foreground headroom is no longer reserved",
                    }
                ]
            },
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "paper_stage": "armed",
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "ready"
    assert payload["what_do_you_need"] == []
    assert payload["managed_controls"][0]["blocker"] == "foreground_app_headroom_reserved"
    assert payload["managed_controls"][0]["managed_control_state"] == "foreground_headroom_reserved_for_unattended_soak"


def test_system_needs_routes_training_quota_blocker_to_storage_quota_guard(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "launch_allowed": False,
                "requested_batch_size": 20,
                "launch_blockers": ["storage_quota_hard_breach"],
                "recommended_prep_commands": [
                    ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
                    ["./scripts/ops/opsctl.sh", "storage-quota-guard", "--json"],
                ],
                "storage_quota_training_gate": {"status": "blocked", "blocked_families": ["decisions"]},
            },
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "needs_action"
    assert payload["what_do_you_need"][0]["blocker"] == "training_runtime_storage_quota_hard_breach"
    assert payload["what_do_you_need"][0]["command"][1] == "storage-quota-guard"
    assert "hard-breached" in payload["what_do_you_need"][0]["expected_impact"]


def test_system_needs_routes_governance_quota_blocker_to_telemetry_compactor(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "launch_allowed": False,
                "requested_batch_size": 20,
                "launch_blockers": ["storage_quota_hard_breach"],
                "recommended_prep_commands": [
                    ["./scripts/ops/opsctl.sh", "storage-quota-guard", "--json"],
                ],
                "storage_quota_training_gate": {"status": "blocked", "blocked_families": ["governance_telemetry"]},
            },
        },
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")

    assert payload["overall_status"] == "needs_action"
    assert payload["what_do_you_need"][0]["command"][1] == "governance-telemetry-compactor"
    assert "governance channel telemetry" in payload["what_do_you_need"][0]["expected_impact"]


def test_system_needs_surfaces_hidden_low_grade_layers(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "profitability_grade": "A+",
            "profit_harvest_report_card": {
                "raw_outcome_grade": "B",
                "base_raw_outcome_grade": "D",
            },
            "active_profile_controls": {
                "aggressive": {"profit_grade": "F"},
            },
        },
    )
    _write_json(
        health / "system_self_intelligence_latest.json",
        {"awareness_state_vector": {"grade": "D"}},
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")
    audit = payload["frames_of_reference"]["low_grade_layer_audit"]

    assert payload["overall_status"] == "needs_action"
    assert payload["what_do_you_need"][0]["blocker"] == "low_grade_layers_still_present"
    assert audit["unique_low_grade_layer_count"] == 3
    assert audit["active_blocker_count"] == 3
    assert audit["control_posture_grade"] == "C"
    assert audit["by_category"]["base_evidence_grade"] == 1
    assert audit["by_category"]["profile_profit_grade"] == 1
    assert audit["by_category"]["self_awareness_grade"] == 1
    assert audit["layers"][0]["command"][1] in {"paper-profitability-control", "system-intelligence"}


def test_system_needs_marks_controlled_low_grades_as_a_plus_posture(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "autonomic_resource_governor_latest.json", {"what_do_you_need": {"items": []}})
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "profitability_grade": "A+",
            "low_grade_control_report_card": {"control_posture_grade": "A+", "active_blocker_count": 0},
            "low_grade_layer_summary": {"control_posture_grade": "A+", "active_blocker_count": 0},
            "profit_harvest_report_card": {
                "raw_outcome_grade": "B",
                "base_raw_outcome_grade": "D",
            },
            "remaining_low_grade_layers": [
                {"profile": "aggressive", "active_blocker": False},
            ],
            "active_profile_controls": {
                "aggressive": {"profit_grade": "F"},
            },
        },
    )
    _write_json(
        health / "system_self_intelligence_latest.json",
        {"awareness_state_vector": {"grade": "D", "control_posture_grade": "A+"}},
    )

    payload = system_needs_intelligence.build_payload(tmp_path, fix_log_path=health / "system_needs_fix_log.jsonl")
    audit = payload["frames_of_reference"]["low_grade_layer_audit"]

    assert audit["unique_low_grade_layer_count"] == 3
    assert audit["active_blocker_count"] == 0
    assert audit["control_posture_grade"] == "A+"
    assert {row["control_state"] for row in audit["layers"]} == {
        "raw_harvest_evidence_under_a_plus_control",
        "contained_by_paper_profitability_control",
        "self_awareness_under_a_plus_control",
    }
    assert not any(item.get("blocker") == "low_grade_layers_still_present" for item in payload["what_do_you_need"])
