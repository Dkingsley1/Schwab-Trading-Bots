from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import mlx_intelligence_router as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _package_rows() -> list[dict]:
    return [
        {"package": package, "locked_version": "1.0", "installed_version": "1.0", "status": "ok"}
        for package in src.REQUIRED_PACKAGES
    ]


def test_mlx_intelligence_router_maps_every_mlx_library_to_a_lane(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green"},
            "cotenant_awareness": {"active": True, "mode": "managed_cotenant"},
        },
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "advisory", "throttle_profile": "soft_cap", "memory_pressure_level": "normal"})
    _write_json(health / "quant_model_control_latest.json", {"overall_status": "ready"})
    _write_json(
        tmp_path / "config" / "library_candidate_routes_v1.json",
        {
            "candidate_libraries": [
                {
                    "package": "mlxvm",
                    "lane": "language_reasoning",
                    "runtime_family": "mlx",
                    "priority": "medium",
                    "reason": "pin model revisions",
                    "install_window": "maintenance",
                    "target_functions": ["mlx_model_revision_pinning"],
                },
                {
                    "package": "statsforecast",
                    "lane": "time_series_forecasting",
                    "runtime_family": "python",
                    "priority": "high",
                    "reason": "forecasting",
                    "install_window": "off_hours",
                },
            ]
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "advisory"
    assert payload["library_coverage"]["coverage_ratio"] == 1.0
    assert payload["route_coverage"]["route_coverage_ratio"] == 1.0
    assert payload["library_utilization_matrix"]["mapped_library_ratio"] == 1.0
    assert payload["control_contract"]["uses_all_available_mlx_libraries"] is True
    assert payload["runtime_caps"]["profile"] == "foreground_safe"
    assert payload["runtime_caps"]["compile_mode"] == "direct_stable"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_ROUTER_ENABLED"] == "1"
    assert payload["lane_optimization_summary"]["profile_count"] == len(payload["workload_routes"])
    assert payload["lane_optimization_summary"]["scheduler_mode"] == "bounded_direct_stable"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_MEMORY_TIER"] == "deep_green"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_SCHEDULER_MODE"] == "bounded_direct_stable"
    assert payload["adaptive_reopen_controller"]["enabled"] is True
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_HYSTERESIS_ENABLED"] == "1"
    assert payload["staged_mlx_candidate_matrix"]["candidate_package_count"] == 1
    assert payload["staged_mlx_candidate_routes"][0]["package"] == "mlxvm"
    assert payload["staged_mlx_candidate_routes"][0]["target_functions"] == ["mlx_model_revision_pinning"]
    assert payload["control_contract"]["staged_mlx_candidate_count"] == 1


def test_mlx_intelligence_router_counts_runtime_ahead_packages_as_available(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    rows = _package_rows()
    rows[0]["status"] = "runtime_ahead_of_lock"
    rows[0]["locked_version"] = "0.31.1"
    rows[0]["installed_version"] = "0.31.2"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": rows,
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green"}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "throttle_profile": "observe", "memory_pressure_level": "normal"})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["library_coverage"]["missing_count"] == 0
    assert payload["library_coverage"]["coverage_ratio"] == 1.0
    assert payload["route_coverage"]["blocked_lane_count"] == 0


def test_mlx_intelligence_router_scales_up_on_deep_green_unified_memory(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_free_pct": 92.0,
                "swap_used_gb": 0.25,
                "compressed_store_gb": 2.0,
                "compressor_gb": 0.5,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "throttle_profile": "observe", "memory_pressure_level": "normal", "host_saturation_score": 20.0},
    )

    payload = src.build_payload(tmp_path)
    caps = payload["runtime_caps"]
    tensor_route = next(row for row in payload["workload_routes"] if row["lane"] == "tensor_quant_core")

    assert payload["overall_status"] == "ready"
    assert caps["profile"] == "max_throughput"
    assert caps["mlx_memory_tier"] == "deep_green"
    assert caps["max_concurrent_mlx_jobs"] == 4
    assert caps["tensor_batch_cap"] == 96
    assert caps["embedding_batch_cap"] == 192
    assert caps["compile_mode"] == "direct_stable"
    assert payload["lane_optimization_summary"]["scheduler_mode"] == "parallel_direct_stable"
    assert payload["lane_optimization_summary"]["recommended_queue_order"][0] == "tensor_quant_core"
    assert payload["lane_optimization_summary"]["total_memory_budget_mb"] > 0
    assert payload["lane_optimization_summary"]["admission_token_budget"] == 18
    assert payload["lane_optimization_summary"]["model_cache_budget_mb"] > payload["lane_optimization_summary"]["total_memory_budget_mb"]
    assert tensor_route["optimization_profile"]["run_mode"] == "bounded_direct_stable"
    assert tensor_route["optimization_profile"]["compile_allowed"] is True
    assert tensor_route["optimization_profile"]["queue_tier"] == "hot"
    assert tensor_route["optimization_profile"]["memory_budget_mb"] > 0
    assert tensor_route["optimization_profile"]["token_cost"] == 6
    assert payload["adaptive_reopen_controller"]["reopen_stage"] == "warming_direct_stable"
    assert payload["adaptive_reopen_controller"]["stable_green_windows"] == 1
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_PREWARM_POLICY"] == "prewarm_when_idle"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_SCHEDULER_MODE"] == "parallel_direct_stable"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_REOPEN_STAGE"] == "warming_direct_stable"


def test_mlx_intelligence_router_hysteresis_promotes_after_clean_windows(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_intelligence_router_latest.json",
        {
            "adaptive_reopen_controller": {
                "reopen_stage": "warming_direct_stable",
                "stable_green_windows": 1,
                "pressure_windows": 0,
            }
        },
    )
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_free_pct": 92.0,
                "swap_used_gb": 0.25,
                "compressed_store_gb": 2.0,
                "compressor_gb": 0.5,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "throttle_profile": "observe", "memory_pressure_level": "normal", "host_saturation_score": 20.0},
    )

    payload = src.build_payload(tmp_path)
    adaptive = payload["adaptive_reopen_controller"]

    assert adaptive["stable_green_windows"] == 2
    assert adaptive["clean_windows_required"] == 2
    assert adaptive["reopen_stage"] == "parallel_direct_stable"
    assert adaptive["reopen_allowed"] is True
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_REOPEN_ALLOWED_BY_HYSTERESIS"] == "1"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_TOKEN_BUDGET"] == "18"


def test_mlx_intelligence_router_yields_to_p_core_backlog_contract(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green"}})
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "throttle_profile": "observe",
            "memory_pressure_level": "normal",
            "p_core_runtime_feedback": {
                "active": True,
                "policy": "p_core_preprocess_single_sql_writer",
                "preprocess_worker_budget": 6,
                "p_core_burst_intelligence": {
                    "mode": "burst_6",
                    "selected_workers": 6,
                    "max_budget": 6,
                    "reason": "cool host backlog burst",
                },
                "training_pcore_gate": {"small_targeted_training_allowed_now": False},
            },
        },
    )

    payload = src.build_payload(tmp_path)
    caps = payload["runtime_caps"]
    env = payload["recommended_runtime_env"]

    assert payload["overall_status"] == "advisory"
    assert caps["p_core_allocation_aware"] is True
    assert caps["p_core_contract_source"] == "runtime_throttle_control"
    assert caps["p_core_allocation_mode"] == "burst_6"
    assert caps["p_core_preprocess_workers"] == 6
    assert caps["max_concurrent_mlx_jobs"] == 1
    assert caps["tensor_batch_cap"] == 16
    assert caps["heavy_vlm_enabled"] is False
    assert caps["compile_mode"] == "off"
    assert env["MLX_INTELLIGENCE_PCORE_AWARE"] == "1"
    assert env["MLX_INTELLIGENCE_PCORE_MODE"] == "burst_6"
    assert env["MLX_INTELLIGENCE_PCORE_PREPROCESS_WORKERS"] == "6"
    assert env["MLX_INTELLIGENCE_SCHEDULER_MODE"] == "pcore_yield_single_flight"
    assert env["MLX_INTELLIGENCE_REOPEN_STAGE"] == "single_flight_pcore_hold"
    assert env["MLX_INTELLIGENCE_TOKEN_BUDGET"] == "3"
    assert payload["control_contract"]["p_core_allocation_aware"] is True
    assert payload["control_contract"]["mlx_scheduler_mode"] == "pcore_yield_single_flight"
    assert payload["control_contract"]["mlx_reopen_stage"] == "single_flight_pcore_hold"
    assert payload["lane_optimization_summary"]["allowed_lane_count"] >= 1
    assert payload["lane_optimization_summary"]["recommended_queue_order"][0] == "tensor_quant_core"


def test_mlx_intelligence_router_excludes_incompatible_optional_sidecars(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    package_rows = _package_rows()
    for row in package_rows:
        if row["package"] in {"mlx-data", "mlx-graphs", "mlx-cluster"}:
            row["status"] = "compatibility_excluded"
            row["installed_version"] = None
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": package_rows,
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green"}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "throttle_profile": "observe", "memory_pressure_level": "normal"})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["library_coverage"]["coverage_ratio"] == 1.0
    assert payload["library_coverage"]["missing_count"] == 0
    assert payload["library_coverage"]["compatibility_excluded_packages"] == [
        "mlx-data",
        "mlx-graphs",
        "mlx-cluster",
    ]
    assert payload["route_coverage"]["excluded_lanes"] == ["graph_intelligence", "data_pipeline"]
    assert payload["route_coverage"]["route_coverage_ratio"] == 1.0


def test_mlx_intelligence_router_apply_writes_capped_env(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {"compile_available": True, "compile_smoke_ok": False, "metal_available": True},
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "blocked", "memory_snapshot": {"memory_pressure_state": "red"}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "blocked", "throttle_profile": "protect_live", "memory_pressure_level": "high"})

    payload = src.build_payload(tmp_path)
    override_path = tmp_path / "config" / ".env.mlx_intelligence_router_override"
    result = src.write_outputs(
        payload,
        out_path=tmp_path / "governance" / "health" / "mlx_intelligence_router_latest.json",
        external_context_path=tmp_path / "exports" / "external_context" / "mlx_intelligence_router_latest.json",
        markdown_path=tmp_path / "exports" / "reports" / "operator" / "mlx_intelligence_router_latest.md",
        override_path=override_path,
        apply=True,
    )
    override = override_path.read_text(encoding="utf-8")

    assert payload["overall_status"] == "advisory"
    assert payload["runtime_caps"]["profile"] == "protect_live"
    assert payload["runtime_caps"]["max_concurrent_mlx_jobs"] == 1
    assert payload["runtime_caps"]["compile_mode"] == "off"
    assert result["applied"] is True
    assert "MLX_INTELLIGENCE_ROUTER_ENABLED='1'" in override
    assert "MLX_INTELLIGENCE_PROFILE='protect_live'" in override
    assert "MLX_INTELLIGENCE_MEMORY_TIER='pressure'" in override
    assert "MLX_INTELLIGENCE_SCHEDULER_MODE='protective_hold'" in override
    assert "MLX_INTELLIGENCE_QUEUE_POLICY='score_ordered_pressure_aware_lane_queue'" in override
    assert "MLX_INTELLIGENCE_REOPEN_STAGE='pressure_hold'" in override
    assert "MLX_INTELLIGENCE_TOKEN_BUDGET='0'" in override


def test_mlx_intelligence_router_micro_batches_when_unified_memory_is_guarded(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(health / "mlx_library_upgrade_latest.json", {"ok": True})
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_free_pct": 30.0,
                "swap_used_gb": 3.5,
                "compressed_store_gb": 11.0,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "throttle_profile": "observe", "memory_pressure_level": "normal", "host_saturation_score": 25.0},
    )

    payload = src.build_payload(tmp_path)
    caps = payload["runtime_caps"]
    routes = {row["lane"]: row for row in payload["workload_routes"]}

    assert payload["overall_status"] == "advisory"
    assert caps["mlx_memory_tier"] == "guarded"
    assert caps["profile"] == "sustain"
    assert caps["max_concurrent_mlx_jobs"] == 1
    assert caps["tensor_batch_cap"] == 24
    assert caps["embedding_batch_cap"] == 48
    assert caps["compile_mode"] == "off"
    assert payload["lane_optimization_summary"]["scheduler_mode"] == "micro_batch_priority"
    assert payload["lane_optimization_summary"]["total_memory_budget_mb"] > 0
    assert payload["adaptive_reopen_controller"]["reopen_stage"] == "micro_batch_watch"
    assert routes["embedding_memory"]["optimization_profile"]["allowed_now"] is True
    assert routes["embedding_memory"]["optimization_profile"]["run_mode"] == "micro_batch_only"
    assert routes["embedding_memory"]["optimization_profile"]["admission_policy"] == "micro_batch_admission_with_cooldown"
    assert routes["tensor_quant_core"]["optimization_profile"]["allowed_now"] is False
    assert payload["control_contract"]["unified_memory_tier"] == "guarded"


def test_failed_installer_artifact_does_not_block_verified_runtime(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "mlx_runtime_audit_latest.json",
        {
            "ok": True,
            "package_rows": _package_rows(),
            "runtime": {
                "compile_available": True,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
        },
    )
    _write_json(
        health / "mlx_library_upgrade_latest.json",
        {
            "ok": False,
            "install_result": {
                "ok": False,
                "stderr_tail": "ResolutionImpossible: optional mlx-graphs pin conflict",
            },
        },
    )
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green"}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "throttle_profile": "observe", "memory_pressure_level": "normal"})

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["library_coverage"]["missing_count"] == 0
    assert payload["readiness_repair_plan"]["status"] == "ready"


def test_mlx_intelligence_router_builds_readiness_repair_plan_when_audit_missing(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green"}})
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "throttle_profile": "observe",
            "memory_pressure_level": "normal",
            "p_core_runtime_feedback": {
                "active": True,
                "preprocess_worker_budget": 4,
                "p_core_burst_intelligence": {"mode": "foreground_protect", "selected_workers": 4},
            },
        },
    )

    payload = src.build_payload(tmp_path)
    repair = payload["readiness_repair_plan"]

    assert payload["overall_status"] == "blocked"
    assert repair["status"] == "audit_required"
    assert repair["pcore_safe_to_repair_now"] is False
    assert repair["recommended_commands"][0] == ["./scripts/ops/opsctl.sh", "mlx-audit", "--json"]
    assert any("MLX audit" in action for action in payload["recommended_actions"])
