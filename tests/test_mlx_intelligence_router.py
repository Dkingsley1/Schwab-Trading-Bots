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

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "advisory"
    assert payload["library_coverage"]["coverage_ratio"] == 1.0
    assert payload["route_coverage"]["route_coverage_ratio"] == 1.0
    assert payload["library_utilization_matrix"]["mapped_library_ratio"] == 1.0
    assert payload["control_contract"]["uses_all_available_mlx_libraries"] is True
    assert payload["runtime_caps"]["profile"] == "foreground_safe"
    assert payload["runtime_caps"]["compile_mode"] == "canary_first"
    assert payload["recommended_runtime_env"]["MLX_INTELLIGENCE_ROUTER_ENABLED"] == "1"


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
    assert payload["control_contract"]["p_core_allocation_aware"] is True


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
