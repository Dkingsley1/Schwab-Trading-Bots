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
