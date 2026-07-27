from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import library_utilization_router as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_library_utilization_router_maps_all_non_mlx_packages_and_keeps_mlx_default(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "\n".join(
            [
                "mlx==0.31.1",
                "pandas==3.0.1",
                "duckdb==1.5.0",
                "QuantLib==1.42.1",
                "torch==2.10.0",
                "aiohttp==3.13.3",
                "schwab-py==1.5.1",
                "matplotlib==3.10.8",
                "pytest==9.0.2",
            ]
        ),
        encoding="utf-8",
    )
    health = tmp_path / "governance" / "health"
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "advisory", "throttle_profile": "soft_cap", "memory_pressure_level": "normal"})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "advisory", "runtime_caps": {"max_concurrent_mlx_jobs": 2}})
    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={
            "mlx": "0.31.1",
            "pandas": "3.0.1",
            "duckdb": "1.5.0",
            "quantlib": "1.42.1",
            "torch": "2.10.0",
            "aiohttp": "3.13.3",
            "schwab-py": "1.5.1",
            "matplotlib": "3.10.8",
            "pytest": "9.0.2",
        },
    )

    lanes = payload["library_utilization_matrix"]["package_to_lane"]

    assert payload["overall_status"] == "advisory"
    assert payload["coverage"]["coverage_ratio"] == 1.0
    assert payload["coverage"]["locked_non_mlx_package_count"] == 8
    assert "mlx" not in lanes
    assert lanes["pandas"] == "dataframe_feature_engine"
    assert lanes["duckdb"] == "storage_sql"
    assert lanes["quantlib"] == "quant_derivatives_risk"
    assert lanes["torch"] == "portable_ml_replay"
    assert payload["control_contract"]["uses_all_managed_non_mlx_libraries"] is True
    assert payload["control_contract"]["default_ml_backend"] == "mlx"
    assert payload["recommended_runtime_env"]["PRIMARY_ML_RUNTIME_BACKEND"] == "mlx"
    assert payload["recommended_runtime_env"]["PORTABLE_MODEL_REPLAY_POLICY"] == "canary_or_off_hours_only"


def test_library_utilization_router_apply_writes_env_caps(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("pandas==3.0.1\nonnxruntime==1.24.3\n", encoding="utf-8")
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json", {"overall_status": "blocked", "throttle_profile": "protect_live", "memory_pressure_level": "high"})
    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={"pandas": "3.0.1", "onnxruntime": "1.24.3"},
    )
    override_path = tmp_path / "config" / ".env.library_utilization_router_override"
    result = src.write_outputs(
        payload,
        out_path=tmp_path / "governance" / "health" / "library_utilization_router_latest.json",
        external_context_path=tmp_path / "exports" / "external_context" / "library_utilization_router_latest.json",
        markdown_path=tmp_path / "exports" / "reports" / "operator" / "library_utilization_router_latest.md",
        override_path=override_path,
        apply=True,
    )
    override = override_path.read_text(encoding="utf-8")

    assert payload["runtime_caps"]["profile"] == "protect_live"
    assert payload["runtime_caps"]["max_portable_model_replay_jobs"] == 0
    assert result["applied"] is True
    assert "LIBRARY_UTILIZATION_ROUTER_ENABLED='1'" in override
    assert "PRIMARY_ML_RUNTIME_BACKEND='mlx'" in override


def test_library_utilization_router_routes_runtime_ahead_and_optional_fallback_as_advisory(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "\n".join(
            [
                "pandas==3.0.1",
                "pandas-ta==0.4.71b0",
                "ta==0.11.0",
                "requests==2.32.5",
            ]
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={
            "pandas": "3.0.3",
            "ta": "0.11.0",
            "requests": "2.34.2",
        },
    )
    coverage = payload["coverage"]
    rows = {row["package"]: row for row in payload["package_inventory"]}

    assert payload["overall_status"] == "advisory"
    assert payload["ok"] is True
    assert coverage["missing_runtime_count"] == 0
    assert coverage["runtime_ahead_of_lock_count"] == 2
    assert coverage["optional_fallback_active_count"] == 1
    assert rows["pandas"]["status"] == "runtime_ahead_of_lock"
    assert rows["pandas-ta"]["status"] == "optional_fallback_active"
    assert rows["pandas-ta"]["available_fallback_packages"] == ["ta", "pandas", "numpy"][:2]
    assert "adopt newer runtime packages into the lock after canary evidence instead of marking active routes failed" in payload["recommended_actions"]
