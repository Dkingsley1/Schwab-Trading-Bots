from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import library_upgrade_route_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_library_upgrade_route_control_routes_now_and_plans_reconciliation(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "\n".join(
            [
                "mlx==0.31.1",
                "mlx-lm==0.31.1",
                "pandas==3.0.1",
                "pandas-ta==0.4.71b0",
                "ta==0.11.0",
                "setuptools==82.0.0",
            ]
        ),
        encoding="utf-8",
    )
    health = tmp_path / "governance" / "health"
    _write_json(health / "runtime_throttle_control_latest.json", {"throttle_profile": "soft_cap", "memory_pressure_level": "normal"})
    _write_json(health / "mlx_intelligence_router_latest.json", {"runtime_caps": {"max_concurrent_mlx_jobs": 2}})

    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        python_bin=Path("/venv/bin/python"),
        installed_versions={
            "mlx": "0.31.1",
            "mlx-lm": "0.31.1",
            "pandas": "3.0.3",
            "ta": "0.11.0",
            "setuptools": "81.0.0",
        },
    )
    plan = payload["upgrade_plan"]
    actions = {row["package"]: row for row in plan["actions"]}

    assert payload["overall_status"] == "advisory"
    assert payload["ok"] is True
    assert payload["route_matrix"]["mapped_package_ratio"] == 1.0
    assert plan["hard_blocker_count"] == 0
    assert actions["pandas"]["action"] == "adopt_runtime_version_into_lock_after_canary_evidence"
    assert actions["pandas-ta"]["action"] == "keep_fallback_route_active_and_install_pinned_optional_package_later"
    assert actions["pandas-ta"]["command"] == ["/venv/bin/python", "-m", "pip", "install", "-U", "pandas-ta==0.4.71b0"]
    assert actions["setuptools"]["action"] == "upgrade_runtime_to_locked_version_in_maintenance_window"
    assert payload["recommended_runtime_env"]["LIBRARY_UPGRADE_ROUTE_DEPENDENCY_MUTATION_ALLOWED"] == "0"


def test_library_upgrade_route_control_blocks_missing_required_runtime_package(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("requests==2.32.5\nmlx==0.31.1\n", encoding="utf-8")

    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        python_bin=Path("/venv/bin/python"),
        installed_versions={"mlx": "0.31.1"},
    )
    plan = payload["upgrade_plan"]

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert plan["hard_blocker_count"] == 1
    assert plan["hard_blockers"][0]["package"] == "requests"


def test_library_upgrade_route_control_defers_known_optional_mlx_satellites(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "\n".join(
            [
                "mlx==0.31.1",
                "mlx-cluster==0.0.7",
                "mlx-data==0.2.0",
                "mlx-graphs==0.0.9",
            ]
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        python_bin=Path("/venv/bin/python"),
        installed_versions={"mlx": "0.31.1"},
    )
    actions = {row["package"]: row for row in payload["upgrade_plan"]["actions"]}

    assert payload["overall_status"] == "advisory"
    assert payload["upgrade_plan"]["hard_blocker_count"] == 0
    assert actions["mlx-cluster"]["status"] == "compatibility_excluded_optional"
    assert actions["mlx-cluster"]["action"] == "keep_optional_mlx_satellite_deferred_until_compatible_distribution_exists"


def test_library_upgrade_route_control_apply_writes_env_override(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("mlx==0.31.1\npandas==3.0.1\n", encoding="utf-8")
    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        python_bin=Path("/venv/bin/python"),
        installed_versions={"mlx": "0.31.1", "pandas": "3.0.1"},
    )
    override_path = tmp_path / "config" / ".env.library_upgrade_route_control_override"
    result = src.write_outputs(
        payload,
        out_path=tmp_path / "governance" / "health" / "library_upgrade_route_control_latest.json",
        external_context_path=tmp_path / "exports" / "external_context" / "library_upgrade_route_control_latest.json",
        markdown_path=tmp_path / "exports" / "reports" / "operator" / "library_upgrade_route_control_latest.md",
        override_path=override_path,
        apply=True,
    )
    override = override_path.read_text(encoding="utf-8")

    assert result["applied"] is True
    assert result["dependency_mutation_ran"] is False
    assert "LIBRARY_UPGRADE_ROUTE_CONTROL_ENABLED='1'" in override
    assert "LIBRARY_UPGRADE_ROUTE_DEPENDENCY_MUTATION_ALLOWED='0'" in override
