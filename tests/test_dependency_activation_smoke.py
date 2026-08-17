from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import dependency_activation_smoke as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_dependency_activation_smoke_selects_batch_and_reports_installed_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "config" / "library_candidate_routes_v1.json",
        {
            "candidate_libraries": [
                {"package": "cachetools", "lane": "provider_rate_limit_cache", "runtime_family": "python"},
                {"package": "ruff", "lane": "production_quality_gates", "runtime_family": "python"},
                {"package": "mlxvm", "lane": "language_reasoning", "runtime_family": "mlx"},
            ]
        },
    )
    _write_json(
        project_root / "config" / "library_activation_profiles_v1.json",
        {
            "profile_order": ["live", "ops", "research", "media"],
            "profile_lanes": {
                "live": ["provider_rate_limit_cache"],
                "ops": ["production_quality_gates"],
                "research": ["language_reasoning"],
            },
            "initial_activation_batches": {
                "production_core_safe": ["cachetools"],
                "release_gate_safe": ["ruff"],
                "mlx_off_hours": ["mlxvm"],
            },
        },
    )

    payload = src.build_payload(
        project_root,
        batch="production_core_safe",
        installed_versions={"cachetools": "6.2.0"},
    )

    assert payload["overall_status"] == "ready"
    assert payload["summary"]["selected_candidate_count"] == 1
    assert payload["summary"]["installed_candidate_count"] == 1
    assert payload["candidate_smoke_rows"][0]["activation_profiles"] == ["live"]
    assert payload["candidate_smoke_rows"][0]["status"] == "installed"


def test_dependency_activation_smoke_keeps_missing_candidates_pending_not_failed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "config" / "library_candidate_routes_v1.json",
        {
            "candidate_libraries": [
                {"package": "cachetools", "lane": "provider_rate_limit_cache", "runtime_family": "python"},
                {"package": "ruff", "lane": "production_quality_gates", "runtime_family": "python"},
            ]
        },
    )
    _write_json(
        project_root / "config" / "library_activation_profiles_v1.json",
        {
            "profile_order": ["live", "ops"],
            "profile_lanes": {
                "live": ["provider_rate_limit_cache"],
                "ops": ["production_quality_gates"],
            },
            "initial_activation_batches": {"production_core_safe": ["cachetools", "ruff"]},
        },
    )

    payload = src.build_payload(project_root, batch="production_core_safe", installed_versions={})

    assert payload["ok"] is True
    assert payload["overall_status"] == "pending_install"
    assert payload["summary"]["pending_install_count"] == 2
    assert {row["status"] for row in payload["candidate_smoke_rows"]} == {"pending_install"}
