from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import production_soak_enhancement as src
from tests.test_production_readiness_control import _seed_minimal_project


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_soak_config(project_root: Path) -> Path:
    control_rows = [
        ("sandbox_dependency_activation", 1, "dependency_activation_smoke_runner"),
        ("live_submit_firewall", 2, "live_execution_risk_firewall"),
        ("deterministic_replay_baseline", 3, "deterministic_replay_harness"),
        ("production_release_command", 4, "release_gates"),
        ("telemetry_redaction_canary", 5, "observability_redaction"),
        ("safe_profile_promotion_plan", 6, "dependency_activation_smoke_runner"),
        ("rollback_drill", 7, "incident_and_rollback_system"),
        ("cockpit_operator_surface", 8, "operator_cockpit"),
    ]
    config_path = project_root / "config" / "production_soak_enhancement_v1.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "default_dependency_batch": "production_core_safe",
            "artifact_paths": {
                "replay_baseline": "governance/health/production_readiness_replay_baseline.json",
                "telemetry_canary": "governance/health/telemetry_redaction_canary_latest.json",
                "profile_promotion_plan": "governance/health/safe_profile_promotion_plan_latest.json",
                "rollback_drill": "governance/rollback/production_rollback_drill_latest.json",
            },
            "release_command": (
                "./scripts/ops/opsctl.sh production-soak-enhancement --apply --json --exit-zero "
                "&& ./scripts/ops/opsctl.sh production-readiness --apply --json --exit-zero "
                "&& ./.venv314/bin/python -m pytest -q tests/test_dependency_activation_smoke.py "
                "tests/test_production_readiness_control.py tests/test_production_soak_enhancement.py "
                "tests/test_library_utilization_router.py tests/test_mlx_intelligence_router.py"
            ),
            "soak_enhancements": [
                {
                    "id": item_id,
                    "control_number": control_number,
                    "source_domain": source_domain,
                    "soak_enhancement": f"{item_id} control evidence",
                }
                for item_id, control_number, source_domain in control_rows
            ],
        },
    )
    return config_path


def test_production_soak_enhancement_materializes_eight_guarded_controls_without_live_mutation(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _seed_minimal_project(project_root)
    config_path = _seed_soak_config(project_root)
    cockpit = project_root / "scripts" / "ops" / "operator_cockpit.py"
    cockpit.write_text(
        "production_readiness_control_latest.json\nproduction_soak_enhancement_latest.json\n",
        encoding="utf-8",
    )

    payload = src.build_payload(project_root, config_path=config_path, dependency_batch="production_core_safe")
    controls = {row["id"]: row for row in payload["controls"]}

    assert Path(__file__).name == "test_production_soak_enhancement.py"
    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded"
    assert payload["control_count"] == 8
    assert payload["blocked_control_count"] == 0
    assert payload["production_readiness"]["live_runtime_promotion_allowed"] is False
    assert payload["control_contract"]["live_runtime_mutated"] is False
    assert payload["control_contract"]["soak_can_run_with_live_orders_disabled"] is True
    assert controls["live_submit_firewall"]["evidence"]["live_order_allowed"] is False
    assert controls["production_release_command"]["evidence"]["command"].find("tests/test_production_soak_enhancement.py") >= 0
    assert controls["cockpit_operator_surface"]["status"] == "ready"


def test_production_soak_enhancement_apply_writes_auxiliary_dry_run_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_minimal_project(project_root)
    config_path = _seed_soak_config(project_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = src.build_payload(project_root, config_path=config_path, dependency_batch="production_core_safe")

    result = src.write_outputs(
        payload,
        project_root=project_root,
        config=config,
        out_path=project_root / "governance" / "health" / "production_soak_enhancement_latest.json",
        markdown_path=project_root / "exports" / "reports" / "operator" / "production_soak_enhancement_latest.md",
        apply=True,
    )
    written = result["auxiliary_artifacts_written"]

    assert set(written) == {"replay_baseline", "telemetry_canary", "profile_promotion_plan", "rollback_drill"}
    assert all(Path(path).exists() for path in written.values())
    rollback = json.loads(Path(written["rollback_drill"]).read_text(encoding="utf-8"))
    profile_plan = json.loads(Path(written["profile_promotion_plan"]).read_text(encoding="utf-8"))
    assert rollback["dry_run_only"] is True
    assert profile_plan["live_runtime_mutated"] is False


def test_apply_keeps_mutable_replay_baseline_out_of_tracked_config(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    readiness_config = _seed_minimal_project(project_root)
    config_path = _seed_soak_config(project_root)
    unsafe_baseline = project_root / "config" / "production_readiness_replay_baseline.json"
    unsafe_baseline.write_text('{"sentinel": true}\n', encoding="utf-8")

    readiness = json.loads(readiness_config.read_text(encoding="utf-8"))
    readiness["deterministic_replay"]["baseline_path"] = str(unsafe_baseline)
    _write_json(readiness_config, readiness)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = src.build_payload(project_root, config_path=config_path, dependency_batch="production_core_safe")
    result = src.write_outputs(
        payload,
        project_root=project_root,
        config=config,
        out_path=project_root / "governance" / "health" / "production_soak_enhancement_latest.json",
        markdown_path=project_root / "exports" / "reports" / "operator" / "production_soak_enhancement_latest.md",
        apply=True,
    )

    written_baseline = Path(result["auxiliary_artifacts_written"]["replay_baseline"])
    assert written_baseline == project_root / "governance" / "health" / "production_readiness_replay_baseline.json"
    assert json.loads(unsafe_baseline.read_text(encoding="utf-8")) == {"sentinel": True}
