from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import infrabot_library_self_awareness_control as control


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_required_health(root: Path) -> None:
    health = root / "governance" / "health"
    _write_json(
        health / "library_utilization_router_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "coverage": {
                "managed_non_mlx_package_count": 192,
                "mapped_package_count": 192,
                "coverage_ratio": 1.0,
                "locked_runtime_ok_ratio": 1.0,
                "missing_runtime_count": 0,
                "version_mismatch_count": 0,
            },
            "candidate_library_matrix": {
                "candidate_package_count": 137,
                "mapped_candidate_ratio": 1.0,
                "runtime_family_counts": {"python": 125, "mlx": 12},
            },
        },
    )
    _write_json(
        health / "library_upgrade_route_control_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "upgrade_plan": {
                "mode": "route_now_plan_upgrades_without_mutating_dependencies",
                "soak_dependency_mutation_allowed": False,
                "hard_blocker_count": 0,
                "actionable_package_count": 0,
            },
            "route_matrix": {
                "route_count": 18,
                "blocked_route_count": 0,
                "degraded_route_count": 0,
            },
        },
    )
    _write_json(
        health / "mlx_intelligence_router_latest.json",
        {
            "overall_status": "advisory",
            "ok": True,
            "blocked_lane_count": 0,
            "runtime_caps": {
                "max_concurrent_mlx_jobs": 1,
                "compile_mode": "direct_stable",
                "mlx_reopen_controller": {"allowed": True, "mode": "single_light_job_yielding_to_pcore"},
            },
        },
    )
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "overall": {
                "status": "ok",
                "ok": True,
                "attention": [],
                "raw_attention": [],
                "attention_tiers": {"critical": [], "degraded": [], "watch": [], "advisory": []},
                "managed_attention": ["storage_quota_guard_needs_work"],
                "managed_controls": [
                    {
                        "attention": "storage_quota_guard_needs_work",
                        "managed_by": "unattended_soak_readiness",
                        "soak_ready": True,
                        "paper_armed": True,
                        "when_to_unmanage": "surface if storage becomes a hard blocker",
                    }
                ],
                "soak_management_context": {
                    "soak_ready": True,
                    "soak_grade": "A+",
                    "paper_armed": True,
                },
            }
        },
    )
    _write_json(health / "unattended_soak_readiness_latest.json", {"overall_status": "ready", "ok": True, "blockers": []})
    _write_json(
        health / "production_level_upgrade_hardener_control_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "grade": "A+",
            "ready_count": 20,
            "raw_profitability_truth_preserved": True,
            "live_execution_authority": False,
        },
    )
    duplicate_storage_command = ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"]
    _write_json(
        health / "infrastructure_autofix_bot_latest.json",
        {
            "overall_status": "degraded",
            "ok": False,
            "repair_plan": [
                {"name": "storage", "reason": "pending_lines", "cmd": duplicate_storage_command},
                {"name": "storage_again", "reason": "same", "cmd": duplicate_storage_command},
            ],
            "advisory_repair_plan": [
                {
                    "name": "source",
                    "reason": "source_verification",
                    "cmd": ["./scripts/ops/opsctl.sh", "source-verification-refresh", "--apply", "--json"],
                }
            ],
        },
    )
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "overall_status": "blocked",
            "ok": False,
            "repair_plan": [{"name": "repair", "cmd": duplicate_storage_command}],
        },
    )
    _write_json(
        health / "system_drift_autopilot_latest.json",
        {
            "overall_status": "blocked",
            "ok": False,
            "repair_plan": [
                {"surface": "adaptive", "reason": "blocked", "cmd": ["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"]}
            ],
        },
    )
    _write_json(health / "infrabot_adaptive_governor_latest.json", {"overall_status": "guarded"})


def test_ready_fixture_is_a_plus_and_does_not_mutate_dependencies(tmp_path: Path) -> None:
    _write_required_health(tmp_path)

    payload = control.build_payload(tmp_path, config_path=PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json")

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["control_contract"]["live_execution_authority"] is False
    assert payload["control_contract"]["dependency_mutation_allowed_during_soak"] is False
    assert payload["library_upgrade_scope"]["dependency_mutation_allowed_during_soak"] is False
    assert payload["library_upgrade_scope"]["hard_upgrade_blockers"] == 0
    assert payload["library_upgrade_scope"]["configured_candidate_additions"]
    assert any(row["status"] == "managed_soak_advisory" for row in payload["self_awareness_need_brief"])


def test_missing_library_router_is_a_hard_blocker(tmp_path: Path) -> None:
    _write_required_health(tmp_path)
    (tmp_path / "governance" / "health" / "library_utilization_router_latest.json").unlink()

    payload = control.build_payload(tmp_path, config_path=PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json")

    assert payload["ok"] is False
    assert payload["overall_status"] == "needs_work"
    assert any(str(blocker).startswith("library_utilization_router:artifact_missing") for blocker in payload["blockers"])
    hard_needs = [row for row in payload["self_awareness_need_brief"] if row["status"] == "hard_blocker"]
    assert hard_needs
    assert hard_needs[0]["authority_boundary"] == "safe_repair_or_read_only_no_live_execution_no_dependency_mutation"


def test_repair_commands_are_deduped_and_storage_is_single_writer(tmp_path: Path) -> None:
    _write_required_health(tmp_path)

    payload = control.build_payload(tmp_path, config_path=PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json")
    commands = payload["infrabot_efficiency_plan"]["commands"]
    storage_commands = [row for row in commands if row["command"][:2] == ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot"]]

    assert len(storage_commands) == 1
    assert storage_commands[0]["lane"] == "storage_writer"
    assert storage_commands[0]["max_parallel"] == 1
    assert storage_commands[0]["single_writer_or_pressure_sensitive"] is True


def test_raw_profitability_recovery_lane_is_self_awareness_owned(tmp_path: Path) -> None:
    _write_required_health(tmp_path)

    payload = control.build_payload(tmp_path, config_path=PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json")
    raw_commands = [
        row
        for row in payload["infrabot_efficiency_plan"]["commands"]
        if row["lane"] == "raw_profitability_recovery"
    ]
    command_tuples = {tuple(row["command"]) for row in raw_commands}

    assert payload["infrabot_efficiency_plan"]["lane_counts"]["raw_profitability_recovery"] >= 6
    assert ("./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json") in command_tuples
    assert ("./scripts/ops/opsctl.sh", "master-grandmaster-train", "--apply", "--json") in command_tuples
    assert ("./scripts/ops/opsctl.sh", "live-canary-readiness", "--apply", "--json") in command_tuples
    assert {row["authority_boundary"] for row in raw_commands} == {
        "safe_repair_or_read_only_no_live_execution_no_dependency_mutation"
    }


def test_library_scope_stages_candidates_but_requires_maintenance_for_installs(tmp_path: Path) -> None:
    _write_required_health(tmp_path)

    payload = control.build_payload(tmp_path, config_path=PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json")
    scope = payload["library_upgrade_scope"]

    assert scope["existing_python_packages_managed"] == 192
    assert scope["existing_python_coverage_ratio"] == 1.0
    assert scope["candidate_python_packages"] == 125
    assert scope["candidate_mlx_packages"] == 12
    assert scope["upgrade_safety_contract"]["stage_new_candidates_without_installing"] is True
    assert scope["upgrade_safety_contract"]["pip_install_or_lock_rewrite_ran"] is False
    assert scope["upgrade_safety_contract"]["maintenance_required_for_dependency_mutation"] is True
