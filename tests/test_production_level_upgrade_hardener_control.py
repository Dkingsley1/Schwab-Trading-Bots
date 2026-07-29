from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.ops import production_level_upgrade_hardener_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cmd(name: str) -> list[str]:
    return ["./scripts/ops/opsctl.sh", name, "--json"]


def _item(control_id: str, group: str, path: str) -> dict:
    return {
        "control_id": control_id,
        "group": group,
        "title": control_id.replace("_", " ").title(),
        "commands": [_cmd(control_id)],
        "requirements": [{"path": path, "ready_statuses": ["ready"], "truthy_paths": ["ok"]}],
    }


def _base_config(project_root: Path) -> Path:
    config = {
        "schema_version": 1,
        "target_grade": "A+",
        "control_contract": {
            "live_execution_authority": False,
            "raw_profitability_truth_must_remain_visible": True,
        },
        "items": [],
    }
    for index in range(10):
        config["items"].append(_item(f"prod_{index}", "production_upgrade", f"governance/health/prod_{index}.json"))
    for index in range(10):
        config["items"].append(_item(f"hard_{index}", "hardener", f"governance/health/hard_{index}.json"))
    path = project_root / "config" / "production_level_upgrade_hardener_v1.json"
    _write_json(path, config)
    for item in config["items"]:
        _write_json(project_root / item["requirements"][0]["path"], {"overall_status": "ready", "ok": True})
    return path


def test_control_reports_a_plus_when_all_twenty_items_are_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["item_count"] == 20
    assert payload["group_counts"]["production_upgrade"]["ready"] == 10
    assert payload["group_counts"]["hardener"]["ready"] == 10
    assert payload["live_execution_authority"] is False
    assert payload["control_contract"]["live_execution_authority"] is False


def test_production_depth_catalog_has_fifty_controls_per_system_section(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)

    payload = src.build_payload(project_root, config_path=config_path)
    catalog = payload["production_depth_catalog"]

    assert catalog["grade"] == "A+"
    assert catalog["controls_per_section_target"] == 50
    assert catalog["section_count"] >= 12
    assert catalog["total_control_count"] == catalog["section_count"] * 50
    assert catalog["unique_control_count"] == catalog["total_control_count"]
    assert catalog["live_execution_authority"] is False
    assert payload["quality_checks"]["production_depth_fifty_controls_each"] is True
    assert payload["control_contract"]["production_depth_total_control_count"] == catalog["total_control_count"]
    section_ids = {section["section_id"] for section in catalog["sections"]}
    assert {
        "broker_auth_execution",
        "paper_trading_profitability",
        "storage_ingestion_backpressure",
        "infrastructure_self_healing",
    } <= section_ids
    assert all(section["control_count"] == 50 for section in catalog["sections"])
    assert all(
        control["authority_boundary"] == "safe_repair_or_read_only_no_live_execution"
        for section in catalog["sections"]
        for control in section["controls"]
    )


def test_missing_artifact_keeps_repair_command_and_blocks_a_plus(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)
    (project_root / "governance" / "health" / "prod_3.json").unlink()

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "needs_work"
    assert payload["grade"] == "A"
    assert any(blocker.startswith("prod_3:") for blocker in payload["blockers"])
    assert _cmd("prod_3") in payload["ordered_repair_commands"]


def test_raw_profitability_explainer_preserves_raw_truth(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = {
        "schema_version": 1,
        "target_grade": "A+",
        "control_contract": {"live_execution_authority": False, "raw_profitability_truth_must_remain_visible": True},
        "items": [
            {
                "control_id": "raw_profitability_explainer",
                "group": "production_upgrade",
                "title": "Raw Profitability Explainer",
                "commands": [_cmd("paper-profitability-control")],
                "custom_checks": ["raw_profitability_truth"],
                "requirements": [
                    {
                        "path": "governance/health/paper_profitability_control_latest.json",
                        "ready_statuses": ["protective_tightening"],
                        "truthy_paths": [
                            "ok",
                            "grade_transparency_contract.no_live_trade_authority",
                            "raw_d_recovery_ladder_contract.raw_grade_remains_evidence_based",
                        ],
                    }
                ],
            }
        ]
        + [_item(f"prod_{index}", "production_upgrade", f"governance/health/prod_{index}.json") for index in range(9)]
        + [_item(f"hard_{index}", "hardener", f"governance/health/hard_{index}.json") for index in range(10)],
    }
    config_path = project_root / "config" / "production_level_upgrade_hardener_v1.json"
    _write_json(config_path, config)
    _write_json(
        project_root / "governance" / "health" / "paper_profitability_control_latest.json",
        {
            "overall_status": "protective_tightening",
            "ok": True,
            "raw_profitability_grade": "D",
            "controlled_profitability_grade": "A+",
            "profitability_display_grade": "A+ controlled / D raw",
            "grade_transparency_contract": {"no_live_trade_authority": True},
            "raw_d_recovery_ladder_contract": {"raw_grade_remains_evidence_based": True},
        },
    )
    for item in config["items"][1:]:
        _write_json(project_root / item["requirements"][0]["path"], {"overall_status": "ready", "ok": True})

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["grade"] == "A+"
    assert payload["raw_profitability_truth_preserved"] is True
    raw_row = next(row for row in payload["items"] if row["control_id"] == "raw_profitability_explainer")
    assert raw_row["ready"] is True


def test_storage_soft_quota_degraded_without_hard_breach_is_managed_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["items"][0] = {
        "control_id": "storage_soft_quota_escalator",
        "group": "production_upgrade",
        "title": "Storage Soft Quota Escalator",
        "commands": [_cmd("storage-quota-guard")],
        "custom_checks": ["storage_soft_quota_escalator"],
        "requirements": [
            {
                "path": "governance/health/storage_quota_guard_latest.json",
                "ready_statuses": ["ready", "degraded"],
                "zero_count_paths": ["quota_summary.hard_breaches"],
                "required_paths": [
                    "active_hot_buffer_containment.hot_path_green",
                    "active_hot_buffer_containment.hot_lane_control_active",
                ],
            },
            {"path": "governance/health/ingestion_storage_control_latest.json", "ready_statuses": ["ready"], "truthy_paths": ["ok"]},
        ],
    }
    _write_json(config_path, config)
    _write_json(
        project_root / "governance" / "health" / "storage_quota_guard_latest.json",
        {
            "overall_status": "degraded",
            "quota_summary": {"hard_breaches": 0, "soft_breaches": 1},
            "active_hot_buffer_containment": {"hot_path_green": False, "hot_lane_control_active": True},
        },
    )
    _write_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json", {"overall_status": "ready", "ok": True})

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "ready"
    row = next(item for item in payload["items"] if item["control_id"] == "storage_soft_quota_escalator")
    assert row["ready"] is True
    assert row["custom_rows"][0]["evidence"]["hot_path_green"] is False
    assert row["custom_rows"][0]["evidence"]["managed_degraded_visible"] is True


def test_no_fake_green_guard_accepts_honest_degraded_dashboard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["items"][0] = {
        "control_id": "no_fake_green_guard",
        "group": "production_upgrade",
        "title": "No Fake Green Guard",
        "commands": [_cmd("runtime-gate-dashboard")],
        "custom_checks": ["no_fake_green_dashboard"],
        "requirements": [
            {
                "path": "governance/health/runtime_gate_dashboard_latest.json",
                "existence_only": True,
                "truthy_paths": ["overall.soak_management_context.enabled"],
                "required_paths": [
                    "overall.raw_attention",
                    "overall.forensic_attention",
                    "overall.managed_attention",
                    "overall.managed_controls",
                ],
            }
        ],
    }
    _write_json(config_path, config)
    _write_json(
        project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json",
        {
            "overall": {
                "ok": False,
                "raw_attention": ["storage_quota_guard_needs_work"],
                "forensic_attention": ["storage_quota_guard_needs_work"],
                "managed_attention": ["runtime_snapshot_cache_control_needs_work"],
                "managed_controls": [{"attention": "runtime_snapshot_cache_control_needs_work"}],
                "soak_management_context": {"enabled": True},
            }
        },
    )

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "ready"
    row = next(item for item in payload["items"] if item["control_id"] == "no_fake_green_guard")
    assert row["ready"] is True
    assert row["custom_rows"][0]["evidence"]["honest_degraded_visible"] is True


def test_safe_coordinating_infrabot_and_idle_writer_states_are_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["items"][0] = {
        "control_id": "autonomous_repair_planner",
        "group": "production_upgrade",
        "title": "Autonomous Repair Planner",
        "commands": [_cmd("production-quality")],
        "requirements": [
            {
                "path": "governance/health/infrastructure_autofix_bot_latest.json",
                "ready_statuses": ["ready", "degraded"],
                "required_paths": ["repair_plan", "post_apply_recheck"],
            },
            {
                "path": "governance/health/infrabot_adaptive_governor_latest.json",
                "ready_statuses": ["ready", "guarded", "coordinating"],
                "falsey_paths": ["safety_guard.live_execution_authority"],
                "required_paths": ["safety_guard", "capability_registry"],
            },
            {
                "path": "governance/health/production_quality_control_latest.json",
                "ready_statuses": ["ready", "coordinating", "blocked"],
                "truthy_paths": ["quality_checks.safe_apply_only"],
                "falsey_paths": ["live_execution_authority"],
            },
        ],
    }
    config["items"][1] = {
        "control_id": "single_writer_enforcement",
        "group": "production_upgrade",
        "title": "Single Writer Enforcement",
        "commands": [_cmd("writer-cycle-coordinator")],
        "requirements": [
            {
                "path": "governance/health/writer_cycle_coordinator_latest.json",
                "ready_statuses": ["ready", "writer_active", "idle"],
                "truthy_paths": ["ok"],
                "required_paths": ["writer_state_before", "writer_state_after_remediation", "summary"],
            },
            {
                "path": "governance/health/writer_process_intelligence_latest.json",
                "ready_statuses": ["ready", "advisory"],
                "required_paths": ["writer_health", "process_playbook", "safety_envelope"],
            },
        ],
    }
    _write_json(config_path, config)
    _write_json(
        project_root / "governance" / "health" / "infrastructure_autofix_bot_latest.json",
        {"overall_status": "degraded", "repair_plan": [], "post_apply_recheck": {"enabled": False}},
    )
    _write_json(
        project_root / "governance" / "health" / "infrabot_adaptive_governor_latest.json",
        {
            "overall_status": "coordinating",
            "safety_guard": {"live_execution_authority": False},
            "capability_registry": {"integration_contract": {"safe_apply_only": True}},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "production_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "live_execution_authority": False,
            "quality_checks": {"safe_apply_only": True},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "idle",
            "ok": True,
            "writer_state_before": {"active": False},
            "writer_state_after_remediation": {"active": False},
            "summary": {"actionable": 0},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {"state": "idle"},
            "process_playbook": [],
            "safety_envelope": {"single_writer_only": True},
        },
    )

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "ready"
    assert payload["items"][0]["ready"] is True
    assert payload["items"][1]["ready"] is True


def test_live_execution_double_lock_accepts_intentional_guarded_firewall(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _base_config(project_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["items"][0] = {
        "control_id": "live_execution_double_lock",
        "group": "production_upgrade",
        "title": "Live Execution Double Lock",
        "commands": [_cmd("production-readiness")],
        "custom_checks": ["live_execution_double_lock"],
        "requirements": [
            {
                "path": "governance/health/production_readiness_control_latest.json",
                "ready_statuses": ["guarded"],
                "truthy_paths": ["ok", "live_money_production_bar_ready"],
                "zero_count_paths": ["blocked_domain_count"],
            }
        ],
    }
    _write_json(config_path, config)
    _write_json(
        project_root / "governance" / "health" / "production_readiness_control_latest.json",
        {
            "overall_status": "guarded",
            "ok": True,
            "live_money_production_bar_ready": True,
            "blocked_domain_count": 0,
            "domains": [
                {
                    "name": "live_execution_risk_firewall",
                    "status": "ready_guarded",
                    "blockers": ["live_execution_not_armed", "market_data_only_active"],
                    "evidence": {"execution_armed": False, "market_data_only": True, "live_order_allowed": False},
                }
            ],
        },
    )

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "ready"
    assert payload["items"][0]["custom_rows"][0]["ready"] is True


def test_source_mutation_dynamic_check_uses_git_cleanliness(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_text(project_root / "README.md")
    _write_text(project_root / "master_bot_registry.json", "{}\n")
    subprocess.run(["git", "init"], cwd=project_root, check=True, text=True, capture_output=True)
    subprocess.run(["git", "add", "README.md", "master_bot_registry.json"], cwd=project_root, check=True, text=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )
    config_path = _base_config(project_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["items"][0] = {
        "control_id": "source_mutation_runtime_firewall",
        "group": "production_upgrade",
        "title": "Source Mutation Runtime Firewall",
        "commands": [_cmd("source-mutation-guard")],
        "dynamic_checks": ["source_mutation_guard"],
    }
    _write_json(config_path, config)
    subprocess.run(["git", "add", "config/production_level_upgrade_hardener_v1.json"], cwd=project_root, check=True, text=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "add control config"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["items"][0]["ready"] is True
    assert payload["items"][0]["dynamic_rows"][0]["name"] == "source_mutation_guard"
