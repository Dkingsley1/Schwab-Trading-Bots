from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import system_drift_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_guarded_paper_health_fast(health_root: Path) -> None:
    _write_json(
        health_root / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )


def test_system_drift_guard_treats_operator_gated_command_surface_as_ready(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "command_validity_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "metrics": {
                "blocked_entry_count": 0,
                "smoke_failure_count": 0,
                "runtime_smoke_failure_count": 0,
                "operator_gated_entry_count": 12,
            },
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "command_validity",
                "family": "command_surface",
                "artifact_path": artifact,
                "kind": "command_validity",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "ready"


def test_system_drift_guard_marks_stale_artifact_degraded(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "commands_hygiene_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "ready",
            "ok": True,
            "timestamp_utc": "2020-01-01T00:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "commands_hygiene",
                "family": "command_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 1,
                "repair_commands": [["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["metrics"]["stale_surface_count"] == 1
    assert payload["surfaces"][0]["stale"] is True


def test_system_drift_guard_manages_optional_stale_report_during_green_paper_soak(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "report_pdf_bundle_latest.json"
    _write_guarded_paper_health_fast(health_root)
    _write_json(
        artifact,
        {
            "overall_status": "ready",
            "ok": True,
            "timestamp_utc": "2020-01-01T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "report_pdf_bundle",
                "family": "reporting_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 120,
                "guarded_paper_stale_advisory": True,
                "repair_commands": [["./scripts/ops/opsctl.sh", "report-pdfs", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    row = payload["surfaces"][0]
    assert payload["overall_status"] == "ready"
    assert row["status"] == "ready"
    assert row["stale"] is True
    assert row["managed_stale"] is True
    assert row["recovery_deferred_reason"] == "guarded_paper_optional_report_stale"


def test_system_drift_guard_treats_written_commands_hygiene_apply_as_ready(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "commands_hygiene_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "commands_changed": True,
            "runbook_changed": False,
            "apply_results": {"commands_md_written": True, "runbook_written": False},
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "commands_hygiene",
                "family": "command_surface",
                "artifact_path": artifact,
                "kind": "commands_hygiene",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["surfaces"][0]["status"] == "ready"


def test_system_drift_guard_treats_ok_watch_surface_as_ready(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "paper_execution_truth_layer_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "watch",
            "ok": True,
            "failed_checks": [],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "paper_execution_truth_layer",
                "family": "paper_trading_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 180,
                "repair_commands": [["./scripts/ops/opsctl.sh", "paper-truth", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "ready"


def test_system_drift_guard_downgrades_pressure_deferred_blocker(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "adaptive_regression_guard_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "blocked",
            "ok": False,
            "pressure_deferred_count": 2,
            "critical_regression_count": 2,
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "adaptive_regression_guard",
                "family": "governance_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "degraded"
    assert payload["surfaces"][0]["recovery_deferred"] is True


def test_system_drift_guard_downgrades_planned_architecture_repair(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "system_architecture_autopilot_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "blocked",
            "ok": False,
            "apply": True,
            "execute_safe_repairs": False,
            "safe_repair_step_count": 4,
            "attempt_count": 0,
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "system_architecture_autopilot",
                "family": "architecture_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "safe_repairs_planned_not_executed"


def test_system_drift_guard_downgrades_guarded_architecture_recovery_debt(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "architecture_upgrade_scoreboard_latest.json"
    _write_guarded_paper_health_fast(health_root)
    _write_json(
        artifact,
        {
            "overall_status": "blocked",
            "ok": False,
            "rows": [
                {"slug": "self_healing_ops_plane", "status": "blocked"},
                {"slug": "immutable_incident_review", "status": "blocked"},
            ],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "architecture_upgrade_scoreboard",
                "family": "architecture_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 90,
                "repair_commands": [["python", "scripts/ops/architecture_upgrade_scoreboard.py", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_architecture_recovery_debt"


def test_system_drift_guard_downgrades_guarded_master_infra_recovery_debt(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "master_infrastructure_supervisor_latest.json"
    _write_guarded_paper_health_fast(health_root)
    _write_json(
        artifact,
        {
            "overall_status": "blocked",
            "ok": False,
            "checks": [
                {"name": "external_drive_route_health", "status": "blocked"},
                {"name": "governance_artifact_freshness", "status": "blocked"},
                {"name": "self_auditing_infra_bots", "status": "blocked"},
            ],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "master_infrastructure_supervisor",
                "family": "infrastructure_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "master-infra-supervisor", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "ready"
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_infrastructure_recovery_debt"


def test_system_drift_guard_marks_guarded_scoreboard_warning_debt_ready(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "architecture_upgrade_scoreboard_latest.json"
    _write_guarded_paper_health_fast(health_root)
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "ok": False,
            "rows": [
                {"slug": "self_healing_ops_plane", "status": "degraded"},
                {"slug": "immutable_incident_review", "status": "degraded"},
            ],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "architecture_upgrade_scoreboard",
                "family": "architecture_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 90,
                "repair_commands": [["python", "scripts/ops/architecture_upgrade_scoreboard.py", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "ready"
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_architecture_scoreboard_advisory_debt"


def test_system_drift_guard_keeps_managed_recovery_deferred_row_ready_when_stale(
    monkeypatch,
    tmp_path: Path,
) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "architecture_upgrade_scoreboard_latest.json"
    _write_guarded_paper_health_fast(health_root)
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "ok": False,
            "rows": [
                {"slug": "self_healing_ops_plane", "status": "degraded"},
                {"slug": "immutable_incident_review", "status": "degraded"},
            ],
            "timestamp_utc": "2020-01-01T00:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "architecture_upgrade_scoreboard",
                "family": "architecture_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 90,
                "repair_commands": [["python", "scripts/ops/architecture_upgrade_scoreboard.py", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["metrics"]["stale_surface_count"] == 1
    assert payload["surfaces"][0]["status"] == "ready"
    assert payload["surfaces"][0]["stale"] is True
    assert payload["surfaces"][0]["managed_stale"] is True
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_architecture_scoreboard_advisory_debt"


def test_system_drift_guard_marks_guarded_incident_closeout_warning_debt_ready(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    artifact = health_root / "incident_closeout_autopilot_latest.json"
    _write_guarded_paper_health_fast(health_root)
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "ok": False,
            "open_incident_count": 1,
            "bounded_incident_backlog": True,
            "bounded_closeout_path_ready": True,
            "closeout_score": 90.0,
            "recoverable_runtime_clearance": True,
            "recoverable_review_gate": True,
            "blocking_surfaces": [
                {"surface": "runtime_clearance", "severity": "warning"},
                {"surface": "incident_review", "severity": "warning"},
            ],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "incident_closeout",
                "family": "governance_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "incident-closeout", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "ready"
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_incident_closeout_advisory_debt"


def test_system_drift_guard_marks_guarded_self_reference_loop_ready(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_guarded_paper_health_fast(health_root)
    artifacts = {
        "system_architecture_contract_graph": health_root / "system_architecture_contract_graph_latest.json",
        "system_architecture_autopilot": health_root / "system_architecture_autopilot_latest.json",
        "infrastructure_autofix": health_root / "infrastructure_autofix_bot_latest.json",
        "master_infrastructure_supervisor": health_root / "master_infrastructure_supervisor_latest.json",
    }
    _write_json(
        artifacts["system_architecture_contract_graph"],
        {
            "overall_status": "degraded",
            "ok": False,
            "blocked_node_count": 0,
            "blocked_edge_count": 0,
            "degraded_nodes": ["system_drift_guard", "system_self_model"],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )
    _write_json(
        artifacts["system_architecture_autopilot"],
        {
            "overall_status": "degraded",
            "ok": False,
            "final_graph": {"blocked_node_count": 0, "blocked_edge_count": 0},
            "repair_plan": [{"node_id": "system_drift_guard"}, {"node_id": "system_self_model"}],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )
    _write_json(
        artifacts["infrastructure_autofix"],
        {
            "overall_status": "degraded",
            "ok": False,
            "repair_plan": [{"name": "master_infrastructure_supervisor"}],
            "attempts": [{"rc": 0}],
            "failed_attempt_count": 0,
            "hard_failed_attempt_count": 0,
            "operator_followups": [],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )
    _write_json(
        artifacts["master_infrastructure_supervisor"],
        {
            "overall_status": "degraded",
            "ok": False,
            "checks": [{"name": "governance_artifact_freshness", "status": "degraded"}],
            "metrics": {"blocked_check_count": 0, "degraded_check_count": 1},
            "platform_posture": {"operating_posture": "coherent"},
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": name,
                "family": "architecture_surface" if "architecture" in name else "infrastructure_surface",
                "artifact_path": path,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", name.replace("_", "-"), "--json"]],
            }
            for name, path in artifacts.items()
        ],
    )

    payload = src.build_payload(tmp_path)
    reasons = {row["name"]: row["recovery_deferred_reason"] for row in payload["surfaces"]}

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert {row["status"] for row in payload["surfaces"]} == {"ready"}
    assert reasons["system_architecture_contract_graph"] == "guarded_paper_architecture_self_reference_debt"
    assert reasons["system_architecture_autopilot"] == "guarded_paper_architecture_autopilot_self_reference_debt"
    assert reasons["infrastructure_autofix"] == "guarded_paper_infrastructure_autofix_advisory_debt"
    assert reasons["master_infrastructure_supervisor"] == "guarded_paper_infrastructure_self_reference_debt"


def test_system_drift_guard_marks_blocked_architecture_self_reference_loop_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_guarded_paper_health_fast(health_root)
    graph = health_root / "system_architecture_contract_graph_latest.json"
    autopilot = health_root / "system_architecture_autopilot_latest.json"
    _write_json(
        graph,
        {
            "overall_status": "blocked",
            "ok": False,
            "blocked_node_count": 1,
            "blocked_edge_count": 0,
            "authority_violation_count": 0,
            "blocked_nodes": ["system_drift_guard"],
            "degraded_nodes": [],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )
    _write_json(
        autopilot,
        {
            "overall_status": "blocked",
            "ok": False,
            "final_graph": {
                "blocked_node_count": 1,
                "blocked_edge_count": 0,
                "blocked_nodes": ["system_drift_guard"],
            },
            "repair_plan": [{"node_id": "system_drift_guard"}],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "system_architecture_contract_graph",
                "family": "architecture_surface",
                "artifact_path": graph,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--json"]],
            },
            {
                "name": "system_architecture_autopilot",
                "family": "architecture_surface",
                "artifact_path": autopilot,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--json"]],
            },
        ],
    )

    payload = src.build_payload(tmp_path)
    reasons = {row["name"]: row["recovery_deferred_reason"] for row in payload["surfaces"]}

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert reasons["system_architecture_contract_graph"] == "guarded_paper_architecture_self_reference_debt"
    assert reasons["system_architecture_autopilot"] == "guarded_paper_architecture_autopilot_self_reference_debt"


def test_system_drift_guard_marks_executed_architecture_autopilot_self_reference_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_guarded_paper_health_fast(health_root)
    artifact = health_root / "system_architecture_autopilot_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "ok": False,
            "execute_safe_repairs": True,
            "attempt_count": 2,
            "final_graph": {
                "blocked_node_count": 0,
                "blocked_edge_count": 0,
                "degraded_nodes": ["system_drift_guard"],
            },
            "repair_plan": [{"node_id": "adaptive_regression_guard"}, {"node_id": "system_drift_guard"}],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "system_architecture_autopilot",
                "family": "architecture_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["surfaces"][0]["status"] == "ready"
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_architecture_autopilot_self_reference_debt"


def test_system_drift_guard_marks_converged_architecture_repair_plan_self_reference_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_guarded_paper_health_fast(health_root)
    artifact = health_root / "system_architecture_autopilot_latest.json"
    _write_json(
        artifact,
        {
            "overall_status": "degraded",
            "ok": False,
            "execute_safe_repairs": True,
            "attempt_count": 3,
            "attempts": [
                {"node_id": "storage_control", "rc": 0},
                {"node_id": "system_drift_guard", "rc": 0},
                {"node_id": "system_self_model", "rc": 0},
            ],
            "final_graph": {
                "blocked_node_count": 0,
                "degraded_node_count": 2,
                "blocked_edge_count": 0,
                "authority_violation_count": 0,
                "blocked_nodes": [],
                "degraded_nodes": ["system_drift_guard", "system_self_model"],
            },
            "repair_plan": [
                {"node_id": "storage_control"},
                {"node_id": "system_drift_guard"},
                {"node_id": "system_self_model"},
            ],
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        src,
        "surface_specs",
        lambda _root: [
            {
                "name": "system_architecture_autopilot",
                "family": "architecture_surface",
                "artifact_path": artifact,
                "status_key": "overall_status",
                "ok_key": "ok",
                "max_age_minutes": 30,
                "repair_commands": [["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"]],
            }
        ],
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["degraded_surface_count"] == 0
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_architecture_autopilot_self_reference_debt"
