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

    assert payload["overall_status"] == "degraded"
    assert payload["metrics"]["blocked_surface_count"] == 0
    assert payload["surfaces"][0]["recovery_deferred_reason"] == "guarded_paper_infrastructure_recovery_debt"
