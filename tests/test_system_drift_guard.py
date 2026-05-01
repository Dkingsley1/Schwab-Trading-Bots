from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import system_drift_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
