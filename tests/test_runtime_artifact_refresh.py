import json
from pathlib import Path

from scripts.ops import runtime_artifact_refresh


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_artifact_refresh_reports_recovered_and_blocked_outputs(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    ready_path = health / "ready_latest.json"
    blocked_path = health / "blocked_latest.json"
    missing_path = health / "missing_latest.json"

    specs = [
        {"name": "ready_artifact", "payload_path": ready_path, "cmd": ["ready"]},
        {"name": "blocked_artifact", "payload_path": blocked_path, "cmd": ["blocked"]},
        {"name": "missing_artifact", "payload_path": missing_path, "cmd": ["missing"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        name = spec["name"]
        if name == "ready_artifact":
            _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": True, "overall_status": "ready"})
            return {"cmd": list(spec["cmd"]), "rc": 0, "payload": {"ok": True, "overall_status": "ready"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        if name == "blocked_artifact":
            _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": False, "overall_status": "blocked"})
            return {"cmd": list(spec["cmd"]), "rc": 2, "payload": {"ok": False, "overall_status": "blocked"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        return {"cmd": list(spec["cmd"]), "rc": 1, "payload": {}, "stdout_tail": "", "stderr_tail": "boom", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "blocked"
    assert payload["artifacts_recovered_count"] == 2
    assert payload["blocked_step_count"] == 1
    assert payload["error_step_count"] == 1
    assert payload["missing_before"] == ["ready_artifact", "blocked_artifact", "missing_artifact"]
    assert payload["missing_after"] == ["missing_artifact"]


def test_runtime_artifact_refresh_is_degraded_when_outputs_exist_but_one_is_blocked(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    ready_path = health / "ready_latest.json"
    blocked_path = health / "blocked_latest.json"

    specs = [
        {"name": "ready_artifact", "payload_path": ready_path, "cmd": ["ready"]},
        {"name": "blocked_artifact", "payload_path": blocked_path, "cmd": ["blocked"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        name = spec["name"]
        if name == "ready_artifact":
            _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": True, "overall_status": "ready"})
            return {"cmd": list(spec["cmd"]), "rc": 0, "payload": {"ok": True, "overall_status": "ready"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": False, "overall_status": "blocked"})
        return {"cmd": list(spec["cmd"]), "rc": 2, "payload": {"ok": False, "overall_status": "blocked"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["ok"] is True
    assert payload["overall_status"] == "degraded"
    assert payload["missing_after"] == []
    assert payload["error_step_count"] == 0


def test_runtime_artifact_refresh_step_specs_include_training_storage_and_hardening_contracts(tmp_path: Path) -> None:
    specs = runtime_artifact_refresh._step_specs(tmp_path)
    names = [row["name"] for row in specs]

    assert "training_lineage_manifest" in names
    assert "training_quality_control" in names
    assert "portfolio_capacity_curve_report" in names
    assert "cross_host_parity_report" in names
    assert "cost_telemetry" in names
    assert "broker_readiness" in names
    assert "session_ready" in names
    assert "storage_failback_sync" in names
    assert "promotion_autopilot_packet" in names
    assert "canary_rollout_guard" in names
    assert "ingestion_storage_control" in names
    assert "storage_resilience_control" in names
    assert "security_evidence_autofix" in names
    assert "security_audit" in names
    assert "incident_closeout_autopilot" in names
    assert "live_canary_control" in names
    assert "live_readiness_smoke" in names
    assert "runtime_throttle_control" in names
    assert "chrome_headless_guard" in names
    assert "multiple_testing_guard" in names

    chrome_spec = next(row for row in specs if row["name"] == "chrome_headless_guard")
    assert "--apply" in chrome_spec["cmd"]
