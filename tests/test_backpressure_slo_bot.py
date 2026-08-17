import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backpressure_slo_bot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_backpressure_slo_bot_applies_governor_when_profile_drift_is_detected(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "estimated_core_drain_minutes": 41.2,
                "estimated_total_drain_minutes": 226.8,
            },
            "storage": {
                "retention_debt_gb": 63.765,
            },
        },
    )
    _write_json(health / "ingestion_storage_governor_latest.json", {"profile": "steady_state"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "storage_pressure": {"retention_debt_gb": 63.765, "severe_backpressure_overload": True},
            "priority_shards": [
                {
                    "shard": "explanations",
                    "latency_limit_multiplier": 1.727,
                    "storage_breached": True,
                    "latency_breached": True,
                }
            ],
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "ingestion_storage_governor.py" in joined:
            payload = {"ok": True, "profile": "critical_backpressure"}
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "blocked"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 7.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True)

    assert payload["overall_status"] == "applied"
    assert payload["recommended_profile"] == "critical_backpressure"
    assert payload["profile_drift"] is True
    assert payload["signals"]["breached_priority_shards"] == ["explanations"]
    assert payload["steps"]["ingestion_storage_governor"]["status"] == "ok"
    assert payload["summary"]["governor_profile_after_apply"] == "critical_backpressure"


def test_backpressure_slo_bot_stays_stable_when_slos_are_green(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {
                "estimated_core_drain_minutes": 4.0,
                "estimated_total_drain_minutes": 12.0,
            },
            "storage": {
                "retention_debt_gb": 0.0,
            },
        },
    )
    _write_json(health / "ingestion_storage_governor_latest.json", {"profile": "steady_state"})
    _write_json(health / "health_gates_latest.json", {"priority_shards": []})

    payload = src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "stable"
    assert payload["actionable"] is False
    assert payload["recommended_profile"] == "steady_state"


def test_backpressure_slo_bot_flags_queue_health_without_priority_latency(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.41,
            "backpressure": {
                "core_pending_lines": 6400,
                "estimated_core_drain_minutes": 8.0,
                "estimated_total_drain_minutes": 22.0,
                "stale_stage_pending_lines": 0,
            },
            "storage": {"retention_debt_gb": 0.0},
            "steady_state": {
                "quality_score": 81.5,
                "quality_label": "watch",
                "targets": {
                    "pressure_index": 0.25,
                    "core_pending_lines": 5000,
                    "estimated_total_drain_minutes": 15.0,
                    "stale_stage_pending_lines": 0,
                    "retention_debt_gb": 0.25,
                },
                "target_status": {
                    "steady_state_ready": False,
                    "target_breach_count": 3,
                    "target_breaches": [
                        "pressure_index",
                        "core_pending_lines",
                        "estimated_total_drain_minutes",
                    ],
                },
            },
        },
    )
    _write_json(health / "ingestion_storage_governor_latest.json", {"profile": "steady_state"})
    _write_json(health / "health_gates_latest.json", {"priority_shards": []})

    payload = src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "ready"
    assert payload["queue_health_actionable"] is True
    assert payload["priority_latency_actionable"] is False
    assert payload["recommended_profile"] == "elevated_backpressure"
    assert payload["signals"]["steady_state_target_breaches"] == [
        "pressure_index",
        "core_pending_lines",
        "estimated_total_drain_minutes",
    ]
    assert payload["summary"]["backpressure_quality_score"] == 81.5


def test_backpressure_slo_bot_ignores_non_storage_hard_gates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {
                "estimated_core_drain_minutes": 4.0,
                "estimated_total_drain_minutes": 12.0,
            },
            "storage": {"retention_debt_gb": 0.0},
            "steady_state": {
                "quality_score": 98.0,
                "quality_label": "excellent",
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000, "estimated_total_drain_minutes": 15.0},
                "target_status": {"steady_state_ready": True, "target_breach_count": 0, "target_breaches": []},
            },
        },
    )
    _write_json(health / "ingestion_storage_governor_latest.json", {"profile": "steady_state"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {
                "collector_contracts": True,
                "ingestion_backpressure_overload": False,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "priority_shards": [],
        },
    )

    payload = src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "stable"
    assert payload["actionable"] is False
    assert payload["recommended_profile"] == "steady_state"
