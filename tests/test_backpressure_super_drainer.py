import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backpressure_super_drainer as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _ready_drainer_payload() -> dict:
    return {
        "overall_status": "ready",
        "ready_drainer_count": 1,
        "blocked_reasons": [],
        "active_drainer": {
            "name": "core_decision_drainer",
            "status": "ready",
            "live_window_safe": True,
        },
        "candidate_drainers": [
            {"name": "core_decision_drainer", "status": "ready"},
            {"name": "api_ingress_drainer", "status": "idle"},
        ],
        "next_drainer_queue": [],
    }


def test_super_drainer_previews_bounded_waves_without_parallel_writer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {"core_pending_lines": 81000, "total_pending_lines": 81000},
        },
    )

    monkeypatch.setattr(src.drainer_src, "build_payload", lambda *args, **kwargs: _ready_drainer_payload())
    monkeypatch.setattr(
        src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": True, "live_drainer_ready": True},
    )
    monkeypatch.setattr(src.coordinator_src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False})

    payload = src.build_payload(project_root, apply=False, max_waves=4, target_pending_lines=5000)

    assert payload["overall_status"] == "ready"
    assert payload["guardrails"]["single_writer_only"] is True
    assert payload["guardrails"]["starts_parallel_sql_writers"] is False
    assert payload["settings"]["planned_wave_count"] == 2
    assert payload["active_drainer"] == "core_decision_drainer"
    assert payload["self_intelligence_contract"]["included_in_system_self_model"] is True
    assert payload["drainer_strategy"]["pressure_class"] == "elevated"
    assert payload["grandmaster_context_packet"]["active_drainer"] == "core_decision_drainer"
    assert payload["drainer_intelligence_layer"]["decision_packet"]["action"] == "run_bounded_wave"
    assert payload["grandmaster_context_packet"]["intelligence_action"] == "run_bounded_wave"
    assert "backpressure_super_drainer" in payload["assigned_infrabots"]
    assert "drainer_intelligence_layer" in payload["assigned_infrabots"]


def test_super_drainer_applies_one_wave_until_target_is_cleared(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    snapshots = iter(
        [
            {"total_pending_lines": 42000, "core_pending_lines": 42000},
            {"total_pending_lines": 42000, "core_pending_lines": 42000},
            {"total_pending_lines": 4000, "core_pending_lines": 4000},
        ]
    )

    monkeypatch.setattr(src, "_storage_snapshot", lambda _project_root: next(snapshots))
    monkeypatch.setattr(src.drainer_src, "build_payload", lambda *args, **kwargs: _ready_drainer_payload())
    monkeypatch.setattr(
        src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": True, "live_drainer_ready": True},
    )
    monkeypatch.setattr(src.coordinator_src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False})

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "writer_cycle_coordinator.py" in joined:
            payload = {
                "overall_status": "applied_with_followups",
                "summary": {"partial_progress": True, "writer_merged_rows_delta": 38000},
                "writer_state_after_wait": {"merged_rows_this_cycle": 38000},
            }
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "ready"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, max_waves=3, target_pending_lines=5000)

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["waves_run"] == 1
    assert payload["summary"]["pending_lines_delta"] == 38000
    assert payload["stop_reason"] == "target_cleared"
    assert payload["waves"][0]["coordinator_step"]["status"] == "applied_with_followups"


def test_super_drainer_stops_when_wave_makes_no_progress(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    snapshots = iter(
        [
            {"total_pending_lines": 42000, "core_pending_lines": 42000},
            {"total_pending_lines": 42000, "core_pending_lines": 42000},
            {"total_pending_lines": 42000, "core_pending_lines": 42000},
        ]
    )

    monkeypatch.setattr(src, "_storage_snapshot", lambda _project_root: next(snapshots))
    monkeypatch.setattr(src.drainer_src, "build_payload", lambda *args, **kwargs: _ready_drainer_payload())
    monkeypatch.setattr(
        src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": True, "live_drainer_ready": True},
    )
    monkeypatch.setattr(src.coordinator_src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False})

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        payload = {"overall_status": "apply_failed", "summary": {"writer_merged_rows_delta": 0}}
        return {"cmd": cmd, "rc": 2, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, max_waves=3, target_pending_lines=5000)

    assert payload["overall_status"] == "apply_failed"
    assert payload["summary"]["waves_run"] == 1
    assert payload["summary"]["any_progress"] is False
    assert payload["stop_reason"] == "progress_stalled"


def test_super_drainer_writes_memory_feedback(tmp_path: Path) -> None:
    payload = {
        "timestamp_utc": "2026-05-04T12:00:00+00:00",
        "overall_status": "applied",
        "active_drainer": "core_decision_drainer",
        "target_met_final": True,
        "drainer_strategy": {"pressure_class": "elevated"},
        "summary": {
            "initial_pending_lines": 42000,
            "final_pending_lines": 4000,
            "pending_lines_delta": 38000,
            "waves_run": 1,
            "progress_waves": 1,
            "stop_reason": "target_cleared",
        },
    }
    memory_path = tmp_path / "governance" / "health" / "backpressure_super_drainer_memory_latest.json"

    memory = src.write_memory(payload, memory_path)

    assert memory_path.exists()
    assert memory["latest_event"]["active_drainer"] == "core_decision_drainer"
    assert memory["recent_progress_rate"] == 1.0
    assert memory["memory_contract"].startswith("remember_drainer_waves")
