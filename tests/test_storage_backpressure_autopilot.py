from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import storage_backpressure_autopilot as autopilot_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_storage_backpressure_autopilot_builds_coordinated_plan(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 75000,
                "estimated_total_drain_minutes": 95.0,
            },
            "storage": {
                "retention_debt_gb": 5.8,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 2,
            },
            "data_integrity": {"sql_invalid_lines": 1},
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "critical_backpressure",
            "recommended_actions": ["apply the governor"],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": ["run the drain window"],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "focus": {
                "focus_shards": ["explanations"],
                "targeted_retention_debt_gb": 3.1,
                "severe_focus": True,
            },
            "recommended_actions": ["drain explanation debt"],
        },
    )

    payload = autopilot_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert payload["overall_status"] == "ready"
    assert names == [
        "backpressure_slo_bot",
        "writer_cycle_coordinator",
        "retention_debt_sheriff",
    ]
    coordinator_cmd = next(row["cmd"] for row in payload["repair_plan"] if row["name"] == "writer_cycle_coordinator")
    sheriff_cmd = next(row["cmd"] for row in payload["repair_plan"] if row["name"] == "retention_debt_sheriff")
    assert "--maintenance-force" in coordinator_cmd
    assert "--force" in sheriff_cmd
    assert payload["metrics"]["repair_step_count"] == 3
    assert payload["previews"]["retention_debt_sheriff"]["severe_focus"] is True


def test_storage_backpressure_autopilot_fast_paths_completed_writer_handoff(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 18000,
                "estimated_total_drain_minutes": 65.0,
            },
            "storage": {"retention_debt_gb": 0.0},
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": False,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "handoff_needed",
            "actionable": True,
            "drain_ready": False,
            "maintenance_ready": False,
            "writer_state_after_wait": {
                "active": True,
                "active_source": "completed_lock_handoff_needed",
                "complete_lock_handoff_needed": True,
                "current_step": "complete",
            },
            "summary": {"completed_writer_lock_handoff_needed": True},
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(project_root, apply=False)
    coordinator = next(row for row in payload["repair_plan"] if row["name"] == "writer_cycle_coordinator")

    assert "--handoff-only" in coordinator["cmd"]
    assert "--poll-seconds" not in coordinator["cmd"]
    assert "completed_writer_lock_handoff_pending" in coordinator["reason"]
    assert coordinator["timeout_sec"] == 90


def test_attempt_record_accepts_completed_writer_handoff_release() -> None:
    record = autopilot_src._attempt_record(
        {
            "payload": {
                "bot": "writer_cycle_coordinator",
                "overall_status": "handoff_released",
            },
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    )

    assert record["status"] == "ok"
    assert record["overall_status"] == "handoff_released"


def test_storage_backpressure_autopilot_quick_bounded_mode_limits_drainer_ttl(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.334,
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 5156,
                "estimated_total_drain_minutes": 12.0,
            },
            "storage": {"retention_debt_gb": 0.0, "backlog_drain_recommended_now": False},
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "active_drainer": {"name": "core_decision_drainer"},
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": False,
            "drain_ready": False,
            "maintenance_ready": False,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        poll_seconds=5.0,
        wait_timeout_seconds=75.0,
        command_timeout_seconds=180,
        backpressure_command_timeout_seconds=120,
        max_cycles=1,
    )

    drainer = next(row for row in payload["repair_plan"] if row["name"] == "backpressure_drainer_fleet")
    ttl_index = drainer["cmd"].index("--ttl-seconds") + 1
    assert drainer["cmd"][ttl_index] == "75"
    assert drainer["timeout_sec"] == 120
    assert payload["timing_contract"]["mode"] == "quick_bounded"
    assert payload["metrics"]["quick_bounded_mode"] is True


def test_storage_backpressure_autopilot_quick_bounded_caps_uniform_and_lane_timeouts(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    bot_logs_root = tmp_path / "bot_logs"
    bot_logs_root.mkdir(parents=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 1.25,
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 76000,
                "estimated_total_drain_minutes": 70.0,
                "raw_live": {
                    "total_pending_lines": 76000,
                    "core_pending_lines": 36000,
                    "oldest_pending_age_seconds": 1200.0,
                },
            },
            "storage": {
                "retention_debt_gb": 0.0,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "scan_roots": [{"path": str(bot_logs_root)}],
            "raw_summary": {
                "compression_candidate_count": 3,
                "compression_candidate_gb": 12.0,
            },
        },
    )

    monkeypatch.setattr(
        autopilot_src,
        "_uniform_process_refresh",
        lambda *args, **kwargs: {
            "enabled": True,
            "reason": "test_uniform_wants_longer_window",
            "env_overrides": {
                "BACKLOG_DRAIN_UNIFORM_WRITER_POLL_SECONDS": "6",
                "BACKLOG_DRAIN_UNIFORM_WAIT_TIMEOUT_SECONDS": "240",
                "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": "3",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "900",
            },
            "payload": {},
            "write_result": {},
        },
    )
    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": True, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        poll_seconds=5.0,
        wait_timeout_seconds=75.0,
        command_timeout_seconds=180,
        backpressure_command_timeout_seconds=120,
        max_cycles=1,
    )

    assert payload["timing_contract"]["mode"] == "quick_bounded"
    assert payload["timing_contract"]["wait_timeout_seconds"] == 75
    assert payload["timing_contract"]["command_timeout_seconds"] == 180
    assert payload["timing_contract"]["max_cycles"] == 1
    timeouts = {row["name"]: row["timeout_sec"] for row in payload["repair_plan"]}
    assert timeouts["backpressure_slo_bot"] == 180
    assert timeouts["writer_cycle_coordinator"] == 180
    assert timeouts["raw_training_manifest_refresh"] == 180


def test_storage_backpressure_autopilot_applies_only_needed_lanes(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    storage_path = project_root / "governance" / "health" / "ingestion_storage_control_latest.json"
    _write_json(
        storage_path,
        {
            "overall_status": "needs_work",
            "severity": "high",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 42000,
                "estimated_total_drain_minutes": 70.0,
            },
            "storage": {
                "retention_debt_gb": 0.2,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "elevated_backpressure",
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {
                "focus_shards": [],
                "targeted_retention_debt_gb": 0.0,
                "severe_focus": False,
            },
            "recommended_actions": [],
        },
    )

    seen: list[str] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        script = Path(cmd[1]).name
        seen.append(script)
        if script == "writer_cycle_coordinator.py":
            _write_json(
                storage_path,
                {
                    "overall_status": "ready",
                    "severity": "elevated",
                    "backpressure": {
                        "pending_lines_threshold": 15000,
                        "total_pending_lines": 18000,
                        "estimated_total_drain_minutes": 12.0,
                    },
                    "storage": {
                        "retention_debt_gb": 0.0,
                        "backlog_drain_recommended_now": False,
                        "backlog_quarantine_candidate_files": 0,
                    },
                    "data_integrity": {"sql_invalid_lines": 0},
                },
            )
        return {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": {
                "bot": script.replace(".py", ""),
                "overall_status": "applied",
                "ok": True,
            },
        }

    monkeypatch.setattr(autopilot_src, "_run_json", _fake_run_json)

    payload = autopilot_src.build_payload(
        project_root,
        apply=True,
        poll_seconds=5.0,
        wait_timeout_seconds=30.0,
        command_timeout_seconds=60,
        backpressure_command_timeout_seconds=20,
    )

    assert payload["overall_status"] == "applied"
    assert payload["metrics"]["attempted_step_count"] == 2
    assert seen == [
        "backpressure_slo_bot.py",
        "writer_cycle_coordinator.py",
    ]
    assert payload["metrics"]["cycle_count"] == 1


def test_storage_backpressure_autopilot_repeats_cycles_until_targets_clear(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    storage_path = health / "ingestion_storage_control_latest.json"

    def _write_storage(total_pending_lines: int, retention_debt_gb: float, *, overall_status: str, severity: str) -> None:
        _write_json(
            storage_path,
            {
                "overall_status": overall_status,
                "severity": severity,
                "backpressure": {
                    "pending_lines_threshold": 15000,
                    "total_pending_lines": total_pending_lines,
                    "estimated_total_drain_minutes": 95.0,
                },
                "storage": {
                    "retention_debt_gb": retention_debt_gb,
                    "backlog_drain_recommended_now": total_pending_lines > 0,
                    "backlog_quarantine_candidate_files": 0,
                },
                "data_integrity": {"sql_invalid_lines": 0},
            },
        )

    _write_storage(76000, 6.4, overall_status="blocked", severity="critical")

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "critical_backpressure",
            "recommended_actions": ["apply the governor"],
        },
    )

    def _coordinator_preview(*_args, **_kwargs) -> dict:
        current = json.loads(storage_path.read_text(encoding="utf-8"))
        total_pending_lines = current["backpressure"]["total_pending_lines"]
        return {
            "overall_status": "ready",
            "actionable": total_pending_lines > 0,
            "drain_ready": total_pending_lines > 0,
            "maintenance_ready": True,
            "recommended_actions": ["run the drain window"],
        }

    monkeypatch.setattr(autopilot_src.coordinator_src, "build_payload", _coordinator_preview)

    def _sheriff_preview(*_args, **_kwargs) -> dict:
        current = json.loads(storage_path.read_text(encoding="utf-8"))
        retention_debt_gb = float(current["storage"]["retention_debt_gb"])
        return {
            "overall_status": "ready" if retention_debt_gb > 0.0 else "idle",
            "actionable": retention_debt_gb > 0.0,
            "focus": {
                "focus_shards": ["explanations"] if retention_debt_gb > 0.0 else [],
                "targeted_retention_debt_gb": retention_debt_gb,
                "severe_focus": retention_debt_gb >= 5.0,
            },
            "recommended_actions": ["drain explanation debt"] if retention_debt_gb > 0.0 else [],
        }

    monkeypatch.setattr(autopilot_src.sheriff_src, "build_payload", _sheriff_preview)

    seen: list[list[str]] = []
    cycle_counter = {"count": 0}

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        seen.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "backpressure_slo_bot.py":
            payload = {"bot": "backpressure_slo_bot", "overall_status": "applied", "ok": True}
        elif script == "writer_cycle_coordinator.py":
            cycle_counter["count"] += 1
            if cycle_counter["count"] == 1:
                _write_storage(26000, 1.2, overall_status="needs_work", severity="high")
            else:
                _write_storage(15000, 0.1, overall_status="ready", severity="elevated")
            payload = {"bot": "writer_cycle_coordinator", "overall_status": "applied", "ok": True}
        elif script == "retention_debt_sheriff.py":
            payload = {"bot": "retention_debt_sheriff", "overall_status": "applied", "ok": True}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": payload,
        }

    monkeypatch.setattr(autopilot_src, "_run_json", _fake_run_json)

    payload = autopilot_src.build_payload(
        project_root,
        apply=True,
        poll_seconds=5.0,
        wait_timeout_seconds=30.0,
        command_timeout_seconds=60,
        backpressure_command_timeout_seconds=20,
        max_cycles=3,
        target_pending_lines=20000,
        target_retention_debt_gb=0.25,
    )

    assert payload["overall_status"] == "applied"
    assert payload["clearance_state"]["cleared"] is True
    assert payload["metrics"]["cycle_count"] == 2
    assert payload["metrics"]["attempted_step_count"] == 6
    assert payload["cycle_records"][0]["progress"]["progress_observed"] is True
    assert payload["cycle_records"][1]["clearance_after"]["cleared"] is True
    writer_cmds = [cmd for cmd in seen if Path(cmd[1]).name == "writer_cycle_coordinator.py"]
    assert len(writer_cmds) == 2
    assert "--maintenance-force" in writer_cmds[0]
    sheriff_cmds = [cmd for cmd in seen if Path(cmd[1]).name == "retention_debt_sheriff.py"]
    assert sheriff_cmds
    assert "--force" in sheriff_cmds[0]


def test_storage_backpressure_autopilot_forces_writer_focus_when_core_backlog_is_concentrated(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "severity": "critical",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 760000,
                "estimated_total_drain_minutes": 120.0,
            },
            "storage": {
                "retention_debt_gb": 0.1,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 760000,
            "pending_lines_total": 760000,
            "top_pending_files": [
                {"source_rel": "governance/execution_lanes/execution_results_20260422.jsonl", "pending_lines": 310000},
                {"source_rel": "governance/execution_lanes/execution_promotions_20260422.jsonl", "pending_lines": 280000},
                {"source_rel": "governance/execution_lanes/execution_intents_20260422.jsonl", "pending_lines": 90000},
            ],
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "critical_backpressure",
            "recommended_actions": ["apply the governor"],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": ["run the drain window"],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {
                "focus_shards": [],
                "targeted_retention_debt_gb": 0.0,
                "severe_focus": False,
            },
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(project_root, apply=False)

    assert payload["core_focus"]["concentrated_core_backlog"] is True
    coordinator_cmd = next(row["cmd"] for row in payload["repair_plan"] if row["name"] == "writer_cycle_coordinator")
    assert "--maintenance-force" in coordinator_cmd
    assert payload["metrics"]["core_focus_concentrated"] is True


def test_storage_backpressure_autopilot_includes_drainer_fleet_for_focused_backlog(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 64000,
                "estimated_total_drain_minutes": 120.0,
            },
            "storage": {
                "retention_debt_gb": 0.0,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 64000,
            "pending_lines_total": 64000,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 32000,
                    "oldest_pending_age_seconds": 1800.0,
                },
                {
                    "source_rel": "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 30000,
                    "oldest_pending_age_seconds": 1800.0,
                },
            ],
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "critical_backpressure",
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {
                "focus_shards": [],
                "targeted_retention_debt_gb": 0.0,
                "severe_focus": False,
            },
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "backpressure_drainer_fleet" in names
    assert names.index("backpressure_drainer_fleet") < names.index("writer_cycle_coordinator")
    assert payload["previews"]["backpressure_drainer_fleet"]["active_drainer"] == "core_decision_drainer"
    assert payload["metrics"]["drainer_ready_count"] >= 1


def test_storage_backpressure_autopilot_adds_bounded_raw_training_compaction_lane(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    raw_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    raw_root.mkdir(parents=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.12,
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 2200,
                "estimated_total_drain_minutes": 5.0,
            },
            "storage": {
                "retention_debt_gb": 0.0,
                "backlog_drain_recommended_now": False,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "needs_work",
            "scan_roots": [{"path": str(raw_root), "exists": True, "protected": False}],
            "raw_summary": {
                "compression_candidate_count": 30,
                "compression_candidate_gb": 48.0,
            },
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": False,
            "drain_ready": False,
            "maintenance_ready": False,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        raw_training_max_files=5,
        raw_training_max_gb=8.0,
        raw_training_min_candidate_gb=8.0,
        raw_training_bot_logs_min_free_gb=0.1,
        raw_training_local_min_free_gb=0.1,
    )

    names = [row["name"] for row in payload["repair_plan"]]
    assert names == ["raw_training_compaction"]
    raw_cmd = payload["repair_plan"][0]["cmd"]
    assert "raw_training_compaction_intelligence.py" in raw_cmd[1]
    assert "--max-files" in raw_cmd
    assert "--max-gb" in raw_cmd
    assert "--bot-logs-root" in raw_cmd
    assert str(raw_root) in raw_cmd
    assert payload["previews"]["raw_training_compaction"]["actionable"] is True
    assert payload["metrics"]["raw_training_actionable"] is True


def test_storage_backpressure_autopilot_blocks_raw_compaction_when_pressure_hot(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    raw_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    raw_root.mkdir(parents=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.95,
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 2200,
                "estimated_total_drain_minutes": 5.0,
            },
            "storage": {
                "retention_debt_gb": 0.0,
                "backlog_drain_recommended_now": False,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "needs_work",
            "scan_roots": [{"path": str(raw_root), "exists": True, "protected": False}],
            "raw_summary": {
                "compression_candidate_count": 30,
                "compression_candidate_gb": 48.0,
            },
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": False,
            "drain_ready": False,
            "maintenance_ready": False,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        raw_training_pressure_ceiling=0.6,
        raw_training_bot_logs_min_free_gb=0.1,
        raw_training_local_min_free_gb=0.1,
    )

    names = [row["name"] for row in payload["repair_plan"]]
    assert names == ["raw_training_manifest_refresh"]
    assert payload["previews"]["raw_training_compaction"]["actionable"] is False
    assert payload["previews"]["raw_training_compaction"]["manifest_refresh_actionable"] is True
    assert "storage_pressure_above_raw_compaction_ceiling" in payload["previews"]["raw_training_compaction"]["blockers"]
    manifest_cmd = payload["repair_plan"][0]["cmd"]
    assert "--apply" not in manifest_cmd
    assert "raw_training_compaction_intelligence.py" in manifest_cmd[1]


def test_storage_backpressure_autopilot_refreshes_storage_guard_during_emergency_disk_guard(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    raw_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    raw_root.mkdir(parents=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 12.0,
            "storage_plane_contract": {
                "phase": "emergency_disk_guard",
                "disk_contract": {"external_available_gb": 1.5, "external_used_percent": 99.8},
            },
            "storage_efficiency_contract": {
                "active": True,
                "raw_compaction_required": True,
                "manifest_first_required": True,
                "adaptive_raw_training_wave": {
                    "manifest_refresh_required": True,
                    "compaction_apply_allowed_now": False,
                    "max_files": 0,
                    "max_gb": 0.0,
                },
            },
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 80000,
                "core_pending_lines": 40000,
                "estimated_total_drain_minutes": 120.0,
            },
            "storage": {"retention_debt_gb": 0.0, "backlog_drain_recommended_now": False, "backlog_quarantine_candidate_files": 0},
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "needs_work",
            "scan_roots": [{"path": str(raw_root), "exists": True, "protected": False}],
            "raw_summary": {"compression_candidate_count": 30, "compression_candidate_gb": 48.0},
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "drain_ready": False, "maintenance_ready": False},
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        raw_training_pressure_ceiling=0.6,
        raw_training_bot_logs_min_free_gb=0.1,
        raw_training_local_min_free_gb=0.1,
    )

    names = [row["name"] for row in payload["repair_plan"]]
    assert "botlogs_space_recovery" in names
    assert "data_collection_storage_guard" in names
    assert "raw_training_manifest_refresh" in names
    assert "raw_training_compaction" not in names
    assert names.index("botlogs_space_recovery") < names.index("data_collection_storage_guard")
    assert payload["metrics"]["storage_plane_phase"] == "emergency_disk_guard"
    assert payload["metrics"]["storage_emergency_disk_guard"] is True


def test_storage_backpressure_autopilot_runs_space_recovery_for_reserve_rebuild(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    raw_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    raw_root.mkdir(parents=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 8.0,
            "storage_plane_contract": {
                "phase": "storage_reserve_rebuild",
                "disk_contract": {"external_available_gb": 33.0, "external_used_percent": 96.4},
            },
            "storage_efficiency_contract": {
                "active": True,
                "metrics": {
                    "safe_space_recovery_selected_gb": 7.95,
                    "safe_space_recovery_deficit_gb": 31.0,
                    "storage_reserve_rebuild_required": True,
                },
                "adaptive_raw_training_wave": {
                    "manifest_refresh_required": True,
                    "compaction_apply_allowed_now": False,
                    "max_files": 0,
                    "max_gb": 0.0,
                },
            },
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 80000,
                "core_pending_lines": 40000,
                "estimated_total_drain_minutes": 120.0,
            },
            "storage": {"retention_debt_gb": 0.0, "backlog_drain_recommended_now": False, "backlog_quarantine_candidate_files": 0},
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "needs_work",
            "scan_roots": [{"path": str(raw_root), "exists": True, "protected": False}],
            "raw_summary": {"compression_candidate_count": 30, "compression_candidate_gb": 48.0},
        },
    )
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "safe_space_recovery": {
                "enabled": True,
                "candidate_count": 78,
                "candidate_gb": 210.0,
                "selected_count": 2,
                "selected_gb": 7.95,
                "target_free_gb": 64.0,
                "target_free_deficit_gb": 31.0,
                "effective_max_delete_gb": 8.0,
                "reserve_rebuild_required": True,
            }
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "drain_ready": False, "maintenance_ready": False},
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        raw_training_pressure_ceiling=0.6,
        raw_training_bot_logs_min_free_gb=0.1,
        raw_training_local_min_free_gb=0.1,
    )

    names = [row["name"] for row in payload["repair_plan"]]
    assert "botlogs_space_recovery" in names
    assert "data_collection_storage_guard" not in names
    assert payload["metrics"]["storage_plane_phase"] == "storage_reserve_rebuild"
    assert payload["metrics"]["botlogs_space_recovery_reserve_rebuild_required"] is True
    assert payload["previews"]["botlogs_space_recovery"]["target_free_gb"] == 64.0


def test_storage_backpressure_autopilot_allows_raw_compaction_when_only_sql_overlay_is_hot(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    raw_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    raw_root.mkdir(parents=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 4.1,
            "backpressure": {
                "core_pending_lines": 61463,
                "total_pending_lines": 62187,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 3335,
                    "total_pending_lines": 4039,
                    "oldest_pending_age_seconds": 553.671,
                },
                "pending_lines_threshold": 15000,
                "estimated_total_drain_minutes": 94.0,
            },
            "storage": {
                "retention_debt_gb": 0.0,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "needs_work",
            "scan_roots": [{"path": str(raw_root), "exists": True, "protected": False}],
            "raw_summary": {
                "compression_candidate_count": 30,
                "compression_candidate_gb": 48.0,
            },
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": False,
            "drain_ready": False,
            "maintenance_ready": False,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    payload = autopilot_src.build_payload(
        project_root,
        apply=False,
        raw_training_pressure_ceiling=0.6,
        raw_training_bot_logs_min_free_gb=0.1,
        raw_training_local_min_free_gb=0.1,
    )

    names = [row["name"] for row in payload["repair_plan"]]
    assert "raw_training_compaction" in names
    assert payload["previews"]["raw_training_compaction"]["actionable"] is True
    assert payload["previews"]["raw_training_compaction"]["overlay_only_pressure"] is True
    assert payload["previews"]["raw_training_compaction"]["pressure_source"] == "sql_overlay_ignored_for_raw_compaction"


def test_storage_backpressure_autopilot_adopts_uniform_turbo_plus_process(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.649,
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 439,
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 33.5,
                },
                "sql_overlay": {
                    "grade": "B",
                    "pressure_ratio": 0.649,
                    "core_pending_lines": 9739,
                    "total_pending_lines": 9739,
                    "oldest_pending_age_seconds": 68.399,
                    "used_for_pressure": True,
                },
                "truth_gap": {"pending_line_delta": 9300},
            },
            "backlog_relief_contract": {"active_issue_ids": ["sparse_huge_jsonl_files"]},
            "stale_pending_locator": {
                "oldest_sources": [
                    {
                        "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 7989,
                        "oldest_pending_age_seconds": 68.399,
                    }
                ]
            },
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 9739,
                "estimated_total_drain_minutes": 8.0,
            },
            "storage": {"retention_debt_gb": 0.0, "backlog_drain_recommended_now": False},
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "advisory",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 2, "memory_status": "soft_guard"},
            "storage_accelerator_contract": {
                "p_core_preprocess_workers": 3,
                "max_shard_writer_lanes": 8,
                "catch_up_wave_controller": {
                    "enabled": True,
                    "max_waves": 5,
                    "max_seconds_per_writer_cycle": 120,
                },
            },
            "single_writer_tuning_contract": {"hot_batch_size": 120000, "queue_batch_size": 120000},
        },
    )
    _write_json(
        health / "backlog_pump_infrabots_latest.json",
        {
            "overall_status": "advisory",
            "bots": {
                "shard_hotness_router_bot": {
                    "control_env": {"SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": "crypto_trading"},
                    "focused_sources": [
                        {
                            "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                            "shard": "crypto_trading",
                            "pressure_lane": "core",
                            "pending_lines": 7989,
                            "oldest_pending_age_seconds": 68.399,
                        }
                    ],
                },
                "catch_up_wave_budget_bot": {
                    "control_env": {
                        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "5",
                        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "120",
                    }
                },
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer"})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {
                "active": True,
                "current_step": "shard_linking",
                "shard_writer_lane_contract": {
                    "selected_shard_writer_lanes": 2,
                    "max_shard_writer_lanes": 2,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard"},
            "snapshot": {
                "pressure_level": "normal",
                "pressure_kind": "normal",
                "swap_used_gb": 2.0,
                "compressed_pressure_gb": 1.0,
                "pages_throttled": 0,
            },
            "workload_guidance": {"p_core_preprocess_worker_cap": 4},
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "ready", "actionable": False, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "idle", "ready_drainer_count": 0, "recommended_actions": []},
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": False,
            "drain_ready": False,
            "maintenance_ready": False,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {"focus_shards": [], "targeted_retention_debt_gb": 0.0, "severe_focus": False},
            "recommended_actions": [],
        },
    )

    uniform_env_keys = set(autopilot_src.uniform_src.env_dict(autopilot_src.uniform_src.build_payload(project_root)).keys())
    old_env = {key: os.environ.get(key) for key in uniform_env_keys}
    try:
        payload = autopilot_src.build_payload(
            project_root,
            apply=True,
            poll_seconds=20.0,
            wait_timeout_seconds=900.0,
            command_timeout_seconds=120,
            max_cycles=1,
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    uniform_process = payload["uniform_process"]
    speed_contract = uniform_process["payload"]["speed_contract"]

    assert payload["repair_plan"] == []
    assert uniform_process["enabled"] is True
    assert uniform_process["write_result"]["out_path"] == str(health / "backlog_drain_uniform_process_latest.json")
    assert uniform_process["write_result"]["override_path"] == str(project_root / "config" / ".env.backlog_drain_uniform_override")
    assert speed_contract["mode"] == "turbo_plus_single_writer_catchup"
    assert uniform_process["env_overrides"]["BACKLOG_DRAIN_TURBO_PLUS_ENABLED"] == "1"
    assert uniform_process["env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "health_fast,writer_progress,crypto_trading"
    assert uniform_process["env_overrides"]["SQL_LINK_SERVICE_HOT_BATCH_SIZE"] == "420000"
    assert payload["timing_contract"]["poll_seconds"] == 6
    assert payload["timing_contract"]["wait_timeout_seconds"] == 240
    assert payload["timing_contract"]["command_timeout_seconds"] == 210
    assert payload["timing_contract"]["max_cycles"] == 2
    assert payload["always_armed_contract"]["uniform_process_enabled"] is True
