import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import computer_task_intelligence as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_health(
    project_root: Path,
    *,
    resource: dict | None = None,
    memory: dict | None = None,
    runtime: dict | None = None,
    storage: dict | None = None,
    drainer: dict | None = None,
    switchboard: dict | None = None,
    process: dict | None = None,
) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "resource_guard_latest.json",
        resource
        or {
            "creative_session_kind": "none",
            "creative_session_level": "none",
            "creative_apps": [],
            "memory_pressure_state": "green",
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        memory
        or {
            "overall_status": "ready",
            "creative_session": {"active": False, "kind": "none", "level": "none"},
            "cotenant_awareness": {"active": False, "open_apps": [], "co_running_classes": []},
            "memory_snapshot": {"memory_pressure_state": "green"},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        runtime or {"overall_status": "ready", "host_saturation_score": 25.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        storage
        or {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 1000,
                "oldest_pending_age_seconds": 0.0,
                "pending_lines_threshold": 15000,
                "raw_live": {"line_estimation": {"sparse_large_line_pending_lines": 0, "sparse_large_line_pending_bytes": 0}},
            },
        },
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        drainer or {"backlog_section_scorecard": {"overall_grade": "A", "overall_score": 96.0}},
    )
    _write_json(
        health / "mode_switchboard_mission_control_latest.json",
        switchboard or {"operator_mode": {"selected_mode": "overnight_heavy"}},
    )
    _write_json(health / "process_watchdog_latest.json", process or {"status": [], "alerts": []})


def test_computer_task_intelligence_detects_music_dev_backlog_and_caps(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_health(
        project_root,
        resource={
            "creative_session_kind": "music_playback",
            "creative_session_level": "active",
            "creative_apps": ["iTunes"],
            "memory_pressure_state": "green",
        },
        memory={
            "overall_status": "blocked",
            "creative_session": {"active": True, "kind": "music_playback", "level": "active", "apps": ["iTunes"]},
            "cotenant_awareness": {"active": True, "open_apps": ["PyCharm", "Safari", "iTunes"], "co_running_classes": ["developer", "browser"], "co_running_level": "heavy_competition"},
            "memory_snapshot": {"memory_pressure_state": "green"},
        },
        runtime={"overall_status": "blocked", "host_saturation_score": 58.0, "compute_pressure_level": "elevated", "memory_pressure_level": "normal"},
        storage={
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 29644,
                "total_pending_lines": 30344,
                "oldest_pending_age_seconds": 425751.0,
                "pending_lines_threshold": 15000,
                "raw_live": {"line_estimation": {"sparse_large_line_pending_lines": 1006, "sparse_large_line_pending_bytes": 933910723}},
            },
        },
        drainer={"backlog_section_scorecard": {"overall_grade": "C", "overall_score": 73.7}},
        switchboard={"operator_mode": {"selected_mode": "daily_driver"}},
    )

    payload = src.build_payload(project_root, refresh_computer=False)
    env = payload["recommended_env_overrides"]
    unison = payload["computer_unison_contract"]
    needs = payload["a_grade_lift_contract"]["needs"]

    assert payload["task_profile"]["primary_task"] == "music_playback"
    assert "developer_work" in payload["task_profile"]["active_tasks"]
    assert "backlog_drain" in payload["task_profile"]["active_tasks"]
    assert env["SYSTEM_OPERATOR_MODE_REQUESTED"] == "daily_driver"
    assert env["TRAINING_RUNTIME_PAUSED_FOR_COMPUTER_TASK"] == "1"
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.35"
    assert env["COMPUTER_TASK_FOREGROUND_HARDENING_ACTIVE"] == "1"
    assert env["BOT_CPU_SCHEDULER_INTENT"] == "daily_driver_no_background_writer"
    assert env["SQL_LINK_WRITER_BACKGROUND_POLICY"] == "0"
    assert env["SQL_LINK_WRITER_NICE"] == "3"
    assert env["SLEEVE_NICE_BASELINE"] == "4"
    assert env["SLEEVE_NICE_SPECIALIZED"] == "8"
    assert env["COMPUTER_UNISON_CONTRACT_ACTIVE"] == "1"
    assert env["COMPUTER_RESOURCE_INTENT"] == "yield_to_foreground"
    assert env["COMPUTER_DO_NOT_TOUCH_VOLUMES"] == "/Volumes/VIDEO"
    assert unison["resource_intent"] == "yield_to_foreground"
    assert unison["safety_contract"]["does_not_touch_video_volume"] is True
    assert "/Volumes/VIDEO" in unison["safety_contract"]["do_not_touch_volumes"]
    assert payload["process_coordination"]["active"] is True
    assert any(row["section_id"] == "backlog_interference" for row in needs)
    assert "core_pending_lines <= 5000" in next(row for row in needs if row["section_id"] == "backlog_interference")["a_grade_exit_criteria"]


def test_computer_task_intelligence_uses_performance_writer_for_backlog_drain(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_health(
        project_root,
        storage={
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 32000,
                "total_pending_lines": 35000,
                "oldest_pending_age_seconds": 7200.0,
                "pending_lines_threshold": 15000,
                "raw_live": {"line_estimation": {"sparse_large_line_pending_lines": 2000, "sparse_large_line_pending_bytes": 800000000}},
            },
        },
        drainer={"backlog_section_scorecard": {"overall_grade": "D", "overall_score": 55.0}},
        switchboard={"operator_mode": {"selected_mode": "trading_focus"}},
    )

    payload = src.build_payload(project_root, refresh_computer=False)
    env = payload["recommended_env_overrides"]
    classes = {row["class_id"]: row for row in payload["process_coordination"]["process_classes"]}

    assert payload["task_profile"]["primary_task"] == "backlog_drain"
    assert env["BOT_CPU_SCHEDULER_INTENT"] == "performance_core_backlog_drain"
    assert env["SQL_LINK_WRITER_BACKGROUND_POLICY"] == "0"
    assert env["SQL_LINK_WRITER_NICE"] == "0"
    assert env["SLEEVE_NICE_BASELINE"] == "0"
    assert classes["single_writer"]["target_nice"] == 0
    assert classes["drainer_accelerator"]["target_nice"] == 0


def test_stale_process_context_infrabot_ignores_old_app_context(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_health(
        project_root,
        resource={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "creative_session_kind": "none",
            "creative_session_level": "none",
            "creative_apps": [],
            "co_running_apps": [],
            "co_running_classes": [],
            "co_running_cpu_sum": 0.0,
            "memory_pressure_state": "green",
        },
        memory={
            "timestamp_utc": "2020-01-01T00:00:00+00:00",
            "overall_status": "blocked",
            "creative_session": {"active": True, "kind": "music_playback", "level": "active", "apps": ["iTunes"]},
            "cotenant_awareness": {
                "active": True,
                "open_apps": ["PyCharm", "Safari", "iTunes"],
                "co_running_classes": ["developer", "browser"],
                "co_running_level": "heavy_competition",
                "co_running_cpu_sum": 250.0,
            },
            "co_running_session": {"active": True, "apps": ["PyCharm"], "classes": ["developer"], "cpu_sum": 250.0},
            "memory_snapshot": {"memory_pressure_state": "green"},
        },
        storage={
            "overall_status": "blocked",
            "backpressure": {
                "core_pending_lines": 22000,
                "total_pending_lines": 26000,
                "oldest_pending_age_seconds": 7200.0,
                "pending_lines_threshold": 15000,
                "raw_live": {"line_estimation": {"sparse_large_line_pending_lines": 500, "sparse_large_line_pending_bytes": 1000000}},
            },
        },
        drainer={"backlog_section_scorecard": {"overall_grade": "D", "overall_score": 55.0}},
        switchboard={"operator_mode": {"selected_mode": "trading_focus"}},
    )

    payload = src.build_payload(project_root, refresh_computer=False)
    session = payload["session_context"]
    infrabot = payload["stale_process_context_infrabot"]
    env = payload["recommended_env_overrides"]

    assert payload["task_profile"]["primary_task"] == "backlog_drain"
    assert "music_playback" not in payload["task_profile"]["active_tasks"]
    assert session["open_apps"] == []
    assert session["co_running_classes"] == []
    assert infrabot["status"] == "cleared_stale_context"
    assert infrabot["ignored_memory_efficiency_app_context"] is True
    assert env["STALE_PROCESS_CONTEXT_INFRABOT_ACTIVE"] == "1"
    assert env["STALE_PROCESS_CONTEXT_CLEARED"] == "1"
    assert env["PROCESS_CONTEXT_SOURCE"] == "fresh_resource_guard"


def test_computer_task_intelligence_allows_overnight_when_clean(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(src, "_is_off_hours", lambda now=None: True)
    project_root = tmp_path / "project"
    _seed_health(project_root)

    payload = src.build_payload(project_root, refresh_computer=False)
    env = payload["recommended_env_overrides"]

    assert payload["task_profile"]["primary_task"] == "overnight_research"
    assert payload["normal_use_scorecard"]["overall_grade"] == "A"
    assert payload["computer_unison_contract"]["resource_intent"] == "use_idle_headroom"
    assert payload["computer_unison_contract"]["preemption_level"] == "observe"
    assert env["COMPUTER_RESOURCE_INTENT"] == "use_idle_headroom"
    assert env["SYSTEM_OPERATOR_MODE_REQUESTED"] == "overnight_heavy"
    assert env["TRAINING_RUNTIME_PAUSED_FOR_COMPUTER_TASK"] == "0"
    assert env["ROSTER_EXPANSION_ALLOWED"] == "1"


def test_computer_task_override_file_is_shell_quoted(tmp_path: Path) -> None:
    path = tmp_path / ".env.computer_task_override"

    changed = src._write_override(path, {"COMPUTER_TASK_PROFILE": "music playback", "ASYNC_PIPELINE_WORKERS": "2"})

    text = path.read_text(encoding="utf-8")
    assert changed is True
    assert "COMPUTER_TASK_PROFILE='music playback'" in text
    assert "ASYNC_PIPELINE_WORKERS=2" in text
    assert src._write_override(path, {"COMPUTER_TASK_PROFILE": "music playback", "ASYNC_PIPELINE_WORKERS": "2"}) is False


def test_process_coordination_renices_only_matching_background_processes(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=True, text=True, timeout=0):  # noqa: ANN001
        calls.append([str(part) for part in cmd])
        if cmd[:2] == ["ps", "-axo"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "PID NI %CPU %MEM COMMAND\n"
                    "101 0 22.5 0.8 /bin/python scripts/run_shadow_training_loop.py --broker schwab\n"
                    "102 9 10.0 0.4 /bin/python scripts/ops/sql_link_shard_manager.py\n"
                    "103 20 9.0 0.2 /bin/python scripts/ops/live_macro_auto_watch.py\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(src.subprocess, "run", fake_run)
    policy = {
        "active": True,
        "max_processes": 4,
        "process_classes": [
            {"class_id": "market_runtime", "target_nice": 14, "patterns": ["scripts/run_shadow_training_loop.py"]},
            {"class_id": "single_writer", "target_nice": 12, "patterns": ["scripts/ops/sql_link_shard_manager.py"]},
            {"class_id": "macro_media_capture", "target_nice": 14, "patterns": ["scripts/ops/live_macro_auto_watch.py"]},
        ],
        "contract": {"renice_only": True},
    }

    result = src._coordinate_background_processes(policy, apply=True)

    assert result["matched_process_count"] == 3
    assert result["renice_attempted"] == 2
    assert result["renice_succeeded"] == 2
    assert ["renice", "-n", "14", "-p", "101"] in calls
    assert ["renice", "-n", "3", "-p", "102"] in calls
    assert result["actions"][1]["renice_delta"] == 3
