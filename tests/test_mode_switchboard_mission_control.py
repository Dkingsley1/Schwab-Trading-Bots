import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import mode_switchboard_mission_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_switchboard_health(
    project_root: Path,
    *,
    runtime_throttle: dict | None = None,
    memory_efficiency: dict | None = None,
    storage: dict | None = None,
    drainer_intelligence: dict | None = None,
    computer_task: dict | None = None,
) -> Path:
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {"broker_ready": True, "session_ready": True, "paper_lane_fresh": True, "live_lane_running": False},
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "ready"}, "release_contract": {"shared_host_training_resume_allowed": True}},
    )
    _write_json(health / "runtime_access_mode_latest.json", {"mode": "native"})
    _write_json(health / "portable_brain_contract_latest.json", {"host_contract": {"host_profile": "max_throughput"}})
    _write_json(
        health / "process_watchdog_latest.json",
        {"status": [{"name": "shadow_watchdog", "running": 1}, {"name": "paper_execution_lane", "running": 1}]},
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        runtime_throttle or {"overall_status": "ready", "host_saturation_score": 42.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        memory_efficiency or {"overall_status": "ready", "creative_session": {"active": False, "level": "none"}, "cotenant_awareness": {"active": False}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        storage or {"overall_status": "ready", "backpressure": {"total_pending_lines": 8000, "core_pending_lines": 4000, "pending_lines_threshold": 15000}},
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        drainer_intelligence or {"backlog_section_scorecard": {"overall_grade": "A", "overall_score": 93.0}},
    )
    _write_json(health / "computer_task_intelligence_latest.json", computer_task or {})
    return health


def test_mode_switchboard_mission_control_tracks_shadow_paper_and_live(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_switchboard_health(project_root)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["mode_counts"]["active"] == 2
    assert payload["control_surface"]["host_profile"] == "max_throughput"
    assert any(row["mode"] == "shadow" and row["active"] for row in payload["modes"])
    assert any(row["mode"] == "paper" and row["active"] for row in payload["modes"])
    assert any(row["mode"] == "live" and row["ready"] for row in payload["modes"])
    assert payload["operator_mode"]["selected_mode"] in {"trading_focus", "overnight_heavy"}


def test_operator_mode_daily_driver_caps_foreground_and_backlog(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_OPERATOR_MODE_REQUESTED", raising=False)
    project_root = tmp_path / "project"
    _seed_switchboard_health(
        project_root,
        runtime_throttle={"overall_status": "blocked", "host_saturation_score": 84.2, "compute_pressure_level": "high", "memory_pressure_level": "normal"},
        memory_efficiency={
            "overall_status": "constrained",
            "creative_session": {"active": True, "kind": "music_playback", "level": "active"},
            "cotenant_awareness": {"active": True, "co_running_level": "interactive", "open_apps": ["Music"]},
        },
        storage={"overall_status": "blocked", "backpressure": {"total_pending_lines": 48000, "core_pending_lines": 39297, "pending_lines_threshold": 15000}},
        drainer_intelligence={"backlog_section_scorecard": {"overall_grade": "C", "overall_score": 72.4}},
    )

    operator_mode = src.build_payload(project_root)["operator_mode"]
    env = operator_mode["env_overrides"]
    controls = {row["control"]: row for row in operator_mode["six_point_taming_contract"]}

    assert operator_mode["selected_mode"] == "daily_driver"
    assert env["DAILY_DRIVER_MODE_ACTIVE"] == "1"
    assert env["BACKLOG_INTAKE_GOVERNOR_ACTIVE"] == "1"
    assert env["FOREGROUND_APP_GOVERNOR_ACTIVE"] == "1"
    assert env["TRAINING_RUNTIME_PAUSED_BY_OPERATOR_MODE"] == "1"
    assert env["HEAVY_COLLECTORS_PAUSED_BY_OPERATOR_MODE"] == "1"
    assert env["ROSTER_EXPANSION_ALLOWED"] == "0"
    assert controls["daily_driver_mode"]["active"] is True
    assert controls["backlog_intake_governor"]["active"] is True
    assert controls["foreground_app_contract"]["active"] is True


def test_operator_mode_uses_computer_task_budget_when_available(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_OPERATOR_MODE_REQUESTED", raising=False)
    project_root = tmp_path / "project"
    _seed_switchboard_health(
        project_root,
        runtime_throttle={"overall_status": "ready", "host_saturation_score": 48.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
        memory_efficiency={"overall_status": "ready", "creative_session": {"active": True, "kind": "music_playback", "level": "active"}, "cotenant_awareness": {"active": True, "open_apps": ["Music"]}},
        storage={"overall_status": "blocked", "backpressure": {"total_pending_lines": 22000, "core_pending_lines": 18000, "pending_lines_threshold": 15000}},
        computer_task={
            "task_profile": {"primary_task": "music_playback"},
            "normal_use_budget": {
                "requested_operator_mode": "daily_driver",
                "collector_intake_ratio": "0.40",
                "coinbase_snapshot_workers": "1",
                "async_pipeline_workers": "2",
                "quant_model_workers": "1",
                "training_allowed": False,
                "heavy_collectors_allowed": False,
                "report_refresh_allowed": False,
                "live_feed_lines": "25",
                "live_feed_follow_files": "6",
            },
        },
    )

    operator_mode = src.build_payload(project_root)["operator_mode"]
    env = operator_mode["env_overrides"]

    assert operator_mode["computer_task_budget_used"] is True
    assert operator_mode["computer_task_profile"] == "music_playback"
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.20"
    assert env["LIVE_FEED_HEAVY_DEFAULT_LINES"] == "25"
    assert env["LIVE_FEED_HEAVY_MAX_FOLLOW_FILES"] == "6"


def test_operator_mode_uses_hardened_computer_task_env_when_available(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_OPERATOR_MODE_REQUESTED", raising=False)
    project_root = tmp_path / "project"
    _seed_switchboard_health(
        project_root,
        runtime_throttle={"overall_status": "blocked", "host_saturation_score": 59.0, "compute_pressure_level": "elevated", "memory_pressure_level": "normal"},
        memory_efficiency={"overall_status": "blocked", "creative_session": {"active": True, "kind": "music_playback", "level": "active"}, "cotenant_awareness": {"active": True, "open_apps": ["Music"]}},
        storage={"overall_status": "blocked", "backpressure": {"total_pending_lines": 30344, "core_pending_lines": 29644, "pending_lines_threshold": 15000}},
        computer_task={
            "task_profile": {"primary_task": "music_playback"},
            "normal_use_budget": {"requested_operator_mode": "daily_driver", "collector_intake_ratio": "0.40", "async_pipeline_workers": "2", "live_feed_lines": "25", "live_feed_follow_files": "6"},
            "recommended_env_overrides": {
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.35",
                "ASYNC_PIPELINE_WORKERS": "1",
                "LIVE_FEED_HEAVY_DEFAULT_LINES": "20",
                "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES": "5",
                "COMPUTER_TASK_FOREGROUND_HARDENING_ACTIVE": "1",
                "COMPUTER_UNISON_CONTRACT_ACTIVE": "1",
                "COMPUTER_RESOURCE_INTENT": "yield_to_foreground",
                "COMPUTER_FRICTION_INDEX": "38",
                "COMPUTER_PREEMPTION_LEVEL": "protect",
                "COMPUTER_PROTECTED_TASKS": "music_playback,developer_work",
                "COMPUTER_DO_NOT_TOUCH_VOLUMES": "/Volumes/VIDEO",
                "MACOS_NORMAL_USE_FIRST": "1",
                "SUPPORT_TELEMETRY_SHED_ACTIVE": "1",
                "OPS_SUPPORT_JOB_NICE": "14",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "30",
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "300",
            },
        },
    )

    operator_mode = src.build_payload(project_root)["operator_mode"]
    env = operator_mode["env_overrides"]

    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.20"
    assert env["ASYNC_PIPELINE_WORKERS"] == "1"
    assert env["LIVE_FEED_HEAVY_DEFAULT_LINES"] == "20"
    assert env["LIVE_FEED_HEAVY_MAX_FOLLOW_FILES"] == "5"
    assert env["COMPUTER_TASK_FOREGROUND_HARDENING_ACTIVE"] == "1"
    assert env["COMPUTER_UNISON_CONTRACT_ACTIVE"] == "1"
    assert env["COMPUTER_RESOURCE_INTENT"] == "yield_to_foreground"
    assert env["COMPUTER_PREEMPTION_LEVEL"] == "protect"
    assert env["COMPUTER_DO_NOT_TOUCH_VOLUMES"] == "/Volumes/VIDEO"
    assert env["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "30"
    assert env["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "300"
    assert operator_mode["computer_unison_used"] is True
    assert operator_mode["computer_resource_intent"] == "yield_to_foreground"


def test_operator_mode_denies_overnight_heavy_under_pressure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_OPERATOR_MODE_REQUESTED", "overnight-heavy")
    project_root = tmp_path / "project"
    _seed_switchboard_health(
        project_root,
        runtime_throttle={"overall_status": "blocked", "host_saturation_score": 82.0, "compute_pressure_level": "high", "memory_pressure_level": "normal"},
        memory_efficiency={"overall_status": "ready", "creative_session": {"active": True, "kind": "music_playback", "level": "active"}, "cotenant_awareness": {"active": False}},
        storage={"overall_status": "blocked", "backpressure": {"total_pending_lines": 30000, "core_pending_lines": 25000, "pending_lines_threshold": 15000}},
    )

    operator_mode = src.build_payload(project_root)["operator_mode"]

    assert operator_mode["selected_mode"] == "daily_driver"
    assert operator_mode["reason"] == "overnight_heavy_denied_by_foreground_or_pressure"
    assert operator_mode["env_overrides"]["OVERNIGHT_HEAVY_MODE_ACTIVE"] == "0"


def test_operator_mode_allows_overnight_heavy_when_clean(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_OPERATOR_MODE_REQUESTED", raising=False)
    monkeypatch.setattr(src, "_is_off_hours", lambda now=None: True)
    project_root = tmp_path / "project"
    _seed_switchboard_health(
        project_root,
        runtime_throttle={"overall_status": "ready", "host_saturation_score": 36.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
        memory_efficiency={"overall_status": "ready", "creative_session": {"active": False, "level": "none"}, "cotenant_awareness": {"active": False}},
        storage={"overall_status": "ready", "backpressure": {"total_pending_lines": 4200, "core_pending_lines": 2400, "pending_lines_threshold": 15000}},
        drainer_intelligence={"backlog_section_scorecard": {"overall_grade": "A", "overall_score": 95.0}},
    )

    operator_mode = src.build_payload(project_root)["operator_mode"]
    env = operator_mode["env_overrides"]

    assert operator_mode["selected_mode"] == "overnight_heavy"
    assert env["OVERNIGHT_HEAVY_MODE_ACTIVE"] == "1"
    assert env["TRAINING_RUNTIME_PAUSED_BY_OPERATOR_MODE"] == "0"
    assert env["HEAVY_COLLECTORS_PAUSED_BY_OPERATOR_MODE"] == "0"
    assert env["ROSTER_EXPANSION_ALLOWED"] == "1"


def test_operator_mode_override_file_is_stable_and_shell_quoted(tmp_path: Path) -> None:
    override_path = tmp_path / ".env.operator_mode_override"

    changed = src._write_override(
        override_path,
        {
            "SYSTEM_OPERATOR_MODE": "daily_driver",
            "SYSTEM_OPERATOR_MODE_REASON": "foreground backlog",
        },
    )

    text = override_path.read_text(encoding="utf-8")
    assert changed is True
    assert "SYSTEM_OPERATOR_MODE=daily_driver" in text
    assert "SYSTEM_OPERATOR_MODE_REASON='foreground backlog'" in text
    assert src._write_override(
        override_path,
        {
            "SYSTEM_OPERATOR_MODE": "daily_driver",
            "SYSTEM_OPERATOR_MODE_REASON": "foreground backlog",
        },
    ) is False
