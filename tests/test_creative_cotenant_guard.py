import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import creative_cotenant_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _fixture_resource_snapshot(project_root: Path, **_: object) -> dict:
    return json.loads(
        (project_root / "governance" / "health" / "resource_guard_latest.json").read_text(encoding="utf-8")
    )


def test_creative_cotenant_guard_applies_override_and_dedupes(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "creative_apps_active": True,
            "creative_app_count": 1,
            "creative_apps": ["Logic Pro"],
            "creative_session_level": "active",
            "editing_app_cpu_sum": 48.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {
                "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
                "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
                "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE": "pro_balanced",
            },
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.2},
    )
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {"overall_status": "blocked", "throttle_profile": "protect_live", "host_saturation_score": 91.0},
    )

    pid_snapshots = iter([[101, 202, 303], [303]])
    monkeypatch.setattr(src, "_refresh_resource_guard_snapshot", _fixture_resource_snapshot)
    monkeypatch.setattr(src, "_pgrep_matching_pids", lambda _pattern: next(pid_snapshots))
    monkeypatch.setattr(src, "_parent_pid_for_pid", lambda _pid: 0)
    monkeypatch.setattr(src, "_terminate_pids", lambda pids: [{"pid": pid, "ok": True} for pid in pids])

    override_path = tmp_path / "config" / ".env.memory_efficiency_override"
    payload = src.build_payload(tmp_path, apply=True, override_path=override_path)

    assert payload["memory_efficiency"]["changed"] is True
    assert payload["memory_efficiency"]["recommended_profile"] == "pro_balanced"
    assert payload["paper_execution_lane"]["count_before"] == 3
    assert payload["paper_execution_lane"]["count_after"] == 1
    assert payload["paper_execution_lane"]["keep_pid"] == 303
    assert payload["paper_execution_lane"]["extra_pids"] == [101, 202]
    assert "memory_efficiency_override_updated" in payload["actions"]
    assert "paper_execution_lane_deduped" in payload["actions"]
    assert override_path.exists() is True


def test_paper_lane_singleton_prefers_current_launcher_child(tmp_path: Path, monkeypatch) -> None:
    pid_snapshots = iter([[101, 202, 303], [202]])
    monkeypatch.setattr(src, "_pgrep_matching_pids", lambda _pattern: next(pid_snapshots))
    monkeypatch.setattr(src, "_parent_pid_for_pid", lambda pid: {101: 11, 202: 22, 303: 33}.get(pid, 0))
    monkeypatch.setattr(
        src,
        "_command_for_pid",
        lambda pid: f"python {tmp_path}/scripts/run_all_sleeves.py" if pid == 22 else "python stale_parent.py",
    )
    monkeypatch.setattr(src, "_terminate_pids", lambda pids: [{"pid": pid, "ok": True} for pid in pids])

    payload = src.build_paper_lane_singleton(tmp_path, apply=True)

    assert payload["ok"] is True
    assert payload["paper_execution_lane"]["keep_pid"] == 202
    assert payload["paper_execution_lane"]["extra_pids"] == [101, 303]
    assert payload["paper_execution_lane"]["parent_owned_pids"] == [202]
    assert payload["live_execution_allowed"] is False


def test_creative_cotenant_guard_protects_music_playback(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "creative_apps_active": True,
            "creative_app_count": 1,
            "creative_apps": ["Music"],
            "creative_session_level": "active",
            "creative_session_kind": "music_playback",
            "editing_app_cpu_sum": 4.5,
            "music_playback_cpu": 4.5,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe"},
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.2},
    )
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "throttle_profile": "protect_live", "host_saturation_score": 65.0},
    )

    monkeypatch.setattr(src, "_refresh_resource_guard_snapshot", _fixture_resource_snapshot)
    monkeypatch.setattr(src, "_pgrep_matching_pids", lambda _pattern: [101])
    monkeypatch.setattr(src, "_matching_heavy_research_processes", lambda: [{"pid": 202, "pattern": "scripts/run_shadow_training_loop.py", "command": "loop"}])
    monkeypatch.setattr(src, "_terminate_pids", lambda pids: [{"pid": pid, "ok": True} for pid in pids])

    payload = src.build_payload(tmp_path, apply=False, override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["creative_mode"]["active"] is True
    assert payload["creative_mode"]["kind"] == "music_playback"
    assert payload["memory_efficiency"]["recommended_profile"] == "air_safe"
    assert payload["memory_efficiency"]["recommended_env_overrides"]["AUDIO_PLAYBACK_PRIORITY"] == "1"
    assert payload["pause_contract"]["env_contract"]["TRAINING_RUNTIME_PAUSED_FOR_CREATIVE"] == "1"
    assert payload["heavy_research_pause"]["active"] is True
    assert payload["heavy_research_pause"]["terminate_processes"] is True
    assert payload["heavy_research_pause"]["action"] == "sigterm_optional_heavy_research"
    assert payload["heavy_research_pause"]["terminated_count"] == 0
    assert payload["creative_mode"]["audio_regression_guard_active"] is True
    assert "music_audio_regression_guard_active" in payload["actions"]
    assert "audio_playback_protection" in payload["controller_contract"]["scope"]
    assert "audio_regression_guard" in payload["controller_contract"]["scope"]


def test_creative_cotenant_guard_reports_missing_paper_lane_without_spawning(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {"memory_pressure_state": "green", "memory_pressure_kind": "none"},
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {"applied_tier": "pro_balanced", "env_overrides": {}, "hardware": {"memory_gb": 32.0}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.0},
    )

    monkeypatch.setattr(src, "_refresh_resource_guard_snapshot", _fixture_resource_snapshot)
    monkeypatch.setattr(src, "_pgrep_matching_pids", lambda _pattern: [])

    payload = src.build_payload(tmp_path, apply=False, override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["paper_execution_lane"]["count_before"] == 0
    assert payload["paper_execution_lane"]["count_after"] == 0
    assert payload["overall_status"] == "needs_work"
    assert "paper_execution_lane_missing" in payload["actions"]
