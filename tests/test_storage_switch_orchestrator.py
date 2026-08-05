from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import storage_switch_orchestrator as src


def test_build_payload_quiesce_only_stops_and_switches_without_restart(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    seen: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        timeout_sec: int,
        env_overrides: dict[str, str] | None = None,
    ) -> dict:
        seen.append(list(cmd))
        payload: dict[str, object] = {}
        if "storage_failback_sync.py" in cmd[-2]:
            payload = {"mode": "local_fallback", "active_root": str(project_root / "local_fallback_storage")}
        return {
            "cmd": list(cmd),
            "rc": 0,
            "duration_ms": 5.0,
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
            "timed_out": False,
        }

    monkeypatch.setattr(src, "_run_command", _fake_run)
    monkeypatch.setattr(src.writer_src, "writer_state_snapshot", lambda project_root: {"active": True, "writer_pid": 123})
    monkeypatch.setattr(
        src.writer_src,
        "_wait_for_writer_idle",
        lambda project_root, poll_seconds, wait_timeout_seconds: {
            "requested": True,
            "completed": True,
            "timed_out": False,
            "attempts": 1,
            "waited_seconds": 0.2,
            "final_state": {"active": False, "writer_pid": None},
        },
    )

    payload = src.build_payload(
        project_root,
        target_mode="local",
        restart=False,
        eject=False,
        quiesce_only=True,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "quiesced_switched"
    assert payload["achieved_target_mode"] is True
    assert payload["quiesce_only"] is True
    assert "stop_stack" in payload["steps"]
    assert "storage_failback_sync" in payload["steps"]
    assert "feed_refresh" not in payload["steps"]
    assert seen[0][-1] == "stop"
    assert "storage_failback_sync.py" in seen[1][-2]


def test_build_payload_no_restart_skips_stop_and_restart(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        timeout_sec: int,
        env_overrides: dict[str, str] | None = None,
    ) -> dict:
        calls.append(list(cmd))
        payload = {"mode": "external", "active_root": "/Volumes/BOT_LOGS/schwab_trading_bot"}
        return {
            "cmd": list(cmd),
            "rc": 0,
            "duration_ms": 5.0,
            "payload": payload if "storage_failback_sync.py" in cmd[-2] else {},
            "stdout_tail": "",
            "stderr_tail": "",
            "timed_out": False,
        }

    monkeypatch.setattr(src, "_run_command", _fake_run)
    monkeypatch.setattr(src.writer_src, "writer_state_snapshot", lambda project_root: {"active": False, "writer_pid": None})

    payload = src.build_payload(
        project_root,
        target_mode="external",
        restart=False,
        eject=False,
        quiesce_only=False,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "switched"
    assert payload["steps"].keys() == {"storage_failback_sync", "storage_split_brain_reconciler"}
    assert all(cmd[-1] != "stop" for cmd in calls)
    json.dumps(payload, ensure_ascii=True)


def test_build_payload_no_restart_blocks_active_writer_before_route_mutation(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    calls: list[list[str]] = []
    monkeypatch.setattr(
        src,
        "_run_command",
        lambda cmd, **kwargs: calls.append(list(cmd)) or {"cmd": list(cmd), "rc": 0, "timed_out": False},
    )
    monkeypatch.setattr(
        src.writer_src,
        "writer_state_snapshot",
        lambda project_root: {"active": True, "writer_pid": 123},
    )

    payload = src.build_payload(
        project_root,
        target_mode="external",
        restart=False,
        eject=False,
        quiesce_only=False,
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked_writer_active"
    assert payload["route_mutation_performed"] is False
    assert calls == []
    assert not (project_root / "config" / ".env.storage_override").exists()
