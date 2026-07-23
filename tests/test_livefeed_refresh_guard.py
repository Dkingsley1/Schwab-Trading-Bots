from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.ops import livefeed_refresh_guard as src


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_livefeed_refresh_guard_validates_all_dry_run_routes(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "livefeed_local_latest.json",
        {"status": "running", "alive": True, "health_writer": True, "writer_mode": "local_mirror", "source": "all", "heavy": 0},
    )

    def fake_run(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        if "livefeed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=livefeed-refresh\nsource=all\nmirror_only=1\n"
        elif "live-feed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=live-feed-refresh\nsource=all\nmirror_only=1\n"
        else:
            source = command[command.index("--source") + 1]
            stdout = f"feed_refresh_dry_run=1\ncmd=feed-refresh\nsource={source}\nmirror_only=0\n"
        return {"command": command, "returncode": 0, "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(project_root, out_path=project_root / "guard.json")

    assert payload["overall_status"] == "ready"
    assert payload["route_ok_count"] == 6
    assert payload["contract"]["validates_all_refresh_routes"] is True


def test_livefeed_refresh_guard_blocks_broken_alias(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "livefeed_local_latest.json",
        {"status": "running", "alive": True, "health_writer": True, "writer_mode": "local_mirror", "source": "all"},
    )

    def fake_run(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        if "live-feed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=live-feed-refresh\n"
        elif "livefeed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=livefeed-refresh\nsource=all\nmirror_only=1\n"
        else:
            source = command[command.index("--source") + 1]
            stdout = f"feed_refresh_dry_run=1\ncmd=feed-refresh\nsource={source}\nmirror_only=0\n"
        return {"command": command, "returncode": 0, "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(project_root, out_path=project_root / "guard.json")

    assert payload["overall_status"] == "blocked"
    assert "route_failed:live_feed_refresh_alias" in payload["blockers"]


def test_livefeed_refresh_guard_apply_requires_local_mirror_ready(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "livefeed_local_latest.json",
        {"status": "running", "alive": True, "health_writer": True, "writer_mode": "local_mirror", "source": "all"},
    )

    def fake_run(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        if "livefeed-refresh" in command and "--dry-run" not in command:
            stdout = "livefeed_refresh_completed source=all local_mirror=ready\n"
        elif "livefeed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=livefeed-refresh\nsource=all\nmirror_only=1\nlocal_mirror=refresh\n"
        elif "live-feed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=live-feed-refresh\nsource=all\nmirror_only=1\nlocal_mirror=refresh\n"
        else:
            source = command[command.index("--source") + 1]
            stdout = f"feed_refresh_dry_run=1\ncmd=feed-refresh\nsource={source}\nmirror_only=0\n"
        return {"command": command, "returncode": 0, "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(project_root, apply=True, out_path=project_root / "guard.json")

    assert payload["overall_status"] == "ready"
    assert payload["apply_result"]["ok"] is True
    assert payload["contract"]["apply_refreshes_local_livefeed_mirror"] is True


def test_livefeed_refresh_guard_blocks_unsupervised_health_writer(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "livefeed_local_latest.json",
        {"status": "running", "alive": True, "source": "main", "heavy": 1},
    )

    def fake_run(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        if "livefeed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=livefeed-refresh\nsource=all\nmirror_only=1\n"
        elif "live-feed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=live-feed-refresh\nsource=all\nmirror_only=1\n"
        else:
            source = command[command.index("--source") + 1]
            stdout = f"feed_refresh_dry_run=1\ncmd=feed-refresh\nsource={source}\nmirror_only=0\n"
        return {"command": command, "returncode": 0, "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(project_root, out_path=project_root / "guard.json")

    assert payload["overall_status"] == "blocked"
    assert "livefeed_health_writer_not_supervised" in payload["blockers"]


def test_livefeed_refresh_guard_detects_heavy_viewer_without_supervised_mirror(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "livefeed_local_latest.json",
        {
            "status": "running",
            "alive": True,
            "pid": 999999,
            "health_writer": True,
            "writer_mode": "local_mirror",
            "source": "main",
            "skipped_file_count": 0,
        },
    )

    def fake_run(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        if "livefeed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=livefeed-refresh\nsource=all\nmirror_only=1\n"
        elif "live-feed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=live-feed-refresh\nsource=all\nmirror_only=1\n"
        else:
            source = command[command.index("--source") + 1]
            stdout = f"feed_refresh_dry_run=1\ncmd=feed-refresh\nsource={source}\nmirror_only=0\n"
        return {"command": command, "returncode": 0, "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_command", fake_run)
    monkeypatch.setattr(
        src,
        "_process_snapshot",
        lambda project_root, source, health_pid=None: {
            "source": source,
            "health_pid": health_pid,
            "health_pid_alive": False,
            "local_mirror_process_count": 0,
            "heavy_process_count": 1,
            "guarded_heavy_process_count": 0,
            "process_count": 1,
            "local_mirror_processes": [],
            "heavy_processes": [{"pid": 123, "command": "live_feed_tail.sh --source main --heavy"}],
            "guarded_heavy_processes": [],
        },
    )

    payload = src.build_payload(project_root, out_path=project_root / "guard.json")

    assert payload["overall_status"] == "blocked"
    assert payload["health"]["operating_mode"] == "operator_heavy_viewer_only"
    assert "livefeed_health_pid_not_running" in payload["blockers"]
    assert "livefeed_supervised_mirror_missing_while_heavy_active" in payload["blockers"]
    assert payload["recommended_actions"] == ["./scripts/ops/opsctl.sh livefeed-refresh-guard --apply --json"]
    assert payload["degradation"]["operator_heavy_viewer_count"] == 1


def test_livefeed_refresh_guard_accepts_rotated_health_helper_when_mirror_alive(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "livefeed_local_latest.json",
        {
            "status": "running",
            "alive": True,
            "pid": 999999,
            "health_writer": True,
            "writer_mode": "local_mirror",
            "source": "main",
            "skipped_file_count": 0,
        },
    )

    def fake_run(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        if "livefeed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=livefeed-refresh\nsource=all\nmirror_only=1\n"
        elif "live-feed-refresh" in command:
            stdout = "feed_refresh_dry_run=1\ncmd=live-feed-refresh\nsource=all\nmirror_only=1\n"
        else:
            source = command[command.index("--source") + 1]
            stdout = f"feed_refresh_dry_run=1\ncmd=feed-refresh\nsource={source}\nmirror_only=0\n"
        return {"command": command, "returncode": 0, "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_command", fake_run)
    monkeypatch.setattr(
        src,
        "_process_snapshot",
        lambda project_root, source, health_pid=None: {
            "source": source,
            "health_pid": health_pid,
            "health_pid_alive": False,
            "local_mirror_process_count": 1,
            "heavy_process_count": 1,
            "guarded_heavy_process_count": 0,
            "process_count": 2,
            "local_mirror_processes": [{"pid": 123, "command": "live_feed_tail.sh --source main --lines 120 --no-color"}],
            "heavy_processes": [{"pid": 456, "command": "live_feed_tail.sh --source main --heavy"}],
            "guarded_heavy_processes": [],
        },
    )

    payload = src.build_payload(project_root, out_path=project_root / "guard.json")

    assert payload["overall_status"] == "ready"
    assert payload["health"]["ok"] is True
    assert payload["health"]["pid_known_dead"] is True
    assert payload["health"]["pid_rotated_to_helper"] is True
    assert "livefeed_health_pid_not_running" not in payload["blockers"]
    assert payload["degradation"]["supervised_local_mirror"] is True
