import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import process_watchdog as pw


def test_refresh_runtime_reports_uses_lightweight_commands(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    one_numbers = project_root / "exports" / "one_numbers" / "one_numbers_summary.json"
    one_numbers.parent.mkdir(parents=True, exist_ok=True)
    paper_performance = project_root / "governance" / "health" / "paper_performance_latest.json"
    paper_performance.parent.mkdir(parents=True, exist_ok=True)
    backpressure = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    divergence = project_root / "governance" / "health" / "data_source_divergence_latest.json"
    daily_summary = project_root / "exports" / "sql_reports" / f"daily_runtime_summary_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    daily_summary.parent.mkdir(parents=True, exist_ok=True)
    daily_summary.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)

    calls: list[list[str]] = []

    def _fake_file_age(path: Path) -> float:
        if path in {one_numbers, paper_performance, backpressure, divergence}:
            return 999999.0
        return 0.0

    def _fake_run(cmd: list[str]):
        calls.append(cmd)
        if "build_one_numbers_report.py" in str(cmd[1:]):
            one_numbers.write_text(json.dumps({"generated_utc": "2026-03-31T21:00:00+00:00"}), encoding="utf-8")
            return 0, "", ""
        if "paper_performance_report.py" in str(cmd[1:]):
            paper_performance.write_text(json.dumps({"timestamp_utc": "2026-03-31T21:00:00+00:00"}), encoding="utf-8")
            return 0, "", ""
        if "ingestion_backpressure_guard.py" in str(cmd[1:]):
            backpressure.write_text(json.dumps({"timestamp_utc": "2026-03-31T21:00:00+00:00"}), encoding="utf-8")
            return 0, "", ""
        if "data_source_divergence_bot.py" in str(cmd[1:]):
            divergence.write_text(json.dumps({"timestamp_utc": "2026-03-31T21:00:00+00:00", "ok": True}), encoding="utf-8")
            return 0, "", ""
        return 0, "{}", ""

    monkeypatch.setattr(pw, "_file_age_seconds", _fake_file_age)
    monkeypatch.setattr(pw, "_run", _fake_run)
    monkeypatch.setattr(pw, "_proc_running", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(pw, "_resource_guard_allows_job", lambda job_name, profile="optional": (True, f"{job_name}:{profile}:ok"))

    out = pw._refresh_runtime_reports(max_age_seconds=60)

    assert out["one_numbers"]["refreshed"] is True
    assert out["paper_performance"]["refreshed"] is True
    assert out["ingestion_backpressure"]["refreshed"] is True
    assert out["data_source_divergence"]["refreshed"] is True
    assert any(
        "build_one_numbers_report.py" in " ".join(cmd)
        and "--lightweight" in cmd
        and "--no-sql-write" in cmd
        for cmd in calls
    )
    assert any("paper_performance_report.py" in " ".join(cmd) and "--json-only" in cmd for cmd in calls)
    assert any("ingestion_backpressure_guard.py" in " ".join(cmd) and "--json" in cmd for cmd in calls)
    assert any("data_source_divergence_bot.py" in " ".join(cmd) and "--json" in cmd for cmd in calls)


def test_refresh_runtime_reports_flags_stuck_refresh_process(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    one_numbers = project_root / "exports" / "one_numbers" / "one_numbers_summary.json"
    one_numbers.parent.mkdir(parents=True, exist_ok=True)
    paper_performance = project_root / "governance" / "health" / "paper_performance_latest.json"
    paper_performance.parent.mkdir(parents=True, exist_ok=True)
    backpressure = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    backpressure.write_text("{}", encoding="utf-8")
    divergence = project_root / "governance" / "health" / "data_source_divergence_latest.json"
    divergence.write_text("{}", encoding="utf-8")
    daily_summary = project_root / "exports" / "sql_reports" / f"daily_runtime_summary_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    daily_summary.parent.mkdir(parents=True, exist_ok=True)
    daily_summary.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("OPS_WATCHDOG_REFRESH_STUCK_SECONDS", "300")
    monkeypatch.setattr(
        pw,
        "_file_age_seconds",
        lambda path: 999999.0 if path == one_numbers else 0.0,
    )
    monkeypatch.setattr(pw, "_resource_guard_allows_job", lambda job_name, profile="optional": (True, f"{job_name}:{profile}:ok"))
    monkeypatch.setattr(pw, "_proc_running", lambda pattern, exclude_patterns=None: 1 if "build_one_numbers_report.py" in pattern else 0)
    monkeypatch.setattr(pw, "_proc_elapsed_seconds", lambda pattern, exclude_patterns=None: 1200.0 if "build_one_numbers_report.py" in pattern else None)
    monkeypatch.setattr(pw, "_run", lambda cmd: (_ for _ in ()).throw(AssertionError("refresh should not rerun while a stuck process is still present")))

    out = pw._refresh_runtime_reports(max_age_seconds=60)

    assert out["one_numbers"]["refreshed"] is False
    assert out["one_numbers"]["error"] == "refresh_stuck_suspected"
    assert out["one_numbers"]["running_seconds"] == 1200.0


def test_run_returns_timeout_payload(monkeypatch) -> None:
    def _fake_run(*_args, **_kwargs):
        exc = subprocess.TimeoutExpired(
            cmd=["fake-helper", "--json"],
            timeout=12.0,
        )
        exc.output = "partial stdout\n"
        exc.stderr = "partial stderr\n"
        raise exc

    monkeypatch.setattr(pw.subprocess, "run", _fake_run)

    rc, stdout, stderr = pw._run(["fake-helper", "--json"], timeout_seconds=12.0)

    assert rc == 124
    assert stdout == "partial stdout"
    assert "timeout_after_seconds=12.0" in stderr
    assert "partial stderr" in stderr


def test_build_execution_lane_target_uses_paper_health_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pw, "PROJECT_ROOT", tmp_path)

    target = pw._build_execution_lane_target("paper", heartbeat_max_age_seconds=240)

    assert target["name"] == "execution_lane_paper"
    assert target["pattern"] == "scripts/run_execution_lane.py --mode paper"
    assert target["cmd"][-2:] == ["--mode", "paper"]
    assert str(target["heartbeat_glob"]).endswith("execution_lane_paper_latest.json")
    assert target["heartbeat_max_age_seconds"] == 240
