import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.nightly_resilience_check as nightly


def test_resolves_launchd_watchdog_log_path(tmp_path: Path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    fake_project = tmp_path / "project"
    log_path = fake_home / "Library" / "Logs" / "schwab_trading_bot" / "launchd_watchdog" / "shadow_watchdog.out.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("watchdog check\n", encoding="utf-8")

    monkeypatch.setattr(nightly.Path, "home", lambda: fake_home)
    monkeypatch.setattr(nightly, "PROJECT_ROOT", fake_project)

    assert nightly._resolve_watchdog_log() == log_path


def test_process_backed_stale_logs_are_warnings(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(nightly, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(nightly, "_pgrep_count", lambda pattern: 1 if "shadow_watchdog" in pattern else 2)
    monkeypatch.setattr(nightly, "_resolve_watchdog_log", lambda: None)
    monkeypatch.setattr(nightly, "_resolve_all_sleeves_log", lambda: tmp_path / "old_all_sleeves.log")
    monkeypatch.setattr(nightly, "_fresh_minutes", lambda path: 999.0)
    monkeypatch.setattr(nightly, "_count_keyword", lambda path, keyword: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_resilience_check.py",
            "--out-file",
            str(tmp_path / "nightly.json"),
            "--event-file",
            str(tmp_path / "nightly.jsonl"),
            "--json",
        ],
    )

    rc = nightly.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["failed_checks"] == []
    assert sorted(payload["warnings"]) == [
        "all_sleeves_log_stale_loops_running",
        "watchdog_log_stale_process_running",
    ]


def test_process_watchdog_artifact_certifies_current_watchdog(tmp_path: Path, monkeypatch, capsys) -> None:
    fake_project = tmp_path / "project"
    health = fake_project / "governance" / "health"
    health.mkdir(parents=True)
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "watchdog_intelligence": {
                    "overall_status": "ready",
                    "active_issue_count": 0,
                    "restart_storm_count": 0,
                    "alert_count": 0,
                    "target_count": 2,
                    "healthy_target_count": 2,
                },
                "status": [
                    {"name": "all_sleeves", "process_live": True, "heartbeat_ok": True},
                    {"name": "coinbase_loop", "process_live": True, "heartbeat_ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(nightly, "PROJECT_ROOT", fake_project)
    monkeypatch.setattr(nightly, "_pgrep_count", lambda pattern: 0 if "shadow_watchdog" in pattern else 2)
    monkeypatch.setattr(nightly, "_resolve_watchdog_log", lambda: None)
    monkeypatch.setattr(nightly, "_resolve_all_sleeves_log", lambda: None)
    monkeypatch.setattr(nightly, "_count_keyword", lambda path, keyword: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_resilience_check.py",
            "--out-file",
            str(tmp_path / "nightly.json"),
            "--event-file",
            str(tmp_path / "nightly.jsonl"),
            "--json",
        ],
    )

    rc = nightly.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["failed_checks"] == []
    assert payload["metrics"]["process_watchdog_certified"] is True
    assert "watchdog_log_stale_process_watchdog_certified" in payload["warnings"]


def test_process_watchdog_artifact_certifies_current_all_sleeves_when_wrapper_log_is_stale(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fake_project = tmp_path / "project"
    health = fake_project / "governance" / "health"
    health.mkdir(parents=True)
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "watchdog_intelligence": {
                    "overall_status": "ready",
                    "active_issue_count": 0,
                    "restart_storm_count": 0,
                    "alert_count": 0,
                    "target_count": 2,
                    "healthy_target_count": 2,
                },
                "status": [
                    {"name": "all_sleeves", "process_live": True, "heartbeat_ok": True},
                    {"name": "coinbase_loop", "process_live": True, "heartbeat_ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(nightly, "PROJECT_ROOT", fake_project)
    monkeypatch.setattr(nightly, "_pgrep_count", lambda pattern: 1 if "run_all_sleeves" in pattern else 0)
    monkeypatch.setattr(nightly, "_resolve_watchdog_log", lambda: None)
    monkeypatch.setattr(nightly, "_resolve_all_sleeves_log", lambda: tmp_path / "old_all_sleeves.log")
    monkeypatch.setattr(
        nightly,
        "_fresh_minutes",
        lambda path: 0.0 if str(path).endswith("process_watchdog_latest.json") else 999.0,
    )
    monkeypatch.setattr(nightly, "_count_keyword", lambda path, keyword: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_resilience_check.py",
            "--out-file",
            str(tmp_path / "nightly.json"),
            "--event-file",
            str(tmp_path / "nightly.jsonl"),
            "--json",
        ],
    )

    rc = nightly.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["failed_checks"] == []
    assert "all_sleeves_log_stale_process_watchdog_certified" in payload["warnings"]
