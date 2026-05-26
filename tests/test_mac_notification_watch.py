import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts" / "ops"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import mac_notification_watch as watch


def test_power_event_candidates_include_recent_clamshell_sleep(monkeypatch) -> None:
    now = datetime.now(timezone.utc).astimezone()
    sleep_stamp = now.strftime("%Y-%m-%d %H:%M:%S %z")
    open_stamp = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S %z")
    monkeypatch.setattr(
        watch,
        "_recent_pmset_lines",
        lambda limit=watch.PMSET_POWER_LOG_TAIL_LINES: [
            f"{sleep_stamp} Sleep                Entering Sleep state due to 'Clamshell Sleep':TCPKeepAlive=active Using Batt (Charge:100%) 5 secs",
            f"{open_stamp} Assertions           PID 358(powerd) Created UserIsActive \"com.apple.powermanagement.lidopen\" 00:00:00  id:0x0x9000092e5 [System: PrevIdle PrevDisp PrevSleep DeclUser kCPU kDisp]",
        ],
    )
    candidates = watch._power_event_candidates(24 * 60 * 60)

    assert any(key.startswith("power_clamshell_sleep:") for key, _ in candidates)
    assert any("MacBook lid closed" in message for _, message in candidates)
    assert any(key.startswith("power_lid_open:") for key, _ in candidates)


def test_power_event_severity_and_heading() -> None:
    close_key = "power_clamshell_sleep:2026-03-27T20:26:14+00:00"
    open_key = "power_lid_open:2026-03-27T21:16:05+00:00"

    assert watch._event_severity(close_key, "") == "critical"
    assert watch._event_severity(open_key, "") == "info"
    assert watch._notification_heading(close_key, "") == ("Trading Bot Critical", "Laptop Closed")
    assert watch._notification_heading(open_key, "") == ("Trading Bot Incident", "Laptop Opened")


def test_recent_pmset_lines_handles_timeout(monkeypatch) -> None:
    monkeypatch.setattr(watch, "_PMSET_POWER_LOG_CACHE", None)

    def _timeout(*args, **kwargs):
        raise watch.subprocess.TimeoutExpired(cmd="pmset", timeout=1.0)

    monkeypatch.setattr(watch.subprocess, "run", _timeout)

    assert watch._recent_pmset_lines() == []


def test_recent_pmset_lines_uses_short_cache(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(watch, "_PMSET_POWER_LOG_CACHE", None)
    monkeypatch.setenv(watch.PMSET_POWER_LOG_CACHE_SECONDS_ENV, "90")
    monkeypatch.setattr(watch.time, "monotonic", lambda: 100.0 + len(calls))

    class _Result:
        returncode = 0
        stdout = "2026-05-22 08:00:00 -0400 Sleep                Entering Sleep state due to 'Clamshell Sleep'\n"

    def _run(*args, **kwargs):
        calls.append(args)
        return _Result()

    monkeypatch.setattr(watch.subprocess, "run", _run)

    first = watch._recent_pmset_lines()
    second = watch._recent_pmset_lines()

    assert first == second
    assert len(calls) == 1


def test_recent_pmset_lines_returns_cached_lines_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(watch, "_PMSET_POWER_LOG_CACHE", (10.0, ["cached"]))
    monkeypatch.setattr(watch.time, "monotonic", lambda: 200.0)

    def _timeout(*args, **kwargs):
        raise watch.subprocess.TimeoutExpired(cmd="pmset", timeout=1.0)

    monkeypatch.setattr(watch.subprocess, "run", _timeout)

    assert watch._recent_pmset_lines() == ["cached"]


def test_notification_body_adds_action_hint_for_tripwire() -> None:
    key = "tripwire:all_sleeves"
    body = watch._notification_body(key, "Tripwire triggered for all_sleeves")

    assert "Tripwire triggered for all_sleeves" in body
    assert "Inspect: incident_timeline_latest.json" in body
    assert "Action: keep live halted and inspect the tripwire incidents." in body


def test_all_sleeves_down_suppressed_when_launcher_is_recently_starting(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "overall_status": "degraded",
                "phase": "starting",
                "launcher_pid": 12345,
                "running_job_count": 3,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALL_SLEEVES_LAUNCHER_PATH", launcher)

    event = watch._all_sleeves_down_event(
        {"status": [{"name": "all_sleeves", "running": 0, "heartbeat_ok": False, "alt_running": 0}]},
        900.0,
    )

    assert event is None


def test_all_sleeves_down_suppressed_during_watchdog_restart_handoff(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat(),
                "overall_status": "stopped",
                "phase": "stopped",
                "launcher_pid": 0,
                "running_job_count": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALL_SLEEVES_LAUNCHER_PATH", launcher)

    event = watch._all_sleeves_down_event(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": [
                {
                    "name": "all_sleeves",
                    "running": 0,
                    "heartbeat_ok": False,
                    "alt_running": 0,
                    "restarted_pid": 12345,
                    "restart_reason": "process_missing",
                }
            ],
        },
        900.0,
    )

    assert event is None


def test_all_sleeves_down_suppressed_when_fanout_hold_intentionally_blocks_restart(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(watch, "ALL_SLEEVES_LAUNCHER_PATH", launcher)

    event = watch._all_sleeves_down_event(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": [
                {
                    "name": "all_sleeves",
                    "running": 0,
                    "heartbeat_ok": False,
                    "alt_running": 0,
                    "restart_skipped": "startup_not_ready",
                    "reason": "process_fanout_guard_active",
                }
            ],
        },
        900.0,
    )

    assert event is None


def test_all_sleeves_down_suppressed_when_creative_pause_is_intentional(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(watch, "ALL_SLEEVES_LAUNCHER_PATH", launcher)

    event = watch._all_sleeves_down_event(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "creative_cotenant_pause": {
                "active": True,
                "reason": "music_playback",
                "creative_session_kind": "music_playback",
                "creative_session_level": "active",
            },
            "watchdog_intelligence": {
                "notification_policy": {"suppress_intentional_holds": True},
                "exact_needs": [
                    {
                        "target": "all_sleeves",
                        "status": "intentional_hold",
                        "blocker": "music_playback",
                    }
                ],
            },
            "status": [
                {
                    "name": "all_sleeves",
                    "running": 0,
                    "heartbeat_ok": False,
                    "alt_running": 0,
                    "paused_by_creative_cotenant_guard": True,
                    "restart_skipped": "creative_cotenant_pause_active",
                    "reason": "music_playback",
                }
            ],
        },
        900.0,
    )

    assert event is None


def test_all_sleeves_restart_storm_suppressed_while_launcher_is_recovering(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "overall_status": "degraded",
                "phase": "starting",
                "launcher_pid": 12345,
                "running_job_count": 9,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALL_SLEEVES_LAUNCHER_PATH", launcher)

    event = watch._restart_storm_event({"restart_storms": [{"name": "all_sleeves"}]}, 900.0)

    assert event is None


def test_notification_body_adds_action_hint_for_guardrail_warning() -> None:
    key = "critical_alert:warn:latest_default"
    message = "Margin Guard [Aggressive / Schwab]\nPosition limit reached"

    body = watch._notification_body(key, message)

    assert "Margin Guard [Aggressive / Schwab]" in body
    assert "Position limit reached" in body
    assert "Inspect:" in body
    assert "Action: review the guardrail and keep the lane constrained." in body


def test_notification_group_key_compacts_critical_alert_variants() -> None:
    message = "Margin Guard [Aggressive / Schwab]\nPosition limit reached"
    key_a = "critical_alert:warn:latest_default"
    key_b = "critical_alert:warn:latest_other"

    assert watch._notification_group_key(key_a, message) == watch._notification_group_key(key_b, message)


def test_global_halt_clear_event_surfaces_recent_auto_clear(monkeypatch, tmp_path: Path) -> None:
    halt_recovery = tmp_path / "shadow_watchdog_halt_recovery_latest.json"
    halt_recovery.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "action": "halt_auto_cleared",
                "halt_reason": "incident_auto_halt",
                "decision_reason": "eligible",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "HALT_RECOVERY_PATH", halt_recovery)

    event = watch._global_halt_clear_event(watch._read_json(halt_recovery), 900.0)

    assert event is not None
    key, message = event
    assert key == "global_halt_cleared"
    assert "cleared automatically" in message
    assert watch._event_severity(key, message) == "info"
    assert watch._notification_heading(key, message) == ("Trading Bot Incident", "Global Halt Cleared")


def test_incident_auto_halt_clear_event_is_informational() -> None:
    event = watch._incident_auto_halt_event(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "halt_cleared",
            "ok": True,
            "halt": False,
            "clear_streak": 3,
        },
        900.0,
    )

    assert event == ("incident_auto_halt_cleared", "Incident auto-halt cleared itself\nClear streak: 3")
    assert watch._event_severity(event[0], event[1]) == "info"


def test_halt_clear_events_share_imessage_allowlist_family() -> None:
    allowlist = watch._parse_imessage_event_allowlist("global_halt,incident_auto_halt")

    assert watch._imessage_event_allowed("global_halt_cleared", allowlist) is True
    assert watch._imessage_event_allowed("incident_auto_halt_cleared", allowlist) is True


def test_swap_pressure_event_surfaces_restart_advisory() -> None:
    event = watch._swap_pressure_event(
        {
            "notification": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "swap_pressure_restart_advisory",
                "current_tier": "pause_research",
                "message": "Swap pressure is pause_research at 18.4 GB; restart PyCharm when convenient.",
            }
        },
        900.0,
    )

    assert event is not None
    key, message = event
    assert key == "swap_pressure:swap_pressure_restart_advisory:pause_research"
    assert "restart PyCharm" in message
    assert watch._event_severity(key, message) == "warn"
    assert watch._notification_heading(key, message) == ("Trading Bot Warning", "Swap Pressure")
