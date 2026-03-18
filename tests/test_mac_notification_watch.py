from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import scripts.ops.mac_notification_watch as watch


class _Result:
    returncode = 0
    stdout = ""
    stderr = ""


def test_notify_sends_imessage_when_enabled(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _Result()

    monkeypatch.setattr(watch.subprocess, "run", fake_run)

    result = watch._notify(
        "Trading Bot Incident",
        "Tripwire triggered for schwab_parallel",
        imessage_enabled=True,
        imessage_recipient="me@example.com",
    )

    assert result["mac"]["returncode"] == 0
    assert result["imessage_attempted"] is True
    assert result["imessage"]["recipient"] == "me@example.com"
    assert len(calls) == 2
    assert calls[0][0][:2] == ["osascript", "-e"]
    assert calls[1][0] == [
        "osascript",
        "-",
        "me@example.com",
        "Trading Bot Incident\nTripwire triggered for schwab_parallel",
    ]
    assert 'tell application "Messages"' in calls[1][1]["input"]
    assert calls[1][1]["text"] is True


def test_notify_skips_imessage_without_recipient(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _Result()

    monkeypatch.setattr(watch.subprocess, "run", fake_run)

    result = watch._notify(
        "Trading Bot Incident",
        "Restart storm detected: coinbase_loop",
        imessage_enabled=True,
        imessage_recipient="",
    )

    assert result["imessage_attempted"] is False
    assert result["imessage"] is None
    assert len(calls) == 1
    assert calls[0][0][:2] == ["osascript", "-e"]


def test_notify_suppresses_imessage_below_min_severity(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _Result()

    monkeypatch.setattr(watch.subprocess, "run", fake_run)

    result = watch._notify(
        "Trading Bot Warning",
        "Margin Guard [Default / Coinbase]\nfutures_margin_headroom_insufficient",
        imessage_enabled=True,
        imessage_recipient="me@example.com",
        imessage_min_severity="critical",
        severity="warn",
    )

    assert result["imessage_attempted"] is False
    assert result["imessage"] is None
    assert len(calls) == 1
    assert calls[0][0][:2] == ["osascript", "-e"]


def test_imessage_event_allowlist_blocks_tripwire_but_keeps_requested_events() -> None:
    allowlist = watch._parse_imessage_event_allowlist(
        "all_sleeves_down,global_halt,incident_auto_halt,preflight_critical,storage_mount_missing,critical_alert"
    )

    assert watch._imessage_event_allowed("tripwire:schwab_parallel", allowlist) is False
    assert watch._imessage_event_allowed("restart_storm:all_sleeves", allowlist) is False
    assert watch._imessage_event_allowed("all_sleeves_down", allowlist) is True
    assert watch._imessage_event_allowed("storage_mount_missing", allowlist) is True
    assert watch._imessage_event_allowed(
        "critical_alert:critical:critical_latest_intraday_aggressive_equities_schwab",
        allowlist,
    ) is True


def test_notify_mac_escapes_quotes(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _Result()

    monkeypatch.setattr(watch.subprocess, "run", fake_run)

    watch._notify_mac('Trading "Bot" Incident', 'Tripwire active: "schwab_parallel"')

    assert len(calls) == 1
    assert '\\"Bot\\"' in calls[0][0][2]
    assert '\\"schwab_parallel\\"' in calls[0][0][2]


def test_incident_auto_halt_event_reports_recent_failure() -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "halt": False,
        "failed_checks": ["daily_verify_not_ok", "promotion_quality_gate_not_ok"],
    }

    event = watch._incident_auto_halt_event(payload, 900.0)

    assert event == (
        "incident_auto_halt",
        "Incident auto-halt failed\nChecks: daily_verify_not_ok,promotion_quality_gate_not_ok",
    )


def test_preflight_critical_event_ignores_cleared_payload() -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "broker": "schwab",
        "failed_checks": [],
        "cleared": True,
    }

    assert watch._preflight_critical_event(payload, 900.0) is None


def test_critical_alert_events_include_recent_critical_file(tmp_path, monkeypatch) -> None:
    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "severity": "critical",
        "event": "lane_kill_switch_engaged",
        "message": "lane=day cooldown=240s",
        "profile": "aggressive",
        "broker": "schwab",
    }
    (alerts_dir / "critical_latest_aggressive_equities_schwab.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALERTS_DIR", alerts_dir)

    events = watch._critical_alert_events(900.0)

    assert events == [
        (
            "critical_alert:critical:critical_latest_aggressive_equities_schwab",
            "Lane Kill Switch [Aggressive / Schwab]\nlane=day cooldown=240s",
        )
    ]


def test_event_candidates_ignore_stale_alert_payloads(monkeypatch) -> None:
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    monkeypatch.setattr(watch, "ALERTS_DIR", Path("/nonexistent"))
    monkeypatch.setattr(
        watch,
        "_read_json",
        lambda path: {"timestamp_utc": stale_ts, "failed_checks": ["disk_free"], "broker": "schwab"}
        if path == watch.PREFLIGHT_CRITICAL_PATH
        else {},
    )

    events = watch._event_candidates(30.0)

    assert events == []


def test_notify_imessage_captures_applescript_error(monkeypatch) -> None:
    class _ErrorResult:
        returncode = 1
        stdout = ""
        stderr = "Messages got an error"

    monkeypatch.setattr(watch.subprocess, "run", lambda *args, **kwargs: _ErrorResult())

    result = watch._notify_imessage("Trading Bot Incident", "Tripwire active", "me@example.com")

    assert result == {
        "channel": "imessage",
        "recipient": "me@example.com",
        "returncode": 1,
        "stdout": "",
        "stderr": "Messages got an error",
    }


def test_notification_heading_distinguishes_warning_and_critical() -> None:
    assert watch._notification_heading(
        "critical_alert:warn:critical_latest_default_crypto_coinbase",
        "Futures Margin Guard [Default / Coinbase]\nfutures_margin_headroom_insufficient",
    ) == ("Trading Bot Warning", "Guardrail Warning")
    assert watch._notification_heading(
        "critical_alert:critical:critical_latest_intraday_aggressive_equities_schwab",
        "Lane Kill Switch [Intraday Aggressive / Schwab]\nlane=day cooldown=240s",
    ) == ("Trading Bot Critical", "Critical Guardrail")

def test_storage_route_is_treated_as_critical() -> None:
    assert watch._event_severity("storage_mount_missing", "Storage route unavailable: /Volumes/BOT_LOGS") == "critical"
    assert watch._notification_heading(
        "storage_mount_missing",
        "Storage route unavailable: /Volumes/BOT_LOGS",
    ) == ("Trading Bot Critical", "Storage Route")

def test_parse_imessage_event_allowlist_supports_all_token() -> None:
    assert watch._parse_imessage_event_allowlist("all") == {"*"}





