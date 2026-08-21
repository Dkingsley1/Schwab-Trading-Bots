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


def test_recent_pmset_lines_handles_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(watch, "_PMSET_POWER_LOG_CACHE", None)
    cache_path = tmp_path / "pmset_cache.json"
    monkeypatch.setattr(watch, "DEFAULT_PMSET_CACHE_PATH", cache_path)

    def _timeout(*args, **kwargs):
        raise watch.subprocess.TimeoutExpired(cmd="pmset", timeout=1.0)

    monkeypatch.setattr(watch.subprocess, "run", _timeout)

    assert watch._recent_pmset_lines() == []
    assert json.loads(cache_path.read_text(encoding="utf-8"))["lines"] == []


def test_recent_pmset_lines_uses_short_cache(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(watch, "_PMSET_POWER_LOG_CACHE", None)
    monkeypatch.setattr(watch, "DEFAULT_PMSET_CACHE_PATH", tmp_path / "pmset_cache.json")
    monkeypatch.setenv(watch.PMSET_POWER_LOG_CACHE_SECONDS_ENV, "90")
    monkeypatch.setenv(watch.PMSET_SKIP_UNDER_PRESSURE_ENV, "0")
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
    assert calls[0][0] == ["/usr/bin/pmset", "-g", "log"]


def test_recent_pmset_lines_uses_disk_cache_across_runs(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "pmset_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "lines": ["cached-a", "cached-b"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "_PMSET_POWER_LOG_CACHE", None)
    monkeypatch.setattr(watch, "DEFAULT_PMSET_CACHE_PATH", cache_path)
    monkeypatch.setenv(watch.PMSET_POWER_LOG_CACHE_SECONDS_ENV, "90")

    def _run(*args, **kwargs):
        raise AssertionError("pmset should not run when disk cache is fresh")

    monkeypatch.setattr(watch.subprocess, "run", _run)

    assert watch._recent_pmset_lines() == ["cached-a", "cached-b"]


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


def test_notification_allowlist_honors_reason_aliases() -> None:
    allowlist = watch._parse_event_allowlist("tripwire,storage_critical,health_gate_critical,auth_expired")

    assert watch._notification_event_allowed("tripwire:all_sleeves", allowlist) is True
    assert watch._notification_event_allowed("storage_mount_missing", allowlist) is True
    assert watch._notification_event_allowed("critical_alert:critical:critical_latest_default_crypto_coinbase", allowlist) is True
    assert watch._notification_event_allowed("auth_lease:critical:interactive_refresh_required", allowlist) is True
    assert watch._notification_event_allowed("creative_mode:creative_mode_active:music", allowlist) is False


def test_auth_lease_event_surfaces_interactive_schwab_refresh() -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    event = watch._auth_lease_event(
        {
            "timestamp_utc": stamp,
            "overall_status": "blocked",
            "lease_state": "critical",
            "broker_state": {
                "auth_reason": "OAuthError: invalid_grant: Refresh token is invalid, expired or revoked",
            },
        },
        {
            "timestamp_utc": stamp,
            "overall_status": "blocked",
            "token": {"ready": False},
            "findings": ["token_not_ready:token_expired"],
            "operator_followups": ["./scripts/ops/opsctl.sh token-refresh-interactive --force --json"],
        },
        900.0,
    )

    assert event is not None
    key, message = event
    assert key == "auth_lease:critical:interactive_refresh_required"
    assert "Schwab sign-in is required" in message
    assert "Paper execution and broker reconciliation are paused" in message
    assert watch._event_severity(key, message) == "critical"
    assert watch._notification_heading(key, message) == ("Trading Bot Critical", "Schwab Authorization")
    assert "token-refresh-interactive" in watch._notification_body(key, message)


def test_auth_lease_event_surfaces_warning_without_claiming_paper_is_paused() -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    event = watch._auth_lease_event(
        {
            "timestamp_utc": stamp,
            "overall_status": "degraded",
            "lease_state": "warning",
        },
        {"timestamp_utc": stamp, "overall_status": "ready", "token": {"ready": True}},
        900.0,
    )

    assert event == (
        "auth_lease:warn:lease_warning",
        "Schwab authorization is nearing its critical lease floor.\nPaper collection remains active.",
    )
    assert watch._event_severity(event[0], event[1]) == "warn"
    assert watch._notification_heading(event[0], event[1]) == ("Trading Bot Warning", "Schwab Authorization")


def test_auth_lease_event_clears_when_current_contract_is_ready() -> None:
    stamp = datetime.now(timezone.utc).isoformat()

    assert watch._auth_lease_event(
        {"timestamp_utc": stamp, "overall_status": "ready", "lease_state": "healthy"},
        {"timestamp_utc": stamp, "overall_status": "ready", "token": {"ready": True}},
        900.0,
    ) is None


def test_auth_lease_event_ignores_stale_blocked_supervisor_when_lease_is_current() -> None:
    now = datetime.now(timezone.utc)

    assert watch._auth_lease_event(
        {"timestamp_utc": now.isoformat(), "overall_status": "ready", "lease_state": "healthy"},
        {
            "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
            "overall_status": "blocked",
            "token": {"ready": False},
            "operator_followups": ["./scripts/ops/opsctl.sh token-refresh-interactive --force --json"],
        },
        900.0,
    ) is None


def test_auth_notification_repeat_floor_defaults_to_thirty_minutes(monkeypatch) -> None:
    monkeypatch.delenv(watch.AUTH_MIN_REPEAT_SECONDS_ENV, raising=False)

    assert watch._event_repeat_seconds("auth_lease:critical:blocked", 300.0) == 1800.0
    assert watch._event_repeat_seconds("tripwire:all_sleeves", 300.0) == 300.0


def test_critical_alert_events_suppress_training_done(monkeypatch, tmp_path: Path) -> None:
    alerts = tmp_path / "alerts"
    alerts.mkdir()
    (alerts / "critical_latest_training.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "severity": "critical",
                "event": "retrain_finished",
                "message": "Training done: status=completed_successfully",
            }
        ),
        encoding="utf-8",
    )
    (alerts / "critical_latest_guardrail.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "severity": "critical",
                "event": "lane_kill_switch_engaged",
                "message": "lane=futures cooldown=220s",
                "profile": "default",
                "broker": "coinbase",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALERTS_DIR", alerts)
    monkeypatch.setenv(watch.SUPPRESS_TRAINING_DONE_ENV, "1")

    events = watch._critical_alert_events(900.0)

    assert len(events) == 1
    assert "Lane Kill Switch" in events[0][1]


def test_critical_alert_events_suppress_expired_lane_cooldown(monkeypatch, tmp_path: Path) -> None:
    alerts = tmp_path / "alerts"
    alerts.mkdir()
    now = datetime.now(timezone.utc)
    (alerts / "critical_latest_guardrail.json").write_text(
        json.dumps(
            {
                "timestamp_utc": (now - timedelta(seconds=240)).isoformat(),
                "severity": "critical",
                "event": "lane_kill_switch_engaged",
                "message": "lane=futures cooldown=220s",
                "profile": "default",
                "broker": "coinbase",
                "details": {"lane": "futures", "cooldown_seconds": 220},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALERTS_DIR", alerts)

    assert watch._critical_alert_events(900.0) == []


def test_critical_alert_events_keep_active_lane_cooldown(monkeypatch, tmp_path: Path) -> None:
    alerts = tmp_path / "alerts"
    alerts.mkdir()
    now = datetime.now(timezone.utc)
    (alerts / "critical_latest_guardrail.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "severity": "critical",
                "event": "lane_kill_switch_engaged",
                "message": "lane=futures cooldown=220s",
                "profile": "default",
                "broker": "coinbase",
                "details": {
                    "lane": "futures",
                    "cooldown_seconds": 220,
                    "cooldown_until_utc": (now + timedelta(seconds=120)).isoformat(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "ALERTS_DIR", alerts)

    events = watch._critical_alert_events(900.0)

    assert len(events) == 1
    assert "Lane Kill Switch" in events[0][1]


def test_notify_attempts_imessage_when_enabled_and_severity_matches(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(watch, "_notify_mac", lambda *args, **kwargs: {"channel": "mac", "returncode": 0})

    def fake_imessage(title: str, body: str, recipient: str) -> dict:
        calls.append((title, body, recipient))
        return {"channel": "imessage", "recipient": recipient, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(watch, "_notify_imessage", fake_imessage)

    delivery = watch._notify(
        "Trading Bot Critical",
        "Tripwire active",
        imessage_enabled=True,
        imessage_recipient="dan@example.com",
        imessage_min_severity="critical",
        severity="critical",
    )

    assert delivery["imessage_attempted"] is True
    assert delivery["imessage"]["returncode"] == 0
    assert calls == [("Trading Bot Critical", "Tripwire active", "dan@example.com")]


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


def test_storage_event_ignores_stale_unavailable_mount() -> None:
    event = watch._storage_event(
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "external_available": False,
            "mount_root": "/Volumes/BOT_LOGS",
        },
        900.0,
    )

    assert event is None


def test_storage_event_surfaces_recent_unavailable_mount() -> None:
    event = watch._storage_event(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "external_available": False,
            "external_unavailable_reason": "low_space",
            "mount_root": "/Volumes/BOT_LOGS",
        },
        900.0,
    )

    assert event == ("storage_mount_missing", "Storage route unavailable: /Volumes/BOT_LOGS (external low space)")
