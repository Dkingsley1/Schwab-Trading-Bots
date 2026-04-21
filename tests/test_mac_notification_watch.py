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


def test_notification_body_adds_action_hint_for_tripwire() -> None:
    key = "tripwire:all_sleeves"
    body = watch._notification_body(key, "Tripwire triggered for all_sleeves")

    assert "Tripwire triggered for all_sleeves" in body
    assert "Action: keep live halted and inspect the tripwire incidents." in body


def test_notification_body_adds_action_hint_for_guardrail_warning() -> None:
    key = "critical_alert:warn:latest_default"
    message = "Margin Guard [Aggressive / Schwab]\nPosition limit reached"

    body = watch._notification_body(key, message)

    assert "Margin Guard [Aggressive / Schwab]" in body
    assert "Position limit reached" in body
    assert "Action: review the guardrail and keep the lane constrained." in body
