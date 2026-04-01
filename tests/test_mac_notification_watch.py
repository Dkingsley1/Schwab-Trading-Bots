import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts" / "ops"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import mac_notification_watch as watch


def test_power_event_candidates_include_recent_clamshell_sleep(monkeypatch) -> None:
    monkeypatch.setattr(
        watch,
        "_recent_pmset_lines",
        lambda limit=watch.PMSET_POWER_LOG_TAIL_LINES: [
            "2026-03-27 16:26:14 -0400 Sleep                Entering Sleep state due to 'Clamshell Sleep':TCPKeepAlive=active Using Batt (Charge:100%) 5 secs",
            "2026-03-27 17:16:05 -0400 Assertions           PID 358(powerd) Created UserIsActive \"com.apple.powermanagement.lidopen\" 00:00:00  id:0x0x9000092e5 [System: PrevIdle PrevDisp PrevSleep DeclUser kCPU kDisp]",
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
