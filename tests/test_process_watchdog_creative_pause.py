import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import process_watchdog as watchdog


def test_creative_pause_resolves_and_forgives_coinbase_restart_debt() -> None:
    events = [
        {
            "name": "coinbase_loop",
            "event": "restart",
            "ts_epoch": float(100 + index),
        }
        for index in range(6)
    ]
    status_rows = [
        {
            "name": "coinbase_loop",
            "running": 0,
            "heartbeat_ok": False,
            "paused_by_creative_cotenant_guard": True,
            "creative_pause_reason": "music_playback",
        }
    ]

    active, recent = watchdog._resolved_restart_storms(
        events=events,
        status_rows=status_rows,
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=900,
        now_epoch=1000.0,
    )
    kept, forgiveness = watchdog._forgive_resolved_restart_debt(events, recent)

    assert active == []
    assert recent[0]["resolved"] is True
    assert recent[0]["resolution_reason"] == "music_playback"
    assert kept == []
    assert forgiveness["removed_event_count"] == 6
