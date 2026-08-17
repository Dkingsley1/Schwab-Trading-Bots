import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import process_watchdog as watchdog


def test_soft_music_playback_does_not_suppress_read_only_sleeve_collection() -> None:
    pause = {
        "active": True,
        "creative_session_level": "active",
        "creative_session_kind": "music_playback",
        "hard_pause_terminate_processes": False,
        "hard_pause_action": "lightweight_pause_contract_refresh",
    }

    assert watchdog._creative_pause_suppresses_target("all_sleeves", pause) is False


def test_hard_music_pause_still_suppresses_read_only_sleeve_collection() -> None:
    pause = {
        "active": True,
        "creative_session_level": "active",
        "creative_session_kind": "music_playback_hot",
        "hard_pause_terminate_processes": True,
        "hard_pause_action": "sigterm_optional_heavy_research",
    }

    assert watchdog._creative_pause_suppresses_target("all_sleeves", pause) is True


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


def test_sql_writer_active_progress_resolves_restart_storm_debt() -> None:
    events = [
        {
            "name": "sql_link_writer",
            "event": "restart",
            "ts_epoch": float(100 + index),
        }
        for index in range(5)
    ]
    status_rows = [
        {
            "name": "sql_link_writer",
            "running": 1,
            "heartbeat_ok": True,
            "heartbeat_age_seconds": 15,
            "heartbeat_max_age_seconds": 120,
            "writer_recovered_ok": True,
            "live_execution_critical": False,
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

    assert active == []
    assert recent[0]["resolved"] is True
    assert recent[0]["impact"] == "storage_writer"
    assert recent[0]["blocks_execution_clear"] is False
    assert recent[0]["resolution_reason"] == "sql_writer_active_progress_recovered"


def test_restart_storm_isolation_follows_explicit_execution_clearance_flag() -> None:
    contract = watchdog._restart_storm_isolation_contract(
        [
            {
                "name": "sql_link_writer",
                "quarantinable": False,
                "blocks_execution_clear": False,
            },
            {
                "name": "execution_lane_live",
                "quarantinable": False,
                "blocks_execution_clear": True,
            },
        ]
    )

    assert contract["isolated_targets"] == ["sql_link_writer"]
    assert contract["execution_blocking_targets"] == ["execution_lane_live"]
    assert contract["all_active_storms_isolated"] is False
