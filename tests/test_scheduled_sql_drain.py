from datetime import datetime, timedelta, timezone

import pytest

from scripts.ops.scheduled_sql_drain import follow_through

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def check(observation=None, **changes):
    args = dict(
        cycles=1,
        elapsed_seconds=20,
        cycle_seconds=15,
        rows_written=100,
        cycle_ok=True,
        refresh_ok=True,
        interval_seconds=120,
        now=NOW,
    )
    args.update(changes)
    return follow_through(
        observation
        or dict(
            timestamp_utc=NOW.isoformat(),
            pending_lines=3000,
            pending_lines_total=4000,
            oldest_pending_age_seconds=90,
        ),
        **args
    )


def test_useful_scheduled_writer_follows_through_without_waiting_for_launchd():
    result = check()
    assert result["continue"] is True
    assert result["delay_seconds"] == 15


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"rows_written": 0}, "no_measured_progress"),
        ({"cycle_ok": False}, "cycle_incomplete"),
        ({"refresh_ok": False}, "observation_incomplete"),
        ({"cycles": 8}, "bounded_window_complete"),
        ({"elapsed_seconds": 150}, "bounded_window_complete"),
        ({"cycle_seconds": 120}, "bounded_window_complete"),
    ],
)
def test_follow_through_is_bounded_and_requires_progress(changes, reason):
    result = check(**changes)
    assert result["continue"] is False
    assert result["reason"] == reason


@pytest.mark.parametrize(
    "key,value",
    [
        ("pending_lines", True),
        ("pending_lines", "3000"),
        ("pending_lines_total", 1),
        ("oldest_pending_age_seconds", float("nan")),
        ("timestamp_utc", (NOW - timedelta(seconds=31)).isoformat()),
        ("timestamp_utc", (NOW + timedelta(seconds=1)).isoformat()),
        ("timestamp_utc", "2026-09-15T00:00:00"),
    ],
)
def test_unknown_or_stale_observation_cannot_drive_more_work(key, value):
    observation = dict(
        timestamp_utc=NOW.isoformat(),
        pending_lines=3000,
        pending_lines_total=4000,
        oldest_pending_age_seconds=90,
    )
    observation[key] = value
    result = check(observation)
    assert not result["continue"]
    assert result["reason"] == "observation_incomplete"


def test_near_empty_target_stops_but_small_aged_hot_tail_gets_service():
    observation = dict(
        timestamp_utc=NOW.isoformat(),
        pending_lines=20,
        pending_lines_total=50,
        oldest_pending_age_seconds=10,
    )
    assert check(observation)["reason"] == "near_empty_target"
    observation["oldest_pending_age_seconds"] = 61
    assert check(observation)["continue"]
