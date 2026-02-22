import time

from scripts.shadow_watchdog import Target, _can_restart


def test_watchdog_restart_rate_limit() -> None:
    t = Target(name="x", match="x", start_cmd="echo hi")
    now = time.time()

    assert _can_restart(t, now, max_restarts=2, window_seconds=60)
    t.restart_times.append(now)
    assert _can_restart(t, now, max_restarts=2, window_seconds=60)
    t.restart_times.append(now)
    assert not _can_restart(t, now, max_restarts=2, window_seconds=60)
