"""Bounded follow-through policy for the existing scheduled SQL writer."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def follow_through(
    observation: dict[str, Any],
    *,
    cycles: int,
    elapsed_seconds: float,
    cycle_seconds: float,
    rows_written: int,
    cycle_ok: bool,
    refresh_ok: bool,
    interval_seconds: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Keep one admitted writer useful without turning launchd into a daemon."""
    now = now or datetime.now(timezone.utc)
    delay = min(max(float(interval_seconds), 5.0), 15.0)
    result: dict[str, Any] = {
        "continue": False,
        "reason": "cycle_incomplete",
        "cycles_completed": cycles,
        "max_cycles": 8,
        "admission_window_seconds": 180,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "delay_seconds": delay,
        "observation_timestamp_utc": observation.get("timestamp_utc"),
    }
    if not cycle_ok:
        return result
    if rows_written <= 0:
        result["reason"] = "no_measured_progress"
        return result
    if cycles >= 8 or elapsed_seconds + delay + max(cycle_seconds * 1.5, 30) >= 180:
        result["reason"] = "bounded_window_complete"
        return result
    try:
        stamp = datetime.fromisoformat(
            str(observation["timestamp_utc"]).replace("Z", "+00:00")
        )
        age = (now - stamp).total_seconds()
        core = observation["pending_lines"]
        total = observation["pending_lines_total"]
        oldest = observation["oldest_pending_age_seconds"]
        valid = (
            refresh_ok
            and 0 <= age <= 30
            and type(core) is int
            and type(total) is int
            and 0 <= core <= total
            and type(oldest) in (int, float)
            and math.isfinite(oldest)
            and oldest >= 0
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        result["reason"] = "observation_incomplete"
        return result
    result.update(
        core_pending_lines=core,
        total_pending_lines=total,
        oldest_pending_age_seconds=oldest,
    )
    pending = core > 1000 or total > 2500 or (core > 0 and oldest > 60)
    result["continue"] = pending
    result["reason"] = "fresh_debt_after_progress" if pending else "near_empty_target"
    return result
