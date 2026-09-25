from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo


def extended_equity_session_state(*, session: str, now: datetime) -> dict[str, Any]:
    """Conservative Schwab AM/PM windows, never overnight or queued orders."""
    result: dict[str, Any] = {
        "ready": False,
        "session": session,
        "calendar_id": "XNYS",
        "state": "closed",
        "blocker": "extended_equity_session_closed",
    }
    try:
        import exchange_calendars
        import pandas as pd

        if session not in {"AM", "PM"} or now.tzinfo is None:
            raise ValueError("explicit extended session and timezone required")
        eastern = ZoneInfo("America/New_York")
        local = now.astimezone(eastern)
        day = pd.Timestamp(local.date())
        calendar = exchange_calendars.get_calendar("XNYS")
        if not calendar.is_session(day):
            result["blocker"] = "extended_equity_non_trading_day"
            return result
        opened = calendar.session_open(day).to_pydatetime().astimezone(eastern)
        closed = calendar.session_close(day).to_pydatetime().astimezone(eastern)
        # Do not infer broker-specific extended hours on an exceptional session.
        if opened.time() != time(9, 30) or closed.time() != time(16):
            result["blocker"] = "extended_equity_exceptional_session_not_supported"
            return result
        start, end = (
            (time(7), time(9, 25)) if session == "AM" else (time(16, 5), time(20))
        )
        begin = datetime.combine(local.date(), start, eastern)
        finish = datetime.combine(local.date(), end, eastern)
        remaining = (finish - local).total_seconds()
        ready = begin <= local < finish and remaining >= 75
        result.update(
            ready=ready,
            state="open" if ready else "closed",
            session_label=local.date().isoformat(),
            open_utc=begin.astimezone(timezone.utc).isoformat(),
            close_utc=finish.astimezone(timezone.utc).isoformat(),
            remaining_seconds=max(0, remaining),
            minimum_remaining_seconds=75,
            blocker=(
                ""
                if ready
                else (
                    "extended_equity_closeout_buffer_active"
                    if begin <= local < finish
                    else "extended_equity_session_closed"
                )
            ),
        )
    except Exception as exc:
        result.update(
            state="unknown",
            blocker=f"extended_equity_calendar_unavailable:{type(exc).__name__}",
        )
    return result
