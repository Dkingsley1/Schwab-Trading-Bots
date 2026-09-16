"""Fresh hot-lane pressure evidence for retrying a failed daily check."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

MAX_OBSERVATION_AGE_SECONDS = 300


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else None


def ingestion_observation_ready(
    payload: Any, *, after_utc: Any, now: datetime | None = None
) -> bool:
    if not isinstance(payload, dict):
        return False
    observed = _timestamp(payload.get("timestamp_utc"))
    after = _timestamp(after_utc)
    current = now or datetime.now(timezone.utc)
    if observed is None or after is None or current.tzinfo is None:
        return False
    if not after < observed <= current:
        return False
    if (current - observed).total_seconds() > MAX_OBSERVATION_AGE_SECONDS:
        return False
    for name in ("overload", "line_pressure", "age_pressure", "ema_pressure"):
        if payload.get(name) is not False:
            return False
    if (
        type(payload.get("file_pressure")) is not bool
        or type(payload.get("trend_up")) is not bool
    ):
        return False
    if payload["file_pressure"] and payload["trend_up"]:
        return False
    for name in ("pending_lines", "pending_files", "files_scanned"):
        if type(payload.get(name)) is not int or payload[name] < 0:
            return False
    for name in (
        "pending_lines_threshold",
        "pending_files_threshold",
        "oldest_age_threshold_seconds",
    ):
        if type(payload.get(name)) is not int or payload[name] <= 0:
            return False
    for name in ("oldest_pending_age_seconds", "ema_pending_lines"):
        value = payload.get(name)
        if type(value) not in (int, float) or value < 0:
            return False
        try:
            if not math.isfinite(value):
                return False
        except OverflowError:
            return False
    if payload["pending_lines"] >= payload["pending_lines_threshold"]:
        return False
    selection = payload.get("scan_selection")
    if not isinstance(selection, dict):
        return False
    for name in ("selected_files", "discovered_relevant_files", "max_files"):
        if type(selection.get(name)) is not int or selection[name] <= 0:
            return False
    # This is the owner's bounded hot-lane selection, not full archive coverage.
    return bool(
        payload["files_scanned"]
        == selection["selected_files"]
        <= min(selection["discovered_relevant_files"], selection["max_files"])
    )
