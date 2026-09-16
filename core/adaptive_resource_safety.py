"""Read-only headroom projections; may tighten existing limits, never widen them."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _timestamp(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return result.astimezone(timezone.utc) if result.tzinfo is not None else None


def adaptive_headroom_guard(
    resource: dict[str, Any],
    previous: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    source = _timestamp(
        resource.get("source_timestamp_utc", resource.get("timestamp_utc"))
    )
    producer = _timestamp(resource.get("timestamp_utc"))
    fresh = bool(
        source is not None
        and producer is not None
        and 0 <= (now - source).total_seconds() <= 120
        and 0 <= (now - producer).total_seconds() <= 120
        and resource.get("input_evidence_ready") is not False
    )
    thresholds = resource.get("memory_pressure_thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    prior_source = _timestamp(previous.get("source_timestamp_utc"))
    elapsed = (
        (source - prior_source).total_seconds() if source and prior_source else None
    )
    prior_valid = bool(
        fresh
        and previous.get("input_evidence_ready") is True
        and elapsed is not None
        and 0 <= elapsed <= 120
    )
    prior_metrics = previous.get("metrics")
    prior_metrics = prior_metrics if isinstance(prior_metrics, dict) else {}
    metrics = {}
    triggers = []
    invalid_metrics = []
    for key, threshold_key, minimum_drop, unit_max in (
        ("memory_available_pct", "yellow_available_pct", 5.0, 100.0),
        ("memory_free_pct", "yellow_free_pct", 5.0, 100.0),
        ("local_disk_free_gb", "yellow_local_disk_gb", 2.0, None),
    ):
        value, floor = _number(resource.get(key)), _number(
            thresholds.get(threshold_key)
        )
        if (key in resource and value is None) or (
            threshold_key in thresholds and floor is None
        ):
            invalid_metrics.append(key)
        if unit_max and (
            (value is not None and value > unit_max)
            or (floor is not None and floor > unit_max)
        ):
            invalid_metrics.append(key)
        if value is None or floor is None or key in invalid_metrics:
            continue
        prior_row = prior_metrics.get(key)
        prior_row = prior_row if isinstance(prior_row, dict) else {}
        prior_value = _number(prior_row.get("value"))
        rate = None
        # Source time, not publication time, defines an independent measurement.
        if prior_valid and 5 <= elapsed <= 120 and prior_value is not None:
            rate = (value - prior_value) / elapsed
        projected = value + min(rate or 0.0, 0.0) * 40
        forecast = bool(
            rate is not None
            and prior_value - value >= minimum_drop
            and value > floor
            and projected < floor
        )
        breached = value < floor
        metrics[key] = {
            "value": value,
            "warning_floor": floor,
            "rate_per_second": round(rate, 6) if rate is not None else None,
            "projected_40_seconds": round(max(projected, 0), 3),
            "forecast_crossing": forecast,
        }
        if fresh and (breached or forecast):
            triggers.append(
                f"{key}:{'warning_floor' if breached else 'falling_headroom'}"
            )
    if invalid_metrics:
        fresh = False
    prior_active = previous.get("active") is True
    clear_since = (
        _timestamp(previous.get("clear_since_source_timestamp_utc"))
        if prior_active
        else None
    )
    if clear_since and not (prior_source and clear_since <= prior_source <= source):
        clear_since = None
    if not fresh or triggers:
        clear_since = None
    elif prior_active and clear_since is None:
        clear_since = source
    recovering = bool(
        fresh
        and not triggers
        and prior_active
        and clear_since is not None
        and (source - clear_since).total_seconds() < 60
    )
    active = bool(not fresh or triggers or recovering)
    return {
        "input_evidence_ready": fresh,
        "source_timestamp_utc": source.isoformat() if source else None,
        "state": (
            "unavailable"
            if not fresh
            else "guarded" if triggers else "recovering" if recovering else "clear"
        ),
        "active": active,
        "minimum_memory_pressure_level": (
            "high" if not fresh else "elevated" if active else "normal"
        ),
        "clear_since_source_timestamp_utc": (
            clear_since.isoformat() if clear_since else None
        ),
        "projection_horizon_seconds": 40,
        "recovery_dwell_seconds": 60,
        "metrics": metrics,
        "invalid_metrics": invalid_metrics,
        "reasons": triggers or (["headroom_recovery_dwell"] if recovering else []),
        "authority": "tighten_existing_resource_controls_only",
    }
