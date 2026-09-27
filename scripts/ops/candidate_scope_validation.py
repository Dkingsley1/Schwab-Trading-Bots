#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_POLICY_RELATIVE_PATH = "config/candidate_scope_validation_v1.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _as_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _parse_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _numeric_version(raw: Any) -> tuple[int, ...] | None:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)*)\s*", str(raw or ""))
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _legacy_policy(
    required_scopes: Iterable[str], required_hours: float
) -> dict[str, Any]:
    scopes = sorted({str(scope) for scope in required_scopes if str(scope).strip()})
    return {
        "schema_version": 1,
        "policy_id": "legacy_uniform_elapsed_fallback",
        "calendar": {
            "calendar_id": "XNYS",
            "library": "exchange-calendars",
            "minimum_version": "0",
        },
        "tiers": {
            "legacy_uniform": {
                "required_hours": max(float(required_hours), 0.0),
                "required_completed_sessions": 0,
                "blocks_promotion": True,
            }
        },
        "scope_tiers": {scope: "legacy_uniform" for scope in scopes},
        "unknown_scope_tier": "legacy_uniform",
        "accounting": {
            "require_both_elapsed_hours_and_completed_sessions": True,
            "planned_maintenance_preserves_prior_credit": True,
            "planned_maintenance_offline_time_earns_credit": False,
            "interrupted_market_session_earns_session_credit": False,
        },
        "authority": {
            "live_execution_authority": False,
            "policy_changes_order_permissions": False,
        },
    }


def load_scope_validation_policy(
    project_root: Path,
    production_config: Mapping[str, Any],
    *,
    required_scopes: Iterable[str],
    legacy_required_hours: float = 720.0,
) -> dict[str, Any]:
    candidate = _as_dict(production_config.get("candidate"))
    configured = str(
        candidate.get("scope_validation_policy_path") or DEFAULT_POLICY_RELATIVE_PATH
    ).strip()
    required = bool(candidate.get("require_scope_validation_policy", False))
    path = Path(configured)
    if not path.is_absolute():
        path = project_root / path
    loaded = _load_json(path) if path.is_file() else {}
    fallback = not loaded and not required
    policy = (
        _legacy_policy(required_scopes, legacy_required_hours) if fallback else loaded
    )

    errors: list[str] = []
    tiers = _as_dict(policy.get("tiers"))
    scope_tiers = _as_dict(policy.get("scope_tiers"))
    unknown_tier = str(policy.get("unknown_scope_tier") or "").strip()
    calendar = _as_dict(policy.get("calendar"))
    if not policy:
        errors.append("scope_validation_policy_missing")
    if _as_int(policy.get("schema_version"), 0) != 1:
        errors.append("scope_validation_schema_version_invalid")
    if not str(policy.get("policy_id") or "").strip():
        errors.append("scope_validation_policy_id_missing")
    if not tiers:
        errors.append("scope_validation_tiers_missing")
    if not scope_tiers:
        errors.append("scope_validation_scope_tiers_missing")
    if not unknown_tier or unknown_tier not in tiers:
        errors.append("scope_validation_unknown_scope_tier_invalid")
    if not str(calendar.get("calendar_id") or "").strip():
        errors.append("scope_validation_calendar_id_missing")
    for tier_id, raw_tier in sorted(tiers.items()):
        tier = _as_dict(raw_tier)
        if not tier:
            errors.append(f"scope_validation_tier_invalid:{tier_id}")
            continue
        if _as_float(tier.get("required_hours"), -1.0) < 0.0:
            errors.append(f"scope_validation_tier_hours_invalid:{tier_id}")
        if _as_int(tier.get("required_completed_sessions"), -1) < 0:
            errors.append(f"scope_validation_tier_sessions_invalid:{tier_id}")
    for scope, tier_id in sorted(scope_tiers.items()):
        if str(tier_id) not in tiers:
            errors.append(f"scope_validation_scope_tier_unknown:{scope}={tier_id}")

    return {
        "ready": not errors,
        "required": required,
        "loaded": bool(loaded),
        "fallback_legacy_policy": fallback,
        "path": str(path),
        "relative_path": configured,
        "policy": policy,
        "errors": errors,
    }


def _maintenance_intervals(
    windows: Iterable[Mapping[str, Any]],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for row in windows:
        start = _parse_utc(row.get("offline_start_utc") or row.get("start_utc"))
        end = _parse_utc(row.get("offline_end_utc") or row.get("end_utc"))
        if start is None or end is None:
            continue
        start = max(start, window_start)
        end = min(end, window_end)
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    merged: list[tuple[datetime, datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def maintenance_overlap_hours(
    windows: Iterable[Mapping[str, Any]],
    *,
    window_start: datetime | None,
    window_end: datetime,
) -> float:
    if window_start is None or window_end <= window_start:
        return 0.0
    return (
        sum(
            (end - start).total_seconds()
            for start, end in _maintenance_intervals(
                windows, window_start=window_start, window_end=window_end
            )
        )
        / 3600.0
    )


def load_planned_maintenance_windows(
    project_root: Path, *, now: datetime
) -> list[dict[str, Any]]:
    directory = project_root / "governance" / "maintenance_events"
    if not directory.is_dir():
        return []
    windows: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("latest_"):
            continue
        payload = _load_json(path)
        classification = str(payload.get("classification") or "").strip().lower()
        accounting = _as_dict(payload.get("soak_accounting"))
        if not classification.startswith("planned_"):
            continue
        if str(payload.get("status") or "").strip().lower() != "completed":
            continue
        if bool(
            accounting.get(
                "counts_as_system_degradation",
                payload.get("counts_as_system_degradation", True),
            )
        ):
            continue
        if bool(
            accounting.get(
                "counts_as_trading_system_failure",
                payload.get("counts_as_trading_system_failure", True),
            )
        ):
            continue
        offline = _as_dict(payload.get("actual_offline_window"))
        start = _parse_utc(offline.get("offline_start_utc") or offline.get("start_utc"))
        end = _parse_utc(offline.get("offline_end_utc") or offline.get("end_utc"))
        if start is None or end is None or end <= start or start >= now:
            continue
        end = min(end, now)
        event_id = str(payload.get("event_id") or path.stem)
        windows[event_id] = {
            "event_id": event_id,
            "classification": classification,
            "title": str(payload.get("title") or event_id),
            "offline_start_utc": start.isoformat(),
            "offline_end_utc": end.isoformat(),
            "duration_hours": round((end - start).total_seconds() / 3600.0, 6),
            "source_path": str(path),
            "resets_candidate_clock": False,
            "earns_active_runtime_credit": False,
        }
    return sorted(windows.values(), key=lambda row: row["offline_start_utc"])


def _calendar_schedule(
    *,
    calendar_id: str,
    minimum_version: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    try:
        import exchange_calendars
        import pandas as pd

        package_version = metadata.version("exchange-calendars")
    except Exception as exc:
        return {
            "ready": False,
            "library": "exchange-calendars",
            "library_version": "",
            "calendar_id": calendar_id,
            "sessions": [],
            "errors": [f"exchange_calendars_unavailable:{type(exc).__name__}"],
        }
    errors: list[str] = []
    installed_version = _numeric_version(package_version)
    required_version = _numeric_version(minimum_version or "0")
    if installed_version is None:
        errors.append(f"exchange_calendars_version_invalid:{package_version}")
    elif required_version is None:
        errors.append(f"exchange_calendars_minimum_version_invalid:{minimum_version}")
    else:
        width = max(len(installed_version), len(required_version))
        installed_version = installed_version + (0,) * (width - len(installed_version))
        required_version = required_version + (0,) * (width - len(required_version))
        if installed_version < required_version:
            errors.append(
                f"exchange_calendars_version_below_floor:{package_version}<{minimum_version}"
            )
    rows: list[dict[str, Any]] = []
    try:
        calendar = exchange_calendars.get_calendar(calendar_id)
        schedule = calendar.schedule.loc[
            pd.Timestamp(start.date()) : pd.Timestamp(end.date())
        ]
        for session_label, values in schedule.iterrows():
            open_value = values.get("open")
            close_value = values.get("close")
            opened = open_value.to_pydatetime()
            closed = close_value.to_pydatetime()
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            if closed.tzinfo is None:
                closed = closed.replace(tzinfo=timezone.utc)
            rows.append(
                {
                    "session": str(session_label.date()),
                    "open": opened.astimezone(timezone.utc),
                    "close": closed.astimezone(timezone.utc),
                }
            )
    except Exception as exc:
        errors.append(f"exchange_calendar_schedule_failed:{type(exc).__name__}")
    if not rows:
        errors.append("exchange_calendar_schedule_empty")
    return {
        "ready": not errors,
        "library": "exchange-calendars",
        "library_version": package_version,
        "calendar_id": calendar_id,
        "sessions": rows,
        "errors": errors,
    }


def _session_overlaps_maintenance(
    opened: datetime,
    closed: datetime,
    maintenance_windows: Iterable[Mapping[str, Any]],
) -> bool:
    for row in maintenance_windows:
        start = _parse_utc(row.get("offline_start_utc") or row.get("start_utc"))
        end = _parse_utc(row.get("offline_end_utc") or row.get("end_utc"))
        if (
            start is not None
            and end is not None
            and min(closed, end) > max(opened, start)
        ):
            return True
    return False


def _grade(score: float, complete: bool) -> str:
    if complete and score >= 100.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def evaluate_scope_validation(
    project_root: Path,
    production_config: Mapping[str, Any],
    *,
    scope_windows_started_utc: Mapping[str, Any],
    required_scopes: Iterable[str],
    candidate_ready: bool,
    now: datetime,
    maintenance_windows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required = sorted({str(scope) for scope in required_scopes if str(scope).strip()})
    legacy_hours = _as_float(
        _as_dict(production_config.get("soak")).get("required_hours"), 720.0
    )
    loaded = load_scope_validation_policy(
        project_root,
        production_config,
        required_scopes=required,
        legacy_required_hours=legacy_hours,
    )
    policy = _as_dict(loaded.get("policy"))
    tiers = _as_dict(policy.get("tiers"))
    scope_tiers = _as_dict(policy.get("scope_tiers"))
    unknown_tier = str(policy.get("unknown_scope_tier") or "")
    windows = {
        str(scope): _parse_utc(value)
        for scope, value in scope_windows_started_utc.items()
    }
    maintenance = list(maintenance_windows or [])
    required_starts = [windows.get(scope) for scope in required if windows.get(scope)]
    max_required_hours = max(
        (
            _as_float(
                _as_dict(tiers.get(str(scope_tiers.get(scope) or unknown_tier))).get(
                    "required_hours"
                ),
                legacy_hours,
            )
            for scope in required
        ),
        default=legacy_hours,
    )
    calendar_policy = _as_dict(policy.get("calendar"))
    schedule_start = min(required_starts) if required_starts else now
    schedule_end = now + timedelta(days=max(120, int(max_required_hours / 24.0) + 45))
    sessions_required = any(
        _as_int(
            _as_dict(tiers.get(str(scope_tiers.get(scope) or unknown_tier))).get(
                "required_completed_sessions"
            ),
            0,
        )
        > 0
        for scope in required
    )
    calendar = (
        _calendar_schedule(
            calendar_id=str(calendar_policy.get("calendar_id") or "XNYS"),
            minimum_version=str(calendar_policy.get("minimum_version") or "0"),
            start=schedule_start,
            end=schedule_end,
        )
        if sessions_required
        else {
            "ready": True,
            "library": str(calendar_policy.get("library") or "exchange-calendars"),
            "library_version": "not_required",
            "calendar_id": str(calendar_policy.get("calendar_id") or "XNYS"),
            "sessions": [],
            "errors": [],
        }
    )
    schedule = _as_list(calendar.get("sessions"))

    rows: list[dict[str, Any]] = []
    for scope in sorted(set(windows) | set(scope_tiers) | set(required)):
        configured_tier = str(scope_tiers.get(scope) or "")
        unknown_scope = not configured_tier
        tier_id = configured_tier or unknown_tier
        tier = _as_dict(tiers.get(tier_id))
        blocks_promotion = bool(tier.get("blocks_promotion", True))
        required_for_promotion = scope in required and blocks_promotion
        required_hours = max(_as_float(tier.get("required_hours"), legacy_hours), 0.0)
        required_sessions = max(_as_int(tier.get("required_completed_sessions"), 0), 0)
        started = windows.get(scope)
        observed_hours = (
            max((now - started).total_seconds() / 3600.0, 0.0)
            if started is not None
            else 0.0
        )
        maintenance_hours = maintenance_overlap_hours(
            maintenance, window_start=started, window_end=now
        )
        active_hours = max(observed_hours - maintenance_hours, 0.0)
        credited_hours = (
            active_hours if candidate_ready and loaded.get("ready") else 0.0
        )
        eligible_sessions = [
            row for row in schedule if started is not None and row["open"] >= started
        ]
        completed_sessions = [
            row
            for row in eligible_sessions
            if row["close"] <= now
            and not _session_overlaps_maintenance(
                row["open"], row["close"], maintenance
            )
        ]
        interrupted_sessions = [
            row
            for row in eligible_sessions
            if row["close"] <= now
            and _session_overlaps_maintenance(row["open"], row["close"], maintenance)
        ]
        credited_sessions = (
            len(completed_sessions)
            if candidate_ready and loaded.get("ready") and calendar.get("ready")
            else 0
        )
        hours_ready = required_hours <= 0.0 or credited_hours >= required_hours
        sessions_ready = (
            required_sessions <= 0 or credited_sessions >= required_sessions
        )
        window_ready = started is not None or not required_for_promotion
        row_ready = bool(
            not required_for_promotion
            or (
                loaded.get("ready")
                and candidate_ready
                and window_ready
                and hours_ready
                and sessions_ready
                and (calendar.get("ready") or required_sessions == 0)
            )
        )
        hour_ratio = (
            1.0 if required_hours <= 0 else min(credited_hours / required_hours, 1.0)
        )
        session_ratio = (
            1.0
            if required_sessions <= 0
            else min(credited_sessions / required_sessions, 1.0)
        )
        progress = 1.0 if not required_for_promotion else min(hour_ratio, session_ratio)
        valid_future_sessions = [
            row
            for row in eligible_sessions
            if not _session_overlaps_maintenance(row["open"], row["close"], maintenance)
        ]
        session_eligible_at = (
            valid_future_sessions[required_sessions - 1]["close"]
            if required_sessions > 0 and len(valid_future_sessions) >= required_sessions
            else None
        )
        hour_eligible_at = (
            started + timedelta(hours=required_hours + maintenance_hours)
            if started is not None
            else None
        )
        eligibility_candidates = [
            value
            for value in (hour_eligible_at, session_eligible_at)
            if value is not None
        ]
        earliest_eligible = (
            max(eligibility_candidates) if eligibility_candidates else None
        )
        blockers: list[str] = []
        if required_for_promotion:
            if not loaded.get("ready"):
                blockers.append("scope_validation_policy_not_ready")
            if not candidate_ready:
                blockers.append("candidate_not_current")
            if started is None:
                blockers.append("scope_window_missing")
            if not hours_ready:
                blockers.append("required_elapsed_hours_pending")
            if not sessions_ready:
                blockers.append("required_completed_sessions_pending")
            if required_sessions > 0 and not calendar.get("ready"):
                blockers.append("market_calendar_not_ready")
        rows.append(
            {
                "scope": scope,
                "tier": tier_id or "invalid",
                "unknown_scope_fail_closed": unknown_scope,
                "required_for_promotion": required_for_promotion,
                "window_started_utc": started.isoformat() if started else "",
                "required_hours": required_hours,
                "observed_hours": round(observed_hours, 6),
                "planned_maintenance_excluded_hours": round(maintenance_hours, 6),
                "credited_hours": round(credited_hours, 6),
                "required_completed_sessions": required_sessions,
                "observed_completed_sessions": len(completed_sessions),
                "credited_completed_sessions": credited_sessions,
                "interrupted_sessions_excluded": len(interrupted_sessions),
                "hours_ready": hours_ready,
                "sessions_ready": sessions_ready,
                "ready": row_ready,
                "progress_percent": round(progress * 100.0, 3),
                "earliest_eligible_utc": (
                    earliest_eligible.isoformat() if earliest_eligible else ""
                ),
                "blockers": blockers,
            }
        )

    required_rows = [row for row in rows if row["required_for_promotion"]]
    complete = bool(
        required_rows
        and loaded.get("ready")
        and candidate_ready
        and all(row["ready"] for row in required_rows)
    )
    score = (
        sum(_as_float(row.get("progress_percent")) for row in required_rows)
        / len(required_rows)
        if required_rows
        else 0.0
    )
    blocking_rows = [row for row in required_rows if not row["ready"]]
    bottleneck = (
        min(required_rows, key=lambda row: _as_float(row.get("progress_percent")))
        if required_rows
        else {}
    )
    return {
        "policy_id": str(policy.get("policy_id") or "missing"),
        "policy_ready": bool(loaded.get("ready")),
        "policy_required": bool(loaded.get("required")),
        "policy_path": loaded.get("path"),
        "policy_relative_path": loaded.get("relative_path"),
        "policy_errors": list(loaded.get("errors") or []),
        "legacy_fallback_active": bool(loaded.get("fallback_legacy_policy")),
        "candidate_ready": candidate_ready,
        "scope_aware_validation_complete": complete,
        "promotion_elapsed_complete": complete,
        "score": round(score, 3),
        "grade": _grade(score, complete),
        "required_scopes": required,
        "required_scope_count": len(required_rows),
        "ready_required_scope_count": sum(1 for row in required_rows if row["ready"]),
        "blocking_scopes": [row["scope"] for row in blocking_rows],
        "bottleneck_scope": str(bottleneck.get("scope") or ""),
        "bottleneck_tier": str(bottleneck.get("tier") or ""),
        "bottleneck_progress_percent": _as_float(
            bottleneck.get("progress_percent"), 0.0
        ),
        "scope_results": rows,
        "calendar": {
            "ready": bool(calendar.get("ready")),
            "library": calendar.get("library"),
            "library_version": calendar.get("library_version"),
            "calendar_id": calendar.get("calendar_id"),
            "errors": list(calendar.get("errors") or []),
        },
        "planned_maintenance_event_count": len(maintenance),
        "accounting": _as_dict(policy.get("accounting")),
        "authority": {
            **_as_dict(policy.get("authority")),
            "live_execution_authority": False,
            "policy_changes_order_permissions": False,
        },
    }
