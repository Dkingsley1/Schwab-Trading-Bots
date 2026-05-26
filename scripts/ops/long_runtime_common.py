#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import deque
from datetime import date, datetime, timedelta, time, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo is standard on supported runtimes
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc

STATUS_RANK = {
    "ready": 0,
    "ok": 0,
    "active": 0,
    "warn": 1,
    "thin": 1,
    "needs_coverage": 1,
    "needs_work": 1,
    "degraded": 2,
    "inactive": 2,
    "missing": 2,
    "blocked": 3,
    "critical": 3,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_recent_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(int(limit), 1))
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return list(rows)


def payload_timestamp(payload: dict[str, Any], path: Path | None = None) -> datetime | None:
    for key in ("timestamp_utc", "updated_at_utc", "updated_at", "created_at", "ended_utc", "started_utc"):
        parsed = parse_iso_utc(payload.get(key))
        if parsed is not None:
            return parsed
    if path is None:
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def payload_age_minutes(payload: dict[str, Any], path: Path | None = None, *, now: datetime | None = None) -> float | None:
    ts = payload_timestamp(payload, path)
    if ts is None:
        return None
    current = now or utc_now()
    return max((current - ts).total_seconds() / 60.0, 0.0)


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def ordered_unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def status_rank(status: str) -> int:
    return STATUS_RANK.get(str(status or "").strip().lower(), 1)


def bool_status(ok: bool) -> str:
    return "ready" if ok else "blocked"


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    offset = (weekday - current.weekday()) % 7
    return current + timedelta(days=offset + (nth - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_date(year: int) -> date:
    # Meeus/Jones/Butcher Gregorian computus.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_equity_market_holiday(day: date) -> str:
    year = day.year
    holidays = {
        _observed_fixed_holiday(year, 1, 1): "new_years_day",
        _nth_weekday(year, 1, 0, 3): "martin_luther_king_jr_day",
        _nth_weekday(year, 2, 0, 3): "washingtons_birthday",
        _easter_date(year) - timedelta(days=2): "good_friday",
        _last_weekday(year, 5, 0): "memorial_day",
        _observed_fixed_holiday(year, 6, 19): "juneteenth",
        _observed_fixed_holiday(year, 7, 4): "independence_day",
        _nth_weekday(year, 9, 0, 1): "labor_day",
        _nth_weekday(year, 11, 3, 4): "thanksgiving_day",
        _observed_fixed_holiday(year, 12, 25): "christmas_day",
    }
    return holidays.get(day, "")


def eastern_off_hours_window(
    *,
    now: datetime | None = None,
    start_local: time = time(16, 15),
    end_local: time = time(9, 20),
) -> dict[str, Any]:
    current = (now or utc_now()).astimezone(LOCAL_TZ)
    local_clock = current.timetz().replace(tzinfo=None)
    is_weekend = current.weekday() >= 5
    holiday_name = us_equity_market_holiday(current.date())
    market_holiday = bool(holiday_name)
    active = bool(is_weekend or market_holiday or local_clock >= start_local or local_clock < end_local)
    return {
        "active": active,
        "is_weekend": is_weekend,
        "market_holiday": market_holiday,
        "market_holiday_name": holiday_name,
        "timezone": "America/New_York",
        "local_time": current.isoformat(),
        "window_start_local": start_local.strftime("%H:%M"),
        "window_end_local": end_local.strftime("%H:%M"),
        "label": "off_hours" if active else "market_hours",
    }
