from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from core.accountability import safe_write_json_atomic


def twelve_data_guard_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return root / "governance" / "health" / "fx_twelve_data_guard_latest.json"


def load_twelve_data_guard(project_root: str | Path) -> Dict[str, Any]:
    path = twelve_data_guard_path(project_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_twelve_data_failure(*, code: str = "", message: str = "") -> str:
    code_text = str(code or "").strip()
    lowered = str(message or "").lower()
    daily_fragments = (
        "run out of api credits for the day",
        "api credits were used",
        "current limit being",
        "daily limit",
        "daily quota",
    )
    if code_text == "429" and any(fragment in lowered for fragment in daily_fragments):
        return "daily_quota"
    if code_text == "429" or "too many requests" in lowered or "rate limit" in lowered:
        return "rate_limit"
    return ""


def _next_daily_quota_reset_ts(now_ts: float) -> float:
    reset_hour = min(max(int(os.getenv("FX_TWELVE_DATA_DAILY_QUOTA_RESET_HOUR_UTC", "0") or 0), 0), 23)
    reset_minute = min(max(int(os.getenv("FX_TWELVE_DATA_DAILY_QUOTA_RESET_MINUTE_UTC", "0") or 0), 0), 59)
    reset_grace_seconds = max(
        float(os.getenv("FX_TWELVE_DATA_DAILY_QUOTA_RESET_GRACE_SECONDS", "300") or 300.0),
        0.0,
    )
    now_dt = datetime.fromtimestamp(float(now_ts), tz=timezone.utc)
    reset_dt = now_dt.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
    if reset_dt.timestamp() <= float(now_ts):
        reset_dt += timedelta(days=1)
    return reset_dt.timestamp() + reset_grace_seconds


def mark_twelve_data_cooldown(
    *,
    project_root: str | Path,
    kind: str,
    code: str = "",
    message: str = "",
    symbol: str = "",
    source: str = "",
    now_ts: float | None = None,
) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    now_value = float(time.time() if now_ts is None else now_ts)
    prior = load_twelve_data_guard(root)
    if str(kind or "").strip().lower() == "daily_quota":
        cooldown_until_ts = _next_daily_quota_reset_ts(now_value)
    else:
        cooldown_until_ts = now_value + max(
            float(os.getenv("FX_TWELVE_DATA_RATE_LIMIT_COOLDOWN_SECONDS", "900") or 900.0),
            1.0,
        )

    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.fromtimestamp(now_value, tz=timezone.utc).isoformat(),
        "source": str(source or "").strip(),
        "symbol": str(symbol or "").strip().upper(),
        "kind": str(kind or "").strip().lower() or "rate_limit",
        "code": str(code or "").strip(),
        "message": str(message or "").strip(),
        "cooldown_until_ts": float(cooldown_until_ts),
        "cooldown_until_utc": datetime.fromtimestamp(cooldown_until_ts, tz=timezone.utc).isoformat(),
        "failure_count": int(prior.get("failure_count", 0) or 0) + 1,
        "last_failure_utc": datetime.fromtimestamp(now_value, tz=timezone.utc).isoformat(),
    }
    safe_write_json_atomic(
        str(twelve_data_guard_path(root)),
        payload,
        project_root=str(root),
        source="fx_twelve_data_guard",
        indent=2,
        marker=True,
    )
    return twelve_data_cooldown_status(root, now_ts=now_value)


def twelve_data_cooldown_status(project_root: str | Path, *, now_ts: float | None = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    state = load_twelve_data_guard(root)
    now_value = float(time.time() if now_ts is None else now_ts)
    cooldown_until_ts = float(state.get("cooldown_until_ts", 0.0) or 0.0)
    remaining_seconds = max(cooldown_until_ts - now_value, 0.0)
    return {
        **state,
        "active": bool(cooldown_until_ts > now_value),
        "remaining_seconds": float(remaining_seconds),
    }
