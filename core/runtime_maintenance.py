from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


FLAG_NAME = "RUNTIME_MAINTENANCE_HOLD.flag"
MAINTENANCE_HOLD_TOKEN_ENV = "SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(raw: Any) -> datetime | None:
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


def maintenance_hold_path(project_root: str | Path) -> Path:
    override = str(os.getenv("RUNTIME_MAINTENANCE_HOLD_PATH", "") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(project_root).resolve() / "governance" / "health" / FLAG_NAME


def maintenance_hold_snapshot(project_root: str | Path, *, now_utc: datetime | None = None) -> dict[str, Any]:
    path = maintenance_hold_path(project_root)
    now = now_utc or _utc_now()
    exists = True
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        decoded = {}
        exists = False
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "active": True,
            "expired": False,
            "valid": False,
            "reason": "unreadable_maintenance_hold_fail_closed",
            "error": f"{type(exc).__name__}:{exc}",
        }

    payload = decoded if isinstance(decoded, dict) else {}
    if not exists:
        return {
            "path": str(path),
            "exists": False,
            "active": False,
            "expired": False,
            "valid": True,
            "reason": "",
        }

    expires_at = _parse_timestamp(payload.get("expires_at_utc"))
    expired = bool(expires_at is not None and now >= expires_at)
    return {
        "path": str(path),
        "exists": True,
        "active": not expired,
        "expired": expired,
        "valid": True,
        "reason": str(payload.get("reason") or "runtime_maintenance"),
        "owner": str(payload.get("owner") or ""),
        "token": str(payload.get("token") or ""),
        "engaged_at_utc": str(payload.get("engaged_at_utc") or ""),
        "expires_at_utc": str(payload.get("expires_at_utc") or ""),
        "ttl_seconds": int(payload.get("ttl_seconds", 0) or 0),
        "payload": payload,
    }


def maintenance_hold_token_authorized(snapshot: dict[str, Any], *, token: str = "") -> bool:
    expected = str(snapshot.get("token") or "").strip()
    supplied = str(token or os.getenv(MAINTENANCE_HOLD_TOKEN_ENV, "") or "").strip()
    return bool(
        snapshot.get("active", False)
        and snapshot.get("valid", False)
        and expected
        and supplied
        and supplied == expected
    )


def engage_maintenance_hold(
    project_root: str | Path,
    *,
    reason: str,
    owner: str = "",
    ttl_seconds: int = 8 * 60 * 60,
) -> dict[str, Any]:
    path = maintenance_hold_path(project_root)
    now = _utc_now()
    ttl = max(int(ttl_seconds), 60)
    payload = {
        "schema_version": 1,
        "engaged_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(seconds=ttl)).isoformat(),
        "ttl_seconds": ttl,
        "reason": str(reason or "runtime_maintenance"),
        "owner": str(owner or os.getenv("USER") or "operator"),
        "token": secrets.token_hex(16),
        "blocks": [
            "runtime_launches",
            "watchdog_restarts",
            "sqlite_writers",
            "guarded_maintenance_jobs",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return maintenance_hold_snapshot(project_root)


def release_maintenance_hold(project_root: str | Path, *, expected_token: str = "") -> dict[str, Any]:
    before = maintenance_hold_snapshot(project_root)
    path = maintenance_hold_path(project_root)
    current_token = str(before.get("token") or "")
    if expected_token and current_token and expected_token != current_token:
        return {
            **before,
            "released": False,
            "release_error": "maintenance_hold_token_mismatch",
        }
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        return {
            **before,
            "released": False,
            "release_error": f"{type(exc).__name__}:{exc}",
        }
    return {
        **maintenance_hold_snapshot(project_root),
        "released": True,
        "previous": before,
    }
