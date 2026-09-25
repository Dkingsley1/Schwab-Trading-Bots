from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


FLAG_NAME = "STACK_RESTART_IN_PROGRESS.flag"
DEFAULT_TTL_SECONDS = 15 * 60


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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def stack_restart_fence_path(project_root: str | Path) -> Path:
    override = str(os.getenv("STACK_RESTART_FENCE_PATH", "") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(project_root).resolve() / "governance" / "health" / FLAG_NAME


def stack_restart_fence_snapshot(
    project_root: str | Path,
    *,
    now_utc: datetime | None = None,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> dict[str, Any]:
    path = stack_restart_fence_path(project_root)
    now = now_utc or _utc_now()
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "path": str(path),
            "exists": False,
            "active": False,
            "expired": False,
            "owner_alive": False,
            "valid": True,
            "reason": "",
        }
    except Exception as exc:
        try:
            age_seconds = max(now.timestamp() - path.stat().st_mtime, 0.0)
        except OSError:
            age_seconds = 0.0
        expired = bool(age_seconds >= DEFAULT_TTL_SECONDS)
        return {
            "path": str(path),
            "exists": True,
            "active": not expired,
            "expired": expired,
            "owner_alive": False,
            "valid": False,
            "reason": (
                "unreadable_stack_restart_fence_expired"
                if expired
                else "unreadable_stack_restart_fence_fail_closed"
            ),
            "age_seconds": round(age_seconds, 3),
            "error": f"{type(exc).__name__}:{exc}",
        }

    payload = decoded if isinstance(decoded, dict) else {}
    owner_pid = int(payload.get("owner_pid", 0) or 0)
    expires_at = _parse_timestamp(payload.get("expires_at_utc"))
    expired = bool(expires_at is not None and now >= expires_at)
    owner_alive = bool(pid_alive(owner_pid))
    valid = bool(owner_pid > 0 and expires_at is not None and payload.get("token"))
    active = bool(valid and owner_alive and not expired)
    if active:
        reason = str(payload.get("reason") or "stack_restart_in_progress")
    elif expired:
        reason = "stack_restart_fence_expired"
    elif valid and not owner_alive:
        reason = "stack_restart_owner_not_alive"
    else:
        reason = "stack_restart_fence_invalid"
    return {
        "path": str(path),
        "exists": True,
        "active": active,
        "expired": expired,
        "owner_alive": owner_alive,
        "valid": valid,
        "reason": reason,
        "owner": str(payload.get("owner") or ""),
        "owner_pid": owner_pid,
        "token": str(payload.get("token") or ""),
        "engaged_at_utc": str(payload.get("engaged_at_utc") or ""),
        "expires_at_utc": str(payload.get("expires_at_utc") or ""),
        "ttl_seconds": int(payload.get("ttl_seconds", 0) or 0),
        "payload": payload,
    }


def engage_stack_restart_fence(
    project_root: str | Path,
    *,
    owner_pid: int,
    owner: str = "start_stack",
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    path = stack_restart_fence_path(project_root)
    existing = stack_restart_fence_snapshot(project_root)
    if bool(existing.get("active", False)):
        return {**existing, "acquired": False, "acquire_error": "stack_restart_already_in_progress"}
    if bool(existing.get("exists", False)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            return {**existing, "acquired": False, "acquire_error": f"stale_fence_cleanup_failed:{type(exc).__name__}:{exc}"}

    now = _utc_now()
    ttl = max(int(ttl_seconds), 60)
    payload = {
        "schema_version": 1,
        "engaged_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(seconds=ttl)).isoformat(),
        "ttl_seconds": ttl,
        "reason": "stack_restart_in_progress",
        "owner": str(owner or "start_stack"),
        "owner_pid": int(owner_pid),
        "token": secrets.token_hex(16),
        "blocks": [
            "peer_supervisor_recovery",
            "watchdog_worker_spawns",
            "hot_standby_fallback_starts",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        current = stack_restart_fence_snapshot(project_root)
        return {**current, "acquired": False, "acquire_error": "stack_restart_race_lost"}
    except Exception as exc:
        return {
            **stack_restart_fence_snapshot(project_root),
            "acquired": False,
            "acquire_error": f"stack_restart_fence_create_failed:{type(exc).__name__}:{exc}",
        }
    return {**stack_restart_fence_snapshot(project_root), "acquired": True}


def release_stack_restart_fence(project_root: str | Path, *, expected_token: str) -> dict[str, Any]:
    before = stack_restart_fence_snapshot(project_root)
    path = stack_restart_fence_path(project_root)
    current_token = str(before.get("token") or "")
    supplied = str(expected_token or "")
    if not supplied or not current_token or supplied != current_token:
        return {**before, "released": False, "release_error": "stack_restart_fence_token_mismatch"}
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        return {**before, "released": False, "release_error": f"{type(exc).__name__}:{exc}"}
    return {**stack_restart_fence_snapshot(project_root), "released": True, "previous": before}
