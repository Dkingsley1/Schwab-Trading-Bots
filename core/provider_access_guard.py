from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

from core.accountability import safe_write_json_atomic


_HTTP_STATUS_RE = re.compile(r"(?:http_status_|status(?:_code)?[=: ]+)(\d{3})", re.IGNORECASE)


def provider_http_status_code(value: Any) -> int:
    if isinstance(value, dict):
        for key in ("status_code", "http_status", "provider_status_code"):
            try:
                code = int(value.get(key, 0) or 0)
            except Exception:
                code = 0
            if 100 <= code <= 599:
                return code
        value = value.get("error") or value.get("reason") or value
    match = _HTTP_STATUS_RE.search(str(value or ""))
    return int(match.group(1)) if match else 0


def _safe_provider_name(provider: str) -> str:
    token = re.sub(r"[^a-z0-9_.-]+", "_", str(provider or "provider").strip().lower())
    return token.strip("._") or "provider"


def provider_access_state_path(project_root: str | Path, provider: str) -> Path:
    return Path(project_root) / "governance" / "health" / f"provider_access_guard_{_safe_provider_name(provider)}_latest.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _epoch_from_iso(value: Any) -> float:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _iso_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def provider_cooldown_seconds(provider: str, status_code: int) -> int:
    name = _safe_provider_name(provider).upper().replace("-", "_").replace(".", "_")
    code = int(status_code or 0)
    if code == 403:
        default = "900"
    elif code in {401, 429}:
        default = "300"
    else:
        default = "180"
    raw = os.getenv(f"{name}_HTTP_{code}_COOLDOWN_SECONDS", default)
    try:
        return max(int(float(raw)), 15)
    except Exception:
        return int(default)


def provider_access_status(
    project_root: str | Path,
    provider: str,
    *,
    now_ts: float | None = None,
) -> Dict[str, Any]:
    now = float(time.time() if now_ts is None else now_ts)
    path = provider_access_state_path(project_root, provider)
    payload = _load_json(path)
    until_ts = 0.0
    try:
        until_ts = float(payload.get("cooldown_until_epoch", 0.0) or 0.0)
    except Exception:
        until_ts = 0.0
    if until_ts <= 0.0:
        until_ts = _epoch_from_iso(payload.get("cooldown_until_utc"))
    active = bool(until_ts > now and str(payload.get("state") or "").lower() == "cooldown")
    return {
        **payload,
        "provider": _safe_provider_name(provider),
        "active": active,
        "remaining_seconds": max(int(until_ts - now), 0) if active else 0,
        "cooldown_until_epoch": float(until_ts),
        "path": str(path),
    }


def activate_provider_cooldown(
    project_root: str | Path,
    provider: str,
    *,
    status_code: int,
    reason: str,
    symbol: str = "",
    profile: str = "",
    domain: str = "",
    cooldown_seconds: int | None = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    name = _safe_provider_name(provider)
    state_path = provider_access_state_path(root, name)
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    seconds = max(
        int(cooldown_seconds if cooldown_seconds is not None else provider_cooldown_seconds(name, status_code)),
        15,
    )
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            current = provider_access_status(root, name, now_ts=now)
            current_until = float(current.get("cooldown_until_epoch", 0.0) or 0.0)
            until_ts = max(current_until, now + seconds)
            first_denial = str(current.get("first_denial_utc") or "")
            if not first_denial or not bool(current.get("active", False)):
                first_denial = _iso_from_epoch(now)
            payload: Dict[str, Any] = {
                "timestamp_utc": _iso_from_epoch(now),
                "schema_version": 1,
                "ok": False,
                "overall_status": "degraded",
                "provider": name,
                "state": "cooldown",
                "active": True,
                "status_code": int(status_code or 0),
                "reason": str(reason or "provider_access_denied"),
                "symbol": str(symbol or ""),
                "profile": str(profile or ""),
                "domain": str(domain or ""),
                "owner_pid": int(os.getpid()),
                "first_denial_utc": first_denial,
                "last_denial_utc": _iso_from_epoch(now),
                "denial_count": int(current.get("denial_count", 0) or 0) + 1,
                "cooldown_seconds": int(max(until_ts - now, 0.0)),
                "cooldown_until_epoch": float(until_ts),
                "cooldown_until_utc": _iso_from_epoch(until_ts),
                "next_probe_after_utc": _iso_from_epoch(until_ts),
                "policy": "fleet_wide_cooldown_stops_provider_fanout_after_access_denial",
            }
            safe_write_json_atomic(
                str(state_path),
                payload,
                project_root=str(root),
                source="provider_access_guard.activate",
            )
            return provider_access_status(root, name, now_ts=now)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def mark_provider_recovered(
    project_root: str | Path,
    provider: str,
    *,
    evidence: str,
    force: bool = False,
) -> Dict[str, Any]:
    root = Path(project_root)
    name = _safe_provider_name(provider)
    state_path = provider_access_state_path(root, name)
    current = provider_access_status(root, name)
    if (
        not current
        or str(current.get("state") or "") != "cooldown"
        or (bool(current.get("active", False)) and not force)
    ):
        return current
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            current = provider_access_status(root, name)
            if bool(current.get("active", False)) and not force:
                return current
            now = time.time()
            payload = {
                **{key: value for key, value in current.items() if key not in {"path", "remaining_seconds"}},
                "timestamp_utc": _iso_from_epoch(now),
                "ok": True,
                "overall_status": "ready",
                "state": "ready",
                "active": False,
                "remaining_seconds": 0,
                "recovered_utc": _iso_from_epoch(now),
                "recovery_evidence": str(evidence or "successful_provider_request"),
                "forced_recovery_from_verified_request": bool(force),
            }
            safe_write_json_atomic(
                str(state_path),
                payload,
                project_root=str(root),
                source="provider_access_guard.recovered",
            )
            return provider_access_status(root, name, now_ts=now)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _snapshot_path(project_root: str | Path, provider: str, symbol: str) -> Path:
    symbol_key = hashlib.sha1(str(symbol).strip().upper().encode("utf-8")).hexdigest()[:20]
    return (
        Path(project_root)
        / "governance"
        / "health"
        / "provider_market_snapshot_cache"
        / _safe_provider_name(provider)
        / f"{symbol_key}.json"
    )


def load_shared_market_snapshot(
    project_root: str | Path,
    provider: str,
    symbol: str,
    *,
    max_age_seconds: float,
    now_ts: float | None = None,
) -> Dict[str, Any] | None:
    now = float(time.time() if now_ts is None else now_ts)
    payload = _load_json(_snapshot_path(project_root, provider, symbol))
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else None
    try:
        age = max(now - float(payload.get("timestamp_epoch", 0.0) or 0.0), 0.0)
    except Exception:
        age = float("inf")
    if snapshot is None or age > max(float(max_age_seconds), 0.0):
        return None
    return {**snapshot, "shared_provider_cache_hit": 1.0, "shared_provider_cache_age_seconds": float(age)}


def write_shared_market_snapshot(
    project_root: str | Path,
    provider: str,
    symbol: str,
    snapshot: Dict[str, Any],
) -> bool:
    root = Path(project_root)
    now = time.time()
    return bool(
        safe_write_json_atomic(
            str(_snapshot_path(root, provider, symbol)),
            {
                "timestamp_utc": _iso_from_epoch(now),
                "timestamp_epoch": float(now),
                "provider": _safe_provider_name(provider),
                "symbol": str(symbol).strip().upper(),
                "snapshot": dict(snapshot or {}),
            },
            project_root=str(root),
            source="provider_access_guard.market_snapshot_cache",
            marker=False,
        )
    )


@contextmanager
def provider_request_slot(
    project_root: str | Path,
    provider: str,
    symbol: str,
    *,
    slot_count: int = 4,
    wait_seconds: float = 20.0,
) -> Iterator[None]:
    root = Path(project_root)
    name = _safe_provider_name(provider)
    slots = max(int(slot_count), 1)
    slot = int(hashlib.sha1(str(symbol).strip().upper().encode("utf-8")).hexdigest()[:8], 16) % slots
    lock_path = root / "governance" / "health" / f"provider_request_slot_{name}_{slot}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(float(wait_seconds), 0.1)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if provider_access_status(root, name).get("active", False):
                    raise RuntimeError(f"{name}_provider_cooldown_active")
                time.sleep(0.025)
        if not acquired:
            raise RuntimeError(f"{name}_provider_request_slot_timeout")
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
