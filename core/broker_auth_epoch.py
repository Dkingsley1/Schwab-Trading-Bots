from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _expiry_epoch(payload: dict[str, Any]) -> float | None:
    candidates = [payload]
    for key in ("token", "tokens"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for candidate in candidates:
        for key in ("expires_at", "expiresAt", "expires_epoch"):
            try:
                value = float(candidate.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def token_epoch(path: Path) -> dict[str, Any]:
    """Return a nonsecret identity for the currently installed broker token."""
    token_path = Path(path)
    try:
        stat = token_path.stat()
    except OSError:
        return {
            "present": False,
            "id": "missing",
            "mtime_ns": 0,
            "mtime_epoch": 0.0,
            "size_bytes": 0,
        }

    expires_at: float | None = None
    try:
        payload = json.loads(token_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        expires_at = _expiry_epoch(payload)

    identity = ":".join(
        (
            str(int(stat.st_dev)),
            str(int(stat.st_ino)),
            str(int(stat.st_mtime_ns)),
            str(int(stat.st_size)),
            str(int(expires_at or 0.0)),
        )
    )
    return {
        "present": True,
        "id": hashlib.sha256(identity.encode("ascii")).hexdigest()[:20],
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_epoch": float(stat.st_mtime),
        "size_bytes": int(stat.st_size),
        "expires_at_epoch": expires_at,
    }


def token_epoch_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if not bool(current.get("present", False)):
        return False
    if not bool(previous.get("present", False)):
        return True
    return str(previous.get("id") or "") != str(current.get("id") or "")
