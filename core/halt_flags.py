from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.accountability import safe_write_json_atomic


def write_halt_flag_atomic(
    path: str | Path,
    payload: dict[str, Any],
    *,
    project_root: str = "",
    source: str = "halt_flag",
) -> bool:
    return safe_write_json_atomic(
        str(Path(path)),
        payload,
        project_root=project_root,
        source=source,
        marker=False,
    )


def inspect_halt_flag(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    out: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
        "payload": {},
        "reason": "",
        "valid": False,
        "error": "",
        "size_bytes": 0,
    }
    if not target.exists():
        return out

    try:
        out["size_bytes"] = int(target.stat().st_size)
    except Exception:
        out["size_bytes"] = 0

    try:
        raw = target.read_text(encoding="utf-8")
    except Exception as exc:
        out["error"] = f"read_error:{type(exc).__name__}"
        return out

    if not raw.strip():
        out["error"] = "empty_payload"
        return out

    try:
        payload = json.loads(raw)
    except Exception as exc:
        out["error"] = f"invalid_json:{type(exc).__name__}"
        return out

    if not isinstance(payload, dict):
        out["error"] = f"payload_not_object:{type(payload).__name__}"
        return out

    out["payload"] = payload
    out["reason"] = str(payload.get("reason") or "").strip()
    out["valid"] = bool(out["reason"])
    if not out["valid"]:
        out["error"] = "missing_reason"
    return out
