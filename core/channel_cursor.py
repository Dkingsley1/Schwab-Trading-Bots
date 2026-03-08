from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from core.path_registry import channel_cursor_path


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def get_channel_cursor(project_root: str | Path, consumer: str, channel: str) -> Dict[str, Any]:
    path = Path(channel_cursor_path(project_root, consumer, channel))
    payload = _load_json(path)
    if not payload:
        return {
            "consumer": str(consumer or ""),
            "channel": str(channel or ""),
            "acked_message_id": "",
            "acked_at": "",
            "updated_at": "",
            "cursor_version": 1,
            "path": str(path),
        }
    payload.setdefault("path", str(path))
    return payload


def ack_channel_cursor(
    project_root: str | Path,
    consumer: str,
    channel: str,
    *,
    acked_message_id: str,
    acked_at: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    path = Path(channel_cursor_path(project_root, consumer, channel))
    payload = {
        "consumer": str(consumer or ""),
        "channel": str(channel or ""),
        "acked_message_id": str(acked_message_id or ""),
        "acked_at": str(acked_at or _now_utc()),
        "updated_at": _now_utc(),
        "cursor_version": 1,
    }
    if extra:
        payload["extra"] = dict(extra)
    _atomic_write_json(path, payload)
    payload["path"] = str(path)
    return payload
