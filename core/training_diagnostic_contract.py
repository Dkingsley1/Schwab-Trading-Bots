"""Producer freshness and identity-bound contracts for training evidence."""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


def materialization_contract_valid(contract: Mapping[str, Any], bot_id: str) -> bool:
    if not isinstance(contract, Mapping) or not bot_id or contract.get("bot_id") != bot_id:
        return False
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    try:
        digest = hashlib.sha256(json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return False
    return bool(contract.get("contract_sha256") == digest)


def diagnostic_age_hours(payload: Mapping[str, Any], now: datetime) -> float | None:
    raw = next(
        (payload[key] for key in ("generated_at_utc", "generated_utc", "timestamp_utc", "timestamp") if payload.get(key)),
        None,
    )
    if not raw:
        return None
    try:
        produced = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if produced.tzinfo is None:
            produced = produced.replace(tzinfo=timezone.utc)
        seconds = (now.astimezone(timezone.utc) - produced.astimezone(timezone.utc)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds / 3600.0 if seconds >= 0.0 else None
