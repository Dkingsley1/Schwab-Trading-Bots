from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


def _parse_utc(value: Any) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("event timestamp is required")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EventTimePolicy:
    allowed_lateness_seconds: float = 30.0
    max_future_skew_seconds: float = 5.0
    dedupe_capacity: int = 10000

    def __post_init__(self) -> None:
        if self.allowed_lateness_seconds < 0:
            raise ValueError("allowed lateness cannot be negative")
        if self.max_future_skew_seconds < 0:
            raise ValueError("future skew cannot be negative")
        if self.dedupe_capacity < 1:
            raise ValueError("dedupe capacity must be positive")


@dataclass(frozen=True)
class EventTimeDecision:
    accepted: bool
    disposition: str
    reason: str
    stream_id: str
    event_id: str
    event_time_utc: str
    observed_at_utc: str
    watermark_utc: str
    lateness_seconds: float
    payload_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventTimeGuard:
    def __init__(self, policy: EventTimePolicy | None = None) -> None:
        self.policy = policy or EventTimePolicy()
        self._streams: dict[str, dict[str, Any]] = {}

    def _state(self, stream_id: str) -> dict[str, Any]:
        key = str(stream_id or "").strip()
        if not key:
            raise ValueError("stream_id is required")
        return self._streams.setdefault(
            key,
            {
                "max_event_time": None,
                "seen": OrderedDict(),
                "accepted_count": 0,
                "quarantine_count": 0,
            },
        )

    def ingest(
        self,
        *,
        stream_id: str,
        event_id: str,
        event_time_utc: Any,
        payload: Any,
        observed_at_utc: Any | None = None,
    ) -> dict[str, Any]:
        state = self._state(stream_id)
        key = str(event_id or "").strip()
        if not key:
            raise ValueError("event_id is required")
        event_time = _parse_utc(event_time_utc)
        observed = _parse_utc(observed_at_utc or datetime.now(timezone.utc).isoformat())
        payload_hash = _hash(payload)
        seen: OrderedDict[str, str] = state["seen"]
        prior_hash = seen.get(key)
        max_event_time: datetime | None = state["max_event_time"]
        watermark = (
            max_event_time - timedelta(seconds=self.policy.allowed_lateness_seconds)
            if max_event_time is not None
            else event_time - timedelta(seconds=self.policy.allowed_lateness_seconds)
        )
        disposition = "on_time"
        reason = "accepted"
        accepted = True
        if prior_hash == payload_hash:
            accepted = False
            disposition = "duplicate"
            reason = "exact_duplicate"
        elif prior_hash is not None:
            accepted = False
            disposition = "quarantine"
            reason = "event_identity_payload_conflict"
        elif event_time > observed + timedelta(
            seconds=self.policy.max_future_skew_seconds
        ):
            accepted = False
            disposition = "quarantine"
            reason = "event_time_future_skew"
        elif max_event_time is not None and event_time < watermark:
            accepted = False
            disposition = "quarantine"
            reason = "event_arrived_after_watermark"
        elif max_event_time is not None and event_time < max_event_time:
            disposition = "out_of_order_within_bound"

        if prior_hash is None:
            seen[key] = payload_hash
            seen.move_to_end(key)
            while len(seen) > self.policy.dedupe_capacity:
                seen.popitem(last=False)
        if accepted:
            state["accepted_count"] += 1
            if max_event_time is None or event_time > max_event_time:
                state["max_event_time"] = event_time
                max_event_time = event_time
            watermark = max_event_time - timedelta(
                seconds=self.policy.allowed_lateness_seconds
            )
        elif disposition == "quarantine":
            state["quarantine_count"] += 1

        lateness = max((observed - event_time).total_seconds(), 0.0)
        material = {
            "accepted": accepted,
            "disposition": disposition,
            "reason": reason,
            "stream_id": str(stream_id),
            "event_id": key,
            "event_time_utc": _iso(event_time),
            "observed_at_utc": _iso(observed),
            "watermark_utc": _iso(watermark),
            "lateness_seconds": round(lateness, 6),
            "payload_sha256": payload_hash,
        }
        decision = EventTimeDecision(**material, receipt_sha256=_hash(material))
        return decision.to_dict()

    def stream_status(self, stream_id: str) -> dict[str, Any]:
        state = self._state(stream_id)
        max_event_time: datetime | None = state["max_event_time"]
        watermark = (
            max_event_time - timedelta(seconds=self.policy.allowed_lateness_seconds)
            if max_event_time is not None
            else None
        )
        return {
            "stream_id": str(stream_id),
            "max_event_time_utc": _iso(max_event_time) if max_event_time else "",
            "watermark_utc": _iso(watermark) if watermark else "",
            "dedupe_entries": len(state["seen"]),
            "accepted_count": int(state["accepted_count"]),
            "quarantine_count": int(state["quarantine_count"]),
        }

    def snapshot(self) -> dict[str, Any]:
        streams: dict[str, Any] = {}
        for stream_id, state in sorted(self._streams.items()):
            max_event_time: datetime | None = state["max_event_time"]
            streams[stream_id] = {
                "max_event_time_utc": _iso(max_event_time) if max_event_time else "",
                "seen": list(state["seen"].items()),
                "accepted_count": int(state["accepted_count"]),
                "quarantine_count": int(state["quarantine_count"]),
            }
        material = {
            "schema_version": 1,
            "policy": asdict(self.policy),
            "streams": streams,
        }
        return {**material, "snapshot_sha256": _hash(material)}

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "EventTimeGuard":
        material = {
            key: value
            for key, value in dict(snapshot).items()
            if key != "snapshot_sha256"
        }
        if str(snapshot.get("snapshot_sha256") or "") != _hash(material):
            raise ValueError("event-time snapshot checksum mismatch")
        if int(snapshot.get("schema_version") or 0) != 1:
            raise ValueError("event-time snapshot schema unsupported")
        guard = cls(EventTimePolicy(**dict(snapshot.get("policy") or {})))
        for stream_id, raw_state in dict(snapshot.get("streams") or {}).items():
            state = guard._state(str(stream_id))
            row = dict(raw_state or {})
            state["max_event_time"] = (
                _parse_utc(row.get("max_event_time_utc"))
                if row.get("max_event_time_utc")
                else None
            )
            state["seen"] = OrderedDict(
                (str(key), str(value)) for key, value in row.get("seen") or []
            )
            state["accepted_count"] = int(row.get("accepted_count") or 0)
            state["quarantine_count"] = int(row.get("quarantine_count") or 0)
        return guard
