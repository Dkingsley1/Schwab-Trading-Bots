from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"filled", "canceled", "rejected", "expired"}
UNRESOLVED_STATES = {
    "reserved",
    "submitting",
    "submit_unknown",
    "acknowledged",
    "open",
    "partially_filled",
    "cancel_pending",
    "cancel_unknown",
}
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "reserved": {"submitting", "rejected"},
    "submitting": {"acknowledged", "submit_unknown", "rejected"},
    "submit_unknown": {"acknowledged", "open", "partially_filled", "filled", "cancel_pending", "canceled", "rejected", "expired"},
    "acknowledged": {"open", "partially_filled", "filled", "cancel_pending", "canceled", "rejected", "expired"},
    "open": {"partially_filled", "filled", "cancel_pending", "canceled", "rejected", "expired"},
    "partially_filled": {"partially_filled", "filled", "cancel_pending", "canceled", "expired"},
    "cancel_pending": {"open", "partially_filled", "filled", "canceled", "expired", "cancel_unknown"},
    "cancel_unknown": {"open", "partially_filled", "filled", "cancel_pending", "canceled", "expired"},
    "filled": set(),
    "canceled": set(),
    "rejected": set(),
    "expired": set(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_broker_state(raw: Any, *, filled_quantity: float = 0.0, requested_quantity: float = 0.0) -> str:
    status = str(raw or "").strip().upper().replace(" ", "_")
    aliases = {
        "PENDING_ACTIVATION": "acknowledged",
        "PENDING_ACKNOWLEDGEMENT": "acknowledged",
        "AWAITING_PARENT_ORDER": "acknowledged",
        "AWAITING_CONDITION": "acknowledged",
        "QUEUED": "open",
        "WORKING": "open",
        "ACCEPTED": "open",
        "PENDING_REPLACE": "open",
        "PENDING_CANCEL": "cancel_pending",
        "REPLACED": "open",
        "PARTIALLY_FILLED": "partially_filled",
        "PARTIAL_FILL": "partially_filled",
        "FILLED": "filled",
        "EXECUTED": "filled",
        "CANCELED": "canceled",
        "CANCELLED": "canceled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
    }
    normalized = aliases.get(status, "")
    if normalized == "open" and filled_quantity > 0.0:
        normalized = "partially_filled"
    if requested_quantity > 0.0 and filled_quantity >= requested_quantity:
        normalized = "filled"
    return normalized or "submit_unknown"


class LiveOrderLedger:
    """Durable, fail-closed order intent and broker-state ledger.

    A reserved intent is never submitted twice. If the process loses certainty after
    broker submission, the intent remains ``submit_unknown`` until reconciliation
    proves its broker state.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    broker_order_id TEXT NOT NULL DEFAULT '',
                    requested_quantity REAL NOT NULL DEFAULT 0,
                    filled_quantity REAL NOT NULL DEFAULT 0,
                    average_fill_price REAL NOT NULL DEFAULT 0,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT ''
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_order_intents_broker_order_id
                    ON order_intents(broker_order_id)
                    WHERE broker_order_id <> '';
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(intent_id) REFERENCES order_intents(intent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_order_events_intent_id ON order_events(intent_id, event_id);
                """
            )

    @staticmethod
    def _event_hash(event: dict[str, Any]) -> str:
        return _sha256_text(_canonical_json(event))

    def _append_event(
        self,
        conn: sqlite3.Connection,
        *,
        intent_id: str,
        from_state: str,
        to_state: str,
        details: dict[str, Any] | None = None,
        timestamp_utc: str | None = None,
    ) -> str:
        row = conn.execute("SELECT event_hash FROM order_events ORDER BY event_id DESC LIMIT 1").fetchone()
        previous_hash = str(row["event_hash"]) if row else ""
        event = {
            "intent_id": intent_id,
            "timestamp_utc": timestamp_utc or _utc_now(),
            "from_state": from_state,
            "to_state": to_state,
            "details": details or {},
            "previous_event_hash": previous_hash,
        }
        event_hash = self._event_hash(event)
        conn.execute(
            """
            INSERT INTO order_events (
                intent_id, timestamp_utc, from_state, to_state,
                details_json, previous_event_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                event["timestamp_utc"],
                from_state,
                to_state,
                _canonical_json(event["details"]),
                previous_hash,
                event_hash,
            ),
        )
        return event_hash

    def reserve(
        self,
        *,
        intent_id: str,
        payload: dict[str, Any],
        requested_quantity: float = 0.0,
    ) -> dict[str, Any]:
        key = str(intent_id or "").strip()
        if not key:
            raise ValueError("intent_id is required")
        payload_json = _canonical_json(payload)
        payload_hash = _sha256_text(payload_json)
        now = _utc_now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM order_intents WHERE intent_id = ?", (key,)).fetchone()
            if existing is not None:
                conflict = str(existing["payload_hash"]) != payload_hash
                conn.rollback()
                return {
                    "reserved": False,
                    "duplicate": not conflict,
                    "conflict": conflict,
                    "intent_id": key,
                    "state": str(existing["state"]),
                    "broker_order_id": str(existing["broker_order_id"]),
                    "reason": "intent_payload_conflict" if conflict else "intent_already_reserved",
                }
            conn.execute(
                """
                INSERT INTO order_intents (
                    intent_id, payload_hash, payload_json, state, requested_quantity,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, 'reserved', ?, ?, ?)
                """,
                (key, payload_hash, payload_json, max(float(requested_quantity or 0.0), 0.0), now, now),
            )
            self._append_event(
                conn,
                intent_id=key,
                from_state="",
                to_state="reserved",
                details={"payload_hash": payload_hash},
                timestamp_utc=now,
            )
            conn.commit()
            return {
                "reserved": True,
                "duplicate": False,
                "conflict": False,
                "intent_id": key,
                "state": "reserved",
                "broker_order_id": "",
                "reason": "reserved",
            }
        finally:
            conn.close()

    def transition(
        self,
        *,
        intent_id: str,
        to_state: str,
        broker_order_id: str = "",
        filled_quantity: float | None = None,
        average_fill_price: float | None = None,
        last_error: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = str(intent_id or "").strip()
        target = str(to_state or "").strip().lower()
        if target not in ALLOWED_TRANSITIONS:
            raise ValueError(f"unsupported order state: {target}")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM order_intents WHERE intent_id = ?", (key,)).fetchone()
            if row is None:
                conn.rollback()
                raise KeyError(f"unknown intent_id: {key}")
            current = str(row["state"])
            if target == current:
                conn.rollback()
                return self._row_dict(row)
            if target not in ALLOWED_TRANSITIONS.get(current, set()):
                conn.rollback()
                raise ValueError(f"illegal order transition: {current}->{target}")
            now = _utc_now()
            broker_id = str(broker_order_id or row["broker_order_id"] or "").strip()
            filled = float(row["filled_quantity"] if filled_quantity is None else max(float(filled_quantity), 0.0))
            average = float(row["average_fill_price"] if average_fill_price is None else max(float(average_fill_price), 0.0))
            conn.execute(
                """
                UPDATE order_intents
                SET state = ?, broker_order_id = ?, filled_quantity = ?,
                    average_fill_price = ?, updated_at_utc = ?, last_error = ?
                WHERE intent_id = ?
                """,
                (target, broker_id, filled, average, now, str(last_error or ""), key),
            )
            self._append_event(
                conn,
                intent_id=key,
                from_state=current,
                to_state=target,
                details={
                    **(details or {}),
                    "broker_order_id": broker_id,
                    "filled_quantity": filled,
                    "average_fill_price": average,
                    "last_error": str(last_error or ""),
                },
                timestamp_utc=now,
            )
            updated = conn.execute("SELECT * FROM order_intents WHERE intent_id = ?", (key,)).fetchone()
            conn.commit()
            return self._row_dict(updated)
        finally:
            conn.close()

    def mark_submitting(self, intent_id: str) -> dict[str, Any]:
        return self.transition(intent_id=intent_id, to_state="submitting")

    def mark_submit_result(
        self,
        *,
        intent_id: str,
        acknowledged: bool,
        broker_order_id: str = "",
        error: str = "",
        definitively_rejected: bool = False,
    ) -> dict[str, Any]:
        confirmed = bool(acknowledged and str(broker_order_id or "").strip())
        return self.transition(
            intent_id=intent_id,
            to_state="acknowledged" if confirmed else "rejected" if definitively_rejected else "submit_unknown",
            broker_order_id=broker_order_id,
            last_error=error,
            details={
                "broker_acknowledged": confirmed,
                "definitively_rejected": bool(definitively_rejected),
            },
        )

    def record_broker_update(
        self,
        *,
        broker_order_id: str,
        broker_status: Any,
        filled_quantity: float = 0.0,
        average_fill_price: float = 0.0,
    ) -> dict[str, Any]:
        broker_id = str(broker_order_id or "").strip()
        if not broker_id:
            raise ValueError("broker_order_id is required")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM order_intents WHERE broker_order_id = ?", (broker_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown broker_order_id: {broker_id}")
        target = normalize_broker_state(
            broker_status,
            filled_quantity=max(float(filled_quantity or 0.0), 0.0),
            requested_quantity=max(float(row["requested_quantity"] or 0.0), 0.0),
        )
        current = str(row["state"])
        if target == "submit_unknown" and current != "submitting":
            return self._row_dict(row)
        return self.transition(
            intent_id=str(row["intent_id"]),
            to_state=target,
            broker_order_id=broker_id,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            details={"broker_status": str(broker_status or "")},
        )

    def mark_cancel_pending(self, broker_order_id: str) -> dict[str, Any]:
        row = self.get_by_broker_order_id(broker_order_id)
        if not row:
            raise KeyError(f"unknown broker_order_id: {broker_order_id}")
        return self.transition(intent_id=str(row["intent_id"]), to_state="cancel_pending", broker_order_id=broker_order_id)

    def mark_cancel_unknown(self, broker_order_id: str, *, error: str = "") -> dict[str, Any]:
        row = self.get_by_broker_order_id(broker_order_id)
        if not row:
            raise KeyError(f"unknown broker_order_id: {broker_order_id}")
        return self.transition(
            intent_id=str(row["intent_id"]),
            to_state="cancel_unknown",
            broker_order_id=broker_order_id,
            last_error=error,
            details={"cancel_outcome_known": False},
        )

    def reconcile_ambiguous(
        self,
        *,
        intent_id: str,
        resolution: str,
        evidence: str,
        broker_order_id: str = "",
        filled_quantity: float = 0.0,
        average_fill_price: float = 0.0,
    ) -> dict[str, Any]:
        key = str(intent_id or "").strip()
        proof = str(evidence or "").strip()
        requested = str(resolution or "").strip().lower()
        if len(proof) < 12:
            raise ValueError("reconciliation evidence must be at least 12 characters")
        row = self.get(key)
        if not row:
            raise KeyError(f"unknown intent_id: {key}")
        current = str(row.get("state") or "")
        if current not in {"submitting", "submit_unknown", "cancel_unknown"}:
            raise ValueError(f"intent is not ambiguous: {current}")

        if requested == "not_submitted":
            target = "open" if current == "cancel_unknown" else "rejected"
        else:
            target = requested
        allowed_resolutions = {"open", "partially_filled", "filled", "canceled", "rejected", "expired"}
        if target not in allowed_resolutions:
            raise ValueError(f"unsupported reconciliation resolution: {requested}")

        broker_id = str(broker_order_id or row.get("broker_order_id") or "").strip()
        if target in {"open", "partially_filled", "filled"} and not broker_id:
            raise ValueError(f"broker_order_id is required for resolution: {target}")
        if current == "submitting" and target != "rejected":
            self.transition(
                intent_id=key,
                to_state="submit_unknown",
                last_error="reconciliation_required_after_interrupted_submit",
                details={"manual_reconciliation_started": True, "evidence": proof},
            )
        return self.transition(
            intent_id=key,
            to_state=target,
            broker_order_id=broker_id,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            details={
                "manual_reconciliation": True,
                "requested_resolution": requested,
                "evidence": proof,
            },
        )

    @staticmethod
    def _row_dict(row: sqlite3.Row | None) -> dict[str, Any]:
        return dict(row) if row is not None else {}

    def get(self, intent_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM order_intents WHERE intent_id = ?", (str(intent_id or "").strip(),)).fetchone()
        return self._row_dict(row)

    def get_by_broker_order_id(self, broker_order_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM order_intents WHERE broker_order_id = ?",
                (str(broker_order_id or "").strip(),),
            ).fetchone()
        return self._row_dict(row)

    def unresolved(self) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in UNRESOLVED_STATES)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM order_intents WHERE state IN ({placeholders}) ORDER BY created_at_utc",
                tuple(sorted(UNRESOLVED_STATES)),
            ).fetchall()
        return [self._row_dict(row) for row in rows]

    def verify_event_chain(self) -> dict[str, Any]:
        errors: list[str] = []
        previous = ""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM order_events ORDER BY event_id").fetchall()
        for row in rows:
            details = json.loads(str(row["details_json"] or "{}"))
            event = {
                "intent_id": str(row["intent_id"]),
                "timestamp_utc": str(row["timestamp_utc"]),
                "from_state": str(row["from_state"]),
                "to_state": str(row["to_state"]),
                "details": details,
                "previous_event_hash": str(row["previous_event_hash"]),
            }
            if str(row["previous_event_hash"]) != previous:
                errors.append(f"previous_hash_mismatch_event={row['event_id']}")
            expected = self._event_hash(event)
            if str(row["event_hash"]) != expected:
                errors.append(f"event_hash_mismatch_event={row['event_id']}")
            previous = str(row["event_hash"])
        return {
            "ok": not errors,
            "event_count": len(rows),
            "chain_head": previous,
            "errors": errors,
            "unresolved_count": len(self.unresolved()),
        }
