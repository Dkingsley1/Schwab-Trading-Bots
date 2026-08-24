from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


TRANSITIONS = {
    ("NEW", "EXECUTION"): "EXECUTED",
    ("EXECUTED", "CONFIRMATION"): "CONFIRMED",
    ("CONFIRMED", "ALLOCATION"): "ALLOCATED",
    ("CONFIRMED", "SETTLEMENT"): "SETTLED",
    ("ALLOCATED", "SETTLEMENT"): "SETTLED",
    ("EXECUTED", "CANCELLATION"): "CANCELED",
    ("CONFIRMED", "CANCELLATION"): "CANCELED",
    ("ALLOCATED", "CANCELLATION"): "CANCELED",
    ("CONFIRMED", "EXERCISE"): "EXERCISED",
    ("ALLOCATED", "EXERCISE"): "EXERCISED",
    ("CONFIRMED", "EXPIRATION"): "EXPIRED",
    ("ALLOCATED", "EXPIRATION"): "EXPIRED",
}


@dataclass
class TradeLifecycle:
    trade_id: str
    product_id: str
    economics_sha256: str
    derivative: bool = False
    state: str = "NEW"
    receipts: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, event: Mapping[str, Any]) -> dict[str, Any]:
        event_id = str(event.get("event_id") or "").strip()
        event_type = str(event.get("event_type") or "").strip().upper()
        if not event_id or not event_type:
            return self._reject("event_identity_missing")
        if str(event.get("trade_id") or "") != self.trade_id:
            return self._reject("trade_identity_mismatch")
        if str(event.get("product_id") or "") != self.product_id:
            return self._reject("product_identity_mismatch")
        if str(event.get("economics_sha256") or "") != self.economics_sha256:
            return self._reject("economic_terms_mutated")

        event_hash = _digest(dict(event))
        for receipt in self.receipts:
            if receipt["event_id"] == event_id:
                if receipt["event_sha256"] == event_hash:
                    return {
                        "accepted": True,
                        "disposition": "duplicate_idempotent",
                        "receipt": receipt,
                    }
                return self._reject("duplicate_event_conflict")

        next_state = TRANSITIONS.get((self.state, event_type))
        if next_state is None:
            return self._reject("illegal_lifecycle_transition")
        if event_type in {"EXERCISE", "EXPIRATION"} and not self.derivative:
            return self._reject("derivative_event_for_non_derivative_product")

        before = {
            "trade_id": self.trade_id,
            "product_id": self.product_id,
            "economics_sha256": self.economics_sha256,
            "state": self.state,
        }
        after = {**before, "state": next_state}
        receipt = {
            "event_id": event_id,
            "event_type": event_type,
            "event_sha256": event_hash,
            "previous_receipt_sha256": (
                self.receipts[-1]["receipt_sha256"] if self.receipts else None
            ),
            "before_state_sha256": _digest(before),
            "after_state_sha256": _digest(after),
            "from_state": self.state,
            "to_state": next_state,
        }
        receipt["receipt_sha256"] = _digest(receipt)
        self.state = next_state
        self.receipts.append(receipt)
        return {"accepted": True, "disposition": "applied", "receipt": receipt}

    def verify_chain(self) -> dict[str, Any]:
        errors: list[str] = []
        previous = None
        for index, receipt in enumerate(self.receipts):
            row = dict(receipt)
            stored = row.pop("receipt_sha256", None)
            if _digest(row) != stored:
                errors.append(f"receipt_digest_mismatch:{index}")
            if receipt.get("previous_receipt_sha256") != previous:
                errors.append(f"receipt_parent_mismatch:{index}")
            previous = receipt.get("receipt_sha256")
        return {
            "ok": not errors,
            "errors": errors,
            "event_count": len(self.receipts),
            "state": self.state,
            "execution_authority": False,
        }

    @staticmethod
    def _reject(reason: str) -> dict[str, Any]:
        return {"accepted": False, "disposition": reason, "execution_authority": False}
