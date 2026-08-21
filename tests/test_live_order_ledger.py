import sqlite3
from pathlib import Path

import pytest

from core.live_order_ledger import LiveOrderLedger
from scripts.ops import live_order_ledger_control as control


def _reserve(ledger: LiveOrderLedger, intent_id: str = "decision-1", quantity: float = 10.0) -> dict:
    return ledger.reserve(
        intent_id=intent_id,
        payload={"symbol": "AAPL", "action": "BUY", "quantity": quantity},
        requested_quantity=quantity,
    )


def test_reservation_is_transactionally_idempotent_and_detects_conflicts(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")

    assert _reserve(ledger)["reserved"] is True
    duplicate = _reserve(ledger)
    conflict = ledger.reserve(
        intent_id="decision-1",
        payload={"symbol": "MSFT", "action": "BUY", "quantity": 10.0},
        requested_quantity=10.0,
    )

    assert duplicate["reserved"] is False
    assert duplicate["duplicate"] is True
    assert duplicate["reason"] == "intent_already_reserved"
    assert conflict["conflict"] is True
    assert conflict["reason"] == "intent_payload_conflict"


def test_unknown_submit_cannot_be_reserved_or_submitted_again(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")
    unknown = ledger.mark_submit_result(intent_id="decision-1", acknowledged=False, error="timeout")

    assert unknown["state"] == "submit_unknown"
    assert _reserve(ledger)["reason"] == "intent_already_reserved"
    payload = control.build_payload(tmp_path, ledger_path=ledger.path)
    assert payload["ok"] is False
    assert payload["submit_unknown_count"] == 1
    assert "broker_submit_outcome_unknown" in payload["blockers"]


def test_deterministic_client_rejection_is_terminal_not_ambiguous(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")

    rejected = ledger.mark_submit_result(
        intent_id="decision-1",
        acknowledged=False,
        error="http_status_401",
        definitively_rejected=True,
    )

    assert rejected["state"] == "rejected"
    assert ledger.unresolved() == []


def test_ambiguous_submit_reconciliation_requires_evidence(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")
    ledger.mark_submit_result(intent_id="decision-1", acknowledged=False, error="timeout")

    try:
        ledger.reconcile_ambiguous(
            intent_id="decision-1",
            resolution="not_submitted",
            evidence="short",
        )
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("short reconciliation evidence should fail closed")

    reconciled = ledger.reconcile_ambiguous(
        intent_id="decision-1",
        resolution="not_submitted",
        evidence="Broker order history proves no order was accepted",
    )
    assert reconciled["state"] == "rejected"
    assert ledger.verify_event_chain()["ok"] is True


def test_broker_updates_reconcile_partial_fill_and_terminal_fill(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")
    acknowledged = ledger.mark_submit_result(
        intent_id="decision-1",
        acknowledged=True,
        broker_order_id="broker-1",
    )
    assert acknowledged["state"] == "acknowledged"

    partial = ledger.record_broker_update(
        broker_order_id="broker-1",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=4.0,
        average_fill_price=100.25,
    )
    filled = ledger.record_broker_update(
        broker_order_id="broker-1",
        broker_status="FILLED",
        filled_quantity=10.0,
        average_fill_price=100.30,
    )

    assert partial["state"] == "partially_filled"
    assert partial["filled_quantity"] == 4.0
    assert filled["state"] == "filled"
    assert ledger.unresolved() == []
    assert ledger.verify_event_chain()["ok"] is True


def test_repeated_partial_fill_is_a_monotonic_material_update(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")
    ledger.mark_submit_result(intent_id="decision-1", acknowledged=True, broker_order_id="broker-1")
    ledger.record_broker_update(
        broker_order_id="broker-1",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=2.0,
        average_fill_price=100.0,
    )
    updated = ledger.record_broker_update(
        broker_order_id="broker-1",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=7.0,
        average_fill_price=100.1,
    )

    assert updated["state"] == "partially_filled"
    assert updated["filled_quantity"] == 7.0
    assert ledger.verify_integrity()["ok"] is True


def test_fill_regression_overfill_and_broker_identity_mutation_fail_immediately(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")
    ledger.mark_submit_result(intent_id="decision-1", acknowledged=True, broker_order_id="broker-1")
    ledger.record_broker_update(
        broker_order_id="broker-1",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=4.0,
        average_fill_price=100.0,
    )

    with pytest.raises(ValueError, match="cannot_decrease"):
        ledger.transition(intent_id="decision-1", to_state="partially_filled", filled_quantity=3.0)
    with pytest.raises(ValueError, match="cannot_exceed"):
        ledger.transition(intent_id="decision-1", to_state="filled", filled_quantity=11.0)
    with pytest.raises(ValueError, match="immutable"):
        ledger.transition(
            intent_id="decision-1",
            to_state="partially_filled",
            broker_order_id="different-broker-id",
        )


def test_ambiguous_cancel_requires_broker_reconciliation(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")
    ledger.mark_submit_result(intent_id="decision-1", acknowledged=True, broker_order_id="broker-1")
    ledger.record_broker_update(broker_order_id="broker-1", broker_status="WORKING")
    ledger.mark_cancel_pending("broker-1")
    pending_payload = control.build_payload(tmp_path, ledger_path=ledger.path)
    assert pending_payload["ok"] is False
    assert "broker_cancel_pending_reconciliation" in pending_payload["blockers"]
    unknown = ledger.mark_cancel_unknown("broker-1", error="network_lost")

    assert unknown["state"] == "cancel_unknown"
    payload = control.build_payload(tmp_path, ledger_path=ledger.path)
    assert payload["ok"] is False
    assert payload["cancel_unknown_count"] == 1
    assert "broker_cancel_outcome_unknown" in payload["blockers"]

    reconciled = ledger.record_broker_update(broker_order_id="broker-1", broker_status="WORKING")
    assert reconciled["state"] == "open"


def test_event_chain_tampering_is_detected(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")

    with sqlite3.connect(str(ledger.path)) as conn:
        conn.execute("UPDATE order_events SET details_json = ? WHERE event_id = 1", ('{"tampered":true}',))

    integrity = ledger.verify_event_chain()
    assert integrity["ok"] is False
    assert any("event_hash_mismatch" in error for error in integrity["errors"])


def test_materialized_intent_state_tampering_is_detected(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)
    ledger.mark_submitting("decision-1")

    with sqlite3.connect(str(ledger.path)) as conn:
        conn.execute("UPDATE order_intents SET state = 'filled' WHERE intent_id = 'decision-1'")

    integrity = ledger.verify_integrity()
    assert integrity["ok"] is False
    assert integrity["state_mismatch_count"] == 1
    assert "intent_materialized_state_mismatch:decision-1" in integrity["errors"]


def test_payload_hash_tampering_is_detected_by_full_integrity_probe(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    _reserve(ledger)

    with sqlite3.connect(str(ledger.path)) as conn:
        conn.execute("UPDATE order_intents SET payload_json = '{\"tampered\":true}' WHERE intent_id = 'decision-1'")

    integrity = ledger.verify_integrity()
    payload = control.build_payload(tmp_path, ledger_path=ledger.path)
    assert integrity["ok"] is False
    assert integrity["payload_hash_mismatch_count"] == 1
    assert payload["ok"] is False
    assert "order_ledger_integrity_invalid" in payload["blockers"]


def test_ledger_enforces_wal_full_sync_and_foreign_keys(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")

    integrity = ledger.verify_integrity()

    assert integrity["ok"] is True
    assert integrity["sqlite"]["journal_mode"] == "wal"
    assert integrity["sqlite"]["synchronous"] >= 2
    assert integrity["sqlite"]["foreign_key_error_count"] == 0
