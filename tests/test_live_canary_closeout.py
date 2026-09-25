from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.live_execution_envelope import build_live_execution_envelope
from core.live_order_ledger import LiveOrderLedger
from core.order_intent import build_order_intent_evidence
from scripts.ops.live_canary_closeout import build_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads(
    (PROJECT_ROOT / "config" / "live_canary_micro_policy_v1.json").read_text()
)
POLICY = json.loads(
    (PROJECT_ROOT / "config" / "live_canary_graduation_v1.json").read_text()
)
CANDIDATE_ID = "candidate-closeout-test"
ACCOUNT_REFERENCE = "opaque-live-account-reference"
BROKER_ORDER_ID = "broker-order-reference-that-must-remain-private"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _paths(tmp_path: Path) -> dict[str, Path]:
    policy_path = tmp_path / "config" / "live_canary_graduation_v1.json"
    plan_path = tmp_path / "config" / "live_canary_micro_policy_v1.json"
    ledger_path = tmp_path / "governance" / "runtime" / "live_orders.sqlite3"
    receipts_path = tmp_path / "governance" / "evidence" / "closeouts.jsonl"
    account_study_path = (
        tmp_path / "governance" / "health" / "account_position_study_latest.json"
    )
    _write(policy_path, POLICY)
    _write(plan_path, PLAN)
    return {
        "policy": policy_path,
        "plan": plan_path,
        "ledger": ledger_path,
        "receipts": receipts_path,
        "account_study": account_study_path,
    }


def _filled_entry(ledger_path: Path) -> tuple[str, datetime]:
    intent_id = "live-canary-entry-one"
    price = 30.0
    quote_time = datetime.now(timezone.utc)
    preflight = {
        "ready": True,
        "receipt_sha256": "a" * 64,
        "account_policy_key": PLAN["account_policy_key"],
        "execution_route_id": PLAN["execution_route_id"],
        "settled_cash_broker_visible_usd": 200.0,
        "regime_bucket": "risk_on",
        "regime_receipt_sha256": "b" * 64,
    }
    intent = build_order_intent_evidence(
        decision_id=intent_id,
        symbol="SCHD",
        action="BUY",
        quantity=1.0,
        strategy=PLAN["execution_route_id"],
        asset_type="EQUITY",
        limit_price=price,
        quote_snapshot={
            "timestamp_utc": quote_time.isoformat(),
            "bid_price": price,
            "ask_price": price,
            "last_price": price,
            "spread_bps": 1.0,
            "quote_age_ms": 0.0,
            "source_provider": "schwab_api",
            "source_venue": "XNYS",
            "snapshot_id": "snapshot-closeout-test",
        },
        expected_fill={"expected_fill_price": price},
        risk_decision={"ok": True, "gate": "ready", "reason": "ready"},
    )
    order_request = {
        "symbol": "SCHD",
        "action": "BUY",
        "quantity": 1.0,
        "asset_type": "EQUITY",
        "limit_price": price,
        "account_reference": ACCOUNT_REFERENCE,
        "order_spec": {"orderType": "LIMIT", "price": "30.00"},
    }
    envelope = build_live_execution_envelope(
        intent_evidence=intent,
        order_request=order_request,
        candidate_id=CANDIDATE_ID,
        broker="schwab",
        account_reference=ACCOUNT_REFERENCE,
        account_snapshot_evidence={
            "broker_position_snapshot_sha256": "c" * 64,
            "broker_position_snapshot_captured_at_utc": quote_time.isoformat(),
            "broker_position_snapshot_quantity": 0.0,
            "live_canary_preflight_receipt": preflight,
        },
        policy_sha256="d" * 64,
        ttl_seconds=60.0,
        created_at_utc=quote_time,
    )
    ledger = LiveOrderLedger(ledger_path)
    reservation = ledger.reserve(
        intent_id=intent_id,
        requested_quantity=1.0,
        payload={
            "broker": "schwab",
            "account_reference": ACCOUNT_REFERENCE,
            "symbol": "SCHD",
            "action": "BUY",
            "quantity": 1.0,
            "order_spec": order_request["order_spec"],
            "mode_invariant_intent": intent,
            "live_execution_envelope": envelope,
        },
    )
    assert reservation["reserved"] is True
    ledger.mark_submitting(intent_id)
    ledger.mark_submit_result(
        intent_id=intent_id,
        acknowledged=True,
        broker_order_id=BROKER_ORDER_ID,
    )
    ledger.record_broker_update(
        broker_order_id=BROKER_ORDER_ID,
        broker_status="FILLED",
        filled_quantity=1.0,
        average_fill_price=price,
    )
    filled_at = datetime.fromisoformat(
        ledger.events(intent_id=intent_id)[-1]["timestamp_utc"]
    ).astimezone(timezone.utc)
    return intent_id, filled_at


def _account_study(timestamp: datetime, *, cash_usd: float = 169.98) -> dict:
    return {
        "schema_version": 3,
        "timestamp_utc": timestamp.isoformat(),
        "accounts": [
            {
                "account_policy_key": PLAN["account_policy_key"],
                "borrowing_allowed": False,
                "flags": {"closing_only": False},
                "account_capability_truth": {
                    "balance_truth": {
                        "cash_balance": cash_usd,
                        "cash_available_for_trading": 1000.0,
                        "pending_deposits": 0.0,
                    },
                    "debit_truth": {
                        "accrued_interest": 0.0,
                        "interest_bearing_borrowing_confirmed": False,
                    },
                    "broker_call_truth": {"in_call": False},
                    "position_collateral_truth": {"uncovered_short_option_count": 0},
                },
            }
        ],
        "positions": [
            {
                "account_policy_key": PLAN["account_policy_key"],
                "asset_type": "EQUITY",
                "symbol": "SCHD",
                "quantity": 1.0,
            }
        ],
    }


def _build(tmp_path: Path, paths: dict[str, Path], **kwargs: object) -> dict:
    return build_payload(
        tmp_path,
        policy_path=paths["policy"],
        plan_path=paths["plan"],
        ledger_path=paths["ledger"],
        receipts_path=paths["receipts"],
        account_study_path=paths["account_study"],
        **kwargs,
    )


def test_missing_first_canary_is_ready_idle_not_degradation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    payload = _build(tmp_path, paths)

    assert payload["control_ok"] is True
    assert payload["overall_status"] == "ready_idle"
    assert payload["phase"] == "awaiting_filled_canary"
    assert payload["capture_eligible"] is False
    assert payload["live_execution_authority"] is False


def test_closeout_requires_fresh_exact_account_truth_and_redacts_ids(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    intent_id, filled_at = _filled_entry(paths["ledger"])
    account_at = filled_at + timedelta(seconds=1)
    _write(paths["account_study"], _account_study(account_at))

    preview = _build(
        tmp_path,
        paths,
        intent_id=intent_id,
        now=account_at + timedelta(seconds=1),
    )
    captured = _build(
        tmp_path,
        paths,
        intent_id=intent_id,
        capture=True,
        now=account_at + timedelta(seconds=2),
    )

    assert preview["overall_status"] == "ready_to_capture"
    assert preview["capture_eligible"] is True
    assert preview["reconciliation"]["post_cash_source"] == "broker_cash_balance"
    assert captured["overall_status"] == "captured"
    assert captured["captured"] is True
    persisted = paths["receipts"].read_text(encoding="utf-8")
    assert ACCOUNT_REFERENCE not in persisted
    assert BROKER_ORDER_ID not in persisted
    assert paths["receipts"].stat().st_mode & 0o077 == 0

    duplicate = _build(
        tmp_path,
        paths,
        intent_id=intent_id,
        capture=True,
        now=account_at + timedelta(seconds=3),
    )
    assert duplicate["control_ok"] is False
    assert "closeout_receipt_already_exists_for_intent" in duplicate["blockers"]
    assert len(paths["receipts"].read_text(encoding="utf-8").splitlines()) == 1


def test_account_study_that_predates_fill_cannot_be_captured(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    intent_id, filled_at = _filled_entry(paths["ledger"])
    account_at = filled_at - timedelta(seconds=1)
    _write(paths["account_study"], _account_study(account_at))

    payload = _build(
        tmp_path,
        paths,
        intent_id=intent_id,
        capture=True,
        now=filled_at + timedelta(seconds=2),
    )

    assert payload["control_ok"] is True
    assert payload["overall_status"] == "reconciliation_pending"
    assert payload["captured"] is False
    assert "account_position_study_predates_fill" in payload["pending_evidence"]
    assert not paths["receipts"].exists()
