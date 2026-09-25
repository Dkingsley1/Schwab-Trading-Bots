from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.live_canary_graduation import (
    build_live_canary_closeout_receipt,
    evaluate_live_canary_graduation,
)
from core.live_execution_envelope import build_live_execution_envelope
from core.live_order_ledger import LiveOrderLedger
from core.order_intent import build_order_intent_evidence, canonical_payload_sha256
from scripts.ops.live_canary_preflight import CONFIRMATION_PHRASE, _issue_allowlist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads(
    (PROJECT_ROOT / "config" / "live_canary_micro_policy_v1.json").read_text()
)
POLICY = json.loads(
    (PROJECT_ROOT / "config" / "live_canary_graduation_v1.json").read_text()
)
CANDIDATE_ID = "candidate-live-canary-test"
ACCOUNT_REFERENCE = "opaque-test-account-reference"


def _filled_order(
    ledger: LiveOrderLedger,
    *,
    intent_id: str,
    action: str,
    price: float,
    filled_at_utc: str,
    regime: str,
    cash_reconciled: bool = True,
) -> dict:
    symbol = "SCHD"
    quantity = 1.0
    preflight = {
        "ready": True,
        "receipt_sha256": "a" * 64,
        "account_policy_key": PLAN["account_policy_key"],
        "execution_route_id": PLAN["execution_route_id"],
    }
    intent = build_order_intent_evidence(
        decision_id=intent_id,
        symbol=symbol,
        action=action,
        quantity=quantity,
        strategy=PLAN["execution_route_id"],
        asset_type="EQUITY",
        limit_price=price,
        quote_snapshot={
            "timestamp_utc": filled_at_utc,
            "bid_price": price,
            "ask_price": price,
            "last_price": price,
            "spread_bps": 1.0,
            "quote_age_ms": 0.0,
            "source_provider": "schwab_api",
            "source_venue": "XNYS",
            "snapshot_id": "snapshot-1",
        },
        expected_fill={"expected_fill_price": price},
        risk_decision={"ok": True, "gate": "ready", "reason": "ready"},
    )
    order_request = {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "asset_type": "EQUITY",
        "limit_price": price,
        "account_reference": ACCOUNT_REFERENCE,
        "order_spec": {"orderType": "LIMIT", "price": f"{price:.2f}"},
    }
    envelope = build_live_execution_envelope(
        intent_evidence=intent,
        order_request=order_request,
        candidate_id=CANDIDATE_ID,
        broker="schwab",
        account_reference=ACCOUNT_REFERENCE,
        account_snapshot_evidence={
            "broker_position_snapshot_sha256": "b" * 64,
            "broker_position_snapshot_captured_at_utc": filled_at_utc,
            "broker_position_snapshot_quantity": 0.0,
            "live_canary_preflight_receipt": preflight,
        },
        policy_sha256="c" * 64,
        ttl_seconds=60.0,
        created_at_utc=datetime.fromisoformat(filled_at_utc),
    )
    reservation = ledger.reserve(
        intent_id=intent_id,
        payload={
            "broker": "schwab",
            "account_reference": ACCOUNT_REFERENCE,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_spec": order_request["order_spec"],
            "mode_invariant_intent": intent,
            "live_execution_envelope": envelope,
        },
        requested_quantity=quantity,
    )
    assert reservation["reserved"] is True
    ledger.mark_submitting(intent_id)
    broker_order_id = f"broker-{intent_id}"
    ledger.mark_submit_result(
        intent_id=intent_id,
        acknowledged=True,
        broker_order_id=broker_order_id,
    )
    row = ledger.record_broker_update(
        broker_order_id=broker_order_id,
        broker_status="FILLED",
        filled_quantity=quantity,
        average_fill_price=price,
    )
    final_event = ledger.events(intent_id=intent_id)[-1]
    is_entry = action == "BUY"
    return build_live_canary_closeout_receipt(
        intent_id=intent_id,
        intent_payload_sha256=row["payload_hash"],
        final_order_event_sha256=final_event["event_hash"],
        broker_order_id=broker_order_id,
        candidate_id=CANDIDATE_ID,
        account_policy_key=PLAN["account_policy_key"],
        execution_route_id=PLAN["execution_route_id"],
        account_reference_sha256=envelope["component_hashes"][
            "account_reference_sha256"
        ],
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=price,
        filled_at_utc=filled_at_utc,
        broker_fees_usd=0.01,
        fee_evidence_source="broker_transaction_receipt",
        pre_position_quantity=0.0 if is_entry else 1.0,
        post_position_quantity=1.0 if is_entry else 0.0,
        pre_settled_cash_usd=200.0,
        post_settled_cash_usd=200.0 - price if is_entry else 200.0 + price,
        order_reconciled=True,
        position_reconciled=True,
        cash_reconciled=cash_reconciled,
        position_delta_verified=True,
        reduce_only_exit=not is_entry,
        regime_bucket=regime,
        regime_receipt_sha256="d" * 64,
        benchmark_return_bps=5.0 if not is_entry else None,
        benchmark_receipt_sha256="e" * 64 if not is_entry else "",
        account_study_sha256="f" * 64,
        account_study_timestamp_utc=filled_at_utc,
    )


def _evaluate(ledger: LiveOrderLedger, receipts: list[dict]) -> dict:
    return evaluate_live_canary_graduation(
        canary_plan=PLAN,
        graduation_policy=POLICY,
        order_intents=ledger.intents(),
        order_events=ledger.events(),
        closeout_receipts=receipts,
        ledger_integrity=ledger.verify_integrity(),
        current_candidate_id=CANDIDATE_ID,
    )


def test_missing_earned_evidence_is_ready_idle_not_degradation(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    payload = _evaluate(ledger, [])
    assert payload["control_ok"] is True
    assert payload["overall_status"] == "ready_idle"
    assert payload["phase"] == "awaiting_first_canary"
    assert payload["pending_evidence"] == ["first_live_canary_not_run"]
    assert payload["live_execution_authority"] is False


def test_billion_scale_is_cataloged_but_institutional_classes_fail_closed(
    tmp_path: Path,
) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    payload = _evaluate(ledger, [])
    ladder = payload["capital_ladder"]
    tiers = {row["tier"]: row for row in ladder["evaluations"]}

    assert ladder["cataloged_tier_count"] == 22
    assert ladder["maximum_cataloged_capital_usd"] == 1_000_000_000
    assert ladder["enabled_operating_classes"] == ["personal_brokerage"]
    assert (
        tiers["micro_validation"]["scale_governance"]["operating_class_enabled"] is True
    )
    assert (
        tiers["advanced_1000000"]["scale_governance"]["operating_class_enabled"]
        is False
    )
    billion = tiers["large_institutional_1000000000"]
    assert billion["operator_review_eligible"] is False
    assert billion["scale_governance"]["operating_class"] == "large_institutional"
    assert billion["scale_governance"]["missing_controls"]
    assert billion["requirements"]["operating_class_enabled"] is False
    assert billion["requirements"]["required_scale_controls_evidenced"] is False
    assert payload["live_execution_authority"] is False


def test_incomplete_closeout_stays_in_reconciliation_without_graduating(
    tmp_path: Path,
) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    receipt = _filled_order(
        ledger,
        intent_id="entry-pending-cash",
        action="BUY",
        price=30.0,
        filled_at_utc="2026-08-24T15:00:00+00:00",
        regime="risk_on",
        cash_reconciled=False,
    )
    payload = _evaluate(ledger, [receipt])
    assert payload["control_ok"] is True
    assert payload["phase"] == "first_canary_reconciliation"
    assert payload["first_canary"]["reconciled"] is False
    assert "closeout_cash_reconciliation_pending" in payload["pending_evidence"]


def test_first_reconciled_fill_never_unlocks_follow_on_order_automatically(
    tmp_path: Path,
) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    receipt = _filled_order(
        ledger,
        intent_id="entry-one",
        action="BUY",
        price=30.0,
        filled_at_utc="2026-08-24T15:00:00+00:00",
        regime="risk_on",
    )
    payload = _evaluate(ledger, [receipt])
    assert payload["first_canary"]["reconciled"] is True
    assert payload["first_canary"]["automatic_follow_on_order_allowed"] is False
    assert payload["phase"] == "round_trip_and_economic_validation"
    assert payload["stage_progression"]["highest_completed_stage"] == 0


def test_three_positive_reconciled_round_trips_reach_stage_two_review_only(
    tmp_path: Path,
) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    receipts: list[dict] = []
    for index, day in enumerate((24, 25, 26), start=1):
        regime = "risk_on" if index != 2 else "defensive"
        receipts.append(
            _filled_order(
                ledger,
                intent_id=f"entry-{index}",
                action="BUY",
                price=30.0,
                filled_at_utc=f"2026-08-{day:02d}T14:30:00+00:00",
                regime=regime,
            )
        )
        receipts.append(
            _filled_order(
                ledger,
                intent_id=f"exit-{index}",
                action="SELL",
                price=30.2,
                filled_at_utc=f"2026-08-{day:02d}T19:30:00+00:00",
                regime=regime,
            )
        )
    payload = _evaluate(ledger, receipts)
    assert payload["control_ok"] is True
    assert payload["metrics"]["reconciled_round_trip_count"] == 3
    assert payload["metrics"]["independent_trading_day_count"] == 3
    assert payload["metrics"]["distinct_regime_bucket_count"] == 2
    assert payload["metrics"]["total_post_cost_pnl_usd"] > 0.0
    assert payload["stage_progression"]["highest_completed_stage"] == 1
    assert payload["phase"] == "stage_2_operator_review"
    assert payload["stage_progression"]["automatic_stage_progression"] is False
    assert payload["capital_ladder"]["automatic_scaling"] is False
    assert (
        payload["capital_ladder"]["evaluations"][1]["operator_review_eligible"] is False
    )


def test_tampered_closeout_receipt_fails_closed(tmp_path: Path) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    receipt = _filled_order(
        ledger,
        intent_id="entry-tampered",
        action="BUY",
        price=30.0,
        filled_at_utc="2026-08-24T15:00:00+00:00",
        regime="risk_on",
    )
    receipt["fill_price"] = 1.0
    payload = _evaluate(ledger, [receipt])
    assert payload["control_ok"] is False
    assert payload["phase"] == "blocked"
    assert "closeout_receipt_hash_mismatch" in payload["blockers"]
    assert "closeout_receipt_fill_price_mismatch" in payload["blockers"]


def test_unsealed_regime_label_cannot_count_as_regime_evidence(
    tmp_path: Path,
) -> None:
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    entry = _filled_order(
        ledger,
        intent_id="entry-unsealed-regime",
        action="BUY",
        price=30.0,
        filled_at_utc="2026-08-24T15:00:00+00:00",
        regime="risk_on",
    )
    exit_receipt = _filled_order(
        ledger,
        intent_id="exit-unsealed-regime",
        action="SELL",
        price=30.2,
        filled_at_utc="2026-08-24T19:30:00+00:00",
        regime="risk_on",
    )
    for receipt in (entry, exit_receipt):
        receipt["regime_receipt_sha256"] = ""
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = canonical_payload_sha256(unsigned)

    payload = _evaluate(ledger, [entry, exit_receipt])
    assert payload["control_ok"] is True
    assert payload["metrics"]["reconciled_round_trip_count"] == 1
    assert payload["metrics"]["distinct_regime_bucket_count"] == 0
    assert "regime_receipt_evidence_pending" in payload["pending_evidence"]


def test_stage_two_allowlist_requires_earned_stage_one_graduation(
    tmp_path: Path,
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "governance" / "runtime").mkdir(parents=True)
    (tmp_path / "config" / "live_canary_micro_policy_v1.json").write_text(
        json.dumps(PLAN), encoding="utf-8"
    )
    (tmp_path / "config" / "live_canary_graduation_v1.json").write_text(
        json.dumps(POLICY), encoding="utf-8"
    )
    (tmp_path / "config" / "production_readiness_control_v1.json").write_text(
        json.dumps(
            {
                "live_execution_risk_firewall": {
                    "canary_plan_path": "config/live_canary_micro_policy_v1.json"
                }
            }
        ),
        encoding="utf-8",
    )
    (
        tmp_path / "governance" / "runtime" / "production_candidate_state.json"
    ).write_text(json.dumps({"candidate_id": CANDIDATE_ID}), encoding="utf-8")
    result = _issue_allowlist(
        tmp_path,
        stage=2,
        duration_minutes=60,
        confirmation=CONFIRMATION_PHRASE,
        confirm_all=True,
    )
    assert result["ok"] is False
    assert result["error"] == "prior_canary_stage_graduation_not_earned"
    assert result["completed_stage"] == 0
    assert result["live_execution_authority"] is False
