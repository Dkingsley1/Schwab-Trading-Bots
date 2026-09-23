import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.broker_test_accounting import reconcile_test_accounting
from core.supervised_broker_test import account_digest

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 17, 18, tzinfo=timezone.utc)


@pytest.fixture
def evidence():
    plan = json.loads((ROOT / "config/supervised_broker_test_v1.json").read_text())
    baseline_time = NOW - timedelta(days=1)
    baseline = {
        "state": "observed",
        "timestamp_utc": baseline_time.isoformat(),
        "source": "schwab_currentBalances.cashBalance",
        "balance_usd": "850.14",
        "broker_payload_sha256": "baseline-hash",
        "account_reference_sha256": account_digest("private-account"),
    }
    payload = {
        "action": "BUY",
        "symbol": "O",
        "test_id": plan["test_id"],
        "account_policy_key": plan["account_policy_key"],
        "account_reference_sha256": account_digest("private-account"),
        "baseline_cash_observation": baseline,
    }
    order = {
        "state": "filled",
        "filled_quantity": 5,
        "average_fill_price": "56.93",
        "broker_order_id": "private-order",
        "created_at_utc": (baseline_time + timedelta(seconds=1)).isoformat(),
        "payload_json": json.dumps(payload),
    }
    trade = {
        "activityId": "private-transaction",
        "orderId": "private-order",
        "type": "TRADE",
        "status": "VALID",
        "time": (baseline_time + timedelta(seconds=3)).isoformat(),
        "settlementDate": NOW.isoformat(),
        "settlementStatus": "SETTLED",
        "netAmount": "-284.65",
        "transferItems": [
            {
                "amount": 5,
                "price": "56.93",
                "instrument": {"symbol": "O", "assetType": "EQUITY"},
            }
        ],
    }
    return {
        "plan": plan,
        "orders": [order],
        "transactions": {
            "source_complete": True,
            "rows": [trade],
            "window_start_utc": baseline_time.isoformat(),
            "window_end_utc": NOW.isoformat(),
        },
        "reference": "private-account",
        "position_consistent": True,
        "now": NOW,
        "cash_observation": {
            **baseline,
            "timestamp_utc": NOW.isoformat(),
            "balance_usd": "565.49",
        },
    }


def test_full_accounting_needs_cash_execution_settlement_and_position(evidence):
    result = reconcile_test_accounting(**evidence)
    assert result["state"] == "reconciled"
    assert result["account_cash_reconciled"]
    assert result["trade_cash_reconciled"]
    assert result["orders"][0]["effective_charges_usd"] == "0.00"
    assert result["orders"][0]["fee_breakdown_certified"] is False
    assert result["cash_funding_authority"] is False
    assert "private-transaction" not in json.dumps(result)
    assert "private-order" not in json.dumps(result)
    assert "private-account" not in json.dumps(result)


def test_legacy_cash_proxy_cannot_be_backfilled_by_numeric_match(evidence):
    payload = json.loads(evidence["orders"][0]["payload_json"])
    del payload["baseline_cash_observation"]
    payload["baseline_cash_usd"] = 850.14
    evidence["orders"][0]["payload_json"] = json.dumps(payload)
    result = reconcile_test_accounting(**evidence)
    assert result["trade_cash_reconciled"]
    assert not result["account_cash_reconciled"]
    assert (
        "comparable_pretrade_and_posttrade_cash_evidence_required" in result["pending"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "PENDING"),
        ("orderId", "different"),
        ("type", "DIVIDEND_OR_INTEREST"),
        ("netAmount", "NaN"),
        ("netAmount", "284.65"),
        ("netAmount", "-280"),
        ("activityId", ""),
        ("time", (NOW + timedelta(seconds=1)).isoformat()),
    ],
)
def test_invalid_trade_never_reconciles(evidence, field, value):
    evidence["transactions"]["rows"][0][field] = value
    result = reconcile_test_accounting(**evidence)
    assert not result["trade_cash_reconciled"]
    assert result["state"] == "pending"


@pytest.mark.parametrize(
    "change",
    [
        {"amount": 4},
        {"amount": -5},
        {"price": "56.92"},
        {"instrument": {"symbol": "SCHD", "assetType": "EQUITY"}},
        {"instrument": {"symbol": "O", "assetType": "OPTION"}},
    ],
)
def test_transaction_must_match_actual_equity_execution(evidence, change):
    evidence["transactions"]["rows"][0]["transferItems"][0].update(change)
    assert not reconcile_test_accounting(**evidence)["trade_cash_reconciled"]


def test_duplicates_do_not_double_cash_or_fees(evidence):
    evidence["transactions"]["rows"] *= 2
    assert reconcile_test_accounting(**evidence)["state"] == "reconciled"
    conflicting = copy.deepcopy(evidence["transactions"]["rows"][0])
    conflicting["netAmount"] = "-284.66"
    evidence["transactions"]["rows"].append(conflicting)
    assert not reconcile_test_accounting(**evidence)["trade_cash_reconciled"]


def test_partial_execution_postings_and_effective_charges(evidence):
    first = evidence["transactions"]["rows"][0]
    first["netAmount"] = "-113.88"
    first["transferItems"][0]["amount"] = 2
    second = copy.deepcopy(first)
    second.update(activityId="second", netAmount="-170.81")
    second["transferItems"][0]["amount"] = 3
    evidence["transactions"]["rows"].append(second)
    evidence["cash_observation"]["balance_usd"] = "565.45"
    result = reconcile_test_accounting(**evidence)
    assert result["state"] == "reconciled"
    assert result["orders"][0]["effective_charges_usd"] == "0.04"


@pytest.mark.parametrize(
    "case",
    [
        "unsettled",
        "missing_settlement",
        "wrong_cash",
        "incomplete",
        "stale",
        "wrong_account",
        "position_mismatch",
        "missing_net",
        "unknown_transaction",
    ],
)
def test_accounting_remains_pending_for_missing_evidence(evidence, case):
    if case == "unsettled":
        evidence["transactions"]["rows"][0]["settlementDate"] = (
            NOW + timedelta(days=1)
        ).isoformat()
    elif case == "missing_settlement":
        del evidence["transactions"]["rows"][0]["settlementDate"]
    elif case == "wrong_cash":
        evidence["cash_observation"]["balance_usd"] = "900"
    elif case == "incomplete":
        evidence["transactions"]["source_complete"] = False
    elif case == "stale":
        evidence["cash_observation"]["timestamp_utc"] = (
            NOW - timedelta(seconds=31)
        ).isoformat()
    elif case == "wrong_account":
        evidence["reference"] = "another-account"
    elif case == "position_mismatch":
        evidence["position_consistent"] = False
    elif case == "missing_net":
        del evidence["transactions"]["rows"][0]["netAmount"]
    else:
        evidence["transactions"]["rows"].append(
            {
                "activityId": "other",
                "status": "UNKNOWN",
                "netAmount": 100,
                "time": NOW.isoformat(),
            }
        )
    assert reconcile_test_accounting(**evidence)["state"] == "pending"


def test_other_account_activity_is_in_cash_equation_not_trade_profit(evidence):
    evidence["transactions"]["rows"].append(
        {
            "activityId": "deposit",
            "status": "VALID",
            "type": "ACH",
            "time": NOW.isoformat(),
            "netAmount": "100",
        }
    )
    evidence["cash_observation"]["balance_usd"] = "665.49"
    result = reconcile_test_accounting(**evidence)
    assert result["state"] == "reconciled"
    assert result["orders"][0]["net_cash_usd"] == "-284.65"


def test_scheduled_settlement_date_alone_is_not_settlement_confirmation(evidence):
    trade = evidence["transactions"]["rows"][0]
    trade["settlementDate"] = "2026-09-17"
    del trade["settlementStatus"]
    result = reconcile_test_accounting(**evidence)
    assert result["orders"][0]["settlement_date_reached"] is True
    assert result["settlement_observed"] is False
    assert result["state"] == "pending"
