from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import contextlib
from datetime import datetime

import pytest

from core.decision_price_evidence import digest
from core.live_order_ledger import LiveOrderLedger
from core.order_intent import canonical_payload_sha256
from core.schd_bot_handoff import prepare_handoff, validate_handoff
from core.schd_market_evidence import candle_context_receipt, provider_quote_features
from core.supervised_broker_test import (
    PURPOSE,
    approval_phrase,
    build_market_request,
    dispatch_once,
    intent_id,
    lifecycle_check,
    reconcile_order,
    request_fields,
    validate_policy,
)
from tests.test_schd_native_decision import fixture_data, packet_from

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ready(tmp_path, fixture_data):
    row, candidate, market, now = deepcopy(fixture_data)
    packet = packet_from(tmp_path, [row], candidate, market, now)
    plan = json.loads((ROOT / "config/supervised_schd_broker_test_v1.json").read_text())
    quote = {
        "symbol": "SCHD",
        "source_provider": "schwab_api",
        "realtime": True,
        "transport": {"ok": True},
        "provider_timestamp_utc": now.isoformat(),
        "bid_price": 32.99,
        "ask_price": 33.0,
        "bid_size": 100,
        "ask_size": 100,
        "bid_timestamp_utc": now.isoformat(),
        "ask_timestamp_utc": now.isoformat(),
    }
    handoff = prepare_handoff(
        plan, packet, quote, action="BUY", candidate_id=packet["candidate_id"], now=now
    )
    assert handoff["ready"], handoff["blockers"]
    return plan, packet, quote, now, handoff


@pytest.mark.parametrize(
    "mutation",
    [
        "hold",
        "gate",
        "basis",
        "stale_decision",
        "future_quote",
        "stale_quote",
        "wide_spread",
        "price_jump",
        "candidate",
        "missing_native",
        "source_gap",
    ],
)
def test_native_market_evidence_fails_closed(ready, mutation):
    plan, packet, quote, now, _ = deepcopy(ready)
    candidate = packet["candidate_id"]
    if mutation == "hold":
        packet["decision"]["action"] = "HOLD"
    if mutation == "gate":
        packet["decision"]["gates"]["fixture_gate"] = False
    if mutation == "basis":
        packet["price_basis"] = "unknown"
    if mutation == "stale_decision":
        now += timedelta(seconds=121)
    if mutation == "future_quote":
        quote["provider_timestamp_utc"] = (now + timedelta(seconds=1)).isoformat()
    if mutation == "stale_quote":
        quote["provider_timestamp_utc"] = (now - timedelta(seconds=16)).isoformat()
    if mutation == "wide_spread":
        quote["ask_price"] = 34
    if mutation == "price_jump":
        quote.update(bid_price=33.99, ask_price=34)
    if mutation == "candidate":
        candidate = "wrong"
    if mutation == "missing_native":
        packet.pop("native_validation")
    if mutation == "source_gap":
        packet["native_validation"]["blockers"] = ["native_malformed_row"]
    handoff = prepare_handoff(
        plan, packet, quote, action="BUY", candidate_id=candidate, now=now
    )
    assert not handoff["ready"]


def assessment(plan, packet, handoff, now):
    return {
        "timestamp_utc": now.isoformat(),
        "purpose": PURPOSE,
        "technical_ready": True,
        "operator_attestation_ready": True,
        "operator_submit_ready": True,
        "blockers": [],
        "request_sha256": canonical_payload_sha256(handoff["request"]),
        "policy_sha256": canonical_payload_sha256(plan),
        "candidate_id": packet["candidate_id"],
        "account_reference_sha256": "test-account",
        "position_quantity": 0,
        "settled_cash_usd": 100,
        "bot_handoff": handoff["receipt"],
        "market_order_price_risk_reviewed": True,
    }


@pytest.mark.parametrize(
    "change", ["none", "phrase", "receipt", "risk", "request", "stale", "unknown"]
)
def test_market_dispatch_requires_fresh_exact_approval_and_is_once(
    tmp_path, ready, change
):
    plan, packet, quote, now, handoff = deepcopy(ready)
    review = assessment(plan, packet, handoff, now)
    request = handoff["request"]
    phrase = approval_phrase(plan, request)
    calls = []
    if change == "phrase":
        phrase = "yes"
    if change == "receipt":
        review["bot_handoff"] = {}
    if change == "risk":
        review["market_order_price_risk_reviewed"] = False
    if change == "request":
        review["request_sha256"] = "changed"
    check_time = now + timedelta(seconds=16) if change == "stale" else now
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")

    def dispatch(spec):
        calls.append(spec)
        if change == "unknown":
            raise TimeoutError()
        return {"ok": True, "status_code": 201, "order_id": "fixture-only-1"}

    kwargs = dict(
        plan=plan,
        request=request,
        ledger=ledger,
        assessment=review,
        approved_phrase=phrase,
        approved_at=now,
        now=check_time,
        dispatch=dispatch,
    )
    result = dispatch_once(**kwargs)
    assert len(calls) == (1 if change in {"none", "unknown"} else 0)
    if calls:
        assert result["broker_mutation_attempted"]
        kwargs["ledger"] = LiveOrderLedger(tmp_path / "orders.sqlite3")
        assert not dispatch_once(**kwargs)["ok"]
        assert len(calls) == 1
        assert calls[0]["orderType"] == "MARKET" and "price" not in calls[0]


def test_market_fill_uses_executions_not_estimate(tmp_path, ready):
    plan, packet, quote, now, handoff = ready
    ledger = LiveOrderLedger(tmp_path / "orders.sqlite3")
    dispatch_once(
        plan=plan,
        request=handoff["request"],
        ledger=ledger,
        assessment=assessment(plan, packet, handoff, now),
        approved_phrase=approval_phrase(plan, handoff["request"]),
        approved_at=now,
        now=now,
        dispatch=lambda _: {
            "ok": True,
            "status_code": 201,
            "order_id": "fixture-only-1",
        },
    )
    broker = dict(
        handoff["request"],
        orderId="fixture-only-1",
        status="FILLED",
        filledQuantity=1,
        orderActivityCollection=[
            {
                "activityType": "EXECUTION",
                "executionLegs": [{"quantity": 1, "price": 33.02}],
            }
        ],
    )
    row = reconcile_order(ledger, ledger.get(intent_id(plan, "BUY")), broker)
    assert row["average_fill_price"] == 33.02
    broker["orderActivityCollection"] = []
    with pytest.raises(ValueError):
        reconcile_order(ledger, row, broker)
    request = build_market_request(plan, action="SELL")
    blockers = lifecycle_check(
        plan,
        request,
        ledger,
        account_reference="wrong",
        position_quantity=1,
        unencumbered_quantity=1,
        account_captured_at=(now + timedelta(seconds=1)).isoformat(),
    )
    assert "verified_test_entry_required_for_sell" in blockers


def test_market_template_cannot_expand_scope(ready):
    plan, _, _, _, handoff = ready
    for field, value in (
        ("session", "PM"),
        ("price", "33"),
        ("duration", "GOOD_TILL_CANCEL"),
    ):
        with pytest.raises(ValueError):
            request_fields(plan, dict(handoff["request"], **{field: value}))
    expanded = deepcopy(handoff["request"])
    expanded["orderLegCollection"][0]["quantity"] = 2
    with pytest.raises(ValueError):
        request_fields(plan, expanded)
    o_plan = json.loads((ROOT / "config/supervised_broker_test_v1.json").read_text())
    with pytest.raises(ValueError):
        build_market_request(o_plan, action="BUY")
    changed = deepcopy(plan)
    changed["bot_market_order_contract"]["native_decision_required"] = False
    with pytest.raises(ValueError):
        validate_policy(changed)


def test_provider_times_are_never_fabricated(ready):
    _, _, _, now, _ = ready
    payload = {
        "SCHD": {
            "symbol": "SCHD",
            "realtime": True,
            "quote": {
                "quoteTime": now.timestamp() * 1000,
                "bidTime": (now.timestamp() - 1) * 1000,
                "askTime": (now.timestamp() + 1) * 1000,
                "lastPrice": 33,
            },
        }
    }
    observed = provider_quote_features(payload, "SCHD", now=now)
    assert observed["provider_quote_ts_utc"] == now.timestamp()
    assert "provider_ask_ts_utc" not in observed
    assert observed["provider_quote_last_price"] == 33
    assert "provider_quote_ts_utc" not in provider_quote_features({}, "SCHD", now=now)
    assert (
        provider_quote_features({"SCHD": []}, "SCHD", now=now)[
            "provider_quote_realtime_norm"
        ]
        == 0
    )


def test_native_collector_keeps_raw_quote_when_history_replaces_last(ready):
    from scripts import run_shadow_training_loop as loop

    _, _, _, now, _ = ready
    payload = {
        "SCHD": {
            "symbol": "SCHD",
            "realtime": True,
            "quote": {
                "quoteTime": now.timestamp() * 1000,
                "lastPrice": 99,
                "bidPrice": 32.99,
                "askPrice": 33.0,
                "bidSize": 100,
                "askSize": 100,
            },
        }
    }

    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    client = SimpleNamespace(
        get_quote=lambda symbol: Response(payload),
        get_price_history_every_minute=lambda *a, **k: Response(
            {
                "candles": [
                    {
                        "close": 33,
                        "open": 33,
                        "high": 33.1,
                        "low": 32.9,
                        "volume": 1000,
                    },
                ]
            }
        ),
    )
    snapshot = loop._market_snapshot_from_schwab(client, "SCHD")
    assert snapshot["last_price"] == 33
    assert snapshot["provider_quote_last_price"] == 99
    assert snapshot["provider_quote_ts_utc"] == now.timestamp()


def test_context_receipt_binds_capture_and_rejects_stale_and_protected(
    tmp_path, fixture_data
):
    _, _, market, now = deepcopy(fixture_data)
    path = tmp_path / "governance/rehearsals/schd/market_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(market))
    receipt = candle_context_receipt(tmp_path, "SCHD", now=now)
    assert receipt["capture_sha256"] == digest(market)
    assert (
        candle_context_receipt(tmp_path, "SCHD", now=now + timedelta(minutes=6))[
            "state"
        ]
        == "stale_or_future"
    )
    path.unlink()
    path.symlink_to("/Volumes/VIDEO/do-not-touch.json")
    assert candle_context_receipt(tmp_path, "SCHD", now=now)["state"] == "unavailable"


def test_checklist_never_issues_attestation_or_connects(monkeypatch, ready):
    from scripts.ops import supervised_broker_test as cli

    plan, _, _, _, _ = ready
    monkeypatch.setattr(cli, "load", lambda *a: plan)
    monkeypatch.setattr(cli, "connect", lambda *a, **k: pytest.fail("connected"))
    monkeypatch.setattr(
        cli, "_issue_attestation", lambda *a, **k: pytest.fail("attested")
    )
    result = cli.run(
        SimpleNamespace(
            command="attestation-checklist",
            symbol="SCHD",
            session="NORMAL",
            bot_market=True,
            quantity=None,
            limit_price=None,
        )
    )
    assert result["state"] == "checklist_only_not_attested"
    assert "market_order_price_risk_reviewed" in result["required_confirmations"]
    assert not result["live_execution_authority"]


@pytest.mark.parametrize("changed", [False, True])
def test_interactive_handoff_rechecks_decision_before_mock_dispatch(
    tmp_path, monkeypatch, ready, changed
):
    from scripts.ops import supervised_broker_test as cli
    from scripts.ops import schd_bot_handoff as reader

    plan, packet, quote, now, handoff = deepcopy(ready)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "load",
        lambda root, path: (
            plan if path.endswith("supervised_schd_broker_test_v1.json") else {}
        ),
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    for key in (
        "ALLOW_ORDER_EXECUTION",
        "EXECUTION_LANE_LIVE_ENABLED",
        "TOP_BOT_ENABLE_LIVE_EXECUTION",
    ):
        monkeypatch.setenv(key, "0")
    observations = []

    def observe(*a, **k):
        result = deepcopy(handoff)
        observations.append(result)
        if changed and len(observations) >= 3:
            result["receipt"]["decision_binding_sha256"] = "new-decision"
        return result

    monkeypatch.setattr(reader, "observe_handoff", observe)
    monkeypatch.setattr(cli, "test_lock", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(
        cli, "component_action_guard", lambda *a, **k: contextlib.nullcontext()
    )
    calls = []
    trader = SimpleNamespace(
        _fetch_live_quote=lambda **k: {},
        broker_adapter=SimpleNamespace(
            place_order_candidates=lambda **k: [("place_order", (), k)]
        ),
        _invoke_client_candidates=lambda **k: calls.append(k)
        or {"ok": True, "status_code": 201, "order_id": "fixture-only"},
    )
    monkeypatch.setattr(cli, "connect", lambda *a, **k: (trader, "test", quote))
    monkeypatch.setattr(cli, "_quote_summary", lambda *a, **k: quote)
    ledger = LiveOrderLedger(tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(cli, "open_ledger", lambda *a: ledger)
    monkeypatch.setattr(cli, "broker_inventory", lambda *a: {})
    monkeypatch.setattr(
        cli, "assessment", lambda *a, **k: assessment(plan, packet, handoff, now)
    )
    monkeypatch.setattr(cli, "_refresh_account_study", lambda **k: {"ok": True})
    monkeypatch.setattr(cli, "_issue_attestation", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(cli, "broker_cash_observation", lambda *a: {})
    monkeypatch.setattr(
        cli, "settle_order", lambda *a: {"state": "not_observed_fixture"}
    )
    monkeypatch.setattr(cli, "observe", lambda *a, **k: {})

    def answer(prompt):
        if "Current settled" in prompt:
            return "280.81"
        if "Type exactly" in prompt:
            return approval_phrase(plan, handoff["request"])
        return "yes"

    monkeypatch.setattr("builtins.input", answer)
    args = SimpleNamespace(
        command="submit",
        symbol="SCHD",
        session="NORMAL",
        bot_market=True,
        quantity=None,
        limit_price=None,
        action="BUY",
    )
    if changed:
        with pytest.raises(ValueError, match="decision_changed"):
            cli.run(args)
        assert not calls and not ledger.get(intent_id(plan, "BUY"))
    else:
        assert cli.run(args)["ok"]
        assert len(calls) == 1
    assert len(observations) == 3
