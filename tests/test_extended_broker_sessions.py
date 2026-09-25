import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.equity_order_sessions import extended_equity_session_state
from core.live_canary_preflight import evaluate_live_canary_preflight
from core.live_execution_envelope import file_sha256
from core.live_order_ledger import LiveOrderLedger
from core.order_intent import canonical_payload_sha256
from core.supervised_broker_test import (
    PURPOSE,
    approval_phrase,
    attestation_path,
    build_request,
    dispatch_once,
    intent_id,
    lifecycle_check,
    policy_path,
    reconcile_order,
    validate_policy,
)
from scripts.ops import supervised_broker_test as cli
from scripts.ops import live_canary_preflight as preflight_cli
from tests.test_supervised_broker_test import ready, order_payload
from tests.test_supervised_broker_test_cli import setup
from tests.test_live_canary_preflight import _write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return json.loads((ROOT / policy_path("SCHD")).read_text())


@pytest.mark.parametrize(
    "session,stamp,expected",
    [
        ("AM", "2026-09-23T10:59:59+00:00", False),
        ("AM", "2026-09-23T11:00:00+00:00", True),
        ("AM", "2026-09-23T13:23:45+00:00", True),
        ("AM", "2026-09-23T13:23:46+00:00", False),
        ("AM", "2026-09-23T13:25:00+00:00", False),
        ("PM", "2026-09-23T20:04:59+00:00", False),
        ("PM", "2026-09-23T20:05:00+00:00", True),
        ("PM", "2026-09-23T23:58:45+00:00", True),
        ("PM", "2026-09-24T00:00:00+00:00", False),
        ("AM", "2026-01-05T12:00:00+00:00", True),
        ("AM", "2026-01-05T11:59:59+00:00", False),
        ("PM", "2026-09-26T21:00:00+00:00", False),
        ("AM", "2026-12-25T13:00:00+00:00", False),
        ("PM", "2026-11-27T22:00:00+00:00", False),
        ("AM", "2026-11-27T13:00:00+00:00", False),
        ("SEAMLESS", "2026-09-23T21:00:00+00:00", False),
    ],
)
def test_calendar_dst_holidays_boundaries(session, stamp, expected):
    assert (
        extended_equity_session_state(
            session=session, now=datetime.fromisoformat(stamp)
        )["ready"]
        is expected
    )


def test_calendar_error_fails_closed(monkeypatch):
    import exchange_calendars

    monkeypatch.setattr(
        exchange_calendars,
        "get_calendar",
        lambda *a: (_ for _ in ()).throw(ValueError("missing")),
    )
    result = extended_equity_session_state(session="PM", now=datetime.now(timezone.utc))
    assert not result["ready"] and result["state"] == "unknown"


@pytest.mark.parametrize("session", ["AM", "PM", "NORMAL"])
def test_explicit_session_sealed_into_template_and_confirmation(plan, session):
    req = build_request(
        plan, action="BUY", quantity=1, limit_price="33.00", session=session
    )
    assert req["session"] == session and req["duration"] == "DAY"
    assert approval_phrase(plan, req).endswith(f"SESSION {session} DAY")
    assert (
        cli.propose_entry(plan, {}, now=datetime.now(timezone.utc), session=session)[
            "request"
        ]
        == {}
    )


@pytest.mark.parametrize(
    "mutation", ["id", "size", "autonomy", "sessions", "risk", "budget"]
)
def test_no_scope_expansion(plan, mutation):
    if mutation == "id":
        plan["test_id"] = "another_attempt"
    elif mutation == "size":
        plan["hard_limits"]["max_order_quantity"] = 2
    elif mutation == "autonomy":
        plan["authority"]["automatic_sell"] = True
    elif mutation == "sessions":
        plan["activation_contract"]["allowed_sessions"].append("SEAMLESS")
    elif mutation == "risk":
        plan["activation_contract"]["extended_hours_risk_confirmation_required"] = False
    elif mutation == "budget":
        plan["account_capital_usd"] = 300
    with pytest.raises(ValueError):
        validate_policy(plan)


def test_old_o_policy_and_production_stay_normal_only(tmp_path):
    old = json.loads((ROOT / policy_path("O")).read_text())
    for session in ("AM", "PM"):
        with pytest.raises(ValueError):
            build_request(
                old, action="BUY", quantity=1, limit_price=57, session=session
            )
        with pytest.raises(ValueError, match="normal-session-only"):
            evaluate_live_canary_preflight(tmp_path, symbol="SCHD", session=session)


@pytest.fixture
def extended_setup(setup, plan):
    root, kwargs = setup
    now = datetime(2026, 9, 23, 21, tzinfo=timezone.utc)
    old = kwargs["now"]
    # Only isolated fixture evidence, never live runtime files.
    for path in root.rglob("*.json"):
        payload = json.loads(
            path.read_text()
            .replace(old.isoformat(), now.isoformat())
            .replace(
                (old + timedelta(minutes=5)).isoformat(),
                (now + timedelta(minutes=5)).isoformat(),
            )
        )
        _write(path, payload)
    manifest_path = root / "governance/releases/immutable_release_manifest_latest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("manifest_sha256")
    manifest["freeze_window"]["ends_at_utc"] = (now + timedelta(days=1)).isoformat()
    manifest["manifest_sha256"] = canonical_payload_sha256(manifest)
    _write(manifest_path, manifest)
    _write(root / policy_path("SCHD"), plan)
    att = json.loads((root / attestation_path()).read_text())
    att.update(
        session="PM",
        extended_hours_risk_reviewed=True,
        test_policy_sha256=file_sha256(root / policy_path("SCHD")),
        trading_tax_ledger_sha256=file_sha256(
            root / "governance/tax/trading_tax_ledger_2026_latest.json"
        ),
        account_study_sha256=file_sha256(root / cli.STUDY_PATH),
    )
    _write(root / attestation_path("SCHD"), att, mode=0o600)
    kwargs.update(
        plan=plan,
        now=now,
        request=build_request(
            plan, action="BUY", quantity=1, limit_price=33, session="PM"
        ),
        inventory={"ok": True, "open_order_count": 0, "timestamp_utc": now.isoformat()},
    )
    kwargs["quote"].update(
        symbol="SCHD",
        bid_price=33,
        ask_price=33.01,
        provider_timestamp_utc=now.isoformat(),
        bid_timestamp_utc=now.isoformat(),
        ask_timestamp_utc=now.isoformat(),
        bid_size=1,
        ask_size=1,
    )
    return root, kwargs


def test_extended_preflight_ready_without_weakening_production(extended_setup):
    root, kwargs = extended_setup
    result = cli.assessment(root, **kwargs)
    assert result["blockers"] == []
    assert result["operator_submit_ready"] is True
    assert result["preflight"]["equity_session"]["session"] == "PM"
    assert result["live_execution_authority"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "O"),
        ("bid_size", 0),
        ("ask_size", 0),
        ("bid_timestamp_utc", "2026-09-23T20:59:44+00:00"),
        ("ask_timestamp_utc", "2026-09-23T21:00:01+00:00"),
        ("ask_price", 34),
        ("realtime", False),
    ],
)
def test_extended_quote_rejects_stale_side_or_missing_liquidity(
    extended_setup, field, value
):
    root, kwargs = extended_setup
    kwargs["quote"][field] = value
    assert not cli.assessment(root, **kwargs)["technical_ready"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("session", "AM"),
        ("extended_hours_risk_reviewed", False),
        ("test_policy_sha256", "wrong"),
    ],
)
def test_attestation_is_session_and_policy_bound(extended_setup, field, value):
    root, kwargs = extended_setup
    path = root / attestation_path("SCHD")
    payload = json.loads(path.read_text())
    payload[field] = value
    _write(path, payload, mode=0o600)
    result = cli.assessment(root, **kwargs)
    assert not result["operator_submit_ready"]
    assert result["preflight"]["operator_attestation_blockers"]


def test_issue_requires_explicit_extended_risk_before_writing(tmp_path):
    result = preflight_cli._issue_attestation(
        tmp_path,
        settled_cash_usd=565.49,
        duration_minutes=5,
        confirmation=preflight_cli.CONFIRMATION_PHRASE,
        confirm_all=True,
        purpose=PURPOSE,
        test_symbol="SCHD",
        session="PM",
    )
    assert not result["ok"] and not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "mode", ["ready", "expired_quote", "wrong_session", "closed", "wrong_phrase"]
)
def test_dispatch_revalidates_session_and_quote_without_real_broker(
    extended_setup, mode
):
    root, kwargs = extended_setup
    review = cli.assessment(root, **kwargs)
    now = kwargs["now"]
    phrase = approval_phrase(kwargs["plan"], kwargs["request"])
    if mode == "expired_quote":
        now += timedelta(seconds=15)
    elif mode == "closed":
        now = datetime(2026, 9, 24, 0, tzinfo=timezone.utc)
    elif mode == "wrong_session":
        review["preflight"]["equity_session"]["session"] = "AM"
    elif mode == "wrong_phrase":
        phrase = phrase.replace("PM", "AM")
    calls = []
    result = dispatch_once(
        plan=kwargs["plan"],
        request=kwargs["request"],
        ledger=kwargs["ledger"],
        assessment=review,
        approved_phrase=phrase,
        approved_at=kwargs["now"],
        now=now,
        dispatch=lambda spec: calls.append(spec)
        or {"ok": True, "status_code": 201, "order_id": "mock-schd"},
    )
    assert result["ok"] is (mode == "ready")
    assert len(calls) == (1 if mode == "ready" else 0)
    if mode == "ready":
        again = dispatch_once(
            plan=kwargs["plan"],
            request=kwargs["request"],
            ledger=LiveOrderLedger(kwargs["ledger"].path),
            assessment=review,
            approved_phrase=phrase,
            approved_at=now,
            now=now,
            dispatch=lambda spec: pytest.fail("duplicate"),
        )
        assert not again["ok"]


def test_no_unattended_extended_submit(tmp_path, plan, monkeypatch):
    _write(tmp_path / policy_path("SCHD"), plan)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli, "connect", lambda *a, **k: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="interactive_operator"):
        cli.run(
            argparse.Namespace(
                command="submit",
                symbol="SCHD",
                session="PM",
                action="BUY",
                quantity="1",
                limit_price="33",
                json=True,
            )
        )


@pytest.mark.parametrize(
    "field,value", [("session", "NORMAL"), ("duration", "GOOD_TILL_CANCEL")]
)
def test_reconciliation_binds_broker_session_and_duration(extended_setup, field, value):
    root, kwargs = extended_setup
    plan, req, ledger, now = (kwargs[k] for k in ("plan", "request", "ledger", "now"))
    review = cli.assessment(root, **kwargs)
    dispatch_once(
        plan=plan,
        request=req,
        ledger=ledger,
        assessment=review,
        approved_phrase=approval_phrase(plan, req),
        approved_at=now,
        now=now,
        dispatch=lambda spec: {"ok": True, "status_code": 201, "order_id": "mock-schd"},
    )
    broker = order_payload(req, filled=1, fill_price=33, broker_id="mock-schd")
    broker[field] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        reconcile_order(ledger, ledger.get(intent_id(plan, "BUY")), broker)
