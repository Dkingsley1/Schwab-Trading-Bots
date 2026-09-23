import argparse
import contextlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.live_canary_preflight import (
    evaluate_live_canary_preflight,
    required_operator_confirmations,
)
from core.live_execution_envelope import file_sha256
from core.live_order_ledger import LiveOrderLedger
from core.supervised_broker_test import PURPOSE, build_request, intent_id
from scripts.ops import supervised_broker_test as cli
from tests.test_live_canary_preflight import _seed_ready_preflight, _write
from tests.test_supervised_broker_test import filled_entry, submit, order_payload

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "mutation", ["none", "drift", "chain", "coverage", "changed_state", "dirty"]
)
def test_source_acceptance_binds_content_and_history_not_precommit_head(
    tmp_path, monkeypatch, mutation
):
    candidate = {
        "candidate_id": "test-candidate",
        "accepted_git_head": "precommit-head",
        "overall_sha256": "a" * 64,
        "event_chain_head": "b" * 64,
    }
    checked = {
        "state": dict(candidate),
        "current": {"overall_sha256": candidate["overall_sha256"]},
        "candidate_drift": False,
        "operation_error": "",
        "source_coverage": {"ready": True},
        "event_chain": {"ok": True, "chain_head": candidate["event_chain_head"]},
    }
    _write(tmp_path / "config/production_excellence_v1.json", {"candidate": {}})
    calls = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kw: calls.append(command)
        or SimpleNamespace(stdout=" M core/file.py" if mutation == "dirty" else ""),
    )
    monkeypatch.setattr(
        cli.production_excellence_control, "manage_candidate", lambda *args: checked
    )
    if mutation == "drift":
        checked["candidate_drift"] = True
    elif mutation == "chain":
        checked["event_chain"]["chain_head"] = "wrong"
    elif mutation == "coverage":
        checked["source_coverage"]["ready"] = False
    elif mutation == "changed_state":
        checked["state"]["candidate_id"] = "different-candidate"
    blockers = cli.current_source_blockers(tmp_path, candidate)
    assert bool(blockers) == (mutation != "none")
    assert len(calls) == 1 and calls[0][1] == "status"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)
    plan = json.loads((ROOT / cli.POLICY_PATH).read_text())
    _write(tmp_path / cli.POLICY_PATH, plan)
    for relative in ("config/account_policy_registry.json", cli.STUDY_PATH):
        path = tmp_path / relative
        payload = json.loads(
            path.read_text().replace("schwab_cash_account_1", "schwab_roth_ira_primary")
        )
        if relative.startswith("config"):
            payload["account_slots"][0].update(
                cash_only_live_budget=True,
                existing_positions_authority="observe_only",
                account_type="roth_ira",
            )
        else:
            row = payload["accounts"][0]
            row.update(operator_account_kind="roth_ira", tax_wrapper="roth_ira")
            row["account_capability_truth"]["operator_classification"] = {
                "account_kind": "roth_ira",
                "tax_wrapper": "roth_ira",
            }
            row["account_capability_truth"]["balance_truth"].update(
                cash_balance=850.14, cash_available_for_trading=850.14
            )
            payload["positions"] = []
        _write(path, payload)
    attestation = json.loads(
        (
            tmp_path / "governance/runtime/live_canary_operator_attestation.json"
        ).read_text()
    )
    attestation.update(
        purpose=PURPOSE,
        account_policy_key=plan["account_policy_key"],
        settled_cash_usd=850.14,
        account_study_sha256=file_sha256(tmp_path / cli.STUDY_PATH),
        test_policy_sha256=file_sha256(tmp_path / cli.POLICY_PATH),
        issued_at_utc=now.isoformat(),
        expires_at_utc=(now + timedelta(minutes=5)).isoformat(),
    )
    for key in required_operator_confirmations("roth_ira"):
        attestation[key] = True
    attestation["broker_open_orders_reviewed"] = True
    attestation["no_concurrent_manual_orders_confirmed"] = True
    _write(
        tmp_path / "governance/runtime/supervised_broker_test_attestation.json",
        attestation,
        mode=0o600,
    )
    _write(
        tmp_path / "governance/health/local_storage_reserve_guard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "local_storage_reserve": {
                "pressure_active": False,
                "hard_block": False,
                "pressure_free_gb": 64,
            },
        },
    )
    _write(
        tmp_path / "governance/health/production_excellence_control_latest.json",
        {"ten_out_of_ten_ready": False, "live_money_consideration_ready": False},
    )
    _write(
        tmp_path / "governance/health/continuous_soak_integrity_control_latest.json",
        {"scope_aware_validation_complete": False},
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(cli, "current_source_blockers", lambda *args: [])
    monkeypatch.setattr(
        cli, "evaluate_component_action", lambda *args, **kwargs: {"ok": True}
    )
    monkeypatch.setattr(
        cli.shutil, "disk_usage", lambda *args: SimpleNamespace(free=100 * 1024**3)
    )
    for key in (
        "OPERATOR_STOP",
        "GLOBAL_TRADING_HALT",
        "ALLOW_ORDER_EXECUTION",
        "EXECUTION_LANE_LIVE_ENABLED",
        "TOP_BOT_ENABLE_LIVE_EXECUTION",
    ):
        monkeypatch.setenv(key, "0")
    ledger = LiveOrderLedger(tmp_path / cli.LEDGER_PATH)
    request = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    quote = {
        "source_provider": "schwab_api",
        "realtime": True,
        "transport": {"ok": True},
        "provider_timestamp_utc": now.isoformat(),
        "bid_price": 57.09,
        "ask_price": 57.10,
        "source_venue": "XNYS",
        "snapshot_id": "f" * 64,
    }
    kwargs = dict(
        plan=plan,
        request=request,
        quote=quote,
        reference=env["SCHWAB_ACCOUNT_HASH"],
        ledger=ledger,
        inventory={"ok": True, "open_order_count": 0, "timestamp_utc": now.isoformat()},
        now=now,
    )
    return tmp_path, kwargs


def test_technical_test_is_separate_from_production_soak_and_profitability(setup):
    root, kwargs = setup
    result = cli.assessment(root, **kwargs)
    assert result["blockers"] == []
    assert result["operator_submit_ready"] is True
    assert result["production_promotion_credit"] is False
    assert result["live_execution_authority"] is False
    assert (
        json.loads(
            (
                root / "governance/health/production_excellence_control_latest.json"
            ).read_text()
        )["ten_out_of_ten_ready"]
        is False
    )


@pytest.mark.parametrize("realtime", [True, False])
def test_native_quote_adapter_output_is_consumed_without_relabeling(setup, realtime):
    root, kwargs = setup
    now = kwargs["now"]
    raw = {
        "ok": True,
        "status_code": 200,
        "quote_snapshot": {
            "bid_price": 57.09,
            "ask_price": 57.10,
            "raw_payload": {
                "O": {
                    "realtime": realtime,
                    "quote": {
                        "bidTime": int(now.timestamp() * 1000),
                        "askTime": int(now.timestamp() * 1000),
                        "askMICId": "XNYS",
                    },
                }
            },
        },
    }
    kwargs["quote"] = cli._quote_summary(raw, symbol="O", now=now)
    assert kwargs["quote"]["source_provider"] == "schwab_api"
    proposal = cli.propose_entry(kwargs["plan"], kwargs["quote"], now=now)
    assert (proposal["state"] == "proposed") is realtime
    result = cli.assessment(root, **kwargs)
    assert result["operator_submit_ready"] is realtime


@pytest.mark.parametrize(
    "path,update,expected",
    [
        (
            "governance/risk/risk_service_boundary_latest.json",
            {"ok": False},
            "risk_service_boundary_not_ready",
        ),
        (
            "governance/health/live_order_ledger_control_latest.json",
            {"unresolved_intent_count": 1},
            "live_order_ledger_not_ready",
        ),
        (
            "governance/health/release_freeze_guard_latest.json",
            {"immutable_release_boundary": {}},
            "immutable_release_boundary_not_ready",
        ),
        (
            "governance/health/local_storage_reserve_guard_latest.json",
            {"local_storage_reserve": {"pressure_active": True, "hard_block": False}},
            "current_storage_write_headroom_required",
        ),
        (
            "governance/health/SCHWAB_BROKER_BOUNDARY_QUARANTINE.json",
            {"active": True},
            "schwab_broker_boundary_quarantine_active",
        ),
    ],
)
def test_technical_safety_gates_remain_required(setup, path, update, expected):
    root, kwargs = setup
    value = cli.load(root, path)
    _write(root / path, {**value, **update})
    result = cli.assessment(root, **kwargs)
    assert expected in result["blockers"]
    assert not result["technical_ready"]


def test_operator_attestation_is_separate_and_not_auto_issued(setup):
    root, kwargs = setup
    (root / "governance/runtime/supervised_broker_test_attestation.json").unlink()
    result = cli.assessment(root, **kwargs)
    assert result["technical_ready"]
    assert not result["operator_submit_ready"]
    assert not result["operator_attestation_ready"]
    assert not (
        root / "governance/runtime/supervised_broker_test_attestation.json"
    ).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "expired_quote",
        "future_account",
        "open_orders",
        "wrong_account",
        "above_bid",
        "dirty_source",
    ],
)
def test_bad_current_execution_evidence_fails_closed(setup, monkeypatch, mutation):
    root, kwargs = setup
    if mutation == "expired_quote":
        kwargs["quote"]["provider_timestamp_utc"] = (
            kwargs["now"] - timedelta(seconds=16)
        ).isoformat()
    elif mutation == "future_account":
        study = cli.load(root, cli.STUDY_PATH)
        study["timestamp_utc"] = (kwargs["now"] + timedelta(seconds=3)).isoformat()
        _write(root / cli.STUDY_PATH, study)
    elif mutation == "open_orders":
        kwargs["inventory"]["open_order_count"] = 1
    elif mutation == "wrong_account":
        kwargs["reference"] = "another-account"
    elif mutation == "above_bid":
        kwargs["quote"]["bid_price"] = 57.08
    else:
        monkeypatch.setattr(
            cli,
            "current_source_blockers",
            lambda *args: ["test_source_release_not_clean"],
        )
    result = cli.assessment(root, **kwargs)
    assert result["technical_blockers"]
    assert result["operator_submit_ready"] is False


def test_sell_preflight_does_not_drop_safety_blockers(setup):
    root, kwargs = setup
    (root / "governance/runtime/supervised_broker_test_attestation.json").unlink()
    result = evaluate_live_canary_preflight(
        root,
        symbol="O",
        action="SELL",
        account_reference=kwargs["reference"],
        purpose=PURPOSE,
        now=kwargs["now"],
    )
    assert not result["ready"]
    assert "live_canary_operator_attestation_missing_or_invalid" in result["blockers"]


def test_test_attestation_cannot_satisfy_production_preflight(setup):
    root, kwargs = setup
    test_path = root / "governance/runtime/supervised_broker_test_attestation.json"
    _write(
        root / "governance/runtime/live_canary_operator_attestation.json",
        json.loads(test_path.read_text()),
        mode=0o600,
    )
    result = evaluate_live_canary_preflight(
        root, symbol="SCHD", account_reference=kwargs["reference"], now=kwargs["now"]
    )
    assert "operator_attestation_purpose_mismatch" in result["blockers"]


def test_no_unattended_submit_or_broker_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    _write(tmp_path / cli.POLICY_PATH, json.loads((ROOT / cli.POLICY_PATH).read_text()))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        cli, "connect", lambda *args: pytest.fail("must reject before connecting")
    )
    args = argparse.Namespace(
        command="submit", action="BUY", quantity=None, limit_price=None, json=True
    )
    with pytest.raises(ValueError, match="interactive_operator_terminal_required"):
        cli.run(args)


def test_explicit_pm_limit_label_and_extended_risk_checklist(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    policy = "config/supervised_schd_broker_test_v1.json"
    _write(tmp_path / policy, json.loads((ROOT / policy).read_text()))
    args = argparse.Namespace(command="attestation-checklist", symbol="SCHD",
        session="PM", action="BUY", quantity=1, limit_price="33.33", json=True,
        bot_market=False)
    checklist = cli.run(args)
    assert "extended_hours_risk_reviewed" in checklist["required_confirmations"]
    assert "market_order_price_risk_reviewed" not in checklist["required_confirmations"]
    monkeypatch.setattr(cli, "connect", lambda *a, **k: (object(), "test", {}))
    monkeypatch.setattr(cli, "open_ledger", lambda *a: object())
    monkeypatch.setattr(cli, "broker_inventory", lambda *a: {})
    monkeypatch.setattr(cli, "assessment", lambda *a, **k: {
        "technical_ready": False, "technical_blockers": ["retained_test_blocker"],
        "request": k["request"], "live_execution_authority": False})
    args.command = "preview"
    result = cli.run(args)
    assert result["price_proposal"]["state"] == "operator_specified_limit"
    assert result["request"]["price"] == "33.33"
    assert result["request"]["session"] == "PM"
    assert result["technical_blockers"] == ["retained_test_blocker"]
    assert not result["live_execution_authority"]


def test_protected_symlink_rejected_before_target_io(tmp_path):
    (tmp_path / "redirect").symlink_to("/Volumes/VIDEO/private")
    with pytest.raises(ValueError, match="unsafe_test_path"):
        cli.local_path(tmp_path, "redirect/file.json")


def test_account_order_keywords_match_schwab_sdk():
    # The SDK signature is get_order(order_id, account_hash), not account first.
    BaseClient = pytest.importorskip("schwab.client.base").BaseClient
    import inspect

    signature = inspect.signature(BaseClient.get_order)
    signature.bind(None, account_hash="test-account", order_id="test-order")
    inspect.signature(BaseClient.cancel_order).bind(
        None, account_hash="test-account", order_id="test-order"
    )


def test_cancel_deadline_dispatches_once_and_reconciles(tmp_path, monkeypatch):
    plan = json.loads((ROOT / cli.POLICY_PATH).read_text())
    ledger = LiveOrderLedger(tmp_path / "ledger.sqlite3")
    submit(plan, ledger)
    request = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs)
        if kwargs["operation"] == "cancel_order":
            return {"ok": True}
        return {
            "ok": True,
            "response_payload": order_payload(request, status="CANCELED", filled=0),
        }

    trader = SimpleNamespace(_invoke_client_candidates=invoke)
    result = cli.settle_order(
        trader, ledger, intent_id(plan, "BUY"), "roth-test-hash", deadline_seconds=0
    )
    assert result["state"] == "canceled"
    assert [call["operation"] for call in calls] == ["cancel_order", "get_order"]
    assert calls[0]["candidates"] == [
        (
            "cancel_order",
            (),
            {"account_hash": "roth-test-hash", "order_id": "broker-test-1"},
        )
    ]
    assert ledger.verify_integrity()["ok"]


def test_failed_cancel_and_failed_read_retain_ambiguity(tmp_path):
    plan = json.loads((ROOT / cli.POLICY_PATH).read_text())
    ledger = LiveOrderLedger(tmp_path / "ledger.sqlite3")
    submit(plan, ledger)
    calls = []
    trader = SimpleNamespace(
        _invoke_client_candidates=lambda **kwargs: calls.append(kwargs["operation"])
        or {"ok": False}
    )
    result = cli.settle_order(
        trader, ledger, intent_id(plan, "BUY"), "roth-test-hash", deadline_seconds=0
    )
    assert result["state"] == "cancel_unknown"
    assert result["manual_broker_check_required"]
    assert calls == ["cancel_order", "get_order"]


def test_observed_dividend_requires_symbol_bound_broker_data(tmp_path):
    plan = json.loads((ROOT / cli.POLICY_PATH).read_text())
    ledger = LiveOrderLedger(tmp_path / "ledger.sqlite3")
    filled_entry(plan, ledger)
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    rows = [
        {
            "activityId": "private-id",
            "type": "DIVIDEND_OR_INTEREST",
            "description": "CASH DIVIDEND",
            "symbol": "O",
            "netAmount": 1.35,
            "time": now.isoformat(),
        }
    ]
    trader = SimpleNamespace(
        _invoke_client_candidates=lambda **kwargs: {
            "ok": True,
            "response": SimpleNamespace(json=lambda: rows),
        }
    )
    result = cli.dividend_observations(trader, "roth-test-hash", plan, ledger, now=now)
    assert result["state"] == "observed"
    assert result["events"][0]["event_id"] != "private-id"
    del rows[0]["symbol"]
    result = cli.dividend_observations(trader, "roth-test-hash", plan, ledger, now=now)
    assert result["state"] == "incomplete"
    assert result["events"] == []


@pytest.mark.parametrize("cash", ["565.49", 0, None, "NaN", True])
def test_cash_observation_requires_explicit_finite_broker_balance(cash):
    calls = []
    trader = SimpleNamespace(
        _invoke_client_candidates=lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "response": SimpleNamespace(
                json=lambda: {
                    "securitiesAccount": {"currentBalances": {"cashBalance": cash}}
                }
            ),
        }
    )
    result = cli.broker_cash_observation(trader, "private-reference")
    assert (result["state"] == "observed") == (
        cash in ("565.49", 0) and not isinstance(cash, bool)
    )
    assert calls[0]["operation"] == "get_account"
    assert "private-reference" not in json.dumps(result)
    if result["state"] == "observed":
        assert result["settled_cash_certified"] is False


def test_observe_unused_scope_reuses_native_technical_preflight(setup, monkeypatch):
    root, kwargs = setup
    calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    trader = SimpleNamespace(
        _fetch_live_quote=lambda **kw: calls.append("quote") or kwargs["quote"]
    )
    monkeypatch.setattr(
        cli,
        "connect",
        lambda *args, **kw: (trader, kwargs["reference"], kwargs["quote"]),
    )
    monkeypatch.setattr(
        cli, "observe", lambda *args, **kw: {"purchase_scope": {"entry_attempts": 0}}
    )
    monkeypatch.setattr(cli, "_quote_summary", lambda *args, **kw: kwargs["quote"])
    monkeypatch.setattr(
        cli,
        "broker_inventory",
        lambda *args: calls.append("inventory") or kwargs["inventory"],
    )

    def preflight(*args, **kw):
        calls.append("preflight")
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_id": "test",
            "policy_sha256": "test-policy",
            "technical_ready": False,
            "technical_blockers": ["halt_flags_active"],
        }

    monkeypatch.setattr(cli, "assessment", preflight)
    result = cli.run(argparse.Namespace(command="observe"))
    assert calls == ["quote", "inventory", "preflight"]
    assert result["proposal_preflight"]["technical_ready"] is False
    assert not kwargs["ledger"].intents()


def test_account_transaction_query_is_unfiltered_and_rejects_truncation(tmp_path):
    plan = json.loads((ROOT / cli.POLICY_PATH).read_text())
    ledger = LiveOrderLedger(tmp_path / "ledger.sqlite3")
    filled_entry(plan, ledger)
    calls = []
    rows = [{}] * 1000
    trader = SimpleNamespace(
        _invoke_client_candidates=lambda **kw: calls.append(kw)
        or {"ok": True, "response": SimpleNamespace(json=lambda: rows)}
    )
    result = cli.transaction_observations(
        trader, "roth-test-hash", plan, ledger, now=datetime.now(timezone.utc)
    )
    assert not result["source_complete"]
    assert "symbol" not in calls[0]["candidates"][0][2]
    rows.clear()
    result = cli.transaction_observations(
        trader,
        "roth-test-hash",
        plan,
        ledger,
        now=datetime.now(timezone.utc) + timedelta(days=60),
    )
    assert not result["source_complete"]


@pytest.mark.parametrize(
    "scenario", ["confirmed", "declined", "changed_evidence", "autonomy_enabled"]
)
def test_interactive_flow_uses_only_mock_broker_and_exact_confirmation(
    setup, monkeypatch, scenario
):
    root, kwargs = setup
    calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        cli, "component_action_guard", lambda *args, **kw: contextlib.nullcontext()
    )
    trader = SimpleNamespace(
        _fetch_live_quote=lambda **kw: kwargs["quote"],
        broker_adapter=SimpleNamespace(
            place_order_candidates=lambda **kw: [("place_order", (), kw)]
        ),
        _invoke_client_candidates=lambda **kw: calls.append(kw)
        or {"ok": True, "status_code": 201, "order_id": "mock-order"},
    )
    monkeypatch.setattr(
        cli,
        "connect",
        lambda *args, **kw: (trader, kwargs["reference"], kwargs["quote"]),
    )
    monkeypatch.setattr(cli, "broker_inventory", lambda *args: kwargs["inventory"])
    monkeypatch.setattr(cli, "_quote_summary", lambda *args, **kw: kwargs["quote"])
    monkeypatch.setattr(cli, "_refresh_account_study", lambda **kw: {"ok": True})
    attestations = []
    monkeypatch.setattr(
        cli,
        "_issue_attestation",
        lambda *args, **kw: attestations.append(kw) or {"ok": True},
    )
    monkeypatch.setattr(cli, "settle_order", lambda *args: {"state": "submitted"})
    monkeypatch.setattr(
        cli, "observe", lambda *args, **kw: {"state": "reconciliation_pending"}
    )
    real_assessment = cli.assessment

    def assess(*args, **kw):
        result = real_assessment(*args, **kw)
        if scenario == "changed_evidence" and attestations:
            result.update(
                operator_submit_ready=False, blockers=["fresh_quote_required"]
            )
        return result

    monkeypatch.setattr(cli, "assessment", assess)
    confirmations = ["yes"] * (len(required_operator_confirmations("roth_ira")) + 2)
    answers = iter(
        confirmations
        + ["850.14", cli.approval_phrase(kwargs["plan"], kwargs["request"])]
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "no" if scenario == "declined" else next(answers),
    )
    args = argparse.Namespace(
        command="submit", action="BUY", quantity=None, limit_price=None, json=True
    )
    if scenario == "autonomy_enabled":
        monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
        with pytest.raises(ValueError, match="autonomous_execution_disabled"):
            cli.run(args)
    elif scenario == "declined":
        with pytest.raises(ValueError, match="operator_confirmation_incomplete"):
            cli.run(args)
    else:
        result = cli.run(args)
        mutations = [call for call in calls if call["operation"] != "get_account"]
        assert bool(mutations) == (scenario == "confirmed")
        assert attestations[0]["purpose"] == PURPOSE
        if scenario == "confirmed":
            assert result["ok"]
            assert [call["operation"] for call in calls] == [
                "get_account",
                "place_order",
            ]
            assert mutations[0]["candidates"][0][2]["order_spec"] == kwargs["request"]
            assert not kwargs["ledger"].get(intent_id(kwargs["plan"], "SELL"))
    if scenario in {"declined", "autonomy_enabled"}:
        assert not calls and not attestations and not kwargs["ledger"].intents()
