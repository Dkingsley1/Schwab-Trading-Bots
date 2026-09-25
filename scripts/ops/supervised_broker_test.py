#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brokers.capability_contract import evaluate_order_request
from core.broker_test_accounting import reconcile_test_accounting
from core.accountability import safe_write_json_atomic
from core.live_canary_preflight import (
    evaluate_live_canary_preflight,
    required_operator_confirmations,
)
from core.live_execution_envelope import (
    build_live_execution_envelope,
    file_sha256,
    verify_live_execution_envelope,
)
from core.live_order_ledger import LiveOrderLedger, TERMINAL_STATES
from core.order_intent import build_order_intent_evidence, canonical_payload_sha256
from core.storage_router import inspect_storage_path
from core.supervised_broker_test import (
    AUTHORITY,
    PURPOSE,
    account_digest,
    attestation_path,
    approval_phrase,
    build_request,
    build_market_request,
    dispatch_once,
    fresh,
    holding_observation,
    intent_id,
    intent_payload,
    lifecycle_check,
    number,
    policy_path,
    propose_entry,
    reconcile_order,
    request_fields,
    timestamp,
    validate_policy,
    validate_session,
)
from core.system_role_contracts import component_action_guard, evaluate_component_action
from scripts.brokers.schwab.common import build_schwab_trader
from scripts.ops.live_canary_dress_rehearsal import (
    _account_context,
    _quote_summary,
    _read_only_environment,
    _refresh_account_study,
    _refresh_technical_evidence,
    _resolve_account_references,
    _restore_environment,
)
from scripts.ops.live_canary_preflight import CONFIRMATION_PHRASE, _issue_attestation
from scripts.ops.schwab_account_snapshot_refresh import _quiet_auth
from scripts.ops.schwab_tax_ledger_refresh import _kind, _symbol
from scripts.ops import production_excellence_control

POLICY_PATH = "config/supervised_broker_test_v1.json"
LEDGER_PATH = "governance/runtime/live_order_ledger.sqlite3"
STUDY_PATH = "governance/health/account_position_study_latest.json"
REPORT_PATH = "governance/health/supervised_broker_test_latest.json"


def local_path(root: Path, relative: str) -> Path:
    path = root / relative
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"}:
        raise ValueError(f"unsafe_test_path:{route.get('status')}")
    return path


def load(root: Path, relative: str) -> dict[str, Any]:
    try:
        result = json.loads(local_path(root, relative).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def check_evidence_routes(root: Path, symbol: str = "O") -> None:
    firewall = load(root, "config/production_readiness_control_v1.json").get(
        "live_execution_risk_firewall", {}
    )
    for key, value in firewall.items():
        if isinstance(value, str) and (
            key.endswith("_path") or key.endswith("_artifact")
        ):
            local_path(
                root, value.replace("{year}", str(datetime.now(timezone.utc).year))
            )
    for relative in (
        STUDY_PATH,
        LEDGER_PATH,
        LEDGER_PATH + "-wal",
        LEDGER_PATH + "-shm",
        policy_path(symbol),
        attestation_path(symbol),
    ):
        local_path(root, relative)
    release = load(root, "governance/health/release_freeze_guard_latest.json")
    manifest = release.get("immutable_release_boundary", {}).get("manifest_path")
    if manifest:
        local_path(root, str(manifest))


def current_source_blockers(root: Path, candidate: Mapping[str, Any]) -> list[str]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ["test_source_integrity_unavailable"]
    reasons = []
    if status.stdout.strip():
        reasons.append("test_source_release_not_clean")
    config = load(root, "config/production_excellence_v1.json")
    if not config:
        return reasons + ["test_candidate_policy_missing"]
    for path in production_excellence_control._candidate_paths(root, config):
        local_path(root, str(path))
    # Acceptance precedes the commit; bind content and history, not the old HEAD.
    checked = production_excellence_control.manage_candidate(root, config)
    current = checked.get("current", {})
    chain = checked.get("event_chain", {})
    if (
        not candidate.get("candidate_id")
        or checked.get("state") != dict(candidate)
        or checked.get("candidate_drift") is not False
        or checked.get("operation_error")
        or not current.get("overall_sha256")
        or current.get("overall_sha256") != candidate.get("overall_sha256")
        or checked.get("source_coverage", {}).get("ready") is not True
        or chain.get("ok") is not True
        or not chain.get("chain_head")
        or chain.get("chain_head") != candidate.get("event_chain_head")
    ):
        reasons.append("test_source_not_accepted_candidate")
    return reasons


def assessment(
    root: Path,
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    quote: Mapping[str, Any],
    reference: str,
    ledger: LiveOrderLedger,
    inventory: Mapping[str, Any],
    now: datetime,
    bot_handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_policy(plan)
    check_evidence_routes(root, plan["symbol"])
    action, qty, limit = request_fields(plan, request)
    candidate = load(root, "governance/runtime/production_candidate_state.json")
    study = load(root, STUDY_PATH)
    account = _account_context(
        study,
        account_policy_key=plan["account_policy_key"],
        symbol=plan["symbol"],
        canary_symbols={plan["symbol"]},
    )
    preflight = evaluate_live_canary_preflight(
        root,
        symbol=plan["symbol"],
        action=action,
        account_reference=reference,
        purpose=PURPOSE,
        session=request["session"],
        now=now,
    )
    blockers = list(preflight["blockers"])
    market_order = request["orderType"] == "MARKET"
    if market_order:
        from core.schd_bot_handoff import validate_handoff

        blockers.extend(
            validate_handoff(
                bot_handoff or {},
                request=request,
                candidate_id=candidate.get("candidate_id"),
                now=now,
            )
        )
    blockers.extend(current_source_blockers(root, candidate))
    if not reference or not preflight.get("account_reference_matches"):
        blockers.append("pinned_roth_account_required")
    if not fresh(study.get("timestamp_utc"), now, 30):
        blockers.append("fresh_account_observation_required")
    if (
        inventory.get("ok") is not True
        or inventory.get("open_order_count") != 0
        or not fresh(inventory.get("timestamp_utc"), now, 30)
    ):
        blockers.append("fresh_empty_broker_open_order_inventory_required")
    firewall = load(root, "config/production_readiness_control_v1.json").get(
        "live_execution_risk_firewall", {}
    )
    for key in (
        "OPERATOR_STOP",
        "GLOBAL_TRADING_HALT",
        "ALLOW_ORDER_EXECUTION",
        "EXECUTION_LANE_LIVE_ENABLED",
        "TOP_BOT_ENABLE_LIVE_EXECUTION",
    ):
        if str(os.environ.get(key, "0")).lower() in {"1", "true", "yes", "on"}:
            blockers.append(f"test_runtime_flag_conflict:{key}")
    for relative in firewall.get("halt_flags", []):
        if local_path(root, relative).exists():
            blockers.append("halt_flags_active")
    for relative in firewall.get("required_safety_flags", []):
        if not local_path(root, relative).exists():
            blockers.append("required_safety_flag_missing")
    quarantine = load(
        root,
        str(
            firewall.get("schwab_boundary_quarantine_artifact")
            or "governance/health/SCHWAB_BROKER_BOUNDARY_QUARANTINE.json"
        ),
    )
    if quarantine.get("active"):
        blockers.append("schwab_broker_boundary_quarantine_active")
    reserve = load(root, "governance/health/local_storage_reserve_guard_latest.json")
    storage = reserve.get("local_storage_reserve", {})
    if (
        not fresh(reserve.get("timestamp_utc"), now, 300)
        or storage.get("pressure_active") is not False
        or storage.get("hard_block") is not False
    ):
        blockers.append("current_storage_write_headroom_required")
    pressure_floor = max(float(storage.get("pressure_free_gb") or 64), 64)
    if (
        shutil.disk_usage(local_path(root, "governance/runtime")).free / 1024**3
        < pressure_floor
    ):
        blockers.append("current_storage_below_pressure_floor")
    role = evaluate_component_action(
        root,
        component_id="live_execution_gateway",
        action="live_submit",
        state_domain="live_order_submission",
    )
    if not role.get("ok"):
        blockers.append("system_role_contract_live_submit_denied")
    context = account.get("covered_position_safety", {}).get(
        "collateral_by_underlying", []
    )
    unencumbered = next(
        (
            row["unencumbered_equity_shares"]
            for row in context
            if row.get("underlying") == plan["symbol"]
        ),
        0,
    )
    blockers.extend(
        lifecycle_check(
            plan,
            request,
            ledger,
            account_reference=reference,
            position_quantity=account["candidate_symbol_quantity"],
            unencumbered_quantity=unencumbered,
            account_captured_at=str(study.get("timestamp_utc") or ""),
        )
    )
    if (
        quote.get("source_provider") != "schwab_api"
        or quote.get("realtime") is not True
        or quote.get("transport", {}).get("ok") is not True
    ):
        blockers.append("realtime_schwab_quote_required")
    if not fresh(quote.get("provider_timestamp_utc"), now, 15):
        blockers.append("fresh_quote_required")
    extended_quote_expiry = ""
    if request["session"] != "NORMAL":
        if quote.get("symbol") != plan["symbol"]:
            blockers.append("extended_quote_symbol_mismatch")
        for side in ("bid", "ask"):
            try:
                age = (
                    now - timestamp(quote.get(f"{side}_timestamp_utc"))
                ).total_seconds()
                valid = 0 <= age <= 15 and number(quote.get(f"{side}_size", 0)) > 0
            except (ValueError, TypeError):
                valid = False
            if not valid:
                blockers.append(f"fresh_extended_{side}_and_size_required")
        try:
            extended_quote_expiry = (
                min(
                    timestamp(quote.get(f"{side}_timestamp_utc"))
                    for side in ("bid", "ask")
                )
                + timedelta(seconds=15)
            ).isoformat()
        except (ValueError, TypeError):
            pass
    try:
        bid, ask = number(quote.get("bid_price")), number(quote.get("ask_price"))
        spread = (ask - bid) / ((ask + bid) / 2) * 10000
        if (
            bid <= 0
            or ask <= bid
            or spread > number(plan["hard_limits"]["max_spread_bps"])
        ):
            blockers.append("quote_spread_invalid")
        if not market_order and action == "BUY" and limit > bid:
            blockers.append("buy_limit_above_fresh_bid_no_chase")
        if request["session"] != "NORMAL" and abs(limit - bid) / bid * 10000 > number(
            plan["hard_limits"]["max_limit_distance_bps"]
        ):
            blockers.append("extended_limit_too_far_from_bid")
        if (
            not market_order
            and action == "SELL"
            and limit
            < bid * (1 - number(plan["hard_limits"]["max_limit_distance_bps"]) / 10000)
        ):
            blockers.append("sell_limit_too_far_below_bid")
    except (ValueError, ArithmeticError):
        bid = ask = spread = number(0)
        blockers.append("quote_prices_invalid")
    funding_price = ask * number("1.0035") if market_order else limit
    if market_order and funding_price * qty + number(
        plan["hard_limits"]["cost_reserve_usd"]
    ) > number(plan["account_capital_usd"]):
        blockers.append("estimated_market_cost_above_test_budget")
    if market_order and action == "SELL":
        entry = ledger.get(intent_id(plan, "BUY"))
        if bid <= number(entry.get("average_fill_price", 0)):
            blockers.append("sell_quote_not_above_verified_entry_price")
    if action == "BUY" and number(
        account["settled_cash_broker_visible_usd"]
    ) < funding_price * qty + number(plan["hard_limits"]["cost_reserve_usd"]):
        blockers.append("test_not_fully_cash_funded")
    order_request = {
        "symbol": plan["symbol"],
        "action": action,
        "quantity": qty,
        "limit_price": float(limit),
        "asset_type": "EQUITY",
        "account_reference": reference,
        "order_spec": dict(request),
    }
    capability = evaluate_order_request(
        "schwab", order_request, mode="live", require_production_eligible=True
    )
    if not capability.get("ok"):
        blockers.append("broker_capability_contract_blocked")
    quote_snapshot = {
        "timestamp_utc": quote.get("provider_timestamp_utc"),
        "bid_price": float(bid),
        "ask_price": float(ask),
        "source_provider": quote.get("source_provider"),
        "source_venue": quote.get("source_venue"),
        "snapshot_id": quote.get("snapshot_id"),
    }
    intent = build_order_intent_evidence(
        decision_id=intent_id(plan, action),
        symbol=plan["symbol"],
        action=action,
        quantity=qty,
        strategy=PURPOSE,
        limit_price=float(limit),
        quote_snapshot=quote_snapshot,
        risk_decision={
            "ok": not blockers,
            "gate": "supervised_test_technical_preflight",
            "reason": "ready" if not blockers else "blocked",
        },
    )
    envelope = build_live_execution_envelope(
        intent_evidence=intent,
        order_request=order_request,
        candidate_id=str(candidate.get("candidate_id") or ""),
        broker="schwab",
        account_reference=reference,
        policy_sha256=file_sha256(local_path(root, policy_path(plan["symbol"]))),
        account_snapshot_evidence={
            "broker_position_snapshot_sha256": file_sha256(
                local_path(root, STUDY_PATH)
            ),
            "broker_position_snapshot_captured_at_utc": study.get("timestamp_utc"),
            "live_canary_preflight_receipt": preflight,
        },
        created_at_utc=now,
        ttl_seconds=15,
    )
    verification = verify_live_execution_envelope(
        envelope,
        expected_candidate_id=str(candidate.get("candidate_id") or ""),
        expected_account_reference=reference,
        expected_policy_sha256=file_sha256(
            local_path(root, policy_path(plan["symbol"]))
        ),
        now_utc=now,
        require_affirmative_risk_decision=True,
        require_quote_provenance=True,
        allowed_quote_providers=("schwab_api",),
        require_canary_preflight_receipt=True,
        expected_account_policy_key=plan["account_policy_key"],
    )
    blockers.extend(verification["blockers"])
    operator_blockers = set(preflight.get("operator_attestation_blockers", []))
    derived = {
        "order_intent_risk_decision_not_approved",
        "live_canary_preflight_receipt_not_ready",
    }
    technical_blockers = [
        value for value in blockers if value not in operator_blockers | derived
    ]
    return {
        "timestamp_utc": now.isoformat(),
        "purpose": PURPOSE,
        "test_id": plan["test_id"],
        "candidate_id": candidate.get("candidate_id"),
        "account_reference_sha256": account_digest(reference),
        "request_sha256": canonical_payload_sha256(request),
        "policy_sha256": canonical_payload_sha256(plan),
        "extended_quote_expires_at_utc": extended_quote_expiry,
        "technical_ready": not technical_blockers,
        "operator_attestation_ready": preflight["operator_attestation_ready"],
        "operator_submit_ready": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "technical_blockers": list(dict.fromkeys(technical_blockers)),
        "position_quantity": account["candidate_symbol_quantity"],
        "settled_cash_usd": account["settled_cash_broker_visible_usd"],
        "request": dict(request),
        "bot_handoff": dict(bot_handoff or {}),
        "market_price_not_guaranteed": market_order,
        "preflight": preflight,
        "envelope": envelope,
        "production_validation": "separate_not_waived_or_credited",
        **AUTHORITY,
    }


def open_ledger(root: Path) -> LiveOrderLedger:
    path = local_path(root, LEDGER_PATH)
    if not path.is_file():
        raise ValueError("existing_native_order_ledger_required")
    local_path(root, LEDGER_PATH + "-wal")
    local_path(root, LEDGER_PATH + "-shm")
    return LiveOrderLedger(path)


def broker_inventory(trader: Any, reference: str) -> dict[str, Any]:
    result = trader._invoke_client_candidates(
        operation="get_orders_snapshot",
        candidates=trader.broker_adapter.orders_snapshot_candidates(
            account_reference=reference
        ),
        context={"purpose": PURPOSE},
    )
    raw = result.get("response")
    try:
        rows = raw.json()
    except Exception:
        rows = None
    terminal = {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}
    valid = (
        result.get("ok") is True
        and isinstance(rows, list)
        and len(rows) < 500
        and all(isinstance(row, dict) and row.get("status") for row in rows)
    )
    return {
        "ok": valid,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "open_order_count": (
            sum(str(row["status"]).upper() not in terminal for row in rows)
            if valid
            else None
        ),
        "lookback_days": 60,
        "older_open_orders_require_broker_ui_review": True,
    }


def connect(
    root: Path, plan: Mapping[str, Any], *, refresh_technical: bool = True
) -> tuple[Any, str, dict[str, Any]]:
    check_evidence_routes(root, plan["symbol"])
    technical = _refresh_technical_evidence() if refresh_technical else {"ok": True}
    refreshed = _refresh_account_study(quiet_auth=True)
    if not technical.get("ok") or not refreshed.get("ok"):
        raise ValueError("native_technical_or_account_refresh_failed")
    registry = load(root, "config/account_policy_registry.json")
    reference, expected, _ = _resolve_account_references(plan=plan, registry=registry)
    if not reference or reference != expected:
        raise ValueError("designated_roth_account_binding_required")
    slot = next(
        row
        for row in registry["account_slots"]
        if row.get("account_policy_key") == plan["account_policy_key"]
    )
    for name in slot.get("env_names", []):
        if (
            str(name).endswith("_HASH")
            and name != "SCHWAB_ACCOUNT_HASH"
            and not os.environ.get(name)
        ):
            os.environ[name] = expected
    os.environ["SCHWAB_ACCOUNT_HASH"] = reference
    os.environ["SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER"] = "0"
    trader = build_schwab_trader(root, mode="shadow")
    _quiet_auth(trader, quiet=True)
    trader.live_account_hash = reference
    quote = _quote_summary(
        trader._fetch_live_quote(symbol=plan["symbol"]),
        symbol=plan["symbol"],
        now=datetime.now(timezone.utc),
    )
    return trader, reference, quote


def fetch_and_record(
    trader: Any, ledger: LiveOrderLedger, row: Mapping[str, Any], reference: str
) -> dict[str, Any]:
    result = trader._invoke_client_candidates(
        operation="get_order",
        candidates=[
            (
                "get_order",
                (),
                {"account_hash": reference, "order_id": str(row["broker_order_id"])},
            )
        ],
        context={"purpose": PURPOSE},
    )
    if not result.get("ok"):
        raise ValueError("broker_order_read_failed_reconciliation_required")
    return reconcile_order(ledger, row, result.get("response_payload", {}))


def settle_order(
    trader: Any,
    ledger: LiveOrderLedger,
    key: str,
    reference: str,
    *,
    deadline_seconds: int = 60,
) -> dict[str, Any]:
    row = ledger.get(key)
    age = max(
        (datetime.now(timezone.utc) - timestamp(row["created_at_utc"])).total_seconds(),
        0,
    )
    deadline = time.monotonic() + max(deadline_seconds - age, 0)
    if not row.get("broker_order_id"):
        return {
            "state": row.get("state"),
            "reconciliation_required": True,
            "manual_broker_check_required": True,
        }
    while time.monotonic() < deadline:
        try:
            row = fetch_and_record(trader, ledger, row, reference)
        except (ValueError, OSError):
            break
        if row["state"] in TERMINAL_STATES:
            return {
                "state": row["state"],
                "filled_quantity": row["filled_quantity"],
                "average_fill_price": row["average_fill_price"],
                "account_reconciliation_required": True,
            }
        time.sleep(min(2, max(deadline - time.monotonic(), 0)))
    broker_id = str(row["broker_order_id"])
    if row["state"] not in {"cancel_pending", "cancel_unknown"}:
        ledger.mark_cancel_pending(broker_id)
        try:
            result = trader._invoke_client_candidates(
                operation="cancel_order",
                candidates=[
                    (
                        "cancel_order",
                        (),
                        {"account_hash": reference, "order_id": broker_id},
                    )
                ],
                context={"purpose": PURPOSE},
            )
        except Exception:
            result = {"ok": False}
        if not result.get("ok"):
            ledger.mark_cancel_unknown(
                broker_id, error="test_deadline_cancel_outcome_unknown"
            )
    try:
        row = fetch_and_record(trader, ledger, ledger.get(key), reference)
    except (ValueError, OSError):
        return {
            "state": ledger.get(key)["state"],
            "reconciliation_required": True,
            "manual_broker_check_required": True,
            "automatic_retry_allowed": False,
        }
    return {
        "state": row["state"],
        "reconciliation_required": row["state"] not in TERMINAL_STATES,
        "account_reconciliation_required": True,
        "automatic_retry_allowed": False,
    }


def broker_cash_observation(trader: Any, reference: str) -> dict[str, Any]:
    result = trader._invoke_client_candidates(
        operation="get_account",
        candidates=[("get_account", (), {"account_hash": reference})],
        context={"purpose": PURPOSE},
    )
    try:
        raw = result["response"].json()
        cash = number(raw["securitiesAccount"]["currentBalances"]["cashBalance"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"state": "explicit_current_cash_balance_unavailable"}
    if result.get("ok") is not True:
        return {"state": "broker_cash_read_failed"}
    return {
        "state": "observed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "account_reference_sha256": account_digest(reference),
        "source": "schwab_currentBalances.cashBalance",
        "balance_usd": str(cash),
        "broker_payload_sha256": canonical_payload_sha256(raw),
        "settled_cash_certified": False,
    }


def observe(
    root: Path,
    *,
    plan: Mapping[str, Any],
    trader: Any,
    reference: str,
    ledger: LiveOrderLedger,
) -> dict[str, Any]:
    for side in ("BUY", "SELL"):
        row = ledger.get(intent_id(plan, side))
        if row:
            if intent_payload(row).get("account_reference_sha256") != account_digest(
                reference
            ):
                raise ValueError("test_account_mismatch")
            if row.get("broker_order_id"):
                fetch_and_record(trader, ledger, row, reference)
    refreshed = _refresh_account_study(quiet_auth=True)
    if not refreshed.get("ok"):
        raise ValueError("account_refresh_failed")
    study = load(root, STUDY_PATH)
    context = _account_context(
        study,
        account_policy_key=plan["account_policy_key"],
        symbol=plan["symbol"],
        canary_symbols={plan["symbol"]},
    )
    if not context["account_found"]:
        raise ValueError("designated_roth_account_truth_missing")
    cash = broker_cash_observation(trader, reference)
    now = datetime.now(timezone.utc)
    source = transaction_observations(trader, reference, plan, ledger, now=now)
    now = datetime.now(timezone.utc)
    transactions = dividend_observations(
        trader, reference, plan, ledger, now=now, source=source
    )
    result = holding_observation(
        plan=plan,
        ledger=ledger,
        account_reference=reference,
        position_quantity=context["candidate_symbol_quantity"],
        account_captured_at=str(study.get("timestamp_utc") or ""),
        now=now,
        dividend_events=transactions["events"],
        transactions=source,
    )
    result["dividend_tracking"] = {
        key: value for key, value in transactions.items() if key != "events"
    }
    result["cash_reconciliation"] = "not_certified_by_position_observation"
    orders = [ledger.get(intent_id(plan, side)) for side in ("BUY", "SELL")]
    result["accounting"] = reconcile_test_accounting(
        plan=plan,
        orders=[row for row in orders if row],
        transactions=source,
        reference=reference,
        position_consistent=not result["blockers"]
        and result["state"]
        in {"holding_observed", "round_trip_observed", "entry_not_filled"},
        cash_observation=cash,
        now=now,
    )
    entry = orders[0]
    result["purchase_scope"] = {
        "test_policy_sha256": canonical_payload_sha256(plan),
        "account_policy_key": plan["account_policy_key"],
        "symbol": plan["symbol"],
        "entry_attempts": int(bool(entry)),
        "entry_state": entry.get("state", "not_started"),
        "entry_created_at_utc": entry.get("created_at_utc"),
        "entry_gross_usd": str(
            number(entry.get("filled_quantity", 0))
            * number(entry.get("average_fill_price", 0))
        ),
        "position_quantity": context["candidate_symbol_quantity"],
        "funding_proxy_usd": context["settled_cash_broker_visible_usd"],
        "settled_cash_certified": False,
    }
    return result


def transaction_observations(
    trader: Any,
    reference: str,
    plan: Mapping[str, Any],
    ledger: LiveOrderLedger,
    *,
    now: datetime,
) -> dict[str, Any]:
    entry = ledger.get(intent_id(plan, "BUY"))
    if not entry or number(entry.get("filled_quantity", 0)) <= 0:
        return {"state": "not_started", "rows": [], "source_complete": False}
    baseline = intent_payload(entry).get("baseline_cash_observation", {})
    origin = timestamp(baseline.get("timestamp_utc") or entry["created_at_utc"])
    start = max(origin, now - timedelta(days=59))
    result = trader._invoke_client_candidates(
        operation="get_transactions",
        candidates=[
            (
                "get_transactions",
                (),
                {
                    "account_hash": reference,
                    "start_date": start,
                    "end_date": now,
                },
            )
        ],
        context={"purpose": PURPOSE},
    )
    try:
        rows = result["response"].json()
    except Exception:
        rows = None
    if (
        not result.get("ok")
        or not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
    ):
        return {
            "state": "broker_transaction_read_unavailable",
            "rows": [],
            "source_complete": False,
        }
    return {
        "rows": rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "account_reference_sha256": account_digest(reference),
        "source_complete": len(rows) < 1000 and start == origin,
        "window_start_utc": start.isoformat(),
        "window_end_utc": now.isoformat(),
    }


def dividend_observations(
    trader: Any,
    reference: str,
    plan: Mapping[str, Any],
    ledger: LiveOrderLedger,
    *,
    now: datetime,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = (
        source
        if source is not None
        else transaction_observations(trader, reference, plan, ledger, now=now)
    )
    if not source.get("window_start_utc"):
        return {
            "state": source.get("state", "incomplete"),
            "events": [],
            "source_complete": False,
        }
    rows = source["rows"]
    start = timestamp(source["window_start_utc"])
    events = []
    unresolved = 0
    for row in rows:
        if _kind(row, action="UNKNOWN") != "dividend":
            continue
        if _symbol(row) and _symbol(row) != plan["symbol"]:
            continue
        event_id = str(row.get("activityId") or row.get("transactionId") or "")
        when = row.get("transactionDate") or row.get("time") or row.get("tradeDate")
        try:
            amount = number(row.get("netAmount"))
            observed = timestamp(when)
        except (TypeError, ValueError):
            unresolved += 1
            continue
        if (
            not event_id
            or _symbol(row) != plan["symbol"]
            or not start <= observed <= now
            or amount < 0
        ):
            unresolved += 1
            continue
        events.append(
            {
                "event_id": account_digest(event_id),
                "account_policy_key": plan["account_policy_key"],
                "symbol": plan["symbol"],
                "tax_event_kind": "dividend",
                "transaction_date": observed.isoformat(),
                "amount_usd": str(amount),
                "broker_payload_sha256": canonical_payload_sha256(row),
            }
        )
    complete = not unresolved and source.get("source_complete") is True
    return {
        "state": "observed" if events else "not_observed" if complete else "incomplete",
        "events": events,
        "source_complete": complete,
        "window_start_utc": start.isoformat(),
        "window_end_utc": now.isoformat(),
        "unresolved_dividend_rows": unresolved,
        "tax_characterization": "not_certified",
    }


@contextlib.contextmanager
def test_lock(root: Path):
    path = local_path(root, "governance/locks/supervised_broker_test.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = PROJECT_ROOT
    symbol = getattr(args, "symbol", "O")
    session = getattr(args, "session", "NORMAL")
    plan = load(root, policy_path(symbol))
    validate_policy(plan)
    validate_session(plan, session)
    bot_market = getattr(args, "bot_market", False)
    if bot_market and (
        symbol != "SCHD"
        or session != "NORMAL"
        or args.quantity is not None
        or args.limit_price is not None
    ):
        raise ValueError(
            "bot market test requires SCHD NORMAL with fixed one-share quantity and no limit override"
        )
    if args.command == "status":
        return {
            "purpose": PURPOSE,
            "state": "operator_controlled_not_armed",
            "plan": plan,
            "selected_session": session,
            "connected": False,
            **AUTHORITY,
        }
    if args.command == "attestation-checklist":
        return {
            "purpose": PURPOSE,
            "state": "checklist_only_not_attested",
            "account": plan["account_policy_key"],
            "required_confirmations": list(required_operator_confirmations("roth_ira"))
            + ["broker_open_orders_reviewed", "no_concurrent_manual_orders_confirmed"]
            + (["extended_hours_risk_reviewed"] if session in {"AM", "PM"} else [])
            + (["market_order_price_risk_reviewed"] if bot_market else []),
            "current_settled_cash_required": True,
            "confirmation_each_order": True,
            "issue_only_after_technical_ready": True,
            **AUTHORITY,
        }
    if args.command == "readiness":
        previous = _read_only_environment()
        try:
            check_evidence_routes(root, symbol)
            technical = _refresh_technical_evidence()
            preflight = evaluate_live_canary_preflight(
                root,
                symbol=symbol,
                action=args.action,
                purpose=PURPOSE,
                session=session,
            )
            candidate = load(root, "governance/runtime/production_candidate_state.json")
            source_blockers = current_source_blockers(root, candidate)
            blockers = list(
                dict.fromkeys(
                    technical["blockers"] + preflight["blockers"] + source_blockers
                )
            )
            return {
                "purpose": PURPOSE,
                "state": "readiness_observation_not_authorization",
                "ok": not blockers,
                "blockers": blockers,
                "technical_refresh": technical,
                "preflight": preflight,
                "source_blockers": source_blockers,
                **AUTHORITY,
            }
        finally:
            _restore_environment(previous)
    if args.command == "submit" and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("interactive_operator_terminal_required_no_unattended_submit")
    if args.command == "submit" and any(
        str(os.environ.get(key, "0")).lower() in {"1", "true", "yes", "on"}
        for key in (
            "ALLOW_ORDER_EXECUTION",
            "EXECUTION_LANE_LIVE_ENABLED",
            "TOP_BOT_ENABLE_LIVE_EXECUTION",
        )
    ):
        raise ValueError("test_requires_autonomous_execution_disabled")
    registry = load(root, "config/account_policy_registry.json")
    names = {"SCHWAB_ACCOUNT_HASH", "SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER"}
    for slot in registry.get("account_slots", []):
        if slot.get("account_policy_key") == plan["account_policy_key"]:
            names.update(
                name
                for name in slot.get("env_names", [])
                if str(name).endswith("_HASH")
            )
    previous = _read_only_environment()
    previous.update({name: os.environ.get(name) for name in names})
    try:
        if bot_market:
            from core.decision_price_evidence import build_evidence
            from scripts.ops.schd_bot_handoff import observe_handoff

            observed = observe_handoff(
                root, plan, {}, action=args.action, now=datetime.now(timezone.utc)
            )
            packet = observed.get("receipt", {}).get("packet")
            evidence_blockers = (
                build_evidence(packet, now=datetime.now(timezone.utc))["blockers"]
                if packet
                else observed["blockers"]
            )
            if packet and packet["decision"]["action"] != args.action:
                evidence_blockers = list(evidence_blockers) + [
                    "bot_action_does_not_match_requested_side"
                ]
            if evidence_blockers:
                return {
                    "state": "blocked",
                    "purpose": PURPOSE,
                    "blockers": evidence_blockers,
                    "native_decision": packet.get("decision") if packet else None,
                    **AUTHORITY,
                }
        lease_path = (
            load(root, "config/system_role_contracts_v1.json")
            .get("action_leases", {})
            .get("live_submit", {})
            .get("path", "governance/locks/live_order_submission_authority.lock")
        )
        local_path(root, lease_path)
        writer = (
            component_action_guard(
                root,
                component_id="live_execution_gateway",
                action="live_submit",
                state_domain="live_order_submission",
            )
            if args.command == "submit"
            else contextlib.nullcontext()
        )
        with test_lock(root), writer:
            trader, reference, quote = connect(
                root, plan, refresh_technical=args.command != "observe"
            )
            ledger = open_ledger(root)
            if args.command == "observe":
                result = observe(
                    root, plan=plan, trader=trader, reference=reference, ledger=ledger
                )
                if result.get("purchase_scope", {}).get("entry_attempts") == 0:
                    quote = _quote_summary(
                        trader._fetch_live_quote(symbol=plan["symbol"]),
                        symbol=plan["symbol"],
                        now=datetime.now(timezone.utc),
                    )
                    draft = propose_entry(plan, quote, now=datetime.now(timezone.utc))
                    if draft.get("state") == "proposed":
                        review = assessment(
                            root,
                            plan=plan,
                            request=draft["request"],
                            quote=quote,
                            reference=reference,
                            ledger=ledger,
                            inventory=broker_inventory(trader, reference),
                            now=datetime.now(timezone.utc),
                        )
                        result["proposal_preflight"] = {
                            key: review[key]
                            for key in (
                                "timestamp_utc",
                                "candidate_id",
                                "policy_sha256",
                                "technical_ready",
                                "technical_blockers",
                            )
                        }
                result["quote"] = quote
                return result
            proposal = propose_entry(
                plan, quote, now=datetime.now(timezone.utc), session=session
            )
            handoff = None
            if bot_market:
                request = build_market_request(plan, action=args.action)
                handoff = observe_handoff(
                    root,
                    plan,
                    quote,
                    action=args.action,
                    now=datetime.now(timezone.utc),
                )["receipt"]
            elif (
                args.action == "BUY"
                and args.limit_price is None
                and args.quantity is None
            ):
                if proposal["state"] != "proposed":
                    return proposal
                request = proposal["request"]
            elif args.limit_price is None or args.quantity is None:
                raise ValueError(
                    "explicit SELL or overridden order needs both quantity and limit price"
                )
            else:
                request = build_request(
                    plan,
                    action=args.action,
                    quantity=args.quantity,
                    limit_price=args.limit_price,
                    session=session,
                )
            inventory = broker_inventory(trader, reference)
            review = assessment(
                root,
                plan=plan,
                request=request,
                quote=quote,
                reference=reference,
                ledger=ledger,
                inventory=inventory,
                now=datetime.now(timezone.utc),
                bot_handoff=handoff,
            )
            review["price_proposal"] = (
                {
                    "state": "native_bot_market_request",
                    "price_guaranteed": False,
                    "automatic_sell": False,
                }
                if bot_market
                else (
                    {"state": "operator_specified_limit", "request": request,
                     "blockers": [], **AUTHORITY}
                    if args.limit_price is not None and args.quantity is not None
                    else proposal
                )
            )
            if args.command == "preview" or not review["technical_ready"]:
                return review
            from core.live_execution_switch import check_live_execution_switch

            switch_context = {"purpose": PURPOSE, "symbol": plan["symbol"], "session": request["session"]}
            switch_check = check_live_execution_switch(
                root, broker="schwab", operation="place_order", context=switch_context,
            )
            if not switch_check["allowed"]:
                return {"ok": False, "state": "blocked", "blockers": switch_check["blockers"],
                        "broker_mutation_attempted": False, **AUTHORITY}
            phrase = approval_phrase(plan, request)
            print(
                json.dumps(
                    {
                        "request": request,
                        "account": plan["account_policy_key"],
                        "budget_usd": plan["account_capital_usd"],
                        "cost_reserve_usd": plan["hard_limits"]["cost_reserve_usd"],
                        "buy_and_hold": plan["investment_style"] == "buy_and_hold",
                        "cancel_if_unfilled_seconds": 60,
                    },
                    indent=2,
                )
            )
            print(
                "Confirm each reviewed account/risk requirement, then this exact order. SELL is a separate test, never automatic."
            )
            if bot_market:
                print(
                    "MARKET execution price is not guaranteed. The $100 check is a preflight estimate, not a broker price cap; even a SELL quoted above entry can fill lower."
                )
                if (
                    input("market_order_price_risk_reviewed [yes/no]: ").strip().lower()
                    != "yes"
                ):
                    raise ValueError("market_order_price_risk_confirmation_required")
            if session != "NORMAL":
                print(
                    "Extended hours: lower liquidity, wider spreads, partial/no fills, and prices that may differ across venues. No market fallback or overnight carry."
                )
                if (
                    input("extended_hours_risk_reviewed [yes/no]: ").strip().lower()
                    != "yes"
                ):
                    raise ValueError("extended_hours_risk_confirmation_required")
            for field in required_operator_confirmations("roth_ira") + (
                "broker_open_orders_reviewed",
                "no_concurrent_manual_orders_confirmed",
            ):
                if input(f"{field} [yes/no]: ").strip().lower() != "yes":
                    raise ValueError("operator_confirmation_incomplete")
            settled_cash = number(
                input("Current settled cash shown by Schwab (USD): ").strip()
            )
            if settled_cash < number(plan["account_capital_usd"]):
                raise ValueError("settled_cash_below_test_budget")
            approved = input(f"Type exactly: {phrase}\n").strip()
            if approved != phrase:
                raise ValueError("exact_order_confirmation_required")
            approved_at = datetime.now(timezone.utc)
            # Refresh again after human review; never change the approved price or size.
            refreshed = _refresh_account_study(quiet_auth=True)
            if not refreshed.get("ok"):
                raise ValueError("post_confirmation_account_refresh_failed")
            issued = _issue_attestation(
                root,
                settled_cash_usd=float(settled_cash),
                duration_minutes=5,
                confirmation=CONFIRMATION_PHRASE,
                confirm_all=True,
                confirm_retirement_account_risk=True,
                purpose=PURPOSE,
                test_symbol=symbol,
                session=session,
                confirm_extended_hours_risk=session != "NORMAL",
            )
            if not issued.get("ok"):
                raise ValueError("test_attestation_not_issued")
            inventory = broker_inventory(trader, reference)
            cash_baseline = broker_cash_observation(trader, reference)
            quote = _quote_summary(
                trader._fetch_live_quote(symbol=plan["symbol"]),
                symbol=plan["symbol"],
                now=datetime.now(timezone.utc),
            )
            if bot_market:
                refreshed_handoff = observe_handoff(
                    root,
                    plan,
                    quote,
                    action=args.action,
                    now=datetime.now(timezone.utc),
                )
                if refreshed_handoff["receipt"].get(
                    "decision_binding_sha256"
                ) != handoff.get("decision_binding_sha256"):
                    raise ValueError(
                        "bot_decision_changed_during_confirmation_review_again"
                    )
                handoff = refreshed_handoff["receipt"]
            review = assessment(
                root,
                plan=plan,
                request=request,
                quote=quote,
                reference=reference,
                ledger=ledger,
                inventory=inventory,
                now=datetime.now(timezone.utc),
                bot_handoff=handoff,
            )
            if not review["operator_submit_ready"]:
                return review
            review["cash_balance_observation"] = cash_baseline
            review["market_order_price_risk_reviewed"] = bot_market
            switch_check = check_live_execution_switch(
                root, broker="schwab", operation="place_order", context=switch_context,
            )
            if not switch_check["allowed"]:
                return {"ok": False, "state": "blocked", "blockers": switch_check["blockers"],
                        "broker_mutation_attempted": False, **AUTHORITY}
            result = dispatch_once(
                plan=plan,
                request=request,
                ledger=ledger,
                assessment=review,
                approved_phrase=approved,
                approved_at=approved_at,
                now=datetime.now(timezone.utc),
                dispatch=lambda spec: trader._invoke_client_candidates(
                    operation="place_order",
                    candidates=trader.broker_adapter.place_order_candidates(
                        account_reference=reference, order_spec=spec
                    )[:1],
                    context={
                        "purpose": PURPOSE,
                        "symbol": plan["symbol"],
                        "action": args.action,
                        "session": request["session"],
                    },
                ),
            )
            if result.get("broker_mutation_attempted"):
                result["order_closeout"] = settle_order(
                    trader, ledger, result["intent_id"], reference
                )
                result["holding_observation"] = observe(
                    root, plan=plan, trader=trader, reference=reference, ledger=ledger
                )
            return result
    finally:
        _restore_environment(previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Operator-only broker functionality test. Default status is offline; no autonomous orders or production promotion."
    )
    parser.add_argument(
        "command",
        choices=(
            "status",
            "preview",
            "submit",
            "observe",
            "readiness",
            "attestation-checklist",
        ),
        nargs="?",
        default="status",
    )
    parser.add_argument("--action", choices=("BUY", "SELL"), default="BUY")
    parser.add_argument("--symbol", choices=("O", "SCHD"), default="O")
    parser.add_argument("--session", choices=("NORMAL", "AM", "PM"), default="NORMAL")
    parser.add_argument("--quantity")
    parser.add_argument("--limit-price")
    parser.add_argument(
        "--bot-market",
        action="store_true",
        help="Native SCHD decision, one share, regular session, separate operator confirmation per order",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "purpose": PURPOSE,
            "state": "blocked",
            "error": type(exc).__name__,
            "reason": (
                str(exc)
                if isinstance(exc, ValueError)
                else "test_failed_check_native_order_ledger_and_broker_before_retry"
            ),
            "manual_broker_check_required": args.command == "submit",
            **AUTHORITY,
        }
    if args.command not in {"status", "attestation-checklist", "readiness"}:
        try:
            report_path = (
                REPORT_PATH
                if args.symbol == "O"
                else "governance/health/supervised_schd_broker_test_latest.json"
            )
            destination = local_path(PROJECT_ROOT, report_path)
            written = safe_write_json_atomic(
                str(destination),
                result,
                project_root=str(PROJECT_ROOT),
                source="supervised_broker_test",
            )
            if written is False:
                result["report_persistence_failed"] = True
        except Exception:
            result["report_persistence_failed"] = True
    print(json.dumps(result, ensure_ascii=True, indent=None if args.json else 2))
    return (
        2
        if result.get("blockers")
        or result.get("state") == "blocked"
        or result.get("operator_submit_ready") is False
        or result.get("ok") is False
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
