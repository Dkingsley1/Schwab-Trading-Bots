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
    approval_phrase,
    build_request,
    dispatch_once,
    fresh,
    holding_observation,
    intent_id,
    intent_payload,
    lifecycle_check,
    number,
    propose_entry,
    reconcile_order,
    request_fields,
    timestamp,
    validate_policy,
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


def check_evidence_routes(root: Path) -> None:
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
        POLICY_PATH,
        "governance/runtime/supervised_broker_test_attestation.json",
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
) -> dict[str, Any]:
    validate_policy(plan)
    check_evidence_routes(root)
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
        now=now,
    )
    blockers = list(preflight["blockers"])
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
        quote.get("source_provider") != "schwab"
        or quote.get("realtime") is not True
        or quote.get("transport", {}).get("ok") is not True
    ):
        blockers.append("realtime_schwab_quote_required")
    if not fresh(quote.get("provider_timestamp_utc"), now, 15):
        blockers.append("fresh_quote_required")
    try:
        bid, ask = number(quote.get("bid_price")), number(quote.get("ask_price"))
        spread = (ask - bid) / ((ask + bid) / 2) * 10000
        if (
            bid <= 0
            or ask <= bid
            or spread > number(plan["hard_limits"]["max_spread_bps"])
        ):
            blockers.append("quote_spread_invalid")
        if action == "BUY" and limit > bid:
            blockers.append("buy_limit_above_fresh_bid_no_chase")
        if action == "SELL" and limit < bid * (
            1 - number(plan["hard_limits"]["max_limit_distance_bps"]) / 10000
        ):
            blockers.append("sell_limit_too_far_below_bid")
    except (ValueError, ArithmeticError):
        bid = ask = spread = number(0)
        blockers.append("quote_prices_invalid")
    if action == "BUY" and number(
        account["settled_cash_broker_visible_usd"]
    ) < limit * qty + number(plan["hard_limits"]["cost_reserve_usd"]):
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
        "source_provider": "schwab",
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
        policy_sha256=file_sha256(local_path(root, POLICY_PATH)),
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
        expected_policy_sha256=file_sha256(local_path(root, POLICY_PATH)),
        now_utc=now,
        require_affirmative_risk_decision=True,
        require_quote_provenance=True,
        allowed_quote_providers=("schwab",),
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
        "technical_ready": not technical_blockers,
        "operator_attestation_ready": preflight["operator_attestation_ready"],
        "operator_submit_ready": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "technical_blockers": list(dict.fromkeys(technical_blockers)),
        "position_quantity": account["candidate_symbol_quantity"],
        "settled_cash_usd": account["settled_cash_broker_visible_usd"],
        "request": dict(request),
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
    check_evidence_routes(root)
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
    now = datetime.now(timezone.utc)
    transactions = dividend_observations(trader, reference, plan, ledger, now=now)
    result = holding_observation(
        plan=plan,
        ledger=ledger,
        account_reference=reference,
        position_quantity=context["candidate_symbol_quantity"],
        account_captured_at=str(study.get("timestamp_utc") or ""),
        now=now,
        dividend_events=transactions["events"],
    )
    result["dividend_tracking"] = {
        key: value for key, value in transactions.items() if key != "events"
    }
    result["cash_reconciliation"] = "not_certified_by_position_observation"
    return result


def dividend_observations(
    trader: Any,
    reference: str,
    plan: Mapping[str, Any],
    ledger: LiveOrderLedger,
    *,
    now: datetime,
) -> dict[str, Any]:
    entry = ledger.get(intent_id(plan, "BUY"))
    if not entry or number(entry.get("filled_quantity", 0)) <= 0:
        return {"state": "not_started", "events": [], "source_complete": False}
    start = max(timestamp(entry["created_at_utc"]), now - timedelta(days=59))
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
                    "symbol": plan["symbol"],
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
            "events": [],
            "source_complete": False,
        }
    events = []
    unresolved = 0
    for row in rows:
        if _kind(row, action="UNKNOWN") != "dividend":
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
    complete = (
        not unresolved
        and len(rows) < 1000
        and start == timestamp(entry["created_at_utc"])
    )
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
    plan = load(root, POLICY_PATH)
    validate_policy(plan)
    if args.command == "status":
        return {
            "purpose": PURPOSE,
            "state": "operator_controlled_not_armed",
            "plan": plan,
            "connected": False,
            **AUTHORITY,
        }
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
                return observe(
                    root, plan=plan, trader=trader, reference=reference, ledger=ledger
                )
            proposal = propose_entry(plan, quote, now=datetime.now(timezone.utc))
            if (
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
            )
            review["price_proposal"] = proposal
            if args.command == "preview" or not review["technical_ready"]:
                return review
            phrase = approval_phrase(plan, request)
            print(
                json.dumps(
                    {
                        "request": request,
                        "account": plan["account_policy_key"],
                        "budget_usd": 300,
                        "cost_reserve_usd": 1,
                        "buy_and_hold": args.action == "BUY",
                        "cancel_if_unfilled_seconds": 60,
                    },
                    indent=2,
                )
            )
            print(
                "Confirm each reviewed account/risk requirement, then this exact order. SELL is a separate test, never automatic."
            )
            for field in required_operator_confirmations("roth_ira") + (
                "broker_open_orders_reviewed",
                "no_concurrent_manual_orders_confirmed",
            ):
                if input(f"{field} [yes/no]: ").strip().lower() != "yes":
                    raise ValueError("operator_confirmation_incomplete")
            settled_cash = number(
                input("Current settled cash shown by Schwab (USD): ").strip()
            )
            if settled_cash < 300:
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
            )
            if not issued.get("ok"):
                raise ValueError("test_attestation_not_issued")
            inventory = broker_inventory(trader, reference)
            quote = _quote_summary(
                trader._fetch_live_quote(symbol=plan["symbol"]),
                symbol=plan["symbol"],
                now=datetime.now(timezone.utc),
            )
            review = assessment(
                root,
                plan=plan,
                request=request,
                quote=quote,
                reference=reference,
                ledger=ledger,
                inventory=inventory,
                now=datetime.now(timezone.utc),
            )
            if not review["operator_submit_ready"]:
                return review
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
        choices=("status", "preview", "submit", "observe"),
        nargs="?",
        default="status",
    )
    parser.add_argument("--action", choices=("BUY", "SELL"), default="BUY")
    parser.add_argument("--quantity")
    parser.add_argument("--limit-price")
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
    if args.command != "status":
        try:
            destination = local_path(PROJECT_ROOT, REPORT_PATH)
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
