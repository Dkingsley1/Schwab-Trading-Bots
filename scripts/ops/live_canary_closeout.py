#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import write_payload

from core.live_canary_graduation import (
    ENTRY_ACTIONS,
    EXIT_ACTIONS,
    build_live_canary_closeout_receipt,
    extract_live_canary_intent_identity,
)
from core.live_order_ledger import LiveOrderLedger

DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "live_canary_graduation_v1.json"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "config" / "live_canary_micro_policy_v1.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "live_canary_closeout_latest.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _project_path(root: Path, raw: Any, fallback: str) -> Path:
    path = Path(str(raw or fallback))
    return path if path.is_absolute() else root / path


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _existing_receipts(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"closeout_receipts_unreadable:{type(exc).__name__}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"closeout_receipt_invalid_json:line={line_number}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"closeout_receipt_not_object:line={line_number}")
            continue
        rows.append(payload)
    return rows, errors


def _account_truth(
    account_study: Mapping[str, Any],
    *,
    account_policy_key: str,
    symbol: str,
) -> dict[str, Any]:
    accounts = (
        account_study.get("accounts")
        if isinstance(account_study.get("accounts"), list)
        else []
    )
    account = next(
        (
            dict(row)
            for row in accounts
            if isinstance(row, Mapping)
            and str(row.get("account_policy_key") or "").strip() == account_policy_key
        ),
        {},
    )
    positions = (
        account_study.get("positions")
        if isinstance(account_study.get("positions"), list)
        else []
    )
    quantity = sum(
        float(_number(row.get("quantity")) or 0.0)
        for row in positions
        if isinstance(row, Mapping)
        and str(row.get("account_policy_key") or "").strip() == account_policy_key
        and str(row.get("symbol") or "").strip().upper() == symbol
        and str(row.get("asset_type") or "").strip().upper() == "EQUITY"
    )
    capability = (
        account.get("account_capability_truth")
        if isinstance(account.get("account_capability_truth"), Mapping)
        else {}
    )
    balance = (
        capability.get("balance_truth")
        if isinstance(capability.get("balance_truth"), Mapping)
        else {}
    )
    calls = (
        capability.get("broker_call_truth")
        if isinstance(capability.get("broker_call_truth"), Mapping)
        else {}
    )
    collateral = (
        capability.get("position_collateral_truth")
        if isinstance(capability.get("position_collateral_truth"), Mapping)
        else {}
    )
    flags = account.get("flags") if isinstance(account.get("flags"), Mapping) else {}
    cash_balance = _number(balance.get("cash_balance"))
    if cash_balance is None:
        cash_balance = _number(account.get("cash_balance"))
    if cash_balance is None:
        cash_balance = _number(balance.get("cash_available_for_trading"))
    cash = float(cash_balance or 0.0)
    debit = (
        capability.get("debit_truth")
        if isinstance(capability.get("debit_truth"), Mapping)
        else {}
    )
    safety_violations: list[str] = []
    if not account:
        safety_violations.append("designated_canary_account_truth_missing")
    if bool(flags.get("closing_only", False)):
        safety_violations.append("canary_account_closing_only")
    if bool(calls.get("in_call", False)):
        safety_violations.append("canary_account_in_broker_call")
    if float(_number(balance.get("pending_deposits")) or 0.0) > 0.0:
        safety_violations.append("canary_account_pending_deposit")
    if float(_number(collateral.get("uncovered_short_option_count")) or 0.0) > 0.0:
        safety_violations.append("canary_account_uncovered_short_options")
    if bool(account.get("borrowing_allowed", False)):
        safety_violations.append("canary_account_borrowing_authority_present")
    if (
        bool(debit.get("interest_bearing_borrowing_confirmed", False))
        or float(_number(debit.get("accrued_interest")) or 0.0) > 0.0
    ):
        safety_violations.append("canary_account_interest_bearing_debit_present")
    return {
        "account_found": bool(account),
        "position_quantity": quantity,
        "cash_usd": cash,
        "cash_source": (
            "broker_cash_balance"
            if _number(balance.get("cash_balance")) is not None
            else (
                "account_cash_balance"
                if _number(account.get("cash_balance")) is not None
                else "broker_cash_available_for_trading_fallback"
            )
        ),
        "safety_violations": safety_violations,
    }


def _append_private_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8", closefd=False) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(
                json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    intent_id: str = "",
    capture: bool = False,
    policy_path: Path | None = None,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
    receipts_path: Path | None = None,
    account_study_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    policy_file = policy_path or root / "config" / DEFAULT_POLICY_PATH.name
    plan_file = plan_path or root / "config" / DEFAULT_PLAN_PATH.name
    policy = _load_json(policy_file)
    plan = _load_json(plan_file)
    evidence = (
        policy.get("evidence") if isinstance(policy.get("evidence"), dict) else {}
    )
    ledger_file = ledger_path or _project_path(
        root,
        evidence.get("live_order_ledger_path"),
        "governance/runtime/live_order_ledger.sqlite3",
    )
    receipts_file = receipts_path or _project_path(
        root,
        evidence.get("closeout_receipts_path"),
        "governance/evidence/live_canary_closeout_receipts.jsonl",
    )
    account_study_file = account_study_path or _project_path(
        root,
        evidence.get("account_position_study_path"),
        "governance/health/account_position_study_latest.json",
    )
    hard_errors: list[str] = []
    pending: list[str] = []
    if not policy:
        hard_errors.append("live_canary_graduation_policy_missing_or_invalid")
    if not plan:
        hard_errors.append("live_canary_plan_missing_or_invalid")
    existing_receipts, receipt_errors = _existing_receipts(receipts_file)
    hard_errors.extend(receipt_errors)
    existing_intent_ids = {
        str(row.get("intent_id") or "").strip() for row in existing_receipts
    }

    if not ledger_file.exists():
        pending.append("live_order_ledger_not_created")
        intents: list[dict[str, Any]] = []
        ledger = None
        integrity: dict[str, Any] = {}
    else:
        ledger = LiveOrderLedger(ledger_file)
        integrity = ledger.verify_integrity()
        if integrity.get("ok") is not True:
            hard_errors.append("live_order_ledger_integrity_invalid")
        intents = ledger.intents()

    requested_intent = str(intent_id or "").strip()
    if requested_intent:
        selected = next(
            (
                row
                for row in intents
                if str(row.get("intent_id") or "").strip() == requested_intent
            ),
            {},
        )
        if not selected:
            hard_errors.append("requested_live_canary_intent_not_found")
    else:
        selected = next(
            (
                row
                for row in intents
                if str(row.get("state") or "").strip() == "filled"
                and str(row.get("intent_id") or "").strip() not in existing_intent_ids
            ),
            {},
        )

    if not selected:
        if not hard_errors:
            pending.append("filled_live_canary_intent_pending")
        return {
            "schema_version": 1,
            "timestamp_utc": current.isoformat(),
            "ok": not hard_errors,
            "control_ok": not hard_errors,
            "overall_status": "blocked" if hard_errors else "ready_idle",
            "phase": "blocked" if hard_errors else "awaiting_filled_canary",
            "capture_requested": bool(capture),
            "capture_eligible": False,
            "captured": False,
            "pending_evidence": _ordered_unique(pending),
            "blockers": _ordered_unique(hard_errors),
            "live_execution_authority": False,
            "stage_progression_authority": False,
            "capital_scaling_authority": False,
        }

    selected_intent_id = str(selected.get("intent_id") or "").strip()
    if selected_intent_id in existing_intent_ids:
        hard_errors.append("closeout_receipt_already_exists_for_intent")
    if str(selected.get("state") or "").strip() != "filled":
        pending.append("selected_live_canary_intent_not_filled")

    identity, identity_errors = extract_live_canary_intent_identity(selected)
    hard_errors.extend(identity_errors)
    if str(identity.get("account_policy_key") or "") != str(
        plan.get("account_policy_key") or ""
    ):
        hard_errors.append("closeout_account_policy_mismatch")
    if str(identity.get("execution_route_id") or "") != str(
        plan.get("execution_route_id") or ""
    ):
        hard_errors.append("closeout_execution_route_mismatch")

    events = ledger.events(intent_id=selected_intent_id) if ledger is not None else []
    final_event = events[-1] if events else {}
    event_details = (
        final_event.get("details")
        if isinstance(final_event.get("details"), Mapping)
        else {}
    )
    broker_status = str(event_details.get("broker_status") or "").strip().upper()
    filled_at = _parse_timestamp(final_event.get("timestamp_utc"))
    order_reconciled = bool(
        final_event
        and final_event.get("to_state") == "filled"
        and broker_status in {"FILLED", "EXECUTED"}
        and integrity.get("ok") is True
    )
    if not order_reconciled:
        pending.append("broker_order_reconciliation_pending")
    if filled_at is None:
        hard_errors.append("filled_order_event_timestamp_invalid")

    account_study = _load_json(account_study_file)
    account_study_at = _parse_timestamp(
        account_study.get("timestamp_utc")
        or account_study.get("generated_at_utc")
        or account_study.get("updated_at_utc")
    )
    account_study_hash = _file_sha256(account_study_file)
    if not account_study or not account_study_hash:
        pending.append("fresh_account_position_study_pending")
    max_study_age = max(
        float(_number(evidence.get("max_account_study_age_seconds")) or 300.0), 1.0
    )
    max_snapshot_delay = max(
        float(
            _number(evidence.get("max_account_snapshot_delay_after_fill_seconds"))
            or 300.0
        ),
        1.0,
    )
    if account_study_at is None:
        pending.append("account_position_study_timestamp_invalid")
    else:
        if account_study_at > current + timedelta(seconds=5):
            pending.append("account_position_study_timestamp_in_future")
        if (current - account_study_at).total_seconds() > max_study_age:
            pending.append("account_position_study_stale_for_closeout")
        if filled_at is not None:
            delay = (account_study_at - filled_at).total_seconds()
            if delay < 0.0:
                pending.append("account_position_study_predates_fill")
            elif delay > max_snapshot_delay:
                pending.append(
                    "account_position_study_too_late_for_isolated_cash_proof"
                )

    symbol = str(identity.get("symbol") or "").strip().upper()
    account_truth = _account_truth(
        account_study,
        account_policy_key=str(identity.get("account_policy_key") or ""),
        symbol=symbol,
    )
    hard_errors.extend(account_truth["safety_violations"])
    pre_position = _number(identity.get("pre_position_quantity"))
    pre_cash = _number(identity.get("pre_settled_cash_usd"))
    post_position = _number(account_truth.get("position_quantity"))
    post_cash = _number(account_truth.get("cash_usd"))
    if pre_position is None:
        pending.append("sealed_pre_fill_position_pending")
    if pre_cash is None:
        pending.append("sealed_pre_fill_cash_pending")

    action = str(identity.get("action") or "").strip().upper()
    quantity = float(_number(identity.get("quantity")) or 0.0)
    fill_price = float(_number(selected.get("average_fill_price")) or 0.0)
    fee_floor = max(
        float(_number(evidence.get("conservative_fee_floor_usd_per_order")) or 0.02),
        0.0,
    )
    position_tolerance = max(
        float(
            _number(evidence.get("position_reconciliation_tolerance_quantity"))
            or 0.000001
        ),
        0.0,
    )
    cash_tolerance = max(
        float(_number(evidence.get("cash_reconciliation_tolerance_usd")) or 0.25),
        0.0,
    )
    signed_quantity = quantity if action in ENTRY_ACTIONS else -quantity
    expected_post_position = (
        pre_position + signed_quantity if pre_position is not None else None
    )
    position_reconciled = bool(
        expected_post_position is not None
        and post_position is not None
        and abs(post_position - expected_post_position) <= position_tolerance
    )
    if not position_reconciled:
        pending.append("account_position_delta_reconciliation_pending")

    cash_direction = -1.0 if action in ENTRY_ACTIONS else 1.0
    expected_post_cash = (
        pre_cash + cash_direction * fill_price * quantity - fee_floor
        if pre_cash is not None and fill_price > 0.0 and quantity > 0.0
        else None
    )
    cash_reconciled = bool(
        expected_post_cash is not None
        and post_cash is not None
        and abs(post_cash - expected_post_cash) <= cash_tolerance
    )
    if not cash_reconciled:
        pending.append("account_cash_delta_reconciliation_pending")

    unresolved = bool(ledger.unresolved()) if ledger is not None else False
    if unresolved:
        hard_errors.append("unresolved_live_broker_operation_present")
    reduce_only_exit = bool(
        action in EXIT_ACTIONS
        and pre_position is not None
        and pre_position >= quantity - position_tolerance
        and position_reconciled
    )
    if action in EXIT_ACTIONS and not reduce_only_exit:
        hard_errors.append("canary_exit_not_verified_reduce_only")

    receipt = build_live_canary_closeout_receipt(
        intent_id=selected_intent_id,
        intent_payload_sha256=str(selected.get("payload_hash") or ""),
        final_order_event_sha256=str(final_event.get("event_hash") or ""),
        broker_order_id=str(selected.get("broker_order_id") or ""),
        candidate_id=str(identity.get("candidate_id") or ""),
        account_policy_key=str(identity.get("account_policy_key") or ""),
        execution_route_id=str(identity.get("execution_route_id") or ""),
        account_reference_sha256=str(identity.get("account_reference_sha256") or ""),
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=fill_price,
        filled_at_utc=filled_at.isoformat() if filled_at is not None else "",
        broker_fees_usd=fee_floor,
        fee_evidence_source="conservative_policy_floor",
        pre_position_quantity=float(pre_position or 0.0),
        post_position_quantity=float(post_position or 0.0),
        pre_settled_cash_usd=float(pre_cash or 0.0),
        post_settled_cash_usd=float(post_cash or 0.0),
        order_reconciled=order_reconciled,
        position_reconciled=position_reconciled,
        cash_reconciled=cash_reconciled,
        position_delta_verified=position_reconciled,
        reduce_only_exit=reduce_only_exit,
        unresolved_broker_operation=unresolved,
        safety_violations=account_truth["safety_violations"],
        regime_bucket=str(identity.get("regime_bucket") or "unknown"),
        regime_receipt_sha256=str(identity.get("regime_receipt_sha256") or ""),
        account_study_sha256=account_study_hash,
        account_study_timestamp_utc=(
            account_study_at.isoformat() if account_study_at is not None else ""
        ),
    )
    hard_errors = _ordered_unique(hard_errors)
    pending = _ordered_unique(pending)
    capture_eligible = not hard_errors and not pending
    captured = False
    if capture and capture_eligible:
        _append_private_jsonl(receipts_file, receipt)
        captured = True

    return {
        "schema_version": 1,
        "timestamp_utc": current.isoformat(),
        "ok": not hard_errors,
        "control_ok": not hard_errors,
        "overall_status": (
            "blocked"
            if hard_errors
            else (
                "captured"
                if captured
                else (
                    "ready_to_capture" if capture_eligible else "reconciliation_pending"
                )
            )
        ),
        "phase": (
            "blocked"
            if hard_errors
            else (
                "closeout_complete"
                if captured
                else (
                    "awaiting_capture"
                    if capture_eligible
                    else "post_fill_reconciliation"
                )
            )
        ),
        "intent_id_sha256": hashlib.sha256(
            selected_intent_id.encode("utf-8")
        ).hexdigest(),
        "symbol": symbol,
        "action": action,
        "capture_requested": bool(capture),
        "capture_eligible": capture_eligible,
        "captured": captured,
        "receipt_sha256": str(receipt.get("receipt_sha256") or ""),
        "reconciliation": {
            "order_reconciled": order_reconciled,
            "position_reconciled": position_reconciled,
            "cash_reconciled": cash_reconciled,
            "pre_position_quantity": pre_position,
            "post_position_quantity": post_position,
            "expected_post_position_quantity": expected_post_position,
            "pre_cash_usd": pre_cash,
            "post_cash_usd": post_cash,
            "post_cash_source": str(account_truth.get("cash_source") or ""),
            "expected_post_cash_usd": expected_post_cash,
            "conservative_fee_floor_usd": fee_floor,
            "reduce_only_exit": reduce_only_exit,
        },
        "sources": {
            "policy_path": str(policy_file),
            "plan_path": str(plan_file),
            "ledger_path": str(ledger_file),
            "account_position_study_path": str(account_study_file),
            "account_position_study_sha256": account_study_hash,
            "closeout_receipts_path": str(receipts_file),
        },
        "pending_evidence": pending,
        "blockers": hard_errors,
        "live_execution_authority": False,
        "stage_progression_authority": False,
        "capital_scaling_authority": False,
        "broker_mutation_attempted": False,
        "contract": {
            "append_only": True,
            "raw_account_reference_persisted": False,
            "raw_broker_order_id_persisted": False,
            "manual_reconciliation_override_allowed": False,
            "automatic_follow_on_order_allowed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or append one immutable post-fill canary closeout receipt from "
            "the durable ledger and a fresh exact-account position study. This "
            "command performs no broker mutation and grants no live authority."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--intent-id", default="")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--account-study", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()

    def resolved(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else root / path

    payload = build_payload(
        root,
        intent_id=str(args.intent_id or ""),
        capture=bool(args.capture),
        policy_path=resolved(args.policy),
        plan_path=resolved(args.plan),
        ledger_path=resolved(args.ledger),
        receipts_path=resolved(args.receipts),
        account_study_path=resolved(args.account_study),
    )
    out_path = (
        resolved(args.out) or root / "governance" / "health" / DEFAULT_OUT_PATH.name
    )
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_canary_closeout "
            f"status={payload.get('overall_status', 'unknown')} "
            f"eligible={int(bool(payload.get('capture_eligible', False)))} "
            f"captured={int(bool(payload.get('captured', False)))}"
        )
    if payload.get("control_ok") is not True:
        return 2
    if args.capture and payload.get("captured") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
