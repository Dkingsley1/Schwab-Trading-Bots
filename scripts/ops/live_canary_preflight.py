#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.live_canary_preflight import (
    RETIREMENT_TAX_WRAPPERS,
    evaluate_live_canary_preflight,
    required_operator_confirmations,
)
from scripts.ops.live_canary_graduation import build_payload as build_graduation_payload

CONFIRMATION_PHRASE = "I CONFIRM SUPERVISED LIVE CANARY"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)


def _issue_attestation(
    project_root: Path,
    *,
    settled_cash_usd: float,
    duration_minutes: int,
    confirmation: str,
    confirm_all: bool,
    confirm_retirement_account_risk: bool = False,
) -> dict[str, Any]:
    if confirmation.strip() != CONFIRMATION_PHRASE or not confirm_all:
        return {
            "ok": False,
            "error": "explicit_operator_confirmation_required",
            "required_phrase": CONFIRMATION_PHRASE,
            "live_execution_authority": False,
        }
    readiness = _load_json(
        project_root / "config" / "production_readiness_control_v1.json"
    )
    policy = readiness.get("live_execution_risk_firewall")
    policy = policy if isinstance(policy, dict) else {}
    plan_path = project_root / str(
        policy.get("canary_plan_path") or "config/live_canary_micro_policy_v1.json"
    )
    plan = _load_json(plan_path)
    registry_path = project_root / str(
        policy.get("account_policy_registry_path")
        or "config/account_policy_registry.json"
    )
    registry = _load_json(registry_path)
    candidate_path = project_root / str(
        policy.get("production_candidate_state_path")
        or "governance/runtime/production_candidate_state.json"
    )
    candidate = _load_json(candidate_path)
    account_study_path = project_root / str(
        policy.get("account_position_study_path")
        or "governance/health/account_position_study_latest.json"
    )
    account_study = _load_json(account_study_path)
    attestation_path = project_root / str(
        policy.get("live_canary_operator_attestation_path")
        or "governance/runtime/live_canary_operator_attestation.json"
    )
    tax_path = project_root / str(
        policy.get("trading_tax_ledger_path")
        or "governance/tax/trading_tax_ledger_{year}_latest.json"
    ).replace("{year}", str(datetime.now(timezone.utc).year))

    account_policy_key = str(plan.get("account_policy_key") or "").strip()
    route_id = str(plan.get("execution_route_id") or "").strip()
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    slots = (
        registry.get("account_slots")
        if isinstance(registry.get("account_slots"), list)
        else []
    )
    account_slot = next(
        (
            dict(row)
            for row in slots
            if isinstance(row, Mapping)
            and str(row.get("account_policy_key") or "").strip() == account_policy_key
        ),
        {},
    )
    account_reference_env = str(
        policy.get("account_reference_env") or "SCHWAB_ACCOUNT_HASH"
    )
    account_reference = str(os.getenv(account_reference_env, "") or "").strip()
    expected_hashes = [
        str(os.getenv(str(name), "") or "").strip()
        for name in account_slot.get("env_names", [])
        if str(name or "").strip().endswith("_HASH")
        and str(os.getenv(str(name), "") or "").strip()
    ]
    if not account_reference or account_reference not in expected_hashes:
        return {
            "ok": False,
            "error": "live_account_not_designated_canary_account",
            "account_policy_key": account_policy_key,
            "live_execution_authority": False,
        }

    account_rows = (
        account_study.get("accounts")
        if isinstance(account_study.get("accounts"), list)
        else []
    )
    account_row = next(
        (
            dict(row)
            for row in account_rows
            if isinstance(row, Mapping)
            and str(row.get("account_policy_key") or "").strip() == account_policy_key
        ),
        {},
    )
    capability = (
        account_row.get("account_capability_truth")
        if isinstance(account_row.get("account_capability_truth"), Mapping)
        else {}
    )
    operator_classification = (
        capability.get("operator_classification")
        if isinstance(capability.get("operator_classification"), Mapping)
        else {}
    )
    tax_wrapper = (
        str(
            operator_classification.get("tax_wrapper")
            or account_row.get("tax_wrapper")
            or "unknown"
        )
        .strip()
        .lower()
    )
    retirement_account = tax_wrapper in RETIREMENT_TAX_WRAPPERS
    if retirement_account and not confirm_retirement_account_risk:
        return {
            "ok": False,
            "error": "explicit_retirement_account_risk_confirmation_required",
            "account_policy_key": account_policy_key,
            "tax_wrapper": tax_wrapper,
            "required_flag": "--confirm-retirement-account-risk",
            "live_execution_authority": False,
        }
    balance = (
        capability.get("balance_truth")
        if isinstance(capability.get("balance_truth"), Mapping)
        else {}
    )
    broker_visible_cash = max(
        _safe_float(balance.get("cash_available_for_trading"), 0.0),
        _safe_float(balance.get("cash_balance"), account_row.get("cash_balance", 0.0)),
    )
    required_cash = max(
        _safe_float(plan.get("account_capital_usd"), 0.0),
        _safe_float(account_slot.get("canary_cap_usd"), 0.0),
    )
    if (
        not account_row
        or settled_cash_usd < required_cash
        or broker_visible_cash < required_cash
        or _safe_float(balance.get("pending_deposits"), 0.0) > 0.0
    ):
        return {
            "ok": False,
            "error": "settled_cash_not_confirmed_by_operator_and_broker",
            "account_policy_key": account_policy_key,
            "required_cash_usd": required_cash,
            "broker_visible_cash_usd": broker_visible_cash,
            "live_execution_authority": False,
        }

    activation = (
        plan.get("activation_contract")
        if isinstance(plan.get("activation_contract"), Mapping)
        else {}
    )
    max_hours = max(
        _safe_float(activation.get("max_operator_attestation_hours"), 4.0), 0.0
    )
    duration = min(max(int(duration_minutes), 1), max(int(max_hours * 60), 1))
    issued = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "account_policy_key": account_policy_key,
        "execution_route_id": route_id,
        "account_reference_sha256": hashlib.sha256(
            account_reference.encode("utf-8")
        ).hexdigest(),
        "account_study_sha256": _file_sha256(account_study_path),
        "trading_tax_ledger_sha256": _file_sha256(tax_path),
        "issued_at_utc": issued.isoformat(),
        "expires_at_utc": (issued + timedelta(minutes=duration)).isoformat(),
        "settled_cash_usd": float(settled_cash_usd),
        "broker_visible_cash_usd": float(broker_visible_cash),
        "broker_ui_checked_at_utc": issued.isoformat(),
        "tax_wrapper": tax_wrapper,
        "retirement_account": retirement_account,
        "live_execution_authority": False,
        "operator_release_still_required": True,
        "policy": "short-lived human attestation can only satisfy preflight evidence; it cannot arm live execution",
    }
    for field in required_operator_confirmations(tax_wrapper):
        payload[field] = True
    _atomic_private_json(attestation_path, payload)
    return {
        "ok": True,
        "attestation_path": str(attestation_path),
        "expires_at_utc": payload["expires_at_utc"],
        "account_policy_key": account_policy_key,
        "live_execution_authority": False,
    }


def _issue_allowlist(
    project_root: Path,
    *,
    stage: int,
    duration_minutes: int,
    confirmation: str,
    confirm_all: bool,
) -> dict[str, Any]:
    if confirmation.strip() != CONFIRMATION_PHRASE or not confirm_all:
        return {
            "ok": False,
            "error": "explicit_operator_confirmation_required",
            "required_phrase": CONFIRMATION_PHRASE,
            "live_execution_authority": False,
        }
    readiness = _load_json(
        project_root / "config" / "production_readiness_control_v1.json"
    )
    policy = readiness.get("live_execution_risk_firewall")
    policy = policy if isinstance(policy, dict) else {}
    plan_path = project_root / str(
        policy.get("canary_plan_path") or "config/live_canary_micro_policy_v1.json"
    )
    plan = _load_json(plan_path)
    stage_row = next(
        (
            dict(row)
            for row in plan.get("stages", [])
            if isinstance(row, Mapping) and int(row.get("stage", 0) or 0) == stage
        ),
        {},
    )
    symbols = [
        str(item or "").strip().upper()
        for item in stage_row.get("symbols", [])
        if str(item or "").strip()
    ]
    if not stage_row or not symbols:
        return {
            "ok": False,
            "error": "canary_stage_invalid",
            "live_execution_authority": False,
        }
    graduation: dict[str, Any] = {}
    require_prior_graduation = bool(
        (
            plan.get("activation_contract")
            if isinstance(plan.get("activation_contract"), Mapping)
            else {}
        ).get("require_prior_stage_graduation_for_stage_above_one", True)
    )
    if stage > 1 and require_prior_graduation:
        graduation = build_graduation_payload(project_root)
        completed_stage = int(
            (
                graduation.get("stage_progression")
                if isinstance(graduation.get("stage_progression"), Mapping)
                else {}
            ).get("highest_completed_stage", 0)
            or 0
        )
        if graduation.get("control_ok") is not True or completed_stage < stage - 1:
            return {
                "ok": False,
                "error": "prior_canary_stage_graduation_not_earned",
                "requested_stage": int(stage),
                "required_completed_stage": int(stage - 1),
                "completed_stage": completed_stage,
                "graduation_phase": str(graduation.get("phase") or "missing"),
                "graduation_blockers": list(graduation.get("blockers") or []),
                "graduation_pending_evidence": list(
                    graduation.get("pending_evidence") or []
                ),
                "live_execution_authority": False,
            }
    preflight = evaluate_live_canary_preflight(
        project_root,
        symbol=symbols[0],
        action="BUY",
    )
    if not preflight.get("ready", False):
        return {
            "ok": False,
            "error": "live_canary_preflight_not_ready",
            "blockers": preflight.get("blockers", []),
            "live_execution_authority": False,
        }
    activation = (
        plan.get("activation_contract")
        if isinstance(plan.get("activation_contract"), Mapping)
        else {}
    )
    max_minutes = max(
        int(_safe_float(activation.get("max_allowlist_duration_hours"), 4.0) * 60),
        1,
    )
    duration = min(max(int(duration_minutes), 1), max_minutes)
    issued = datetime.now(timezone.utc)
    candidate = _load_json(
        project_root
        / str(
            policy.get("production_candidate_state_path")
            or "governance/runtime/production_candidate_state.json"
        )
    )
    attestation_path = project_root / str(
        policy.get("live_canary_operator_attestation_path")
        or "governance/runtime/live_canary_operator_attestation.json"
    )
    allowlist_path = project_root / str(
        policy.get("canary_allowlist_path")
        or "governance/runtime/live_canary_allowlist.json"
    )
    payload = {
        "schema_version": 1,
        "enabled": True,
        "candidate_id": str(candidate.get("candidate_id") or "").strip(),
        "account_policy_key": str(plan.get("account_policy_key") or "").strip(),
        "execution_route_id": str(plan.get("execution_route_id") or "").strip(),
        "account_reference_sha256": str(
            preflight.get("account_reference_sha256") or ""
        ),
        "operator_attestation_sha256": _file_sha256(attestation_path),
        "stage": int(stage),
        "symbols": symbols,
        "graduation_receipt_sha256": (
            str(graduation.get("graduation_receipt_sha256") or "") if stage > 1 else ""
        ),
        "issued_at_utc": issued.isoformat(),
        "expires_at_utc": (issued + timedelta(minutes=duration)).isoformat(),
        "live_execution_authority": False,
        "operator_release_still_required": True,
    }
    _atomic_private_json(allowlist_path, payload)
    return {
        "ok": True,
        "allowlist_path": str(allowlist_path),
        "stage": int(stage),
        "symbols": symbols,
        "expires_at_utc": payload["expires_at_utc"],
        "live_execution_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate or issue the short-lived supervised live-canary preflight attestation."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--symbol", default="SCHD")
    parser.add_argument("--action", default="BUY")
    parser.add_argument("--issue-attestation", action="store_true")
    parser.add_argument("--issue-allowlist", action="store_true")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--settled-cash-usd", type=float, default=0.0)
    parser.add_argument("--duration-minutes", type=int, default=60)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--confirm-all", action="store_true")
    parser.add_argument("--confirm-retirement-account-risk", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    issue_result: dict[str, Any] = {}
    if args.issue_attestation:
        issue_result = _issue_attestation(
            project_root,
            settled_cash_usd=max(float(args.settled_cash_usd), 0.0),
            duration_minutes=args.duration_minutes,
            confirmation=args.confirmation,
            confirm_all=bool(args.confirm_all),
            confirm_retirement_account_risk=bool(args.confirm_retirement_account_risk),
        )
        if not issue_result.get("ok", False):
            print(json.dumps(issue_result, ensure_ascii=True))
            return 2
    allowlist_issue_result: dict[str, Any] = {}
    if args.issue_allowlist:
        allowlist_issue_result = _issue_allowlist(
            project_root,
            stage=max(int(args.stage), 1),
            duration_minutes=args.duration_minutes,
            confirmation=args.confirmation,
            confirm_all=bool(args.confirm_all),
        )
        if not allowlist_issue_result.get("ok", False):
            print(json.dumps(allowlist_issue_result, ensure_ascii=True))
            return 2

    payload = evaluate_live_canary_preflight(
        project_root,
        symbol=args.symbol,
        action=args.action,
    )
    if issue_result:
        payload["attestation_issue_result"] = issue_result
    if allowlist_issue_result:
        payload["allowlist_issue_result"] = allowlist_issue_result
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_canary_preflight "
            f"ready={int(bool(payload.get('ready', False)))} "
            f"account_policy_key={payload.get('account_policy_key', '')} "
            f"blockers={','.join(payload.get('blockers', []))}"
        )
    return 0 if payload.get("ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
