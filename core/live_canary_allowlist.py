from __future__ import annotations

import json
import hashlib
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return (payload if isinstance(payload, dict) else {}), isinstance(payload, dict)


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _symbols(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        symbol = str(item or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def _allowlist_entry_intents(
    ledger_path: Path,
    *,
    issued_at: datetime | None,
    candidate_id: str,
    account_policy_key: str,
    execution_route_id: str,
) -> tuple[list[dict[str, Any]], str]:
    if issued_at is None:
        return [], "allowlist_issue_time_invalid_for_entry_budget"
    if not ledger_path.exists():
        return [], "live_order_ledger_missing_for_entry_budget"
    try:
        with sqlite3.connect(str(ledger_path), timeout=2.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT intent_id, created_at_utc, state, payload_json
                FROM order_intents
                WHERE created_at_utc >= ?
                ORDER BY created_at_utc, intent_id
                """,
                (issued_at.isoformat(),),
            ).fetchall()
    except (OSError, sqlite3.DatabaseError) as exc:
        return [], f"live_order_ledger_unreadable_for_entry_budget:{type(exc).__name__}"

    matched: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = str(payload.get("action") or "").strip().upper()
        if action not in {"BUY", "BUY_TO_OPEN", "SELL_SHORT", "SELL_TO_OPEN"}:
            continue
        envelope = (
            payload.get("live_execution_envelope")
            if isinstance(payload.get("live_execution_envelope"), dict)
            else {}
        )
        snapshot = (
            envelope.get("account_snapshot_evidence")
            if isinstance(envelope.get("account_snapshot_evidence"), dict)
            else {}
        )
        preflight = (
            snapshot.get("live_canary_preflight_receipt")
            if isinstance(snapshot.get("live_canary_preflight_receipt"), dict)
            else {}
        )
        if str(envelope.get("candidate_id") or "").strip() != candidate_id:
            continue
        if str(preflight.get("account_policy_key") or "").strip() != account_policy_key:
            continue
        if str(preflight.get("execution_route_id") or "").strip() != execution_route_id:
            continue
        matched.append(
            {
                "intent_id_sha256": hashlib.sha256(
                    str(row["intent_id"] or "").encode("utf-8")
                ).hexdigest(),
                "created_at_utc": str(row["created_at_utc"] or ""),
                "state": str(row["state"] or ""),
                "symbol": str(payload.get("symbol") or "").strip().upper(),
                "action": action,
            }
        )
    return matched, ""


def evaluate_live_canary_allowlist(
    project_root: str | Path,
    *,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    env_map = dict(env) if isinstance(env, Mapping) else dict(os.environ)
    readiness, _ = _load_json(root / "config" / "production_readiness_control_v1.json")
    firewall = readiness.get("live_execution_risk_firewall")
    firewall = firewall if isinstance(firewall, dict) else {}
    allowlist_path = _project_path(root, firewall.get("canary_allowlist_path"))
    plan_path = _project_path(
        root,
        firewall.get("canary_plan_path") or "config/live_canary_micro_policy_v1.json",
    )
    candidate_path = _project_path(
        root,
        firewall.get("production_candidate_state_path")
        or "governance/runtime/production_candidate_state.json",
    )
    lifecycle_path = _project_path(
        root,
        firewall.get("symbol_lifecycle_path") or "config/symbol_lifecycle_v1.json",
    )

    allowlist, allowlist_valid_json = _load_json(allowlist_path)
    plan, plan_valid_json = _load_json(plan_path)
    candidate, candidate_valid_json = _load_json(candidate_path)
    lifecycle, lifecycle_valid_json = _load_json(lifecycle_path)

    blockers: list[str] = []
    if not allowlist_path.exists():
        blockers.append("canary_allowlist_missing")
    elif not allowlist_valid_json:
        blockers.append("canary_allowlist_invalid_json")
    if not plan_valid_json:
        blockers.append("canary_plan_invalid")
    if not candidate_valid_json:
        blockers.append("production_candidate_state_invalid")
    if not lifecycle_valid_json:
        blockers.append("symbol_lifecycle_invalid")

    enabled = bool(allowlist.get("enabled", False))
    if allowlist_valid_json and not enabled:
        blockers.append("canary_allowlist_disabled")
    if allowlist_valid_json and int(allowlist.get("schema_version", 0) or 0) != 1:
        blockers.append("canary_allowlist_schema_invalid")
    required_fields = [
        str(item or "").strip()
        for item in plan.get("required_allowlist_fields", [])
        if str(item or "").strip()
    ]
    missing_required_fields = [
        field
        for field in required_fields
        if field not in allowlist
        or allowlist.get(field) is None
        or allowlist.get(field) == ""
    ]
    if allowlist_valid_json and missing_required_fields:
        blockers.append("canary_allowlist_required_fields_missing")

    current_candidate_id = str(candidate.get("candidate_id") or "").strip()
    candidate_accepted_at = _parse_timestamp(candidate.get("accepted_at_utc"))
    if candidate_valid_json and not candidate_accepted_at:
        blockers.append("production_candidate_acceptance_missing")
    allowlist_candidate_id = str(allowlist.get("candidate_id") or "").strip()
    candidate_matches = bool(
        current_candidate_id and allowlist_candidate_id == current_candidate_id
    )
    if allowlist_valid_json and not candidate_matches:
        blockers.append("canary_allowlist_candidate_mismatch")

    account_policy_key = str(plan.get("account_policy_key") or "").strip()
    execution_route_id = str(plan.get("execution_route_id") or "").strip()
    if (
        allowlist_valid_json
        and account_policy_key
        and str(allowlist.get("account_policy_key") or "").strip() != account_policy_key
    ):
        blockers.append("canary_allowlist_account_policy_mismatch")
    if (
        allowlist_valid_json
        and execution_route_id
        and str(allowlist.get("execution_route_id") or "").strip() != execution_route_id
    ):
        blockers.append("canary_allowlist_execution_route_mismatch")
    account_reference_env = str(
        firewall.get("account_reference_env") or "SCHWAB_ACCOUNT_HASH"
    )
    account_reference = str(env_map.get(account_reference_env) or "").strip()
    expected_account_reference_sha256 = (
        hashlib.sha256(account_reference.encode("utf-8")).hexdigest()
        if account_reference
        else ""
    )
    if (
        allowlist_valid_json
        and "account_reference_sha256" in required_fields
        and str(allowlist.get("account_reference_sha256") or "").strip().lower()
        != expected_account_reference_sha256
    ):
        blockers.append("canary_allowlist_account_reference_mismatch")
    attestation_path = _project_path(
        root,
        firewall.get("live_canary_operator_attestation_path")
        or "governance/runtime/live_canary_operator_attestation.json",
    )
    try:
        expected_attestation_sha256 = hashlib.sha256(
            attestation_path.read_bytes()
        ).hexdigest()
    except OSError:
        expected_attestation_sha256 = ""
    if (
        allowlist_valid_json
        and "operator_attestation_sha256" in required_fields
        and str(allowlist.get("operator_attestation_sha256") or "").strip().lower()
        != expected_attestation_sha256
    ):
        blockers.append("canary_allowlist_operator_attestation_mismatch")

    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issued_at = _parse_timestamp(allowlist.get("issued_at_utc"))
    expires_at = _parse_timestamp(allowlist.get("expires_at_utc"))
    activation = (
        plan.get("activation_contract")
        if isinstance(plan.get("activation_contract"), dict)
        else {}
    )
    try:
        max_allowlist_hours = max(
            float(activation.get("max_allowlist_duration_hours", 4.0) or 4.0), 0.0
        )
    except Exception:
        max_allowlist_hours = 4.0
    if allowlist_valid_json and (
        not issued_at or issued_at > now_utc + timedelta(minutes=5)
    ):
        blockers.append("canary_allowlist_issued_at_invalid")
    if (
        allowlist_valid_json
        and issued_at
        and candidate_accepted_at
        and issued_at < candidate_accepted_at
    ):
        blockers.append("canary_allowlist_predates_candidate_acceptance")
    unexpired = bool(expires_at and expires_at > now_utc)
    if allowlist_valid_json and not unexpired:
        blockers.append("canary_allowlist_expired_or_invalid")
    duration_hours = (
        (expires_at - issued_at).total_seconds() / 3600.0
        if issued_at and expires_at
        else 0.0
    )
    if allowlist_valid_json and (
        duration_hours <= 0.0
        or max_allowlist_hours <= 0.0
        or duration_hours > max_allowlist_hours
    ):
        blockers.append("canary_allowlist_duration_exceeds_policy")

    try:
        max_entry_intents = max(
            int(activation.get("max_new_entry_intents_per_allowlist", 1) or 1), 0
        )
    except (TypeError, ValueError):
        max_entry_intents = 0
    enforce_entry_budget = bool(
        activation.get("require_entry_budget_enforcement", False)
    )
    ledger_path = _project_path(
        root,
        activation.get("live_order_ledger_path")
        or "governance/runtime/live_order_ledger.sqlite3",
    )
    allowlist_entry_intents: list[dict[str, Any]] = []
    entry_budget_error = ""
    if allowlist_valid_json and enforce_entry_budget:
        allowlist_entry_intents, entry_budget_error = _allowlist_entry_intents(
            ledger_path,
            issued_at=issued_at,
            candidate_id=current_candidate_id,
            account_policy_key=account_policy_key,
            execution_route_id=execution_route_id,
        )
        if max_entry_intents <= 0:
            blockers.append("canary_allowlist_entry_budget_invalid")
        if entry_budget_error:
            blockers.append(entry_budget_error)
        if max_entry_intents > 0 and len(allowlist_entry_intents) >= max_entry_intents:
            blockers.append("canary_allowlist_entry_budget_exhausted")

    try:
        stage = int(allowlist.get("stage", 0) or 0)
    except Exception:
        stage = 0
    stages = (
        [row for row in plan.get("stages", []) if isinstance(row, dict)]
        if plan_valid_json
        else []
    )
    stage_row = next(
        (row for row in stages if int(row.get("stage", 0) or 0) == stage), {}
    )
    stage_symbols = _symbols(stage_row.get("symbols"))
    allowlist_symbols = _symbols(allowlist.get("symbols"))
    if allowlist_valid_json and not stage_row:
        blockers.append("canary_allowlist_stage_invalid")
    if allowlist_valid_json and not allowlist_symbols:
        blockers.append("canary_allowlist_empty")
    out_of_plan_symbols = sorted(set(allowlist_symbols) - set(stage_symbols))
    if out_of_plan_symbols:
        blockers.append("canary_allowlist_symbol_not_in_stage")

    require_prior_graduation = bool(
        activation.get("require_prior_stage_graduation_for_stage_above_one", True)
    )
    graduation_path = _project_path(
        root,
        activation.get("live_canary_graduation_health_path")
        or "governance/health/live_canary_graduation_latest.json",
    )
    graduation, graduation_valid_json = _load_json(graduation_path)
    graduation_completed_stage = int(
        (
            graduation.get("stage_progression")
            if isinstance(graduation.get("stage_progression"), dict)
            else {}
        ).get("highest_completed_stage", 0)
        or 0
    )
    graduation_receipt = (
        str(graduation.get("graduation_receipt_sha256") or "").strip().lower()
    )
    allowlist_graduation_receipt = (
        str(allowlist.get("graduation_receipt_sha256") or "").strip().lower()
    )
    if allowlist_valid_json and stage > 1 and require_prior_graduation:
        if not graduation_valid_json:
            blockers.append("live_canary_graduation_receipt_missing_or_invalid")
        elif graduation.get("control_ok") is not True:
            blockers.append("live_canary_graduation_control_blocked")
        elif graduation_completed_stage < stage - 1:
            blockers.append("prior_canary_stage_graduation_not_earned")
        if not graduation_receipt or allowlist_graduation_receipt != graduation_receipt:
            blockers.append("canary_allowlist_graduation_receipt_mismatch")

    renamed = (
        lifecycle.get("renamed_symbols")
        if isinstance(lifecycle.get("renamed_symbols"), dict)
        else {}
    )
    deprecated_symbols = sorted(
        symbol
        for symbol in allowlist_symbols
        if symbol in {str(key).upper() for key in renamed}
    )
    if deprecated_symbols:
        blockers.append("canary_allowlist_contains_deprecated_symbol")

    hard_limits = (
        plan.get("hard_limits") if isinstance(plan.get("hard_limits"), dict) else {}
    )
    return {
        "ready": not blockers,
        "blockers": blockers,
        "path": str(allowlist_path),
        "exists": allowlist_path.exists(),
        "enabled": enabled,
        "candidate_id": allowlist_candidate_id,
        "current_candidate_id": current_candidate_id,
        "candidate_accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
        "candidate_matches": candidate_matches,
        "account_policy_key": str(allowlist.get("account_policy_key") or ""),
        "expected_account_policy_key": account_policy_key,
        "execution_route_id": str(allowlist.get("execution_route_id") or ""),
        "expected_execution_route_id": execution_route_id,
        "account_reference_bound": bool(
            expected_account_reference_sha256
            and str(allowlist.get("account_reference_sha256") or "").strip().lower()
            == expected_account_reference_sha256
        ),
        "operator_attestation_bound": bool(
            expected_attestation_sha256
            and str(allowlist.get("operator_attestation_sha256") or "").strip().lower()
            == expected_attestation_sha256
        ),
        "required_fields": required_fields,
        "missing_required_fields": missing_required_fields,
        "issued_at_utc": str(allowlist.get("issued_at_utc") or ""),
        "expires_at_utc": str(allowlist.get("expires_at_utc") or ""),
        "duration_hours": round(float(duration_hours), 6),
        "max_allowlist_duration_hours": float(max_allowlist_hours),
        "unexpired": unexpired,
        "entry_budget": {
            "enforced": enforce_entry_budget,
            "maximum_new_entry_intents": max_entry_intents,
            "consumed_new_entry_intents": len(allowlist_entry_intents),
            "remaining_new_entry_intents": max(
                max_entry_intents - len(allowlist_entry_intents), 0
            ),
            "ledger_path": str(ledger_path),
            "ledger_error": entry_budget_error,
            "intents": allowlist_entry_intents,
            "reduce_only_exits_preserved": True,
        },
        "stage": stage,
        "symbols": allowlist_symbols,
        "stage_symbols": stage_symbols,
        "graduation_path": str(graduation_path),
        "graduation_completed_stage": graduation_completed_stage,
        "graduation_receipt_bound": bool(
            stage <= 1
            or (
                graduation_receipt
                and allowlist_graduation_receipt == graduation_receipt
            )
        ),
        "out_of_plan_symbols": out_of_plan_symbols,
        "deprecated_symbols": deprecated_symbols,
        "plan_path": str(plan_path),
        "plan_status": str(plan.get("status") or "missing"),
        "planned_stages": stages,
        "hard_limits": hard_limits,
        "candidate_state_path": str(candidate_path),
        "symbol_lifecycle_path": str(lifecycle_path),
        "live_execution_authority": False,
    }
