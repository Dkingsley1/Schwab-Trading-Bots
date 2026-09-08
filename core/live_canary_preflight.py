from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

LIVE_ENTRY_ACTIONS = frozenset({"BUY", "BUY_TO_OPEN", "SELL_SHORT", "SELL_TO_OPEN"})
REQUIRED_OPERATOR_CONFIRMATIONS = (
    "settled_cash_confirmed",
    "borrowing_disabled_confirmed",
    "account_restrictions_clear",
    "provider_balance_reviewed",
    "tax_and_settlement_reviewed",
    "manual_cancel_ready",
    "broker_web_or_mobile_ready",
    "operator_supervision_confirmed",
)
RETIREMENT_ACCOUNT_OPERATOR_CONFIRMATIONS = (
    "retirement_account_loss_capacity_reviewed",
    "cross_account_wash_sale_reviewed",
    "retirement_contribution_capacity_not_assumed",
)
RETIREMENT_TAX_WRAPPERS = frozenset({"ira", "roth", "roth_ira", "traditional_ira"})


def required_operator_confirmations(tax_wrapper: Any) -> tuple[str, ...]:
    wrapper = str(tax_wrapper or "").strip().lower()
    if wrapper in RETIREMENT_TAX_WRAPPERS:
        return (
            REQUIRED_OPERATOR_CONFIRMATIONS + RETIREMENT_ACCOUNT_OPERATOR_CONFIRMATIONS
        )
    return REQUIRED_OPERATOR_CONFIRMATIONS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return (payload if isinstance(payload, dict) else {}), isinstance(payload, dict)


def _project_path(project_root: Path, raw: Any, fallback: str) -> Path:
    path = Path(str(raw or fallback))
    return path if path.is_absolute() else project_root / path


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


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


def _payload_age_seconds(payload: Mapping[str, Any], now: datetime) -> float | None:
    timestamp = _parse_timestamp(
        payload.get("timestamp_utc")
        or payload.get("generated_at_utc")
        or payload.get("updated_at_utc")
    )
    if timestamp is None:
        return None
    return max((now - timestamp).total_seconds(), 0.0)


def _fresh_payload(
    payload: Mapping[str, Any], *, now: datetime, max_age_seconds: float
) -> tuple[bool, float | None]:
    age = _payload_age_seconds(payload, now)
    return bool(age is not None and age <= max(max_age_seconds, 0.0)), age


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _equity_session_state(
    *, now: datetime, calendar_id: str, buffer_minutes: float
) -> dict[str, Any]:
    try:
        import exchange_calendars
        import pandas as pd

        calendar = exchange_calendars.get_calendar(calendar_id)
        minute = pd.Timestamp(now).floor("min")
        session_open = bool(calendar.is_open_on_minute(minute, ignore_breaks=False))
        if not session_open:
            return {
                "ready": False,
                "calendar_id": calendar_id,
                "state": "closed",
                "blocker": "primary_equity_session_closed",
            }
        session_label = calendar.minute_to_session(minute)
        opened = calendar.session_open(session_label).to_pydatetime()
        closed = calendar.session_close(session_label).to_pydatetime()
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        buffer = timedelta(minutes=max(buffer_minutes, 0.0))
        inside_buffered_window = opened + buffer <= now <= closed - buffer
        return {
            "ready": inside_buffered_window,
            "calendar_id": calendar_id,
            "state": "open" if inside_buffered_window else "auction_buffer",
            "session_label": str(session_label.date()),
            "open_utc": opened.astimezone(timezone.utc).isoformat(),
            "close_utc": closed.astimezone(timezone.utc).isoformat(),
            "buffer_minutes": float(max(buffer_minutes, 0.0)),
            "blocker": (
                "" if inside_buffered_window else "equity_session_auction_buffer_active"
            ),
        }
    except Exception as exc:
        return {
            "ready": False,
            "calendar_id": calendar_id,
            "state": "unknown",
            "blocker": f"equity_session_calendar_unavailable:{type(exc).__name__}",
        }


def _tax_symbol_review(
    ledger: Mapping[str, Any],
    *,
    symbol: str,
    action: str,
    now: datetime,
    lookback_days: int,
) -> dict[str, Any]:
    symbol_key = str(symbol or "").strip().upper()
    action_key = str(action or "").strip().upper()
    if action_key not in {"BUY", "BUY_TO_OPEN"} or not symbol_key:
        return {
            "ready": True,
            "symbol": symbol_key,
            "recent_disposition_count": 0,
            "review_required_count": 0,
        }
    cutoff = now - timedelta(days=max(int(lookback_days), 1))
    recent: list[dict[str, Any]] = []
    review_required: list[dict[str, Any]] = []
    rows = ledger.get("events") if isinstance(ledger.get("events"), list) else []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("symbol") or "").strip().upper() != symbol_key:
            continue
        event_kind = str(raw.get("tax_event_kind") or "").strip().lower()
        event_action = str(raw.get("action") or "").strip().upper()
        if event_kind not in {
            "disposition",
            "sale",
            "realization",
        } and event_action not in {
            "SELL",
            "SELL_TO_CLOSE",
        }:
            continue
        when = _parse_timestamp(raw.get("transaction_date") or raw.get("timestamp_utc"))
        if when is None or when < cutoff or when > now + timedelta(minutes=5):
            continue
        compact = {
            "event_id": str(raw.get("event_id") or ""),
            "transaction_date": when.isoformat(),
            "tax_treatment": str(raw.get("tax_treatment") or ""),
            "wash_sale_status": str(raw.get("wash_sale_status") or "unknown"),
        }
        recent.append(compact)
        wash_status = compact["wash_sale_status"].strip().lower()
        realized = raw.get("realized_gain_loss_usd", raw.get("realized_pnl_usd"))
        realized_value = _safe_float(realized, 0.0)
        if (
            wash_status not in {"clear", "not_applicable", "no_loss"}
            or realized_value < 0.0
        ):
            review_required.append(compact)
    return {
        "ready": not review_required,
        "symbol": symbol_key,
        "lookback_days": max(int(lookback_days), 1),
        "recent_disposition_count": len(recent),
        "review_required_count": len(review_required),
        "review_required_events": review_required[:20],
    }


def evaluate_live_canary_preflight(
    project_root: str | Path,
    *,
    symbol: str = "",
    action: str = "BUY",
    account_reference: str = "",
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    current = (now or _utc_now()).astimezone(timezone.utc)
    env_map = dict(env) if isinstance(env, Mapping) else dict(os.environ)
    readiness_path = root / "config" / "production_readiness_control_v1.json"
    readiness, readiness_valid = _load_json(readiness_path)
    policy = readiness.get("live_execution_risk_firewall")
    policy = policy if isinstance(policy, dict) else {}
    plan_path = _project_path(
        root,
        policy.get("canary_plan_path"),
        "config/live_canary_micro_policy_v1.json",
    )
    plan, plan_valid = _load_json(plan_path)
    candidate_path = _project_path(
        root,
        policy.get("production_candidate_state_path"),
        "governance/runtime/production_candidate_state.json",
    )
    candidate, candidate_valid = _load_json(candidate_path)
    registry_path = _project_path(
        root,
        policy.get("account_policy_registry_path"),
        "config/account_policy_registry.json",
    )
    registry, registry_valid = _load_json(registry_path)
    account_study_path = _project_path(
        root,
        policy.get("account_position_study_path"),
        "governance/health/account_position_study_latest.json",
    )
    account_study, account_study_valid = _load_json(account_study_path)
    attestation_path = _project_path(
        root,
        policy.get("live_canary_operator_attestation_path"),
        "governance/runtime/live_canary_operator_attestation.json",
    )
    attestation, attestation_valid = _load_json(attestation_path)
    risk_path = _project_path(
        root,
        policy.get("risk_service_boundary_path"),
        "governance/risk/risk_service_boundary_latest.json",
    )
    risk_boundary, risk_valid = _load_json(risk_path)
    release_path = _project_path(
        root,
        policy.get("release_freeze_guard_path"),
        "governance/health/release_freeze_guard_latest.json",
    )
    release_guard, release_valid = _load_json(release_path)
    ledger_path = _project_path(
        root,
        policy.get("live_order_ledger_control_path"),
        "governance/health/live_order_ledger_control_latest.json",
    )
    order_ledger, ledger_valid = _load_json(ledger_path)
    tax_path_value = str(
        policy.get("trading_tax_ledger_path")
        or "governance/tax/trading_tax_ledger_{year}_latest.json"
    ).replace("{year}", str(current.year))
    tax_path = _project_path(root, tax_path_value, tax_path_value)
    tax_ledger, tax_valid = _load_json(tax_path)
    regime_path = _project_path(
        root,
        policy.get("regime_control_plane_path"),
        "governance/health/regime_control_plane_latest.json",
    )
    regime_control, regime_valid = _load_json(regime_path)

    blockers: list[str] = []
    if not readiness_valid:
        blockers.append("production_readiness_policy_invalid")
    if not plan_valid:
        blockers.append("live_canary_plan_invalid")
    if not candidate_valid:
        blockers.append("production_candidate_state_invalid")
    if not registry_valid:
        blockers.append("account_policy_registry_invalid")

    candidate_id = str(candidate.get("candidate_id") or "").strip()
    account_policy_key = str(plan.get("account_policy_key") or "").strip()
    route_id = str(plan.get("execution_route_id") or "").strip()
    if not candidate_id:
        blockers.append("production_candidate_id_missing")
    if not account_policy_key:
        blockers.append("canary_account_policy_key_missing")
    if not route_id:
        blockers.append("canary_execution_route_missing")

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
    if not account_slot:
        blockers.append("designated_canary_account_policy_missing")
    if account_slot and not bool(account_slot.get("canary_candidate", False)):
        blockers.append("designated_account_not_canary_eligible")
    if account_slot and bool(account_slot.get("borrowing_allowed", True)):
        blockers.append("designated_account_allows_borrowing")
    allowed_routes = {
        str(item or "").strip()
        for item in account_slot.get("allowed_live_routes", [])
        if str(item or "").strip()
    }
    if route_id and route_id not in allowed_routes:
        blockers.append("canary_route_not_allowed_for_account")

    account_reference_env = str(
        policy.get("account_reference_env") or "SCHWAB_ACCOUNT_HASH"
    )
    live_account_reference = str(
        account_reference or env_map.get(account_reference_env) or ""
    ).strip()
    candidate_hash_envs = [
        str(item or "").strip()
        for item in account_slot.get("env_names", [])
        if str(item or "").strip().endswith("_HASH")
        and str(item or "").strip() != account_reference_env
    ]
    expected_references = [
        str(env_map.get(name) or "").strip()
        for name in candidate_hash_envs
        if str(env_map.get(name) or "").strip()
    ]
    account_reference_matches = bool(
        live_account_reference
        and any(
            hmac.compare_digest(live_account_reference, expected)
            for expected in expected_references
        )
    )
    if not expected_references:
        blockers.append("designated_canary_account_hash_unavailable")
    if not account_reference_matches:
        blockers.append("live_account_not_designated_canary_account")
    account_reference_sha256 = (
        _sha256_text(live_account_reference) if live_account_reference else ""
    )

    max_account_age = max(
        _safe_float(policy.get("max_account_preflight_age_seconds"), 120.0), 1.0
    )
    account_study_fresh, account_study_age = _fresh_payload(
        account_study, now=current, max_age_seconds=max_account_age
    )
    if not account_study_valid:
        blockers.append("account_position_study_invalid")
    elif not account_study_fresh:
        blockers.append("account_position_study_stale")
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
    if not account_row:
        blockers.append("designated_canary_account_truth_missing")

    capability = (
        account_row.get("account_capability_truth")
        if isinstance(account_row.get("account_capability_truth"), Mapping)
        else {}
    )
    balance_truth = (
        capability.get("balance_truth")
        if isinstance(capability.get("balance_truth"), Mapping)
        else {}
    )
    debit_truth = (
        capability.get("debit_truth")
        if isinstance(capability.get("debit_truth"), Mapping)
        else {}
    )
    call_truth = (
        capability.get("broker_call_truth")
        if isinstance(capability.get("broker_call_truth"), Mapping)
        else {}
    )
    collateral_truth = (
        capability.get("position_collateral_truth")
        if isinstance(capability.get("position_collateral_truth"), Mapping)
        else {}
    )
    flags = (
        account_row.get("flags")
        if isinstance(account_row.get("flags"), Mapping)
        else {}
    )
    account_canary = (
        account_row.get("canary_preflight")
        if isinstance(account_row.get("canary_preflight"), Mapping)
        else {}
    )
    operator_classification = (
        capability.get("operator_classification")
        if isinstance(capability.get("operator_classification"), Mapping)
        else {}
    )
    account_kind = (
        str(
            operator_classification.get("account_kind")
            or account_row.get("operator_account_kind")
            or account_slot.get("account_type")
            or "unknown"
        )
        .strip()
        .lower()
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
    account_constraints = (
        plan.get("account_constraints")
        if isinstance(plan.get("account_constraints"), Mapping)
        else {}
    )
    required_account_kind = (
        str(account_constraints.get("required_account_kind") or "").strip().lower()
    )
    required_tax_wrapper = (
        str(account_constraints.get("required_tax_wrapper") or "").strip().lower()
    )
    if required_account_kind and account_kind != required_account_kind:
        blockers.append("canary_account_kind_mismatch")
    if required_tax_wrapper and tax_wrapper != required_tax_wrapper:
        blockers.append("canary_tax_wrapper_mismatch")
    if bool(account_constraints.get("cash_only", False)) and not bool(
        account_slot.get("cash_only_live_budget", False)
    ):
        blockers.append("canary_cash_only_policy_mismatch")
    required_existing_authority = (
        str(account_constraints.get("existing_positions_authority") or "")
        .strip()
        .lower()
    )
    actual_existing_authority = (
        str(account_slot.get("existing_positions_authority") or "").strip().lower()
    )
    if (
        required_existing_authority
        and actual_existing_authority != required_existing_authority
    ):
        blockers.append("canary_existing_position_authority_mismatch")
    configured_cap = max(
        _safe_float(plan.get("account_capital_usd"), 0.0),
        _safe_float(account_slot.get("canary_cap_usd"), 0.0),
    )
    cash_available = max(
        _safe_float(balance_truth.get("cash_available_for_trading"), 0.0),
        _safe_float(
            balance_truth.get("cash_balance"), account_row.get("cash_balance", 0.0)
        ),
    )
    pending_deposits = _safe_float(balance_truth.get("pending_deposits"), 0.0)
    if configured_cap <= 0.0:
        blockers.append("canary_cash_cap_invalid")
    if cash_available < configured_cap or pending_deposits > 0.0:
        blockers.append("canary_settled_cash_not_broker_visible")
    if bool(flags.get("closing_only", False)):
        blockers.append("canary_account_closing_only")
    if bool(call_truth.get("in_call", False)):
        blockers.append("canary_account_in_broker_call")
    if _safe_float(collateral_truth.get("uncovered_short_option_count"), 0.0) > 0.0:
        blockers.append("canary_account_uncovered_short_options")
    if bool(
        account_row.get(
            "borrowing_allowed", account_slot.get("borrowing_allowed", True)
        )
    ):
        blockers.append("canary_account_borrowing_authority_present")
    for blocker in (
        account_canary.get("blockers", [])
        if isinstance(account_canary.get("blockers"), list)
        else []
    ):
        if str(blocker or "").strip():
            blockers.append(str(blocker).strip())

    attestation_mode_ok = False
    if attestation_path.exists():
        try:
            attestation_mode_ok = (
                stat.S_IMODE(attestation_path.stat().st_mode) & 0o077 == 0
            )
        except OSError:
            attestation_mode_ok = False
    attestation_blockers: list[str] = []
    attestation_tax_ledger_matches = False
    required_confirmations = required_operator_confirmations(tax_wrapper)
    missing_confirmations = list(required_confirmations)
    attestation_issued = _parse_timestamp(attestation.get("issued_at_utc"))
    attestation_expires = _parse_timestamp(attestation.get("expires_at_utc"))
    max_attestation_hours = max(
        _safe_float(
            (
                (plan.get("activation_contract") or {}).get(
                    "max_operator_attestation_hours"
                )
                if isinstance(plan.get("activation_contract"), Mapping)
                else 4.0
            ),
            4.0,
        ),
        0.0,
    )
    if not attestation_valid:
        attestation_blockers.append(
            "live_canary_operator_attestation_missing_or_invalid"
        )
    elif int(attestation.get("schema_version", 0) or 0) != 1:
        attestation_blockers.append("live_canary_operator_attestation_schema_invalid")
    elif not attestation_mode_ok:
        attestation_blockers.append(
            "live_canary_operator_attestation_permissions_unsafe"
        )
    else:
        if (
            attestation_issued is None
            or attestation_expires is None
            or attestation_expires <= current
            or attestation_issued > current + timedelta(minutes=5)
            or attestation_expires <= attestation_issued
            or (attestation_expires - attestation_issued).total_seconds()
            > max_attestation_hours * 3600.0
        ):
            attestation_blockers.append(
                "live_canary_operator_attestation_expired_or_invalid"
            )
        if str(attestation.get("candidate_id") or "").strip() != candidate_id:
            attestation_blockers.append("operator_attestation_candidate_mismatch")
        if (
            str(attestation.get("account_policy_key") or "").strip()
            != account_policy_key
        ):
            attestation_blockers.append("operator_attestation_account_policy_mismatch")
        if str(attestation.get("execution_route_id") or "").strip() != route_id:
            attestation_blockers.append("operator_attestation_route_mismatch")
        if (
            str(attestation.get("account_reference_sha256") or "").strip().lower()
            != account_reference_sha256
        ):
            attestation_blockers.append(
                "operator_attestation_account_reference_mismatch"
            )
        if str(
            attestation.get("account_study_sha256") or ""
        ).strip().lower() != _file_sha256(account_study_path):
            attestation_blockers.append("operator_attestation_account_study_mismatch")
        attestation_tax_ledger_matches = bool(
            str(attestation.get("trading_tax_ledger_sha256") or "").strip().lower()
            == _file_sha256(tax_path)
            and _file_sha256(tax_path)
        )
        if not attestation_tax_ledger_matches:
            attestation_blockers.append("operator_attestation_tax_ledger_mismatch")
        missing_confirmations = [
            field
            for field in required_confirmations
            if attestation.get(field) is not True
        ]
        if missing_confirmations:
            attestation_blockers.append("operator_attestation_confirmations_incomplete")
        if _safe_float(attestation.get("settled_cash_usd"), 0.0) < configured_cap:
            attestation_blockers.append(
                "operator_attested_settled_cash_below_canary_cap"
            )
        if bool(debit_truth.get("requires_broker_ui_confirmation", False)) and not bool(
            attestation.get("provider_balance_reviewed", False)
        ):
            attestation_blockers.append("provider_balance_requires_operator_review")
    blockers.extend(attestation_blockers)

    max_risk_age = max(
        _safe_float(policy.get("max_risk_boundary_age_seconds"), 900.0), 1.0
    )
    risk_fresh, risk_age = _fresh_payload(
        risk_boundary, now=current, max_age_seconds=max_risk_age
    )
    input_health = (
        risk_boundary.get("input_health")
        if isinstance(risk_boundary.get("input_health"), Mapping)
        else {}
    )
    risk_ready = bool(
        risk_valid
        and risk_fresh
        and risk_boundary.get("ok", False)
        and str(risk_boundary.get("overall_status") or "").strip().lower() == "ready"
        and input_health.get("sources_ready", False)
    )
    if not risk_ready:
        blockers.append("risk_service_boundary_not_ready")

    max_ledger_age = max(
        _safe_float(policy.get("max_live_order_ledger_age_seconds"), 120.0), 1.0
    )
    ledger_fresh, ledger_age = _fresh_payload(
        order_ledger, now=current, max_age_seconds=max_ledger_age
    )
    ledger_ready = bool(
        ledger_valid
        and ledger_fresh
        and order_ledger.get("ok", False)
        and str(order_ledger.get("overall_status") or "").strip().lower() == "ready"
        and int(order_ledger.get("unresolved_intent_count", 0) or 0) == 0
    )
    if not ledger_ready:
        blockers.append("live_order_ledger_not_ready")

    max_release_age = max(
        _safe_float(policy.get("max_release_guard_age_seconds"), 900.0), 1.0
    )
    release_fresh, release_age = _fresh_payload(
        release_guard, now=current, max_age_seconds=max_release_age
    )
    immutable_boundary = (
        release_guard.get("immutable_release_boundary")
        if isinstance(release_guard.get("immutable_release_boundary"), Mapping)
        else {}
    )
    git_integrity = (
        release_guard.get("git_integrity")
        if isinstance(release_guard.get("git_integrity"), Mapping)
        else {}
    )
    manifest_path = Path(str(immutable_boundary.get("manifest_path") or ""))
    if manifest_path and not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    release_manifest, manifest_valid_json = _load_json(manifest_path)
    manifest_identity = (
        release_manifest.get("release_identity")
        if isinstance(release_manifest.get("release_identity"), Mapping)
        else {}
    )
    manifest_base = {
        key: release_manifest.get(key)
        for key in (
            "schema_version",
            "created_at_utc",
            "release_identity",
            "rollback",
            "freeze_window",
            "live_execution_authority",
        )
    }
    expected_manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest_base,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    release_manifest_ready = bool(
        manifest_valid_json
        and int(release_manifest.get("schema_version", 0) or 0) == 1
        and release_manifest.get("live_execution_authority") is False
        and str(release_manifest.get("manifest_sha256") or "").strip().lower()
        == expected_manifest_sha256
        and str(manifest_identity.get("commit") or "").strip()
        == str(git_integrity.get("commit") or "").strip()
        and str(manifest_identity.get("tracked_tree_receipt_sha256") or "").strip()
        == str(git_integrity.get("tracked_tree_receipt_sha256") or "").strip()
    )
    release_ready = bool(
        release_valid
        and release_fresh
        and immutable_boundary.get("ready", False)
        and git_integrity.get("ready", False)
        and release_manifest_ready
    )
    if not release_ready:
        blockers.append("immutable_release_boundary_not_ready")

    max_tax_age = max(
        _safe_float(policy.get("max_tax_ledger_age_seconds"), 86400.0), 1.0
    )
    tax_fresh, tax_age = _fresh_payload(
        tax_ledger, now=current, max_age_seconds=max_tax_age
    )
    if not tax_valid or not tax_fresh:
        blockers.append("trading_tax_ledger_not_ready")
    tax_review = _tax_symbol_review(
        tax_ledger,
        symbol=symbol,
        action=action,
        now=current,
        lookback_days=int(
            _safe_float(policy.get("wash_sale_review_lookback_days"), 31.0)
        ),
    )
    tax_operator_review_satisfied = bool(
        attestation_tax_ledger_matches
        and attestation.get("tax_and_settlement_reviewed") is True
    )
    if not bool(tax_review.get("ready", False)) and not tax_operator_review_satisfied:
        blockers.append("same_symbol_wash_sale_review_required")
    tax_review["operator_review_satisfied"] = tax_operator_review_satisfied

    require_session = bool(policy.get("require_primary_equity_session_open", False))
    session_state = (
        _equity_session_state(
            now=current,
            calendar_id=str(policy.get("primary_equity_calendar_id") or "XNYS"),
            buffer_minutes=_safe_float(
                policy.get("equity_auction_buffer_minutes"), 5.0
            ),
        )
        if require_session
        else {"ready": True, "state": "not_enforced", "blocker": ""}
    )
    if require_session and not bool(session_state.get("ready", False)):
        blockers.append(
            str(session_state.get("blocker") or "primary_equity_session_not_ready")
        )

    entry_action = str(action or "").strip().upper() in LIVE_ENTRY_ACTIONS
    unique_blockers = _ordered_unique(blockers if entry_action else [])
    evidence = {
        "account_study_sha256": _file_sha256(account_study_path),
        "operator_attestation_sha256": _file_sha256(attestation_path),
        "risk_boundary_sha256": _file_sha256(risk_path),
        "release_guard_sha256": _file_sha256(release_path),
        "release_manifest_sha256": _file_sha256(manifest_path),
        "live_order_ledger_control_sha256": _file_sha256(ledger_path),
        "trading_tax_ledger_sha256": _file_sha256(tax_path),
        "regime_control_plane_sha256": _file_sha256(regime_path),
    }
    max_regime_age = max(
        _safe_float(policy.get("max_regime_evidence_age_seconds"), 900.0), 1.0
    )
    regime_fresh, regime_age = _fresh_payload(
        regime_control, now=current, max_age_seconds=max_regime_age
    )
    regime_bucket = (
        str(regime_control.get("regime_state") or "unknown").strip().lower()
        if regime_valid and regime_fresh
        else "unknown"
    )
    regime_receipt_sha256 = (
        _file_sha256(regime_path)
        if regime_bucket not in {"", "unknown", "unclassified"}
        else ""
    )
    receipt = {
        "schema_version": 1,
        "evaluated_at_utc": current.isoformat(),
        "ready": not unique_blockers,
        "candidate_id": candidate_id,
        "account_policy_key": account_policy_key,
        "execution_route_id": route_id,
        "account_kind": account_kind,
        "tax_wrapper": tax_wrapper,
        "retirement_account": tax_wrapper in RETIREMENT_TAX_WRAPPERS,
        "account_reference_sha256": account_reference_sha256,
        "symbol": str(symbol or "").strip().upper(),
        "action": str(action or "").strip().upper(),
        "settled_cash_required_usd": configured_cap,
        "settled_cash_broker_visible_usd": cash_available,
        "operator_attested_settled_cash_usd": _safe_float(
            attestation.get("settled_cash_usd"), 0.0
        ),
        "account_reference_matches": account_reference_matches,
        "account_study_age_seconds": account_study_age,
        "risk_boundary_age_seconds": risk_age,
        "live_order_ledger_age_seconds": ledger_age,
        "release_guard_age_seconds": release_age,
        "tax_ledger_age_seconds": tax_age,
        "regime_bucket": regime_bucket,
        "regime_receipt_sha256": regime_receipt_sha256,
        "regime_evidence_age_seconds": regime_age,
        "regime_evidence_fresh": bool(regime_valid and regime_fresh),
        "risk_boundary_ready": risk_ready,
        "live_order_ledger_ready": ledger_ready,
        "immutable_release_boundary_ready": release_ready,
        "immutable_release_manifest_ready": release_manifest_ready,
        "tax_review": tax_review,
        "equity_session": session_state,
        "operator_attestation_confirmations_complete": not missing_confirmations,
        "required_operator_confirmations": list(required_confirmations),
        "missing_operator_confirmations": missing_confirmations,
        "operator_attestation_ready": not attestation_blockers,
        "operator_attestation_blockers": attestation_blockers,
        "provider_balance_review_required": bool(
            debit_truth.get("requires_broker_ui_confirmation", False)
        ),
        "evidence": evidence,
        "blockers": unique_blockers,
        "live_execution_authority": False,
        "policy": "preflight evidence may block live entry but can never grant live execution",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return receipt
