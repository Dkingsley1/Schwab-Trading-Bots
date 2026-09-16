from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from core.order_intent import canonical_payload_sha256, verify_order_intent_evidence

LIVE_EXECUTION_ENVELOPE_SCHEMA_VERSION = 1
MUTATING_BROKER_OPERATIONS = frozenset({"place_order", "replace_order", "cancel_order"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    current = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        return ""


def broker_operation_retry_contract(
    operation: str, configured_attempts: int
) -> dict[str, Any]:
    operation_key = str(operation or "").strip().lower()
    requested = max(int(configured_attempts or 1), 1)
    mutating = operation_key in MUTATING_BROKER_OPERATIONS
    return {
        "operation": operation_key,
        "mutating": mutating,
        "configured_attempts": requested,
        "max_attempts": 1 if mutating else requested,
        "retry_after_dispatch_allowed": not mutating,
        "ambiguous_failure_requires_reconciliation": mutating,
        "policy": (
            "mutating broker calls are attempted once; uncertain outcomes reconcile before any related mutation"
            if mutating
            else "read-only broker calls may use bounded retry, backoff, and jitter"
        ),
    }


def _redacted_order_request(order_request: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(order_request)
    account_reference = str(payload.pop("account_reference", "") or "").strip()
    payload["account_reference_sha256"] = (
        _sha256_text(account_reference) if account_reference else ""
    )
    return payload


def _client_order_id(
    *, candidate_id: str, broker: str, intent_sha256: str, account_reference_sha256: str
) -> str:
    digest = canonical_payload_sha256(
        {
            "candidate_id": str(candidate_id or "").strip(),
            "broker": str(broker or "").strip().lower(),
            "intent_sha256": str(intent_sha256 or "").strip().lower(),
            "account_reference_sha256": str(account_reference_sha256 or "")
            .strip()
            .lower(),
        }
    )
    return f"stb-{digest[:28]}"


def build_live_execution_envelope(
    *,
    intent_evidence: Mapping[str, Any],
    order_request: Mapping[str, Any],
    candidate_id: str,
    broker: str,
    account_reference: str,
    account_snapshot_evidence: Mapping[str, Any],
    policy_sha256: str,
    ttl_seconds: float = 15.0,
    created_at_utc: datetime | None = None,
) -> dict[str, Any]:
    created = created_at_utc or _utc_now()
    ttl = max(float(ttl_seconds or 0.0), 0.001)
    redacted_request = _redacted_order_request(order_request)
    account_reference_hash = (
        _sha256_text(str(account_reference or "").strip())
        if str(account_reference or "").strip()
        else ""
    )
    intent = dict(intent_evidence)
    account_snapshot = dict(account_snapshot_evidence)
    intent_sha256 = str(intent.get("intent_sha256") or "").strip().lower()
    component_hashes = {
        "intent_sha256": intent_sha256,
        "broker_order_request_sha256": canonical_payload_sha256(redacted_request),
        "account_snapshot_sha256": canonical_payload_sha256(account_snapshot),
        "account_reference_sha256": account_reference_hash,
        "policy_sha256": str(policy_sha256 or "").strip().lower(),
    }
    immutable = {
        "schema_version": LIVE_EXECUTION_ENVELOPE_SCHEMA_VERSION,
        "created_at_utc": _iso_utc(created),
        "expires_at_utc": _iso_utc(created + timedelta(seconds=ttl)),
        "candidate_id": str(candidate_id or "").strip(),
        "broker": str(broker or "").strip().lower(),
        "client_order_id": _client_order_id(
            candidate_id=candidate_id,
            broker=broker,
            intent_sha256=intent_sha256,
            account_reference_sha256=account_reference_hash,
        ),
        "intent_evidence": intent,
        "broker_order_request": redacted_request,
        "account_snapshot_evidence": account_snapshot,
        "component_hashes": component_hashes,
        "authority": {
            "live_execution_authority": False,
            "operator_release_still_required": True,
            "envelope_cannot_submit_order": True,
        },
    }
    return {
        **immutable,
        "envelope_sha256": canonical_payload_sha256(immutable),
    }


def _quote_freshness(
    quote: Mapping[str, Any],
    *,
    now_utc: datetime,
    max_quote_age_seconds: float,
    max_future_skew_seconds: float,
) -> tuple[float | None, list[str]]:
    blockers: list[str] = []
    timestamp = _parse_timestamp(quote.get("timestamp_utc"))
    reported_age_ms = quote.get("quote_age_ms")
    reported_age: float | None = None
    try:
        numeric_age = float(reported_age_ms)
        if numeric_age > 0.0:
            reported_age = numeric_age / 1000.0
    except (TypeError, ValueError):
        reported_age = None

    timestamp_age: float | None = None
    if timestamp is not None:
        delta = (now_utc - timestamp).total_seconds()
        if delta < -max(float(max_future_skew_seconds), 0.0):
            blockers.append("quote_timestamp_in_future")
        timestamp_age = max(delta, 0.0)
    if timestamp is None and reported_age is None:
        blockers.append("quote_freshness_missing")
        return None, blockers

    observed_age = max(
        value for value in (timestamp_age, reported_age) if value is not None
    )
    if observed_age > max(float(max_quote_age_seconds), 0.0):
        blockers.append("quote_is_stale")
    return observed_age, blockers


def _quote_spread_bps(quote: Mapping[str, Any]) -> float | None:
    try:
        spread = float(quote.get("spread_bps") or 0.0)
    except (TypeError, ValueError):
        spread = 0.0
    if spread > 0.0:
        return spread
    try:
        bid = float(quote.get("bid_price") or 0.0)
        ask = float(quote.get("ask_price") or 0.0)
    except (TypeError, ValueError):
        return None
    midpoint = (bid + ask) / 2.0
    if bid > 0.0 and ask >= bid and midpoint > 0.0:
        return ((ask - bid) / midpoint) * 10000.0
    return None


def verify_live_execution_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_candidate_id: str = "",
    expected_account_reference: str = "",
    expected_policy_sha256: str = "",
    now_utc: datetime | None = None,
    max_quote_age_seconds: float = 15.0,
    max_account_snapshot_age_seconds: float = 30.0,
    max_spread_bps: float = 75.0,
    max_future_skew_seconds: float = 2.0,
    require_affirmative_risk_decision: bool = False,
    require_quote_provenance: bool = False,
    allowed_quote_providers: tuple[str, ...] = (),
    require_canary_preflight_receipt: bool = False,
    expected_account_policy_key: str = "",
) -> dict[str, Any]:
    blockers: list[str] = []
    current = now_utc or _utc_now()
    intent = (
        envelope.get("intent_evidence")
        if isinstance(envelope.get("intent_evidence"), Mapping)
        else {}
    )
    order_request = (
        envelope.get("broker_order_request")
        if isinstance(envelope.get("broker_order_request"), Mapping)
        else {}
    )
    snapshot = (
        envelope.get("account_snapshot_evidence")
        if isinstance(envelope.get("account_snapshot_evidence"), Mapping)
        else {}
    )
    supplied_hashes = (
        envelope.get("component_hashes")
        if isinstance(envelope.get("component_hashes"), Mapping)
        else {}
    )

    if (
        int(envelope.get("schema_version", 0) or 0)
        != LIVE_EXECUTION_ENVELOPE_SCHEMA_VERSION
    ):
        blockers.append("live_execution_envelope_schema_invalid")
    broker = str(envelope.get("broker") or "").strip().lower()
    if not broker:
        blockers.append("broker_identity_missing")
    intent_verification = verify_order_intent_evidence(intent)
    if not intent_verification.get("ok", False):
        blockers.append("order_intent_evidence_invalid")
    risk_decision = (
        intent.get("risk_decision")
        if isinstance(intent.get("risk_decision"), Mapping)
        else {}
    )
    if require_affirmative_risk_decision and risk_decision.get("ok") is not True:
        blockers.append("order_intent_risk_decision_not_approved")
    if not str(envelope.get("candidate_id") or "").strip():
        blockers.append("candidate_id_missing")
    if (
        expected_candidate_id
        and str(envelope.get("candidate_id") or "").strip()
        != str(expected_candidate_id).strip()
    ):
        blockers.append("candidate_id_mismatch")
    expected_account_hash = (
        _sha256_text(str(expected_account_reference or "").strip())
        if expected_account_reference
        else ""
    )
    if not str(supplied_hashes.get("account_reference_sha256") or "").strip():
        blockers.append("account_reference_hash_missing")
    elif (
        expected_account_hash
        and str(supplied_hashes.get("account_reference_sha256"))
        != expected_account_hash
    ):
        blockers.append("account_reference_hash_mismatch")
    if not snapshot:
        blockers.append("account_snapshot_evidence_missing")
    else:
        snapshot_digest = (
            str(snapshot.get("broker_position_snapshot_sha256") or "").strip().lower()
        )
        if len(snapshot_digest) != 64:
            blockers.append("account_snapshot_sha256_missing")
        snapshot_timestamp = _parse_timestamp(
            snapshot.get("broker_position_snapshot_captured_at_utc")
        )
        if snapshot_timestamp is None:
            blockers.append("account_snapshot_timestamp_missing")
        else:
            snapshot_age = (current - snapshot_timestamp).total_seconds()
            if snapshot_age < -max(float(max_future_skew_seconds), 0.0):
                blockers.append("account_snapshot_timestamp_in_future")
            elif snapshot_age > max(float(max_account_snapshot_age_seconds), 0.0):
                blockers.append("account_snapshot_is_stale")
        if require_canary_preflight_receipt:
            preflight = (
                snapshot.get("live_canary_preflight_receipt")
                if isinstance(snapshot.get("live_canary_preflight_receipt"), Mapping)
                else {}
            )
            if not preflight:
                blockers.append("live_canary_preflight_receipt_missing")
            else:
                if preflight.get("ready") is not True:
                    blockers.append("live_canary_preflight_receipt_not_ready")
                if len(str(preflight.get("receipt_sha256") or "").strip()) != 64:
                    blockers.append("live_canary_preflight_receipt_sha256_missing")
                if (
                    expected_account_policy_key
                    and str(preflight.get("account_policy_key") or "").strip()
                    != str(expected_account_policy_key).strip()
                ):
                    blockers.append("live_canary_preflight_account_policy_mismatch")
                if (
                    str(preflight.get("account_reference_sha256") or "")
                    .strip()
                    .lower()
                    != expected_account_hash
                ):
                    blockers.append("live_canary_preflight_account_reference_mismatch")

    expected_hashes = {
        "intent_sha256": str(intent.get("intent_sha256") or "").strip().lower(),
        "broker_order_request_sha256": canonical_payload_sha256(order_request),
        "account_snapshot_sha256": canonical_payload_sha256(snapshot),
        "account_reference_sha256": str(
            supplied_hashes.get("account_reference_sha256") or ""
        )
        .strip()
        .lower(),
        "policy_sha256": str(supplied_hashes.get("policy_sha256") or "")
        .strip()
        .lower(),
    }
    for key, expected in expected_hashes.items():
        if str(supplied_hashes.get(key) or "").strip().lower() != expected:
            blockers.append(f"{key}_mismatch")
    if len(expected_hashes["policy_sha256"]) != 64:
        blockers.append("policy_sha256_missing")
    elif (
        expected_policy_sha256
        and expected_hashes["policy_sha256"]
        != str(expected_policy_sha256).strip().lower()
    ):
        blockers.append("policy_sha256_mismatch")
    if (
        str(order_request.get("account_reference_sha256") or "").strip().lower()
        != expected_hashes["account_reference_sha256"]
    ):
        blockers.append("broker_order_account_reference_hash_mismatch")

    expected_client_order_id = _client_order_id(
        candidate_id=str(envelope.get("candidate_id") or ""),
        broker=broker,
        intent_sha256=expected_hashes["intent_sha256"],
        account_reference_sha256=expected_hashes["account_reference_sha256"],
    )
    if str(envelope.get("client_order_id") or "").strip() != expected_client_order_id:
        blockers.append("client_order_id_mismatch")

    semantic = (
        intent.get("semantic_order")
        if isinstance(intent.get("semantic_order"), Mapping)
        else {}
    )
    parity_fields = {
        "symbol": str(order_request.get("symbol") or "").strip().upper()
        == str(semantic.get("symbol") or "").strip().upper(),
        "action": str(order_request.get("action") or "").strip().upper()
        == str(semantic.get("action") or "").strip().upper(),
        "asset_type": str(order_request.get("asset_type") or "").strip().upper()
        == str(semantic.get("asset_type") or "").strip().upper(),
    }
    try:
        parity_fields["quantity"] = (
            abs(
                float(order_request.get("quantity") or 0.0)
                - float(semantic.get("quantity") or 0.0)
            )
            <= 1e-9
        )
        request_limit = float(order_request.get("limit_price") or 0.0)
        semantic_limit = float(semantic.get("limit_price") or 0.0)
        parity_fields["limit_price"] = abs(request_limit - semantic_limit) <= 1e-9
    except (TypeError, ValueError):
        parity_fields["quantity"] = False
        parity_fields["limit_price"] = False
    if not all(parity_fields.values()):
        blockers.append("intent_to_broker_payload_parity_failed")

    created = _parse_timestamp(envelope.get("created_at_utc"))
    expires = _parse_timestamp(envelope.get("expires_at_utc"))
    if created is None or expires is None or expires <= created:
        blockers.append("execution_envelope_time_window_invalid")
    else:
        if (created - current).total_seconds() > max(
            float(max_future_skew_seconds), 0.0
        ):
            blockers.append("execution_envelope_created_in_future")
        if current > expires:
            blockers.append("execution_envelope_expired")

    quote = (
        intent.get("quote_snapshot")
        if isinstance(intent.get("quote_snapshot"), Mapping)
        else {}
    )
    quote_age_seconds, quote_blockers = _quote_freshness(
        quote,
        now_utc=current,
        max_quote_age_seconds=max_quote_age_seconds,
        max_future_skew_seconds=max_future_skew_seconds,
    )
    blockers.extend(quote_blockers)
    quote_provider = str(quote.get("source_provider") or "").strip().lower()
    allowed_provider_set = {
        str(item or "").strip().lower()
        for item in allowed_quote_providers
        if str(item or "").strip()
    }
    if require_quote_provenance and not quote_provider:
        blockers.append("quote_source_provider_missing")
    elif allowed_provider_set and quote_provider not in allowed_provider_set:
        blockers.append("quote_source_provider_not_allowed")
    spread_bps = _quote_spread_bps(quote)
    if spread_bps is None:
        blockers.append("quote_spread_evidence_missing")
    elif spread_bps > max(float(max_spread_bps), 0.0):
        blockers.append("spread_exceeds_cap")

    immutable = {
        key: envelope.get(key)
        for key in (
            "schema_version",
            "created_at_utc",
            "expires_at_utc",
            "candidate_id",
            "broker",
            "client_order_id",
            "intent_evidence",
            "broker_order_request",
            "account_snapshot_evidence",
            "component_hashes",
            "authority",
        )
    }
    expected_envelope_sha256 = canonical_payload_sha256(immutable)
    if (
        str(envelope.get("envelope_sha256") or "").strip().lower()
        != expected_envelope_sha256
    ):
        blockers.append("execution_envelope_sha256_mismatch")

    authority = (
        envelope.get("authority")
        if isinstance(envelope.get("authority"), Mapping)
        else {}
    )
    if (
        authority.get("live_execution_authority") is not False
        or authority.get("envelope_cannot_submit_order") is not True
    ):
        blockers.append("execution_envelope_authority_boundary_invalid")

    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "ok": not unique_blockers,
        "blockers": unique_blockers,
        "intent_verification": intent_verification,
        "parity_fields": parity_fields,
        "quote_age_seconds": quote_age_seconds,
        "spread_bps": spread_bps,
        "quote_source_provider": quote_provider,
        "expected_envelope_sha256": expected_envelope_sha256,
        "envelope_sha256": str(envelope.get("envelope_sha256") or ""),
        "client_order_id": str(envelope.get("client_order_id") or ""),
        "expected_client_order_id": expected_client_order_id,
    }
