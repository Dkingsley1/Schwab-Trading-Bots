from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from core.order_intent import canonical_payload_sha256, verify_order_intent_evidence

RECEIPT_SCHEMA_VERSION = 1
ENTRY_ACTIONS = frozenset({"BUY", "BUY_TO_OPEN"})
EXIT_ACTIONS = frozenset({"SELL", "SELL_TO_CLOSE"})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in values if str(item)))


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    return canonical_payload_sha256(unsigned)


def build_live_canary_closeout_receipt(
    *,
    intent_id: str,
    intent_payload_sha256: str,
    final_order_event_sha256: str,
    broker_order_id: str,
    candidate_id: str,
    account_policy_key: str,
    execution_route_id: str,
    account_reference_sha256: str,
    symbol: str,
    action: str,
    quantity: float,
    fill_price: float,
    filled_at_utc: str,
    broker_fees_usd: float,
    fee_evidence_source: str,
    pre_position_quantity: float,
    post_position_quantity: float,
    pre_settled_cash_usd: float,
    post_settled_cash_usd: float,
    order_reconciled: bool,
    position_reconciled: bool,
    cash_reconciled: bool,
    position_delta_verified: bool,
    reduce_only_exit: bool = False,
    unresolved_broker_operation: bool = False,
    safety_violations: Iterable[str] = (),
    regime_bucket: str = "unknown",
    regime_receipt_sha256: str = "",
    benchmark_return_bps: float | None = None,
    benchmark_receipt_sha256: str = "",
    account_study_sha256: str = "",
    account_study_timestamp_utc: str = "",
    source: str = "broker_reconciled_live_canary_closeout",
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": str(source or "").strip(),
        "intent_id": str(intent_id or "").strip(),
        "intent_payload_sha256": str(intent_payload_sha256 or "").strip().lower(),
        "final_order_event_sha256": str(final_order_event_sha256 or "").strip().lower(),
        "broker_order_id_sha256": _sha256_text(str(broker_order_id or "").strip()),
        "candidate_id": str(candidate_id or "").strip(),
        "account_policy_key": str(account_policy_key or "").strip(),
        "execution_route_id": str(execution_route_id or "").strip(),
        "account_reference_sha256": str(account_reference_sha256 or "").strip().lower(),
        "symbol": str(symbol or "").strip().upper(),
        "action": str(action or "").strip().upper(),
        "quantity": float(quantity),
        "fill_price": float(fill_price),
        "filled_at_utc": str(filled_at_utc or "").strip(),
        "broker_fees_usd": float(broker_fees_usd),
        "fee_evidence_source": str(fee_evidence_source or "").strip(),
        "pre_position_quantity": float(pre_position_quantity),
        "post_position_quantity": float(post_position_quantity),
        "position_delta_quantity": round(
            float(post_position_quantity) - float(pre_position_quantity), 12
        ),
        "pre_settled_cash_usd": float(pre_settled_cash_usd),
        "post_settled_cash_usd": float(post_settled_cash_usd),
        "order_reconciled": bool(order_reconciled),
        "position_reconciled": bool(position_reconciled),
        "cash_reconciled": bool(cash_reconciled),
        "position_delta_verified": bool(position_delta_verified),
        "reduce_only_exit": bool(reduce_only_exit),
        "unresolved_broker_operation": bool(unresolved_broker_operation),
        "safety_violations": _ordered_unique(safety_violations),
        "regime_bucket": str(regime_bucket or "unknown").strip().lower(),
        "regime_receipt_sha256": str(regime_receipt_sha256 or "").strip().lower(),
        "benchmark_return_bps": (
            float(benchmark_return_bps) if benchmark_return_bps is not None else None
        ),
        "benchmark_receipt_sha256": str(benchmark_receipt_sha256 or "").strip().lower(),
        "account_study_sha256": str(account_study_sha256 or "").strip().lower(),
        "account_study_timestamp_utc": str(account_study_timestamp_utc or "").strip(),
        "authority": {
            "live_execution_authority": False,
            "stage_progression_authority": False,
            "capital_scaling_authority": False,
            "operator_review_required": True,
        },
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def extract_live_canary_intent_identity(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    payload_json = str(row.get("payload_json") or "")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        payload = {}
        errors.append("intent_payload_json_invalid")
    if not isinstance(payload, dict):
        payload = {}
        errors.append("intent_payload_not_object")
    if _sha256_text(payload_json) != str(row.get("payload_hash") or "").lower():
        errors.append("intent_payload_hash_mismatch")

    envelope = _mapping(payload.get("live_execution_envelope"))
    envelope_hash = str(envelope.get("envelope_sha256") or "").strip().lower()
    unsigned_envelope = dict(envelope)
    unsigned_envelope.pop("envelope_sha256", None)
    if not envelope or envelope_hash != canonical_payload_sha256(unsigned_envelope):
        errors.append("live_execution_envelope_hash_mismatch")

    envelope_intent = _mapping(envelope.get("intent_evidence"))
    stored_intent = _mapping(payload.get("mode_invariant_intent"))
    intent_check = verify_order_intent_evidence(envelope_intent)
    if not intent_check.get("ok", False):
        errors.append("order_intent_evidence_invalid")
    if str(stored_intent.get("intent_sha256") or "") != str(
        envelope_intent.get("intent_sha256") or ""
    ):
        errors.append("stored_and_enveloped_intent_mismatch")

    semantic = _mapping(envelope_intent.get("semantic_order"))
    expected_fill = _mapping(envelope_intent.get("expected_fill"))
    snapshot = _mapping(envelope.get("account_snapshot_evidence"))
    preflight = _mapping(snapshot.get("live_canary_preflight_receipt"))
    component_hashes = _mapping(envelope.get("component_hashes"))
    if preflight.get("ready") is not True:
        errors.append("sealed_canary_preflight_not_ready")
    if len(str(preflight.get("receipt_sha256") or "").strip()) != 64:
        errors.append("sealed_canary_preflight_hash_missing")

    quantity = _number(semantic.get("quantity"))
    payload_quantity = _number(payload.get("quantity"))
    if (
        quantity is None
        or payload_quantity is None
        or abs(quantity - payload_quantity) > 1e-9
    ):
        errors.append("intent_quantity_mismatch")
    symbol = str(semantic.get("symbol") or "").strip().upper()
    action = str(semantic.get("action") or "").strip().upper()
    if symbol != str(payload.get("symbol") or "").strip().upper():
        errors.append("intent_symbol_mismatch")
    if action != str(payload.get("action") or "").strip().upper():
        errors.append("intent_action_mismatch")

    return (
        {
            "candidate_id": str(envelope.get("candidate_id") or "").strip(),
            "account_policy_key": str(
                preflight.get("account_policy_key") or ""
            ).strip(),
            "execution_route_id": str(
                preflight.get("execution_route_id") or ""
            ).strip(),
            "account_reference_sha256": str(
                component_hashes.get("account_reference_sha256") or ""
            )
            .strip()
            .lower(),
            "symbol": symbol,
            "action": action,
            "quantity": float(quantity or 0.0),
            "expected_fill_price": float(
                _number(expected_fill.get("expected_fill_price")) or 0.0
            ),
            "asset_type": str(semantic.get("asset_type") or "").strip().upper(),
            "payload_hash": str(row.get("payload_hash") or "").strip().lower(),
            "pre_position_quantity": _number(
                snapshot.get("broker_position_snapshot_quantity")
            ),
            "pre_settled_cash_usd": _number(
                preflight.get("settled_cash_broker_visible_usd")
            ),
            "regime_bucket": str(preflight.get("regime_bucket") or "unknown")
            .strip()
            .lower(),
            "regime_receipt_sha256": str(preflight.get("regime_receipt_sha256") or "")
            .strip()
            .lower(),
        },
        errors,
    )


def _fill_deviation_bps(action: str, actual: float, expected: float) -> float | None:
    if actual <= 0.0 or expected <= 0.0:
        return None
    return abs(actual - expected) / expected * 10000.0


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None,
    final_event: Mapping[str, Any] | None,
    canary_plan: Mapping[str, Any],
    graduation_policy: Mapping[str, Any],
    stage_symbols: set[str],
) -> dict[str, Any]:
    hard_errors: list[str] = []
    pending: list[str] = []
    evidence_policy = _mapping(graduation_policy.get("evidence"))
    if int(receipt.get("schema_version", 0) or 0) != RECEIPT_SCHEMA_VERSION:
        hard_errors.append("closeout_receipt_schema_invalid")
    if str(receipt.get("receipt_sha256") or "").lower() != _receipt_hash(receipt):
        hard_errors.append("closeout_receipt_hash_mismatch")
    allowed_sources = {
        str(item or "").strip()
        for item in _list(evidence_policy.get("allowed_closeout_sources"))
        if str(item or "").strip()
    }
    if str(receipt.get("source") or "").strip() not in allowed_sources:
        hard_errors.append("closeout_receipt_source_not_allowed")
    if row is None:
        hard_errors.append("closeout_receipt_intent_missing_from_ledger")
        identity: dict[str, Any] = {}
    else:
        identity, identity_errors = extract_live_canary_intent_identity(row)
        hard_errors.extend(identity_errors)
        if str(row.get("state") or "").strip() != "filled":
            hard_errors.append("closeout_receipt_order_not_filled")
        if (
            str(receipt.get("intent_payload_sha256") or "").lower()
            != str(row.get("payload_hash") or "").lower()
        ):
            hard_errors.append("closeout_receipt_intent_payload_mismatch")
        broker_order_id = str(row.get("broker_order_id") or "").strip()
        if not broker_order_id or str(
            receipt.get("broker_order_id_sha256") or ""
        ).lower() != _sha256_text(broker_order_id):
            hard_errors.append("closeout_receipt_broker_order_mismatch")
        fill_price = _number(receipt.get("fill_price"))
        ledger_fill_price = _number(row.get("average_fill_price"))
        if (
            fill_price is None
            or ledger_fill_price is None
            or fill_price <= 0.0
            or abs(fill_price - ledger_fill_price) > 1e-9
        ):
            hard_errors.append("closeout_receipt_fill_price_mismatch")

    if final_event is None:
        hard_errors.append("closeout_receipt_final_order_event_missing")
    else:
        if str(final_event.get("to_state") or "") != "filled":
            hard_errors.append("closeout_receipt_final_order_event_not_filled")
        if (
            str(receipt.get("final_order_event_sha256") or "").lower()
            != str(final_event.get("event_hash") or "").lower()
        ):
            hard_errors.append("closeout_receipt_final_order_event_mismatch")

    expected_identity = {
        "account_policy_key": str(canary_plan.get("account_policy_key") or "").strip(),
        "execution_route_id": str(canary_plan.get("execution_route_id") or "").strip(),
    }
    for key in (
        "candidate_id",
        "account_policy_key",
        "execution_route_id",
        "account_reference_sha256",
        "symbol",
        "action",
    ):
        expected = str(identity.get(key) or "").strip()
        supplied = str(receipt.get(key) or "").strip()
        if key in {"symbol", "action"}:
            expected = expected.upper()
            supplied = supplied.upper()
        if not expected or supplied != expected:
            hard_errors.append(f"closeout_receipt_{key}_mismatch")
    for key, expected in expected_identity.items():
        if expected and str(receipt.get(key) or "").strip() != expected:
            hard_errors.append(f"closeout_receipt_{key}_policy_mismatch")

    quantity = _number(receipt.get("quantity"))
    if (
        quantity is None
        or quantity <= 0.0
        or abs(quantity - float(identity.get("quantity", 0.0))) > 1e-9
    ):
        hard_errors.append("closeout_receipt_quantity_mismatch")
    max_quantity = _number(
        _mapping(canary_plan.get("hard_limits")).get("max_order_quantity")
    )
    if (
        quantity is not None
        and max_quantity is not None
        and quantity > max_quantity + 1e-9
    ):
        hard_errors.append("closeout_receipt_exceeds_current_quantity_limit")
    if str(receipt.get("symbol") or "").upper() not in stage_symbols:
        hard_errors.append("closeout_receipt_symbol_outside_canary_plan")

    fill_price = float(_number(receipt.get("fill_price")) or 0.0)
    max_notional = _number(
        _mapping(canary_plan.get("hard_limits")).get("max_order_notional_usd")
    )
    if (
        quantity is not None
        and max_notional is not None
        and fill_price * quantity > max_notional + 1e-9
    ):
        hard_errors.append("closeout_receipt_exceeds_current_notional_limit")

    filled_at = _parse_timestamp(receipt.get("filled_at_utc"))
    if filled_at is None:
        hard_errors.append("closeout_receipt_fill_timestamp_invalid")
    action = str(receipt.get("action") or "").strip().upper()
    delta = _number(receipt.get("position_delta_quantity"))
    expected_delta = quantity if action in ENTRY_ACTIONS else -float(quantity or 0.0)
    if action not in ENTRY_ACTIONS | EXIT_ACTIONS:
        hard_errors.append("closeout_receipt_action_not_round_trip_eligible")
    elif delta is None or abs(delta - expected_delta) > 1e-9:
        hard_errors.append("closeout_receipt_position_delta_mismatch")

    safety_violations = _ordered_unique(
        str(item or "").strip() for item in _list(receipt.get("safety_violations"))
    )
    if safety_violations:
        hard_errors.append("closeout_receipt_safety_violation")
    if receipt.get("unresolved_broker_operation") is True:
        hard_errors.append("closeout_receipt_broker_state_ambiguous")
    if (
        action in EXIT_ACTIONS
        and _mapping(graduation_policy.get("stage_progression")).get(
            "require_reduce_only_exit", True
        )
        and receipt.get("reduce_only_exit") is not True
    ):
        hard_errors.append("closeout_receipt_exit_not_reduce_only")

    required_flags = {
        "order_reconciled": "closeout_order_reconciliation_pending",
        "position_reconciled": "closeout_position_reconciliation_pending",
        "cash_reconciled": "closeout_cash_reconciliation_pending",
        "position_delta_verified": "closeout_position_delta_verification_pending",
    }
    for field, reason in required_flags.items():
        if receipt.get(field) is not True:
            pending.append(reason)
    fees = _number(receipt.get("broker_fees_usd"))
    accepted_fee_sources = {
        str(item or "").strip()
        for item in _list(evidence_policy.get("accepted_fee_evidence_sources"))
        if str(item or "").strip()
    }
    fee_source = str(receipt.get("fee_evidence_source") or "").strip()
    if fees is None or fees < 0.0 or fee_source not in accepted_fee_sources:
        pending.append("closeout_fee_evidence_pending")

    if len(str(receipt.get("account_study_sha256") or "").strip()) != 64:
        pending.append("closeout_account_study_receipt_pending")
    if _parse_timestamp(receipt.get("account_study_timestamp_utc")) is None:
        pending.append("closeout_account_study_timestamp_pending")

    expected_fill = float(identity.get("expected_fill_price", 0.0) or 0.0)
    deviation = _fill_deviation_bps(action, fill_price, expected_fill)
    if deviation is None:
        pending.append("closeout_expected_fill_evidence_pending")

    actual_fee_sources = {
        str(item or "").strip()
        for item in _list(evidence_policy.get("actual_fee_evidence_sources"))
        if str(item or "").strip()
    }
    regime_bucket = str(receipt.get("regime_bucket") or "").strip().lower()
    regime_hash = str(receipt.get("regime_receipt_sha256") or "").strip().lower()
    verified_regime_bucket = (
        regime_bucket
        if regime_bucket not in {"", "unknown", "unclassified"}
        and len(regime_hash) == 64
        else ""
    )
    progression_pending: list[str] = []
    if not verified_regime_bucket:
        progression_pending.append("regime_receipt_evidence_pending")

    benchmark = _number(receipt.get("benchmark_return_bps"))
    benchmark_hash = str(receipt.get("benchmark_receipt_sha256") or "").strip().lower()
    verified_benchmark = (
        benchmark if benchmark is not None and len(benchmark_hash) == 64 else None
    )
    if benchmark is not None and verified_benchmark is None:
        progression_pending.append("benchmark_receipt_evidence_pending")
    return {
        "usable": not hard_errors and not pending,
        "hard_errors": _ordered_unique(hard_errors),
        "pending": _ordered_unique(pending),
        "identity": identity,
        "filled_at": filled_at,
        "fill_deviation_bps": deviation,
        "actual_fee_evidence": fee_source in actual_fee_sources,
        "safety_violations": safety_violations,
        "verified_regime_bucket": verified_regime_bucket,
        "verified_benchmark_return_bps": verified_benchmark,
        "progression_pending": _ordered_unique(progression_pending),
    }


def _pair_round_trips(
    receipts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    open_entries: dict[tuple[str, ...], deque[dict[str, Any]]] = defaultdict(deque)
    completed: list[dict[str, Any]] = []
    errors: list[str] = []
    ordered = sorted(receipts, key=lambda item: item["_validation"]["filled_at"])
    for receipt in ordered:
        identity = receipt["_validation"]["identity"]
        key = (
            str(identity.get("candidate_id") or ""),
            str(identity.get("account_policy_key") or ""),
            str(identity.get("execution_route_id") or ""),
            str(identity.get("account_reference_sha256") or ""),
            str(identity.get("symbol") or ""),
        )
        action = str(identity.get("action") or "").upper()
        if action in ENTRY_ACTIONS:
            open_entries[key].append(receipt)
            continue
        if action not in EXIT_ACTIONS:
            continue
        quantity = float(identity.get("quantity", 0.0) or 0.0)
        entry: dict[str, Any] | None = None
        while open_entries[key]:
            candidate = open_entries[key].popleft()
            entry_quantity = float(
                candidate["_validation"]["identity"].get("quantity", 0.0) or 0.0
            )
            if abs(entry_quantity - quantity) <= 1e-9:
                entry = candidate
                break
            errors.append("round_trip_quantity_pairing_mismatch")
        if entry is None:
            errors.append("round_trip_exit_without_matched_canary_entry")
            continue
        entry_price = float(entry.get("fill_price", 0.0) or 0.0)
        exit_price = float(receipt.get("fill_price", 0.0) or 0.0)
        fees = float(entry.get("broker_fees_usd", 0.0) or 0.0) + float(
            receipt.get("broker_fees_usd", 0.0) or 0.0
        )
        gross = (exit_price - entry_price) * quantity
        post_cost = gross - fees
        entry_notional = entry_price * quantity
        return_bps = (
            post_cost / entry_notional * 10000.0 if entry_notional > 0.0 else 0.0
        )
        regimes = {
            str(item["_validation"].get("verified_regime_bucket") or "").strip().lower()
            for item in (entry, receipt)
            if str(item["_validation"].get("verified_regime_bucket") or "").strip()
        }
        benchmark = _number(receipt["_validation"].get("verified_benchmark_return_bps"))
        completed.append(
            {
                "round_trip_id": canonical_payload_sha256(
                    {
                        "entry_receipt_sha256": entry.get("receipt_sha256"),
                        "exit_receipt_sha256": receipt.get("receipt_sha256"),
                    }
                ),
                "symbol": key[-1],
                "quantity": quantity,
                "entry_filled_at_utc": str(entry.get("filled_at_utc") or ""),
                "exit_filled_at_utc": str(receipt.get("filled_at_utc") or ""),
                "exit_trading_day": receipt["_validation"]["filled_at"]
                .date()
                .isoformat(),
                "gross_pnl_usd": round(gross, 8),
                "fees_usd": round(fees, 8),
                "post_cost_pnl_usd": round(post_cost, 8),
                "post_cost_return_bps": round(return_bps, 8),
                "maximum_fill_deviation_bps": round(
                    max(
                        float(entry["_validation"].get("fill_deviation_bps") or 0.0),
                        float(receipt["_validation"].get("fill_deviation_bps") or 0.0),
                    ),
                    8,
                ),
                "regime_buckets": sorted(regimes),
                "actual_fee_evidence": bool(
                    entry["_validation"].get("actual_fee_evidence", False)
                    and receipt["_validation"].get("actual_fee_evidence", False)
                ),
                "benchmark_return_bps": benchmark,
                "benchmark_excess_return_bps": (
                    round(return_bps - benchmark, 8) if benchmark is not None else None
                ),
                "fully_reconciled": True,
            }
        )
    return completed, _ordered_unique(errors)


def _round_trip_metrics(round_trips: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        round_trips, key=lambda row: str(row.get("exit_filled_at_utc") or "")
    )
    pnls = [float(row.get("post_cost_pnl_usd", 0.0) or 0.0) for row in ordered]
    returns = [float(row.get("post_cost_return_bps", 0.0) or 0.0) for row in ordered]
    daily: dict[str, float] = defaultdict(float)
    regimes: set[str] = set()
    actual_fee_count = 0
    benchmark_excess: list[float] = []
    peak = 0.0
    cumulative = 0.0
    max_drawdown = 0.0
    losing_streak = 0
    maximum_losing_streak = 0
    for row, pnl in zip(ordered, pnls):
        daily[str(row.get("exit_trading_day") or "unknown")] += pnl
        regimes.update(
            str(item) for item in _list(row.get("regime_buckets")) if str(item)
        )
        actual_fee_count += int(bool(row.get("actual_fee_evidence", False)))
        benchmark_value = _number(row.get("benchmark_excess_return_bps"))
        if benchmark_value is not None:
            benchmark_excess.append(benchmark_value)
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        if pnl < 0.0:
            losing_streak += 1
            maximum_losing_streak = max(maximum_losing_streak, losing_streak)
        else:
            losing_streak = 0
    mean_return = statistics.fmean(returns) if returns else 0.0
    if len(returns) >= 2:
        lcb = mean_return - 1.96 * statistics.stdev(returns) / math.sqrt(len(returns))
    else:
        lcb = None
    positive_days = sum(1 for value in daily.values() if value > 0.0)
    return {
        "reconciled_round_trip_count": len(ordered),
        "profitable_round_trip_count": sum(1 for value in pnls if value > 0.0),
        "independent_trading_day_count": len(daily),
        "positive_trading_day_count": positive_days,
        "positive_trading_day_ratio": (
            round(positive_days / len(daily), 8) if daily else 0.0
        ),
        "distinct_regime_bucket_count": len(regimes),
        "regime_buckets": sorted(regimes),
        "total_post_cost_pnl_usd": round(sum(pnls), 8),
        "mean_post_cost_return_bps": round(mean_return, 8),
        "normal_approx_lcb_95_post_cost_return_bps": (
            round(lcb, 8) if lcb is not None else None
        ),
        "maximum_fill_deviation_bps": round(
            max(
                [
                    float(row.get("maximum_fill_deviation_bps", 0.0) or 0.0)
                    for row in ordered
                ]
                or [0.0]
            ),
            8,
        ),
        "maximum_daily_loss_usd": round(
            max([-value for value in daily.values() if value < 0.0] or [0.0]), 8
        ),
        "maximum_cumulative_drawdown_usd": round(max_drawdown, 8),
        "maximum_consecutive_losing_round_trips": maximum_losing_streak,
        "actual_fee_evidence_round_trip_count": actual_fee_count,
        "benchmark_evidence_round_trip_count": len(benchmark_excess),
        "total_benchmark_excess_return_bps": round(sum(benchmark_excess), 8),
        "daily_post_cost_pnl_usd": {
            key: round(value, 8) for key, value in sorted(daily.items())
        },
    }


def _gate(
    ready: bool, *, observed: Any, required: Any, comparator: str
) -> dict[str, Any]:
    return {
        "ready": bool(ready),
        "observed": observed,
        "required": required,
        "comparator": comparator,
    }


def _scale_governance_for_capital(
    graduation_policy: Mapping[str, Any], capital_usd: float
) -> dict[str, Any]:
    governance = _mapping(graduation_policy.get("scale_governance"))
    enabled = {
        str(value or "").strip()
        for value in _list(governance.get("enabled_operating_classes"))
        if str(value or "").strip()
    }
    classes = sorted(
        [
            dict(row)
            for row in _list(governance.get("operating_classes"))
            if isinstance(row, Mapping)
        ],
        key=lambda row: float(_number(row.get("maximum_cataloged_capital_usd")) or 0.0),
    )
    selected = next(
        (
            row
            for row in classes
            if float(_number(row.get("maximum_cataloged_capital_usd")) or 0.0) + 1e-9
            >= max(float(capital_usd), 0.0)
        ),
        {},
    )
    operating_class = str(selected.get("operating_class") or "").strip()
    required_controls = _ordered_unique(_list(selected.get("required_controls")))
    control_evidence = _mapping(governance.get("control_evidence"))
    missing_controls = [
        control
        for control in required_controls
        if control_evidence.get(control) is not True
    ]
    return {
        "operating_class": operating_class,
        "operating_class_cataloged": bool(selected),
        "operating_class_enabled": bool(operating_class in enabled),
        "maximum_cataloged_capital_usd": float(
            _number(selected.get("maximum_cataloged_capital_usd")) or 0.0
        ),
        "required_controls": required_controls,
        "missing_controls": missing_controls,
        "required_controls_evidenced": not missing_controls,
        "automatic_operating_class_transition": False,
        "external_investor_capital_allowed": False,
        "third_party_account_management_allowed": False,
    }


def evaluate_live_canary_graduation(
    *,
    canary_plan: Mapping[str, Any],
    graduation_policy: Mapping[str, Any],
    order_intents: Iterable[Mapping[str, Any]],
    order_events: Iterable[Mapping[str, Any]],
    closeout_receipts: Iterable[Mapping[str, Any]],
    ledger_integrity: Mapping[str, Any] | None = None,
    source_errors: Iterable[str] = (),
    current_candidate_id: str = "",
) -> dict[str, Any]:
    blockers = _ordered_unique(source_errors)
    expected_policy_id = str(
        graduation_policy.get("expected_canary_policy_id") or ""
    ).strip()
    if int(graduation_policy.get("schema_version", 0) or 0) != 1:
        blockers.append("live_canary_graduation_policy_schema_invalid")
    if (
        not expected_policy_id
        or str(canary_plan.get("policy_id") or "").strip() != expected_policy_id
    ):
        blockers.append("live_canary_graduation_canary_policy_mismatch")
    integrity = _mapping(ledger_integrity)
    if integrity and integrity.get("ok") is not True:
        blockers.append("live_order_ledger_integrity_invalid")

    intents = {
        str(row.get("intent_id") or "").strip(): dict(row) for row in order_intents
    }
    final_events: dict[str, dict[str, Any]] = {}
    for event in order_events:
        intent_id = str(event.get("intent_id") or "").strip()
        if intent_id:
            final_events[intent_id] = dict(event)
    stages = sorted(
        [
            dict(row)
            for row in _list(canary_plan.get("stages"))
            if isinstance(row, Mapping)
        ],
        key=lambda row: int(row.get("stage", 0) or 0),
    )
    stage_symbols = {
        str(symbol or "").strip().upper()
        for row in stages
        for symbol in _list(row.get("symbols"))
        if str(symbol or "").strip()
    }
    stage_one_symbols = (
        {
            str(symbol or "").strip().upper()
            for symbol in _list(stages[0].get("symbols"))
            if str(symbol or "").strip()
        }
        if stages
        else set()
    )

    usable_receipts: list[dict[str, Any]] = []
    receipt_pending: list[str] = []
    progression_receipt_pending: list[str] = []
    receipt_summaries: list[dict[str, Any]] = []
    for raw_receipt in closeout_receipts:
        receipt = dict(raw_receipt)
        intent_id = str(receipt.get("intent_id") or "").strip()
        validation = _validate_receipt(
            receipt,
            row=intents.get(intent_id),
            final_event=final_events.get(intent_id),
            canary_plan=canary_plan,
            graduation_policy=graduation_policy,
            stage_symbols=stage_symbols,
        )
        blockers.extend(validation["hard_errors"])
        receipt_pending.extend(validation["pending"])
        progression_receipt_pending.extend(validation["progression_pending"])
        receipt_summaries.append(
            {
                "receipt_sha256": str(receipt.get("receipt_sha256") or ""),
                "intent_id_sha256": _sha256_text(intent_id),
                "symbol": str(receipt.get("symbol") or "").upper(),
                "action": str(receipt.get("action") or "").upper(),
                "filled_at_utc": str(receipt.get("filled_at_utc") or ""),
                "usable": bool(validation["usable"]),
                "pending": list(validation["pending"]),
                "progression_pending": list(validation["progression_pending"]),
                "hard_errors": list(validation["hard_errors"]),
            }
        )
        if validation["usable"]:
            receipt["_validation"] = validation
            usable_receipts.append(receipt)

    round_trips, pairing_errors = _pair_round_trips(usable_receipts)
    blockers.extend(pairing_errors)
    blockers = _ordered_unique(blockers)
    receipt_pending = _ordered_unique(receipt_pending)
    progression_receipt_pending = _ordered_unique(progression_receipt_pending)
    metrics = _round_trip_metrics(round_trips)

    first_entries = [
        receipt
        for receipt in usable_receipts
        if str(receipt.get("action") or "").upper() in ENTRY_ACTIONS
        and (
            not bool(
                _mapping(graduation_policy.get("first_canary")).get(
                    "require_stage_one_symbol", True
                )
            )
            or str(receipt.get("symbol") or "").upper() in stage_one_symbols
        )
    ]
    first_canary_reconciled = bool(first_entries)
    progression = _mapping(graduation_policy.get("stage_progression"))
    count = int(metrics["reconciled_round_trip_count"])
    days = int(metrics["independent_trading_day_count"])
    regimes = int(metrics["distinct_regime_bucket_count"])
    positive_day_ratio = float(metrics["positive_trading_day_ratio"])
    total_post_cost = float(metrics["total_post_cost_pnl_usd"])
    mean_post_cost_return = float(metrics["mean_post_cost_return_bps"])
    fill_deviation = float(metrics["maximum_fill_deviation_bps"])
    daily_loss = float(metrics["maximum_daily_loss_usd"])
    drawdown = float(metrics["maximum_cumulative_drawdown_usd"])
    losing_streak = int(metrics["maximum_consecutive_losing_round_trips"])
    required_days = int(progression.get("minimum_independent_trading_days", 0) or 0)
    required_regimes = int(progression.get("minimum_distinct_regime_buckets", 0) or 0)
    required_positive_ratio = float(
        progression.get("minimum_positive_day_ratio", 0.0) or 0.0
    )
    minimum_total = float(
        progression.get("minimum_total_post_cost_pnl_usd_exclusive", 0.0) or 0.0
    )
    minimum_mean = float(
        progression.get("minimum_mean_post_cost_return_bps_exclusive", 0.0) or 0.0
    )
    maximum_fill = float(progression.get("maximum_fill_deviation_bps", 0.0) or 0.0)
    maximum_daily_loss = float(progression.get("maximum_daily_loss_usd", 0.0) or 0.0)
    maximum_drawdown = float(
        progression.get("maximum_cumulative_drawdown_usd", 0.0) or 0.0
    )
    maximum_losing_streak = int(
        progression.get("maximum_consecutive_losing_round_trips", 0) or 0
    )
    economic_gates = {
        "independent_trading_days": _gate(
            days >= required_days,
            observed=days,
            required=required_days,
            comparator=">=",
        ),
        "regime_diversity": _gate(
            regimes >= required_regimes,
            observed=regimes,
            required=required_regimes,
            comparator=">=",
        ),
        "positive_day_ratio": _gate(
            positive_day_ratio >= required_positive_ratio,
            observed=positive_day_ratio,
            required=required_positive_ratio,
            comparator=">=",
        ),
        "positive_total_post_cost_pnl": _gate(
            total_post_cost > minimum_total,
            observed=total_post_cost,
            required=minimum_total,
            comparator=">",
        ),
        "positive_mean_post_cost_return": _gate(
            mean_post_cost_return > minimum_mean,
            observed=mean_post_cost_return,
            required=minimum_mean,
            comparator=">",
        ),
        "fill_model_fidelity": _gate(
            maximum_fill <= 0.0 or fill_deviation <= maximum_fill,
            observed=fill_deviation,
            required=maximum_fill,
            comparator="<=",
        ),
        "daily_loss_control": _gate(
            maximum_daily_loss <= 0.0 or daily_loss <= maximum_daily_loss,
            observed=daily_loss,
            required=maximum_daily_loss,
            comparator="<=",
        ),
        "drawdown_control": _gate(
            maximum_drawdown <= 0.0 or drawdown <= maximum_drawdown,
            observed=drawdown,
            required=maximum_drawdown,
            comparator="<=",
        ),
        "loss_streak_control": _gate(
            maximum_losing_streak <= 0 or losing_streak <= maximum_losing_streak,
            observed=losing_streak,
            required=maximum_losing_streak,
            comparator="<=",
        ),
    }
    economic_ready = all(
        bool(gate.get("ready", False)) for gate in economic_gates.values()
    )

    highest_completed_stage = 0
    stage_evaluations: list[dict[str, Any]] = []
    for stage in stages:
        stage_number = int(stage.get("stage", 0) or 0)
        minimum_round_trips = int(
            stage.get("minimum_successful_reconciled_round_trips", 0) or 0
        )
        ready = bool(
            first_canary_reconciled
            and count >= minimum_round_trips
            and economic_ready
            and not blockers
        )
        if ready and stage_number == highest_completed_stage + 1:
            highest_completed_stage = stage_number
        stage_evaluations.append(
            {
                "stage": stage_number,
                "symbols": [str(item).upper() for item in _list(stage.get("symbols"))],
                "minimum_reconciled_round_trips": minimum_round_trips,
                "round_trip_floor_met": count >= minimum_round_trips,
                "economic_validation_met": economic_ready,
                "completed": ready,
                "automatic_progression": False,
            }
        )

    ladder = [
        dict(row)
        for row in _list(graduation_policy.get("capital_ladder"))
        if isinstance(row, Mapping)
    ]
    ladder_names = [str(row.get("tier") or "").strip() for row in ladder]
    ladder_capitals = [
        float(_number(row.get("account_capital_usd")) or 0.0) for row in ladder
    ]
    if (
        not ladder
        or any(not name for name in ladder_names)
        or len(ladder_names) != len(set(ladder_names))
        or any(capital <= 0.0 for capital in ladder_capitals)
        or ladder_capitals != sorted(set(ladder_capitals))
    ):
        blockers.append("capital_ladder_policy_invalid")
    uncataloged_capitals = [
        capital
        for capital in ladder_capitals
        if not _scale_governance_for_capital(graduation_policy, capital).get(
            "operating_class_cataloged", False
        )
    ]
    if uncataloged_capitals:
        blockers.append("capital_ladder_operating_class_coverage_invalid")
    ladder_evaluations: list[dict[str, Any]] = []
    lcb = _number(metrics.get("normal_approx_lcb_95_post_cost_return_bps"))
    actual_fee_count = int(metrics["actual_fee_evidence_round_trip_count"])
    benchmark_count = int(metrics["benchmark_evidence_round_trip_count"])
    benchmark_excess = float(metrics["total_benchmark_excess_return_bps"])
    for index, tier in enumerate(ladder):
        minimum_round_trips = int(tier.get("minimum_reconciled_round_trips", 0) or 0)
        minimum_days = int(tier.get("minimum_independent_trading_days", 0) or 0)
        minimum_regimes = int(tier.get("minimum_distinct_regime_buckets", 0) or 0)
        require_lcb = bool(tier.get("require_positive_normal_approx_lcb_95", False))
        require_benchmark = bool(tier.get("require_positive_benchmark_excess", False))
        require_actual_fees = bool(tier.get("require_actual_fee_evidence", False))
        tier_capital = float(tier.get("account_capital_usd", 0.0) or 0.0)
        scale_governance = _scale_governance_for_capital(
            graduation_policy, tier_capital
        )
        requirements = {
            "round_trips": count >= minimum_round_trips,
            "independent_days": days >= minimum_days,
            "regime_buckets": regimes >= minimum_regimes,
            "economic_validation": economic_ready if index > 0 else True,
            "positive_lcb_95": (not require_lcb) or (lcb is not None and lcb > 0.0),
            "positive_benchmark_excess": (not require_benchmark)
            or (benchmark_count >= minimum_round_trips and benchmark_excess > 0.0),
            "actual_fee_evidence": (not require_actual_fees)
            or actual_fee_count >= minimum_round_trips,
            "no_control_or_safety_blockers": not blockers,
            "operating_class_cataloged": bool(
                scale_governance.get("operating_class_cataloged", False)
            ),
            "operating_class_enabled": bool(
                scale_governance.get("operating_class_enabled", False)
            ),
            "required_scale_controls_evidenced": bool(
                scale_governance.get("required_controls_evidenced", False)
            ),
        }
        review_eligible = all(requirements.values())
        ladder_evaluations.append(
            {
                "tier": str(tier.get("tier") or f"tier_{index}"),
                "proposed_limits": {
                    "account_capital_usd": tier_capital,
                    "max_order_notional_usd": float(
                        tier.get("max_order_notional_usd", 0.0) or 0.0
                    ),
                    "max_order_quantity": float(
                        tier.get("max_order_quantity", 0.0) or 0.0
                    ),
                },
                "policy_requirements": {
                    "minimum_reconciled_round_trips": minimum_round_trips,
                    "minimum_independent_trading_days": minimum_days,
                    "minimum_distinct_regime_buckets": minimum_regimes,
                    "require_positive_normal_approx_lcb_95": require_lcb,
                    "require_positive_benchmark_excess": require_benchmark,
                    "require_actual_fee_evidence": require_actual_fees,
                },
                "scale_governance": scale_governance,
                "requirements": requirements,
                "operator_review_eligible": review_eligible,
                "limits_applied": index == 0,
                "automatic_scaling": False,
            }
        )

    pending_evidence: list[str] = list(receipt_pending)
    filled_intent_count = sum(
        1 for row in intents.values() if str(row.get("state") or "") == "filled"
    )
    if not first_canary_reconciled:
        pending_evidence.append(
            "first_canary_closeout_receipt_pending"
            if filled_intent_count > 0
            else "first_live_canary_not_run"
        )
    first_stage_floor = (
        int(stages[0].get("minimum_successful_reconciled_round_trips", 0) or 0)
        if stages
        else 0
    )
    if first_canary_reconciled and count < first_stage_floor:
        pending_evidence.append("three_fully_reconciled_round_trips_pending")
    if count > 0:
        pending_evidence.extend(progression_receipt_pending)
        pending_evidence.extend(
            f"{name}_pending"
            for name, gate in economic_gates.items()
            if not gate["ready"]
        )
    pending_evidence = _ordered_unique(pending_evidence)

    maximum_stage = max([int(row.get("stage", 0) or 0) for row in stages] or [0])
    next_stage = (
        highest_completed_stage + 1
        if highest_completed_stage < maximum_stage and highest_completed_stage > 0
        else 1 if highest_completed_stage == 0 else None
    )
    if blockers:
        phase = "blocked"
        overall_status = "blocked"
    elif not first_canary_reconciled:
        phase = (
            "first_canary_reconciliation"
            if filled_intent_count > 0 or receipt_summaries
            else "awaiting_first_canary"
        )
        overall_status = "ready_idle"
    elif highest_completed_stage == 0:
        phase = "round_trip_and_economic_validation"
        overall_status = "collecting_evidence"
    elif highest_completed_stage < maximum_stage:
        phase = f"stage_{highest_completed_stage + 1}_operator_review"
        overall_status = "operator_review_required"
    else:
        phase = "capital_ladder_observation"
        overall_status = "operator_review_required"

    receipt_basis = {
        "policy_id": str(graduation_policy.get("policy_id") or ""),
        "canary_policy_id": str(canary_plan.get("policy_id") or ""),
        "candidate_id": str(current_candidate_id or ""),
        "account_policy_key": str(canary_plan.get("account_policy_key") or ""),
        "execution_route_id": str(canary_plan.get("execution_route_id") or ""),
        "ledger_event_chain_head": str(
            _mapping(integrity.get("event_chain")).get("chain_head") or ""
        ),
        "closeout_receipt_sha256": sorted(
            str(row.get("receipt_sha256") or "") for row in usable_receipts
        ),
        "metrics": metrics,
        "highest_completed_stage": highest_completed_stage,
    }
    return {
        "schema_version": 1,
        "policy_id": str(graduation_policy.get("policy_id") or ""),
        "control_ok": not blockers,
        "overall_status": overall_status,
        "phase": phase,
        "identity": {
            "candidate_id": str(current_candidate_id or ""),
            "account_policy_key": str(canary_plan.get("account_policy_key") or ""),
            "execution_route_id": str(canary_plan.get("execution_route_id") or ""),
        },
        "first_canary": {
            "reconciled": first_canary_reconciled,
            "reconciled_entry_fill_count": len(first_entries),
            "automatic_follow_on_order_allowed": False,
        },
        "metrics": metrics,
        "economic_gates": economic_gates,
        "economic_validation_ready": economic_ready,
        "stage_progression": {
            "highest_completed_stage": highest_completed_stage,
            "next_stage_requiring_operator_review": next_stage,
            "evaluations": stage_evaluations,
            "automatic_stage_progression": False,
        },
        "capital_ladder": {
            "active_policy_tier": (
                str(ladder[0].get("tier") or "micro_validation") if ladder else ""
            ),
            "evaluations": ladder_evaluations,
            "cataloged_tier_count": len(ladder_evaluations),
            "maximum_cataloged_capital_usd": max(ladder_capitals, default=0.0),
            "enabled_operating_classes": sorted(
                {
                    str(value or "").strip()
                    for value in _list(
                        _mapping(graduation_policy.get("scale_governance")).get(
                            "enabled_operating_classes"
                        )
                    )
                    if str(value or "").strip()
                }
            ),
            "automatic_scaling": False,
            "loss_chasing": False,
            "martingale": False,
            "policy_commit_required_for_any_limit_change": True,
        },
        "receipts": {
            "source_count": len(receipt_summaries),
            "usable_count": len(usable_receipts),
            "filled_live_order_intent_count": filled_intent_count,
            "summaries": receipt_summaries,
        },
        "pending_evidence": pending_evidence,
        "blockers": blockers,
        "graduation_receipt_sha256": canonical_payload_sha256(receipt_basis),
        "live_execution_authority": False,
        "stage_progression_authority": False,
        "capital_scaling_authority": False,
        "operator_review_required": True,
        "contract": {
            "successful_trade_alone_cannot_graduate": True,
            "profitability_is_not_guaranteed": True,
            "missing_earned_evidence_is_pending_not_system_degradation": True,
            "ambiguous_or_tampered_evidence_fails_closed": True,
            "new_release_attestation_and_allowlist_required_after_review": True,
        },
    }
