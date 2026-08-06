from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


INTENT_SCHEMA_VERSION = 1


def _normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else None
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    return round(numeric, 12) if math.isfinite(numeric) else None


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _normalized(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_quote_snapshot(features: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    meta = metadata or {}

    def first(*keys: str, default: Any = 0.0) -> Any:
        for key in keys:
            if key in features and features.get(key) is not None and features.get(key) != "":
                return features.get(key)
            if key in meta and meta.get(key) is not None and meta.get(key) != "":
                return meta.get(key)
        return default

    return _normalized(
        {
            "timestamp_utc": first(
                "quote_timestamp_utc",
                "snapshot_timestamp_utc",
                "market_timestamp_utc",
                default="",
            ),
            "last_price": first("last_price", "mark_price", "reference_price"),
            "bid_price": first("bid_price", "bid"),
            "ask_price": first("ask_price", "ask"),
            "spread_bps": first("spread_bps", "model_spread_bps"),
            "quote_age_ms": first("quote_age_ms", "market_data_age_ms"),
            "source_quality_norm": first("news_source_quality_norm", "source_quality_norm"),
            "tradeability_norm": first("market_micro_tradeability_score_norm", "tradeability_score"),
            "source_provider": first("source_provider", default=""),
            "source_venue": first("source_venue", default=""),
            "snapshot_id": first("snapshot_id", default=""),
        }
    )


def compact_expected_fill(expected_fill: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = expected_fill or {}
    keys = (
        "expected_fill_price",
        "expected_slippage_bps",
        "partial_fill_ratio",
        "impact_bps",
        "spread_jump_penalty_bps",
        "symbol_curve_multiplier",
        "fill_quality_bucket",
        "paper_execution_status",
    )
    return _normalized({key: payload.get(key, "" if key.endswith("bucket") or key.endswith("status") else 0.0) for key in keys})


def compact_risk_decision(risk_decision: Mapping[str, Any] | None) -> dict[str, Any]:
    decision = risk_decision or {}
    details = decision.get("details") if isinstance(decision.get("details"), Mapping) else {}
    stable_detail_keys = (
        "position_qty",
        "projected_position_qty",
        "order_notional",
        "projected_order_notional",
        "daily_realized_pnl",
        "open_orders_total",
        "open_orders_symbol",
        "reference_price",
        "intended_price",
        "notional_multiplier",
    )
    return _normalized(
        {
            "ok": bool(decision.get("ok", False)),
            "gate": str(decision.get("gate") or "not_evaluated"),
            "reason": str(decision.get("reason") or "not_evaluated"),
            "limits": {key: details.get(key) for key in stable_detail_keys if key in details},
        }
    )


def build_order_intent_evidence(
    *,
    decision_id: str,
    symbol: str,
    action: str,
    quantity: float,
    strategy: str,
    asset_type: str = "EQUITY",
    limit_price: float = 0.0,
    quote_snapshot: Mapping[str, Any] | None = None,
    expected_fill: Mapping[str, Any] | None = None,
    risk_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    semantic_order = _normalized(
        {
            "decision_id": str(decision_id or "").strip(),
            "symbol": str(symbol or "").strip().upper(),
            "action": str(action or "").strip().upper(),
            "quantity": max(float(quantity or 0.0), 0.0),
            "strategy": str(strategy or "default").strip(),
            "asset_type": str(asset_type or "EQUITY").strip().upper(),
            "limit_price": max(float(limit_price or 0.0), 0.0),
        }
    )
    quote = _normalized(dict(quote_snapshot or {}))
    fill = compact_expected_fill(expected_fill)
    risk = compact_risk_decision(risk_decision)
    component_hashes = {
        "semantic_order_sha256": canonical_payload_sha256(semantic_order),
        "quote_snapshot_sha256": canonical_payload_sha256(quote),
        "expected_fill_sha256": canonical_payload_sha256(fill),
        "risk_decision_sha256": canonical_payload_sha256(risk),
    }
    immutable_payload = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "semantic_order": semantic_order,
        "quote_snapshot": quote,
        "expected_fill": fill,
        "risk_decision": risk,
        "component_hashes": component_hashes,
    }
    return {
        **immutable_payload,
        "intent_sha256": canonical_payload_sha256(immutable_payload),
        "adapter_excluded_from_hash": True,
        "mode_excluded_from_hash": True,
    }


def verify_order_intent_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    semantic_order = evidence.get("semantic_order") if isinstance(evidence.get("semantic_order"), Mapping) else {}
    quote = evidence.get("quote_snapshot") if isinstance(evidence.get("quote_snapshot"), Mapping) else {}
    fill = evidence.get("expected_fill") if isinstance(evidence.get("expected_fill"), Mapping) else {}
    risk = evidence.get("risk_decision") if isinstance(evidence.get("risk_decision"), Mapping) else {}
    expected_components = {
        "semantic_order_sha256": canonical_payload_sha256(semantic_order),
        "quote_snapshot_sha256": canonical_payload_sha256(quote),
        "expected_fill_sha256": canonical_payload_sha256(fill),
        "risk_decision_sha256": canonical_payload_sha256(risk),
    }
    supplied_components = evidence.get("component_hashes") if isinstance(evidence.get("component_hashes"), Mapping) else {}
    immutable_payload = {
        "schema_version": int(evidence.get("schema_version", 0) or 0),
        "semantic_order": _normalized(semantic_order),
        "quote_snapshot": _normalized(quote),
        "expected_fill": _normalized(fill),
        "risk_decision": _normalized(risk),
        "component_hashes": expected_components,
    }
    expected_intent = canonical_payload_sha256(immutable_payload)
    errors = [
        key
        for key, value in expected_components.items()
        if str(supplied_components.get(key) or "") != value
    ]
    if str(evidence.get("intent_sha256") or "") != expected_intent:
        errors.append("intent_sha256")
    if int(evidence.get("schema_version", 0) or 0) != INTENT_SCHEMA_VERSION:
        errors.append("schema_version")
    return {
        "ok": not errors,
        "errors": errors,
        "expected_intent_sha256": expected_intent,
        "intent_sha256": str(evidence.get("intent_sha256") or ""),
    }
