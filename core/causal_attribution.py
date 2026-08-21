from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


TRACE_SCHEMA_VERSION = 1
STAGE_ORDER = (
    "source",
    "feature",
    "signal",
    "sizing",
    "risk",
    "execution",
    "cost",
    "outcome",
)
_VOLATILE_KEYS = {
    "timestamp",
    "timestamp_utc",
    "created_at",
    "created_at_utc",
    "updated_at",
    "updated_at_utc",
    "latency_ms",
    "broker_order_id",
    "order_id",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def ensure_trace_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = _mapping(payload.get("trace_context"))
    if str(existing.get("trace_id") or "").startswith("trace_"):
        return existing
    metadata = _mapping(payload.get("metadata"))
    specialization = _mapping(metadata.get("strategy_specialization"))
    seed = {
        "message_id": payload.get("message_id"),
        "intent_id": payload.get("intent_id") or metadata.get("decision_intent_id"),
        "run_id": payload.get("run_id"),
        "iter_id": payload.get("iter_id"),
        "symbol": str(payload.get("symbol") or "").upper(),
        "action": str(payload.get("action") or "HOLD").upper(),
        "quantity": payload.get("quantity"),
        "strategy_id": specialization.get("selected_strategy_id")
        or payload.get("strategy"),
        "source_profile": metadata.get("source_profile"),
    }
    seed_hash = canonical_sha256(seed)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": f"trace_{seed_hash[:24]}",
        "root_seed_sha256": seed_hash,
        "parent_trace_id": str(payload.get("parent_trace_id") or ""),
    }


def _stage(trace_id: str, name: str, payload: Any, parent_hash: str) -> dict[str, Any]:
    payload_hash = canonical_sha256(payload)
    material = {
        "trace_id": trace_id,
        "stage": name,
        "parent_stage_hash": parent_hash,
        "payload_sha256": payload_hash,
    }
    return {**material, "stage_hash": canonical_sha256(material)}


def _first_number(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> float | None:
    for row in rows:
        for key in keys:
            value = row.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def build_attribution(
    *,
    intent: Mapping[str, Any],
    result: Mapping[str, Any],
    gateway: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    features = _mapping(intent.get("features"))
    metadata = _mapping(intent.get("metadata"))
    paper_order = _mapping(result.get("paper_order"))
    execution = _mapping(result.get("execution"))
    sources = [paper_order, execution, dict(result), features, metadata]
    expected_edge_bps = _first_number(
        sources, ("expected_edge_bps", "expected_post_cost_edge_bps", "gross_edge_bps")
    )
    realized_gross_bps = _first_number(
        sources, ("realized_gross_return_bps", "gross_return_bps", "gross_pnl_bps")
    )
    fee_bps = _first_number(sources, ("fee_bps", "fees_bps", "commission_bps"))
    slippage_bps = _first_number(sources, ("slippage_bps", "realized_slippage_bps"))
    impact_bps = _first_number(sources, ("market_impact_bps", "impact_bps"))
    realized_net_bps = _first_number(
        sources, ("post_cost_return_bps", "realized_net_return_bps", "net_return_bps")
    )
    known_costs = [
        value for value in (fee_bps, slippage_bps, impact_bps) if value is not None
    ]
    modeled_cost_bps = sum(known_costs) if known_costs else None
    residual_bps = None
    if (
        realized_gross_bps is not None
        and modeled_cost_bps is not None
        and realized_net_bps is not None
    ):
        residual_bps = realized_net_bps - (realized_gross_bps - modeled_cost_bps)
    values = {
        "expected_edge_bps": expected_edge_bps,
        "realized_gross_bps": realized_gross_bps,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "market_impact_bps": impact_bps,
        "modeled_cost_bps": modeled_cost_bps,
        "realized_net_bps": realized_net_bps,
        "attribution_residual_bps": residual_bps,
    }
    return {
        "schema_version": 1,
        "symbol": str(intent.get("symbol") or "").upper(),
        "action": str(intent.get("action") or "HOLD").upper(),
        "requested_quantity": _first_number([dict(intent)], ("quantity",)),
        "filled_quantity": _first_number(
            sources, ("filled_quantity", "quantity_filled")
        ),
        "risk_allow_execute": _mapping(gateway).get("allow_execute"),
        "risk_reasons": list(_mapping(gateway).get("reasons") or []),
        "values": values,
        "observed_fields": sorted(
            key for key, value in values.items() if value is not None
        ),
        "missing_fields": sorted(key for key, value in values.items() if value is None),
        "no_fabricated_defaults": True,
    }


def build_execution_trace(
    *,
    intent: Mapping[str, Any],
    result: Mapping[str, Any],
    gateway: Mapping[str, Any] | None = None,
    mode: str = "paper",
) -> dict[str, Any]:
    context = ensure_trace_context(intent)
    trace_id = str(context["trace_id"])
    metadata = _mapping(intent.get("metadata"))
    features = _mapping(intent.get("features"))
    specialization = _mapping(metadata.get("strategy_specialization"))
    attribution = build_attribution(intent=intent, result=result, gateway=gateway)
    payloads = {
        "source": {
            "source_broker": metadata.get("source_broker"),
            "source_profile": metadata.get("source_profile"),
            "transport_receipt_sha256": metadata.get("transport_receipt_sha256"),
            "source_receipt_sha256": metadata.get("source_receipt_sha256"),
        },
        "feature": {
            "feature_count": len(features),
            "features_sha256": canonical_sha256(features),
        },
        "signal": {
            "action": intent.get("action"),
            "model_score": intent.get("model_score"),
            "threshold": intent.get("threshold"),
            "strategy_id": specialization.get("selected_strategy_id")
            or intent.get("strategy"),
        },
        "sizing": {
            "quantity": intent.get("quantity"),
            "allocation": metadata.get("allocation"),
        },
        "risk": dict(gateway or {}),
        "execution": {
            "mode": str(mode),
            "status": result.get("status"),
            "paper_realism_status": _mapping(result.get("paper_order")).get(
                "paper_realism_status"
            ),
        },
        "cost": attribution["values"],
        "outcome": {
            "result_status": result.get("status"),
            "filled_quantity": attribution.get("filled_quantity"),
            "realized_net_bps": attribution["values"].get("realized_net_bps"),
        },
    }
    stages: list[dict[str, Any]] = []
    parent_hash = ""
    for name in STAGE_ORDER:
        stage = _stage(trace_id, name, payloads[name], parent_hash)
        stages.append(stage)
        parent_hash = str(stage["stage_hash"])
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_context": context,
        "stage_count": len(stages),
        "stages": stages,
        "chain_head_sha256": parent_hash,
        "attribution": attribution,
    }


def verify_execution_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    stages = [
        dict(row) for row in trace.get("stages") or [] if isinstance(row, Mapping)
    ]
    trace_id = str(_mapping(trace.get("trace_context")).get("trace_id") or "")
    parent = ""
    observed_names: list[str] = []
    for index, row in enumerate(stages):
        observed_names.append(str(row.get("stage") or ""))
        material = {
            "trace_id": trace_id,
            "stage": row.get("stage"),
            "parent_stage_hash": row.get("parent_stage_hash"),
            "payload_sha256": row.get("payload_sha256"),
        }
        if str(row.get("parent_stage_hash") or "") != parent:
            errors.append(f"parent_hash_mismatch:stage={index}")
        if str(row.get("stage_hash") or "") != canonical_sha256(material):
            errors.append(f"stage_hash_mismatch:stage={index}")
        parent = str(row.get("stage_hash") or "")
    if tuple(observed_names) != STAGE_ORDER:
        errors.append("stage_order_or_coverage_invalid")
    if str(trace.get("chain_head_sha256") or "") != parent:
        errors.append("chain_head_mismatch")
    return {"ok": not errors, "errors": errors, "stage_count": len(stages)}
