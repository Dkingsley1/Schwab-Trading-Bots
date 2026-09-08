from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


_MODE_SPECIFIC_KEYS = {
    "account_reference",
    "broker_order_id",
    "created_at",
    "created_at_utc",
    "execution_gateway",
    "latency_ms",
    "mode",
    "order_id",
    "paper_order",
    "result",
    "result_status",
    "source_mode",
    "status_code",
    "target_mode",
    "timestamp_utc",
    "updated_at_utc",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _MODE_SPECIFIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unwrap(row: Mapping[str, Any]) -> dict[str, Any]:
    intent = row.get("intent")
    return dict(intent) if isinstance(intent, Mapping) else dict(row)


def canonical_intent(row: Mapping[str, Any]) -> dict[str, Any]:
    intent = _unwrap(row)
    metadata = _mapping(intent.get("metadata"))
    specialization = _mapping(metadata.get("strategy_specialization"))
    receipt = _mapping(specialization.get("contract_receipt"))
    canonical = {
        "symbol": str(intent.get("symbol") or "").strip().upper(),
        "action": str(intent.get("action") or "HOLD").strip().upper(),
        "quantity": round(float(intent.get("quantity") or 0.0), 10),
        "asset_type": str(
            intent.get("asset_type") or metadata.get("asset_class") or "EQUITY"
        )
        .strip()
        .upper(),
        "strategy_id": str(
            specialization.get("selected_strategy_id")
            or receipt.get("strategy_id")
            or intent.get("strategy")
            or ""
        ),
        "order_spec": _canonical(intent.get("order_spec") or {}),
        "risk_reducing_exit": bool(intent.get("risk_reducing_exit", False)),
        "candidate_id": str(
            metadata.get("production_candidate_id") or receipt.get("candidate_id") or ""
        ),
    }
    return canonical


def comparison_key(row: Mapping[str, Any]) -> str:
    intent = _unwrap(row)
    trace = _mapping(intent.get("trace_context")) or _mapping(row.get("trace_context"))
    for value in (
        trace.get("trace_id"),
        intent.get("intent_id"),
        row.get("intent_message_id"),
        row.get("source_intent_message_id"),
        row.get("parent_message_id"),
        intent.get("message_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return f"semantic_{_hash(canonical_intent(row))[:24]}"


def compare_pair(paper: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any]:
    paper_intent = canonical_intent(paper)
    live_intent = canonical_intent(live)
    differences = [
        key
        for key in sorted(set(paper_intent) | set(live_intent))
        if paper_intent.get(key) != live_intent.get(key)
    ]
    return {
        "ok": not differences,
        "comparison_key": comparison_key(paper),
        "paper_intent_sha256": _hash(paper_intent),
        "live_intent_sha256": _hash(live_intent),
        "differences": differences,
        "allowed_mode_differences": [
            "broker_order_id",
            "fill_price",
            "filled_quantity",
            "latency_ms",
            "fees",
            "slippage",
            "venue_status",
        ],
    }


def compare_record_sets(
    paper_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
    *,
    require_live_for_every_paper: bool = False,
) -> dict[str, Any]:
    paper_index = {comparison_key(row): row for row in paper_rows}
    live_index = {comparison_key(row): row for row in live_rows}
    paired_keys = sorted(set(paper_index) & set(live_index))
    pairs = [compare_pair(paper_index[key], live_index[key]) for key in paired_keys]
    mismatches = [row for row in pairs if not row["ok"]]
    unpaired_paper = sorted(set(paper_index) - set(live_index))
    missing_live = unpaired_paper if require_live_for_every_paper else []
    missing_paper = sorted(set(live_index) - set(paper_index))
    empirical_ready = bool(pairs and not mismatches and not missing_paper)
    if not live_rows:
        status = "awaiting_live_shadow_samples"
    elif mismatches:
        status = "semantic_mismatch"
    elif missing_paper:
        status = "orphan_live_shadow_intent"
    elif require_live_for_every_paper and missing_live:
        status = "partial_shadow_coverage"
    else:
        status = "equivalent"
    return {
        "ok": not mismatches and not missing_paper,
        "structural_ready": True,
        "empirical_ready": empirical_ready,
        "status": status,
        "paper_count": len(paper_rows),
        "live_count": len(live_rows),
        "paired_count": len(pairs),
        "mismatch_count": len(mismatches),
        "missing_live_count": len(missing_live),
        "unpaired_paper_count": len(unpaired_paper),
        "missing_paper_count": len(missing_paper),
        "missing_live_keys": missing_live[:100],
        "missing_paper_keys": missing_paper[:100],
        "pairs": pairs,
    }
