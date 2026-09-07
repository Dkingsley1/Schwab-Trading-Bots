from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

EXECUTION_CONTRACT_VERSION = "execution_intent_contract_v1"
EXECUTION_IDENTITY_SCHEMA_VERSION = 1
TRADE_ACTIONS = frozenset(
    {
        "BUY",
        "SELL",
        "SELL_SHORT",
        "BUY_TO_COVER",
        "BUY_TO_OPEN",
        "BUY_TO_CLOSE",
        "SELL_TO_OPEN",
        "SELL_TO_CLOSE",
        "CLOSE",
        "ROLL",
    }
)


def _text(value: Any, default: str = "") -> str:
    value_text = str(value or "").strip()
    return value_text or default


def _metadata(intent: Dict[str, Any]) -> Dict[str, Any]:
    raw = intent.get("metadata")
    return dict(raw) if isinstance(raw, dict) else {}


def _number_text(value: Any) -> str:
    try:
        return format(float(value), ".12g")
    except (TypeError, ValueError):
        return "0"


def _json_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = b"null"
    return hashlib.sha256(encoded).hexdigest()


def _normalize_asset_class(value: Any) -> str:
    text = _text(value).lower()
    aliases = {
        "stock": "equity",
        "stocks": "equity",
        "equities": "equity",
        "option": "options",
        "future": "futures",
        "cryptocurrency": "crypto",
        "spot_crypto": "crypto",
    }
    return aliases.get(text, text)


def _infer_asset_class(intent: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    declared = _normalize_asset_class(
        metadata.get("asset_class")
        or metadata.get("asset_type")
        or intent.get("asset_class")
        or intent.get("asset_type")
    )
    if declared:
        return declared
    symbol = _text(intent.get("symbol")).upper()
    if metadata.get("options_plan") or any(
        token in symbol for token in (" CALL ", " PUT ")
    ):
        return "options"
    if metadata.get("futures_plan") or symbol.startswith("/"):
        return "futures"
    if symbol.endswith("-USD") or symbol.endswith("-USDC"):
        return "crypto"
    return "equity" if symbol else "unknown"


def _actor_identity(
    intent: Dict[str, Any], metadata: Dict[str, Any]
) -> tuple[str, str, str]:
    strategy = _text(intent.get("strategy") or metadata.get("strategy"))
    strategy_lower = strategy.lower()
    layer = _text(metadata.get("layer") or intent.get("layer")).lower()
    intent_kind = _text(
        intent.get("intent_kind") or metadata.get("intent_kind"), "master"
    ).lower()
    explicit_bot_id = _text(metadata.get("bot_id") or intent.get("bot_id"))

    if layer == "paper_portfolio_consensus" or strategy_lower.startswith(
        "paper_portfolio_consensus"
    ):
        return (
            "portfolio_consensus",
            "paper_portfolio_consensus",
            "constituent_manifest_bound",
        )

    master_layers = {
        "grand_master",
        "master",
        "options_master",
        "futures_master",
        "portfolio_master",
    }
    master_kinds = {
        "master",
        "grand_master",
        "options_master",
        "futures_master",
        "master_options_plan",
        "master_futures_plan",
    }
    if (
        layer in master_layers
        or strategy_lower
        in {"grand_master_bot", "options_master_bot", "futures_master_bot"}
        or (
            intent_kind in master_kinds
            and not explicit_bot_id
            and not strategy_lower.startswith("paper_mirror::")
        )
    ):
        return (
            "hierarchical_master",
            strategy or layer or intent_kind or "hierarchical_master",
            "orchestration_only",
        )

    if not explicit_bot_id and strategy_lower.startswith("paper_mirror::"):
        explicit_bot_id = strategy.split("::", 1)[1].strip()
    if explicit_bot_id:
        return "registered_bot", explicit_bot_id, "registry_bound"

    return "unresolved", strategy or "unresolved", "unbound"


def _canonical_fields(intent: Dict[str, Any]) -> Dict[str, str]:
    metadata = _metadata(intent)
    actor_type, actor_id, authority_model = _actor_identity(intent, metadata)
    source_broker = _text(
        metadata.get("source_broker")
        or intent.get("source_broker")
        or metadata.get("broker")
        or intent.get("broker")
        or metadata.get("provider")
        or intent.get("provider"),
        "unknown",
    ).lower()
    source_profile = _text(
        metadata.get("source_profile")
        or intent.get("source_profile")
        or metadata.get("profile")
        or intent.get("profile")
        or metadata.get("sleeve")
        or intent.get("sleeve"),
        "default",
    ).lower()
    shadow_domain = _text(
        metadata.get("shadow_domain")
        or intent.get("shadow_domain")
        or metadata.get("domain")
        or intent.get("domain"),
        "market_execution",
    ).lower()
    runtime_lane = _text(
        metadata.get("runtime_lane")
        or intent.get("runtime_lane")
        or metadata.get("routing_lane")
        or intent.get("routing_lane")
        or metadata.get("lane")
        or intent.get("lane"),
        source_profile or "default",
    ).lower()
    return {
        "actor_type": actor_type,
        "actor_id": actor_id,
        "authority_model": authority_model,
        "intent_kind": _text(
            intent.get("intent_kind") or metadata.get("intent_kind"), "master"
        ).lower(),
        "strategy": _text(intent.get("strategy") or metadata.get("strategy")),
        "symbol": _text(intent.get("symbol")).upper(),
        "action": _text(intent.get("action") or intent.get("side")).upper(),
        "quantity": _number_text(intent.get("quantity")),
        "model_score": _number_text(intent.get("model_score")),
        "threshold": _number_text(intent.get("threshold")),
        "features_sha256": _json_hash(
            intent.get("features") if isinstance(intent.get("features"), dict) else {}
        ),
        "gates_sha256": _json_hash(
            intent.get("gates") if isinstance(intent.get("gates"), dict) else {}
        ),
        "reasons_sha256": _json_hash(
            intent.get("reasons") if isinstance(intent.get("reasons"), list) else []
        ),
        "source_features_sha256": _text(
            (intent.get("execution_transport") or {}).get("source_features_sha256")
            if isinstance(intent.get("execution_transport"), dict)
            else ""
        ),
        "source_mode": _text(intent.get("source_mode"), "shadow").lower(),
        "target_mode": _text(intent.get("target_mode"), "paper").lower(),
        "source_broker": source_broker,
        "source_profile": source_profile,
        "shadow_domain": shadow_domain,
        "runtime_lane": runtime_lane,
        "asset_class": _infer_asset_class(intent, metadata),
        "production_candidate_id": _text(
            metadata.get("production_candidate_id")
            or intent.get("production_candidate_id")
            or metadata.get("candidate_id")
            or intent.get("candidate_id")
        ),
        "message_id": _text(intent.get("message_id")),
        "parent_message_id": _text(
            intent.get("parent_message_id") or metadata.get("parent_message_id")
        ),
    }


def _receipt(fields: Dict[str, str]) -> str:
    encoded = json.dumps(
        fields,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_execution_intent(
    intent: Dict[str, Any],
    *,
    force_reseal: bool = False,
) -> Dict[str, Any]:
    """Attach one canonical identity to an execution intent without changing its economics."""

    row = dict(intent or {})
    metadata = _metadata(row)
    fields = _canonical_fields(row)
    existing = metadata.get("execution_identity")
    has_existing = isinstance(existing, dict) and bool(existing)
    if force_reseal or not has_existing:
        identity: Dict[str, Any] = {
            "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
            "contract_version": EXECUTION_CONTRACT_VERSION,
            **fields,
            "receipt_sha256": _receipt(fields),
            "sealed_at": "producer" if force_reseal else "legacy_consumer_upgrade",
        }
        metadata["execution_identity"] = identity
    else:
        identity = dict(existing)

    metadata.setdefault("source_broker", fields["source_broker"])
    metadata.setdefault("source_profile", fields["source_profile"])
    metadata.setdefault("shadow_domain", fields["shadow_domain"])
    metadata.setdefault("runtime_lane", fields["runtime_lane"])
    metadata.setdefault("asset_class", fields["asset_class"])
    if fields["message_id"]:
        metadata.setdefault("execution_message_id", fields["message_id"])
        metadata.setdefault("decision_id", fields["message_id"])
    if fields["parent_message_id"]:
        metadata.setdefault("parent_decision_id", fields["parent_message_id"])
    row["metadata"] = metadata
    row["execution_contract_version"] = EXECUTION_CONTRACT_VERSION
    row["execution_actor_type"] = fields["actor_type"]
    row["execution_actor_id"] = fields["actor_id"]
    row["execution_authority_model"] = fields["authority_model"]
    row["source_broker"] = fields["source_broker"]
    row["source_profile"] = fields["source_profile"]
    row["shadow_domain"] = fields["shadow_domain"]
    row["runtime_lane"] = fields["runtime_lane"]
    row["routing_lane"] = fields["runtime_lane"]
    row["asset_class"] = fields["asset_class"]
    if fields["production_candidate_id"]:
        row["production_candidate_id"] = fields["production_candidate_id"]
    return row


def validate_execution_intent_contract(
    intent: Dict[str, Any],
    *,
    target_mode: str = "",
) -> Dict[str, Any]:
    metadata = _metadata(intent)
    identity = metadata.get("execution_identity")
    expected = _canonical_fields(intent)
    reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(identity, dict) or not identity:
        reasons.append("execution_identity_missing")
        identity = {}
    else:
        if _text(identity.get("contract_version")) != EXECUTION_CONTRACT_VERSION:
            reasons.append("execution_contract_version_mismatch")
        try:
            identity_schema_version = int(identity.get("schema_version") or 0)
        except (TypeError, ValueError):
            identity_schema_version = 0
        if identity_schema_version != EXECUTION_IDENTITY_SCHEMA_VERSION:
            reasons.append("execution_identity_schema_version_mismatch")
        if _text(identity.get("receipt_sha256")) != _receipt(expected):
            reasons.append("execution_identity_receipt_mismatch")
        for key, expected_value in expected.items():
            if _text(identity.get(key)) != expected_value:
                reasons.append(f"execution_identity_{key}_mismatch")

    if not expected["symbol"]:
        reasons.append("execution_symbol_missing")
    if not expected["message_id"]:
        reasons.append("execution_message_id_missing")
    if expected["action"] not in TRADE_ACTIONS:
        reasons.append("execution_action_not_tradeable")
    if (
        expected["actor_type"] == "unresolved"
        or expected["authority_model"] == "unbound"
    ):
        reasons.append("execution_actor_unresolved")
    if expected["source_broker"] == "unknown":
        warnings.append("execution_source_broker_unknown")
    if expected["asset_class"] == "unknown":
        warnings.append("execution_asset_class_unknown")
    if _text(target_mode or expected["target_mode"]).lower() == "live":
        if expected["source_broker"] == "unknown":
            reasons.append("live_execution_source_broker_unknown")
        if expected["production_candidate_id"] == "":
            reasons.append("live_execution_candidate_identity_missing")

    reasons = list(dict.fromkeys(reasons))
    warnings = list(dict.fromkeys(warnings))
    return {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "identity": dict(identity),
        "expected_identity": {
            **expected,
            "receipt_sha256": _receipt(expected),
        },
    }


def execution_provenance(intent: Dict[str, Any]) -> Dict[str, Any]:
    fields = _canonical_fields(intent)
    metadata = _metadata(intent)
    identity = metadata.get("execution_identity")
    return {
        "symbol": fields["symbol"],
        "action": fields["action"],
        "strategy": fields["strategy"],
        "intent_kind": fields["intent_kind"],
        "target_mode": fields["target_mode"],
        "source_broker": fields["source_broker"],
        "provider": fields["source_broker"],
        "source_profile": fields["source_profile"],
        "profile": fields["source_profile"],
        "shadow_domain": fields["shadow_domain"],
        "domain": fields["shadow_domain"],
        "runtime_lane": fields["runtime_lane"],
        "routing_lane": fields["runtime_lane"],
        "asset_class": fields["asset_class"],
        "execution_actor_type": fields["actor_type"],
        "execution_actor_id": fields["actor_id"],
        "execution_authority_model": fields["authority_model"],
        "production_candidate_id": fields["production_candidate_id"],
        "execution_identity": dict(identity) if isinstance(identity, dict) else {},
    }
