from __future__ import annotations

import json
import math
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from core.brokers.base import BrokerAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "config" / "broker_capability_contracts_v1.json"
_ACTIVE_ORDER_STATES = {"paper", "live"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def load_capability_contracts(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("broker capability contract must be an object")
    validate_capability_contracts(payload)
    return payload


def validate_capability_contracts(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("broker capability contract schema must be version 1")
    brokers = _mapping(payload.get("brokers"))
    if not brokers:
        raise ValueError("broker capability contract defines no brokers")
    for broker_name, raw in brokers.items():
        broker = _mapping(raw)
        if not _mapping(broker.get("declared_capabilities")):
            raise ValueError(f"broker declared capabilities missing: {broker_name}")
        pools = _mapping(broker.get("rate_limit_pools"))
        if not pools or any(float(value or 0) <= 0 for value in pools.values()):
            raise ValueError(f"broker rate-limit pools invalid: {broker_name}")
        for mode in _ACTIVE_ORDER_STATES:
            contract = _mapping(broker.get(mode))
            if (
                not contract
                or "enabled" not in contract
                or "production_eligible" not in contract
            ):
                raise ValueError(
                    f"broker mode contract incomplete: {broker_name}:{mode}"
                )
            if contract.get("enabled"):
                for key in ("asset_classes", "order_types", "time_in_force"):
                    if not contract.get(key):
                        raise ValueError(
                            f"broker mode {key} missing: {broker_name}:{mode}"
                        )
            if mode == "paper" and contract.get("production_eligible"):
                raise ValueError(
                    f"paper mode cannot be production eligible: {broker_name}"
                )


def broker_contract(
    broker_name: str, *, contracts: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    active = (
        dict(contracts)
        if isinstance(contracts, Mapping)
        else load_capability_contracts()
    )
    brokers = _mapping(active.get("brokers"))
    name = str(broker_name or "").strip().lower()
    row = _mapping(brokers.get(name))
    if not row:
        raise KeyError(f"broker capability contract missing: {name or 'unknown'}")
    return row


def _normalize_asset_type(value: Any, order_spec: Mapping[str, Any]) -> str:
    aliases = {
        "STOCK": "EQUITY",
        "EQUITIES": "EQUITY",
        "FUTURES": "FUTURE",
        "OPTIONS": "OPTION",
        "CRYPTOCURRENCY": "CRYPTO",
    }
    candidates = [value, order_spec.get("assetType"), order_spec.get("asset_type")]
    legs = order_spec.get("orderLegCollection")
    if isinstance(legs, list):
        for leg in legs:
            if not isinstance(leg, Mapping):
                continue
            instrument = _mapping(leg.get("instrument"))
            candidates.extend(
                (instrument.get("assetType"), instrument.get("asset_type"))
            )
    for candidate in candidates:
        normalized = str(candidate or "").strip().upper()
        if normalized:
            return aliases.get(normalized, normalized)
    return "EQUITY"


def _decimal_places(value: Any) -> int:
    try:
        decimal = Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return 999
    return max(-decimal.as_tuple().exponent, 0)


def evaluate_order_request(
    broker_name: str,
    request: Mapping[str, Any],
    *,
    mode: str,
    contracts: Mapping[str, Any] | None = None,
    require_production_eligible: bool = False,
) -> dict[str, Any]:
    broker = broker_contract(broker_name, contracts=contracts)
    normalized_mode = str(mode or "").strip().lower()
    reasons: list[str] = []
    if normalized_mode not in _ACTIVE_ORDER_STATES:
        reasons.append("unsupported_execution_mode")
        mode_contract: dict[str, Any] = {}
    else:
        mode_contract = _mapping(broker.get(normalized_mode))
    if not mode_contract.get("enabled", False):
        reasons.append(f"{normalized_mode or 'unknown'}_execution_not_supported")
    if require_production_eligible and not mode_contract.get(
        "production_eligible", False
    ):
        reasons.append("broker_mode_not_production_eligible")

    order_spec = _mapping(request.get("order_spec"))
    asset_type = _normalize_asset_type(request.get("asset_type"), order_spec)
    order_type = (
        str(order_spec.get("orderType") or order_spec.get("order_type") or "MARKET")
        .strip()
        .upper()
    )
    time_in_force = (
        str(
            order_spec.get("duration")
            or order_spec.get("timeInForce")
            or order_spec.get("time_in_force")
            or "DAY"
        )
        .strip()
        .upper()
    )
    action = str(request.get("action") or "").strip().upper()
    symbol = str(request.get("symbol") or "").strip().upper()
    try:
        quantity = float(request.get("quantity") or 0.0)
    except (TypeError, ValueError):
        quantity = 0.0
    quantity_policy = _mapping(broker.get("quantity"))
    minimum_quantity = float(quantity_policy.get("minimum") or 0.0)
    if not symbol:
        reasons.append("symbol_missing")
    if action not in {
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
    }:
        reasons.append("action_not_supported")
    if not math.isfinite(quantity) or quantity < minimum_quantity:
        reasons.append("quantity_below_contract_minimum")
    if _decimal_places(request.get("quantity")) > int(
        quantity_policy.get("maximum_decimals") or 0
    ):
        reasons.append("quantity_precision_exceeds_contract")
    if asset_type not in {
        str(value).upper() for value in mode_contract.get("asset_classes") or []
    }:
        reasons.append(f"asset_class_not_supported_for_{normalized_mode}:{asset_type}")
    if order_type not in {
        str(value).upper() for value in mode_contract.get("order_types") or []
    }:
        reasons.append(f"order_type_not_supported_for_{normalized_mode}:{order_type}")
    if time_in_force not in {
        str(value).upper() for value in mode_contract.get("time_in_force") or []
    }:
        reasons.append(
            f"time_in_force_not_supported_for_{normalized_mode}:{time_in_force}"
        )

    price = request.get("limit_price")
    if not price:
        price = order_spec.get("price") or order_spec.get("stopPrice") or 0.0
    if order_type in {"LIMIT", "STOP_LIMIT"}:
        try:
            price_number = float(price)
        except (TypeError, ValueError):
            price_number = 0.0
        price_policy = _mapping(broker.get("price"))
        if not math.isfinite(price_number) or price_number < float(
            price_policy.get("minimum") or 0.0
        ):
            reasons.append("limit_price_below_contract_minimum")
        if _decimal_places(price) > int(price_policy.get("maximum_decimals") or 0):
            reasons.append("price_precision_exceeds_contract")

    return {
        "ok": not reasons,
        "broker": str(broker_name or "").strip().lower(),
        "mode": normalized_mode,
        "asset_type": asset_type,
        "order_type": order_type,
        "time_in_force": time_in_force,
        "production_eligible": bool(mode_contract.get("production_eligible", False)),
        "implementation": str(mode_contract.get("implementation") or "unknown"),
        "reasons": reasons,
    }


def adapter_conformance_report(
    adapter: BrokerAdapter, *, contracts: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    broker = broker_contract(adapter.name, contracts=contracts)
    declared = _mapping(broker.get("declared_capabilities"))
    actual = asdict(adapter.capabilities)
    mismatches = [
        key
        for key, expected in sorted(declared.items())
        if bool(actual.get(key, False)) != bool(expected)
    ]
    market_data_transport = str(broker.get("market_data_transport") or "").strip()
    method_checks = {
        "supports_market_data": bool(adapter.quote_candidates(symbol="SPY"))
        or market_data_transport == "authenticated_client_direct",
        "supports_account_discovery": bool(adapter.account_numbers_candidates()),
        "supports_account_snapshot": bool(
            adapter.accounts_snapshot_candidates(
                account_reference="contract-test", allow_global_fallback=False
            )
        ),
        "supports_positions": bool(
            adapter.position_candidates(account_reference="contract-test")
        ),
        "supports_order_place": bool(
            adapter.place_order_candidates(
                account_reference="contract-test", order_spec={"orderType": "MARKET"}
            )
        ),
        "supports_order_replace": bool(
            adapter.replace_order_candidates(
                account_reference="contract-test",
                order_id="contract-order",
                order_spec={"orderType": "LIMIT"},
            )
        ),
        "supports_order_cancel": bool(
            adapter.cancel_order_candidates(
                account_reference="contract-test", order_id="contract-order"
            )
        ),
        "supports_order_fetch": bool(
            adapter.fetch_order_candidates(
                account_reference="contract-test", order_id="contract-order"
            )
        ),
    }
    method_mismatches = [
        key
        for key, has_candidates in method_checks.items()
        if bool(actual.get(key, False)) != bool(has_candidates)
    ]
    return {
        "ok": not mismatches and not method_mismatches,
        "broker": adapter.name,
        "capability_mismatches": mismatches,
        "candidate_method_mismatches": method_mismatches,
        "method_checks": method_checks,
        "paper_implementation": str(
            _mapping(broker.get("paper")).get("implementation") or ""
        ),
        "live_implementation": str(
            _mapping(broker.get("live")).get("implementation") or ""
        ),
        "live_production_eligible": bool(
            _mapping(broker.get("live")).get("production_eligible", False)
        ),
    }


def all_adapter_conformance(
    *, contracts: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    from core.brokers.registry import available_broker_names, build_broker_adapter

    active = (
        dict(contracts)
        if isinstance(contracts, Mapping)
        else load_capability_contracts()
    )
    rows = [
        adapter_conformance_report(build_broker_adapter(name), contracts=active)
        for name in available_broker_names()
    ]
    return {
        "ok": all(row["ok"] for row in rows),
        "broker_count": len(rows),
        "brokers": rows,
    }
