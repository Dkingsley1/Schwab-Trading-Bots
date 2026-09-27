#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from typing import Any

BALANCE_SECTIONS = {
    "current": "currentBalances",
    "projected": "projectedBalances",
    "initial": "initialBalances",
}
SENSITIVE_KEY_PARTS = (
    "accountnumber",
    "account_number",
    "accountreference",
    "hash",
    "token",
    "secret",
)
KNOWN_ACCOUNT_FIELDS = {
    "type",
    "isClosingOnlyRestricted",
    "isDayTrader",
    "isIntradayMargin",
    "isPortfolioMargin",
    "pfcbFlag",
    "roundTrips",
}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _alias_text(alias: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(alias.get(key) or "").strip()
        if value:
            return value
    return ""


def _canonical_token(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")


def _canonical_trading_access(alias: dict[str, Any]) -> str:
    raw = _alias_text(
        alias,
        "trading_access",
        "margin_access",
        "trading_type",
        "operator_trading_type",
    )
    token = _canonical_token(raw)
    aliases = {
        "cash": "cash",
        "cash_only": "cash",
        "limited": "limited_margin",
        "limited_margin": "limited_margin",
        "ira_limited_margin": "limited_margin",
        "margin": "full_margin",
        "full_margin": "full_margin",
        "reg_t_margin": "full_margin",
        "portfolio_margin": "portfolio_margin",
    }
    return aliases.get(token, "unknown")


def _primitive_provider_fields(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key)
        lowered = key_text.lower().replace("-", "_")
        if any(part in lowered for part in SENSITIVE_KEY_PARTS):
            continue
        if (
            isinstance(value, bool)
            or value is None
            or isinstance(value, (int, float, str))
        ):
            out[key_text] = value
    return dict(sorted(out.items()))


def provider_position_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields = _primitive_provider_fields(
        {key: value for key, value in row.items() if not str(key).startswith("_")}
    )
    fields.pop("instrument", None)
    instrument = (
        row.get("instrument") if isinstance(row.get("instrument"), dict) else {}
    )
    safe_instrument = _primitive_provider_fields(instrument)
    safe_instrument.pop("cusip", None)
    return {
        "position": fields,
        "instrument": safe_instrument,
    }


def normalize_operator_classification(alias: dict[str, Any] | None) -> dict[str, Any]:
    raw = alias if isinstance(alias, dict) else {}
    trading_access = _canonical_trading_access(raw)
    account_kind = (
        _canonical_token(
            _alias_text(raw, "operator_account_kind", "account_kind", "kind")
        )
        or "unknown"
    )
    tax_wrapper = _canonical_token(_alias_text(raw, "tax_wrapper", "ownership_wrapper"))
    if not tax_wrapper:
        if account_kind in {"roth", "roth_ira"}:
            tax_wrapper = "roth_ira"
        elif account_kind in {"traditional_ira", "ira"}:
            tax_wrapper = "traditional_ira"
        elif account_kind in {"cash", "taxable", "brokerage"}:
            tax_wrapper = "taxable"
        else:
            tax_wrapper = "unknown"

    verified = _as_bool(raw.get("operator_verified"), False) or _canonical_token(
        raw.get("verification_source")
    ) in {"operator", "broker_document", "broker_ui"}
    requested_borrowing = _as_bool(
        raw.get("borrowing_allowed"),
        trading_access in {"full_margin", "portfolio_margin"},
    )
    conflicts: list[str] = []
    if trading_access in {"cash", "limited_margin"} and requested_borrowing:
        conflicts.append("borrowing_requested_for_non_borrowing_access_model")
    borrowing_allowed = bool(
        requested_borrowing and trading_access in {"full_margin", "portfolio_margin"}
    )
    margin_interest_possible = bool(borrowing_allowed)
    allowed_routes = sorted(
        {
            str(item).strip()
            for item in _as_list(raw.get("allowed_live_routes"))
            if str(item or "").strip()
        }
    )
    canary_cap_usd = max(_safe_float(raw.get("canary_cap_usd"), 0.0), 0.0)
    classification_complete = bool(
        verified
        and account_kind != "unknown"
        and tax_wrapper != "unknown"
        and trading_access != "unknown"
        and not conflicts
    )
    return {
        "account_policy_key": _alias_text(raw, "account_policy_key", "policy_key"),
        "operator_account_label": _alias_text(
            raw, "operator_account_label", "label", "name"
        ),
        "account_kind": account_kind,
        "tax_wrapper": tax_wrapper,
        "tax_treatment": _canonical_token(raw.get("tax_treatment"))
        or (
            "taxable"
            if tax_wrapper == "taxable"
            else "tax_advantaged" if tax_wrapper != "unknown" else "unknown"
        ),
        "trading_access": trading_access,
        "limited_margin": trading_access == "limited_margin",
        "borrowing_allowed": borrowing_allowed,
        "margin_interest_possible": margin_interest_possible,
        "cash_only_live_budget": _as_bool(
            raw.get("cash_only_live_budget"),
            trading_access in {"cash", "limited_margin"},
        ),
        "short_stock_allowed": (
            _as_bool(raw.get("short_stock_allowed"), False)
            if borrowing_allowed
            else False
        ),
        "option_access": _canonical_token(raw.get("option_access")) or "unknown",
        "existing_positions_authority": _canonical_token(
            raw.get("existing_positions_authority")
        )
        or "observe_only",
        "canary_candidate": _as_bool(raw.get("canary_candidate"), False),
        "canary_cap_usd": round(canary_cap_usd, 2),
        "allowed_live_routes": allowed_routes,
        "operator_verified": verified,
        "verification_source": _canonical_token(raw.get("verification_source"))
        or "unverified",
        "verified_at_utc": str(raw.get("verified_at_utc") or ""),
        "classification_complete": classification_complete,
        "classification_conflicts": conflicts,
        "live_execution_authority": False,
    }


def _balance_sections(securities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _primitive_provider_fields(securities.get(provider_key))
        for name, provider_key in BALANCE_SECTIONS.items()
    }


def _first_balance_value(
    balances: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    *,
    sections: tuple[str, ...] = ("current", "projected", "initial"),
) -> float:
    for section in sections:
        row = balances.get(section, {})
        for key in keys:
            if key in row and row.get(key) not in {None, ""}:
                return _safe_float(row.get(key), 0.0)
    return 0.0


def _position_collateral_truth(positions: list[dict[str, Any]]) -> dict[str, Any]:
    equity_by_symbol: dict[str, float] = {}
    short_options: list[dict[str, Any]] = []
    option_close_mark = 0.0
    for row in positions:
        asset_type = str(row.get("asset_type") or "").upper()
        underlying = (
            str(row.get("underlying") or row.get("symbol") or "").strip().upper()
        )
        quantity = _safe_float(row.get("quantity"), 0.0)
        market_value = _safe_float(row.get("market_value"), 0.0)
        if asset_type == "EQUITY" and underlying:
            equity_by_symbol[underlying] = equity_by_symbol.get(underlying, 0.0) + max(
                quantity, 0.0
            )
        elif asset_type == "OPTION" and quantity < 0.0:
            contracts = abs(quantity)
            option_close_mark += abs(min(market_value, 0.0))
            short_options.append(
                {
                    "underlying": underlying,
                    "contracts": contracts,
                    "shares_required": contracts * 100.0,
                }
            )
    uncovered = []
    for row in short_options:
        shares = equity_by_symbol.get(str(row.get("underlying") or ""), 0.0)
        if shares + 1e-9 < _safe_float(row.get("shares_required"), 0.0):
            uncovered.append(str(row.get("underlying") or "UNKNOWN"))
    return {
        "short_option_contract_count": round(
            sum(_safe_float(row.get("contracts"), 0.0) for row in short_options), 6
        ),
        "short_option_close_mark_estimate": round(option_close_mark, 4),
        "covered_short_option_count": len(short_options) - len(uncovered),
        "uncovered_short_option_count": len(uncovered),
        "uncovered_underlyings": sorted(set(uncovered)),
    }


def build_account_capability_truth(
    securities: dict[str, Any],
    *,
    alias: dict[str, Any] | None,
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    classification = normalize_operator_classification(alias)
    provider_account = _primitive_provider_fields(securities)
    provider_account.pop("accountNumber", None)
    balances = _balance_sections(securities)
    cash_balance = _first_balance_value(balances, ("cashBalance", "totalCash"))
    cash_available = _first_balance_value(balances, ("cashAvailableForTrading",))
    available_funds = _first_balance_value(balances, ("availableFunds",))
    available_nonmargin = _first_balance_value(
        balances, ("availableFundsNonMarginableTrade",)
    )
    buying_power = _first_balance_value(balances, ("buyingPower", "stockBuyingPower"))
    buying_power_nonmargin = _first_balance_value(
        balances, ("buyingPowerNonMarginableTrade",)
    )
    intraday_buying_power = _first_balance_value(
        balances,
        (
            "intradayBuyingPowerAmount",
            "intradayMarginBuyingPower",
            "dayTradingBuyingPower",
        ),
    )
    accrued_interest = _first_balance_value(balances, ("accruedInterest",))
    margin_balance = _first_balance_value(balances, ("marginBalance", "margin"))
    pending_deposits = _first_balance_value(balances, ("pendingDeposits",))
    maintenance_call = _first_balance_value(balances, ("maintenanceCall",))
    reg_t_call = _first_balance_value(balances, ("regTCall",))
    day_trade_call = _first_balance_value(
        balances,
        ("dayTradingBuyingPowerCall", "dayTradingEquityCall"),
    )
    collateral = _position_collateral_truth(positions)

    negative_margin_balance = margin_balance < -0.005
    if not negative_margin_balance and accrued_interest <= 0.0:
        debit_status = "no_debit_indicated"
        borrowing_confirmed = False
    elif accrued_interest > 0.0:
        debit_status = "interest_accrual_present_requires_broker_review"
        borrowing_confirmed = classification.get("borrowing_allowed", False)
    elif classification.get("trading_access") in {"cash", "limited_margin"}:
        debit_status = "negative_provider_balance_not_confirmed_as_borrowing"
        borrowing_confirmed = False
    else:
        debit_status = "possible_margin_debit_requires_broker_confirmation"
        borrowing_confirmed = False

    canary_cap = _safe_float(classification.get("canary_cap_usd"), 0.0)
    canary_candidate = bool(classification.get("canary_candidate", False))
    cash_proxy_ready = bool(
        canary_cap > 0.0
        and cash_balance + 0.005 >= canary_cap
        and pending_deposits <= 0.0
    )
    canary_blockers: list[str] = []
    if canary_candidate:
        if not classification.get("classification_complete", False):
            canary_blockers.append("operator_account_classification_incomplete")
        if not classification.get("account_policy_key"):
            canary_blockers.append("account_policy_binding_missing")
        if classification.get("borrowing_allowed", False):
            canary_blockers.append("borrowing_not_disabled_for_canary")
        if not classification.get("cash_only_live_budget", False):
            canary_blockers.append("cash_only_canary_budget_not_enforced")
        if not classification.get("allowed_live_routes"):
            canary_blockers.append("canary_route_scope_missing")
        if canary_cap <= 0.0:
            canary_blockers.append("canary_cap_missing")
        elif not cash_proxy_ready:
            canary_blockers.append("canary_cash_not_funded_and_settled")
        if bool(provider_account.get("isClosingOnlyRestricted", False)):
            canary_blockers.append("account_closing_only_restricted")
        if maintenance_call or reg_t_call or day_trade_call:
            canary_blockers.append("broker_account_call_present")
        if collateral.get("uncovered_short_option_count", 0):
            canary_blockers.append("uncovered_short_option_present")

    unknown_account_fields = sorted(set(provider_account) - KNOWN_ACCOUNT_FIELDS)
    return {
        "schema_version": 1,
        "operator_classification": classification,
        "provider_account": {
            "fields": provider_account,
            "provider_account_type": str(provider_account.get("type") or "").upper(),
            "provider_type_is_not_borrowing_authority": True,
            "unknown_safe_field_names": unknown_account_fields,
        },
        "provider_balances": balances,
        "provider_field_inventory": {
            "account": sorted(provider_account),
            **{name: sorted(row) for name, row in balances.items()},
        },
        "balance_truth": {
            "cash_balance": round(cash_balance, 4),
            "cash_available_for_trading": round(cash_available, 4),
            "available_funds": round(available_funds, 4),
            "available_funds_nonmarginable": round(available_nonmargin, 4),
            "buying_power": round(buying_power, 4),
            "buying_power_nonmarginable": round(buying_power_nonmargin, 4),
            "intraday_buying_power": round(intraday_buying_power, 4),
            "pending_deposits": round(pending_deposits, 4),
            "cash_funding_source": "broker_cash_balance_proxy",
            "settled_cash_claimed": False,
        },
        "debit_truth": {
            "provider_margin_balance": round(margin_balance, 4),
            "provider_margin_balance_magnitude": round(
                abs(min(margin_balance, 0.0)), 4
            ),
            "accrued_interest": round(accrued_interest, 4),
            "status": debit_status,
            "interest_bearing_borrowing_confirmed": bool(borrowing_confirmed),
            "operator_borrowing_authority": bool(
                classification.get("borrowing_allowed", False)
            ),
            "requires_broker_ui_confirmation": debit_status != "no_debit_indicated",
            "negative_margin_balance_is_not_option_close_mark": True,
        },
        "broker_call_truth": {
            "maintenance_call": round(maintenance_call, 4),
            "reg_t_call": round(reg_t_call, 4),
            "day_trading_call": round(day_trade_call, 4),
            "in_call": bool(maintenance_call or reg_t_call or day_trade_call),
        },
        "position_collateral_truth": collateral,
        "canary_preflight": {
            "designated_candidate": canary_candidate,
            "account_policy_key": classification.get("account_policy_key"),
            "configured_cap_usd": round(canary_cap, 2),
            "cash_proxy_ready": cash_proxy_ready,
            "account_preflight_ready": bool(canary_candidate and not canary_blockers),
            "blockers": canary_blockers,
            "allowed_live_routes": classification.get("allowed_live_routes", []),
            "existing_positions_authority": classification.get(
                "existing_positions_authority"
            ),
            "live_execution_authority": False,
            "activation_policy": "account_truth_can_block_but_never_grant_live_execution",
        },
        "redaction": {
            "raw_account_number_emitted": False,
            "raw_account_hash_emitted": False,
        },
    }
