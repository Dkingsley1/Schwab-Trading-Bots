#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "trading_tax_policy_us_federal_2026.json"
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "trading_tax_profile.json"
DEFAULT_ACCOUNT_CONTEXT_PATH = PROJECT_ROOT / "governance" / "health" / "account_policy_context_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "trading_tax_estimate_latest.json"

FILING_STATUS_ALIASES = {
    "single": "single",
    "married_filing_jointly": "married_filing_jointly",
    "mfj": "married_filing_jointly",
    "married_joint": "married_filing_jointly",
    "married_filing_separately": "married_filing_separately",
    "mfs": "married_filing_separately",
    "married_separate": "married_filing_separately",
    "head_of_household": "head_of_household",
    "hoh": "head_of_household",
    "qualifying_surviving_spouse": "qualifying_surviving_spouse",
    "qss": "qualifying_surviving_spouse",
    "qualifying_widow": "qualifying_surviving_spouse",
}

PAPER_ENVIRONMENTS = {"paper", "shadow", "simulation", "simulated", "backtest", "research"}
ACTUAL_ENVIRONMENTS = {"actual", "broker", "live", "real", "production"}
TAX_ADVANTAGED_TREATMENTS = {
    "tax_advantaged",
    "roth",
    "roth_ira",
    "traditional_ira",
    "ira",
    "401k",
    "403b",
    "hsa",
}


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _number(raw: Any) -> float | None:
    if raw in {None, ""}:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        parsed = _number(value)
        if parsed is not None:
            return parsed
    return None


def _round(raw: Any, digits: int = 2) -> float | None:
    value = _number(raw)
    return round(value, digits) if value is not None else None


def _parse_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return date.fromisoformat(text[:10])
        except Exception:
            return None


def _now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _filing_status(raw: Any) -> str:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return FILING_STATUS_ALIASES.get(key, "unknown")


def _tax_treatment(raw: Any) -> str:
    value = str(raw or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if value in TAX_ADVANTAGED_TREATMENTS:
        return "tax_advantaged"
    if value in {"taxable", "brokerage", "individual", "joint_taxable"}:
        return "taxable"
    return "unknown"


def _validate_brackets(rows: Any, *, upper_key: str) -> list[str]:
    issues: list[str] = []
    brackets = _list(rows)
    if not brackets:
        return ["brackets_missing"]
    previous = -1.0
    for index, row in enumerate(brackets):
        item = _dict(row)
        rate = _number(item.get("rate"))
        upper = _number(item.get(upper_key))
        if rate is None or rate < 0.0 or rate > 1.0:
            issues.append(f"invalid_rate_at_{index}")
        if upper is None:
            if index != len(brackets) - 1:
                issues.append(f"open_band_before_end_at_{index}")
            continue
        if upper <= previous:
            issues.append(f"non_monotonic_upper_band_at_{index}")
        previous = upper
    if _number(_dict(brackets[-1]).get(upper_key)) is not None:
        issues.append("final_band_must_be_open")
    return issues


def validate_policy(policy: dict[str, Any], *, requested_tax_year: int) -> dict[str, Any]:
    issues: list[str] = []
    tax_year = int(_number(policy.get("tax_year")) or 0)
    if tax_year != int(requested_tax_year):
        issues.append("tax_year_mismatch")
    if str(policy.get("jurisdiction") or "") != "US_FEDERAL":
        issues.append("unsupported_jurisdiction")
    if not str(policy.get("verification_status") or "").startswith("verified"):
        issues.append("policy_not_verified")

    ordinary = _dict(policy.get("ordinary_income_brackets"))
    preferred = _dict(policy.get("preferential_capital_gain_brackets"))
    for status in sorted(set(FILING_STATUS_ALIASES.values())):
        issues.extend(
            f"ordinary_{status}_{issue}"
            for issue in _validate_brackets(ordinary.get(status), upper_key="up_to_usd")
        )
        issues.extend(
            f"preferred_{status}_{issue}"
            for issue in _validate_brackets(preferred.get(status), upper_key="up_to_taxable_income_usd")
        )

    niit = _dict(policy.get("net_investment_income_tax"))
    niit_rate = _number(niit.get("rate"))
    if niit_rate is None or niit_rate < 0.0 or niit_rate > 1.0:
        issues.append("invalid_niit_rate")
    section_1256 = _dict(policy.get("section_1256"))
    long_fraction = _number(section_1256.get("long_term_fraction"))
    short_fraction = _number(section_1256.get("short_term_fraction"))
    if long_fraction is None or short_fraction is None or abs(long_fraction + short_fraction - 1.0) > 1e-9:
        issues.append("invalid_section_1256_split")

    sources = _list(policy.get("source_references"))
    if not sources or any("irs.gov" not in str(_dict(row).get("url") or "") for row in sources):
        issues.append("official_source_attestation_missing")
    return {
        "ok": not issues,
        "status": "verified" if not issues else "blocked",
        "tax_year": tax_year,
        "issues": sorted(set(issues)),
        "source_count": len(sources),
    }


def _account_tax_treatments(profile: dict[str, Any], account_context: dict[str, Any]) -> dict[str, str]:
    treatments: dict[str, str] = {}
    for key, value in _dict(profile.get("account_tax_treatment_by_label")).items():
        label = str(key or "").strip()
        if label:
            treatments[label] = _tax_treatment(value)

    context = _dict(account_context.get("account_policy_context"))
    for row in _list(context.get("configured_account_slots")):
        item = _dict(row)
        treatment = _tax_treatment(item.get("tax_treatment"))
        for key in ("account_label", "account_policy_key"):
            label = str(item.get(key) or "").strip()
            if label and label not in treatments:
                treatments[label] = treatment
    return treatments


def _event_environment(row: dict[str, Any]) -> str:
    if bool(row.get("paper", False)):
        return "paper"
    value = str(
        row.get("environment")
        or row.get("execution_environment")
        or row.get("mode")
        or row.get("source_environment")
        or "unknown"
    ).strip().lower()
    if value in PAPER_ENVIRONMENTS:
        return "paper"
    if value in ACTUAL_ENVIRONMENTS:
        return "actual"
    return "unknown"


def _event_kind(row: dict[str, Any]) -> str:
    raw = str(
        row.get("tax_event_kind")
        or row.get("event_kind")
        or row.get("income_type")
        or row.get("transaction_type")
        or row.get("type")
        or "unknown"
    ).strip().lower().replace("-", "_").replace(" ", "_")
    description = str(row.get("description") or "").strip().lower()
    if raw in {"capital_gain", "capital_loss", "capital_disposition", "trade_disposition", "sale", "sell"}:
        return "capital_disposition"
    if raw in {"dividend", "cash_dividend", "qualified_dividend", "ordinary_dividend"} or "dividend" in description:
        return "dividend"
    if raw in {"investment_interest_expense", "margin_interest_expense"}:
        return "investment_interest_expense"
    if raw in {"interest", "taxable_interest"} or "interest" in description:
        return "interest"
    if raw in {"section_1256", "futures_mark_to_market", "regulated_futures"}:
        return "section_1256"
    if raw in {"tax_exempt_interest", "municipal_interest"}:
        return "tax_exempt_interest"
    if raw in {"acquisition", "buy", "reinvestment"}:
        return "acquisition"
    if raw in {"fee", "commission", "transfer", "deposit", "withdrawal"}:
        return raw
    return "unknown"


def _event_amount(row: dict[str, Any], *, kind: str) -> tuple[float | None, str]:
    if kind == "capital_disposition":
        direct = _first_number(
            row.get("realized_gain_loss_usd"),
            row.get("taxable_gain_loss_usd"),
            row.get("realized_pnl_usd"),
            row.get("realized_pnl"),
        )
        if direct is not None:
            return direct, "reported_realized_gain_loss"
        proceeds = _first_number(row.get("proceeds_usd"), row.get("sale_proceeds_usd"))
        basis = _first_number(row.get("adjusted_cost_basis_usd"), row.get("cost_basis_usd"))
        fees = _first_number(row.get("fees_usd"), row.get("commissions_and_fees_usd")) or 0.0
        if proceeds is not None and basis is not None:
            return proceeds - basis - fees, "derived_from_proceeds_basis_fees"
        return None, "missing_realized_gain_or_basis"
    amount = _first_number(
        row.get("taxable_amount_usd"),
        row.get("income_amount_usd"),
        row.get("amount_usd"),
        row.get("net_amount_usd"),
    )
    return amount, "reported_taxable_amount" if amount is not None else "missing_taxable_amount"


def _capital_character(row: dict[str, Any], policy: dict[str, Any]) -> tuple[str, str]:
    explicit = str(row.get("tax_character") or row.get("holding_period_class") or "").strip().lower()
    if explicit in {"short", "short_term", "short_term_capital"}:
        return "short_term", "reported"
    if explicit in {"long", "long_term", "long_term_capital"}:
        return "long_term", "reported"
    if explicit in {"ordinary", "ordinary_income"}:
        return "ordinary", "reported"
    if explicit in {"section_1256", "60_40"} and bool(row.get("section_1256_verified", False)):
        return "section_1256", "verified_contract_classification"

    acquired = _parse_date(row.get("acquired_at") or row.get("acquisition_date") or row.get("date_acquired"))
    disposed = _parse_date(row.get("disposed_at") or row.get("disposition_date") or row.get("date_sold"))
    if acquired is not None and disposed is not None and disposed >= acquired:
        minimum = int(_number(policy.get("long_term_holding_period_days_minimum")) or 366)
        return ("long_term" if (disposed - acquired).days >= minimum else "short_term"), "derived_from_dates"
    days = _number(row.get("holding_period_days"))
    if days is not None:
        minimum = int(_number(policy.get("long_term_holding_period_days_minimum")) or 366)
        return ("long_term" if days >= minimum else "short_term"), "derived_from_holding_days"
    return "unknown", "holding_period_missing"


def _empty_buckets() -> dict[str, float]:
    return {
        "ordinary_investment_income": 0.0,
        "short_term_capital": 0.0,
        "long_term_capital": 0.0,
        "qualified_dividends": 0.0,
        "tax_exempt_interest": 0.0,
        "net_investment_income": 0.0,
    }


def _add(bucket: dict[str, float], key: str, amount: float) -> None:
    bucket[key] = float(bucket.get(key, 0.0)) + float(amount)


def _event_date(row: dict[str, Any]) -> date | None:
    return _parse_date(
        row.get("disposed_at")
        or row.get("transaction_date")
        or row.get("timestamp_utc")
        or row.get("date")
    )


def _acquisition_index(events: Iterable[dict[str, Any]]) -> dict[str, list[date]]:
    index: dict[str, list[date]] = {}
    for row in events:
        action = str(row.get("action") or row.get("side") or row.get("transaction_subtype") or "").strip().upper()
        kind = str(row.get("tax_event_kind") or row.get("event_kind") or "").strip().lower()
        if not (action.startswith("BUY") or kind in {"acquisition", "buy", "reinvestment"}):
            continue
        symbol = str(row.get("symbol") or row.get("underlying") or "").strip().upper()
        event_date = _event_date(row)
        if symbol and event_date is not None:
            index.setdefault(symbol, []).append(event_date)
    return index


def _potential_wash_sale(
    row: dict[str, Any],
    *,
    amount: float,
    acquisitions: dict[str, list[date]],
    policy: dict[str, Any],
) -> bool:
    if amount >= 0.0:
        return False
    status = str(row.get("wash_sale_status") or "").strip().lower()
    if status in {"clear", "not_applicable", "adjusted", "verified_clear"}:
        return False
    if bool(row.get("wash_sale_verified_clear", False)):
        return False
    symbol = str(row.get("symbol") or row.get("underlying") or "").strip().upper()
    sold = _event_date(row)
    before = int(_number(policy.get("wash_sale_window_days_before")) or 30)
    after = int(_number(policy.get("wash_sale_window_days_after")) or 30)
    if symbol and sold is not None:
        for acquired in acquisitions.get(symbol, []):
            delta = (acquired - sold).days
            if -before <= delta <= after:
                return True
    return status not in {"not_a_wash_sale", "broker_verified_clear"}


def _classify_events(
    events: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    policy: dict[str, Any],
    account_context: dict[str, Any],
) -> dict[str, Any]:
    lower = _empty_buckets()
    upper = _empty_buckets()
    accounts = _account_tax_treatments(profile, account_context)
    acquisitions = _acquisition_index(events)
    classifications: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    counts = {
        "input_events": len(events),
        "actual_taxable_events": 0,
        "tax_advantaged_events": 0,
        "paper_events": 0,
        "unrealized_events": 0,
        "ignored_nontax_events": 0,
        "unresolved_events": 0,
    }
    paper_realized = 0.0
    unrealized = 0.0

    for index, raw in enumerate(events):
        row = _dict(raw)
        event_id = str(row.get("event_id") or row.get("transaction_id") or f"row_{index + 1}")
        environment = _event_environment(row)
        realization = str(row.get("realization_status") or row.get("tax_status") or "realized").strip().lower()
        kind = _event_kind(row)
        amount, amount_source = _event_amount(row, kind=kind)
        account_label = str(row.get("account_label") or row.get("operator_account_label") or "").strip()
        treatment = _tax_treatment(row.get("tax_treatment") or accounts.get(account_label))

        if environment == "paper":
            counts["paper_events"] += 1
            if amount is not None and realization != "unrealized":
                paper_realized += amount
            classifications.append({
                "event_id": event_id,
                "classification": "paper_excluded_from_tax",
                "amount_usd": _round(amount),
            })
            continue
        if realization in {"unrealized", "open", "not_realized", "mark_to_market_unverified"}:
            counts["unrealized_events"] += 1
            if amount is not None:
                unrealized += amount
            classifications.append({
                "event_id": event_id,
                "classification": "unrealized_not_currently_owed",
                "amount_usd": _round(amount),
            })
            continue
        if environment != "actual":
            unresolved.append({"event_id": event_id, "reason": "execution_environment_unknown"})
            continue
        if kind in {"fee", "commission", "transfer", "deposit", "withdrawal"}:
            counts["ignored_nontax_events"] += 1
            continue
        if treatment == "tax_advantaged":
            counts["tax_advantaged_events"] += 1
            classifications.append({
                "event_id": event_id,
                "classification": "verified_tax_advantaged_current_tax_excluded",
                "amount_usd": _round(amount),
            })
            continue
        if treatment != "taxable":
            unresolved.append({"event_id": event_id, "reason": "account_tax_treatment_unknown"})
            continue
        if kind == "investment_interest_expense":
            unresolved.append({
                "event_id": event_id,
                "reason": "investment_interest_deduction_requires_itemization_and_form_4952_limits",
            })
            continue
        if amount is None:
            unresolved.append({"event_id": event_id, "reason": amount_source})
            continue

        counts["actual_taxable_events"] += 1
        if bool(row.get("tax_amount_provisional", False)):
            unresolved.append({"event_id": event_id, "reason": "provisional_amount_requires_tax_form_reconciliation"})
        classification = ""
        evidence = amount_source
        if kind in {"capital_disposition", "section_1256"}:
            character, character_source = _capital_character(row, policy)
            if kind == "section_1256" and bool(row.get("section_1256_verified", False)):
                character = "section_1256"
                character_source = "verified_contract_classification"
            disallowed = abs(_first_number(row.get("wash_sale_disallowed_loss_usd")) or 0.0)
            taxable_amount = amount + disallowed if amount < 0.0 else amount
            potential_wash = _potential_wash_sale(
                row,
                amount=taxable_amount,
                acquisitions=acquisitions,
                policy=policy,
            )
            if character == "section_1256":
                split = _dict(policy.get("section_1256"))
                long_fraction = float(_number(split.get("long_term_fraction")) or 0.60)
                short_fraction = float(_number(split.get("short_term_fraction")) or 0.40)
                for bucket in (lower, upper):
                    _add(bucket, "long_term_capital", taxable_amount * long_fraction)
                    _add(bucket, "short_term_capital", taxable_amount * short_fraction)
                    _add(bucket, "net_investment_income", taxable_amount)
                classification = "section_1256_60_40"
            elif character == "short_term":
                _add(lower, "short_term_capital", taxable_amount)
                _add(lower, "net_investment_income", taxable_amount)
                if potential_wash:
                    _add(upper, "short_term_capital", max(taxable_amount, 0.0))
                    _add(upper, "net_investment_income", max(taxable_amount, 0.0))
                else:
                    _add(upper, "short_term_capital", taxable_amount)
                    _add(upper, "net_investment_income", taxable_amount)
                classification = "short_term_capital"
            elif character == "long_term":
                _add(lower, "long_term_capital", taxable_amount)
                _add(lower, "net_investment_income", taxable_amount)
                if potential_wash:
                    _add(upper, "long_term_capital", max(taxable_amount, 0.0))
                    _add(upper, "net_investment_income", max(taxable_amount, 0.0))
                else:
                    _add(upper, "long_term_capital", taxable_amount)
                    _add(upper, "net_investment_income", taxable_amount)
                classification = "long_term_capital"
            elif character == "ordinary":
                for bucket in (lower, upper):
                    _add(bucket, "ordinary_investment_income", taxable_amount)
                    _add(bucket, "net_investment_income", taxable_amount)
                classification = "ordinary_income"
            else:
                if taxable_amount >= 0.0:
                    _add(lower, "long_term_capital", taxable_amount)
                    _add(upper, "short_term_capital", taxable_amount)
                else:
                    _add(lower, "short_term_capital", taxable_amount)
                    if not potential_wash:
                        _add(upper, "long_term_capital", taxable_amount)
                _add(lower, "net_investment_income", taxable_amount)
                _add(upper, "net_investment_income", max(taxable_amount, 0.0) if potential_wash else taxable_amount)
                classification = "capital_holding_period_range"
                unresolved.append({"event_id": event_id, "reason": character_source})
            if potential_wash:
                unresolved.append({"event_id": event_id, "reason": "potential_wash_sale_or_cross_account_repurchase"})
            evidence = f"{amount_source};{character_source}"
        elif kind == "dividend":
            qualified = row.get("qualified_dividend")
            if qualified is True or str(qualified).strip().lower() in {"1", "true", "yes"}:
                for bucket in (lower, upper):
                    _add(bucket, "qualified_dividends", amount)
                    _add(bucket, "net_investment_income", amount)
                classification = "qualified_dividend"
            elif qualified is False or str(qualified).strip().lower() in {"0", "false", "no"}:
                for bucket in (lower, upper):
                    _add(bucket, "ordinary_investment_income", amount)
                    _add(bucket, "net_investment_income", amount)
                classification = "ordinary_dividend"
            else:
                _add(lower, "qualified_dividends", amount)
                _add(upper, "ordinary_investment_income", amount)
                for bucket in (lower, upper):
                    _add(bucket, "net_investment_income", amount)
                classification = "dividend_qualification_range"
                unresolved.append({"event_id": event_id, "reason": "qualified_dividend_status_unknown"})
        elif kind == "interest":
            for bucket in (lower, upper):
                _add(bucket, "ordinary_investment_income", amount)
                _add(bucket, "net_investment_income", amount)
            classification = "taxable_interest"
        elif kind == "tax_exempt_interest":
            for bucket in (lower, upper):
                _add(bucket, "tax_exempt_interest", amount)
            classification = "tax_exempt_interest"
        else:
            unresolved.append({"event_id": event_id, "reason": "tax_event_kind_unknown"})
            counts["actual_taxable_events"] -= 1
            continue

        classifications.append({
            "event_id": event_id,
            "account_label": account_label,
            "symbol": str(row.get("symbol") or ""),
            "classification": classification,
            "amount_usd": round(amount, 2),
            "evidence": evidence,
        })

    counts["unresolved_events"] = len(unresolved)
    return {
        "lower": lower,
        "upper": upper,
        "counts": counts,
        "classifications": classifications,
        "unresolved": unresolved,
        "paper_realized_pnl_usd": round(paper_realized, 2),
        "unrealized_pnl_usd": round(unrealized, 2),
        "account_tax_treatments": accounts,
    }


def _ordinary_tax(income: float, brackets: list[Any]) -> float:
    taxable = max(float(income), 0.0)
    tax = 0.0
    previous = 0.0
    for raw in brackets:
        row = _dict(raw)
        rate = float(_number(row.get("rate")) or 0.0)
        upper = _number(row.get("up_to_usd"))
        if upper is None:
            tax += max(taxable - previous, 0.0) * rate
            break
        amount = max(min(taxable, upper) - previous, 0.0)
        tax += amount * rate
        previous = upper
        if taxable <= upper:
            break
    return tax


def _preferred_tax(ordinary_income: float, preferred_income: float, brackets: list[Any]) -> float:
    remaining = max(float(preferred_income), 0.0)
    stack = max(float(ordinary_income), 0.0)
    tax = 0.0
    previous = 0.0
    for raw in brackets:
        row = _dict(raw)
        rate = float(_number(row.get("rate")) or 0.0)
        upper = _number(row.get("up_to_taxable_income_usd"))
        band_start = max(previous, stack)
        if upper is None:
            capacity = remaining
        else:
            capacity = max(upper - band_start, 0.0)
        taxed = min(remaining, capacity)
        tax += taxed * rate
        remaining -= taxed
        if remaining <= 1e-9:
            break
        if upper is not None:
            previous = upper
    return tax


def _net_capital(short_term: float, long_term: float, *, loss_limit: float) -> dict[str, float]:
    st = float(short_term)
    lt = float(long_term)
    if st > 0.0 and lt < 0.0:
        combined = st + lt
        st, lt = (combined, 0.0) if combined >= 0.0 else (0.0, combined)
    elif st < 0.0 and lt > 0.0:
        combined = st + lt
        st, lt = (0.0, combined) if combined >= 0.0 else (combined, 0.0)
    total_loss = abs(min(st, 0.0)) + abs(min(lt, 0.0))
    return {
        "net_short_term_gain_usd": max(st, 0.0),
        "net_long_term_gain_usd": max(lt, 0.0),
        "net_capital_loss_usd": total_loss,
        "current_year_ordinary_loss_deduction_usd": min(total_loss, max(float(loss_limit), 0.0)),
        "estimated_loss_carryforward_usd": max(total_loss - max(float(loss_limit), 0.0), 0.0),
    }


def _scenario_tax(
    bucket: dict[str, float],
    *,
    profile: dict[str, Any],
    policy: dict[str, Any],
    filing_status: str,
) -> dict[str, Any] | None:
    base_ordinary = _number(profile.get("taxable_ordinary_income_before_trading_usd"))
    base_preferred = _number(profile.get("preferential_income_before_trading_usd"))
    if base_ordinary is None or base_preferred is None:
        return None

    st_carry = _number(profile.get("short_term_capital_loss_carryover_usd"))
    lt_carry = _number(profile.get("long_term_capital_loss_carryover_usd"))
    st = float(bucket.get("short_term_capital", 0.0)) - max(st_carry or 0.0, 0.0)
    lt = float(bucket.get("long_term_capital", 0.0)) - max(lt_carry or 0.0, 0.0)
    loss_limits = _dict(policy.get("capital_loss_ordinary_income_deduction_limit_usd"))
    loss_limit = float(_number(loss_limits.get(filing_status)) or _number(loss_limits.get("default")) or 3000.0)
    capital = _net_capital(st, lt, loss_limit=loss_limit)

    ordinary_addition = (
        float(bucket.get("ordinary_investment_income", 0.0))
        + capital["net_short_term_gain_usd"]
        - capital["current_year_ordinary_loss_deduction_usd"]
    )
    preferred_addition = capital["net_long_term_gain_usd"] + float(bucket.get("qualified_dividends", 0.0))
    ordinary_brackets = _list(_dict(policy.get("ordinary_income_brackets")).get(filing_status))
    preferred_brackets = _list(_dict(policy.get("preferential_capital_gain_brackets")).get(filing_status))
    baseline_tax = _ordinary_tax(base_ordinary, ordinary_brackets) + _preferred_tax(
        base_ordinary,
        base_preferred,
        preferred_brackets,
    )
    after_ordinary = max(base_ordinary + ordinary_addition, 0.0)
    after_preferred = max(base_preferred + preferred_addition, 0.0)
    after_tax = _ordinary_tax(after_ordinary, ordinary_brackets) + _preferred_tax(
        after_ordinary,
        after_preferred,
        preferred_brackets,
    )

    niit_policy = _dict(policy.get("net_investment_income_tax"))
    niit_rate = float(_number(niit_policy.get("rate")) or 0.0)
    niit_threshold = _number(_dict(niit_policy.get("magi_threshold_usd")).get(filing_status))
    base_magi = _number(profile.get("modified_adjusted_gross_income_before_trading_usd"))
    base_nii = _number(profile.get("net_investment_income_before_trading_usd"))
    niit_exact = base_magi is not None and base_nii is not None and niit_threshold is not None
    niit_increment = None
    if niit_exact:
        baseline_niit = niit_rate * min(max(base_nii, 0.0), max(base_magi - niit_threshold, 0.0))
        nii_addition = float(bucket.get("net_investment_income", 0.0))
        magi_addition = ordinary_addition + preferred_addition
        after_niit = niit_rate * min(
            max(base_nii + nii_addition, 0.0),
            max(base_magi + magi_addition - niit_threshold, 0.0),
        )
        niit_increment = after_niit - baseline_niit

    income_tax_increment = after_tax - baseline_tax
    return {
        "income_tax_increment_usd": round(income_tax_increment, 2),
        "niit_increment_usd": round(niit_increment, 2) if niit_increment is not None else None,
        "federal_increment_usd": round(income_tax_increment + (niit_increment or 0.0), 2),
        "niit_exact": niit_exact,
        "ordinary_income_addition_usd": round(ordinary_addition, 2),
        "preferential_income_addition_usd": round(preferred_addition, 2),
        "capital_netting": {key: round(value, 2) for key, value in capital.items()},
    }


def _maximum_rate_reserve(bucket: dict[str, float], policy: dict[str, Any], filing_status: str) -> float:
    ordinary_rows = _list(_dict(policy.get("ordinary_income_brackets")).get(filing_status or "single"))
    preferred_rows = _list(_dict(policy.get("preferential_capital_gain_brackets")).get(filing_status or "single"))
    top_ordinary = max((_number(_dict(row).get("rate")) or 0.0 for row in ordinary_rows), default=0.37)
    top_preferred = max((_number(_dict(row).get("rate")) or 0.0 for row in preferred_rows), default=0.20)
    niit = float(_number(_dict(policy.get("net_investment_income_tax")).get("rate")) or 0.038)
    ordinary_positive = max(float(bucket.get("ordinary_investment_income", 0.0)), 0.0) + max(
        float(bucket.get("short_term_capital", 0.0)), 0.0
    )
    preferred_positive = max(float(bucket.get("long_term_capital", 0.0)), 0.0) + max(
        float(bucket.get("qualified_dividends", 0.0)), 0.0
    )
    return ordinary_positive * (top_ordinary + niit) + preferred_positive * (top_preferred + niit)


def _state_estimate(profile: dict[str, Any], taxable_increase: float | None) -> dict[str, Any]:
    residency = _dict(profile.get("tax_residency"))
    model = _dict(profile.get("state_tax_model"))
    state = str(residency.get("state") or "unknown").strip()
    method = str(model.get("method") or "unsupported").strip().lower()
    rate = _number(model.get("effective_rate"))
    if state.lower() in {"unknown", "", "unconfigured"}:
        return {"status": "unconfigured", "state": state or "unknown", "estimate_usd": None}
    if method == "none":
        return {"status": "verified_no_individual_income_tax_model", "state": state, "estimate_usd": 0.0}
    if method == "flat_effective" and rate is not None and taxable_increase is not None:
        return {
            "status": "configured_effective_rate_estimate",
            "state": state,
            "effective_rate": rate,
            "estimate_usd": round(max(taxable_increase, 0.0) * max(rate, 0.0), 2),
        }
    return {"status": "unsupported_state_tax_model", "state": state, "estimate_usd": None}


def evaluate(
    events: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    policy: dict[str, Any],
    account_context: dict[str, Any] | None = None,
    ledger_metadata: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = _now(now)
    requested_year = int(_number(profile.get("tax_year")) or timestamp.year)
    validation = validate_policy(policy, requested_tax_year=requested_year)
    filing_status = _filing_status(profile.get("filing_status"))
    metadata = _dict(ledger_metadata)
    ledger_complete = bool(metadata.get("complete_for_tax_year", False)) and bool(
        metadata.get("all_relevant_accounts_included", False)
    )

    base = {
        "timestamp_utc": timestamp.isoformat(),
        "schema_version": 1,
        "tax_year": requested_year,
        "jurisdiction": "US_FEDERAL",
        "policy_validation": validation,
        "filing_status": filing_status,
        "advice_status": "estimate_only_not_tax_advice",
        "live_execution_permission": False,
    }
    if not validation.get("ok"):
        return {
            **base,
            "ok": False,
            "status": "blocked_tax_policy",
            "estimate_status": "unavailable",
            "hard_blockers": validation.get("issues") or [],
            "tax_owed_from_trading_estimate_usd": None,
        }

    classified = _classify_events(
        events,
        profile=profile,
        policy=policy,
        account_context=_dict(account_context),
    )
    lower_result = _scenario_tax(
        classified["lower"],
        profile=profile,
        policy=policy,
        filing_status=filing_status,
    ) if filing_status != "unknown" else None
    upper_result = _scenario_tax(
        classified["upper"],
        profile=profile,
        policy=policy,
        filing_status=filing_status,
    ) if filing_status != "unknown" else None

    niit_rate = float(_number(_dict(policy.get("net_investment_income_tax")).get("rate")) or 0.038)
    reserve = _maximum_rate_reserve(classified["upper"], policy, filing_status if filing_status != "unknown" else "single")
    federal_low = None
    federal_high = None
    taxable_increase = None
    if lower_result is not None and upper_result is not None:
        lower_niit = lower_result.get("niit_increment_usd")
        upper_niit = upper_result.get("niit_increment_usd")
        low_income_tax = float(lower_result.get("income_tax_increment_usd") or 0.0)
        high_income_tax = float(upper_result.get("income_tax_increment_usd") or 0.0)
        lower_nii = max(float(classified["lower"].get("net_investment_income", 0.0)), 0.0)
        upper_nii = max(float(classified["upper"].get("net_investment_income", 0.0)), 0.0)
        federal_low = low_income_tax + (float(lower_niit) if lower_niit is not None else 0.0)
        federal_high = high_income_tax + (float(upper_niit) if upper_niit is not None else niit_rate * upper_nii)
        range_low = min(federal_low, federal_high)
        range_high = max(federal_low, federal_high)
        federal_low = range_low
        federal_high = range_high
        taxable_increase = max(
            float(upper_result.get("ordinary_income_addition_usd") or 0.0)
            + float(upper_result.get("preferential_income_addition_usd") or 0.0),
            0.0,
        )
        reserve = max(reserve, federal_high)

    state = _state_estimate(profile, taxable_increase)
    unresolved = _list(classified.get("unresolved"))
    required_inputs: list[str] = []
    if filing_status == "unknown":
        required_inputs.append("filing_status")
    for key in (
        "taxable_ordinary_income_before_trading_usd",
        "preferential_income_before_trading_usd",
        "modified_adjusted_gross_income_before_trading_usd",
        "net_investment_income_before_trading_usd",
        "short_term_capital_loss_carryover_usd",
        "long_term_capital_loss_carryover_usd",
    ):
        if _number(profile.get(key)) is None:
            required_inputs.append(key)
    if state.get("estimate_usd") is None:
        required_inputs.append("state_and_local_tax_policy")
    if not ledger_complete:
        required_inputs.append("complete_year_to_date_actual_broker_tax_ledger")
    if unresolved:
        required_inputs.append("resolve_event_level_tax_evidence")

    federal_exact = (
        federal_low is not None
        and federal_high is not None
        and abs(federal_high - federal_low) < 0.01
        and all("niit" not in key for key in required_inputs)
        and not unresolved
    )
    total_exact = federal_exact and state.get("estimate_usd") is not None and ledger_complete
    exact_total = None
    if total_exact:
        exact_total = max(float(federal_high or 0.0) + float(state.get("estimate_usd") or 0.0), 0.0)

    if not events:
        status = "no_tax_ledger_data"
    elif required_inputs:
        status = "needs_taxpayer_or_ledger_evidence"
    else:
        status = "ready"
    payment_threshold = float(_number(policy.get("estimated_payment_review_threshold_usd")) or 1000.0)
    reserve_complete = bool(ledger_complete and not unresolved)
    reported_reserve = max(reserve, 0.0) if reserve_complete else None
    if reported_reserve is None:
        payment_watch = "insufficient_evidence"
    else:
        payment_watch = "review_required" if reported_reserve >= payment_threshold else "below_policy_review_threshold"
    state_amount = _number(state.get("estimate_usd"))
    total_reserve = (
        max(reported_reserve + float(state_amount or 0.0), 0.0)
        if reported_reserve is not None
        else None
    )

    return {
        **base,
        "ok": True,
        "status": status,
        "estimate_status": "exact_estimate_available" if total_exact else "range_or_reserve_only",
        "tax_owed_from_trading_estimate_usd": round(exact_total, 2) if exact_total is not None else None,
        "federal": {
            "estimate_lower_usd": round(max(federal_low, 0.0), 2) if federal_low is not None else None,
            "estimate_upper_usd": round(max(federal_high, 0.0), 2) if federal_high is not None else None,
            "maximum_rate_reserve_usd": round(reported_reserve, 2) if reported_reserve is not None else None,
            "lower_scenario": lower_result,
            "upper_scenario": upper_result,
            "estimate_complete": federal_exact,
        },
        "state_and_local": state,
        "recommended_tax_reserve": {
            "amount_usd": round(total_reserve, 2) if total_reserve is not None else None,
            "includes_state_and_local": total_reserve is not None and state_amount is not None,
            "posture": (
                "conservative_max_rate_reserve_not_amount_owed"
                if total_reserve is not None
                else "unavailable_until_event_and_ledger_evidence_is_complete"
            ),
        },
        "taxable_activity": {
            "lower_scenario_buckets_usd": {key: round(value, 2) for key, value in classified["lower"].items()},
            "upper_scenario_buckets_usd": {key: round(value, 2) for key, value in classified["upper"].items()},
            "counts": classified["counts"],
        },
        "non_taxable_now": {
            "paper_realized_pnl_usd": classified["paper_realized_pnl_usd"],
            "unrealized_pnl_usd": classified["unrealized_pnl_usd"],
            "paper_note": "Paper profits are training evidence and never included in tax owed.",
            "unrealized_note": "Unrealized gains are a planning exposure, not current realized tax owed, except for verified mark-to-market regimes.",
        },
        "ledger_coverage": {
            "complete_for_tax_year": bool(metadata.get("complete_for_tax_year", False)),
            "all_relevant_accounts_included": bool(metadata.get("all_relevant_accounts_included", False)),
            "ledger_complete": ledger_complete,
            "coverage_start": metadata.get("coverage_start"),
            "coverage_end": metadata.get("coverage_end"),
            "source": metadata.get("source"),
        },
        "event_classifications": classified["classifications"][:500],
        "unresolved_evidence": unresolved[:200],
        "required_inputs": sorted(set(required_inputs)),
        "estimated_payment_watch": {
            "status": payment_watch,
            "review_threshold_usd": payment_threshold,
            "reason": (
                "Tax-payment posture is unavailable until unresolved event and ledger evidence is complete."
                if reported_reserve is None
                else "Review withholding or estimated payments when projected incremental liability is material."
            ),
        },
        "reconciliation_contract": {
            "authoritative_documents": ["Form 1099-B", "Form 1099-DIV", "Form 1099-INT", "Form 1099-DA", "Form 6781", "Schedule D"],
            "system_role": "year-to-date estimate and evidence-gap monitor",
            "filing_or_tax_advice": False,
        },
    }


def _load_ledger(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {"source": str(path), "missing": True}
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows, {"source": str(path), "format": "jsonl"}
    payload = load_json(path)
    rows = _list(payload.get("events") or payload.get("transactions") or payload.get("tax_events"))
    metadata = _dict(payload.get("coverage") or payload.get("ledger_metadata"))
    metadata.setdefault("source", str(path))
    return [_dict(row) for row in rows], metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate year-to-date trading tax with evidence-aware federal treatment.")
    parser.add_argument("--tax-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--ledger", default="")
    parser.add_argument("--policy", default="")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--account-context", default=str(DEFAULT_ACCOUNT_CONTEXT_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tax_year = int(args.tax_year)
    ledger_path = (
        Path(args.ledger).expanduser()
        if str(args.ledger or "").strip()
        else PROJECT_ROOT / "governance" / "tax" / f"trading_tax_ledger_{tax_year}_latest.json"
    )
    if str(args.policy or "").strip():
        policy_path = Path(args.policy).expanduser()
    else:
        generated = PROJECT_ROOT / "governance" / "tax" / "regulations" / f"us_federal_{tax_year}.json"
        configured = PROJECT_ROOT / "config" / f"trading_tax_policy_us_federal_{tax_year}.json"
        policy_path = generated if generated.exists() else configured
        if tax_year == 2026 and not policy_path.exists():
            policy_path = DEFAULT_POLICY_PATH
    policy = load_json(policy_path)
    profile = dict(load_json(Path(args.profile).expanduser()))
    profile_year = int(_number(profile.get("tax_year")) or 0)
    if profile_year != tax_year:
        for key in (
            "taxable_ordinary_income_before_trading_usd",
            "preferential_income_before_trading_usd",
            "modified_adjusted_gross_income_before_trading_usd",
            "net_investment_income_before_trading_usd",
            "short_term_capital_loss_carryover_usd",
            "long_term_capital_loss_carryover_usd",
            "prior_year_total_tax_usd",
            "projected_current_year_withholding_and_estimated_payments_usd",
        ):
            profile[key] = None
        profile["profile_rollover_from_tax_year"] = profile_year or None
    profile["tax_year"] = tax_year
    account_context = load_json(Path(args.account_context).expanduser())
    events, metadata = _load_ledger(ledger_path)
    payload = evaluate(
        events,
        profile=profile,
        policy=policy,
        account_context=account_context,
        ledger_metadata=metadata,
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        federal = _dict(payload.get("federal"))
        print(
            "trading_tax_estimator "
            f"status={payload.get('status')} "
            f"tax_year={payload.get('tax_year')} "
            f"federal_low={federal.get('estimate_lower_usd')} "
            f"federal_high={federal.get('estimate_upper_usd')} "
            f"reserve={_dict(payload.get('recommended_tax_reserve')).get('amount_usd')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
