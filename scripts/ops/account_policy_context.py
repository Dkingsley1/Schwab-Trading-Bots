#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "account_policy_context_latest.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "account_policy_registry.json"

FINRA_INTRADAY_MARGIN_EFFECTIVE_DATE = date(2026, 6, 4)
SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE = date(2026, 6, 8)
FINRA_INTRADAY_MARGIN_PHASE_IN_END_DATE = date(2027, 10, 20)

REGULATORY_SOURCE_REFERENCES = [
    {
        "source": "FINRA Regulatory Notice 26-10",
        "url": "https://business.cch.com/srd/RegulatoryNotice26-10_FINRAorg042126.pdf",
        "reason": "FINRA adopted intraday margin standards replacing day-trading margin requirements effective 2026-06-04, with phase-in through 2027-10-20.",
    },
    {
        "source": "SEC Release No. 34-105226",
        "url": "https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf",
        "reason": "SEC approval order for FINRA Rule 4210 amendments replacing PDT/day-trading buying power with intraday margin requirements.",
    },
    {
        "source": "Charles Schwab day-trading rule update",
        "url": "https://www.schwab.com/learn/story/schwab-changes-rules-around-day-trading",
        "reason": "Broker-specific implementation date and account-treatment guidance for Schwab customers.",
    },
    {
        "source": "Investor.gov Pattern Day Trader update",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/glossary/pattern-day-trader",
        "reason": "Investor-facing reminder that brokerage firms may transition at different times during the phase-in period.",
    },
]

DEFAULT_ACCOUNT_SLOTS = [
    {
        "account_policy_key": "schwab_roth_ira_primary",
        "account_label": "Roth IRA",
        "account_type": "roth",
        "tax_treatment": "tax_advantaged",
        "broker": "schwab",
        "env_names": [
            "SCHWAB_ACCOUNT_HASH",
            "SCHWAB_ROTH_ACCOUNT_HASH",
            "SCHWAB_ROTH_IRA_ACCOUNT_HASH",
            "SCHWAB_ROTH_ACCOUNT_NUMBER",
            "SCHWAB_ROTH_IRA_ACCOUNT_NUMBER",
        ],
    },
    {
        "account_policy_key": "schwab_cash_account_1",
        "account_label": "Cash Account 1",
        "account_type": "cash",
        "tax_treatment": "taxable",
        "broker": "schwab",
        "env_names": [
            "SCHWAB_CASH_ACCOUNT_1_HASH",
            "SCHWAB_TAXABLE_ACCOUNT_1_HASH",
            "SCHWAB_CASH_ACCOUNT_1_NUMBER",
            "SCHWAB_TAXABLE_ACCOUNT_1_NUMBER",
        ],
    },
    {
        "account_policy_key": "schwab_cash_account_2",
        "account_label": "Cash Account 2",
        "account_type": "cash",
        "tax_treatment": "taxable",
        "broker": "schwab",
        "env_names": [
            "SCHWAB_CASH_ACCOUNT_2_HASH",
            "SCHWAB_TAXABLE_ACCOUNT_2_HASH",
            "SCHWAB_CASH_ACCOUNT_2_NUMBER",
            "SCHWAB_TAXABLE_ACCOUNT_2_NUMBER",
        ],
    },
]


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _coerce_date(raw: Any | None) -> date:
    if raw is None:
        return datetime.now(timezone.utc).date()
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    text = str(raw).strip()
    if not text:
        return datetime.now(timezone.utc).date()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        return date.fromisoformat(text[:10])


def _pdt_transition_contract(as_of_date: date) -> dict[str, Any]:
    if as_of_date < FINRA_INTRADAY_MARGIN_EFFECTIVE_DATE:
        phase = "legacy_pdt_until_finra_effective_date"
        legacy_pdt_framework_active = True
        schwab_day_trade_count_retired = False
        broker_implementation_watch_required = True
        next_broker_milestone = SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE.isoformat()
    elif as_of_date < SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE:
        phase = "finra_effective_schwab_cutover_pending"
        legacy_pdt_framework_active = True
        schwab_day_trade_count_retired = False
        broker_implementation_watch_required = True
        next_broker_milestone = SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE.isoformat()
    elif as_of_date <= FINRA_INTRADAY_MARGIN_PHASE_IN_END_DATE:
        phase = "schwab_day_trade_count_retired_intraday_margin_phase_in"
        legacy_pdt_framework_active = False
        schwab_day_trade_count_retired = True
        broker_implementation_watch_required = True
        next_broker_milestone = FINRA_INTRADAY_MARGIN_PHASE_IN_END_DATE.isoformat()
    else:
        phase = "intraday_margin_phase_in_complete"
        legacy_pdt_framework_active = False
        schwab_day_trade_count_retired = True
        broker_implementation_watch_required = False
        next_broker_milestone = ""
    return {
        "active": True,
        "as_of_date": as_of_date.isoformat(),
        "phase": phase,
        "finra_effective_date": FINRA_INTRADAY_MARGIN_EFFECTIVE_DATE.isoformat(),
        "schwab_day_trade_count_retire_date": SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE.isoformat(),
        "phase_in_end_date": FINRA_INTRADAY_MARGIN_PHASE_IN_END_DATE.isoformat(),
        "legacy_pdt_framework_active_for_schwab_policy": legacy_pdt_framework_active,
        "schwab_day_trade_count_retired": schwab_day_trade_count_retired,
        "broker_implementation_watch_required": broker_implementation_watch_required,
        "next_broker_milestone": next_broker_milestone,
        "system_posture": (
            "do_not_widen_day_trading_until_broker_intraday_margin_buying_power_is_observed"
            if broker_implementation_watch_required or not schwab_day_trade_count_retired
            else "broker_day_trade_count_retired_but_intraday_margin_buying_power_still_required"
        ),
        "legacy_pdt_count_rule": {
            "day_trade_count_window_business_days": 5,
            "legacy_day_trade_count_threshold": 4,
            "legacy_trade_share_threshold": 0.06,
            "legacy_minimum_equity_usd": 25000,
            "used_for_live_widening": False,
        },
        "new_intraday_margin_rule": {
            "uses_intraday_margin_level": True,
            "requires_real_time_intraday_buying_power_or_margin_deficit_context": True,
            "supplements_regular_margin_requirements": True,
            "covers_intraday_exposure_rather_than_day_trade_frequency": True,
        },
        "broker_specific_controls": {
            "broker": "schwab",
            "stop_counting_day_trades_date": SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE.isoformat(),
            "requires_broker_confirmed_intraday_margin_buying_power": True,
            "requires_house_margin_and_account_eligibility_check": True,
            "live_execution_permission_unchanged": True,
        },
        "legacy_day_trade_count_warning": {
            "active": True,
            "retained_until_broker_intraday_margin_fields_are_observed": True,
            "purpose": "legacy warning and audit context only; not a live widening permission",
        },
        "risk_notes": [
            "Rule change removes the old PDT frequency/equity framework, not the risks of intraday leverage.",
            "Broker house margin, account eligibility, and real-time buying power can still constrain trading.",
            "Cash, IRA, and margin accounts must remain separated in policy and sizing.",
        ],
        "source_references": REGULATORY_SOURCE_REFERENCES,
    }


def _latest_schwab_account_metrics(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    rows = sorted(
        health.glob("broker_truth_*_schwab_latest.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    latest_metrics_record: dict[str, Any] | None = None
    for path in rows:
        payload = load_json(path)
        metrics = _extract_account_metrics_from_broker_truth(payload)
        if not isinstance(metrics, dict) or not metrics:
            continue
        try:
            age_seconds = max(datetime.now(timezone.utc).timestamp() - float(path.stat().st_mtime), 0.0)
        except Exception:
            age_seconds = 0.0
        fetched = payload.get("fetched") if isinstance(payload.get("fetched"), dict) else {}
        record = {
            "source_path": str(path),
            "source_age_seconds": round(age_seconds, 3),
            "metrics": metrics,
            "status": str(payload.get("status") or payload.get("overall_status") or ("ok" if fetched.get("ok") else "")),
        }
        if _intraday_buying_power_from_metrics(metrics)[1]:
            return record
        if latest_metrics_record is None:
            latest_metrics_record = record
    if latest_metrics_record is not None:
        return latest_metrics_record
    return {
        "source_path": "",
        "source_age_seconds": None,
        "metrics": {},
        "status": "missing",
    }


def _first_positive_from_balances(balance_rows: list[dict[str, Any]], keys: list[str]) -> float:
    for row in balance_rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = _safe_float(row.get(key), 0.0)
            if value > 0.0:
                return value
    return 0.0


def _extract_account_metrics_from_broker_truth(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("account_metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    snapshot_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    fetched = payload.get("fetched") if isinstance(payload.get("fetched"), dict) else {}
    if isinstance(fetched.get("payload"), dict):
        snapshot_payload = fetched["payload"]
    account = snapshot_payload.get("securitiesAccount") if isinstance(snapshot_payload, dict) else {}
    if not isinstance(account, dict):
        return {}
    current_balances = account.get("currentBalances") if isinstance(account.get("currentBalances"), dict) else {}
    projected_balances = account.get("projectedBalances") if isinstance(account.get("projectedBalances"), dict) else {}
    initial_balances = account.get("initialBalances") if isinstance(account.get("initialBalances"), dict) else {}
    balance_rows = [current_balances, projected_balances, initial_balances]
    day_trading_buying_power = _first_positive_from_balances(
        balance_rows,
        ["dayTradingBuyingPower", "intradayBuyingPower", "intradayMarginBuyingPower"],
    )
    return {
        "equity": _first_positive_from_balances(balance_rows, ["liquidationValue", "equity", "accountValue"]),
        "cash_balance": _first_positive_from_balances(
            balance_rows,
            ["cashBalance", "cashAvailableForTrading", "moneyMarketFund", "totalCash"],
        ),
        "buying_power": _first_positive_from_balances(
            [current_balances, projected_balances, initial_balances],
            ["buyingPower", "buyingPowerNonMarginableTrade", "stockBuyingPower"],
        ),
        "available_funds": _first_positive_from_balances(
            balance_rows,
            ["availableFunds", "availableFundsNonMarginableTrade", "cashAvailableForTrading"],
        ),
        "dayTradingBuyingPower": day_trading_buying_power,
        "intraday_buying_power": day_trading_buying_power,
    }


def _intraday_buying_power_from_metrics(metrics: dict[str, Any]) -> tuple[float, str]:
    for key in (
        "intraday_buying_power",
        "intraday_margin_buying_power",
        "day_trading_buying_power",
        "dayTradingBuyingPower",
    ):
        value = _safe_float(metrics.get(key), 0.0)
        if value > 0.0:
            return value, key
    return 0.0, ""


def _intraday_margin_probe_contract(
    project_root: Path,
    *,
    as_of_date: date,
    pdt_contract: dict[str, Any],
) -> dict[str, Any]:
    broker_metrics = _latest_schwab_account_metrics(project_root)
    metrics = broker_metrics.get("metrics") if isinstance(broker_metrics.get("metrics"), dict) else {}
    intraday_buying_power, intraday_key = _intraday_buying_power_from_metrics(metrics)
    available_funds = _safe_float(metrics.get("available_funds"), 0.0)
    buying_power = _safe_float(metrics.get("buying_power"), 0.0)
    cash_balance = _safe_float(metrics.get("cash_balance"), 0.0)
    equity = _safe_float(metrics.get("equity"), 0.0)
    observed = bool(intraday_key)
    schwab_cutover_reached = as_of_date >= SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE
    if not schwab_cutover_reached:
        status = "scheduled_pre_schwab_cutover"
    elif observed:
        status = "ready"
    else:
        status = "needs_broker_intraday_margin_probe"
    return {
        "active": True,
        "status": status,
        "probe_required_after": SCHWAB_DAY_TRADE_COUNT_RETIRE_DATE.isoformat(),
        "probe_required_now": bool(schwab_cutover_reached and not observed),
        "broker": "schwab",
        "broker_truth_source_path": broker_metrics.get("source_path", ""),
        "broker_truth_source_age_seconds": broker_metrics.get("source_age_seconds"),
        "broker_truth_status": broker_metrics.get("status", ""),
        "intraday_buying_power_observed": observed,
        "intraday_buying_power_source_key": intraday_key,
        "intraday_buying_power": round(intraday_buying_power, 6),
        "available_funds": round(available_funds, 6),
        "buying_power": round(buying_power, 6),
        "cash_balance": round(cash_balance, 6),
        "equity": round(equity, 6),
        "phase": str(pdt_contract.get("phase") or ""),
        "exact_command": ["./scripts/ops/opsctl.sh", "account-policy-context", "--json"],
        "expected_impact": "keeps live-micro/day-trading widening blocked until Schwab exposes broker-confirmed intraday margin buying power",
        "risk_level": "low",
        "when_to_stop": "stop when intraday_buying_power_observed=true from a fresh Schwab broker-truth account metrics artifact",
    }


def _paper_intraday_margin_deficit_simulator(
    *,
    probe_contract: dict[str, Any],
    pdt_contract: dict[str, Any],
) -> dict[str, Any]:
    intraday_bp = _safe_float(probe_contract.get("intraday_buying_power"), 0.0)
    available_funds = _safe_float(probe_contract.get("available_funds"), 0.0)
    buying_power = _safe_float(probe_contract.get("buying_power"), 0.0)
    simulated_exposure = max(_safe_float(os.getenv("PAPER_INTRADAY_MARGIN_SIM_EXPOSURE_USD"), 0.0), 0.0)
    usable_power = intraday_bp or available_funds or buying_power
    deficit = max(simulated_exposure - usable_power, 0.0)
    return {
        "active": True,
        "mode": "paper_only_intraday_margin_deficit_simulator",
        "status": "ready" if deficit <= 0.0 else "deficit_simulated",
        "live_execution_allowed": False,
        "simulated_intraday_exposure_usd": round(simulated_exposure, 6),
        "usable_intraday_power_proxy_usd": round(usable_power, 6),
        "simulated_margin_deficit_usd": round(deficit, 6),
        "intraday_buying_power_observed": bool(probe_contract.get("intraday_buying_power_observed", False)),
        "phase": str(pdt_contract.get("phase") or ""),
        "policy": "simulate margin deficit in paper only before any future live-micro review",
        "stop_condition": "simulated_margin_deficit_usd is 0 and broker-confirmed intraday buying power is observed",
    }


def _slot_margin_policy(row: dict[str, Any], pdt_contract: dict[str, Any]) -> dict[str, Any]:
    account_type = str(row.get("account_type") or "unknown").strip().lower()
    broker = str(row.get("broker") or "unknown").strip().lower()
    margin_enabled = _as_bool(row.get("margin_enabled"), default=account_type == "margin")
    if account_type in {"cash", "roth", "ira", "traditional_ira"} or not margin_enabled:
        applicability = "not_margin_day_trading_account"
        day_trade_widening_allowed = False
    elif broker == "schwab" and bool(pdt_contract.get("schwab_day_trade_count_retired", False)):
        applicability = "schwab_intraday_margin_framework_pending_buying_power_confirmation"
        day_trade_widening_allowed = False
    else:
        applicability = "legacy_or_unconfirmed_margin_day_trading_policy"
        day_trade_widening_allowed = False
    return {
        "account_policy_key": row.get("account_policy_key"),
        "broker": broker,
        "account_type": account_type,
        "margin_enabled": margin_enabled,
        "pdt_or_intraday_margin_applicability": applicability,
        "legacy_pdt_framework_active_for_slot": bool(pdt_contract.get("legacy_pdt_framework_active_for_schwab_policy", True))
        if broker == "schwab"
        else True,
        "requires_intraday_margin_buying_power_confirmation": margin_enabled,
        "day_trade_widening_allowed": day_trade_widening_allowed,
        "operator_confirmation_required": True,
        "live_execution_permission_unchanged": True,
    }


def _slot_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    env_names = [str(item) for item in _as_list(raw.get("env_names") or raw.get("env_bindings")) if str(item)]
    env_bindings = []
    for raw_name in env_names:
        name = str(raw_name.get("name") if isinstance(raw_name, dict) else raw_name).strip()
        if not name:
            continue
        env_bindings.append({"name": name, "present": bool(os.environ.get(name))})
    return {
        "account_policy_key": str(raw.get("account_policy_key") or raw.get("key") or "unknown_account"),
        "account_label": str(raw.get("account_label") or raw.get("label") or "Unknown Account"),
        "account_type": str(raw.get("account_type") or "unknown"),
        "tax_treatment": str(raw.get("tax_treatment") or "unknown"),
        "broker": str(raw.get("broker") or "unknown"),
        "margin_enabled": _as_bool(raw.get("margin_enabled"), default=str(raw.get("account_type") or "").lower() == "margin"),
        "env_bindings": env_bindings,
        "bot_visible": bool(raw.get("bot_visible", True)),
        "auto_order_enabled": bool(raw.get("auto_order_enabled", False)),
        "requires_operator_confirmation": bool(raw.get("requires_operator_confirmation", True)),
    }


def _load_registry_slots(registry_path: Path) -> tuple[list[dict[str, Any]], bool]:
    registry = load_json(registry_path)
    rows = _as_list(registry.get("account_slots") or registry.get("configured_account_slots"))
    if rows:
        return [_slot_from_raw(row) for row in rows if isinstance(row, dict)], True
    return [_slot_from_raw(row) for row in DEFAULT_ACCOUNT_SLOTS], False


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    as_of_date: Any | None = None,
) -> dict[str, Any]:
    slots, registry_present = _load_registry_slots(registry_path)
    effective_date = _coerce_date(as_of_date)
    pdt_contract = _pdt_transition_contract(effective_date)
    probe_contract = _intraday_margin_probe_contract(project_root, as_of_date=effective_date, pdt_contract=pdt_contract)
    margin_simulator = _paper_intraday_margin_deficit_simulator(
        probe_contract=probe_contract,
        pdt_contract=pdt_contract,
    )
    slot_margin_policies = [_slot_margin_policy(row, pdt_contract) for row in slots]
    margin_slots = sum(1 for row in slot_margin_policies if bool(row.get("margin_enabled", False)))
    roth_slots = sum(1 for row in slots if row.get("account_type") == "roth")
    cash_slots = sum(1 for row in slots if row.get("account_type") == "cash")
    auto_order_enabled = any(bool(row.get("auto_order_enabled", False)) for row in slots)
    missing_bindings = [
        {
            "account_policy_key": row.get("account_policy_key"),
            "env_name": binding.get("name"),
        }
        for row in slots
        for binding in _as_list(row.get("env_bindings"))
        if isinstance(binding, dict) and not bool(binding.get("present", False))
    ]
    next_actions = []
    if missing_bindings:
        next_actions.append("set account hash environment variables when live-micro account binding is intentionally approved")
    if auto_order_enabled:
        next_actions.append("turn off account auto-order flags before running any paper-to-live review")
    if bool(pdt_contract.get("broker_implementation_watch_required", False)):
        next_actions.append("keep Schwab day-trading widening under broker-cutover watch until the June 8 Schwab policy change is verified")
    next_actions.append("require broker-reported intraday margin buying power before any future live day-trading widening")
    if bool(probe_contract.get("probe_required_now", False)):
        next_actions.append("refresh Schwab broker-truth account metrics and confirm intraday buying power before live-micro review")
    overall_status = "ready" if slots and not auto_order_enabled else "blocked"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "account_policy_context": {
            "schema_version": 1,
            "registry_path": str(registry_path),
            "registry_present": registry_present,
            "configured_account_slots": slots,
            "slot_count": len(slots),
            "unmatched_schwab_cash_fallback": {
                "enabled": True,
                "requires_anchor_match": True,
                "account_label_prefix": "Cash Account",
            },
            "redaction_contract": {
                "account_numbers_exposed_in_policy": False,
                "account_hashes_exposed_in_policy": False,
                "bot_context_key": "bot_visible_account_context",
                "auto_order_enabled_default": False,
                "operator_confirmation_default": True,
            },
            "pdt_intraday_margin_transition": pdt_contract,
            "intraday_margin_probe_contract": probe_contract,
            "paper_intraday_margin_deficit_simulator": margin_simulator,
            "slot_margin_policies": slot_margin_policies,
        },
        "coverage": {
            "roth_slots": roth_slots,
            "cash_slots": cash_slots,
            "margin_slots": margin_slots,
            "configured_account_slots": len(slots),
            "target_roth_slots": 1,
            "target_cash_slots": 2,
        },
        "bot_contract": {
            "bots_should_read": "bot_visible_account_context",
            "raw_account_numbers_required_for_policy": False,
            "raw_account_hashes_required_for_policy": False,
            "auto_order_enabled": auto_order_enabled,
            "operator_confirmation_required": True,
            "supported_account_types": sorted({str(row.get("account_type") or "unknown") for row in slots} | {"unknown"}),
            "day_trading_rule_awareness": "finra_intraday_margin_replaces_legacy_pdt",
            "pdt_transition_phase": str(pdt_contract.get("phase") or ""),
            "legacy_pdt_framework_active_for_schwab_policy": bool(
                pdt_contract.get("legacy_pdt_framework_active_for_schwab_policy", True)
            ),
            "schwab_day_trade_count_retired": bool(pdt_contract.get("schwab_day_trade_count_retired", False)),
            "intraday_margin_buying_power_required": True,
            "intraday_margin_buying_power_observed": bool(probe_contract.get("intraday_buying_power_observed", False)),
            "intraday_margin_probe_status": str(probe_contract.get("status") or ""),
            "paper_intraday_margin_simulator_status": str(margin_simulator.get("status") or ""),
            "day_trade_widening_allowed": False,
            "live_execution_permission_unchanged": True,
            "broker_developer_platform_order_limit_policy": "operator_managed_external_throttle_not_internal_scalability_ceiling",
        },
        "missing_env_bindings": missing_bindings[:40],
        "next_actions": next_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh redacted account policy context for bots and income-readiness controls.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--as-of-date", default="", help="Override policy date for testing/regression checks (YYYY-MM-DD).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    registry_path = Path(args.registry_path).expanduser()
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    payload = build_payload(project_root, registry_path=registry_path, as_of_date=args.as_of_date or None)
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "account_policy_context "
            f"status={payload.get('overall_status')} "
            f"slots={payload.get('coverage', {}).get('configured_account_slots')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
