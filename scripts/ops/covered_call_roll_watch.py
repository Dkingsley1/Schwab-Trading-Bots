#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "governance" / "health" / "broker_truth_shared_snapshot_schwab_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "covered_call_roll_watch_latest.json"
DEFAULT_ALERT_LATEST_PATH = PROJECT_ROOT / "governance" / "alerts" / "critical_latest_covered_call_roll_watch.json"
DEFAULT_PREFERENCE_PATH = PROJECT_ROOT / "config" / "covered_call_roll_preferences.json"
DEFAULT_ACCOUNT_ALIAS_PATH = PROJECT_ROOT / "config" / "account_aliases.json"
OCC_SYMBOL_RE = re.compile(r"^(?P<underlying>.+?)\s+(?P<yymmdd>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _parse_iso_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _account_meta(node: dict[str, Any], *, index: int) -> dict[str, Any]:
    meta = node.get("_broker_account") if isinstance(node.get("_broker_account"), dict) else {}
    label = str(meta.get("account_label") or "").strip()
    if not label:
        tail = str(meta.get("account_number_tail") or meta.get("account_reference_tail") or "").strip()
        label = f"account_{index + 1}_{tail}" if tail else f"account_{index + 1}"
    return {
        "account_key": label,
        "account_label": label,
        "account_index": int(meta.get("account_index", index) or index),
        "account_number_tail": str(meta.get("account_number_tail") or "").strip(),
        "account_reference_tail": str(meta.get("account_reference_tail") or "").strip(),
    }


def _annotated_position_rows_from_account(account: dict[str, Any], *, index: int) -> list[dict[str, Any]]:
    sec = account.get("securitiesAccount") if isinstance(account.get("securitiesAccount"), dict) else account
    if not isinstance(sec, dict):
        return []
    meta = _account_meta(sec if isinstance(sec.get("_broker_account"), dict) else account, index=index)
    rows = sec.get("positions")
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["_account_key"] = meta["account_key"]
        item["_account_label"] = meta["account_label"]
        item["_account_index"] = meta["account_index"]
        item["_account_number_tail"] = meta["account_number_tail"]
        item["_account_reference_tail"] = meta["account_reference_tail"]
        out.append(item)
    return out


def _position_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    fetched = snapshot.get("fetched") if isinstance(snapshot.get("fetched"), dict) else {}
    payload = fetched.get("payload") if isinstance(fetched.get("payload"), dict) else {}
    accounts = payload.get("accounts")
    if isinstance(accounts, list):
        out: list[dict[str, Any]] = []
        for idx, account in enumerate(accounts):
            if isinstance(account, dict):
                out.extend(_annotated_position_rows_from_account(account, index=idx))
        return out
    return _annotated_position_rows_from_account(payload, index=0)


def _instrument(row: dict[str, Any]) -> dict[str, Any]:
    inst = row.get("instrument")
    return inst if isinstance(inst, dict) else {}


def _parse_occ_symbol(raw: Any) -> dict[str, Any]:
    symbol = str(raw or "").strip().upper()
    normalized = re.sub(r"\s+", " ", symbol)
    match = OCC_SYMBOL_RE.match(normalized)
    if match is None:
        return {"ok": False, "symbol": symbol}
    yymmdd = match.group("yymmdd")
    try:
        expiration = date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    except Exception:
        return {"ok": False, "symbol": symbol}
    return {
        "ok": True,
        "symbol": symbol,
        "underlying": match.group("underlying").strip().upper(),
        "expiration": expiration,
        "right": "CALL" if match.group("right") == "C" else "PUT",
        "strike": int(match.group("strike")) / 1000.0,
    }


def _row_account_key(row: dict[str, Any]) -> str:
    return str(row.get("_account_key") or row.get("_account_label") or "account_1").strip() or "account_1"


def _account_alias_rows(account_aliases: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(account_aliases, dict):
        return {}
    scoped = account_aliases.get("schwab_accounts")
    rows = scoped if isinstance(scoped, dict) else account_aliases
    return rows if isinstance(rows, dict) else {}


def _account_alias_for(row: dict[str, Any], account_aliases: dict[str, Any] | None) -> dict[str, Any]:
    rows = _account_alias_rows(account_aliases)
    if not rows:
        return {}
    label = str(row.get("_account_label") or row.get("_account_key") or "").strip()
    key = str(row.get("_account_key") or "").strip()
    tail = str(row.get("_account_number_tail") or "").strip()
    ref_tail = str(row.get("_account_reference_tail") or "").strip()
    candidates = {item for item in (label, key, tail, ref_tail, f"tail:{tail}", f"account_tail:{tail}", f"reference_tail:{ref_tail}") if item}
    for candidate in candidates:
        raw = rows.get(candidate)
        if isinstance(raw, dict):
            return raw
    for raw in rows.values():
        if not isinstance(raw, dict):
            continue
        raw_candidates = {
            str(raw.get("account_label") or "").strip(),
            str(raw.get("broker_account_label") or "").strip(),
            str(raw.get("account_key") or "").strip(),
            str(raw.get("account_number_tail") or "").strip(),
            str(raw.get("account_reference_tail") or "").strip(),
        }
        raw_candidates = {item for item in raw_candidates if item}
        if candidates & raw_candidates:
            return raw
    return {}


def _alias_text(alias: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = str(alias.get(key) or "").strip()
        if text:
            return text
    return ""


def _equity_positions(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        inst = _instrument(row)
        if str(inst.get("assetType") or "").upper() != "EQUITY":
            continue
        symbol = str(inst.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        account_key = _row_account_key(row)
        qty = _safe_float(row.get("longQuantity"), 0.0) - _safe_float(row.get("shortQuantity"), 0.0)
        market_value = _safe_float(row.get("marketValue"), 0.0)
        px = market_value / qty if qty > 0.0 and market_value > 0.0 else _safe_float(
            row.get("currentPrice", row.get("marketPrice", row.get("averagePrice"))),
            0.0,
        )
        account_positions = out.setdefault(account_key, {})
        prior = account_positions.get(symbol)
        if prior is not None:
            prior_qty = _safe_float(prior.get("quantity"), 0.0)
            prior_value = _safe_float(prior.get("market_value"), 0.0)
            combined_qty = prior_qty + float(qty)
            combined_value = prior_value + float(market_value)
            prior["quantity"] = float(combined_qty)
            prior["market_value"] = float(combined_value)
            if combined_qty > 0.0 and combined_value > 0.0:
                prior["price"] = float(combined_value / combined_qty)
            elif px > 0.0:
                prior["price"] = float(px)
            continue
        account_positions[symbol] = {
            "quantity": float(qty),
            "market_value": float(market_value),
            "price": float(px),
        }
    return out


def _date_iso(value: date) -> str:
    return value.isoformat()


def _roll_windows(expiration: date, *, today: date, itm_pct: float, args: argparse.Namespace) -> dict[str, Any]:
    early_start = expiration - timedelta(days=_safe_int(args.early_dte, 60))
    primary_start = expiration - timedelta(days=_safe_int(args.primary_start_dte, 45))
    primary_end = expiration - timedelta(days=_safe_int(args.primary_end_dte, 21))
    assignment_watch_start = primary_end
    urgent_start = expiration - timedelta(days=_safe_int(args.urgent_dte, 14))

    early_itm = itm_pct >= float(args.itm_early_pct) / 100.0
    recommended_start = early_start if early_itm else primary_start
    recommended_end = primary_end
    active_start = today if today > recommended_start else recommended_start

    return {
        "recommended_start": _date_iso(recommended_start),
        "recommended_end": _date_iso(recommended_end),
        "active_start": _date_iso(active_start),
        "primary_start": _date_iso(primary_start),
        "primary_end": _date_iso(primary_end),
        "assignment_watch_start": _date_iso(assignment_watch_start),
        "urgent_start": _date_iso(urgent_start),
        "expiration": _date_iso(expiration),
        "early_itm_window_enabled": bool(early_itm),
    }


def _operator_preference_for(underlying: str, preferences: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(preferences, dict):
        return {}
    scoped = preferences.get("covered_call_roll_preferences")
    rows = scoped if isinstance(scoped, dict) else preferences
    key = str(underlying or "").strip().upper()
    raw = rows.get(key) if isinstance(rows, dict) else None
    return raw if isinstance(raw, dict) else {}


def _operator_roll_preference_packet(
    *,
    underlying: str,
    underlying_price: float,
    strike: float,
    preferences: dict[str, Any] | None,
) -> dict[str, Any]:
    pref = _operator_preference_for(underlying, preferences)
    wait_price = _safe_float(
        pref.get("operator_wait_for_underlying_price", pref.get("wait_for_underlying_price")),
        0.0,
    )
    if wait_price <= 0.0:
        return {}
    trigger_hit = bool(underlying_price > 0.0 and underlying_price <= wait_price + 0.005)
    return {
        "active": True,
        "operator_roll_bias": str(pref.get("operator_roll_bias") or "wait_for_pullback_before_early_roll"),
        "wait_for_underlying_price": round(wait_price, 4),
        "trigger_direction": "at_or_below",
        "trigger_hit": trigger_hit,
        "current_underlying_price": round(underlying_price, 4),
        "still_itm_at_trigger": bool(trigger_hit and underlying_price > strike),
        "safety_date_windows_override": True,
        "note": str(pref.get("note") or ""),
    }


def _classify_call(
    *,
    dte: int,
    itm_pct: float,
    underlying_price: float,
    strike: float,
    covered: bool,
    today: date,
    windows: dict[str, Any],
    operator_preference: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str, list[str]]:
    if not covered:
        return "uncovered_short_call", "critical", ["short_call_not_covered_by_100_shares"]

    recommended_start = date.fromisoformat(str(windows["recommended_start"]))
    primary_start = date.fromisoformat(str(windows["primary_start"]))
    primary_end = date.fromisoformat(str(windows["primary_end"]))
    urgent_start = date.fromisoformat(str(windows["urgent_start"]))
    deep_itm = itm_pct >= float(args.deep_itm_pct) / 100.0
    early_itm = itm_pct >= float(args.itm_early_pct) / 100.0
    wait_price = _safe_float(operator_preference.get("wait_for_underlying_price"), 0.0)
    wait_active = bool(operator_preference.get("active") and wait_price > 0.0)
    wait_hit = bool(wait_active and underlying_price > 0.0 and underlying_price <= wait_price + 0.005)

    reasons: list[str] = []
    if deep_itm:
        reasons.append(f"deep_itm={itm_pct:.2%}")
    elif early_itm:
        reasons.append(f"itm={itm_pct:.2%}")
    if today >= urgent_start:
        reasons.append(f"inside_urgent_window dte={dte}")
        return "urgent_roll_window", "critical", reasons
    if today >= primary_end and early_itm:
        reasons.append(f"inside_assignment_watch_window dte={dte}")
        return "assignment_watch_window", "critical", reasons
    if today >= recommended_start:
        reasons.append(f"inside_recommended_roll_window dte={dte}")
        return "roll_window_active", "critical", reasons
    if today >= primary_start:
        reasons.append(f"inside_primary_watch_window dte={dte}")
        return "primary_watch", "warn", reasons
    if wait_active and wait_hit:
        reasons.append(f"operator_wait_price_hit underlying={underlying_price:.2f}<=trigger={wait_price:.2f}")
        if underlying_price > strike:
            reasons.append(f"still_itm={itm_pct:.2%}")
            return "operator_price_review", "critical", reasons
        reasons.append("roll_pressure_relieved_at_or_below_strike")
        return "operator_price_hit_otm_review", "warn", reasons
    if wait_active and early_itm:
        reasons.append(f"operator_wait_price_not_hit underlying={underlying_price:.2f}>trigger={wait_price:.2f}")
        return "operator_wait_price_watch", "warn", reasons
    if early_itm:
        reasons.append(f"pre_window_itm_watch starts={windows['recommended_start']}")
        return "pre_window_itm_watch", "warn", reasons
    return "monitor", "info", ["outside_roll_window"]


def evaluate(
    snapshot: dict[str, Any],
    *,
    today: date,
    args: argparse.Namespace,
    preferences: dict[str, Any] | None = None,
    account_aliases: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _position_rows(snapshot)
    equities = _equity_positions(rows)
    calls: list[dict[str, Any]] = []
    alert_calls: list[dict[str, Any]] = []

    for row in rows:
        inst = _instrument(row)
        if str(inst.get("assetType") or "").upper() != "OPTION":
            continue
        parsed = _parse_occ_symbol(inst.get("symbol"))
        if not parsed.get("ok") or parsed.get("right") != "CALL":
            continue
        short_contracts = _safe_float(row.get("shortQuantity"), 0.0)
        if short_contracts <= 0.0:
            continue
        underlying = str(parsed["underlying"])
        account_key = _row_account_key(row)
        equity = equities.get(account_key, {}).get(underlying, {})
        shares = float(equity.get("quantity", 0.0) or 0.0)
        px = float(equity.get("price", 0.0) or 0.0)
        strike = float(parsed["strike"])
        expiration = parsed["expiration"]
        assert isinstance(expiration, date)
        dte = (expiration - today).days
        if dte < 0:
            continue
        account_alias = _account_alias_for(row, account_aliases)
        covered_shares_required = int(math.ceil(short_contracts * 100.0))
        covered = shares + 1e-9 >= covered_shares_required
        itm_pct = (px / strike) - 1.0 if px > 0.0 and strike > 0.0 else 0.0
        option_market_value = abs(_safe_float(row.get("marketValue"), 0.0))
        mark_per_share = option_market_value / max(short_contracts * 100.0, 1.0)
        avg_credit = _safe_float(row.get("averagePrice"), 0.0)
        pnl_est = (avg_credit - mark_per_share) * short_contracts * 100.0
        windows = _roll_windows(expiration, today=today, itm_pct=itm_pct, args=args)
        operator_preference = _operator_roll_preference_packet(
            underlying=underlying,
            underlying_price=px,
            strike=strike,
            preferences=preferences,
        )
        status, severity, reasons = _classify_call(
            dte=dte,
            itm_pct=itm_pct,
            underlying_price=px,
            strike=strike,
            covered=covered,
            today=today,
            windows=windows,
            operator_preference=operator_preference,
            args=args,
        )
        packet = {
            "underlying": underlying,
            "account_label": str(row.get("_account_label") or account_key),
            "account_index": _safe_int(row.get("_account_index"), 0),
            "operator_account_label": _alias_text(account_alias, "operator_account_label", "label", "name"),
            "operator_account_kind": _alias_text(account_alias, "operator_account_kind", "account_kind", "kind"),
            "operator_trading_type": _alias_text(account_alias, "trading_type", "operator_trading_type"),
            "option_symbol": str(parsed["symbol"]),
            "right": "CALL",
            "strike": round(strike, 4),
            "expiration": _date_iso(expiration),
            "dte": int(dte),
            "short_contracts": float(short_contracts),
            "covered": bool(covered),
            "shares": round(shares, 4),
            "covered_shares_required": covered_shares_required,
            "underlying_price": round(px, 4),
            "moneyness_pct": round(itm_pct * 100.0, 4),
            "short_avg_credit_per_share": round(avg_credit, 4),
            "short_mark_per_share": round(mark_per_share, 4),
            "unrealized_short_option_pnl_est": round(pnl_est, 2),
            "roll_window": windows,
            "operator_roll_preference": operator_preference,
            "status": status,
            "severity": severity,
            "reasons": reasons,
            "operator_note": "Paper advisory only; do not auto-place a roll order.",
        }
        calls.append(packet)
        if severity == "critical" and status != "monitor":
            alert_calls.append(packet)

    if alert_calls:
        overall_status = "critical"
        ok = False
    elif any(
        str(call.get("status")) in {"pre_window_itm_watch", "primary_watch", "operator_wait_price_watch"}
        for call in calls
    ):
        overall_status = "watch"
        ok = True
    else:
        overall_status = "ready"
        ok = True

    next_call = None
    if calls:
        next_call = sorted(
            calls,
            key=lambda item: (
                date.fromisoformat(str((item.get("roll_window") or {}).get("recommended_start", "9999-12-31"))),
                int(item.get("dte", 9999) or 9999),
            ),
        )[0]

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(ok),
        "overall_status": overall_status,
        "source": str(DEFAULT_SNAPSHOT_PATH),
        "operator_preferences_source": str(DEFAULT_PREFERENCE_PATH),
        "account_aliases_source": str(DEFAULT_ACCOUNT_ALIAS_PATH),
        "today": _date_iso(today),
        "position_rows": len(rows),
        "account_count": len({str(row.get("_account_key") or "account_1") for row in rows}) if rows else 0,
        "covered_call_count": len(calls),
        "alert_count": len(alert_calls),
        "next_roll_window": (next_call or {}).get("roll_window", {}) if next_call else {},
        "next_roll_underlying": (next_call or {}).get("underlying", "") if next_call else "",
        "covered_calls": calls,
        "recommended_actions": _recommended_actions(calls, alert_calls),
    }


def _recommended_actions(calls: list[dict[str, Any]], alert_calls: list[dict[str, Any]]) -> list[str]:
    if alert_calls:
        active = alert_calls[0]
        preference = active.get("operator_roll_preference") if isinstance(active.get("operator_roll_preference"), dict) else {}
        if str(active.get("status")) == "operator_price_review":
            return [
                f"{active.get('underlying')} is at or below the operator roll-review trigger {preference.get('wait_for_underlying_price')}",
                "review the covered-call roll ticket manually; auto-ordering remains disabled",
            ]
        return [
            "review the active covered-call roll candidates before market close",
            "keep the decision advisory-only unless an operator explicitly approves an order ticket",
        ]
    if calls:
        call = sorted(calls, key=lambda row: str((row.get("roll_window") or {}).get("recommended_start", "")))[0]
        window = call.get("roll_window") if isinstance(call.get("roll_window"), dict) else {}
        preference = call.get("operator_roll_preference") if isinstance(call.get("operator_roll_preference"), dict) else {}
        if preference.get("active") and not preference.get("trigger_hit"):
            return [
                f"wait for {call.get('underlying')} at or below {preference.get('wait_for_underlying_price')} before voluntary early roll review",
                f"date windows still override: review starts {window.get('recommended_start')} and primary window starts {window.get('primary_start')}",
            ]
        return [
            f"next roll review for {call.get('underlying')} starts {window.get('recommended_start')} and primary window starts {window.get('primary_start')}",
            "continue monitoring moneyness, DTE, ex-dividend/event risk, and available roll credit",
        ]
    return ["no covered short calls detected in the latest Schwab position snapshot"]


def _alert_message(payload: dict[str, Any]) -> str:
    rows = payload.get("covered_calls") if isinstance(payload.get("covered_calls"), list) else []
    active = [row for row in rows if str(row.get("severity")) == "critical"]
    if not active:
        return ""
    row = active[0]
    window = row.get("roll_window") if isinstance(row.get("roll_window"), dict) else {}
    preference = row.get("operator_roll_preference") if isinstance(row.get("operator_roll_preference"), dict) else {}
    if str(row.get("status")) == "operator_price_review":
        return (
            f"{row.get('underlying')} covered call operator roll-review trigger hit\n"
            f"Contract: {row.get('strike')}C exp {row.get('expiration')} ({row.get('dte')} DTE)\n"
            f"Underlying: {row.get('underlying_price')} <= trigger {preference.get('wait_for_underlying_price')}\n"
            f"Moneyness: {row.get('moneyness_pct')}%; advisory only, no auto-order"
        )
    return (
        f"{row.get('underlying')} covered call roll window active\n"
        f"Contract: {row.get('strike')}C exp {row.get('expiration')} ({row.get('dte')} DTE)\n"
        f"Range: {window.get('recommended_start')} to {window.get('recommended_end')}; "
        f"primary {window.get('primary_start')} to {window.get('primary_end')}\n"
        f"Moneyness: {row.get('moneyness_pct')}%; status={row.get('status')}"
    )


def write_alert(payload: dict[str, Any], *, alert_path: Path = DEFAULT_ALERT_LATEST_PATH) -> None:
    message = _alert_message(payload)
    if not message:
        try:
            alert_path.unlink()
        except FileNotFoundError:
            pass
        return
    alert = {
        "timestamp_utc": payload["timestamp_utc"],
        "event": "covered_call_roll_watch",
        "severity": "critical",
        "broker": "schwab",
        "profile": "covered_call_roll_watch",
        "message": message,
        "covered_call_count": payload.get("covered_call_count", 0),
        "alert_count": payload.get("alert_count", 0),
        "next_roll_window": payload.get("next_roll_window", {}),
        "operator_only": True,
        "auto_order_enabled": False,
    }
    write_payload(alert_path, alert)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch held Schwab covered calls and publish roll-window alerts.")
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--alert-file", default=str(DEFAULT_ALERT_LATEST_PATH))
    parser.add_argument("--preference-file", default=str(DEFAULT_PREFERENCE_PATH))
    parser.add_argument("--account-alias-file", default=str(DEFAULT_ACCOUNT_ALIAS_PATH))
    parser.add_argument("--today", default="")
    parser.add_argument("--early-dte", type=int, default=60)
    parser.add_argument("--primary-start-dte", type=int, default=45)
    parser.add_argument("--primary-end-dte", type=int, default=21)
    parser.add_argument("--urgent-dte", type=int, default=14)
    parser.add_argument("--itm-early-pct", type=float, default=2.0)
    parser.add_argument("--deep-itm-pct", type=float, default=8.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    today = _parse_iso_date(args.today) or datetime.now(timezone.utc).date()
    snapshot_path = Path(args.snapshot_path)
    preference_path = Path(args.preference_file)
    account_alias_path = Path(args.account_alias_file)
    preferences = load_json(preference_path)
    account_aliases = load_json(account_alias_path)
    payload = evaluate(
        load_json(snapshot_path),
        today=today,
        args=args,
        preferences=preferences,
        account_aliases=account_aliases,
    )
    payload["source"] = str(snapshot_path)
    payload["operator_preferences_source"] = str(preference_path)
    payload["account_aliases_source"] = str(account_alias_path)
    out_path = Path(args.out_file)
    write_payload(out_path, payload)
    write_alert(payload, alert_path=Path(args.alert_file))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        window = payload.get("next_roll_window") if isinstance(payload.get("next_roll_window"), dict) else {}
        print(
            "covered_call_roll_watch "
            f"status={payload.get('overall_status')} "
            f"covered_calls={payload.get('covered_call_count')} "
            f"next={payload.get('next_roll_underlying') or 'none'} "
            f"range={window.get('recommended_start', '')}..{window.get('recommended_end', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
