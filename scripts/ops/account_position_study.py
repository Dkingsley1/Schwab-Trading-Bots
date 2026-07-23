#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.covered_call_roll_watch import (
        DEFAULT_ACCOUNT_ALIAS_PATH,
        _account_alias_for,
        _alias_text,
        _instrument,
        _parse_occ_symbol,
        _position_rows,
        _safe_float,
    )
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from .covered_call_roll_watch import (
        DEFAULT_ACCOUNT_ALIAS_PATH,
        _account_alias_for,
        _alias_text,
        _instrument,
        _parse_occ_symbol,
        _position_rows,
        _safe_float,
    )
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "governance" / "health" / "broker_truth_shared_snapshot_schwab_latest.json"
DEFAULT_ROLL_WATCH_PATH = PROJECT_ROOT / "governance" / "health" / "covered_call_roll_watch_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "account_position_study_latest.json"
DEFAULT_PROFILES = (
    "aggressive_equities_schwab",
    "conservative_equities_schwab",
    "dividend_equities_schwab",
    "swing_aggressive_equities_schwab",
)
DEFAULT_DECISION_TAIL_BYTES = 32 * 1024 * 1024
SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _position_qty(row: dict[str, Any]) -> float:
    if "netQuantity" in row:
        return _safe_float(row.get("netQuantity"), 0.0)
    if "quantity" in row:
        return _safe_float(row.get("quantity"), 0.0)
    return _safe_float(row.get("longQuantity"), 0.0) - _safe_float(row.get("shortQuantity"), 0.0)


def _position_underlying(row: dict[str, Any]) -> str:
    inst = _instrument(row)
    symbol = str(inst.get("symbol") or "").strip().upper()
    asset_type = str(inst.get("assetType") or "").strip().upper()
    if asset_type == "OPTION":
        explicit = str(inst.get("underlyingSymbol") or "").strip().upper()
        if explicit:
            return explicit
        parsed = _parse_occ_symbol(symbol)
        if parsed.get("ok"):
            return str(parsed.get("underlying") or "").strip().upper()
    return symbol


def _positions(snapshot: dict[str, Any], account_aliases: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _position_rows(snapshot):
        inst = _instrument(row)
        symbol = str(inst.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        qty = _position_qty(row)
        if abs(qty) <= 0.0:
            continue
        alias = _account_alias_for(row, account_aliases)
        asset_type = str(inst.get("assetType") or "EQUITY").strip().upper()
        out.append(
            {
                "account_label": str(row.get("_account_label") or "account_1"),
                "account_index": int(row.get("_account_index", 0) or 0),
                "operator_account_label": _alias_text(alias, "operator_account_label", "label", "name"),
                "operator_account_kind": _alias_text(alias, "operator_account_kind", "account_kind", "kind"),
                "operator_trading_type": _alias_text(alias, "trading_type", "operator_trading_type"),
                "symbol": symbol,
                "underlying": _position_underlying(row),
                "asset_type": asset_type,
                "quantity": round(qty, 6),
                "long_quantity": round(_safe_float(row.get("longQuantity"), 0.0), 6),
                "short_quantity": round(_safe_float(row.get("shortQuantity"), 0.0), 6),
                "market_value": round(_safe_float(row.get("marketValue"), 0.0), 4),
                "average_price": round(_safe_float(row.get("averagePrice"), 0.0), 4),
            }
        )
    return out


def _decision_paths(profiles: list[str], day: str) -> list[Path]:
    root = PROJECT_ROOT / "governance" / "channels" / "decision"
    return [root / profile / f"decision_{day}.jsonl" for profile in profiles]


def _recent_lines(path: Path, *, max_bytes: int = DEFAULT_DECISION_TAIL_BYTES) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(size - max(int(max_bytes), 1), 0)
            handle.seek(start)
            data = handle.read()
    except Exception:
        return []
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return lines


def _latest_decision_context(symbols: set[str], profiles: list[str], day: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not symbols:
        return latest
    symbol_tokens = {symbol.upper() for symbol in symbols if symbol}
    for path in _decision_paths(profiles, day):
        if not path.exists():
            continue
        profile = path.parent.name
        remaining = set(symbol_tokens)
        for line in reversed(_recent_lines(path)):
            if not remaining:
                break
            if not any(token in line for token in remaining):
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol not in remaining:
                continue
            ts = str(row.get("timestamp_utc") or "")
            prior = latest.get(symbol)
            if prior and str(prior.get("timestamp_utc") or "") >= ts:
                remaining.remove(symbol)
                continue
            market = row.get("market") if isinstance(row.get("market"), dict) else {}
            grand = row.get("grand_master_meta") if isinstance(row.get("grand_master_meta"), dict) else {}
            latest[symbol] = {
                "timestamp_utc": ts,
                "profile": profile,
                "market": {
                    "last_price": market.get("last_price"),
                    "prev_close": market.get("prev_close"),
                    "pct_from_close": market.get("pct_from_close"),
                    "mom_5m": market.get("mom_5m"),
                    "vol_30m": market.get("vol_30m"),
                    "range_pos": market.get("range_pos"),
                    "spread_bps": market.get("spread_bps"),
                },
                "stance": {
                    "master_action": row.get("master_action"),
                    "master_score": row.get("master_score"),
                    "master_vote": row.get("master_vote"),
                    "directional_trigger": grand.get("directional_trigger"),
                    "specialist_consensus": grand.get("specialist_consensus"),
                    "sleeve_consensus": grand.get("sleeve_consensus"),
                    "options_master": row.get("options_master"),
                    "futures_master": row.get("futures_master"),
                },
            }
            remaining.remove(symbol)
    return latest


def _fallback_position_context(positions: list[dict[str, Any]], roll_watch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fallback: dict[str, dict[str, Any]] = {}
    for pos in positions:
        if str(pos.get("asset_type") or "").upper() != "EQUITY":
            continue
        underlying = str(pos.get("underlying") or "").strip().upper()
        qty = _safe_float(pos.get("quantity"), 0.0)
        market_value = _safe_float(pos.get("market_value"), 0.0)
        mark = market_value / qty if qty > 0.0 and market_value > 0.0 else 0.0
        if underlying and mark > 0.0:
            fallback.setdefault(
                underlying,
                {
                    "timestamp_utc": "",
                    "profile": "broker_position_mark",
                    "market": {"last_price": round(mark, 4)},
                    "stance": {},
                },
            )
    roll_rows = roll_watch.get("covered_calls") if isinstance(roll_watch.get("covered_calls"), list) else []
    roll_counts: dict[str, int] = {}
    for row in roll_rows:
        if isinstance(row, dict):
            underlying = str(row.get("underlying") or "").strip().upper()
            if underlying:
                roll_counts[underlying] = roll_counts.get(underlying, 0) + 1

    def _roll_rank(context: dict[str, Any]) -> tuple[int, float]:
        stance = context.get("stance") if isinstance(context.get("stance"), dict) else {}
        severity = str(stance.get("roll_watch_severity") or "").strip().lower()
        dte = _safe_float(stance.get("dte"), 999999.0)
        return SEVERITY_RANK.get(severity, 0), -dte

    for row in roll_rows:
        if not isinstance(row, dict):
            continue
        underlying = str(row.get("underlying") or "").strip().upper()
        if not underlying:
            continue
        candidate = {
            "timestamp_utc": str(roll_watch.get("timestamp_utc") or ""),
            "profile": "covered_call_roll_watch",
            "market": {
                "last_price": row.get("underlying_price"),
                "moneyness_pct": row.get("moneyness_pct"),
            },
            "stance": {
                "roll_watch_status": row.get("status"),
                "roll_watch_severity": row.get("severity"),
                "covered_call_count_for_underlying": roll_counts.get(underlying, 1),
                "strike": row.get("strike"),
                "expiration": row.get("expiration"),
                "dte": row.get("dte"),
                "roll_trigger": (row.get("operator_roll_preference") or {}).get("wait_for_underlying_price")
                if isinstance(row.get("operator_roll_preference"), dict)
                else None,
            },
        }
        prior = fallback.get(underlying)
        if not prior or str(prior.get("profile") or "") != "covered_call_roll_watch" or _roll_rank(candidate) > _roll_rank(prior):
            fallback[underlying] = candidate
    return fallback


def _underlying_summary(positions: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pos in positions:
        underlying = str(pos.get("underlying") or "").strip().upper()
        if not underlying:
            continue
        item = rows.setdefault(
            underlying,
            {
                "underlying": underlying,
                "accounts": [],
                "equity_quantity": 0.0,
                "option_contract_net": 0.0,
                "market_value": 0.0,
                "position_symbols": [],
                "chart_context": {},
            },
        )
        label = str(pos.get("account_label") or "")
        if label and label not in item["accounts"]:
            item["accounts"].append(label)
        operator_label = str(pos.get("operator_account_label") or "").strip()
        if operator_label:
            operator_accounts = item.setdefault("operator_accounts", [])
            if operator_label not in operator_accounts:
                operator_accounts.append(operator_label)
        operator_kind = str(pos.get("operator_account_kind") or "").strip()
        if operator_kind:
            account_kinds = item.setdefault("account_kinds", [])
            if operator_kind not in account_kinds:
                account_kinds.append(operator_kind)
        symbol = str(pos.get("symbol") or "")
        if symbol and symbol not in item["position_symbols"]:
            item["position_symbols"].append(symbol)
        qty = _safe_float(pos.get("quantity"), 0.0)
        if str(pos.get("asset_type") or "").upper() == "EQUITY":
            item["equity_quantity"] = round(float(item["equity_quantity"]) + qty, 6)
        elif str(pos.get("asset_type") or "").upper() == "OPTION":
            item["option_contract_net"] = round(float(item["option_contract_net"]) + qty, 6)
        item["market_value"] = round(float(item["market_value"]) + _safe_float(pos.get("market_value"), 0.0), 4)

    for underlying, item in rows.items():
        item["chart_context"] = decisions.get(underlying, {})
        item["accounts"] = sorted(item["accounts"])
        item["operator_accounts"] = sorted(item.get("operator_accounts", []))
        item["account_kinds"] = sorted(item.get("account_kinds", []))
        item["position_symbols"] = sorted(item["position_symbols"])
    return sorted(rows.values(), key=lambda item: abs(_safe_float(item.get("market_value"), 0.0)), reverse=True)


def evaluate(
    *,
    snapshot: dict[str, Any],
    roll_watch: dict[str, Any],
    profiles: list[str],
    day: str,
    account_aliases: dict[str, Any] | None = None,
) -> dict[str, Any]:
    positions = _positions(snapshot, account_aliases=account_aliases)
    underlyings = {str(pos.get("underlying") or "").strip().upper() for pos in positions if pos.get("underlying")}
    fallback = _fallback_position_context(positions, roll_watch)
    decisions = {**fallback, **_latest_decision_context(underlyings, profiles, day)}
    roll_rows = roll_watch.get("covered_calls") if isinstance(roll_watch.get("covered_calls"), list) else []
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "source": str(DEFAULT_SNAPSHOT_PATH),
        "account_aliases_source": str(DEFAULT_ACCOUNT_ALIAS_PATH),
        "position_count": len(positions),
        "account_count": len({str(pos.get("account_label") or "") for pos in positions if pos.get("account_label")}),
        "underlying_count": len(underlyings),
        "decision_context_count": len(decisions),
        "positions": positions,
        "underlyings": _underlying_summary(positions, decisions),
        "covered_call_roll_watch": {
            "overall_status": roll_watch.get("overall_status", ""),
            "covered_call_count": roll_watch.get("covered_call_count", 0),
            "alert_count": roll_watch.get("alert_count", 0),
            "covered_calls": roll_rows,
        },
        "notes": [
            "Account labels are redacted; raw account numbers are not emitted.",
            "Operator account labels come from the local redacted account alias map.",
            "Chart context uses latest local decision market features and does not place orders.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an all-visible-account position and chart-context study artifact.")
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--roll-watch-path", default=str(DEFAULT_ROLL_WATCH_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--account-alias-file", default=str(DEFAULT_ACCOUNT_ALIAS_PATH))
    parser.add_argument("--day", default=_today_key())
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = [item.strip() for item in str(args.profiles or "").split(",") if item.strip()]
    payload = evaluate(
        snapshot=load_json(Path(args.snapshot_path)),
        roll_watch=load_json(Path(args.roll_watch_path)),
        profiles=profiles,
        day=str(args.day or _today_key()),
        account_aliases=load_json(Path(args.account_alias_file)),
    )
    payload["source"] = str(Path(args.snapshot_path))
    payload["account_aliases_source"] = str(Path(args.account_alias_file))
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "account_position_study "
            f"positions={payload['position_count']} "
            f"accounts={payload['account_count']} "
            f"underlyings={payload['underlying_count']} "
            f"decision_context={payload['decision_context_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
