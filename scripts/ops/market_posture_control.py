#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "market_posture_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.market_posture_control_override"

PROFILE_FILES = {
    "bond": "bond_equities_schwab",
    "conservative": "conservative_equities_schwab",
    "default": "default_crypto_schwab",
    "dividend": "dividend_equities_schwab",
    "aggressive": "aggressive_equities_schwab",
    "crypto_futures": "crypto_futures_crypto_schwab",
    "intraday_aggressive": "intraday_aggressive_equities_schwab",
    "long_term_core_etf": "long_term_core_etf_equities_schwab",
    "long_term_dividend": "long_term_dividend_equities_schwab",
    "schwab_futures": "schwab_futures_equities_schwab",
    "swing_aggressive": "swing_aggressive_equities_schwab",
}
AGGRESSIVE_PROFILES = {"aggressive", "intraday_aggressive", "swing_aggressive", "crypto_futures", "schwab_futures"}
DEFENSIVE_PROFILES = {"bond", "conservative", "dividend", "long_term_core_etf", "long_term_dividend"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


def _parse_jsonl_lines(lines: list[bytes], *, limit: int) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(int(limit), 1))
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return list(rows)


def _tail_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
    except Exception:
        return []
    if size <= 0:
        return []

    # Decision rows can be heavily compacted. Sample the recent byte tail so
    # posture control stays a control-plane command instead of a full log scan.
    for window in (16 * 1024 * 1024, 64 * 1024 * 1024):
        try:
            offset = max(size - window, 0)
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(min(size, window))
            lines = data.splitlines()
            if offset > 0 and lines:
                lines = lines[1:]
            rows = _parse_jsonl_lines(lines, limit=limit)
            if rows or offset == 0:
                return rows
        except Exception:
            return []
    return []


def _available_decision_days(project_root: Path) -> list[str]:
    days: set[str] = set()
    channels = sorted(set(PROFILE_FILES.values()))
    for channel in channels:
        root = project_root / "governance" / "channels" / "decision" / channel
        try:
            paths = list(root.glob("decision_*.jsonl"))
        except Exception:
            paths = []
        for path in paths:
            name = path.name
            day = name.removeprefix("decision_").removesuffix(".jsonl")
            if len(day) != 8 or not day.isdigit():
                continue
            try:
                if path.stat().st_size <= 0:
                    continue
            except Exception:
                continue
            days.add(day)
    return sorted(days)


def _decision_day(project_root: Path) -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    decision_days = _available_decision_days(project_root)
    if today in decision_days:
        return today
    if decision_days:
        return decision_days[-1]
    paper = load_json(project_root / "governance" / "health" / "sleeve_profitability_dashboard_latest.json")
    day = str(paper.get("paper_day") or "").strip()
    if day:
        return day
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _action(row: dict[str, Any]) -> str:
    for path in (
        ("master_action",),
        ("master_dispatch", "action"),
        ("master_intent_action",),
        ("execution_guard", "action"),
        ("options_margin_guard", "action"),
        ("futures_margin_guard", "action"),
        ("action",),
    ):
        cursor: Any = row
        for key in path:
            cursor = cursor.get(key) if isinstance(cursor, dict) else None
        text = str(cursor or "").strip().upper()
        if text:
            return text
    return "HOLD"


def _summarize_profile(project_root: Path, *, profile: str, day: str, limit: int) -> dict[str, Any]:
    channel = PROFILE_FILES.get(profile, f"{profile}_equities_schwab")
    path = project_root / "governance" / "channels" / "decision" / channel / f"decision_{day}.jsonl"
    rows = _tail_jsonl(path, limit=limit)
    actions = Counter(_action(row) for row in rows)
    symbols = Counter(str(row.get("symbol") or "").strip().upper() for row in rows if str(row.get("symbol") or "").strip())
    n = max(len(rows), 1)
    risk_off = sum(_safe_float(_as_dict(row.get("flow_awareness_features")).get("flow_risk_off_norm"), 0.0) for row in rows) / n
    risk_on = sum(_safe_float(_as_dict(row.get("flow_awareness_features")).get("flow_risk_on_norm"), 0.0) for row in rows) / n
    defensive = sum(_safe_float(_as_dict(row.get("flow_awareness_features")).get("flow_defensive_rotation_norm"), 0.0) for row in rows) / n
    edge = sum(_safe_float(_as_dict(row.get("allocation_confidence")).get("allocation_trade_edge_norm"), 0.0) for row in rows) / n
    confidence = sum(_safe_float(_as_dict(row.get("allocation_confidence")).get("allocation_confidence_norm"), 0.0) for row in rows) / n
    confirmation = sum(
        max(
            _safe_float(row.get("core_cross_asset_confirmation_norm"), 0.0),
            _safe_float(_as_dict(row.get("lead_lag_features")).get("lead_lag_confirmation_norm"), 0.0),
        )
        for row in rows
    ) / n
    dispatch_qty = sum(_safe_float(_as_dict(row.get("portfolio")).get("dispatch_qty"), 0.0) for row in rows)
    lane_paused = sum(1 for row in rows if bool(_as_dict(row.get("circuit_breakers")).get("lane_kill_switch_active", False)))
    return {
        "profile": profile,
        "path": str(path),
        "exists": path.exists(),
        "rows_sampled": len(rows),
        "action_counts": dict(sorted(actions.items())),
        "hold_ratio": round(actions.get("HOLD", 0) / n, 6),
        "buy_count": int(actions.get("BUY", 0)),
        "sell_count": int(actions.get("SELL", 0)),
        "dispatch_qty_sum": round(float(dispatch_qty), 6),
        "avg_risk_off_norm": round(float(risk_off), 6),
        "avg_risk_on_norm": round(float(risk_on), 6),
        "avg_defensive_rotation_norm": round(float(defensive), 6),
        "avg_allocation_edge_norm": round(float(edge), 6),
        "avg_allocation_confidence_norm": round(float(confidence), 6),
        "avg_confirmation_norm": round(float(confirmation), 6),
        "lane_pause_ratio": round(lane_paused / n, 6),
        "top_symbols": [{"symbol": symbol, "count": count} for symbol, count in symbols.most_common(12)],
    }


def _posture_state(profile_summaries: dict[str, dict[str, Any]], profitability: dict[str, Any]) -> tuple[str, list[str]]:
    bond = profile_summaries.get("bond", {})
    aggressive_rows = [
        row for name, row in profile_summaries.items() if name in AGGRESSIVE_PROFILES and _safe_int(row.get("rows_sampled"), 0) > 0
    ]
    avg_aggressive_edge = (
        sum(_safe_float(row.get("avg_allocation_edge_norm"), 0.0) for row in aggressive_rows) / max(len(aggressive_rows), 1)
        if aggressive_rows
        else 0.0
    )
    avg_aggressive_confirmation = (
        sum(_safe_float(row.get("avg_confirmation_norm"), 0.0) for row in aggressive_rows) / max(len(aggressive_rows), 1)
        if aggressive_rows
        else 0.0
    )
    protective = str(profitability.get("overall_status") or "").strip().lower() == "protective_tightening"
    bond_hold = (
        _safe_int(bond.get("rows_sampled"), 0) > 0
        and _safe_float(bond.get("hold_ratio"), 0.0) >= 0.80
        and abs(_safe_float(bond.get("dispatch_qty_sum"), 0.0)) <= 1e-8
    )
    defensive_present = _safe_float(bond.get("avg_defensive_rotation_norm"), 0.0) >= 0.10
    momentum_faded = avg_aggressive_edge <= 0.08 and avg_aggressive_confirmation <= 0.55

    reasons: list[str] = []
    if protective:
        reasons.append("paper_profitability_control_is_protective_tightening")
    if bond_hold:
        reasons.append("bond_sleeve_is_holding_without_new_dispatch")
    if defensive_present:
        reasons.append("bond_defensive_rotation_signal_present")
    if momentum_faded:
        reasons.append("aggressive_edge_or_confirmation_is_not_clean")

    if protective and bond_hold and momentum_faded:
        return "defensive_hold_momentum_faded", reasons
    if protective:
        return "protective_tightening", reasons
    if bond_hold and defensive_present:
        return "defensive_watch", reasons
    if momentum_faded:
        return "selective_wait_for_confirmation", reasons
    return "balanced_observe", reasons or ["no_defensive_posture_trigger"]


def _env_overrides(state: str) -> dict[str, str]:
    defensive = state in {"defensive_hold_momentum_faded", "protective_tightening", "defensive_watch"}
    return {
        "MARKET_POSTURE_CONTROL_ENABLED": "1",
        "MARKET_POSTURE_STATE": state,
        "MARKET_POSTURE_RISK_APPETITE": "low" if defensive else "selective",
        "MARKET_POSTURE_AGGRESSIVE_ENTRY_MULT": "0.45" if defensive else "0.72",
        "MARKET_POSTURE_BUY_CONFIRMATION_FLOOR": "0.58" if defensive else "0.52",
        "MARKET_POSTURE_BUY_RISK_ON_EDGE_DELTA": "0.08" if defensive else "0.04",
        "MARKET_POSTURE_DEFENSIVE_BUY_RISK_OFF_FLOOR": "0.18" if defensive else "0.12",
        "MARKET_POSTURE_DEFENSIVE_HOLD_OK": "1",
        "MARKET_POSTURE_RERISK_CONFIRMATION_SAMPLES": "2",
        "MARKET_POSTURE_AGGRESSIVE_PROFILES": ",".join(sorted(AGGRESSIVE_PROFILES)),
        "MARKET_POSTURE_DEFENSIVE_PROFILES": ",".join(sorted(DEFENSIVE_PROFILES)),
        "DEFENSIVE_SYMBOL_EVERY_N_ITERS": "1" if defensive else "2",
    }


def _override_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Auto-managed by scripts/ops/market_posture_control.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
    ]
    for key, value in sorted(_as_dict(payload.get("env_overrides")).items()):
        lines.append(f"{key}={shlex.quote(str(value))}")
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT, *, sample_limit: int = 1200) -> dict[str, Any]:
    day = _decision_day(project_root)
    health = project_root / "governance" / "health"
    profitability = load_json(health / "paper_profitability_control_latest.json")
    sleeve_dashboard = load_json(health / "sleeve_profitability_dashboard_latest.json")
    profiles = {
        profile: _summarize_profile(project_root, profile=profile, day=day, limit=sample_limit)
        for profile in PROFILE_FILES
    }
    state, reasons = _posture_state(profiles, profitability)
    overrides = _env_overrides(state)
    defensive_active = state in {"defensive_hold_momentum_faded", "protective_tightening", "defensive_watch"}
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "defensive_posture_active" if defensive_active else "ready",
        "posture_state": state,
        "reasons": reasons,
        "paper_day": day,
        "profile_summaries": profiles,
        "paper_profitability": {
            "overall_status": str(profitability.get("overall_status") or ""),
            "profitability_grade": str(profitability.get("profitability_grade") or ""),
        },
        "sleeve_totals": _as_dict(sleeve_dashboard.get("totals")),
        "env_overrides": overrides,
        "runtime_contract": {
            "paper_only": True,
            "live_execution_allowed": False,
            "preserve_capital_when_momentum_fades": True,
            "do_not_force_bond_buys_without_risk_off_confirmation": True,
            "block_aggressive_new_buys_until_re_risk_confirmation": defensive_active,
            "allow_reduce_only_and_holds": True,
            "rerisk_when": [
                "risk_on exceeds risk_off by MARKET_POSTURE_BUY_RISK_ON_EDGE_DELTA",
                "cross-asset or lead-lag confirmation clears MARKET_POSTURE_BUY_CONFIRMATION_FLOOR",
                "allocation edge is positive and execution guards remain clean",
                "two consecutive posture samples agree",
            ],
        },
        "recommended_actions": [
            "keep the bond/rates sleeve watching and holding while momentum is unclear",
            "avoid widening aggressive sleeves until re-risk confirmation clears",
            "refresh sleeve-pnl and market-posture-control after a trend day or major macro catalyst",
        ],
    }
    return payload


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path | None = None,
    override_path: Path | None = None,
) -> dict[str, Any]:
    out_candidate = out_path or Path("governance") / "health" / "market_posture_control_latest.json"
    override_candidate = override_path or Path("config") / ".env.market_posture_control_override"
    out = out_candidate if out_candidate.is_absolute() else project_root / out_candidate
    override = override_candidate if override_candidate.is_absolute() else project_root / override_candidate
    write_payload(out, payload)
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(_override_text(payload), encoding="utf-8")
    payload = dict(payload)
    payload["apply_result"] = {
        "applied": True,
        "health_path": str(out),
        "override_path": str(override),
    }
    write_payload(out, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish and apply market posture controls for momentum-faded defensive regimes.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=1200)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--override-file", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(PROJECT_ROOT, sample_limit=max(int(args.sample_limit), 1))
    if args.apply:
        payload = apply_payload(PROJECT_ROOT, payload, out_path=args.out_file, override_path=args.override_file)
    else:
        payload["apply_result"] = {
            "applied": False,
            "health_path": str(args.out_file if args.out_file.is_absolute() else PROJECT_ROOT / args.out_file),
            "override_path": str(args.override_file if args.override_file.is_absolute() else PROJECT_ROOT / args.override_file),
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "market_posture_control "
            f"status={payload.get('overall_status')} "
            f"state={payload.get('posture_state')} "
            f"day={payload.get('paper_day')}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
