#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_profitability_dashboard_latest.json"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _grade_from_net(net: float, realized: float, unrealized: float, executions: int) -> str:
    if executions <= 0:
        return "N/A"
    if net >= 250.0 and realized >= 0.0:
        return "A+"
    if net >= 100.0 and realized >= 0.0:
        return "A"
    if net >= 25.0:
        return "B"
    if net >= 0.0:
        return "C"
    if unrealized < 0.0 and abs(unrealized) > max(abs(realized) * 1.5, 50.0):
        return "D"
    return "C-"


def _action_for_row(row: dict[str, Any]) -> str:
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    realized = _safe_float(row.get("ending_realized_pnl_total"), 0.0)
    unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
    executions = _safe_int(row.get("executions"), 0)
    if executions <= 0:
        return "collect_live_heartbeat_or_wait_for_fills"
    if net > 0.0 and unrealized > max(realized, 0.0) * 2.0 and unrealized > 50.0:
        return "harvest_partial_winners_if_runner_protection_clears"
    if net < 0.0 and unrealized < 0.0:
        return "contain_new_adds_and_replay_unrealized_drag"
    if net > 0.0 and realized >= unrealized:
        return "monitor_and_allow_clean_strategy_promotion"
    return "keep_collecting_and_refresh_profitability_control"


def _dashboard_row(row: dict[str, Any], profile_controls: dict[str, Any]) -> dict[str, Any]:
    profile = str(row.get("profile") or "unknown").strip().lower() or "unknown"
    realized = _safe_float(row.get("ending_realized_pnl_total"), 0.0)
    unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    executions = _safe_int(row.get("executions"), 0)
    win_rate_raw = row.get("win_rate")
    control = _as_dict(profile_controls.get(profile))
    return {
        "profile": profile,
        "data_status": str(row.get("data_status") or ""),
        "day_utc": str(row.get("day_utc") or ""),
        "executions": executions,
        "realized_pnl_total": round(realized, 6),
        "unrealized_pnl_total": round(unrealized, 6),
        "net_pnl_total": round(net, 6),
        "win_rate": None if win_rate_raw is None else round(_safe_float(win_rate_raw), 6),
        "winning_strategy_count": _safe_int(row.get("winning_strategy_count"), 0),
        "losing_strategy_count": _safe_int(row.get("losing_strategy_count"), 0),
        "grade": _grade_from_net(net, realized, unrealized, executions),
        "control_action": str(control.get("action") or ""),
        "harvest_action": _action_for_row(row),
        "top_winning_strategies": _as_list(row.get("top_winning_strategies"))[:3],
        "top_losing_strategies": _as_list(row.get("top_losing_strategies"))[:3],
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, max_rows: int = 40) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    paper = load_json(health / "paper_performance_latest.json")
    profitability = load_json(health / "paper_profitability_control_latest.json")
    sleeves = [row for row in _as_list(paper.get("sleeve_latest")) if isinstance(row, dict)]
    profile_controls = _as_dict(profitability.get("active_profile_controls"))
    rows = [_dashboard_row(row, profile_controls) for row in sleeves]
    rows.sort(key=lambda row: _safe_float(row.get("net_pnl_total"), 0.0), reverse=True)
    totals = {
        "sleeve_count": len(rows),
        "execution_count": sum(_safe_int(row.get("executions"), 0) for row in rows),
        "realized_pnl_total": round(sum(_safe_float(row.get("realized_pnl_total"), 0.0) for row in rows), 6),
        "unrealized_pnl_total": round(sum(_safe_float(row.get("unrealized_pnl_total"), 0.0) for row in rows), 6),
        "net_pnl_total": round(sum(_safe_float(row.get("net_pnl_total"), 0.0) for row in rows), 6),
    }
    weak_rows = [row for row in rows if str(row.get("grade") or "") in {"C-", "D", "F"}]
    harvest_rows = [row for row in rows if "harvest" in str(row.get("harvest_action") or "")]
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(rows),
        "overall_status": "ready" if rows else "no_paper_sleeve_rows",
        "paper_day": str(_as_dict(paper.get("day")).get("day_utc") or paper.get("day") or ""),
        "profitability_status": str(profitability.get("overall_status") or ""),
        "profitability_grade": str(profitability.get("profitability_grade") or ""),
        "totals": totals,
        "top_sleeves": rows[: min(max(int(max_rows), 1), 40)],
        "bottom_sleeves": list(reversed(rows[-min(max(int(max_rows), 1), 40) :])) if rows else [],
        "weak_sleeve_count": len(weak_rows),
        "harvest_attention_count": len(harvest_rows),
        "harvest_attention": harvest_rows[:12],
        "recommended_actions": [
            "run paper-profitability-control --apply before widening any sleeve" if weak_rows else "",
            "use harvest_attention rows to convert unrealized winners into realized paper gains" if harvest_rows else "",
            "refresh paper-performance daily and compare sleeve net, realized, and unrealized totals",
        ],
        "contract": {
            "mode": "per_sleeve_profitability_dashboard_v1",
            "read_only": True,
            "live_execution_allowed": False,
            "source_artifacts": [
                str(health / "paper_performance_latest.json"),
                str(health / "paper_profitability_control_latest.json"),
            ],
        },
    }
    payload["recommended_actions"] = [str(item) for item in payload["recommended_actions"] if str(item)]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only per-sleeve paper profitability dashboard.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), max_rows=max(int(args.max_rows), 1))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        totals = _as_dict(payload.get("totals"))
        print(
            "sleeve_profitability_dashboard "
            f"status={payload.get('overall_status')} "
            f"sleeves={totals.get('sleeve_count', 0)} "
            f"net={_safe_float(totals.get('net_pnl_total'), 0.0):.2f}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
