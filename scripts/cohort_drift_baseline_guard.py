#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "cohort_drift_baseline_history.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "cohort_drift_baseline_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    except Exception:
        return []
    return rows


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _cohort_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    profile = str(raw_row.get("profile") or "").strip().lower()
    family = str(raw_row.get("strategy_family") or raw_row.get("family") or profile).strip().lower() or "unknown"
    timeframe = str(raw_row.get("timeframe") or raw_row.get("lane") or profile).strip().lower() or "unknown"
    venue = str(raw_row.get("venue") or raw_row.get("execution_venue") or raw_row.get("broker") or "unknown").strip().lower() or "unknown"
    tca_summary = raw_row.get("tca_summary") if isinstance(raw_row.get("tca_summary"), dict) else {}
    poor_fill_count = _to_int(tca_summary.get("poor_or_fair_fill_count"), 0)
    return {
        "cohort_key": f"{family}|{timeframe}|{venue}",
        "profile": profile,
        "family": family,
        "timeframe": timeframe,
        "venue": venue,
        "day_utc": str(raw_row.get("day_utc") or "").strip(),
        "current_day_available": bool(raw_row.get("current_day_available", False)),
        "data_status": str(raw_row.get("data_status") or "").strip().lower(),
        "executions": _to_int(raw_row.get("executions"), 0),
        "ending_timestamp_utc": str(raw_row.get("ending_timestamp_utc") or "").strip(),
        "ending_realized_pnl_total": _to_float(raw_row.get("ending_realized_pnl_total"), 0.0),
        "ending_unrealized_pnl_total": _to_float(raw_row.get("ending_unrealized_pnl_total"), 0.0),
        "win_rate": _to_float(raw_row.get("win_rate"), 0.0),
        "ending_net_pnl_total": _to_float(raw_row.get("ending_net_pnl_total"), 0.0),
        "flat_strategy_count": _to_int(raw_row.get("flat_strategy_count"), 0),
        "non_flat_strategy_count": _to_int(raw_row.get("non_flat_strategy_count"), 0),
        "mean_slippage_gap_bps": _to_float(tca_summary.get("mean_slippage_gap_bps"), 0.0),
        "poor_or_fair_fill_count": poor_fill_count,
    }


def build_payload(
    *,
    paper_performance: dict[str, Any],
    history_rows: list[dict[str, Any]],
    lookback_snapshots: int,
    min_history_points: int,
    min_executions: int,
    max_win_rate_drop: float,
    max_pnl_drop: float,
    max_slippage_gap_bps_increase: float,
    min_current_day_execution_completeness_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    sleeve_latest = paper_performance.get("sleeve_latest") if isinstance(paper_performance.get("sleeve_latest"), list) else []
    current_rows = [_cohort_row(row) for row in sleeve_latest if isinstance(row, dict) and str(row.get("profile") or "").strip()]
    snapshot_entry = {
        "timestamp_utc": now,
        "schema_version": 1,
        "cohorts": current_rows,
    }

    prior_snapshots = history_rows[-max(int(lookback_snapshots), 0):]
    drifted_rows: list[dict[str, Any]] = []
    deferred_current_day_rows: list[dict[str, Any]] = []
    warmup_count = 0

    for row in current_rows:
        row_day_utc = str(row.get("day_utc") or "").strip()
        prior_matches_by_day: dict[str, dict[str, Any]] = {}
        for snapshot in prior_snapshots:
            cohorts = snapshot.get("cohorts") if isinstance(snapshot.get("cohorts"), list) else []
            for prior in cohorts:
                if not isinstance(prior, dict) or str(prior.get("cohort_key") or "") != row["cohort_key"]:
                    continue
                prior_day_utc = str(prior.get("day_utc") or "").strip()
                if row_day_utc and prior_day_utc == row_day_utc:
                    continue
                prior_matches_by_day[prior_day_utc or f"snapshot:{len(prior_matches_by_day)}"] = prior
        prior_matches = list(prior_matches_by_day.values())[-max(int(lookback_snapshots), 0) :]
        if len(prior_matches) < int(min_history_points):
            warmup_count += 1
            continue

        baseline_executions = median(_to_int(item.get("executions"), 0) for item in prior_matches)
        baseline = {
            "win_rate": median(_to_float(item.get("win_rate"), 0.0) for item in prior_matches),
            "ending_net_pnl_total": median(_to_float(item.get("ending_net_pnl_total"), 0.0) for item in prior_matches),
            "mean_slippage_gap_bps": median(_to_float(item.get("mean_slippage_gap_bps"), 0.0) for item in prior_matches),
        }
        current_day_execution_floor = max(
            float(int(min_executions)),
            float(baseline_executions) * max(float(min_current_day_execution_completeness_ratio), 0.0),
        )
        current_day_incomplete = bool(
            row.get("current_day_available", False)
            and baseline_executions > 0
            and float(row.get("executions", 0) or 0) < float(current_day_execution_floor)
        )
        if current_day_incomplete:
            deferred_current_day_rows.append(
                {
                    **row,
                    "baseline_executions": int(round(float(baseline_executions))),
                    "current_day_execution_floor": round(float(current_day_execution_floor), 6),
                    "deferred_reason": "current_day_execution_completeness",
                }
            )
            continue

        current_day_mark_to_market_open = bool(
            row.get("current_day_available", False)
            and str(row.get("data_status") or "") == "current"
            and int(row.get("non_flat_strategy_count", 0) or 0) > 0
            and abs(float(row.get("ending_unrealized_pnl_total", 0.0) or 0.0))
            > abs(float(row.get("ending_realized_pnl_total", 0.0) or 0.0))
            and abs(float(row.get("ending_realized_pnl_total", 0.0) or 0.0)) <= float(max_pnl_drop)
        )
        if current_day_mark_to_market_open:
            deferred_current_day_rows.append(
                {
                    **row,
                    "baseline_executions": int(round(float(baseline_executions))),
                    "current_day_execution_floor": round(float(current_day_execution_floor), 6),
                    "deferred_reason": "current_day_mark_to_market_open_positions",
                }
            )
            continue

        win_rate_drop = baseline["win_rate"] - row["win_rate"]
        pnl_drop = baseline["ending_net_pnl_total"] - row["ending_net_pnl_total"]
        slippage_increase = row["mean_slippage_gap_bps"] - baseline["mean_slippage_gap_bps"]

        severe = bool(
            row["executions"] >= int(min_executions)
            and (
                win_rate_drop > float(max_win_rate_drop)
                or pnl_drop > float(max_pnl_drop)
                or slippage_increase > float(max_slippage_gap_bps_increase)
            )
        )
        if severe:
            drifted_rows.append(
                {
                    **row,
                    "baseline": baseline,
                    "drift": {
                        "win_rate_drop": round(win_rate_drop, 6),
                        "pnl_drop": round(pnl_drop, 6),
                        "slippage_gap_bps_increase": round(slippage_increase, 6),
                    },
                }
            )

    if not current_rows:
        ok = False
        overall_status = "blocked"
        failed_checks = ["no_paper_performance_cohorts"]
    else:
        ok = not drifted_rows
        if drifted_rows:
            overall_status = "needs_work"
        elif warmup_count == len(current_rows):
            overall_status = "warmup"
        else:
            overall_status = "ready"
        failed_checks = ["cohort_drift_detected"] if drifted_rows else []

    payload = {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "failed_checks": failed_checks,
        "summary": {
            "cohort_count": len(current_rows),
            "warmup_cohort_count": warmup_count,
            "deferred_current_day_cohort_count": len(deferred_current_day_rows),
            "severe_cohort_count": len(drifted_rows),
        },
        "deferred_current_day_cohorts": deferred_current_day_rows[:25],
        "drifted_cohorts": drifted_rows[:25],
        "current_cohorts": current_rows[:40],
        "thresholds": {
            "lookback_snapshots": int(lookback_snapshots),
            "min_history_points": int(min_history_points),
            "min_executions": int(min_executions),
            "max_win_rate_drop": float(max_win_rate_drop),
            "max_pnl_drop": float(max_pnl_drop),
            "max_slippage_gap_bps_increase": float(max_slippage_gap_bps_increase),
            "min_current_day_execution_completeness_ratio": float(min_current_day_execution_completeness_ratio),
        },
    }
    return payload, snapshot_entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Track cohort-level paper drift against rolling baselines.")
    parser.add_argument("--paper-performance-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"))
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--lookback-snapshots", type=int, default=7)
    parser.add_argument("--min-history-points", type=int, default=3)
    parser.add_argument("--min-executions", type=int, default=10)
    parser.add_argument("--max-win-rate-drop", type=float, default=0.12)
    parser.add_argument("--max-pnl-drop", type=float, default=5.0)
    parser.add_argument("--max-slippage-gap-bps-increase", type=float, default=2.5)
    parser.add_argument("--min-current-day-execution-completeness-ratio", type=float, default=0.5)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    history_path = Path(args.history_file)
    payload, snapshot_entry = build_payload(
        paper_performance=_load_json(Path(args.paper_performance_file)),
        history_rows=_load_history(history_path),
        lookback_snapshots=int(args.lookback_snapshots),
        min_history_points=int(args.min_history_points),
        min_executions=int(args.min_executions),
        max_win_rate_drop=float(args.max_win_rate_drop),
        max_pnl_drop=float(args.max_pnl_drop),
        max_slippage_gap_bps_increase=float(args.max_slippage_gap_bps_increase),
        min_current_day_execution_completeness_ratio=float(args.min_current_day_execution_completeness_ratio),
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot_entry, ensure_ascii=True) + "\n")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary", {})
        print(
            "cohort_drift_baseline_guard "
            f"ok={str(payload['ok']).lower()} "
            f"cohorts={int(summary.get('cohort_count', 0) or 0)} "
            f"severe={int(summary.get('severe_cohort_count', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
