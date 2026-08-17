#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "allocator" / "portfolio_capacity_curve_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _normalize(text: Any, default: str) -> str:
    value = str(text or "").strip().lower()
    return value or default


def _recent_paper_rows(project_root: Path, limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paper_root = project_root / "exports" / "paper_broker_bridge" / "paper"
    files = sorted(
        [*paper_root.glob("*.jsonl"), *paper_root.glob("*.jsonl.gz")],
        key=lambda path: (path.stat().st_mtime if path.exists() else 0.0, path.name),
    )
    for path in reversed(files):
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    raw_lines = handle.read().splitlines()
            else:
                raw_lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in reversed(raw_lines):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= limit:
                return rows
    return rows


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    intents_file: Path | None = None,
    paper_performance_file: Path | None = None,
    execution_budget_file: Path | None = None,
) -> dict[str, Any]:
    intents_path = intents_file or (project_root / "governance" / "allocator" / "portfolio_candidate_intents_latest.json")
    performance_path = paper_performance_file or (project_root / "governance" / "health" / "paper_performance_latest.json")
    execution_budget_path = execution_budget_file or (project_root / "governance" / "risk" / "execution_budget_latest.json")
    intents_payload = _load_json(intents_path)
    performance = _load_json(performance_path)
    execution_budget = _load_json(execution_budget_path)

    sleeve_latest = performance.get("sleeve_latest") if isinstance(performance.get("sleeve_latest"), list) else []
    slippage_by_profile = {
        str(row.get("profile") or "").strip().lower(): _safe_float(((row.get("tca_summary") or {}).get("mean_slippage_gap_bps")), 0.0)
        for row in sleeve_latest
        if isinstance(row, dict)
    }
    poor_fill_by_profile = {
        str(row.get("profile") or "").strip().lower(): int(((row.get("tca_summary") or {}).get("poor_or_fair_fill_count")) or 0)
        for row in sleeve_latest
        if isinstance(row, dict)
    }

    global_actions = max(_safe_float(((execution_budget.get("global") or {}).get("max_total_actions_per_hour")), 60.0), 1.0)
    budget_scale = max(min(global_actions / 120.0, 1.0), 0.35)

    intents = intents_payload.get("intents") if isinstance(intents_payload.get("intents"), list) else []
    if not intents:
        fallback_rows = _recent_paper_rows(project_root)
        intents = [
            {
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "sleeve": str(((row.get("metadata") or {}).get("source_profile") or row.get("profile") or "default")).strip().lower(),
                "capacity_fraction": max(min(_safe_float(row.get("tradeability_score"), 1.0), 1.0), 0.2),
                "forward_cost_bps": max(_safe_float(row.get("slippage_gap_bps"), 0.0), 0.0),
                "venue": "primary",
                "clock_bucket": "intraday",
                "regime": "normal",
            }
            for row in fallback_rows
            if isinstance(row, dict) and str(row.get("symbol") or "").strip()
        ]
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in intents:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        sleeve = _normalize(row.get("sleeve") or row.get("profile"), "default")
        venue = _normalize(row.get("venue"), "primary")
        clock_bucket = _normalize(row.get("clock_bucket") or row.get("session_bucket"), "all_day")
        regime = _normalize(row.get("regime"), "normal")
        base_capacity = max(min(_safe_float(row.get("capacity_fraction"), 1.0), 1.0), 0.05)
        forward_cost_bps = max(slippage_by_profile.get(sleeve, 0.0), _safe_float(row.get("forward_cost_bps"), 0.0))
        forward_cost_bps += poor_fill_by_profile.get(sleeve, 0) * 0.5
        cost_scale = max(0.25, 1.0 - min(forward_cost_bps, 60.0) / 100.0)
        recommended_capacity_fraction = round(max(min(base_capacity * cost_scale * budget_scale, 1.0), 0.05), 6)
        key = (symbol, venue, clock_bucket, regime)
        current = by_key.get(key)
        candidate = {
            "symbol": symbol,
            "venue": venue,
            "clock_bucket": clock_bucket,
            "regime": regime,
            "profile": sleeve,
            "forward_cost_bps": round(forward_cost_bps, 4),
            "recommended_capacity_fraction": recommended_capacity_fraction,
            "input_capacity_fraction": round(base_capacity, 6),
            "budget_scale": round(budget_scale, 6),
        }
        if current is None or candidate["recommended_capacity_fraction"] < float(current.get("recommended_capacity_fraction", 1.0)):
            by_key[key] = candidate

    curves = sorted(by_key.values(), key=lambda row: (float(row.get("recommended_capacity_fraction", 1.0)), row.get("symbol", "")))
    constrained = [row for row in curves if float(row.get("recommended_capacity_fraction", 1.0)) < 0.95]
    summary = {
        "curve_count": len(curves),
        "constrained_curve_count": len(constrained),
        "venue_count": len({str(row.get("venue") or "") for row in curves}),
        "clock_bucket_count": len({str(row.get("clock_bucket") or "") for row in curves}),
        "regime_count": len({str(row.get("regime") or "") for row in curves}),
        "allocator_ready": bool(curves),
        "top_constrained_symbols": [str(row.get("symbol") or "") for row in constrained[:10]],
    }
    overall_status = "ready" if curves else "degraded"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": bool(curves),
        "overall_status": overall_status,
        "summary": summary,
        "curves": curves[:200],
        "source_files": {
            "intents": str(intents_path),
            "paper_performance": str(performance_path),
            "execution_budget": str(execution_budget_path),
        },
        "top_actions": [
            "feed allocator scoring through venue, clock-bucket, and regime-aware capacity curves before order emission",
            "use realized sleeve slippage to shape forward capacity fractions instead of leaving capacity implicit",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish symbol-by-venue-by-time capacity curves for portfolio allocation.")
    parser.add_argument("--intents-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "portfolio_candidate_intents_latest.json"))
    parser.add_argument("--paper-performance-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"))
    parser.add_argument("--execution-budget-file", default=str(PROJECT_ROOT / "governance" / "risk" / "execution_budget_latest.json"))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        intents_file=Path(args.intents_file).expanduser(),
        paper_performance_file=Path(args.paper_performance_file).expanduser(),
        execution_budget_file=Path(args.execution_budget_file).expanduser(),
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "portfolio_capacity_curve_report "
            f"overall_status={payload.get('overall_status', '')} "
            f"curve_count={int(((payload.get('summary') or {}).get('curve_count') or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
