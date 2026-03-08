#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _iter_tail_jsonl(path: Path, max_bytes: int) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".gz":
        yield from _iter_jsonl(path)
        return

    max_bytes = max(int(max_bytes), 262144)
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        seek_pos = max(size - max_bytes, 0)
        fh.seek(seek_pos)
        blob = fh.read().decode("utf-8", errors="ignore")

    lines = blob.splitlines()
    if seek_pos > 0 and lines:
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def _lane_from_dirname(name: str) -> str:
    lower = (name or "").lower()
    if "long_term" in lower:
        return "long_term"
    if "futures" in lower or "crypto" in lower:
        return "futures"
    if "options" in lower:
        return "options"
    if "swing" in lower:
        return "swing"
    if "intraday" in lower or "aggressive" in lower or "day" in lower:
        return "day"
    return "equities"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified cross-lane scorecard from governance decision logs.")
    ap.add_argument("--lookback-hours", type=float, default=24.0)
    ap.add_argument("--max-rows-scan", type=int, default=int(250000))
    ap.add_argument("--tail-bytes", type=int, default=int(6_000_000))
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cutoff = _now_utc() - timedelta(hours=max(float(args.lookback_hours), 1.0))

    files: List[Path] = []
    min_day = (cutoff - timedelta(days=1)).strftime("%Y%m%d")
    for root in (PROJECT_ROOT / "governance").glob("shadow_*"):
        for path in sorted(root.glob("master_control_*.jsonl"), reverse=True) + sorted(root.glob("master_control_*.jsonl.gz"), reverse=True):
            name = path.name
            day = ""
            if name.startswith("master_control_"):
                tail = name.replace("master_control_", "", 1)
                day = tail.split(".", 1)[0]
            if len(day) == 8 and day.isdigit() and day < min_day:
                continue
            files.append(path)

    lane_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_total = 0
    rows_used = 0

    max_rows_scan = max(int(args.max_rows_scan), 1000)
    stop_scan = False
    for path in files:
        if stop_scan:
            break
        default_lane = _lane_from_dirname(path.parent.name)
        row_iter = _iter_tail_jsonl(path, int(args.tail_bytes)) if path.suffix != ".gz" else _iter_jsonl(path)
        for row in row_iter:
            rows_total += 1
            if rows_total >= max_rows_scan:
                stop_scan = True
            ts = _parse_ts(str(row.get("timestamp_utc", "")))
            if ts is None or ts < cutoff:
                if stop_scan:
                    break
                continue
            lane = str(((row.get("lane_allocator") or {}).get("lane") if isinstance(row.get("lane_allocator"), dict) else "") or "").strip().lower()
            if not lane:
                lane = default_lane
            lane_rows[lane].append(row)
            rows_used += 1
            if stop_scan:
                break

    per_lane: Dict[str, Dict[str, Any]] = {}
    for lane, rows in lane_rows.items():
        if not rows:
            continue
        action_counts: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        master_scores: List[float] = []
        slippages: List[float] = []
        blocked_intent = 0
        exec_guard_block = 0
        portfolio_risk_block = 0
        manual_active = 0
        options_roll = 0
        futures_roll = 0

        for row in rows:
            action = str(row.get("master_action", "HOLD") or "HOLD").upper()
            if action not in action_counts:
                action = "HOLD"
            action_counts[action] += 1
            master_scores.append(_safe_float(row.get("master_score"), 0.5))
            slippages.append(_safe_float(((row.get("execution_sim") or {}).get("slippage_bps")), 0.0))
            if bool(row.get("master_guard_blocked_intent", False)):
                blocked_intent += 1
            if not bool(((row.get("execution_guard") or {}).get("ok", True))):
                exec_guard_block += 1
            if bool(((row.get("portfolio_risk_engine") or {}).get("blocked", False))):
                portfolio_risk_block += 1
            if bool(((row.get("manual_trade_reconcile") or {}).get("active", False))):
                manual_active += 1
            if str(((row.get("options_roll_manager") or {}).get("directive", "none"))).lower() != "none":
                options_roll += 1
            if str(((row.get("futures_roll_manager") or {}).get("directive", "none"))).lower() != "none":
                futures_roll += 1

        n = max(len(rows), 1)
        per_lane[lane] = {
            "rows": len(rows),
            "buy_rows": action_counts["BUY"],
            "sell_rows": action_counts["SELL"],
            "hold_rows": action_counts["HOLD"],
            "avg_master_score": round(sum(master_scores) / n, 6),
            "avg_slippage_bps": round(sum(slippages) / n, 4),
            "intent_block_rate": round(blocked_intent / n, 6),
            "execution_guard_block_rate": round(exec_guard_block / n, 6),
            "portfolio_risk_block_rate": round(portfolio_risk_block / n, 6),
            "manual_reconcile_active_rate": round(manual_active / n, 6),
            "options_roll_rate": round(options_roll / n, 6),
            "futures_roll_rate": round(futures_roll / n, 6),
        }

    payload = {
        "timestamp_utc": _now_utc().isoformat(),
        "lookback_hours": float(args.lookback_hours),
        "rows_scanned": int(rows_total),
        "rows_used": int(rows_used),
        "max_rows_scan": int(max_rows_scan),
        "lanes": per_lane,
        "lane_count": len(per_lane),
        "ok": rows_used > 0,
    }

    out_json = PROJECT_ROOT / "governance" / "health" / "unified_lane_scorecard_latest.json"
    out_md = PROJECT_ROOT / "exports" / "sql_reports" / "unified_lane_scorecard_latest.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Unified Lane Scorecard ({payload['timestamp_utc']})")
    lines.append("")
    lines.append(f"- Lookback hours: {payload['lookback_hours']}")
    lines.append(f"- Rows used/scanned: {payload['rows_used']}/{payload['rows_scanned']}")
    lines.append("")
    if not per_lane:
        lines.append("No rows found in lookback window.")
    else:
        for lane in sorted(per_lane.keys()):
            row = per_lane[lane]
            lines.append(f"## {lane}")
            lines.append(f"- rows: {row['rows']}")
            lines.append(f"- buy/sell/hold: {row['buy_rows']}/{row['sell_rows']}/{row['hold_rows']}")
            lines.append(f"- avg_master_score: {row['avg_master_score']}")
            lines.append(f"- avg_slippage_bps: {row['avg_slippage_bps']}")
            lines.append(f"- intent_block_rate: {row['intent_block_rate']}")
            lines.append(f"- execution_guard_block_rate: {row['execution_guard_block_rate']}")
            lines.append(f"- portfolio_risk_block_rate: {row['portfolio_risk_block_rate']}")
            lines.append(f"- manual_reconcile_active_rate: {row['manual_reconcile_active_rate']}")
            lines.append(f"- options_roll_rate: {row['options_roll_rate']}")
            lines.append(f"- futures_roll_rate: {row['futures_roll_rate']}")
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if args.json:
        print(json.dumps(payload))
    else:
        print(f"unified_lane_scorecard ok={payload['ok']} lanes={payload['lane_count']} rows={payload['rows_used']}")
        print(out_json)
        print(out_md)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
