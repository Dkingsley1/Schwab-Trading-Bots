import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly report: why promotion gate is blocked.")
    parser.add_argument("--readiness-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_latest.json"))
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--diagnostics-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "canary_diagnostics_latest.json"))
    parser.add_argument("--regime-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "regime_segmented_latest.json"))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--out-json", default=str(PROJECT_ROOT / "governance" / "audits" / "weekly_gate_blockers_latest.json"))
    parser.add_argument("--out-md", default=str(PROJECT_ROOT / "governance" / "audits" / "weekly_gate_blockers_latest.md"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    readiness = _load(Path(args.readiness_file))
    gate = _load(Path(args.gate_file))
    diagnostics = _load(Path(args.diagnostics_file))
    regime = _load(Path(args.regime_file))

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(int(args.lookback_days), 1))

    hist = []
    for row in _read_jsonl(Path(args.history_jsonl)):
        ts = _parse_ts(str(row.get("timestamp_utc", "")))
        if ts is None or ts < cutoff:
            continue
        hist.append(row)

    fail_series = [float((r or {}).get("fail_share", 1.0) or 1.0) for r in hist]
    fail_delta = round((fail_series[-1] - fail_series[0]), 6) if len(fail_series) >= 2 else 0.0
    trend = "improving" if fail_delta < 0 else ("worsening" if fail_delta > 0 else "flat")

    top_bots = diagnostics.get("top_failing_bots") if isinstance(diagnostics.get("top_failing_bots"), list) else []
    top_bots = [r for r in top_bots if isinstance(r, dict)][:10]

    fail_by_segment = readiness.get("failed_by_segment") if isinstance(readiness.get("failed_by_segment"), dict) else {}
    reg_rows = regime.get("segments") if isinstance(regime.get("segments"), dict) else {}
    seg_table = []
    for seg, fail_count in fail_by_segment.items():
        pass_rate = float((reg_rows.get(seg, {}) if isinstance(reg_rows.get(seg), dict) else {}).get("pass_rate", 0.0) or 0.0)
        seg_table.append({"segment": seg, "fail_count": int(fail_count or 0), "pass_rate": round(pass_rate, 6)})
    seg_table.sort(key=lambda r: (-int(r["fail_count"]), float(r["pass_rate"]), str(r["segment"])))

    recommendations = [
        "Keep promotion gate strict until fail_share trend is improving.",
        "Prioritize retrain on top recurring failed bots (canary diagnostics list).",
        "Use regime-focused retrain on segments with highest fail_count and lowest pass_rate.",
        "Avoid adding newly immature bots into promotion math until graduation gate passes.",
    ]

    payload = {
        "timestamp_utc": now.isoformat(),
        "lookback_days": int(args.lookback_days),
        "current_gate": {
            "promote_ok": bool(gate.get("promote_ok", False)),
            "fail_share": float(gate.get("fail_share", 1.0) or 1.0),
            "max_fail_share": float(((gate.get("thresholds") or {}) if isinstance(gate.get("thresholds"), dict) else {}).get("max_fail_share", 0.25) or 0.25),
            "considered_bots": int(gate.get("considered_bots", 0) or 0),
            "failed_bots": int(gate.get("failed_bots", 0) or 0),
        },
        "trend": {
            "samples": len(fail_series),
            "fail_share_series": fail_series,
            "fail_share_delta": fail_delta,
            "direction": trend,
        },
        "top_failing_bots": top_bots,
        "segment_blockers": seg_table[:10],
        "recommendations": recommendations,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        f"# Weekly Gate Blocker Report ({payload['timestamp_utc']})",
        f"- promote_ok: {payload['current_gate']['promote_ok']}",
        f"- fail_share: {payload['current_gate']['fail_share']:.6f} / {payload['current_gate']['max_fail_share']:.6f}",
        f"- considered_bots: {payload['current_gate']['considered_bots']}",
        f"- failed_bots: {payload['current_gate']['failed_bots']}",
        f"- trend({int(args.lookback_days)}d): {trend} delta={fail_delta:.6f}",
        "",
        "## Top Bot Blockers",
    ]
    for row in top_bots:
        lines.append(f"- {row.get('bot_id')} fail_days={row.get('fail_days')}")

    lines.append("")
    lines.append("## Segment Blockers")
    for row in seg_table[:10]:
        lines.append(
            f"- {row.get('segment')}: fail_count={row.get('fail_count')} pass_rate={float(row.get('pass_rate', 0.0)):.4f}"
        )

    lines.append("")
    lines.append("## Recommendations")
    for rec in recommendations:
        lines.append(f"- {rec}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "weekly_gate_blocker_report "
            f"promote_ok={str(payload['current_gate']['promote_ok']).lower()} "
            f"trend={trend}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
