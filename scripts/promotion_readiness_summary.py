import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTS = {
    "trend": ["trend", "breakout", "donchian"],
    "mean_revert": ["mean_revert", "vwap", "bollinger", "keltner"],
    "shock": ["flash", "shock", "event", "crash", "anomaly"],
    "liquidity": ["liquidity", "spread", "order_flow", "microstructure"],
}


def _segment(bot_id: str) -> str:
    b = (bot_id or "").lower()
    for seg, keys in SEGMENTS.items():
        if any(k in b for k in keys):
            return seg
    return "other"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_recent_jsonl(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-max(limit, 1) :]


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily promotion readiness summary and history.")
    parser.add_argument("--gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--walk-forward-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--latest-out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_latest.json"))
    parser.add_argument("--fail-list-out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_fail_bots_latest.json"))
    parser.add_argument("--trend-window", type=int, default=14)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    gate = _load(Path(args.gate_file))
    wf = _load(Path(args.walk_forward_file))

    thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
    fail_examples = gate.get("fail_examples") if isinstance(gate.get("fail_examples"), list) else []

    failed_bots = [str((r or {}).get("bot_id", "")).strip() for r in fail_examples if str((r or {}).get("bot_id", "")).strip()]
    failed_bots = sorted(set(failed_bots))

    wf_bots = wf.get("bots") if isinstance(wf.get("bots"), dict) else {}
    seg_counts: dict[str, int] = {}
    for bot_id in failed_bots:
        row = wf_bots.get(bot_id, {}) if isinstance(wf_bots, dict) else {}
        seg = _segment(bot_id)
        # prefer explicit tag if upstream ever adds it
        if isinstance(row, dict) and isinstance(row.get("segment"), str) and row.get("segment"):
            seg = str(row.get("segment")).strip().lower()
        seg_counts[seg] = seg_counts.get(seg, 0) + 1

    now = datetime.now(timezone.utc).isoformat()
    fail_share = float(gate.get("fail_share", 1.0) or 1.0)
    max_fail_share = float(thresholds.get("max_fail_share", 0.25) or 0.25)
    readiness_margin = round(max_fail_share - fail_share, 6)

    latest_row = {
        "timestamp_utc": now,
        "promote_ok": bool(gate.get("promote_ok", False)),
        "coverage_ok": bool(gate.get("coverage_ok", False)),
        "considered_bots": int(gate.get("considered_bots", 0) or 0),
        "failed_bots": int(gate.get("failed_bots", 0) or 0),
        "fail_share": round(fail_share, 6),
        "max_fail_share": round(max_fail_share, 6),
        "readiness_margin": readiness_margin,
        "thresholds": {
            "min_forward_mean": thresholds.get("min_forward_mean"),
            "min_delta": thresholds.get("min_delta"),
            "min_runs_per_bot": thresholds.get("min_runs_per_bot"),
            "min_considered_bots": thresholds.get("min_considered_bots"),
        },
        "top_fail_examples": fail_examples[:10],
        "failed_bots_list": failed_bots,
        "failed_by_segment": seg_counts,
    }

    history_path = Path(args.history_jsonl)
    _append_jsonl(history_path, latest_row)
    hist = _read_recent_jsonl(history_path, max(int(args.trend_window), 2))

    fail_trend = [float((r or {}).get("fail_share", 1.0) or 1.0) for r in hist]
    considered_trend = [int((r or {}).get("considered_bots", 0) or 0) for r in hist]
    fail_delta = round((fail_trend[-1] - fail_trend[0]), 6) if len(fail_trend) >= 2 else 0.0

    payload = {
        **latest_row,
        "trend_window": len(hist),
        "trend": {
            "fail_share_series": fail_trend,
            "considered_bots_series": considered_trend,
            "fail_share_delta": fail_delta,
            "direction": "improving" if fail_delta < 0 else ("worsening" if fail_delta > 0 else "flat"),
        },
    }

    latest_out = Path(args.latest_out)
    latest_out.parent.mkdir(parents=True, exist_ok=True)
    latest_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    fail_out = Path(args.fail_list_out)
    fail_out.parent.mkdir(parents=True, exist_ok=True)
    fail_out.write_text(json.dumps({
        "timestamp_utc": now,
        "failed_bots": failed_bots,
        "failed_by_segment": seg_counts,
    }, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "promotion_readiness "
            f"promote_ok={payload['promote_ok']} "
            f"fail_share={payload['fail_share']:.6f}/{payload['max_fail_share']:.6f} "
            f"margin={payload['readiness_margin']:.6f} "
            f"trend={payload['trend']['direction']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
