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
    rows: list[dict] = []
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
    parser = argparse.ArgumentParser(description="Build promotion bottleneck focus plan for targeted retrain.")
    parser.add_argument("--readiness-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_latest.json"))
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--diagnostics-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "canary_diagnostics_latest.json"))
    parser.add_argument("--regime-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "regime_segmented_latest.json"))
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--top-bots", type=int, default=20)
    parser.add_argument("--top-segments", type=int, default=2)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_bottleneck_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    readiness = _load(Path(args.readiness_file))
    diagnostics = _load(Path(args.diagnostics_file))
    regime = _load(Path(args.regime_file))

    fail_by_segment = readiness.get("failed_by_segment") if isinstance(readiness.get("failed_by_segment"), dict) else {}
    seg_rows = regime.get("segments") if isinstance(regime.get("segments"), dict) else {}

    ranked_segments = []
    for seg, fail_count in fail_by_segment.items():
        pass_rate = float((seg_rows.get(seg, {}) if isinstance(seg_rows.get(seg), dict) else {}).get("pass_rate", 0.0) or 0.0)
        score = float(fail_count or 0) * (1.0 + max(0.0, 0.5 - pass_rate))
        ranked_segments.append({
            "segment": str(seg),
            "fail_count": int(fail_count or 0),
            "pass_rate": round(pass_rate, 6),
            "priority_score": round(score, 6),
        })
    ranked_segments.sort(key=lambda r: (-float(r["priority_score"]), -int(r["fail_count"]), r["segment"]))
    top_segments = ranked_segments[: max(int(args.top_segments), 1)]

    top_bots_raw = diagnostics.get("top_failing_bots") if isinstance(diagnostics.get("top_failing_bots"), list) else []
    top_bots = []
    for row in top_bots_raw[: max(int(args.top_bots), 1)]:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        top_bots.append({"bot_id": bot_id, "fail_days": int(row.get("fail_days", 0) or 0)})

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(int(args.lookback_days), 1))
    trend_rows = []
    for row in _read_jsonl(Path(args.history_jsonl)):
        ts = _parse_ts(str(row.get("timestamp_utc", "")))
        if ts is None or ts < cutoff:
            continue
        trend_rows.append(row)

    fail_series = [float((r or {}).get("fail_share", 1.0) or 1.0) for r in trend_rows]
    fail_delta = round((fail_series[-1] - fail_series[0]), 6) if len(fail_series) >= 2 else 0.0
    trend_direction = "improving" if fail_delta < 0 else ("worsening" if fail_delta > 0 else "flat")

    regime_focus = ",".join([str(r["segment"]) for r in top_segments if str(r.get("segment", "")).strip()])
    canary_top_n = max(len(top_bots), 10)
    max_targets = min(max(12, len(top_bots) * 2), 30)

    payload = {
        "timestamp_utc": now.isoformat(),
        "lookback_days": int(args.lookback_days),
        "trend": {
            "fail_share_series": fail_series,
            "fail_share_delta": fail_delta,
            "direction": trend_direction,
        },
        "current": {
            "promote_ok": bool(readiness.get("promote_ok", False)),
            "fail_share": float(readiness.get("fail_share", 1.0) or 1.0),
            "max_fail_share": float(readiness.get("max_fail_share", 0.25) or 0.25),
            "readiness_margin": float(readiness.get("readiness_margin", -1.0) or -1.0),
        },
        "top_segments": top_segments,
        "top_failing_bots": top_bots,
        "recommended_retrain_profile": {
            "RETRAIN_REGIME_FOCUS": regime_focus,
            "RETRAIN_CANARY_PRIORITY_TOP_N": int(canary_top_n),
            "RETRAIN_MAX_TARGETS": int(max_targets),
            "RETRAIN_MIN_MODEL_AGE_HOURS": 12 if trend_direction != "improving" else 18,
        },
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "promotion_bottleneck_focus "
            f"direction={trend_direction} "
            f"focus={regime_focus or 'none'} "
            f"top_bots={len(top_bots)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
