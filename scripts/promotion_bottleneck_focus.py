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


def _strategy_bot_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if "::" in text:
        return text.split("::", 1)[1].strip().lower()
    return text.lower()


def _failure_reason_summary(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    marker = "runtime_training_quality_guard_failed"
    if marker in text:
        text = text.split(marker, 1)[-1].strip()
    return text[:240]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build promotion bottleneck focus plan for targeted retrain.")
    parser.add_argument("--readiness-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_latest.json"))
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--diagnostics-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "canary_diagnostics_latest.json"))
    parser.add_argument("--regime-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "regime_segmented_latest.json"))
    parser.add_argument("--training-success-file", default=str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"))
    parser.add_argument("--paper-performance-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"))
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--top-bots", type=int, default=20)
    parser.add_argument("--top-segments", type=int, default=2)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_bottleneck_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    readiness = _load(Path(args.readiness_file))
    diagnostics = _load(Path(args.diagnostics_file))
    regime = _load(Path(args.regime_file))
    training_success = _load(Path(args.training_success_file))
    paper_performance = _load(Path(args.paper_performance_file))

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
    sleeve_rows = paper_performance.get("sleeve_latest") if isinstance(paper_performance.get("sleeve_latest"), list) else []
    bot_to_sleeves: dict[str, list[str]] = {}
    weak_sleeves = []
    for sleeve in sleeve_rows:
        if not isinstance(sleeve, dict):
            continue
        profile = str(sleeve.get("profile") or "").strip().lower()
        if not profile:
            continue
        win_rate_raw = sleeve.get("win_rate")
        weak_sleeves.append(
            {
                "profile": profile,
                "ending_net_pnl_total": round(float(sleeve.get("ending_net_pnl_total", 0.0) or 0.0), 6),
                "win_rate": (round(float(win_rate_raw), 6) if win_rate_raw is not None else None),
                "winning_strategy_count": int(sleeve.get("winning_strategy_count", 0) or 0),
                "losing_strategy_count": int(sleeve.get("losing_strategy_count", 0) or 0),
                "flat_strategy_count": int(sleeve.get("flat_strategy_count", 0) or 0),
            }
        )
        ranked = []
        for key in ("top_losing_strategies", "top_winning_strategies"):
            values = sleeve.get(key) if isinstance(sleeve.get(key), list) else []
            ranked.extend(values)
        for row in ranked:
            if not isinstance(row, dict):
                continue
            bot_id = _strategy_bot_id(str(row.get("strategy") or ""))
            if not bot_id:
                continue
            bot_to_sleeves.setdefault(bot_id, [])
            if profile not in bot_to_sleeves[bot_id]:
                bot_to_sleeves[bot_id].append(profile)
    weak_sleeves.sort(
        key=lambda row: (
            1.0 if row.get("win_rate") is None else float(row.get("win_rate")),
            float(row.get("ending_net_pnl_total", 0.0) or 0.0),
            row.get("profile", ""),
        )
    )

    top_bots = []
    seen_bots: set[str] = set()
    training_failures = training_success.get("failure_details") if isinstance(training_success.get("failure_details"), list) else []
    latest_training_failures = []
    for row in training_failures[: max(int(args.top_bots), 1)]:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        seen_bots.add(bot_id)
        latest_training_failures.append(
            {
                "bot_id": bot_id,
                "latest_reason": _failure_reason_summary(str(row.get("reason") or "")),
                "observed_live_sleeves": bot_to_sleeves.get(bot_id, []),
            }
        )
        top_bots.append(
            {
                "bot_id": bot_id,
                "fail_days": int(row.get("fail_days", 0) or 0),
                "latest_reason": _failure_reason_summary(str(row.get("reason") or "")),
                "observed_live_sleeves": bot_to_sleeves.get(bot_id, []),
            }
        )

    for row in top_bots_raw[: max(int(args.top_bots), 1)]:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id or bot_id in seen_bots:
            continue
        seen_bots.add(bot_id)
        top_bots.append(
            {
                "bot_id": bot_id,
                "fail_days": int(row.get("fail_days", 0) or 0),
                "observed_live_sleeves": bot_to_sleeves.get(bot_id, []),
            }
        )

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
    targeted_bot_ids = [str(row.get("bot_id") or "").strip() for row in top_bots if str(row.get("bot_id") or "").strip()]

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
        "latest_training_failures": latest_training_failures,
        "weak_sleeves": weak_sleeves[:5],
        "recommended_retrain_profile": {
            "RETRAIN_REGIME_FOCUS": regime_focus,
            "RETRAIN_CANARY_PRIORITY_TOP_N": int(canary_top_n),
            "RETRAIN_MAX_TARGETS": int(max_targets),
            "RETRAIN_MIN_MODEL_AGE_HOURS": 12 if trend_direction != "improving" else 18,
            "RETRAIN_INCLUDE_BOT_IDS": ",".join(targeted_bot_ids),
            "RETRAIN_SKIP_MASTER_UPDATE": bool(targeted_bot_ids),
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
