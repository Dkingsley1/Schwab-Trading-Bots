import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTS = {
    "trend": ["trend", "breakout", "donchian", "momentum", "dmi", "relative_strength", "gap_open", "seasonal"],
    "mean_revert": ["mean_revert", "vwap", "bollinger", "keltner", "bond", "dividend", "yield", "income", "defensive", "drip", "compound", "quality", "allocator", "risk_budget"],
    "shock": ["flash", "shock", "event", "crash", "anomaly", "macro", "inflation", "pmi", "rates", "credit", "futures", "term_structure", "vol", "news"],
    "liquidity": ["liquidity", "spread", "order_flow", "microstructure", "execution", "latency", "position_1m_3m"],
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


def _fail_priority(row: dict) -> tuple[float, float, str]:
    bot_id = str((row or {}).get("bot_id", "")).strip()
    failed = (row or {}).get("failed_gates", {}) if isinstance((row or {}).get("failed_gates"), dict) else {}
    forward_mean = float((row or {}).get("forward_mean", 0.0) or 0.0)
    delta = float((row or {}).get("delta", 0.0) or 0.0)
    severity = 0.0
    if bool(failed.get("forward_mean")):
        severity += max(0.53 - forward_mean, 0.0) + 0.5
    if bool(failed.get("delta")):
        severity += max((-0.01) - delta, 0.0) + 0.75
    if bool(failed.get("trading_quality_score")):
        severity += 0.25
    if bool(failed.get("overfit_gap")):
        severity += 0.2
    return (-severity, float((row or {}).get("runs", 0) or 0.0), bot_id)


def _recommended_regime_focus(seg_counts: dict[str, int], top_n: int = 2) -> str:
    ranked = sorted(seg_counts.items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))
    picks = [seg for seg, _ in ranked if seg in {"trend", "mean_revert", "shock", "liquidity", "other"}]
    return ",".join(picks[: max(int(top_n), 1)])


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
    near_pass_examples = gate.get("near_pass_examples") if isinstance(gate.get("near_pass_examples"), list) else []

    failed_bots = [str((r or {}).get("bot_id", "")).strip() for r in fail_examples if str((r or {}).get("bot_id", "")).strip()]
    failed_bots = sorted(set(failed_bots))
    near_pass_bots = [str((r or {}).get("bot_id", "")).strip() for r in near_pass_examples if str((r or {}).get("bot_id", "")).strip()]
    near_pass_bots = sorted(set(near_pass_bots))

    wf_bots = wf.get("bots") if isinstance(wf.get("bots"), dict) else {}
    seg_counts: dict[str, int] = {}
    fail_examples_scoped: list[dict] = []
    for fail_row in fail_examples:
        if not isinstance(fail_row, dict):
            continue
        bot_id = str(fail_row.get("bot_id", "")).strip()
        if not bot_id:
            continue
        wf_row = wf_bots.get(bot_id, {}) if isinstance(wf_bots, dict) else {}
        seg = _segment(bot_id)
        # prefer explicit tag if upstream ever adds it
        if isinstance(wf_row, dict) and isinstance(wf_row.get("segment"), str) and wf_row.get("segment"):
            seg = str(wf_row.get("segment")).strip().lower()
        seg_counts[seg] = seg_counts.get(seg, 0) + 1
        fail_examples_scoped.append({**fail_row, **{"segment": seg}})

    prioritized_fail_examples = sorted(fail_examples_scoped, key=_fail_priority)
    recommended_bot_ids = [str(row.get("bot_id", "")).strip() for row in prioritized_fail_examples[:3] if str(row.get("bot_id", "")).strip()]
    recommended_regime_focus = _recommended_regime_focus(seg_counts)
    canary_watchlist = [str((row or {}).get("bot_id", "")).strip() for row in near_pass_examples[:5] if str((row or {}).get("bot_id", "")).strip()]

    now = datetime.now(timezone.utc).isoformat()
    raw_fail_share = gate.get("fail_share", 1.0)
    fail_share = float(1.0 if raw_fail_share is None else raw_fail_share)
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
        "near_pass_examples": near_pass_examples[:10],
        "failed_bots_list": failed_bots,
        "near_pass_bots_list": near_pass_bots,
        "failed_by_segment": seg_counts,
        "recommended_retrain": {
            "include_bot_ids": recommended_bot_ids,
            "regime_focus": recommended_regime_focus,
            "top_fail_segments": sorted(seg_counts.items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))[:3],
            "watchlist_bot_ids": canary_watchlist,
        },
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
