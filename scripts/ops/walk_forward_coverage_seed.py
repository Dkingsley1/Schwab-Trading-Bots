#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "walk_forward" / "coverage_seed_latest.json"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "governance" / "walk_forward" / "coverage_seed_queue.jsonl"
DEFAULT_WALK_FORWARD_PATH = PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _load_walk_forward_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("bots") if isinstance(payload.get("bots"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for bot_id, row in rows.items():
        text = str(bot_id or "").strip().lower()
        if text and isinstance(row, dict):
            out[text] = row
    return out


def _queue_bucket(bot_role: str) -> str:
    role = str(bot_role or "").strip().lower()
    if "signal" in role:
        return "signal"
    if "infrastructure" in role:
        return "infrastructure"
    if "options" in role:
        return "options"
    return "general"


def build_payload(project_root: Path = PROJECT_ROOT, *, limit: int = 8) -> dict[str, Any]:
    readiness = _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    requalification = _load_json(project_root / "governance" / "health" / "training_requalification_latest.json")
    walk_forward_rows = _load_walk_forward_rows(project_root / "governance" / "walk_forward" / "walk_forward_latest.json")
    thresholds = readiness.get("thresholds") if isinstance(readiness.get("thresholds"), dict) else {}
    considered_bots = _safe_int(readiness.get("considered_bots"), 0)
    min_considered_bots = max(_safe_int(thresholds.get("min_considered_bots"), 4), 1)
    min_runs_per_bot = max(_safe_int(thresholds.get("min_runs_per_bot"), 12), 1)
    coverage_shortfall_bots = max(min_considered_bots - considered_bots, _safe_int(readiness.get("coverage_shortfall_bots"), 0))
    readiness_margin = _safe_float(readiness.get("readiness_margin"), 0.0)
    blocking_reasons = [str(item).strip() for item in (readiness.get("blocking_reasons") or []) if str(item).strip()]
    staged_candidates = requalification.get("top_reactivation_ready") if isinstance(requalification.get("top_reactivation_ready"), list) else []
    all_candidates = requalification.get("top_candidates") if isinstance(requalification.get("top_candidates"), list) else []
    candidates: list[dict[str, Any]] = []
    for row in list(staged_candidates) + list(all_candidates):
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        if any(str(existing.get("bot_id") or "").strip().lower() == bot_id for existing in candidates):
            continue
        candidates.append(row)
    seed_rows: list[dict[str, Any]] = []
    queue_counts: dict[str, int] = {}
    total_recommended_runs = 0
    for row in candidates[: max(int(limit), coverage_shortfall_bots, 1)]:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        bot_role = str(row.get("bot_role") or "")
        queue_bucket = _queue_bucket(bot_role)
        actions = [str(raw or "").strip() for raw in list(row.get("actions") or []) if str(raw or "").strip()]
        needs_repair = "repair_runtime_inputs" in actions or "refresh_training_diagnostics" in actions
        priority = _safe_float(row.get("priority"), 0.0)
        walk_forward_row = walk_forward_rows.get(bot_id, {})
        current_runs = _safe_int(row.get("walk_forward_runs"), _safe_int(walk_forward_row.get("runs"), 0))
        runs_remaining = max(int(min_runs_per_bot) - current_runs, 0)
        recommended_runs = runs_remaining + (2 if coverage_shortfall_bots >= min_considered_bots else 0)
        total_recommended_runs += int(recommended_runs)
        queue_counts[queue_bucket] = int(queue_counts.get(queue_bucket, 0) + 1)
        seed_rows.append(
            {
                "bot_id": bot_id,
                "bot_role": bot_role,
                "queue_bucket": queue_bucket,
                "priority": priority,
                "coverage_pressure": round(priority * max(float(coverage_shortfall_bots), 1.0), 3),
                "current_runs": int(current_runs),
                "runs_remaining": int(runs_remaining),
                "recommended_runs": int(recommended_runs),
                "needs_runtime_input_repair": needs_repair,
                "actions": _ordered_unique(
                    actions
                    + [
                    "generate_walk_forward_runs",
                    "refresh_promotion_gate",
                    "recheck_promotion_quality_gate",
                    ]
                ),
            }
        )
    overall_status = "ready" if coverage_shortfall_bots <= 0 else "needs_coverage"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "ok": True,
        "overall_status": overall_status,
        "coverage_shortfall_bots": coverage_shortfall_bots,
        "considered_bots": considered_bots,
        "min_considered_bots": min_considered_bots,
        "min_runs_per_bot": min_runs_per_bot,
        "readiness_margin": round(readiness_margin, 6),
        "blocking_reasons": blocking_reasons,
        "shortfall_run_budget": int(coverage_shortfall_bots * min_runs_per_bot),
        "standing_queue": {
            "seed_queue_size": len(seed_rows),
            "total_recommended_runs": int(total_recommended_runs),
            "queue_bucket_counts": queue_counts,
            "repair_before_seed_count": sum(1 for row in seed_rows if bool(row.get("needs_runtime_input_repair", False))),
        },
        "seed_queue": seed_rows,
        "recommended_actions": _ordered_unique(
            [
                "keep a standing walk-forward seed queue so promotion coverage is built continuously instead of only during retrain windows",
                "favor requalification-ready bots first so seed cycles improve both coverage and roster quality",
                "repair runtime inputs before seeding coverage for candidates that still lack usable diagnostics",
                "treat coverage shortfall as a run-budget problem: keep enough seed runs queued to satisfy the promotion floor every day",
            ]
            + (["promotion remains blocked until coverage shortfall is cleared"] if coverage_shortfall_bots > 0 else [])
        ),
    }
    return payload


def _write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously seed walk-forward coverage for promotable bots.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--queue-out", default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--write-queue", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), limit=int(args.limit))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.write_queue:
        _write_queue(Path(args.queue_out).expanduser(), list(payload.get("seed_queue") or []))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "walk_forward_coverage_seed "
            f"coverage_shortfall_bots={int(payload.get('coverage_shortfall_bots', 0) or 0)} "
            f"seed_queue={len(payload.get('seed_queue') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
