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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "roster_resilience_planner_latest.json"


CURRENT_REGIME_REPLACEMENT_HINTS: list[dict[str, Any]] = [
    {
        "bot_id": "brain_refinery_v101_guard_heavy_regime_memory",
        "bot_role": "signal_sub_bot",
        "fit_regimes": ["risk_off_shock", "risk_off_trend", "fragile_transition"],
        "priority": 96.0,
        "rationale": "Guard-heavy memory sleeve for elevated stress and unstable follow-through.",
        "activation_path": "new_runtime_candidate",
    },
    {
        "bot_id": "brain_refinery_v102_open_drive_liquidity_pressure",
        "bot_role": "signal_sub_bot",
        "fit_regimes": ["risk_off_shock", "open_drive_stress", "thin_liquidity"],
        "priority": 94.0,
        "rationale": "Open-drive pressure sleeve for disorderly opens, gap continuation, and fast liquidity deterioration.",
        "activation_path": "new_runtime_candidate",
    },
    {
        "bot_id": "brain_refinery_v105_feed_consensus_execution_guard",
        "bot_role": "infrastructure_sub_bot",
        "fit_regimes": ["risk_off_shock", "thin_liquidity", "event_volatility"],
        "priority": 93.0,
        "rationale": "Consensus execution guard to keep degraded inputs from leaking into active traffic.",
        "activation_path": "new_runtime_candidate",
    },
    {
        "bot_id": "brain_refinery_v106_cross_asset_regime_stability_guard",
        "bot_role": "infrastructure_sub_bot",
        "fit_regimes": ["risk_off_shock", "fragile_transition", "risk_off_trend"],
        "priority": 92.0,
        "rationale": "Cross-asset stability guard to confirm or reject stressed regime shifts before sleeve thaw.",
        "activation_path": "new_runtime_candidate",
    },
    {
        "bot_id": "brain_refinery_v104_futures_event_followthrough",
        "bot_role": "signal_sub_bot",
        "fit_regimes": ["risk_off_shock", "event_volatility", "fragile_transition"],
        "priority": 90.0,
        "rationale": "Event follow-through sleeve for futures-led repricing and macro shock continuation.",
        "activation_path": "new_runtime_candidate",
    },
]


def _normalize_regime(value: Any) -> str:
    return str(value or "").strip().lower()


def _regime_fit_score(fit_regimes: list[Any], live_regime: str) -> int:
    live = _normalize_regime(live_regime)
    if not live:
        return 0
    preferred = {_normalize_regime(item) for item in fit_regimes if _normalize_regime(item)}
    if live in preferred:
        return 3
    if live.startswith("risk_off") and {"bearish", "fragile_transition", "event_volatility"} & preferred:
        return 2
    return 0


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _registry_row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = row
    return out


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    registry = load_json(project_root / "master_bot_registry.json")
    supportability = load_json(health_root / "supportability_control_latest.json")
    requalification = load_json(health_root / "training_requalification_latest.json")
    coverage_seed = load_json(walk_root / "coverage_seed_latest.json")
    new_bot_graduation = load_json(walk_root / "new_bot_graduation_latest.json")
    roster_expansion = load_json(health_root / "roster_expansion_slots_latest.json")
    regime_control = load_json(health_root / "regime_control_plane_latest.json")

    registry_map = _registry_row_map(_registry_rows(registry))
    supportability_row = supportability.get("supportability") if isinstance(supportability.get("supportability"), dict) else {}
    teacher_student = supportability.get("teacher_student") if isinstance(supportability.get("teacher_student"), dict) else {}
    teacher_quality = supportability.get("teacher_quality") if isinstance(supportability.get("teacher_quality"), dict) else {}
    active_bots = int(supportability_row.get("active_bots", 0) or 0)
    active_supportable_bots = int(supportability_row.get("active_supportable_bots", 0) or 0)
    teacher_count = int(teacher_student.get("teacher_count", 0) or 0)
    students_without_teachers = int(teacher_student.get("students_without_teachers", 0) or 0)
    elite_teacher_count = int(teacher_quality.get("elite_teacher_count", 0) or 0)
    reactivation_ready_count = int(requalification.get("reactivation_ready_count", 0) or 0)
    top_candidates = requalification.get("top_candidates") if isinstance(requalification.get("top_candidates"), list) else []
    mature_bots = int(((new_bot_graduation.get("maturity") or {}).get("mature_bots", 0)) or 0)
    coverage_shortfall_bots = int(coverage_seed.get("coverage_shortfall_bots", 0) or 0)
    roster_expansion_summary = roster_expansion.get("summary") if isinstance(roster_expansion.get("summary"), dict) else {}
    planned_slot_count = int(roster_expansion_summary.get("planned_slot_count", 0) or 0)
    registered_slot_count = int(roster_expansion_summary.get("registered_slot_count", 0) or 0)
    missing_slot_count = int(roster_expansion_summary.get("missing_slot_count", 0) or 0)
    critical_slots_missing = list(roster_expansion_summary.get("critical_slots_missing") or [])
    live_regime = str(regime_control.get("regime_state") or roster_expansion_summary.get("live_regime") or "").strip() or "unknown"

    bench_depth = active_supportable_bots + reactivation_ready_count + mature_bots
    supportable_replacement_shortlist = [
        {
            "bot_id": str(row.get("bot_id") or ""),
            "bot_role": str(row.get("bot_role") or ""),
            "priority": float(row.get("priority", 0.0) or 0.0),
            "walk_forward_status": str(row.get("walk_forward_status") or ""),
            "actions": list(row.get("actions") or [])[:4],
        }
        for row in top_candidates
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    ][:5]
    regime_fit_replacements: list[dict[str, Any]] = []
    for hint in CURRENT_REGIME_REPLACEMENT_HINTS:
        fit_score = _regime_fit_score(list(hint.get("fit_regimes") or []), live_regime)
        if fit_score <= 0:
            continue
        bot_id = str(hint.get("bot_id") or "").strip().lower()
        registry_row = registry_map.get(bot_id, {})
        regime_fit_replacements.append(
            {
                "bot_id": bot_id,
                "bot_role": str(hint.get("bot_role") or ""),
                "priority": float(hint.get("priority", 0.0) or 0.0),
                "reason": str(registry_row.get("reason") or hint.get("activation_path") or ""),
                "promotion_reason": str(registry_row.get("promotion_reason") or ""),
                "active": bool(registry_row.get("active", False)),
                "lifecycle_state": str(registry_row.get("lifecycle_state") or ""),
                "regime_fit_score": fit_score,
                "fit_regimes": list(hint.get("fit_regimes") or []),
                "rationale": str(hint.get("rationale") or ""),
            }
        )
    regime_fit_replacements.sort(
        key=lambda row: (
            -int(row.get("regime_fit_score", 0) or 0),
            -float(row.get("priority", 0.0) or 0.0),
            str(row.get("bot_id") or ""),
        )
    )
    regime_priority_slots = [
        {
            "bot_id": str(row.get("bot_id") or ""),
            "bot_role": str(row.get("bot_role") or ""),
            "priority": str(row.get("priority") or ""),
            "slot_label": str(row.get("slot_label") or ""),
            "registered": bool(row.get("registered", False)),
            "regime_fit_score": int(row.get("regime_fit_score", 0) or 0),
        }
        for row in (roster_expansion.get("current_regime_priority_slots") or [])
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    ][:5]
    a_plus_contract = {
        "active_supportable_target": 3,
        "bench_depth_target": 5,
        "coverage_shortfall_target": 0,
        "active_supportable_bots": active_supportable_bots,
        "bench_depth": bench_depth,
        "coverage_shortfall_bots": coverage_shortfall_bots,
        "a_plus_ready": bool(active_supportable_bots >= 3 and bench_depth >= 5 and coverage_shortfall_bots <= 0),
    }
    overall_status = "ready"
    if bench_depth < 2 or teacher_count == 0:
        overall_status = "blocked"
    elif coverage_shortfall_bots > 0 or students_without_teachers > 0 or elite_teacher_count == 0 or missing_slot_count > 0:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "promote at least two supportable active bots before trusting a multi-week run" if active_supportable_bots < 2 else "",
            "pair infrastructure and signal students with qualified teachers so the bench can actually rotate" if teacher_count == 0 or students_without_teachers > 0 else "",
            "promote or reactivate at least one elite teacher so bot-quality mentorship is grounded in a proven performer" if elite_teacher_count == 0 else "",
            "refresh top reactivation candidates to widen the fallback bench" if reactivation_ready_count == 0 else "",
            "close walk-forward coverage debt so replacement bots are truly ready to take traffic" if coverage_shortfall_bots > 0 else "",
            "graduate at least three supportable replacements before calling the bench A+ resilient" if not a_plus_contract["a_plus_ready"] else "",
            "sync the planned roster expansion slots into the registry so the bench roadmap reflects the intended 10-slot buildout" if missing_slot_count > 0 else "",
            "train the critical roster expansion slots so infrastructure, signal, and options gaps close in a balanced order" if registered_slot_count > 0 else "",
            "stage the current regime-fit runtime candidates before inventing new sleeves" if regime_fit_replacements else "",
            f"bias the {live_regime} bench toward {', '.join(str(row.get('bot_id') or '') for row in regime_fit_replacements[:3])}" if regime_fit_replacements and _normalize_regime(live_regime).startswith("risk_off") else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "bench": {
            "active_bots": active_bots,
            "active_supportable_bots": active_supportable_bots,
            "teacher_count": teacher_count,
            "elite_teacher_count": elite_teacher_count,
            "students_without_teachers": students_without_teachers,
            "reactivation_ready_count": reactivation_ready_count,
            "mature_bots": mature_bots,
            "bench_depth": bench_depth,
        },
        "coverage": {
            "coverage_shortfall_bots": coverage_shortfall_bots,
            "seed_queue_size": int(((coverage_seed.get("standing_queue") or {}).get("seed_queue_size", 0)) or 0),
        },
        "replacement_shortlist": supportable_replacement_shortlist,
        "current_regime": {
            "live_regime": live_regime,
            "regime_fit_replacement_count": len(regime_fit_replacements),
            "regime_fit_replacements": regime_fit_replacements[:5],
            "regime_priority_slots": regime_priority_slots,
        },
        "a_plus_contract": a_plus_contract,
        "roster_expansion": {
            "planned_slot_count": planned_slot_count,
            "registered_slot_count": registered_slot_count,
            "missing_slot_count": missing_slot_count,
            "critical_slots_missing": critical_slots_missing[:4],
        },
        "infra_bots": ["roster_expansion_slots", "roster_resilience_planner", "supportability_control", "training_requalification_lane", "walk_forward_coverage_seed"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan roster depth, fallback champions, and teacher coverage for long runtime windows.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "roster_resilience_planner "
            f"overall_status={payload.get('overall_status', '')} "
            f"bench_depth={int(((payload.get('bench') or {}).get('bench_depth', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
