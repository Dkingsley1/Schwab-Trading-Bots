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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "roster_expansion_slots_latest.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"

DEFAULT_SLOT_SPECS: list[dict[str, Any]] = [
    {
        "bot_id": "brain_refinery_v107_infra_teacher_execution_quality_champion",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Infrastructure Teacher Champion",
        "slot_kind": "teacher_champion",
        "priority": "critical",
        "objective": "Create a proven infrastructure champion that can mentor risk-budget, execution-quality, and allocator students.",
        "target_functions": ["teacher_quality_guard", "bot_quality_autopilot", "supportability_control", "execution_lifecycle"],
        "preferred_regimes": ["all_weather", "fragile_transition", "risk_off_shock"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v99_defensive_dividend_concentration"],
        "rationale": "Closes the uncovered infrastructure teacher gap so infrastructure students stop borrowing mismatched mentors.",
    },
    {
        "bot_id": "brain_refinery_v108_signal_regime_fallback_champion",
        "bot_role": "signal_sub_bot",
        "slot_label": "Signal Fallback Champion",
        "slot_kind": "fallback_champion",
        "priority": "critical",
        "objective": "Add a slower, supportable signal champion that can carry traffic when faster intraday variants are on probation.",
        "target_functions": ["training_quality_control", "promotion_gate", "roster_resilience"],
        "preferred_regimes": ["risk_off_trend", "mixed_transition", "thin_liquidity"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal"],
        "rationale": "Improves roster resilience by replacing fragile single-bot dependence with a supportable fallback champion.",
    },
    {
        "bot_id": "brain_refinery_v109_defensive_options_risk_off_teacher",
        "bot_role": "options_sub_bot",
        "slot_label": "Defensive Options Teacher",
        "slot_kind": "teacher_champion",
        "priority": "critical",
        "objective": "Build a high-quality defensive options teacher for hedging, convexity, and risk-off overlays.",
        "target_functions": ["teacher_quality_guard", "distill_new_bots", "paper_execution_calibration"],
        "preferred_regimes": ["risk_off_shock", "bearish", "event_volatility"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration"],
        "rationale": "Gives the system a second defensive options mentor instead of relying on a single options teacher family.",
    },
    {
        "bot_id": "brain_refinery_v110_runtime_input_resilience_guard",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Runtime Input Resilience Guard",
        "slot_kind": "infra_guard",
        "priority": "high",
        "objective": "Detect and down-rank incomplete runtime inputs before weak snapshots cascade into retrains or promotions.",
        "target_functions": ["training_runtime_control", "runtime_snapshot_cache_control", "daily_verify_auto_remediation"],
        "preferred_regimes": ["all_weather", "thin_liquidity"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v99_defensive_dividend_concentration"],
        "rationale": "Supports the runtime-repair lane that is currently blocking requalification and coverage seeding.",
    },
    {
        "bot_id": "brain_refinery_v111_slippage_capacity_limiter",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Slippage Capacity Limiter",
        "slot_kind": "execution_guard",
        "priority": "high",
        "objective": "Make execution realism regime-aware by limiting entries when slippage, spread, or queue pressure drift outside budget.",
        "target_functions": ["paper_execution_calibration", "execution_lifecycle", "risk_service"],
        "preferred_regimes": ["thin_liquidity", "event_volatility", "risk_off_shock"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration", "brain_refinery_v10_seasonal"],
        "rationale": "Improves paper-to-live fidelity and gives infrastructure students a high-value execution-quality mentor target.",
    },
    {
        "bot_id": "brain_refinery_v112_cross_asset_regime_confirmation",
        "bot_role": "signal_sub_bot",
        "slot_label": "Cross-Asset Regime Confirmation",
        "slot_kind": "regime_signal",
        "priority": "high",
        "objective": "Confirm or reject equity impulses using rates, FX, crypto, and breadth so regime shifts do not whipsaw the active roster.",
        "target_functions": ["regime_control_plane", "walk_forward_coverage_seed", "promotion_gate"],
        "preferred_regimes": ["mixed_transition", "fragile_transition", "risk_off_trend"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal"],
        "rationale": "Adds regime memory and reduces the chance that one noisy sleeve dominates the signal stack.",
    },
    {
        "bot_id": "brain_refinery_v113_sector_breadth_rotation_fallback",
        "bot_role": "signal_sub_bot",
        "slot_label": "Sector Breadth Rotation Fallback",
        "slot_kind": "fallback_champion",
        "priority": "high",
        "objective": "Use sector breadth and leadership rotation to supply a calmer fallback signal when intraday momentum degrades.",
        "target_functions": ["roster_resilience", "walk_forward_coverage_seed", "promotion_gate"],
        "preferred_regimes": ["risk_off_trend", "mixed_transition", "slow_rotation"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal"],
        "rationale": "Broadens the signal bench with a less fragile style than the current ultrafast-heavy active path.",
    },
    {
        "bot_id": "brain_refinery_v114_overnight_gap_defense_overlay",
        "bot_role": "options_sub_bot",
        "slot_label": "Overnight Gap Defense Overlay",
        "slot_kind": "defensive_overlay",
        "priority": "high",
        "objective": "Add an options overlay focused on gap risk, overnight shock, and open-drive reversal protection.",
        "target_functions": ["paper_execution_calibration", "daily_verify_auto_remediation", "risk_service"],
        "preferred_regimes": ["event_volatility", "open_drive_stress", "risk_off_shock"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration"],
        "rationale": "Extends the defensive options bench into a concrete gap-risk use case that the current stack lacks.",
    },
    {
        "bot_id": "brain_refinery_v115_macro_event_repricing_relay",
        "bot_role": "signal_sub_bot",
        "slot_label": "Macro Event Repricing Relay",
        "slot_kind": "event_signal",
        "priority": "medium",
        "objective": "Capture follow-through after macro and futures shocks without requiring the fastest intraday bot to shoulder the whole move.",
        "target_functions": ["macro_bulletin", "regime_control_plane", "walk_forward_coverage_seed"],
        "preferred_regimes": ["event_volatility", "risk_off_trend", "fragile_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal"],
        "rationale": "Turns macro and futures context into a supportable signal slot rather than leaving it as ad hoc event handling.",
    },
    {
        "bot_id": "brain_refinery_v116_drawdown_circuit_allocator",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Drawdown Circuit Allocator",
        "slot_kind": "allocator_guard",
        "priority": "medium",
        "objective": "Reallocate or throttle sleeves when drawdown, correlation, and supportability signals deteriorate together.",
        "target_functions": ["supportability_control", "risk_service", "portfolio_allocator"],
        "preferred_regimes": ["risk_off_shock", "fragile_transition", "all_weather"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v99_defensive_dividend_concentration"],
        "rationale": "Adds a practical infrastructure allocator slot that can protect multi-week runtime stability when one sleeve degrades.",
    },
    {
        "bot_id": "brain_refinery_v117_iv_skew_dislocation_overlay",
        "bot_role": "options_sub_bot",
        "slot_label": "IV Skew Dislocation Overlay",
        "slot_kind": "vol_surface_overlay",
        "priority": "high",
        "objective": "Exploit or defend against skew dislocations when downside hedging demand and dealer positioning diverge from realized stress.",
        "target_functions": ["collect_options_flow_context", "paper_execution_calibration", "risk_service"],
        "preferred_regimes": ["event_volatility", "risk_off_shock", "fragile_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration"],
        "rationale": "Adds a dedicated skew-aware overlay so the options bench is not limited to broad defensive rotation and term-structure logic.",
    },
    {
        "bot_id": "brain_refinery_v118_earnings_convexity_event_overlay",
        "bot_role": "options_sub_bot",
        "slot_label": "Earnings Convexity Overlay",
        "slot_kind": "event_overlay",
        "priority": "medium",
        "objective": "Model pre-event convexity and post-event vol crush so the stack can hedge or lean into single-name earnings dislocations more deliberately.",
        "target_functions": ["macro_bulletin", "collect_options_flow_context", "paper_execution_calibration"],
        "preferred_regimes": ["event_volatility", "open_drive_stress", "mixed_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration", "brain_refinery_v31_defensive_rotation"],
        "rationale": "Extends the options roster beyond index-style defense into a repeatable single-name event playbook.",
    },
    {
        "bot_id": "brain_refinery_v119_put_call_stress_reversal_overlay",
        "bot_role": "options_sub_bot",
        "slot_label": "Put/Call Stress Reversal Overlay",
        "slot_kind": "stress_reversal_overlay",
        "priority": "medium",
        "objective": "React to extreme put-call, gamma, and flow imbalance conditions that often precede violent squeezes or air-pocket reversals.",
        "target_functions": ["collect_options_flow_context", "regime_control_plane", "risk_service"],
        "preferred_regimes": ["risk_off_shock", "thin_liquidity", "mixed_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration", "brain_refinery_v27_term_structure_vol"],
        "rationale": "Gives the options sleeve a dedicated stress-reversal specialist instead of forcing broader defensive bots to cover that lane.",
    },
]


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


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


def _teacher_index(payload: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    rows = payload.get("qualified_teachers") if isinstance(payload.get("qualified_teachers"), list) else []
    by_role: dict[str, list[str]] = {}
    overall: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        role = str(row.get("bot_role") or "unknown").strip() or "unknown"
        if not bot_id:
            continue
        if bot_id not in overall:
            overall.append(bot_id)
        bucket = by_role.setdefault(role, [])
        if bot_id not in bucket:
            bucket.append(bot_id)
    return by_role, overall


def _gap_by_role(payload: dict[str, Any]) -> dict[str, int]:
    rows = ((payload.get("teacher_student") or {}).get("teacher_gap_by_role") or []) if isinstance(payload, dict) else []
    out: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("student_role") or "unknown").strip() or "unknown"
        out[role] = _safe_int(row.get("missing_assignments"), 0)
    return out


def _seed_queue_by_role(payload: dict[str, Any]) -> dict[str, int]:
    rows = payload.get("seed_queue") if isinstance(payload.get("seed_queue"), list) else []
    out: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("bot_role") or "unknown").strip() or "unknown"
        out[role] = int(out.get(role, 0) + 1)
    return out


def _priority_value(priority: str) -> int:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return order.get(str(priority or "").strip().lower(), 4)


def _normalize_regime(value: Any) -> str:
    return str(value or "").strip().lower()


def _regime_fit_score(preferred_regimes: list[Any], live_regime: str) -> int:
    live = _normalize_regime(live_regime)
    if not live:
        return 0
    preferred = {_normalize_regime(item) for item in preferred_regimes if _normalize_regime(item)}
    if live in preferred:
        return 3
    if "all_weather" in preferred:
        return 1
    if live.startswith("risk_off") and {"bearish", "fragile_transition", "event_volatility"} & preferred:
        return 2
    return 0


def _slot_registry_row(slot: dict[str, Any]) -> dict[str, Any]:
    return {
        "bot_id": str(slot.get("bot_id") or ""),
        "bot_role": str(slot.get("bot_role") or "unknown"),
        "active": False,
        "reason": "planned_roster_expansion_slot",
        "weight": 0.0,
        "preference_score": 0.0,
        "quality_score": 0.0,
        "test_accuracy": None,
        "candidate_test_accuracy": None,
        "candidate_quality_score": 0.0,
        "previous_best_accuracy": None,
        "no_improvement_streak": 0,
        "deleted_from_rotation": False,
        "delete_reason": "",
        "promoted": False,
        "promotion_reason": "planned_roster_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "inactive",
        "slot_label": str(slot.get("slot_label") or ""),
        "slot_kind": str(slot.get("slot_kind") or ""),
        "slot_priority": str(slot.get("priority") or "medium"),
        "slot_objective": str(slot.get("objective") or ""),
        "target_functions": list(slot.get("target_functions") or []),
        "preferred_regimes": list(slot.get("preferred_regimes") or []),
        "bootstrap_teacher_bot_ids": list(slot.get("bootstrap_teacher_bot_ids") or []),
    }


def _refresh_registry_summary(payload: dict[str, Any]) -> None:
    rows = _registry_rows(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    active_rows = [row for row in rows if bool(row.get("active", False))]
    summary["total_bots"] = len(rows)
    summary["active_bots"] = len(active_rows)
    summary["inactive_bots"] = max(len(rows) - len(active_rows), 0)
    summary["active_signal_sub_bots"] = sum(1 for row in rows if bool(row.get("active", False)) and str(row.get("bot_role") or "") == "signal_sub_bot")
    summary["active_infrastructure_sub_bots"] = sum(1 for row in rows if bool(row.get("active", False)) and str(row.get("bot_role") or "") == "infrastructure_sub_bot")
    summary["inactive_signal_sub_bots"] = sum(1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "signal_sub_bot")
    summary["inactive_infrastructure_sub_bots"] = sum(1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "infrastructure_sub_bot")
    payload["summary"] = summary
    payload["updated_at_utc"] = iso_now()


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = load_json(project_root / "master_bot_registry.json")
    teacher_quality = load_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json")
    supportability = load_json(project_root / "governance" / "health" / "supportability_control_latest.json")
    coverage_seed = load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")
    regime_control = load_json(project_root / "governance" / "health" / "regime_control_plane_latest.json")

    registry_map = _registry_row_map(_registry_rows(registry))
    teacher_by_role, teacher_overall = _teacher_index(teacher_quality)
    gap_by_role = _gap_by_role(supportability)
    seed_by_role = _seed_queue_by_role(coverage_seed)
    live_regime = str(regime_control.get("regime_state") or "").strip() or "unknown"

    role_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    teacher_anchor_missing_roles: set[str] = set()
    slots: list[dict[str, Any]] = []
    critical_slots_missing: list[str] = []
    regime_priority_slots: list[dict[str, Any]] = []
    registered_count = 0
    teacher_anchor_ready_count = 0

    for spec in sorted(
        DEFAULT_SLOT_SPECS,
        key=lambda row: (
            -_regime_fit_score(list(row.get("preferred_regimes") or []), live_regime),
            _priority_value(str(row.get("priority") or "")),
            str(row.get("bot_id") or ""),
        ),
    ):
        bot_id = str(spec.get("bot_id") or "").strip().lower()
        role = str(spec.get("bot_role") or "unknown").strip() or "unknown"
        role_counts[role] = int(role_counts.get(role, 0) + 1)
        priority = str(spec.get("priority") or "medium").strip().lower() or "medium"
        priority_counts[priority] = int(priority_counts.get(priority, 0) + 1)
        row = registry_map.get(bot_id, {})
        registered = bool(row)
        if registered:
            registered_count += 1
        same_role_teachers = list(teacher_by_role.get(role) or [])
        bootstrap_teachers = same_role_teachers[:3] if same_role_teachers else list((spec.get("bootstrap_teacher_bot_ids") or teacher_overall[:3])[:3])
        teacher_anchor_ready = bool(same_role_teachers)
        if teacher_anchor_ready:
            teacher_anchor_ready_count += 1
        else:
            teacher_anchor_missing_roles.add(role)
        if priority == "critical" and not registered:
            critical_slots_missing.append(bot_id)
        regime_fit_score = _regime_fit_score(list(spec.get("preferred_regimes") or []), live_regime)
        slot_row = {
            **spec,
            "registered": registered,
            "registry_reason": str(row.get("reason") or "") if registered else "",
            "registry_promotion_reason": str(row.get("promotion_reason") or "") if registered else "",
            "same_role_teacher_bot_ids": same_role_teachers[:3],
            "bootstrap_teacher_bot_ids": bootstrap_teachers,
            "teacher_anchor_ready": teacher_anchor_ready,
            "current_teacher_gap": int(gap_by_role.get(role, 0)),
            "current_seed_queue_count": int(seed_by_role.get(role, 0)),
            "current_live_regime": live_regime,
            "regime_fit": regime_fit_score > 0,
            "regime_fit_score": regime_fit_score,
        }
        slots.append(slot_row)
        if regime_fit_score > 0:
            regime_priority_slots.append(
                {
                    "bot_id": bot_id,
                    "bot_role": role,
                    "slot_label": str(spec.get("slot_label") or ""),
                    "priority": priority,
                    "registered": registered,
                    "regime_fit_score": regime_fit_score,
                    "preferred_regimes": list(spec.get("preferred_regimes") or []),
                    "rationale": str(spec.get("rationale") or ""),
                }
            )

    planned_slot_count = len(DEFAULT_SLOT_SPECS)
    missing_slot_count = max(planned_slot_count - registered_count, 0)
    overall_status = "ready" if missing_slot_count <= 0 else "degraded"

    recommended_actions = ordered_unique(
        [
            "sync the planned roster expansion slots into the master registry so the bench roadmap is explicit" if missing_slot_count > 0 else "",
            "train the critical infrastructure teacher slot first so infrastructure students stop relying on cross-role mentors",
            "upgrade the signal fallback champion and defensive options teacher before adding more student-only bots",
            "seed walk-forward coverage for the new signal and infrastructure slots as soon as runtime inputs are healthy",
            "treat the slippage, runtime-input, and drawdown allocator slots as infrastructure prerequisites for multi-week uptime",
            f"prioritize the current {live_regime} buildout around "
            + ", ".join(str(row.get("bot_id") or "") for row in regime_priority_slots[:3])
            if regime_priority_slots
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": missing_slot_count <= 0,
        "overall_status": overall_status,
        "summary": {
            "planned_slot_count": planned_slot_count,
            "registered_slot_count": registered_count,
            "missing_slot_count": missing_slot_count,
            "critical_slot_count": int(priority_counts.get("critical", 0)),
            "critical_slots_missing": critical_slots_missing,
            "teacher_anchor_ready_count": teacher_anchor_ready_count,
            "teacher_anchor_missing_roles": sorted(teacher_anchor_missing_roles),
            "role_counts": role_counts,
            "priority_counts": priority_counts,
            "live_regime": live_regime,
            "regime_fit_slot_count": len(regime_priority_slots),
        },
        "teacher_anchor_pool": {
            "overall": teacher_overall[:6],
            "by_role": {role: ids[:3] for role, ids in teacher_by_role.items()},
        },
        "roster_slots": slots,
        "current_regime_priority_slots": regime_priority_slots[:5],
        "infra_bots": [
            "roster_expansion_slots",
            "roster_resilience_planner",
            "teacher_quality_guard",
            "bot_quality_autopilot",
            "walk_forward_coverage_seed",
        ],
        "recommended_actions": recommended_actions,
        "registry_sync": {
            "applied": False,
            "added_slots": 0,
            "backup_path": "",
        },
    }


def apply_registry(project_root: Path = PROJECT_ROOT, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_json(registry_path)
    rows = _registry_rows(registry)
    row_map = _registry_row_map(rows)
    missing_slots = [spec for spec in DEFAULT_SLOT_SPECS if str(spec.get("bot_id") or "").strip().lower() not in row_map]
    backup_path = ""
    if missing_slots:
        lifecycle_dir = project_root / "governance" / "lifecycle"
        lifecycle_dir.mkdir(parents=True, exist_ok=True)
        stamp = iso_now().replace(":", "").replace("+00:00", "Z")
        backup = lifecycle_dir / f"master_bot_registry.roster_expansion_backup_{stamp}.json"
        if registry_path.exists():
            backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path = str(backup)
        rows.extend(_slot_registry_row(spec) for spec in missing_slots)
        registry["sub_bots"] = rows
        _refresh_registry_summary(registry)
        registry_path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "applied": bool(missing_slots),
        "added_slots": len(missing_slots),
        "added_bot_ids": [str(spec.get("bot_id") or "") for spec in missing_slots],
        "backup_path": backup_path,
        "registry_path": str(registry_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track and register the planned roster expansion slots for teacher coverage, fallback champions, and infrastructure guards.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply-registry", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out_file).expanduser()
    registry_path = Path(args.registry).expanduser()

    apply_result = {"applied": False, "added_slots": 0, "backup_path": "", "added_bot_ids": [], "registry_path": str(registry_path)}
    if args.apply_registry:
        apply_result = apply_registry(project_root, registry_path=registry_path)

    payload = build_payload(project_root)
    payload["registry_sync"] = apply_result
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "roster_expansion_slots "
            f"overall_status={payload.get('overall_status', '')} "
            f"registered_slots={int(((payload.get('summary') or {}).get('registered_slot_count', 0) or 0))} "
            f"missing_slots={int(((payload.get('summary') or {}).get('missing_slot_count', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
