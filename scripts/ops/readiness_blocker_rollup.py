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
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT = Path("governance/health/readiness_blocker_rollup_latest.json")
SCHEMA_VERSION = 1

ROOT_DEFINITIONS: dict[str, dict[str, str]] = {
    "candidate_source_drift": {
        "title": "Unaccepted frozen-candidate source drift",
        "fix_class": "engineering_review",
        "action": "review the complete change set and accept one candidate generation with an audited reason",
    },
    "readiness_artifact_freshness": {
        "title": "Readiness evidence freshness",
        "fix_class": "engineering",
        "action": "refresh the bounded readiness evidence lane and repair any producer that cannot publish atomically",
    },
    "candidate_soak_time": {
        "title": "Frozen-candidate soak duration",
        "fix_class": "elapsed_time",
        "action": "keep the unchanged candidate collecting until the seven-day checkpoint and 30-day window elapse",
    },
    "independent_fill_evidence": {
        "title": "Independent paper/replay fill evidence",
        "fix_class": "evidence",
        "action": "acquire provenance-verified broker-paper or market-replay fills after the candidate cutoff",
    },
    "promotion_candidate_coverage": {
        "title": "Promotion candidate walk-forward coverage",
        "fix_class": "evidence",
        "action": "advance runtime-approved staged bots and route sample-starved bots back to labeled collection",
    },
    "raw_profitability_evidence": {
        "title": "Candidate-bound raw profitability evidence",
        "fix_class": "outcome_evidence",
        "action": "continue selective paper trading until post-cost breadth, effective samples, confidence bounds, and raw PnL clear honestly",
    },
    "canary_cohort_evidence": {
        "title": "Candidate-bound canary cohort evidence",
        "fix_class": "evidence",
        "action": "continue schema-v2 canary and baseline collection until both cohorts and the edge confidence bound clear",
    },
    "institutional_operations": {
        "title": "Institutional operations contract",
        "fix_class": "engineering_or_evidence",
        "action": "inspect the production-readiness failed checks that are not downstream of another known root",
    },
    "unclassified_readiness_blocker": {
        "title": "Unclassified readiness blocker",
        "fix_class": "engineering",
        "action": "classify this blocker before automated remediation or live-money consideration",
    },
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _root_for_pillar(pillar_id: str, check_id: str, *, raw_profitability_active: bool) -> str:
    if pillar_id == "p01_frozen_candidate":
        return "candidate_source_drift"
    if pillar_id == "p02_clean_30_day_soak":
        if check_id in {"seven_day_checkpoint", "thirty_day_window"}:
            return "candidate_soak_time"
        if check_id == "soak_artifact_fresh":
            return "readiness_artifact_freshness"
        if check_id == "soak_runtime_ready":
            return "readiness_artifact_freshness"
        if check_id == "soak_candidate_frozen":
            return "candidate_source_drift"
    if pillar_id == "p05_independent_fill_evidence":
        return "independent_fill_evidence"
    if pillar_id == "p06_real_promotion_candidates":
        return "promotion_candidate_coverage"
    if pillar_id == "p07_profitability_evidence":
        return "raw_profitability_evidence"
    if pillar_id == "p08_controlled_canary_graduation":
        return "canary_cohort_evidence"
    if pillar_id == "p10_institutional_operations":
        return "raw_profitability_evidence" if raw_profitability_active else "institutional_operations"
    return "unclassified_readiness_blocker"


def _root_for_readiness_section(section_id: str) -> str:
    if section_id == "paper_profitability_control":
        return "raw_profitability_evidence"
    if section_id == "continuous_soak":
        return "candidate_soak_time"
    return "unclassified_readiness_blocker"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    excellence = load_json(health / "production_excellence_control_latest.json")
    live_money = load_json(health / "live_money_readiness_contract_latest.json")
    slo = load_json(health / "production_quality_slo_guard_latest.json")
    accrual = load_json(health / "readiness_evidence_accrual_latest.json")
    profitability = load_json(health / "paper_profitability_control_latest.json")
    roots: dict[str, dict[str, Any]] = {}

    def add(root_id: str, *, surface: str, symptom: str, evidence: Any = None, priority: str = "high") -> None:
        definition = ROOT_DEFINITIONS.get(root_id, ROOT_DEFINITIONS["unclassified_readiness_blocker"])
        row = roots.setdefault(
            root_id,
            {
                "root_id": root_id,
                "title": definition["title"],
                "fix_class": definition["fix_class"],
                "recommended_action": definition["action"],
                "priority": priority,
                "downstream_surfaces": [],
                "symptoms": [],
                "evidence": [],
            },
        )
        if priority == "critical":
            row["priority"] = "critical"
        row["downstream_surfaces"].append(surface)
        row["symptoms"].append(symptom)
        if evidence not in (None, "", [], {}):
            row["evidence"].append(evidence)

    slo_lanes = [row for row in _as_list(slo.get("breached_lanes")) + _as_list(slo.get("warning_lanes")) if isinstance(row, dict)]
    raw_profitability_active = any(str(row.get("lane_id") or "") == "raw_profitability_recovery" for row in slo_lanes)
    for lane in slo_lanes:
        lane_id = str(lane.get("lane_id") or "")
        root_id = "raw_profitability_evidence" if lane_id == "raw_profitability_recovery" else "unclassified_readiness_blocker"
        add(
            root_id,
            surface=f"production_quality_slo:{lane_id or 'unknown'}",
            symptom=str(lane.get("status") or "active"),
            evidence={"blocking_reasons": lane.get("blocking_reasons") or [], "active_minutes": lane.get("active_minutes")},
            priority="critical" if str(lane.get("status") or "") == "breach" else "high",
        )

    for pillar in _as_list(excellence.get("pillars")):
        if not isinstance(pillar, dict):
            continue
        pillar_id = str(pillar.get("pillar_id") or "")
        for check_id in [str(item) for item in _as_list(pillar.get("failed_checks")) if str(item).strip()]:
            root_id = _root_for_pillar(pillar_id, check_id, raw_profitability_active=raw_profitability_active)
            add(
                root_id,
                surface=f"production_excellence:{pillar_id}",
                symptom=check_id,
                evidence={"pillar_grade": pillar.get("grade"), "pillar_score": pillar.get("score")},
                priority="critical" if pillar_id == "p01_frozen_candidate" else "high",
            )

    grade_summary = _as_dict(live_money.get("grade_summary"))
    for section_id in [str(item) for item in _as_list(grade_summary.get("below_floor_sections")) if str(item).strip()]:
        root_id = _root_for_readiness_section(section_id)
        add(root_id, surface=f"live_money_readiness:{section_id}", symptom="section_below_A", evidence=section_id)
    runway = _as_dict(live_money.get("transition_runway"))
    for pillar in _as_list(runway.get("pillars")):
        if not isinstance(pillar, dict) or bool(pillar.get("ready", False)):
            continue
        blocked_sections = [str(item) for item in _as_list(pillar.get("blocked_sections"))]
        pillar_id = str(pillar.get("pillar_id") or "unknown")
        root_id = (
            "raw_profitability_evidence"
            if "paper_profitability_control" in blocked_sections
            else "candidate_soak_time"
            if pillar_id == "continuous_soak" or "continuous_soak" in blocked_sections
            else "unclassified_readiness_blocker"
        )
        add(
            root_id,
            surface=f"transition_runway:{pillar_id}",
            symptom=str(pillar.get("runway_status") or "blocked"),
            evidence=pillar.get("blockers") or [],
        )

    stalled_mapping = {
        "independent_fills": "independent_fill_evidence",
        "considered_bots": "promotion_candidate_coverage",
        "promotion_candidates": "promotion_candidate_coverage",
        "post_cost_samples": "raw_profitability_evidence",
        "post_cost_days": "raw_profitability_evidence",
        "post_cost_symbols": "raw_profitability_evidence",
        "post_cost_effective_samples": "raw_profitability_evidence",
        "raw_net_pnl": "raw_profitability_evidence",
        "canary_samples": "canary_cohort_evidence",
        "baseline_samples": "canary_cohort_evidence",
        "canary_independent_days": "canary_cohort_evidence",
        "canary_effective_samples": "canary_cohort_evidence",
        "soak_elapsed_hours": "candidate_soak_time",
    }
    for metric_id in [str(item) for item in _as_list(accrual.get("stalled_metric_ids")) if str(item).strip()]:
        add(
            stalled_mapping.get(metric_id, "unclassified_readiness_blocker"),
            surface=f"evidence_accrual:{metric_id}",
            symptom="evidence_counter_stalled",
            priority="critical",
        )

    if str(profitability.get("raw_profitability_grade") or profitability.get("profitability_grade") or "").upper() not in {"A", "A+"}:
        add(
            "raw_profitability_evidence",
            surface="paper_profitability_control:raw_grade",
            symptom="raw_profitability_grade_below_A",
            evidence={
                "grade": profitability.get("raw_profitability_grade") or profitability.get("profitability_grade"),
                "net_pnl": _as_dict(_as_dict(profitability.get("a_plus_target_contract")).get("current")).get("net_pnl"),
            },
        )

    root_rows: list[dict[str, Any]] = []
    for row in roots.values():
        row["downstream_surfaces"] = ordered_unique(row["downstream_surfaces"])
        row["symptoms"] = ordered_unique(row["symptoms"])
        dedup_evidence: list[Any] = []
        seen: set[str] = set()
        for item in row["evidence"]:
            key = json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            dedup_evidence.append(item)
        row["evidence"] = dedup_evidence
        row["downstream_surface_count"] = len(row["downstream_surfaces"])
        row["symptom_count"] = len(row["symptoms"])
        root_rows.append(row)
    root_rows.sort(key=lambda row: (0 if row["priority"] == "critical" else 1, -_safe_int(row["downstream_surface_count"]), row["root_id"]))
    downstream_count = sum(_safe_int(row.get("symptom_count"), 0) for row in root_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": "ready" if not root_rows else "blocked",
        "ok": not root_rows,
        "unique_root_cause_count": len(root_rows),
        "downstream_blocker_count": downstream_count,
        "duplicate_symptom_reduction_count": max(downstream_count - len(root_rows), 0),
        "critical_root_cause_count": sum(1 for row in root_rows if row["priority"] == "critical"),
        "root_causes": root_rows,
        "causal_summary": [
            f"{row['root_id']} -> {', '.join(row['downstream_surfaces'])}"
            for row in root_rows
        ],
        "control_contract": {
            "one_root_can_explain_multiple_downstream_grades": True,
            "duplicate_blockers_not_counted_as_independent_failures": True,
            "unknown_blockers_fail_closed_until_classified": True,
            "evidence_debt_is_not_reported_as_an_engineering_defect": True,
            "live_execution_authority": False,
        },
        "recommended_actions": [row["recommended_action"] for row in root_rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll duplicate readiness symptoms up to causal root blockers.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(project_root)
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "readiness_blocker_rollup "
            f"status={payload['overall_status']} roots={payload['unique_root_cause_count']} "
            f"downstream={payload['downstream_blocker_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
