#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_quality_control_latest.json"

IMPROVEMENT_SPECS: list[dict[str, str]] = [
    {"key": "runtime_input_coverage", "name": "Runtime Input Coverage", "category": "supportability"},
    {"key": "lane_specific_training", "name": "Lane-Specific Training", "category": "dataset_shape"},
    {"key": "label_and_abstention_calibration", "name": "Label And Abstention Calibration", "category": "labels"},
    {"key": "stale_active_diagnostics", "name": "Stale Active Diagnostics", "category": "supportability"},
    {"key": "promotion_coverage", "name": "Promotion Coverage", "category": "rollout"},
    {"key": "paper_loss_feedback", "name": "Paper Loss Feedback", "category": "feedback"},
    {"key": "ingestion_health_guard", "name": "Ingestion Health Guard", "category": "data_ops"},
    {"key": "snapshot_freshness", "name": "Runtime Snapshot Freshness", "category": "dataset_shape"},
    {"key": "lane_dominance_cap", "name": "Lane Dominance Cap", "category": "dataset_shape"},
    {"key": "symbol_concentration_cap", "name": "Symbol Concentration Cap", "category": "dataset_shape"},
    {"key": "active_supportability", "name": "Active Supportability Score", "category": "supportability"},
    {"key": "active_diagnostic_sla", "name": "Active Diagnostic SLA", "category": "supportability"},
    {"key": "targeted_retrain_shortlist", "name": "Targeted Retrain Shortlist", "category": "remediation"},
    {"key": "active_probation_isolation", "name": "Active Probation Isolation", "category": "remediation"},
    {"key": "lane_lookback_guidance", "name": "Lane Lookback Guidance", "category": "dataset_shape"},
    {"key": "research_candidate_backlog", "name": "Research Candidate Backlog", "category": "portfolio"},
    {"key": "report_and_dashboard_integration", "name": "Report And Dashboard Integration", "category": "observability"},
    {"key": "feature_store_lineage", "name": "Feature Store Lineage", "category": "data_ops"},
    {"key": "experiment_replayability", "name": "Experiment Replayability", "category": "rollout"},
    {"key": "multiple_testing_control", "name": "Multiple Testing Control", "category": "research"},
    {"key": "decay_monitoring", "name": "Decay Monitoring", "category": "research"},
    {"key": "ingestion_drain_time_guard", "name": "Ingestion Drain-Time Guard", "category": "data_ops"},
    {"key": "storage_retention_hygiene", "name": "Storage Retention Hygiene", "category": "data_ops"},
    {"key": "training_requalification_lane", "name": "Training Requalification Lane", "category": "rollout"},
    {"key": "continuous_coverage_seed", "name": "Continuous Coverage Seed", "category": "rollout"},
    {"key": "calibration_abstention_control", "name": "Calibration Abstention Control", "category": "labels"},
]

_BASE_LOOKBACK_BY_LANE = {
    "intraday_aggressive": 30,
    "aggressive": 30,
    "swing_aggressive": 45,
    "crypto": 21,
    "crypto_futures": 21,
    "fx": 30,
    "futures": 30,
    "conservative": 45,
    "dividend": 60,
    "bond": 60,
    "paper": 14,
    "other": 30,
}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_latest_jsonl_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            rows = [line.strip() for line in handle if line.strip()]
    except Exception:
        return {}
    for raw in reversed(rows):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _mode_to_lane(mode: str) -> str:
    text = str(mode or "").strip().lower()
    if "crypto_futures" in text:
        return "crypto_futures"
    if "crypto" in text:
        return "crypto"
    if "intraday" in text:
        return "intraday_aggressive"
    if "swing" in text:
        return "swing_aggressive"
    if "dividend" in text:
        return "dividend"
    if "bond" in text:
        return "bond"
    if "fx" in text:
        return "fx"
    if "futures" in text:
        return "futures"
    if "conservative" in text:
        return "conservative"
    if "aggressive" in text:
        return "aggressive"
    if "paper" in text:
        return "paper"
    return "other"


def _lane_rows(snapshot: Dict[str, Any]) -> list[dict[str, Any]]:
    coverage = snapshot.get("coverage") if isinstance(snapshot.get("coverage"), dict) else {}
    top_modes = coverage.get("top_modes") if isinstance(coverage.get("top_modes"), list) else []
    lane_totals: Dict[str, int] = {}
    for row in top_modes:
        if not isinstance(row, dict):
            continue
        lane = _mode_to_lane(str(row.get("mode") or ""))
        lane_totals[lane] = lane_totals.get(lane, 0) + _safe_int(row.get("row_count"), 0)
    rows = [{"lane": lane, "row_count": row_count} for lane, row_count in lane_totals.items() if row_count > 0]
    rows.sort(key=lambda item: (-int(item["row_count"]), str(item["lane"])))
    return rows


def _lane_lookback_guidance(lane_rows: list[dict[str, Any]], total_rows: int) -> dict[str, int]:
    guidance: dict[str, int] = {}
    total = max(int(total_rows), 1)
    for row in lane_rows:
        lane = str(row.get("lane") or "")
        share = _safe_float(row.get("row_count"), 0.0) / total
        base = int(_BASE_LOOKBACK_BY_LANE.get(lane, _BASE_LOOKBACK_BY_LANE["other"]))
        if share >= 0.30:
            recommended = max(base - 7, 14)
        elif share <= 0.05:
            recommended = min(base + 15, 90)
        else:
            recommended = base
        guidance[lane] = int(recommended)
    return guidance


def _weak_sleeves(paper_payload: Dict[str, Any]) -> list[dict[str, Any]]:
    sleeves = paper_payload.get("sleeve_latest") if isinstance(paper_payload.get("sleeve_latest"), list) else []
    rows: list[dict[str, Any]] = []
    for row in sleeves:
        if not isinstance(row, dict):
            continue
        pnl = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        win_rate = row.get("win_rate")
        win_rate_value = _safe_float(win_rate, -1.0) if win_rate is not None else None
        if pnl < 0.0 or (win_rate_value is not None and win_rate_value < 0.45):
            rows.append(
                {
                    "profile": str(row.get("profile") or "").strip().lower(),
                    "ending_net_pnl_total": round(pnl, 6),
                    "win_rate": (round(win_rate_value, 6) if win_rate_value is not None else None),
                }
            )
    rows.sort(key=lambda item: (float(item.get("ending_net_pnl_total", 0.0) or 0.0), item.get("profile", "")))
    return rows


def _build_improvement(
    spec: dict[str, str],
    *,
    status: str,
    priority: int,
    summary: str,
    recommendation: str,
    metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": spec["key"],
        "name": spec["name"],
        "category": spec["category"],
        "implemented": True,
        "status": status,
        "priority": int(priority),
        "summary": summary,
        "recommendation": recommendation,
        "metric": metric or {},
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"

    registry_audit = _load_json(health_root / "training_registry_audit_latest.json")
    label_audit = _load_json(health_root / "training_label_audit_latest.json")
    snapshot = _load_json(health_root / "runtime_training_snapshot_latest.json")
    promotion_quality = _load_json(health_root / "promotion_quality_gate_latest.json")
    promotion_readiness = _load_json(walk_root / "promotion_readiness_latest.json")
    training_report = _load_json(health_root / "training_report_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")
    paper_performance = _load_json(health_root / "paper_performance_latest.json")
    feature_store_manifest = _load_json(project_root / "governance" / "feature_store" / "latest.json")
    multiple_testing_guard = _load_json(project_root / "governance" / "research" / "multiple_testing_guard_latest.json")
    decay_monitor = _load_json(project_root / "governance" / "research" / "decay_monitor_latest.json")
    replay_hash_registry_guard = _load_json(health_root / "replay_hash_registry_guard_latest.json")
    ingestion_storage_control = _load_json(health_root / "ingestion_storage_control_latest.json")
    training_requalification = _load_json(health_root / "training_requalification_latest.json")
    coverage_seed = _load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")
    calibration_control = _load_json(health_root / "calibration_abstention_control_latest.json")
    roster_resilience = _load_json(health_root / "roster_resilience_planner_latest.json")
    experiment_latest = _load_latest_jsonl_row(project_root / "governance" / "experiments" / "experiment_registry.jsonl")

    active_bots = _safe_int(registry_audit.get("registry_active_bots"), 0)
    active_sample_starved = registry_audit.get("active_sample_starved") if isinstance(registry_audit.get("active_sample_starved"), list) else []
    active_quality_failed = registry_audit.get("active_quality_failed") if isinstance(registry_audit.get("active_quality_failed"), list) else []
    active_stale = registry_audit.get("active_stale_diagnostics") if isinstance(registry_audit.get("active_stale_diagnostics"), list) else []
    supportability_counts = registry_audit.get("supportability_counts") if isinstance(registry_audit.get("supportability_counts"), dict) else {}
    tier_counts = registry_audit.get("tier_counts") if isinstance(registry_audit.get("tier_counts"), dict) else {}
    top_actions = label_audit.get("top_actions") if isinstance(label_audit.get("top_actions"), list) else []
    recommendation_counts = label_audit.get("recommendation_counts") if isinstance(label_audit.get("recommendation_counts"), dict) else {}

    snapshot_rows = _safe_int(snapshot.get("row_count"), 0)
    snapshot_sequences = _safe_int(snapshot.get("sequence_count"), 0)
    snapshot_ts = _parse_iso_utc(snapshot.get("timestamp_utc"))
    snapshot_age_hours = max((now - snapshot_ts).total_seconds() / 3600.0, 0.0) if snapshot_ts is not None else None

    coverage = snapshot.get("coverage") if isinstance(snapshot.get("coverage"), dict) else {}
    top_modes = coverage.get("top_modes") if isinstance(coverage.get("top_modes"), list) else []
    top_symbols = coverage.get("top_symbols") if isinstance(coverage.get("top_symbols"), list) else []
    top_mode_share = (_safe_float((top_modes[0] if top_modes else {}).get("row_count"), 0.0) / max(snapshot_rows, 1)) if snapshot_rows > 0 else 0.0
    top3_symbol_share = (
        sum(_safe_int(row.get("row_count"), 0) for row in top_symbols[:3]) / max(snapshot_rows, 1)
        if snapshot_rows > 0
        else 0.0
    )
    lane_rows = _lane_rows(snapshot)
    lane_guidance = _lane_lookback_guidance(lane_rows, snapshot_rows)
    top_lane_share = (_safe_float((lane_rows[0] if lane_rows else {}).get("row_count"), 0.0) / max(snapshot_rows, 1)) if snapshot_rows > 0 else 0.0

    weak_sleeve_rows = _weak_sleeves(paper_performance)
    promotion_thresholds = promotion_readiness.get("thresholds") if isinstance(promotion_readiness.get("thresholds"), dict) else {}
    considered_bots = _safe_int(promotion_readiness.get("considered_bots"), 0)
    min_considered_bots = _safe_int(promotion_thresholds.get("min_considered_bots"), 4)
    considered_gap = max(min_considered_bots - considered_bots, 0)
    promotion_ready = bool(promotion_readiness.get("promote_ok", False))
    training_confirmed = bool(((training_report.get("summary") or {}).get("confirmed_training_success", False)))
    health_hard_gate = bool(health_gates.get("hard_gate_triggered", False))
    feature_store_ok = bool(feature_store_manifest.get("ok", False))
    feature_store_lineage_ok = bool(
        feature_store_ok
        and str(((feature_store_manifest.get("dataset_contract") or {}).get("rows_sha256") or "")).strip()
        and bool(((feature_store_manifest.get("point_in_time_contract") or {}).get("dataset_join_keys")))
    )
    experiment_replayability = experiment_latest.get("replayability") if isinstance(experiment_latest.get("replayability"), dict) else {}
    exact_replay_ready = bool(experiment_replayability.get("exact_replay_ready", False))
    replay_hash_guard_ok = bool(replay_hash_registry_guard.get("ok", False))
    multiple_testing_ok = bool(multiple_testing_guard.get("ok", False))
    decay_status = str(decay_monitor.get("overall_status") or "")
    ingestion_storage_status = str(ingestion_storage_control.get("overall_status") or "")
    backpressure_block = ingestion_storage_control.get("backpressure") if isinstance(ingestion_storage_control.get("backpressure"), dict) else {}
    storage_block = ingestion_storage_control.get("storage") if isinstance(ingestion_storage_control.get("storage"), dict) else {}
    estimated_core_drain_minutes = backpressure_block.get("estimated_core_drain_minutes")
    estimated_total_drain_minutes = backpressure_block.get("estimated_total_drain_minutes")
    retention_debt_gb = _safe_float(storage_block.get("retention_debt_gb"), _safe_float(health_gates.get("storage_pressure", {}).get("retention_debt_gb"), 0.0))

    active_supportable = max(
        _safe_int(supportability_counts.get("supportable_active"), 0)
        + _safe_int(supportability_counts.get("artifact_backed_active"), 0),
        0,
    )
    if active_supportable <= 0:
        active_supportable = max(
            active_bots
            - _safe_int(tier_counts.get("active_stale"), 0)
            - _safe_int(tier_counts.get("active_repair"), 0)
            - _safe_int(tier_counts.get("active_probation"), 0),
            0,
        )
    supportability_score = round((active_supportable / max(active_bots, 1)) * 100.0, 2) if active_bots > 0 else 0.0

    refresh_diagnostics_bot_ids = [str((row or {}).get("bot_id") or "").strip().lower() for row in active_stale if str((row or {}).get("bot_id") or "").strip()]
    repair_runtime_input_bot_ids = [str((row or {}).get("bot_id") or "").strip().lower() for row in active_sample_starved if str((row or {}).get("bot_id") or "").strip()]
    quality_probation_bot_ids = [str((row or {}).get("bot_id") or "").strip().lower() for row in active_quality_failed if str((row or {}).get("bot_id") or "").strip()]
    targeted_retrain_bot_ids = []
    for bot_id in repair_runtime_input_bot_ids + quality_probation_bot_ids:
        if bot_id and bot_id not in targeted_retrain_bot_ids:
            targeted_retrain_bot_ids.append(bot_id)
    weak_sleeve_count = _safe_int(decay_monitor.get("weak_sleeve_count"), len(weak_sleeve_rows))
    reactivation_ready_count = _safe_int(training_requalification.get("reactivation_ready_count"), 0)
    bench_depth = _safe_int(((roster_resilience.get("bench") or {}).get("bench_depth")), 0)
    roster_a_plus_ready = bool(((roster_resilience.get("a_plus_contract") or {}).get("a_plus_ready", False)))
    coverage_seed_queue = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []
    calibration_recommendations = calibration_control.get("recommendations") if isinstance(calibration_control.get("recommendations"), list) else []
    calibration_recommendations += (
        calibration_control.get("family_recommendations")
        if isinstance(calibration_control.get("family_recommendations"), list)
        else []
    )

    improvements: list[dict[str, Any]] = []
    specs = {spec["key"]: spec for spec in IMPROVEMENT_SPECS}
    improvements.append(
        _build_improvement(
            specs["runtime_input_coverage"],
            status=("blocked" if repair_runtime_input_bot_ids else "ready"),
            priority=(3 if repair_runtime_input_bot_ids else 0),
            summary=f"Active runtime input gaps: {len(repair_runtime_input_bot_ids)} bots",
            recommendation=("rebuild runtime inputs and rerun targeted retrain for active zero-sample bots" if repair_runtime_input_bot_ids else "keep active bots on the current runtime snapshot path"),
            metric={"repair_runtime_input_bot_count": len(repair_runtime_input_bot_ids)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["lane_specific_training"],
            status=("needs_work" if top_lane_share >= 0.35 else "ready"),
            priority=(2 if top_lane_share >= 0.35 else 0),
            summary=f"Top lane share={top_lane_share:.3f}",
            recommendation=("train by lane and keep dominant lanes on shorter lookbacks to avoid swamping slower sleeves" if top_lane_share >= 0.35 else "lane mix is healthy enough for current targeted retrains"),
            metric={"top_lane_share": round(top_lane_share, 6), "lane_count": len(lane_rows)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["label_and_abstention_calibration"],
            status=("needs_work" if top_actions else "ready"),
            priority=(2 if top_actions else 0),
            summary=("Top label actions: " + ", ".join(top_actions[:3])) if top_actions else "No major label-action recommendations surfaced",
            recommendation=("apply the audit’s top label and abstention actions before widening the dataset" if top_actions else "keep current label and abstention settings"),
            metric={"top_actions": top_actions[:5], "recommendation_counts": recommendation_counts},
        )
    )
    improvements.append(
        _build_improvement(
            specs["stale_active_diagnostics"],
            status=("blocked" if refresh_diagnostics_bot_ids else "ready"),
            priority=(3 if refresh_diagnostics_bot_ids else 0),
            summary=f"Active stale diagnostics: {len(refresh_diagnostics_bot_ids)} bots",
            recommendation=("refresh diagnostics or downgrade unsupported active bots" if refresh_diagnostics_bot_ids else "all active diagnostics are current enough"),
            metric={"refresh_diagnostics_bot_count": len(refresh_diagnostics_bot_ids)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["promotion_coverage"],
            status=("blocked" if considered_gap > 0 or not promotion_ready else "ready"),
            priority=(3 if considered_gap > 0 or not promotion_ready else 0),
            summary=f"considered_bots={considered_bots} required={min_considered_bots}",
            recommendation=("increase evaluated walk-forward coverage before trusting new promotions" if considered_gap > 0 or not promotion_ready else "promotion coverage is healthy"),
            metric={"considered_bots": considered_bots, "min_considered_bots": min_considered_bots, "considered_gap": considered_gap},
        )
    )
    improvements.append(
        _build_improvement(
            specs["paper_loss_feedback"],
            status=("needs_work" if weak_sleeve_rows else "ready"),
            priority=(1 if weak_sleeve_rows else 0),
            summary=f"Weak sleeve count={len(weak_sleeve_rows)}",
            recommendation=("feed weak sleeve losses back into hard-negative or threshold-tuning work" if weak_sleeve_rows else "paper sleeves are not currently surfacing major loss feedback"),
            metric={"weak_sleeves": weak_sleeve_rows[:6]},
        )
    )
    improvements.append(
        _build_improvement(
            specs["ingestion_health_guard"],
            status=("blocked" if health_hard_gate else "ready"),
            priority=(3 if health_hard_gate else 0),
            summary=f"health_gate_triggered={str(health_hard_gate).lower()}",
            recommendation=("stabilize ingestion and storage before trusting large retrain cycles" if health_hard_gate else "ingestion health is clear for training work"),
            metric={"health_gate_triggered": health_hard_gate},
        )
    )
    improvements.append(
        _build_improvement(
            specs["snapshot_freshness"],
            status=("blocked" if snapshot_rows <= 0 or snapshot_age_hours is None or snapshot_age_hours > 36.0 else "ready"),
            priority=(3 if snapshot_rows <= 0 else 1 if snapshot_age_hours is None or snapshot_age_hours > 36.0 else 0),
            summary=f"snapshot_rows={snapshot_rows} age_hours={round(snapshot_age_hours, 3) if snapshot_age_hours is not None else 'missing'}",
            recommendation=("rebuild the runtime snapshot before retraining" if snapshot_rows <= 0 or snapshot_age_hours is None or snapshot_age_hours > 36.0 else "snapshot freshness is healthy"),
            metric={"snapshot_rows": snapshot_rows, "snapshot_sequences": snapshot_sequences, "snapshot_age_hours": round(snapshot_age_hours, 3) if snapshot_age_hours is not None else None},
        )
    )
    improvements.append(
        _build_improvement(
            specs["lane_dominance_cap"],
            status=("needs_work" if top_mode_share >= 0.25 else "ready"),
            priority=(2 if top_mode_share >= 0.25 else 0),
            summary=f"Top mode share={top_mode_share:.3f}",
            recommendation=("cap the dominant mode in retrains or use lane-specific batches" if top_mode_share >= 0.25 else "mode concentration is acceptable"),
            metric={"top_mode_share": round(top_mode_share, 6), "top_mode": str((top_modes[0] if top_modes else {}).get("mode") or "")},
        )
    )
    improvements.append(
        _build_improvement(
            specs["symbol_concentration_cap"],
            status=("needs_work" if top3_symbol_share >= 0.12 else "ready"),
            priority=(1 if top3_symbol_share >= 0.12 else 0),
            summary=f"Top3 symbol share={top3_symbol_share:.3f}",
            recommendation=("downweight dominant symbols or extend lookback on underrepresented lanes" if top3_symbol_share >= 0.12 else "symbol concentration is acceptable"),
            metric={"top3_symbol_share": round(top3_symbol_share, 6)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["active_supportability"],
            status=("blocked" if supportability_score < 50.0 else "needs_work" if supportability_score < 80.0 else "ready"),
            priority=(3 if supportability_score < 50.0 else 2 if supportability_score < 80.0 else 0),
            summary=f"Active supportability score={supportability_score:.2f}",
            recommendation=("raise the share of active bots with fresh diagnostics and usable samples" if supportability_score < 80.0 else "active bot supportability is healthy"),
            metric={"active_supportability_score": supportability_score, "active_supportable_bots": active_supportable, "active_bots": active_bots},
        )
    )
    improvements.append(
        _build_improvement(
            specs["active_diagnostic_sla"],
            status=("blocked" if _safe_int(tier_counts.get("active_stale"), 0) > 0 else "ready"),
            priority=(3 if _safe_int(tier_counts.get("active_stale"), 0) > 0 else 0),
            summary=f"active_stale={_safe_int(tier_counts.get('active_stale'), 0)}",
            recommendation=("keep active diagnostics within the freshness SLA before considering retrain success meaningful" if _safe_int(tier_counts.get("active_stale"), 0) > 0 else "diagnostic freshness SLA is healthy"),
            metric={"active_stale_count": _safe_int(tier_counts.get("active_stale"), 0)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["targeted_retrain_shortlist"],
            status="ready",
            priority=(1 if targeted_retrain_bot_ids else 0),
            summary=f"Targeted retrain candidates={len(targeted_retrain_bot_ids)}",
            recommendation=("use the shortlist for targeted retrain instead of broad full-registry sweeps" if targeted_retrain_bot_ids else "no targeted retrain shortlist is currently needed"),
            metric={"targeted_retrain_bot_ids": targeted_retrain_bot_ids[:12]},
        )
    )
    improvements.append(
        _build_improvement(
            specs["active_probation_isolation"],
            status=("needs_work" if quality_probation_bot_ids else "ready"),
            priority=(2 if quality_probation_bot_ids else 0),
            summary=f"Active probation bots={len(quality_probation_bot_ids)}",
            recommendation=("keep quality-failing active bots on probation or observe-only lanes until thresholds recover" if quality_probation_bot_ids else "no active probation bots currently need isolation"),
            metric={"quality_probation_bot_ids": quality_probation_bot_ids[:12]},
        )
    )
    improvements.append(
        _build_improvement(
            specs["lane_lookback_guidance"],
            status="ready",
            priority=0,
            summary=f"Lane guidance generated for {len(lane_guidance)} lanes",
            recommendation="apply lane-specific lookback days during targeted retrains",
            metric={"lane_lookback_days": lane_guidance},
        )
    )
    improvements.append(
        _build_improvement(
            specs["research_candidate_backlog"],
            status=("needs_work" if _safe_int(tier_counts.get("research_candidate"), 0) >= 5 else "ready"),
            priority=(1 if _safe_int(tier_counts.get("research_candidate"), 0) >= 5 else 0),
            summary=f"Research candidates={_safe_int(tier_counts.get('research_candidate'), 0)}",
            recommendation=("triage or batch research candidates so they do not dilute supportability work" if _safe_int(tier_counts.get("research_candidate"), 0) >= 5 else "research backlog is manageable"),
            metric={"research_candidate_count": _safe_int(tier_counts.get("research_candidate"), 0)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["report_and_dashboard_integration"],
            status="ready",
            priority=0,
            summary="Training quality control is published to health artifacts and designed for report/dashboard integration",
            recommendation="keep using the control artifact as the single source of truth for training remediation",
            metric={"artifact_path": str(DEFAULT_OUT_PATH)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["feature_store_lineage"],
            status=("ready" if feature_store_lineage_ok else "blocked"),
            priority=(0 if feature_store_lineage_ok else 2),
            summary=f"feature_store_ok={str(feature_store_ok).lower()}",
            recommendation=("keep the feature-store manifest current before broad retrains" if feature_store_lineage_ok else "publish and refresh a canonical feature-store manifest before trusting large-scale retrains"),
            metric={
                "feature_store_ok": feature_store_ok,
                "lineage_schema_version": _safe_int(feature_store_manifest.get("lineage_schema_version"), 0),
                "lane_partition_count": len(feature_store_manifest.get("lane_partitions") or []),
            },
        )
    )
    improvements.append(
        _build_improvement(
            specs["experiment_replayability"],
            status=("ready" if exact_replay_ready and replay_hash_guard_ok else "blocked"),
            priority=(0 if exact_replay_ready and replay_hash_guard_ok else 2),
            summary=f"exact_replay_ready={str(exact_replay_ready).lower()} replay_hash_guard_ok={str(replay_hash_guard_ok).lower()}",
            recommendation=("keep dataset/model/replay bundle hashes attached to training decisions" if exact_replay_ready and replay_hash_guard_ok else "repair replay hash drift or missing experiment bundle hashes before promotion"),
            metric={
                "latest_experiment_id": str(experiment_latest.get("experiment_id") or ""),
                "exact_replay_ready": exact_replay_ready,
                "replay_hash_guard_ok": replay_hash_guard_ok,
            },
        )
    )
    improvements.append(
        _build_improvement(
            specs["multiple_testing_control"],
            status=("ready" if multiple_testing_ok else "blocked"),
            priority=(0 if multiple_testing_ok else 2),
            summary=f"multiple_testing_ok={str(multiple_testing_ok).lower()}",
            recommendation=("keep a consistent correction family across replay and promotion batches" if multiple_testing_ok else "publish a multiple-testing guard before widening replay or threshold-search experimentation"),
            metric={
                "family_size": _safe_int(multiple_testing_guard.get("family_size"), 0),
                "correction_method": str(multiple_testing_guard.get("correction_method") or ""),
                "regime_segments": multiple_testing_guard.get("regime_segments") if isinstance(multiple_testing_guard.get("regime_segments"), list) else [],
            },
        )
    )
    improvements.append(
        _build_improvement(
            specs["decay_monitoring"],
            status=("ready" if decay_status == "ready" else "needs_work" if decay_status == "needs_work" else "blocked"),
            priority=(0 if decay_status == "ready" else 1 if decay_status == "needs_work" else 2),
            summary=f"decay_status={decay_status or 'missing'} weak_sleeves={weak_sleeve_count}",
            recommendation=("keep using paper and replay decay signals as a training input" if decay_status == "ready" else "fold weak-sleeve and decay-monitor findings into targeted retrain or probation decisions"),
            metric={
                "weak_sleeve_count": weak_sleeve_count,
                "history_days_available": _safe_int(decay_monitor.get("history_days_available"), 0),
                "pnl_slope": decay_monitor.get("pnl_slope"),
            },
        )
    )
    improvements.append(
        _build_improvement(
            specs["ingestion_drain_time_guard"],
            status=(
                "blocked"
                if ingestion_storage_status == "blocked"
                or (estimated_core_drain_minutes is not None and _safe_float(estimated_core_drain_minutes) > 30.0)
                or (estimated_total_drain_minutes is not None and _safe_float(estimated_total_drain_minutes) > 180.0)
                else "ready"
            ),
            priority=(
                2
                if ingestion_storage_status == "blocked"
                or (estimated_core_drain_minutes is not None and _safe_float(estimated_core_drain_minutes) > 30.0)
                or (estimated_total_drain_minutes is not None and _safe_float(estimated_total_drain_minutes) > 180.0)
                else 0
            ),
            summary=(
                f"core_drain_minutes={round(_safe_float(estimated_core_drain_minutes), 3) if estimated_core_drain_minutes is not None else 'missing'} "
                f"total_drain_minutes={round(_safe_float(estimated_total_drain_minutes), 3) if estimated_total_drain_minutes is not None else 'missing'}"
            ),
            recommendation=("keep core and total ingestion drain time within training-safe budgets" if ingestion_storage_status == "ready" else "reduce backlog and drain time before large retrains so the dataset stops lagging reality"),
            metric={
                "severity": str(ingestion_storage_control.get("severity") or ""),
                "estimated_core_drain_minutes": estimated_core_drain_minutes,
                "estimated_total_drain_minutes": estimated_total_drain_minutes,
            },
        )
    )
    improvements.append(
        _build_improvement(
            specs["storage_retention_hygiene"],
            status=("blocked" if retention_debt_gb > 0.0 else "ready"),
            priority=(2 if retention_debt_gb > 0.0 else 0),
            summary=f"retention_debt_gb={retention_debt_gb:.3f}",
            recommendation=("keep retention debt at zero so stale explanation and attribution shards do not skew training freshness" if retention_debt_gb <= 0.0 else "run retention, compaction, and shard-splitting remediation before trusting the next retrain cycle"),
            metric={
                "retention_debt_gb": round(retention_debt_gb, 3),
                "top_actions": ingestion_storage_control.get("top_actions") if isinstance(ingestion_storage_control.get("top_actions"), list) else [],
            },
        )
    )
    improvements.append(
        _build_improvement(
            specs["training_requalification_lane"],
            status=("needs_work" if reactivation_ready_count <= 0 and active_bots < max(min_considered_bots, 1) else "ready"),
            priority=(1 if reactivation_ready_count <= 0 and active_bots < max(min_considered_bots, 1) else 0),
            summary=f"reactivation_ready_count={reactivation_ready_count}",
            recommendation=("keep the requalification lane stocked so active coverage can recover from hygiene downgrades" if reactivation_ready_count > 0 else "build requalification-ready candidates before the active roster gets too thin"),
            metric={"reactivation_ready_count": reactivation_ready_count, "candidate_count": _safe_int(training_requalification.get("candidate_count"), 0)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["continuous_coverage_seed"],
            status=("blocked" if considered_gap > 0 and not coverage_seed_queue else "ready"),
            priority=(2 if considered_gap > 0 and not coverage_seed_queue else 0),
            summary=f"coverage_seed_queue={len(coverage_seed_queue)}",
            recommendation=("keep a standing seed queue feeding walk-forward coverage" if coverage_seed_queue else "generate a walk-forward seed queue so promotion coverage is built continuously instead of intermittently"),
            metric={"coverage_seed_queue_size": len(coverage_seed_queue), "coverage_shortfall_bots": _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0)},
        )
    )
    improvements.append(
        _build_improvement(
            specs["calibration_abstention_control"],
            status=("needs_work" if calibration_recommendations else "ready"),
            priority=(1 if calibration_recommendations else 0),
            summary=f"calibration_recommendations={len(calibration_recommendations)}",
            recommendation=("apply learned abstention calibration before widening the dataset or retrain scope" if calibration_recommendations else "no active calibration or abstention remediations are pending"),
            metric={"recommendations": calibration_recommendations[:6], "overall_status": str(calibration_control.get("overall_status") or "")},
        )
    )

    improvements.sort(key=lambda row: (-int(row["priority"]), str(row["key"])))
    blocked_count = sum(1 for row in improvements if row["status"] == "blocked")
    needs_work_count = sum(1 for row in improvements if row["status"] == "needs_work")

    quality_score = 100.0
    quality_score -= min(len(repair_runtime_input_bot_ids) * 3.0, 30.0)
    quality_score -= min(len(refresh_diagnostics_bot_ids) * 2.0, 30.0)
    quality_score -= min(len(quality_probation_bot_ids) * 6.0, 18.0)
    quality_score -= min(float(considered_gap) * 6.0, 18.0)
    if not training_confirmed:
        quality_score -= 8.0
    if health_hard_gate:
        quality_score -= 8.0
    if not feature_store_lineage_ok:
        quality_score -= 10.0
    if not exact_replay_ready or not replay_hash_guard_ok:
        quality_score -= 8.0
    if not multiple_testing_ok:
        quality_score -= 6.0
    if decay_status == "needs_work":
        quality_score -= 4.0
    if decay_status not in {"ready", "needs_work"}:
        quality_score -= 6.0
    if ingestion_storage_status == "blocked":
        quality_score -= 8.0
    if retention_debt_gb > 0.0:
        quality_score -= min(retention_debt_gb * 2.0, 10.0)
    if reactivation_ready_count <= 0 and active_bots < max(min_considered_bots, 1):
        quality_score -= 4.0
    if bench_depth < 3:
        quality_score -= 4.0
    if considered_gap > 0 and not coverage_seed_queue:
        quality_score -= 6.0
    if calibration_recommendations:
        quality_score -= min(len(calibration_recommendations) * 2.0, 6.0)
    if top_lane_share >= 0.35:
        quality_score -= min((top_lane_share - 0.35) * 40.0, 8.0)
    if top3_symbol_share >= 0.12:
        quality_score -= min((top3_symbol_share - 0.12) * 40.0, 6.0)
    quality_score = max(round(quality_score, 2), 0.0)
    promotion_confidence_ready = bool(
        quality_score >= 85.0
        and promotion_ready
        and exact_replay_ready
        and replay_hash_guard_ok
        and feature_store_lineage_ok
        and multiple_testing_ok
        and str(calibration_control.get("overall_status") or "").strip().lower() == "ready"
        and roster_a_plus_ready
    )

    if blocked_count > 0:
        overall_status = "blocked"
        ok = False
    elif needs_work_count > 0:
        overall_status = "needs_attention"
        ok = False
    else:
        overall_status = "ready"
        ok = True

    top_priorities = [row["key"] for row in improvements if int(row["priority"]) >= 2][:6]
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "training_quality_score": quality_score,
        "implemented_improvement_count": len(IMPROVEMENT_SPECS),
        "improvement_status_counts": {
            "blocked": blocked_count,
            "needs_work": needs_work_count,
            "ready": len(IMPROVEMENT_SPECS) - blocked_count - needs_work_count,
        },
        "top_priorities": top_priorities,
        "supportability": {
            "active_bots": active_bots,
            "active_supportable_bots": active_supportable,
            "active_supportability_score": supportability_score,
            "supportability_counts": supportability_counts,
            "tier_counts": tier_counts,
        },
        "dataset_shape": {
            "snapshot_rows": snapshot_rows,
            "snapshot_sequences": snapshot_sequences,
            "snapshot_age_hours": round(snapshot_age_hours, 3) if snapshot_age_hours is not None else None,
            "top_mode_share": round(top_mode_share, 6),
            "top_lane_share": round(top_lane_share, 6),
            "top3_symbol_share": round(top3_symbol_share, 6),
            "lane_rows": lane_rows[:12],
            "lane_lookback_days": lane_guidance,
        },
        "targeted_actions": {
            "refresh_diagnostics_bot_ids": refresh_diagnostics_bot_ids[:20],
            "repair_runtime_input_bot_ids": repair_runtime_input_bot_ids[:20],
            "quality_probation_bot_ids": quality_probation_bot_ids[:20],
            "targeted_retrain_bot_ids": targeted_retrain_bot_ids[:20],
            "weak_sleeves": weak_sleeve_rows[:8],
            "top_label_actions": top_actions[:8],
        },
        "rollout": {
            "training_confirmed": training_confirmed,
            "promotion_ready": promotion_ready,
            "promotion_confidence_ready": promotion_confidence_ready,
            "considered_bots": considered_bots,
            "min_considered_bots": min_considered_bots,
            "considered_gap": considered_gap,
            "exact_replay_ready": exact_replay_ready,
            "replay_hash_guard_ok": replay_hash_guard_ok,
        },
        "a_plus_contract": {
            "training_quality_target": 90.0,
            "quality_score": quality_score,
            "promotion_confidence_ready": promotion_confidence_ready,
            "bench_depth": bench_depth,
            "roster_a_plus_ready": roster_a_plus_ready,
        },
        "data_ops": {
            "health_gate_triggered": health_hard_gate,
            "training_report_overall_status": str(training_report.get("overall_status") or ""),
            "promotion_quality_ok": bool(promotion_quality.get("ok", False)),
            "feature_store_status": str(feature_store_manifest.get("overall_status") or ""),
            "ingestion_storage_status": ingestion_storage_status,
            "retention_debt_gb": round(retention_debt_gb, 3),
            "estimated_core_drain_minutes": estimated_core_drain_minutes,
            "estimated_total_drain_minutes": estimated_total_drain_minutes,
        },
        "research": {
            "multiple_testing_status": str(multiple_testing_guard.get("overall_status") or ""),
            "correction_method": str(multiple_testing_guard.get("correction_method") or ""),
            "family_size": _safe_int(multiple_testing_guard.get("family_size"), 0),
            "decay_status": decay_status,
            "weak_sleeve_count": weak_sleeve_count,
        },
        "improvements": improvements,
        "source_files": {
            "training_registry_audit": str(health_root / "training_registry_audit_latest.json"),
            "training_label_audit": str(health_root / "training_label_audit_latest.json"),
            "runtime_training_snapshot": str(health_root / "runtime_training_snapshot_latest.json"),
            "promotion_readiness": str(walk_root / "promotion_readiness_latest.json"),
            "promotion_quality_gate": str(health_root / "promotion_quality_gate_latest.json"),
            "training_report": str(health_root / "training_report_latest.json"),
            "health_gates": str(health_root / "health_gates_latest.json"),
            "paper_performance": str(health_root / "paper_performance_latest.json"),
            "feature_store_manifest": str(project_root / "governance" / "feature_store" / "latest.json"),
            "multiple_testing_guard": str(project_root / "governance" / "research" / "multiple_testing_guard_latest.json"),
            "decay_monitor": str(project_root / "governance" / "research" / "decay_monitor_latest.json"),
            "replay_hash_registry_guard": str(health_root / "replay_hash_registry_guard_latest.json"),
            "ingestion_storage_control": str(health_root / "ingestion_storage_control_latest.json"),
            "experiment_registry": str(project_root / "governance" / "experiments" / "experiment_registry.jsonl"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a unified training quality control artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_quality_control "
            f"status={payload['overall_status']} "
            f"score={payload['training_quality_score']:.2f} "
            f"priorities={','.join(payload['top_priorities']) if payload['top_priorities'] else 'none'}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
