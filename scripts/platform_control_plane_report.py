#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.execution_simulator import ExecutionSimResult
except Exception:
    ExecutionSimResult = None


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_latest_jsonl_row(path: Path) -> dict[str, Any]:
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


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path.resolve())


def _artifact_entry(project_root: Path, path: Path, *, present: bool | None = None, details: str = "") -> dict[str, Any]:
    return {
        "path": _rel(project_root, path),
        "present": bool(path.exists()) if present is None else bool(present),
        "details": str(details or ""),
    }


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _iter_recent_paper_rows(project_root: Path, *, max_rows: int) -> list[dict[str, Any]]:
    files = [Path(p) for p in glob.glob(str(project_root / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_*.jsonl"))]
    rows: list[dict[str, Any]] = []
    for path in sorted(files)[-2:]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except Exception:
            continue
    return rows[-max(int(max_rows), 1) :]


def _execution_simulator_capabilities() -> dict[str, Any]:
    fields = set()
    if ExecutionSimResult is not None:
        fields = set(getattr(ExecutionSimResult, "__dataclass_fields__", {}).keys())
    capabilities = {
        "queue_position": "queue_position_ratio" in fields,
        "partial_fills": "partial_fill_ratio" in fields,
        "cancels": "cancel_probability" in fields,
        "borrow": "borrow_fee_bps" in fields,
        "fees": "fee_bps" in fields,
        "venue_rules": "venue_rule_penalty_bps" in fields,
        "latency": "latency_ms" in fields,
    }
    return {
        "field_count": len(fields),
        "fields": sorted(fields),
        "capabilities": capabilities,
        "capability_count": sum(1 for value in capabilities.values() if value),
    }


def _domain(slug: str, title: str, score: float, summary: str, evidence: list[dict[str, Any]], gaps: list[str], next_actions: list[str]) -> dict[str, Any]:
    bounded_score = max(0.0, min(round(float(score), 2), 100.0))
    if bounded_score >= 80.0:
        status = "strong"
    elif bounded_score >= 60.0:
        status = "advancing"
    elif bounded_score >= 40.0:
        status = "partial"
    else:
        status = "thin"
    if bounded_score >= 85.0:
        readiness_tier = "institutional_candidate"
    elif bounded_score >= 70.0:
        readiness_tier = "industry_leaning"
    elif bounded_score >= 50.0:
        readiness_tier = "professional"
    else:
        readiness_tier = "foundational"
    return {
        "slug": slug,
        "title": title,
        "score": bounded_score,
        "status": status,
        "readiness_tier": readiness_tier,
        "upgrade_required": bounded_score < 80.0,
        "summary": summary,
        "evidence": evidence,
        "gaps": _ordered_unique(gaps),
        "next_actions": _ordered_unique(next_actions),
    }


def build_report(project_root: Path, *, max_rows: int) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"
    risk_root = project_root / "governance" / "risk"

    training = _load(health_root / "training_success_latest.json")
    scorecard = _load(health_root / "retrain_scorecard_latest.json")
    paper = _load(health_root / "paper_performance_latest.json")
    event_store = _load(health_root / "point_in_time_event_store_latest.json")
    provider_mesh = _load(health_root / "provider_mesh_latest.json")
    service_control_plane = _load(health_root / "service_control_plane_latest.json")
    broker = _load(health_root / "broker_readiness_latest.json")
    token_guard = _load(health_root / "premarket_token_guard_latest.json")
    counterfactual = _load(health_root / "counterfactual_replay_latest.json")
    live_readiness = _load(health_root / "live_readiness_smoke_latest.json")
    backpressure = _load(health_root / "ingestion_backpressure_latest.json")
    resource_guard = _load(health_root / "resource_guard_latest.json")
    lifecycle = _load(project_root / "governance" / "lifecycle" / "model_lifecycle_latest.json")
    registry_training_audit = _load(health_root / "training_registry_audit_latest.json")
    label_audit = _load(health_root / "training_label_audit_latest.json")
    champion = _load(champion_root / "registry.json")
    readiness = _load(walk_root / "promotion_readiness_latest.json")
    audit = _load(project_root / "governance" / "audits" / "registry_mutation_latest.json")
    sql_progress = _load(health_root / "sql_link_service_progress_latest.json")
    sql_service = _load(health_root / "sql_link_service_latest.json")
    storage_maintenance = _load(health_root / "storage_maintenance_latest.json")
    feature_versions = _load(project_root / "governance" / "feature_versions" / "latest.json")
    paper_replay = _load(health_root / "paper_replay_drill_latest.json")
    replay_end_to_end = _load(health_root / "replay_end_to_end_latest.json")
    replay_hash_registry = _load(health_root / "replay_hash_registry_guard_latest.json")
    promotion_quality_gate = _load(health_root / "promotion_quality_gate_latest.json")
    training_quality = _load(health_root / "training_quality_control_latest.json")
    derived_state = _load(health_root / "derived_state_latest.json")
    portfolio_risk = _load(risk_root / "portfolio_risk_latest.json")
    execution_budget = _load(risk_root / "execution_budget_latest.json")
    risk_service_boundary = _load(risk_root / "risk_service_boundary_latest.json")
    security_audit = _load(health_root / "security_audit_latest.json")
    secret_scan = _load(health_root / "secret_scan_latest.json")
    session_ready = _load(health_root / "session_ready_latest.json")
    daily_verify = _load(health_root / "daily_auto_verify_latest.json")
    daily_verify_remediation = _load(health_root / "daily_verify_auto_remediation_bot_latest.json")
    live_reconciliation_slo = _load(health_root / "live_reconciliation_slo_latest.json")
    paper_reconciliation_slo = _load(health_root / "paper_reconciliation_slo_latest.json")
    slo_burn = _load(health_root / "slo_burn_latest.json")
    snapshot_coverage = _load(health_root / "snapshot_coverage_latest.json")
    runtime_training_snapshot = _load(health_root / "runtime_training_snapshot_latest.json")
    replay_feature_ablation = _load(health_root / "replay_feature_ablation_latest.json")
    ingestion_priority_queue = _load(health_root / "ingestion_priority_queue_latest.json")
    storage_split_brain = _load(health_root / "storage_split_brain_reconciler_latest.json")
    storage_resilience = _load(health_root / "storage_resilience_control_latest.json")
    operator_cockpit = _load(health_root / "operator_cockpit_latest.json")
    execution_lab = _load(health_root / "execution_lab_latest.json")
    calibration_control = _load(health_root / "calibration_abstention_control_latest.json")
    training_requalification = _load(health_root / "training_requalification_latest.json")
    content_store = _load(project_root / "governance" / "content_store" / "latest.json")
    portfolio_allocator_service = _load(project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json")
    coverage_seed = _load(walk_root / "coverage_seed_latest.json")
    state_snapshot_drill = _load(project_root / "exports" / "state_snapshot_drills" / "latest.json")
    sleeve_slo = _load(project_root / "governance" / "watchdog" / "sleeve_slo_latest.json")
    experiment_latest = _load_latest_jsonl_row(project_root / "governance" / "experiments" / "experiment_registry.jsonl")

    feature_store_manifest_path = project_root / "governance" / "feature_store" / "latest.json"
    feature_store_manifest = _load(feature_store_manifest_path)
    multiple_testing_guard_path = project_root / "governance" / "research" / "multiple_testing_guard_latest.json"
    multiple_testing_guard = _load(multiple_testing_guard_path)
    decay_monitor_path = project_root / "governance" / "research" / "decay_monitor_latest.json"
    decay_monitor = _load(decay_monitor_path)
    migration_manifest_path = project_root / "governance" / "migrations" / "latest.json"
    migration_manifest = _load(migration_manifest_path)
    rbac_manifest = project_root / "governance" / "security" / "rbac_roles.json"
    codeowners_candidates = [
        project_root / ".github" / "CODEOWNERS",
        project_root / "CODEOWNERS",
    ]
    ci_workflow = project_root / ".github" / "workflows" / "ci_guardrails.yml"
    runbook = project_root / "scripts" / "runbook.sh"
    rollback_bundle = project_root / "scripts" / "release_ops.sh"
    schema_violation_logs = sorted(project_root.glob("governance/events/channel_schema_violations_*.jsonl"))
    mutation_journals = sorted(project_root.glob("governance/audits/registry_mutation_journal_*.jsonl*"))
    backup_restore_events = sorted(project_root.glob("governance/watchdog/backup_restore_events.jsonl*"))

    rows = _iter_recent_paper_rows(project_root, max_rows=max_rows)
    symbol_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    tca_slippage_gap: list[float] = []
    allocation_confidence: list[float] = []
    allocation_conflict: list[float] = []
    tradeability: list[float] = []
    for row in rows:
        symbol_counts[str(row.get("symbol") or "UNKNOWN").strip().upper()] += 1
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        profile_counts[str(meta.get("source_profile") or row.get("profile") or "default").strip().lower()] += 1
        tca_slippage_gap.append(_safe_float(row.get("slippage_gap_bps"), 0.0))
        allocation_confidence.append(_safe_float(row.get("allocation_confidence_scale"), 0.0))
        allocation_conflict.append(_safe_float(row.get("allocation_conflict_norm"), 0.0))
        tradeability.append(_safe_float(row.get("tradeability_score"), 0.0))

    sleeve_latest = paper.get("sleeve_latest") if isinstance(paper.get("sleeve_latest"), list) else []
    tca_rows = [
        {
            "profile": str(row.get("profile") or ""),
            "mean_slippage_gap_bps": float(((row.get("tca_summary") or {}).get("mean_slippage_gap_bps", 0.0) or 0.0)),
            "poor_or_fair_fill_count": int(((row.get("tca_summary") or {}).get("poor_or_fair_fill_count", 0) or 0)),
        }
        for row in sleeve_latest
        if isinstance(row, dict)
    ]
    tca_rows.sort(key=lambda row: (abs(float(row.get("mean_slippage_gap_bps", 0.0) or 0.0)), row.get("profile", "")), reverse=True)

    artifact_hygiene = {
        "ok": bool(lifecycle.get("ok", False)),
        "hard_missing_active_artifacts": int(lifecycle.get("missing_active_artifacts", 0) or 0),
        "missing_log_only_artifacts": int(lifecycle.get("missing_log_only_artifacts", 0) or 0),
        "missing_active_artifacts_total": int(lifecycle.get("missing_active_artifacts_total", lifecycle.get("missing_active_artifacts", 0)) or 0),
        "stale_active_training_diagnostics": int(lifecycle.get("stale_active_training_diagnostics", 0) or 0),
        "repair_fixed_count": int(((lifecycle.get("repair") or {}).get("fixed_count", 0) or 0)),
        "registry_updated": bool(((lifecycle.get("repair") or {}).get("registry_updated", False))),
    }

    simulator_caps = _execution_simulator_capabilities()
    lineage = scorecard.get("lineage") if isinstance(scorecard.get("lineage"), dict) else {}
    feature_file_hashes = feature_versions.get("file_hashes") if isinstance(feature_versions.get("file_hashes"), dict) else {}
    experiment_replayability = experiment_latest.get("replayability") if isinstance(experiment_latest.get("replayability"), dict) else {}
    codeowners_exists = any(path.exists() for path in codeowners_candidates)

    domains: list[dict[str, Any]] = []

    lineage_score = 0.0
    if lineage:
        lineage_score += 20.0
    if feature_file_hashes:
        lineage_score += 15.0
    if str(feature_versions.get("env_hash") or "").strip():
        lineage_score += 10.0
    if bool(event_store.get("ok", False)) and int(event_store.get("event_count", 0) or 0) > 0:
        lineage_score += 15.0
    if snapshot_coverage:
        lineage_score += 10.0
    if runtime_training_snapshot:
        lineage_score += 10.0
    if bool(feature_store_manifest.get("ok", False)):
        lineage_score += 20.0
    elif feature_store_manifest:
        lineage_score += 10.0
    else:
        lineage_score = min(lineage_score, 72.0)
    lineage_gaps = []
    if not feature_store_manifest:
        lineage_gaps.append("No canonical feature-store manifest exists under governance/feature_store, so point-in-time features are still managed as artifacts rather than a true store.")
    elif not bool(feature_store_manifest.get("ok", False)):
        lineage_gaps.append("Feature-store manifest exists but is not yet green, so point-in-time lineage is still not fully sealed.")
    if not snapshot_coverage:
        lineage_gaps.append("Snapshot coverage artifact is missing, which makes per-lane point-in-time completeness harder to prove.")
    if not lineage:
        lineage_gaps.append("Retrain lineage payload is missing from retrain_scorecard_latest.json.")
    domains.append(
        _domain(
            "point_in_time_data_lineage",
            "Point-in-time data lineage and a true feature store",
            lineage_score,
            "Lineage hashes, event-store normalization, training snapshots, and a feature-store manifest now exist, but point-in-time serving and governance contracts still need tightening.",
            [
                _artifact_entry(project_root, health_root / "retrain_scorecard_latest.json", present=bool(lineage), details=f"lineage_keys={len(lineage)}"),
                _artifact_entry(project_root, project_root / "governance" / "feature_versions" / "latest.json", present=bool(feature_file_hashes), details=f"file_hashes={len(feature_file_hashes)}"),
                _artifact_entry(project_root, health_root / "point_in_time_event_store_latest.json", present=bool(event_store.get("ok", False)), details=f"event_count={int(event_store.get('event_count', 0) or 0)}"),
                _artifact_entry(project_root, feature_store_manifest_path, present=bool(feature_store_manifest), details=f"ok={bool(feature_store_manifest.get('ok', False))}"),
            ],
            lineage_gaps,
            [
                "Publish a canonical feature-store manifest with dataset partitions, join keys, and effective-time windows.",
                "Promote snapshot coverage into a required point-in-time completeness gate before retrain and promotion.",
            ],
        )
    )

    experiment_score = 0.0
    if experiment_latest:
        experiment_score += 15.0
    if str(experiment_replayability.get("bundle_hash") or "").strip():
        experiment_score += 20.0
    if str(experiment_replayability.get("dataset_hash") or "").strip():
        experiment_score += 15.0
    if str(experiment_replayability.get("model_hash") or "").strip():
        experiment_score += 15.0
    if str(experiment_replayability.get("replay_hash") or "").strip():
        experiment_score += 15.0
    if bool(replay_end_to_end.get("ok", False)):
        experiment_score += 10.0
    if bool(paper_replay.get("ok", False)):
        experiment_score += 10.0
    if bool(replay_hash_registry.get("ok", False)):
        experiment_score += 10.0
    if str(content_store.get("manifest_hash") or "").strip():
        experiment_score += 10.0
    if not bool(experiment_replayability.get("exact_replay_ready", False)):
        experiment_score = min(experiment_score, 68.0)
    if replay_hash_registry.get("ok") is False:
        experiment_score = min(experiment_score, 62.0)
    experiment_gaps = []
    if not experiment_latest:
        experiment_gaps.append("No immutable experiment-registry row is available yet.")
    if not bool(experiment_replayability.get("exact_replay_ready", False)):
        experiment_gaps.append("Latest experiment row is missing one or more of dataset/model/replay hashes, so exact replayability is not fully sealed.")
    if replay_hash_registry.get("ok") is False:
        experiment_gaps.append("Replay hash registry guard is currently failing, which means immutable replay expectations are drifting.")
    domains.append(
        _domain(
            "immutable_experiment_tracking",
            "Immutable experiment tracking with dataset/model hashes and exact replayability",
            experiment_score,
            "Experiment tracking is now hash-centric, but exact replayability still depends on the latest recorded bundle being complete and the replay hash registry staying green.",
            [
                _artifact_entry(project_root, project_root / "governance" / "experiments" / "experiment_registry.jsonl", present=bool(experiment_latest), details=f"latest_experiment_id={str(experiment_latest.get('experiment_id') or '')}"),
                _artifact_entry(project_root, health_root / "paper_replay_drill_latest.json", present=bool(paper_replay.get("ok", False)), details=f"replay_hash={str(paper_replay.get('replay_hash') or '')[:12]}"),
                _artifact_entry(project_root, health_root / "replay_end_to_end_latest.json", present=bool(replay_end_to_end.get("ok", False)), details=f"replay_hash={str(replay_end_to_end.get('replay_hash') or '')[:12]}"),
                _artifact_entry(project_root, health_root / "replay_hash_registry_guard_latest.json", present=bool(replay_hash_registry), details=f"ok={bool(replay_hash_registry.get('ok', False))}"),
            ],
            experiment_gaps,
            [
                "Record experiment rows with dataset, model, and replay hashes for every training and promotion candidate, not just session starts/stops.",
                "Keep replay hash registry healthy before promotion so immutable replay expectations stay anchored.",
            ],
        )
    )

    simulator_score = 15.0 if bool(tca_rows) else 0.0
    simulator_score += float(simulator_caps.get("capability_count", 0) or 0) * 10.0
    if bool(execution_lab.get("ok", False)):
        simulator_score += 15.0
    simulator_score = min(simulator_score, 100.0)
    if not bool(simulator_caps.get("capabilities", {}).get("queue_position", False)):
        simulator_score = min(simulator_score, 58.0)
    simulator_gaps = []
    if not bool(simulator_caps.get("capabilities", {}).get("queue_position", False)):
        simulator_gaps.append("Execution simulator still lacks queue-position telemetry.")
    if not bool(simulator_caps.get("capabilities", {}).get("venue_rules", False)):
        simulator_gaps.append("Execution simulator still lacks venue-rule penalties.")
    if not bool(tca_rows):
        simulator_gaps.append("Paper performance report is not yet publishing TCA sleeve summaries.")
    domains.append(
        _domain(
            "high_fidelity_simulator",
            "A much higher-fidelity simulator: queue position, partial fills, cancels, borrow, fees, venue rules, and latency",
            simulator_score,
            "The simulator now exposes queue position, partial fills, cancel probability, borrow drag, fee drag, venue penalties, and latency, but it still needs deeper venue-specific market microstructure to feel institutional.",
            [
                _artifact_entry(project_root, project_root / "core" / "execution_simulator.py", present=bool(simulator_caps.get("capability_count", 0)), details=f"capabilities={int(simulator_caps.get('capability_count', 0) or 0)}"),
                _artifact_entry(project_root, health_root / "paper_performance_latest.json", present=bool(tca_rows), details=f"tca_profiles={len(tca_rows)}"),
            ],
            simulator_gaps,
            [
                "Add venue-specific fill models for opening auctions, halts, and partial-cancel behavior by broker and market.",
                "Promote simulator telemetry into replay and paper-performance reports so model research sees queue/cancel/borrow drag directly.",
            ],
        )
    )

    portfolio_score = 0.0
    if derived_state:
        portfolio_score += 25.0
    if portfolio_risk:
        portfolio_score += 25.0
    if execution_budget:
        portfolio_score += 20.0
    if portfolio_allocator_service:
        portfolio_score += 15.0
    if bool(symbol_counts):
        portfolio_score += 10.0
    if bool(profile_counts):
        portfolio_score += 10.0
    if portfolio_score > 0.0:
        portfolio_score += 10.0
    portfolio_score = min(portfolio_score, 100.0)
    portfolio_score = min(portfolio_score, 74.0)
    domains.append(
        _domain(
            "portfolio_construction",
            "A real portfolio-construction layer above the signal bots: netting, factor controls, capacity, exposure budgets",
            portfolio_score,
            "Portfolio risk and execution budgets exist, but the portfolio layer is still mostly budget- and exposure-oriented rather than a full factor- and capacity-aware allocator.",
            [
                _artifact_entry(project_root, health_root / "derived_state_latest.json", present=bool(derived_state), details=f"risk_level={str(derived_state.get('risk_level') or '')}"),
                _artifact_entry(project_root, risk_root / "portfolio_risk_latest.json", present=bool(portfolio_risk), details=f"risk_level={str(portfolio_risk.get('risk_level') or '')}"),
                _artifact_entry(project_root, risk_root / "execution_budget_latest.json", present=bool(execution_budget), details=f"gross_risk_budget={portfolio_risk.get('gross_risk_budget', execution_budget.get('gross_risk_budget'))}"),
            ],
            [
                "There is no evidence of a factor-neutral or optimizer-driven portfolio construction layer yet.",
                "Capacity controls are still expressed mostly as budgets and limits rather than symbol- and venue-aware portfolio capacity curves.",
            ],
            [
                "Add a portfolio allocator that nets sleeve intents and enforces factor, sector, and regime exposure budgets before orders are emitted.",
                "Introduce per-symbol and per-venue capacity curves so portfolio construction can reason about crowding and execution impact.",
            ],
        )
    )

    risk_score = 0.0
    if portfolio_risk:
        risk_score += 20.0
    if execution_budget:
        risk_score += 15.0
    if live_reconciliation_slo:
        risk_score += 15.0
    if paper_reconciliation_slo:
        risk_score += 10.0
    if (project_root / "scripts" / "global_risk_killswitch.py").exists():
        risk_score += 15.0
    if (project_root / "scripts" / "incident_auto_halt.py").exists():
        risk_score += 15.0
    if token_guard:
        risk_score += 10.0
    if risk_service_boundary:
        risk_score += 15.0
    risk_score = min(risk_score, 100.0)
    risk_score = min(risk_score, 78.0)
    domains.append(
        _domain(
            "independent_risk_services",
            "Independent pre-trade and post-trade risk services that are not embedded in the strategy path",
            risk_score,
            "The repo has meaningful pre-trade and post-trade safety services, but they are still tightly coupled to the same codebase and do not yet look like separately deployable risk services.",
            [
                _artifact_entry(project_root, risk_root / "portfolio_risk_latest.json", present=bool(portfolio_risk), details=f"risk_score={portfolio_risk.get('risk_score')}"),
                _artifact_entry(project_root, risk_root / "execution_budget_latest.json", present=bool(execution_budget), details=f"max_total_actions_per_hour={execution_budget.get('max_total_actions_per_hour')}"),
                _artifact_entry(project_root, health_root / "live_reconciliation_slo_latest.json", present=bool(live_reconciliation_slo), details=f"ok={bool(live_reconciliation_slo.get('ok', False))}"),
                _artifact_entry(project_root, project_root / "scripts" / "global_risk_killswitch.py"),
            ],
            [
                "Pre-trade and post-trade controls still live inside the main repo rather than as separately isolated services with independent deploy surfaces.",
                "Risk-service evidence is stronger on limits and reconciliation than on truly independent service boundaries.",
            ],
            [
                "Split pre-trade approval and post-trade reconciliation into independent service endpoints with their own deploy and audit surfaces.",
                "Publish risk-service contracts so strategies cannot bypass portfolio, execution-budget, or reconciliation checks.",
            ],
        )
    )

    tca_score = 0.0
    if bool(tca_rows):
        tca_score += 35.0
    if bool(rows):
        tca_score += 15.0
    if execution_budget:
        tca_score += 15.0
    if tradeability:
        tca_score += 10.0
    if allocation_conflict:
        tca_score += 10.0
    if allocation_confidence:
        tca_score += 10.0
    tca_score = min(tca_score, 100.0)
    tca_score = min(tca_score, 68.0)
    domains.append(
        _domain(
            "transaction_cost_and_capacity",
            "Better transaction-cost and capacity models by symbol, venue, and time of day",
            tca_score,
            "TCA and tradeability signals exist in paper telemetry, but capacity is still inferred indirectly and there is not yet a first-class symbol-by-venue-by-time cost model.",
            [
                _artifact_entry(project_root, health_root / "paper_performance_latest.json", present=bool(tca_rows), details=f"tca_profiles={len(tca_rows)}"),
                _artifact_entry(project_root, risk_root / "execution_budget_latest.json", present=bool(execution_budget), details=f"execution_budget_present={bool(execution_budget)}"),
            ],
            [
                "Capacity is not yet modeled as a first-class surface by symbol, venue, and time of day.",
                "TCA is stronger on realized slippage summaries than on forward-looking capacity and venue microstructure curves.",
            ],
            [
                "Add capacity curves by symbol, venue, and clock bucket so research and execution budgets share the same cost model.",
                "Promote TCA breakdowns into paper and replay artifacts by venue, order type, and volatility regime.",
            ],
        )
    )

    research_score = 0.0
    if counterfactual:
        research_score += 20.0
    if replay_feature_ablation:
        research_score += 15.0
    if readiness:
        research_score += 15.0
    if promotion_quality_gate:
        research_score += 10.0
    if training_quality:
        research_score += 10.0
    if label_audit:
        research_score += 10.0
    if registry_training_audit:
        research_score += 10.0
    if bool(multiple_testing_guard.get("ok", False)):
        research_score += 10.0
    elif multiple_testing_guard:
        research_score += 5.0
    if bool(decay_monitor.get("ok", False)):
        research_score += 10.0
    elif decay_monitor:
        research_score += 5.0
    research_score = min(research_score, 100.0)
    if not multiple_testing_guard:
        research_score = min(research_score, 65.0)
    domains.append(
        _domain(
            "statistical_research_discipline",
            "Stronger statistical research discipline: multiple-testing control, regime-segmented validation, and decay monitoring",
            research_score,
            "The research stack already supports replay, ablation, promotion gates, and counterfactual analysis, but it still needs explicit multiple-testing and decay-monitoring artifacts to look institutional.",
            [
                _artifact_entry(project_root, health_root / "counterfactual_replay_latest.json", present=bool(counterfactual), details=f"keys={len(counterfactual)}"),
                _artifact_entry(project_root, health_root / "replay_feature_ablation_latest.json", present=bool(replay_feature_ablation), details=f"keys={len(replay_feature_ablation)}"),
                _artifact_entry(project_root, walk_root / "promotion_readiness_latest.json", present=bool(readiness), details=f"promote_ok={bool(readiness.get('promote_ok', False))}"),
                _artifact_entry(project_root, multiple_testing_guard_path, present=bool(multiple_testing_guard), details=f"ok={bool(multiple_testing_guard.get('ok', False))}"),
                _artifact_entry(project_root, decay_monitor_path, present=bool(decay_monitor), details=f"status={str(decay_monitor.get('overall_status') or '')}"),
            ],
            [
                *([] if multiple_testing_guard else ["No dedicated multiple-testing control artifact was found."]),
                *([] if decay_monitor else ["No dedicated decay-monitor artifact was found."]),
                *([] if bool(multiple_testing_guard.get("ok", False)) or not multiple_testing_guard else ["Multiple-testing guard exists but is not currently green."]),
                *([] if str(decay_monitor.get("overall_status") or "") in {"ready", "needs_work"} or not decay_monitor else ["Decay monitor exists but is not publishing a readable status."]),
            ],
            [
                "Add a multiple-testing guard artifact that records hypothesis families, correction method, and false-discovery limits for each research batch.",
                "Add decay-monitor reports by lane and regime so promoted bots can be retired before forward performance silently erodes.",
            ],
        )
    )

    governance_score = 0.0
    if champion:
        governance_score += 20.0
    if promotion_quality_gate:
        governance_score += 15.0
    if readiness:
        governance_score += 15.0
    if lifecycle:
        governance_score += 15.0
    if audit:
        governance_score += 15.0
    if rollback_bundle.exists():
        governance_score += 10.0
    approval_path = champion_root / "PROMOTION_APPROVED.flag"
    if approval_path.exists():
        governance_score += 10.0
    governance_score = min(governance_score, 100.0)
    governance_score = min(governance_score, 78.0)
    domains.append(
        _domain(
            "formal_model_governance",
            "Formal model governance: approvals, champion/challenger, rollback bundles, and promotion committees in code",
            governance_score,
            "Champion/challenger, promotion gates, lifecycle hygiene, and rollback entry points are real, but governance is still mostly operator-centric rather than committee- and approval-workflow-driven.",
            [
                _artifact_entry(project_root, champion_root / "registry.json", present=bool(champion), details=f"stages={len(champion.get('stages', []) or [])}"),
                _artifact_entry(project_root, walk_root / "promotion_readiness_latest.json", present=bool(readiness), details=f"promote_ok={bool(readiness.get('promote_ok', False))}"),
                _artifact_entry(project_root, health_root / "promotion_quality_gate_latest.json", present=bool(promotion_quality_gate), details=f"ok={bool(promotion_quality_gate.get('ok', False))}"),
                _artifact_entry(project_root, rollback_bundle, details="rollback entry point"),
            ],
            [
                "Promotion governance is stronger on gates and registries than on explicit committee approvals and sign-off policy.",
                "Rollback exists, but there is not yet a fully bundled promotion packet with committee metadata and deployment attestations.",
            ],
            [
                "Add committee-style promotion packets that capture reviewers, rationale, rollback bundle hashes, and deployment approvals.",
                "Treat champion/challenger transitions as signed governance events with mandatory approval metadata.",
            ],
        )
    )

    security_score = 0.0
    if security_audit:
        security_score += 20.0
        if bool(security_audit.get("ok", False)):
            security_score += 10.0
    if bool(secret_scan):
        security_score += 15.0
        if int(secret_scan.get("findings_count", 0) or 0) == 0:
            security_score += 5.0
    if bool(mutation_journals):
        security_score += 15.0
    if (project_root / "scripts" / "shadow_preflight.py").exists():
        security_score += 15.0
    if ci_workflow.exists():
        security_score += 10.0
    if approval_path.exists():
        security_score += 5.0
    if rbac_manifest.exists():
        security_score += 20.0
    if rbac_manifest.exists() and bool(security_audit.get("ok", False)) and int(secret_scan.get("findings_count", 0) or 0) == 0:
        security_score += 5.0
    security_score = min(security_score, 100.0)
    if not rbac_manifest.exists():
        security_score = min(security_score, 58.0)
    domains.append(
        _domain(
            "security_and_compliance",
            "Security and compliance hardening: RBAC, secret rotation, tamper-evident audit logs, and strict paper/live separation",
            security_score,
            "Secrets scanning, audit journaling, RBAC, and paper/live separation are now represented as first-class controls, but the repo still needs stronger freshness and live-ops compliance discipline to look institutional.",
            [
                _artifact_entry(project_root, health_root / "security_audit_latest.json", present=bool(security_audit), details=f"ok={bool(security_audit.get('ok', False))} status={str(security_audit.get('overall_status') or '')}"),
                _artifact_entry(project_root, health_root / "secret_scan_latest.json", present=bool(secret_scan), details=f"findings={int(secret_scan.get('findings_count', 0) or 0)}"),
                _artifact_entry(project_root, project_root / "scripts" / "shadow_preflight.py", details="paper/live separation checks"),
                _artifact_entry(project_root, rbac_manifest, details="expected RBAC manifest"),
            ],
            [
                *([] if rbac_manifest.exists() else ["No RBAC manifest or role-policy registry was found."]),
                *([] if security_audit else ["No dedicated security hardening audit artifact was found."]),
                *([] if int(secret_scan.get("findings_count", 0) or 0) == 0 else ["Secret scan is not currently clean, so live hardening is still exposed to accidental credential drift."]),
            ],
            [
                "Add an RBAC manifest for live actions, promotion approvals, deletion lanes, and emergency controls.",
                "Promote secret-rotation and audit-journal freshness into a dedicated security control-plane artifact.",
            ],
        )
    )

    reliability_score = 0.0
    if daily_verify:
        reliability_score += 10.0
        if bool(daily_verify.get("ok", False)):
            reliability_score += 10.0
    if state_snapshot_drill:
        reliability_score += 10.0
        if bool(state_snapshot_drill.get("ok", False)):
            reliability_score += 10.0
    if resource_guard:
        reliability_score += 10.0
        if bool(resource_guard.get("ok", False)):
            reliability_score += 5.0
    if storage_maintenance:
        reliability_score += 10.0
        if str(storage_maintenance.get("reason") or "").strip().lower() not in {"resource_guard_failed", "resource_guard_blocked"}:
            reliability_score += 5.0
    if bool(backup_restore_events):
        reliability_score += 10.0
    if session_ready:
        reliability_score += 10.0
        if bool(session_ready.get("ok", False)):
            reliability_score += 5.0
    if live_readiness:
        reliability_score += 10.0
        if bool(live_readiness.get("ok", False)):
            reliability_score += 5.0
    if ingestion_priority_queue:
        reliability_score += 10.0
    if storage_resilience:
        reliability_score += 10.0
        if bool(storage_resilience.get("ok", False)):
            reliability_score += 5.0
    if daily_verify_remediation:
        reliability_score += 5.0
    reliability_score = min(reliability_score, 100.0)
    domains.append(
        _domain(
            "reliability_engineering",
            "Deeper reliability engineering: restore drills, chaos testing, machine-independent deployment, and queue durability",
            reliability_score,
            "Reliability is one of the stronger parts of the system thanks to watchdogs, restore drills, and maintenance lanes, but machine-independent deployment and explicit chaos drills still need work.",
            [
                _artifact_entry(project_root, health_root / "daily_auto_verify_latest.json", present=bool(daily_verify), details=f"ok={bool(daily_verify.get('ok', False))}"),
                _artifact_entry(project_root, project_root / "exports" / "state_snapshot_drills" / "latest.json", present=bool(state_snapshot_drill), details=f"ok={bool(state_snapshot_drill.get('ok', False))}"),
                _artifact_entry(project_root, health_root / "storage_maintenance_latest.json", present=bool(storage_maintenance), details=f"reason={str(storage_maintenance.get('reason') or '')}"),
                _artifact_entry(project_root, health_root / "resource_guard_latest.json", present=bool(resource_guard), details=f"profile={str(resource_guard.get('profile') or '')}"),
            ],
            [
                "No explicit chaos-drill artifact was found.",
                "Deployment still appears optimized for the current machine more than for machine-independent environment promotion.",
            ],
            [
                "Add scheduled chaos drills for writer loss, storage path failover, and stale-artifact lane failure.",
                "Publish a machine-independent deployment bundle so restore drills can be replayed off-host with the same runtime contract.",
            ],
        )
    )

    observability_score = 0.0
    if session_ready:
        observability_score += 15.0
    if live_reconciliation_slo:
        observability_score += 15.0
    if paper_reconciliation_slo:
        observability_score += 10.0
    if sleeve_slo:
        observability_score += 15.0
    if slo_burn:
        observability_score += 10.0
    if health_root.joinpath("one_numbers_latest.json").exists():
        observability_score += 10.0
    if (project_root / "scripts" / "ops" / "runtime_gate_dashboard.py").exists():
        observability_score += 10.0
    if (project_root / "scripts" / "ops" / "crash_report_digest.py").exists():
        observability_score += 10.0
    if operator_cockpit:
        observability_score += 10.0
    observability_score = min(observability_score, 100.0)
    observability_score = min(observability_score, 82.0)
    domains.append(
        _domain(
            "observability_and_slo",
            "Better observability: SLOs, paging rules, incident timelines, cost telemetry, and golden-signal dashboards",
            observability_score,
            "The system already has good operator-facing health summaries and SLO guards, but cost telemetry and a fuller incident-timeline surface still need strengthening.",
            [
                _artifact_entry(project_root, health_root / "session_ready_latest.json", present=bool(session_ready), details=f"ok={bool(session_ready.get('ok', False))}"),
                _artifact_entry(project_root, health_root / "live_reconciliation_slo_latest.json", present=bool(live_reconciliation_slo), details=f"ok={bool(live_reconciliation_slo.get('ok', False))}"),
                _artifact_entry(project_root, project_root / "governance" / "watchdog" / "sleeve_slo_latest.json", present=bool(sleeve_slo), details=f"ok={bool(sleeve_slo.get('ok', False))}"),
                _artifact_entry(project_root, project_root / "scripts" / "ops" / "runtime_gate_dashboard.py"),
            ],
            [
                "No dedicated cost-telemetry artifact was found.",
                "Observability is strong on health snapshots but still lighter on full incident timeline and cost accounting surfaces.",
            ],
            [
                "Add cost telemetry for market data, storage, training, and sidecar backend usage into the operator dashboard.",
                "Publish an incident timeline artifact that stitches alerts, watchdog actions, and operator interventions into one review surface.",
            ],
        )
    )

    developer_score = 0.0
    if ci_workflow.exists():
        developer_score += 25.0
    if runbook.exists():
        developer_score += 15.0
    if (project_root / "COMMANDS.md").exists():
        developer_score += 15.0
    if (project_root / "README.md").exists():
        developer_score += 10.0
    if bool(schema_violation_logs):
        developer_score += 10.0
    if codeowners_exists:
        developer_score += 15.0
    if migration_manifest:
        developer_score += 15.0
    if (project_root / "scripts" / "dependency_guard.py").exists():
        developer_score += 10.0
    developer_score = min(developer_score, 100.0)
    if not codeowners_exists or not migration_manifest:
        developer_score = min(developer_score, 62.0)
    domains.append(
        _domain(
            "developer_process",
            "Team-grade developer process: CI gates, schema contracts, migration discipline, code owners, and runbooks",
            developer_score,
            "CI, commands, runbooks, schema tracking, CODEOWNERS, and a migration manifest are now in the repo, but the process still needs stricter team enforcement to feel institutional.",
            [
                _artifact_entry(project_root, ci_workflow, details="CI guardrails workflow"),
                _artifact_entry(project_root, runbook, details="terminal runbook helper"),
                _artifact_entry(project_root, project_root / "COMMANDS.md", details="operator command surface"),
                _artifact_entry(project_root, codeowners_candidates[0], present=codeowners_exists, details="expected CODEOWNERS policy"),
                _artifact_entry(project_root, migration_manifest_path, present=bool(migration_manifest), details=f"ok={bool(migration_manifest.get('ok', False))}"),
            ],
            [
                *([] if codeowners_exists else ["No CODEOWNERS policy file was found."]),
                *([] if migration_manifest else ["Migration discipline is not yet surfaced as a first-class governance artifact."]),
            ],
            [
                "Add CODEOWNERS and ownership policy for strategy, risk, ops, and storage subsystems.",
                "Introduce migration manifests for schema changes, artifact version bumps, and replay compatibility changes.",
            ],
        )
    )

    domain_status_counts = Counter(domain["status"] for domain in domains)
    overall_score = round(sum(float(domain.get("score", 0.0) or 0.0) for domain in domains) / max(len(domains), 1), 2)
    if overall_score >= 82.0 and int(domain_status_counts.get("thin", 0)) == 0:
        overall_status = "industry_leaning"
    elif overall_score >= 68.0 and int(domain_status_counts.get("thin", 0)) <= 1:
        overall_status = "advancing"
    elif overall_score >= 52.0:
        overall_status = "upgrade_required"
    else:
        overall_status = "gap_heavy"

    sorted_domains = sorted(domains, key=lambda row: (float(row.get("score", 0.0) or 0.0), str(row.get("slug") or "")))
    top_priorities = _ordered_unique(
        [
            action
            for domain in sorted_domains
            if float(domain.get("score", 0.0) or 0.0) < 80.0
            for action in list(domain.get("next_actions") or [])[:2]
        ]
    )[:12]
    critical_gaps = _ordered_unique(
        [
            gap
            for domain in sorted_domains
            if float(domain.get("score", 0.0) or 0.0) < 70.0
            for gap in list(domain.get("gaps") or [])[:2]
        ]
    )[:12]

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "point_in_time_data_lineage": scorecard.get("lineage", {}),
        "model_registry_and_rollout": {
            "training_reason": str(training.get("reason") or ""),
            "promotion_status": str(training.get("promotion_status") or ""),
            "trained_ok_but_not_promotable": bool(training.get("trained_ok_but_not_promotable", False)),
            "promotion_middle_lane_active": bool(training.get("trained_ok_but_not_promotable", False)),
            "champion": champion.get("champion", {}),
            "stages": champion.get("stages", []),
            "readiness": readiness,
            "lifecycle": lifecycle,
            "training_registry_audit": registry_training_audit,
            "training_label_audit": label_audit,
            "artifact_hygiene": artifact_hygiene,
        },
        "transaction_cost_analysis": {
            "top_profiles_by_slippage_gap": tca_rows[:6],
            "mean_slippage_gap_bps_recent": round(sum(tca_slippage_gap) / max(len(tca_slippage_gap), 1), 6),
        },
        "portfolio_risk_engine": {
            "top_symbols": [{"symbol": symbol, "count": int(count)} for symbol, count in symbol_counts.most_common(8)],
            "top_profiles": [{"profile": profile, "count": int(count)} for profile, count in profile_counts.most_common(8)],
        },
        "broker_reconciliation_layer": {
            "broker_ready": bool(broker.get("ready_for_open", False)),
            "token_guard_ok": bool(token_guard.get("ok", False)),
            "broker_readiness": broker,
            "preopen_dashboard": live_readiness.get("preopen_dashboard", {}),
        },
        "research_replay_stack": {
            "counterfactual_replay": counterfactual,
            "paper_replay_drill": paper_replay,
            "replay_end_to_end": replay_end_to_end,
            "latest_experiment": experiment_latest,
            "content_store": content_store,
        },
        "point_in_time_event_store": event_store,
        "provider_mesh": provider_mesh,
        "service_control_plane": service_control_plane,
        "automated_post_trade_attribution": {
            "weak_sleeves": [
                {
                    "profile": str(row.get("profile") or ""),
                    "ending_net_pnl_total": float(row.get("ending_net_pnl_total", 0.0) or 0.0),
                    "top_loss_causes": row.get("top_loss_causes", []),
                }
                for row in sleeve_latest
                if isinstance(row, dict)
            ][:6],
        },
        "champion_challenger": champion,
        "ops_hardening": {
            "broker_readiness": broker,
            "token_guard": token_guard,
            "live_readiness_smoke": live_readiness,
            "memory_hygiene": live_readiness.get("memory_hygiene", {}),
            "resource_guard": resource_guard,
            "artifact_hygiene": artifact_hygiene,
            "security_audit": security_audit,
        },
        "storage_sql_backlog_shaping": {
            "pending_lines": int(backpressure.get("pending_lines", 0) or 0),
            "pending_lines_deferred": int(backpressure.get("pending_lines_deferred", 0) or 0),
            "pending_lines_cold": int(backpressure.get("pending_lines_cold", 0) or 0),
            "cold_lane_recommendation": str(backpressure.get("cold_lane_recommendation") or ""),
            "top_cold_pending_files": backpressure.get("top_cold_pending_files", []),
            "sql_sync": {
                "status": str(sql_progress.get("status") or ""),
                "current_step": str(sql_progress.get("current_step") or ""),
                "completed_shard_count": int(sql_progress.get("completed_shard_count", 0) or 0),
                "completed_merge_count": int(sql_progress.get("completed_merge_count", 0) or 0),
                "merged_rows_this_cycle": int(sql_progress.get("merged_rows_this_cycle", 0) or 0),
                "sqlite_wal_size_gb": float(sql_service.get("sqlite_wal_size_gb", 0.0) or 0.0),
                "storage_maintenance_reason": str(storage_maintenance.get("reason") or ""),
            },
        },
        "immutable_audit_trail": audit,
        "capital_allocation_intelligence": {
            "avg_allocation_confidence_scale": round(sum(allocation_confidence) / max(len(allocation_confidence), 1), 6),
            "avg_allocation_conflict_norm": round(sum(allocation_conflict) / max(len(allocation_conflict), 1), 6),
            "avg_tradeability_score": round(sum(tradeability) / max(len(tradeability), 1), 6),
            "portfolio_allocator_service": portfolio_allocator_service,
            "risk_service_boundary": risk_service_boundary,
        },
        "morning_control_plane": {
            "feed_and_broker_readiness": {
                "broker_ready": bool(broker.get("ready_for_open", False)),
                "token_guard_ok": bool(token_guard.get("ok", False)),
                "live_readiness_ok": bool(live_readiness.get("ok", False)),
            },
            "training_readiness": {
                "active_sample_starved": len(registry_training_audit.get("active_sample_starved", []) or []),
                "active_quality_failed": len(registry_training_audit.get("active_quality_failed", []) or []),
                "active_stale_diagnostics": len(registry_training_audit.get("active_stale_diagnostics", []) or []),
                "tier_counts": registry_training_audit.get("tier_counts", {}),
                "supportability_counts": registry_training_audit.get("supportability_counts", {}),
                "top_label_actions": label_audit.get("top_actions", []),
            },
            "sql_storage": {
                "pending_lines": int(backpressure.get("pending_lines", 0) or 0),
                "pending_lines_cold": int(backpressure.get("pending_lines_cold", 0) or 0),
                "sql_sync_status": str(sql_progress.get("status") or ""),
                "sql_sync_step": str(sql_progress.get("current_step") or ""),
                "sqlite_wal_size_gb": float(sql_service.get("sqlite_wal_size_gb", 0.0) or 0.0),
                "storage_maintenance_reason": str(storage_maintenance.get("reason") or ""),
                "ingestion_priority_queue": ingestion_priority_queue,
                "storage_resilience_control": storage_resilience,
                "storage_split_brain_reconciler": storage_split_brain,
            },
        },
        "training_upgrade_lane": {
            "training_requalification": training_requalification,
            "coverage_seed": coverage_seed,
            "calibration_abstention_control": calibration_control,
        },
        "execution_lab": execution_lab,
        "operator_cockpit": operator_cockpit,
        "daily_verify_auto_remediation_bot": daily_verify_remediation,
        "institutional_readiness": {
            "overall_score": overall_score,
            "overall_status": overall_status,
            "domain_count": len(domains),
            "status_counts": dict(domain_status_counts),
            "weakest_domains": [
                {
                    "slug": str(domain.get("slug") or ""),
                    "title": str(domain.get("title") or ""),
                    "score": float(domain.get("score", 0.0) or 0.0),
                }
                for domain in sorted_domains[:5]
            ],
            "top_priorities": top_priorities,
            "critical_gaps": critical_gaps,
        },
        "institutional_domains": domains,
        "institutional_domains_by_slug": {
            str(domain.get("slug") or ""): domain for domain in domains
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a professional control-plane summary from live artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "platform_control_plane_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_report(Path(args.project_root).resolve(), max_rows=int(args.max_rows))
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        readiness = payload.get("institutional_readiness") if isinstance(payload.get("institutional_readiness"), dict) else {}
        print(
            "platform_control_plane "
            f"trained_reason={payload.get('model_registry_and_rollout', {}).get('training_reason', '')} "
            f"event_count={int(((payload.get('point_in_time_event_store') or {}).get('event_count', 0) or 0))} "
            f"readiness_score={float(readiness.get('overall_score', 0.0) or 0.0):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
