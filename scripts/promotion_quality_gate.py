import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_THRESHOLDS_FILE = PROJECT_ROOT / "governance" / "ops_thresholds.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _promotion_defaults() -> dict[str, Any]:
    payload = _load_json(OPS_THRESHOLDS_FILE)
    gates = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), dict) else {}
    return gates.get("promotion_gate") if isinstance(gates.get("promotion_gate"), dict) else {}


def _promotion_scope_active(promotion_gate: dict[str, Any], graduation_gate: dict[str, Any]) -> bool:
    considered = int(promotion_gate.get("considered_bots", 0) or 0)
    graduation_scope = int(graduation_gate.get("graduation_scope_active_count", 0) or 0)
    return bool(
        promotion_gate.get("promote_ok", False)
        or considered > 0
        or graduation_scope > 0
    )


def _graduation_effective_ok(graduation_gate: dict[str, Any], promotion_gate: dict[str, Any]) -> bool:
    if bool(graduation_gate.get("ok", False)):
        return True
    pass_examples = promotion_gate.get("pass_examples") if isinstance(promotion_gate.get("pass_examples"), list) else []
    return bool(promotion_gate.get("promote_ok", False) and pass_examples)


def _effective_min_considered(promotion_gate: dict[str, Any], configured_min: int) -> int:
    effective_thresholds = (
        promotion_gate.get("effective_thresholds")
        if isinstance(promotion_gate.get("effective_thresholds"), dict)
        else {}
    )
    raw_value = effective_thresholds.get("min_considered_bots", configured_min)
    try:
        return max(int(raw_value or configured_min), 1)
    except Exception:
        return max(int(configured_min or 1), 1)


def _promotion_candidate_ids(promotion_gate: dict[str, Any]) -> set[str]:
    ids = {
        str(raw or "").strip()
        for raw in (promotion_gate.get("considered_bot_ids") or [])
        if str(raw or "").strip()
    }
    for key in ("pass_examples", "near_pass_examples", "fail_examples"):
        rows = promotion_gate.get(key) if isinstance(promotion_gate.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get("bot_id") or "").strip()
            if bot_id:
                ids.add(bot_id)
    return ids


def _new_bot_admission_relevant_blockers(
    admission_guard: dict[str, Any],
    promotion_gate: dict[str, Any],
) -> tuple[list[str], list[str]]:
    candidate_ids = sorted(_promotion_candidate_ids(promotion_gate))
    blocking_rows = (
        admission_guard.get("blocking_candidates")
        if isinstance(admission_guard.get("blocking_candidates"), list)
        else []
    )
    blocking_ids = sorted(
        {
            str((row or {}).get("bot_id") or "").strip()
            for row in blocking_rows
            if isinstance(row, dict) and str((row or {}).get("bot_id") or "").strip()
        }
    )
    if not candidate_ids:
        return blocking_ids, candidate_ids
    candidate_set = set(candidate_ids)
    return [bot_id for bot_id in blocking_ids if bot_id in candidate_set], candidate_ids


def _resolve_daily_verify_failures(
    daily_verify: dict[str, Any],
    *,
    graduation_gate: dict[str, Any],
    owner_guard: dict[str, Any],
    admission_guard: dict[str, Any],
    feature_store_ready: bool,
    schema_compatibility_guard: dict[str, Any],
    golden_replay_guard: dict[str, Any],
    cohort_drift_guard: dict[str, Any],
    replay_hash_registry_gate: dict[str, Any],
    probation_guard: dict[str, Any],
    promotion_packet: dict[str, Any],
    promotion_gate: dict[str, Any],
    snapshot_coverage_guard: dict[str, Any],
    data_source_divergence_guard: dict[str, Any],
    artifact_freshness_guard: dict[str, Any],
    nightly_resilience_guard: dict[str, Any],
    state_snapshot_drill: dict[str, Any],
    db_integrity_guard: dict[str, Any],
    execution_queue_stress_guard: dict[str, Any],
    paper_reconciliation_slo_guard: dict[str, Any],
    paper_execution_truth_layer: dict[str, Any] | None = None,
    paper_execution_calibration: dict[str, Any] | None = None,
    resource_guard: dict[str, Any] | None = None,
    ignored_failed_checks: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    failed = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    unresolved: list[str] = []
    resolved: list[str] = []
    promotion_scope_active = _promotion_scope_active(promotion_gate, graduation_gate)
    ignored = {str(item or "").strip() for item in (ignored_failed_checks or set()) if str(item or "").strip()}
    resource_guard = resource_guard or {}
    for item in failed:
        name = str(item or "").strip()
        if name in ignored:
            resolved.append(name)
            continue
        if name == "incomplete_run_recovered":
            resolved.append(name)
            continue
        if name == "new_bot_graduation_gate" and _graduation_effective_ok(graduation_gate, promotion_gate):
            resolved.append(name)
            continue
        if name == "bot_support_owner_guard" and bool(owner_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "new_bot_admission_guard" and (
            bool(admission_guard.get("ok", False))
            or not _new_bot_admission_relevant_blockers(admission_guard, promotion_gate)[0]
        ):
            resolved.append(name)
            continue
        if name == "execution_queue_stress_bot" and bool(execution_queue_stress_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "resource_guard" and bool(
            resource_guard.get("resource_guard_ok", resource_guard.get("ok", False))
        ):
            resolved.append(name)
            continue
        if name == "feature_store_manifest" and feature_store_ready:
            resolved.append(name)
            continue
        if name == "retrain_schema_compatibility_guard" and bool(schema_compatibility_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "nightly_resilience_check" and (bool(nightly_resilience_guard.get("ok", False)) or not promotion_scope_active):
            resolved.append(name)
            continue
        if name == "state_snapshot_drill" and bool(state_snapshot_drill.get("ok", False)):
            resolved.append(name)
            continue
        if name == "db_integrity" and bool(db_integrity_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "golden_replay_regression_guard" and bool(golden_replay_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "cohort_drift_baseline_guard" and bool(cohort_drift_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "replay_hash_registry_guard" and bool(replay_hash_registry_gate.get("ok", False)):
            resolved.append(name)
            continue
        if name == "paper_reconciliation_slo_guard" and bool(paper_reconciliation_slo_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "paper_execution_truth_layer" and bool((paper_execution_truth_layer or {}).get("ok", False)):
            resolved.append(name)
            continue
        if name == "paper_execution_calibration_report" and bool((paper_execution_calibration or {}).get("ok", False)):
            resolved.append(name)
            continue
        if name == "snapshot_coverage_sentinel" and bool(snapshot_coverage_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "data_source_divergence_bot" and bool(data_source_divergence_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "artifact_freshness" and bool(artifact_freshness_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "champion_challenger_probation_guard" and bool(probation_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "promotion_packet_builder" and (
            bool(promotion_packet.get("ok", False)) or not promotion_scope_active
        ):
            resolved.append(name)
            continue
        if name == "promotion_quality_gate" and (bool(promotion_gate.get("promote_ok", False)) or not promotion_scope_active):
            resolved.append(name)
            continue
        unresolved.append(name)
    # The quality gate should not recursively block on daily_auto_verify
    # echoing the quality gate's own failure state back into this check.
    if "promotion_quality_gate" in unresolved:
        unresolved = [name for name in unresolved if name != "promotion_quality_gate"]
        if "promotion_quality_gate" not in resolved:
            resolved.append("promotion_quality_gate")
    return unresolved, resolved


def evaluate_quality(
    promotion_gate: dict[str, Any],
    daily_verify: dict[str, Any],
    graduation_gate: dict[str, Any],
    leak_overfit: dict[str, Any],
    replay_gate: dict[str, Any],
    replay_hash_registry_gate: dict[str, Any],
    reconciliation_slo: dict[str, Any] | None = None,
    feature_store_manifest: dict[str, Any] | None = None,
    bot_support_owner_guard: dict[str, Any] | None = None,
    new_bot_admission_guard: dict[str, Any] | None = None,
    retrain_schema_compatibility_guard: dict[str, Any] | None = None,
    golden_replay_regression_guard: dict[str, Any] | None = None,
    cohort_drift_baseline_guard: dict[str, Any] | None = None,
    champion_challenger_probation_guard: dict[str, Any] | None = None,
    promotion_packet: dict[str, Any] | None = None,
    snapshot_coverage_guard: dict[str, Any] | None = None,
    data_source_divergence_guard: dict[str, Any] | None = None,
    artifact_freshness_guard: dict[str, Any] | None = None,
    nightly_resilience_guard: dict[str, Any] | None = None,
    state_snapshot_drill: dict[str, Any] | None = None,
    db_integrity_guard: dict[str, Any] | None = None,
    execution_queue_stress_guard: dict[str, Any] | None = None,
    paper_reconciliation_slo_guard: dict[str, Any] | None = None,
    paper_execution_truth_layer: dict[str, Any] | None = None,
    paper_execution_calibration: dict[str, Any] | None = None,
    resource_guard: dict[str, Any] | None = None,
    *,
    max_fail_share: float,
    min_considered_bots: int,
    require_replay: bool,
    require_reconciliation_slo: bool,
    ignore_daily_verify_failed_checks: set[str] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    failed: list[str] = []
    has_reconciliation_slo = reconciliation_slo is not None
    has_feature_store_manifest = feature_store_manifest is not None
    has_bot_support_owner_guard = bot_support_owner_guard is not None
    has_new_bot_admission_guard = new_bot_admission_guard is not None
    has_schema_compatibility_guard = retrain_schema_compatibility_guard is not None
    has_golden_replay_guard = golden_replay_regression_guard is not None
    has_cohort_drift_guard = cohort_drift_baseline_guard is not None
    has_probation_guard = champion_challenger_probation_guard is not None
    has_promotion_packet = promotion_packet is not None
    reconciliation_slo = reconciliation_slo or {}
    feature_store_manifest = feature_store_manifest or {}
    bot_support_owner_guard = bot_support_owner_guard or {}
    new_bot_admission_guard = new_bot_admission_guard or {}
    retrain_schema_compatibility_guard = retrain_schema_compatibility_guard or {}
    golden_replay_regression_guard = golden_replay_regression_guard or {}
    cohort_drift_baseline_guard = cohort_drift_baseline_guard or {}
    champion_challenger_probation_guard = champion_challenger_probation_guard or {}
    promotion_packet = promotion_packet or {}
    snapshot_coverage_guard = snapshot_coverage_guard or {}
    data_source_divergence_guard = data_source_divergence_guard or {}
    artifact_freshness_guard = artifact_freshness_guard or {}
    nightly_resilience_guard = nightly_resilience_guard or {}
    state_snapshot_drill = state_snapshot_drill or {}
    db_integrity_guard = db_integrity_guard or {}
    execution_queue_stress_guard = execution_queue_stress_guard or {}
    paper_reconciliation_slo_guard = paper_reconciliation_slo_guard or {}
    has_paper_execution_truth_layer = paper_execution_truth_layer is not None
    paper_execution_truth_layer = paper_execution_truth_layer or {}
    paper_execution_calibration = paper_execution_calibration or {}
    resource_guard = resource_guard or {}

    considered = int(promotion_gate.get("considered_bots", 0) or 0)
    configured_min_considered_bots = max(int(min_considered_bots or 1), 1)
    effective_min_considered_bots = _effective_min_considered(promotion_gate, configured_min_considered_bots)
    raw_fail_share = promotion_gate.get("fail_share", 1.0)
    fail_share = float(1.0 if raw_fail_share is None else raw_fail_share)
    promote_ok = bool(promotion_gate.get("promote_ok", False))
    point_in_time_contract = (
        feature_store_manifest.get("point_in_time_contract")
        if isinstance(feature_store_manifest.get("point_in_time_contract"), dict)
        else {}
    )
    contract_hashes = feature_store_manifest.get("contract_hashes") if isinstance(feature_store_manifest.get("contract_hashes"), dict) else {}
    feature_store_ready = bool(
        feature_store_manifest.get("ok", False)
        and (
            feature_store_manifest.get("strict_ok", False)
            or feature_store_manifest.get("strict_seed_ready", False)
        )
        and (
            point_in_time_contract.get("complete", False)
            or point_in_time_contract.get("seed_ready", False)
        )
        and str(contract_hashes.get("dataset_manifest_sha256") or "").strip()
    )
    unresolved_daily_verify, resolved_daily_verify = _resolve_daily_verify_failures(
        daily_verify,
        graduation_gate=graduation_gate,
        owner_guard=bot_support_owner_guard,
        admission_guard=new_bot_admission_guard,
        feature_store_ready=feature_store_ready,
        schema_compatibility_guard=retrain_schema_compatibility_guard,
        golden_replay_guard=golden_replay_regression_guard,
        cohort_drift_guard=cohort_drift_baseline_guard,
        replay_hash_registry_gate=replay_hash_registry_gate,
        probation_guard=champion_challenger_probation_guard,
        promotion_packet=promotion_packet,
        promotion_gate=promotion_gate,
        snapshot_coverage_guard=snapshot_coverage_guard,
        data_source_divergence_guard=data_source_divergence_guard,
        artifact_freshness_guard=artifact_freshness_guard,
        nightly_resilience_guard=nightly_resilience_guard,
        state_snapshot_drill=state_snapshot_drill,
        db_integrity_guard=db_integrity_guard,
        execution_queue_stress_guard=execution_queue_stress_guard,
        paper_reconciliation_slo_guard=paper_reconciliation_slo_guard,
        paper_execution_truth_layer=paper_execution_truth_layer,
        paper_execution_calibration=paper_execution_calibration,
        resource_guard=resource_guard,
        ignored_failed_checks=ignore_daily_verify_failed_checks,
    )

    promotion_scope_active = _promotion_scope_active(promotion_gate, graduation_gate)
    graduation_effective_ok = _graduation_effective_ok(graduation_gate, promotion_gate)

    if promotion_scope_active:
        if not promote_ok:
            failed.append("promotion_gate_blocked")
        if considered < effective_min_considered_bots:
            failed.append("insufficient_considered_bots")
        if fail_share > float(max_fail_share):
            failed.append("fail_share_above_limit")

    if unresolved_daily_verify:
        failed.append("daily_verify_not_ok")

    if not graduation_effective_ok:
        failed.append("new_bot_graduation_not_ok")

    if has_bot_support_owner_guard and promotion_scope_active and not bool(bot_support_owner_guard.get("ok", False)):
        failed.append("bot_support_owner_contract_not_ok")

    admission_relevant_blocking_ids, admission_candidate_ids = _new_bot_admission_relevant_blockers(
        new_bot_admission_guard,
        promotion_gate,
    )
    if has_new_bot_admission_guard and not bool(new_bot_admission_guard.get("ok", False)) and admission_relevant_blocking_ids:
        failed.append("new_bot_admission_not_ok")

    if has_feature_store_manifest and not feature_store_ready:
        failed.append("feature_store_manifest_not_ready")

    if has_schema_compatibility_guard and not bool(retrain_schema_compatibility_guard.get("ok", False)):
        failed.append("retrain_schema_compatibility_not_ok")

    if has_golden_replay_guard and promotion_scope_active and not bool(golden_replay_regression_guard.get("ok", False)):
        failed.append("golden_replay_regression_not_ok")

    if has_cohort_drift_guard and not bool(cohort_drift_baseline_guard.get("ok", False)):
        failed.append("cohort_drift_baseline_not_ok")

    if not bool(leak_overfit.get("ok", False)):
        failed.append("leak_overfit_not_ok")

    if require_replay and promotion_scope_active and not bool(replay_gate.get("ok", False)):
        failed.append("replay_determinism_not_ok")

    if has_probation_guard and promotion_scope_active and not bool(champion_challenger_probation_guard.get("ok", False)):
        failed.append("champion_challenger_probation_not_ok")

    if require_reconciliation_slo and has_reconciliation_slo and promotion_scope_active and not bool(reconciliation_slo.get("ok", False)):
        failed.append("reconciliation_slo_not_ok")

    if promotion_scope_active and has_promotion_packet and not bool(promotion_packet.get("ok", False)):
        failed.append("promotion_packet_not_ready")

    if promotion_scope_active and has_paper_execution_truth_layer and not bool(paper_execution_truth_layer.get("ok", False)):
        failed.append("paper_execution_truth_layer_not_ok")

    details = {
        "promotion": {
            "promote_ok": promote_ok,
            "promotion_scope_active": promotion_scope_active,
            "considered_bots": considered,
            "min_considered_bots": effective_min_considered_bots,
            "configured_min_considered_bots": configured_min_considered_bots,
            "fail_share": round(fail_share, 6),
        },
        "daily_verify_ok": len(unresolved_daily_verify) == 0,
        "daily_verify_unresolved_failed_checks": unresolved_daily_verify,
        "daily_verify_resolved_failed_checks": resolved_daily_verify,
        "graduation_ok": bool(graduation_gate.get("ok", False)),
        "graduation_effective_ok": bool(graduation_effective_ok),
        "bot_support_owner_guard_ok": (bool(bot_support_owner_guard.get("ok", False)) if has_bot_support_owner_guard else None),
        "new_bot_admission_ok": (bool(new_bot_admission_guard.get("ok", False)) if has_new_bot_admission_guard else None),
        "new_bot_admission_relevant_blocking_ids": admission_relevant_blocking_ids,
        "promotion_candidate_ids": admission_candidate_ids,
        "feature_store_manifest_ready": (feature_store_ready if has_feature_store_manifest else None),
        "retrain_schema_compatibility_ok": (
            bool(retrain_schema_compatibility_guard.get("ok", False)) if has_schema_compatibility_guard else None
        ),
        "golden_replay_regression_ok": (
            bool(golden_replay_regression_guard.get("ok", False)) if has_golden_replay_guard else None
        ),
        "cohort_drift_baseline_ok": (
            bool(cohort_drift_baseline_guard.get("ok", False)) if has_cohort_drift_guard else None
        ),
        "leak_overfit_ok": bool(leak_overfit.get("ok", False)),
        "replay_ok": bool(replay_gate.get("ok", False)),
        "replay_hash_registry_ok": bool(replay_hash_registry_gate.get("ok", False)),
        "champion_challenger_probation_ok": (
            bool(champion_challenger_probation_guard.get("ok", False)) if has_probation_guard else None
        ),
        "reconciliation_slo_ok": (bool(reconciliation_slo.get("ok", False)) if has_reconciliation_slo else None),
        "promotion_packet_ok": (bool(promotion_packet.get("ok", False)) if has_promotion_packet else None),
        "snapshot_coverage_ok": bool(snapshot_coverage_guard.get("ok", False)) if snapshot_coverage_guard else None,
        "data_source_divergence_ok": bool(data_source_divergence_guard.get("ok", False)) if data_source_divergence_guard else None,
        "artifact_freshness_ok": bool(artifact_freshness_guard.get("ok", False)) if artifact_freshness_guard else None,
        "nightly_resilience_ok": bool(nightly_resilience_guard.get("ok", False)) if nightly_resilience_guard else None,
        "state_snapshot_drill_ok": bool(state_snapshot_drill.get("ok", False)) if state_snapshot_drill else None,
        "db_integrity_ok": bool(db_integrity_guard.get("ok", False)) if db_integrity_guard else None,
        "execution_queue_stress_ok": bool(execution_queue_stress_guard.get("ok", False)) if execution_queue_stress_guard else None,
        "paper_execution_truth_layer_ok": (
            bool(paper_execution_truth_layer.get("ok", False)) if has_paper_execution_truth_layer else None
        ),
        "paper_execution_truth_layer_status": (
            str(paper_execution_truth_layer.get("overall_status") or "") if has_paper_execution_truth_layer else None
        ),
        "paper_execution_truth_layer_failed_checks": (
            paper_execution_truth_layer.get("failed_checks", []) if has_paper_execution_truth_layer else None
        ),
    }
    return len(failed) == 0, failed, details


def main() -> int:
    defaults = _promotion_defaults()
    parser = argparse.ArgumentParser(description="Stricter promotion quality gate.")
    parser.add_argument("--promotion-gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--daily-verify-file", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--graduation-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--bot-support-owner-file", default=str(PROJECT_ROOT / "governance" / "health" / "bot_support_owner_guard_latest.json"))
    parser.add_argument("--new-bot-admission-file", default=str(PROJECT_ROOT / "governance" / "health" / "new_bot_admission_guard_latest.json"))
    parser.add_argument("--leak-overfit-file", default=str(PROJECT_ROOT / "governance" / "health" / "leak_overfit_guard_latest.json"))
    parser.add_argument("--replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_end_to_end_latest.json"))
    parser.add_argument("--replay-hash-registry-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"))
    parser.add_argument("--feature-store-manifest", default=str(PROJECT_ROOT / "governance" / "feature_store" / "latest.json"))
    parser.add_argument("--schema-compatibility-file", default=str(PROJECT_ROOT / "governance" / "health" / "retrain_schema_compatibility_latest.json"))
    parser.add_argument("--golden-replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "golden_replay_regression_latest.json"))
    parser.add_argument("--cohort-drift-file", default=str(PROJECT_ROOT / "governance" / "health" / "cohort_drift_baseline_latest.json"))
    parser.add_argument("--probation-guard-file", default=str(PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_latest.json"))
    parser.add_argument("--reconciliation-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json"))
    parser.add_argument("--paper-reconciliation-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_reconciliation_slo_latest.json"))
    parser.add_argument("--paper-execution-truth-layer-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_execution_truth_layer_latest.json"))
    parser.add_argument("--paper-execution-calibration-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json"))
    parser.add_argument("--promotion-packet-file", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packet_latest.json"))
    parser.add_argument("--snapshot-coverage-file", default=str(PROJECT_ROOT / "governance" / "health" / "snapshot_coverage_latest.json"))
    parser.add_argument("--data-source-divergence-file", default=str(PROJECT_ROOT / "governance" / "health" / "data_source_divergence_latest.json"))
    parser.add_argument("--artifact-freshness-file", default=str(PROJECT_ROOT / "governance" / "health" / "artifact_freshness_slo_latest.json"))
    parser.add_argument("--nightly-resilience-file", default=str(PROJECT_ROOT / "governance" / "health" / "nightly_resilience_latest.json"))
    parser.add_argument("--state-snapshot-drill-file", default=str(PROJECT_ROOT / "exports" / "state_snapshot_drills" / "latest.json"))
    parser.add_argument("--db-integrity-file", default=str(PROJECT_ROOT / "governance" / "health" / "sqlite_maintenance_latest.json"))
    parser.add_argument("--execution-queue-stress-file", default=str(PROJECT_ROOT / "governance" / "health" / "execution_queue_stress_latest.json"))
    parser.add_argument("--resource-guard-file", default=str(PROJECT_ROOT / "governance" / "health" / "resource_guard_latest.json"))
    parser.add_argument("--max-fail-share", type=float, default=float(defaults.get("max_fail_share", 0.25)))
    parser.add_argument("--min-considered-bots", type=int, default=int(defaults.get("min_considered_bots", 4)))
    parser.add_argument("--require-replay", action="store_true", default=True)
    parser.add_argument("--no-require-replay", dest="require_replay", action="store_false")
    parser.add_argument("--require-reconciliation-slo", action="store_true", default=True)
    parser.add_argument("--no-require-reconciliation-slo", dest="require_reconciliation_slo", action="store_false")
    parser.add_argument("--ignore-daily-verify-check", action="append", default=[])
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    promotion = _load_json(Path(args.promotion_gate_file))
    daily_verify = _load_json(Path(args.daily_verify_file))
    graduation = _load_json(Path(args.graduation_file))
    owner_guard = _load_json(Path(args.bot_support_owner_file))
    new_bot_admission = _load_json(Path(args.new_bot_admission_file))
    leak_overfit = _load_json(Path(args.leak_overfit_file))
    replay = _load_json(Path(args.replay_file))
    replay_hash_registry = _load_json(Path(args.replay_hash_registry_file))
    feature_store_manifest = _load_json(Path(args.feature_store_manifest))
    schema_compatibility = _load_json(Path(args.schema_compatibility_file))
    golden_replay = _load_json(Path(args.golden_replay_file))
    cohort_drift = _load_json(Path(args.cohort_drift_file))
    probation_guard = _load_json(Path(args.probation_guard_file))
    reconciliation = _load_json(Path(args.reconciliation_file))
    paper_reconciliation = _load_json(Path(args.paper_reconciliation_file))
    paper_execution_truth_layer = _load_json(Path(args.paper_execution_truth_layer_file))
    paper_execution_calibration = _load_json(Path(args.paper_execution_calibration_file))
    promotion_packet = _load_json(Path(args.promotion_packet_file))
    snapshot_coverage = _load_json(Path(args.snapshot_coverage_file))
    data_source_divergence = _load_json(Path(args.data_source_divergence_file))
    artifact_freshness = _load_json(Path(args.artifact_freshness_file))
    nightly_resilience = _load_json(Path(args.nightly_resilience_file))
    state_snapshot_drill = _load_json(Path(args.state_snapshot_drill_file))
    db_integrity = _load_json(Path(args.db_integrity_file))
    execution_queue_stress = _load_json(Path(args.execution_queue_stress_file))
    resource_guard = _load_json(Path(args.resource_guard_file))

    ok, failed_checks, details = evaluate_quality(
        promotion,
        daily_verify,
        graduation,
        leak_overfit,
        replay,
        replay_hash_registry,
        reconciliation,
        feature_store_manifest=feature_store_manifest,
        bot_support_owner_guard=owner_guard,
        new_bot_admission_guard=new_bot_admission,
        retrain_schema_compatibility_guard=schema_compatibility,
        golden_replay_regression_guard=golden_replay,
        cohort_drift_baseline_guard=cohort_drift,
        champion_challenger_probation_guard=probation_guard,
        promotion_packet=promotion_packet,
        snapshot_coverage_guard=snapshot_coverage,
        data_source_divergence_guard=data_source_divergence,
        artifact_freshness_guard=artifact_freshness,
        nightly_resilience_guard=nightly_resilience,
        state_snapshot_drill=state_snapshot_drill,
        db_integrity_guard=db_integrity,
        execution_queue_stress_guard=execution_queue_stress,
        paper_reconciliation_slo_guard=paper_reconciliation,
        paper_execution_truth_layer=paper_execution_truth_layer,
        paper_execution_calibration=paper_execution_calibration,
        resource_guard=resource_guard,
        max_fail_share=float(args.max_fail_share),
        min_considered_bots=int(args.min_considered_bots),
        require_replay=bool(args.require_replay),
        require_reconciliation_slo=bool(args.require_reconciliation_slo),
        ignore_daily_verify_failed_checks={str(item or "").strip() for item in list(args.ignore_daily_verify_check or []) if str(item or "").strip()},
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "failed_checks": failed_checks,
        "thresholds": {
            "max_fail_share": float(args.max_fail_share),
            "min_considered_bots": int(details.get("promotion", {}).get("min_considered_bots", args.min_considered_bots)),
            "configured_min_considered_bots": int(args.min_considered_bots),
            "require_replay": bool(args.require_replay),
            "require_reconciliation_slo": bool(args.require_reconciliation_slo),
        },
        "details": details,
        "source_files": {
            "promotion_gate": str(args.promotion_gate_file),
            "daily_verify": str(args.daily_verify_file),
            "graduation": str(args.graduation_file),
            "bot_support_owner_guard": str(args.bot_support_owner_file),
            "new_bot_admission": str(args.new_bot_admission_file),
            "leak_overfit": str(args.leak_overfit_file),
            "replay": str(args.replay_file),
            "replay_hash_registry": str(args.replay_hash_registry_file),
            "feature_store_manifest": str(args.feature_store_manifest),
            "retrain_schema_compatibility": str(args.schema_compatibility_file),
            "golden_replay_regression": str(args.golden_replay_file),
            "cohort_drift_baseline": str(args.cohort_drift_file),
            "probation_guard": str(args.probation_guard_file),
            "reconciliation": str(args.reconciliation_file),
            "paper_reconciliation": str(args.paper_reconciliation_file),
            "paper_execution_truth_layer": str(args.paper_execution_truth_layer_file),
            "paper_execution_calibration": str(args.paper_execution_calibration_file),
            "promotion_packet": str(args.promotion_packet_file),
            "snapshot_coverage": str(args.snapshot_coverage_file),
            "data_source_divergence": str(args.data_source_divergence_file),
            "artifact_freshness": str(args.artifact_freshness_file),
            "nightly_resilience": str(args.nightly_resilience_file),
            "state_snapshot_drill": str(args.state_snapshot_drill_file),
            "db_integrity": str(args.db_integrity_file),
            "execution_queue_stress": str(args.execution_queue_stress_file),
            "resource_guard": str(args.resource_guard_file),
        },
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        checks = ",".join(failed_checks) if failed_checks else "none"
        print(f"promotion_quality_gate_ok={int(ok)} failed_checks={checks}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
