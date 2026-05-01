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


def _resolve_daily_verify_failures(
    daily_verify: dict[str, Any],
    *,
    graduation_gate: dict[str, Any],
    owner_guard: dict[str, Any],
    admission_guard: dict[str, Any],
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
    ignored_failed_checks: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    failed = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    unresolved: list[str] = []
    resolved: list[str] = []
    promotion_scope_active = _promotion_scope_active(promotion_gate, graduation_gate)
    ignored = {str(item or "").strip() for item in (ignored_failed_checks or set()) if str(item or "").strip()}
    for item in failed:
        name = str(item or "").strip()
        if name in ignored:
            resolved.append(name)
            continue
        if name == "incomplete_run_recovered":
            resolved.append(name)
            continue
        if name == "new_bot_graduation_gate" and bool(graduation_gate.get("ok", False)):
            resolved.append(name)
            continue
        if name == "bot_support_owner_guard" and bool(owner_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "new_bot_admission_guard" and bool(admission_guard.get("ok", False)):
            resolved.append(name)
            continue
        if name == "retrain_schema_compatibility_guard" and bool(schema_compatibility_guard.get("ok", False)):
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

    considered = int(promotion_gate.get("considered_bots", 0) or 0)
    raw_fail_share = promotion_gate.get("fail_share", 1.0)
    fail_share = float(1.0 if raw_fail_share is None else raw_fail_share)
    promote_ok = bool(promotion_gate.get("promote_ok", False))
    unresolved_daily_verify, resolved_daily_verify = _resolve_daily_verify_failures(
        daily_verify,
        graduation_gate=graduation_gate,
        owner_guard=bot_support_owner_guard,
        admission_guard=new_bot_admission_guard,
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
        ignored_failed_checks=ignore_daily_verify_failed_checks,
    )
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

    promotion_scope_active = _promotion_scope_active(promotion_gate, graduation_gate)

    if promotion_scope_active:
        if not promote_ok:
            failed.append("promotion_gate_blocked")
        if considered < int(min_considered_bots):
            failed.append("insufficient_considered_bots")
        if fail_share > float(max_fail_share):
            failed.append("fail_share_above_limit")

    if unresolved_daily_verify:
        failed.append("daily_verify_not_ok")

    if not bool(graduation_gate.get("ok", False)):
        failed.append("new_bot_graduation_not_ok")

    if has_bot_support_owner_guard and promotion_scope_active and not bool(bot_support_owner_guard.get("ok", False)):
        failed.append("bot_support_owner_contract_not_ok")

    if has_new_bot_admission_guard and not bool(new_bot_admission_guard.get("ok", False)):
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

    details = {
        "promotion": {
            "promote_ok": promote_ok,
            "promotion_scope_active": promotion_scope_active,
            "considered_bots": considered,
            "fail_share": round(fail_share, 6),
        },
        "daily_verify_ok": len(unresolved_daily_verify) == 0,
        "daily_verify_unresolved_failed_checks": unresolved_daily_verify,
        "daily_verify_resolved_failed_checks": resolved_daily_verify,
        "graduation_ok": bool(graduation_gate.get("ok", False)),
        "bot_support_owner_guard_ok": (bool(bot_support_owner_guard.get("ok", False)) if has_bot_support_owner_guard else None),
        "new_bot_admission_ok": (bool(new_bot_admission_guard.get("ok", False)) if has_new_bot_admission_guard else None),
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
    parser.add_argument("--promotion-packet-file", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packet_latest.json"))
    parser.add_argument("--snapshot-coverage-file", default=str(PROJECT_ROOT / "governance" / "health" / "snapshot_coverage_latest.json"))
    parser.add_argument("--data-source-divergence-file", default=str(PROJECT_ROOT / "governance" / "health" / "data_source_divergence_latest.json"))
    parser.add_argument("--artifact-freshness-file", default=str(PROJECT_ROOT / "governance" / "health" / "artifact_freshness_slo_latest.json"))
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
    promotion_packet = _load_json(Path(args.promotion_packet_file))
    snapshot_coverage = _load_json(Path(args.snapshot_coverage_file))
    data_source_divergence = _load_json(Path(args.data_source_divergence_file))
    artifact_freshness = _load_json(Path(args.artifact_freshness_file))

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
            "min_considered_bots": int(args.min_considered_bots),
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
            "promotion_packet": str(args.promotion_packet_file),
            "snapshot_coverage": str(args.snapshot_coverage_file),
            "data_source_divergence": str(args.data_source_divergence_file),
            "artifact_freshness": str(args.artifact_freshness_file),
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
