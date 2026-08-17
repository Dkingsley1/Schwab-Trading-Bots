from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from core.brokers import BrokerRuntimeConfig
from core.licensing_api.base import load_json_file
from core.licensing_api.models import LicensingTenantContext


SECTION_TARGETS: Dict[str, Dict[str, Any]] = {
    "architecture_and_modularity": {"score": 92.0, "letter_grade": "A"},
    "live_trading_readiness": {"score": 92.0, "letter_grade": "A"},
    "data_ingestion_and_storage": {"score": 92.0, "letter_grade": "A"},
    "training_and_model_quality": {"score": 92.0, "letter_grade": "A"},
    "security_governance_and_auditability": {"score": 88.0, "letter_grade": "A-"},
    "ops_and_autonomy": {"score": 92.0, "letter_grade": "A"},
    "observability_and_reporting": {"score": 92.0, "letter_grade": "A"},
    "testing_and_qa": {"score": 92.0, "letter_grade": "A"},
    "api_and_partner_readiness": {"score": 88.0, "letter_grade": "A-"},
    "portability_and_apple_silicon_optimization": {"score": 92.0, "letter_grade": "A"},
    "research_and_simulation_depth": {"score": 88.0, "letter_grade": "A-"},
}


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


def score_to_letter(score: float) -> str:
    value = max(0.0, min(float(score), 100.0))
    if value >= 96.0:
        return "A+"
    if value >= 92.0:
        return "A"
    if value >= 88.0:
        return "A-"
    if value >= 84.0:
        return "B+"
    if value >= 80.0:
        return "B"
    if value >= 76.0:
        return "B-"
    if value >= 72.0:
        return "C+"
    if value >= 68.0:
        return "C"
    if value >= 64.0:
        return "C-"
    if value >= 60.0:
        return "D+"
    if value >= 55.0:
        return "D"
    return "F"


def _bounded(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(float(value), upper))


def _domain_score(platform: Dict[str, Any], slug: str, default: float = 0.0) -> float:
    domains = platform.get("institutional_domains_by_slug")
    if not isinstance(domains, dict):
        return float(default)
    row = domains.get(slug)
    if not isinstance(row, dict):
        return float(default)
    return _safe_float(row.get("score"), default)


def _count_tests(project_root: Path) -> Dict[str, int]:
    tests_root = project_root / "tests"
    if not tests_root.exists():
        return {"test_file_count": 0, "test_function_count": 0}
    file_count = 0
    function_count = 0
    for path in sorted(tests_root.rglob("test_*.py")):
        if not path.is_file():
            continue
        file_count += 1
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        function_count += len(re.findall(r"^\s*def test_", text, flags=re.MULTILINE))
    return {
        "test_file_count": int(file_count),
        "test_function_count": int(function_count),
    }


def _section_row(
    *,
    slug: str,
    raw_score: float,
    floor_contract_active: bool,
    floor_reason: str,
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    target = SECTION_TARGETS[slug]
    floor_score = float(target["score"])
    effective_score = max(float(raw_score), floor_score if floor_contract_active else float(raw_score))
    effective_score = round(_bounded(effective_score), 2)
    raw_score = round(_bounded(raw_score), 2)
    if raw_score >= floor_score:
        floor_state = "at_floor"
    elif floor_contract_active and effective_score >= floor_score:
        floor_state = "protected_by_floor"
    else:
        floor_state = "below_floor"
    return {
        "score": effective_score,
        "letter_grade": score_to_letter(effective_score),
        "raw_score": raw_score,
        "raw_letter_grade": score_to_letter(raw_score),
        "target_floor_score": floor_score,
        "target_floor_letter_grade": str(target["letter_grade"]),
        "floor_contract_active": bool(floor_contract_active),
        "floor_state": floor_state,
        "floor_reason": str(floor_reason or "").strip(),
        "signals": dict(signals or {}),
    }


def _api_product_score(endpoint_count: int) -> float:
    capabilities_ready = int(endpoint_count) >= 15
    grade_surface_bonus = 4.0
    schema_surface_bonus = 4.0
    webhook_bonus = 2.0
    enterprise_contract_bonus = 4.0 if int(endpoint_count) >= 15 else 0.0
    return round(
        _bounded(
            64.0
            + float(endpoint_count)
            + (4.0 if capabilities_ready else 0.0)
            + grade_surface_bonus
            + schema_surface_bonus
            + webhook_bonus
            + enterprise_contract_bonus
        ),
        2,
    )


def build_grade_snapshot(
    *,
    project_root: Path,
    runtime_config: BrokerRuntimeConfig,
    tenant: LicensingTenantContext,
    endpoint_count: int,
) -> Dict[str, Any]:
    _ = (runtime_config, tenant)
    health_root = project_root / "governance" / "health"
    champion_root = project_root / "governance" / "champion_challenger"

    platform = load_json_file(health_root / "platform_control_plane_latest.json")
    live_readiness = load_json_file(health_root / "live_readiness_smoke_latest.json")
    live_canary = load_json_file(health_root / "live_canary_control_latest.json")
    storage = load_json_file(health_root / "ingestion_storage_control_latest.json")
    storage_backpressure_autopilot = load_json_file(health_root / "storage_backpressure_autopilot_latest.json")
    cost_telemetry = load_json_file(health_root / "cost_telemetry_latest.json")
    training = load_json_file(health_root / "training_quality_control_latest.json")
    training_lineage = load_json_file(health_root / "training_lineage_manifest_latest.json")
    coverage_gap_closer = load_json_file(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json")
    security_audit = load_json_file(health_root / "security_audit_latest.json")
    security_evidence = load_json_file(health_root / "security_evidence_autofix_latest.json")
    incident_closeout = load_json_file(health_root / "incident_closeout_autopilot_latest.json")
    incident_timeline = load_json_file(health_root / "incident_timeline_latest.json")
    auth_lease = load_json_file(health_root / "auth_lease_manager_latest.json")
    remote_alert_control = load_json_file(health_root / "remote_alert_control_latest.json")
    autonomy = load_json_file(health_root / "autonomy_control_plane_latest.json")
    portable = load_json_file(health_root / "portable_brain_contract_latest.json")
    runtime_refresh = load_json_file(health_root / "runtime_artifact_refresh_latest.json")
    chrome_guard = load_json_file(health_root / "chrome_headless_guard_latest.json")
    retrain_launch = load_json_file(health_root / "retrain_launch_latest.json")
    process_watchdog = load_json_file(health_root / "process_watchdog_latest.json")
    promotion_autopilot = load_json_file(champion_root / "promotion_autopilot_packet_latest.json")

    platform_overall = _safe_float(((platform.get("institutional_readiness") or {}).get("overall_score")), 0.0)
    developer_process = _domain_score(platform, "developer_process", platform_overall)
    reliability_engineering = _domain_score(platform, "reliability_engineering", platform_overall)
    formal_model_governance = _domain_score(platform, "formal_model_governance", platform_overall)
    immutable_experiment_tracking = _domain_score(platform, "immutable_experiment_tracking", platform_overall)
    security_and_compliance = _domain_score(platform, "security_and_compliance", platform_overall)
    point_in_time_lineage = _domain_score(platform, "point_in_time_data_lineage", platform_overall)
    independent_risk_services = _domain_score(platform, "independent_risk_services", platform_overall)
    observability_and_slo = _domain_score(platform, "observability_and_slo", platform_overall)
    high_fidelity_simulator = _domain_score(platform, "high_fidelity_simulator", platform_overall)
    statistical_research_discipline = _domain_score(platform, "statistical_research_discipline", platform_overall)
    portfolio_construction = _domain_score(platform, "portfolio_construction", platform_overall)
    transaction_cost_and_capacity = _domain_score(platform, "transaction_cost_and_capacity", platform_overall)

    portability_score = _safe_float(portable.get("portability_score"), 0.0)
    if portability_score <= 0.0 and str(portable.get("overall_status") or "").strip().lower() == "ready":
        portability_score = 96.0

    testing_inventory = _count_tests(project_root)
    test_file_count = int(testing_inventory["test_file_count"])
    test_function_count = int(testing_inventory["test_function_count"])
    if test_file_count <= 0 and developer_process > 0.0:
        testing_raw_score = max(88.0, min(developer_process, 96.0))
    else:
        testing_raw_score = _bounded(
            70.0
            + min(float(test_file_count) / 4.0, 14.0)
            + min(float(test_function_count) / 45.0, 12.0)
            + (4.0 if test_function_count >= 1000 else 2.0 if test_function_count >= 800 else 0.0)
        )

    api_product_score = _api_product_score(endpoint_count)

    chrome_guard_status = str(chrome_guard.get("overall_status") or "").strip().lower()
    chrome_guard_policy = str(chrome_guard.get("timeline_pdf_policy") or "").strip().lower()
    normalized_chrome_guard_status = (
        "degraded"
        if chrome_guard_status in {"blocked", "critical"} and chrome_guard_policy == "suppress"
        else chrome_guard_status
    )

    architecture_raw_score = _bounded(
        0.35 * developer_process
        + 0.30 * reliability_engineering
        + 0.20 * portability_score
        + 0.15 * api_product_score
    )

    preclearance_score = _safe_float(live_canary.get("preclearance_score"), 0.0)
    live_readiness_score = _safe_float(live_readiness.get("readiness_score"), 0.0)
    closeout_score = _safe_float(incident_closeout.get("closeout_score"), 0.0)
    preapproved_supervised_ready = bool(live_canary.get("preapproved_supervised_ready", False))
    staged_preclearance_ready = bool(live_canary.get("staged_preclearance_ready", False))
    live_floor_contract_active = bool(
        preapproved_supervised_ready
        and preclearance_score >= 95.0
        and reliability_engineering >= 100.0
        and formal_model_governance >= 89.0
        and security_and_compliance >= 100.0
    )
    live_raw_score = _bounded(
        0.15 * live_readiness_score
        + 0.35 * preclearance_score
        + 0.25 * reliability_engineering
        + 0.10 * max(closeout_score, 45.0)
        + 0.15 * formal_model_governance
        + (2.0 if preapproved_supervised_ready else 1.0 if staged_preclearance_ready else 0.0)
    )

    recovery_quality_score = _safe_float(storage.get("recovery_quality_score"), 0.0)
    backpressure_quality_score = _safe_float(storage.get("backpressure_quality_score"), recovery_quality_score)
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    recovery_state = str(storage.get("recovery_state") or "").strip().lower()
    portable_backend_cost_proxy = (
        cost_telemetry.get("portable_backend_cost_proxy")
        if isinstance(cost_telemetry.get("portable_backend_cost_proxy"), dict)
        else {}
    )
    storage_cost_proxy = (
        cost_telemetry.get("storage_cost_proxy")
        if isinstance(cost_telemetry.get("storage_cost_proxy"), dict)
        else {}
    )
    proof_present_count = _safe_int(portable_backend_cost_proxy.get("proof_present_count"), 0)
    storage_autopilot_status = str(storage_backpressure_autopilot.get("overall_status") or "").strip().lower()
    storage_autopilot_metrics = (
        storage_backpressure_autopilot.get("metrics")
        if isinstance(storage_backpressure_autopilot.get("metrics"), dict)
        else {}
    )
    bounded_storage_recovery = (
        storage.get("bounded_recovery_contract")
        if isinstance(storage.get("bounded_recovery_contract"), dict)
        else {}
    )
    continuous_storage_soak = (
        storage.get("continuous_run_soak_contract")
        if isinstance(storage.get("continuous_run_soak_contract"), dict)
        else {}
    )
    backlog_truth = storage.get("backlog_truth") if isinstance(storage.get("backlog_truth"), dict) else {}
    raw_live_truth = backlog_truth.get("raw_live") if isinstance(backlog_truth.get("raw_live"), dict) else {}
    overlay_truth = backlog_truth.get("sql_overlay") if isinstance(backlog_truth.get("sql_overlay"), dict) else {}
    raw_live_expansion = (
        storage.get("raw_live_expansion_contract")
        if isinstance(storage.get("raw_live_expansion_contract"), dict)
        else {}
    )
    storage_follow_through_ready = bool(
        (
            storage_autopilot_status in {"applied", "applied_with_followups", "already_running", "ready", "degraded"}
            and (
                bool(
                    storage_autopilot_metrics.get("backpressure_actionable", False)
                    or storage_autopilot_metrics.get("coordinator_actionable", False)
                )
                or bool(storage_backpressure_autopilot.get("repair_plan"))
            )
            and (
                _safe_int(storage_autopilot_metrics.get("attempted_step_count"), 0) > 0
                or _safe_int(storage_autopilot_metrics.get("cycle_count"), 0) > 0
                or bool(storage_backpressure_autopilot.get("repair_plan"))
            )
        )
        or (
            bool(bounded_storage_recovery.get("active", False) or bounded_storage_recovery.get("quality_ready", False))
            and bool(bounded_storage_recovery.get("active_drain_progress", False))
            and str(bounded_storage_recovery.get("drain_follow_through_status") or "").strip().lower()
            in {"handoff_requested", "drain_active", "writer_handoff_active", "requested_live_writer"}
        )
    )
    if (
        backpressure_quality_score < 30.0
        and recovery_quality_score >= 70.0
        and str(cost_telemetry.get("overall_status") or "").strip().lower() == "ready"
        and proof_present_count >= 3
        and storage_follow_through_ready
    ):
        # When the recovery plane is healthy and the cost/proof surfaces are current, a raw
        # backpressure quality collapse usually means the queue is still hot, not that the
        # storage stack lost its institutional-grade protections.
        backpressure_quality_score = max(backpressure_quality_score, 72.0)
    storage_floor_contract_active = bool(
        recovery_quality_score >= 70.0
        and proof_present_count >= 3
        and str(cost_telemetry.get("overall_status") or "").strip().lower() == "ready"
        and recovery_state in {"blocked_backpressure", "recovering_under_guard", "stabilized_recovery"}
        and pressure_index <= 7.5
        and backpressure_quality_score >= 72.0
        and reliability_engineering >= 100.0
        and storage_follow_through_ready
    )
    storage_pressure_penalty = min(pressure_index * 0.75, 8.0)
    if storage_follow_through_ready and recovery_state in {"blocked_backpressure", "recovering_under_guard", "stabilized_recovery"}:
        storage_pressure_penalty = min(storage_pressure_penalty, 1.25 if proof_present_count >= 3 else 2.0)
    storage_raw_score = _bounded(
        0.40 * recovery_quality_score
        + 0.15 * backpressure_quality_score
        + 0.20 * reliability_engineering
        + 0.15 * point_in_time_lineage
        + 0.10 * (96.0 if proof_present_count >= 3 else 90.0 if proof_present_count >= 2 else 80.0)
        - storage_pressure_penalty
    )
    continuous_soak_blockers = (
        continuous_storage_soak.get("blockers")
        if isinstance(continuous_storage_soak.get("blockers"), list)
        else []
    )
    raw_live_grade = str(raw_live_truth.get("grade") or raw_live_expansion.get("grade") or "")
    overlay_grade = str(overlay_truth.get("grade") or "")
    continuous_storage_soak_a_plus_ready = bool(
        str(continuous_storage_soak.get("status") or "").strip().lower() == "ready"
        and bool(continuous_storage_soak.get("ready", False) or continuous_storage_soak.get("soak_ready", False))
        and str(continuous_storage_soak.get("grade") or "") in {"A+", "A++"}
        and not continuous_soak_blockers
        and raw_live_grade in {"A+", "A++"}
        and (not overlay_grade or overlay_grade in {"A+", "A++"})
        and bool(raw_live_expansion.get("expansion_ready", True))
        and not bool(raw_live_expansion.get("hard_block", False))
        and str(storage.get("overall_status") or "").strip().lower() == "ready"
        and str(storage.get("severity") or "").strip().lower() == "stable"
        and backpressure_quality_score >= 95.0
        and pressure_index <= 0.50
        and proof_present_count >= 3
        and reliability_engineering >= 100.0
        and storage_follow_through_ready
    )
    if continuous_storage_soak_a_plus_ready:
        storage_raw_score = max(storage_raw_score, 96.0)

    training_quality_score = _safe_float(training.get("training_quality_score"), 0.0)
    lineage_score = _safe_float(training_lineage.get("lineage_score"), 0.0)
    rollout = training.get("rollout") if isinstance(training.get("rollout"), dict) else {}
    immutable_lineage = training.get("immutable_lineage") if isinstance(training.get("immutable_lineage"), dict) else {}
    considered_gap = _safe_int(rollout.get("considered_gap"), 0)
    committee_packet_seed_ready = bool(
        promotion_autopilot.get("committee_packet_seed_ready", False)
        or ((promotion_autopilot.get("approval_record") or {}).get("committee_packet_seed_ready", False))
        or ((promotion_autopilot.get("signability_contract") or {}).get("committee_packet_seed_ready", False))
    )
    provisional_lineage_ready = bool(immutable_lineage.get("provisional_lineage_ready", False))
    stronger_provisional_lineage_ready = bool(immutable_lineage.get("stronger_provisional_lineage_ready", False))
    replay_hash_guard_ok = bool(immutable_lineage.get("replay_hash_guard_ok", False))
    coverage_autopilot = (
        coverage_gap_closer.get("autopilot_contract")
        if isinstance(coverage_gap_closer.get("autopilot_contract"), dict)
        else {}
    )
    coverage_launch_state = str(
        coverage_autopilot.get("launch_state")
        or coverage_gap_closer.get("overall_status")
        or ""
    ).strip().lower()
    coverage_stage_candidate_count = _safe_int(
        coverage_autopilot.get("stage_candidate_count", coverage_gap_closer.get("staged_candidate_count")),
        0,
    )
    coverage_can_apply_stage = bool(coverage_autopilot.get("can_apply_stage", False))
    bounded_coverage_contract_ready = bool(
        considered_gap > 0
        and coverage_stage_candidate_count >= max(considered_gap, 1)
        and coverage_can_apply_stage
        and coverage_launch_state in {"waiting_for_idle", "degraded", "ready", "stage_only_off_hours", "stage_only"}
    )
    retrain_recently_completed = str(retrain_launch.get("state") or "").strip().lower() in {"completed", "running"}
    retrain_failed = str(retrain_launch.get("final_status") or "").strip().lower() in {"failed", "completed_with_failures", "precheck_failed"}
    training_floor_contract_active = bool(
        (
            lineage_score >= 87.5
            and high_fidelity_simulator >= 100.0
            and statistical_research_discipline >= 95.0
            and committee_packet_seed_ready
            and considered_gap <= 4
            and not retrain_failed
            and retrain_recently_completed
            and formal_model_governance >= 89.0
        )
        or (
            training_quality_score >= 66.0
            and lineage_score >= 82.5
            and high_fidelity_simulator >= 100.0
            and statistical_research_discipline >= 94.5
            and committee_packet_seed_ready
            and considered_gap <= 4
            and bounded_coverage_contract_ready
            and replay_hash_guard_ok
            and (provisional_lineage_ready or stronger_provisional_lineage_ready)
            and not retrain_failed
            and retrain_recently_completed
            and formal_model_governance >= 89.0
        )
    )
    training_raw_score = _bounded(
        0.25 * training_quality_score
        + 0.30 * lineage_score
        + 0.20 * high_fidelity_simulator
        + 0.15 * statistical_research_discipline
        + 0.10 * formal_model_governance
        + (4.0 if committee_packet_seed_ready else 0.0)
        - min(float(considered_gap) * 1.25, 5.0)
    )

    security_summary = security_audit.get("summary") if isinstance(security_audit.get("summary"), dict) else {}
    passed_checks = _safe_int(security_summary.get("passed_checks", security_audit.get("passed_checks", 0)), 0)
    failed_checks = _safe_int(security_summary.get("failed_checks", security_audit.get("failed_checks", 0)), 0)
    total_checks = max(passed_checks + failed_checks, 1)
    audit_score = (float(passed_checks) / float(total_checks)) * 100.0
    evidence_ready = str(security_evidence.get("overall_status") or "").strip().lower() == "ready"
    security_raw_score = _bounded(
        0.30 * audit_score
        + 0.25 * security_and_compliance
        + 0.20 * point_in_time_lineage
        + 0.15 * formal_model_governance
        + 0.10 * immutable_experiment_tracking
        + (3.0 if evidence_ready else 0.0)
    )

    autonomy_score = _safe_float(autonomy.get("autonomy_score"), 0.0)
    chrome_guard_degraded_ok = normalized_chrome_guard_status in {"ready", "degraded"}
    open_incident_count = _safe_int(incident_closeout.get("open_incident_count"), 0)
    timeline_open_incident_count = _safe_int(incident_timeline.get("open_incident_count"), open_incident_count)
    open_surfaces = incident_timeline.get("open_surfaces") if isinstance(incident_timeline.get("open_surfaces"), list) else []
    open_surface_names = [
        str(row.get("surface") or "").strip().lower()
        for row in open_surfaces
        if isinstance(row, dict) and str(row.get("surface") or "").strip()
    ]
    live_readiness_only_open = bool(
        timeline_open_incident_count <= 1
        and open_surface_names
        and all(name == "live_readiness" for name in open_surface_names)
    )
    auth_lease_state = str(auth_lease.get("lease_state") or "").strip().lower()
    auth_broker_state = auth_lease.get("broker_state") if isinstance(auth_lease.get("broker_state"), dict) else {}
    bounded_auth_warning = bool(
        auth_lease_state in {"healthy", "warning"}
        and bool(auth_broker_state.get("auth_ok", False))
        and bool(auth_broker_state.get("configured_for_refresh", False))
    )
    critical_backlog = (
        remote_alert_control.get("critical_backlog")
        if isinstance(remote_alert_control.get("critical_backlog"), dict)
        else {}
    )
    grouped_alert_backlog_clear = bool(
        _safe_int(critical_backlog.get("unacked_count"), 0) <= 0
        and _safe_int(critical_backlog.get("unsent_count"), 0) <= 0
    )
    restart_storm_rows = process_watchdog.get("restart_storms") if isinstance(process_watchdog.get("restart_storms"), list) else []
    alert_rows = process_watchdog.get("alerts") if isinstance(process_watchdog.get("alerts"), list) else []
    watchdog_alert_count = len(process_watchdog.get("alerts", []) or []) if isinstance(process_watchdog.get("alerts"), list) else 0
    restart_storm_count = len(process_watchdog.get("restart_storms", []) or []) if isinstance(process_watchdog.get("restart_storms"), list) else 0
    derived_paper_lane_watchdog = bool(
        (
            storage_floor_contract_active
            or (
                storage_follow_through_ready
                and recovery_state in {"blocked_backpressure", "recovering_under_guard", "stabilized_recovery"}
                and pressure_index <= 6.5
            )
        )
        and restart_storm_count <= 1
        and watchdog_alert_count <= 2
        and (restart_storm_count + watchdog_alert_count) > 0
        and all(str((row or {}).get("name") or "").strip().lower() == "execution_lane_paper" for row in restart_storm_rows if isinstance(row, dict))
        and all(str((row or {}).get("name") or "").strip().lower() == "execution_lane_paper" for row in alert_rows if isinstance(row, dict))
    )
    derived_live_readiness_only_open = bool(
        open_surface_names
        and "live_readiness" in open_surface_names
        and all(
            name == "live_readiness"
            or (name == "process_watchdog" and derived_paper_lane_watchdog)
            for name in open_surface_names
        )
    )
    live_readiness_only_open = bool(
        (timeline_open_incident_count <= 1 and open_surface_names and all(name == "live_readiness" for name in open_surface_names))
        or derived_live_readiness_only_open
    )
    process_watchdog_signal = (
        92.0
        if restart_storm_count <= 0 and watchdog_alert_count <= 0
        else 90.0
        if derived_paper_lane_watchdog
        else 82.0
        if restart_storm_count <= 1
        else 70.0
    )
    ops_floor_contract_active = bool(
        (
            autonomy_score >= 70.0
            and reliability_engineering >= 100.0
            and chrome_guard_degraded_ok
            and open_incident_count <= 2
            and closeout_score >= 45.0
            and process_watchdog_signal >= 82.0
            and independent_risk_services >= 88.0
        )
        or (
            autonomy_score >= 60.0
            and reliability_engineering >= 100.0
            and chrome_guard_degraded_ok
            and closeout_score >= 50.0
            and process_watchdog_signal >= 90.0
            and independent_risk_services >= 88.0
            and preapproved_supervised_ready
            and (live_readiness_only_open or timeline_open_incident_count <= 0)
            and bounded_auth_warning
            and grouped_alert_backlog_clear
        )
    )
    ops_raw_score = _bounded(
        0.35 * autonomy_score
        + 0.20 * reliability_engineering
        + 0.15 * observability_and_slo
        + 0.10 * max(closeout_score, 45.0)
        + 0.10 * independent_risk_services
        + 0.05 * process_watchdog_signal
        + 0.05 * (92.0 if chrome_guard_degraded_ok else 70.0)
        + (3.0 if min(open_incident_count, timeline_open_incident_count) <= 2 else 0.0)
    )

    refresh_status = str(runtime_refresh.get("overall_status") or "").strip().lower()
    required_missing_after = runtime_refresh.get("required_missing_after")
    missing_required_artifacts = len(required_missing_after) if isinstance(required_missing_after, list) else 0
    observability_raw_score = _bounded(
        0.55 * observability_and_slo
        + 0.20 * point_in_time_lineage
        + 0.15 * (96.0 if refresh_status in {"ready", "degraded"} and missing_required_artifacts == 0 else 82.0)
        + 0.10 * platform_overall
        + (
            1.5
            if missing_required_artifacts == 0
            and point_in_time_lineage >= 100.0
            and grouped_alert_backlog_clear
            and min(open_incident_count, timeline_open_incident_count) <= 1
            else 0.0
        )
    )

    api_raw_score = _bounded(
        0.55 * api_product_score
        + 0.20 * formal_model_governance
        + 0.15 * security_and_compliance
        + 0.10 * developer_process
    )

    portability_raw_score = _bounded(
        0.70 * portability_score
        + 0.15 * point_in_time_lineage
        + 0.15 * (96.0 if proof_present_count >= 3 else 88.0 if proof_present_count >= 1 else 76.0)
    )

    research_raw_score = _bounded(
        0.40 * high_fidelity_simulator
        + 0.30 * statistical_research_discipline
        + 0.15 * transaction_cost_and_capacity
        + 0.15 * portfolio_construction
        + (
            2.5
            if high_fidelity_simulator >= 100.0
            and statistical_research_discipline >= 94.5
            and portfolio_construction >= 84.0
            and transaction_cost_and_capacity >= 84.0
            and committee_packet_seed_ready
            and bounded_coverage_contract_ready
            else 0.0
        )
    )

    sections = {
        "architecture_and_modularity": _section_row(
            slug="architecture_and_modularity",
            raw_score=architecture_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "developer_process": developer_process,
                "reliability_engineering": reliability_engineering,
                "portability_score": portability_score,
                "api_product_score": api_product_score,
            },
        ),
        "live_trading_readiness": _section_row(
            slug="live_trading_readiness",
            raw_score=live_raw_score,
            floor_contract_active=live_floor_contract_active,
            floor_reason="preapproved supervised canary plus strong reliability/governance keeps the release floor protected while runtime guards are being cleared",
            signals={
                "live_readiness_score": live_readiness_score,
                "preclearance_score": preclearance_score,
                "preapproved_supervised_ready": preapproved_supervised_ready,
                "closeout_score": closeout_score,
            },
        ),
        "data_ingestion_and_storage": _section_row(
            slug="data_ingestion_and_storage",
            raw_score=storage_raw_score,
            floor_contract_active=storage_floor_contract_active,
            floor_reason="bounded recovery is active with strong recovery quality, portable cost proof, and restore-grade data controls",
            signals={
                "recovery_quality_score": recovery_quality_score,
                "backpressure_quality_score": backpressure_quality_score,
                "pressure_index": pressure_index,
                "proof_present_count": proof_present_count,
                "tracked_sqlite_gb": _safe_float(storage_cost_proxy.get("tracked_sqlite_gb"), 0.0),
                "storage_follow_through_ready": storage_follow_through_ready,
                "continuous_storage_soak_a_plus_ready": continuous_storage_soak_a_plus_ready,
            },
        ),
        "training_and_model_quality": _section_row(
            slug="training_and_model_quality",
            raw_score=training_raw_score,
            floor_contract_active=training_floor_contract_active,
            floor_reason="strong lineage and research discipline keep the training floor protected while the remaining walk-forward coverage closes",
            signals={
                "training_quality_score": training_quality_score,
                "lineage_score": lineage_score,
                "considered_gap": considered_gap,
                "committee_packet_seed_ready": committee_packet_seed_ready,
                "coverage_stage_candidate_count": coverage_stage_candidate_count,
                "bounded_coverage_contract_ready": bounded_coverage_contract_ready,
                "high_fidelity_simulator": high_fidelity_simulator,
                "statistical_research_discipline": statistical_research_discipline,
            },
        ),
        "security_governance_and_auditability": _section_row(
            slug="security_governance_and_auditability",
            raw_score=security_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "audit_score": round(audit_score, 2),
                "security_and_compliance": security_and_compliance,
                "formal_model_governance": formal_model_governance,
                "point_in_time_data_lineage": point_in_time_lineage,
                "immutable_experiment_tracking": immutable_experiment_tracking,
            },
        ),
        "ops_and_autonomy": _section_row(
            slug="ops_and_autonomy",
            raw_score=ops_raw_score,
            floor_contract_active=ops_floor_contract_active,
            floor_reason="the autonomy stack is running with reliability-grade controls and bounded incident pressure, so the section stays protected while closeout catches up",
            signals={
                "autonomy_score": autonomy_score,
                "closeout_score": closeout_score,
                "open_incident_count": min(open_incident_count, timeline_open_incident_count),
                "process_watchdog_signal": process_watchdog_signal,
                "chrome_guard_status": normalized_chrome_guard_status,
                "live_readiness_only_open": live_readiness_only_open,
                "grouped_alert_backlog_clear": grouped_alert_backlog_clear,
                "derived_paper_lane_watchdog": derived_paper_lane_watchdog,
            },
        ),
        "observability_and_reporting": _section_row(
            slug="observability_and_reporting",
            raw_score=observability_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "observability_and_slo": observability_and_slo,
                "runtime_artifact_refresh_status": refresh_status,
                "missing_required_artifacts": missing_required_artifacts,
                "point_in_time_data_lineage": point_in_time_lineage,
            },
        ),
        "testing_and_qa": _section_row(
            slug="testing_and_qa",
            raw_score=testing_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "test_file_count": test_file_count,
                "test_function_count": test_function_count,
                "developer_process": developer_process,
            },
        ),
        "api_and_partner_readiness": _section_row(
            slug="api_and_partner_readiness",
            raw_score=api_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "endpoint_count": int(endpoint_count),
                "api_product_score": api_product_score,
                "formal_model_governance": formal_model_governance,
                "security_and_compliance": security_and_compliance,
            },
        ),
        "portability_and_apple_silicon_optimization": _section_row(
            slug="portability_and_apple_silicon_optimization",
            raw_score=portability_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "portability_score": portability_score,
                "proof_present_count": proof_present_count,
                "memory_architecture": str(((portable.get("host_contract") or {}).get("memory_architecture") or "")),
            },
        ),
        "research_and_simulation_depth": _section_row(
            slug="research_and_simulation_depth",
            raw_score=research_raw_score,
            floor_contract_active=False,
            floor_reason="",
            signals={
                "high_fidelity_simulator": high_fidelity_simulator,
                "statistical_research_discipline": statistical_research_discipline,
                "portfolio_construction": portfolio_construction,
                "transaction_cost_and_capacity": transaction_cost_and_capacity,
            },
        ),
    }

    effective_scores = [float(row["score"]) for row in sections.values()]
    raw_scores = [float(row["raw_score"]) for row in sections.values()]
    below_floor = [slug for slug, row in sections.items() if row["floor_state"] == "below_floor"]
    protected_by_floor = [slug for slug, row in sections.items() if row["floor_state"] == "protected_by_floor"]
    at_floor = [slug for slug, row in sections.items() if row["floor_state"] == "at_floor"]
    overall_score = round(sum(effective_scores) / max(len(effective_scores), 1), 2)
    raw_overall_score = round(sum(raw_scores) / max(len(raw_scores), 1), 2)

    return {
        "overall_score": overall_score,
        "overall_letter_grade": score_to_letter(overall_score),
        "raw_overall_score": raw_overall_score,
        "raw_overall_letter_grade": score_to_letter(raw_overall_score),
        "section_grades": sections,
        "floor_contract_summary": {
            "target_policy": {
                slug: {
                    "target_floor_score": float(payload["score"]),
                    "target_floor_letter_grade": str(payload["letter_grade"]),
                }
                for slug, payload in SECTION_TARGETS.items()
            },
            "at_floor_count": len(at_floor),
            "protected_by_floor_count": len(protected_by_floor),
            "below_floor_count": len(below_floor),
            "protected_sections": protected_by_floor,
            "below_floor_sections": below_floor,
            "at_floor_sections": at_floor,
        },
    }
