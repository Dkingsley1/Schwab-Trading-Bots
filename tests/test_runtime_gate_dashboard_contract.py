import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

MODULE_PATH = Path(
    "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/runtime_gate_dashboard.py"
)
spec = importlib.util.spec_from_file_location(
    "runtime_gate_dashboard_contract", MODULE_PATH
)
runtime_gate_dashboard = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runtime_gate_dashboard)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_runtime_gate_dashboard_marks_missing_sections_with_explicit_contract_state(
    tmp_path,
):
    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["overall_status"] == payload["overall"]["status"]
    assert payload["ok"] == payload["overall"]["ok"]
    assert payload["runtime"]["artifact_status"] == "missing"
    assert payload["runtime"]["artifact_reason"] == "artifact_missing"
    assert payload["runtime"]["mode"] == "unknown"
    assert payload["apple_silicon"]["artifact_status"] == "missing"
    assert payload["memory"]["artifact_status"] == "missing"
    assert payload["training"]["artifact_status"] == "missing"
    assert payload["platform"]["artifact_status"] == "missing"


def test_runtime_gate_dashboard_manages_paper_soak_auth_warning_attention(tmp_path):
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "paper_armed": True,
            "paper_stage": "armed",
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )
    auth_path = health / "auth_lease_manager_latest.json"
    supervisor_path = health / "schwab_auth_supervisor_latest.json"
    _write_json(
        auth_path,
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {
                "expires_in_seconds": 1120,
                "critical_lease_seconds": 600,
                "token_lease_grace": True,
            },
            "broker_state": {
                "broker_ready": True,
                "broker_operable": True,
                "network_ok": True,
                "auth_ok": False,
                "auth_probe_ok": False,
            },
        },
    )
    _write_json(
        supervisor_path,
        {"overall_status": "ready", "ok": True, "paper_soak_auth_operable": True},
    )

    artifacts = {
        "auth_lease_manager": {
            "path": str(auth_path),
            "summary": {"overall_status": "degraded"},
        },
        "schwab_auth_supervisor": {
            "path": str(supervisor_path),
            "summary": {"overall_status": "ready"},
        },
    }
    context = runtime_gate_dashboard._dashboard_soak_context(tmp_path)
    reason = runtime_gate_dashboard._attention_managed_by_green_soak(
        "auth_lease_manager_needs_work",
        artifacts,
        context,
    )

    assert context["enabled"] is True
    assert context["guarded_health_ready"] is True
    assert (
        reason == "schwab_auth_warning_managed_while_token_above_paper_readiness_floor"
    )


def test_dashboard_soak_context_accepts_guarded_ready_health_fast(tmp_path):
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "guarded_ready",
            "operational_readiness": {"guarded_paper": {"ok": True, "status": "ready"}},
        },
    )

    context = runtime_gate_dashboard._dashboard_soak_context(tmp_path)

    assert context["enabled"] is True
    assert context["guarded_health_ready"] is True


def test_runtime_gate_dashboard_manages_bounded_transient_backlog_attention(tmp_path):
    health = tmp_path / "governance" / "health"
    storage_path = health / "ingestion_storage_control_latest.json"
    storage_payload = {
        "overall_status": "ready",
        "severity": "elevated",
        "pressure_index": 0.926,
        "continuous_run_soak_contract": {
            "status": "blocked",
            "ready": False,
            "soak_ready": False,
            "blockers": ["steady_state_targets_not_clear"],
        },
        "bounded_recovery_contract": {
            "route_verified": True,
            "active_drain_progress": True,
            "drain_delta_signal_observed": True,
            "hard_gate_active": False,
            "effective_hard_gate_active": False,
        },
        "data_integrity": {
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_oversize_payloads": 0,
            "sql_overlay_ops_write_failures": 0,
        },
        "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        "backpressure": {
            "raw_live": {
                "core_pending_lines": 3902,
                "total_pending_lines": 4916,
                "oldest_pending_age_seconds": 222.349,
            }
        },
    }
    _write_json(storage_path, storage_payload)
    artifacts = {
        "ingestion_storage_control": {
            "path": str(storage_path),
            "summary": {
                "overall_status": "ready",
                "severity": "elevated",
                "pressure_index": 0.926,
            },
        },
        "external_backlog_drain": {
            "summary": {
                "overall_status": "drain_active",
                "recommended_now": True,
                "aged_candidate_files": 0,
                "writer_busy": False,
            }
        },
    }

    assert runtime_gate_dashboard._ingestion_soak_ready_for_dashboard(artifacts) is True
    reason = runtime_gate_dashboard._attention_managed_by_green_soak(
        "external_backlog_drain_recommended",
        artifacts,
        {"enabled": True},
    )
    assert reason == "external_backlog_handoff_managed_while_ingestion_soak_is_green"


def _teacher_quality_artifact(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "governance" / "distillation" / "teacher_quality_latest.json"
    payload = {
        "overall_status": "blocked",
        "summary": {
            "qualified_teacher_count": 0,
            "elite_teacher_count": 0,
            "teaching_enabled": False,
        },
        "overfitting_awareness": {
            "overall_status": "guarded",
            "risk_bot_count": 0,
            "hard_risk_bot_count": 0,
            "blocked_teacher_count": 3,
            "blocked_status_counts": {
                "registry_only_guarded": 1,
                "insufficient_evidence": 2,
            },
        },
    }
    _write_json(path, payload)
    return path, payload


def test_runtime_gate_dashboard_manages_fail_closed_unqualified_teacher_pool(tmp_path):
    path, _ = _teacher_quality_artifact(tmp_path)
    artifacts = {"teacher_quality_guard": {"path": str(path), "stale": False}}

    reason = runtime_gate_dashboard._attention_managed_by_green_soak(
        "teacher_quality_guard_blocked",
        artifacts,
        {"enabled": True},
    )

    assert (
        reason
        == "teacher_qualification_deferred_while_unqualified_teachers_are_fail_closed"
    )


def test_runtime_gate_dashboard_does_not_manage_unsafe_or_unknown_teacher_blocks(
    tmp_path,
):
    path, payload = _teacher_quality_artifact(tmp_path)
    artifacts = {"teacher_quality_guard": {"path": str(path), "stale": False}}
    unsafe_variants = (
        ("risk_bot_count", 1),
        ("unknown_teacher_state", 1),
    )

    for status_name, status_count in unsafe_variants:
        candidate = json.loads(json.dumps(payload))
        if status_name == "risk_bot_count":
            candidate["overfitting_awareness"]["risk_bot_count"] = status_count
        else:
            candidate["overfitting_awareness"]["blocked_teacher_count"] += status_count
            candidate["overfitting_awareness"]["blocked_status_counts"][
                status_name
            ] = status_count
        _write_json(path, candidate)
        assert (
            runtime_gate_dashboard._attention_managed_by_green_soak(
                "teacher_quality_guard_blocked",
                artifacts,
                {"enabled": True},
            )
            == ""
        )


def test_runtime_gate_dashboard_downgrades_bot_quality_autopilot_evidence_pending(
    tmp_path,
):
    artifacts = {
        "bot_quality_autopilot": {
            "stale": False,
            "summary": {
                "overall_status": "blocked",
                "quality_queue": 12,
                "qualified_teacher_count": 0,
                "teacher_quality_status": "collecting_evidence",
                "coverage_shortfall_bots": 4,
                "rerouted_targeted_retrain_count": 7,
                "minimum_sample_count": 200,
            },
        },
        "retrain_artifact_freshness": {
            "summary": {"sample_sufficiency_failed_checks": ["paper_replay"]},
        },
        "daily_auto_verify": {
            "summary": {
                "operational_ok": True,
                "effective_failed_checks": [
                    "feature_store_manifest",
                    "retrain_schema_compatibility_guard",
                    "promotion_quality_gate",
                ],
            },
        },
    }

    assert (
        runtime_gate_dashboard._bot_quality_autopilot_deferred_for_evidence_collection(
            artifacts
        )
    )
    assert (
        runtime_gate_dashboard._attention_tier("bot_quality_autopilot_evidence_pending")
        == "advisory"
    )


def test_runtime_gate_dashboard_keeps_evidence_debt_out_of_degraded_attention():
    assert runtime_gate_dashboard._artifact_waiting_on_evidence_not_degraded(
        "bot_profitability_scalability_control",
        "ready_with_evidence_debt",
        {
            "control_grade": "A+",
            "profitability_claim_ready": False,
            "live_allocation_ready": False,
        },
    )
    assert runtime_gate_dashboard._artifact_waiting_on_evidence_not_degraded(
        "sleeve_scalability_selector",
        "ready_with_evidence_debt",
        {
            "control_ready": True,
            "recommendation_ready": False,
            "paper_execution_authority": False,
            "live_execution_authority": False,
        },
    )
    assert runtime_gate_dashboard._artifact_waiting_on_evidence_not_degraded(
        "master_grandmaster_evidence_v2",
        "ready_with_evidence_debt",
        {
            "structural_grade": "A+",
            "paper_coordination_ready": True,
            "automatic_live_promotion_allowed": False,
        },
    )
    assert not runtime_gate_dashboard._artifact_waiting_on_evidence_not_degraded(
        "bot_profitability_scalability_control",
        "blocked_integrity",
        {
            "control_grade": "A+",
            "profitability_claim_ready": False,
            "live_allocation_ready": False,
        },
    )


def test_runtime_gate_dashboard_surfaces_guarded_paper_execution_separately_from_collection(
    tmp_path,
):
    health = tmp_path / "governance" / "health"
    timestamp = datetime.now(timezone.utc).isoformat()
    _write_json(
        health / "all_sleeves_launcher_latest.json",
        {
            "timestamp_utc": timestamp,
            "overall_status": "guarded_ready",
            "expected_job_count": 6,
            "running_job_count": 5,
            "launcher_readiness_contract": {
                "collection_fanout_ready": True,
                "paper_execution_ready": False,
                "readiness_status": "guarded_execution_blocked",
                "execution_attention": ["paper_executor_safety_parked"],
            },
        },
    )
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "timestamp_utc": timestamp,
            "overall_status": "ready",
            "collector_count": 42,
            "total_observations": 12345,
            "collection_coverage_score": 98.5,
            "data_quality_score": 99.0,
        },
    )

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "paper_execution_safety_guard_active" in payload["overall"]["attention"]
    assert payload["execution_runtime"]["collection_fanout_ready"] is True
    assert payload["execution_runtime"]["paper_execution_ready"] is False
    assert payload["data_quality_dimensions"]["collector_count"] == 42
    assert payload["data_quality_dimensions"]["total_observations"] == 12345
    assert (
        payload["data_quality_dimensions"]["collector_coverage_quality_score"] == 98.5
    )
    assert (
        "paper_execution_safety_guard_active"
        in payload["overall"]["attention_tiers"]["advisory"]
    )


def test_runtime_gate_dashboard_contains_evidence_and_context_debt() -> None:
    attention = [
        "paper_execution_safety_guard_active",
        "source_verification_context_debt",
        "promotion_not_ready",
    ]
    tiers = runtime_gate_dashboard._attention_tiers(attention)

    containment = runtime_gate_dashboard._degradation_containment_contract(
        attention,
        tiers,
        {},
        [],
    )

    assert containment["status"] == "contained"
    assert containment["uncontained_count"] == 0
    assert containment["hot_path_blocked"] is False
    assert {row["domain"] for row in containment["containment_rows"]} >= {
        "execution_safety",
        "market_context",
        "training_promotion",
    }


def test_runtime_gate_dashboard_routes_known_attention_to_owner_commands() -> None:
    actions = runtime_gate_dashboard._remediation_actions(
        [
            "promotion_not_ready",
            "roster_resilience_planner_needs_work",
            "coordination_state_control_needs_work",
            "sql_link_service_not_ok",
            "bot_quality_autopilot_evidence_pending",
            "remote_alert_control_needs_work",
        ]
    )
    by_attention = {row["attention"]: row for row in actions}

    assert by_attention["promotion_not_ready"]["owner"] == "promotion_quality_gate"
    assert by_attention["promotion_not_ready"]["command"][1] == "promotion-quality-gate"
    assert (
        by_attention["roster_resilience_planner_needs_work"]["owner"]
        == "roster_resilience_planner"
    )
    assert (
        by_attention["roster_resilience_planner_needs_work"]["command"][1]
        == "roster-resilience"
    )
    assert (
        by_attention["coordination_state_control_needs_work"]["owner"]
        == "coordination_state_control"
    )
    assert (
        by_attention["coordination_state_control_needs_work"]["command"][1]
        == "coordination-status"
    )
    assert by_attention["sql_link_service_not_ok"]["owner"] == "sql_link_service"
    assert by_attention["sql_link_service_not_ok"]["command"][1] == "sql-link-service"
    assert (
        by_attention["bot_quality_autopilot_evidence_pending"]["owner"]
        == "bot_quality_autopilot"
    )
    assert (
        by_attention["bot_quality_autopilot_evidence_pending"]["command"][1]
        == "bot-quality-autopilot"
    )
    assert (
        by_attention["remote_alert_control_needs_work"]["owner"]
        == "remote_alert_control"
    )
    assert (
        by_attention["remote_alert_control_needs_work"]["command"][1]
        == "remote-alert-control"
    )
    assert all(row["owner"] != "operator_review" for row in actions)


def test_runtime_gate_dashboard_ops_smoothing_separates_collection_from_promotion() -> (
    None
):
    artifacts = {
        "all_sleeves_launcher": {
            "summary": {
                "collection_fanout_ready": True,
                "paper_execution_ready": False,
            }
        },
        "ingestion_storage_control": {
            "summary": {
                "overall_status": "ready",
                "severity": "stable",
                "pressure_index": 0.1,
            }
        },
        "health_gates": {"summary": {"hard_gate_triggered": False}},
        "global_killswitch": {"summary": {"halt": False}},
        "source_verification": {
            "summary": {
                "overall_status": "degraded",
                "decision_critical_sources_ready": True,
                "decision_context_debt": ["decision_context_mesh"],
            }
        },
        "bot_profitability_scalability_control": {
            "summary": {
                "overall_status": "ready_with_evidence_debt",
                "live_allocation_ready": False,
            }
        },
        "sleeve_scalability_selector": {
            "summary": {
                "overall_status": "ready_with_evidence_debt",
                "recommendation_ready": False,
            }
        },
        "master_grandmaster_evidence_v2": {
            "summary": {
                "overall_status": "ready_with_evidence_debt",
                "automatic_live_promotion_allowed": False,
                "promotion_blockers": ["profitability_evidence_ready"],
            }
        },
    }

    smoothing = runtime_gate_dashboard._ops_smoothing_contract(
        artifacts,
        ["source_verification_context_debt", "promotion_not_ready"],
        runtime_gate_dashboard._attention_tiers(
            ["source_verification_context_debt", "promotion_not_ready"]
        ),
        [],
    )

    assert smoothing["status"] == "ready_to_collect_evidence"
    assert smoothing["safe_to_collect_evidence"] is True
    assert smoothing["safe_to_promote_or_live_trade"] is False
    assert "decision_context_mesh" in smoothing["source_context_debt"]
    assert (
        "bot_profitability_scalability_control" in smoothing["evidence_debt_surfaces"]
    )
