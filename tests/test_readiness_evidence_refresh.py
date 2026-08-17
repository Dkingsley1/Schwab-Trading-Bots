import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import readiness_evidence_refresh as refresh


def test_refresh_profiles_are_bounded_and_keep_required_ordering() -> None:
    accrual = [row["name"] for row in refresh.profile_steps("accrual")]
    dashboard = [row["name"] for row in refresh.profile_steps("dashboard")]
    production = [row["name"] for row in refresh.profile_steps("production")]

    assert accrual == [
        "market_replay_fill_capture",
        "runtime_training_snapshot",
        "point_in_time_event_store",
        "snapshot_coverage",
        "feature_store_manifest",
        "collector_contracts",
        "source_verification",
        "capability_materialization",
        "collector_capability_control",
        "provider_mesh",
        "independent_fill_acquisition",
        "paper_execution_calibration",
        "paper_performance",
        "paper_profitability_control",
        "readiness_evidence_accrual",
    ]
    assert len(dashboard) < len(refresh.default_steps())
    assert len(production) < len(refresh.default_steps())
    assert dashboard.index("market_replay_fill_capture") < dashboard.index("independent_fill_acquisition")
    assert dashboard.index("runtime_training_snapshot") < dashboard.index("feature_store_manifest")
    assert dashboard.index("snapshot_coverage") < dashboard.index("feature_store_manifest")
    assert dashboard.index("feature_store_manifest") < dashboard.index("collector_contracts")
    assert dashboard.index("collector_contracts") < dashboard.index("source_verification")
    assert dashboard.index("source_verification") < dashboard.index("capability_materialization")
    assert dashboard.index("capability_materialization") < dashboard.index("collector_capability_control")
    assert dashboard.index("source_verification") < dashboard.index("provider_mesh")
    assert dashboard.index("collector_capability_control") < dashboard.index("provider_mesh")
    assert dashboard.index("storage_retention_unison") < dashboard.index("notification_escalation_ladder")
    assert dashboard.index("state_snapshot_restore_drill") < dashboard.index("storage_resilience_control")
    assert dashboard.index("storage_resilience_control") < dashboard.index("ingestion_storage_control")
    assert dashboard.index("ingestion_storage_control") < dashboard.index("blackstart_recovery")
    assert dashboard.index("blackstart_recovery") < dashboard.index("unattended_soak_readiness")
    assert dashboard.index("notification_escalation_ladder") < dashboard.index("unattended_soak_readiness")
    assert dashboard.index("paper_execution_calibration") < dashboard.index("readiness_evidence_accrual")
    dashboard_steps = {row["name"]: row for row in refresh.profile_steps("dashboard")}
    assert set(dashboard_steps["storage_retention_unison"]["allowed_returncodes"]) == {0, 2}

    required_pillar_owners = {
        "memory_pressure_intelligence",
        "autonomic_resource_governor",
        "unattended_soak_readiness",
        "chaos_drill_coordinator",
        "risk_service_boundary",
        "coherent_training_profitability_refresh",
        "promotion_quality_gate",
        "canary_rollout",
        "live_canary_control",
        "content_addressed_store",
        "production_readiness",
        "production_excellence",
        "system_drift_guard",
        "master_infrastructure_supervisor",
    }
    assert required_pillar_owners.issubset(set(production))
    assert production.index("memory_pressure_intelligence") < production.index("autonomic_resource_governor")
    assert production.index("autonomic_resource_governor") < production.index("coherent_training_profitability_refresh")
    assert production.index("coherent_training_profitability_refresh") < production.index("promotion_candidate_advancement")
    assert production.index("one_numbers_report") < production.index("portfolio_risk_ledger")
    assert production.index("portfolio_risk_ledger") < production.index("execution_budget")
    assert production.index("execution_budget") < production.index("risk_service_boundary")
    assert production.index("content_addressed_store") < production.index("storage_disaster_recovery")
    assert production.index("secret_scan") < production.index("security_evidence_autofix")
    assert production.index("security_evidence_autofix") < production.index("security_audit")
    assert production.index("state_snapshot_restore_drill") < production.index("storage_resilience_control")
    assert production.index("storage_resilience_control") < production.index("ingestion_storage_control")
    assert production.index("ingestion_storage_control") < production.index("blackstart_recovery")
    assert production.index("storage_disaster_recovery") < production.index("blackstart_recovery")
    production_steps = {row["name"]: row for row in refresh.profile_steps("production")}
    assert production_steps["coherent_training_profitability_refresh"]["args"] == [
        "--scope",
        "training-profitability",
        "--skip-dashboard",
        "--json",
    ]
    assert production_steps["content_addressed_store"]["args"] == ["--no-gc", "--json"]
    assert production_steps["security_evidence_autofix"]["args"] == ["--json"]
    assert production_steps["security_evidence_autofix"]["max_age_minutes"] == 60.0
    assert production_steps["security_audit"]["depends_on"] == [
        "secret_scan",
        "security_evidence_autofix",
    ]
    assert "--apply" in production_steps["storage_disaster_recovery"]["args"]
    assert set(production_steps["live_canary_control"]["allowed_returncodes"]) == {0, 2}
    assert production_steps["livefeed_refresh_guard"]["args"] == ["--apply", "--json"]
    assert production_steps["livefeed_refresh_guard"]["max_age_minutes"] == 15.0
    assert production_steps["stateful_storage_regression_guard"]["args"] == ["--apply", "--json"]
    assert production_steps["codex_project_guard"]["args"] == ["--staged", "--json"]
    assert production_steps["incident_closeout"]["artifact"] == (
        "governance/health/incident_closeout_autopilot_latest.json"
    )
    assert set(production_steps["system_drift_guard"]["depends_on"]) == {
        "one_numbers_regression_guard",
        "codex_project_guard",
        "coinbase_api_health",
        "incident_closeout",
        "section_grade_guard",
        "adaptive_regression_guard",
        "system_architecture_contract_graph",
        "system_architecture_autopilot",
    }
    assert set(production_steps["system_architecture_contract_graph"]["depends_on"]) >= {
        "system_drift_registry",
        "schwab_indicator_intelligence",
        "system_expansion_execution",
        "distributed_cell_architecture",
        "architecture_hardening",
    }
    assert production_steps["master_infrastructure_supervisor"]["depends_on"] == ["system_drift_guard"]
    assert production_steps["system_self_model_settled"]["depends_on"] == ["master_infrastructure_supervisor"]
    assert production_steps["system_architecture_contract_graph_settled"]["depends_on"] == ["system_self_model_settled"]
    assert production_steps["system_architecture_autopilot_settled"]["depends_on"] == [
        "system_architecture_contract_graph_settled"
    ]
    assert production.index("master_infrastructure_supervisor") < production.index("system_self_model_settled")
    assert production.index("system_self_model_settled") < production.index("system_architecture_contract_graph_settled")
    assert production.index("system_architecture_contract_graph_settled") < production.index(
        "system_architecture_autopilot_settled"
    )


def test_accrual_collectors_are_bounded_and_evidence_only() -> None:
    steps = {row["name"]: row for row in refresh.profile_steps("accrual")}

    replay_capture = steps["market_replay_fill_capture"]
    assert replay_capture["args"] == [
        "--apply",
        "--max-bytes-per-observation-file",
        "8388608",
        "--json",
    ]
    snapshot = steps["runtime_training_snapshot"]
    assert snapshot["max_age_minutes"] == 15.0
    assert snapshot["args"] == [
        "--reuse-if-fresh-minutes",
        "15",
        "--incremental-max-runtime-seconds",
        "30",
        "--incremental-max-candidate-rows",
        "5000",
        "--json",
    ]
    assert set(steps["feature_store_manifest"]["allowed_returncodes"]) == {0, 2}
    assert set(steps["snapshot_coverage"]["allowed_returncodes"]) == {0, 2}
    assert set(steps["paper_execution_calibration"]["allowed_returncodes"]) == {0, 2}
    assert "--include-data-plane" in steps["collector_contracts"]["args"]


NOW = datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spec(artifact: str, *, allowed=(0,), max_age=15) -> dict:
    return {
        "name": "test_step",
        "script": "scripts/test_step.py",
        "artifact": artifact,
        "args": ["--json"],
        "max_age_minutes": max_age,
        "allowed_returncodes": list(allowed),
        "depends_on": [],
    }


def test_fresh_artifact_is_not_recomputed(tmp_path: Path) -> None:
    artifact = tmp_path / "governance" / "health" / "test_latest.json"
    _write(artifact, {"timestamp_utc": (NOW - timedelta(minutes=2)).isoformat()})

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("fresh step should not execute")

    payload = refresh.refresh(
        tmp_path,
        steps=[_spec("governance/health/test_latest.json")],
        runner=should_not_run,
        now=NOW,
    )

    assert payload["ok"] is True
    assert payload["fresh_step_count"] == 1
    assert payload["refreshed_step_count"] == 0


def test_due_step_accepts_evidence_pending_return_code(tmp_path: Path) -> None:
    artifact = tmp_path / "governance" / "health" / "test_latest.json"

    def runner(*_args, **_kwargs):
        _write(artifact, {"timestamp_utc": NOW.isoformat(), "overall_status": "evidence_pending", "ok": False})
        return {"rc": 2, "stdout": json.dumps({"overall_status": "evidence_pending", "ok": False}), "stderr": "", "timed_out": False}

    payload = refresh.refresh(
        tmp_path,
        steps=[_spec("governance/health/test_latest.json", allowed=(0, 2))],
        runner=runner,
        now=NOW,
    )

    assert payload["overall_status"] == "ready"
    assert payload["refreshed_step_count"] == 1
    assert payload["operational_failures"] == []
    assert payload["steps"][0]["published_status"] == "evidence_pending"


def test_timeout_is_an_operational_failure(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        return {"rc": 124, "stdout": "", "stderr": "timeout", "timed_out": True}

    payload = refresh.refresh(
        tmp_path,
        steps=[_spec("governance/health/missing.json")],
        runner=runner,
        now=NOW,
    )

    assert payload["ok"] is False
    assert payload["operational_failures"] == ["test_step"]


def test_refresh_report_cooldown_returns_without_rewriting(tmp_path: Path) -> None:
    out = tmp_path / "governance" / "health" / "readiness_evidence_refresh_latest.json"
    _write(out, {"timestamp_utc": (NOW - timedelta(minutes=2)).isoformat(), "overall_status": "ready", "ok": True})

    payload = refresh.refresh(tmp_path, steps=[], now=NOW)

    assert payload["refresh_skipped"] is True
    assert payload["write_latest"] is False
    assert payload["refresh_skip_reason"] == "cooldown_active"


def test_unattended_soak_runs_after_all_freshness_dependencies() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}

    assert set(steps["unattended_soak_readiness"]["depends_on"]) == {
        "storage_retention_unison",
        "notification_escalation_ladder",
        "livefeed_refresh_guard",
        "storage_resilience_control",
        "ingestion_storage_control",
        "blackstart_recovery",
        "capability_materialization",
        "collector_capability_control",
        "provider_mesh",
    }
    for name in steps["unattended_soak_readiness"]["depends_on"]:
        assert steps[name]["max_age_minutes"] < 180
    assert "--apply" not in steps["storage_retention_unison"]["args"]


def test_profitability_firewall_runs_after_all_hardening_evidence_producers() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}
    dependencies = set(steps["profitability_evidence_firewall"]["depends_on"])

    assert {
        "paper_execution_calibration",
        "paper_profitability_control",
        "execution_queue_stress",
        "multiple_testing_guard",
        "decay_monitor",
        "profitability_independent_validator",
        "profitability_holdout_vault",
        "profitability_benchmark_capture",
        "profitability_benchmark_hurdle",
    }.issubset(dependencies)
    assert "profitability_evidence_firewall" in steps["production_excellence"]["depends_on"]


def test_production_quality_refreshes_health_gates_before_derived_controls() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}

    assert steps["health_gates"]["script"] == "scripts/health_gates.py"
    assert steps["health_gates"]["max_age_minutes"] <= 60
    assert "health_gates" in steps["production_quality_control"]["depends_on"]


def test_runtime_self_awareness_refreshes_in_dependency_order() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}

    assert steps["memory_pressure_intelligence"]["max_age_minutes"] <= 15
    assert steps["autonomic_resource_governor"]["depends_on"] == ["memory_pressure_intelligence"]
    assert steps["bot_needs_intelligence"]["depends_on"] == ["training_quality_control"]
    assert set(steps["training_runtime_control"]["depends_on"]) == {
        "memory_pressure_intelligence",
        "autonomic_resource_governor",
        "training_quality_control",
        "bot_needs_intelligence",
    }
    assert "training_runtime_control" in steps["promotion_candidate_advancement"]["depends_on"]
    assert set(steps["architecture_upgrade_scoreboard"]["depends_on"]) == {
        "production_excellence",
        "training_runtime_control",
        "autonomy_control_plane",
    }
    assert set(steps["autonomy_control_plane"]["depends_on"]) == {
        "production_excellence",
        "training_runtime_control",
    }
    assert set(steps["system_needs_intelligence"]["depends_on"]) == {
        "readiness_blocker_rollup",
        "architecture_upgrade_scoreboard",
        "training_runtime_control",
        "memory_pressure_intelligence",
        "uniform_hardening_contract",
    }


def test_uniform_hardening_runs_after_critical_freshness_producers_and_before_readiness() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}

    assert set(steps["runtime_paper_regression_guard"]["depends_on"]) == {
        "paper_truth_dependency_refresh",
        "health_gates",
    }
    assert set(steps["source_verification_autorefresh"]["depends_on"]) == {
        "autonomic_resource_governor",
        "health_gates",
    }
    assert set(steps["source_verification"]["depends_on"]) == {"collector_contracts"}
    assert set(steps["capability_materialization"]["depends_on"]) == {"source_verification"}
    assert set(steps["collector_capability_control"]["depends_on"]) == {
        "collector_contracts",
        "source_verification",
        "capability_materialization",
    }
    assert set(steps["provider_mesh"]["depends_on"]) == {
        "collector_capability_control",
        "source_verification",
    }
    assert "--apply" in steps["source_verification_autorefresh"]["args"]
    assert "source_verification_autorefresh" in steps["paper_truth_dependency_refresh"]["depends_on"]
    assert "runtime_paper_regression_guard" in steps["production_quality_control"]["depends_on"]
    assert set(steps["uniform_hardening_contract"]["depends_on"]) == {
        "coherent_training_profitability_refresh",
        "production_quality_slo",
        "live_readiness_smoke",
        "runtime_paper_regression_guard",
        "source_verification_autorefresh",
        "training_runtime_control",
        "profitability_evidence_firewall",
    }
    assert "uniform_hardening_contract" in steps["production_readiness"]["depends_on"]
