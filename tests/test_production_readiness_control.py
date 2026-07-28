from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import production_readiness_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_minimal_project(project_root: Path) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        project_root / "config" / "library_candidate_routes_v1.json",
        {
            "candidate_libraries": [
                {"package": "cachetools", "lane": "provider_rate_limit_cache", "runtime_family": "python"},
            ]
        },
    )
    _write_json(
        project_root / "config" / "library_activation_profiles_v1.json",
        {
            "profile_order": ["live", "ops", "research", "media"],
            "profile_lanes": {"live": ["provider_rate_limit_cache"]},
            "initial_activation_batches": {"production_core_safe": ["cachetools"]},
        },
    )
    _write_text(project_root / "config" / "requirements.lock.txt", "cachetools==6.2.0\n")
    _write_text(project_root / "config" / ".env.library_utilization_router_override")
    for profile in ("live", "ops", "research"):
        _write_text(project_root / "config" / "runtime_profiles" / f"{profile}.lock.txt")
    _write_text(project_root / "governance" / "health" / "PAPER_TRADE_LOCK.flag")
    for script_path in (
        "scripts/ops/dependency_activation_smoke.py",
        "scripts/ops/library_utilization_router.py",
        "scripts/ops/mlx_intelligence_router.py",
        "scripts/ops/promotion_pipeline.py",
        "scripts/secret_scan.py",
    ):
        _write_text(project_root / script_path)
    for artifact_path in (
        "governance/health/library_utilization_router_latest.json",
        "governance/health/mlx_intelligence_router_latest.json",
        "governance/health/live_readiness_smoke_latest.json",
        "governance/health/promotion_quality_gate_latest.json",
        "governance/health/golden_replay_regression_latest.json",
        "governance/health/shadow_replay_diff_latest.json",
        "governance/health/replay_hash_registry_guard_latest.json",
        "governance/health/point_in_time_event_store_latest.json",
        "governance/health/retrain_schema_compatibility_latest.json",
        "governance/health/jsonl_sql_ingestion_health_schema_violations_latest.json",
        "governance/health/provider_mesh_latest.json",
        "governance/health/artifact_freshness_slo_latest.json",
        "governance/health/production_quality_slo_guard_latest.json",
        "governance/health/live_reconciliation_slo_latest.json",
        "governance/health/paper_reconciliation_slo_latest.json",
        "governance/health/backpressure_slo_bot_latest.json",
        "governance/health/incident_review_packet_latest.json",
    ):
        _write_json(project_root / artifact_path, {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "health" / "remote_alert_control_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "content_store" / "latest.json", {"timestamp_utc": now, "ok": True, "artifact_count": 3, "skipped_blob_count": 0})
    _write_json(project_root / "governance" / "health" / "source_verification_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "health" / "security_audit_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "health" / "secret_scan_latest.json", {"timestamp_utc": now, "findings_count": 0})
    _write_json(project_root / "governance" / "health" / "auth_lease_manager_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "health" / "storage_disaster_recovery_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(
        project_root / "governance" / "health" / "blackstart_recovery_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "ok": True,
            "production_grade_ready": True,
            "blocked_stage_count": 0,
        },
    )
    _write_json(
        project_root / "governance" / "health" / "paper_execution_truth_layer_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "ok": True,
            "a_plus_ready": True,
            "blocked_gates": 0,
        },
    )
    _write_json(
        project_root / "governance" / "health" / "telemetry_redaction_canary_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        project_root / "governance" / "health" / "live_canary_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "canary_weight_ok": True,
            "live_lane_should_be_read_only": True,
            "live_money_contract_enforced": True,
            "live_money_contract_hard_block": True,
        },
    )
    config_path = project_root / "config" / "production_readiness_control_v1.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "control_policy": "test_policy",
            "default_dependency_activation_batch": "production_core_safe",
            "live_execution_risk_firewall": {
                "market_data_only_default": True,
                "allow_order_execution_env": "ALLOW_ORDER_EXECUTION",
                "market_data_only_env": "MARKET_DATA_ONLY",
                "max_single_order_notional": 1000.0,
                "max_daily_loss": 250.0,
                "max_order_quantity": 100,
                "max_quote_age_seconds": 15.0,
                "max_spread_bps": 75.0,
                "required_safety_flags": ["governance/health/PAPER_TRADE_LOCK.flag"],
                "halt_flags": ["governance/health/OPERATOR_STOP.flag", "governance/health/GLOBAL_TRADING_HALT.flag"],
                "order_intent_path": "governance/health/live_order_intents_latest.json",
            },
            "deterministic_replay": {
                "required_artifacts": [
                    "governance/health/golden_replay_regression_latest.json",
                    "governance/health/shadow_replay_diff_latest.json",
                    "governance/health/replay_hash_registry_guard_latest.json",
                ],
                "fingerprint_paths": [
                    "config/requirements.lock.txt",
                    "config/library_candidate_routes_v1.json",
                    "config/library_activation_profiles_v1.json",
                    "config/production_readiness_control_v1.json",
                ],
                "baseline_path": "governance/health/production_readiness_replay_baseline.json",
            },
            "observability_redaction": {
                "redaction_patterns": [
                    "(?i)bearer\\s+[a-z0-9._-]+",
                    "(?i)(account_id|order_id)\\s*[:=]\\s*['\\\"]?[a-z0-9-]{6,}",
                ],
                "redaction_samples": [
                    {"name": "token", "input": "Authorization: Bearer abc123.private", "must_not_contain": ["abc123.private"]},
                    {"name": "account", "input": "account_id=123456789", "must_not_contain": ["123456789"]},
                ],
            },
            "release_gates": {
                "required_scripts": [
                    "scripts/ops/dependency_activation_smoke.py",
                    "scripts/ops/library_utilization_router.py",
                    "scripts/ops/mlx_intelligence_router.py",
                    "scripts/ops/promotion_pipeline.py",
                    "scripts/secret_scan.py",
                ],
                "required_artifacts": [
                    "governance/health/library_utilization_router_latest.json",
                    "governance/health/mlx_intelligence_router_latest.json",
                    "governance/health/live_readiness_smoke_latest.json",
                    "governance/health/promotion_quality_gate_latest.json",
                ],
            },
            "data_integrity_gates": {
                "required_artifacts": [
                    "governance/health/point_in_time_event_store_latest.json",
                    "governance/health/retrain_schema_compatibility_latest.json",
                    "governance/health/jsonl_sql_ingestion_health_schema_violations_latest.json",
                    "governance/health/provider_mesh_latest.json",
                    "governance/health/artifact_freshness_slo_latest.json",
                ]
            },
            "incident_rollback": {
                "snapshot_paths": [
                    "config/requirements.lock.txt",
                    "config/library_candidate_routes_v1.json",
                    "config/library_activation_profiles_v1.json",
                    "config/production_readiness_control_v1.json",
                ],
                "promotion_packet_paths": ["governance/health/incident_review_packet_latest.json"],
                "rollback_manifest_path": "governance/rollback/production_rollback_manifest_latest.json",
                "rollback_commands": ["./scripts/ops/opsctl.sh global-halt-status --json"],
            },
            "slo_error_budget": {
                "target_success_ratio": 0.999,
                "required_artifacts": [
                    "governance/health/production_quality_slo_guard_latest.json",
                    "governance/health/live_reconciliation_slo_latest.json",
                    "governance/health/paper_reconciliation_slo_latest.json",
                    "governance/health/backpressure_slo_bot_latest.json",
                    "governance/health/artifact_freshness_slo_latest.json",
                ],
            },
            "live_money_production_bar": {
                "enabled": True,
                "require_for_live_canary": True,
                "require_read_only_pre_canary": True,
                "required_domain_statuses": {
                    "dependency_activation_smoke_runner": ["ready"],
                    "live_execution_risk_firewall": ["ready_guarded"],
                    "deterministic_replay_harness": ["ready"],
                    "observability_redaction": ["ready"],
                    "release_gates": ["ready"],
                    "data_integrity_gates": ["ready"],
                    "incident_and_rollback_system": ["ready"],
                    "slo_error_budget_policy": ["ready"],
                },
                "required_capabilities": [
                    {
                        "capability_id": "external_alert_supervision",
                        "artifact": "governance/health/remote_alert_control_latest.json",
                        "ready_statuses": ["ready"],
                        "max_age_hours": 24,
                        "truthy_keys": ["ok"],
                    },
                    {
                        "capability_id": "immutable_evidence_store",
                        "artifact": "governance/content_store/latest.json",
                        "ready_statuses": ["ok", "ready"],
                        "max_age_hours": 24,
                        "truthy_keys": ["ok"],
                        "max_count_by_key": {"skipped_blob_count": 0},
                    },
                    {
                        "capability_id": "secret_scan_zero_findings",
                        "artifact": "governance/health/secret_scan_latest.json",
                        "ready_statuses": ["ready", "ok"],
                        "max_age_hours": 24,
                        "zero_count_keys": ["findings_count"],
                    },
                    {
                        "capability_id": "blackstart_recovery",
                        "artifact": "governance/health/blackstart_recovery_latest.json",
                        "ready_statuses": ["ready"],
                        "max_age_hours": 24,
                        "truthy_keys": ["ok", "production_grade_ready"],
                        "zero_count_keys": ["blocked_stage_count"],
                    },
                    {
                        "capability_id": "live_canary_governor_read_only_cap",
                        "artifact": "governance/health/live_canary_control_latest.json",
                        "ready_statuses": ["blocked", "ready", "guarded"],
                        "max_age_hours": 24,
                        "truthy_keys": [
                            "canary_weight_ok",
                            "live_lane_should_be_read_only",
                            "live_money_contract_enforced",
                            "live_money_contract_hard_block",
                        ],
                    },
                ],
            },
        },
    )
    return config_path


def test_production_readiness_control_covers_all_domains_and_keeps_live_guarded(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _seed_minimal_project(project_root)

    payload = src.build_payload(
        project_root,
        config_path=config_path,
        installed_versions={"cachetools": "6.2.0"},
        env={"ALLOW_ORDER_EXECUTION": "0"},
    )
    domains = {row["name"]: row for row in payload["domains"]}

    assert payload["domain_count"] == 9
    assert payload["blocked_domain_count"] == 0
    assert payload["overall_status"] == "guarded"
    assert payload["live_runtime_promotion_allowed"] is False
    assert payload["live_money_production_bar_ready"] is True
    assert payload["live_money_canary_consideration_ready"] is True
    assert payload["control_contract"]["covers_controls_1_through_8"] is True
    assert payload["control_contract"]["covers_live_money_production_bar"] is True
    assert domains["dependency_activation_smoke_runner"]["status"] == "ready"
    assert domains["live_execution_risk_firewall"]["status"] == "ready_guarded"
    assert domains["live_money_production_bar"]["status"] == "ready"
    assert domains["observability_redaction"]["status"] == "ready"
    assert domains["incident_and_rollback_system"]["status"] == "ready"


def test_production_readiness_control_blocks_unsafe_live_order_intent(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _seed_minimal_project(project_root)
    order_path = project_root / "governance" / "health" / "live_order_intents_latest.json"
    _write_json(
        order_path,
        {
            "orders": [
                {
                    "symbol": "SPY",
                    "side": "BUY",
                    "quantity": 250,
                    "limit_price": 500.0,
                    "quote_age_seconds": 60.0,
                    "spread_bps": 120.0,
                }
            ]
        },
    )

    payload = src.build_payload(
        project_root,
        config_path=config_path,
        order_intents_path=order_path,
        installed_versions={"cachetools": "6.2.0"},
        env={"ALLOW_ORDER_EXECUTION": "1"},
    )
    firewall = {row["name"]: row for row in payload["domains"]}["live_execution_risk_firewall"]

    assert payload["overall_status"] == "blocked"
    assert firewall["status"] == "blocked"
    assert "live_execution_risk_firewall:order_0:notional_exceeds_cap" in payload["blockers"]
    assert "quote_is_stale" in firewall["evidence"]["checked_orders"][0]["reasons"]


def test_production_readiness_control_blocks_missing_live_money_capability(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = _seed_minimal_project(project_root)
    (project_root / "governance" / "content_store" / "latest.json").unlink()

    payload = src.build_payload(
        project_root,
        config_path=config_path,
        installed_versions={"cachetools": "6.2.0"},
        env={"ALLOW_ORDER_EXECUTION": "0"},
    )
    production_bar = {row["name"]: row for row in payload["domains"]}["live_money_production_bar"]

    assert payload["overall_status"] == "blocked"
    assert payload["live_money_production_bar_ready"] is False
    assert payload["live_money_canary_consideration_ready"] is False
    assert production_bar["status"] == "blocked"
    assert "live_money_production_bar:immutable_evidence_store:immutable_evidence_store_missing" in payload["blockers"]
