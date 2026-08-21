import json
from pathlib import Path

from scripts.ops import sleeve_ingestion_production_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_artifacts(
    project_root: Path,
    *,
    backlog_class: str = "managed_deferred_backlog_with_stale_support_tail",
    backlog_active: bool = True,
    safe_to_auto_apply: bool = True,
    paper_allowed: bool = True,
    live_blocked: bool = True,
    collector_count: int = 183,
    observed_count: int = 183,
    unmanaged_zero: int = 0,
    direct_execution: int = 0,
    live_trading: int = 0,
    allow_order_execution: str = "0",
    market_data_only: str = "1",
    live_execution_allowed: bool = False,
    write_coverage: bool = True,
) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "overall_status": "ready",
            "collector_count": collector_count,
            "effective_bots_with_observations": observed_count,
            "bots_with_observations": observed_count,
            "unmanaged_zero_observation_count": unmanaged_zero,
            "zero_observation_count": unmanaged_zero,
            "total_observations": 242963,
            "training_ready_count": 4,
            "collection_coverage_score": 100.0 if unmanaged_zero == 0 else 92.0,
            "data_quality_score": 100.0,
            "zero_observation_repair_lane": {"active": unmanaged_zero > 0},
        },
    )
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "counts_after": {
                "non_deleted_bots": 1742,
                "data_collection_active_bots": 1742,
                "paper_live_data_enabled_bots": 1584,
                "collection_until_standard_bots": 158,
                "direct_execution_allowed_bots": direct_execution,
                "live_trading_enabled_bots": live_trading,
            },
            "safety_contract": {
                "paper_trade_lock": "1",
                "market_data_only": market_data_only,
                "allow_order_execution": allow_order_execution,
                "live_execution_allowed": live_execution_allowed,
            },
        },
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "operational_readiness": {
                "guarded_paper": {"status": "ready", "blockers": [], "paper_ramp_stage": "armed"}
            },
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "status": "ready",
                    "ok": True,
                    "child_process_count": 42,
                    "child_fanout_ok": True,
                    "heartbeat_ok": True,
                }
            },
        },
    )
    _write_json(
        health / "ingestion_priority_queue_latest.json",
        {
            "queue_depth": 20,
            "items_synced": 20,
            "lane_counts": {
                "core": {"pending_lines": 38667, "adaptive_quota_share": 0.8},
                "deferred": {"pending_lines": 15863698, "adaptive_quota_share": 0.2},
                "cold": {"pending_lines": 0, "adaptive_quota_share": 0.0},
            },
            "dispatch_plan": [{"source_rel": "decisions/a.jsonl"}],
        },
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {
            "high_backlog_control": {
                "active": backlog_active,
                "class": backlog_class,
                "severity": "watch" if backlog_active else "stable",
                "production_grade_contract": {
                    "state": "production_owned" if safe_to_auto_apply else "needs_operator",
                    "grade": "A+" if safe_to_auto_apply else "D",
                    "score": 100 if safe_to_auto_apply else 62,
                    "missing": [] if safe_to_auto_apply else ["safe_to_auto_apply"],
                },
                "automation_contract": {
                    "safe_to_auto_apply": safe_to_auto_apply,
                    "repair_plan_names": ["defer_support_tail"] if safe_to_auto_apply else [],
                },
                "paper_soak_boundary": {"allowed_with_advisory": paper_allowed},
                "live_money_boundary": {"blocked": live_blocked},
                "next_system_action": "continue managed off-hours drain",
            }
        },
    )
    if write_coverage:
        _write_json(
            health / "sleeve_strategy_coverage_latest.json",
            {
                "overall_status": "ready",
                "ok": True,
                "sleeve_count": 48,
                "active_runtime_sleeve_count": 48,
                "strategy_count": 48,
                "missing_runtime_sleeves": [],
                "strategy_covered_needs_launcher": [],
            },
        )
    _write_json(
        health / "collector_capability_control_latest.json",
        {
            "overall_status": "ready_with_coverage_debt",
            "ok": True,
            "paper_soak_ready": True,
            "live_promotion_ready": False,
            "routing_receipt_sha256": "route-receipt",
            "summary": {
                "assignment_count": 1781,
                "bot_binding_count": 1781,
            },
            "ingestion_routing_contract": {
                "policy_id": "sleeve_ingestion_routing_v2",
                "policy_receipt_sha256": "policy-receipt",
                "routing_artifact_receipt_sha256": "route-receipt",
                "decision_policy_id": "institutional_decision_flow_sleeve_playbooks_v4",
                "decision_stage": "02_data_qualification",
                "decision_family_count": 15,
                "runtime_route_count": 25,
                "runtime_paper_ready_route_count": 20,
                "runtime_live_ready_route_count": 4,
                "average_profile_route_quality": 0.91,
                "paper_data_debt_blocks_global_collection": False,
                "live_data_debt_blocks_candidate_promotion": True,
                "transport_contract": {
                    "idempotency_required": True,
                    "payload_digest_required": True,
                    "source_timestamp_required": True,
                    "bounded_response_size_required": True,
                    "retry_only_transient_failures": True,
                    "respect_retry_after": True,
                    "redact_query_parameters_from_receipts": True,
                    "watermark_on_success": True,
                    "dead_letter_after_retry_exhaustion": True,
                },
            },
            "ingestion_authority_contract": {
                "changes_strategy_signal": False,
                "launches_collectors": False,
                "fetches_external_data": False,
                "mutates_bot_registry": False,
                "paper_execution_authority": False,
                "live_execution_authority": False,
                "automatic_promotion_authority": False,
                "profitability_guaranteed": False,
            },
        },
    )


def test_managed_deferred_backlog_gets_a_plus_manifest_first_control(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    override_path = tmp_path / "config" / ".env.sleeve_ingestion_production_override"

    payload = src.build_payload(tmp_path, apply=True, override_path=override_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["production_grade_contract"]["grade"] == "A+"
    assert payload["production_grade_contract"]["missing"] == []
    assert payload["ingestion_mode_contract"]["mode"] == "production_owned_manifest_first"
    assert payload["ingestion_mode_contract"]["max_active_ratio"] == 0.16
    assert payload["ingestion_mode_contract"]["paper_soak_allowed"] is True
    assert payload["ingestion_mode_contract"]["live_money_blocked"] is True
    assert payload["data_tier_contract"]["deferred_budget"] == "0"
    assert "idempotency_key" in payload["sleeve_event_envelope_contract"]["required_fields"]
    assert "payload_digest" in payload["sleeve_event_envelope_contract"]["required_fields"]
    assert "ingestion_route_receipt_sha256" in payload["sleeve_event_envelope_contract"]["required_fields"]
    assert payload["decision_aligned_routing_contract"]["policy_id"] == "sleeve_ingestion_routing_v2"
    assert payload["decision_aligned_routing_contract"]["all_bots_route_bound"] is True

    text = override_path.read_text(encoding="utf-8")
    assert "SLEEVE_INGESTION_MODE=production_owned_manifest_first" in text
    assert "SLEEVE_INGESTION_MAX_ACTIVE_RATIO=0.16" in text
    assert "SLEEVE_INGESTION_EVENT_ENVELOPE_REQUIRED=1" in text
    assert "SLEEVE_INGESTION_IDEMPOTENCY_REQUIRED=1" in text
    assert "SLEEVE_INGESTION_ROUTE_RECEIPT_REQUIRED=1" in text
    assert "SLEEVE_INGESTION_ROUTE_ENFORCEMENT=1" in text
    assert "SLEEVE_INGESTION_ROUTE_MAX_AGE_MINUTES=30" in text
    assert "SLEEVE_INGESTION_ROUTING_POLICY_ID=sleeve_ingestion_routing_v2" in text
    assert "MARKET_DATA_ONLY=1" in text
    assert "ALLOW_ORDER_EXECUTION=0" in text
    assert payload["source_freshness_contract"]["all_required_fresh"] is True


def test_hot_core_backlog_forces_low_duty_manifest_first_mode(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, backlog_class="hot_path_backpressure", paper_allowed=False)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["production_grade_contract"]["grade"] == "A+"
    assert payload["ingestion_mode_contract"]["mode"] == "hot_path_recovery_manifest_first"
    assert payload["ingestion_mode_contract"]["max_active_ratio"] == 0.1
    assert payload["ingestion_mode_contract"]["paper_soak_allowed"] is False
    assert payload["ingestion_mode_contract"]["live_money_blocked"] is True
    assert payload["control_env_recommendations"]["HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG"] == "1"


def test_unmanaged_zero_observations_route_to_targeted_repair(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, observed_count=181, unmanaged_zero=2)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["ingestion_mode_contract"]["mode"] == "targeted_observation_repair"
    assert payload["ingestion_mode_contract"]["max_active_ratio"] == 0.08
    assert "observation_coverage_ready" in payload["production_grade_contract"]["missing"]
    assert "zero_observation_repair_clear" in payload["production_grade_contract"]["missing"]


def test_missing_sleeve_coverage_artifact_blocks_production_claim(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, write_coverage=False)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["production_grade_contract"]["grade"] == "B+"
    assert "sleeve_coverage_ready" in payload["production_grade_contract"]["missing"]
    assert "source_artifacts_fresh" in payload["production_grade_contract"]["missing"]
    assert payload["source_freshness_contract"]["stale_or_missing"] == ["sleeve_coverage"]


def test_live_execution_boundary_drift_blocks_sleeve_ingestion(tmp_path: Path) -> None:
    _write_artifacts(
        tmp_path,
        direct_execution=1,
        live_trading=1,
        allow_order_execution="1",
        market_data_only="0",
        live_execution_allowed=True,
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["production_grade_contract"]["grade"] == "D"
    assert payload["ingestion_mode_contract"]["mode"] == "blocked_live_execution_boundary"
    assert payload["ingestion_mode_contract"]["max_active_ratio"] == 0
    assert "live_execution_locked" in payload["production_grade_contract"]["missing"]
