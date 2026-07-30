from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import use_mode_compliance_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_personal_ready(project_root: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    health = project_root / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"status": "ready", "ok": True, "blockers": []},
                "live_execution": {"status": "blocked_read_only", "ok": False},
            },
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "status": "ready",
                    "ok": True,
                    "child_process_count": 16,
                }
            },
            "collection": {
                "overall_status": "ready",
                "effective_bots_with_observations": 183,
                "total_observations": 12345,
            },
            "storage": {"pressure_index": 0.01},
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"timestamp_utc": now, "overall_status": "ready", "lease_state": "healthy"})
    _write_json(health / "schwab_auth_supervisor_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(
        health / "broker_readiness_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "ready_for_open": True, "auth_ok": True, "network_ok": True},
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"timestamp_utc": now, "overall_status": "ready", "failed_checks": []})
    _write_json(health / "paper_execution_truth_layer_latest.json", {"timestamp_utc": now, "overall_status": "ready", "failed_checks": []})
    _write_json(health / "paper_400_ramp_latest.json", {"timestamp_utc": now, "overall_status": "ready", "stage": "armed", "blockers": []})
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "effective_bots_with_observations": 183,
            "bots_with_observations": 183,
            "total_observations": 12345,
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"timestamp_utc": now, "overall_status": "ready", "pressure_index": 0.01})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "live_orders_blocked": True, "market_data_only": True},
    )
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "controlled_profitability_grade": "A+",
            "raw_profitability_grade": "D",
        },
    )


def _seed_operator_grade_ready(project_root: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    health = project_root / "governance" / "health"
    _write_json(
        health / "a_plus_operating_packet_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "ok": True,
            "overall_score": 100.0,
            "a_plus_ready": True,
            "lane_count": 10,
            "a_plus_lane_count": 10,
            "non_a_plus_lane_count": 0,
            "blocker_count": 0,
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "ok": True,
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
        },
    )
    _write_json(health / "autonomy_control_plane_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True, "autonomy_score": 98.5})
    _write_json(health / "storage_disaster_recovery_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "blackstart_recovery_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 2,
            "queue_depth": 125,
        },
    )
    _write_json(
        health / "live_canary_readiness_contract_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blocked_milestones": ["m01_continuous_soak_no_hard_blockers"],
            "authority_boundaries": {"live_execution_authority": False},
            "live_money_canary_milestones": [
                {"milestone_id": "m11_use_mode_and_commercial_boundary", "ready": True, "blockers": []}
            ],
        },
    )
    _write_json(
        health / "commercial_readiness_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "commercial_product_mode": "personal_only",
            "commercial_intent": False,
            "commercial_release_blocked": False,
            "grade": "A+",
            "blockers": [],
        },
    )
    _write_json(health / "security_audit_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "secret_scan_latest.json", {"timestamp_utc": now, "overall_status": "ready", "findings_count": 0})
    _write_json(health / "telemetry_redaction_canary_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})


def test_use_mode_guard_grades_clean_personal_paper_use_a_plus(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)

    payload = src.build_payload(project_root, env={})

    assert payload["overall_status"] == "ready"
    assert payload["use_mode"] == "personal"
    assert payload["personal_use"]["grade"] == "A+"
    assert payload["personal_use"]["perfect_personal_use_ready"] is True
    assert payload["personal_use"]["personal_live_money_ready"] is False
    assert payload["commercial_use"]["commercial_use_intent_detected"] is False
    assert payload["authority_boundaries"]["live_execution_authority"] is False
    assert payload["authority_boundaries"]["does_not_enable_live_execution"] is True


def test_use_mode_guard_accepts_managed_deferred_backlog_for_personal_operator_soak(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)
    now = datetime.now(timezone.utc).isoformat()
    health = project_root / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {
                    "status": "ready",
                    "ok": True,
                    "blockers": [],
                    "storage_relief_contract": {
                        "active": True,
                        "status": "managed_deferred_backlog_waiting_for_off_hours",
                        "core_pending_lines": 809,
                        "support_pending_lines": 8246,
                        "deferred_pending_lines": 15899259,
                        "total_pending_lines": 15908314,
                        "backlog_drain_status": "waiting_for_off_hours",
                    },
                },
                "live_execution": {"status": "blocked_read_only", "ok": False},
            },
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "status": "ready",
                    "ok": True,
                    "child_process_count": 16,
                }
            },
            "collection": {
                "overall_status": "ready",
                "effective_bots_with_observations": 183,
                "total_observations": 12345,
            },
            "storage": {"pressure_index": 53.028},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 53.028,
            "backpressure": {
                "core_pending_lines": 809,
                "support_pending_lines": 8246,
                "deferred_pending_lines": 15899259,
                "total_pending_lines": 15908314,
            },
            "storage": {"backlog_drain_status": "waiting_for_off_hours"},
        },
    )

    payload = src.build_payload(project_root, env={})
    storage_row = next(row for row in payload["personal_use"]["criteria"] if row["criterion_id"] == "storage_pressure_clean")

    assert payload["personal_use"]["grade"] == "A+"
    assert payload["personal_use"]["perfect_personal_use_ready"] is True
    assert storage_row["ready"] is True
    assert storage_row["evidence"]["managed_deferred_backlog_relief"]["managed"] is True


def test_use_mode_guard_promotes_clean_personal_to_operator_grade_autonomy(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)
    _seed_operator_grade_ready(project_root)
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"overall_status": "ready", "ok": True, "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"overall_status": "ready", "ok": True, "failed_checks": []})

    payload = src.build_payload(project_root, env={})

    strength = payload["personal_use"]["operator_grade_personal_autonomy"]
    assert payload["personal_use"]["grade"] == "A+"
    assert strength["ready"] is True
    assert strength["tier"] == "operator_grade_personal_autonomy"
    assert strength["score"] == 100.0
    assert strength["blockers"] == []
    assert payload["authority_boundaries"]["live_execution_authority"] is False


def test_operator_grade_personal_autonomy_blocks_without_a_plus_packet(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)
    _seed_operator_grade_ready(project_root)
    _write_json(
        project_root / "governance" / "health" / "a_plus_operating_packet_latest.json",
        {"overall_status": "needs_work", "ok": False, "a_plus_ready": False, "non_a_plus_lane_count": 1},
    )
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"overall_status": "ready", "ok": True, "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"overall_status": "ready", "ok": True, "failed_checks": []})

    payload = src.build_payload(project_root, env={})

    strength = payload["personal_use"]["operator_grade_personal_autonomy"]
    assert payload["personal_use"]["perfect_personal_use_ready"] is True
    assert strength["ready"] is False
    assert "a_plus_operating_packet_all_lanes:a_plus_operating_packet_not_ready" in strength["blockers"]
    assert "a_plus_operating_packet_all_lanes:non_a_plus_lanes=1" in strength["blockers"]


def test_use_mode_guard_blocks_customer_execution_without_review(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)

    payload = src.build_payload(
        project_root,
        env={
            "SYSTEM_USE_MODE": "commercial_software",
            "CUSTOMER_ORDER_EXECUTION_ENABLED": "1",
        },
    )

    assert payload["overall_status"] == "blocked"
    assert payload["commercial_use"]["commercial_use_intent_detected"] is True
    assert payload["commercial_use"]["commercial_clearance_status"] == "blocked_requires_compliance_review"
    assert "broker_dealer_review_not_approved" in payload["commercial_use"]["blockers"]
    assert "broker_dealer_customer_execution_review_required" in payload["commercial_use"]["blockers"]
    assert payload["authority_boundaries"]["customer_order_execution_allowed"] is False
    assert payload["authority_boundaries"]["live_execution_authority"] is False


def test_use_mode_guard_hard_blocks_customer_funds_and_custody(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)

    payload = src.build_payload(
        project_root,
        env={
            "CUSTOMER_FUNDS_ENABLED": "true",
            "CUSTODY_ENABLED": "true",
            "COMMERCIAL_LEGAL_REVIEW_APPROVED": "1",
            "COMMERCIAL_COMPLIANCE_REVIEW_APPROVED": "1",
        },
    )

    assert payload["overall_status"] == "blocked"
    assert "customer_funds_or_custody_hard_block" in payload["commercial_use"]["hard_blockers"]
    assert "customer_funds_or_custody_not_allowed_without_registered_reviewed_program" in payload["commercial_use"]["hard_blockers"]
    assert payload["authority_boundaries"]["customer_funds_allowed"] is False
    assert payload["authority_boundaries"]["custody_allowed"] is False


def test_use_mode_guard_blocks_paid_signals_and_marketing_until_reviews_exist(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_personal_ready(project_root)

    payload = src.build_payload(
        project_root,
        env={
            "PAID_SIGNALS_ENABLED": "1",
            "PERFORMANCE_MARKETING_ENABLED": "1",
        },
    )

    assert payload["overall_status"] == "blocked"
    assert "investment_adviser_review_not_approved" in payload["commercial_use"]["blockers"]
    assert "marketing_review_not_approved" in payload["commercial_use"]["blockers"]
    assert payload["authority_boundaries"]["performance_claims_allowed_without_review"] is False
