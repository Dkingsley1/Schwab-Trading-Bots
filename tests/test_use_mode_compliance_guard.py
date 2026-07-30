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
