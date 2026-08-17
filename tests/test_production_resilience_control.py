import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import production_resilience_control as resilience


NOW = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed(tmp_path: Path) -> Path:
    config = json.loads(resilience.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / resilience.DEFAULT_CONFIG_PATH.name
    _write(config_path, config)
    for spec in config["sections"]:
        owner = tmp_path / spec["owner_source"]
        owner.parent.mkdir(parents=True, exist_ok=True)
        owner.write_text("# owner\n", encoding="utf-8")
    health = tmp_path / "governance" / "health"
    timestamp = NOW.isoformat()
    _write(
        health / "soak_reliability_sentinel_latest.json",
        {
            "timestamp_utc": timestamp,
            "ok": True,
            "blockers": [],
            "safety_contract": {
                "always_on_observation": True,
                "heavy_maintenance_separated": True,
                "exact_refresh_allowlist_only": True,
                "automatic_live_orders": False,
            },
            "bounded_repair": {
                "max_actions_per_cycle": 2,
                "cooldown_seconds": 300,
                "max_failures_before_circuit": 2,
                "circuit_open_seconds": 3600,
                "open_circuits": [],
            },
        },
    )
    _write(
        health / "profitability_evidence_firewall_latest.json",
        {
            "timestamp_utc": timestamp,
            "semantic_ok_scope": "control_implementation",
            "control_implementation_ready": True,
            "economic_evidence_ready": True,
            "live_promotion_ready": True,
            "promotion_evidence_ready": True,
            "control_grade": "A+",
            "economic_evidence_grade": "A+",
            "raw_profitability_grade": "A",
            "raw_profitability_grade_overridden": False,
            "grading_contract": {
                "generic_control_ok_must_not_be_interpreted_as_economic_readiness": True,
                "future_profitability_is_not_guaranteed": True,
            },
        },
    )
    _write(
        health / "control_surface_ownership_latest.json",
        {
            "timestamp_utc": timestamp,
            "ok": True,
            "control_count": 12,
            "duplicate_resource_paths": [],
            "control_contract": {
                "one_declared_writer_per_resource": True,
                "owners_are_source_backed": True,
                "mutable_automation_is_coordinated": True,
            },
        },
    )
    _write(
        health / "release_freeze_guard_latest.json",
        {
            "timestamp_utc": timestamp,
            "immutable_release_boundary": {
                "ready": True,
                "rollback_ready": True,
                "requires_clean_worktree": True,
                "requires_upstream_synchronization": True,
            },
            "git_integrity": {"ready": True},
        },
    )
    _write(
        health / "chaos_drill_coordinator_latest.json",
        {
            "timestamp_utc": timestamp,
            "ok": True,
            "required_drill_count": 10,
            "verified_drill_count": 10,
            "overdue_drills": [],
            "recovery_slo_breaches": [],
            "drill_program": {"automation_ready": True, "recovery_slo_met": True},
            "evidence_scope": {"live_execution_authority": False},
        },
    )
    _write(
        health / "live_order_ledger_control_latest.json",
        {
            "timestamp_utc": timestamp,
            "ok": True,
            "unresolved_intent_count": 0,
            "integrity": {"ok": True},
            "contract": {
                "transactional_reservation_before_submit": True,
                "ambiguous_submit_never_auto_retried": True,
                "foreign_key_integrity_required": True,
                "wal_full_sync_required": True,
                "event_state_materialization_must_match": True,
            },
        },
    )
    _write(
        health / "storage_disaster_recovery_latest.json",
        {
            "timestamp_utc": timestamp,
            "recovery_objectives": {
                "ready": True,
                "rpo": {"target_minutes": 720, "met": True},
                "rto": {"target_seconds": 30, "met": True},
                "evidence_receipt_sha256": "a" * 64,
                "paper_collection_blocked": False,
            },
        },
    )
    _write(
        health / "independent_runtime_monitor_latest.json",
        {
            "timestamp_utc": timestamp,
            "local_monitor_ready": True,
            "production_monitor_ready": True,
            "implementation_boundary": {
                "stdlib_only": True,
                "imports_trading_runtime": False,
                "runs_as_separate_launchd_process": True,
                "automatic_repairs": False,
                "automatic_orders": False,
            },
        },
    )
    return config_path


def test_all_ten_sections_produce_framework_aware_live_eligibility(tmp_path: Path) -> None:
    config = _seed(tmp_path)

    payload = resilience.build_payload(tmp_path, config_path=config, now=NOW)

    assert payload["framework_awareness_ready"] is True
    assert payload["implementation_grade"] == "A+"
    assert payload["paper_soak_ready"] is True
    assert payload["live_promotion_ready"] is True
    assert payload["live_promotion_readiness_percent"] == 100.0
    assert payload["authority_contract"]["live_execution_authority"] is False


def test_dirty_release_blocks_live_promotion_without_interrupting_paper_soak(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    path = tmp_path / "governance" / "health" / "release_freeze_guard_latest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["immutable_release_boundary"]["ready"] = False
    payload["git_integrity"]["ready"] = False
    _write(path, payload)

    result = resilience.build_payload(tmp_path, config_path=config, now=NOW)

    assert result["paper_soak_ready"] is True
    assert result["live_promotion_ready"] is False
    assert result["live_promotion_readiness_percent"] == 90.0


def test_local_deadman_failure_blocks_unattended_paper_claim(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    path = tmp_path / "governance" / "health" / "independent_runtime_monitor_latest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["local_monitor_ready"] = False
    payload["production_monitor_ready"] = False
    _write(path, payload)

    result = resilience.build_payload(tmp_path, config_path=config, now=NOW)

    assert result["paper_soak_ready"] is False
    assert result["overall_status"] == "blocked"


def test_profitability_evidence_debt_is_honest_and_does_not_stop_collection(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    path = tmp_path / "governance" / "health" / "profitability_evidence_firewall_latest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["economic_evidence_ready"] = False
    payload["live_promotion_ready"] = False
    payload["promotion_evidence_ready"] = False
    payload["economic_evidence_grade"] = "D"
    _write(path, payload)

    result = resilience.build_payload(tmp_path, config_path=config, now=NOW)

    assert result["paper_soak_ready"] is True
    assert result["live_promotion_ready"] is False
    section = next(row for row in result["sections"] if row["section_id"].startswith("10_"))
    assert section["details"]["economic_evidence_grade"] == "D"
