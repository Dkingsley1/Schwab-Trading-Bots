import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import live_feed_status_contract as contract


NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
STAMP = NOW.isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _ready_fixture(project_root: Path) -> Path:
    health = project_root / "governance" / "health"
    _write(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "strict_all_clear": True,
            "repair_backlog_active": False,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "ok": True,
                    "status": "ready",
                    "child_process_count": 14,
                    "child_fanout_ok": True,
                }
            },
            "collection": {
                "collector_count": 183,
                "effective_bots_with_observations": 183,
                "unmanaged_zero_observation_count": 0,
                "total_observations": 400000,
            },
        },
    )
    _write(
        health / "broker_readiness_latest.json",
        {
            "timestamp_utc": STAMP,
            "ready_for_open": True,
            "network_ok": True,
            "auth_ok": True,
            "token_expires_in_seconds": 1700,
            "preflight_checks": {"refresh_needed_after": False},
            "warnings": [],
        },
    )
    _write(
        health / "auth_lease_manager_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "lease_state": "healthy",
            "broker_state": {
                "broker_ready": True,
                "network_ok": True,
                "auth_ok": True,
                "auth_probe_ok": True,
            },
            "lease_budget": {"expires_in_seconds": 1700, "probe_backed": True},
        },
    )
    _write(
        health / "schwab_auth_supervisor_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "token": {
                "ready": True,
                "refresh_needed": False,
                "expires_in_seconds": 1700,
                "age_seconds": 100,
                "min_expires_seconds": 1500,
                "min_ready_expires_seconds": 900,
            },
        },
    )
    _write(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.1,
            "pending_lines_threshold": 15000,
            "backpressure": {
                "core_pending_lines": 80,
                "total_pending_lines": 100,
                "oldest_pending_age_seconds": 10,
                "raw_live": {"core_pending_lines": 80, "total_pending_lines": 100},
            },
            "storage": {"backlog_drain_status": "idle"},
        },
    )
    _write(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "throttle_profile": "soft_cap",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "host_saturation_score": 20,
            "runtime_saturation_governor_v2": {
                "collector_policy": {"mode": "sustain"},
                "paper_live_data_policy": {
                    "paper_execution_allowed": True,
                    "paper_execution_consumer_paused": False,
                },
            },
        },
    )
    _write(health / "paper_400_ramp_latest.json", {"timestamp_utc": STAMP, "overall_status": "ready", "ok": True, "stage": "armed"})
    _write(
        health / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "overall_grade": "A+",
            "overall_score": 100,
            "safe_to_leave_unattended": True,
        },
    )
    return health


def test_ready_contract_reports_fresh_consistent_runtime(tmp_path: Path) -> None:
    _ready_fixture(tmp_path)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)

    assert payload["overall_status"] == "ready"
    assert payload["schema_version"] == 3
    assert payload["visibility_status"] == "ready"
    assert payload["operational_status"] == "ready"
    assert payload["headline_status"] == "ready"
    assert payload["guarded_paper_status"] == "ready"
    assert payload["safe_to_leave_unattended"] is True
    assert payload["operator_summary"]["root_cause"] == "none"
    assert payload["operator_summary"]["next_action"] == "none"
    assert payload["operator_summary"]["paper_impact"] == "none"
    assert payload["active_operational_rows"] == {}
    assert payload["managed_operational_watches"] == []
    assert payload["fresh_source_count"] == 8
    assert payload["source_count"] == 9
    assert payload["rows"]["production_excellence"]["status"] == "missing"
    assert payload["rows"]["production_excellence"]["paper_impact"] == "none"
    assert payload["contradictions"] == []
    assert payload["rows"]["auth"]["status"] == "ready"
    assert payload["rows"]["auth"]["consistency"] == "consistent"
    assert payload["rows"]["collection"]["sleeve_children"] == 14
    assert payload["rows"]["storage"]["strict_status"] == "ready"
    assert payload["rows"]["throttle"]["cause"] == "none"
    assert payload["rows"]["system"]["action"] == "none"
    assert payload["rows"]["throttle"]["action"] == "none"
    assert payload["rows"]["soak"]["effective_safe"] is True
    assert payload["rows"]["soak"]["action"] == "none"
    assert payload["rows"]["soak"]["warning_count"] == 0


def test_managed_throttle_advisory_is_visible_without_false_remediation(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    throttle = json.loads((health / "runtime_throttle_control_latest.json").read_text(encoding="utf-8"))
    throttle["overall_status"] = "advisory"
    throttle["soft_cap_advisory_reclassification"] = {
        "active": True,
        "to_status": "advisory",
        "reason": "full_force_paper_and_research_pressure_is_soak_guarded_advisory",
    }
    _write(health / "runtime_throttle_control_latest.json", throttle)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    row = payload["rows"]["throttle"]
    lines = contract.format_status_lines(payload)

    assert row["status"] == "advisory"
    assert row["managed_advisory"] is True
    assert row["policy_reason"] == "full_force_paper_and_research_pressure_is_soak_guarded_advisory"
    assert row["action"] == "none"
    assert any("[throttle] level=watch status=advisory" in line for line in lines)
    assert any("managed=true" in line and "action=none" in line for line in lines)


def test_managed_high_compute_reports_owner_and_zero_paper_impact(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    throttle = json.loads((health / "runtime_throttle_control_latest.json").read_text(encoding="utf-8"))
    throttle.update(
        {
            "overall_status": "advisory",
            "compute_pressure_level": "high",
            "host_saturation_score": 82.0,
            "soft_cap_advisory_reclassification": {
                "active": True,
                "to_status": "advisory",
                "reason": "research_training_pressure_is_already_niced_and_guarded_advisory",
            },
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
            },
            "host_pressure_attribution": {
                "dominant_bucket": "bot_owned",
                "external_pressure_dominant": True,
                "research_hot_low_priority": True,
                "storage_writer_hot": True,
            },
        }
    )
    _write(health / "runtime_throttle_control_latest.json", throttle)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    row = payload["rows"]["throttle"]
    lines = contract.format_status_lines(payload)

    assert payload["operational_status"] == "ready"
    assert payload["active_operational_rows"] == {}
    assert payload["managed_operational_watches"] == ["throttle"]
    assert row["cause"] == "managed_compute_pressure"
    assert row["managed_control"] is True
    assert row["paper_state"] == "allowed"
    assert row["impact"] == "none"
    assert row["pressure_owner"] == "external"
    assert row["action"] == "none"
    assert any(
        "[throttle] level=watch status=advisory" in line
        and "cause=managed_compute_pressure" in line
        and "impact=none" in line
        and "owner=external" in line
        for line in lines
    )


def test_fx_auth_cooldown_with_context_fallback_is_managed_and_visible(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    cooldown_until = NOW.timestamp() + 3600
    cooldown = {
        "timestamp_utc": STAMP,
        "active": True,
        "kind": "auth",
        "cooldown_until_ts": cooldown_until,
        "credential_action_required": True,
    }
    _write(
        health / "fx_shadow_session_latest.json",
        {
            "timestamp_utc": STAMP,
            "mode": "forex_session_context_only",
            "session": {
                "provider": {
                    "enabled": True,
                    "available": False,
                    "reason": "provider_cooldown:auth",
                    "cooldown": cooldown,
                }
            },
        },
    )
    _write(
        health / "data_ingress_latest_fx_equities_schwab.json",
        {
            "timestamp_utc": STAMP,
            "loop_state": "forex_session_context_only",
            "iter_total_requests": 0,
            "iter_error_rate": 0.0,
        },
    )
    _write(health / "fx_twelve_data_guard_latest.json", cooldown)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    row = payload["rows"]["fx_provider"]
    lines = contract.format_status_lines(payload)

    assert payload["headline_status"] == "ready"
    assert payload["safe_to_leave_unattended"] is True
    assert payload["managed_operational_watches"] == ["fx_provider"]
    assert row["status"] == "watch"
    assert row["managed_fallback"] is True
    assert row["impact"] == "fx_realtime_deferred"
    assert row["action"] == "renew-twelve-data-key"
    assert any(
        "[fx-provider] level=watch status=watch" in line
        and "fallback=true" in line
        and "credential=true" in line
        and "impact=fx_realtime_deferred" in line
        for line in lines
    )


def test_fx_realtime_error_storm_without_fallback_blocks_headline(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    _write(
        health / "fx_shadow_session_latest.json",
        {
            "timestamp_utc": STAMP,
            "mode": "live_forex_quotes",
            "session": {"provider": {"enabled": True, "available": True, "reason": "available"}},
        },
    )
    _write(
        health / "data_ingress_latest_fx_equities_schwab.json",
        {
            "timestamp_utc": STAMP,
            "loop_state": "running",
            "iter_total_requests": 6,
            "iter_error_rate": 1.0,
        },
    )

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    headline = contract.format_status_lines(payload)[0]

    assert payload["headline_status"] == "blocked"
    assert payload["safe_to_leave_unattended"] is False
    assert payload["active_operational_rows"]["fx_provider"] == "blocked"
    assert payload["operator_summary"]["attention_owner"] == "fx_provider"
    assert payload["operator_summary"]["root_cause"] == "fx_realtime_ingestion_error_rate_high"
    assert payload["operator_summary"]["domain_impact"] == "fx_observations_failed"
    assert payload["operator_summary"]["next_action"] == "fx-provider-fallback"
    assert "impact=fx_observations_failed" in headline
    assert "paper_impact=none" in headline


def test_bounded_storage_watch_remains_safe_for_paper_soak_without_false_contradiction(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    storage_path = health / "ingestion_storage_control_latest.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage.update({"pressure_index": 0.507, "severity": "stable"})
    storage["backpressure"] = {
        "core_pending_lines": 829,
        "total_pending_lines": 1866,
        "oldest_pending_age_seconds": 121,
        "raw_live": {"core_pending_lines": 829, "total_pending_lines": 1866},
    }
    storage["storage"] = {"backlog_drain_status": "drain_active"}
    _write(storage_path, storage)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    lines = contract.format_status_lines(payload)

    assert payload["overall_status"] == "ready"
    assert payload["contradictions"] == []
    assert payload["rows"]["storage"]["status"] == "watch"
    assert payload["rows"]["storage"]["managed_bounded_backlog"] is True
    assert payload["rows"]["throttle"]["cause"] == "bounded_storage_watch"
    assert payload["rows"]["throttle"]["recovery"] == "storage_drain_active"
    assert payload["rows"]["throttle"]["managed_control"] is True
    assert payload["rows"]["soak"]["status"] == "ready"
    assert payload["rows"]["soak"]["effective_safe"] is True
    assert payload["rows"]["soak"]["managed_storage_watch"] is True
    assert payload["operational_status"] == "ready"
    assert payload["managed_operational_watches"] == ["storage"]
    assert any("[storage] level=watch status=watch" in line and "managed=true" in line for line in lines)
    assert any("[soak] level=ok status=ready" in line and "watch=bounded_backlog" in line for line in lines)


def test_auth_reconciles_superseded_pre_refresh_warning(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    broker = json.loads((health / "broker_readiness_latest.json").read_text(encoding="utf-8"))
    broker["warnings"] = ["token_expiring_soon:1400.0"]
    _write(health / "broker_readiness_latest.json", broker)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    auth = payload["rows"]["auth"]

    assert auth["status"] == "ready"
    assert auth["consistency"] == "reconciled"
    assert auth["active_warning_count"] == 0
    assert auth["superseded_warning_count"] == 1


def test_auth_blocks_on_fresh_source_conflict(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    broker = json.loads((health / "broker_readiness_latest.json").read_text(encoding="utf-8"))
    broker["ready_for_open"] = False
    broker["auth_ok"] = False
    _write(health / "broker_readiness_latest.json", broker)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)

    assert payload["rows"]["auth"]["status"] == "blocked"
    assert payload["operational_status"] == "blocked"
    assert payload["rows"]["auth"]["consistency"] == "conflict"
    assert "auth_sources_disagree" in payload["contradictions"]


def test_fresh_paper_ramp_disagreement_blocks_walkaway_headline(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    _write(
        health / "paper_400_ramp_latest.json",
        {
            "timestamp_utc": STAMP,
            "ok": False,
            "armed": False,
            "stage": "blocked",
            "blockers": ["runtime_capacity_not_ready_for_400_paper"],
        },
    )

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    headline = contract.format_status_lines(payload)[0]

    assert payload["visibility_status"] == "degraded"
    assert payload["operational_status"] == "blocked"
    assert payload["headline_status"] == "blocked"
    assert payload["guarded_paper_status"] == "blocked"
    assert payload["safe_to_leave_unattended"] is False
    assert payload["rows"]["paper_ramp"]["sources_disagree"] is True
    assert "paper_sources_disagree" in payload["contradictions"]
    assert payload["operator_summary"]["attention_owner"] == "paper_ramp"
    assert payload["operator_summary"]["next_action"] == "paper-400-ramp"
    assert "level=alert" in headline
    assert "paper=blocked" in headline
    assert "walkaway=false" in headline


def test_runtime_paper_pause_overrides_ready_health_and_ramp(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    throttle_path = health / "runtime_throttle_control_latest.json"
    throttle = json.loads(throttle_path.read_text(encoding="utf-8"))
    throttle["overall_status"] = "blocked"
    throttle["runtime_saturation_governor_v2"]["paper_live_data_policy"] = {
        "paper_execution_allowed": False,
        "paper_execution_consumer_paused": True,
        "paper_execution_pause_reason": "paper_execution_cpu_pressure",
    }
    _write(throttle_path, throttle)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    headline = contract.format_status_lines(payload)[0]

    assert payload["guarded_paper_status"] == "blocked"
    assert payload["rows"]["system"]["health_paper_status"] == "ready"
    assert payload["rows"]["system"]["paper_ramp_status"] == "ready"
    assert payload["rows"]["soak"]["effective_safe"] is False
    assert payload["headline_status"] == "blocked"
    assert payload["safe_to_leave_unattended"] is False
    assert "paper_runtime_policy_disagrees" in payload["contradictions"]
    assert payload["operator_summary"]["attention_owner"] == "throttle"
    assert payload["operator_summary"]["root_cause"] == "paper_execution_cpu_pressure"
    assert payload["operator_summary"]["paper_impact"] == "paper_blocked"
    assert "visibility=degraded" in headline
    assert "impact=paper_paused" in headline
    assert "paper_impact=paper_blocked" in headline


def test_current_storage_recovery_supersedes_old_safe_soak_snapshot(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    fast = json.loads((health / "health_fast_latest.json").read_text(encoding="utf-8"))
    fast["overall_status"] = "degraded"
    fast["strict_all_clear"] = False
    fast["repair_backlog_active"] = True
    fast["operational_readiness"]["guarded_paper"] = {
        "ok": False,
        "status": "blocked",
        "blockers": ["storage_pending_above_threshold"],
    }
    _write(health / "health_fast_latest.json", fast)
    _write(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": STAMP,
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 1.05,
            "pending_lines_threshold": 15000,
            "estimated_total_drain_minutes": 2.5,
            "backpressure": {
                "core_pending_lines": 15000,
                "total_pending_lines": 16500,
                "oldest_pending_age_seconds": 120,
                "raw_live": {"core_pending_lines": 2500, "total_pending_lines": 3500},
            },
            "storage": {"backlog_drain_status": "drain_active"},
        },
    )
    throttle = json.loads((health / "runtime_throttle_control_latest.json").read_text(encoding="utf-8"))
    throttle["overall_status"] = "degraded"
    throttle["runtime_saturation_governor_v2"]["paper_live_data_policy"] = {
        "paper_execution_allowed": False,
        "paper_execution_consumer_paused": True,
        "paper_execution_pause_reason": "paper_ramp_blocked",
    }
    _write(health / "runtime_throttle_control_latest.json", throttle)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)

    assert payload["overall_status"] == "degraded"
    assert payload["rows"]["storage"]["status"] == "recovering"
    assert payload["rows"]["storage"]["strict_status"] == "blocked"
    assert payload["rows"]["storage"]["effective_total_pending_lines"] == 3500
    assert payload["rows"]["throttle"]["cause"] == "storage_backlog"
    assert payload["rows"]["throttle"]["recovery"] == "storage_drain_active"
    assert payload["rows"]["soak"]["effective_safe"] is False
    assert "soak_snapshot_superseded_by_current_health" in payload["contradictions"]


def test_stale_secondary_auth_source_is_watch_not_false_block(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    broker = json.loads((health / "broker_readiness_latest.json").read_text(encoding="utf-8"))
    broker["timestamp_utc"] = "2026-08-03T11:00:00+00:00"
    _write(health / "broker_readiness_latest.json", broker)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)

    assert payload["overall_status"] == "degraded"
    assert payload["rows"]["auth"]["status"] == "watch"
    assert payload["rows"]["auth"]["reason"] == "stale_broker_readiness"
    assert payload["rows"]["auth"]["impact"] == "none"


def test_stale_unattended_soak_cannot_publish_walkaway_ready(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    soak_path = health / "unattended_soak_readiness_latest.json"
    soak = json.loads(soak_path.read_text(encoding="utf-8"))
    soak["timestamp_utc"] = "2026-08-03T10:00:00+00:00"
    _write(soak_path, soak)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    headline = contract.format_status_lines(payload)[0]

    assert payload["visibility_status"] == "degraded"
    assert payload["rows"]["soak"]["status"] == "degraded"
    assert payload["rows"]["soak"]["effective_safe"] is False
    assert payload["headline_status"] == "degraded"
    assert payload["safe_to_leave_unattended"] is False
    assert payload["operator_summary"]["attention_owner"] == "soak"
    assert "unattended_soak" in payload["stale_sources"]
    assert "walkaway=false" in headline


def test_formatted_lines_expose_cause_recovery_impact_and_action(tmp_path: Path) -> None:
    _ready_fixture(tmp_path)
    payload = contract.build_status_snapshot(tmp_path, now=NOW)

    lines = contract.format_status_lines(payload)
    joined = "\n".join(lines)

    assert len(lines) == 9
    assert "[status-contract]" in joined
    assert "[system]" in joined
    assert "[collection]" in joined
    assert "[auth]" in joined
    assert "[storage]" in joined
    assert "[throttle]" in joined
    assert "[soak]" in joined
    assert "[production-excellence]" in joined
    assert "cause=" in joined
    assert "recovery=" in joined
    assert "impact=" in joined
    assert "action=" in joined
    assert "walkaway=true" in lines[0]
    assert "active_issues=0" in lines[0]
    assert "managed_watches=none" in lines[0]
    assert "oldest=" in lines[0]
    assert "live=locked_read_only" in joined
    assert "level=ok" in joined
    assert "None" not in joined


def test_exit_code_uses_operational_headline_not_source_visibility() -> None:
    assert contract.status_exit_code({"overall_status": "ready", "headline_status": "blocked"}) == 2
    assert contract.status_exit_code({"overall_status": "degraded", "headline_status": "ready"}) == 0


def test_storage_reports_source_attributed_overlay_without_mislabeling_throttle(tmp_path: Path) -> None:
    health = _ready_fixture(tmp_path)
    storage_path = health / "ingestion_storage_control_latest.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage.update(
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 5.5,
            "recovery_state": "blocked_backpressure",
            "backlog_truth": {
                "authoritative_mode": "overlay_source_attributed",
                "raw_live": {"grade": "A+"},
                "sql_overlay": {"grade": "F", "used_for_pressure": True},
            },
            "stale_pending_locator": {
                "status": "attributed",
                "stale_source_count": 1,
                "oldest_sources": [
                    {
                        "source_rel": "governance/channels/decision/intraday_aggressive_equities_schwab/decision_20260803.jsonl",
                        "shard": "trading",
                        "pending_lines": 600,
                    }
                ],
            },
            "backpressure": {
                "core_pending_lines": 600,
                "total_pending_lines": 700,
                "oldest_pending_age_seconds": 1320,
                "raw_live": {"core_pending_lines": 4, "total_pending_lines": 100},
            },
            "storage": {"backlog_drain_status": "drain_active"},
        }
    )
    _write(storage_path, storage)

    payload = contract.build_status_snapshot(tmp_path, now=NOW)
    storage_row = payload["rows"]["storage"]

    assert payload["overall_status"] == "degraded"
    assert storage_row["truth_mode"] == "overlay_source_attributed"
    assert storage_row["raw_grade"] == "A+"
    assert storage_row["overlay_grade"] == "F"
    assert storage_row["stale_source_count"] == 1
    assert storage_row["oldest_source"].startswith("intraday_aggressive_equities_schwab/")
    assert storage_row["cause"] == "stale_sql_overlay"
    assert storage_row["impact"] == "strict_live_gate_only"
    assert payload["rows"]["throttle"]["status"] == "ready"
    assert payload["rows"]["throttle"]["cause"] == "strict_storage_backlog"
    assert payload["rows"]["soak"]["effective_safe"] is False
