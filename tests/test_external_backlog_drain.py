import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import external_backlog_drain as src
from scripts.ops import sql_link_shard_manager as shard_manager


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_sql_link_shard_manager_can_ignore_stale_active_request(tmp_path: Path, monkeypatch) -> None:
    request_path = tmp_path / "sql_link_service_request_latest.json"
    _write_json(
        request_path,
        {
            "active": True,
            "request_kind": "backpressure_drainer_fleet",
            "requested_at": "2026-05-28T00:03:19+00:00",
            "expires_utc": "2099-01-01T00:00:00+00:00",
            "reason": "runtime_channel_drainer",
            "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "runtime,crypto_runtime,health_fast"},
        },
    )

    monkeypatch.delenv("SQL_LINK_SERVICE_IGNORE_ACTIVE_REQUEST", raising=False)
    assert shard_manager._load_active_request(request_path)["env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "runtime,crypto_runtime,health_fast"

    monkeypatch.setenv("SQL_LINK_SERVICE_IGNORE_ACTIVE_REQUEST", "1")
    assert shard_manager._load_active_request(request_path) == {}


def test_external_backlog_drain_drops_blank_shard_path_filters() -> None:
    profile, env = src._drain_env(
        {},
        critical=True,
        off_hours_active=True,
        core_focus={
            "concentrated": True,
            "top3_pending_lines": 90000,
            "top3_share": 0.95,
            "hotspots": [
                {
                    "source_rel": "decisions/shadow_bond_equities/trade_decisions_20260612.jsonl",
                    "pending_lines": 90000,
                    "age_seconds": 1800.0,
                }
            ],
        },
        backpressure={
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_bond_equities/decision_explanations_20260612.jsonl",
                    "shard": "crypto_explanations",
                    "pending_lines": 80000,
                    "oldest_pending_age_seconds": 1800.0,
                }
            ],
        },
    )

    assert profile == "offhours_external_backlog_drain"
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"] == (
        "decisions/shadow_bond_equities/trade_decisions_20260612.jsonl"
    )
    assert env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS"] == (
        "decision_explanations/shadow_bond_equities/decision_explanations_20260612.jsonl"
    )
    assert "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS" not in env
    assert "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS" not in env
    assert "SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS" not in env
    assert all(value.strip() for key, value in env.items() if key.endswith("_PATH_CONTAINS"))


def test_fresh_managed_overlay_replaces_larger_raw_backlog() -> None:
    raw = {
        "timestamp_utc": "2026-08-05T13:03:30+00:00",
        "pending_lines": 553061,
        "pending_lines_total": 625624,
        "pending_lines_deferred": 72563,
        "top_pending_files": [
            {"source_rel": "decisions/stale.jsonl", "pending_lines": 553061},
        ],
    }
    storage = {
        "timestamp_utc": "2026-08-05T13:03:34+00:00",
        "backpressure": {
            "overlay_adjusted": True,
            "core_pending_lines": 12838,
            "deferred_pending_lines": 0,
            "cold_pending_lines": 0,
            "total_pending_lines": 12838,
            "oldest_pending_age_seconds": 35.0,
        },
        "stale_pending_locator": {
            "top_pending_sources": [
                {
                    "source_rel": "decisions/fresh.jsonl",
                    "pending_lines": 12838,
                    "pressure_lane": "core",
                }
            ],
            "oldest_sources": [],
        },
    }

    effective = src._backpressure_with_storage_overlay(raw, storage)

    assert effective["pending_lines"] == 12838
    assert effective["pending_lines_total"] == 12838
    assert effective["pending_lines_deferred"] == 0
    assert [row["source_rel"] for row in effective["top_pending_files"]] == ["decisions/fresh.jsonl"]
    assert effective["_storage_overlay_authoritative"] is True


def test_fresh_zero_overlay_retires_raw_backlog_but_stale_zero_does_not() -> None:
    raw = {
        "timestamp_utc": "2026-08-05T13:03:30+00:00",
        "pending_lines": 50000,
        "pending_lines_total": 50000,
        "top_pending_files": [{"source_rel": "decisions/raw.jsonl", "pending_lines": 50000}],
    }
    fresh_zero = {
        "timestamp_utc": "2026-08-05T13:03:35+00:00",
        "backpressure": {
            "overlay_adjusted": True,
            "core_pending_lines": 0,
            "deferred_pending_lines": 0,
            "cold_pending_lines": 0,
            "total_pending_lines": 0,
        },
        "stale_pending_locator": {"top_pending_sources": [], "oldest_sources": []},
    }
    stale_zero = {**fresh_zero, "timestamp_utc": "2026-08-05T12:58:00+00:00"}

    effective_fresh = src._backpressure_with_storage_overlay(raw, fresh_zero)
    effective_stale = src._backpressure_with_storage_overlay(raw, stale_zero)

    assert effective_fresh["pending_lines_total"] == 0
    assert effective_fresh["top_pending_files"] == []
    assert effective_fresh["_storage_overlay_authoritative_zero"] is True
    assert effective_stale["pending_lines_total"] == 50000
    assert effective_stale["_storage_overlay_rejected_reason"] == "managed_overlay_older_than_raw_backpressure"


def test_newer_lower_raw_estimate_does_not_erase_conservative_overlay_debt() -> None:
    raw = {
        "timestamp_utc": "2026-08-05T13:36:27+00:00",
        "pending_lines": 769,
        "pending_lines_total": 21749,
    }
    storage = {
        "timestamp_utc": "2026-08-05T13:35:48+00:00",
        "backpressure": {
            "overlay_adjusted": True,
            "core_pending_lines": 91419,
            "deferred_pending_lines": 20974,
            "total_pending_lines": 112393,
        },
        "stale_pending_locator": {
            "top_pending_sources": [
                {
                    "source_rel": "governance/events/signal_generation_20260805.jsonl",
                    "pending_lines": 91419,
                    "pressure_lane": "core",
                }
            ],
            "oldest_sources": [],
        },
    }

    effective = src._backpressure_with_storage_overlay(raw, storage)

    assert effective["pending_lines"] == 91419
    assert effective["pending_lines_total"] == 112393
    assert effective["_storage_overlay_authoritative"] is True
    assert effective["_storage_overlay_freshness"]["raw_is_stricter"] is False


def test_apply_parks_writer_when_fresh_managed_overlay_is_zero(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    now = datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)
    raw = {
        "timestamp_utc": "2026-08-05T12:59:50+00:00",
        "pending_lines": 50000,
        "pending_lines_total": 50000,
        "top_pending_files": [{"source_rel": "decisions/raw.jsonl", "pending_lines": 50000}],
    }
    _write_json(health / "ingestion_backpressure_latest.json", raw)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": "2026-08-05T13:00:00+00:00",
            "backpressure": {
                "overlay_adjusted": True,
                "core_pending_lines": 0,
                "deferred_pending_lines": 0,
                "cold_pending_lines": 0,
                "total_pending_lines": 0,
            },
            "stale_pending_locator": {"top_pending_sources": [], "oldest_sources": []},
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 0})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(
        health / "sql_link_service_request_latest.json",
        {
            "active": True,
            "request_kind": "external_backlog_drain",
            "requested_at": "2026-08-05T12:58:00+00:00",
            "expires_utc": "2026-08-05T13:15:00+00:00",
            "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "trading,governance"},
        },
    )
    monkeypatch.setattr(src.governor_src, "build_payload", lambda *args, **kwargs: {"profile": "stable", "env_overrides": {}})
    calls: list[str] = []

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path=None, env_overrides=None, timeout_seconds=None) -> dict:
        joined = " ".join(cmd)
        calls.append(joined)
        if "ingestion_backpressure_guard.py" in joined:
            payload = raw
        elif "ingestion_priority_queue.py" in joined:
            payload = {"queue_depth": 0}
        else:
            raise AssertionError(f"unexpected command: {joined}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 1.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, now_utc=now)

    assert payload["apply_executed"] is False
    assert payload["material_drain_recommended"] is False
    assert payload["steps"]["material_drain_gate"]["status"] == "skipped"
    assert not any("sql_link_shard_manager.py" in call for call in calls)
    retired = json.loads((health / "sql_link_service_request_latest.json").read_text(encoding="utf-8"))
    assert retired["active"] is False
    assert retired["retired_reason"] == "fresh_managed_overlay_reports_zero_backlog"


def test_external_backlog_drain_memory_caps_override_operator_burst_width(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "memory_efficiency_control_latest.json",
        {
            "overall_status": "advisory",
            "recommended_profile": "air_safe",
            "reasons": ["compressed_memory_high"],
            "compressed_memory_relief_contract": {"active": True, "managed": True},
            "recommended_env_overrides": {
                "BACKLOG_PCORE_PREPROCESS_WORKERS": "1",
                "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "1",
                "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "1",
                "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "1",
            },
        },
    )

    capped, contract = src._apply_memory_safety_caps(
        project_root,
        {
            "BACKLOG_PCORE_PREPROCESS_WORKERS": "7",
            "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "7",
            "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "7",
            "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "8",
            "BACKLOG_ACCELERATOR_PREPROCESS_WORKERS": "7",
            "BACKLOG_ACCELERATOR_TARGET_PLANNED_SHARDS": "7",
        },
    )

    assert contract["active"] is True
    assert contract["applied"] is True
    assert {capped[key] for key in src._MEMORY_SAFETY_WORKER_KEYS} == {"1"}
    assert capped["BACKLOG_ACCELERATOR_PREPROCESS_WORKERS"] == "1"
    assert capped["BACKLOG_ACCELERATOR_TARGET_PLANNED_SHARDS"] == "1"


def test_external_backlog_drain_builds_offhours_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 12000,
            "pending_lines_total": 510000,
            "pending_lines_deferred": 410000,
            "pending_lines_cold": 180000,
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/events/api_calls_20260406.jsonl",
                    "pending_lines": 210000,
                    "oldest_pending_age_seconds": 14400.0,
                }
            ],
            "top_cold_pending_files": [
                {
                    "source_rel": "governance/shadow_pnl_attribution_20260406.jsonl",
                    "pending_lines": 180000,
                    "oldest_pending_age_seconds": 25200.0,
                }
            ],
            "top_pending_files": [
                {
                    "source_rel": "governance/execution_lanes/execution_results_20260406.jsonl",
                    "pending_lines": 240000,
                    "oldest_pending_age_seconds": 6400.0,
                },
                {
                    "source_rel": "governance/execution_lanes/execution_promotions_20260406.jsonl",
                    "pending_lines": 180000,
                    "oldest_pending_age_seconds": 6300.0,
                },
                {
                    "source_rel": "governance/execution_lanes/execution_intents_20260406.jsonl",
                    "pending_lines": 90000,
                    "oldest_pending_age_seconds": 6200.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 14})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 2.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["recommended_now"] is True
    assert payload["drain_profile"] == "offhours_external_backlog_drain"
    assert payload["aged_candidate_files"] == 2
    assert payload["drain_overrides"]["deferred_files_budget"] == 6
    assert payload["drain_overrides"]["sql_interval_seconds"] == 12
    assert payload["drain_overrides"]["hot_batch_size"] == 240000
    assert payload["drain_overrides"]["preferred_shards"] == ["governance", "health_fast", "support_watchdog"]
    assert payload["drain_overrides"]["governance_max_files"] == 14
    assert payload["drain_overrides"]["governance_max_lines_per_file"] == 64000
    assert payload["drain_overrides"]["resource_guard_optional_max_load_per_core"] == 12.0
    assert payload["drain_overrides"]["governance_path_focus"] == [
        "governance/execution_lanes/execution_results_20260406.jsonl",
        "governance/execution_lanes/execution_promotions_20260406.jsonl",
        "governance/execution_lanes/execution_intents_20260406.jsonl",
    ]
    assert payload["core_focus_concentrated"] is True
    assert payload["core_focus_top3_pending_lines"] == 510000
    assert any("compact or archive" in item for item in payload["top_actions"])
    assert any("dominant core backlog files" in item for item in payload["top_actions"])
    assert any("governance shard pinned" in item for item in payload["top_actions"])


def test_external_backlog_drain_accepts_pinned_local_hot_storage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 20,
            "pending_lines_total": 40,
            "pending_lines_deferred": 20,
            "pending_lines_cold": 0,
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "storage_mount_guard_latest.json",
        {
            "external_available": False,
            "storage_mode": "external",
            "hot_storage_available": True,
            "external_required_for_hot_path": False,
            "probe_skipped_external_io": True,
        },
    )
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["reported_storage_mode"] == "external"
    assert payload["storage_mode"] == "local_fallback"
    assert payload["local_hot_storage_ready"] is True
    assert payload["routed_hot_storage_ready"] is True
    assert "routed_hot_storage_unavailable" not in payload["blocked_reasons"]


def test_external_backlog_drain_waits_cleanly_for_market_hours_guard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 800,
            "pending_lines_total": 4200,
            "pending_lines_deferred": 3400,
            "pending_lines_cold": 0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/paper/trade_decisions_20260504.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 120.0,
                }
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/runtime/default_crypto_schwab/runtime_20260504.jsonl",
                    "pending_lines": 3400,
                    "oldest_pending_age_seconds": 180.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 4})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 4, 14, 30, tzinfo=timezone.utc),
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "waiting_for_off_hours"
    assert payload["material_drain_recommended"] is True
    assert payload["recommended_now"] is False
    assert payload["blocked_reasons"] == ["market_hours_guard"]
    assert payload["hard_blocked_reasons"] == []
    assert payload["soft_blocked_reasons"] == ["market_hours_guard"]
    assert payload["waiting_for_off_hours"] is True


def test_external_backlog_drain_pins_decision_channels_to_trading_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 66344,
            "pending_lines_total": 66344,
            "pending_lines_deferred": 1888,
            "pending_lines_cold": 0,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 30675,
                    "oldest_pending_age_seconds": 1200.0,
                },
                {
                    "source_rel": "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 29664,
                    "oldest_pending_age_seconds": 1180.0,
                },
                {
                    "source_rel": "governance/channels/decision/intraday_aggressive_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 3541,
                    "oldest_pending_age_seconds": 200.0,
                },
                {
                    "source_rel": "governance/events/premarket_token_guard_20260430.jsonl",
                    "pending_lines": 1,
                    "oldest_pending_age_seconds": 9000.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 8})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "storage_pressure": {"retention_debt_gb": 0.4}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 30, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["core_focus_concentrated"] is True
    assert payload["drain_overrides"]["preferred_shards"] == ["trading", "health_fast", "support_watchdog"]
    assert payload["drain_overrides"]["governance_path_focus"] == []
    assert payload["drain_overrides"]["trading_max_files"] == 16
    assert payload["drain_overrides"]["trading_max_lines_per_file"] == 64000
    assert payload["drain_overrides"]["trading_path_focus"] == [
        "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
        "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
    ]
    assert any("trading shard pinned" in item for item in payload["top_actions"])


def test_external_backlog_drain_includes_crypto_trading_for_crypto_decision_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 9600,
            "pending_lines_total": 9600,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260502.jsonl",
                    "pending_lines": 6200,
                    "oldest_pending_age_seconds": 12.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260502.jsonl",
                    "pending_lines": 2600,
                    "oldest_pending_age_seconds": 8.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 2, 21, 0, tzinfo=timezone.utc),
    )

    assert "crypto_trading" in payload["drain_overrides"]["preferred_shards"]
    assert payload["drain_overrides"]["preferred_shards"].index("crypto_trading") < payload["drain_overrides"]["preferred_shards"].index("trading")


def test_external_backlog_drain_focuses_near_hard_core_expansion_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 43991,
            "pending_lines_total": 44565,
            "pending_lines_deferred": 574,
            "pending_lines_cold": 0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/paper/trade_decisions_20260503.jsonl",
                    "pending_lines": 33735,
                    "oldest_pending_age_seconds": 60.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260503.jsonl",
                    "pending_lines": 8613,
                    "oldest_pending_age_seconds": 22092.0,
                },
                {
                    "source_rel": "paper_trades_paper.jsonl",
                    "pending_lines": 799,
                    "oldest_pending_age_seconds": 0.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 3, 13, 45, tzinfo=timezone.utc),
    )

    assert payload["core_focus_concentrated"] is True
    assert payload["drain_overrides"]["preferred_shards"][:4] == [
        "trading",
        "crypto_trading",
        "health_fast",
        "support_watchdog",
    ]
    assert payload["drain_overrides"]["trading_path_focus"] == [
        "decisions/paper/trade_decisions_20260503.jsonl",
    ]
    assert payload["drain_overrides"]["crypto_trading_path_focus"] == [
        "decisions/shadow_crypto/trade_decisions_20260503.jsonl",
    ]
    assert payload["drain_overrides"]["shard_link_timeout_seconds"] == 420


def test_external_backlog_drain_treats_market_holiday_as_safe_storage_window(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1000,
            "pending_lines_total": 700000,
            "pending_lines_deferred": 690000,
            "pending_lines_cold": 0,
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_crypto/decision_explanations_20260525.jsonl",
                    "pending_lines": 334000,
                    "oldest_pending_age_seconds": 36000.0,
                },
                {
                    "source_rel": "decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260525.jsonl",
                    "pending_lines": 324000,
                    "oldest_pending_age_seconds": 36000.0,
                },
                {
                    "source_rel": "decision_explanations/shadow_fx_equities/decision_explanations_20260525.jsonl",
                    "pending_lines": 30000,
                    "oldest_pending_age_seconds": 36000.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 20})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 25, 13, 30, tzinfo=timezone.utc),
    )

    assert payload["blocked_reasons"] == []
    assert payload["off_hours_window"]["active"] is True
    assert payload["off_hours_window"]["market_holiday"] is True
    assert payload["off_hours_window"]["market_holiday_name"] == "memorial_day"
    assert payload["recommended_now"] is True
    assert payload["drain_overrides"]["preferred_shards"][:2] == ["crypto_explanations", "explanations"]
    assert payload["drain_overrides"]["crypto_explanations_max_lines_per_file"] == 64000
    assert payload["drain_overrides"]["crypto_explanations_path_focus"] == [
        "decision_explanations/shadow_crypto/decision_explanations_20260525.jsonl",
        "decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260525.jsonl",
    ]
    assert payload["drain_overrides"]["explanations_path_focus"] == [
        "decision_explanations/shadow_fx_equities/decision_explanations_20260525.jsonl",
    ]
    assert any("crypto explanations shard pinned" in item for item in payload["top_actions"])


def test_external_backlog_drain_uses_storage_overlay_leaders_for_focus(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 39811,
            "pending_lines_total": 44060,
            "pending_lines_deferred": 4249,
            "pending_lines_cold": 0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/write_failures_20260525.jsonl",
                    "pending_lines": 32618,
                    "oldest_pending_age_seconds": 13558.506,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "backpressure": {
                "core_pending_lines": 1904980,
                "deferred_pending_lines": 4249,
                "cold_pending_lines": 0,
                "support_pending_lines": 77119,
                "total_pending_lines": 1986348,
                "overlay_adjusted": True,
                "oldest_pending_age_seconds": 36480.298,
            },
            "stale_pending_locator": {
                "top_pending_sources": [
                    {
                        "source_rel": "governance/events/signal_generation_20260525.jsonl",
                        "shard": "governance",
                        "pressure_lane": "core",
                        "pending_lines": 1314510,
                        "oldest_pending_age_seconds": 114.839,
                    },
                    {
                        "source_rel": "decisions/shadow_crypto/trade_decisions_20260525.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 298525,
                        "oldest_pending_age_seconds": 36425.104,
                    },
                    {
                        "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260525.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 291945,
                        "oldest_pending_age_seconds": 36480.298,
                    },
                    {
                        "source_rel": "governance/channels/risk/crypto_futures_crypto_schwab/risk_20260525.jsonl",
                        "shard": "risk_support",
                        "pressure_lane": "support",
                        "pending_lines": 39103,
                        "oldest_pending_age_seconds": 36409.237,
                    },
                ],
                "oldest_sources": [],
            },
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 20})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 25, 13, 40, tzinfo=timezone.utc),
    )

    assert payload["storage_overlay_focus"]["active"] is True
    assert payload["storage_overlay_focus"]["adjusted"] is True
    assert payload["backpressure_before"]["total_pending_lines"] == 1986348
    assert payload["raw_backpressure_before"]["total_pending_lines"] == 44060
    assert payload["core_focus_top3_pending_lines"] == 1904980
    assert payload["drain_overrides"]["preferred_shards"][:5] == [
        "governance",
        "crypto_trading",
        "risk_support",
        "health_fast",
        "support_watchdog",
    ]
    assert payload["drain_overrides"]["governance_path_focus"] == [
        "governance/events/signal_generation_20260525.jsonl",
        "governance/events/write_failures_20260525.jsonl",
    ]
    assert payload["drain_overrides"]["crypto_trading_path_focus"] == [
        "decisions/shadow_crypto/trade_decisions_20260525.jsonl",
        "decisions/shadow_crypto_futures_crypto/trade_decisions_20260525.jsonl",
    ]
    assert payload["drain_overrides"]["risk_support_path_focus"] == [
        "governance/channels/risk/crypto_futures_crypto_schwab/risk_20260525.jsonl",
    ]
    assert payload["drain_overrides"]["risk_support_max_lines_per_file"] == 160000
    assert any("risk-support shard pinned" in item for item in payload["top_actions"])


def test_external_backlog_drain_routes_overlay_explanations_by_shard(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 2000,
            "pending_lines_total": 2500,
            "pending_lines_deferred": 500,
            "pending_lines_cold": 0,
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "backpressure": {
                "core_pending_lines": 2000,
                "deferred_pending_lines": 500,
                "cold_pending_lines": 90000,
                "total_pending_lines": 92500,
                "overlay_adjusted": True,
                "oldest_pending_age_seconds": 36000.0,
            },
            "stale_pending_locator": {
                "top_pending_sources": [
                    {
                        "source_rel": "decision_explanations/shadow_neural_operator_surrogates_equities/decision_explanations_20260525.jsonl",
                        "shard": "crypto_explanations",
                        "pressure_lane": "cold",
                        "pending_lines": 64000,
                        "oldest_pending_age_seconds": 36000.0,
                    }
                ],
                "oldest_sources": [],
            },
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 20})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 25, 13, 40, tzinfo=timezone.utc),
    )

    assert payload["drain_overrides"]["preferred_shards"][0] == "crypto_explanations"
    assert payload["drain_overrides"]["crypto_explanations_path_focus"] == [
        "decision_explanations/shadow_neural_operator_surrogates_equities/decision_explanations_20260525.jsonl",
    ]
    assert payload["drain_overrides"]["crypto_explanations_max_lines_per_file"] == 64000


def test_external_backlog_drain_does_not_recommend_broad_sweep_for_tiny_hot_queue(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 182,
            "pending_lines_total": 184,
            "pending_lines_deferred": 2,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 2,
            "top_pending_files": [
                {
                    "source_rel": "decisions/paper/trade_decisions_20260503.jsonl",
                    "pending_lines": 155,
                    "oldest_pending_age_seconds": 145.0,
                }
            ],
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/failover_events.jsonl",
                    "pending_lines": 2,
                    "oldest_pending_age_seconds": 18.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 3, 14, 5, tzinfo=timezone.utc),
    )

    assert payload["recommended_now"] is False
    assert payload["material_drain_recommended"] is False
    assert payload["core_focus_concentrated"] is False


def test_external_backlog_drain_recommends_sparse_large_jsonl_byte_window(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 240,
            "pending_lines_total": 240,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "line_estimation": {
                "sparse_large_line_pending_lines": 90,
                "sparse_large_line_pending_bytes": 96 * 1024 * 1024,
            },
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260525.jsonl",
                    "pending_lines": 90,
                    "oldest_pending_age_seconds": 20.0,
                    "sparse_large_line": True,
                    "estimated_pending_bytes": 96 * 1024 * 1024,
                    "estimated_avg_bytes_per_line": 950000.0,
                    "file_size_bytes": 9_000_000_000,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 25, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["recommended_now"] is True
    assert payload["material_drain_recommended"] is True
    assert payload["backpressure_after"]["sparse_large_line_pending_bytes"] == 96 * 1024 * 1024
    assert payload["drain_overrides"]["sparse_large_decision_drain"] is True
    assert payload["drain_overrides"]["ingest_max_bytes_per_file"] == 128 * 1024 * 1024
    assert payload["drain_overrides"]["sqlite_batch_max_bytes"] == 32 * 1024 * 1024
    assert payload["drain_overrides"]["crypto_trading_path_focus"] == [
        "governance/channels/decision/default_crypto_schwab/decision_20260525.jsonl"
    ]


def test_external_backlog_drain_writes_handoff_when_resource_guard_blocks_focused_core_backlog(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    backpressure_payload = {
        "pending_lines": 64000,
        "pending_lines_total": 64000,
        "pending_lines_deferred": 0,
        "pending_lines_cold": 0,
        "top_pending_files": [
            {
                "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
                "pending_lines": 31000,
                "oldest_pending_age_seconds": 1800.0,
            },
            {
                "source_rel": "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
                "pending_lines": 30000,
                "oldest_pending_age_seconds": 1800.0,
            },
        ],
    }
    _write_json(health / "ingestion_backpressure_latest.json", backpressure_payload)
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    lock_path = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=4321 started=2026-04-30T22:00:00+00:00 cmd=sql_link_shard_manager", encoding="utf-8")

    monkeypatch.setattr(src, "SQL_WRITER_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        joined = " ".join(cmd)
        if "ingestion_backpressure_guard.py" in joined:
            payload = backpressure_payload
        elif "ingestion_priority_queue.py" in joined:
            payload = {"queue_depth": 2}
        elif "resource_guard.py" in joined:
            payload = {"ok": False, "resource_guard_ok": False, "resource_guard_reasons": ["memory_pressure_yellow"]}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(
        project_root,
        apply=True,
        follow_through=True,
        now_utc=datetime(2026, 4, 30, 22, 5, tzinfo=timezone.utc),
    )

    assert payload["apply_executed"] is False
    assert payload["blocked_reasons"] == ["resource_guard_blocked"]
    assert payload["follow_through"]["status"] == "handoff_requested"
    assert payload["follow_through"]["progress_state"] == "requested_live_writer_after_resource_guard"
    assert payload["service_request"]["request_kind"] == "external_backlog_drain"
    assert payload["service_request"]["reason"].endswith(":resource_guard_handoff")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("trading,")
    assert "SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS" in payload["service_request"]["env_overrides"]


def test_external_backlog_drain_calls_out_stale_stage_archive_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 200,
            "pending_lines_total": 120200,
            "pending_lines_deferred": 120000,
            "pending_lines_cold": 119900,
            "top_deferred_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl",
                    "pending_lines": 119900,
                    "oldest_pending_age_seconds": 572800.0,
                }
            ],
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl",
                    "pending_lines": 119900,
                    "oldest_pending_age_seconds": 572800.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 3})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["stale_stage_candidate_files"] == 1
    assert payload["stale_stage_candidate_pending_lines"] == 119900
    assert any(str(row.get("candidate_action")) == "reap_or_archive_stale_stage" for row in payload["hotspots"])
    assert any("staged stale artifacts" in item for item in payload["top_actions"])


def test_external_backlog_drain_calls_out_watchdog_support_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 400,
            "pending_lines_total": 182400,
            "pending_lines_deferred": 182000,
            "pending_lines_support_telemetry": 180000,
            "pending_lines_cold": 0,
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/failover_events.jsonl",
                    "pending_lines": 180000,
                    "oldest_pending_age_seconds": 90.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["support_watchdog_candidate_files"] == 1
    assert payload["support_watchdog_candidate_pending_lines"] == 180000
    assert any(str(row.get("candidate_action")) == "drain_support_watchdog" for row in payload["hotspots"])
    assert any("support shard" in item for item in payload["top_actions"])


def test_external_backlog_drain_apply_executes_and_refreshes_backlog(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10000,
            "pending_lines_total": 300000,
            "pending_lines_deferred": 220000,
            "pending_lines_cold": 80000,
            "top_pending_files": [
                {
                    "source_rel": "governance/execution_lanes/execution_results_20260406.jsonl",
                    "pending_lines": 140000,
                    "oldest_pending_age_seconds": 6400.0,
                },
                {
                    "source_rel": "governance/execution_lanes/execution_promotions_20260406.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 6300.0,
                },
                {
                    "source_rel": "governance/execution_lanes/execution_intents_20260406.jsonl",
                    "pending_lines": 60000,
                    "oldest_pending_age_seconds": 6200.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 12})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})

    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {
            "profile": "critical_backpressure",
            "env_overrides": {
                "SQL_LINK_SERVICE_PRIMARY_DB": str(project_root / "data" / "jsonl_link.sqlite3"),
                "BOT_CHANNEL_QUEUE_DB": str(project_root / "data" / "bot_channel_queue.sqlite3"),
                "SQL_LINK_SERVICE_QUEUE_DB": str(project_root / "data" / "bot_channel_queue.sqlite3"),
                "INGEST_MAX_DEFERRED_FILES": "0",
                "JSONL_SQL_MAX_COLD_LANE_FILES": "0",
            },
        },
    )

    seen: list[str] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        joined = " ".join(cmd)
        seen.append(joined)
        if "ingestion_backpressure_guard.py" in joined and "before" not in joined:
            if len([row for row in seen if "ingestion_backpressure_guard.py" in row]) == 1:
                payload = {
                    "pending_lines": 10000,
                    "pending_lines_total": 300000,
                    "pending_lines_deferred": 220000,
                    "pending_lines_cold": 80000,
                }
            else:
                payload = {
                    "pending_lines": 4000,
                    "pending_lines_total": 180000,
                    "pending_lines_deferred": 120000,
                    "pending_lines_cold": 56000,
                }
        elif "ingestion_priority_queue.py" in joined:
            if len([row for row in seen if "ingestion_priority_queue.py" in row]) == 1:
                payload = {"queue_depth": 12}
            else:
                payload = {"queue_depth": 7}
        elif "resource_guard.py" in joined:
            payload = {"ok": True}
        elif "sql_link_shard_manager.py" in joined:
            assert env_overrides is not None
            assert env_overrides["INGEST_MAX_DEFERRED_FILES"] == "6"
            assert env_overrides["JSONL_SQL_MAX_COLD_LANE_FILES"] == "2"
            assert env_overrides["SQL_LINK_SERVICE_IGNORE_ACTIVE_REQUEST"] == "1"
            assert env_overrides["SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB"] == "0.25"
            assert env_overrides["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "90"
            assert env_overrides["SQL_LINK_SERVICE_AUTO_HOT_RETENTION"] == "0"
            assert env_overrides["SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION"] == "0"
            assert env_overrides["SQL_LINK_SERVICE_SHARD_RUNTIME_STATE_CHECKPOINT_LINES"] == "1500"
            assert env_overrides["SQL_LINK_SERVICE_SHARDS"].startswith("governance,")
            assert env_overrides["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"] == "14"
            assert env_overrides["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE"] == "64000"
            assert env_overrides["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"] == ",".join(
                [
                    "governance/execution_lanes/execution_results_20260406.jsonl",
                    "governance/execution_lanes/execution_promotions_20260406.jsonl",
                    "governance/execution_lanes/execution_intents_20260406.jsonl",
                ]
            )
            payload = {"ok": True, "reason": "ok"}
        elif "sqlite_performance_maintenance.py" in joined:
            assert timeout_seconds == 20.0
            payload = {"ok": True}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 3, "staged_files": 2}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 1, "deleted_files": 1}}
        elif "data_retention_policy.py" in joined:
            payload = {"ok": True, "deleted": 9}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["apply_executed"] is True
    assert payload["drain_delta"]["deferred_pending_lines"] == 100000
    assert payload["drain_delta"]["cold_pending_lines"] == 24000
    assert payload["queue_depth_after"] == 7
    assert payload["steps"]["sql_link_shard_manager"]["status"] == "ok"


def test_external_backlog_drain_follow_through_retries_busy_writer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines": 50, "pending_lines_total": 100, "pending_lines_deferred": 40, "pending_lines_cold": 10})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    lock_path = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=4321 started=2026-04-06T20:00:00+00:00 cmd=sql_link_shard_manager", encoding="utf-8")

    monkeypatch.setattr(src, "SQL_WRITER_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )
    monkeypatch.setattr(src.time_mod, "sleep", lambda seconds: None)

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        joined = " ".join(cmd)
        if "ingestion_backpressure_guard.py" in joined:
            payload = {"pending_lines": 50, "pending_lines_total": 100, "pending_lines_deferred": 40, "pending_lines_cold": 10}
        elif "ingestion_priority_queue.py" in joined:
            payload = {"queue_depth": 2}
        elif "resource_guard.py" in joined:
            payload = {"ok": True}
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": False, "reason": "writer_lock_busy", "busy": True}
        elif "sqlite_performance_maintenance.py" in joined:
            payload = {"ok": True}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "staged_files": 0}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "deleted_files": 0}}
        elif "data_retention_policy.py" in joined:
            payload = {"ok": True, "deleted": 0}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(
        project_root,
        apply=True,
        follow_through=True,
        poll_seconds=0.1,
        wait_timeout_seconds=1.0,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["follow_through"]["requested"] is True
    assert payload["follow_through"]["completed"] is True
    assert payload["follow_through"]["attempts"] == 1
    assert payload["follow_through"]["status"] == "handoff_requested"
    assert payload["follow_through"]["progress_state"] == "requested_live_writer"
    assert payload["steps"]["sql_link_shard_manager_initial"]["status"] == "busy"
    assert payload["steps"]["sql_link_service_request"]["status"] == "ok"
    assert payload["service_request"]["request_kind"] == "external_backlog_drain"
    assert (project_root / "governance" / "health" / "sql_link_service_request_latest.json").exists()


def test_external_backlog_drain_preserves_focused_drainer_handoff(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    request_path = health / "sql_link_service_request_latest.json"
    _write_json(
        request_path,
        {
            "timestamp_utc": "2026-04-06T20:59:00+00:00",
            "active": True,
            "request_kind": "backpressure_drainer_fleet",
            "reason": "backpressure_drainer_fleet:stale_decision_log_drainer",
            "requested_at": "2026-04-06T20:59:00+00:00",
            "expires_utc": "2026-04-06T21:20:00+00:00",
            "assigned_pressure_lane": "stale_decision_log_backpressure",
            "env_overrides": {
                "SQL_LINK_SERVICE_SHARDS": "trading,aggressive_trading,crypto_trading,health_fast",
                "SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS": "governance/channels/decision/conservative_equities_schwab/decision_20260406.jsonl",
            },
        },
    )
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines": 50, "pending_lines_total": 100, "pending_lines_deferred": 40, "pending_lines_cold": 10})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    lock_path = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=4321 started=2026-04-06T20:00:00+00:00 cmd=sql_link_shard_manager", encoding="utf-8")

    monkeypatch.setattr(src, "SQL_WRITER_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )
    monkeypatch.setattr(src.time_mod, "sleep", lambda seconds: None)

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        joined = " ".join(cmd)
        if "ingestion_backpressure_guard.py" in joined:
            payload = {"pending_lines": 50, "pending_lines_total": 100, "pending_lines_deferred": 40, "pending_lines_cold": 10}
        elif "ingestion_priority_queue.py" in joined:
            payload = {"queue_depth": 2}
        elif "resource_guard.py" in joined:
            payload = {"ok": True}
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": False, "reason": "writer_lock_busy", "busy": True}
        elif "sqlite_performance_maintenance.py" in joined:
            payload = {"ok": True}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "staged_files": 0}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "deleted_files": 0}}
        elif "data_retention_policy.py" in joined:
            payload = {"ok": True, "deleted": 0}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(
        project_root,
        apply=True,
        follow_through=True,
        poll_seconds=0.1,
        wait_timeout_seconds=1.0,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["follow_through"]["status"] == "handoff_requested"
    assert payload["service_request"]["request_kind"] == "backpressure_drainer_fleet"
    assert payload["service_request"]["preserved_existing_request"] is True
    persisted = json.loads(request_path.read_text(encoding="utf-8"))
    assert persisted["request_kind"] == "backpressure_drainer_fleet"
    assert persisted["reason"] == "backpressure_drainer_fleet:stale_decision_log_drainer"


def test_follow_through_retry_marks_progressing_timeout(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    lock_path = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=4321 started=2026-04-06T20:00:00+00:00 cmd=sql_link_shard_manager", encoding="utf-8")

    base_now = datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc)

    class _FakeDatetime:
        current = base_now

        @classmethod
        def now(cls, tz=None):
            value = cls.current
            return value if tz is None else value.astimezone(tz)

    attempts = {"count": 0}

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        attempts["count"] += 1
        _FakeDatetime.current += timedelta(seconds=0.6)
        payload = {
            "ok": False,
            "reason": "writer_lock_busy",
            "busy": True,
            "current_step": "merge_primary",
            "completed_shard_count": attempts["count"],
            "completed_merge_count": 0,
            "merged_rows_this_cycle": attempts["count"] * 100,
        }
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "SQL_WRITER_LOCK_PATH", lock_path)
    monkeypatch.setattr(src, "datetime", _FakeDatetime)
    monkeypatch.setattr(src, "_run_json_command", _fake_run)
    monkeypatch.setattr(src.time_mod, "sleep", lambda seconds: None)

    result = src._follow_through_retry(
        project_root=project_root,
        health_root=health,
        drain_env={},
        poll_seconds=0.1,
        wait_timeout_seconds=1.0,
    )

    assert result["completed"] is False
    assert result["status"] == "timed_out"
    assert result["progress_state"] == "progressing"
    assert result["progress_observed"] is True
    assert result["progress_events"] >= 1


def test_run_json_command_returns_after_hard_child_timeout(tmp_path: Path) -> None:
    result = src._run_json_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        timeout_seconds=1.0,
    )

    assert result["timed_out"] is True
    assert result["rc"] == 124
    assert result["duration_ms"] < 5000
