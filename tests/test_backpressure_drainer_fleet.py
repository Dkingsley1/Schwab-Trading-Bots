import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backpressure_drainer_fleet as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_backpressure_drainer_fleet_routes_concentrated_decisions_to_trading(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 64455,
            "pending_lines_total": 64472,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 30675,
                    "oldest_pending_age_seconds": 7200.0,
                },
                {
                    "source_rel": "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 29664,
                    "oldest_pending_age_seconds": 7200.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["shards"][:2] == ["trading", "aggressive_trading"]
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("trading,")
    assert "conservative_equities_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_routes_crypto_decisions_to_crypto_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1200,
            "pending_lines_total": 1200,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260501.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 900.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260501.jsonl",
                    "pending_lines": 400,
                    "oldest_pending_age_seconds": 600.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["shards"][:1] == ["crypto_trading"]
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("crypto_trading,")
    assert "SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS" not in payload["active_env_overrides"]
    assert "default_crypto_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_writes_single_writer_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1000,
            "pending_lines_total": 1000,
            "pending_lines_support_telemetry": 1000,
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/failover_events.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 300.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["service_request"]["request_kind"] == "backpressure_drainer_fleet"
    assert payload["service_request"]["reason"].endswith(":support_watchdog_drainer")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "support_watchdog,health_fast"
    assert (health / "sql_link_service_request_latest.json").exists()


def test_backpressure_drainer_fleet_guards_cold_stage_during_market_hours(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120000,
            "pending_lines_cold": 120000,
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/a.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "blocked"
    assert payload["blocked_reasons"] == ["market_hours_guard"]
    assert not (health / "sql_link_service_request_latest.json").exists()


def test_backpressure_drainer_fleet_can_force_cold_stage_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120000,
            "pending_lines_cold": 120000,
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/a.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        force_live_window=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "cold_stage_drainer"
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("data,")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS"].endswith("a.jsonl")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE"] == "64000"
