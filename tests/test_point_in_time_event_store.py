import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.point_in_time_event_store as event_store


def test_point_in_time_event_store_normalizes_recent_events(tmp_path) -> None:
    project_root = tmp_path / "project"
    event_dir = project_root / "governance" / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / "live_macro_media_events_20260401.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-01T14:55:30+00:00",
                "event_type": "live_macro_media_ingest",
                "category": "live_macro_media",
                "source": "C-SPAN",
                "speaker": "Supreme Court coverage",
                "symbols": ["SPY", "QQQ"],
                "market_broad_market": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = event_store.build_event_store(project_root, limit=20)

    assert payload["ok"] is True
    assert payload["event_count"] == 1
    assert payload["events"][0]["event_type"] == "live_macro_media_ingest"
    assert payload["events"][0]["category"] == "live_macro_media"


def test_point_in_time_event_store_infers_categories_when_missing(tmp_path) -> None:
    project_root = tmp_path / "project"
    event_dir = project_root / "governance" / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / "live_macro_events_20260401.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T19:00:00+00:00",
                        "event_type": "publish_from_media_ingest",
                        "source": "White House",
                        "speaker": "Donald Trump",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T15:00:00+00:00",
                        "event_type": "live_macro_media_ingest",
                        "source": "C-SPAN",
                        "speaker": "Supreme Court coverage",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T14:00:00+00:00",
                        "event_type": "premarket_token_guard_20260401",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T13:30:00+00:00",
                        "event_type": "options_surface_refresh",
                        "source": "options ingest",
                        "market_signal_types": ["gamma_flip_distance"],
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T13:15:00+00:00",
                        "event_type": "futures_roll_watch",
                        "source": "futures desk",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T13:00:00+00:00",
                        "event_type": "dividend_reinvest_policy",
                        "source": "income sleeve",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T12:45:00+00:00",
                        "event_type": "long_term_dca_rebalance",
                        "source": "allocation engine",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T12:30:00+00:00",
                        "event_type": "training_success_latest",
                        "source": "retrain controller",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T12:15:00+00:00",
                        "event_type": "data_retention_vacuum",
                        "source": "storage control",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-01T12:00:00+00:00",
                        "event_type": "sql_backpressure_watch",
                        "source": "sql_link",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = event_store.build_event_store(project_root, limit=20)

    categories = [row["category"] for row in payload["events"]]
    assert "policy_macro" in categories
    assert "legal_policy" in categories
    assert "broker_readiness" in categories
    assert "options_event" in categories
    assert "futures_event" in categories
    assert "dividend_event" in categories
    assert "long_term_allocation" in categories
    assert "training_control" in categories
    assert "storage_control" in categories
    assert "ingestion_control" in categories


def test_point_in_time_event_store_ingests_health_artifacts_and_context_syncs(tmp_path) -> None:
    project_root = tmp_path / "project"
    health_dir = project_root / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "ingestion_backpressure_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:00:00+00:00",
                "source": "sql linker",
                "notes": ["queue pressure elevated"],
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "storage_route_status_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:01:00+00:00",
                "source": "storage router",
                "warnings": ["failback pending"],
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "collector_contracts_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:02:00+00:00",
                "source": "collector contracts",
                "soft_failures": ["market_micro_context"],
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "source_verification_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:03:00+00:00",
                "source": "source verification",
                "notes": ["coverage drift"],
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "broker_truth_aggressive_equities_schwab_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:04:00+00:00",
                "source": "schwab",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "shadow_watchdog_halt_recovery_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:05:00+00:00",
                "source": "halt recovery",
                "errors": ["global_halt"],
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "sec_edgar_sync_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:06:00+00:00",
                "source": "SEC Edgar",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "fx_market_context_sync_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:07:00+00:00",
                "source": "FX context",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )

    payload = event_store.build_event_store(project_root, limit=40)

    categories = {row["category"] for row in payload["events"]}
    assert "ingestion_control" in categories
    assert "storage_control" in categories
    assert "source_quality" in categories
    assert "broker_readiness" in categories
    assert "tradeability" in categories
    assert "filing" in categories
    assert "fx_context" in categories


def test_point_in_time_event_store_dedupes_repeated_hourly_broker_events(tmp_path) -> None:
    project_root = tmp_path / "project"
    event_dir = project_root / "governance" / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / "premarket_token_guard_20260401.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"timestamp_utc": "2026-04-01T14:00:00+00:00", "event_type": "premarket_token_guard_20260401"}),
                json.dumps({"timestamp_utc": "2026-04-01T14:05:00+00:00", "event_type": "premarket_token_guard_20260401"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = event_store.build_event_store(project_root, limit=20)

    assert payload["event_count"] == 1
    assert payload["category_counts"]["broker_readiness"] == 1


def test_point_in_time_event_store_tracks_latest_by_category(tmp_path) -> None:
    project_root = tmp_path / "project"
    health_dir = project_root / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "service_control_plane_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:10:00+00:00",
                "source": "service_control_plane",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "provider_mesh_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-16T20:09:00+00:00",
                "source": "provider_mesh",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )

    payload = event_store.build_event_store(project_root, limit=20)

    assert payload["latest_by_category"]["control_plane"]["event_type"] == "service_control_plane_latest"


def test_point_in_time_event_store_quarantines_future_and_invalid_events(tmp_path) -> None:
    project_root = tmp_path / "project"
    event_dir = project_root / "governance" / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / "live_macro_events_20260810.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_utc": "2026-08-10T12:00:00+00:00",
                        "event_type": "valid_macro",
                        "source": "official",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-08-10T13:30:00+00:00",
                        "event_type": "future_macro",
                        "source": "bad_clock",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "not-a-timestamp",
                        "event_type": "invalid_macro",
                        "source": "malformed",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = event_store.build_event_store(
        project_root,
        limit=20,
        now=datetime(2026, 8, 10, 12, 30, tzinfo=timezone.utc),
    )

    contract = payload["point_in_time_contract"]
    assert payload["ok"] is False
    assert payload["event_count"] == 1
    assert payload["quarantined_event_count"] == 2
    assert contract["future_event_count"] == 1
    assert contract["invalid_timestamp_count"] == 1
    assert contract["point_in_time_only"] is False
    assert len(contract["source_manifest_sha256"]) == 64
