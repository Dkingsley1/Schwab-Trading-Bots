from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "data_intelligence_layer.py"
spec = importlib.util.spec_from_file_location("data_intelligence_layer", MODULE_PATH)
dil = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(dil)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_source_health(root: Path) -> None:
    health = root / "governance" / "health"
    rows = []
    for source_id in (
        "schwab_symbol_news",
        "market_quote_profiles",
        "market_micro_context",
        "crypto_market_context",
        "ticker_news_context",
    ):
        rows.append(
            {
                "source_id": source_id,
                "verification_status": "single_source_verified",
                "ok": True,
                "fresh": True,
                "source_confidence_score": 0.88,
            }
        )
    _write_json(
        health / "source_verification_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "overall": {
                "unverified_sources": [],
                "stale_sources": [],
                "mean_source_confidence_score": 0.88,
            },
            "sources": rows,
        },
    )
    _write_json(
        health / "crypto_market_context_sync_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "tracked_symbols": 12,
            "ok_source_count": 7,
            "source_count": 9,
            "compared_assets": 8,
            "news_row_count": 40,
        },
    )
    _write_json(
        health / "schwab_symbol_news_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "attempted_symbol_count": 120,
            "requested_symbol_count": 120,
            "symbols_with_news": 100,
            "total_news_items": 900,
            "coverage_ratio": 0.8333,
        },
    )
    _write_json(
        health / "ticker_news_context_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "symbols_with_news": 110,
            "requested_symbol_count": 120,
            "total_news_items": 1200,
            "ok_source_count": 3,
            "source_count": 3,
        },
    )
    _write_json(
        health / "coinbase_api_health_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "public_market_data": {
                "ok": True,
                "symbol": "BTC-USD",
                "latency_ms": 40.0,
                "snapshot_requested": True,
            },
        },
    )
    _write_json(health / "market_micro_sync_latest.json", {"ok": True, "overall_status": "ready"})


def _write_route_db(root: Path) -> Path:
    db = root / "data" / "jsonl_link.sqlite3"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY,
            ingested_at TEXT,
            source_broker TEXT,
            source_provider TEXT,
            source_venue TEXT,
            asset_class TEXT,
            routing_lane TEXT,
            source_quality_label TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    schwab = [
        (now, "schwab", "schwab", "schwab", "equities", "schwab_equities", "broker_native")
        for _ in range(600)
    ]
    coinbase = [
        (now, "coinbase", "coinbase", "coinbase", "crypto", "coinbase_crypto", "exchange_native")
        for _ in range(300)
    ]
    conn.executemany(
        """
        INSERT INTO jsonl_records (
            ingested_at, source_broker, source_provider, source_venue,
            asset_class, routing_lane, source_quality_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        schwab + coinbase,
    )
    conn.commit()
    conn.close()
    return db


def test_build_payload_scores_route_coverage_and_expands_when_calm(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = _write_route_db(tmp_path)

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
    )

    assert payload["overall_status"] == "ready"
    assert payload["route_coverage"]["rows_total"] == 900
    assert payload["route_coverage"]["by_family"]["schwab"] == 600
    assert payload["route_coverage"]["by_family"]["coinbase"] == 300
    assert payload["source_scorecards"]["schwab"]["overall_status"] == "ready"
    assert payload["source_scorecards"]["coinbase"]["overall_status"] == "ready"
    assert payload["degradation_intelligence"]["mode"] == "max"
    assert payload["degradation_intelligence"]["overall_status"] == "ready"
    assert payload["training_label_bridge"]["source_label_weighting_enabled"] is True
    assert payload["volume_plan"]["profile"] == "expanded"
    assert payload["volume_plan"]["override_env"]["CRYPTO_MARKET_CONTEXT_COINBASE_QUOTE_MAX_ASSETS"] == "36"
    assert any("crypto-market-sync" in row["shell"] for row in payload["volume_plan"]["collector_commands"])


def test_apply_writes_managed_data_volume_override(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = _write_route_db(tmp_path)
    override = tmp_path / "config" / ".env.data_intelligence_override"

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=override,
        apply=True,
    )

    assert payload["applied"] is True
    text = override.read_text(encoding="utf-8")
    assert "DATA_VOLUME_PROFILE=expanded" in text
    assert "DATA_DEGRADATION_INTELLIGENCE_MODE=max" in text
    assert "DATA_SOURCE_BAD_DATA_DOWNWEIGHT_ENABLED=1" in text
    assert "TRAINING_SOURCE_LABEL_WEIGHTING_ENABLED=1" in text
    assert "CRYPTO_MARKET_CONTEXT_MAX_SYMBOLS=36" in text
    assert "SCHWAB_SYMBOL_NEWS_LIMIT_PER_SYMBOL=80" in text
    assert "TICKER_NEWS_LIMIT_PER_SYMBOL=20" in text


def test_runtime_pressure_downshifts_volume_profile(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = _write_route_db(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "host_saturation_score": 72.0, "compute_pressure_level": "high"},
    )

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
        apply=True,
    )

    assert payload["volume_plan"]["profile"] == "conservative"
    assert payload["volume_plan"]["override_env"]["CRYPTO_MARKET_CONTEXT_COINBASE_QUOTE_MAX_ASSETS"] == "18"
    assert "host_saturation_high" in payload["volume_plan"]["profile_reasons"]


def test_advisory_runtime_with_clean_storage_allows_guarded_expanded_volume(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = _write_route_db(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "advisory",
            "host_saturation_score": 42.18,
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "effective_raw_live": {
                    "total_pending_lines": 0,
                    "source": "fresh_empty_sql_ingestion_overlay",
                },
                "effective_raw_live_source": "fresh_empty_sql_ingestion_overlay",
                "pending_lines_threshold": 15000,
            },
        },
    )

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
        apply=True,
    )

    assert payload["volume_plan"]["profile"] == "expanded"
    assert payload["volume_plan"]["override_env"]["DATA_VOLUME_PULLS_DEFERRED"] == "0"
    assert payload["volume_plan"]["override_env"]["CRYPTO_MARKET_CONTEXT_COINBASE_QUOTE_MAX_ASSETS"] == "36"
    assert "compute_pressure_level_high" not in payload["volume_plan"]["profile_reasons"]
    assert "compute_pressure_high_guarded_by_runtime_advisory_storage_clear" in payload["volume_plan"]["profile_reasons"]


def test_payload_json_fallback_infers_routes_before_sql_column_migration(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = tmp_path / "data" / "jsonl_link.sqlite3"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY,
            ingested_at TEXT,
            source_rel TEXT,
            payload_json TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO jsonl_records (ingested_at, source_rel, payload_json) VALUES (?, ?, ?)",
        [
            (now, "logs/schwab_equities/live.jsonl", json.dumps({"symbol": "SPY"})),
            (now, "logs/coinbase_crypto/live.jsonl", json.dumps({"symbol": "BTC-USD"})),
            (now, "exports/external_context/crypto_market_context.jsonl", json.dumps({"symbol": "ETH-USD"})),
            (
                now,
                "governance/events/signal_generation_20260624.jsonl",
                json.dumps(
                    {
                        "event": "signal_generation",
                        "signal_quality": "bad_signal",
                        "source_path": str(
                            tmp_path
                            / "decisions"
                            / "shadow_intraday_aggressive_equities"
                            / "trade_decisions_20260624.jsonl"
                        ),
                        "symbol": "VGIT",
                        "strategy": "master_futures_bot",
                    }
                ),
            ),
        ],
    )
    conn.commit()
    conn.close()

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
    )

    assert payload["route_coverage"]["coverage_mode"] == "payload_json_source_rel_inference"
    assert payload["route_coverage"]["by_family"]["schwab"] == 2
    assert payload["route_coverage"]["by_family"]["coinbase"] == 1
    assert payload["route_coverage"]["by_family"]["crypto"] == 1
    assert payload["route_coverage"]["by_asset_class"]["equities"] == 2
    assert "route_label_columns_missing:source_broker,source_provider,source_venue,asset_class,routing_lane,source_quality_label" in payload["route_coverage"]["coverage_gaps"]


def test_route_coverage_merges_materialized_rows_with_payload_fallback(tmp_path: Path, monkeypatch) -> None:
    _write_source_health(tmp_path)
    monkeypatch.setenv("DATA_INTELLIGENCE_ENABLE_LARGE_DB_PAYLOAD_INFERENCE", "1")
    monkeypatch.setenv("DATA_INTELLIGENCE_PAYLOAD_INFERENCE_DB_SIZE_GB", "0.000001")
    db = tmp_path / "data" / "jsonl_link.sqlite3"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY,
            ingested_at TEXT,
            source_rel TEXT,
            payload_json TEXT,
            source_broker TEXT,
            source_provider TEXT,
            source_venue TEXT,
            asset_class TEXT,
            routing_lane TEXT,
            source_quality_label TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO jsonl_records (
            ingested_at, source_rel, payload_json, source_broker, source_provider,
            source_venue, asset_class, routing_lane, source_quality_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                now,
                "governance/channels/api/default_crypto_coinbase/api_20260624.jsonl",
                json.dumps({"symbol": "BTC-USD"}),
                "coinbase",
                "coinbase",
                "coinbase",
                "crypto",
                "coinbase_crypto",
                "exchange_native",
            ),
            (
                now,
                "governance/events/signal_generation_20260624.jsonl",
                json.dumps(
                    {
                        "event": "signal_generation",
                        "source_path": str(
                            tmp_path
                            / "decisions"
                            / "shadow_intraday_aggressive_equities"
                            / "trade_decisions_20260624.jsonl"
                        ),
                        "symbol": "VGIT",
                    }
                ),
                "",
                "",
                "",
                "",
                "",
                "",
            ),
        ],
    )
    conn.commit()
    conn.close()

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
    )

    assert payload["route_coverage"]["coverage_mode"] == "materialized_route_columns_latest_known_id_sample_plus_payload_json_source_rel_inference"
    assert payload["route_coverage"]["by_family"]["schwab"] == 1
    assert payload["route_coverage"]["by_family"]["coinbase"] == 1
    assert payload["source_scorecards"]["coinbase"]["route_rows"] == 1


def test_severe_backpressure_defers_heavy_pulls(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = _write_route_db(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {"total_pending_lines": 70000, "pending_lines_threshold": 15000},
        },
    )

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
        apply=True,
    )

    assert payload["overall_status"] == "degraded"
    assert payload["volume_plan"]["profile"] == "deferred"
    assert payload["volume_plan"]["override_env"]["DATA_VOLUME_PULLS_DEFERRED"] == "1"
    assert len(payload["volume_plan"]["collector_commands"]) == 2


def test_degradation_intelligence_downweights_degraded_source_suspects(tmp_path: Path) -> None:
    _write_source_health(tmp_path)
    db = _write_route_db(tmp_path)
    health = tmp_path / "governance" / "health"
    source_payload = json.loads((health / "source_verification_latest.json").read_text(encoding="utf-8"))
    source_payload["sources"].append(
        {
            "source_id": "public_macro_feeds",
            "verification_status": "single_source_verified",
            "ok": True,
            "fresh": True,
            "source_confidence_score": 0.60,
            "notes": [
                "partial_sources=1/4",
                "fred_warnings=9",
                "official_macro_context_verified_partial_public_feeds",
            ],
            "evidence": {
                "ok_sources": 1,
                "total_sources": 4,
                "official_macro_context_verified_partial_public_feeds": True,
            },
        }
    )
    _write_json(health / "source_verification_latest.json", source_payload)

    payload = dil.build_payload(
        tmp_path,
        db_path=db,
        out_path=tmp_path / "governance" / "health" / "data_intelligence_layer_latest.json",
        override_path=tmp_path / "config" / ".env.data_intelligence_override",
    )

    degradation = payload["degradation_intelligence"]
    suspects = {row["source_id"]: row for row in degradation["degraded_source_suspects"]}
    assert degradation["overall_status"] == "degraded"
    assert "public_macro_feeds" in suspects
    assert suspects["public_macro_feeds"]["label_weight"] <= 0.55
    assert payload["training_label_bridge"]["source_label_weight_overrides"]["public_macro_feeds"] <= 0.55
