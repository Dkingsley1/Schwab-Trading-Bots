from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import source_verification_report as svr


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_build_source_verification_payload_classifies_sources(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "data_source_divergence_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "compared_buckets": 124,
            "worst_relative_spread": 0.01,
            "max_relative_spread": 0.03,
            "cross_profile": {"ok": False, "offenders": [{"symbol": "GS"}]},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "symbols_requested": 2,
            "symbols_with_chain": 2,
            "symbols_with_metrics": 2,
            "sources": {
                "polygon": {"ok": True},
                "unusual_whales_api": {"ok": True},
                "unusual_whales_export": {"ok": False},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "macro_crosscheck_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "passed_checks": 4,
            "total_checks": 4,
            "notes": ["official_treasury_fallback=html_page_parse"],
            "checks": {
                "artifacts_fresh": {"ok": True},
                "bls_dual_source": {"ok": True},
                "bea_dual_source": {"ok": True},
                "treasury_dual_source": {"ok": True},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "crypto_market_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "tracked_symbols": 7,
            "tracked_assets": 7,
            "ok_source_count": 6,
            "source_count": 7,
            "compared_assets": 4,
            "sources": {
                "deribit": {"ok": True},
                "kraken": {"ok": True},
                "hyperliquid": {"ok": True},
                "coinmetrics": {"ok": True},
                "defillama": {"ok": True},
                "etherscan": {"ok": True},
                "coingecko": {"ok": False},
            },
        },
    )
    _write_json(
        tmp_path / "exports" / "external_feeds" / "latest_status.json",
        {
            "timestamp_utc": fresh_ts,
            "bls": {"ok": True},
            "census": {"ok": True},
            "fred": {"ok": True, "warnings": ["gold request failed"]},
            "bea": {"ok": True},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "sources": {
                "federal_reserve": {"ok": True},
                "treasury": {"ok": True, "fallback": "html_page_parse"},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "schwab_education_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "ok_source_count": 6,
            "source_count": 6,
            "item_count": 54,
            "page_item_count": 22,
            "channel_item_count": 32,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "fx_market_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "ok_source_count": 2,
            "source_count": 2,
            "official_pairs": 4,
            "proxy_symbols_observed": 4,
            "proxy_agreement_norm": 0.71,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "sources": {
                "local_micro": {"ok": True, "symbol_count": 77},
                "treasury_auctions": {"ok": True, "rows": 12},
                "finra_short_volume": {"ok": True, "symbol_count": 79},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "sec_edgar_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "ticker_map_ok": True,
            "requested_symbols": 20,
            "resolved_symbols": 20,
            "tracked_symbols": 20,
            "error_count": 0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "extended_quant_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "tracked_symbols": 23,
            "sources": {
                "cftc_cot": {"ok": True},
                "nyfed_sofr": {"ok": True, "averages_error": "HTTP Error 400: Bad Request"},
                "cboe": {"ok": True},
                "nasdaq_threshold": {"ok": True},
            },
        },
    )

    payload = svr.build_source_verification_payload(tmp_path)

    counts = payload["overall"]["counts"]
    assert counts["cross_verified"] == 5
    assert counts["single_source_verified"] == 6
    assert counts["single_source_unverified"] == 0
    assert payload["overall"]["all_verified"] is True

    rows = {row["source_id"]: row for row in payload["sources"]}
    assert rows["market_quote_profiles"]["verification_status"] == "cross_verified"
    assert "cross_profile_residual_offenders=1" in rows["market_quote_profiles"]["notes"]
    assert rows["polygon_unusual_whales_options_context"]["verification_status"] == "cross_verified"
    assert rows["macro_crossstack"]["verification_status"] == "cross_verified"
    assert rows["crypto_market_context"]["verification_status"] == "cross_verified"
    assert rows["public_macro_feeds"]["verification_status"] == "single_source_verified"


def test_build_source_verification_payload_marks_stale_sources_unverified(tmp_path: Path) -> None:
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "data_source_divergence_latest.json",
        {"timestamp_utc": stale_ts, "ok": True, "cross_profile": {"ok": True, "offenders": []}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "options_flow_context_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "sources": {"polygon": {"ok": False}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "macro_crosscheck_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "passed_checks": 0, "total_checks": 4},
    )
    _write_json(
        tmp_path / "governance" / "health" / "crypto_market_context_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "ok_source_count": 1, "source_count": 7, "compared_assets": 0},
    )
    _write_json(
        tmp_path / "exports" / "external_feeds" / "latest_status.json",
        {"timestamp_utc": stale_ts, "bls": {"ok": False}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "sources": {"treasury": {"ok": False}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "schwab_education_context_sync_latest.json",
        {
            "timestamp_utc": stale_ts,
            "ok": False,
            "ok_source_count": 0,
            "source_count": 6,
            "item_count": 0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "fx_market_context_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "ok_source_count": 0, "source_count": 2, "official_pairs": 0, "proxy_symbols_observed": 0, "proxy_agreement_norm": 0.0},
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "sources": {"local_micro": {"ok": False}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "sec_edgar_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "ticker_map_ok": False, "error_count": 1},
    )
    _write_json(
        tmp_path / "governance" / "health" / "extended_quant_context_sync_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "sources": {"cftc_cot": {"ok": False}}},
    )

    payload = svr.build_source_verification_payload(tmp_path)

    assert payload["overall"]["all_verified"] is False
    assert payload["overall"]["counts"]["single_source_unverified"] == 11


def test_build_source_verification_payload_treats_export_only_options_flow_as_unverified(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "overall_status": "degraded",
            "context_profile": "unusual_whales_overlay_only",
            "symbols_requested": 1,
            "symbols_with_chain": 0,
            "symbols_with_metrics": 1,
            "coverage_score": 0.45,
            "coverage": {"polygon_backbone_ok": False, "context_profile": "unusual_whales_overlay_only"},
            "sources": {
                "polygon": {"ok": False, "errors": ["polygon_api_key_missing"]},
                "unusual_whales_api": {"ok": False},
                "unusual_whales_export": {"ok": True},
            },
        },
    )

    row = svr._options_flow_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert row["verification_status"] == "single_source_unverified"
    assert "context_profile=unusual_whales_overlay_only" in row["notes"]
    assert row["evidence"]["polygon_backbone_ok"] is False


def test_build_source_verification_payload_accepts_polygon_primary_options_flow_without_overlay(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "overall_status": "ready",
            "context_profile": "polygon_primary_only",
            "symbols_requested": 1,
            "symbols_with_chain": 1,
            "symbols_with_metrics": 1,
            "coverage_score": 0.9,
            "coverage": {"polygon_backbone_ok": True, "context_profile": "polygon_primary_only"},
            "sources": {
                "polygon": {"ok": True, "required": True, "contract_participates": True},
                "unusual_whales_api": {"ok": False, "required": False, "expected": False, "contract_participates": False},
                "unusual_whales_export": {"ok": False, "required": False, "expected": False, "contract_participates": False},
            },
        },
    )

    row = svr._options_flow_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert row["verification_status"] == "single_source_verified"
    assert "context_profile=polygon_primary_only" not in row["notes"]
    assert row["evidence"]["unusual_whales_expected"] is False


def test_build_source_verification_payload_tolerates_one_partial_source_for_macro_and_micro(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "sources": {
                "bls_calendar": {"ok": True},
                "federal_reserve_calendar": {"ok": False},
                "federal_reserve": {"ok": True},
                "treasury": {"ok": True},
                "bls": {"ok": True},
                "bea": {"ok": True},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "sources": {
                "local_micro": {"ok": True, "symbol_count": 77},
                "treasury_auctions": {"ok": True, "rows": 12},
                "finra_short_volume": {"ok": True, "symbol_count": 79},
                "nasdaq_trade_halts": {"ok": False, "rows": 0},
            },
        },
    )

    macro_row = svr._official_macro_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))
    micro_row = svr._market_micro_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert macro_row["verification_status"] == "single_source_verified"
    assert "partial_sources=5/6" in macro_row["notes"]
    assert micro_row["verification_status"] == "single_source_verified"
    assert "partial_sources=3/4" in micro_row["notes"]
