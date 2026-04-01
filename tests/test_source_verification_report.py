from __future__ import annotations

import json
import sys
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
    _write_json(
        tmp_path / "governance" / "health" / "data_source_divergence_latest.json",
        {
            "timestamp_utc": "2026-03-20T21:31:43+00:00",
            "ok": True,
            "compared_buckets": 124,
            "worst_relative_spread": 0.01,
            "max_relative_spread": 0.03,
            "cross_profile": {"ok": False, "offenders": [{"symbol": "GS"}]},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "tastytrade_context_sync_latest.json",
        {
            "timestamp_utc": "2026-03-20T23:01:34+00:00",
            "ok": True,
            "sandbox": True,
            "symbols_requested": 2,
            "symbols_with_chain": 2,
            "symbols_with_metrics": 0,
            "alignment_ok": True,
            "alignment_compared": 0,
            "alignment_reference_only": 2,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "macro_crosscheck_latest.json",
        {
            "timestamp_utc": "2026-03-20T22:10:00+00:00",
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
            "timestamp_utc": "2026-03-20T23:20:00+00:00",
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
            "timestamp_utc": "2026-03-20T17:58:23+00:00",
            "bls": {"ok": True},
            "census": {"ok": True},
            "fred": {"ok": True, "warnings": ["gold request failed"]},
            "bea": {"ok": True},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {
            "timestamp_utc": "2026-03-20T21:40:22+00:00",
            "ok": True,
            "sources": {
                "federal_reserve": {"ok": True},
                "treasury": {"ok": True, "fallback": "html_page_parse"},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": "2026-03-20T17:59:02+00:00",
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
            "timestamp_utc": "2026-03-20T21:00:56+00:00",
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
            "timestamp_utc": "2026-03-20T21:00:56+00:00",
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
    assert counts["cross_verified"] == 4
    assert counts["single_source_verified"] == 5
    assert counts["single_source_unverified"] == 0
    assert payload["overall"]["all_verified"] is True

    rows = {row["source_id"]: row for row in payload["sources"]}
    assert rows["market_quote_profiles"]["verification_status"] == "cross_verified"
    assert "cross_profile_residual_offenders=1" in rows["market_quote_profiles"]["notes"]
    assert rows["tastytrade_options_context"]["verification_status"] == "cross_verified"
    assert "reference_only_alignment" in rows["tastytrade_options_context"]["notes"]
    assert rows["macro_crossstack"]["verification_status"] == "cross_verified"
    assert rows["crypto_market_context"]["verification_status"] == "cross_verified"
    assert rows["public_macro_feeds"]["verification_status"] == "single_source_verified"


def test_build_source_verification_payload_marks_stale_sources_unverified(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "data_source_divergence_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": True, "cross_profile": {"ok": True, "offenders": []}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "tastytrade_context_sync_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "alignment_ok": False},
    )
    _write_json(
        tmp_path / "governance" / "health" / "macro_crosscheck_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "passed_checks": 0, "total_checks": 4},
    )
    _write_json(
        tmp_path / "governance" / "health" / "crypto_market_context_sync_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "ok_source_count": 1, "source_count": 7, "compared_assets": 0},
    )
    _write_json(
        tmp_path / "exports" / "external_feeds" / "latest_status.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "bls": {"ok": False}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "sources": {"treasury": {"ok": False}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "sources": {"local_micro": {"ok": False}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "sec_edgar_sync_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "ticker_map_ok": False, "error_count": 1},
    )
    _write_json(
        tmp_path / "governance" / "health" / "extended_quant_context_sync_latest.json",
        {"timestamp_utc": "2026-03-10T00:00:00+00:00", "ok": False, "sources": {"cftc_cot": {"ok": False}}},
    )

    payload = svr.build_source_verification_payload(tmp_path)

    assert payload["overall"]["all_verified"] is False
    assert payload["overall"]["counts"]["single_source_unverified"] == 9
