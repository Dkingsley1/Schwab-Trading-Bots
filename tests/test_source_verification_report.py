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


def test_free_equity_reference_refresh_command_is_bounded(tmp_path: Path) -> None:
    command = svr._refresh_command_for_source(tmp_path, "free_equity_reference_context")

    assert command[:2] == [str(tmp_path / "scripts" / "ops" / "opsctl.sh"), "free-equity-reference-sync"]
    assert command[command.index("--max-symbols") + 1] == "40"
    assert command[command.index("--timeout") + 1] == "2.5"
    assert command[command.index("--max-runtime-seconds") + 1] == "45"


def _write_fed_2026_stress_scenario_files(root: Path) -> None:
    scenario_dir = root / "config" / "stress_scenarios"
    _write_json(
        scenario_dir / "fed_2026_supervisory_severely_adverse.json",
        {
            "schema_version": 1,
            "scenario_id": "fed_2026_supervisory_severely_adverse",
            "source": {
                "publisher": "Board of Governors of the Federal Reserve System",
                "url": "https://www.federalreserve.gov/publications/2026-stress-test-scenarios.htm",
                "retrieved_date": "2026-05-01",
            },
            "key_stress_anchors": {"unemployment_peak_pct": 10.0, "vix_peak": 72.0},
            "domestic_variables": {
                "columns": ["date", "unemployment_rate", "market_volatility_index"],
                "rows": [["2026-Q1", 5.9, 59.7], ["2026-Q2", 7.2, 72.0]],
            },
            "international_variables": {
                "columns": ["date", "euro_area_real_gdp_growth"],
                "rows": [["2026-Q1", -8.6], ["2026-Q2", -8.5]],
            },
        },
    )
    _write_json(
        scenario_dir / "fed_2026_stress_modules.json",
        {
            "schema_version": 1,
            "module_map_id": "fed_2026_supervisory_stress_modules",
            "scenario_id": "fed_2026_supervisory_severely_adverse",
            "module_count": 10,
            "source": {
                "publisher": "Board of Governors of the Federal Reserve System",
                "url": "https://www.federalreserve.gov/publications/2026-stress-test-scenarios.htm",
                "retrieved_date": "2026-05-01",
            },
            "usage_policy": {"direct_execution_allowed": False},
            "stress_modules": [
                {"module_id": "fed_2026_equity_crash_volatility_spike", "primary_series": ["dow_jones_total_stock_market_index"], "internal_feature_keys": ["fed_2026_equity_crash_vol_spike_norm"]},
                {"module_id": "fed_2026_corporate_credit_spread_blowout", "primary_series": ["bbb_corporate_yield"], "internal_feature_keys": ["fed_2026_credit_spread_blowout_norm"]},
                {"module_id": "fed_2026_housing_price_shock", "primary_series": ["house_price_index"], "internal_feature_keys": ["fed_2026_housing_price_shock_norm"]},
                {"module_id": "fed_2026_commercial_real_estate_shock", "primary_series": ["commercial_real_estate_price_index"], "internal_feature_keys": ["fed_2026_cre_price_shock_norm"]},
                {"module_id": "fed_2026_unemployment_recession_shock", "primary_series": ["unemployment_rate"], "internal_feature_keys": ["fed_2026_unemployment_recession_norm"]},
                {"module_id": "fed_2026_global_recession_deflation_shock", "primary_series": ["euro_area_real_gdp_growth"], "internal_feature_keys": ["fed_2026_global_recession_deflation_norm"]},
                {"module_id": "fed_2026_commodity_inflation_shock", "primary_series": ["cpi_inflation_rate"], "internal_feature_keys": ["fed_2026_commodity_inflation_shock_norm"]},
                {"module_id": "fed_2026_treasury_yield_shock", "primary_series": ["ten_year_treasury_yield"], "internal_feature_keys": ["fed_2026_treasury_yield_shock_norm"]},
                {"module_id": "fed_2026_us_dollar_stress", "primary_series": ["euro_area_usd_per_euro"], "internal_feature_keys": ["fed_2026_usd_stress_norm"]},
                {"module_id": "fed_2026_counterparty_default_contagion_shock", "primary_series": ["market_volatility_index"], "internal_feature_keys": ["fed_2026_counterparty_default_contagion_norm"]},
            ],
        },
    )
    _write_json(
        scenario_dir / "fed_2026_source_plumbing.json",
        {
            "schema_version": 1,
            "plumbing_id": "fed_2026_supervisory_source_plumbing",
            "scenario_id": "fed_2026_supervisory_severely_adverse",
            "series_map": {
                "domestic_variables": {
                    "unemployment_rate": "fed_2026_unemployment_stress_norm",
                    "market_volatility_index": "fed_2026_vix_stress_norm",
                },
                "international_variables": {
                    "euro_area_real_gdp_growth": "fed_2026_euro_area_growth_norm",
                },
            },
            "market_proxy_symbols": {"equity_beta": ["SPY"], "volatility": ["VIXY"]},
            "stress_module_map": {
                "fed_2026_equity_crash_volatility_spike": "fed_2026_equity_crash_vol_spike_norm",
                "fed_2026_corporate_credit_spread_blowout": "fed_2026_credit_spread_blowout_norm",
                "fed_2026_housing_price_shock": "fed_2026_housing_price_shock_norm",
                "fed_2026_commercial_real_estate_shock": "fed_2026_cre_price_shock_norm",
                "fed_2026_unemployment_recession_shock": "fed_2026_unemployment_recession_norm",
                "fed_2026_global_recession_deflation_shock": "fed_2026_global_recession_deflation_norm",
                "fed_2026_commodity_inflation_shock": "fed_2026_commodity_inflation_shock_norm",
                "fed_2026_treasury_yield_shock": "fed_2026_treasury_yield_shock_norm",
                "fed_2026_us_dollar_stress": "fed_2026_usd_stress_norm",
                "fed_2026_counterparty_default_contagion_shock": "fed_2026_counterparty_default_contagion_norm",
            },
            "internal_feature_keys": [
                "quant_macro_stress_2026_driver_norm",
                "quant_fed_2026_scenario_integrity_norm",
                "quant_fed_2026_equity_crash_vol_spike_norm",
                "quant_fed_2026_credit_spread_blowout_norm",
                "fed_2026_source_plumbing_map",
            ],
            "governance_targets": ["source_verification", "point_in_time_event_store", "replay_hash_registry"],
        },
    )


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
        tmp_path / "governance" / "health" / "free_equity_reference_context_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "requested_symbol_count": 5,
            "symbols_with_reference": 5,
            "sources": {
                "yahoo_chart": {"ok": True},
                "nasdaq_quote": {"ok": True},
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
    _write_json(
        tmp_path / "governance" / "health" / "public_policy_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "context_profile": "official_free_public_policy_liquidity",
            "ok_source_count": 3,
            "source_count": 3,
            "countries": ["USA", "CHN", "JPN", "DEU", "GBR"],
            "sources": {
                "treasury_debt_to_penny": {"ok": True, "record_date": "2026-06-11"},
                "treasury_avg_interest_rates": {"ok": True, "record_date": "2026-05-31"},
                "world_bank_indicators": {
                    "ok": True,
                    "lastupdated": "2026-04-08",
                    "indicator_success_count": 5,
                    "value_count": 25,
                },
            },
            "features": {
                "us_public_debt_to_worldbank_gdp_proxy": 1.36,
                "treasury_avg_interest_rate_pct": 3.46,
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "schwab_symbol_news_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "overall_status": "ready",
            "auth_ok": True,
            "requested_symbol_count": 500,
            "attempted_symbol_count": 500,
            "symbols_with_news": 37,
            "total_news_items": 92,
            "coverage_ratio": 0.074,
            "method_counts": {"get_news": 500},
            "source_counts": {"Schwab Network": 92},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ticker_news_context_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "overall_status": "ready",
            "requested_symbol_count": 500,
            "symbols_with_news": 112,
            "total_news_items": 420,
            "coverage_ratio": 0.224,
            "ok_source_count": 5,
            "source_count": 6,
            "source_counts": {"Yahoo Finance": 330, "CoinDesk": 44, "Schwab Network": 46},
            "sources": {
                "yahoo_finance_symbol_rss": {"ok": True},
                "coindesk": {"ok": True},
                "cointelegraph": {"ok": True},
                "decrypt": {"ok": True},
                "the_block": {"ok": True},
                "bitcoin_magazine": {"ok": False},
            },
        },
    )
    _write_fed_2026_stress_scenario_files(tmp_path)

    payload = svr.build_source_verification_payload(tmp_path)

    counts = payload["overall"]["counts"]
    assert counts["cross_verified"] == 5
    assert counts["single_source_verified"] == 11
    assert counts["single_source_unverified"] == 0
    assert payload["overall"]["all_verified"] is True

    rows = {row["source_id"]: row for row in payload["sources"]}
    assert rows["market_quote_profiles"]["verification_status"] == "cross_verified"
    assert "cross_profile_residual_offenders=1" in rows["market_quote_profiles"]["notes"]
    assert rows["options_context_mesh"]["verification_status"] == "cross_verified"
    assert "polygon_unusual_whales_options_context" in rows["options_context_mesh"]["aliases"]
    assert rows["options_context_mesh"]["source_confidence_score"] > 0.8
    assert rows["macro_crossstack"]["verification_status"] == "cross_verified"
    assert rows["crypto_market_context"]["verification_status"] == "cross_verified"
    assert rows["free_equity_reference_context"]["verification_status"] == "single_source_verified"
    assert rows["public_macro_feeds"]["verification_status"] == "single_source_verified"
    assert rows["schwab_symbol_news"]["verification_status"] == "single_source_verified"
    assert rows["ticker_news_context"]["verification_status"] == "single_source_verified"
    assert rows["public_policy_context"]["verification_status"] == "single_source_verified"
    assert rows["public_policy_context"]["evidence"]["world_bank_value_count"] == 25
    assert rows["fed_2026_supervisory_stress_scenario"]["verification_status"] == "single_source_verified"
    assert rows["fed_2026_supervisory_stress_scenario"]["evidence"]["internal_feature_count"] >= 3
    assert rows["fed_2026_supervisory_stress_scenario"]["evidence"]["stress_module_count"] == 10
    assert rows["fed_2026_supervisory_stress_scenario"]["evidence"]["stress_module_map_count"] == 10
    assert payload["source_confidence_summary"]["low_confidence_source_count"] == 0


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
        tmp_path / "governance" / "health" / "free_equity_reference_context_latest.json",
        {"timestamp_utc": stale_ts, "ok": False, "symbols_with_reference": 0, "sources": {"yahoo_chart": {"ok": False}}},
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
    assert payload["overall"]["counts"]["single_source_unverified"] == 16


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


def test_options_flow_verification_accepts_intentionally_unconfigured_optional_credentials(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": False,
            "overall_status": "blocked",
            "auth_issue": "options_flow_credentials_missing",
            "operator_action_required": True,
            "symbols_requested": 2,
            "symbols_with_chain": 0,
            "symbols_with_metrics": 0,
            "sources": {
                "polygon": {"ok": False, "required": True, "expected": True, "contract_participates": True},
                "unusual_whales_api": {"ok": False, "required": False, "expected": False, "contract_participates": False},
                "unusual_whales_export": {
                    "ok": False,
                    "required": False,
                    "expected": False,
                    "configured": False,
                    "contract_participates": False,
                },
            },
        },
    )

    row = svr._options_flow_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert row["verification_status"] == "single_source_verified"
    assert row["ok"] is True
    assert "optional_options_flow_credentials_not_configured" in row["notes"]
    assert row["evidence"]["optional_unconfigured"] is True
    assert svr._row_has_actionable_notes(row) is False


def test_options_flow_verification_accepts_free_option_chain_profile(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "overall_status": "ready",
            "context_profile": "free_options_chain_only",
            "symbols_requested": 2,
            "symbols_with_chain": 2,
            "symbols_with_metrics": 2,
            "coverage": {
                "context_profile": "free_options_chain_only",
                "free_options_chain_ok": True,
                "polygon_backbone_ok": False,
            },
            "sources": {
                "polygon": {"ok": False, "required": True, "expected": True, "contract_participates": True},
                "yahoo_options_chain": {"ok": True, "symbol_count": 2, "contract_participates": True},
                "cboe_delayed_options": {"ok": False, "contract_participates": True},
                "unusual_whales_api": {"ok": False, "expected": False, "contract_participates": False},
                "unusual_whales_export": {"ok": False, "expected": False, "contract_participates": False},
            },
        },
    )

    row = svr._options_flow_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert row["verification_status"] == "single_source_verified"
    assert row["evidence"]["free_options_chain_ok"] is True
    assert row["evidence"]["options_backbone_ok"] is True
    assert "polygon_api_key_missing" not in row["notes"]
    assert "context_profile=free_options_chain_only" not in row["notes"]
    assert svr._row_has_actionable_notes(row) is False


def test_cross_verified_crypto_source_warnings_are_not_actionable() -> None:
    row = {
        "verification_status": "cross_verified",
        "notes": ["partial_sources=16/18", "source_warnings=1"],
        "evidence": {"warning_count": 1},
    }

    assert svr._row_has_actionable_notes(row) is False


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


def test_market_micro_verification_accepts_nasdaq_when_finra_is_degraded(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "sources": {
                "local_micro": {"ok": True, "symbol_count": 5},
                "treasury_auctions": {"ok": True, "rows": 12},
                "finra_short_volume": {"ok": False, "symbol_count": 0},
                "nasdaq_trade_halts": {"ok": True, "rows": 0},
            },
        },
    )

    micro_row = svr._market_micro_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert micro_row["verification_status"] == "single_source_verified"
    assert micro_row["evidence"]["critical_sources"]["external_micro_reference"] is True
    assert micro_row["evidence"]["critical_sources"]["finra_short_volume"] is False


def test_market_micro_verification_accepts_market_closed_local_micro_fallback(tmp_path: Path) -> None:
    sunday_utc = datetime(2026, 5, 24, 15, 30, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": sunday_utc.isoformat(),
            "ok": True,
            "sources": {
                "local_micro": {"ok": False, "symbol_count": 0},
                "treasury_auctions": {"ok": True, "rows": 12},
                "finra_short_volume": {"ok": True, "symbol_count": 392},
                "nasdaq_trade_halts": {"ok": True, "rows": 0, "contract_participates": False},
            },
        },
    )

    micro_row = svr._market_micro_row(tmp_path / "governance" / "health", sunday_utc)

    assert micro_row["verification_status"] == "single_source_verified"
    assert "local_micro_absent_market_closed" in micro_row["notes"]
    assert micro_row["evidence"]["market_closed_local_micro_fallback"] is True
    assert micro_row["evidence"]["effective_ok_sources"] == 3


def test_market_micro_verification_accepts_holiday_pause_fallback(tmp_path: Path) -> None:
    holiday_utc = datetime(2026, 5, 25, 17, 30, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": holiday_utc.isoformat(),
            "ok": True,
            "sources": {
                "local_micro": {"ok": False, "symbol_count": 0},
                "treasury_auctions": {"ok": True, "rows": 12},
                "finra_short_volume": {"ok": True, "symbol_count": 392},
                "nasdaq_trade_halts": {"ok": True, "rows": 0, "contract_participates": False},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "data_ingress_latest_default_equities_schwab.json",
        {
            "timestamp_utc": holiday_utc.isoformat(),
            "loop_state": "paused_session_gate",
            "pause_reason": "holiday",
        },
    )

    micro_row = svr._market_micro_row(tmp_path / "governance" / "health", holiday_utc)

    assert micro_row["verification_status"] == "single_source_verified"
    assert "local_micro_absent_market_closed" in micro_row["notes"]
    assert micro_row["evidence"]["holiday_pause_observed"] is True


def test_fx_market_verification_accepts_holiday_official_rate_fallback(tmp_path: Path) -> None:
    holiday_utc = datetime(2026, 5, 25, 17, 30, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "governance" / "health" / "fx_market_context_sync_latest.json",
        {
            "timestamp_utc": holiday_utc.isoformat(),
            "ok": True,
            "ok_source_count": 4,
            "source_count": 5,
            "official_pairs": 6,
            "proxy_symbols_observed": 0,
            "proxy_agreement_norm": 0.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "data_ingress_latest_default_equities_schwab.json",
        {
            "timestamp_utc": holiday_utc.isoformat(),
            "loop_state": "paused_session_gate",
            "pause_reason": "holiday",
        },
    )

    fx_row = svr._fx_market_row(tmp_path / "governance" / "health", holiday_utc)

    assert fx_row["verification_status"] == "single_source_verified"
    assert "market_proxy_absent_market_closed" in fx_row["notes"]
    assert fx_row["evidence"]["official_rate_only_holiday_fallback"] is True
    assert svr._row_has_actionable_notes(fx_row) is False


def test_fx_market_verification_counts_official_sources_when_skip_payload_has_null_totals(tmp_path: Path) -> None:
    market_open_utc = datetime(2026, 6, 23, 16, 0, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "governance" / "health" / "fx_market_context_sync_latest.json",
        {
            "timestamp_utc": market_open_utc.isoformat(),
            "ok": True,
            "skipped": True,
            "proxy_symbols_observed": 0,
            "sources": {
                "ecb": {"ok": True, "rows": 61},
                "fed_h10": {"ok": True, "pair_count": 6},
                "macro_cross_asset": {"ok": False},
                "market_proxy": {"ok": False, "symbols": 0},
                "twelve_data": {"ok": False, "configured": True, "pairs_ok": 0},
            },
        },
    )

    fx_row = svr._fx_market_row(tmp_path / "governance" / "health", market_open_utc)

    assert fx_row["verification_status"] == "single_source_verified"
    assert fx_row["evidence"]["ok_sources"] == 2
    assert fx_row["evidence"]["total_sources"] == 5
    assert fx_row["evidence"]["official_pairs"] == 6
    assert "official_reference_rates_only_direct_fx_unavailable" in fx_row["notes"]
    assert svr._row_has_actionable_notes(fx_row) is False


def test_public_macro_feeds_use_official_macro_as_authoritative_fallback(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "exports" / "external_feeds" / "latest_status.json",
        {
            "timestamp_utc": fresh_ts,
            "bls": {"ok": True},
            "census": {"ok": False},
            "fred": {"ok": False, "warnings": ["GDP failed", "UNRATE failed"]},
            "bea": {"ok": False},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "sources": {
                "bls_calendar": {"ok": True},
                "federal_reserve_calendar": {"ok": True},
                "federal_reserve": {"ok": True},
                "treasury": {"ok": True},
                "bls": {"ok": True},
                "bea": {"ok": True},
            },
        },
    )

    row = svr._external_feeds_row(tmp_path, datetime.now(timezone.utc))

    assert row["verification_status"] == "single_source_verified"
    assert row["ok"] is True
    assert "partial_sources=1/4" in row["notes"]
    assert "official_macro_context_verified_partial_public_feeds" in row["notes"]
    assert row["evidence"]["official_macro_context_verified_partial_public_feeds"] is True
    assert row["evidence"]["raw_public_ok_sources"] == 1
    assert row["evidence"]["raw_public_total_sources"] == 4
    assert row["evidence"]["effective_ok_sources"] >= row["evidence"]["official_macro_min_ok_sources_required"]
    assert row["evidence"]["effective_ok_sources"] > row["evidence"]["raw_public_ok_sources"]
    assert row["evidence"]["effective_total_sources"] >= row["evidence"]["effective_ok_sources"]
    assert svr._row_has_actionable_notes(row) is False


def test_public_policy_context_accepts_partial_world_bank_with_treasury_policy_core(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "public_policy_context_sync_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": False,
            "context_profile": "official_free_public_policy_liquidity",
            "countries": ["USA", "CHN", "JPN", "DEU", "GBR"],
            "sources": {
                "treasury_debt_to_penny": {"ok": True, "record_date": "2026-06-18"},
                "treasury_avg_interest_rates": {"ok": True, "record_date": "2026-05-31"},
                "world_bank_indicators": {
                    "ok": False,
                    "indicator_count": 5,
                    "indicator_success_count": 4,
                    "value_count": 29,
                    "lastupdated": "2026-04-08",
                    "errors": {"BN.CAB.XOKA.GD.ZS": "The read operation timed out"},
                },
            },
            "features": {
                "us_public_debt_to_worldbank_gdp_proxy": 1.36,
                "treasury_avg_interest_rate_pct": 3.31,
            },
        },
    )

    row = svr._public_policy_context_row(tmp_path / "governance" / "health", datetime.now(timezone.utc))

    assert row["verification_status"] == "single_source_verified"
    assert row["ok"] is True
    assert row["evidence"]["effective_ok_sources"] == 3
    assert row["evidence"]["world_bank_partial_verified"] is True
    assert "world_bank_indicators_partial=4/5" in row["notes"]
    assert svr._row_has_actionable_notes(row) is False
