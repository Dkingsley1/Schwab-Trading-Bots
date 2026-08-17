from __future__ import annotations

from datetime import datetime, timezone

from core.global_central_bank_context import (
    CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    assess_central_bank_cross_source_context,
)
from scripts.synchronize_global_central_bank_context import build_central_bank_cross_source_context


NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)


def _global_context() -> dict:
    return {
        "timestamp_utc": NOW.isoformat(),
        "contract": {
            "tier_1_minimum_ratio": 1.0,
            "important_bank_minimum_ratio": 1.0,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
        "methodology": {
            "point_in_time_only": True,
            "missing_values_are_not_zero_filled": True,
        },
        "coverage": {
            "tier_1_coverage_ratio": 1.0,
            "important_bank_coverage_ratio": 1.0,
            "future_observation_selected": False,
            "source_failures": [],
        },
        "global_features": {key: 0.5 for key in GLOBAL_CENTRAL_BANK_FEATURE_KEYS},
        "banks": {
            "european_central_bank": {
                "bank_id": "european_central_bank",
                "name": "European Central Bank",
                "bis_area_code": "XM",
                "world_bank_area_code": "EMU",
                "currency": "EUR",
                "tier": 1,
                "weight": 1.0,
                "region": "europe",
                "groups": ["g5", "advanced"],
                "bot_domains": ["rates", "fx", "global_macro"],
                "official_policy_url": "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html",
                "policy_rate": {
                    "rate_percent": 3.0,
                    "change_bps_30d": -25.0,
                    "change_bps_90d": -50.0,
                    "days_since_last_change": 14,
                    "observation_date": "2026-08-14",
                    "fresh": True,
                },
                "balance_sheet": {
                    "total_assets_usd_billions": 7900.0,
                    "quarter_over_quarter_change_pct": 1.2,
                    "year_over_year_change_pct": -1.0,
                    "observation_date": "2026-06-30",
                    "fresh": True,
                },
                "ready": True,
            }
        },
    }


def _fx_context(*, reference_date: str = "2026-08-14", severity: str = "none") -> dict:
    return {
        "timestamp_utc": NOW.isoformat(),
        "derived": {
            "currency_reference_rates": {
                "EUR": {
                    "date": reference_date,
                    "units_per_eur": 1.0,
                }
            },
            "currency_reference_changes": {"EUR": 0.01},
            "canonical_reconciliation": {
                "EURUSD": {"divergence_severity": severity},
            },
        },
    }


def _public_policy(*, period: str = "2025") -> dict:
    return {
        "timestamp_utc": NOW.isoformat(),
        "sources": {
            "world_bank_indicators": {
                "indicators": {
                    "inflation_cpi_annual_pct": {
                        "values": {"EMU": {"value": 2.2, "date": period, "country_name": "Euro area"}}
                    },
                    "current_account_pct_gdp": {
                        "values": {"EMU": {"value": 2.5, "date": period, "country_name": "Euro area"}}
                    },
                }
            }
        },
    }


def _macro_cross_asset() -> dict:
    return {
        "timestamp_utc": NOW.isoformat(),
        "cross_asset": {"vix": 20.0, "high_yield_oas_bps": 3.5},
    }


def _build(*, fx: dict | None = None, public_policy: dict | None = None, macro: dict | None = None) -> dict:
    return build_central_bank_cross_source_context(
        global_central_banks=_global_context(),
        fx_market=fx or {},
        public_policy=public_policy or {},
        official_macro={},
        macro_cross_asset=macro or {},
        central_bank_liquidity={},
        now=NOW,
    )


def test_cross_source_router_joins_lineage_and_symbol_routes() -> None:
    payload = _build(fx=_fx_context(), public_policy=_public_policy(), macro=_macro_cross_asset())
    bank = payload["banks"]["european_central_bank"]

    assert bank["synchronized_ready"] is True
    assert set(bank["joined_source_ids"]) == {"fx_market_context", "macro_cross_asset", "public_policy_context"}
    assert bank["cross_source_link_count"] >= 3
    assert all(link["point_in_time"] for link in bank["lineage"])
    policy_lineage = next(link for link in bank["lineage"] if link["dimension"] == "policy_rate")
    assert policy_lineage["source_url"] == "https://data.bis.org/topics/CBPOL"
    assert policy_lineage["publisher_url"].startswith("https://www.ecb.europa.eu/")
    assert payload["routing"]["symbol_to_bank_ids"]["FXE"] == ["european_central_bank"]
    assert payload["symbol_features"]["FXE"]["central_bank_sync_fx_coverage_norm"] == 1.0
    assert payload["symbol_features"]["FXE"]["central_bank_sync_macro_coverage_norm"] == 1.0
    assert set(payload["global_features"]) == set(CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS)
    assert assess_central_bank_cross_source_context(payload, now_utc=NOW)["ready"] is True
    assert payload["contract"]["live_execution_authority"] is False


def test_raw_bank_record_alone_cannot_claim_synchronization() -> None:
    payload = _build()
    bank = payload["banks"]["european_central_bank"]

    assert bank["cross_source_link_count"] == 0
    assert bank["synchronized_ready"] is False
    assert payload["coverage"]["synchronized_bank_coverage_ratio"] == 0.0
    assert payload["coverage"]["banks_without_distinct_cross_source"] == ["european_central_bank"]
    assert "synchronized_bank_coverage_below_contract" in assess_central_bank_cross_source_context(
        payload, now_utc=NOW
    )["reasons"]


def test_future_observation_is_excluded_from_routes() -> None:
    payload = _build(fx=_fx_context(reference_date="2026-08-16"))
    bank = payload["banks"]["european_central_bank"]
    fx_lineage = next(link for link in bank["lineage"] if link["dimension"] == "fx_transmission")

    assert fx_lineage["observation_in_future"] is True
    assert fx_lineage["point_in_time"] is False
    assert bank["synchronized_ready"] is False
    assert payload["coverage"]["future_observations_excluded"] == {
        "european_central_bank": ["fx_transmission"]
    }
    assert "FXE" not in payload["routing"]["symbol_to_bank_ids"]


def test_hard_provider_conflict_blocks_affected_bank_and_context() -> None:
    payload = _build(fx=_fx_context(severity="high"), public_policy=_public_policy())
    bank = payload["banks"]["european_central_bank"]
    assessment = assess_central_bank_cross_source_context(payload, now_utc=NOW)

    assert bank["synchronized_ready"] is False
    assert bank["hard_conflicts"] == ["european_central_bank:fx_provider_divergence:EURUSD"]
    assert "FXE" not in payload["routing"]["symbol_to_bank_ids"]
    assert assessment["ready"] is False
    assert "hard_source_conflicts_present" in assessment["reasons"]


def test_missing_macro_values_remain_omitted_with_coverage_companion() -> None:
    payload = _build(fx=_fx_context())
    bank = payload["banks"]["european_central_bank"]

    assert "sovereign_macro" not in bank["usable_features"]
    assert "policy_inflation_alignment_norm" not in bank["derived_features"]
    assert payload["symbol_features"]["FXE"]["central_bank_sync_macro_coverage_norm"] == 0.0
