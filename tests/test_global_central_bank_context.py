from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone

from core.global_central_bank_context import (
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    assess_global_central_bank_context,
)
from scripts.collect_global_central_bank_context import build_global_central_bank_context


def _series_xml(rows: list[tuple[str, list[tuple[str, float]]]]) -> str:
    series = []
    for area, observations in rows:
        obs = "".join(
            f'<Obs TIME_PERIOD="{period}" OBS_VALUE="{value}" />'
            for period, value in observations
        )
        series.append(
            f'<Series REF_AREA="{area}" FREQ="D" SOURCE_REF="official" '
            f'TITLE="{area} series">{obs}</Series>'
        )
    return (
        "<Root><Extracted>2026-08-15T12:00:00Z</Extracted>"
        "<ReportingEnd>2026-08-15</ReportingEnd>"
        f"<DataSet>{''.join(series)}</DataSet></Root>"
    )


def _registry() -> dict:
    return {
        "coverage_thresholds": {
            "policy_rate_max_age_days": 45,
            "balance_sheet_max_age_days": 240,
            "tier_1_minimum_ratio": 1.0,
            "important_bank_minimum_ratio": 1.0,
        },
        "banks": [
            {
                "bank_id": "federal_reserve",
                "name": "Federal Reserve",
                "bis_area_code": "US",
                "world_bank_area_code": "USA",
                "currency": "USD",
                "tier": 1,
                "weight": 0.6,
                "region": "north_america",
                "groups": ["g5", "advanced"],
                "policy_framework": "policy_rate",
                "policy_rate_required": True,
                "balance_sheet_required": True,
                "official_policy_url": "https://www.federalreserve.gov/monetarypolicy.htm",
                "bot_domains": ["rates", "global_macro"],
            },
            {
                "bank_id": "european_central_bank",
                "name": "European Central Bank",
                "bis_area_code": "XM",
                "world_bank_area_code": "EMU",
                "currency": "EUR",
                "tier": 1,
                "weight": 0.4,
                "region": "europe",
                "groups": ["g5", "advanced"],
                "policy_framework": "policy_rate",
                "policy_rate_required": True,
                "balance_sheet_required": True,
                "official_policy_url": "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html",
                "bot_domains": ["rates", "fx", "global_macro"],
            },
        ],
    }


def test_global_central_bank_context_is_point_in_time_and_complete() -> None:
    policy_xml = _series_xml(
        [
            ("US", [("2026-04-01", 5.0), ("2026-07-01", 4.75), ("2026-08-14", 4.5), ("2026-08-20", 9.99)]),
            ("XM", [("2026-04-01", 3.5), ("2026-07-01", 3.25), ("2026-08-14", 3.0)]),
        ]
    )
    assets_xml = _series_xml(
        [
            ("US", [("2025-Q2", 7000.0), ("2025-Q3", 6900.0), ("2025-Q4", 6800.0), ("2026-Q1", 6750.0), ("2026-Q2", 6800.0), ("2026-Q3", 9999.0)]),
            ("XM", [("2025-Q2", 8000.0), ("2025-Q3", 7900.0), ("2025-Q4", 7850.0), ("2026-Q1", 7800.0), ("2026-Q2", 7900.0)]),
        ]
    )
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)

    payload = build_global_central_bank_context(
        policy_xml=policy_xml,
        assets_xml=assets_xml,
        registry=_registry(),
        as_of_date=date(2026, 8, 15),
        collected_at=now,
    )

    fed = payload["banks"]["federal_reserve"]
    assert fed["policy_rate"]["rate_percent"] == 4.5
    assert fed["policy_rate"]["change_bps_30d"] == -25.0
    assert fed["balance_sheet"]["observation_period"] == "2026-Q2"
    assert payload["coverage"]["future_observations_excluded"]["policy_rates"]["US"] == ["2026-08-20"]
    assert payload["coverage"]["future_observations_excluded"]["balance_sheets"]["US"] == ["2026-Q3"]
    assert payload["coverage"]["future_observation_selected"] is False
    assert set(payload["global_features"]) == set(GLOBAL_CENTRAL_BANK_FEATURE_KEYS)
    assert assess_global_central_bank_context(payload, now_utc=now)["ready"] is True
    assert payload["contract"]["live_execution_authority"] is False
    assert payload["contract"]["automatic_promotion_authority"] is False


def test_non_rate_framework_can_be_ready_without_forcing_a_policy_rate() -> None:
    registry = _registry()
    registry["coverage_thresholds"].update({"tier_1_minimum_ratio": 0.0, "important_bank_minimum_ratio": 1.0})
    registry["banks"] = [
        {
            "bank_id": "monetary_authority_of_singapore",
            "name": "Monetary Authority of Singapore",
            "bis_area_code": "SG",
            "world_bank_area_code": "SGP",
            "currency": "SGD",
            "tier": 2,
            "weight": 1.0,
            "region": "asia",
            "groups": ["advanced"],
            "policy_framework": "exchange_rate_band",
            "policy_rate_required": False,
            "balance_sheet_required": True,
            "official_policy_url": "https://www.mas.gov.sg/monetary-policy",
            "bot_domains": ["fx", "global_macro"],
        }
    ]
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    payload = build_global_central_bank_context(
        policy_xml=_series_xml([("US", [("2026-08-14", 4.5)])]),
        assets_xml=_series_xml([("SG", [("2026-Q1", 500.0), ("2026-Q2", 510.0)])]),
        registry=registry,
        as_of_date=now.date(),
        collected_at=now,
    )

    row = payload["banks"]["monetary_authority_of_singapore"]
    assert row["policy_rate"] == {}
    assert row["ready"] is True
    assert row["missing_required_dimensions"] == []


def test_global_contract_rejects_authority_drift() -> None:
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    payload = {
        "timestamp_utc": now.isoformat(),
        "contract": {
            "tier_1_minimum_ratio": 0.8,
            "important_bank_minimum_ratio": 0.85,
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
    }
    drifted = deepcopy(payload)
    drifted["contract"]["live_execution_authority"] = True

    assessment = assess_global_central_bank_context(drifted, now_utc=now)

    assert assessment["ready"] is False
    assert "live_execution_authority_not_locked" in assessment["reasons"]
