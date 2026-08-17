from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_public_policy_context as ppc


def _world_bank_payload(indicator_id: str) -> list[Any]:
    values = {
        "NY.GDP.MKTP.CD": {
            "USA": 28750956130731.2,
            "CHN": 18743803170827.2,
            "JPN": 4027597523550.58,
            "DEU": 4685592577804.69,
            "GBR": 3686033044482.13,
        },
        "FP.CPI.TOTL.ZG": {"USA": 2.9, "CHN": 0.2, "JPN": 2.7, "DEU": 2.2, "GBR": 3.3},
        "BN.CAB.XOKA.GD.ZS": {"USA": -4.1, "CHN": 2.2, "JPN": 4.7, "DEU": 5.8, "GBR": -3.0},
        "GC.DOD.TOTL.GD.ZS": {"USA": 118.0, "CHN": 86.0, "JPN": 214.0, "DEU": 64.0, "GBR": 131.0},
        "FR.INR.RINR": {"USA": 1.2, "CHN": 5.1, "JPN": -0.3, "DEU": 0.4, "GBR": 0.9},
    }[indicator_id]
    rows = []
    for country, value in values.items():
        rows.append(
            {
                "indicator": {"id": indicator_id, "value": indicator_id},
                "country": {"id": country[:2], "value": country},
                "countryiso3code": country,
                "date": "2024",
                "value": value,
            }
        )
    return [{"page": 1, "pages": 1, "per_page": 80, "total": len(rows), "lastupdated": "2026-04-08"}, rows]


def test_collect_public_policy_context_builds_official_free_source_lane(monkeypatch) -> None:
    def fake_http_json(url: str, *, user_agent: str, timeout: float) -> tuple[Any | None, str | None]:
        assert user_agent
        assert timeout > 0
        if "debt_to_penny" in url:
            return (
                {
                    "data": [
                        {
                            "record_date": "2026-06-11",
                            "debt_held_public_amt": "31594717601039.83",
                            "intragov_hold_amt": "7626067516278.85",
                            "tot_pub_debt_out_amt": "39220785117318.68",
                        },
                        {
                            "record_date": "2026-06-10",
                            "debt_held_public_amt": "31592369611735.25",
                            "intragov_hold_amt": "7620896668005.91",
                            "tot_pub_debt_out_amt": "39213266279741.16",
                        },
                    ]
                },
                None,
            )
        if "avg_interest_rates" in url:
            return (
                {
                    "data": [
                        {
                            "record_date": "2026-05-31",
                            "security_desc": "Treasury Bills",
                            "avg_interest_rate_amt": "3.690",
                        },
                        {
                            "record_date": "2026-05-31",
                            "security_desc": "Treasury Notes",
                            "avg_interest_rate_amt": "3.248",
                        },
                    ]
                },
                None,
            )
        if "api.worldbank.org" in url:
            indicator_id = url.split("/indicator/", 1)[1].split("?", 1)[0]
            return _world_bank_payload(indicator_id), None
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(ppc, "_safe_http_json", fake_http_json)
    monkeypatch.setattr(ppc, "_cached_world_bank_indicators", lambda: {})

    payload, status = ppc.collect_public_policy_context(
        countries=["USA", "CHN", "JPN", "DEU", "GBR"],
        user_agent="test-agent",
        timeout=1.0,
    )

    assert payload["provider"] == "public_policy_context"
    assert status["ok"] is True
    assert status["ok_source_count"] == 3
    assert status["source_count"] == 3
    assert status["sources"]["treasury_debt_to_penny"]["daily_change_usd"] > 0
    assert status["sources"]["world_bank_indicators"]["value_count"] == 25
    assert status["features"]["us_public_debt_to_worldbank_gdp_proxy"] > 1.0
    assert status["features"]["treasury_avg_interest_rate_pct"] == (3.690 + 3.248) / 2.0


def test_collect_public_policy_context_keeps_required_sources_explicit(monkeypatch) -> None:
    def fake_http_json(url: str, *, user_agent: str, timeout: float) -> tuple[Any | None, str | None]:
        if "avg_interest_rates" in url:
            return None, "temporary timeout"
        if "debt_to_penny" in url:
            return (
                {
                    "data": [
                        {
                            "record_date": "2026-06-11",
                            "debt_held_public_amt": "31594717601039.83",
                            "intragov_hold_amt": "7626067516278.85",
                            "tot_pub_debt_out_amt": "39220785117318.68",
                        }
                    ]
                },
                None,
            )
        if "api.worldbank.org" in url:
            indicator_id = url.split("/indicator/", 1)[1].split("?", 1)[0]
            return _world_bank_payload(indicator_id), None
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(ppc, "_safe_http_json", fake_http_json)
    monkeypatch.setattr(ppc, "_cached_world_bank_indicators", lambda: {})
    monkeypatch.setattr(ppc, "_cached_treasury_avg_interest", lambda: None)

    _, status = ppc.collect_public_policy_context(
        countries=["USA", "CHN", "JPN", "DEU", "GBR"],
        user_agent="test-agent",
        timeout=1.0,
    )

    assert status["ok"] is True
    assert status["required_sources_ok"] is True
    assert status["ok_source_count"] == 2
    assert status["sources"]["treasury_avg_interest_rates"]["ok"] is False


def test_collect_public_policy_context_uses_cached_monthly_treasury_rate(monkeypatch) -> None:
    def fake_http_json(url: str, *, user_agent: str, timeout: float) -> tuple[Any | None, str | None]:
        if "avg_interest_rates" in url:
            return None, "temporary treasury disconnect"
        if "debt_to_penny" in url:
            return (
                {
                    "data": [
                        {
                            "record_date": "2026-06-11",
                            "debt_held_public_amt": "31594717601039.83",
                            "intragov_hold_amt": "7626067516278.85",
                            "tot_pub_debt_out_amt": "39220785117318.68",
                        }
                    ]
                },
                None,
            )
        if "api.worldbank.org" in url:
            indicator_id = url.split("/indicator/", 1)[1].split("?", 1)[0]
            return _world_bank_payload(indicator_id), None
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(ppc, "_safe_http_json", fake_http_json)
    monkeypatch.setattr(ppc, "_cached_world_bank_indicators", lambda: {})
    monkeypatch.setattr(
        ppc,
        "_cached_treasury_avg_interest",
        lambda: ppc._source_status(
            "treasury_avg_interest_rates",
            ok=True,
            url=ppc.TREASURY_AVG_INTEREST_URL,
            record_date="2026-05-31",
            row_count=16,
            avg_interest_rate_pct=3.3186875,
            rates_by_security={},
            cached_fallback=True,
            cache_reason="live_treasury_avg_interest_rates_unavailable",
        ),
    )

    _, status = ppc.collect_public_policy_context(
        countries=["USA", "CHN", "JPN", "DEU", "GBR"],
        user_agent="test-agent",
        timeout=1.0,
    )

    cached_source = status["sources"]["treasury_avg_interest_rates"]
    assert status["ok"] is True
    assert status["ok_source_count"] == 3
    assert cached_source["ok"] is True
    assert cached_source["cached_fallback"] is True
    assert cached_source["live_error"] == "temporary treasury disconnect"
    assert status["features"]["treasury_avg_interest_rate_pct"] == 3.3186875


def test_world_bank_timeout_recovers_with_smaller_country_chunks(monkeypatch) -> None:
    def fake_http_json(url: str, *, user_agent: str, timeout: float) -> tuple[Any | None, str | None]:
        indicator_id = url.split("/indicator/", 1)[1].split("?", 1)[0]
        country_path = url.split("/country/", 1)[1].split("/indicator/", 1)[0]
        countries = country_path.split(";")
        if len(countries) > ppc.WORLD_BANK_FALLBACK_COUNTRY_CHUNK:
            return None, "primary timeout"
        payload = _world_bank_payload(indicator_id)
        payload[1] = [row for row in payload[1] if row["countryiso3code"] in countries]
        return payload, None

    monkeypatch.setattr(ppc, "_safe_http_json", fake_http_json)
    monkeypatch.setattr(ppc, "_cached_world_bank_indicators", lambda: {})
    monkeypatch.setattr(ppc, "WORLD_BANK_FALLBACK_COUNTRY_CHUNK", 2)

    result = ppc._fetch_world_bank_indicators(
        countries=["USA", "CHN", "JPN", "DEU", "GBR"],
        user_agent="test-agent",
        timeout=1.0,
    )

    assert result["ok"] is True
    assert result["indicator_success_count"] == 5
    assert len(result["fallback_used_indicators"]) == 5


def test_world_bank_cache_is_labeled_and_fills_transient_indicator_gap(monkeypatch) -> None:
    def fake_http_json(url: str, *, user_agent: str, timeout: float) -> tuple[Any | None, str | None]:
        indicator_id = url.split("/indicator/", 1)[1].split("?", 1)[0]
        if indicator_id == "FR.INR.RINR":
            return None, "temporary timeout"
        return _world_bank_payload(indicator_id), None

    cached = _world_bank_payload("FR.INR.RINR")
    cached_rows = {
        row["countryiso3code"]: {
            "date": row["date"],
            "value": row["value"],
            "country_name": row["country"]["value"],
        }
        for row in cached[1]
    }
    monkeypatch.setattr(ppc, "_safe_http_json", fake_http_json)
    monkeypatch.setattr(
        ppc,
        "_cached_world_bank_indicators",
        lambda: {
            "age_days": 1.0,
            "timestamp_utc": "2026-08-05T12:00:00+00:00",
            "indicators": {
                "real_interest_rate_pct": {
                    "lastupdated": "2026-04-08",
                    "values": cached_rows,
                }
            },
        },
    )

    result = ppc._fetch_world_bank_indicators(
        countries=["USA", "CHN", "JPN", "DEU", "GBR"],
        user_agent="test-agent",
        timeout=1.0,
    )

    assert result["ok"] is True
    assert "FR.INR.RINR" in result["cache_used_indicators"]
    cached_value = result["indicators"]["real_interest_rate_pct"]["values"]["USA"]
    assert cached_value["cached"] is True
    assert cached_value["cache_timestamp_utc"] == "2026-08-05T12:00:00+00:00"
