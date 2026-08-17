from datetime import datetime, timezone

from scripts.collect_analyst_consensus_context import (
    _claim_daily_request,
    _fetch_nasdaq_forecast,
    build_analyst_consensus_payload,
    parse_alpha_vantage_estimates,
    parse_nasdaq_earnings_forecast,
)


def _provider_row(symbol: str, now: datetime) -> dict:
    payload = {
        "symbol": symbol,
        "estimates": [
            {
                "date": "2026-09-30",
                "horizon": "current quarter",
                "epsEstimateAverage": "2.00",
                "epsEstimateHigh": "2.40",
                "epsEstimateLow": "1.60",
                "epsEstimateAnalystCount": "20",
                "epsEstimateAverage7DaysAgo": "1.95",
                "epsEstimateAverage30DaysAgo": "1.90",
                "epsEstimateRevisionUpTrailing7Days": "3",
                "epsEstimateRevisionDownTrailing7Days": "1",
                "epsEstimateRevisionUpTrailing30Days": "6",
                "epsEstimateRevisionDownTrailing30Days": "2",
                "revenueEstimateAverage": "1000",
                "revenueEstimateHigh": "1100",
                "revenueEstimateLow": "900",
                "revenueEstimateAnalystCount": "18",
            }
        ],
    }
    return parse_alpha_vantage_estimates(payload, symbol=symbol, collected_at_utc=now)


def _config() -> dict:
    return {
        "context_id": "analyst_consensus_context_v1",
        "provider_documentation_url": "https://www.alphavantage.co/documentation/#earnings-estimates",
        "minimum_covered_symbols": 2,
        "minimum_universe_coverage_ratio": 1.0,
        "minimum_revision_history_ratio": 1.0,
        "maximum_requests_per_day": 25,
        "authority_contract": {
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
    }


def test_alpha_vantage_parser_preserves_consensus_and_revision_history() -> None:
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    row = _provider_row("AAPL", now)
    assert row["symbol"] == "AAPL"
    assert row["collected_at_utc"] == now.isoformat()
    estimate = row["estimates"][0]
    assert estimate["eps_analyst_count"] == 20.0
    assert estimate["eps_average_30_days_ago"] == 1.9
    assert estimate["eps_revision_up_30_days"] == 6.0


def test_consensus_payload_earns_direct_readiness_from_real_coverage() -> None:
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    payload = build_analyst_consensus_payload(
        config=_config(),
        universe_symbols=["AAPL", "MSFT"],
        cache_rows={"AAPL": _provider_row("AAPL", now), "MSFT": _provider_row("MSFT", now)},
        now_utc=now,
        provider_enabled=True,
        provider_configured=True,
    )
    assert payload["direct_evidence_ready"] is True
    assert payload["coverage"]["coverage_ratio"] == 1.0
    assert payload["coverage"]["revision_history_ratio"] == 1.0
    assert payload["derived"]["symbol_features"]["AAPL"]["consensus_revision_direction_norm"] > 0.5
    assert payload["authority_contract"]["live_execution_authority"] is False


def test_missing_or_thin_consensus_cannot_claim_readiness() -> None:
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    payload = build_analyst_consensus_payload(
        config=_config(),
        universe_symbols=["AAPL", "MSFT"],
        cache_rows={"AAPL": _provider_row("AAPL", now)},
        now_utc=now,
        provider_enabled=False,
        provider_configured=False,
    )
    assert payload["direct_evidence_ready"] is False
    assert "minimum_symbol_coverage_not_met" in payload["consumer_contract"]["reasons"]
    assert "universe_coverage_ratio_not_met" in payload["consumer_contract"]["reasons"]


def test_daily_request_quota_is_persistent_and_resets_by_utc_day(tmp_path) -> None:
    first_day = datetime(2026, 8, 15, 23, 59, tzinfo=timezone.utc)
    claimed, state = _claim_daily_request(
        tmp_path,
        now=first_day,
        maximum_requests=2,
    )
    assert claimed is True
    assert state["requests_claimed"] == 1

    claimed, state = _claim_daily_request(
        tmp_path,
        now=first_day,
        maximum_requests=2,
    )
    assert claimed is True
    assert state["requests_remaining"] == 0

    claimed, state = _claim_daily_request(
        tmp_path,
        now=first_day,
        maximum_requests=2,
    )
    assert claimed is False
    assert state["requests_claimed"] == 2

    claimed, state = _claim_daily_request(
        tmp_path,
        now=datetime(2026, 8, 16, 0, 1, tzinfo=timezone.utc),
        maximum_requests=2,
    )
    assert claimed is True
    assert state["date_utc"] == "2026-08-16"
    assert state["requests_claimed"] == 1


def test_nasdaq_parser_preserves_dispersion_counts_and_zero_revisions() -> None:
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    row = parse_nasdaq_earnings_forecast(
        {
            "data": {
                "symbol": "aapl",
                "quarterlyForecast": {
                    "rows": [
                        {
                            "fiscalEnd": "Sep 2026",
                            "consensusEPSForecast": 1.98,
                            "highEPSForecast": 2.09,
                            "lowEPSForecast": 1.91,
                            "noOfEstimates": 8,
                            "up": 0,
                            "down": 0,
                        }
                    ]
                },
                "yearlyForecast": {"rows": []},
            },
            "status": {"rCode": 200},
        },
        symbol="AAPL",
        collected_at_utc=now,
    )
    assert row["provider"] == "nasdaq_analyst_forecast"
    assert row["symbol"] == "AAPL"
    assert row["estimates"][0]["eps_analyst_count"] == 8.0
    assert row["estimates"][0]["eps_revision_up_30_days"] == 0.0


def test_full_governed_universe_requires_all_sixteen_symbols() -> None:
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    symbols = [f"S{index:02d}" for index in range(16)]
    config = _config() | {
        "expected_universe_symbol_count": 16,
        "required_symbols": symbols,
        "minimum_covered_symbols": 16,
        "minimum_universe_coverage_ratio": 1.0,
        "minimum_revision_history_ratio": 1.0,
    }
    rows = {symbol: _provider_row(symbol, now) for symbol in symbols}
    incomplete = build_analyst_consensus_payload(
        config=config,
        universe_symbols=symbols,
        cache_rows={symbol: row for symbol, row in rows.items() if symbol != symbols[-1]},
        now_utc=now,
        provider_enabled=True,
        provider_configured=True,
    )
    assert incomplete["direct_evidence_ready"] is False
    assert incomplete["coverage"]["missing_required_symbols"] == [symbols[-1]]

    complete = build_analyst_consensus_payload(
        config=config,
        universe_symbols=symbols,
        cache_rows=rows,
        now_utc=now,
        provider_enabled=True,
        provider_configured=True,
    )
    assert complete["direct_evidence_ready"] is True
    assert complete["coverage"]["covered_symbol_count"] == 16
    assert complete["coverage"]["revision_history_ratio"] == 1.0


def test_nasdaq_transport_is_bounded_and_uses_curl(monkeypatch) -> None:
    captured: dict = {}

    class Result:
        returncode = 0
        stderr = ""
        stdout = '{"data":{"symbol":"aapl"},"status":{"rCode":200}}'

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr("scripts.collect_analyst_consensus_context.subprocess.run", fake_run)
    payload = _fetch_nasdaq_forecast(
        "https://api.nasdaq.com/api/analyst/{symbol}/earnings-forecast",
        symbol="AAPL",
        timeout_seconds=10,
    )
    assert payload["status"]["rCode"] == 200
    assert captured["command"][0] == "/usr/bin/curl"
    assert "--max-time" in captured["command"]
    assert captured["kwargs"]["timeout"] == 25
