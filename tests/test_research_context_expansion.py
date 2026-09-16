import csv
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import runtime_training_common as rtc
from core.research_context_expansion import (
    COLLECTOR_IDS,
    RUNTIME_RESEARCH_CONTEXT_FEATURE_KEYS,
    research_context_ready,
)
from scripts.collect_research_context_expansion import (
    build_bis_global_liquidity_context,
    build_cross_asset_breadth_context,
    build_earnings_event_context,
    build_fixed_income_trace_context,
    build_futures_curve_context,
    build_options_greeks_surface_context,
    build_portfolio_factor_risk_context,
    build_tape_liquidity_context,
)

NOW = datetime(2026, 8, 24, 18, 0, tzinfo=timezone.utc)
EXPECTED_COLLECTORS = {
    "bis_global_liquidity_context",
    "cross_asset_breadth_context",
    "earnings_event_context",
    "fixed_income_trace_context",
    "futures_curve_context",
    "options_greeks_surface_context",
    "portfolio_factor_risk_context",
    "tape_liquidity_context",
}


def _row(
    symbol: str,
    features: dict,
    *,
    minute: int = 0,
    asset_class: str = "equity",
    lane: str = "equity",
) -> dict:
    return {
        "timestamp_utc": (NOW + timedelta(minutes=minute)).isoformat(),
        "symbol": symbol,
        "asset_class": asset_class,
        "features": features,
        "data_route": {"routing_lane": lane, "source_quality_score": 0.94},
    }


def _capabilities(payload: dict) -> dict[str, dict]:
    return {row["capability_id"]: row for row in payload["capabilities"]}


def _assert_observation_only(payload: dict) -> None:
    assert payload["snapshot_receipt_sha256"]
    assert payload["authority_contract"] == {
        "observation_only": True,
        "paper_execution_authority": False,
        "live_execution_authority": False,
        "automatic_promotion_authority": False,
        "registry_mutation_authority": False,
    }
    assert (
        payload["evidence_contract"]["missing_dimensions_are_omitted_not_zero_filled"]
        is True
    )
    assert all(row["proof_receipt_sha256"] for row in payload["capabilities"])


def test_exactly_eight_unique_context_collectors_are_declared() -> None:
    assert len(COLLECTOR_IDS) == 8
    assert set(COLLECTOR_IDS) == EXPECTED_COLLECTORS
    assert len(RUNTIME_RESEARCH_CONTEXT_FEATURE_KEYS) >= 50


def test_cross_asset_collector_earns_all_capabilities_from_aligned_series() -> None:
    scales = {
        "SPY": 0.0010,
        "QQQ": 0.0013,
        "IWM": 0.0008,
        "TLT": -0.0005,
        "XLE": 0.0006,
        "XLK": 0.0014,
        "XLF": -0.0002,
    }
    rows = [
        _row(symbol, {"pct_from_close": scale * (index + 1)}, minute=index * 5)
        for index in range(5)
        for symbol, scale in scales.items()
    ]

    payload = build_cross_asset_breadth_context(rows, now=NOW + timedelta(minutes=25))

    assert payload["usable_capability_count"] == payload["capability_count"] == 7
    assert payload["metrics"]["symbol_count"] == len(scales)
    _assert_observation_only(payload)


def test_tape_collector_normalizes_raw_volatility_and_relative_vwap() -> None:
    rows = [
        _row(
            symbol,
            {
                "spread_bps": 4.0 + index,
                "quote_age_ms": 500.0 + index,
                "vol_30m": 0.008,
                "last_price": 101.0,
                "vwap_60m": 100.0,
                "bid_size": 100.0,
                "ask_size": 80.0,
                "data_quality_quote_agreement_norm": 0.96,
            },
        )
        for index, symbol in enumerate(("SPY", "QQQ", "IWM"))
    ]

    payload = build_tape_liquidity_context(rows, now=NOW)
    features = payload["derived"]["symbol_features"]["SPY"]

    assert payload["usable_capability_count"] == payload["capability_count"] == 6
    assert features["research_tape_realized_volatility_norm"] == pytest.approx(0.1)
    assert features["research_tape_vwap_state_norm"] == pytest.approx(2.0 / 3.0)
    _assert_observation_only(payload)


def test_options_collector_requires_a_direct_available_chain() -> None:
    features = {
        "options_chain_available": 1.0,
        "options_contract_count_norm": 0.8,
        "options_iv_atm": 0.25,
        "options_iv_atm_norm": 0.3,
        "options_iv_skew_norm": 0.6,
        "options_iv_term_structure_norm": 0.4,
        "options_delta_abs_mean_norm": 0.5,
        "options_gamma_mean_norm": 0.2,
        "options_theta_abs_mean_norm": 0.3,
        "options_vega_mean_norm": 0.4,
        "options_open_interest_total": 10_000,
        "options_open_interest_norm": 0.7,
        "options_realized_volatility_norm": 0.25,
        "options_iv_realized_spread_norm": 0.65,
    }

    payload = build_options_greeks_surface_context([_row("SPY", features)], now=NOW)

    assert payload["usable_capability_count"] == payload["capability_count"] == 8
    assert payload["metrics"]["symbols_with_direct_chain"] == 1
    _assert_observation_only(payload)


def test_futures_collector_does_not_turn_default_zeros_into_evidence() -> None:
    features = {
        "futures_expiry_days": 30.0,
        "futures_term_structure_norm": 0.0,
        "futures_basis_bps": 0.0,
        "futures_basis_bps_norm": 0.0,
        "futures_roll_yield_norm": 0.0,
        "futures_open_interest": 0.0,
        "futures_open_interest_norm": 0.0,
        "futures_session_volume_profile_norm": 0.0,
        "futures_calendar_spread_curve_norm": 0.0,
        "futures_basis_dislocation_norm": 0.0,
    }

    payload = build_futures_curve_context(
        [_row("/ES", features, asset_class="futures", lane="futures")],
        now=NOW,
    )
    capabilities = _capabilities(payload)

    assert capabilities["expiry_state"]["usable"] is True
    assert capabilities["futures_term_structure"]["usable"] is False
    assert capabilities["futures_basis"]["usable"] is False
    assert capabilities["roll_yield"]["usable"] is False
    assert capabilities["calendar_spreads"]["usable"] is False
    assert (
        "research_futures_term_structure_norm"
        not in payload["derived"]["symbol_features"]["/ES"]
    )
    _assert_observation_only(payload)


def test_earnings_collector_requires_capability_specific_analyst_fields() -> None:
    rows = [
        _row("AAPL", {"calendar_feed_available": 1.0, "calendar_earnings_7d_norm": 0.7})
    ]
    sec_payload = {
        "timestamp_utc": NOW.isoformat(),
        "status": {"ok": True},
        "symbol_rows": [{"symbol": "AAPL", "earnings_7d": 1, "guidance_7d": 2}],
    }
    analyst_payload = {
        "timestamp_utc": NOW.isoformat(),
        "ok": True,
        "symbols": {"AAPL": {"estimates": [{"period": "2026-Q3"}]}},
        "derived": {"symbol_features": {"AAPL": {}}},
    }

    missing = build_earnings_event_context(
        rows,
        now=NOW,
        sec_payload=sec_payload,
        analyst_payload=analyst_payload,
    )
    missing_capabilities = _capabilities(missing)
    assert missing_capabilities["estimate_revisions"]["usable"] is False
    assert missing_capabilities["estimate_dispersion"]["usable"] is False

    analyst_payload["derived"]["symbol_features"]["AAPL"] = {
        "consensus_revision_direction_norm": 0.65,
        "consensus_dispersion_norm": 0.3,
    }
    complete = build_earnings_event_context(
        rows,
        now=NOW,
        sec_payload=sec_payload,
        analyst_payload=analyst_payload,
    )
    assert complete["usable_capability_count"] == complete["capability_count"] == 5
    _assert_observation_only(complete)


def test_portfolio_collector_combines_positions_with_aligned_market_series() -> None:
    scales = {
        "SPY": 0.0010,
        "QQQ": 0.0012,
        "IWM": 0.0008,
        "AAPL": 0.0014,
        "MSFT": 0.0011,
    }
    rows = [
        _row(
            symbol,
            {"pct_from_close": scale * (index + 1), "spread_bps": 3.0 + scales[symbol]},
            minute=index * 5,
        )
        for index in range(5)
        for symbol, scale in scales.items()
    ]
    account_payload = {
        "timestamp_utc": NOW.isoformat(),
        "ok": True,
        "positions": [
            {"symbol": "AAPL", "market_value": 6_000.0},
            {"symbol": "MSFT", "market_value": 4_000.0},
        ],
    }

    payload = build_portfolio_factor_risk_context(
        rows, now=NOW, account_payload=account_payload
    )

    assert payload["usable_capability_count"] == payload["capability_count"] == 6
    assert (
        payload["derived"]["symbol_features"]["AAPL"]["research_portfolio_weight_norm"]
        == 0.6
    )
    _assert_observation_only(payload)


def test_finra_collector_materializes_fixed_income_breadth_and_liquidity() -> None:
    def fetch_json(url: str, **_kwargs):
        if "corporateMarketBreadth" in url:
            return [
                {
                    "tradeReportDate": "2026-08-21",
                    "advances": 120,
                    "declines": 80,
                    "totalTradeCount": 1_100,
                },
                {
                    "tradeReportDate": "2026-08-20",
                    "advances": 100,
                    "declines": 90,
                    "totalTradeCount": 1_000,
                },
            ]
        return [
            {
                "tradeReportDate": "2026-08-21",
                "dealerCustomerVolume": 2_000_000,
                "atsInterdealerVolume": 1_000_000,
            }
        ]

    payload = build_fixed_income_trace_context(
        now=NOW,
        fetch_json=fetch_json,
        access_token="fixture-token",
    )

    assert payload["usable_capability_count"] == payload["capability_count"] == 4
    assert payload["sources"]["finra_corporate_market_breadth"]["ok"] is True
    assert payload["derived"]["global_features"]["research_trace_available_norm"] == 1.0
    _assert_observation_only(payload)


def test_finra_collector_does_not_request_data_without_public_credential() -> None:
    def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("FINRA must not be called without a configured bearer token")

    payload = build_fixed_income_trace_context(
        now=NOW,
        fetch_json=unexpected_fetch,
        access_token="",
    )

    assert payload["ok"] is False
    assert payload["usable_capability_count"] == 0
    assert payload["warnings"] == ["finra_public_api_credential_required"]
    assert payload["sources"]["finra_corporate_market_breadth"]["credential_configured"] is False


def _bis_fixture(*, descriptive_headers: bool = False) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "TIME_PERIOD:Time period" if descriptive_headers else "TIME_PERIOD",
            "COUNTERPARTY_COUNTRY:Country"
            if descriptive_headers
            else "COUNTERPARTY_COUNTRY",
            "CURRENCY:Currency" if descriptive_headers else "CURRENCY",
            "SERIES:Series" if descriptive_headers else "SERIES",
            "OBS_VALUE:Observation Value" if descriptive_headers else "OBS_VALUE",
        ]
    )
    for index in range(12):
        writer.writerow(
            ["2026-Q1", f"C{index:02d}", "USD", f"S{index:02d}", 100 + index]
        )
        writer.writerow(
            ["2026-Q2", f"C{index:02d}", "USD", f"S{index:02d}", 105 + index]
        )
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("WS_GLI.csv", buffer.getvalue())
    return raw.getvalue()


def test_bis_parser_accepts_official_descriptive_column_headers() -> None:
    payload = build_bis_global_liquidity_context(
        now=NOW,
        fetch_bytes=lambda *_args, **_kwargs: _bis_fixture(
            descriptive_headers=True
        ),
    )

    assert payload["usable_capability_count"] == 3


def test_bis_collector_materializes_bounded_global_liquidity_context() -> None:
    payload = build_bis_global_liquidity_context(
        now=NOW,
        fetch_bytes=lambda *_args, **_kwargs: _bis_fixture(),
    )

    assert payload["usable_capability_count"] == payload["capability_count"] == 3
    assert payload["metrics"]["series_count"] == 12
    assert payload["metrics"]["bounded"] is True
    assert payload["derived"]["global_features"]["research_bis_available_norm"] == 1.0
    _assert_observation_only(payload)


def test_runtime_rejects_stale_tampered_or_authoritative_context(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            symbol,
            {
                "spread_bps": 4.0,
                "quote_age_ms": 500.0,
                "vol_30m": 0.004,
                "last_price": 101.0,
                "vwap_60m": 100.0,
                "bid_size": 100.0,
                "ask_size": 80.0,
                "data_quality_quote_agreement_norm": 0.95,
            },
        )
        for symbol in ("SPY", "QQQ", "IWM")
    ]
    payload = build_tape_liquidity_context(rows, now=datetime.now(timezone.utc))
    output = (
        tmp_path / "exports" / "external_context" / "tape_liquidity_context_latest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")

    context = rtc._load_runtime_gap_fill_context(tmp_path)
    assert context["external_global_features"]["research_tape_available_norm"] == 1.0
    assert research_context_ready(payload, "tape_liquidity_context") is True
    assert (
        research_context_ready(
            payload,
            "tape_liquidity_context",
            now=datetime.fromisoformat(payload["timestamp_utc"])
            + timedelta(minutes=121),
        )
        is False
    )

    payload["derived"]["global_features"]["research_tape_available_norm"] = 0.25
    output.write_text(json.dumps(payload), encoding="utf-8")
    tampered = rtc._load_runtime_gap_fill_context(tmp_path)
    assert "research_tape_available_norm" not in tampered["external_global_features"]

    payload["authority_contract"]["live_execution_authority"] = True
    assert research_context_ready(payload, "tape_liquidity_context") is False
