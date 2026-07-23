from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_options_flow_context as options_flow


def test_collect_options_flow_context_combines_polygon_and_unusual_whales(monkeypatch) -> None:
    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)

    def _fake_polygon_get(path: str, *, api_key: str, query, user_agent: str, timeout: float):  # type: ignore[no-untyped-def]
        if path.startswith("/v2/snapshot/locale/us/markets/stocks/tickers/"):
            return (
                {
                    "ticker": {
                        "lastTrade": {"p": 510.0},
                        "prevDay": {"c": 502.0},
                        "day": {"v": 2_500_000, "h": 512.0, "l": 500.0},
                        "todaysChangePerc": 1.6,
                    }
                },
                None,
            )
        return (
            {
                "results": [
                    {"contract_type": "call", "expiration_date": today.isoformat(), "strike_price": 510.0},
                    {"contract_type": "put", "expiration_date": tomorrow.isoformat(), "strike_price": 505.0},
                ]
            },
            None,
        )

    def _fake_unusual_whales_get(path: str, *, api_key: str, query, user_agent: str, timeout: float):  # type: ignore[no-untyped-def]
        if path.endswith("/iv-rank"):
            return ({"data": {"iv_rank": 62, "implied_volatility": 48}}, None)
        if path.endswith("/max-pain"):
            return ({"data": {"max_pain": 508.0}}, None)
        if path.endswith("/oi-change"):
            return (
                {
                    "data": [
                        {"option_symbol": "SPY260414C00510000", "oi_change": 1800, "curr_oi": 7500, "volume": 6200},
                        {"option_symbol": "SPY260415P00505000", "oi_change": 1200, "curr_oi": 5100, "volume": 4100},
                    ]
                },
                None,
            )
        if path.endswith("/net-prem-ticks"):
            return ({"data": [{"net_call_premium": 2_500_000, "net_put_premium": 1_000_000, "net_premium": 1_500_000}]}, None)
        return (None, "unknown_endpoint")

    monkeypatch.setattr(options_flow, "_polygon_get", _fake_polygon_get)
    monkeypatch.setattr(options_flow, "_unusual_whales_get", _fake_unusual_whales_get)

    payload, status = options_flow.collect_options_flow_context(
        polygon_api_key="polygon-key",
        unusual_whales_api_key="uw-key",
        unusual_whales_export_path="",
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        polygon_contract_limit=50,
    )

    assert status["ok"] is True
    assert status["symbols_with_polygon"] == 1
    assert status["symbols_with_unusual_whales"] == 1
    assert status["symbols_with_metrics"] == 1
    assert payload["derived"]["symbol_features"]["SPY"]["tasty_iv_rank_norm"] > 0.0
    assert payload["derived"]["symbol_features"]["SPY"]["tasty_max_pain_proximity_norm"] > 0.0
    assert payload["derived"]["symbol_features"]["SPY"]["source_confidence_norm"] > 0.0
    assert payload["derived"]["symbol_features"]["SPY"]["schema_confidence_norm"] > 0.0
    assert payload["sources"]["polygon"]["ok"] is True
    assert payload["sources"]["unusual_whales_api"]["ok"] is True
    assert payload["sources"]["polygon"]["source_confidence_norm"] > 0.0
    assert payload["sources"]["unusual_whales_api"]["schema_confidence_norm"] > 0.0
    assert status["sources"]["polygon"]["ok"] is True
    assert status["sources"]["unusual_whales_api"]["ok"] is True
    assert "source_contracts" in status
    assert payload["collection_contract"]["provider_confidence_norm"] > 0.0


def test_collect_options_flow_context_marks_missing_credentials() -> None:
    payload, status = options_flow.collect_options_flow_context(
        polygon_api_key="",
        unusual_whales_api_key="",
        unusual_whales_export_path="",
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        polygon_contract_limit=50,
        free_sources_enabled=False,
    )

    assert status["ok"] is False
    assert status["operator_action_required"] is True
    assert status["auth_issue"] == "options_flow_credentials_missing"
    assert payload["sources"]["session"]["recommended_action"] == "set_polygon_api_key"


def test_collect_options_flow_context_uses_free_option_chain_without_paid_credentials(monkeypatch) -> None:
    today = datetime.now(timezone.utc).date()

    def _fake_yahoo(symbol: str, *, user_agent: str, timeout: float, contract_limit: int):  # type: ignore[no-untyped-def]
        return (
            {
                "ticker": {
                    "lastTrade": {"p": 510.0},
                    "prevDay": {"c": 500.0},
                    "day": {"v": 3_000_000, "h": 512.0, "l": 498.0},
                    "todaysChangePerc": 2.0,
                }
            },
            {"iv_rank": 52.0, "implied_volatility": 41.0},
            [
                {"contract_type": "call", "expiration_date": today.isoformat(), "strike_price": 510.0, "volume": 1200},
                {"contract_type": "put", "expiration_date": today.isoformat(), "strike_price": 505.0, "volume": 900},
            ],
            {"ok": True, "contract_count": 2, "source_confidence_norm": 0.72, "schema_confidence_norm": 0.82},
        )

    monkeypatch.setattr(options_flow, "_polygon_get", lambda *args, **kwargs: (None, "polygon_api_key_missing"))
    monkeypatch.setattr(options_flow, "_unusual_whales_get", lambda *args, **kwargs: (None, "unusual_whales_api_key_missing"))
    monkeypatch.setattr(options_flow, "_collect_yahoo_options_chain", _fake_yahoo)
    monkeypatch.setattr(options_flow, "_collect_cboe_options_chain", lambda *args, **kwargs: ({}, None, [], {"ok": False, "error": "not_available"}))

    payload, status = options_flow.collect_options_flow_context(
        polygon_api_key="",
        unusual_whales_api_key="",
        unusual_whales_export_path="",
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        polygon_contract_limit=50,
    )

    assert status["ok"] is True
    assert status["overall_status"] == "ready"
    assert status["context_profile"] == "free_options_chain_only"
    assert status["symbols_with_free_options"] == 1
    assert status["coverage"]["free_options_chain_ok"] is True
    assert payload["sources"]["yahoo_options_chain"]["ok"] is True
    assert payload["sources"]["session"]["recommended_action"] == ""
    assert payload["derived"]["symbol_features"]["SPY"]["tasty_iv_rank_norm"] > 0.0


def test_collect_options_flow_context_accepts_unusual_whales_export_without_api(monkeypatch, tmp_path: Path) -> None:
    export_path = tmp_path / "uw_export.json"
    export_path.write_text(
        json.dumps(
            {
                "symbols": {
                    "SPY": {
                        "iv_rank": {"iv_rank": 62, "implied_volatility": 48},
                        "max_pain": {"max_pain": 508.0},
                        "oi_change": {
                            "rows": [
                                {"option_symbol": "SPY260414C00510000", "oi_change": 1800, "curr_oi": 7500, "volume": 6200},
                                {"option_symbol": "SPY260415P00505000", "oi_change": 1200, "curr_oi": 5100, "volume": 4100},
                            ]
                        },
                        "net_prem_ticks": {
                            "rows": [
                                {"net_call_premium": 2_500_000, "net_put_premium": 1_000_000, "net_premium": 1_500_000}
                            ]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def _fake_polygon_get(path: str, *, api_key: str, query, user_agent: str, timeout: float):  # type: ignore[no-untyped-def]
        return (None, "polygon_api_key_missing")

    def _fake_unusual_whales_get(path: str, *, api_key: str, query, user_agent: str, timeout: float):  # type: ignore[no-untyped-def]
        return (None, "unusual_whales_api_key_missing")

    monkeypatch.setattr(options_flow, "_polygon_get", _fake_polygon_get)
    monkeypatch.setattr(options_flow, "_unusual_whales_get", _fake_unusual_whales_get)

    payload, status = options_flow.collect_options_flow_context(
        polygon_api_key="",
        unusual_whales_api_key="",
        unusual_whales_export_path=str(export_path),
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        polygon_contract_limit=50,
        unusual_whales_export_min_stable_seconds=0,
        free_sources_enabled=False,
    )

    assert status["ok"] is True
    assert status["overall_status"] == "degraded"
    assert status["context_profile"] == "unusual_whales_overlay_only"
    assert status["operator_action_required"] is True
    assert status["symbols_with_polygon"] == 0
    assert status["symbols_with_unusual_whales"] == 0
    assert status["symbols_with_metrics"] == 1
    assert payload["derived"]["symbol_features"]["SPY"]["tasty_iv_rank_norm"] > 0.0
    assert payload["sources"]["unusual_whales_export"]["ok"] is True
    assert payload["sources"]["unusual_whales_export"]["symbol_count"] == 1
    assert payload["sources"]["unusual_whales_export"]["source_confidence_norm"] > 0.0
    assert payload["sources"]["unusual_whales_export"]["freshness_norm"] > 0.0
    assert status["sources"]["unusual_whales_export"]["ok"] is True
    assert payload["sources"]["session"]["recommended_action"] == "set_polygon_api_key"


def test_collect_options_flow_context_surfaces_bad_export_parse_failures(tmp_path: Path) -> None:
    export_path = tmp_path / "uw_export.json"
    export_path.write_text("{bad json", encoding="utf-8")

    payload, status = options_flow.collect_options_flow_context(
        polygon_api_key="",
        unusual_whales_api_key="",
        unusual_whales_export_path=str(export_path),
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        polygon_contract_limit=50,
        unusual_whales_export_min_stable_seconds=0,
        free_sources_enabled=False,
    )

    assert status["ok"] is False
    assert status["auth_issue"] == "options_flow_export_unusable"
    assert any(str(item).startswith("parse_failed:") for item in payload["sources"]["unusual_whales_export"]["errors"])
    assert payload["sources"]["session"]["recommended_action"] == "repair_unusual_whales_export_or_set_polygon_api_key"


def test_collect_options_flow_context_treats_polygon_as_primary_when_optional_overlay_is_unconfigured(monkeypatch) -> None:
    today = datetime.now(timezone.utc).date()

    def _fake_polygon_get(path: str, *, api_key: str, query, user_agent: str, timeout: float):  # type: ignore[no-untyped-def]
        if path.startswith("/v2/snapshot/locale/us/markets/stocks/tickers/"):
            return (
                {
                    "ticker": {
                        "lastTrade": {"p": 510.0},
                        "prevDay": {"c": 502.0},
                        "day": {"v": 2_500_000, "h": 512.0, "l": 500.0},
                        "todaysChangePerc": 1.6,
                    }
                },
                None,
            )
        return (
            {
                "results": [
                    {"contract_type": "call", "expiration_date": today.isoformat(), "strike_price": 510.0},
                ]
            },
            None,
        )

    monkeypatch.setattr(options_flow, "_polygon_get", _fake_polygon_get)
    monkeypatch.setattr(options_flow, "_unusual_whales_get", lambda *args, **kwargs: (None, "unusual_whales_api_key_missing"))

    payload, status = options_flow.collect_options_flow_context(
        polygon_api_key="polygon-key",
        unusual_whales_api_key="",
        unusual_whales_export_path="",
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        polygon_contract_limit=50,
        free_sources_enabled=False,
    )

    assert status["ok"] is True
    assert status["overall_status"] == "ready"
    assert status["context_profile"] == "polygon_primary_only"
    assert status.get("operator_action_required") is not True
    assert payload["sources"]["session"]["unusual_whales_expected"] is False
    assert payload["sources"]["session"]["recommended_action"] == ""


def test_inspect_unusual_whales_export_supports_row_inference_directory_layout(tmp_path: Path) -> None:
    export_dir = tmp_path / "uw_exports"
    export_dir.mkdir()
    (export_dir / "options_flow_20260415.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"dataset": "iv_rank", "symbol": "SPY", "iv_rank": 61, "implied_volatility": 47}),
                json.dumps({"dataset": "max_pain", "symbol": "SPY", "max_pain": 507.0}),
                json.dumps({"dataset": "oi_change", "symbol": "SPY", "option_symbol": "SPY260415C00510000", "oi_change": 900, "curr_oi": 3200}),
                json.dumps({"dataset": "net_premium", "symbol": "SPY", "net_call_premium": 800000, "net_put_premium": 300000, "net_premium": 500000}),
            ]
        ),
        encoding="utf-8",
    )

    payload, inspection = options_flow.inspect_unusual_whales_export(
        str(export_dir),
        max_age_seconds=21600,
        min_stable_seconds=0,
    )

    assert inspection["usable"] is True
    assert inspection["adapter"] == "row_inference"
    assert payload["symbols"]["SPY"]["iv_rank"]["iv_rank"] == 61
    assert inspection["dataset_symbol_counts"]["net_prem_ticks"] == 1


def test_inspect_unusual_whales_export_waits_for_stable_candidates(tmp_path: Path) -> None:
    export_path = tmp_path / "uw_export.json"
    export_path.write_text(json.dumps({"symbols": {"SPY": {"iv_rank": {"iv_rank": 55}}}}), encoding="utf-8")

    payload, inspection = options_flow.inspect_unusual_whales_export(
        str(export_path),
        max_age_seconds=21600,
        min_stable_seconds=3600,
    )

    assert payload == {}
    assert inspection["usable"] is False
    assert "candidate_not_stable_yet" in inspection["issues"]
