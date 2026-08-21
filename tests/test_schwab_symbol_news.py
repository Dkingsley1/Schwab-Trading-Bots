from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts import collect_schwab_symbol_news as schwab_news
from scripts.collect_schwab_symbol_news import (
    _symbol_features,
    build_payload,
    dedupe_items,
    extract_news_items,
    load_ticker_universe,
    load_ticker_universe_with_policy,
    normalize_news_item,
)


def test_load_ticker_universe_includes_expanded_hut() -> None:
    symbols, groups, source = load_ticker_universe()

    assert "HUT" in symbols
    assert "ACWI" in symbols
    assert "GFF" in symbols
    assert "1000" not in symbols
    assert len(symbols) >= 1000
    assert groups["HUT"]
    assert groups["ACWI"]
    assert source


def test_load_ticker_universe_defers_slow_tier_when_storage_pressure_active(tmp_path, monkeypatch) -> None:
    for key in (
        "TICKER_UNIVERSE_SLOW_TIER_DEFER_ON_STORAGE_PRESSURE",
        "TICKER_UNIVERSE_SLOW_SYMBOLS",
        "TICKER_UNIVERSE_STANDARD_SYMBOLS",
        "TICKER_UNIVERSE_STORAGE_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    health_root = tmp_path / "governance" / "health"
    health_root.mkdir(parents=True)
    (health_root / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "blocked",
                "severity": "critical",
                "backpressure": {
                    "effective_raw_live": {
                        "core_pending_lines": 18000,
                        "total_pending_lines": 42000,
                        "stale_stage_pending_lines": 0,
                        "oldest_pending_age_seconds": 900,
                    },
                    "pending_lines_threshold": 5000,
                    "total_pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
                "steady_state": {
                    "targets": {
                        "core_pending_lines": 5000,
                        "total_pending_lines": 15000,
                        "oldest_pending_age_seconds": 240,
                    },
                    "target_status": {
                        "steady_state_ready": False,
                        "target_breach_count": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    symbols, groups, source, policy = load_ticker_universe_with_policy(project_root=tmp_path)

    assert len(symbols) == 500
    assert policy["mode"] == "slow_tier_deferred_for_storage_pressure"
    assert policy["deferred_symbol_count"] == 500
    assert policy["storage_pressure_active"] is True
    assert source.endswith("+slow_tier_deferred")
    assert "NVDA" in symbols
    assert "GFF" not in symbols
    assert groups["NVDA"]


def test_extract_and_normalize_nested_schwab_news_payload() -> None:
    payload = {
        "HUT": {
            "headlines": [
                {
                    "storyHeadline": "HUT rallies after Bitcoin miner update",
                    "sourceName": "Schwab Network",
                    "storyUrl": "https://example.com/hut",
                    "publishedDate": "2026-06-02T14:00:00Z",
                    "relatedSymbols": ["HUT", "BTC-USD"],
                }
            ]
        }
    }

    rows = extract_news_items(payload, "HUT")
    normalized = dedupe_items(normalize_news_item(row, symbol="HUT", source_method="get_news") for row in rows)

    assert len(normalized) == 1
    assert normalized[0]["headline"] == "HUT rallies after Bitcoin miner update"
    assert normalized[0]["publisher"] == "Schwab Network"
    assert normalized[0]["source_method"] == "get_news"
    assert "HUT" in normalized[0]["symbols"]


def test_build_payload_preview_has_all_symbols_without_auth() -> None:
    payload = build_payload(authenticate=False)

    assert payload["overall_status"] == "preview_only"
    assert payload["requested_symbol_count"] >= 500
    assert payload["universe_policy"]["slow_tier_defer_on_storage_pressure"] is True
    assert payload["safety_contract"]["live_execution_allowed"] is False
    assert "/Volumes/VIDEO" in payload["safety_contract"]["protected_volumes"]


def test_symbol_news_features_include_catalyst_and_sentiment() -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "items": [
            {
                "headline": "NVDA beats earnings and raises guidance",
                "publisher": "Schwab Network",
                "published": now,
                "symbols": ["NVDA"],
            }
        ]
    }
    rows = [
        normalize_news_item(row, symbol="NVDA", source_method="get_news")
        for row in extract_news_items(payload, "NVDA")
    ]
    built = dedupe_items(rows)
    features = _symbol_features("NVDA", built, now_ts=datetime.now(timezone.utc).timestamp(), max_items=10)

    assert len(built) == 1
    assert built[0]["headline"].startswith("NVDA beats")
    assert features["schwab_symbol_news_available"] == 1.0
    assert features["news_sentiment"] > 0.0
    assert features["schwab_news_catalyst_earnings_norm"] > 0.0


def test_public_fallback_only_refreshes_without_broker_auth(tmp_path) -> None:
    context_path = tmp_path / "exports" / "external_context" / "schwab_education_context_latest.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    context_path.write_text(
        json.dumps(
            {
                "timestamp_utc": now,
                "items": [
                    {
                        "headline": "NVDA earnings update",
                        "source": "Schwab Network",
                        "published_at": now,
                        "symbols": ["NVDA"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_payload(
        project_root=tmp_path,
        symbols_arg="NVDA,SPY",
        public_fallback_only=True,
    )

    assert payload["ok"] is True
    assert payload["auth_required"] is False
    assert payload["auth_ok"] is True
    assert payload["fallback_source_contract"]["fresh"] is True
    assert payload["symbols"]["NVDA"]["source_method"] == "schwab_public_context_fallback"


def test_authenticated_no_endpoint_fallback_publishes_source_contract(tmp_path, monkeypatch) -> None:
    context_path = tmp_path / "exports" / "external_context" / "schwab_education_context_latest.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    context_path.write_text(
        json.dumps(
            {
                "timestamp_utc": now,
                "items": [
                    {
                        "headline": "NVDA earnings update",
                        "source": "Schwab Network",
                        "published_at": now,
                        "symbols": ["NVDA"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        @staticmethod
        def news_candidates(*, symbol: str, limit: int) -> list[tuple[str, tuple, dict]]:
            return []

    class FakeTrader:
        broker_adapter = FakeAdapter()

        @staticmethod
        def authenticate() -> object:
            return object()

    monkeypatch.setattr(schwab_news.BaseTrader, "from_env", lambda **_kwargs: FakeTrader())

    payload = build_payload(
        project_root=tmp_path,
        symbols_arg="NVDA,SPY",
        max_runtime_seconds=10,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready_public_schwab_fallback"
    assert payload["fallback_active"] is True
    assert payload["fallback_source_contract"]["fresh"] is True
    assert payload["fallback_source_contract"]["path"] == str(context_path)
