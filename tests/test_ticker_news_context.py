from __future__ import annotations

from pathlib import Path
from typing import Any

import scripts.collect_ticker_news_context as ticker_news


def test_rss_items_parse_source_metadata() -> None:
    rss = """<?xml version="1.0"?>
    <rss><channel><item>
      <title>HUT rallies after Bitcoin mining update</title>
      <description>Hut 8 shares climb as bitcoin miners move.</description>
      <link>https://example.com/hut</link>
      <pubDate>Tue, 02 Jun 2026 14:00:00 GMT</pubDate>
    </item></channel></rss>
    """

    rows = ticker_news._rss_items(
        rss,
        publisher="Yahoo Finance",
        source_id="yahoo_finance_HUT",
        source_confidence_norm=0.7,
    )

    assert len(rows) == 1
    assert rows[0]["publisher"] == "Yahoo Finance"
    assert rows[0]["published_at"].startswith("2026-06-02T14:00:00")
    assert ticker_news._matches_symbol(rows[0], "HUT")


def test_build_payload_maps_mocked_news_sources(monkeypatch: Any, tmp_path: Path) -> None:
    def fake_fetch_text(url: str, **kwargs: Any) -> dict[str, Any]:
        if "feeds.finance.yahoo.com" in url:
            symbol = "NVDA"
            if "HUT" in url:
                symbol = "HUT"
            elif "NVDA" in url:
                symbol = "NVDA"
            return {
                "ok": True,
                "text": f"""<rss><channel><item>
                    <title>{symbol} beats earnings and rallies</title>
                    <description>{symbol} has a fresh catalyst.</description>
                    <link>https://example.com/{symbol.lower()}</link>
                    <pubDate>Tue, 02 Jun 2026 14:00:00 GMT</pubDate>
                </item></channel></rss>""",
                "status_code": 200,
                "duration_ms": 1.0,
            }
        if "coindesk" in url:
            return {
                "ok": True,
                "text": """<rss><channel><item>
                    <title>Bitcoin miners rally as crypto market firms</title>
                    <description>Bitcoin mining equities are moving.</description>
                    <link>https://example.com/crypto</link>
                    <pubDate>Tue, 02 Jun 2026 14:00:00 GMT</pubDate>
                </item></channel></rss>""",
                "status_code": 200,
                "duration_ms": 1.0,
            }
        return {"ok": False, "text": "", "error": "unexpected"}

    monkeypatch.setattr(ticker_news, "fetch_text", fake_fetch_text)

    payload = ticker_news.build_payload(
        project_root=tmp_path,
        symbols_arg="HUT,NVDA",
        max_runtime_seconds=10,
        timeout_seconds=1,
        sleep_seconds=0,
        include_optional_global_feeds=False,
        include_existing_schwab=False,
    )

    assert payload["overall_status"] == "ready"
    assert payload["symbols_with_news"] == 2
    assert payload["symbols"]["HUT"]["item_count"] > 0
    assert payload["derived"]["news_symbol_features"]["NVDA"]["ticker_news_available"] == 1.0
    assert payload["safety_contract"]["live_execution_allowed"] is False
    assert "/Volumes/VIDEO" in payload["safety_contract"]["protected_volumes"]
