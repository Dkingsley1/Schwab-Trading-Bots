from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/collect_schwab_education_context.py")
spec = importlib.util.spec_from_file_location("collect_schwab_education_context", MODULE_PATH)
schwab_education_context = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(schwab_education_context)


def test_extract_page_items_keeps_relevant_schwab_links() -> None:
    html = """
    <html><body>
      <a href="/coaching/ondemand-webcasts">On-demand Webcasts</a>
      <a href="/learn/trading/article-one">Article | Apr 12, 2026</a>
      <a href="/privacy/privacy-policy">Privacy</a>
      <a href="https://www.youtube.com/@CharlesSchwab">Charles Schwab YouTube</a>
    </body></html>
    """

    rows = schwab_education_context._extract_page_items(
        {"id": "test_page", "title": "Test Page", "url": "https://www.schwab.com/coaching/webcasts", "publisher": "Charles Schwab"},
        html,
        max_items=10,
    )

    urls = {row["url"] for row in rows}
    assert "https://www.schwab.com/coaching/ondemand-webcasts" in urls
    assert "https://www.youtube.com/@CharlesSchwab" in urls
    assert all("privacy" not in url for url in urls)


def test_build_payload_combines_page_and_channel_sources(monkeypatch) -> None:
    published = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        schwab_education_context,
        "_fetch_text_result",
        lambda url, source_name, user_agent, timeout: {
            "ok": True,
            "text": (
                f'<a href="/learn/trading/options">Apple options setup for AAPL | Apr 13, 2026</a>'
                '<a href="https://www.youtube.com/@SchwabNetwork">Schwab Network</a>'
            ),
            "fetched_utc": published,
            "source_confidence_norm": 0.97,
            "schema_confidence_norm": 0.93,
            "freshness_norm": 1.0,
        },
    )
    monkeypatch.setattr(
        schwab_education_context,
        "_channel_items",
        lambda spec, max_items, timeout: (
            [
                {
                    "headline": f'{spec["publisher"]} live archive on S&P 500 rates and Apple',
                    "title": f'{spec["publisher"]} live archive on S&P 500 rates and Apple',
                    "url": f'{spec["url"]}/watch',
                    "publisher": spec["publisher"],
                    "source": spec["publisher"],
                    "source_id": spec["id"],
                    "channel": spec["channel_name"],
                    "content_type": "youtube_stream",
                    "publishedDate": published,
                }
            ],
            {
                "ok": True,
                "videos_ok": True,
                "streams_ok": True,
                "item_count": 1,
                "errors": [],
                "fetched_utc": published,
                "source_confidence_norm": 0.95,
                "schema_confidence_norm": 0.9,
                "freshness_norm": 1.0,
            },
        ),
    )

    payload, status = schwab_education_context.build_payload(
        timeout_seconds=10.0,
        user_agent="test-agent",
        max_page_items=5,
        max_channel_items=5,
    )

    assert status["ok"] is True
    assert status["ok_source_count"] == status["source_count"]
    assert payload["provider"] == "schwab_education_context"
    assert payload["collection_contract"]["provider_confidence_norm"] > 0.0
    assert payload["collection_contract"]["source_contracts"]["schwab_live_webcasts"]["source_confidence_norm"] > 0.0
    assert status["source_contracts"]["schwab_network_youtube"]["schema_confidence_norm"] > 0.0
    assert payload["derived"]["news_features"]["news_source_quality_norm"] > 0.0
    assert payload["derived"]["news_features"]["news_items_24h"] > 0.0
    assert payload["derived"]["content_type_counts"]["youtube_stream"] >= 1
    assert payload["derived"]["global_features"]["schwab_education_video_share_norm"] > 0.0
    assert payload["derived"]["symbol_features"]["AAPL"]["news_available"] > 0.0
    assert payload["derived"]["symbol_features"]["AAPL"]["schwab_education_symbol_frequency_norm"] > 0.0
    assert payload["items"][0]["symbols"]
    assert payload["items"][0]["source_confidence_norm"] > 0.0
    assert payload["items"][0]["schema_confidence_norm"] > 0.0
    assert payload["items"][0]["freshness_norm"] > 0.0


def test_yt_playlist_limits_fetch_and_caps_timeout(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run(cmd, capture_output, text, check, timeout):  # type: ignore[no-untyped-def]
        calls.append({"cmd": list(cmd), "timeout": timeout})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps({"entries": [{"id": "abc123", "title": "Clip"}]}),
            stderr="",
        )

    monkeypatch.setattr(schwab_education_context, "YT_DLP_BIN", "/opt/homebrew/bin/yt-dlp")
    monkeypatch.setattr(schwab_education_context.subprocess, "run", _fake_run)

    payload, error = schwab_education_context._yt_playlist(
        "https://www.youtube.com/@SchwabNetwork",
        "videos",
        timeout=120.0,
        playlist_limit=5,
    )

    assert error is None
    assert payload["entries"][0]["id"] == "abc123"
    assert calls[0]["timeout"] == schwab_education_context.YT_PLAYLIST_TIMEOUT_MAX_SECONDS
    assert "--playlist-end" in calls[0]["cmd"]
    assert "5" in calls[0]["cmd"]
