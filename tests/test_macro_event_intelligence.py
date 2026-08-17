import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import macro_event_intelligence as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_macro_event_intelligence_upgrades_replay_completed_event(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    _write_json(health / "macro_auto_watch_status.json", {"live_detected": False, "post_live_replay_completed": True})
    _write_json(health / "macro_auto_watch_state.json", {"source": "C-SPAN", "speaker": "Kevin Warsh"})
    _write_json(health / "live_macro_media_status.json", {"status": "ready"})
    _write_json(
        live_macro_path,
        {
            "source": "C-SPAN",
            "speaker": "Kevin Warsh",
            "stance": "hawkish",
            "sentiment_hint": -0.75,
            "shock_hint": 1.0,
            "summary": "hawkish remarks on inflation and rates",
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path)

    assert payload["overall_status"] == "ready"
    assert payload["transcript_quality"] == "full_replay"
    assert payload["market_relevance"] == "high"
    assert payload["replay_contract"]["post_replay_summary_ready"] is True


def test_macro_event_intelligence_uses_media_quality_before_replay(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(health / "macro_auto_watch_status.json", {"live_detected": True, "post_live_replay_completed": False})
    _write_json(health / "macro_auto_watch_state.json", {"source": "C-SPAN", "speaker": "Kevin Warsh"})
    _write_json(health / "live_macro_media_status.json", {"status": "running", "transcript_quality_norm": 0.81, "cue_count": 12, "asr_backend": "mlx_whisper"})
    _write_json(
        media_latest_path,
        {"transcript_quality_norm": 0.81, "transcript_cue_match_norm": 0.74, "cue_count": 12, "asr_backend": "mlx_whisper"},
    )
    _write_json(
        live_macro_path,
        {
            "source": "C-SPAN",
            "speaker": "Kevin Warsh",
            "stance": "hawkish",
            "sentiment_hint": -0.75,
            "shock_hint": 1.0,
            "summary": "hawkish remarks on inflation and rates",
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    assert payload["overall_status"] == "ready"
    assert payload["transcript_quality"] == "aligned_transcript"
    assert payload["transcript_quality_score"] == 0.81
    assert payload["cue_match_score"] == 0.74
    assert payload["transcript_source"] == "mlx_whisper"


def test_macro_event_intelligence_recognizes_manual_replay_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(health / "macro_auto_watch_status.json", {"live_detected": False, "post_live_replay_completed": False})
    _write_json(health / "macro_auto_watch_state.json", {"source": "C-SPAN", "speaker": "Kevin Warsh", "post_live_replay_completed": False})
    _write_json(
        health / "live_macro_media_status.json",
        {
            "ok": True,
            "analysis_file": "/tmp/analysis.json",
            "transcript_quality_norm": 0.53,
            "cue_count": 28,
            "asr_backend": "mlx_whisper",
        },
    )
    _write_json(
        media_latest_path,
        {
            "ok": True,
            "analysis_file": "/tmp/analysis.json",
            "transcript_quality_norm": 0.53,
            "cue_count": 28,
            "asr_backend": "mlx_whisper",
        },
    )
    _write_json(
        live_macro_path,
        {
            "source": "C-SPAN",
            "speaker": "Kevin Warsh",
            "stance": "dovish",
            "sentiment_hint": 0.19,
            "shock_hint": 1.0,
            "items": [
                {
                    "headline": "Kevin Warsh full video transcript replay",
                    "summary": "cleaner second pass after the live stream ended",
                }
            ],
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    assert payload["overall_status"] == "ready"
    assert payload["transcript_quality"] == "full_replay"
    assert payload["media_status"] == "ready"
    assert payload["replay_contract"]["replay_completed"] is True


def test_macro_event_intelligence_ignores_stale_media_from_different_event(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(health / "macro_auto_watch_status.json", {"live_detected": False, "post_live_replay_completed": False})
    _write_json(health / "macro_auto_watch_state.json", {})
    _write_json(
        health / "live_macro_media_status.json",
        {
            "ok": True,
            "source": "CGTN",
            "speaker": "Jensen Huang",
            "timestamp_utc": "2026-05-14T17:22:20+00:00",
            "transcript_quality_norm": 0.9,
            "cue_count": 20,
            "asr_backend": "mlx_whisper",
        },
    )
    _write_json(media_latest_path, {"ok": True, "source": "CGTN", "timestamp_utc": "2026-05-14T17:22:20+00:00", "transcript_quality_norm": 0.9})
    _write_json(
        live_macro_path,
        {
            "source": "Company Earnings Call",
            "published": "2026-05-20T14:28:36+00:00",
            "stance": "mixed",
            "sentiment_hint": -0.2,
            "shock_hint": 0.85,
            "items": [
                {
                    "headline": "NVIDIA Q1 FY2027 earnings today",
                    "summary": "results and call are scheduled today",
                    "published": "2026-05-20T14:28:36+00:00",
                }
            ],
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    assert payload["overall_status"] == "ready"
    assert payload["media_context_status"] == "stale_or_different_event"
    assert payload["transcript_quality"] == "live_excerpt"
    assert payload["transcript_quality_score"] == 0.0


def test_macro_event_intelligence_reports_unverified_schwab_calendar(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(
        health / "macro_auto_watch_status.json",
        {
            "correlate_with_schwab_calendar": True,
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "client_has_no_calendar_methods",
            "calendar_correlation_source": "schwab.client",
        },
    )
    _write_json(health / "macro_auto_watch_state.json", {})
    _write_json(health / "live_macro_media_status.json", {})
    _write_json(media_latest_path, {})
    _write_json(
        live_macro_path,
        {
            "source": "Company Earnings Call",
            "published": "2026-05-20T14:28:36+00:00",
            "shock_hint": 0.85,
            "items": [{"headline": "NVIDIA Q1 FY2027 earnings today", "summary": "official event source is active"}],
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    assert payload["calendar_verification"]["status"] == "unverified"
    assert payload["calendar_verification"]["reason"] == "client_has_no_calendar_methods"
    assert any("Schwab calendar verification" in action for action in payload["recommended_actions"])


def test_macro_event_intelligence_rejects_fomc_calendar_for_nvda_earnings(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(
        health / "macro_auto_watch_status.json",
        {
            "correlate_with_schwab_calendar": True,
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": True,
            "calendar_correlation_reason": "matched_event",
            "calendar_correlation_source": "federalreserve.gov",
            "calendar_event_title": "FOMC Press Conference",
            "calendar_event_time_utc": "2026-06-17T18:30:00+00:00",
            "calendar_matched_terms": ["FOMC", "Federal Reserve"],
        },
    )
    _write_json(health / "macro_auto_watch_state.json", {})
    _write_json(health / "live_macro_media_status.json", {})
    _write_json(media_latest_path, {})
    _write_json(
        live_macro_path,
        {
            "template": "earnings_call",
            "source": "Company Earnings Call",
            "published": "2026-05-20T20:05:00+00:00",
            "shock_hint": 0.9,
            "symbols": ["NVDA", "QQQ", "SMH"],
            "items": [{"headline": "NVIDIA Q1 FY2027 post-earnings move watch", "summary": "watch after-hours reaction and next-session continuation"}],
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    verification = payload["calendar_verification"]
    assert verification["ok"] is False
    assert verification["status"] == "unverified"
    assert verification["mismatch"] is True
    assert verification["reason"] == "calendar_event_mismatch:matched_event"
    assert verification["event_title"] == "FOMC Press Conference"
    assert any("NVIDIA IR preset" in action for action in payload["recommended_actions"])


def test_macro_event_intelligence_rejects_fomc_calendar_for_spacex_ipo(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(
        health / "macro_auto_watch_status.json",
        {
            "correlate_with_schwab_calendar": True,
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": True,
            "calendar_correlation_reason": "already_matched_for_stream",
            "calendar_correlation_source": "federalreserve.gov",
            "calendar_event_title": "FOMC Press Conference",
            "calendar_event_time_utc": "2026-06-17T18:30:00+00:00",
            "calendar_matched_terms": ["FOMC", "Federal Reserve"],
        },
    )
    _write_json(health / "macro_auto_watch_state.json", {})
    _write_json(health / "live_macro_media_status.json", {})
    _write_json(media_latest_path, {})
    _write_json(
        live_macro_path,
        {
            "template": "generic",
            "source": "IPO event prep",
            "published": "2026-06-11T11:40:00+00:00",
            "shock_hint": 1.0,
            "symbols": ["SPCX", "TSLA", "RKLB", "QQQ"],
            "items": [
                {
                    "headline": "SpaceX IPO watch: SPCX expected to begin trading Friday June 12",
                    "summary": "monitor high-volatility IPO first print and related space proxies",
                }
            ],
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    verification = payload["calendar_verification"]
    assert verification["ok"] is False
    assert verification["status"] == "unverified"
    assert verification["mismatch"] is True
    assert verification["mismatch_expected_terms"] == ["spacex", "spcx", "ipo"]
    assert verification["reason"] == "calendar_event_mismatch:already_matched_for_stream"
    assert any("active bulletin" in action for action in payload["recommended_actions"])


def test_macro_event_intelligence_treats_official_release_as_event_evidence(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    live_macro_path = tmp_path / "live_macro_latest.json"
    media_latest_path = tmp_path / "media_latest.json"
    _write_json(health / "macro_auto_watch_status.json", {})
    _write_json(health / "macro_auto_watch_state.json", {})
    _write_json(health / "live_macro_media_status.json", {})
    _write_json(media_latest_path, {})
    _write_json(
        live_macro_path,
        {
            "template": "earnings_call",
            "source": "NVIDIA IR",
            "speaker": "NVIDIA management",
            "published": "2026-05-20T20:20:00+00:00",
            "url": "https://www.globenewswire.com/news-release/2026/05/20/3298888/0/en/nvidia-announces-financial-results-for-first-quarter-fiscal-2027.html",
            "shock_hint": 1.0,
            "symbols": ["NVDA", "QQQ", "SMH"],
            "items": [{"headline": "NVIDIA Q1 FY2027 post-earnings move watch", "summary": "official results release and next-session continuation watch"}],
        },
    )

    payload = src.build_payload(project_root, live_macro_path=live_macro_path, media_latest_path=media_latest_path)

    assert payload["overall_status"] == "ready"
    assert payload["transcript_quality"] == "official_release"
    assert not any("caption-aligned" in action for action in payload["recommended_actions"])
