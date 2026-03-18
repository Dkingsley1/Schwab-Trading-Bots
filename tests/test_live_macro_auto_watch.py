import argparse
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/live_macro_auto_watch.py")
spec = importlib.util.spec_from_file_location("live_macro_auto_watch", MODULE_PATH)
live_macro_auto_watch = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(live_macro_auto_watch)


def _args(**overrides):
    defaults = {
        "youtube_url": "",
        "youtube_channel_url": "",
        "template": "powell",
        "speaker": "Jerome Powell",
        "source": "Federal Reserve",
        "symbols": "",
        "poll_seconds": 45.0,
        "lookback_seconds": 240.0,
        "replay_full_video": False,
        "replay_window_seconds": 300.0,
        "expires_hours": 4.0,
        "out_file": "/tmp/live_macro_latest.json",
        "status_file": "/tmp/macro_auto_watch_status.json",
        "state_file": "/tmp/macro_auto_watch_state.json",
        "pid_file": "/tmp/macro_auto_watch.pid",
        "correlate_with_schwab_calendar": False,
        "trigger_media_ingest_on_live": False,
        "trigger_media_ingest_before_minutes": 0.0,
        "media_ingest_language": "en",
        "media_ingest_audio_format": "mp3",
        "media_ingest_asr_backend": "auto",
        "media_ingest_asr_model": "",
        "media_ingest_cookies_from_browser": "",
        "media_ingest_wait_buffer_seconds": 1800.0,
        "media_ingest_force_redownload": False,
        "once": False,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_normalize_channel_url_strips_live_and_streams_suffixes():
    assert live_macro_auto_watch._normalize_channel_url("https://www.youtube.com/@federalreserve/live") == "https://www.youtube.com/@federalreserve"
    assert live_macro_auto_watch._normalize_channel_url("https://www.youtube.com/@federalreserve/streams") == "https://www.youtube.com/@federalreserve"


def test_resolve_stream_target_reports_upcoming_channel_stream(monkeypatch):
    def _fake_run(args, timeout=90):
        assert args[-1].endswith("/live")
        return {
            "returncode": 1,
            "stdout": "null\n",
            "stderr": "ERROR: [youtube:tab] @federalreserve: The channel is not currently live\n",
            "payload": None,
        }

    monkeypatch.setattr(live_macro_auto_watch, "_run_yt_dlp_json", _fake_run)
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_probe_channel_streams",
        lambda channel_url: {
            "probe_url": f"{channel_url}/streams",
            "upcoming": {
                "title": "FOMC Press Conference, March 19, 2026",
                "video_url": "https://www.youtube.com/watch?v=upcoming123",
                "live_status": "is_upcoming",
            },
            "latest": {
                "title": "FOMC Press Conference, March 18, 2026",
                "video_url": "https://www.youtube.com/watch?v=prior123",
                "live_status": "was_live",
            },
        },
    )

    target = live_macro_auto_watch._resolve_stream_target(_args(youtube_channel_url="https://www.youtube.com/@federalreserve"))

    assert target["stream_state"] == "upcoming_detected"
    assert target["channel_url"] == "https://www.youtube.com/@federalreserve"
    assert target["upcoming_title"] == "FOMC Press Conference, March 19, 2026"
    assert target["upcoming_video_url"] == "https://www.youtube.com/watch?v=upcoming123"
    assert target["latest_live_status"] == "was_live"


def test_resolve_stream_target_prefers_live_channel_video(monkeypatch):
    def _fake_run(args, timeout=90):
        assert args[-1].endswith("/live")
        return {
            "returncode": 0,
            "stdout": '{"webpage_url":"https://www.youtube.com/watch?v=live123","live_status":"is_live","title":"Jerome Powell live remarks"}\n',
            "stderr": "",
            "payload": {
                "webpage_url": "https://www.youtube.com/watch?v=live123",
                "live_status": "is_live",
                "title": "Jerome Powell live remarks",
            },
        }

    monkeypatch.setattr(live_macro_auto_watch, "_run_yt_dlp_json", _fake_run)
    monkeypatch.setattr(live_macro_auto_watch, "_probe_channel_streams", lambda channel_url: (_ for _ in ()).throw(AssertionError("streams probe should not be needed")))

    target = live_macro_auto_watch._resolve_stream_target(_args(youtube_channel_url="https://www.youtube.com/@federalreserve"))

    assert target["stream_state"] == "live"
    assert target["resolved_video_url"] == "https://www.youtube.com/watch?v=live123"
    assert target["stream_title"] == "Jerome Powell live remarks"


def test_run_once_waiting_channel_state_is_healthy_and_dedupes_status_events(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_resolve_stream_target",
        lambda args: {
            "mode": "channel",
            "stream_state": "awaiting_live_stream",
            "channel_url": "https://www.youtube.com/@federalreserve",
            "live_probe_url": "https://www.youtube.com/@federalreserve/live",
            "streams_probe_url": "https://www.youtube.com/@federalreserve/streams",
            "latest_title": "Most Recent Fed Stream",
            "latest_video_url": "https://www.youtube.com/watch?v=prior123",
            "latest_live_status": "was_live",
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "append_live_macro_event",
        lambda **kwargs: events.append(kwargs) or "/tmp/live_macro_events.jsonl",
    )

    state = {}
    args = _args(youtube_channel_url="https://www.youtube.com/@federalreserve")
    out_path = tmp_path / "live_macro_latest.json"

    first = live_macro_auto_watch._run_once(args, state, out_path)
    second = live_macro_auto_watch._run_once(args, state, out_path)

    assert first["ok"] is True
    assert first["stream_state"] == "awaiting_live_stream"
    assert first["updated_bulletin"] is False
    assert first["youtube_channel_url"] == "https://www.youtube.com/@federalreserve"
    assert len(events) == 1
    assert events[0]["event_type"] == "channel_status"
    assert second["ok"] is True
    assert len(events) == 1


def test_classify_text_uses_expanded_keyword_sets_without_carry_forward():
    hawkish = live_macro_auto_watch._classify_text(
        "We need more confidence that inflation is coming down before it is appropriate to cut and policy needs to remain restrictive.",
        "neutral",
        0.0,
        allow_carry_forward=False,
    )
    dovish = live_macro_auto_watch._classify_text(
        "Disinflation is continuing, the labor market has cooled, and policy can become less restrictive over time.",
        "neutral",
        0.0,
        allow_carry_forward=False,
    )

    assert hawkish["stance"] == "hawkish"
    assert "need more confidence" in [hit["token"] for hit in hawkish["hawkish_hits"]]
    assert dovish["stance"] == "dovish"
    assert "disinflation" in [hit["token"] for hit in dovish["dovish_hits"]]


def test_run_once_full_video_replay_uses_whole_transcript_and_records_scores(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(live_macro_auto_watch, "CUE_EVENTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_resolve_stream_target",
        lambda args: {
            "mode": "video",
            "stream_state": "live",
            "resolved_video_url": "https://www.youtube.com/watch?v=replay123",
            "channel_url": "",
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "append_live_macro_event",
        lambda **kwargs: events.append(kwargs) or "/tmp/live_macro_events.jsonl",
    )

    def _fake_fetch(url, work_dir):
        path = Path(work_dir) / "replay.vtt"
        path.write_text(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:05.000\n"
            "Disinflation is continuing and the labor market has cooled.\n\n"
            "00:05:10.000 --> 00:05:16.000\n"
            "Policy can become less restrictive over time.\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(live_macro_auto_watch, "_fetch_caption_snapshot", _fake_fetch)

    state = {"last_stance": "hawkish", "last_sentiment_hint": -0.75}
    args = _args(
        youtube_url="https://www.youtube.com/watch?v=replay123",
        replay_full_video=True,
        once=True,
        json=True,
    )
    out_path = tmp_path / "live_macro_latest.json"

    status = live_macro_auto_watch._run_once(args, state, out_path)

    assert status["analysis_mode"] == "full_video_replay"
    assert status["stance"] == "dovish"
    assert status["classification_reason"] == "keyword_match"
    assert status["window_count"] >= 1
    assert "disinflation" in status["dovish_keywords"]
    assert status["cue_archive_file"].endswith("live_macro_cues_latest.json")
    assert Path(status["cue_archive_file"]).exists()
    assert status["cue_events_file"].endswith(".jsonl")
    assert Path(status["cue_events_file"]).exists()
    assert events[0]["event_type"] == "video_replay"


def test_run_once_channel_live_triggers_media_ingest_and_calendar_correlation_once(monkeypatch, tmp_path):
    events = []
    launches = []
    correlations = []
    monkeypatch.setattr(live_macro_auto_watch, "CUE_EVENTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_resolve_stream_target",
        lambda args: {
            "mode": "channel",
            "stream_state": "live",
            "resolved_video_url": "https://www.youtube.com/watch?v=fedlive123",
            "channel_url": "https://www.youtube.com/@federalreserve",
            "stream_title": "FOMC Press Conference",
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "append_live_macro_event",
        lambda **kwargs: events.append(kwargs) or "/tmp/live_macro_events.jsonl",
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_launch_media_ingest_for_stream",
        lambda args, youtube_url, wait_for_live_seconds=0.0: launches.append((youtube_url, wait_for_live_seconds)) or {
            "media_ingest_triggered": True,
            "media_ingest_reason": "triggered_on_live_stream_detected",
            "media_ingest_pid": 12345,
            "media_ingest_log": "/tmp/macro_media_ingest.log",
            "media_ingest_command": ["python", "live_macro_media_ingest.py"],
            "media_ingest_wait_for_live_seconds": wait_for_live_seconds,
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_fetch_schwab_calendar_correlation",
        lambda args, target: correlations.append(target["resolved_video_url"]) or {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": True,
            "calendar_correlation_reason": "matched_event",
            "calendar_correlation_source": "schwab.get_economic_calendar",
            "calendar_match_score": 8.5,
            "calendar_matched_terms": ["powell", "fomc"],
            "calendar_event_title": "FOMC Press Conference",
            "calendar_event_time_utc": "2026-03-18T18:30:00+00:00",
            "calendar_event_minutes_delta": 5.0,
            "calendar_features": {"calendar_fomc_event_norm": 1.0},
        },
    )

    def _fake_fetch(url, work_dir):
        path = Path(work_dir) / "live.vtt"
        path.write_text(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:05.000\n"
            "Disinflation is continuing.\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(live_macro_auto_watch, "_fetch_caption_snapshot", _fake_fetch)

    state = {}
    args = _args(
        youtube_channel_url="https://www.youtube.com/@federalreserve",
        trigger_media_ingest_on_live=True,
        correlate_with_schwab_calendar=True,
    )
    out_path = tmp_path / "live_macro_latest.json"

    first = live_macro_auto_watch._run_once(args, state, out_path)
    second = live_macro_auto_watch._run_once(args, state, out_path)

    assert launches == [("https://www.youtube.com/watch?v=fedlive123", 0.0)]
    assert correlations == ["https://www.youtube.com/watch?v=fedlive123"]
    assert first["media_ingest_triggered"] is True
    assert first["calendar_correlation_ok"] is True
    assert first["calendar_event_title"] == "FOMC Press Conference"
    assert second["media_ingest_triggered"] is False
    assert second["media_ingest_reason"] == "already_triggered_for_stream"
    assert second["calendar_correlation_reason"] == "already_matched_for_stream"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["calendar_correlation_ok"] is True
    assert payload["items"][0]["calendar_event_title"] == "FOMC Press Conference"


def test_run_once_upcoming_channel_arms_media_ingest_inside_calendar_window(monkeypatch, tmp_path):
    events = []
    launches = []
    correlations = []
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_resolve_stream_target",
        lambda args: {
            "mode": "channel",
            "stream_state": "upcoming_detected",
            "resolved_video_url": "https://www.youtube.com/watch?v=upcomingfed123",
            "upcoming_video_url": "https://www.youtube.com/watch?v=upcomingfed123",
            "channel_url": "https://www.youtube.com/@federalreserve",
            "stream_title": "FOMC Press Conference",
            "upcoming_title": "FOMC Press Conference",
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "append_live_macro_event",
        lambda **kwargs: events.append(kwargs) or "/tmp/live_macro_events.jsonl",
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_launch_media_ingest_for_stream",
        lambda args, youtube_url, wait_for_live_seconds=0.0: launches.append((youtube_url, wait_for_live_seconds)) or {
            "media_ingest_triggered": True,
            "media_ingest_reason": "armed_from_calendar_prelive_window",
            "media_ingest_pid": 777,
            "media_ingest_log": "/tmp/macro_media_ingest_prelive.log",
            "media_ingest_command": ["python", "live_macro_media_ingest.py"],
            "media_ingest_wait_for_live_seconds": wait_for_live_seconds,
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_fetch_schwab_calendar_correlation",
        lambda args, target: correlations.append(target["resolved_video_url"]) or {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": True,
            "calendar_correlation_reason": "matched_event",
            "calendar_correlation_source": "schwab.get_economic_calendar",
            "calendar_match_score": 9.0,
            "calendar_event_title": "FOMC Press Conference",
            "calendar_event_time_utc": "2026-03-18T20:05:00+00:00",
            "calendar_event_minutes_delta": 3.0,
            "calendar_matched_terms": ["fomc", "press conference"],
            "calendar_features": {"calendar_fomc_event_norm": 1.0},
        },
    )
    monkeypatch.setattr(live_macro_auto_watch.time, "time", lambda: 1773864120.0)

    state = {}
    args = _args(
        youtube_channel_url="https://www.youtube.com/@federalreserve",
        trigger_media_ingest_on_live=True,
        trigger_media_ingest_before_minutes=5.0,
        correlate_with_schwab_calendar=True,
    )
    out_path = tmp_path / "live_macro_latest.json"

    status = live_macro_auto_watch._run_once(args, state, out_path)

    assert launches
    assert launches[0][0] == "https://www.youtube.com/watch?v=upcomingfed123"
    assert launches[0][1] >= 180.0
    assert correlations == ["https://www.youtube.com/watch?v=upcomingfed123"]
    assert status["stream_state"] == "upcoming_detected"
    assert status["media_ingest_triggered"] is True
    assert status["media_ingest_trigger_mode"] == "prelive"
    assert status["media_ingest_reason"] == "armed_from_calendar_prelive_window"
    assert status["calendar_event_title"] == "FOMC Press Conference"


def test_run_once_waiting_channel_can_arm_from_live_probe_url(monkeypatch, tmp_path):
    launches = []
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_resolve_stream_target",
        lambda args: {
            "mode": "channel",
            "stream_state": "awaiting_live_stream",
            "resolved_video_url": "",
            "channel_url": "https://www.youtube.com/@federalreserve",
            "live_probe_url": "https://www.youtube.com/@federalreserve/live",
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "append_live_macro_event",
        lambda **kwargs: "/tmp/live_macro_events.jsonl",
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_launch_media_ingest_for_stream",
        lambda args, youtube_url, wait_for_live_seconds=0.0: launches.append((youtube_url, wait_for_live_seconds)) or {
            "media_ingest_triggered": True,
            "media_ingest_reason": "armed_from_calendar_prelive_window",
            "media_ingest_pid": 901,
            "media_ingest_log": "/tmp/macro_media_ingest_waiting.log",
            "media_ingest_command": ["python", "live_macro_media_ingest.py"],
            "media_ingest_wait_for_live_seconds": wait_for_live_seconds,
        },
    )
    monkeypatch.setattr(
        live_macro_auto_watch,
        "_fetch_schwab_calendar_correlation",
        lambda args, target: {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": True,
            "calendar_correlation_reason": "matched_event",
            "calendar_event_title": "Fed Speech",
            "calendar_event_time_utc": "2026-03-18T20:05:00+00:00",
            "calendar_event_minutes_delta": 4.0,
        },
    )
    monkeypatch.setattr(live_macro_auto_watch.time, "time", lambda: 1773864060.0)

    args = _args(
        youtube_channel_url="https://www.youtube.com/@federalreserve",
        trigger_media_ingest_on_live=True,
        trigger_media_ingest_before_minutes=10.0,
        correlate_with_schwab_calendar=True,
    )
    status = live_macro_auto_watch._run_once(args, {}, tmp_path / "live_macro_latest.json")

    assert launches
    assert launches[0][0] == "https://www.youtube.com/@federalreserve/live"
    assert status["media_ingest_triggered"] is True
    assert status["media_ingest_trigger_mode"] == "prelive"


def test_fetch_federal_reserve_calendar_correlation_matches_without_api_key(monkeypatch):
    sample_html = """
    <html><body>
    <h4>FOMC Meetings</h4>
    <h6>Time:</h6>
    <h6>Release Date(s):</h6>
    <p>2:30 p.m.</p>
    <p>FOMC Press Conference</p>
    <p>18</p>
    <h4>Speeches</h4>
    <p>1:30 p.m.</p>
    <p>Speech - Chair Jerome H. Powell</p>
    <p>Brief Award Acceptance Remarks</p>
    <p>21</p>
    </body></html>
    """

    class FixedDateTime(__import__("datetime").datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 18, 18, 20, 0, tzinfo=__import__("datetime").timezone.utc)

    monkeypatch.setattr(live_macro_auto_watch, "datetime", FixedDateTime)
    monkeypatch.setattr(live_macro_auto_watch, "_fetch_text_url", lambda url, timeout_seconds=15.0: sample_html if "2026-march" in url else "")

    args = _args(
        youtube_channel_url="https://www.youtube.com/@federalreserve",
        speaker="Federal Reserve",
        source="Federal Reserve",
    )
    target = {
        "channel_url": "https://www.youtube.com/@federalreserve",
        "stream_title": "FOMC Press Conference",
    }

    result = live_macro_auto_watch._fetch_federal_reserve_calendar_correlation(args, target)

    assert result["calendar_correlation_ok"] is True
    assert result["calendar_correlation_source"] == "federalreserve.gov"
    assert result["calendar_event_title"] == "FOMC Press Conference"
    assert result["calendar_event_time_utc"].startswith("2026-03-18T14:30:00") is False
