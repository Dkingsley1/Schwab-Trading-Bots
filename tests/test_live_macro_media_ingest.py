import argparse
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/live_macro_media_ingest.py")
spec = importlib.util.spec_from_file_location("live_macro_media_ingest", MODULE_PATH)
live_macro_media_ingest = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(live_macro_media_ingest)


def _args(tmp_path: Path, **overrides):
    defaults = {
        "youtube_url": "https://www.youtube.com/watch?v=test123",
        "template": "powell",
        "speaker": "Jerome Powell",
        "source": "Federal Reserve",
        "language": "en",
        "audio_format": "mp3",
        "asr_backend": "auto",
        "asr_model": "",
        "media_root": str(tmp_path / "media"),
        "cue_archive_file": str(tmp_path / "live_macro_cues_latest.json"),
        "status_file": str(tmp_path / "live_macro_media_status.json"),
        "cookies_from_browser": "",
        "wait_for_live_seconds": 0.0,
        "retry_interval_seconds": 5.0,
        "force_redownload": False,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_align_transcript_segments_to_cues_uses_overlap_and_nearest():
    segments = [
        {"segment_index": 0, "start_seconds": 10.0, "end_seconds": 14.0, "text": "inflation has come down"},
        {"segment_index": 1, "start_seconds": 40.0, "end_seconds": 45.0, "text": "policy needs to remain restrictive"},
    ]
    cues = [
        {"cue_index": 0, "start_seconds": 9.0, "end_seconds": 13.0, "text": "inflation has come down"},
        {"cue_index": 1, "start_seconds": 44.0, "end_seconds": 46.0, "text": "remain restrictive"},
    ]

    rows = live_macro_media_ingest._align_transcript_segments_to_cues(segments, cues)

    assert rows[0]["cue_indices"] == [0]
    assert rows[0]["text_overlap_ratio"] == 1.0
    assert rows[1]["nearest_cue_index"] == 1
    assert rows[1]["matched_cue_count"] == 1


def test_run_ingest_writes_artifacts_and_caption_feature_fallback(monkeypatch, tmp_path):
    audio_file = tmp_path / "captured.mp3"
    audio_file.write_bytes(b"fake-audio")

    monkeypatch.setattr(
        live_macro_media_ingest,
        "_extract_video_metadata",
        lambda youtube_url, cookies_from_browser: {"id": "powell123", "title": "Powell Presser"},
    )
    monkeypatch.setattr(live_macro_media_ingest, "_capture_audio", lambda *args, **kwargs: audio_file)
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_transcribe_audio",
        lambda *args, **kwargs: {"ok": False, "backend": "auto", "error": "missing_backend", "text": "", "segments": []},
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_load_caption_cues",
        lambda cue_archive_path, youtube_url: {
            "cue_archive_file": str(cue_archive_path),
            "cue_count": 2,
            "matched_video": True,
            "cues": [
                {"cue_index": 0, "start_seconds": 0.0, "end_seconds": 3.0, "text": "inflation has come down"},
                {"cue_index": 1, "start_seconds": 4.0, "end_seconds": 8.0, "text": "policy can become less restrictive"},
            ],
        },
    )
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_FEATURES_DIR", tmp_path / "training")

    args = _args(tmp_path)
    status = live_macro_media_ingest.run_ingest(args)

    assert status["ok"] is True
    assert status["video_id"] == "powell123"
    assert status["asr_ok"] is False
    assert status["learning_ready"] is True
    assert Path(status["transcript_file"]).exists()
    assert Path(status["alignment_file"]).exists()
    assert Path(status["training_features_file"]).exists()

    rows = [json.loads(line) for line in Path(status["training_features_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["source_type"] == "caption_cue"
    assert rows[0]["stance"] in {"dovish", "neutral", "mixed", "hawkish"}


def test_yt_dlp_command_includes_cookies_when_requested():
    cmd = live_macro_media_ingest._yt_dlp_command(["--dump-single-json", "https://example.com"], cookies_from_browser="chrome")
    assert "--cookies-from-browser" in cmd
    assert "chrome" in cmd


def test_capture_audio_with_wait_retries_until_success(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.mp3"
    attempts = {"count": 0}

    def _fake_capture(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("not_live_yet")
        audio_file.write_bytes(b"audio")
        return audio_file

    monkeypatch.setattr(live_macro_media_ingest, "_capture_audio", _fake_capture)
    monkeypatch.setattr(live_macro_media_ingest.time, "sleep", lambda seconds: None)

    result, tries = live_macro_media_ingest._capture_audio_with_wait(
        "https://www.youtube.com/watch?v=test123",
        tmp_path,
        "vid123",
        audio_format="mp3",
        force_redownload=False,
        cookies_from_browser="",
        wait_for_live_seconds=60.0,
        retry_interval_seconds=5.0,
    )

    assert result == audio_file
    assert tries == 3
