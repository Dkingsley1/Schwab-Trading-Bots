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
        "symbols": "",
        "language": "en",
        "audio_format": "mp3",
        "asr_backend": "auto",
        "asr_model": "",
        "media_root": str(tmp_path / "media"),
        "cue_archive_file": str(tmp_path / "live_macro_cues_latest.json"),
        "status_file": str(tmp_path / "live_macro_media_status.json"),
        "out_file": str(tmp_path / "live_macro_latest.json"),
        "expires_hours": 6.0,
        "cookies_from_browser": "",
        "wait_for_live_seconds": 0.0,
        "retry_interval_seconds": 5.0,
        "retain_policy": "all",
        "publish_bulletin": False,
        "min_actionable_score": 0.75,
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
        lambda youtube_url, cookies_from_browser: ({"id": "powell123", "title": "Powell Presser"}, ""),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_capture_audio_with_wait",
        lambda *args, **kwargs: (audio_file, 1, {"strategy": "existing_output", "cookie_mode": "public"}),
    )
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
    assert status["retained"] is True
    assert status["market_actionable"] is True
    assert status["transcript_quality_norm"] > 0.0
    assert status["source_priority_norm"] > 0.0
    assert Path(status["transcript_file"]).exists()
    assert Path(status["alignment_file"]).exists()
    assert Path(status["training_features_file"]).exists()

    rows = [json.loads(line) for line in Path(status["training_features_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["source_type"] == "caption_cue"
    assert rows[0]["stance"] in {"dovish", "neutral", "mixed", "hawkish"}
    assert rows[0]["transcript_quality_norm"] == status["transcript_quality_norm"]
    assert rows[0]["event_resolution_join_key"].startswith("live_macro:")


def test_analyze_market_usefulness_finds_generic_event_signals():
    analysis = live_macro_media_ingest._analyze_market_usefulness(
        template="generic",
        title="Presidential Address",
        speaker="Donald Trump",
        source="White House",
        transcript_text="We are opening talks, pursuing a ceasefire, and expect to withdraw within weeks while ensuring freedom of navigation through the Strait of Hormuz.",
        transcript_segments=[
            {
                "segment_index": 0,
                "start_seconds": 10.0,
                "end_seconds": 18.0,
                "text": "We are opening talks and expect to withdraw within weeks.",
            },
            {
                "segment_index": 1,
                "start_seconds": 19.0,
                "end_seconds": 26.0,
                "text": "We will ensure freedom of navigation through the Strait of Hormuz.",
            },
        ],
        cues=[],
        fallback_symbols=["SPY"],
        min_actionable_score=0.75,
    )

    assert analysis["actionable"] is True
    assert "deescalation_timeline" in analysis["signal_types"]
    assert "oil_shipping_supply" in analysis["signal_types"]
    assert "USO" in analysis["symbols"]


def test_analyze_market_usefulness_company_templates_require_symbols():
    analysis = live_macro_media_ingest._analyze_market_usefulness(
        template="earnings_call",
        title="AAPL Q2 Earnings Call",
        speaker="Tim Cook",
        source="Apple",
        transcript_text="We are raising guidance and seeing margin expansion across the business.",
        transcript_segments=[
            {
                "segment_index": 0,
                "start_seconds": 2.0,
                "end_seconds": 9.0,
                "text": "We are raising guidance and seeing margin expansion across the business.",
            }
        ],
        cues=[],
        fallback_symbols=[],
        min_actionable_score=0.75,
    )

    assert analysis["broad_market"] is False
    assert analysis["actionable"] is False
    assert analysis["blocked_reason"] == "missing_symbols_for_company_event"
    assert "positive_guidance" in analysis["signal_types"]


def test_analyze_market_usefulness_finds_legal_policy_signals():
    analysis = live_macro_media_ingest._analyze_market_usefulness(
        template="legal_policy",
        title="Supreme Court administrative law ruling",
        speaker="Supreme Court coverage",
        source="Court coverage",
        transcript_text="The Court limited agency authority, invoked the major questions doctrine, and set aside the rule for the administrative agency.",
        transcript_segments=[
            {
                "segment_index": 0,
                "start_seconds": 4.0,
                "end_seconds": 12.0,
                "text": "The Court limited agency authority and invoked the major questions doctrine.",
            },
            {
                "segment_index": 1,
                "start_seconds": 13.0,
                "end_seconds": 20.0,
                "text": "The justices set aside the rule for the administrative agency.",
            },
        ],
        cues=[],
        fallback_symbols=["SPY"],
        min_actionable_score=0.70,
    )

    assert analysis["actionable"] is True
    assert "agency_authority_limited" in analysis["signal_types"]
    assert "XLF" in analysis["symbols"]
    assert analysis["broad_market"] is True


def test_dedupe_signals_collapses_duplicate_legal_hits() -> None:
    signals = [
        {"signal_type": "agency_authority_limited", "score": 0.9, "excerpt": "The Court limited agency authority and invoked the major questions doctrine.", "segment_index": 0},
        {"signal_type": "agency_authority_limited", "score": 0.8, "excerpt": "The Court limited agency authority and invoked the major questions doctrine.", "segment_index": 1},
    ]

    deduped = live_macro_media_ingest._dedupe_signals(signals)

    assert len(deduped) == 1


def test_market_confirmation_requires_multi_segment_support_for_legal_policy() -> None:
    confirmation = live_macro_media_ingest._market_confirmation_state(
        template="legal_policy",
        signals=[{"segment_index": 0, "signal_type": "agency_authority_limited"}],
        source_profile={"priority_norm": 0.92},
        transcript_quality={"quality_norm": 0.71},
        event_resolution_join={"ready_norm": 1.0},
    )

    assert confirmation["required"] is True
    assert confirmation["confirmed"] is False
    assert confirmation["high_conviction_allowed"] is False


def test_source_priority_profile_and_official_candidates_support_legal_policy():
    profile = live_macro_media_ingest._source_priority_profile(
        template="legal_policy",
        title="Supreme Court oral argument coverage",
        speaker="Supreme Court / C-SPAN legal coverage",
        source="C-SPAN",
        youtube_url="https://www.youtube.com/watch?v=test123",
    )
    candidates = live_macro_media_ingest._official_source_candidates(
        template="legal_policy",
        source="Supreme Court / C-SPAN legal coverage",
        title="Supreme Court oral argument coverage",
    )

    assert profile["tier"] == "authoritative_broadcast"
    assert profile["priority_norm"] >= 0.90
    assert any("supremecourt.gov" in item["url"] for item in candidates)


def test_run_ingest_discards_non_actionable_media_when_requested(monkeypatch, tmp_path):
    audio_file = tmp_path / "captured.mp3"
    audio_file.write_bytes(b"fake-audio")

    monkeypatch.setattr(
        live_macro_media_ingest,
        "_extract_video_metadata",
        lambda youtube_url, cookies_from_browser: ({"id": "generic123", "title": "Routine Remarks"}, ""),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_capture_audio_with_wait",
        lambda *args, **kwargs: (audio_file, 1, {"strategy": "existing_output", "cookie_mode": "public"}),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_transcribe_audio",
        lambda *args, **kwargs: {
            "ok": True,
            "backend": "mlx_whisper",
            "model": "tiny",
            "language": "en",
            "text": "Thank you everyone for being here tonight.",
            "segments": [
                {
                    "segment_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "text": "Thank you everyone for being here tonight.",
                }
            ],
        },
    )
    monkeypatch.setattr(live_macro_media_ingest, "_load_caption_cues", lambda *args, **kwargs: {"cue_archive_file": "", "cue_count": 0, "matched_video": False, "cues": []})
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_FEATURES_DIR", tmp_path / "training")

    args = _args(tmp_path, template="generic", speaker="Donald Trump", source="White House", retain_policy="actionable_only")
    status = live_macro_media_ingest.run_ingest(args)

    assert status["ok"] is True
    assert status["retained"] is False
    assert status["scrapped_non_actionable"] is True
    assert status["transcript_file"] == ""
    assert not audio_file.exists()
    assert not (Path(args.media_root) / "generic123").exists()


def test_run_ingest_publishes_company_scoped_bulletin(monkeypatch, tmp_path):
    audio_file = tmp_path / "captured.mp3"
    audio_file.write_bytes(b"fake-audio")

    monkeypatch.setattr(
        live_macro_media_ingest,
        "_extract_video_metadata",
        lambda youtube_url, cookies_from_browser: ({"id": "earnings123", "title": "AAPL Q2 Earnings Call"}, ""),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_capture_audio_with_wait",
        lambda *args, **kwargs: (audio_file, 1, {"strategy": "existing_output", "cookie_mode": "public"}),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_transcribe_audio",
        lambda *args, **kwargs: {
            "ok": True,
            "backend": "mlx_whisper",
            "model": "tiny",
            "language": "en",
            "text": "We are raising guidance and seeing margin expansion with strong demand.",
            "segments": [
                {
                    "segment_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 6.0,
                    "text": "We are raising guidance and seeing margin expansion with strong demand.",
                }
            ],
        },
    )
    monkeypatch.setattr(live_macro_media_ingest, "_load_caption_cues", lambda *args, **kwargs: {"cue_archive_file": "", "cue_count": 0, "matched_video": False, "cues": []})
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_FEATURES_DIR", tmp_path / "training")

    args = _args(
        tmp_path,
        template="earnings_call",
        speaker="Tim Cook",
        source="Apple",
        symbols="AAPL",
        publish_bulletin=True,
    )
    status = live_macro_media_ingest.run_ingest(args)
    bulletin = json.loads(Path(args.out_file).read_text(encoding="utf-8"))

    assert status["market_actionable"] is True
    assert status["market_broad_market"] is False
    assert bulletin["broad_market"] is False
    assert bulletin["symbols"] == ["AAPL"]
    assert "positive_guidance" in bulletin["signal_types"]


def test_run_ingest_publishes_bulletin_when_actionable(monkeypatch, tmp_path):
    audio_file = tmp_path / "captured.mp3"
    audio_file.write_bytes(b"fake-audio")

    monkeypatch.setattr(
        live_macro_media_ingest,
        "_extract_video_metadata",
        lambda youtube_url, cookies_from_browser: ({"id": "speech123", "title": "National Address"}, ""),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_capture_audio_with_wait",
        lambda *args, **kwargs: (audio_file, 1, {"strategy": "existing_output", "cookie_mode": "public"}),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_transcribe_audio",
        lambda *args, **kwargs: {
            "ok": True,
            "backend": "mlx_whisper",
            "model": "tiny",
            "language": "en",
            "text": "We expect to withdraw within weeks and secure the Strait of Hormuz.",
            "segments": [
                {
                    "segment_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 6.0,
                    "text": "We expect to withdraw within weeks and secure the Strait of Hormuz.",
                }
            ],
        },
    )
    monkeypatch.setattr(live_macro_media_ingest, "_load_caption_cues", lambda *args, **kwargs: {"cue_archive_file": "", "cue_count": 0, "matched_video": False, "cues": []})
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_FEATURES_DIR", tmp_path / "training")

    args = _args(tmp_path, template="generic", speaker="Donald Trump", source="White House", publish_bulletin=True)
    status = live_macro_media_ingest.run_ingest(args)
    bulletin = json.loads(Path(args.out_file).read_text(encoding="utf-8"))

    assert status["bulletin_published"] is True
    assert bulletin["active"] is True
    assert set(bulletin["signal_types"]) == {"deescalation_timeline", "oil_shipping_supply"}
    assert bulletin["items"][0]["actionable_score"] >= 0.75
    assert bulletin["derived"]["news_features"]["news_source_quality_norm"] > 0.0
    assert bulletin["event_resolution_join"]["join_key"].startswith("live_macro:")


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
        return audio_file, {"strategy": "existing_output", "cookie_mode": "public"}

    monkeypatch.setattr(live_macro_media_ingest, "_capture_audio", _fake_capture)
    monkeypatch.setattr(live_macro_media_ingest.time, "sleep", lambda seconds: None)

    result, tries, context = live_macro_media_ingest._capture_audio_with_wait(
        "https://www.youtube.com/watch?v=test123",
        tmp_path,
        "vid123",
        audio_format="mp3",
        force_redownload=False,
        cookies_from_browser="",
        prefer_live_from_start=False,
        wait_for_live_seconds=60.0,
        retry_interval_seconds=5.0,
    )

    assert result == audio_file
    assert tries == 3
    assert context["strategy"] == "existing_output"


def test_capture_audio_prefers_live_from_start_when_requested(monkeypatch, tmp_path):
    commands = []

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, cwd, capture_output, text, check, timeout):
        commands.append(list(cmd))
        (tmp_path / "vid123.mp3").write_bytes(b"audio")
        return _FakeProc()

    monkeypatch.setattr(live_macro_media_ingest.subprocess, "run", _fake_run)

    audio_path, context = live_macro_media_ingest._capture_audio(
        "https://www.youtube.com/watch?v=test123",
        tmp_path,
        "vid123",
        audio_format="mp3",
        force_redownload=False,
        cookies_from_browser="",
        prefer_live_from_start=True,
    )

    assert audio_path == (tmp_path / "vid123.mp3")
    assert "--live-from-start" in commands[0]
    assert context["strategy"].startswith("live_from_start")


def test_prepend_bootstrap_cues_uses_early_captions_when_transcript_starts_late():
    merged, summary = live_macro_media_ingest._prepend_bootstrap_cues(
        [
            {
                "segment_index": 0,
                "start_seconds": 14.0,
                "end_seconds": 18.0,
                "text": "Tonight I want to update the nation.",
            }
        ],
        [
            {"cue_index": 0, "start_seconds": 0.0, "end_seconds": 3.0, "text": "My fellow Americans,"},
            {"cue_index": 1, "start_seconds": 4.0, "end_seconds": 8.0, "text": "thank you for joining me tonight."},
        ],
    )

    assert summary["prepended_count"] == 2
    assert merged[0]["source_type"] == "caption_bootstrap_segment"
    assert merged[1]["source_type"] == "caption_bootstrap_segment"
    assert merged[2]["text"] == "Tonight I want to update the nation."


def test_annotate_speaker_turns_extracts_labels_and_cleans_repeated_phrases():
    annotated, turns, summary = live_macro_media_ingest._annotate_speaker_turns(
        [
            {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "text": "SEN. MORENO: How do you let 10 million people in? How do you let 10 million people in?",
            },
            {
                "segment_index": 1,
                "start_seconds": 5.0,
                "end_seconds": 7.0,
                "text": "When you are trying to make sure everybody has jobs?",
            },
            {
                "segment_index": 2,
                "start_seconds": 9.0,
                "end_seconds": 12.0,
                "text": "MR. WARSH: It is as it is and that is the policy.",
            },
        ],
        default_speaker="Supreme Court / C-SPAN legal coverage",
    )

    assert annotated[0]["speaker_label"] == "SEN. MORENO"
    assert annotated[0]["text"] == "How do you let 10 million people in?"
    assert annotated[1]["speaker_label"] == "SEN. MORENO"
    assert annotated[2]["speaker_label"] == "MR. WARSH"
    assert summary["speaker_turn_count"] == 2
    assert turns[0]["speaker_label"] == "SEN. MORENO"
    assert turns[1]["speaker_label"] == "MR. WARSH"


def test_run_ingest_records_speaker_turn_metadata(monkeypatch, tmp_path):
    audio_file = tmp_path / "captured.mp3"
    audio_file.write_bytes(b"fake-audio")

    monkeypatch.setattr(
        live_macro_media_ingest,
        "_extract_video_metadata",
        lambda youtube_url, cookies_from_browser: ({"id": "hearing123", "title": "Warsh Hearing"}, ""),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_capture_audio_with_wait",
        lambda *args, **kwargs: (audio_file, 1, {"strategy": "existing_output", "cookie_mode": "public"}),
    )
    monkeypatch.setattr(
        live_macro_media_ingest,
        "_transcribe_audio",
        lambda *args, **kwargs: {
            "ok": True,
            "backend": "mlx_whisper",
            "model": "tiny",
            "language": "en",
            "text": "SEN. MORENO: How do you let 10 million people in? MR. WARSH: It is as it is and that is the policy.",
            "segments": [
                {
                    "segment_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 4.0,
                    "text": "SEN. MORENO: How do you let 10 million people in?",
                },
                {
                    "segment_index": 1,
                    "start_seconds": 5.0,
                    "end_seconds": 8.0,
                    "text": "MR. WARSH: It is as it is and that is the policy.",
                },
            ],
        },
    )
    monkeypatch.setattr(live_macro_media_ingest, "_load_caption_cues", lambda *args, **kwargs: {"cue_archive_file": "", "cue_count": 0, "matched_video": False, "cues": []})
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(live_macro_media_ingest, "DEFAULT_FEATURES_DIR", tmp_path / "training")

    status = live_macro_media_ingest.run_ingest(_args(tmp_path, template="legal_policy", speaker="Supreme Court / C-SPAN legal coverage", source="C-SPAN"))
    transcript_payload = json.loads(Path(status["transcript_file"]).read_text(encoding="utf-8"))

    assert status["speaker_turn_count"] == 2
    assert set(status["speakers_detected"]) == {"MR. WARSH", "SEN. MORENO"}
    assert transcript_payload["speaker_turn_count"] == 2
    assert set(transcript_payload["speakers_detected"]) == {"MR. WARSH", "SEN. MORENO"}
