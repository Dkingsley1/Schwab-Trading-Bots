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
