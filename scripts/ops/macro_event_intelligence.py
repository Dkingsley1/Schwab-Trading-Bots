#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "macro_event_intelligence_latest.json"
DEFAULT_LIVE_MACRO_PATH = Path("/Volumes/BOT_LOGS/schwab_trading_bot/data/external_context/live_macro_latest.json")
DEFAULT_MEDIA_LATEST_PATH = DEFAULT_LIVE_MACRO_PATH.parent / "live_macro_media" / "latest.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_replay_text(markers: list[Any]) -> str:
    parts: list[str] = []
    for marker in markers:
        if isinstance(marker, str):
            text = marker.strip()
            if text:
                parts.append(text.lower())
    return " ".join(parts)


def _infer_manual_replay_completed(live_macro: dict[str, Any], media: dict[str, Any], media_latest: dict[str, Any]) -> bool:
    replay_text = _collect_replay_text(
        [
            live_macro.get("headline"),
            live_macro.get("summary"),
            live_macro.get("content"),
            live_macro.get("analysis_mode"),
            live_macro.get("mode"),
            *((item or {}).get("headline") for item in live_macro.get("items", []) if isinstance(item, dict)),
            *((item or {}).get("summary") for item in live_macro.get("items", []) if isinstance(item, dict)),
        ]
    )
    replay_marked = any(
        token in replay_text
        for token in (
            "full video transcript replay",
            "full video replay",
            "post-live replay",
            "post live replay",
        )
    )
    media_ready = bool(
        media.get("ok", False)
        or media_latest.get("ok", False)
        or media.get("analysis_file")
        or media_latest.get("analysis_file")
        or media.get("transcript_file")
        or media_latest.get("transcript_file")
    )
    return replay_marked and media_ready


def _infer_media_status(media: dict[str, Any], media_latest: dict[str, Any]) -> str:
    explicit = str(media.get("status") or media_latest.get("status") or "").strip().lower()
    if explicit:
        return explicit
    if bool(media.get("ok", False) or media_latest.get("ok", False)):
        if any(
            source.get(key)
            for source in (media, media_latest)
            for key in ("analysis_file", "transcript_file", "alignment_file", "audio_file")
        ):
            return "ready"
        return "running"
    return "missing"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    live_macro_path: Path = DEFAULT_LIVE_MACRO_PATH,
    media_latest_path: Path = DEFAULT_MEDIA_LATEST_PATH,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    status = load_json(health_root / "macro_auto_watch_status.json")
    state = load_json(health_root / "macro_auto_watch_state.json")
    media = load_json(health_root / "live_macro_media_status.json")
    media_latest = load_json(media_latest_path)
    live_macro = load_json(live_macro_path)

    live_detected = bool(status.get("live_detected", False) or state.get("live_detected", False))
    replay_pending = bool(status.get("post_live_replay_pending", False) or state.get("post_live_replay_pending", False))
    replay_completed = bool(status.get("post_live_replay_completed", False) or state.get("post_live_replay_completed", False))
    replay_completed = replay_completed or _infer_manual_replay_completed(live_macro, media, media_latest)
    transcript_quality = "missing"
    excerpt = str(live_macro.get("excerpt") or live_macro.get("summary") or "").strip()
    transcript_quality_norm = max(
        _safe_float(media_latest.get("transcript_quality_norm"), 0.0),
        _safe_float(media.get("transcript_quality_norm"), 0.0),
    )
    cue_match_norm = max(
        _safe_float(media_latest.get("transcript_cue_match_norm"), 0.0),
        _safe_float(media.get("transcript_cue_match_norm"), 0.0),
    )
    cue_count = int(
        max(
            _safe_float(media_latest.get("cue_count"), 0.0),
            _safe_float(media.get("cue_count"), 0.0),
        )
    )
    transcript_source = str(
        media_latest.get("asr_backend")
        or media.get("asr_backend")
        or live_macro.get("transcript_source")
        or ""
    ).strip()
    if transcript_quality_norm >= 0.75 and cue_count > 0:
        transcript_quality = "aligned_transcript"
    elif transcript_quality_norm >= 0.55:
        transcript_quality = "asr_transcript"
    if excerpt and transcript_quality == "missing":
        transcript_quality = "live_excerpt"
    if replay_completed:
        transcript_quality = "full_replay"

    speaker = str(live_macro.get("speaker") or state.get("speaker") or "").strip()
    source = str(live_macro.get("source") or state.get("source") or "").strip()
    stance = str(live_macro.get("stance") or "").strip().lower()
    sentiment_hint = _safe_float(live_macro.get("sentiment_hint"), 0.0)
    shock_hint = _safe_float(live_macro.get("shock_hint"), 0.0)
    media_status = _infer_media_status(media, media_latest)

    overall_status = "ready" if (live_detected or replay_completed or media_status in {"running", "ready"} or transcript_quality_norm >= 0.55) else "degraded"
    market_relevance = "high" if abs(sentiment_hint) >= 0.5 or shock_hint >= 0.8 else ("medium" if abs(sentiment_hint) >= 0.2 else "low")

    recommended_actions = ordered_unique(
        [
            "keep the live ingest running until the event ends so the second-pass replay has a continuous source record" if live_detected else "",
            "run the completed-video replay pass before treating the live sentiment as final" if replay_pending and not replay_completed else "",
            "promote the post-replay summary into the macro bulletin lane when transcript quality reaches full_replay" if replay_completed else "",
            "prefer caption-aligned or replay transcripts over raw live excerpts before escalating macro sentiment into the trading lane" if transcript_quality in {"missing", "live_excerpt"} else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "source": source,
        "speaker": speaker,
        "stance": stance,
        "sentiment_hint": round(sentiment_hint, 4),
        "shock_hint": round(shock_hint, 4),
        "market_relevance": market_relevance,
        "transcript_quality": transcript_quality,
        "transcript_quality_score": round(transcript_quality_norm, 4),
        "cue_match_score": round(cue_match_norm, 4),
        "cue_count": cue_count,
        "transcript_source": transcript_source,
        "live_detected": live_detected,
        "media_status": media_status,
        "excerpt": excerpt,
        "replay_contract": {
            "replay_pending": replay_pending,
            "replay_completed": replay_completed,
            "full_video_required": bool(live_detected or replay_pending),
            "post_replay_summary_ready": replay_completed,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a cleaner event-to-trade intelligence contract across live ingest and replay.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--live-macro-path", default=str(DEFAULT_LIVE_MACRO_PATH))
    parser.add_argument("--media-latest-path", default=str(DEFAULT_MEDIA_LATEST_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        live_macro_path=Path(args.live_macro_path).expanduser(),
        media_latest_path=Path(args.media_latest_path).expanduser(),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "macro_event_intelligence "
            f"overall_status={payload.get('overall_status', '')} "
            f"market_relevance={payload.get('market_relevance', '')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
