#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.storage_mounts import resolve_external_storage
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from core.storage_mounts import resolve_external_storage
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "macro_event_intelligence_latest.json"
DEFAULT_LIVE_MACRO_PATH = resolve_external_storage().external_root / "data" / "external_context" / "live_macro_latest.json"
DEFAULT_MEDIA_LATEST_PATH = DEFAULT_LIVE_MACRO_PATH.parent / "live_macro_media" / "latest.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _first_item(live_macro: dict[str, Any]) -> dict[str, Any]:
    items = live_macro.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                return item
    return {}


def _event_excerpt(live_macro: dict[str, Any]) -> str:
    item = _first_item(live_macro)
    for key in ("excerpt", "summary", "content", "headline"):
        text = str(live_macro.get(key) or item.get(key) or "").strip()
        if text:
            return text
    return ""


def _media_current_for_event(live_macro: dict[str, Any], media: dict[str, Any]) -> bool:
    if not media:
        return True
    item = _first_item(live_macro)
    event_ts = _parse_dt(live_macro.get("published") or item.get("published") or live_macro.get("timestamp_utc"))
    media_ts = _parse_dt(media.get("timestamp_utc") or media.get("published"))
    if event_ts and media_ts and media_ts < event_ts:
        return False

    event_url = str(live_macro.get("url") or item.get("url") or "").strip().lower()
    media_url = str(media.get("youtube_url") or media.get("video_url") or media.get("resolved_video_url") or "").strip().lower()
    if event_url and media_url and event_url == media_url:
        return True

    event_source = str(live_macro.get("source") or item.get("source") or "").strip().lower()
    media_source = str(media.get("source") or "").strip().lower()
    event_speaker = str(live_macro.get("speaker") or item.get("speaker") or "").strip().lower()
    media_speaker = str(media.get("speaker") or "").strip().lower()
    if event_source and media_source and (event_source in media_source or media_source in event_source):
        return True
    if event_speaker and media_speaker and (event_speaker in media_speaker or media_speaker in event_speaker):
        return True

    return not bool(event_ts and media_ts)


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


def _calendar_verification(status: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(status.get("calendar_correlation_enabled", False) or status.get("correlate_with_schwab_calendar", False))
    ok = bool(status.get("calendar_correlation_ok", False))
    reason = str(status.get("calendar_correlation_reason") or ("disabled" if not enabled else "")).strip()
    source = str(status.get("calendar_correlation_source") or "").strip()
    if ok:
        verification_status = "verified"
    elif enabled:
        verification_status = "unverified"
    else:
        verification_status = "not_requested"
    return {
        "enabled": enabled,
        "ok": ok,
        "status": verification_status,
        "reason": reason,
        "source": source,
        "event_title": str(status.get("calendar_event_title") or "").strip(),
        "event_time_utc": str(status.get("calendar_event_time_utc") or "").strip(),
        "event_minutes_delta": _safe_float(status.get("calendar_event_minutes_delta"), 0.0),
        "matched_terms": [str(item) for item in status.get("calendar_matched_terms", []) if str(item).strip()]
        if isinstance(status.get("calendar_matched_terms"), list)
        else [],
    }


def _event_identity_text(live_macro: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("template", "source", "speaker", "headline", "summary", "content", "excerpt"):
        value = str(live_macro.get(key) or "").strip()
        if value:
            parts.append(value)
    symbols = live_macro.get("symbols")
    if isinstance(symbols, list):
        parts.extend(str(symbol or "").strip() for symbol in symbols if str(symbol or "").strip())
    items = live_macro.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("source", "speaker", "headline", "summary", "content", "publisher"):
                value = str(item.get(key) or "").strip()
                if value:
                    parts.append(value)
            item_symbols = item.get("symbols")
            if isinstance(item_symbols, list):
                parts.extend(str(symbol or "").strip() for symbol in item_symbols if str(symbol or "").strip())
    return " ".join(parts).lower()


def _calendar_identity_text(verification: dict[str, Any]) -> str:
    parts = [
        str(verification.get("source") or ""),
        str(verification.get("event_title") or ""),
        str(verification.get("reason") or ""),
    ]
    matched_terms = verification.get("matched_terms")
    if isinstance(matched_terms, list):
        parts.extend(str(term or "") for term in matched_terms)
    return " ".join(part for part in parts if part).lower()


def _guard_calendar_verification_event_match(
    live_macro: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    if not verification.get("enabled"):
        return verification
    live_text = _event_identity_text(live_macro)
    event_is_earnings = "earnings" in live_text or "earnings_call" in live_text or "company earnings call" in live_text
    event_is_nvda = "nvda" in live_text or "nvidia" in live_text
    if not (event_is_earnings and event_is_nvda):
        return verification

    calendar_text = _calendar_identity_text(verification)
    calendar_looks_fed = any(token in calendar_text for token in ("fomc", "federal reserve", "fed "))
    calendar_has_nvda = "nvda" in calendar_text or "nvidia" in calendar_text
    if not calendar_looks_fed or calendar_has_nvda:
        return verification

    guarded = dict(verification)
    prior_reason = str(guarded.get("reason") or "wrong_calendar_event").strip()
    guarded.update(
        {
            "ok": False,
            "status": "unverified",
            "reason": f"calendar_event_mismatch:{prior_reason}",
            "mismatch": True,
            "mismatch_expected_terms": ["nvidia", "nvda", "earnings"],
            "mismatch_actual_terms": [
                token
                for token in ("federal reserve", "fomc")
                if token in calendar_text
            ],
        }
    )
    return guarded


def _official_release_context(live_macro: dict[str, Any]) -> bool:
    text = _event_identity_text(live_macro)
    item = _first_item(live_macro)
    url = str(live_macro.get("url") or item.get("url") or "").strip().lower()
    official_source = any(token in text for token in ("nvidia ir", "investor relations"))
    official_url = any(token in url for token in ("nvidia.com", "nvidianews.nvidia.com", "globenewswire.com"))
    earnings_context = any(token in text for token in ("earnings", "results", "guidance", "post-earnings"))
    return bool(earnings_context and (official_source or official_url))


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
    media_context_status = "aligned"
    if not _media_current_for_event(live_macro, media):
        media = {}
        media_context_status = "stale_or_different_event"
    if not _media_current_for_event(live_macro, media_latest):
        media_latest = {}
        media_context_status = "stale_or_different_event"

    live_detected = bool(status.get("live_detected", False) or state.get("live_detected", False))
    replay_pending = bool(status.get("post_live_replay_pending", False) or state.get("post_live_replay_pending", False))
    replay_completed = bool(status.get("post_live_replay_completed", False) or state.get("post_live_replay_completed", False))
    replay_completed = replay_completed or _infer_manual_replay_completed(live_macro, media, media_latest)
    transcript_quality = "missing"
    excerpt = _event_excerpt(live_macro)
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
    if transcript_quality == "live_excerpt" and _official_release_context(live_macro):
        transcript_quality = "official_release"
    if replay_completed:
        transcript_quality = "full_replay"

    speaker = str(live_macro.get("speaker") or state.get("speaker") or "").strip()
    source = str(live_macro.get("source") or state.get("source") or "").strip()
    stance = str(live_macro.get("stance") or "").strip().lower()
    sentiment_hint = _safe_float(live_macro.get("sentiment_hint"), 0.0)
    shock_hint = _safe_float(live_macro.get("shock_hint"), 0.0)
    media_status = _infer_media_status(media, media_latest)
    calendar_verification = _guard_calendar_verification_event_match(live_macro, _calendar_verification(status))

    bulletin_present = bool(excerpt or _first_item(live_macro))
    overall_status = "ready" if (bulletin_present or live_detected or replay_completed or media_status in {"running", "ready"} or transcript_quality_norm >= 0.55) else "degraded"
    market_relevance = "high" if abs(sentiment_hint) >= 0.5 or shock_hint >= 0.8 else ("medium" if abs(sentiment_hint) >= 0.2 else "low")

    recommended_actions = ordered_unique(
        [
            "keep the live ingest running until the event ends so the second-pass replay has a continuous source record" if live_detected else "",
            "run the completed-video replay pass before treating the live sentiment as final" if replay_pending and not replay_completed else "",
            "promote the post-replay summary into the macro bulletin lane when transcript quality reaches full_replay" if replay_completed else "",
            "prefer caption-aligned or replay transcripts over raw live excerpts before escalating macro sentiment into the trading lane" if transcript_quality in {"missing", "live_excerpt"} else "",
            "treat Schwab calendar verification as unconfirmed and rely on the official event source until the Schwab calendar adapter exposes an earnings-calendar method"
            if calendar_verification.get("status") == "unverified"
            else "",
            "retarget the macro auto watcher to the NVIDIA IR preset before using calendar verification for this earnings event"
            if bool(calendar_verification.get("mismatch", False))
            else "",
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
        "media_context_status": media_context_status,
        "calendar_verification": calendar_verification,
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
