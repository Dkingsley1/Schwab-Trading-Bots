#!/usr/bin/env python3
import argparse
import atexit
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.live_macro_bulletin import DEFAULT_OUT_PATH, append_live_macro_event, build_live_macro_payload


STATUS_PATH = PROJECT_ROOT / "governance" / "health" / "macro_auto_watch_status.json"
STATE_PATH = PROJECT_ROOT / "governance" / "health" / "macro_auto_watch_state.json"
PID_PATH = PROJECT_ROOT / "governance" / "health" / "macro_auto_watch.pid"
CUE_EVENTS_DIR = PROJECT_ROOT / "governance" / "events"
DEFAULT_CUE_ARCHIVE_BASENAME = "live_macro_cues_latest.json"
YT_DLP_BIN = shutil.which("yt-dlp") or "/opt/homebrew/bin/yt-dlp"
MEDIA_INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "live_macro_media_ingest.py"
LOG_DIR = PROJECT_ROOT / "logs"
_STREAMTEXT_URL_RE = re.compile(r"https://www\.streamtext\.net/player\?[^\s\"']+")
_TIME_LINE_RE = re.compile(r"^(?:\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.)|to be announced)$", re.IGNORECASE)
_DAY_LINE_RE = re.compile(r"^\d{1,2}$")
_FED_SECTION_HEADERS = {
    "Speeches",
    "FOMC Meetings",
    "Board Meetings",
    "Beige Book",
    "Statistical Releases",
    "Testimony",
    "Other",
}
_FED_LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone(timedelta(hours=-5))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _append_jsonl(path: Path, row: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return str(path)


def _parse_vtt_timestamp(raw: str) -> float:
    text = str(raw or "").strip()
    if not text:
        return 0.0
    hours = 0
    parts = text.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2].replace(",", "."))
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1].replace(",", "."))
    else:
        return 0.0
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def _parse_vtt(path: Path) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_lines: List[str] = []

    def _flush() -> None:
        nonlocal current_start, current_end, current_lines
        text = " ".join(line.strip() for line in current_lines if line.strip())
        if current_start is not None and current_end is not None and text:
            if not cues or cues[-1]["text"] != text:
                cues.append({"start": current_start, "end": current_end, "text": text})
        current_start = None
        current_end = None
        current_lines = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip("\ufeff")
        if "-->" in line:
            _flush()
            start_raw, end_raw = [chunk.strip() for chunk in line.split("-->", 1)]
            current_start = _parse_vtt_timestamp(start_raw.split(" ", 1)[0])
            current_end = _parse_vtt_timestamp(end_raw.split(" ", 1)[0])
        elif not line.strip():
            _flush()
        elif current_start is not None:
            current_lines.append(line.strip())
    _flush()
    return cues


def _latest_caption_text(cues: List[Dict[str, Any]], lookback_seconds: float) -> str:
    if not cues:
        return ""
    last_end = float(cues[-1]["end"])
    chunks: List[str] = []
    seen: set[str] = set()
    for cue in reversed(cues):
        if (last_end - float(cue["end"])) > max(float(lookback_seconds), 30.0):
            break
        text = str(cue["text"] or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append(text)
    chunks.reverse()
    return " ".join(chunks).strip()


def _summary_from_text(text: str, max_words: int = 18) -> str:
    words = [w for w in str(text or "").split() if w]
    return " ".join(words[: max(max_words, 6)]).strip()


_HAWKISH_TERM_WEIGHTS = {
    "higher for longer": 3.0,
    "restrictive": 1.5,
    "stay restrictive": 3.0,
    "maintain restrictive": 3.0,
    "policy needs to remain restrictive": 3.0,
    "policy remains restrictive": 2.5,
    "no rush to cut": 3.0,
    "not in a hurry to cut": 3.0,
    "not appropriate to cut": 3.0,
    "premature to ease": 3.0,
    "not cutting": 2.5,
    "higher rates": 2.0,
    "rates need to stay high": 3.0,
    "inflation remains too high": 3.0,
    "inflation is too high": 3.0,
    "inflation risks remain elevated": 3.0,
    "elevated inflation": 2.5,
    "sticky inflation": 2.5,
    "upside risk": 1.5,
    "upward pressure on prices": 2.0,
    "need more confidence": 2.0,
    "cannot declare victory": 2.5,
    "strong labor market": 1.5,
    "resilient labor market": 1.5,
    "resilient economy": 1.0,
    "price stability": 1.0,
}

_DOVISH_TERM_WEIGHTS = {
    "disinflation": 3.0,
    "inflation has eased": 2.5,
    "inflation has come down": 2.5,
    "inflation is coming down": 2.5,
    "cooling inflation": 2.5,
    "less restrictive": 3.0,
    "policy can become less restrictive": 3.0,
    "can ease": 2.0,
    "ease policy": 2.5,
    "policy easing": 2.5,
    "room to ease": 3.0,
    "cut rates": 3.0,
    "lower rates": 3.0,
    "downside risk": 2.0,
    "downside risks to growth": 2.5,
    "growth softening": 2.5,
    "slowing growth": 2.0,
    "below trend growth": 2.0,
    "softening labor": 2.5,
    "labor market cooling": 2.5,
    "labor market has cooled": 2.5,
    "weakening labor market": 2.5,
    "balanced risks": 1.5,
    "risks have become more balanced": 1.5,
    "policy is well positioned": 1.0,
}


def _score_weighted_terms(lowered_text: str, weights: Dict[str, float]) -> tuple[float, List[Dict[str, Any]]]:
    score = 0.0
    hits: List[Dict[str, Any]] = []
    for token, weight in weights.items():
        if token in lowered_text:
            score += float(weight)
            hits.append({"token": token, "weight": float(weight)})
    return score, hits


def _sentiment_from_scores(hawkish_score: float, dovish_score: float) -> float:
    total = max(float(hawkish_score) + float(dovish_score), 1.0)
    diff = float(dovish_score) - float(hawkish_score)
    sentiment = max(-0.75, min(0.75, 0.75 * (diff / total)))
    return round(sentiment, 4)


def _classify_text(
    text: str,
    previous_stance: str,
    previous_sentiment: float,
    *,
    allow_carry_forward: bool,
) -> Dict[str, Any]:
    lowered = str(text or "").lower()
    hawkish_score, hawkish_hits = _score_weighted_terms(lowered, _HAWKISH_TERM_WEIGHTS)
    dovish_score, dovish_hits = _score_weighted_terms(lowered, _DOVISH_TERM_WEIGHTS)
    total_score = hawkish_score + dovish_score
    diff = dovish_score - hawkish_score

    if total_score <= 0.0:
        if allow_carry_forward and previous_stance in {"hawkish", "dovish"}:
            return {
                "stance": previous_stance,
                "sentiment_hint": round(float(previous_sentiment), 4),
                "confidence": 0.15,
                "reason": "carry_forward_no_keyword_match",
                "hawkish_score": 0.0,
                "dovish_score": 0.0,
                "hawkish_hits": [],
                "dovish_hits": [],
            }
        return {
            "stance": "neutral",
            "sentiment_hint": 0.0,
            "confidence": 0.0,
            "reason": "no_keyword_match",
            "hawkish_score": 0.0,
            "dovish_score": 0.0,
            "hawkish_hits": [],
            "dovish_hits": [],
        }

    if hawkish_score > 0.0 and dovish_score > 0.0 and abs(diff) <= max(1.5, total_score * 0.20):
        return {
            "stance": "mixed",
            "sentiment_hint": _sentiment_from_scores(hawkish_score, dovish_score),
            "confidence": round(min(0.65, abs(diff) / max(total_score, 1.0)), 4),
            "reason": "mixed_keyword_balance",
            "hawkish_score": round(hawkish_score, 4),
            "dovish_score": round(dovish_score, 4),
            "hawkish_hits": hawkish_hits,
            "dovish_hits": dovish_hits,
        }

    stance = "dovish" if diff > 0.0 else "hawkish"
    confidence = round(min(1.0, abs(diff) / max(total_score, 1.0)), 4)
    return {
        "stance": stance,
        "sentiment_hint": _sentiment_from_scores(hawkish_score, dovish_score),
        "confidence": confidence,
        "reason": "keyword_match",
        "hawkish_score": round(hawkish_score, 4),
        "dovish_score": round(dovish_score, 4),
        "hawkish_hits": hawkish_hits,
        "dovish_hits": dovish_hits,
    }


def _fetch_caption_snapshot(url: str, work_dir: Path) -> Path:
    cmd = [
        YT_DLP_BIN,
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "en.*",
        "--sub-format",
        "vtt",
        "--no-playlist",
        "--no-progress",
        "--no-warnings",
        "--output",
        str(work_dir / "%(id)s.%(ext)s"),
        str(url),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "yt-dlp_failed").strip()[-1200:])
    candidates = sorted(work_dir.glob("*.vtt"))
    if not candidates:
        external_caption_url = _external_caption_url_from_video(url)
        if external_caption_url:
            raise RuntimeError(f"no_vtt_generated external_caption_url={external_caption_url}")
        raise RuntimeError("no_vtt_generated")
    return candidates[-1]


def _all_caption_text(cues: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    seen: set[str] = set()
    for cue in cues:
        text = str(cue.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append(text)
    return " ".join(chunks).strip()


def _windowed_caption_texts(cues: List[Dict[str, Any]], window_seconds: float) -> List[str]:
    if not cues:
        return []
    span = max(float(window_seconds), 60.0)
    windows: List[str] = []
    window_start = float(cues[0]["start"])
    current: List[str] = []
    seen: set[str] = set()

    def _flush() -> None:
        nonlocal current, seen
        text = " ".join(current).strip()
        if text:
            windows.append(text)
        current = []
        seen = set()

    for cue in cues:
        cue_start = float(cue["start"])
        if current and (cue_start - window_start) >= span:
            _flush()
            window_start = cue_start
        text = str(cue.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        current.append(text)
    _flush()
    return windows


def _archive_caption_cues(
    *,
    cues: List[Dict[str, Any]],
    out_path: Path,
    youtube_url: str,
    youtube_channel_url: str,
    analysis_mode: str,
    text_hash: str,
    replay_full_video: bool,
) -> Dict[str, Any]:
    cue_archive_path = out_path.with_name(DEFAULT_CUE_ARCHIVE_BASENAME)
    cue_events_path = CUE_EVENTS_DIR / f"live_macro_caption_cues_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    cue_rows: List[Dict[str, Any]] = []

    for idx, cue in enumerate(cues):
        cue_text = str(cue.get("text") or "").strip()
        if not cue_text:
            continue
        cue_row = {
            "timestamp_utc": _now_iso(),
            "category": "live_macro_caption_cue",
            "analysis_mode": analysis_mode,
            "replay_full_video": bool(replay_full_video),
            "youtube_url": youtube_url,
            "youtube_channel_url": youtube_channel_url,
            "cue_index": int(idx),
            "start_seconds": round(float(cue.get("start", 0.0) or 0.0), 3),
            "end_seconds": round(float(cue.get("end", 0.0) or 0.0), 3),
            "duration_seconds": round(max(float(cue.get("end", 0.0) or 0.0) - float(cue.get("start", 0.0) or 0.0), 0.0), 3),
            "text": cue_text,
            "text_hash": text_hash,
        }
        cue_rows.append(cue_row)

    cue_archive_payload = {
        "timestamp_utc": _now_iso(),
        "category": "live_macro_caption_cues",
        "analysis_mode": analysis_mode,
        "replay_full_video": bool(replay_full_video),
        "youtube_url": youtube_url,
        "youtube_channel_url": youtube_channel_url,
        "cue_count": len(cue_rows),
        "text_hash": text_hash,
        "cues": cue_rows,
    }
    _write_json(cue_archive_path, cue_archive_payload)
    cue_events_file = ""
    for cue_row in cue_rows:
        cue_events_file = _append_jsonl(cue_events_path, cue_row)
    return {
        "cue_archive_file": str(cue_archive_path),
        "cue_events_file": cue_events_file,
        "cue_count": len(cue_rows),
    }


def _normalize_channel_url(url: str) -> str:
    channel_url = str(url or "").strip().rstrip("/")
    for suffix in ("/live", "/streams"):
        if channel_url.endswith(suffix):
            channel_url = channel_url[: -len(suffix)]
            break
    return channel_url.rstrip("/")


def _youtube_watch_url(raw_url: str) -> str:
    text = str(raw_url or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return f"https://www.youtube.com/watch?v={text}"


def _load_json_output(raw: str) -> Optional[Dict[str, Any]]:
    for line in reversed(str(raw or "").splitlines()):
        candidate = line.strip()
        if not candidate or candidate == "null":
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _run_yt_dlp_json(args: List[str], *, timeout: int = 90) -> Dict[str, Any]:
    proc = subprocess.run(
        [YT_DLP_BIN, *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "payload": _load_json_output(proc.stdout),
    }


def _external_caption_url_from_video(url: str) -> str:
    probe = _run_yt_dlp_json(["--dump-single-json", "--no-playlist", "--no-warnings", str(url)], timeout=120)
    payload = probe.get("payload") or {}
    description = str(payload.get("description") or "")
    match = _STREAMTEXT_URL_RE.search(description)
    return match.group(0) if match else ""


def _probe_channel_streams(channel_url: str) -> Dict[str, Any]:
    streams_url = f"{_normalize_channel_url(channel_url)}/streams"
    probe = _run_yt_dlp_json(["--flat-playlist", "--dump-single-json", "--no-warnings", streams_url], timeout=120)
    payload = probe.get("payload") or {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {"probe_url": streams_url, "upcoming": None, "latest": None}

    live_states = {"is_live", "live"}
    upcoming_states = {"is_upcoming", "upcoming"}
    latest: Optional[Dict[str, Any]] = None
    upcoming: Optional[Dict[str, Any]] = None

    for entry in entries[:8]:
        if not isinstance(entry, dict):
            continue
        video_url = _youtube_watch_url(str(entry.get("webpage_url") or entry.get("url") or entry.get("id") or ""))
        if not video_url:
            continue
        item = {
            "video_url": video_url,
            "title": str(entry.get("title") or ""),
            "live_status": str(entry.get("live_status") or "").lower(),
            "channel": str(entry.get("channel") or ""),
            "release_timestamp": entry.get("release_timestamp"),
            "timestamp": entry.get("timestamp"),
        }
        if latest is None:
            latest = item
        if item["live_status"] in live_states:
            return {"probe_url": streams_url, "live": item, "upcoming": None, "latest": latest}
        if upcoming is None and item["live_status"] in upcoming_states:
            upcoming = item

    return {"probe_url": streams_url, "upcoming": upcoming, "latest": latest}


def _resolve_stream_target(args: argparse.Namespace) -> Dict[str, Any]:
    if args.youtube_url:
        return {
            "mode": "video",
            "stream_state": "live",
            "youtube_url": args.youtube_url,
            "resolved_video_url": args.youtube_url,
            "channel_url": "",
        }

    channel_url = _normalize_channel_url(args.youtube_channel_url)
    live_probe_url = f"{channel_url}/live"
    probe = _run_yt_dlp_json(["--dump-single-json", "--no-playlist", "--no-warnings", live_probe_url], timeout=90)
    payload = probe.get("payload") or {}
    stderr_blob = " ".join([str(probe.get("stdout") or ""), str(probe.get("stderr") or "")]).lower()
    live_status = str(payload.get("live_status") or "").lower() if isinstance(payload, dict) else ""
    resolved_video_url = _youtube_watch_url(str(payload.get("webpage_url") or payload.get("original_url") or payload.get("url") or ""))
    title = str(payload.get("title") or payload.get("fulltitle") or "")

    if resolved_video_url and (live_status in {"is_live", "live"} or bool(payload.get("is_live"))):
        return {
            "mode": "channel",
            "stream_state": "live",
            "youtube_url": resolved_video_url,
            "resolved_video_url": resolved_video_url,
            "channel_url": channel_url,
            "live_probe_url": live_probe_url,
            "stream_title": title,
        }

    streams_probe = _probe_channel_streams(channel_url)
    live_entry = streams_probe.get("live")
    if isinstance(live_entry, dict):
        return {
            "mode": "channel",
            "stream_state": "live",
            "youtube_url": str(live_entry.get("video_url") or ""),
            "resolved_video_url": str(live_entry.get("video_url") or ""),
            "channel_url": channel_url,
            "live_probe_url": live_probe_url,
            "streams_probe_url": streams_probe.get("probe_url"),
            "stream_title": str(live_entry.get("title") or ""),
        }

    status: Dict[str, Any] = {
        "mode": "channel",
        "stream_state": "awaiting_live_stream",
        "youtube_url": "",
        "resolved_video_url": "",
        "channel_url": channel_url,
        "live_probe_url": live_probe_url,
        "streams_probe_url": streams_probe.get("probe_url"),
    }
    if "not currently live" not in stderr_blob and int(probe.get("returncode", 1)) != 0:
        status["probe_warning"] = (str(probe.get("stderr") or probe.get("stdout") or "yt_dlp_channel_probe_failed").strip())[-400:]
    upcoming = streams_probe.get("upcoming")
    latest = streams_probe.get("latest")
    if isinstance(upcoming, dict):
        status["stream_state"] = "upcoming_detected"
        status["upcoming_title"] = str(upcoming.get("title") or "")
        status["upcoming_video_url"] = str(upcoming.get("video_url") or "")
        status["resolved_video_url"] = str(upcoming.get("video_url") or "")
        status["stream_title"] = str(upcoming.get("title") or "")
    if isinstance(latest, dict):
        status["latest_title"] = str(latest.get("title") or "")
        status["latest_video_url"] = str(latest.get("video_url") or "")
        status["latest_live_status"] = str(latest.get("live_status") or "")
    return status


def _launch_media_ingest_for_stream(args: argparse.Namespace, *, youtube_url: str, wait_for_live_seconds: float = 0.0) -> Dict[str, Any]:
    if not MEDIA_INGEST_SCRIPT.exists():
        raise RuntimeError("live_macro_media_ingest_script_missing")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"macro_media_ingest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        sys.executable,
        str(MEDIA_INGEST_SCRIPT),
        "--youtube-url",
        str(youtube_url),
        "--template",
        str(args.template),
        "--speaker",
        str(args.speaker),
        "--source",
        str(args.source),
        "--language",
        str(args.media_ingest_language),
        "--audio-format",
        str(args.media_ingest_audio_format),
        "--asr-backend",
        str(args.media_ingest_asr_backend),
    ]
    if str(args.media_ingest_asr_model or "").strip():
        cmd.extend(["--asr-model", str(args.media_ingest_asr_model)])
    if str(args.media_ingest_cookies_from_browser or "").strip():
        cmd.extend(["--cookies-from-browser", str(args.media_ingest_cookies_from_browser)])
    if max(float(wait_for_live_seconds or 0.0), 0.0) > 0.0:
        cmd.extend(["--wait-for-live-seconds", str(round(float(wait_for_live_seconds), 3))])
    if bool(args.media_ingest_force_redownload):
        cmd.append("--force-redownload")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now_iso()}] starting macro media ingest for {youtube_url}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    return {
        "media_ingest_triggered": True,
        "media_ingest_reason": "triggered_on_live_stream_detected",
        "media_ingest_pid": int(proc.pid),
        "media_ingest_log": str(log_path),
        "media_ingest_command": cmd,
        "media_ingest_wait_for_live_seconds": round(max(float(wait_for_live_seconds or 0.0), 0.0), 3),
    }


def _parse_iso_to_epoch(raw: Any) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _media_ingest_candidate_url(target: Dict[str, Any]) -> str:
    for key in ("resolved_video_url", "upcoming_video_url", "live_probe_url"):
        value = str(target.get(key) or "").strip()
        if value:
            return value
    return ""


def _media_ingest_identity(target: Dict[str, Any], calendar_correlation: Dict[str, Any]) -> str:
    event_time_utc = str(calendar_correlation.get("calendar_event_time_utc") or "").strip()
    channel_url = str(target.get("channel_url") or "").strip()
    candidate_url = _media_ingest_candidate_url(target)
    day_bucket = datetime.now(timezone.utc).strftime("%Y%m%d")
    if event_time_utc and channel_url:
        return f"{channel_url}|{event_time_utc}"
    if candidate_url and not candidate_url.endswith("/live"):
        return candidate_url
    if channel_url:
        return f"{channel_url}|{day_bucket}"
    return candidate_url or day_bucket


def _maybe_trigger_media_ingest(
    args: argparse.Namespace,
    state: Dict[str, Any],
    target: Dict[str, Any],
    calendar_correlation: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "media_ingest_enabled": bool(args.trigger_media_ingest_on_live),
        "media_ingest_triggered": False,
        "media_ingest_trigger_mode": "",
    }
    trigger_prelive_minutes = max(float(args.trigger_media_ingest_before_minutes or 0.0), 0.0)
    stream_state = str(target.get("stream_state") or "")
    if not bool(args.trigger_media_ingest_on_live):
        result["media_ingest_reason"] = "disabled"
        return result

    youtube_url = _media_ingest_candidate_url(target)
    if not youtube_url:
        result["media_ingest_reason"] = "missing_video_url"
        return result

    attempt_identity = _media_ingest_identity(target, calendar_correlation)
    attempted_identity = str(state.get("last_media_ingest_attempt_identity") or "")
    if attempted_identity and attempted_identity == attempt_identity:
        prior_result = str(state.get("last_media_ingest_attempt_result") or "")
        result.update(
            {
                "media_ingest_reason": "already_triggered_for_stream" if prior_result in {"triggered", "armed_prelive"} else (prior_result or "already_attempted_for_stream"),
                "media_ingest_pid": state.get("last_media_ingest_pid"),
                "media_ingest_log": str(state.get("last_media_ingest_log") or ""),
                "media_ingest_triggered_at_utc": str(state.get("last_media_ingest_triggered_utc") or ""),
                "media_ingest_error": str(state.get("last_media_ingest_error") or ""),
                "media_ingest_trigger_mode": str(state.get("last_media_ingest_trigger_mode") or ""),
                "media_ingest_wait_for_live_seconds": state.get("last_media_ingest_wait_for_live_seconds"),
            }
        )
        return result

    wait_for_live_seconds = 0.0
    trigger_reason = "live_stream_detected"
    result["media_ingest_prelive_window_minutes"] = trigger_prelive_minutes
    if stream_state in {"upcoming_detected", "awaiting_live_stream"}:
        if trigger_prelive_minutes <= 0.0:
            result["media_ingest_reason"] = "prelive_disabled"
            return result
        if not bool(calendar_correlation.get("calendar_correlation_ok")):
            result["media_ingest_reason"] = "calendar_not_matched"
            return result
        event_time_ts = _parse_iso_to_epoch(calendar_correlation.get("calendar_event_time_utc"))
        if event_time_ts is None:
            result["media_ingest_reason"] = "missing_calendar_time"
            return result
        seconds_until_event = event_time_ts - time.time()
        result["media_ingest_seconds_until_event"] = round(float(seconds_until_event), 3)
        if seconds_until_event > (trigger_prelive_minutes * 60.0):
            result["media_ingest_reason"] = "outside_prelive_window"
            return result
        wait_for_live_seconds = max(seconds_until_event, 0.0) + max(float(args.media_ingest_wait_buffer_seconds or 0.0), 0.0)
        trigger_reason = "calendar_prelive_arm"
        result["media_ingest_trigger_mode"] = "prelive"
    elif stream_state == "live":
        result["media_ingest_trigger_mode"] = "live"
    else:
        result["media_ingest_reason"] = f"unsupported_stream_state:{stream_state or 'unknown'}"
        return result

    attempted_at_utc = _now_iso()
    state["last_media_ingest_attempt_video_url"] = youtube_url
    state["last_media_ingest_attempt_identity"] = attempt_identity
    state["last_media_ingest_attempted_utc"] = attempted_at_utc

    try:
        launch = _launch_media_ingest_for_stream(args, youtube_url=youtube_url, wait_for_live_seconds=wait_for_live_seconds)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        state.update(
            {
                "last_media_ingest_attempt_result": f"{trigger_reason}_error",
                "last_media_ingest_error": error,
                "last_media_ingest_pid": None,
                "last_media_ingest_log": "",
                "last_media_ingest_trigger_mode": result.get("media_ingest_trigger_mode") or "",
                "last_media_ingest_wait_for_live_seconds": round(wait_for_live_seconds, 3),
            }
        )
        result.update(
            {
                "media_ingest_reason": f"{trigger_reason}_error",
                "media_ingest_error": error,
            }
        )
        append_live_macro_event(
            event_type="media_ingest_trigger_error",
            payload={
                "timestamp_utc": attempted_at_utc,
                "youtube_url": youtube_url,
                "youtube_channel_url": str(target.get("channel_url") or ""),
                "speaker": args.speaker,
                "source": args.source,
                "error": error,
            },
            out_file=out_path,
            extra={
                "youtube_url": youtube_url,
                "youtube_channel_url": str(target.get("channel_url") or ""),
                "stream_title": str(target.get("stream_title") or ""),
                "trigger_reason": trigger_reason,
            },
        )
        return result

    state.update(
        {
            "last_media_ingest_attempt_result": "armed_prelive" if trigger_reason == "calendar_prelive_arm" else "triggered",
            "last_media_ingest_trigger_video_url": youtube_url,
            "last_media_ingest_triggered_utc": attempted_at_utc,
            "last_media_ingest_pid": launch["media_ingest_pid"],
            "last_media_ingest_log": launch["media_ingest_log"],
            "last_media_ingest_error": "",
            "last_media_ingest_trigger_mode": result.get("media_ingest_trigger_mode") or "",
            "last_media_ingest_wait_for_live_seconds": launch.get("media_ingest_wait_for_live_seconds"),
        }
    )
    result.update(launch)
    result["media_ingest_triggered_at_utc"] = attempted_at_utc
    result["media_ingest_command"] = " ".join(launch["media_ingest_command"])
    result["media_ingest_reason"] = "armed_from_calendar_prelive_window" if trigger_reason == "calendar_prelive_arm" else "triggered_on_live_stream_detected"
    append_live_macro_event(
        event_type="media_ingest_prelive_arm" if trigger_reason == "calendar_prelive_arm" else "media_ingest_trigger",
        payload={
            "timestamp_utc": attempted_at_utc,
            "youtube_url": youtube_url,
            "youtube_channel_url": str(target.get("channel_url") or ""),
            "speaker": args.speaker,
            "source": args.source,
            "media_ingest_pid": launch["media_ingest_pid"],
            "media_ingest_log": launch["media_ingest_log"],
        },
        out_file=out_path,
        extra={
            "youtube_url": youtube_url,
            "youtube_channel_url": str(target.get("channel_url") or ""),
            "stream_title": str(target.get("stream_title") or ""),
            "trigger_reason": trigger_reason,
        },
    )
    return result


def _calendar_row_get_ci(row: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    key_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in key_map:
            return key_map[key.lower()]
    return None


def _calendar_parse_epoch(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1e12:
            value /= 1000.0
        if value > 1e9:
            return value
        return None
    text = str(raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        value = float(text)
        if value > 1e12:
            value /= 1000.0
        if value > 1e9:
            return value
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _iter_dict_nodes(node: Any) -> Any:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            for child in _iter_dict_nodes(value):
                yield child
    elif isinstance(node, list):
        for item in node:
            for child in _iter_dict_nodes(item):
                yield child


def _calendar_event_text(row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in (
        "title",
        "name",
        "event",
        "eventtype",
        "category",
        "description",
        "headline",
        "symbol",
        "ticker",
        "country",
        "importance",
        "impact",
        "type",
        "eventclass",
    ):
        value = _calendar_row_get_ci(row, key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    return " ".join(chunks).lower()


class _VisibleTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if str(tag).lower() in {"br", "p", "div", "li", "tr", "td", "th", "section", "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if str(tag).lower() in {"p", "div", "li", "tr", "td", "th", "section", "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        text = str(data or "")
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return "".join(self._chunks)


def _fetch_text_url(url: str, *, timeout_seconds: float = 15.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "schwab-trading-bot/1.0"})
    with urllib.request.urlopen(request, timeout=max(float(timeout_seconds or 15.0), 1.0)) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    return str(raw or "")


def _is_federal_reserve_target(args: argparse.Namespace, target: Dict[str, Any]) -> bool:
    text_blob = " ".join(
        [
            str(args.source or ""),
            str(args.speaker or ""),
            str(args.youtube_channel_url or ""),
            str(target.get("channel_url") or ""),
            str(target.get("stream_title") or ""),
        ]
    ).lower()
    return any(token in text_blob for token in ("federal reserve", "@federalreserve", "jerome powell", "fomc", "fed"))


def _fed_month_calendar_url(month_dt: datetime) -> str:
    return f"https://www.federalreserve.gov/newsevents/{month_dt.year}-{month_dt.strftime('%B').lower()}.htm"


def _extract_visible_lines_from_html(html_text: str) -> List[str]:
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", str(html_text or ""))
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    parser = _VisibleTextHTMLParser()
    parser.feed(cleaned)
    text = html.unescape(parser.text())
    lines: List[str] = []
    for raw in text.splitlines():
        normalized = re.sub(r"\s+", " ", str(raw or "")).strip()
        if normalized:
            lines.append(normalized)
    return lines


def _time_line_to_epoch(month_dt: datetime, day_text: str, time_text: str) -> Optional[float]:
    try:
        day = int(str(day_text or "").strip())
    except Exception:
        return None
    time_clean = str(time_text or "").strip().lower()
    if not time_clean or time_clean == "to be announced":
        return None
    try:
        parsed_time = datetime.strptime(time_clean.replace(".", ""), "%I:%M %p")
    except Exception:
        try:
            parsed_time = datetime.strptime(time_clean, "%I:%M %p")
        except Exception:
            return None
    try:
        event_dt = datetime(month_dt.year, month_dt.month, day, parsed_time.hour, parsed_time.minute, tzinfo=_FED_LOCAL_TZ)
    except Exception:
        return None
    return event_dt.astimezone(timezone.utc).timestamp()


def _parse_federal_reserve_calendar_page(html_text: str, *, page_url: str, month_dt: datetime) -> List[Dict[str, Any]]:
    lines = _extract_visible_lines_from_html(html_text)
    rows: List[Dict[str, Any]] = []
    current_section = ""
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line in _FED_SECTION_HEADERS:
            current_section = line
            idx += 1
            continue
        if current_section not in {"Speeches", "FOMC Meetings", "Board Meetings"}:
            idx += 1
            continue
        if line in {"Time:", "Release Date(s):"}:
            idx += 1
            continue
        if not _TIME_LINE_RE.match(line):
            idx += 1
            continue

        time_line = line
        idx += 1
        block: List[str] = []
        while idx < len(lines):
            probe = lines[idx]
            if probe in _FED_SECTION_HEADERS or _TIME_LINE_RE.match(probe):
                break
            block.append(probe)
            idx += 1

        if not block:
            continue
        day_pos = None
        for pos in range(len(block) - 1, -1, -1):
            if _DAY_LINE_RE.match(block[pos]):
                day_pos = pos
                break
        if day_pos is None:
            continue

        day_text = block[day_pos]
        detail_lines = [item for item in block[:day_pos] if str(item or "").strip().lower() != "watch live"]
        if not detail_lines:
            continue
        title = str(detail_lines[0] or "").strip()
        description = " ".join(str(item or "").strip() for item in detail_lines[1:] if str(item or "").strip()).strip()
        event_epoch = _time_line_to_epoch(month_dt, day_text, time_line)
        row = {
            "title": title,
            "headline": title,
            "description": description,
            "eventtype": current_section,
            "category": current_section,
            "impact": "high" if current_section == "FOMC Meetings" else "medium",
            "dateutc": datetime.fromtimestamp(event_epoch, tz=timezone.utc).isoformat() if event_epoch is not None else "",
            "time": time_line,
            "day": str(day_text),
            "url": page_url,
            "source": "federalreserve.gov",
        }
        rows.append(row)
    return rows


def _fetch_federal_reserve_calendar_correlation(args: argparse.Namespace, target: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_federal_reserve_target(args, target):
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "not_applicable",
        }

    try:
        from core.derivatives_features import summarize_calendar_payload
    except Exception as exc:
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": f"fed_import_error:{type(exc).__name__}",
        }

    now_dt = datetime.now(timezone.utc)
    month_candidates = [
        datetime(now_dt.year, now_dt.month, 1, tzinfo=timezone.utc),
        (datetime(now_dt.year + (1 if now_dt.month == 12 else 0), 1 if now_dt.month == 12 else now_dt.month + 1, 1, tzinfo=timezone.utc)),
    ]
    page_rows: List[Dict[str, Any]] = []
    source_urls: List[str] = []
    for month_dt in month_candidates:
        page_url = _fed_month_calendar_url(month_dt)
        try:
            html_text = _fetch_text_url(page_url, timeout_seconds=12.0)
        except Exception:
            continue
        source_urls.append(page_url)
        page_rows.extend(_parse_federal_reserve_calendar_page(html_text, page_url=page_url, month_dt=month_dt))

    if not page_rows:
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "fed_calendar_fetch_failed",
            "calendar_correlation_source": "federalreserve.gov",
        }

    now_ts = now_dt.timestamp()
    scored_rows: List[Dict[str, Any]] = []
    for row in page_rows:
        score_row = _score_calendar_match(row, args=args, target=target, now_ts=now_ts)
        scored_rows.append({"row": row, "score": score_row})
    scored_rows.sort(key=lambda item: float(item["score"].get("score", 0.0)), reverse=True)
    best = scored_rows[0]
    best_score = float(best["score"].get("score", 0.0))
    best_row = best["row"]
    best_score_row = best["score"]
    calendar_features = summarize_calendar_payload(page_rows, now_ts=now_ts)

    if best_score < 3.0:
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "fed_calendar_no_match",
            "calendar_correlation_source": "federalreserve.gov",
            "calendar_match_score": best_score,
            "calendar_features": calendar_features,
            "calendar_source_urls": source_urls,
        }

    return {
        "calendar_correlation_enabled": True,
        "calendar_correlation_ok": True,
        "calendar_correlation_reason": "matched_event",
        "calendar_correlation_source": "federalreserve.gov",
        "calendar_match_score": best_score,
        "calendar_matched_terms": list(best_score_row.get("matched_terms") or []),
        "calendar_event_title": str(best_row.get("title") or ""),
        "calendar_event_time_utc": str(best_score_row.get("event_time_utc") or ""),
        "calendar_event_minutes_delta": best_score_row.get("minutes_delta"),
        "calendar_event_impact": str(best_score_row.get("impact") or ""),
        "calendar_event_text": str(best_score_row.get("text") or ""),
        "calendar_features": calendar_features,
        "calendar_source_urls": source_urls,
    }


def _calendar_rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        rows.extend([row for row in payload if isinstance(row, dict)])
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in {"events", "items", "calendar", "data", "results", "scheduledevents", "economicevents", "earnings"} and isinstance(value, list):
                rows.extend([row for row in value if isinstance(row, dict)])
    seen_ids: set[int] = set()
    for node in _iter_dict_nodes(payload):
        key_set = {str(key).lower() for key in node.keys()}
        if not key_set.intersection({"eventdate", "date", "timestamp", "startdate", "datetime", "time", "dateutc"}):
            continue
        node_id = id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        rows.append(node)
    return rows[:500]


def _calendar_target_terms(args: argparse.Namespace, target: Dict[str, Any]) -> List[Dict[str, Any]]:
    speaker = str(args.speaker or "").strip().lower()
    stream_title = str(target.get("stream_title") or "").strip().lower()
    terms: List[Dict[str, Any]] = []
    for token, weight in (
        (speaker, 4.0),
        ("jerome powell", 4.0),
        ("powell", 3.0),
        ("federal reserve", 3.0),
        ("fed", 2.0),
        ("fomc", 4.0),
        ("press conference", 3.0),
        ("rate decision", 2.0),
        ("statement", 1.0),
    ):
        value = str(token or "").strip().lower()
        if value and all(existing["token"] != value for existing in terms):
            terms.append({"token": value, "weight": float(weight)})
    if stream_title:
        for fragment in (stream_title, stream_title.replace("live remarks", "").strip(), stream_title.replace("live", "").strip()):
            value = str(fragment or "").strip().lower()
            if len(value) >= 8 and all(existing["token"] != value for existing in terms):
                terms.append({"token": value, "weight": 3.5})
    return terms


def _score_calendar_match(row: Dict[str, Any], *, args: argparse.Namespace, target: Dict[str, Any], now_ts: float) -> Dict[str, Any]:
    text = _calendar_event_text(row)
    matched_terms: List[str] = []
    score = 0.0
    for term in _calendar_target_terms(args, target):
        token = str(term["token"])
        if token and token in text:
            matched_terms.append(token)
            score += float(term["weight"])

    event_ts = None
    for key in ("eventdate", "startdate", "datetime", "date", "timestamp", "time", "dateutc"):
        event_ts = _calendar_parse_epoch(_calendar_row_get_ci(row, key))
        if event_ts is not None:
            break

    minutes_delta = None
    event_time_utc = ""
    if event_ts is not None:
        minutes_delta = abs(event_ts - now_ts) / 60.0
        event_time_utc = datetime.fromtimestamp(event_ts, tz=timezone.utc).isoformat()
        score += max(0.0, 4.0 - min(minutes_delta / 60.0, 4.0))

    impact = str(_calendar_row_get_ci(row, "impact", "importance") or "").lower()
    if any(token in impact for token in ("high", "critical", "major", "red")):
        score += 1.5

    return {
        "score": round(score, 4),
        "matched_terms": matched_terms,
        "text": text,
        "event_time_utc": event_time_utc,
        "minutes_delta": round(float(minutes_delta), 3) if minutes_delta is not None else None,
        "impact": impact,
    }


def _try_schwab_calendar_method(client: Any, method_name: str, days_ahead: int) -> Optional[Any]:
    method = getattr(client, method_name, None)
    if not callable(method):
        return None

    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=max(days_ahead, 1))
    candidates = [
        ((), {"start_datetime": now_utc, "end_datetime": end_utc}),
        ((), {"symbol": "SPY", "start_datetime": now_utc, "end_datetime": end_utc}),
        ((), {"symbols": "SPY", "start_datetime": now_utc, "end_datetime": end_utc}),
        ((), {"symbol": "SPY"}),
        ((), {"symbols": "SPY"}),
        ((), {}),
    ]

    for method_args, method_kwargs in candidates:
        try:
            response = method(*method_args, **method_kwargs)
        except TypeError:
            continue
        except Exception:
            continue
        if response is None:
            continue
        if hasattr(response, "json"):
            try:
                return response.json()
            except Exception:
                continue
        return response
    return None


def _fetch_schwab_calendar_correlation(args: argparse.Namespace, target: Dict[str, Any]) -> Dict[str, Any]:
    now_ts = datetime.now(timezone.utc).timestamp()
    try:
        from core.base_trader import BaseTrader
        from core.derivatives_features import summarize_calendar_payload
    except Exception as exc:
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": f"import_error:{type(exc).__name__}",
        }

    api_key = os.getenv("SCHWAB_API_KEY", "").strip()
    secret = os.getenv("SCHWAB_SECRET", "").strip()
    redirect = os.getenv("SCHWAB_REDIRECT", "https://127.0.0.1:8080").strip()
    if not api_key or not secret or api_key == "YOUR_KEY_HERE" or secret == "YOUR_SECRET_HERE":
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "missing_credentials",
        }

    trader = BaseTrader(api_key, secret, redirect, mode="shadow")
    trader.token_path = str(PROJECT_ROOT / "token.json")
    try:
        client = trader.authenticate()
    except Exception as exc:
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": f"auth_error:{type(exc).__name__}",
        }

    payload = None
    method_used = ""
    candidate_methods = ("get_market_calendar", "get_calendar", "get_events", "get_market_events", "get_economic_calendar")
    available_methods = [method_name for method_name in candidate_methods if callable(getattr(client, method_name, None))]
    if not available_methods:
        fallback = _fetch_federal_reserve_calendar_correlation(args, target)
        if str(fallback.get("calendar_correlation_reason") or "") != "not_applicable":
            return fallback
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "client_has_no_calendar_methods",
            "calendar_correlation_source": "schwab.client",
        }
    for method_name in candidate_methods:
        payload = _try_schwab_calendar_method(client, method_name, 2)
        if payload is not None:
            method_used = method_name
            break

    if payload is None:
        fallback = _fetch_federal_reserve_calendar_correlation(args, target)
        if str(fallback.get("calendar_correlation_reason") or "") != "not_applicable":
            return fallback
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "calendar_fetch_failed",
        }

    rows = _calendar_rows_from_payload(payload)
    if not rows:
        fallback = _fetch_federal_reserve_calendar_correlation(args, target)
        if str(fallback.get("calendar_correlation_reason") or "") != "not_applicable":
            return fallback
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "calendar_rows_empty",
            "calendar_correlation_source": f"schwab.{method_used}",
        }

    scored_rows: List[Dict[str, Any]] = []
    for row in rows:
        score_row = _score_calendar_match(row, args=args, target=target, now_ts=now_ts)
        scored_rows.append({"row": row, "score": score_row})
    scored_rows.sort(key=lambda item: float(item["score"].get("score", 0.0)), reverse=True)
    best = scored_rows[0]
    best_score = float(best["score"].get("score", 0.0))
    best_row = best["row"]
    best_score_row = best["score"]
    calendar_features = summarize_calendar_payload(payload, now_ts=now_ts)

    if best_score < 3.0:
        fallback = _fetch_federal_reserve_calendar_correlation(args, target)
        if bool(fallback.get("calendar_correlation_ok")):
            return fallback
        return {
            "calendar_correlation_enabled": True,
            "calendar_correlation_ok": False,
            "calendar_correlation_reason": "no_match",
            "calendar_correlation_source": f"schwab.{method_used}",
            "calendar_match_score": best_score,
            "calendar_features": calendar_features,
        }

    event_title = str(_calendar_row_get_ci(best_row, "title", "name", "event", "headline") or "").strip()
    return {
        "calendar_correlation_enabled": True,
        "calendar_correlation_ok": True,
        "calendar_correlation_reason": "matched_event",
        "calendar_correlation_source": f"schwab.{method_used}",
        "calendar_match_score": best_score,
        "calendar_matched_terms": list(best_score_row.get("matched_terms") or []),
        "calendar_event_title": event_title,
        "calendar_event_time_utc": str(best_score_row.get("event_time_utc") or ""),
        "calendar_event_minutes_delta": best_score_row.get("minutes_delta"),
        "calendar_event_impact": str(best_score_row.get("impact") or ""),
        "calendar_event_text": str(best_score_row.get("text") or ""),
        "calendar_features": calendar_features,
    }


def _maybe_correlate_schwab_calendar(
    args: argparse.Namespace,
    state: Dict[str, Any],
    target: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "calendar_correlation_enabled": bool(args.correlate_with_schwab_calendar),
    }
    if not bool(args.correlate_with_schwab_calendar):
        result["calendar_correlation_reason"] = "disabled"
        return result

    youtube_url = _media_ingest_candidate_url(target)
    attempt_identity = _media_ingest_identity(target, {})
    attempted_identity = str(state.get("last_calendar_correlation_identity") or "")
    if attempted_identity and attempted_identity == attempt_identity:
        prior_result = str(state.get("last_calendar_correlation_result") or "")
        result.update(
            {
                "calendar_correlation_ok": bool(state.get("last_calendar_correlation_ok")),
                "calendar_correlation_reason": "already_matched_for_stream" if prior_result == "matched_event" else (prior_result or "already_attempted_for_stream"),
                "calendar_correlation_source": str(state.get("last_calendar_correlation_source") or ""),
                "calendar_match_score": state.get("last_calendar_match_score"),
                "calendar_event_title": str(state.get("last_calendar_event_title") or ""),
                "calendar_event_time_utc": str(state.get("last_calendar_event_time_utc") or ""),
                "calendar_event_minutes_delta": state.get("last_calendar_event_minutes_delta"),
                "calendar_matched_terms": list(state.get("last_calendar_matched_terms") or []),
            }
        )
        return result

    correlation = _fetch_schwab_calendar_correlation(args, target)
    state.update(
        {
            "last_calendar_correlation_video_url": youtube_url,
            "last_calendar_correlation_identity": attempt_identity,
            "last_calendar_correlation_ok": bool(correlation.get("calendar_correlation_ok")),
            "last_calendar_correlation_result": str(correlation.get("calendar_correlation_reason") or ""),
            "last_calendar_correlation_source": str(correlation.get("calendar_correlation_source") or ""),
            "last_calendar_match_score": correlation.get("calendar_match_score"),
            "last_calendar_event_title": str(correlation.get("calendar_event_title") or ""),
            "last_calendar_event_time_utc": str(correlation.get("calendar_event_time_utc") or ""),
            "last_calendar_event_minutes_delta": correlation.get("calendar_event_minutes_delta"),
            "last_calendar_matched_terms": list(correlation.get("calendar_matched_terms") or []),
        }
    )
    append_live_macro_event(
        event_type="calendar_correlation",
        payload={
            "timestamp_utc": _now_iso(),
            "youtube_url": youtube_url,
            "youtube_channel_url": str(target.get("channel_url") or ""),
            "speaker": args.speaker,
            "source": args.source,
            "calendar_correlation": correlation,
        },
        out_file=out_path,
        extra={
            "youtube_url": youtube_url,
            "youtube_channel_url": str(target.get("channel_url") or ""),
            "stream_title": str(target.get("stream_title") or ""),
            "calendar_correlation_reason": str(correlation.get("calendar_correlation_reason") or ""),
            "calendar_correlation_ok": bool(correlation.get("calendar_correlation_ok")),
        },
    )
    result.update(correlation)
    return result


def _run_once(args: argparse.Namespace, state: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    previous_stance = str(state.get("last_stance", "neutral") or "neutral")
    previous_sentiment = float(state.get("last_sentiment_hint", 0.0) or 0.0)
    target = _resolve_stream_target(args)
    stream_state = str(target.get("stream_state") or "")
    calendar_correlation: Dict[str, Any] = {}
    if args.correlate_with_schwab_calendar and stream_state in {"awaiting_live_stream", "upcoming_detected", "live"}:
        calendar_correlation = _maybe_correlate_schwab_calendar(args, state, target, out_path)
    trigger_status = _maybe_trigger_media_ingest(args, state, target, calendar_correlation, out_path)

    if stream_state != "live":
        timestamp_utc = _now_iso()
        status = {
            "timestamp_utc": timestamp_utc,
            "ok": True,
            "youtube_url": "",
            "youtube_channel_url": str(target.get("channel_url") or ""),
            "resolved_video_url": str(target.get("resolved_video_url") or ""),
            "template": args.template,
            "speaker": args.speaker,
            "stream_state": stream_state or "awaiting_live_stream",
            "correlate_with_schwab_calendar": bool(args.correlate_with_schwab_calendar),
            "trigger_media_ingest_on_live": bool(args.trigger_media_ingest_on_live),
            "trigger_media_ingest_before_minutes": max(float(args.trigger_media_ingest_before_minutes or 0.0), 0.0),
            "updated_bulletin": False,
            "out_file": str(out_path),
        }
        for key in (
            "live_probe_url",
            "streams_probe_url",
            "upcoming_title",
            "upcoming_video_url",
            "latest_title",
            "latest_video_url",
            "latest_live_status",
            "probe_warning",
        ):
            value = target.get(key)
            if value:
                status[key] = value
        status.update(trigger_status)
        if calendar_correlation:
            status.update(calendar_correlation)
        transition_key = "|".join(
            [
                status["stream_state"],
                str(status.get("upcoming_video_url") or ""),
                str(status.get("latest_video_url") or ""),
            ]
        )
        if transition_key != str(state.get("last_stream_transition_key", "") or ""):
            status["events_file"] = append_live_macro_event(
                event_type="channel_status",
                payload=status,
                out_file=out_path,
                extra={
                    "youtube_channel_url": str(target.get("channel_url") or ""),
                    "stream_state": status["stream_state"],
                },
        )
        state.update(
            {
                "last_run_utc": timestamp_utc,
                "last_stream_state": status["stream_state"],
                "last_stream_transition_key": transition_key,
                "last_channel_url": str(target.get("channel_url") or ""),
            }
        )
        return status

    with tempfile.TemporaryDirectory(prefix="macro_auto_watch_") as tmp_dir:
        caption_path = _fetch_caption_snapshot(str(target.get("resolved_video_url") or ""), Path(tmp_dir))
        cues = _parse_vtt(caption_path)

    analysis_mode = "live_snapshot"
    caption_text = _latest_caption_text(cues, lookback_seconds=args.lookback_seconds)
    window_texts: List[str] = []
    window_stance_counts: Dict[str, int] = {}
    if args.replay_full_video:
        analysis_mode = "full_video_replay"
        caption_text = _all_caption_text(cues)
        window_texts = _windowed_caption_texts(cues, window_seconds=args.replay_window_seconds)
    if not caption_text:
        raise RuntimeError("empty_caption_snapshot")

    text_hash = hashlib.sha256(caption_text.encode("utf-8")).hexdigest()
    classification = _classify_text(
        caption_text,
        previous_stance,
        previous_sentiment,
        allow_carry_forward=not bool(args.replay_full_video),
    )
    if window_texts:
        for text in window_texts:
            window_classification = _classify_text(text, "neutral", 0.0, allow_carry_forward=False)
            stance_key = str(window_classification.get("stance") or "neutral")
            window_stance_counts[stance_key] = int(window_stance_counts.get(stance_key, 0)) + 1
    stance = str(classification["stance"])
    sentiment_hint = float(classification["sentiment_hint"])
    summary = _summary_from_text(caption_text, max_words=24 if args.replay_full_video else 18)
    cue_archive = _archive_caption_cues(
        cues=cues,
        out_path=out_path,
        youtube_url=str(target.get("resolved_video_url") or ""),
        youtube_channel_url=str(target.get("channel_url") or ""),
        analysis_mode=analysis_mode,
        text_hash=text_hash,
        replay_full_video=bool(args.replay_full_video),
    )
    payload = build_live_macro_payload(
        template=args.template,
        headline=f"{args.speaker} {'full video transcript replay' if args.replay_full_video else 'live transcript update'}",
        summary=summary,
        content=caption_text,
        speaker=args.speaker,
        source=args.source,
        url=str(target.get("resolved_video_url") or args.youtube_url or args.youtube_channel_url or ""),
        symbols=args.symbols,
        expires_hours=args.expires_hours,
        stance=stance,
        impact="critical",
        sentiment_hint_override=sentiment_hint,
        shock_hint_override=1.0,
    )
    if calendar_correlation:
        for key, value in calendar_correlation.items():
            if key == "calendar_features" and isinstance(value, dict):
                payload[key] = dict(value)
                if payload.get("items"):
                    payload["items"][0][key] = dict(value)
            elif key.startswith("calendar_"):
                payload[key] = value
                if payload.get("items"):
                    payload["items"][0][key] = value

    updated = text_hash != str(state.get("last_text_hash", "") or "")
    if updated:
        _write_json(out_path, payload)

    events_file = append_live_macro_event(
        event_type="video_replay" if args.replay_full_video else "auto_inference",
        payload=payload,
        out_file=out_path,
        extra={
            "updated_bulletin": bool(updated),
            "youtube_url": str(target.get("resolved_video_url") or ""),
            "youtube_channel_url": str(target.get("channel_url") or ""),
            "analysis_mode": analysis_mode,
            "caption_excerpt": summary,
            "caption_text": caption_text,
            "caption_text_length": len(caption_text),
            "caption_cue_count": len(cues),
            "text_hash": text_hash,
            "stance_confidence": float(classification.get("confidence", 0.0) or 0.0),
            "classification_reason": str(classification.get("reason") or ""),
            "hawkish_score": float(classification.get("hawkish_score", 0.0) or 0.0),
            "dovish_score": float(classification.get("dovish_score", 0.0) or 0.0),
            "hawkish_keywords": [hit["token"] for hit in classification.get("hawkish_hits", [])],
            "dovish_keywords": [hit["token"] for hit in classification.get("dovish_hits", [])],
            "window_count": len(window_texts),
            "window_stance_counts": window_stance_counts,
            "cue_archive_file": cue_archive["cue_archive_file"],
            "cue_events_file": cue_archive["cue_events_file"],
        },
    )

    status = {
        "timestamp_utc": _now_iso(),
        "ok": True,
        "youtube_url": str(target.get("resolved_video_url") or ""),
        "youtube_channel_url": str(target.get("channel_url") or ""),
        "resolved_video_url": str(target.get("resolved_video_url") or ""),
        "template": args.template,
        "speaker": args.speaker,
        "analysis_mode": analysis_mode,
        "stream_state": "live",
        "correlate_with_schwab_calendar": bool(args.correlate_with_schwab_calendar),
        "trigger_media_ingest_on_live": bool(args.trigger_media_ingest_on_live),
        "trigger_media_ingest_before_minutes": max(float(args.trigger_media_ingest_before_minutes or 0.0), 0.0),
        "stance": stance,
        "sentiment_hint": sentiment_hint,
        "stance_confidence": float(classification.get("confidence", 0.0) or 0.0),
        "classification_reason": str(classification.get("reason") or ""),
        "hawkish_score": float(classification.get("hawkish_score", 0.0) or 0.0),
        "dovish_score": float(classification.get("dovish_score", 0.0) or 0.0),
        "hawkish_keywords": [hit["token"] for hit in classification.get("hawkish_hits", [])],
        "dovish_keywords": [hit["token"] for hit in classification.get("dovish_hits", [])],
        "shock_hint": 1.0,
        "updated_bulletin": bool(updated),
        "out_file": str(out_path),
        "caption_excerpt": summary,
        "caption_text_length": len(caption_text),
        "caption_cue_count": len(cues),
        "window_count": len(window_texts),
        "window_stance_counts": window_stance_counts,
        "cue_archive_file": cue_archive["cue_archive_file"],
        "cue_events_file": cue_archive["cue_events_file"],
        "text_hash": text_hash,
        "events_file": events_file,
    }
    status.update(trigger_status)
    status.update(calendar_correlation)
    state.update(
        {
            "last_run_utc": status["timestamp_utc"],
            "last_text_hash": text_hash,
            "last_stance": stance,
            "last_sentiment_hint": sentiment_hint,
            "last_excerpt": summary,
            "last_out_file": str(out_path),
            "last_stream_state": "live",
            "last_stream_transition_key": str(target.get("resolved_video_url") or ""),
            "last_channel_url": str(target.get("channel_url") or ""),
        }
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll YouTube auto-captions and keep the live macro bulletin updated.")
    parser.add_argument("--youtube-url")
    parser.add_argument("--youtube-channel-url")
    parser.add_argument("--template", choices=("powell", "fed", "generic"), default="powell")
    parser.add_argument("--speaker", default="Jerome Powell")
    parser.add_argument("--source", default="Federal Reserve")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--poll-seconds", type=float, default=45.0)
    parser.add_argument("--lookback-seconds", type=float, default=240.0)
    parser.add_argument("--replay-full-video", action="store_true")
    parser.add_argument("--replay-window-seconds", type=float, default=300.0)
    parser.add_argument("--expires-hours", type=float, default=4.0)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--status-file", default=str(STATUS_PATH))
    parser.add_argument("--state-file", default=str(STATE_PATH))
    parser.add_argument("--pid-file", default=str(PID_PATH))
    parser.add_argument("--correlate-with-schwab-calendar", action="store_true")
    parser.add_argument("--trigger-media-ingest-on-live", action="store_true")
    parser.add_argument("--trigger-media-ingest-before-minutes", type=float, default=0.0)
    parser.add_argument("--media-ingest-language", default="en")
    parser.add_argument("--media-ingest-audio-format", default="mp3")
    parser.add_argument("--media-ingest-asr-backend", choices=("auto", "mlx_whisper"), default="auto")
    parser.add_argument("--media-ingest-asr-model", default="")
    parser.add_argument("--media-ingest-cookies-from-browser", default=os.getenv("LIVE_MACRO_COOKIES_FROM_BROWSER", ""))
    parser.add_argument("--media-ingest-wait-buffer-seconds", type=float, default=1800.0)
    parser.add_argument("--media-ingest-force-redownload", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.youtube_url and not args.youtube_channel_url:
        raise SystemExit("youtube input required: pass --youtube-url or --youtube-channel-url")
    if args.youtube_url and args.youtube_channel_url:
        raise SystemExit("pass either --youtube-url or --youtube-channel-url, not both")
    if args.replay_full_video and not args.once:
        raise SystemExit("--replay-full-video requires --once or the macro-replay wrapper")

    out_path = Path(args.out_file).expanduser().resolve()
    status_path = Path(args.status_file).expanduser().resolve()
    state_path = Path(args.state_file).expanduser().resolve()
    pid_path = Path(args.pid_file).expanduser().resolve()

    if not YT_DLP_BIN or not Path(YT_DLP_BIN).exists():
        raise SystemExit("yt-dlp_not_found")

    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: pid_path.unlink(missing_ok=True))

    state = _read_json(state_path)

    def _record(status: Dict[str, Any]) -> None:
        _write_json(status_path, status)
        _write_json(state_path, state)
        if args.json or args.once:
            print(json.dumps(status, ensure_ascii=True, indent=2))

    while True:
        try:
            status = _run_once(args, state, out_path)
        except Exception as exc:
            status = {
                "timestamp_utc": _now_iso(),
                "ok": False,
                "youtube_url": args.youtube_url or "",
                "youtube_channel_url": args.youtube_channel_url or "",
                "template": args.template,
                "speaker": args.speaker,
                "error": f"{type(exc).__name__}:{exc}",
                "out_file": str(out_path),
            }
            status["events_file"] = append_live_macro_event(
                event_type="auto_error",
                payload=status,
                out_file=out_path,
                extra={
                    "youtube_url": args.youtube_url or "",
                    "youtube_channel_url": args.youtube_channel_url or "",
                },
            )
        _record(status)
        if args.once:
            return 0 if status.get("ok") else 1
        time.sleep(max(float(args.poll_seconds), 15.0))


if __name__ == "__main__":
    raise SystemExit(main())
