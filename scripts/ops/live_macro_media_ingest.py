#!/usr/bin/env python3
import argparse
import html
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.live_macro_auto_watch import YT_DLP_BIN, _classify_text
from scripts.ops.live_macro_bulletin import DEFAULT_OUT_PATH, LIVE_MACRO_TEMPLATES, append_live_macro_event, build_live_macro_payload
from core.transcript_cleanup import (
    clean_transcript_text as _shared_clean_transcript_text,
    collapse_repeated_phrases as _shared_collapse_repeated_phrases,
    split_speaker_label as _shared_split_speaker_label,
)


FFMPEG_BIN = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
DEFAULT_MEDIA_ROOT = DEFAULT_OUT_PATH.parent / "live_macro_media"
DEFAULT_CUE_ARCHIVE_PATH = DEFAULT_OUT_PATH.with_name("live_macro_cues_latest.json")
DEFAULT_STATUS_PATH = PROJECT_ROOT / "governance" / "health" / "live_macro_media_status.json"
DEFAULT_EVENT_DIR = PROJECT_ROOT / "governance" / "events"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "training" / "live_macro_audio_features"

_DEFAULT_GENERIC_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "USO",
    "XLE",
    "XLI",
    "GLD",
    "TLT",
    "UUP",
]

_COMPANY_EVENT_TEMPLATES = {"earnings_call", "ceo_interview", "analyst_day"}
_POLICY_EVENT_TEMPLATES = {"powell", "fed", "policy_testimony", "legal_policy"}
_VTT_INLINE_TAG_RE = re.compile(r"</?c(?:\.[^>]*)?>", re.IGNORECASE)
_VTT_TIMESTAMP_TAG_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}\.\d{3}>")
_HTMLISH_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_NOISE_RE = re.compile(r"\[(?:applause|music|laughter|captions?[^\]]*|captioning[^\]]*)\]", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_SPEAKER_LABEL_RE = re.compile(
    r"^(?P<label>(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}|[A-Z]{1,4}\.)"
    r"(?:\s+(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}|[A-Z]{1,4}\.)){0,4})"
    r":\s*(?P<body>.*)$"
)
_SPEAKER_PREFIXES = {
    "MR",
    "MRS",
    "MS",
    "MISS",
    "DR",
    "SEN",
    "SENATOR",
    "REP",
    "REPRESENTATIVE",
    "CHAIR",
    "CHAIRMAN",
    "CHAIRWOMAN",
    "JUSTICE",
    "JUDGE",
    "GOV",
    "GOVERNOR",
    "PRESIDENT",
    "SECRETARY",
    "HOST",
    "MODERATOR",
    "VOICE",
}

_ACTIONABLE_SIGNAL_RULES = {
    "generic": [
        {
            "signal_type": "military_escalation",
            "direction": "risk_off",
            "shock_hint": 1.0,
            "symbols": ["USO", "XLE", "GLD", "UUP", "TLT", "SPY", "QQQ"],
            "tokens": (
                "additional strikes",
                "new strikes",
                "retaliation",
                "retaliate",
                "expand the operation",
                "expand our operation",
                "troops",
                "military action",
                "airstrikes",
                "missile",
                "attack",
                "bombing",
            ),
        },
        {
            "signal_type": "deescalation_timeline",
            "direction": "risk_on",
            "shock_hint": 0.85,
            "symbols": ["SPY", "QQQ", "IWM", "TLT", "USO", "XLE"],
            "tokens": (
                "withdraw within weeks",
                "withdraw within days",
                "leave within weeks",
                "end the war",
                "ceasefire",
                "truce",
                "de-escalation",
                "deescalation",
                "pause in fighting",
                "negotiations",
                "talks underway",
                "diplomatic solution",
            ),
        },
        {
            "signal_type": "oil_shipping_supply",
            "direction": "risk_off",
            "shock_hint": 0.95,
            "symbols": ["USO", "XLE", "DAL", "AAL", "UAL", "GLD", "SPY"],
            "tokens": (
                "strait of hormuz",
                "shipping lanes",
                "shipping lane",
                "maritime traffic",
                "tankers",
                "oil supply",
                "energy supply",
                "pipeline",
                "navigation",
                "shipping routes",
                "freedom of navigation",
            ),
        },
        {
            "signal_type": "sanctions_trade_restrictions",
            "direction": "risk_off",
            "shock_hint": 0.78,
            "symbols": ["SPY", "QQQ", "XLI", "XLE", "UUP"],
            "tokens": (
                "sanctions",
                "secondary sanctions",
                "embargo",
                "export restrictions",
                "export controls",
                "tariffs",
                "duties",
                "trade restrictions",
            ),
        },
    ],
    "policy_testimony": [
        {
            "signal_type": "liquidity_support",
            "direction": "risk_on",
            "shock_hint": 0.78,
            "symbols": ["TLT", "IEF", "SPY", "QQQ", "XLF"],
            "tokens": (
                "slow the runoff",
                "reduce the pace of runoff",
                "reserve balances remain ample",
                "liquidity support",
                "support market functioning",
                "reduce auction sizes",
                "lower issuance",
                "stabilize funding markets",
            ),
        },
        {
            "signal_type": "issuance_pressure",
            "direction": "risk_off",
            "shock_hint": 0.82,
            "symbols": ["TLT", "IEF", "UUP", "SPY", "QQQ"],
            "tokens": (
                "increase auction sizes",
                "higher issuance",
                "term premium",
                "persistent inflation pressure",
                "funding pressure",
                "debt ceiling",
                "treasury supply",
            ),
        },
    ],
    "legal_policy": [
        {
            "signal_type": "agency_authority_limited",
            "direction": "risk_on",
            "shock_hint": 0.72,
            "symbols": ["XLF", "KRE", "XLE", "XLI", "SPY", "QQQ"],
            "tokens": (
                "agency authority",
                "administrative power",
                "major questions doctrine",
                "rulemaking authority",
                "curtail the agency",
                "limit the agency",
                "vacate the rule",
                "set aside the rule",
                "strike down the rule",
            ),
        },
        {
            "signal_type": "enforcement_authority_upheld",
            "direction": "risk_off",
            "shock_hint": 0.76,
            "symbols": ["XLF", "XLV", "XLE", "QQQ", "SPY"],
            "tokens": (
                "uphold the rule",
                "allow the agency",
                "enforcement authority",
                "uphold the enforcement",
                "agency may proceed",
                "agency can enforce",
                "cfpb",
                "sec",
                "epa",
                "ftc",
            ),
        },
        {
            "signal_type": "sector_specific_ruling",
            "direction": "risk_off",
            "shock_hint": 0.68,
            "symbols": ["XLV", "XLE", "XLI", "XLF", "SPY"],
            "tokens": (
                "drug pricing",
                "medicare",
                "health care",
                "pipeline",
                "energy permitting",
                "environmental review",
                "bank fees",
                "payment network",
                "antitrust",
                "competition case",
            ),
        },
    ],
    "earnings_call": [
        {
            "signal_type": "positive_guidance",
            "direction": "risk_on",
            "shock_hint": 0.84,
            "symbols": [],
            "tokens": (
                "raise guidance",
                "raised guidance",
                "raising guidance",
                "strong demand",
                "demand remains strong",
                "backlog growth",
                "bookings growth",
                "margin expansion",
                "pricing power",
                "dividend increase",
                "increase our buyback",
            ),
        },
        {
            "signal_type": "negative_guidance",
            "direction": "risk_off",
            "shock_hint": 0.92,
            "symbols": [],
            "tokens": (
                "cut guidance",
                "lower guidance",
                "reduced guidance",
                "demand softness",
                "demand weakness",
                "slowing demand",
                "margin pressure",
                "inventory build",
                "inventory correction",
                "weaker consumer",
                "order slowdown",
            ),
        },
    ],
    "ceo_interview": [
        {
            "signal_type": "management_confidence",
            "direction": "risk_on",
            "shock_hint": 0.74,
            "symbols": [],
            "tokens": (
                "business is strong",
                "strong pipeline",
                "accelerating demand",
                "healthy backlog",
                "better than expected",
                "seeing strength",
                "margin improvement",
                "strategic alternatives",
                "partnership interest",
            ),
        },
        {
            "signal_type": "management_warning",
            "direction": "risk_off",
            "shock_hint": 0.84,
            "symbols": [],
            "tokens": (
                "cautious outlook",
                "slower demand",
                "weak consumer",
                "margin pressure",
                "product delay",
                "regulatory scrutiny",
                "supply constraints",
                "restructuring charge",
                "softness continues",
            ),
        },
    ],
    "analyst_day": [
        {
            "signal_type": "target_raise",
            "direction": "risk_on",
            "shock_hint": 0.78,
            "symbols": [],
            "tokens": (
                "raise our target",
                "increase our target",
                "long-term target",
                "margin target",
                "free cash flow target",
                "capacity expansion",
                "ai demand",
                "datacenter demand",
                "revenue growth target",
            ),
        },
        {
            "signal_type": "target_cut",
            "direction": "risk_off",
            "shock_hint": 0.84,
            "symbols": [],
            "tokens": (
                "reduce our target",
                "lower our target",
                "delay target",
                "slower ramp",
                "market share pressure",
                "cost overruns",
                "margin headwind",
                "softer outlook",
            ),
        },
    ],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return str(path)


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return cleaned.strip("._-") or "macro_media"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    total = max(len(left_tokens | right_tokens), 1)
    return round(overlap / total, 4)


def _yt_dlp_command(extra_args: List[str], *, cookies_from_browser: str = "") -> List[str]:
    cmd = [YT_DLP_BIN]
    cookie_value = str(cookies_from_browser or "").strip()
    if cookie_value:
        cmd.extend(["--cookies-from-browser", cookie_value])
    cmd.extend(extra_args)
    return cmd


def _format_yt_error(message: str, *, cookies_from_browser: str = "") -> str:
    text = str(message or "").strip()
    if "HTTP Error 403" in text and not str(cookies_from_browser or "").strip():
        text = f"{text} (retry with --cookies-from-browser chrome or --cookies-from-browser safari)"
    return text[-1200:]


def _cookie_attempts(cookies_from_browser: str) -> List[str]:
    attempts: List[str] = []
    primary = str(cookies_from_browser or "").strip()
    for candidate in (primary, ""):
        normalized = str(candidate or "").strip()
        if normalized in attempts:
            continue
        attempts.append(normalized)
    return attempts


def _cookie_mode_label(cookies_from_browser: str) -> str:
    normalized = str(cookies_from_browser or "").strip()
    if not normalized:
        return "public"
    return f"cookies:{normalized}"


def _metadata_prefers_live_from_start(metadata: Dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return False
    live_status = str(metadata.get("live_status") or "").strip().lower()
    return bool(metadata.get("is_live")) or live_status in {"is_live", "live", "is_upcoming", "upcoming"}


def _captured_outputs(audio_dir: Path, video_id: str) -> List[Path]:
    out: List[Path] = []
    for path in sorted(audio_dir.glob(f"{video_id}.*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".part", ".ytdl", ".tmp"}:
            continue
        out.append(path)
    return out


def _clear_capture_outputs(audio_dir: Path, video_id: str) -> None:
    for path in sorted(audio_dir.glob(f"{video_id}.*")):
        if path.is_file():
            path.unlink(missing_ok=True)


def _extract_video_metadata(youtube_url: str, *, cookies_from_browser: str) -> tuple[Dict[str, Any], str]:
    errors: List[str] = []
    for active_cookies in _cookie_attempts(cookies_from_browser):
        cmd = _yt_dlp_command(
            ["--dump-single-json", "--no-playlist", "--no-warnings", str(youtube_url)],
            cookies_from_browser=active_cookies,
        )
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=180)
        if proc.returncode != 0:
            errors.append(
                f"{_cookie_mode_label(active_cookies)}:{_format_yt_error(proc.stderr or proc.stdout or 'yt_dlp_metadata_failed', cookies_from_browser=active_cookies)}"
            )
            continue
        payload = json.loads(proc.stdout or "{}")
        if isinstance(payload, dict) and payload:
            return payload, active_cookies
        errors.append(f"{_cookie_mode_label(active_cookies)}:yt_dlp_metadata_failed")
    raise RuntimeError("; ".join(errors)[-1200:] or "yt_dlp_metadata_failed")


def _capture_audio(
    youtube_url: str,
    audio_dir: Path,
    video_id: str,
    *,
    audio_format: str,
    force_redownload: bool,
    cookies_from_browser: str,
    prefer_live_from_start: bool,
) -> tuple[Path, Dict[str, Any]]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing = _captured_outputs(audio_dir, video_id)
    if existing and not force_redownload:
        return existing[0], {"strategy": "existing_output", "cookie_mode": "unchanged"}

    output_template = audio_dir / f"{video_id}.%(ext)s"
    if force_redownload:
        _clear_capture_outputs(audio_dir, video_id)

    strategies: List[tuple[str, str, List[str]]] = []
    if prefer_live_from_start:
        strategies.extend(
            [
                ("live_from_start_audio_only", "bestaudio/best", ["--live-from-start"]),
                ("live_from_start_progressive", "best[protocol=https][acodec!=none][vcodec!=none]/18", ["--live-from-start"]),
            ]
        )
    strategies.extend(
        [
            ("audio_only_extract", "bestaudio/best", []),
            ("progressive_extract", "best[protocol=https][acodec!=none][vcodec!=none]/18", []),
        ]
    )
    errors: List[str] = []
    for active_cookies in _cookie_attempts(cookies_from_browser):
        for strategy_name, format_selector, strategy_args in strategies:
            cmd = _yt_dlp_command(
                [
                    *strategy_args,
                    "-f",
                    format_selector,
                    "--extract-audio",
                    "--audio-format",
                    audio_format,
                    "--no-playlist",
                    "--no-progress",
                    "--no-warnings",
                    "--output",
                    str(output_template),
                    str(youtube_url),
                ],
                cookies_from_browser=active_cookies,
            )
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=7200)
            if proc.returncode == 0:
                captured = _captured_outputs(audio_dir, video_id)
                if captured:
                    return captured[0], {
                        "strategy": strategy_name,
                        "cookie_mode": _cookie_mode_label(active_cookies),
                        "cookies_from_browser": active_cookies,
                    }
                errors.append(f"{_cookie_mode_label(active_cookies)}:{strategy_name}:audio_capture_missing_output")
            else:
                errors.append(
                    f"{_cookie_mode_label(active_cookies)}:{strategy_name}:{_format_yt_error(proc.stderr or proc.stdout or 'audio_capture_failed', cookies_from_browser=active_cookies)}"
                )
            _clear_capture_outputs(audio_dir, video_id)
    raise RuntimeError("; ".join(errors)[-1600:] or "audio_capture_failed")


def _capture_audio_with_wait(
    youtube_url: str,
    audio_dir: Path,
    video_id: str,
    *,
    audio_format: str,
    force_redownload: bool,
    cookies_from_browser: str,
    prefer_live_from_start: bool,
    wait_for_live_seconds: float,
    retry_interval_seconds: float,
) -> tuple[Path, int, Dict[str, Any]]:
    attempts = 0
    deadline = time.time() + max(float(wait_for_live_seconds or 0.0), 0.0)
    while True:
        attempts += 1
        try:
            audio_path, capture_context = _capture_audio(
                youtube_url,
                audio_dir,
                video_id,
                audio_format=audio_format,
                force_redownload=force_redownload,
                cookies_from_browser=cookies_from_browser,
                prefer_live_from_start=prefer_live_from_start,
            )
            return audio_path, attempts, capture_context
        except Exception:
            if time.time() >= deadline or max(float(wait_for_live_seconds or 0.0), 0.0) <= 0.0:
                raise
            time.sleep(max(float(retry_interval_seconds or 15.0), 5.0))


def _prepend_bootstrap_cues(
    transcript_segments: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
    *,
    max_gap_seconds: float = 90.0,
    min_transcript_start_seconds: float = 8.0,
    max_cues: int = 8,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    segment_rows = [dict(row) for row in list(transcript_segments or []) if isinstance(row, dict)]
    cue_rows = [dict(row) for row in list(cues or []) if isinstance(row, dict)]
    if not segment_rows or not cue_rows:
        return segment_rows, {"prepended_count": 0, "bootstrap_gap_seconds": 0.0}

    first_segment = segment_rows[0]
    first_start = float(first_segment.get("start_seconds", 0.0) or 0.0)
    if first_start < max(float(min_transcript_start_seconds or 0.0), 0.0):
        return segment_rows, {"prepended_count": 0, "bootstrap_gap_seconds": 0.0}

    first_text = str(first_segment.get("text") or "").strip()
    prepended: List[Dict[str, Any]] = []
    lower_bound = max(first_start - max(float(max_gap_seconds or 0.0), 0.0), 0.0)
    for cue in cue_rows:
        cue_text = str(cue.get("text") or "").strip()
        cue_start = float(cue.get("start_seconds", 0.0) or 0.0)
        cue_end = float(cue.get("end_seconds", 0.0) or 0.0)
        if not cue_text:
            continue
        if cue_end >= max(first_start - 1.0, 0.0):
            continue
        if cue_start < lower_bound:
            continue
        if _overlap_ratio(cue_text, first_text) >= 0.8:
            continue
        prepended.append(
            {
                "segment_index": -100000 + int(cue.get("cue_index", len(prepended))),
                "start_seconds": cue_start,
                "end_seconds": cue_end,
                "text": cue_text,
                "source_type": "caption_bootstrap_segment",
            }
        )

    if not prepended:
        return segment_rows, {"prepended_count": 0, "bootstrap_gap_seconds": 0.0}

    prepended = prepended[-max(int(max_cues or 0), 1) :]
    gap_seconds = round(max(first_start - float(prepended[-1].get("end_seconds", 0.0) or 0.0), 0.0), 3)
    return prepended + segment_rows, {"prepended_count": len(prepended), "bootstrap_gap_seconds": gap_seconds}


def _normalize_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if not isinstance(raw_segments, list):
        return segments
    for idx, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue
        segment_text = _clean_transcript_text(raw.get("text"))
        if not segment_text:
            continue
        segment = {
            "segment_index": int(raw.get("id", idx) if raw.get("id") is not None else idx),
            "start_seconds": round(float(raw.get("start", 0.0) or 0.0), 3),
            "end_seconds": round(float(raw.get("end", 0.0) or 0.0), 3),
            "text": segment_text,
            "avg_logprob": raw.get("avg_logprob"),
            "no_speech_prob": raw.get("no_speech_prob"),
            "compression_ratio": raw.get("compression_ratio"),
            "words": [],
        }
        words = raw.get("words")
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                word_text = _clean_transcript_text(word.get("word"))
                if not word_text:
                    continue
                segment["words"].append(
                    {
                        "word": word_text,
                        "start_seconds": round(float(word.get("start", 0.0) or 0.0), 3),
                        "end_seconds": round(float(word.get("end", 0.0) or 0.0), 3),
                        "probability": word.get("probability"),
                    }
                )
        segments.append(segment)
    return segments


def _transcribe_with_mlx_whisper(audio_path: Path, *, asr_model: str, language: str) -> Dict[str, Any]:
    mlx_whisper = importlib.import_module("mlx_whisper")
    kwargs: Dict[str, Any] = {"word_timestamps": True}
    if asr_model:
        kwargs["path_or_hf_repo"] = asr_model
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(audio_path), **kwargs)
    if not isinstance(result, dict):
        raise RuntimeError("mlx_whisper_invalid_result")
    segments = _normalize_segments(result.get("segments"))
    transcript_text = str(result.get("text") or " ".join(segment["text"] for segment in segments)).strip()
    return {
        "ok": True,
        "backend": "mlx_whisper",
        "model": asr_model or "default",
        "language": str(result.get("language") or language or ""),
        "text": transcript_text,
        "segments": segments,
    }


def _transcribe_audio(audio_path: Path, *, asr_backend: str, asr_model: str, language: str) -> Dict[str, Any]:
    requested = str(asr_backend or "auto").lower()
    backends = ["mlx_whisper"] if requested in {"auto", "mlx_whisper"} else [requested]
    errors: List[str] = []

    for backend in backends:
        try:
            if backend == "mlx_whisper":
                return _transcribe_with_mlx_whisper(audio_path, asr_model=asr_model, language=language)
            errors.append(f"unsupported_backend:{backend}")
        except ModuleNotFoundError as exc:
            errors.append(f"{backend}:missing:{exc}")
        except Exception as exc:
            errors.append(f"{backend}:error:{type(exc).__name__}:{exc}")

    return {
        "ok": False,
        "backend": requested,
        "error": "; ".join(errors) if errors else "no_asr_backend_available",
        "text": "",
        "segments": [],
    }


def _load_caption_cues(cue_archive_path: Path, youtube_url: str) -> Dict[str, Any]:
    if not cue_archive_path.exists():
        return {"cue_archive_file": str(cue_archive_path), "cue_count": 0, "cues": [], "matched_video": False}
    payload = json.loads(cue_archive_path.read_text(encoding="utf-8"))
    cues = payload.get("cues") if isinstance(payload, dict) else None
    if not isinstance(cues, list):
        cues = []
    cue_rows: List[Dict[str, Any]] = []
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        cue_rows.append(
            {
                "cue_index": int(cue.get("cue_index", len(cue_rows))),
                "start_seconds": round(float(cue.get("start_seconds", 0.0) or 0.0), 3),
                "end_seconds": round(float(cue.get("end_seconds", 0.0) or 0.0), 3),
                "text": _clean_transcript_text(cue.get("text")),
            }
        )
    matched_video = str(payload.get("youtube_url") or "").strip() == str(youtube_url or "").strip()
    return {
        "cue_archive_file": str(cue_archive_path),
        "cue_count": len(cue_rows),
        "cues": cue_rows,
        "matched_video": matched_video,
    }


def _align_transcript_segments_to_cues(
    segments: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cue_midpoints = {
        int(cue["cue_index"]): (float(cue["start_seconds"]) + float(cue["end_seconds"])) / 2.0 for cue in cues
    }

    for segment in segments:
        seg_start = float(segment.get("start_seconds", 0.0) or 0.0)
        seg_end = float(segment.get("end_seconds", 0.0) or 0.0)
        seg_mid = (seg_start + seg_end) / 2.0
        overlapping = [
            cue
            for cue in cues
            if float(cue.get("end_seconds", 0.0) or 0.0) >= seg_start and float(cue.get("start_seconds", 0.0) or 0.0) <= seg_end
        ]
        nearest = None
        if cues:
            nearest = min(cues, key=lambda cue: abs(seg_mid - cue_midpoints[int(cue["cue_index"])]))
        cue_bundle = overlapping or ([nearest] if nearest is not None else [])
        aligned_text = " ".join(str(cue.get("text") or "").strip() for cue in cue_bundle if str(cue.get("text") or "").strip()).strip()
        distance_seconds = round(abs(seg_mid - cue_midpoints[int(nearest["cue_index"])]), 3) if nearest is not None else None
        rows.append(
            {
                "segment_index": int(segment.get("segment_index", len(rows))),
                "start_seconds": seg_start,
                "end_seconds": seg_end,
                "text": str(segment.get("text") or "").strip(),
                "cue_indices": [int(cue["cue_index"]) for cue in cue_bundle],
                "matched_cue_count": len(cue_bundle),
                "nearest_cue_index": int(nearest["cue_index"]) if nearest is not None else None,
                "nearest_cue_distance_seconds": distance_seconds,
                "aligned_cue_text": aligned_text,
                "text_overlap_ratio": _overlap_ratio(str(segment.get("text") or ""), aligned_text),
            }
        )
    return rows


def _build_training_feature_rows(
    *,
    youtube_url: str,
    video_id: str,
    speaker: str,
    source: str,
    transcript_segments: List[Dict[str, Any]],
    alignment_rows: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
    source_profile: Dict[str, Any],
    transcript_quality: Dict[str, Any],
    event_resolution_join: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, segment in enumerate(transcript_segments):
        alignment = alignment_rows[idx] if idx < len(alignment_rows) else {}
        classification = _classify_text(str(segment.get("text") or ""), "neutral", 0.0, allow_carry_forward=False)
        rows.append(
            {
                "timestamp_utc": _now_iso(),
                "category": "live_macro_training_feature",
                "source_type": str(segment.get("source_type") or "transcript_segment"),
                "youtube_url": youtube_url,
                "video_id": video_id,
                "speaker": speaker,
                "source": source,
                "segment_index": int(segment.get("segment_index", idx)),
                "segment_speaker": str(segment.get("speaker_label") or speaker),
                "speaker_turn_index": int(segment.get("speaker_turn_index", 0) or 0),
                "start_seconds": float(segment.get("start_seconds", 0.0) or 0.0),
                "end_seconds": float(segment.get("end_seconds", 0.0) or 0.0),
                "text": str(segment.get("text") or "").strip(),
                "aligned_cue_text": str(alignment.get("aligned_cue_text") or ""),
                "cue_indices": list(alignment.get("cue_indices") or []),
                "nearest_cue_distance_seconds": alignment.get("nearest_cue_distance_seconds"),
                "text_overlap_ratio": alignment.get("text_overlap_ratio"),
                "stance": str(classification.get("stance") or "neutral"),
                "sentiment_hint": float(classification.get("sentiment_hint", 0.0) or 0.0),
                "confidence": float(classification.get("confidence", 0.0) or 0.0),
                "hawkish_score": float(classification.get("hawkish_score", 0.0) or 0.0),
                "dovish_score": float(classification.get("dovish_score", 0.0) or 0.0),
                "hawkish_keywords": [hit["token"] for hit in classification.get("hawkish_hits", [])],
                "dovish_keywords": [hit["token"] for hit in classification.get("dovish_hits", [])],
                "source_priority_tier": str(source_profile.get("tier") or "secondary"),
                "source_priority_norm": float(source_profile.get("priority_norm", 0.0) or 0.0),
                "official_source_norm": float(source_profile.get("official_source_norm", 0.0) or 0.0),
                "transcript_quality_norm": float(transcript_quality.get("quality_norm", 0.0) or 0.0),
                "cue_match_quality_norm": float(transcript_quality.get("cue_match_ratio", 0.0) or 0.0),
                "duplicate_cluster_norm": float(transcript_quality.get("duplicate_cluster_norm", 0.0) or 0.0),
                "event_resolution_join_key": str(event_resolution_join.get("join_key") or ""),
                "event_resolution_ready_norm": float(event_resolution_join.get("ready_norm", 0.0) or 0.0),
            }
        )

    if rows:
        return rows

    for cue in cues:
        classification = _classify_text(str(cue.get("text") or ""), "neutral", 0.0, allow_carry_forward=False)
        rows.append(
            {
                "timestamp_utc": _now_iso(),
                "category": "live_macro_training_feature",
                "source_type": "caption_cue",
                "youtube_url": youtube_url,
                "video_id": video_id,
                "speaker": speaker,
                "source": source,
                "segment_index": int(cue.get("cue_index", len(rows))),
                "segment_speaker": str(cue.get("speaker_label") or speaker),
                "speaker_turn_index": int(cue.get("speaker_turn_index", 0) or 0),
                "start_seconds": float(cue.get("start_seconds", 0.0) or 0.0),
                "end_seconds": float(cue.get("end_seconds", 0.0) or 0.0),
                "text": str(cue.get("text") or "").strip(),
                "aligned_cue_text": str(cue.get("text") or "").strip(),
                "cue_indices": [int(cue.get("cue_index", len(rows)))],
                "nearest_cue_distance_seconds": 0.0,
                "text_overlap_ratio": 1.0,
                "stance": str(classification.get("stance") or "neutral"),
                "sentiment_hint": float(classification.get("sentiment_hint", 0.0) or 0.0),
                "confidence": float(classification.get("confidence", 0.0) or 0.0),
                "hawkish_score": float(classification.get("hawkish_score", 0.0) or 0.0),
                "dovish_score": float(classification.get("dovish_score", 0.0) or 0.0),
                "hawkish_keywords": [hit["token"] for hit in classification.get("hawkish_hits", [])],
                "dovish_keywords": [hit["token"] for hit in classification.get("dovish_hits", [])],
                "source_priority_tier": str(source_profile.get("tier") or "secondary"),
                "source_priority_norm": float(source_profile.get("priority_norm", 0.0) or 0.0),
                "official_source_norm": float(source_profile.get("official_source_norm", 0.0) or 0.0),
                "transcript_quality_norm": float(transcript_quality.get("quality_norm", 0.0) or 0.0),
                "cue_match_quality_norm": float(transcript_quality.get("cue_match_ratio", 0.0) or 0.0),
                "duplicate_cluster_norm": float(transcript_quality.get("duplicate_cluster_norm", 0.0) or 0.0),
                "event_resolution_join_key": str(event_resolution_join.get("join_key") or ""),
                "event_resolution_ready_norm": float(event_resolution_join.get("ready_norm", 0.0) or 0.0),
            }
        )
    return rows


def _artifact_paths(media_root: Path, video_id: str) -> Dict[str, Path]:
    run_root = media_root / _slug(video_id)
    return {
        "run_root": run_root,
        "audio_dir": run_root / "audio",
        "summary_file": run_root / "summary.json",
        "transcript_file": run_root / "transcript.json",
        "alignment_file": run_root / "alignment.json",
        "analysis_file": run_root / "market_analysis.json",
        "latest_file": media_root / "latest.json",
    }


def _parse_symbols(raw: Any) -> List[str]:
    out: List[str] = []
    if isinstance(raw, list):
        values = raw
    else:
        values = str(raw or "").replace("|", ",").split(",")
    for item in values:
        symbol = str(item or "").strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _excerpt(text: str, max_words: int = 24) -> str:
    words = [chunk for chunk in str(text or "").split() if chunk]
    return " ".join(words[: max(max_words, 8)]).strip()


def _norm_compare_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(token or "").lower())


def _collapse_repeated_phrases(text: str, *, max_window: int = 12) -> str:
    return _shared_collapse_repeated_phrases(text)


def _clean_transcript_text(text: Any) -> str:
    cleaned = _shared_clean_transcript_text(str(text or ""))
    cleaned = cleaned.replace(">>", " ")
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return _MULTISPACE_RE.sub(" ", cleaned).strip(" -")


def _split_speaker_label(text: Any) -> tuple[str, str]:
    return _shared_split_speaker_label(str(text or ""))


def _annotate_speaker_turns(rows: List[Dict[str, Any]], *, default_speaker: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    current_speaker = str(default_speaker or "").strip()
    current_turn: Optional[Dict[str, Any]] = None
    speaker_segment_counts: Dict[str, int] = {}

    for idx, raw in enumerate(list(rows or [])):
        if not isinstance(raw, dict):
            continue
        detected_speaker, body = _split_speaker_label(raw.get("text"))
        cleaned_text = body if detected_speaker else _clean_transcript_text(raw.get("text"))
        if not cleaned_text:
            continue
        speaker_name = detected_speaker or current_speaker or str(default_speaker or "").strip()
        if detected_speaker:
            current_speaker = detected_speaker

        start_seconds = round(float(raw.get("start_seconds", 0.0) or 0.0), 3)
        end_seconds = round(float(raw.get("end_seconds", 0.0) or 0.0), 3)
        segment_index = int(raw.get("segment_index", raw.get("cue_index", idx)))

        row = dict(raw)
        row["text"] = cleaned_text
        row["speaker_label"] = speaker_name
        row["speaker_detected"] = bool(detected_speaker)
        annotated.append(row)
        speaker_segment_counts[speaker_name] = int(speaker_segment_counts.get(speaker_name, 0)) + 1

        if (
            current_turn
            and str(current_turn.get("speaker_label") or "") == speaker_name
            and (start_seconds - float(current_turn.get("end_seconds", start_seconds) or start_seconds)) <= 18.0
        ):
            current_turn["end_seconds"] = end_seconds
            current_turn["segment_indices"].append(segment_index)
            current_turn["text"] = _collapse_repeated_phrases(
                " ".join(
                    chunk
                    for chunk in (str(current_turn.get("text") or "").strip(), cleaned_text)
                    if chunk
                ).strip()
            )
            row["speaker_turn_index"] = int(current_turn.get("speaker_turn_index", len(turns) - 1))
            continue

        current_turn = {
            "speaker_turn_index": len(turns),
            "speaker_label": speaker_name,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "segment_indices": [segment_index],
            "text": cleaned_text,
        }
        turns.append(current_turn)
        row["speaker_turn_index"] = int(current_turn["speaker_turn_index"])

    primary_speakers = [
        speaker
        for speaker, _count in sorted(
            speaker_segment_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]
    return annotated, turns, {
        "speaker_turn_count": len(turns),
        "speakers_detected": primary_speakers,
        "speaker_segment_counts": speaker_segment_counts,
    }


def _impact_label(shock_hint: float) -> str:
    value = float(shock_hint or 0.0)
    if value >= 0.9:
        return "critical"
    if value >= 0.72:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _source_provenance_profile(
    *,
    declared_source: str,
    youtube_url: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    source_text = str(declared_source or "").strip()
    channel_url = str(metadata.get("channel_url") or metadata.get("uploader_url") or "").strip()
    channel_name = str(metadata.get("channel") or metadata.get("uploader") or "").strip()
    title = str(metadata.get("title") or metadata.get("fulltitle") or "").strip()
    webpage_url = str(metadata.get("webpage_url") or youtube_url or "").strip()
    merged = " ".join([channel_url, channel_name, title, webpage_url, str(youtube_url or "")]).lower()
    source_key = re.sub(r"[^a-z0-9]+", "", source_text.lower())
    aliases = {
        "cspan": ("cspan", "c-span", "@cspan"),
        "federalreserve": ("federal reserve", "federalreserve", "@federalreserve", "fomc"),
        "schwabnetwork": ("schwab network", "@schwabnetwork"),
        "charlesschwab": ("charles schwab", "@charlesschwab"),
        "ustreasury": ("u.s. treasury", "us treasury", "@ustreasury"),
        "whitehouse": ("white house", "whitehouse", "@whitehouse"),
    }
    match_tokens = [token for token in aliases.get(source_key, (source_text.lower(),)) if token and token in merged]
    channel_match = bool(match_tokens)
    if not source_text:
        status = "missing_declared_source"
    elif not channel_url and not channel_name and not webpage_url:
        status = "missing_capture_metadata"
    elif channel_match:
        status = "matched"
    else:
        status = "source_channel_mismatch"
    return {
        "declared_source": source_text,
        "capture_channel_url": channel_url,
        "capture_channel_name": channel_name,
        "capture_video_url": webpage_url,
        "capture_title": title,
        "source_channel_match": channel_match,
        "source_provenance_status": status,
        "source_match_tokens": match_tokens,
    }


def _source_priority_profile(
    *,
    template: str,
    title: str,
    speaker: str,
    source: str,
    youtube_url: str,
) -> Dict[str, Any]:
    template_key = str(template or "generic").strip().lower() or "generic"
    source_text = str(source or "").strip().lower()
    title_text = str(title or "").strip().lower()
    speaker_text = str(speaker or "").strip().lower()
    url_text = str(youtube_url or "").strip().lower()
    merged = " ".join(chunk for chunk in (source_text, title_text, speaker_text, url_text) if chunk)
    broadcast_tokens = ("c-span", "cspan", "cnbc", "bloomberg", "yahoo finance", "fox business", "msnbc")

    official = False
    tier = "secondary"
    priority_norm = 0.62

    if any(token in merged for token in ("supreme court", "scotus", "supremecourt.gov")):
        official = "supremecourt.gov" in merged
        tier = "official" if official else "authoritative_broadcast"
        priority_norm = 1.0 if official else 0.92
    elif any(
        token in merged
        for token in (
            "charles schwab",
            "schwab coaching",
            "schwab network",
            "schwab.com",
            "schwabnetwork.com",
            "@charlesschwab",
            "@schwabnetwork",
        )
    ):
        official = any(
            token in merged
            for token in (
                "charles schwab",
                "schwab coaching",
                "schwab.com",
                "schwabnetwork.com",
                "@charlesschwab",
                "@schwabnetwork",
            )
        )
        tier = "official" if official else "authoritative_broadcast"
        priority_norm = 0.96 if official else 0.88
    elif any(token in merged for token in ("federal reserve", "powell", "fomc", "federalreserve.gov")):
        official = any(token in merged for token in ("federal reserve", "federalreserve.gov"))
        tier = "official" if official else "authoritative_broadcast"
        priority_norm = 0.98 if official else 0.90
    elif any(token in merged for token in ("treasury", "home.treasury.gov")):
        official = any(token in merged for token in ("home.treasury.gov", "u.s. treasury"))
        tier = "official" if official else "authoritative_broadcast"
        priority_norm = 0.97 if official else 0.88
    elif any(token in merged for token in ("white house", "whitehouse.gov")):
        official = any(token in merged for token in ("whitehouse.gov", "white house"))
        tier = "official" if official else "authoritative_broadcast"
        priority_norm = 0.97 if official else 0.88
    elif template_key in _COMPANY_EVENT_TEMPLATES and source_text and not any(token in merged for token in broadcast_tokens):
        official = True
        tier = "company_official"
        priority_norm = 0.96
    elif any(token in merged for token in broadcast_tokens):
        tier = "authoritative_broadcast"
        priority_norm = 0.90 if ("c-span" in merged or "cspan" in merged) else 0.84
    elif any(token in merged for token in ("reuters", "associated press", "ap", "wall street journal", "financial times")):
        tier = "trusted_media"
        priority_norm = 0.82

    return {
        "tier": tier,
        "priority_norm": round(_clamp01(priority_norm), 4),
        "official_source_norm": round(1.0 if official else (0.70 if tier == "authoritative_broadcast" else 0.0), 4),
        "official_source_detected": bool(official),
    }


def _official_source_candidates(*, template: str, source: str, title: str) -> List[Dict[str, str]]:
    template_key = str(template or "generic").strip().lower() or "generic"
    source_text = str(source or "").strip().lower()
    title_text = str(title or "").strip().lower()
    merged = f"{source_text} {title_text}".strip()
    candidates: List[Dict[str, str]] = []

    if template_key == "legal_policy" and any(token in merged for token in ("supreme court", "scotus")):
        term_year = datetime.now(timezone.utc).year
        candidates.append(
            {
                "label": "Supreme Court argument audio",
                "kind": "official_audio",
                "url": f"https://www.supremecourt.gov/oral_arguments/argument_audio/{term_year}",
            }
        )
    if template_key in {"powell", "fed", "policy_testimony"} or "federal reserve" in merged or "powell" in merged:
        candidates.append(
            {
                "label": "Federal Reserve news and events",
                "kind": "official_events",
                "url": "https://www.federalreserve.gov/newsevents.htm",
            }
        )
    if "schwab" in merged:
        candidates.extend(
            [
                {
                    "label": "Schwab Coaching live webcasts",
                    "kind": "official_events",
                    "url": "https://www.schwab.com/coaching/webcasts",
                },
                {
                    "label": "Schwab Coaching on-demand webcasts",
                    "kind": "official_archive",
                    "url": "https://www.schwab.com/coaching/ondemand-webcasts",
                },
                {
                    "label": "Schwab Network",
                    "kind": "official_media",
                    "url": "https://www.schwab.com/schwab-network",
                },
            ]
        )
    if "treasury" in merged:
        candidates.append(
            {
                "label": "Treasury news",
                "kind": "official_news",
                "url": "https://home.treasury.gov/news",
            }
        )
    if "white house" in merged:
        candidates.append(
            {
                "label": "White House briefing room",
                "kind": "official_news",
                "url": "https://www.whitehouse.gov/briefing-room/",
            }
        )
    if template_key in _COMPANY_EVENT_TEMPLATES and str(source or "").strip():
        candidates.append(
            {
                "label": "SEC company search",
                "kind": "official_filing_search",
                "url": f"https://www.sec.gov/edgar/search/#/q={quote_plus(str(source or '').strip())}",
            }
        )
    return candidates


def _transcript_quality_report(
    *,
    transcript_segments: List[Dict[str, Any]],
    alignment_rows: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    segment_rows = list(transcript_segments or [])
    cue_rows = list(cues or [])
    text_rows = segment_rows if segment_rows else cue_rows
    texts = [str(row.get("text") or "").strip().lower() for row in text_rows if str(row.get("text") or "").strip()]
    unique_ratio = (len(set(texts)) / max(len(texts), 1)) if texts else 0.0
    duplicate_cluster_norm = _clamp01(1.0 - unique_ratio) if texts else 1.0
    repeated_pairs = 0
    for prev, curr in zip(texts, texts[1:]):
        if prev and curr and prev == curr:
            repeated_pairs += 1
    stale_caption_ratio = (repeated_pairs / max(len(texts) - 1, 1)) if len(texts) >= 2 else 0.0

    overlap_values = [
        float(row.get("text_overlap_ratio", 0.0) or 0.0)
        for row in alignment_rows
        if isinstance(row, dict)
    ]
    avg_overlap = sum(overlap_values) / max(len(overlap_values), 1) if overlap_values else (1.0 if cue_rows and not segment_rows else 0.0)
    cue_match_ratio = (
        sum(1 for row in alignment_rows if list(row.get("cue_indices") or [])) / max(len(alignment_rows), 1)
        if alignment_rows
        else (1.0 if cue_rows and not segment_rows else 0.0)
    )
    timing_distances = [
        abs(float(row.get("nearest_cue_distance_seconds", 0.0) or 0.0))
        for row in alignment_rows
        if row.get("nearest_cue_distance_seconds") is not None
    ]
    timing_alignment_norm = 1.0 - _clamp01((sum(timing_distances) / max(len(timing_distances), 1)) / 12.0) if timing_distances else (1.0 if cue_rows else 0.0)
    word_count = sum(len(text.split()) for text in texts)
    text_length_norm = _clamp01(word_count / 220.0)
    novelty_norm = _clamp01((0.70 * unique_ratio) + (0.30 * (1.0 - stale_caption_ratio))) if texts else 0.0
    quality_norm = _clamp01(
        0.30 * avg_overlap
        + 0.22 * cue_match_ratio
        + 0.18 * timing_alignment_norm
        + 0.15 * text_length_norm
        + 0.15 * novelty_norm
    )
    return {
        "segment_count": len(segment_rows),
        "cue_count": len(cue_rows),
        "avg_overlap_norm": round(_clamp01(avg_overlap), 4),
        "cue_match_ratio": round(_clamp01(cue_match_ratio), 4),
        "timing_alignment_norm": round(_clamp01(timing_alignment_norm), 4),
        "text_length_norm": round(_clamp01(text_length_norm), 4),
        "duplicate_cluster_norm": round(_clamp01(max(duplicate_cluster_norm, stale_caption_ratio)), 4),
        "stale_caption_ratio": round(_clamp01(stale_caption_ratio), 4),
        "novelty_norm": round(_clamp01(novelty_norm), 4),
        "quality_norm": round(_clamp01(quality_norm), 4),
    }


def _event_resolution_join_spec(
    *,
    template: str,
    youtube_url: str,
    video_id: str,
    symbols: List[str],
    broad_market: bool,
) -> Dict[str, Any]:
    template_key = str(template or "generic").strip().lower() or "generic"
    benchmarks = list(_DEFAULT_GENERIC_SYMBOLS if broad_market else symbols[:4])
    return {
        "join_key": f"live_macro:{_slug(video_id)}",
        "template": template_key,
        "youtube_url": youtube_url,
        "symbols": list(symbols),
        "benchmarks": benchmarks,
        "windows_minutes": [5, 30, 60, 240, 1440],
        "metrics": [
            "return_bps",
            "spread_change_bps",
            "relative_volume_norm",
            "slippage_bps",
            "fill_quality_norm",
        ],
        "ready_norm": 1.0 if symbols else (0.75 if broad_market else 0.0),
    }


def _session_news_feature_flags() -> Dict[str, float]:
    local_now = datetime.now().astimezone()
    minutes = (local_now.hour * 60) + local_now.minute
    premarket = 1.0 if 240 <= minutes < 570 else 0.0
    intraday = 1.0 if 570 <= minutes <= 960 else 0.0
    after_hours = 1.0 if 960 < minutes <= 1200 else 0.0
    return {
        "news_premarket_norm": premarket,
        "news_intraday_norm": intraday,
        "news_after_hours_norm": after_hours,
    }


def _topic_feature_map(signal_types: List[str]) -> Dict[str, float]:
    signals = {str(item or "").strip().lower() for item in signal_types if str(item or "").strip()}
    return {
        "news_topic_earnings_norm": 1.0 if any("earnings" in sig or "margin" in sig or "demand" in sig for sig in signals) else 0.0,
        "news_topic_guidance_norm": 1.0 if any("guidance" in sig or "outlook" in sig for sig in signals) else 0.0,
        "news_topic_mna_norm": 1.0 if any("mna" in sig or "takeover" in sig or "merger" in sig or "acquisition" in sig for sig in signals) else 0.0,
        "news_topic_regulatory_norm": 1.0
        if any(
            token in sig
            for sig in signals
            for token in ("legal", "policy", "regulatory", "sanctions", "authority", "rule", "agency")
        )
        else 0.0,
    }


def _derived_external_context_payload(
    *,
    market_analysis: Dict[str, Any],
    source_profile: Dict[str, Any],
    transcript_quality: Dict[str, Any],
    event_resolution_join: Dict[str, Any],
) -> Dict[str, Any]:
    signal_types = list(market_analysis.get("signal_types") or [])
    topic_features = _topic_feature_map(signal_types)
    session_features = _session_news_feature_flags()
    source_quality_norm = _clamp01(
        max(
            float(source_profile.get("priority_norm", 0.0) or 0.0),
            float(source_profile.get("official_source_norm", 0.0) or 0.0),
        )
    )
    entity_relevance = 1.0 if (not bool(market_analysis.get("broad_market")) and market_analysis.get("symbols")) else (0.82 if bool(market_analysis.get("broad_market")) else 0.55)
    base_news_features = {
        "news_source_quality_norm": round(source_quality_norm, 4),
        "news_entity_relevance_norm": round(_clamp01(entity_relevance), 4),
        "news_novelty_norm": float(transcript_quality.get("novelty_norm", 0.0) or 0.0),
        "news_duplicate_cluster_norm": float(transcript_quality.get("duplicate_cluster_norm", 0.0) or 0.0),
        **topic_features,
        **session_features,
    }
    symbol_features = {
        str(symbol or "").strip().upper(): dict(base_news_features)
        for symbol in list(market_analysis.get("symbols") or [])
        if str(symbol or "").strip()
    }
    calendar_features = {
        "calendar_event_proximity_norm": round(_clamp01(float(market_analysis.get("shock_hint", 0.0) or 0.0)), 4),
        "calendar_high_impact_24h_norm": round(_clamp01(float(market_analysis.get("actionable_score", 0.0) or 0.0)), 4),
        "calendar_macro_event_norm": 1.0 if bool(market_analysis.get("broad_market")) else 0.35,
    }
    return {
        "news_features": base_news_features,
        "news_symbol_features": symbol_features,
        "calendar_features": calendar_features,
        "event_resolution_join": event_resolution_join,
        "source_profile": source_profile,
        "transcript_quality": transcript_quality,
    }


def _collect_actionable_signal_matches(
    *,
    text: str,
    rows: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def _token_hit_positions(lowered_text: str, token: str) -> List[str]:
        token_text = str(token or "").strip().lower()
        if not token_text:
            return []
        parts = [re.escape(part) for part in token_text.split() if part]
        if not parts:
            return []
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
        return [match.group(0) for match in re.finditer(pattern, lowered_text, flags=re.IGNORECASE)]

    out: List[Dict[str, Any]] = []
    if not str(text or "").strip():
        return out
    for idx, row in enumerate(rows or []):
        row_text = str(row.get("text") or "").strip()
        lowered = row_text.lower()
        if not lowered:
            continue
        for rule in rules:
            hits = [token for token in rule["tokens"] if _token_hit_positions(lowered, str(token))]
            if not hits:
                continue
            out.append(
                {
                    "signal_type": str(rule["signal_type"]),
                    "direction": str(rule["direction"]),
                    "shock_hint": float(rule["shock_hint"]),
                    "score": round(float(rule["shock_hint"]) * min(1.0, 0.6 + (0.15 * len(hits))), 4),
                    "token_hits": hits,
                    "excerpt": _excerpt(row_text, max_words=26),
                    "segment_index": int(row.get("segment_index", idx)),
                    "start_seconds": row.get("start_seconds"),
                    "end_seconds": row.get("end_seconds"),
                    "symbols": list(rule["symbols"]),
                }
            )
    return out


def _dedupe_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _norm_excerpt(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return " ".join(words[:16]).strip()

    seen: set[tuple[str, str, int]] = set()
    out: List[Dict[str, Any]] = []
    for signal in sorted(signals, key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("signal_type") or ""))):
        excerpt = _norm_excerpt(str(signal.get("excerpt") or ""))
        segment_bucket = int(float(signal.get("segment_index", 0) or 0) // 2)
        key = (str(signal.get("signal_type") or ""), excerpt, segment_bucket)
        if key in seen:
            continue
        seen.add(key)
        out.append(signal)
    return out


def _market_confirmation_state(
    *,
    template: str,
    signals: List[Dict[str, Any]],
    source_profile: Dict[str, Any],
    transcript_quality: Dict[str, Any],
    event_resolution_join: Dict[str, Any],
) -> Dict[str, Any]:
    template_key = str(template or "generic").strip().lower() or "generic"
    required = template_key == "legal_policy"
    distinct_segments = len(
        {
            int(signal.get("segment_index", 0) or 0)
            for signal in signals
            if isinstance(signal, dict)
        }
    )
    source_ok = float(source_profile.get("priority_norm", 0.0) or 0.0) >= 0.88
    transcript_ok = float(transcript_quality.get("quality_norm", 0.0) or 0.0) >= 0.45
    segment_ok = distinct_segments >= 2
    ready = float(event_resolution_join.get("ready_norm", 0.0) or 0.0) >= 0.75
    confirmed = (not required) or bool(source_ok and transcript_ok and segment_ok and ready)
    return {
        "required": bool(required),
        "ready": bool(ready),
        "confirmed": bool(confirmed),
        "distinct_segments": int(distinct_segments),
        "high_conviction_allowed": bool(confirmed),
        "reason": (
            "confirmed_by_source_quality_and_multi_segment_support"
            if confirmed
            else (
                "reaction_join_pending"
                if required and ready
                else "source_or_transcript_confirmation_too_weak"
            )
        ),
    }


def _analyze_market_usefulness(
    *,
    template: str,
    title: str,
    speaker: str,
    source: str,
    transcript_text: str,
    transcript_segments: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
    fallback_symbols: List[str],
    min_actionable_score: float,
) -> Dict[str, Any]:
    template_key = str(template or "generic").strip().lower() or "generic"
    rows = list(transcript_segments or []) or list(cues or [])
    if not rows and str(transcript_text or "").strip():
        rows = [
            {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "text": str(transcript_text or "").strip(),
            }
        ]
    joined_text = str(transcript_text or " ".join(str(row.get("text") or "") for row in rows)).strip()
    classification = _classify_text(joined_text, "neutral", 0.0, allow_carry_forward=False)

    signals: List[Dict[str, Any]] = []
    template_rules = list(_ACTIONABLE_SIGNAL_RULES.get(template_key, []))
    if template_key in _POLICY_EVENT_TEMPLATES:
        stance = str(classification.get("stance") or "neutral")
        confidence = float(classification.get("confidence", 0.0) or 0.0)
        if stance in {"hawkish", "dovish", "mixed"} and confidence >= 0.2:
            signals.append(
                {
                    "signal_type": f"policy_{stance}",
                    "direction": "risk_off" if stance == "hawkish" else ("risk_on" if stance == "dovish" else "mixed"),
                    "shock_hint": max(0.55, min(0.92, abs(float(classification.get("sentiment_hint", 0.0) or 0.0)) + 0.35)),
                    "score": round(max(0.55, confidence), 4),
                    "token_hits": [hit["token"] for hit in classification.get("hawkish_hits", []) + classification.get("dovish_hits", [])],
                    "excerpt": _excerpt(joined_text, max_words=30),
                    "segment_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "symbols": ["SPY", "QQQ", "TLT", "GLD", "UUP"],
                }
            )
    if template_rules:
        signals.extend(_collect_actionable_signal_matches(text=joined_text, rows=rows, rules=template_rules))

    signals = _dedupe_signals(signals)
    total_score = round(sum(float(signal.get("score", 0.0) or 0.0) for signal in signals), 4)
    risk_on = sum(float(signal.get("score", 0.0) or 0.0) for signal in signals if str(signal.get("direction") or "") == "risk_on")
    risk_off = sum(float(signal.get("score", 0.0) or 0.0) for signal in signals if str(signal.get("direction") or "") == "risk_off")
    directional_total = max(risk_on + risk_off, 1e-6)
    sentiment_hint = round(max(-0.95, min(0.95, (risk_on - risk_off) / directional_total)), 4) if signals else float(classification.get("sentiment_hint", 0.0) or 0.0)
    shock_hint = round(
        max(
            float(classification.get("confidence", 0.0) or 0.0) if template_key in _POLICY_EVENT_TEMPLATES else 0.0,
            max((float(signal.get("shock_hint", 0.0) or 0.0) for signal in signals), default=0.0),
        ),
        4,
    )
    requires_symbol_scope = template_key in _COMPANY_EVENT_TEMPLATES
    broad_market = template_key in _POLICY_EVENT_TEMPLATES or template_key == "generic"
    symbols = list(fallback_symbols or (_DEFAULT_GENERIC_SYMBOLS if broad_market else []))
    for signal in signals:
        for symbol in list(signal.get("symbols") or []):
            sym = str(symbol or "").strip().upper()
            if sym and sym not in symbols:
                symbols.append(sym)

    blocked_reason = ""
    actionable = bool(signals) and (total_score >= max(float(min_actionable_score or 0.0), 0.5) or shock_hint >= 0.72)
    if requires_symbol_scope and not symbols:
        actionable = False
        blocked_reason = "missing_symbols_for_company_event"

    signal_types = [str(signal.get("signal_type") or "") for signal in signals]
    top_excerpts = [str(signal.get("excerpt") or "").strip() for signal in signals[:3] if str(signal.get("excerpt") or "").strip()]
    detail_summary = "; ".join(top_excerpts)
    headline_core = title.strip() or f"{speaker} speech".strip() or "market speech"
    if signal_types:
        headline_suffix = ", ".join(signal_types[:2]).replace("_", " ")
        headline = f"{headline_core}: {headline_suffix}"
    else:
        headline = headline_core

    return {
        "template": str(template or "generic"),
        "title": title,
        "speaker": speaker,
        "source": source,
        "actionable": actionable,
        "actionable_score": total_score,
        "min_actionable_score": float(min_actionable_score or 0.0),
        "signal_count": len(signals),
        "signal_types": signal_types,
        "signals": signals,
        "headline": headline,
        "summary": detail_summary or _excerpt(joined_text, max_words=24),
        "content_excerpt": _excerpt(joined_text, max_words=48),
        "sentiment_hint": sentiment_hint,
        "shock_hint": shock_hint,
        "impact": _impact_label(shock_hint),
        "symbols": symbols,
        "broad_market": broad_market,
        "requires_symbol_scope": requires_symbol_scope,
        "blocked_reason": blocked_reason,
        "classification": classification,
    }


def _cleanup_non_actionable_outputs(
    artifact_paths: Dict[str, Path],
    *,
    audio_path: Optional[Path],
    training_features_file: str,
) -> None:
    if training_features_file:
        Path(training_features_file).unlink(missing_ok=True)
    if audio_path is not None:
        audio_path.unlink(missing_ok=True)
    shutil.rmtree(artifact_paths["run_root"], ignore_errors=True)


def run_ingest(args: argparse.Namespace) -> Dict[str, Any]:
    media_root = Path(args.media_root).expanduser().resolve()
    cue_archive_path = Path(args.cue_archive_file).expanduser().resolve()
    status_path = Path(args.status_file).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()
    cookies_from_browser = str(args.cookies_from_browser or "").strip()
    wait_for_live_seconds = max(float(args.wait_for_live_seconds or 0.0), 0.0)
    retry_interval_seconds = max(float(args.retry_interval_seconds or 15.0), 5.0)
    retain_policy = str(args.retain_policy or "all").strip().lower()
    fallback_symbols = _parse_symbols(args.symbols)

    metadata_attempts = 0
    metadata_cookie_mode = _cookie_mode_label(cookies_from_browser)
    metadata_cookie_value = cookies_from_browser
    metadata_deadline = time.time() + wait_for_live_seconds
    while True:
        metadata_attempts += 1
        try:
            metadata, metadata_cookie_value = _extract_video_metadata(args.youtube_url, cookies_from_browser=cookies_from_browser)
            metadata_cookie_mode = _cookie_mode_label(metadata_cookie_value)
            break
        except Exception:
            if time.time() >= metadata_deadline or wait_for_live_seconds <= 0.0:
                raise
            time.sleep(retry_interval_seconds)
    video_id = str(metadata.get("id") or hashlib.sha256(str(args.youtube_url).encode("utf-8")).hexdigest()[:12])
    artifact_paths = _artifact_paths(media_root, video_id)
    title = str(metadata.get("title") or video_id)
    prefer_live_from_start = _metadata_prefers_live_from_start(metadata) or wait_for_live_seconds > 0.0

    audio_path, audio_capture_attempts, audio_capture_context = _capture_audio_with_wait(
        args.youtube_url,
        artifact_paths["audio_dir"],
        video_id,
        audio_format=args.audio_format,
        force_redownload=bool(args.force_redownload),
        cookies_from_browser=metadata_cookie_value,
        prefer_live_from_start=prefer_live_from_start,
        wait_for_live_seconds=wait_for_live_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )
    transcript = _transcribe_audio(
        audio_path,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        language=args.language,
    )
    cue_payload = _load_caption_cues(cue_archive_path, args.youtube_url)
    cues, cue_turns, cue_speaker_summary = _annotate_speaker_turns(
        list(cue_payload.get("cues") or []),
        default_speaker=args.speaker,
    )
    transcript_segments, bootstrap_summary = _prepend_bootstrap_cues(list(transcript.get("segments") or []), cues)
    transcript_segments, speaker_turns, speaker_summary = _annotate_speaker_turns(
        transcript_segments,
        default_speaker=args.speaker,
    )
    transcript_text = " ".join(
        str(row.get("text") or "").strip()
        for row in transcript_segments
        if str(row.get("text") or "").strip()
    ).strip()
    if not transcript_text:
        transcript_text = " ".join(
            str(row.get("text") or "").strip()
            for row in cues
            if str(row.get("text") or "").strip()
        ).strip()
    if not transcript_text:
        transcript_text = _clean_transcript_text(transcript.get("text"))
    if not transcript_text and bootstrap_summary.get("prepended_count"):
        bootstrap_text = " ".join(
            str(row.get("text") or "").strip()
            for row in transcript_segments[: int(bootstrap_summary["prepended_count"])]
            if str(row.get("text") or "").strip()
        ).strip()
        transcript_text = " ".join(chunk for chunk in (bootstrap_text, transcript_text) if chunk).strip()
    alignment_rows = _align_transcript_segments_to_cues(transcript_segments, cues)
    market_analysis = _analyze_market_usefulness(
        template=args.template,
        title=title,
        speaker=args.speaker,
        source=args.source,
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
        cues=cues,
        fallback_symbols=fallback_symbols,
        min_actionable_score=float(args.min_actionable_score or 0.0),
    )
    source_profile = _source_priority_profile(
        template=args.template,
        title=title,
        speaker=args.speaker,
        source=args.source,
        youtube_url=args.youtube_url,
    )
    source_provenance = _source_provenance_profile(
        declared_source=args.source,
        youtube_url=args.youtube_url,
        metadata=metadata,
    )
    official_source_candidates = _official_source_candidates(
        template=args.template,
        source=args.source,
        title=title,
    )
    transcript_quality = _transcript_quality_report(
        transcript_segments=transcript_segments,
        alignment_rows=alignment_rows,
        cues=cues,
    )
    event_resolution_join = _event_resolution_join_spec(
        template=args.template,
        youtube_url=args.youtube_url,
        video_id=video_id,
        symbols=list(market_analysis.get("symbols") or fallback_symbols),
        broad_market=bool(market_analysis.get("broad_market")),
    )
    market_confirmation = _market_confirmation_state(
        template=args.template,
        signals=list(market_analysis.get("signals") or []),
        source_profile=source_profile,
        transcript_quality=transcript_quality,
        event_resolution_join=event_resolution_join,
    )
    market_analysis["market_confirmation"] = market_confirmation
    market_analysis["high_conviction"] = bool(market_analysis.get("actionable")) and bool(market_confirmation.get("high_conviction_allowed", True))
    if bool(market_analysis.get("actionable")) and bool(market_confirmation.get("required")) and not bool(market_confirmation.get("confirmed")):
        market_analysis["actionable_score"] = round(float(market_analysis.get("actionable_score", 0.0) or 0.0) * 0.85, 4)
    training_feature_rows = _build_training_feature_rows(
        youtube_url=args.youtube_url,
        video_id=video_id,
        speaker=args.speaker,
        source=args.source,
        transcript_segments=transcript_segments,
        alignment_rows=alignment_rows,
        cues=cues,
        source_profile=source_profile,
        transcript_quality=transcript_quality,
        event_resolution_join=event_resolution_join,
    )
    derived_context = _derived_external_context_payload(
        market_analysis=market_analysis,
        source_profile=source_profile,
        transcript_quality=transcript_quality,
        event_resolution_join=event_resolution_join,
    )

    transcript_payload = {
        "timestamp_utc": _now_iso(),
        "category": "live_macro_transcript",
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "title": title,
        "speaker": args.speaker,
        "source": args.source,
        "source_provenance": source_provenance,
        "audio_file": str(audio_path),
        "asr_backend": transcript.get("backend"),
        "asr_model": transcript.get("model"),
        "asr_ok": bool(transcript.get("ok")),
        "asr_error": transcript.get("error"),
        "language": transcript.get("language"),
        "text": transcript_text,
        "raw_text": str(transcript.get("text") or ""),
        "segment_count": len(transcript_segments),
        "segments": transcript_segments,
        "speaker_turn_count": int(speaker_summary.get("speaker_turn_count", len(speaker_turns)) or 0),
        "speakers_detected": list(speaker_summary.get("speakers_detected") or []),
        "speaker_turns": speaker_turns,
    }
    alignment_payload = {
        "timestamp_utc": _now_iso(),
        "category": "live_macro_alignment",
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "speaker": args.speaker,
        "source": args.source,
        "source_provenance": source_provenance,
        "cue_archive_file": cue_payload.get("cue_archive_file"),
        "cue_count": int(cue_payload.get("cue_count", 0) or 0),
        "cue_archive_matched_video": bool(cue_payload.get("matched_video")),
        "alignment_count": len(alignment_rows),
        "transcript_quality": transcript_quality,
        "rows": alignment_rows,
    }
    analysis_payload = {
        "timestamp_utc": _now_iso(),
        "category": "live_macro_market_analysis",
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "title": title,
        "speaker": args.speaker,
        "source": args.source,
        "source_provenance": source_provenance,
        "source_profile": source_profile,
        "official_source_candidates": official_source_candidates,
        "transcript_quality": transcript_quality,
        "event_resolution_join": event_resolution_join,
        **market_analysis,
    }

    features_file = DEFAULT_FEATURES_DIR / datetime.now(timezone.utc).strftime("%Y%m%d") / f"{_slug(video_id)}_training_features.jsonl"
    features_events_file = ""
    retained = True
    scrapped_non_actionable = False
    bulletin_events_file = ""
    bulletin_published = False
    bulletin_payload: Dict[str, Any] = {}

    if retain_policy == "actionable_only" and not bool(market_analysis.get("actionable")):
        retained = False
        scrapped_non_actionable = True
        _cleanup_non_actionable_outputs(artifact_paths, audio_path=audio_path, training_features_file=features_events_file)
    else:
        if features_file.exists():
            features_file.unlink()
        for row in training_feature_rows:
            features_events_file = _append_jsonl(features_file, row)

        _write_json(artifact_paths["transcript_file"], transcript_payload)
        _write_json(artifact_paths["alignment_file"], alignment_payload)
        _write_json(artifact_paths["analysis_file"], analysis_payload)

    if bool(args.publish_bulletin) and bool(market_analysis.get("actionable")):
        bulletin_payload = build_live_macro_payload(
            template=args.template,
            headline=str(market_analysis.get("headline") or title or f"{args.speaker} speech"),
            summary=str(market_analysis.get("summary") or ""),
            content=str(market_analysis.get("content_excerpt") or ""),
            speaker=args.speaker,
            source=args.source,
            url=args.youtube_url,
            symbols=list(market_analysis.get("symbols") or fallback_symbols),
            published=_now_iso(),
            expires_hours=float(args.expires_hours or 4.0),
            stance="neutral",
            impact=str(market_analysis.get("impact") or "high"),
            broad_market=bool(market_analysis.get("broad_market")),
            sentiment_hint_override=float(market_analysis.get("sentiment_hint", 0.0) or 0.0),
            shock_hint_override=float(market_analysis.get("shock_hint", 0.0) or 0.0),
            channel="live_macro_media_ingest",
        )
        bulletin_payload["actionable_score"] = float(market_analysis.get("actionable_score", 0.0) or 0.0)
        bulletin_payload["signal_types"] = list(market_analysis.get("signal_types") or [])
        bulletin_payload["derived"] = derived_context
        bulletin_payload["source_profile"] = source_profile
        bulletin_payload["source_provenance"] = source_provenance
        bulletin_payload["transcript_quality"] = transcript_quality
        bulletin_payload["official_source_candidates"] = official_source_candidates
        bulletin_payload["event_resolution_join"] = event_resolution_join
        if isinstance(bulletin_payload.get("items"), list) and bulletin_payload["items"]:
            bulletin_payload["items"][0]["signal_types"] = list(market_analysis.get("signal_types") or [])
            bulletin_payload["items"][0]["actionable_score"] = float(market_analysis.get("actionable_score", 0.0) or 0.0)
            bulletin_payload["items"][0]["source_provenance"] = source_provenance
        _write_json(out_path, bulletin_payload)
        bulletin_events_file = append_live_macro_event(
            event_type="publish_from_media_ingest",
            payload=bulletin_payload,
            out_file=out_path,
            extra={
                "youtube_url": args.youtube_url,
                "video_id": video_id,
                "speaker": args.speaker,
                "source": args.source,
            },
        )
        bulletin_published = True

    summary = {
        "timestamp_utc": _now_iso(),
        "ok": True,
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "title": title,
        "speaker": args.speaker,
        "source": args.source,
        "source_provenance": source_provenance,
        "audio_file": str(audio_path) if retained else "",
        "audio_bytes": int(audio_path.stat().st_size) if audio_path.exists() else 0,
        "audio_ext": audio_path.suffix.lower(),
        "ffmpeg_available": bool(FFMPEG_BIN and Path(FFMPEG_BIN).exists()),
        "cookies_from_browser": cookies_from_browser,
        "metadata_cookie_mode": metadata_cookie_mode,
        "audio_capture_cookie_mode": str(audio_capture_context.get("cookie_mode") or metadata_cookie_mode),
        "audio_capture_strategy": str(audio_capture_context.get("strategy") or ""),
        "wait_for_live_seconds": wait_for_live_seconds,
        "retry_interval_seconds": retry_interval_seconds,
        "metadata_attempts": metadata_attempts,
        "audio_capture_attempts": audio_capture_attempts,
        "prefer_live_from_start": bool(prefer_live_from_start),
        "bootstrap_cue_segments_prepended": int(bootstrap_summary.get("prepended_count", 0) or 0),
        "bootstrap_cue_gap_seconds": float(bootstrap_summary.get("bootstrap_gap_seconds", 0.0) or 0.0),
        "asr_backend": transcript.get("backend"),
        "asr_ok": bool(transcript.get("ok")),
        "asr_error": transcript.get("error"),
        "transcript_file": str(artifact_paths["transcript_file"]),
        "alignment_file": str(artifact_paths["alignment_file"]),
        "cue_archive_file": cue_payload.get("cue_archive_file"),
        "cue_archive_matched_video": bool(cue_payload.get("matched_video")),
        "cue_count": int(cue_payload.get("cue_count", 0) or 0),
        "alignment_count": len(alignment_rows),
        "speaker_turn_count": int(speaker_summary.get("speaker_turn_count", len(speaker_turns)) or 0),
        "speakers_detected": list(speaker_summary.get("speakers_detected") or []),
        "cue_speaker_turn_count": int(cue_speaker_summary.get("speaker_turn_count", len(cue_turns)) or 0),
        "training_feature_count": len(training_feature_rows),
        "training_features_file": features_events_file,
        "learning_ready": bool(training_feature_rows) and retained,
        "retain_policy": retain_policy,
        "retained": retained,
        "scrapped_non_actionable": scrapped_non_actionable,
        "transcript_file": str(artifact_paths["transcript_file"]) if retained else "",
        "alignment_file": str(artifact_paths["alignment_file"]) if retained else "",
        "analysis_file": str(artifact_paths["analysis_file"]) if retained else "",
        "market_actionable": bool(market_analysis.get("actionable")),
        "market_actionable_score": float(market_analysis.get("actionable_score", 0.0) or 0.0),
        "market_signal_count": int(market_analysis.get("signal_count", 0) or 0),
        "market_signal_types": list(market_analysis.get("signal_types") or []),
        "market_sentiment_hint": float(market_analysis.get("sentiment_hint", 0.0) or 0.0),
        "market_shock_hint": float(market_analysis.get("shock_hint", 0.0) or 0.0),
        "market_broad_market": bool(market_analysis.get("broad_market")),
        "market_blocked_reason": str(market_analysis.get("blocked_reason") or ""),
        "market_high_conviction": bool(market_analysis.get("high_conviction")),
        "market_confirmation": market_confirmation,
        "source_priority_tier": str(source_profile.get("tier") or "secondary"),
        "source_priority_norm": float(source_profile.get("priority_norm", 0.0) or 0.0),
        "official_source_norm": float(source_profile.get("official_source_norm", 0.0) or 0.0),
        "official_source_candidates": official_source_candidates,
        "source_channel_match": bool(source_provenance.get("source_channel_match")),
        "source_provenance_status": str(source_provenance.get("source_provenance_status") or ""),
        "transcript_quality_norm": float(transcript_quality.get("quality_norm", 0.0) or 0.0),
        "transcript_cue_match_norm": float(transcript_quality.get("cue_match_ratio", 0.0) or 0.0),
        "transcript_duplicate_cluster_norm": float(transcript_quality.get("duplicate_cluster_norm", 0.0) or 0.0),
        "event_resolution_join": event_resolution_join,
        "bulletin_published": bulletin_published,
        "bulletin_file": str(out_path) if bulletin_published else "",
        "bulletin_events_file": bulletin_events_file,
    }
    if retained:
        _write_json(artifact_paths["summary_file"], summary)
    _write_json(artifact_paths["latest_file"], summary)
    _write_json(status_path, summary)

    event_row = {
        "timestamp_utc": _now_iso(),
        "event_type": "live_macro_media_ingest",
        "category": "live_macro_media",
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "speaker": args.speaker,
        "source": args.source,
        "source_provenance": source_provenance,
        "audio_file": str(audio_path) if retained else "",
        "transcript_file": str(artifact_paths["transcript_file"]) if retained else "",
        "alignment_file": str(artifact_paths["alignment_file"]) if retained else "",
        "training_features_file": features_events_file,
        "cue_archive_file": cue_payload.get("cue_archive_file"),
        "cue_count": int(cue_payload.get("cue_count", 0) or 0),
        "cookies_from_browser": cookies_from_browser,
        "metadata_cookie_mode": metadata_cookie_mode,
        "audio_capture_cookie_mode": str(audio_capture_context.get("cookie_mode") or metadata_cookie_mode),
        "audio_capture_strategy": str(audio_capture_context.get("strategy") or ""),
        "wait_for_live_seconds": wait_for_live_seconds,
        "retry_interval_seconds": retry_interval_seconds,
        "metadata_attempts": metadata_attempts,
        "audio_capture_attempts": audio_capture_attempts,
        "prefer_live_from_start": bool(prefer_live_from_start),
        "bootstrap_cue_segments_prepended": int(bootstrap_summary.get("prepended_count", 0) or 0),
        "bootstrap_cue_gap_seconds": float(bootstrap_summary.get("bootstrap_gap_seconds", 0.0) or 0.0),
        "asr_backend": transcript.get("backend"),
        "asr_ok": bool(transcript.get("ok")),
        "asr_error": transcript.get("error"),
        "learning_ready": bool(training_feature_rows) and retained,
        "retain_policy": retain_policy,
        "retained": retained,
        "scrapped_non_actionable": scrapped_non_actionable,
        "market_actionable": bool(market_analysis.get("actionable")),
        "market_actionable_score": float(market_analysis.get("actionable_score", 0.0) or 0.0),
        "market_signal_types": list(market_analysis.get("signal_types") or []),
        "market_broad_market": bool(market_analysis.get("broad_market")),
        "market_blocked_reason": str(market_analysis.get("blocked_reason") or ""),
        "market_high_conviction": bool(market_analysis.get("high_conviction")),
        "market_confirmation": market_confirmation,
        "source_priority_tier": str(source_profile.get("tier") or "secondary"),
        "source_priority_norm": float(source_profile.get("priority_norm", 0.0) or 0.0),
        "official_source_norm": float(source_profile.get("official_source_norm", 0.0) or 0.0),
        "source_channel_match": bool(source_provenance.get("source_channel_match")),
        "source_provenance_status": str(source_provenance.get("source_provenance_status") or ""),
        "transcript_quality_norm": float(transcript_quality.get("quality_norm", 0.0) or 0.0),
        "speaker_turn_count": int(speaker_summary.get("speaker_turn_count", len(speaker_turns)) or 0),
        "speakers_detected": list(speaker_summary.get("speakers_detected") or []),
        "event_resolution_join": event_resolution_join,
        "bulletin_published": bulletin_published,
        "bulletin_file": str(out_path) if bulletin_published else "",
    }
    summary["events_file"] = _append_jsonl(
        DEFAULT_EVENT_DIR / f"live_macro_media_events_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl",
        event_row,
    )
    _write_json(status_path, summary)
    _write_json(artifact_paths["latest_file"], summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture YouTube macro-event audio, transcribe when available, and align against archived captions.")
    parser.add_argument("--youtube-url", required=True)
    parser.add_argument("--template", choices=LIVE_MACRO_TEMPLATES, default="powell")
    parser.add_argument("--speaker", default="Jerome Powell")
    parser.add_argument("--source", default="Federal Reserve")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--language", default="en")
    parser.add_argument("--audio-format", default="mp3")
    parser.add_argument("--asr-backend", choices=("auto", "mlx_whisper"), default="auto")
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--media-root", default=str(DEFAULT_MEDIA_ROOT))
    parser.add_argument("--cue-archive-file", default=str(DEFAULT_CUE_ARCHIVE_PATH))
    parser.add_argument("--status-file", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--expires-hours", type=float, default=6.0)
    parser.add_argument("--cookies-from-browser", default=os.getenv("LIVE_MACRO_COOKIES_FROM_BROWSER", ""))
    parser.add_argument("--wait-for-live-seconds", type=float, default=0.0)
    parser.add_argument("--retry-interval-seconds", type=float, default=15.0)
    parser.add_argument("--retain-policy", choices=("all", "actionable_only"), default="all")
    parser.add_argument("--publish-bulletin", action="store_true")
    parser.add_argument("--min-actionable-score", type=float, default=0.75)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        status = run_ingest(args)
    except Exception as exc:
        status = {
            "timestamp_utc": _now_iso(),
            "ok": False,
            "youtube_url": args.youtube_url,
            "speaker": args.speaker,
            "source": args.source,
            "error": f"{type(exc).__name__}:{exc}",
        }
        _write_json(Path(args.status_file).expanduser().resolve(), status)
        _append_jsonl(
            DEFAULT_EVENT_DIR / f"live_macro_media_events_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl",
            {
                "timestamp_utc": _now_iso(),
                "event_type": "live_macro_media_ingest_error",
                "category": "live_macro_media",
                "youtube_url": args.youtube_url,
                "speaker": args.speaker,
                "source": args.source,
                "error": status["error"],
            },
        )
        if args.json:
            print(json.dumps(status, ensure_ascii=True, indent=2))
        return 1

    if args.json:
        print(json.dumps(status, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
