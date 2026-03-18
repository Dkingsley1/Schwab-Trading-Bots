#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.live_macro_auto_watch import YT_DLP_BIN, _classify_text
from scripts.ops.live_macro_bulletin import DEFAULT_OUT_PATH


FFMPEG_BIN = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
DEFAULT_MEDIA_ROOT = DEFAULT_OUT_PATH.parent / "live_macro_media"
DEFAULT_CUE_ARCHIVE_PATH = DEFAULT_OUT_PATH.with_name("live_macro_cues_latest.json")
DEFAULT_STATUS_PATH = PROJECT_ROOT / "governance" / "health" / "live_macro_media_status.json"
DEFAULT_EVENT_DIR = PROJECT_ROOT / "governance" / "events"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "training" / "live_macro_audio_features"


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


def _extract_video_metadata(youtube_url: str, *, cookies_from_browser: str) -> Dict[str, Any]:
    cmd = _yt_dlp_command(
        ["--dump-single-json", "--no-playlist", "--no-warnings", str(youtube_url)],
        cookies_from_browser=cookies_from_browser,
    )
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(_format_yt_error(proc.stderr or proc.stdout or "yt_dlp_metadata_failed", cookies_from_browser=cookies_from_browser))
    payload = json.loads(proc.stdout or "{}")
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("yt_dlp_metadata_failed")
    return payload


def _capture_audio(
    youtube_url: str,
    audio_dir: Path,
    video_id: str,
    *,
    audio_format: str,
    force_redownload: bool,
    cookies_from_browser: str,
) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(audio_dir.glob(f"{video_id}.*"))
    if existing and not force_redownload:
        return existing[0]

    output_template = audio_dir / f"{video_id}.%(ext)s"
    cmd = _yt_dlp_command(
        [
            "-f",
            "bestaudio/best",
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
        cookies_from_browser=cookies_from_browser,
    )
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(_format_yt_error(proc.stderr or proc.stdout or "audio_capture_failed", cookies_from_browser=cookies_from_browser))
    captured = sorted(audio_dir.glob(f"{video_id}.*"))
    if not captured:
        raise RuntimeError("audio_capture_missing_output")
    return captured[0]


def _capture_audio_with_wait(
    youtube_url: str,
    audio_dir: Path,
    video_id: str,
    *,
    audio_format: str,
    force_redownload: bool,
    cookies_from_browser: str,
    wait_for_live_seconds: float,
    retry_interval_seconds: float,
) -> tuple[Path, int]:
    attempts = 0
    deadline = time.time() + max(float(wait_for_live_seconds or 0.0), 0.0)
    while True:
        attempts += 1
        try:
            audio_path = _capture_audio(
                youtube_url,
                audio_dir,
                video_id,
                audio_format=audio_format,
                force_redownload=force_redownload,
                cookies_from_browser=cookies_from_browser,
            )
            return audio_path, attempts
        except Exception:
            if time.time() >= deadline or max(float(wait_for_live_seconds or 0.0), 0.0) <= 0.0:
                raise
            time.sleep(max(float(retry_interval_seconds or 15.0), 5.0))


def _normalize_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if not isinstance(raw_segments, list):
        return segments
    for idx, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue
        segment_text = str(raw.get("text") or "").strip()
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
                segment["words"].append(
                    {
                        "word": str(word.get("word") or "").strip(),
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
                "text": str(cue.get("text") or "").strip(),
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
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, segment in enumerate(transcript_segments):
        alignment = alignment_rows[idx] if idx < len(alignment_rows) else {}
        classification = _classify_text(str(segment.get("text") or ""), "neutral", 0.0, allow_carry_forward=False)
        rows.append(
            {
                "timestamp_utc": _now_iso(),
                "category": "live_macro_training_feature",
                "source_type": "transcript_segment",
                "youtube_url": youtube_url,
                "video_id": video_id,
                "speaker": speaker,
                "source": source,
                "segment_index": int(segment.get("segment_index", idx)),
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
        "latest_file": media_root / "latest.json",
    }


def run_ingest(args: argparse.Namespace) -> Dict[str, Any]:
    media_root = Path(args.media_root).expanduser().resolve()
    cue_archive_path = Path(args.cue_archive_file).expanduser().resolve()
    status_path = Path(args.status_file).expanduser().resolve()
    cookies_from_browser = str(args.cookies_from_browser or "").strip()
    wait_for_live_seconds = max(float(args.wait_for_live_seconds or 0.0), 0.0)
    retry_interval_seconds = max(float(args.retry_interval_seconds or 15.0), 5.0)

    metadata_attempts = 0
    metadata_deadline = time.time() + wait_for_live_seconds
    while True:
        metadata_attempts += 1
        try:
            metadata = _extract_video_metadata(args.youtube_url, cookies_from_browser=cookies_from_browser)
            break
        except Exception:
            if time.time() >= metadata_deadline or wait_for_live_seconds <= 0.0:
                raise
            time.sleep(retry_interval_seconds)
    video_id = str(metadata.get("id") or hashlib.sha256(str(args.youtube_url).encode("utf-8")).hexdigest()[:12])
    artifact_paths = _artifact_paths(media_root, video_id)
    title = str(metadata.get("title") or video_id)

    audio_path, audio_capture_attempts = _capture_audio_with_wait(
        args.youtube_url,
        artifact_paths["audio_dir"],
        video_id,
        audio_format=args.audio_format,
        force_redownload=bool(args.force_redownload),
        cookies_from_browser=cookies_from_browser,
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
    cues = list(cue_payload.get("cues") or [])
    alignment_rows = _align_transcript_segments_to_cues(list(transcript.get("segments") or []), cues)
    training_feature_rows = _build_training_feature_rows(
        youtube_url=args.youtube_url,
        video_id=video_id,
        speaker=args.speaker,
        source=args.source,
        transcript_segments=list(transcript.get("segments") or []),
        alignment_rows=alignment_rows,
        cues=cues,
    )

    transcript_payload = {
        "timestamp_utc": _now_iso(),
        "category": "live_macro_transcript",
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "title": title,
        "speaker": args.speaker,
        "source": args.source,
        "audio_file": str(audio_path),
        "asr_backend": transcript.get("backend"),
        "asr_model": transcript.get("model"),
        "asr_ok": bool(transcript.get("ok")),
        "asr_error": transcript.get("error"),
        "language": transcript.get("language"),
        "text": str(transcript.get("text") or ""),
        "segment_count": len(list(transcript.get("segments") or [])),
        "segments": list(transcript.get("segments") or []),
    }
    alignment_payload = {
        "timestamp_utc": _now_iso(),
        "category": "live_macro_alignment",
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "speaker": args.speaker,
        "source": args.source,
        "cue_archive_file": cue_payload.get("cue_archive_file"),
        "cue_count": int(cue_payload.get("cue_count", 0) or 0),
        "cue_archive_matched_video": bool(cue_payload.get("matched_video")),
        "alignment_count": len(alignment_rows),
        "rows": alignment_rows,
    }
    features_file = DEFAULT_FEATURES_DIR / datetime.now(timezone.utc).strftime("%Y%m%d") / f"{_slug(video_id)}_training_features.jsonl"
    if features_file.exists():
        features_file.unlink()
    features_events_file = ""
    for row in training_feature_rows:
        features_events_file = _append_jsonl(features_file, row)

    _write_json(artifact_paths["transcript_file"], transcript_payload)
    _write_json(artifact_paths["alignment_file"], alignment_payload)

    summary = {
        "timestamp_utc": _now_iso(),
        "ok": True,
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "title": title,
        "speaker": args.speaker,
        "source": args.source,
        "audio_file": str(audio_path),
        "audio_bytes": int(audio_path.stat().st_size) if audio_path.exists() else 0,
        "audio_ext": audio_path.suffix.lower(),
        "ffmpeg_available": bool(FFMPEG_BIN and Path(FFMPEG_BIN).exists()),
        "cookies_from_browser": cookies_from_browser,
        "wait_for_live_seconds": wait_for_live_seconds,
        "retry_interval_seconds": retry_interval_seconds,
        "metadata_attempts": metadata_attempts,
        "audio_capture_attempts": audio_capture_attempts,
        "asr_backend": transcript.get("backend"),
        "asr_ok": bool(transcript.get("ok")),
        "asr_error": transcript.get("error"),
        "transcript_file": str(artifact_paths["transcript_file"]),
        "alignment_file": str(artifact_paths["alignment_file"]),
        "cue_archive_file": cue_payload.get("cue_archive_file"),
        "cue_archive_matched_video": bool(cue_payload.get("matched_video")),
        "cue_count": int(cue_payload.get("cue_count", 0) or 0),
        "alignment_count": len(alignment_rows),
        "training_feature_count": len(training_feature_rows),
        "training_features_file": features_events_file,
        "learning_ready": bool(training_feature_rows),
    }
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
        "audio_file": str(audio_path),
        "transcript_file": str(artifact_paths["transcript_file"]),
        "alignment_file": str(artifact_paths["alignment_file"]),
        "training_features_file": features_events_file,
        "cue_archive_file": cue_payload.get("cue_archive_file"),
        "cue_count": int(cue_payload.get("cue_count", 0) or 0),
        "cookies_from_browser": cookies_from_browser,
        "wait_for_live_seconds": wait_for_live_seconds,
        "retry_interval_seconds": retry_interval_seconds,
        "metadata_attempts": metadata_attempts,
        "audio_capture_attempts": audio_capture_attempts,
        "asr_backend": transcript.get("backend"),
        "asr_ok": bool(transcript.get("ok")),
        "asr_error": transcript.get("error"),
        "learning_ready": bool(training_feature_rows),
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
    parser.add_argument("--template", choices=("powell", "fed", "generic"), default="powell")
    parser.add_argument("--speaker", default="Jerome Powell")
    parser.add_argument("--source", default="Federal Reserve")
    parser.add_argument("--language", default="en")
    parser.add_argument("--audio-format", default="mp3")
    parser.add_argument("--asr-backend", choices=("auto", "mlx_whisper"), default="auto")
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--media-root", default=str(DEFAULT_MEDIA_ROOT))
    parser.add_argument("--cue-archive-file", default=str(DEFAULT_CUE_ARCHIVE_PATH))
    parser.add_argument("--status-file", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--cookies-from-browser", default=os.getenv("LIVE_MACRO_COOKIES_FROM_BROWSER", ""))
    parser.add_argument("--wait-for-live-seconds", type=float, default=0.0)
    parser.add_argument("--retry-interval-seconds", type=float, default=15.0)
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
