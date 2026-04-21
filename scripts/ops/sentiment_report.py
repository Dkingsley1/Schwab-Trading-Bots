#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_GLOB = "live_macro_events_*.jsonl"
DEFAULT_LIVE_MACRO_PATH = PROJECT_ROOT / "data" / "external_context" / "live_macro_latest.json"
DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "sentiment_report_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_latest.md"
DEFAULT_HTML_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_latest.html"
DEFAULT_PDF_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_latest.pdf"
DEFAULT_DAILY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_daily_latest.png"
DEFAULT_WEEKLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_weekly_latest.png"
DEFAULT_MONTHLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_monthly_latest.png"
DEFAULT_YEARLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "sentiment_report_yearly_latest.png"
DEFAULT_MEDIA_SUMMARY_GLOB = "*/summary.json"
APP_BROWSER_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)
STANCE_SENTIMENTS = {
    "hawkish": -0.75,
    "bearish": -0.75,
    "dovish": 0.75,
    "bullish": 0.75,
    "neutral": 0.0,
    "mixed": -0.2,
}
EVENT_DATE_RE = re.compile(r"live_macro_events_(\d{8})\.jsonl$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _default_allow_gui_pdf_renderer() -> bool:
    return any(candidate.exists() for candidate in APP_BROWSER_CANDIDATES)


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = (
        os.getenv("SENTIMENT_REPORT_PDF_BIN", "").strip()
        or os.getenv("REPORT_PDF_BUNDLE_PDF_BIN", "").strip()
        or os.getenv("TRAINING_REPORT_PDF_BIN", "").strip()
    )
    if env_override:
        env_bin = Path(env_override).expanduser()
        if env_bin.exists():
            kind = "wkhtmltopdf" if env_bin.name == "wkhtmltopdf" else "browser"
            return str(env_bin), kind

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if wkhtmltopdf:
        return wkhtmltopdf, "wkhtmltopdf"

    for candidate in (
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("microsoft-edge"),
        shutil.which("msedge"),
    ):
        if candidate:
            return candidate, "browser"

    if allow_gui_renderer:
        for candidate in APP_BROWSER_CANDIDATES:
            if candidate.exists():
                return str(candidate), "browser"

    return "", ""


def _render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
    renderer, renderer_kind = _pdf_renderer_binary(allow_gui_renderer=allow_gui_renderer)
    if not renderer:
        return False, "pdf_renderer_not_found"
    html_uri = html_path.resolve().as_uri()
    if renderer_kind == "wkhtmltopdf":
        cmd = [renderer, html_uri, str(pdf_path)]
        rc, out, err = _run(cmd)
    else:
        profile_dir = Path(tempfile.mkdtemp(prefix="sentiment-report-pdf-"))
        try:
            cmd = [
                renderer,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--silent-launch",
                "--no-startup-window",
                "--disable-background-networking",
                "--metrics-recording-only",
                f"--user-data-dir={profile_dir}",
                f"--print-to-pdf={pdf_path}",
                html_uri,
            ]
            rc, out, err = _run(cmd)
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or err or "ok"
    return False, err or out or f"rc={rc}"


def _safe_float(raw: Any, default: float | None = None) -> float | None:
    try:
        return float(raw)
    except Exception:
        return default


def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(float(value), high))


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _day_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d")


def _week_key_from_dt(dt: datetime) -> str:
    current = dt.astimezone(timezone.utc).date()
    return (current - timedelta(days=current.weekday())).strftime("%Y%m%d")


def _month_key_from_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m")


def _year_key_from_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y")


def _display_day(day_key: str) -> str:
    text = str(day_key or "").strip()
    if len(text) != 8:
        return text
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _sentiment_label(value: Any) -> str:
    score = float(value or 0.0)
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def _excerpt(text: Any, *, max_words: int = 18) -> str:
    words = [chunk for chunk in str(text or "").split() if chunk]
    return " ".join(words[: max(max_words, 8)]).strip()


def _rank_counter(counter: Counter[str], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {"name": key, "count": int(count)}
        for key, count in counter.most_common(max(int(limit), 1))
    ]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _event_file_day(path: Path) -> str:
    match = EVENT_DATE_RE.search(path.name)
    return str(match.group(1)) if match else ""


def _cutoff_date(*, selected_day: str, lookback_days: int) -> date:
    return datetime.strptime(str(selected_day), "%Y%m%d").date() - timedelta(days=max(int(lookback_days), 1) - 1)


def _event_files(project_root: Path, *, selected_day: str, lookback_days: int) -> list[Path]:
    cutoff_day = _cutoff_date(selected_day=selected_day, lookback_days=lookback_days).strftime("%Y%m%d")
    rows: list[Path] = []
    for path in sorted((project_root / "governance" / "events").glob(DEFAULT_EVENTS_GLOB)):
        if not path.is_file():
            continue
        file_day = _event_file_day(path)
        if file_day and file_day < cutoff_day:
            continue
        rows.append(path)
    return rows


def _media_summary_files(project_root: Path, *, selected_day: str, lookback_days: int) -> list[Path]:
    selected_date = datetime.strptime(str(selected_day), "%Y%m%d").date()
    cutoff_date = _cutoff_date(selected_day=selected_day, lookback_days=lookback_days)
    media_root = project_root / "data" / "external_context" / "live_macro_media"
    rows: list[Path] = []
    for path in sorted(media_root.glob(DEFAULT_MEDIA_SUMMARY_GLOB)):
        if not path.is_file():
            continue
        payload = _load_json(path)
        ts = _parse_ts(payload.get("timestamp_utc"))
        if ts is None:
            continue
        payload_date = ts.date()
        if payload_date < cutoff_date or payload_date > selected_date:
            continue
        rows.append(path)
    return rows


def _iter_jsonl_dicts(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        yield row
        except Exception:
            continue


def _payload_item(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items")
    if not isinstance(items, list):
        return {}
    for item in items:
        if isinstance(item, dict):
            return item
    return {}


def _first_present_float(*values: Any) -> float | None:
    for value in values:
        parsed = _safe_float(value, None)
        if parsed is not None:
            return float(parsed)
    return None


def _extract_sentiment_point(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    item = _payload_item(payload)
    ts = _parse_ts(row.get("timestamp_utc")) or _parse_ts(payload.get("timestamp_utc")) or _parse_ts(item.get("timestamp_utc"))
    if ts is None:
        return None

    stance = str(
        payload.get("stance")
        or row.get("stance")
        or item.get("stance")
        or ""
    ).strip().lower()
    sentiment = _first_present_float(
        row.get("market_sentiment_hint"),
        row.get("sentiment_hint"),
        payload.get("market_sentiment_hint"),
        payload.get("sentiment_hint"),
        item.get("market_sentiment_hint"),
        item.get("sentiment_hint"),
    )
    if sentiment is None and stance:
        sentiment = STANCE_SENTIMENTS.get(stance)
    if sentiment is None:
        return None

    shock = _first_present_float(
        row.get("market_shock_hint"),
        row.get("shock_hint"),
        payload.get("market_shock_hint"),
        payload.get("shock_hint"),
        item.get("market_shock_hint"),
        item.get("shock_hint"),
    )
    confidence = _first_present_float(
        row.get("stance_confidence"),
        row.get("confidence"),
        payload.get("stance_confidence"),
        payload.get("confidence"),
        item.get("stance_confidence"),
        item.get("confidence"),
    )
    shock_value = _clamp(float(shock if shock is not None else 0.55), low=0.0, high=1.0)
    confidence_value = _clamp(float(confidence if confidence is not None else 0.65), low=0.0, high=1.0)
    sentiment_value = _clamp(float(sentiment), low=-1.0, high=1.0)
    headline = (
        str(payload.get("headline") or "").strip()
        or str(row.get("stream_title") or "").strip()
        or _excerpt(payload.get("summary") or payload.get("content") or row.get("caption_excerpt") or item.get("text") or "")
    )
    source = str(payload.get("source") or row.get("source") or item.get("source") or "").strip()
    speaker = str(payload.get("speaker") or row.get("speaker") or item.get("speaker") or "").strip()
    event_type = str(row.get("event_type") or payload.get("event_type") or "live_macro").strip()
    channel = str(payload.get("channel") or row.get("channel") or "").strip()
    point_key = str(
        row.get("event_resolution_join_key")
        or payload.get("event_resolution_join_key")
        or item.get("event_resolution_join_key")
        or row.get("video_id")
        or payload.get("video_id")
        or item.get("video_id")
        or row.get("youtube_url")
        or payload.get("youtube_url")
        or item.get("youtube_url")
        or f"{ts.isoformat()}::{source}::{headline}"
    ).strip()
    if not stance:
        stance = _sentiment_label(sentiment_value)

    return {
        "timestamp_utc": ts.isoformat(),
        "day_utc": _day_key(ts),
        "week_key": _week_key_from_dt(ts),
        "month_key": _month_key_from_dt(ts),
        "year_key": _year_key_from_dt(ts),
        "point_key": point_key,
        "source_kind": "event_log",
        "event_type": event_type,
        "source": source,
        "speaker": speaker,
        "channel": channel,
        "headline": headline,
        "stance": stance,
        "sentiment_hint": round(float(sentiment_value), 6),
        "shock_hint": round(float(shock_value), 6),
        "confidence": round(float(confidence_value), 6),
        "weight": round(max(0.15, float(shock_value)) * max(0.25, float(confidence_value)), 6),
    }


def _extract_media_summary_point(summary: dict[str, Any], *, source_path: Path) -> dict[str, Any] | None:
    ts = _parse_ts(summary.get("timestamp_utc"))
    sentiment = _first_present_float(
        summary.get("market_sentiment_hint"),
        summary.get("sentiment_hint"),
    )
    if ts is None or sentiment is None:
        return None

    shock = _first_present_float(
        summary.get("market_shock_hint"),
        summary.get("shock_hint"),
        summary.get("market_actionable_score"),
    )
    source_quality = _first_present_float(
        summary.get("source_priority_norm"),
        summary.get("official_source_norm"),
        0.7,
    )
    transcript_quality = _first_present_float(summary.get("transcript_quality_norm"), 0.55)
    confirmation = summary.get("market_confirmation") if isinstance(summary.get("market_confirmation"), dict) else {}
    confirmed = bool(confirmation.get("confirmed")) or bool(summary.get("market_high_conviction"))
    confidence_value = _clamp(
        float(source_quality if source_quality is not None else 0.7)
        * (0.72 + (0.28 * float(transcript_quality if transcript_quality is not None else 0.55)))
        * (1.0 if confirmed else 0.88),
        low=0.25,
        high=1.0,
    )
    shock_value = _clamp(float(shock if shock is not None else 0.55), low=0.0, high=1.0)
    sentiment_value = _clamp(float(sentiment), low=-1.0, high=1.0)
    headline = (
        str(summary.get("title") or "").strip()
        or str(summary.get("headline") or "").strip()
        or _excerpt(summary.get("summary") or summary.get("speaker") or source_path.parent.name)
    )
    source = str(summary.get("source") or "").strip()
    speaker = str(summary.get("speaker") or "").strip()
    point_key = str(
        (
            summary.get("event_resolution_join") or {}
            if isinstance(summary.get("event_resolution_join"), dict)
            else {}
        ).get("join_key")
        or summary.get("video_id")
        or summary.get("youtube_url")
        or source_path.parent.name
    ).strip()
    stance = str(summary.get("stance") or "").strip().lower() or _sentiment_label(sentiment_value)

    return {
        "timestamp_utc": ts.isoformat(),
        "day_utc": _day_key(ts),
        "week_key": _week_key_from_dt(ts),
        "month_key": _month_key_from_dt(ts),
        "year_key": _year_key_from_dt(ts),
        "point_key": point_key,
        "source_kind": "media_summary",
        "event_type": "live_macro_media_summary",
        "source": source,
        "speaker": speaker,
        "channel": "live_macro_media",
        "headline": headline,
        "stance": stance,
        "sentiment_hint": round(float(sentiment_value), 6),
        "shock_hint": round(float(shock_value), 6),
        "confidence": round(float(confidence_value), 6),
        "weight": round(max(0.2, float(shock_value)) * max(0.3, float(confidence_value)), 6),
    }


def _extract_live_snapshot_point(snapshot: dict[str, Any], *, selected_day: str) -> dict[str, Any] | None:
    selected_date = datetime.strptime(str(selected_day), "%Y%m%d").date()
    ts = _parse_ts(snapshot.get("timestamp_utc"))
    sentiment = _first_present_float(
        snapshot.get("market_sentiment_hint"),
        snapshot.get("sentiment_hint"),
    )
    if ts is None or sentiment is None or ts.date() > selected_date:
        return None

    shock = _first_present_float(snapshot.get("market_shock_hint"), snapshot.get("shock_hint"))
    confidence = _first_present_float(
        snapshot.get("source_priority_norm"),
        snapshot.get("official_source_norm"),
        0.65,
    )
    sentiment_value = _clamp(float(sentiment), low=-1.0, high=1.0)
    shock_value = _clamp(float(shock if shock is not None else 0.55), low=0.0, high=1.0)
    confidence_value = _clamp(float(confidence if confidence is not None else 0.65), low=0.25, high=1.0)
    headline = (
        str(snapshot.get("headline") or "").strip()
        or _excerpt(snapshot.get("summary") or snapshot.get("speaker") or snapshot.get("source") or "live macro snapshot")
    )
    point_key = str(
        snapshot.get("event_resolution_join_key")
        or snapshot.get("video_id")
        or snapshot.get("youtube_url")
        or f"snapshot::{_day_key(ts)}::{headline}"
    ).strip()
    stance = str(snapshot.get("stance") or "").strip().lower() or _sentiment_label(sentiment_value)

    return {
        "timestamp_utc": ts.isoformat(),
        "day_utc": _day_key(ts),
        "week_key": _week_key_from_dt(ts),
        "month_key": _month_key_from_dt(ts),
        "year_key": _year_key_from_dt(ts),
        "point_key": point_key,
        "source_kind": "live_snapshot",
        "event_type": "live_macro_snapshot",
        "source": str(snapshot.get("source") or "").strip(),
        "speaker": str(snapshot.get("speaker") or "").strip(),
        "channel": "live_macro_snapshot",
        "headline": headline,
        "stance": stance,
        "sentiment_hint": round(float(sentiment_value), 6),
        "shock_hint": round(float(shock_value), 6),
        "confidence": round(float(confidence_value), 6),
        "weight": round(max(0.2, float(shock_value)) * max(0.3, float(confidence_value)), 6),
    }


def _aggregate_period_rows(
    points: list[dict[str, Any]],
    *,
    key_name: str,
    end_day_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        key = str(point.get(key_name) or "").strip()
        if key:
            grouped[key].append(point)

    rows: list[dict[str, Any]] = []
    previous_avg = 0.0
    for key in sorted(grouped.keys()):
        bucket = sorted(grouped[key], key=lambda row: str(row.get("timestamp_utc") or ""))
        total_weight = sum(max(float(row.get("weight", 0.0) or 0.0), 0.01) for row in bucket)
        weighted_sum = sum(float(row.get("sentiment_hint", 0.0) or 0.0) * max(float(row.get("weight", 0.0) or 0.0), 0.01) for row in bucket)
        avg_sentiment = weighted_sum / max(total_weight, 0.01)
        mean_shock = sum(float(row.get("shock_hint", 0.0) or 0.0) for row in bucket) / max(len(bucket), 1)
        latest = bucket[-1]
        sources = Counter(str(row.get("source") or "unknown") for row in bucket)
        bullish_count = sum(1 for row in bucket if float(row.get("sentiment_hint", 0.0) or 0.0) >= 0.1)
        bearish_count = sum(1 for row in bucket if float(row.get("sentiment_hint", 0.0) or 0.0) <= -0.1)
        neutral_count = len(bucket) - bullish_count - bearish_count
        rows.append(
            {
                key_name: key,
                end_day_name: str(latest.get("day_utc") or ""),
                "end_timestamp_utc": str(latest.get("timestamp_utc") or ""),
                "avg_sentiment_hint": round(float(avg_sentiment), 6),
                "latest_sentiment_hint": round(float(latest.get("sentiment_hint", 0.0) or 0.0), 6),
                "mean_shock_hint": round(float(mean_shock), 6),
                "event_count": int(len(bucket)),
                "bullish_event_count": int(bullish_count),
                "bearish_event_count": int(bearish_count),
                "neutral_event_count": int(neutral_count),
                "change_vs_previous_period": round(float(avg_sentiment - previous_avg), 6),
                "sentiment_label": _sentiment_label(avg_sentiment),
                "top_sources": _rank_counter(sources),
            }
        )
        previous_avg = avg_sentiment
    return rows


def _select_period_summary(
    rows: list[dict[str, Any]],
    *,
    selected_key: str,
    key_name: str,
) -> dict[str, Any]:
    matched = None
    latest = rows[-1] if rows else None
    for row in rows:
        if str(row.get(key_name) or "") == selected_key:
            matched = row
            break
    selected = matched or latest
    if selected is None:
        return {
            key_name: selected_key,
            "available": False,
            "data_status": "no_data",
            "event_count": 0,
            "avg_sentiment_hint": 0.0,
            "latest_sentiment_hint": 0.0,
            "change_vs_previous_period": 0.0,
            "sentiment_label": "neutral",
            "current_period_available": False,
        }
    return {
        **selected,
        "available": True,
        "current_period_available": bool(matched is not None),
        "data_status": "current" if matched is not None else "latest_available",
        "selected_key": selected_key,
    }


def _recent_events(points: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    rows = sorted(points, key=lambda row: str(row.get("timestamp_utc") or ""), reverse=True)
    return rows[: max(int(limit), 1)]


def build_sentiment_report(project_root: Path, *, day: str, lookback_days: int = 365) -> dict[str, Any]:
    selected_date = datetime.strptime(str(day), "%Y%m%d").date()
    event_files = _event_files(project_root, selected_day=day, lookback_days=max(int(lookback_days), 1))
    media_summary_files = _media_summary_files(project_root, selected_day=day, lookback_days=max(int(lookback_days), 1))
    live_macro_latest = _load_json(project_root / "data" / "external_context" / "live_macro_latest.json")
    live_macro_source = live_macro_latest if live_macro_latest else {}

    points: list[dict[str, Any]] = []
    seen_point_keys: set[str] = set()
    event_log_point_count = 0
    media_summary_point_count = 0
    snapshot_fallback_used = False

    def _append_point(point: dict[str, Any] | None) -> bool:
        nonlocal points
        if point is None:
            return False
        point_day = datetime.strptime(str(point["day_utc"]), "%Y%m%d").date()
        if point_day > selected_date:
            return False
        point_key = str(point.get("point_key") or "").strip()
        dedupe_key = point_key or f"{point.get('timestamp_utc', '')}::{point.get('headline', '')}"
        if dedupe_key in seen_point_keys:
            return False
        seen_point_keys.add(dedupe_key)
        points.append(point)
        return True

    for row in _iter_jsonl_dicts(event_files):
        if _append_point(_extract_sentiment_point(row)):
            event_log_point_count += 1

    for path in media_summary_files:
        if _append_point(_extract_media_summary_point(_load_json(path), source_path=path)):
            media_summary_point_count += 1

    if not points:
        snapshot_fallback_used = _append_point(_extract_live_snapshot_point(live_macro_source, selected_day=day))

    points.sort(key=lambda row: str(row.get("timestamp_utc") or ""))

    daily_rows = _aggregate_period_rows(points, key_name="day_utc", end_day_name="day_end_day_utc")
    weekly_rows = _aggregate_period_rows(points, key_name="week_key", end_day_name="week_end_day_utc")
    monthly_rows = _aggregate_period_rows(points, key_name="month_key", end_day_name="month_end_day_utc")
    yearly_rows = _aggregate_period_rows(points, key_name="year_key", end_day_name="year_end_day_utc")

    week_key = _week_key_from_dt(datetime.combine(selected_date, datetime.min.time(), tzinfo=timezone.utc))
    month_key = selected_date.strftime("%Y%m")
    year_key = selected_date.strftime("%Y")
    latest_point = points[-1] if points else {}
    source_files = event_files + media_summary_files

    return {
        "timestamp_utc": _utc_now().isoformat(),
        "schema_version": 2,
        "ok": bool(points),
        "selected_day_utc": day,
        "lookback_days": int(max(int(lookback_days), 1)),
        "source_files_scanned": int(len(source_files)),
        "source_files": [str(path) for path in source_files[:20]],
        "source_breakdown": {
            "event_log_files_scanned": int(len(event_files)),
            "media_summary_files_scanned": int(len(media_summary_files)),
            "event_log_points": int(event_log_point_count),
            "media_summary_points": int(media_summary_point_count),
            "snapshot_fallback_used": bool(snapshot_fallback_used),
        },
        "event_count": int(len(points)),
        "daily_sentiment_series": daily_rows[-90:],
        "weekly_sentiment_series": weekly_rows[-52:],
        "monthly_sentiment_series": monthly_rows[-24:],
        "yearly_sentiment_series": yearly_rows[-12:],
        "day": _select_period_summary(daily_rows, selected_key=day, key_name="day_utc"),
        "week": _select_period_summary(weekly_rows, selected_key=week_key, key_name="week_key"),
        "month": _select_period_summary(monthly_rows, selected_key=month_key, key_name="month_key"),
        "year": _select_period_summary(yearly_rows, selected_key=year_key, key_name="year_key"),
        "recent_events": _recent_events(points),
        "latest_event": latest_point,
        "latest_live_macro_snapshot": {
            "headline": str(live_macro_source.get("headline") or ""),
            "summary": str(live_macro_source.get("summary") or ""),
            "speaker": str(live_macro_source.get("speaker") or ""),
            "source": str(live_macro_source.get("source") or ""),
            "sentiment_hint": _safe_float(
                live_macro_source.get("market_sentiment_hint") or live_macro_source.get("sentiment_hint"),
                0.0,
            ),
            "shock_hint": _safe_float(
                live_macro_source.get("market_shock_hint") or live_macro_source.get("shock_hint"),
                0.0,
            ),
            "stance": str(live_macro_source.get("stance") or ""),
        },
    }


def render_sentiment_graphs(
    payload: dict[str, Any],
    *,
    daily_chart_path: Path,
    weekly_chart_path: Path,
    monthly_chart_path: Path,
    yearly_chart_path: Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "available": False,
            "error": f"matplotlib_unavailable:{type(exc).__name__}:{exc}",
            "daily_png": "",
            "weekly_png": "",
            "monthly_png": "",
            "yearly_png": "",
        }

    daily_rows = [row for row in (payload.get("daily_sentiment_series") or []) if isinstance(row, dict)][-45:]
    weekly_rows = [row for row in (payload.get("weekly_sentiment_series") or []) if isinstance(row, dict)][-20:]
    monthly_rows = [row for row in (payload.get("monthly_sentiment_series") or []) if isinstance(row, dict)][-18:]
    yearly_rows = [row for row in (payload.get("yearly_sentiment_series") or []) if isinstance(row, dict)][-12:]
    for path in (daily_chart_path, weekly_chart_path, monthly_chart_path, yearly_chart_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def _plot(
        rows: list[dict[str, Any]],
        *,
        label_builder,
        value_key: str,
        title: str,
        color: str,
        out_path: Path,
    ) -> None:
        if not rows:
            return
        labels = [label_builder(row) for row in rows]
        values = [float(row.get(value_key, 0.0) or 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, values, color=color, linewidth=2.6, marker="o", markersize=5.8)
        ax.fill_between(labels, values, [0.0] * len(values), color=color, alpha=0.14)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.85)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Avg Sentiment")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    _plot(
        daily_rows,
        label_builder=lambda row: _display_day(str(row.get("day_utc", "")))[5:],
        value_key="avg_sentiment_hint",
        title="Daily Sentiment Trend",
        color="#0f766e",
        out_path=daily_chart_path,
    )
    _plot(
        weekly_rows,
        label_builder=lambda row: _display_day(str(row.get("week_key", "")))[5:],
        value_key="avg_sentiment_hint",
        title="Weekly Sentiment Trend",
        color="#2563eb",
        out_path=weekly_chart_path,
    )
    _plot(
        monthly_rows,
        label_builder=lambda row: str(row.get("month_key", "")),
        value_key="avg_sentiment_hint",
        title="Monthly Sentiment Trend",
        color="#b45309",
        out_path=monthly_chart_path,
    )
    _plot(
        yearly_rows,
        label_builder=lambda row: str(row.get("year_key", "")),
        value_key="avg_sentiment_hint",
        title="Yearly Sentiment Trend",
        color="#0f4c81",
        out_path=yearly_chart_path,
    )

    return {
        "available": bool(daily_rows or weekly_rows or monthly_rows or yearly_rows),
        "daily_png": str(daily_chart_path),
        "weekly_png": str(weekly_chart_path),
        "monthly_png": str(monthly_chart_path),
        "yearly_png": str(yearly_chart_path),
    }


def render_sentiment_markdown(payload: dict[str, Any]) -> str:
    day = payload.get("day") if isinstance(payload.get("day"), dict) else {}
    week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    month = payload.get("month") if isinstance(payload.get("month"), dict) else {}
    year = payload.get("year") if isinstance(payload.get("year"), dict) else {}
    graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
    source_breakdown = payload.get("source_breakdown") if isinstance(payload.get("source_breakdown"), dict) else {}
    recent_events = payload.get("recent_events") if isinstance(payload.get("recent_events"), list) else []
    methodology = _methodology_items(payload)

    lines = [
        "# Sentiment Report",
        "",
        f"- generated_utc: {payload.get('timestamp_utc', '')}",
        f"- selected_day_utc: {payload.get('selected_day_utc', '')}",
        f"- event_count: {int(payload.get('event_count', 0) or 0)}",
        f"- source_files_scanned: {int(payload.get('source_files_scanned', 0) or 0)}",
        f"- event_log_points: {int(source_breakdown.get('event_log_points', 0) or 0)}",
        f"- media_summary_points: {int(source_breakdown.get('media_summary_points', 0) or 0)}",
        f"- snapshot_fallback_used: {bool(source_breakdown.get('snapshot_fallback_used', False))}",
        "",
        "## How It Works",
        "",
    ]
    for item in methodology:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
        "## Day",
        "",
        f"- data_day_utc: {day.get('day_utc', '') or 'n/a'}",
        f"- data_status: {day.get('data_status', '')}",
        f"- avg_sentiment_hint: {float(day.get('avg_sentiment_hint', 0.0) or 0.0):.6f}",
        f"- latest_sentiment_hint: {float(day.get('latest_sentiment_hint', 0.0) or 0.0):.6f}",
        f"- change_vs_previous_period: {float(day.get('change_vs_previous_period', 0.0) or 0.0):.6f}",
        f"- event_count: {int(day.get('event_count', 0) or 0)}",
        "",
        "## Week",
        "",
        f"- week_key: {week.get('week_key', '') or 'n/a'}",
        f"- data_status: {week.get('data_status', '')}",
        f"- avg_sentiment_hint: {float(week.get('avg_sentiment_hint', 0.0) or 0.0):.6f}",
        f"- change_vs_previous_period: {float(week.get('change_vs_previous_period', 0.0) or 0.0):.6f}",
        f"- event_count: {int(week.get('event_count', 0) or 0)}",
        "",
        "## Month",
        "",
        f"- month_key: {month.get('month_key', '') or 'n/a'}",
        f"- data_status: {month.get('data_status', '')}",
        f"- avg_sentiment_hint: {float(month.get('avg_sentiment_hint', 0.0) or 0.0):.6f}",
        f"- change_vs_previous_period: {float(month.get('change_vs_previous_period', 0.0) or 0.0):.6f}",
        f"- event_count: {int(month.get('event_count', 0) or 0)}",
        "",
        "## Year",
        "",
        f"- year_key: {year.get('year_key', '') or 'n/a'}",
        f"- data_status: {year.get('data_status', '')}",
        f"- avg_sentiment_hint: {float(year.get('avg_sentiment_hint', 0.0) or 0.0):.6f}",
        f"- change_vs_previous_period: {float(year.get('change_vs_previous_period', 0.0) or 0.0):.6f}",
        f"- event_count: {int(year.get('event_count', 0) or 0)}",
        "",
        "## Graphs",
        "",
        f"- daily_png: {graphs.get('daily_png', '')}",
        f"- weekly_png: {graphs.get('weekly_png', '')}",
        f"- monthly_png: {graphs.get('monthly_png', '')}",
        f"- yearly_png: {graphs.get('yearly_png', '')}",
        "",
        "## Recent Events",
        "",
        ]
    )
    for row in recent_events:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('timestamp_utc', '')}: "
            f"source={row.get('source', '') or 'n/a'}, "
            f"stance={row.get('stance', '') or 'n/a'}, "
            f"sentiment={float(row.get('sentiment_hint', 0.0) or 0.0):+.4f}, "
            f"shock={float(row.get('shock_hint', 0.0) or 0.0):.4f}, "
            f"headline={str(row.get('headline', '') or 'n/a')}"
        )
    return "\n".join(lines).strip() + "\n"


def _methodology_items(payload: dict[str, Any]) -> list[str]:
    source_breakdown = payload.get("source_breakdown") if isinstance(payload.get("source_breakdown"), dict) else {}
    return [
        (
            "The report reads event history from "
            "`governance/events/live_macro_events_*.jsonl` first, then falls back to imported "
            "`data/external_context/live_macro_media/*/summary.json` records, and only uses the latest "
            "`live_macro_latest.json` snapshot if no historical events are available."
        ),
        (
            "Each event carries a sentiment score in the range `-1.0` to `+1.0`. "
            "Stance is labeled `bullish` at `>= +0.20`, `bearish` at `<= -0.20`, and `neutral` otherwise."
        ),
        (
            "Period averages are weighted rather than flat. The live-event path uses shock and confidence, "
            "while imported summary points also blend source quality, official-source strength, transcript quality, "
            "and confirmation or high-conviction flags."
        ),
        (
            "Daily, weekly, monthly, and yearly lines are all kept separately, so the report can still react "
            "to market-regime shifts instead of smoothing everything into one long average."
        ),
        (
            "If the selected day has no new event, the report will show `latest_available` and surface the most "
            "recent historical regime signal instead of inventing a fresh stance."
        ),
        (
            f"This run used {int(source_breakdown.get('event_log_points', 0) or 0)} event-log points, "
            f"{int(source_breakdown.get('media_summary_points', 0) or 0)} imported summary points, and "
            f"snapshot fallback={str(bool(source_breakdown.get('snapshot_fallback_used', False))).lower()}."
        ),
    ]


def _path_uri(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve().as_uri()


def render_sentiment_html(payload: dict[str, Any], *, source_path: Path, generated_utc: str) -> str:
    day = payload.get("day") if isinstance(payload.get("day"), dict) else {}
    week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    month = payload.get("month") if isinstance(payload.get("month"), dict) else {}
    year = payload.get("year") if isinstance(payload.get("year"), dict) else {}
    graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
    recent_events = payload.get("recent_events") if isinstance(payload.get("recent_events"), list) else []
    source_breakdown = payload.get("source_breakdown") if isinstance(payload.get("source_breakdown"), dict) else {}
    methodology = _methodology_items(payload)

    chart_specs = [
        ("Daily Sentiment Trend", graphs.get("daily_png", "")),
        ("Weekly Sentiment Trend", graphs.get("weekly_png", "")),
        ("Monthly Sentiment Trend", graphs.get("monthly_png", "")),
        ("Yearly Sentiment Trend", graphs.get("yearly_png", "")),
    ]
    chart_cards: list[str] = []
    for title, raw_path in chart_specs:
        uri = _path_uri(raw_path)
        if not uri:
            continue
        chart_cards.append(
            "<section class=\"chart-card\">"
            f"<h2>{html.escape(title)}</h2>"
            f"<img src=\"{html.escape(uri)}\" alt=\"{html.escape(title)}\" />"
            "</section>"
        )

    event_rows: list[str] = []
    for row in recent_events:
        if not isinstance(row, dict):
            continue
        event_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('timestamp_utc', '')))}</td>"
            f"<td>{html.escape(str(row.get('event_type', '') or 'n/a'))}</td>"
            f"<td>{html.escape(str(row.get('source', '') or 'n/a'))}</td>"
            f"<td>{html.escape(str(row.get('speaker', '') or 'n/a'))}</td>"
            f"<td>{html.escape(str(row.get('stance', '') or 'n/a'))}</td>"
            f"<td>{float(row.get('sentiment_hint', 0.0) or 0.0):+.4f}</td>"
            f"<td>{float(row.get('shock_hint', 0.0) or 0.0):.4f}</td>"
            f"<td>{html.escape(str(row.get('headline', '') or 'n/a'))}</td>"
            "</tr>"
        )
    methodology_rows = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in methodology
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Sentiment Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --ink: #1f2933;
      --muted: #66737f;
      --card: #fffaf2;
      --line: #d7ccb9;
      --accent: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, #efe8db 0%, #f7f3ec 100%); color: var(--ink); font: 15px/1.55 Georgia, 'Times New Roman', serif; }}
    .page {{ max-width: 1040px; margin: 0 auto; padding: 34px 24px 48px; }}
    .hero, .section, .chart-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 26px rgba(31, 41, 51, 0.08); }}
    .hero {{ padding: 24px 26px; }}
    .section {{ margin-top: 18px; padding: 18px 22px; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 18px; }}
    .chart-card {{ padding: 16px 18px; }}
    h1, h2, h3 {{ margin: 0 0 10px; font-family: 'Avenir Next', 'Segoe UI', sans-serif; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; }}
    p.meta {{ margin: 0; color: var(--muted); }}
    .path {{ margin-top: 10px; font: 12px/1.4 'SF Mono', 'Menlo', monospace; color: var(--accent); word-break: break-all; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
    .stat-block {{ background: #f7f1e7; border: 1px solid #eadfce; border-radius: 14px; padding: 14px 16px; }}
    .stat-block p {{ margin: 6px 0 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 12px; border: 1px solid #eadfce; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .methodology-list {{ margin: 0; padding-left: 20px; }}
    .methodology-list li {{ margin: 0 0 10px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Sentiment Report</h1>
      <p class="meta">PDF generated {html.escape(generated_utc)}</p>
      <p class="meta">event_count={int(payload.get('event_count', 0) or 0)} | imported_summary_points={int(source_breakdown.get('media_summary_points', 0) or 0)} | snapshot_fallback={str(bool(source_breakdown.get('snapshot_fallback_used', False))).lower()}</p>
      <p class="path">Source: {html.escape(str(source_path))}</p>
    </section>
    <section class="section">
      <h2>How Stance Is Generated</h2>
      <ul class="methodology-list">
        {methodology_rows}
      </ul>
    </section>
    <section class="section">
      <div class="stats">
        <div class="stat-block">
          <h2>Day</h2>
          <p>data_day_utc: {html.escape(str(day.get('day_utc', '') or 'n/a'))}</p>
          <p>status: {html.escape(str(day.get('data_status', '') or 'n/a'))}</p>
          <p>avg_sentiment_hint: {float(day.get('avg_sentiment_hint', 0.0) or 0.0):+.4f}</p>
          <p>event_count: {int(day.get('event_count', 0) or 0)}</p>
        </div>
        <div class="stat-block">
          <h2>Week</h2>
          <p>week_key: {html.escape(str(week.get('week_key', '') or 'n/a'))}</p>
          <p>status: {html.escape(str(week.get('data_status', '') or 'n/a'))}</p>
          <p>avg_sentiment_hint: {float(week.get('avg_sentiment_hint', 0.0) or 0.0):+.4f}</p>
          <p>event_count: {int(week.get('event_count', 0) or 0)}</p>
        </div>
        <div class="stat-block">
          <h2>Month</h2>
          <p>month_key: {html.escape(str(month.get('month_key', '') or 'n/a'))}</p>
          <p>status: {html.escape(str(month.get('data_status', '') or 'n/a'))}</p>
          <p>avg_sentiment_hint: {float(month.get('avg_sentiment_hint', 0.0) or 0.0):+.4f}</p>
          <p>event_count: {int(month.get('event_count', 0) or 0)}</p>
        </div>
        <div class="stat-block">
          <h2>Year</h2>
          <p>year_key: {html.escape(str(year.get('year_key', '') or 'n/a'))}</p>
          <p>status: {html.escape(str(year.get('data_status', '') or 'n/a'))}</p>
          <p>avg_sentiment_hint: {float(year.get('avg_sentiment_hint', 0.0) or 0.0):+.4f}</p>
          <p>event_count: {int(year.get('event_count', 0) or 0)}</p>
        </div>
      </div>
    </section>
    <div class="chart-grid">
      {''.join(chart_cards)}
    </div>
    <section class="section">
      <h2>Recent Sentiment Events</h2>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Event Type</th>
            <th>Source</th>
            <th>Speaker</th>
            <th>Stance</th>
            <th>Sentiment</th>
            <th>Shock</th>
            <th>Headline</th>
          </tr>
        </thead>
        <tbody>
          {''.join(event_rows)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a daily, weekly, monthly, and yearly sentiment report from live macro history and imported media summaries."
    )
    parser.add_argument("--day", default=_utc_now().strftime("%Y%m%d"))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--out-file", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--md-out-file", default=str(DEFAULT_MD_PATH))
    parser.add_argument("--html-out-file", default=str(DEFAULT_HTML_PATH))
    parser.add_argument("--pdf-out-file", default=str(DEFAULT_PDF_PATH))
    parser.add_argument("--daily-chart-file", default=str(DEFAULT_DAILY_CHART_PATH))
    parser.add_argument("--weekly-chart-file", default=str(DEFAULT_WEEKLY_CHART_PATH))
    parser.add_argument("--monthly-chart-file", default=str(DEFAULT_MONTHLY_CHART_PATH))
    parser.add_argument("--yearly-chart-file", default=str(DEFAULT_YEARLY_CHART_PATH))
    parser.add_argument("--allow-gui-pdf-renderer", action=argparse.BooleanOptionalAction, default=_default_allow_gui_pdf_renderer())
    parser.add_argument("--json-only", action="store_true", help="Write the JSON snapshot only and skip charts, markdown, html, and pdf artifacts.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_sentiment_report(PROJECT_ROOT, day=str(args.day), lookback_days=max(int(args.lookback_days), 1))

    out_path = Path(args.out_file)
    md_path = Path(args.md_out_file)
    html_path = Path(args.html_out_file)
    pdf_path = Path(args.pdf_out_file)
    daily_chart_path = Path(args.daily_chart_file)
    weekly_chart_path = Path(args.weekly_chart_file)
    monthly_chart_path = Path(args.monthly_chart_file)
    yearly_chart_path = Path(args.yearly_chart_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json_only:
        payload["graphs"] = {
            "mode": "json_only",
            "daily_png": "",
            "weekly_png": "",
            "monthly_png": "",
            "yearly_png": "",
        }
        payload["pdf"] = {
            "available": False,
            "html_report_path": "",
            "pdf_path": str(pdf_path),
            "detail": "skipped_json_only",
        }
    else:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        payload["graphs"] = render_sentiment_graphs(
            payload,
            daily_chart_path=daily_chart_path,
            weekly_chart_path=weekly_chart_path,
            monthly_chart_path=monthly_chart_path,
            yearly_chart_path=yearly_chart_path,
        )
        generated_utc = str(payload.get("timestamp_utc") or _utc_now().isoformat())
        md_text = render_sentiment_markdown(payload)
        html_text = render_sentiment_html(payload, source_path=out_path, generated_utc=generated_utc)
        md_path.write_text(md_text, encoding="utf-8")
        html_path.write_text(html_text, encoding="utf-8")
        if pdf_path.exists():
            pdf_path.unlink()
        pdf_ok, pdf_detail = _render_pdf_from_html(
            html_path,
            pdf_path,
            allow_gui_renderer=bool(args.allow_gui_pdf_renderer),
        )
        payload["pdf"] = {
            "available": bool(pdf_ok),
            "html_report_path": str(html_path),
            "pdf_path": str(pdf_path),
            "detail": str(pdf_detail),
        }

    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        day_summary = payload.get("day", {}) if isinstance(payload.get("day"), dict) else {}
        week_summary = payload.get("week", {}) if isinstance(payload.get("week"), dict) else {}
        month_summary = payload.get("month", {}) if isinstance(payload.get("month"), dict) else {}
        year_summary = payload.get("year", {}) if isinstance(payload.get("year"), dict) else {}
        print(
            "sentiment_report "
            f"day={day_summary.get('day_utc', '') or 'n/a'} "
            f"day_avg={float(day_summary.get('avg_sentiment_hint', 0.0) or 0.0):+.4f} "
            f"week_avg={float(week_summary.get('avg_sentiment_hint', 0.0) or 0.0):+.4f} "
            f"month_avg={float(month_summary.get('avg_sentiment_hint', 0.0) or 0.0):+.4f} "
            f"year_avg={float(year_summary.get('avg_sentiment_hint', 0.0) or 0.0):+.4f}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
