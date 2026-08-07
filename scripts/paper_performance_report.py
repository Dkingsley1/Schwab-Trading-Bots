#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_PROJECT_ROOT
if str(SOURCE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PROJECT_ROOT))

from scripts.ops.artifact_generation_lock import (
    PAPER_PROFITABILITY_LOCK_ENV,
    paper_profitability_generation_lock,
)
from core.profitability_statistics import clustered_post_cost_statistics

DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_latest.md"
DEFAULT_HTML_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_latest.html"
DEFAULT_PDF_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_latest.pdf"
DEFAULT_DAILY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_daily_latest.png"
DEFAULT_WEEKLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_weekly_latest.png"
DEFAULT_MONTHLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_monthly_latest.png"
DEFAULT_QUARTERLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_quarterly_latest.png"
DEFAULT_SLEEVES_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_sleeves_latest.png"
DEFAULT_SLEEVE_ORDER = (
    "default",
    "conservative",
    "aggressive",
    "intraday_aggressive",
    "swing_aggressive",
    "dividend",
    "dividend_capture",
    "dividend_compound",
    "bond",
    "fx",
    "schwab_futures",
    "crypto_futures",
)
ADVANCED_TELEMETRY_FEATURES = (
    ("core_cross_sectional_rank_norm", "cross_sectional_rank"),
    ("core_event_reaction_norm", "event_reaction"),
    ("core_cross_asset_confirmation_norm", "cross_asset_confirmation"),
    ("day_failed_breakout_risk_norm", "failed_breakout_risk"),
    ("swing_weekly_pullback_quality_norm", "weekly_pullback_quality"),
    ("dividend_payout_stress_gate_norm", "payout_stress_gate"),
    ("long_term_factor_exposure_control_norm", "factor_exposure_control"),
    ("long_term_overlap_rebalance_norm", "overlap_rebalance"),
)
APP_BROWSER_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run(cmd: list[str]) -> tuple[int, str, str]:
    timeout_seconds = float(os.getenv("PAPER_PERFORMANCE_PDF_TIMEOUT_SECONDS", "20") or 20.0)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout_after_{timeout_seconds:.0f}s"
    except Exception as exc:
        return 1, "", str(exc)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sync_profitability_control(
    project_root: Path,
    performance_path: Path,
    *,
    enabled: bool,
) -> dict[str, Any]:
    sync_path = project_root / "governance" / "health" / "paper_profitability_generation_sync_latest.json"
    performance_hash = _file_sha256(performance_path)
    result: dict[str, Any] = {
        "timestamp_utc": _utc_now().isoformat(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "attempted": False,
        "reason": "disabled" if not enabled else "pending",
        "generation_id": performance_hash[:16],
        "paper_performance_path": str(performance_path),
        "paper_performance_sha256": performance_hash,
        "profitability_control_path": str(
            project_root / "governance" / "health" / "paper_profitability_control_latest.json"
        ),
        "lock_coordinated": True,
    }
    if not enabled:
        _atomic_write_json(sync_path, result)
        return result

    control_script = project_root / "scripts" / "ops" / "paper_profitability_control.py"
    if not control_script.exists():
        result.update(
            {
                "attempted": False,
                "reason": "custom_project_root_without_profitability_control",
                "overall_status": "not_applicable",
                "ok": project_root != SOURCE_PROJECT_ROOT,
            }
        )
        _atomic_write_json(sync_path, result)
        return result

    env = os.environ.copy()
    env[PAPER_PROFITABILITY_LOCK_ENV] = "1"
    timeout_seconds = max(float(os.getenv("PAPER_PROFITABILITY_SYNC_TIMEOUT_SECONDS", "30") or 30.0), 1.0)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(control_script),
                "--project-root",
                str(project_root),
                "--apply",
            ],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        rc = int(proc.returncode)
        stdout_tail = str(proc.stdout or "")[-500:]
        stderr_tail = str(proc.stderr or "")[-500:]
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout_tail = str(exc.stdout or "")[-500:]
        stderr_tail = f"timeout_after_{timeout_seconds:.1f}s"
    except Exception as exc:
        rc = 1
        stdout_tail = ""
        stderr_tail = str(exc)[-500:]

    control_path = project_root / "governance" / "health" / "paper_profitability_control_latest.json"
    control = _load_json(control_path)
    source_contract = control.get("paper_performance_input_contract")
    source_contract = source_contract if isinstance(source_contract, dict) else {}
    current_hash = _file_sha256(performance_path)
    synchronized = bool(
        rc == 0
        and current_hash
        and current_hash == performance_hash
        and source_contract.get("usable_for_profitability_grade", False)
        and str(source_contract.get("sha256") or "") == current_hash
    )
    result.update(
        {
            "timestamp_utc": _utc_now().isoformat(),
            "ok": synchronized,
            "overall_status": "ready" if synchronized else "degraded",
            "attempted": True,
            "reason": "hash_bound" if synchronized else "profitability_generation_sync_failed",
            "return_code": rc,
            "duration_seconds": round(time.monotonic() - started, 3),
            "paper_performance_sha256_after": current_hash,
            "profitability_source_sha256": str(source_contract.get("sha256") or ""),
            "source_usable_for_grade": bool(source_contract.get("usable_for_profitability_grade", False)),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    )
    _atomic_write_json(sync_path, result)
    return result


def _default_allow_gui_pdf_renderer() -> bool:
    return os.getenv("PAPER_PERFORMANCE_ALLOW_GUI_PDF_RENDERER", "").strip().lower() in {"1", "true", "yes", "on"}


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = (
        os.getenv("PAPER_PERFORMANCE_PDF_BIN", "").strip()
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
        profile_dir = Path(tempfile.mkdtemp(prefix="paper-performance-pdf-"))
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


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _day_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d")


def _paper_source_files(project_root: Path) -> tuple[list[Path], str]:
    source_groups = [
        (
            "paper_broker_bridge",
            sorted(
                list((project_root / "exports" / "paper_broker_bridge" / "paper").glob("paper_bridge_orders_*.jsonl"))
                + list((project_root / "exports" / "paper_broker_bridge" / "paper").glob("paper_bridge_orders_*.jsonl.gz"))
            ),
        ),
        (
            "trade_logs",
            sorted(
                list((project_root / "exports" / "trade_logs").rglob("paper_trades_*.jsonl"))
                + list((project_root / "exports" / "trade_logs").rglob("paper_trades_*.jsonl.gz"))
            ),
        ),
        (
            "root_paper_trades",
            sorted(list(project_root.glob("paper_trades_*.jsonl")) + list(project_root.glob("paper_trades_*.jsonl.gz"))),
        ),
    ]

    files: list[Path] = []
    kinds: list[str] = []
    seen: set[str] = set()
    for kind, paths in source_groups:
        if paths:
            kinds.append(kind)
        for candidate in paths:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(candidate)
    def _source_priority(item: Path) -> tuple[int, str]:
        text = str(item)
        if "/exports/trade_logs/" in text:
            return 0, text
        if "/exports/paper_broker_bridge/" in text:
            return 2, text
        return 1, text

    files.sort(key=_source_priority)
    return files, ",".join(kinds) if kinds else "none"


def _active_shadow_profiles(project_root: Path, *, day: str) -> dict[str, dict[str, Any]]:
    health_dir = project_root / "governance" / "health"
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(health_dir.glob("shadow_loop_*_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        ts = _parse_ts(payload.get("timestamp_utc"))
        if ts is None or _day_key(ts) != str(day):
            continue
        profile = str(payload.get("profile") or "").strip().lower()
        if not profile:
            continue
        current = latest.get(profile)
        if current is not None:
            current_ts = _parse_ts(current.get("timestamp_utc"))
            if current_ts is not None and ts <= current_ts:
                continue
        latest[profile] = {
            "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
            "state": str(payload.get("state") or "").strip().lower() or "unknown",
            "profile": profile,
            "broker": str(payload.get("broker") or "").strip().lower(),
            "domain": str(payload.get("domain") or "").strip().lower(),
            "symbols_total": int(payload.get("symbols_total", 0) or 0),
            "context_total": int(payload.get("context_total", 0) or 0),
            "pid": int(payload.get("pid", 0) or 0),
        }
    return latest


def _paper_row_signature(row: dict[str, Any]) -> str:
    stable = json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()


def _paper_execution_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    intent = row.get("order_intent_evidence") if isinstance(row.get("order_intent_evidence"), dict) else {}
    semantic_order = intent.get("semantic_order") if isinstance(intent.get("semantic_order"), dict) else {}
    execution_id = str(
        row.get("execution_id")
        or row.get("fill_id")
        or metadata.get("execution_id")
        or metadata.get("fill_id")
        or ""
    ).strip()
    if execution_id:
        return f"execution:{execution_id}"
    decision_id = str(
        row.get("decision_id")
        or metadata.get("decision_id")
        or semantic_order.get("decision_id")
        or ""
    ).strip()
    if decision_id:
        book_id = str(row.get("paper_book_id") or metadata.get("paper_book_id") or "unbound-book").strip()
        return f"paper-decision:{book_id}:{decision_id}"
    return f"content:{_paper_row_signature(row)}"


def _iter_rows(
    files: Iterable[Path],
    *,
    deduplication: dict[str, int] | None = None,
    evidence_through: datetime | None = None,
) -> Iterable[dict[str, Any]]:
    seen_rows: set[str] = set()
    counters = deduplication if deduplication is not None else {}
    counters.setdefault("records_read", 0)
    counters.setdefault("records_emitted", 0)
    counters.setdefault("mirrored_records_suppressed", 0)
    counters.setdefault("post_snapshot_records_deferred", 0)
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    counters["records_read"] += 1
                    timestamp = _parse_ts(row.get("timestamp_utc") or row.get("timestamp"))
                    if evidence_through is not None and timestamp is not None and timestamp > evidence_through:
                        counters["post_snapshot_records_deferred"] += 1
                        continue
                    identity = _paper_execution_identity(row)
                    if identity in seen_rows:
                        counters["mirrored_records_suppressed"] += 1
                        continue
                    seen_rows.add(identity)
                    counters["records_emitted"] += 1
                    yield row
        except Exception:
            continue


def _profile_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        text = str(meta.get("source_profile") or "").strip().lower()
        if text:
            return text
    text = str(row.get("profile") or "").strip().lower()
    return text or "default"


def _strategy_of(row: dict[str, Any]) -> str:
    text = str(row.get("strategy") or "").strip()
    if text:
        return text
    meta = row.get("metadata")
    if isinstance(meta, dict):
        meta_text = str(meta.get("strategy_id") or meta.get("bot_id") or "").strip()
        if meta_text:
            return meta_text
    symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
    action = str(row.get("action") or "").strip().upper() or "UNKNOWN"
    return f"unknown::{symbol}::{action}"


def _pnl_schema_version(row: dict[str, Any]) -> int:
    try:
        return int(row.get("paper_pnl_schema_version", 0) or 0)
    except Exception:
        return 0


def _paper_snapshot_key(row: dict[str, Any]) -> str:
    strategy = _strategy_of(row)
    if _pnl_schema_version(row) < 2:
        return strategy
    book_id = str(row.get("paper_book_id") or "").strip()
    if not book_id:
        book_id = "missing_book_id"
    return f"{book_id}::{strategy}"


def _net_total(row: dict[str, Any]) -> float:
    if _pnl_schema_version(row) >= 2:
        return _safe_float(row.get("paper_strategy_net_pnl_total"), 0.0)
    realized = _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("realized_pnl")))
    unrealized = _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("unrealized_pnl")))
    return float(realized + unrealized)


def _realized_total(row: dict[str, Any]) -> float:
    return _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("realized_pnl")))


def _unrealized_total(row: dict[str, Any]) -> float:
    return _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("unrealized_pnl")))


def _rank_counter(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"name": key, "executions": int(count)}
        for key, count in counter.most_common(limit)
    ]


def _empty_stats() -> dict[str, Any]:
    return {
        "executions": 0,
        "buy_count": 0,
        "sell_count": 0,
        "profiles": Counter(),
        "symbols": Counter(),
        "strategies": Counter(),
    }


def _update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["executions"] = int(stats.get("executions", 0)) + 1
    action = str(row.get("action") or "").upper().strip()
    if action.startswith("BUY"):
        stats["buy_count"] = int(stats.get("buy_count", 0)) + 1
    elif action.startswith("SELL"):
        stats["sell_count"] = int(stats.get("sell_count", 0)) + 1
    stats["profiles"][_profile_of(row)] += 1
    symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
    strategy = str(row.get("strategy") or "").strip() or "unknown"
    stats["symbols"][symbol] += 1
    stats["strategies"][strategy] += 1


def _chart_day_label(day_utc: str) -> str:
    text = str(day_utc or "").strip()
    if len(text) != 8:
        return text
    return f"{text[4:6]}-{text[6:8]}"


def _week_start_key(day_utc: str) -> str:
    dt = datetime.strptime(str(day_utc), "%Y%m%d").date()
    return (dt - timedelta(days=dt.weekday())).strftime("%Y%m%d")


def _month_key(day_utc: str) -> str:
    text = str(day_utc or "").strip()
    if len(text) != 8:
        return text
    return text[:6]


def _quarter_key(day_utc: str) -> str:
    dt = datetime.strptime(str(day_utc), "%Y%m%d").date()
    quarter = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{quarter}"


def _build_history_series(
    all_days: list[str],
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
    stats_by_day: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    prev_net = 0.0
    for dkey in all_days:
        totals = _profile_totals(latest_by_day_profile.get(dkey, {}))
        ending_net = float(totals["ending_net_pnl_total"])
        ending_realized = float(totals["ending_realized_pnl_total"])
        day_stats = stats_by_day.get(dkey, _empty_stats())
        series.append(
            {
                "day_utc": dkey,
                "executions": int(day_stats.get("executions", 0)),
                "ending_net_pnl_total": round(float(ending_net), 6),
                "ending_realized_pnl_total": round(float(ending_realized), 6),
                "change_vs_previous_day": round(float(ending_net - prev_net), 6),
            }
        )
        prev_net = ending_net
    return series


def _build_weekly_history_series(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_period_history_series(
        history_rows,
        key_builder=_week_start_key,
        label_name="week",
    )


def _build_period_history_series(
    history_rows: list[dict[str, Any]],
    *,
    key_builder,
    label_name: str,
) -> list[dict[str, Any]]:
    latest_by_period: dict[str, dict[str, Any]] = {}
    for row in history_rows:
        if not isinstance(row, dict):
            continue
        day_utc = str(row.get("day_utc", "")).strip()
        if len(day_utc) != 8:
            continue
        period_key = str(key_builder(day_utc))
        current = latest_by_period.get(period_key)
        if current is None or day_utc > str(current.get("day_utc", "")):
            latest_by_period[period_key] = row

    period_rows: list[dict[str, Any]] = []
    previous_end = 0.0
    for period_key in sorted(latest_by_period.keys()):
        row = latest_by_period[period_key]
        ending_net = float(row.get("ending_net_pnl_total", 0.0) or 0.0)
        period_rows.append(
            {
                f"{label_name}_key": period_key,
                f"{label_name}_end_day_utc": str(row.get("day_utc", "")),
                "ending_net_pnl_total": round(float(ending_net), 6),
                "change_vs_previous_period": round(float(ending_net - previous_end), 6),
            }
        )
        previous_end = ending_net
    return period_rows


def _build_period_change_series(
    *,
    selected_day: str,
    selected_net: float,
    week_to_date_change: float,
    all_days: list[str],
    history_by_day: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_date = datetime.strptime(selected_day, "%Y%m%d").date()
    period_rows: list[dict[str, Any]] = [
        {
            "label": "WTD",
            "window_days": int(selected_date.weekday() + 1),
            "change": round(float(week_to_date_change), 6),
        }
    ]
    for window_days in (7, 14, 21, 30):
        start_day = (selected_date - timedelta(days=window_days - 1)).strftime("%Y%m%d")
        prior_day = max((d for d in all_days if d < start_day), default="")
        prior_net = float((history_by_day.get(prior_day) or {}).get("ending_net_pnl_total", 0.0) or 0.0)
        available_days = sum(1 for d in all_days if start_day <= d <= selected_day)
        period_rows.append(
            {
                "label": f"{window_days}D",
                "window_days": int(window_days),
                "available_days": int(available_days),
                "change": round(float(selected_net - prior_net), 6),
            }
        )
    return period_rows


def _profile_totals(profile_rows: dict[str, tuple[datetime, dict[str, Any]]] | None) -> dict[str, Any]:
    if not profile_rows:
        return {
            "available": False,
            "ending_timestamp_utc": "",
            "ending_realized_pnl_total": 0.0,
            "ending_unrealized_pnl_total": 0.0,
            "ending_net_pnl_total": 0.0,
            "accounting_scope": "none",
            "paper_book_count": 0,
        }

    latest_ts: datetime | None = None
    realized_sum = 0.0
    unrealized_sum = 0.0

    def _iter_leaves(node: Any) -> Iterable[tuple[datetime, dict[str, Any]]]:
        if isinstance(node, tuple) and len(node) == 2:
            ts, row = node
            if isinstance(ts, datetime) and isinstance(row, dict):
                yield ts, row
            return
        if isinstance(node, dict):
            for child in node.values():
                yield from _iter_leaves(child)

    leaves = list(_iter_leaves(profile_rows))
    schema_v2_leaves = [(ts, row) for ts, row in leaves if _pnl_schema_version(row) >= 2]
    paper_book_count = 0
    accounting_scope = "legacy_strategy_snapshot_sum"
    if schema_v2_leaves:
        accounting_scope = "persistent_profile_book"
        latest_by_book: dict[tuple[str, str], tuple[datetime, dict[str, Any]]] = {}
        for ts, row in schema_v2_leaves:
            profile = str(row.get("paper_profile") or _profile_of(row)).strip().lower() or "default"
            book_id = str(row.get("paper_book_id") or "").strip() or f"missing::{profile}"
            key = (book_id, profile)
            current = latest_by_book.get(key)
            if current is None or ts > current[0]:
                latest_by_book[key] = (ts, row)
        paper_book_count = len(latest_by_book)
        for ts, row in latest_by_book.values():
            realized_sum += _safe_float(row.get("paper_profile_realized_pnl_total"), 0.0)
            unrealized_sum += _safe_float(row.get("paper_profile_unrealized_pnl_total"), 0.0)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
    else:
        for ts, row in leaves:
            realized_sum += _realized_total(row)
            unrealized_sum += _unrealized_total(row)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
    return {
        "available": True,
        "ending_timestamp_utc": latest_ts.isoformat().replace("+00:00", "Z") if latest_ts is not None else "",
        "ending_realized_pnl_total": round(float(realized_sum), 6),
        "ending_unrealized_pnl_total": round(float(unrealized_sum), 6),
        "ending_net_pnl_total": round(float(realized_sum + unrealized_sum), 6),
        "accounting_scope": accounting_scope,
        "paper_book_count": int(paper_book_count),
    }


def _build_sleeve_daily_series(
    all_days: list[str],
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
) -> dict[str, list[dict[str, Any]]]:
    sleeves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles = _ordered_profiles(latest_by_day_profile)
    for profile in profiles:
        prev_net = 0.0
        for dkey in all_days:
            row_map = latest_by_day_profile.get(dkey, {})
            current = row_map.get(profile)
            if current is None:
                continue
            totals = _profile_totals(current)
            ending_realized = float(totals.get("ending_realized_pnl_total", 0.0) or 0.0)
            ending_unrealized = float(totals.get("ending_unrealized_pnl_total", 0.0) or 0.0)
            ending_net = float(totals.get("ending_net_pnl_total", 0.0) or 0.0)
            sleeves[profile].append(
                {
                    "day_utc": dkey,
                    "ending_realized_pnl_total": round(float(ending_realized), 6),
                    "ending_unrealized_pnl_total": round(float(ending_unrealized), 6),
                    "ending_net_pnl_total": round(float(ending_net), 6),
                    "change_vs_previous_day": round(float(ending_net - prev_net), 6),
                }
            )
            prev_net = ending_net
    return {profile: sleeves.get(profile, []) for profile in profiles}


def _ordered_profiles(latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]]) -> list[str]:
    seen = {profile for rows in latest_by_day_profile.values() for profile in rows.keys()}
    ordered = list(DEFAULT_SLEEVE_ORDER)
    extras = sorted(seen.difference(DEFAULT_SLEEVE_ORDER))
    return ordered + extras


def _latest_profile_rows(
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
) -> dict[str, tuple[str, datetime, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]]]:
    latest: dict[str, tuple[str, datetime, dict[str, tuple[datetime, dict[str, Any]]]]] = {}
    for day_key, profile_rows in latest_by_day_profile.items():
        for profile, strategy_rows in profile_rows.items():
            ts_values = [ts for ts, _row in strategy_rows.values()]
            if not ts_values:
                continue
            ts = max(ts_values)
            current = latest.get(profile)
            if current is None or ts > current[1]:
                latest[profile] = (day_key, ts, strategy_rows)
    return latest


def _build_sleeve_period_history(
    sleeve_daily_series: dict[str, list[dict[str, Any]]],
    *,
    key_builder,
    label_name: str,
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for profile, rows in sleeve_daily_series.items():
        payload[profile] = _build_period_history_series(rows, key_builder=key_builder, label_name=label_name)
    return payload


def _sleeve_chart_profiles(sleeve_latest: list[dict[str, Any]]) -> list[str]:
    profiles: list[str] = []
    seen: set[str] = set()
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile", "")).strip()
        if not profile or profile in seen:
            continue
        seen.add(profile)
        profiles.append(profile)
    return profiles


def _sleeve_snapshot_points(sleeve_latest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile", "")).strip()
        if not profile:
            continue
        rows.append(
            {
                "profile": profile,
                "ending_net_pnl_total": float(row.get("ending_net_pnl_total", 0.0) or 0.0),
                "change_vs_previous_day": float(row.get("change_vs_previous_day", 0.0) or 0.0),
                "executions": int(row.get("executions", 0) or 0),
                "win_rate": row.get("win_rate"),
                "winning_strategy_count": int(row.get("winning_strategy_count", 0) or 0),
                "losing_strategy_count": int(row.get("losing_strategy_count", 0) or 0),
                "flat_strategy_count": int(row.get("flat_strategy_count", 0) or 0),
                "risk_adjusted_metric": str(row.get("risk_adjusted_metric", "") or ""),
                "risk_adjusted_metric_value": row.get("risk_adjusted_metric_value"),
                "data_status": str(row.get("data_status", "") or ""),
                "day_utc": str(row.get("day_utc", "") or ""),
            }
        )
    return rows


def _decorate_sleeve_latest_rows(
    sleeve_latest: list[dict[str, Any]],
    *,
    sleeve_daily_series: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        updated = dict(row)
        profile = str(updated.get("profile", "")).strip()
        daily_rows = sleeve_daily_series.get(profile) or []
        latest_day = daily_rows[-1] if daily_rows else {}
        updated["change_vs_previous_day"] = round(float(latest_day.get("change_vs_previous_day", 0.0) or 0.0), 6)
        updated["history_points"] = int(len(daily_rows))
        updated.update(_sleeve_risk_metric(profile, daily_rows))
        rows.append(updated)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _sample_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _candidate_profitability_cutoff(project_root: Path) -> datetime | None:
    try:
        state = json.loads(
            (project_root / "governance" / "runtime" / "production_candidate_state.json").read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None
    windows = state.get("scope_windows_started_utc", {}) if isinstance(state, dict) else {}
    candidates = [
        _parse_ts(windows.get(scope))
        for scope in ("strategy", "execution", "risk", "data", "promotion", "dependencies")
    ]
    return max((value for value in candidates if value is not None), default=None)


def _post_cost_expectancy(
    rows: Iterable[dict[str, Any]],
    *,
    minimum_samples: int = 30,
) -> dict[str, Any]:
    samples: list[tuple[float, float, dict[str, Any]]] = []
    timestamps: list[datetime] = []
    for row in rows:
        if not isinstance(row, dict) or _pnl_schema_version(row) < 2:
            continue
        if "post_cost_pnl_delta" not in row or "post_cost_return_bps" not in row:
            continue
        pnl_delta = _safe_float(row.get("post_cost_pnl_delta"), float("nan"))
        return_bps = _safe_float(row.get("post_cost_return_bps"), float("nan"))
        if not math.isfinite(pnl_delta) or not math.isfinite(return_bps):
            continue
        samples.append((pnl_delta, return_bps, row))
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is not None:
            timestamps.append(ts)

    sample_count = len(samples)
    required = max(int(minimum_samples), 1)
    if not samples:
        return {
            "available": False,
            "status": "no_schema_v2_trade_deltas",
            "sample_count": 0,
            "minimum_samples": required,
            "evidence_sufficient": False,
            "positive_lower_confidence_bound_95": False,
            "promotion_evidence_sufficient": False,
            "positive_clustered_lower_confidence_bound_95": False,
            "promotion_blockers": ["no_post_cost_observations"],
        }

    pnl_values = [item[0] for item in samples]
    return_values = [item[1] for item in samples]
    mean_pnl = _mean(pnl_values)
    mean_return = _mean(return_values)
    pnl_se = _sample_stddev(pnl_values) / math.sqrt(sample_count) if sample_count > 1 else 0.0
    return_se = _sample_stddev(return_values) / math.sqrt(sample_count) if sample_count > 1 else 0.0
    pnl_lcb = mean_pnl - (1.96 * pnl_se)
    return_lcb = mean_return - (1.96 * return_se)
    evidence_sufficient = sample_count >= required
    positive_lcb = evidence_sufficient and pnl_lcb > 0.0 and return_lcb > 0.0
    ordered_samples = sorted(
        samples,
        key=lambda item: _parse_ts(item[2].get("timestamp_utc")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl_delta, _return_bps, _row in ordered_samples:
        cumulative += pnl_delta
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    if not evidence_sufficient:
        status = "insufficient_evidence"
    elif positive_lcb:
        status = "positive_with_95pct_confidence"
    elif mean_pnl > 0.0 and mean_return > 0.0:
        status = "positive_mean_confidence_pending"
    else:
        status = "nonpositive_post_cost_expectancy"

    robust = clustered_post_cost_statistics(
        [item[2] for item in samples],
        minimum_samples=required,
        minimum_days=max(int(os.getenv("PAPER_PROFIT_MIN_INDEPENDENT_DAYS", "7") or 7), 1),
        minimum_symbols=max(int(os.getenv("PAPER_PROFIT_MIN_SYMBOLS", "5") or 5), 1),
        minimum_effective_samples=max(float(os.getenv("PAPER_PROFIT_MIN_EFFECTIVE_SAMPLES", "20") or 20.0), 1.0),
        hypothesis_count=max(len({str(item[2].get("strategy") or "") for item in samples}), 1),
        bootstrap_iterations=max(int(os.getenv("PAPER_PROFIT_BLOCK_BOOTSTRAP_ITERATIONS", "1200") or 1200), 200),
    )
    promotion_sufficient = bool(robust.get("promotion_evidence_sufficient", False))
    promotion_positive = bool(robust.get("positive_clustered_lower_confidence_bound_95", False))
    if not promotion_sufficient:
        promotion_status = "independent_evidence_pending"
    elif promotion_positive:
        promotion_status = "positive_clustered_and_bootstrapped"
    elif mean_pnl > 0.0 and mean_return > 0.0:
        promotion_status = "positive_mean_robust_confidence_pending"
    else:
        promotion_status = "nonpositive_robust_post_cost_expectancy"

    return {
        "available": True,
        "status": status,
        "accounting_scope": "schema_v2_trade_delta",
        "sample_count": int(sample_count),
        "minimum_samples": required,
        "evidence_sufficient": bool(evidence_sufficient),
        "positive_lower_confidence_bound_95": bool(positive_lcb),
        "promotion_evidence_sufficient": promotion_sufficient,
        "positive_clustered_lower_confidence_bound_95": promotion_positive,
        "promotion_status": promotion_status,
        "promotion_blockers": list(robust.get("blockers") or []),
        "robust_statistics": robust,
        "first_sample_timestamp_utc": min(timestamps).isoformat().replace("+00:00", "Z") if timestamps else "",
        "last_sample_timestamp_utc": max(timestamps).isoformat().replace("+00:00", "Z") if timestamps else "",
        "positive_sample_count": int(sum(1 for value in pnl_values if value > 0.0)),
        "positive_sample_rate": round(float(sum(1 for value in pnl_values if value > 0.0) / sample_count), 6),
        "total_post_cost_pnl_delta": round(float(sum(pnl_values)), 6),
        "mean_post_cost_pnl_delta": round(float(mean_pnl), 6),
        "standard_error_post_cost_pnl_delta": round(float(pnl_se), 6),
        "lower_confidence_bound_95_post_cost_pnl_delta": round(float(pnl_lcb), 6),
        "mean_post_cost_return_bps": round(float(mean_return), 6),
        "standard_error_post_cost_return_bps": round(float(return_se), 6),
        "lower_confidence_bound_95_post_cost_return_bps": round(float(return_lcb), 6),
        "max_cumulative_drawdown_post_cost_pnl": round(float(max_drawdown), 6),
        "execution_notional_total": round(
            float(sum(max(_safe_float(item[2].get("execution_notional"), 0.0), 0.0) for item in samples)),
            6,
        ),
        "expected_execution_cost_total": round(
            float(sum(max(_safe_float(item[2].get("expected_execution_cost_amount"), 0.0), 0.0) for item in samples)),
            6,
        ),
    }


def _sleeve_risk_metric(profile: str, daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [
        float(row.get("change_vs_previous_day", 0.0) or 0.0)
        for row in daily_rows
        if isinstance(row, dict)
    ]
    returns = [value for value in returns if math.isfinite(value)]
    normalized_profile = str(profile or "").strip().lower()
    metric: dict[str, Any] = {
        "risk_adjusted_metric": "",
        "risk_adjusted_metric_value": None,
        "risk_adjusted_metric_basis": "daily_pnl_change",
        "risk_adjusted_metric_sample_count": int(len(returns)),
    }
    if "aggressive" in normalized_profile:
        downside = [min(value, 0.0) for value in returns]
        downside_dev = math.sqrt(_mean([value * value for value in downside])) if downside else 0.0
        sortino = (_mean(returns) / downside_dev) if downside_dev > 0.0 else None
        metric.update(
            {
                "risk_adjusted_metric": "sortino_ratio",
                "risk_adjusted_metric_value": round(float(sortino), 6) if sortino is not None else None,
                "sortino_ratio": round(float(sortino), 6) if sortino is not None else None,
                "sortino_downside_deviation": round(float(downside_dev), 6),
            }
        )
    elif "conservative" in normalized_profile:
        stddev = _sample_stddev(returns)
        sharpe = (_mean(returns) / stddev) if stddev > 0.0 else None
        metric.update(
            {
                "risk_adjusted_metric": "sharpe_ratio",
                "risk_adjusted_metric_value": round(float(sharpe), 6) if sharpe is not None else None,
                "sharpe_ratio": round(float(sharpe), 6) if sharpe is not None else None,
                "sharpe_stddev": round(float(stddev), 6),
            }
        )
    return metric


def _sleeve_display_rows(sleeve_latest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status_rank = {"current": 0, "current_live_no_fills": 1, "latest_available": 2, "no_data": 3}
    candidates: list[dict[str, Any]] = []
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        status = str(row.get("data_status", "") or "")
        executions = int(row.get("executions", 0) or 0)
        strategy_count = int(row.get("strategy_count", 0) or 0)
        ending_net = float(row.get("ending_net_pnl_total", 0.0) or 0.0)
        day_change = float(row.get("change_vs_previous_day", 0.0) or 0.0)
        if status == "no_data":
            continue
        if executions <= 0 and strategy_count <= 0 and abs(ending_net) <= 1e-9 and abs(day_change) <= 1e-9:
            continue
        candidates.append(dict(row))

    if not candidates:
        candidates = [dict(row) for row in sleeve_latest if isinstance(row, dict) and str(row.get("data_status", "") or "") != "no_data"]
    if not candidates:
        candidates = [dict(row) for row in sleeve_latest if isinstance(row, dict)]

    candidates.sort(
        key=lambda row: (
            status_rank.get(str(row.get("data_status", "") or ""), 9),
            -float(row.get("ending_net_pnl_total", 0.0) or 0.0),
            -abs(float(row.get("change_vs_previous_day", 0.0) or 0.0)),
            str(row.get("profile", "") or ""),
        )
    )
    return candidates


def _strategy_outcome_summary(
    strategy_rows: dict[str, tuple[datetime, dict[str, Any]]] | None,
) -> dict[str, Any]:
    if not strategy_rows:
        return {
            "strategy_count": 0,
            "winning_strategy_count": 0,
            "losing_strategy_count": 0,
            "flat_strategy_count": 0,
            "non_flat_strategy_count": 0,
            "win_rate": None,
        }

    wins = 0
    losses = 0
    flats = 0
    for _strategy, (_ts, row) in strategy_rows.items():
        net = _net_total(row)
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1
        else:
            flats += 1

    non_flat = wins + losses
    win_rate = round(float(wins / non_flat), 6) if non_flat > 0 else None
    return {
        "strategy_count": int(len(strategy_rows)),
        "winning_strategy_count": int(wins),
        "losing_strategy_count": int(losses),
        "flat_strategy_count": int(flats),
        "non_flat_strategy_count": int(non_flat),
        "win_rate": win_rate,
    }


def _format_win_rate(raw: Any) -> str:
    try:
        if raw is None:
            return "n/a"
        return f"{float(raw) * 100.0:.1f}%"
    except Exception:
        return "n/a"


def _format_optional_float(raw: Any, *, digits: int = 6) -> str:
    if raw is None:
        return "n/a"
    try:
        return f"{float(raw):.{int(digits)}f}"
    except Exception:
        return "n/a"


def _format_strategy_pnl_brief(rows: list[dict[str, Any]], *, positive: bool) -> str:
    if not rows:
        return "n/a"
    parts: list[str] = []
    for row in rows:
        strategy = str(row.get("strategy") or "").strip() or "unknown"
        value = float(row.get("ending_net_pnl_total", 0.0) or 0.0)
        parts.append(f"{strategy}({value:+.2f})")
    return ", ".join(parts)


def _strategy_snapshot_rankings(
    strategy_rows: dict[str, tuple[datetime, dict[str, Any]]] | None,
    *,
    limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    if not strategy_rows:
        return {
            "top_winning_strategies": [],
            "top_losing_strategies": [],
        }

    ranked = [
        {
            "strategy": _strategy_of(row),
            "paper_book_id": str(row.get("paper_book_id") or "").strip(),
            "ending_net_pnl_total": round(float(_net_total(row)), 6),
        }
        for _snapshot_key, (_ts, row) in strategy_rows.items()
    ]
    ranked.sort(key=lambda item: (float(item["ending_net_pnl_total"]), item["strategy"]), reverse=True)
    winners = [row for row in ranked if float(row.get("ending_net_pnl_total", 0.0) or 0.0) > 0.0][: max(int(limit), 1)]
    losers = [
        row
        for row in sorted(ranked, key=lambda item: (float(item["ending_net_pnl_total"]), item["strategy"]))
        if float(row.get("ending_net_pnl_total", 0.0) or 0.0) < 0.0
    ][: max(int(limit), 1)]
    return {
        "top_winning_strategies": winners,
        "top_losing_strategies": losers,
    }


def _session_bucket(ts: datetime | None) -> str:
    if ts is None:
        return "unknown"
    local = ts.astimezone()
    minutes = (local.hour * 60) + local.minute
    if minutes < 570:
        return "premarket"
    if minutes <= 960:
        return "intraday"
    return "after_hours"


def _bucket_norm(raw: Any, *, low: float, high: float) -> str:
    value = _safe_float(raw, 0.0)
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "medium"


def _summarize_latest_feature_telemetry(
    strategy_rows: dict[str, tuple[datetime, dict[str, Any]]] | None,
) -> dict[str, dict[str, Any]]:
    if not strategy_rows:
        return {}

    summary: dict[str, dict[str, Any]] = {}
    for feature_name, label in ADVANCED_TELEMETRY_FEATURES:
        values: list[float] = []
        for _strategy, (_ts, row) in strategy_rows.items():
            if feature_name not in row:
                continue
            values.append(_safe_float(row.get(feature_name), 0.0))
        if not values:
            continue
        mean_norm = sum(values) / max(len(values), 1)
        summary[feature_name] = {
            "label": label,
            "mean_norm": round(float(mean_norm), 6),
            "max_norm": round(float(max(values)), 6),
            "sample_count": int(len(values)),
            "high_count": int(sum(1 for value in values if value >= 0.67)),
            "bucket": _bucket_norm(mean_norm, low=0.33, high=0.67),
        }
    return summary


def _format_feature_telemetry_brief(summary: Any, *, limit: int = 4) -> str:
    if not isinstance(summary, dict) or not summary:
        return "n/a"
    rows: list[tuple[str, float, str]] = []
    for item in summary.values():
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        rows.append(
            (
                label,
                float(item.get("mean_norm", 0.0) or 0.0),
                str(item.get("bucket") or "unknown"),
            )
        )
    if not rows:
        return "n/a"
    rows.sort(key=lambda row: (-row[1], row[0]))
    return ", ".join(f"{label}={mean_norm:.2f}/{bucket}" for label, mean_norm, bucket in rows[: max(int(limit), 1)])


def _summarize_latest_cause_attribution(
    strategy_rows: dict[str, tuple[datetime, dict[str, Any]]] | None,
) -> dict[str, Any]:
    if not strategy_rows:
        return {
            "top_loss_causes": [],
            "tca_summary": {},
        }

    cause_buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "loss_total": 0.0, "net_total": 0.0})
    slippage_gaps: list[float] = []
    expected_slippage: list[float] = []
    realized_slippage: list[float] = []
    partial_fill_ratios: list[float] = []
    poor_fill_count = 0
    for _strategy, (ts, row) in strategy_rows.items():
        net = float(_net_total(row))
        spread_regime = str(row.get("spread_regime") or "unknown")
        event_bucket = _bucket_norm(row.get("event_proximity_norm"), low=0.2, high=0.65)
        tradeability_bucket = _bucket_norm(row.get("tradeability_score"), low=0.45, high=0.70)
        source_quality_bucket = _bucket_norm(row.get("source_quality_norm"), low=0.35, high=0.75)
        conflict_bucket = _bucket_norm(row.get("allocation_conflict_norm"), low=0.2, high=0.55)
        fill_bucket = str(row.get("expected_fill_quality_bucket") or "unknown")
        for cause in (
            f"spread_regime:{spread_regime}",
            f"event_proximity:{event_bucket}",
            f"tradeability:{tradeability_bucket}",
            f"source_quality:{source_quality_bucket}",
            f"session:{_session_bucket(ts)}",
            f"conflict:{conflict_bucket}",
            f"fill_quality:{fill_bucket}",
        ):
            bucket = cause_buckets[cause]
            bucket["count"] += 1
            bucket["net_total"] += net
            if net < 0.0:
                bucket["loss_total"] += abs(net)
        slippage_gaps.append(_safe_float(row.get("slippage_gap_bps"), 0.0))
        expected_slippage.append(_safe_float(row.get("expected_slippage_bps"), 0.0))
        realized_slippage.append(_safe_float(row.get("realized_slippage_bps"), 0.0))
        partial_fill_ratios.append(_safe_float(row.get("expected_partial_fill_ratio"), 1.0))
        if str(fill_bucket).lower() in {"poor", "fair"}:
            poor_fill_count += 1

    ranked_causes = [
        {
            "cause": cause,
            "count": int(values["count"]),
            "loss_total": round(float(values["loss_total"]), 6),
            "net_total": round(float(values["net_total"]), 6),
        }
        for cause, values in cause_buckets.items()
        if float(values["loss_total"]) > 0.0
    ]
    ranked_causes.sort(key=lambda row: (-float(row["loss_total"]), row["cause"]))
    tca_summary = {
        "mean_expected_slippage_bps": round(sum(expected_slippage) / max(len(expected_slippage), 1), 6),
        "mean_realized_slippage_bps": round(sum(realized_slippage) / max(len(realized_slippage), 1), 6),
        "mean_slippage_gap_bps": round(sum(slippage_gaps) / max(len(slippage_gaps), 1), 6),
        "mean_partial_fill_ratio": round(sum(partial_fill_ratios) / max(len(partial_fill_ratios), 1), 6),
        "poor_or_fair_fill_count": int(poor_fill_count),
    }
    return {
        "top_loss_causes": ranked_causes[:5],
        "tca_summary": tca_summary,
    }


def _build_sleeve_latest_summary(
    *,
    day: str,
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
    stats_by_day: dict[str, dict[str, Any]],
    active_shadow_profiles: dict[str, dict[str, Any]] | None = None,
    post_cost_rows_by_profile: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    profile_rows = latest_by_day_profile.get(day, {})
    latest_rows = _latest_profile_rows(latest_by_day_profile)
    live_profiles = active_shadow_profiles or {}
    profile_post_cost_rows = post_cost_rows_by_profile or {}
    rows: list[dict[str, Any]] = []
    seen_profiles = set(_ordered_profiles(latest_by_day_profile))
    ordered_profiles = list(_ordered_profiles(latest_by_day_profile))
    for profile in sorted(live_profiles.keys()):
        if profile not in seen_profiles:
            ordered_profiles.append(profile)
            seen_profiles.add(profile)

    for profile in ordered_profiles:
        current = profile_rows.get(profile)
        source_day = day
        profile_current: dict[str, tuple[datetime, dict[str, Any]]] | None = None
        current_day_available = False
        live_heartbeat = live_profiles.get(profile) or {}
        live_no_fills_yet = False
        if current is not None:
            profile_current = current
            current_day_available = True
        else:
            latest = latest_rows.get(profile)
            if latest is not None:
                source_day, _ts, profile_current = latest
            if live_heartbeat:
                current_day_available = True
                live_no_fills_yet = True
        source_stats = stats_by_day.get(source_day, _empty_stats())
        source_execs = (source_stats.get("profiles") or Counter())
        if profile_current is None:
            rows.append(
                {
                    "profile": profile,
                    "day_utc": day if live_no_fills_yet else "",
                    "current_day_available": bool(live_no_fills_yet),
                    "data_status": "current_live_no_fills" if live_no_fills_yet else "no_data",
                    "executions": 0,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": 0.0,
                    "ending_net_pnl_total": 0.0,
                    "live_shadow_status": str(live_heartbeat.get("state") or ""),
                    "live_shadow_timestamp_utc": str(live_heartbeat.get("timestamp_utc") or ""),
                    "activity_note": "live heartbeat active; no paper fills yet today" if live_no_fills_yet else "",
                    "strategy_count": 0,
                    "winning_strategy_count": 0,
                    "losing_strategy_count": 0,
                    "flat_strategy_count": 0,
                    "non_flat_strategy_count": 0,
                    "win_rate": None,
                    "win_rate_basis": "latest_non_flat_strategy_snapshots",
                    "top_winning_strategies": [],
                    "top_losing_strategies": [],
                    "top_loss_causes": [],
                    "tca_summary": {},
                    "advanced_feature_telemetry": {},
                    "advanced_feature_summary": "live_no_fills_yet" if live_no_fills_yet else "n/a",
                    "post_cost_expectancy": _post_cost_expectancy(profile_post_cost_rows.get(profile, [])),
                }
            )
            continue
        totals = _profile_totals(profile_current)
        outcome = _strategy_outcome_summary(profile_current)
        rankings = _strategy_snapshot_rankings(profile_current)
        attribution = _summarize_latest_cause_attribution(profile_current)
        feature_telemetry = _summarize_latest_feature_telemetry(profile_current)
        ending_realized = float(totals.get("ending_realized_pnl_total", 0.0) or 0.0)
        ending_unrealized = float(totals.get("ending_unrealized_pnl_total", 0.0) or 0.0)
        ending_net = float(totals.get("ending_net_pnl_total", 0.0) or 0.0)
        rows.append(
            {
                "profile": profile,
                "day_utc": source_day,
                "current_day_available": current_day_available,
                "data_status": "current_live_no_fills" if live_no_fills_yet else ("current" if current_day_available else "latest_available"),
                "executions": int(source_stats.get("profiles", Counter()).get(profile, 0) if current_day_available and not live_no_fills_yet else 0 if live_no_fills_yet else source_execs.get(profile, 0)),
                "ending_realized_pnl_total": round(float(ending_realized), 6),
                "ending_unrealized_pnl_total": round(float(ending_unrealized), 6),
                "ending_net_pnl_total": round(float(ending_net), 6),
                "accounting_scope": str(totals.get("accounting_scope") or ""),
                "paper_book_count": int(totals.get("paper_book_count", 0) or 0),
                "strategy_count": int(outcome.get("strategy_count", 0) or 0),
                "winning_strategy_count": int(outcome.get("winning_strategy_count", 0) or 0),
                "losing_strategy_count": int(outcome.get("losing_strategy_count", 0) or 0),
                "flat_strategy_count": int(outcome.get("flat_strategy_count", 0) or 0),
                "non_flat_strategy_count": int(outcome.get("non_flat_strategy_count", 0) or 0),
                "win_rate": outcome.get("win_rate"),
                "win_rate_basis": "latest_non_flat_strategy_snapshots",
                "top_winning_strategies": rankings.get("top_winning_strategies", []),
                "top_losing_strategies": rankings.get("top_losing_strategies", []),
                "top_loss_causes": attribution.get("top_loss_causes", []),
                "tca_summary": attribution.get("tca_summary", {}),
                "advanced_feature_telemetry": feature_telemetry,
                "advanced_feature_summary": _format_feature_telemetry_brief(feature_telemetry),
                "post_cost_expectancy": _post_cost_expectancy(profile_post_cost_rows.get(profile, [])),
                "live_shadow_status": str(live_heartbeat.get("state") or ""),
                "live_shadow_timestamp_utc": str(live_heartbeat.get("timestamp_utc") or ""),
                "activity_note": "live heartbeat active; no paper fills yet today" if live_no_fills_yet else "",
            }
        )
    return rows


def _summarize_day(
    *,
    day: str,
    profile_latest_rows: dict[str, dict[str, tuple[datetime, dict[str, Any]]]] | None,
    stats: dict[str, Any],
    previous_profile_latest_rows: dict[str, dict[str, tuple[datetime, dict[str, Any]]]] | None,
) -> dict[str, Any]:
    totals = _profile_totals(profile_latest_rows)
    previous_totals = _profile_totals(previous_profile_latest_rows)
    if not bool(totals.get("available", False)):
        return {
            "day_utc": day,
            "available": False,
            "executions": int(stats.get("executions", 0)),
            "buy_count": int(stats.get("buy_count", 0)),
            "sell_count": int(stats.get("sell_count", 0)),
            "unique_symbols": int(len(stats.get("symbols", {}))),
            "change_vs_previous_day": 0.0,
            "realized_change_vs_previous_day": 0.0,
            "ending_realized_pnl_total": 0.0,
            "ending_unrealized_pnl_total": 0.0,
            "ending_net_pnl_total": 0.0,
            "top_profiles": _rank_counter(stats.get("profiles", Counter())),
            "top_symbols": _rank_counter(stats.get("symbols", Counter())),
            "top_strategies": _rank_counter(stats.get("strategies", Counter())),
        }

    ending_realized = float(totals.get("ending_realized_pnl_total", 0.0) or 0.0)
    ending_unrealized = float(totals.get("ending_unrealized_pnl_total", 0.0) or 0.0)
    ending_net = float(totals.get("ending_net_pnl_total", 0.0) or 0.0)
    previous_net = float(previous_totals.get("ending_net_pnl_total", 0.0) or 0.0)
    previous_realized = float(previous_totals.get("ending_realized_pnl_total", 0.0) or 0.0)

    return {
        "day_utc": day,
        "available": True,
        "ending_timestamp_utc": str(totals.get("ending_timestamp_utc") or ""),
        "executions": int(stats.get("executions", 0)),
        "buy_count": int(stats.get("buy_count", 0)),
        "sell_count": int(stats.get("sell_count", 0)),
        "unique_symbols": int(len(stats.get("symbols", {}))),
        "change_vs_previous_day": round(float(ending_net - previous_net), 6),
        "realized_change_vs_previous_day": round(float(ending_realized - previous_realized), 6),
        "ending_realized_pnl_total": round(float(ending_realized), 6),
        "ending_unrealized_pnl_total": round(float(ending_unrealized), 6),
        "ending_net_pnl_total": round(float(ending_net), 6),
        "accounting_scope": str(totals.get("accounting_scope") or ""),
        "paper_book_count": int(totals.get("paper_book_count", 0) or 0),
        "top_profiles": _rank_counter(stats.get("profiles", Counter())),
        "top_symbols": _rank_counter(stats.get("symbols", Counter())),
        "top_strategies": _rank_counter(stats.get("strategies", Counter())),
    }


def build_paper_performance_report(project_root: Path, *, day: str, week_days: int = 7) -> dict[str, Any]:
    evidence_through = _utc_now()
    files, source_kind = _paper_source_files(project_root)
    active_shadow_profiles = _active_shadow_profiles(project_root, day=day)
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(dict))
    stats_by_day: dict[str, dict[str, Any]] = defaultdict(_empty_stats)
    post_cost_rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_post_cost_rows: list[dict[str, Any]] = []
    profitability_cutoff = _candidate_profitability_cutoff(project_root)
    deduplication: dict[str, int] = {}

    for row in _iter_rows(
        files,
        deduplication=deduplication,
        evidence_through=evidence_through,
    ):
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None:
            continue
        dkey = _day_key(ts)
        profile = _profile_of(row)
        strategy = _paper_snapshot_key(row)
        _update_stats(stats_by_day[dkey], row)
        if (
            _pnl_schema_version(row) >= 2
            and "post_cost_pnl_delta" in row
            and "post_cost_return_bps" in row
            and (profitability_cutoff is None or ts >= profitability_cutoff)
        ):
            post_cost_rows_by_profile[profile].append(row)
            all_post_cost_rows.append(row)
        current = latest_by_day_profile[dkey][profile].get(strategy)
        if current is None or ts > current[0]:
            latest_by_day_profile[dkey][profile][strategy] = (ts, row)

    all_days = sorted(latest_by_day_profile.keys())
    selected_latest = latest_by_day_profile.get(day)
    previous_day = max((d for d in all_days if d < day), default="")
    previous_latest = latest_by_day_profile.get(previous_day)

    day_summary = _summarize_day(
        day=day,
        profile_latest_rows=selected_latest,
        stats=stats_by_day.get(day, _empty_stats()),
        previous_profile_latest_rows=previous_latest,
    )

    selected_date = datetime.strptime(day, "%Y%m%d").date()
    week_start_date = selected_date - timedelta(days=selected_date.weekday())
    week_start = week_start_date.strftime("%Y%m%d")
    rolling_start = (selected_date - timedelta(days=max(int(week_days), 1) - 1)).strftime("%Y%m%d")
    prior_week_day = max((d for d in all_days if d < week_start), default="")
    prior_rolling_day = max((d for d in all_days if d < rolling_start), default="")

    week_profiles: Counter[str] = Counter()
    week_symbols: Counter[str] = Counter()
    week_strategies: Counter[str] = Counter()
    week_exec = 0
    week_buys = 0
    week_sells = 0
    rolling_series: list[dict[str, Any]] = []

    for dkey in sorted(d for d in stats_by_day.keys() if week_start <= d <= day):
        day_stats = stats_by_day[dkey]
        week_exec += int(day_stats.get("executions", 0))
        week_buys += int(day_stats.get("buy_count", 0))
        week_sells += int(day_stats.get("sell_count", 0))
        week_profiles.update(day_stats.get("profiles", Counter()))
        week_symbols.update(day_stats.get("symbols", Counter()))
        week_strategies.update(day_stats.get("strategies", Counter()))

    history_series = _build_history_series(all_days, latest_by_day_profile, stats_by_day)
    history_by_day = {str(row.get("day_utc", "")): row for row in history_series if isinstance(row, dict)}
    for dkey in sorted(d for d in all_days if rolling_start <= d <= day):
        row = history_by_day.get(dkey)
        if isinstance(row, dict):
            rolling_series.append(
                {
                    "day_utc": dkey,
                    "ending_net_pnl_total": round(float(row.get("ending_net_pnl_total", 0.0) or 0.0), 6),
                    "change_vs_previous_day": round(float(row.get("change_vs_previous_day", 0.0) or 0.0), 6),
                }
            )

    selected_net = float(day_summary.get("ending_net_pnl_total", 0.0) or 0.0)
    selected_realized = float(day_summary.get("ending_realized_pnl_total", 0.0) or 0.0)
    prior_week_net = float((history_by_day.get(prior_week_day) or {}).get("ending_net_pnl_total", 0.0) or 0.0)
    prior_week_realized = float((history_by_day.get(prior_week_day) or {}).get("ending_realized_pnl_total", 0.0) or 0.0)
    prior_rolling_net = float((history_by_day.get(prior_rolling_day) or {}).get("ending_net_pnl_total", 0.0) or 0.0)
    weekly_history_series = _build_weekly_history_series(history_series)
    monthly_history_series = _build_period_history_series(
        history_series,
        key_builder=_month_key,
        label_name="month",
    )
    quarterly_history_series = _build_period_history_series(
        history_series,
        key_builder=_quarter_key,
        label_name="quarter",
    )
    sleeve_daily_series = _build_sleeve_daily_series(all_days, latest_by_day_profile)
    sleeve_weekly_history_series = _build_sleeve_period_history(
        sleeve_daily_series,
        key_builder=_week_start_key,
        label_name="week",
    )
    week_to_date_change = round(float(selected_net - prior_week_net), 6)

    week_summary = {
        "week_start_day_utc": week_start,
        "week_end_day_utc": day,
        "available": bool(day_summary.get("available", False)),
        "executions": int(week_exec),
        "buy_count": int(week_buys),
        "sell_count": int(week_sells),
        "week_to_date_change": week_to_date_change,
        "week_to_date_realized_change": round(float(selected_realized - prior_week_realized), 6),
        "rolling_change_days": int(max(int(week_days), 1)),
        "rolling_change": round(float(selected_net - prior_rolling_net), 6),
        "ending_net_pnl_total": round(float(selected_net), 6),
        "top_profiles": _rank_counter(week_profiles),
        "top_symbols": _rank_counter(week_symbols),
        "top_strategies": _rank_counter(week_strategies),
        "daily_series": rolling_series,
    }

    return {
        "timestamp_utc": _utc_now().isoformat(),
        "schema_version": 1,
        "ok": bool(selected_latest) or bool(active_shadow_profiles),
        "source_kind": source_kind,
        "source_files_scanned": int(len(files)),
        "source_files": [str(path) for path in files[:10]],
        "execution_deduplication": {
            **deduplication,
            "identity_policy": "execution_id_else_paper_book_and_decision_id_else_content_hash",
            "canonical_source_priority": ["trade_logs", "root_paper_trades", "paper_broker_bridge"],
        },
        "active_paper_profile_count_today": int(len(active_shadow_profiles)),
        "active_paper_profiles_today": [dict(active_shadow_profiles[key]) for key in sorted(active_shadow_profiles.keys())],
        "available_days": all_days[-14:],
        "history_daily_series": history_series[-60:],
        "weekly_history_series": weekly_history_series[-16:],
        "monthly_history_series": monthly_history_series[-18:],
        "quarterly_history_series": quarterly_history_series[-16:],
        "sleeve_daily_series": {profile: rows[-60:] for profile, rows in sorted(sleeve_daily_series.items())},
        "sleeve_weekly_history_series": {profile: rows[-16:] for profile, rows in sorted(sleeve_weekly_history_series.items())},
        "sleeve_latest": _decorate_sleeve_latest_rows(
            _build_sleeve_latest_summary(
                day=day,
                latest_by_day_profile=latest_by_day_profile,
                stats_by_day=stats_by_day,
                active_shadow_profiles=active_shadow_profiles,
                post_cost_rows_by_profile=post_cost_rows_by_profile,
            ),
            sleeve_daily_series=sleeve_daily_series,
        ),
        "period_change_series": _build_period_change_series(
            selected_day=day,
            selected_net=float(selected_net),
            week_to_date_change=float(week_to_date_change),
            all_days=all_days,
            history_by_day=history_by_day,
        ),
        "day": day_summary,
        "week": week_summary,
        "post_cost_expectancy": _post_cost_expectancy(all_post_cost_rows),
        "profitability_evidence_window": {
            "candidate_cutoff_utc": profitability_cutoff.isoformat() if profitability_cutoff is not None else "",
            "evidence_through_utc": evidence_through.isoformat(),
            "candidate_filter_active": profitability_cutoff is not None,
            "snapshot_watermark_active": True,
            "policy": "post-cost promotion evidence excludes samples before the latest affected candidate scope window and defers rows after the published scan watermark",
        },
    }


def render_paper_performance_graphs(
    payload: dict[str, Any],
    *,
    daily_chart_path: Path,
    weekly_chart_path: Path,
    monthly_chart_path: Path,
    quarterly_chart_path: Path,
    sleeves_chart_path: Path,
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
            "quarterly_png": "",
            "sleeves_png": "",
        }

    daily_rows = [
        row
        for row in (payload.get("history_daily_series") or [])
        if isinstance(row, dict)
    ]
    weekly_rows = [
        row
        for row in (payload.get("weekly_history_series") or [])
        if isinstance(row, dict)
    ]
    monthly_rows = [
        row
        for row in (payload.get("monthly_history_series") or [])
        if isinstance(row, dict)
    ]
    quarterly_rows = [
        row
        for row in (payload.get("quarterly_history_series") or [])
        if isinstance(row, dict)
    ]
    sleeve_weekly_rows = payload.get("sleeve_weekly_history_series") if isinstance(payload.get("sleeve_weekly_history_series"), dict) else {}
    sleeve_latest = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    display_rows = _sleeve_display_rows(sleeve_latest)
    daily_rows = daily_rows[-30:]
    weekly_rows = weekly_rows[-12:]
    monthly_rows = monthly_rows[-18:]
    quarterly_rows = quarterly_rows[-16:]

    daily_chart_path.parent.mkdir(parents=True, exist_ok=True)
    weekly_chart_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_chart_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly_chart_path.parent.mkdir(parents=True, exist_ok=True)
    sleeves_chart_path.parent.mkdir(parents=True, exist_ok=True)

    if daily_rows:
        labels = [_chart_day_label(str(row.get("day_utc", ""))) for row in daily_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in daily_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#1d4ed8", linewidth=2.6, marker="o", markersize=5.8)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#93c5fd", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Daily Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(daily_chart_path, bbox_inches="tight")
        plt.close(fig)

    if weekly_rows:
        labels = [_chart_day_label(str(row.get("week_end_day_utc", ""))) for row in weekly_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in weekly_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#0f766e", linestyle=":", linewidth=2.4, marker="o", markersize=6.0)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#99f6e4", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Weekly Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(weekly_chart_path, bbox_inches="tight")
        plt.close(fig)

    if monthly_rows:
        labels = [str(row.get("month_key", "")) for row in monthly_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in monthly_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#7c3aed", linestyle=":", linewidth=2.4, marker="o", markersize=6.0)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#ddd6fe", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Monthly Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(monthly_chart_path, bbox_inches="tight")
        plt.close(fig)

    if quarterly_rows:
        labels = [str(row.get("quarter_key", "")) for row in quarterly_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in quarterly_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#b45309", linestyle=":", linewidth=2.4, marker="o", markersize=6.0)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#fde68a", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Quarterly Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(quarterly_chart_path, bbox_inches="tight")
        plt.close(fig)

    display_rows = _sleeve_display_rows(sleeve_latest)
    if display_rows:
        snapshot_points = _sleeve_snapshot_points(display_rows)
        labels = [str(row["profile"]) for row in snapshot_points]
        ending_values = [float(row["ending_net_pnl_total"]) for row in snapshot_points]
        day_change_values = [float(row.get("change_vs_previous_day", 0.0) or 0.0) for row in snapshot_points]
        ypos = list(range(len(labels)))
        fig_height = max(7.0, 3.4 + (0.56 * len(labels)))
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11.2, fig_height),
            dpi=160,
            gridspec_kw={"height_ratios": [1.55, 1.2]},
        )

        def _colors(values: list[float]) -> list[str]:
            out: list[str] = []
            for value in values:
                if value > 0:
                    out.append("#0f766e")
                elif value < 0:
                    out.append("#b91c1c")
                else:
                    out.append("#64748b")
            return out

        def _annotate(ax, values: list[float], rows: list[dict[str, Any]], *, include_wr: bool) -> None:
            for idx, row in enumerate(rows):
                value = float(values[idx])
                direction = "left" if value < 0 else "right"
                offset = -6 if value < 0 else 6
                status = str(row.get("data_status", "") or "current")
                day_tag = str(row.get("day_utc", "") or "")
                status_text = day_tag if status == "current" else f"{status}:{day_tag or 'n/a'}"
                summary = f"{value:.2f} | {status_text} | exec {int(row.get('executions', 0) or 0)}"
                if include_wr:
                    summary += f" | wr {_format_win_rate(row.get('win_rate'))}"
                ax.annotate(
                    summary,
                    xy=(value, idx),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=direction,
                    va="center",
                    fontsize=7.4,
                    color="#1f2933",
                )

        net_ax = axes[0]
        net_ax.barh(ypos, ending_values, color=_colors(ending_values), alpha=0.88)
        net_ax.axvline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        net_ax.set_yticks(ypos)
        net_ax.set_yticklabels(labels, fontsize=8.4)
        net_ax.invert_yaxis()
        net_ax.set_title("Sleeve Net PnL Scoreboard")
        net_ax.set_xlabel("Ending Net PnL")
        net_ax.grid(axis="x", linestyle="--", alpha=0.25)
        max_abs_net = max((abs(v) for v in ending_values), default=0.0)
        pad_net = max(1.0, max_abs_net * 0.14)
        net_ax.set_xlim(min(ending_values + [0.0]) - pad_net, max(ending_values + [0.0]) + pad_net)
        _annotate(net_ax, ending_values, snapshot_points, include_wr=True)

        change_ax = axes[1]
        change_ax.barh(ypos, day_change_values, color=_colors(day_change_values), alpha=0.82)
        change_ax.axvline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        change_ax.set_yticks(ypos)
        change_ax.set_yticklabels(labels, fontsize=8.4)
        change_ax.invert_yaxis()
        change_ax.set_title("Sleeve Day-Over-Day Change")
        change_ax.set_xlabel("Change Vs Previous Day")
        change_ax.grid(axis="x", linestyle="--", alpha=0.25)
        max_abs_change = max((abs(v) for v in day_change_values), default=0.0)
        pad_change = max(1.0, max_abs_change * 0.18)
        change_ax.set_xlim(min(day_change_values + [0.0]) - pad_change, max(day_change_values + [0.0]) + pad_change)
        _annotate(change_ax, day_change_values, snapshot_points, include_wr=False)

        fig.tight_layout()
        fig.savefig(sleeves_chart_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "available": bool(daily_rows or weekly_rows or monthly_rows or quarterly_rows),
        "daily_png": str(daily_chart_path),
        "weekly_png": str(weekly_chart_path),
        "monthly_png": str(monthly_chart_path),
        "quarterly_png": str(quarterly_chart_path),
        "sleeves_png": str(sleeves_chart_path),
    }


def render_paper_performance_markdown(payload: dict[str, Any]) -> str:
    day = payload.get("day") if isinstance(payload.get("day"), dict) else {}
    week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
    sleeve_latest = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    active_profiles = payload.get("active_paper_profiles_today") if isinstance(payload.get("active_paper_profiles_today"), list) else []
    display_rows = _sleeve_display_rows(sleeve_latest)

    lines = [
        "# Paper Performance Report",
        "",
        f"- generated_utc: {payload.get('timestamp_utc', '')}",
        f"- source_kind: {payload.get('source_kind', '')}",
        f"- source_files_scanned: {int(payload.get('source_files_scanned', 0) or 0)}",
        "",
        "## End Of Day",
        "",
        f"- day_utc: {day.get('day_utc', '')}",
        f"- available: {bool(day.get('available', False))}",
        f"- executions: {int(day.get('executions', 0) or 0)}",
        f"- buys/sells: {int(day.get('buy_count', 0) or 0)}/{int(day.get('sell_count', 0) or 0)}",
        f"- ending_realized_pnl_total: {float(day.get('ending_realized_pnl_total', 0.0) or 0.0):.6f}",
        f"- ending_unrealized_pnl_total: {float(day.get('ending_unrealized_pnl_total', 0.0) or 0.0):.6f}",
        f"- ending_net_pnl_total: {float(day.get('ending_net_pnl_total', 0.0) or 0.0):.6f}",
        f"- change_vs_previous_day: {float(day.get('change_vs_previous_day', 0.0) or 0.0):.6f}",
        "",
        "## Week",
        "",
        f"- week_start_day_utc: {week.get('week_start_day_utc', '')}",
        f"- week_end_day_utc: {week.get('week_end_day_utc', '')}",
        f"- executions: {int(week.get('executions', 0) or 0)}",
        f"- week_to_date_change: {float(week.get('week_to_date_change', 0.0) or 0.0):.6f}",
        f"- week_to_date_realized_change: {float(week.get('week_to_date_realized_change', 0.0) or 0.0):.6f}",
        f"- rolling_{int(week.get('rolling_change_days', 7) or 7)}d_change: {float(week.get('rolling_change', 0.0) or 0.0):.6f}",
        "",
        "## Graphs",
        "",
        f"- daily_png: {graphs.get('daily_png', '')}",
        f"- weekly_png: {graphs.get('weekly_png', '')}",
        f"- monthly_png: {graphs.get('monthly_png', '')}",
        f"- quarterly_png: {graphs.get('quarterly_png', '')}",
        f"- sleeves_png: {graphs.get('sleeves_png', '')}",
        "",
        "## Active Paper Lanes Today",
        "",
        f"- active_paper_profile_count_today: {int(payload.get('active_paper_profile_count_today', 0) or 0)}",
    ]

    for item in active_profiles:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- {item.get('profile', '')}: "
            f"state={item.get('state', '')}, "
            f"broker={item.get('broker', '')}, "
            f"heartbeat={item.get('timestamp_utc', '')}, "
            f"symbols={int(item.get('symbols_total', 0) or 0)}"
        )

    lines.append("")
    lines.append("## Sleeve Scoreboard")
    lines.append("")

    for row in display_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('profile', '')}: "
            f"status={row.get('data_status', '')}, "
            f"day={row.get('day_utc', '') or 'n/a'}, "
            f"day_change={float(row.get('change_vs_previous_day', 0.0) or 0.0):.6f}, "
            f"end_net={float(row.get('ending_net_pnl_total', 0.0) or 0.0):.6f}, "
            f"win_rate={_format_win_rate(row.get('win_rate'))}, "
            f"risk_metric={str(row.get('risk_adjusted_metric') or 'n/a')}:"
            f"{_format_optional_float(row.get('risk_adjusted_metric_value'))}, "
            f"wins/losses/flats="
            f"{int(row.get('winning_strategy_count', 0) or 0)}/"
            f"{int(row.get('losing_strategy_count', 0) or 0)}/"
            f"{int(row.get('flat_strategy_count', 0) or 0)}, "
            f"best={_format_strategy_pnl_brief(row.get('top_winning_strategies', []), positive=True)}, "
            f"worst={_format_strategy_pnl_brief(row.get('top_losing_strategies', []), positive=False)}, "
            f"loss_causes={', '.join(str(item.get('cause') or '') for item in (row.get('top_loss_causes') or [])[:3]) or 'n/a'}, "
            f"feature_telemetry={str(row.get('advanced_feature_summary') or 'n/a')}, "
            f"mean_slip_gap_bps={float(((row.get('tca_summary') or {}).get('mean_slippage_gap_bps', 0.0) or 0.0)):.4f}, "
            f"realized={float(row.get('ending_realized_pnl_total', 0.0) or 0.0):.6f}, "
            f"unrealized={float(row.get('ending_unrealized_pnl_total', 0.0) or 0.0):.6f}, "
            f"executions={int(row.get('executions', 0) or 0)}, "
            f"activity_note={str(row.get('activity_note') or 'n/a')}"
        )

    lines.extend(
        [
        "",
        "## Daily Series",
        "",
        ]
    )

    for row in week.get("daily_series", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('day_utc', '')}: end_net={float(row.get('ending_net_pnl_total', 0.0) or 0.0):.6f}, "
            f"change={float(row.get('change_vs_previous_day', 0.0) or 0.0):.6f}"
        )

    return "\n".join(lines).strip() + "\n"


def _path_uri(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve().as_uri()


def render_paper_performance_html(payload: dict[str, Any], *, source_path: Path, generated_utc: str) -> str:
    day = payload.get("day") if isinstance(payload.get("day"), dict) else {}
    week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
    sleeve_latest = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    active_profiles = payload.get("active_paper_profiles_today") if isinstance(payload.get("active_paper_profiles_today"), list) else []
    display_rows = _sleeve_display_rows(sleeve_latest)

    chart_specs = [
        ("Daily Performance", graphs.get("daily_png", "")),
        ("Weekly Performance", graphs.get("weekly_png", "")),
        ("Monthly Performance", graphs.get("monthly_png", "")),
        ("Quarterly Performance", graphs.get("quarterly_png", "")),
        ("Sleeve Scoreboard", graphs.get("sleeves_png", "")),
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

    sleeve_rows = []
    for row in display_rows:
        if not isinstance(row, dict):
            continue
        sleeve_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('profile', '')))}</td>"
            f"<td>{html.escape(str(row.get('data_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('day_utc', '') or 'n/a'))}</td>"
            f"<td>{float(row.get('change_vs_previous_day', 0.0) or 0.0):.6f}</td>"
            f"<td>{int(row.get('executions', 0) or 0)}</td>"
            f"<td>{html.escape(_format_win_rate(row.get('win_rate')))}</td>"
            f"<td>{html.escape(str(row.get('risk_adjusted_metric') or 'n/a'))}</td>"
            f"<td>{html.escape(_format_optional_float(row.get('risk_adjusted_metric_value')))}</td>"
            f"<td>"
            f"{int(row.get('winning_strategy_count', 0) or 0)}/"
            f"{int(row.get('losing_strategy_count', 0) or 0)}/"
            f"{int(row.get('flat_strategy_count', 0) or 0)}"
            f"</td>"
            f"<td>{html.escape(_format_strategy_pnl_brief(row.get('top_winning_strategies', []), positive=True))}</td>"
            f"<td>{html.escape(_format_strategy_pnl_brief(row.get('top_losing_strategies', []), positive=False))}</td>"
            f"<td>{html.escape(', '.join(str(item.get('cause') or '') for item in (row.get('top_loss_causes') or [])[:3]) or 'n/a')}</td>"
            f"<td>{html.escape(str(row.get('advanced_feature_summary') or 'n/a'))}</td>"
            f"<td>{float(row.get('ending_realized_pnl_total', 0.0) or 0.0):.6f}</td>"
            f"<td>{float(row.get('ending_unrealized_pnl_total', 0.0) or 0.0):.6f}</td>"
            f"<td>{float(row.get('ending_net_pnl_total', 0.0) or 0.0):.6f}</td>"
            "</tr>"
        )

    active_rows = []
    for item in active_profiles:
        if not isinstance(item, dict):
            continue
        active_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('profile', '')))}</td>"
            f"<td>{html.escape(str(item.get('state', '')))}</td>"
            f"<td>{html.escape(str(item.get('broker', '')))}</td>"
            f"<td>{html.escape(str(item.get('timestamp_utc', '')))}</td>"
            f"<td>{int(item.get('symbols_total', 0) or 0)}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Paper Performance Report</title>
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
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .stat-block {{ background: #f7f1e7; border: 1px solid #eadfce; border-radius: 14px; padding: 14px 16px; }}
    .stat-block p {{ margin: 6px 0 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 12px; border: 1px solid #eadfce; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Paper Performance Report</h1>
      <p class="meta">PDF generated {html.escape(generated_utc)}</p>
      <p class="path">Source: {html.escape(str(source_path))}</p>
    </section>
    <section class="section">
      <div class="stats">
        <div class="stat-block">
          <h2>End Of Day</h2>
          <p>day_utc: {html.escape(str(day.get('day_utc', '')))}</p>
          <p>executions: {int(day.get('executions', 0) or 0)}</p>
          <p>ending_net_pnl_total: {float(day.get('ending_net_pnl_total', 0.0) or 0.0):.6f}</p>
          <p>change_vs_previous_day: {float(day.get('change_vs_previous_day', 0.0) or 0.0):.6f}</p>
        </div>
        <div class="stat-block">
          <h2>Week</h2>
          <p>week_start_day_utc: {html.escape(str(week.get('week_start_day_utc', '')))}</p>
          <p>week_end_day_utc: {html.escape(str(week.get('week_end_day_utc', '')))}</p>
          <p>week_to_date_change: {float(week.get('week_to_date_change', 0.0) or 0.0):.6f}</p>
          <p>rolling_{int(week.get('rolling_change_days', 7) or 7)}d_change: {float(week.get('rolling_change', 0.0) or 0.0):.6f}</p>
        </div>
      </div>
    </section>
    <div class="chart-grid">
      {''.join(chart_cards)}
    </div>
    <section class="section">
      <h2>Active Paper Lanes Today</h2>
      <p class="meta">active_paper_profile_count_today: {int(payload.get('active_paper_profile_count_today', 0) or 0)}</p>
      <table>
        <thead>
          <tr>
            <th>Lane</th>
            <th>State</th>
            <th>Broker</th>
            <th>Heartbeat</th>
            <th>Symbols</th>
          </tr>
        </thead>
        <tbody>
          {''.join(active_rows)}
        </tbody>
      </table>
    </section>
    <section class="section">
      <h2>Sleeve Scoreboard</h2>
      <table>
        <thead>
          <tr>
            <th>Sleeve</th>
            <th>Status</th>
            <th>Data Day</th>
            <th>Day Change</th>
            <th>Executions</th>
            <th>Win Rate</th>
            <th>Risk Metric</th>
            <th>Risk Value</th>
            <th>W/L/F</th>
            <th>Top Winners</th>
            <th>Top Losers</th>
            <th>Loss Causes</th>
            <th>Feature Telemetry</th>
            <th>Realized</th>
            <th>Unrealized</th>
            <th>Ending Net</th>
          </tr>
        </thead>
        <tbody>
          {''.join(sleeve_rows)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper trading performance snapshot and week-to-date report.")
    ap.add_argument("--day", default=_utc_now().strftime("%Y%m%d"))
    ap.add_argument("--week-days", type=int, default=7)
    ap.add_argument("--out-file", default=str(DEFAULT_JSON_PATH))
    ap.add_argument("--md-out-file", default=str(DEFAULT_MD_PATH))
    ap.add_argument("--html-out-file", default=str(DEFAULT_HTML_PATH))
    ap.add_argument("--pdf-out-file", default=str(DEFAULT_PDF_PATH))
    ap.add_argument("--daily-chart-file", default=str(DEFAULT_DAILY_CHART_PATH))
    ap.add_argument("--weekly-chart-file", default=str(DEFAULT_WEEKLY_CHART_PATH))
    ap.add_argument("--monthly-chart-file", default=str(DEFAULT_MONTHLY_CHART_PATH))
    ap.add_argument("--quarterly-chart-file", default=str(DEFAULT_QUARTERLY_CHART_PATH))
    ap.add_argument("--sleeves-chart-file", default=str(DEFAULT_SLEEVES_CHART_PATH))
    ap.add_argument("--allow-gui-pdf-renderer", action=argparse.BooleanOptionalAction, default=_default_allow_gui_pdf_renderer())
    ap.add_argument("--json-only", action="store_true", help="Write the JSON snapshot only and skip charts/markdown/html/pdf artifacts.")
    ap.add_argument(
        "--sync-profitability-control",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PAPER_PERFORMANCE_SYNC_PROFITABILITY", "1").strip() == "1",
        help="Publish paper performance and its exact-hash profitability control as one coordinated generation.",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    payload = build_paper_performance_report(PROJECT_ROOT, day=str(args.day), week_days=max(int(args.week_days), 1))

    out_path = Path(args.out_file)
    md_path = Path(args.md_out_file)
    html_path = Path(args.html_out_file)
    pdf_path = Path(args.pdf_out_file)
    daily_chart_path = Path(args.daily_chart_file)
    weekly_chart_path = Path(args.weekly_chart_file)
    monthly_chart_path = Path(args.monthly_chart_file)
    quarterly_chart_path = Path(args.quarterly_chart_file)
    sleeves_chart_path = Path(args.sleeves_chart_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.json_only:
        payload["graphs"] = {
            "mode": "json_only",
            "daily_png": "",
            "weekly_png": "",
            "monthly_png": "",
            "quarterly_png": "",
            "sleeves_png": "",
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
        payload["graphs"] = render_paper_performance_graphs(
            payload,
            daily_chart_path=daily_chart_path,
            weekly_chart_path=weekly_chart_path,
            monthly_chart_path=monthly_chart_path,
            quarterly_chart_path=quarterly_chart_path,
            sleeves_chart_path=sleeves_chart_path,
        )
        generated_utc = str(payload.get("timestamp_utc") or _utc_now().isoformat())
        md_text = render_paper_performance_markdown(payload)
        html_text = render_paper_performance_html(payload, source_path=out_path, generated_utc=generated_utc)
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
    canonical_performance_path = PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"
    sync_result: dict[str, Any] = {
        "ok": True,
        "attempted": False,
        "reason": "noncanonical_performance_output",
    }
    with paper_profitability_generation_lock(
        PROJECT_ROOT,
        timeout_seconds=float(os.getenv("PAPER_PROFITABILITY_GENERATION_LOCK_TIMEOUT_SECONDS", "120") or 120.0),
    ):
        existing = _load_json(out_path)
        existing_timestamp = _parse_ts(existing.get("timestamp_utc"))
        payload_timestamp = _parse_ts(payload.get("timestamp_utc"))
        if existing_timestamp is not None and payload_timestamp is not None and existing_timestamp > payload_timestamp:
            sync_result = {
                "ok": True,
                "attempted": False,
                "reason": "newer_generation_already_published",
            }
        else:
            _atomic_write_json(out_path, payload)
            try:
                canonical_output = out_path.resolve(strict=False) == canonical_performance_path.resolve(strict=False)
            except Exception:
                canonical_output = str(out_path) == str(canonical_performance_path)
            if canonical_output:
                sync_result = _sync_profitability_control(
                    PROJECT_ROOT,
                    out_path,
                    enabled=bool(args.sync_profitability_control),
                )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        day_summary = payload.get("day", {}) if isinstance(payload.get("day"), dict) else {}
        week_summary = payload.get("week", {}) if isinstance(payload.get("week"), dict) else {}
        print(
            "paper_performance "
            f"day={day_summary.get('day_utc', '')} "
            f"eod_net={float(day_summary.get('ending_net_pnl_total', 0.0) or 0.0):.4f} "
            f"day_change={float(day_summary.get('change_vs_previous_day', 0.0) or 0.0):.4f} "
            f"wtd_change={float(week_summary.get('week_to_date_change', 0.0) or 0.0):.4f}"
        )
    if not payload.get("ok"):
        return 2
    return 0 if bool(sync_result.get("ok", False)) else 3


if __name__ == "__main__":
    raise SystemExit(main())
