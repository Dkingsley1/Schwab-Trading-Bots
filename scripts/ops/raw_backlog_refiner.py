#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "raw_backlog_refiner_latest.json"
RAW_GREEN_TOTAL = 15_000
RAW_GREEN_CORE = 10_000
RAW_GREEN_DEFERRED = 5_000
RAW_GREEN_AGE_SECONDS = 15 * 60
RAW_WARN_TOTAL = 50_000
RAW_WARN_AGE_SECONDS = 2 * 60 * 60
OVERLAY_WARN_TOTAL = 100_000
SPARSE_PENDING_BYTES_WARN = 64 * 1024 * 1024
TOP_FILE_WARN_LINES = 5_000
TOP_FILE_WARN_AGE_SECONDS = 30 * 60


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _grade(score: float) -> str:
    if score >= 99:
        return "A++"
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 60:
        return "D"
    return "F"


def _status_from_score(score: float) -> str:
    if score >= 90:
        return "ready"
    if score >= 75:
        return "needs_work"
    return "blocked"


def _command(*parts: str) -> list[str]:
    return ["./scripts/ops/opsctl.sh", *parts]


def _parse_json_output(stdout: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_command(
    cmd: list[str],
    *,
    project_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    payload = _parse_json_output(stdout)
    command_ok = bool(not timed_out and (rc == 0 or (rc == 2 and bool(payload))))
    return {
        "cmd": list(cmd),
        "rc": rc,
        "ok": command_ok,
        "surface_ok": payload.get("ok") if isinstance(payload.get("ok"), bool) else None,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "payload_summary": _payload_summary(payload),
        "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
    }


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "ok",
        "overall_status",
        "recommended_now",
        "apply_executed",
        "writer_busy",
        "pressure_index",
        "queue_depth",
    )
    summary = {key: payload.get(key) for key in keys if key in payload}
    backpressure = payload.get("backpressure")
    if isinstance(backpressure, dict):
        summary["backpressure"] = {
            "core": _safe_int(backpressure.get("core_pending_lines"), 0),
            "deferred": _safe_int(backpressure.get("deferred_pending_lines"), 0),
            "cold": _safe_int(backpressure.get("cold_pending_lines"), 0),
            "support": _safe_int(backpressure.get("support_pending_lines"), 0),
            "total": _safe_int(backpressure.get("total_pending_lines"), 0),
        }
    return summary


def _storage_raw_backpressure(storage: dict[str, Any], backpressure: dict[str, Any]) -> dict[str, Any]:
    storage_backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    raw_live = storage_backpressure.get("raw_live") if isinstance(storage_backpressure.get("raw_live"), dict) else {}
    if raw_live:
        return dict(raw_live)
    return {
        "core_pending_lines": _safe_int(backpressure.get("pending_lines"), 0),
        "deferred_pending_lines": _safe_int(backpressure.get("pending_lines_deferred"), 0),
        "cold_pending_lines": _safe_int(backpressure.get("pending_lines_cold"), 0),
        "support_pending_lines": _safe_int(backpressure.get("pending_lines_support_telemetry"), 0),
        "stale_stage_pending_lines": _safe_int(backpressure.get("pending_lines_stale_stage"), 0),
        "total_pending_lines": _safe_int(backpressure.get("pending_lines_total"), 0),
        "oldest_pending_age_seconds": _safe_float(backpressure.get("oldest_pending_age_seconds_total"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)),
        "line_estimation": backpressure.get("line_estimation") if isinstance(backpressure.get("line_estimation"), dict) else {},
    }


def _effective_backpressure(storage: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    storage_backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    return {
        "core_pending_lines": _safe_int(storage_backpressure.get("core_pending_lines"), _safe_int(raw.get("core_pending_lines"), 0)),
        "deferred_pending_lines": _safe_int(storage_backpressure.get("deferred_pending_lines"), _safe_int(raw.get("deferred_pending_lines"), 0)),
        "cold_pending_lines": _safe_int(storage_backpressure.get("cold_pending_lines"), _safe_int(raw.get("cold_pending_lines"), 0)),
        "support_pending_lines": _safe_int(storage_backpressure.get("support_pending_lines"), _safe_int(raw.get("support_pending_lines"), 0)),
        "total_pending_lines": _safe_int(storage_backpressure.get("total_pending_lines"), _safe_int(raw.get("total_pending_lines"), 0)),
        "oldest_pending_age_seconds": _safe_float(storage_backpressure.get("oldest_pending_age_seconds"), _safe_float(raw.get("oldest_pending_age_seconds"), 0.0)),
        "overlay_adjusted": bool(storage_backpressure.get("overlay_adjusted", False)),
    }


def _row_lane(key: str) -> str:
    if key == "top_pending_files":
        return "core"
    if key == "top_deferred_pending_files":
        return "deferred"
    if key == "top_cold_pending_files":
        return "cold"
    if key == "top_support_telemetry_pending_files":
        return "support"
    if key == "top_stale_stage_pending_files":
        return "stale_stage"
    return "unknown"


def _hot_file_rows(backpressure: dict[str, Any], storage: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_source: dict[str, dict[str, Any]] = {}
    for key in (
        "top_pending_files",
        "top_deferred_pending_files",
        "top_cold_pending_files",
        "top_support_telemetry_pending_files",
        "top_stale_stage_pending_files",
    ):
        rows = backpressure.get(key) if isinstance(backpressure.get(key), list) else []
        for raw in rows[:12]:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            if not source_rel or pending_lines <= 0:
                continue
            age_seconds = max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0)
            pending_bytes = max(_safe_int(raw.get("estimated_pending_bytes"), 0), 0)
            row = {
                "source_rel": source_rel,
                "lane": _row_lane(key),
                "pending_lines": pending_lines,
                "oldest_pending_age_seconds": round(age_seconds, 3),
                "estimated_pending_bytes": pending_bytes,
                "sparse_large_line": bool(raw.get("sparse_large_line", False)),
                "line_estimate_method": str(raw.get("line_estimate_method") or ""),
                "candidate_action": _candidate_action(source_rel, _row_lane(key), pending_lines, age_seconds, pending_bytes, bool(raw.get("sparse_large_line", False))),
            }
            current = rows_by_source.get(source_rel)
            if current is None or pending_lines > _safe_int(current.get("pending_lines"), 0):
                rows_by_source[source_rel] = row
            elif age_seconds > _safe_float(current.get("oldest_pending_age_seconds"), 0.0):
                current["oldest_pending_age_seconds"] = round(age_seconds, 3)
                if current.get("candidate_action") == "drain_now":
                    current["candidate_action"] = row["candidate_action"]

    locator = storage.get("stale_pending_locator") if isinstance(storage.get("stale_pending_locator"), dict) else {}
    overlay_rows = locator.get("top_pending_sources") if isinstance(locator.get("top_pending_sources"), list) else []
    for raw in overlay_rows[:8]:
        if not isinstance(raw, dict):
            continue
        source_rel = str(raw.get("source_rel") or "").strip()
        pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
        if not source_rel or pending_lines <= 0:
            continue
        current = rows_by_source.get(source_rel)
        age_seconds = max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0)
        overlay_row = {
            "source_rel": source_rel,
            "lane": str(raw.get("pressure_lane") or "overlay"),
            "shard": str(raw.get("shard") or ""),
            "pending_lines": pending_lines,
            "oldest_pending_age_seconds": round(age_seconds, 3),
            "estimated_pending_bytes": 0,
            "sparse_large_line": False,
            "line_estimate_method": "sql_overlay",
            "candidate_action": _candidate_action(source_rel, str(raw.get("pressure_lane") or "overlay"), pending_lines, age_seconds, 0, False),
            "storage_overlay_source": True,
        }
        if current is None or pending_lines > _safe_int(current.get("pending_lines"), 0):
            rows_by_source[source_rel] = overlay_row
        else:
            current["storage_overlay_source"] = True
            current["shard"] = str(raw.get("shard") or current.get("shard") or "")

    rows = list(rows_by_source.values())
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("pending_lines"), 0),
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
            _safe_int(row.get("estimated_pending_bytes"), 0),
        ),
        reverse=True,
    )
    return rows[:20]


def _candidate_action(
    source_rel: str,
    lane: str,
    pending_lines: int,
    age_seconds: float,
    pending_bytes: int,
    sparse: bool,
) -> str:
    if lane == "stale_stage" or source_rel.startswith("data/stale_stage/"):
        return "stage_or_reap_stale_artifact"
    if sparse or pending_bytes >= SPARSE_PENDING_BYTES_WARN:
        return "byte_window_drain"
    if lane in {"support", "support_telemetry"}:
        return "support_isolated_drain"
    if age_seconds >= TOP_FILE_WARN_AGE_SECONDS and pending_lines >= 100:
        return "stale_tail_catchup"
    if pending_lines >= TOP_FILE_WARN_LINES:
        return "focused_file_drain"
    return "drain_now"


def _score_measure(raw: dict[str, Any]) -> tuple[float, list[str]]:
    total = _safe_int(raw.get("total_pending_lines"), 0)
    core = _safe_int(raw.get("core_pending_lines"), 0)
    deferred = _safe_int(raw.get("deferred_pending_lines"), 0)
    cold = _safe_int(raw.get("cold_pending_lines"), 0)
    stale_stage = _safe_int(raw.get("stale_stage_pending_lines"), 0)
    age = _safe_float(raw.get("oldest_pending_age_seconds"), 0.0)
    penalties = [
        min(max(total - RAW_GREEN_TOTAL, 0) / 2000.0, 25.0),
        min(max(core - RAW_GREEN_CORE, 0) / 1500.0, 20.0),
        min(max(deferred - RAW_GREEN_DEFERRED, 0) / 1000.0, 15.0),
        min(max(cold, 0) / 1000.0, 10.0),
        min(max(stale_stage, 0) / 500.0, 10.0),
        min(max(age - RAW_GREEN_AGE_SECONDS, 0) / 900.0, 20.0),
    ]
    blockers: list[str] = []
    if total > RAW_GREEN_TOTAL:
        blockers.append("raw_total_above_green")
    if core > RAW_GREEN_CORE:
        blockers.append("raw_core_above_green")
    if deferred > RAW_GREEN_DEFERRED:
        blockers.append("raw_deferred_above_green")
    if age > RAW_GREEN_AGE_SECONDS:
        blockers.append("raw_oldest_age_above_green")
    if cold > 0:
        blockers.append("raw_cold_pending")
    if stale_stage > 0:
        blockers.append("raw_stale_stage_pending")
    return max(100.0 - sum(penalties), 0.0), blockers


def _section(
    *,
    section_id: str,
    title: str,
    score: float,
    metrics: dict[str, Any],
    blockers: list[str],
    next_commands: list[list[str]],
    expansions: list[str],
    applied_steps: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "section_id": section_id,
        "title": title,
        "status": _status_from_score(score),
        "grade": _grade(score),
        "score": round(float(score), 3),
        "metrics": metrics,
        "blockers": blockers,
        "next_commands": next_commands,
        "expansions": expansions,
        "applied_steps": applied_steps or [],
    }


def _build_sections(
    *,
    raw: dict[str, Any],
    effective: dict[str, Any],
    storage: dict[str, Any],
    backpressure: dict[str, Any],
    priority_queue: dict[str, Any],
    external_drain: dict[str, Any],
    writer: dict[str, Any],
    pressure_relief: dict[str, Any],
    retention: dict[str, Any],
    sweeper: dict[str, Any],
    reaper: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[list[str]]]:
    raw_score, raw_blockers = _score_measure(raw)
    hot_rows = _hot_file_rows(backpressure, storage)
    sparse = raw.get("line_estimation") if isinstance(raw.get("line_estimation"), dict) else {}
    sparse_pending_bytes = _safe_int(sparse.get("sparse_large_line_pending_bytes"), 0)
    sparse_detected = bool(sparse.get("sparse_large_line_active", False))
    sparse_active = bool(sparse_detected and sparse_pending_bytes >= SPARSE_PENDING_BYTES_WARN)
    overlay_gap = max(_safe_int(effective.get("total_pending_lines"), 0) - _safe_int(raw.get("total_pending_lines"), 0), 0)
    queue_depth = _safe_int(priority_queue.get("queue_depth"), 0)
    raw_total = _safe_int(raw.get("total_pending_lines"), 0)
    raw_oldest_age = _safe_float(raw.get("oldest_pending_age_seconds"), 0.0)
    raw_green_envelope = bool(
        raw_total <= RAW_GREEN_TOTAL
        and raw_oldest_age <= RAW_GREEN_AGE_SECONDS
        and overlay_gap <= OVERLAY_WARN_TOTAL
    )
    sparse_controlled_watch = bool(
        sparse_active
        and raw_total <= RAW_GREEN_TOTAL
        and _safe_int(raw.get("core_pending_lines"), 0) <= RAW_GREEN_CORE
        and raw_oldest_age <= RAW_GREEN_AGE_SECONDS
        and overlay_gap <= 0
    )
    stale_hot_estimate_rows = [
        row
        for row in hot_rows
        if raw_green_envelope
        and _safe_int(row.get("pending_lines"), 0) >= TOP_FILE_WARN_LINES
        and _safe_int(row.get("pending_lines"), 0) > max(raw_total, 0)
    ]
    stale_hot_estimate_sources = {str(row.get("source_rel") or "") for row in stale_hot_estimate_rows}
    hot_scoring_rows = [
        row
        for row in hot_rows
        if str(row.get("source_rel") or "") not in stale_hot_estimate_sources
    ]

    hot_penalty = 0.0
    if hot_scoring_rows:
        hot_penalty += min(_safe_int(hot_scoring_rows[0].get("pending_lines"), 0) / 25_000.0, 25.0)
        hot_penalty += min(_safe_float(hot_scoring_rows[0].get("oldest_pending_age_seconds"), 0.0) / 3600.0, 20.0)
    if sparse_active and not sparse_controlled_watch:
        hot_penalty += min(max(sparse_pending_bytes, 0) / float(128 * 1024 * 1024) * 15.0, 15.0)
    if overlay_gap > OVERLAY_WARN_TOTAL:
        hot_penalty += min(overlay_gap / 250_000.0, 15.0)
    hot_score = max(100.0 - hot_penalty, 0.0)
    hot_blockers = []
    if hot_scoring_rows and _safe_int(hot_scoring_rows[0].get("pending_lines"), 0) >= TOP_FILE_WARN_LINES:
        hot_blockers.append("dominant_hot_file")
    if hot_scoring_rows and _safe_float(hot_scoring_rows[0].get("oldest_pending_age_seconds"), 0.0) >= TOP_FILE_WARN_AGE_SECONDS:
        hot_blockers.append("stale_hot_file")
    if sparse_active and not sparse_controlled_watch:
        hot_blockers.append("sparse_large_jsonl_active")
    if overlay_gap > OVERLAY_WARN_TOTAL:
        hot_blockers.append("sql_overlay_gap")

    writer_state = writer.get("writer_state_before") if isinstance(writer.get("writer_state_before"), dict) else {}
    drain_overrides = external_drain.get("drain_overrides") if isinstance(external_drain.get("drain_overrides"), dict) else {}
    preferred_shards = drain_overrides.get("preferred_shards") if isinstance(drain_overrides.get("preferred_shards"), list) else []
    writer_active = bool(writer_state.get("active", False))
    writer_stale = bool((writer.get("summary") or {}).get("stale_writer_detected", False)) if isinstance(writer.get("summary"), dict) else False
    drain_score = 94.0
    drain_blockers: list[str] = []
    if not preferred_shards and (_safe_int(raw.get("total_pending_lines"), 0) > RAW_GREEN_TOTAL or overlay_gap > OVERLAY_WARN_TOTAL):
        drain_score -= 20.0
        drain_blockers.append("no_focused_drain_plan")
    if writer_stale:
        drain_score -= 30.0
        drain_blockers.append("writer_stale")
    if writer_active:
        drain_score -= 4.0
    if overlay_gap > OVERLAY_WARN_TOTAL:
        drain_score -= min(overlay_gap / 500_000.0, 20.0)
        drain_blockers.append("overlay_catchup_needed")
    drain_score = max(drain_score, 0.0)

    shedding = storage.get("writer_shedding") if isinstance(storage.get("writer_shedding"), dict) else {}
    pressure_active = bool(shedding.get("active", False)) or bool((pressure_relief.get("storage_pressure") or {}).get("active", False)) if isinstance(pressure_relief.get("storage_pressure"), dict) else bool(shedding.get("active", False))
    intake_score = 96.0
    intake_blockers: list[str] = []
    if _safe_int(raw.get("total_pending_lines"), 0) > RAW_GREEN_TOTAL and not pressure_active:
        intake_score -= 22.0
        intake_blockers.append("intake_relief_not_active")
    if _safe_int(raw.get("total_pending_lines"), 0) > RAW_WARN_TOTAL:
        intake_score -= 14.0
        intake_blockers.append("raw_intake_outpaces_drain")
    if sparse_active and not sparse_controlled_watch:
        intake_score -= 6.0
    if queue_depth > 50:
        intake_score -= min(queue_depth / 50.0, 12.0)
        intake_blockers.append("priority_queue_depth_high")
    intake_score = max(intake_score, 0.0)

    retention_deleted = _safe_int(retention.get("deleted"), 0)
    retention_debt_gb = _safe_float(((storage.get("storage") or {}).get("retention_debt_gb") if isinstance(storage.get("storage"), dict) else 0.0), 0.0)
    sweeper_candidates = _safe_int(((sweeper.get("summary") or {}).get("candidate_files") if isinstance(sweeper.get("summary"), dict) else sweeper.get("candidate_files")), 0)
    reaper_candidates = _safe_int(((reaper.get("summary") or {}).get("candidate_files") if isinstance(reaper.get("summary"), dict) else reaper.get("candidate_files")), 0)
    stale_stage_pressure = bool(
        _safe_int(raw.get("stale_stage_pending_lines"), 0) > 0
        or _safe_int(effective.get("stale_stage_pending_lines"), 0) > 0
    )
    cleanup_score = 96.0
    cleanup_blockers: list[str] = []
    if retention_debt_gb > 0:
        cleanup_score -= min(retention_debt_gb * 5.0, 25.0)
        cleanup_blockers.append("retention_debt")
    if sparse_active and not sparse_controlled_watch:
        cleanup_score -= 10.0
        cleanup_blockers.append("sparse_huge_jsonl_cleanup_needed")
    if stale_stage_pressure and sweeper_candidates > 0:
        cleanup_score -= min(sweeper_candidates / 50.0, 10.0)
        cleanup_blockers.append("stale_stage_candidates")
    if stale_stage_pressure and reaper_candidates > 0:
        cleanup_score -= min(reaper_candidates / 50.0, 10.0)
        cleanup_blockers.append("aged_stale_stage_candidates")
    cleanup_score = max(cleanup_score, 0.0)

    sections = {
        "measure_raw_backlog": _section(
            section_id="measure_raw_backlog",
            title="Measure raw backlog",
            score=raw_score,
            metrics={
                "raw_live": raw,
                "effective_backpressure": effective,
                "overlay_gap_lines": overlay_gap,
                "queue_depth": queue_depth,
            },
            blockers=raw_blockers,
            next_commands=[
                _command("ingestion-storage-control", "--json"),
                _command("ingestion-priority-queue", "--json"),
            ],
            expansions=[
                "refresh raw-live and overlay snapshots together so the operator sees file truth and SQL truth separately",
                "grade raw total, raw core, deferred/cold/support, oldest age, sparse pending bytes, and overlay gap independently",
                "write the raw/effective split into this refiner artifact for later before-after comparisons",
            ],
        ),
        "find_raw_hot_files": _section(
            section_id="find_raw_hot_files",
            title="Find raw hot files",
            score=hot_score,
            metrics={
                "hot_file_count": len(hot_rows),
                "top_hot_files": hot_rows[:10],
                "sparse_large_line": sparse,
                "hot_file_map_control": {
                    "raw_green_envelope": raw_green_envelope,
                    "sparse_controlled_watch": sparse_controlled_watch,
                    "stale_hot_estimate_count": len(stale_hot_estimate_rows),
                    "stale_hot_estimate_sources": sorted(stale_hot_estimate_sources)[:10],
                    "scored_hot_file_count": len(hot_scoring_rows),
                    "policy": "when raw/effective totals are green, stale per-file estimates larger than total raw backlog stay visible but do not create blockers",
                },
            },
            blockers=hot_blockers,
            next_commands=[
                _command("ingestion-priority-queue", "--json"),
                _command("external-backlog-drain", "--json"),
            ],
            expansions=[
                "merge raw top files, deferred/cold/support files, and SQL overlay leaders into one ordered map",
                "classify every hot source as focused drain, stale-tail catch-up, byte-window drain, support-isolated drain, or stale-stage cleanup",
                "treat sparse huge JSONL pending bytes as a first-class pressure signal instead of only counting lines",
            ],
        ),
        "drain_refine_raw_files": _section(
            section_id="drain_refine_raw_files",
            title="Drain and refine raw files",
            score=drain_score,
            metrics={
                "writer_active": writer_active,
                "writer_stale": writer_stale,
                "writer_step": str(writer_state.get("current_step") or ""),
                "writer_shards": [
                    _safe_int(writer_state.get("completed_shard_count"), 0),
                    _safe_int(writer_state.get("planned_shard_count"), 0),
                ],
                "writer_rows_this_cycle": _safe_int(writer_state.get("merged_rows_this_cycle"), 0),
                "preferred_shards": preferred_shards,
                "drain_focus": {
                    "governance": drain_overrides.get("governance_path_focus") or [],
                    "trading": drain_overrides.get("trading_path_focus") or [],
                    "crypto_trading": drain_overrides.get("crypto_trading_path_focus") or [],
                    "risk_support": drain_overrides.get("risk_support_path_focus") or [],
                    "explanations": drain_overrides.get("explanations_path_focus") or [],
                    "crypto_explanations": drain_overrides.get("crypto_explanations_path_focus") or [],
                },
            },
            blockers=drain_blockers,
            next_commands=[
                _command("external-backlog-drain", "--apply", "--follow-through", "--wait-timeout-seconds", "900", "--json"),
                _command("writer-cycle-coordinator", "--json"),
            ],
            expansions=[
                "prefer pinned shard/file drains over broad sweeps whenever raw or overlay concentration is high",
                "handoff to the existing single-writer service instead of launching competing writers",
                "keep support/risk/explanation traffic shard-isolated so it cannot crowd hot trading rows",
            ],
        ),
        "reduce_intake_while_draining": _section(
            section_id="reduce_intake_while_draining",
            title="Reduce intake while draining",
            score=intake_score,
            metrics={
                "writer_shedding": shedding,
                "pressure_relief_status": pressure_relief.get("overall_status") or pressure_relief.get("status") or "",
                "queue_depth": queue_depth,
                "raw_total_pending_lines": _safe_int(raw.get("total_pending_lines"), 0),
            },
            blockers=intake_blockers,
            next_commands=[
                _command("pressure-relief", "--apply", "--json"),
                _command("ingestion-storage-governor", "--json"),
            ],
            expansions=[
                "hold trainings/reports/heavy collectors while raw age or overlay catch-up is above target",
                "suppress verbose decision/explanation logs during backlog relief windows",
                "use collector duty cycling and support telemetry shedding before touching live/paper decision safety",
            ],
        ),
        "cleanup_stale_sparse_old_files": _section(
            section_id="cleanup_stale_sparse_old_files",
            title="Clean stale, sparse, and old files",
            score=cleanup_score,
            metrics={
                "retention_debt_gb": round(retention_debt_gb, 3),
                "retention_deleted": retention_deleted,
                "sweeper_candidate_files": sweeper_candidates,
                "reaper_candidate_files": reaper_candidates,
                "sparse_large_line": sparse,
                "sparse_controlled_watch": sparse_controlled_watch,
            },
            blockers=cleanup_blockers,
            next_commands=[
                _command("stale-sweeper", "--json"),
                _command("data-retention", "--apply", "--no-stale-purge", "--skip-sqlite-vacuum", "--json"),
            ],
            expansions=[
                "stage stale artifacts first, then reaper-purge only when explicitly allowed",
                "keep retention and SQLite vacuum separate so cleanup cannot block the active writer unexpectedly",
                "feed sparse pending-byte signals back into drain caps before expanding bot data intake",
            ],
        ),
    }

    top_actions: list[str] = []
    if raw_blockers:
        top_actions.append("refresh raw backlog and keep catch-up active until raw total, core, deferred, and oldest age are all green")
    if hot_blockers:
        top_actions.append("use the hot-file map to pin the writer to dominant stale or sparse sources before broad sweeping")
    if drain_blockers:
        top_actions.append("run the focused external backlog drain and let the single writer finish before starting another writer path")
    if intake_blockers:
        top_actions.append("apply pressure relief so intake slows while the writer catches up")
    if cleanup_blockers:
        top_actions.append("stage stale artifacts and run safe retention after the active drain pass")
    if overlay_gap > OVERLAY_WARN_TOTAL:
        top_actions.append("treat SQL overlay catch-up as separate from raw file backlog; raw may be green while overlay is still draining")
    if not top_actions:
        top_actions.append("keep the refiner in monitor mode; raw backlog is inside the steady-state envelope")

    recommended_commands: list[list[str]] = []
    for section in sections.values():
        for command in section["next_commands"]:
            if command not in recommended_commands:
                recommended_commands.append(command)

    return sections, hot_rows, top_actions[:8], recommended_commands[:10]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    skip_drain: bool = False,
    skip_intake: bool = False,
    skip_cleanup: bool = False,
    allow_stale_reaper: bool = False,
    wait_timeout_seconds: float = 900.0,
    command_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    steps: dict[str, Any] = {}

    if apply:
        steps["measure_raw_backlog"] = _run_command(
            [sys.executable, str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )
        steps["priority_queue_refresh"] = _run_command(
            [sys.executable, str(project_root / "scripts" / "ops" / "ingestion_priority_queue.py"), "--json"],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )
        steps["storage_control_refresh"] = _run_command(
            [sys.executable, str(project_root / "scripts" / "ops" / "ingestion_storage_control.py"), "--json"],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )

    backpressure = _load_json(health_root / "ingestion_backpressure_latest.json")
    storage = _load_json(health_root / "ingestion_storage_control_latest.json")
    priority_queue = _load_json(health_root / "ingestion_priority_queue_latest.json")
    external_drain = _load_json(health_root / "external_backlog_drain_latest.json")
    writer = _load_json(health_root / "writer_cycle_coordinator_latest.json")
    pressure_relief = _load_json(health_root / "pressure_relief_control_latest.json")
    retention = _load_json(health_root / "data_retention_latest.json")
    sweeper = _load_json(health_root / "stale_artifact_sweeper_bot_latest.json")
    reaper = _load_json(health_root / "stale_artifact_reaper_bot_latest.json")
    raw = _storage_raw_backpressure(storage, backpressure)
    effective = _effective_backpressure(storage, raw)

    raw_total = _safe_int(raw.get("total_pending_lines"), 0)
    raw_age = _safe_float(raw.get("oldest_pending_age_seconds"), 0.0)
    overlay_gap = max(_safe_int(effective.get("total_pending_lines"), 0) - raw_total, 0)
    sparse = raw.get("line_estimation") if isinstance(raw.get("line_estimation"), dict) else {}
    sparse_active = bool(sparse.get("sparse_large_line_active", False))
    raw_green_live = bool(
        raw_total <= RAW_GREEN_TOTAL
        and _safe_int(raw.get("core_pending_lines"), 0) <= RAW_GREEN_CORE
        and raw_age <= RAW_GREEN_AGE_SECONDS
        and overlay_gap <= 0
    )
    needs_intake = bool(raw_total > RAW_GREEN_TOTAL or raw_age > RAW_WARN_AGE_SECONDS or (sparse_active and not raw_green_live) or overlay_gap > OVERLAY_WARN_TOTAL)
    needs_drain = bool(raw_total > RAW_GREEN_TOTAL or raw_age > RAW_GREEN_AGE_SECONDS or overlay_gap > OVERLAY_WARN_TOTAL)
    needs_cleanup = bool(
        (sparse_active and not raw_green_live)
        or _safe_int(raw.get("cold_pending_lines"), 0) > 0
        or _safe_int(raw.get("stale_stage_pending_lines"), 0) > 0
    )

    if apply and needs_intake and not skip_intake:
        steps["apply_intake_pressure_relief"] = _run_command(
            [sys.executable, str(project_root / "scripts" / "ops" / "pressure_relief_control.py"), "--apply", "--json"],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )
        pressure_relief = _load_json(health_root / "pressure_relief_control_latest.json")

    if apply and needs_drain and not skip_drain:
        steps["apply_focused_drain"] = _run_command(
            [
                sys.executable,
                str(project_root / "scripts" / "ops" / "external_backlog_drain.py"),
                "--apply",
                "--follow-through",
                "--wait-timeout-seconds",
                str(int(max(wait_timeout_seconds, 1))),
                "--json",
            ],
            project_root=project_root,
            timeout_seconds=max(command_timeout_seconds, min(wait_timeout_seconds + 30.0, wait_timeout_seconds + command_timeout_seconds)),
        )
        external_drain = _load_json(health_root / "external_backlog_drain_latest.json")

    if apply and needs_cleanup and not skip_cleanup:
        steps["stage_stale_artifacts"] = _run_command(
            [sys.executable, str(project_root / "scripts" / "ops" / "stale_artifact_sweeper_bot.py"), "--json"],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )
        steps["safe_retention_pass"] = _run_command(
            [
                sys.executable,
                str(project_root / "scripts" / "data_retention_policy.py"),
                "--apply",
                "--no-stale-purge",
                "--skip-sqlite-vacuum",
                "--json",
            ],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )
        if allow_stale_reaper:
            steps["allowed_stale_reaper_pass"] = _run_command(
                [sys.executable, str(project_root / "scripts" / "ops" / "stale_artifact_reaper_bot.py"), "--json"],
                project_root=project_root,
                timeout_seconds=command_timeout_seconds,
            )
        retention = _load_json(health_root / "data_retention_latest.json")
        sweeper = _load_json(health_root / "stale_artifact_sweeper_bot_latest.json")
        reaper = _load_json(health_root / "stale_artifact_reaper_bot_latest.json")

    if apply:
        steps["post_storage_control_refresh"] = _run_command(
            [sys.executable, str(project_root / "scripts" / "ops" / "ingestion_storage_control.py"), "--json"],
            project_root=project_root,
            timeout_seconds=command_timeout_seconds,
        )
        storage = _load_json(health_root / "ingestion_storage_control_latest.json") or storage
        backpressure = _load_json(health_root / "ingestion_backpressure_latest.json") or backpressure
        priority_queue = _load_json(health_root / "ingestion_priority_queue_latest.json") or priority_queue
        raw = _storage_raw_backpressure(storage, backpressure)
        effective = _effective_backpressure(storage, raw)

    sections, hot_rows, top_actions, recommended_commands = _build_sections(
        raw=raw,
        effective=effective,
        storage=storage,
        backpressure=backpressure,
        priority_queue=priority_queue,
        external_drain=external_drain,
        writer=writer,
        pressure_relief=pressure_relief,
        retention=retention,
        sweeper=sweeper,
        reaper=reaper,
    )
    section_scores = [float(section.get("score", 0.0) or 0.0) for section in sections.values()]
    overall_score = min(section_scores) if section_scores else 0.0
    hard_blockers = [
        blocker
        for section in sections.values()
        for blocker in list(section.get("blockers") or [])
        if blocker
    ]
    raw_green = bool(
        _safe_int(raw.get("total_pending_lines"), 0) <= RAW_GREEN_TOTAL
        and _safe_int(raw.get("core_pending_lines"), 0) <= RAW_GREEN_CORE
        and _safe_float(raw.get("oldest_pending_age_seconds"), 0.0) <= RAW_GREEN_AGE_SECONDS
        and max(_safe_int(effective.get("total_pending_lines"), 0) - _safe_int(raw.get("total_pending_lines"), 0), 0) <= 0
    )
    if raw_green and not hard_blockers:
        overall_score = max(overall_score, 99.0)
    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": bool(overall_score >= 75.0),
        "overall_status": _status_from_score(overall_score),
        "overall_grade": _grade(overall_score),
        "overall_score": round(overall_score, 3),
        "apply_requested": bool(apply),
        "apply_executed": bool(steps),
        "policy": {
            "single_writer_only": True,
            "hard_delete_requires_allow_stale_reaper": True,
            "protected_volumes": ["/Volumes/VIDEO"],
            "live_execution_unchanged": "paper_read_only",
        },
        "summary": {
            "raw_total_pending_lines": _safe_int(raw.get("total_pending_lines"), 0),
            "raw_core_pending_lines": _safe_int(raw.get("core_pending_lines"), 0),
            "raw_deferred_pending_lines": _safe_int(raw.get("deferred_pending_lines"), 0),
            "raw_cold_pending_lines": _safe_int(raw.get("cold_pending_lines"), 0),
            "raw_support_pending_lines": _safe_int(raw.get("support_pending_lines"), 0),
            "raw_oldest_pending_age_seconds": round(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 3),
            "effective_total_pending_lines": _safe_int(effective.get("total_pending_lines"), 0),
            "effective_core_pending_lines": _safe_int(effective.get("core_pending_lines"), 0),
            "overlay_gap_lines": max(_safe_int(effective.get("total_pending_lines"), 0) - _safe_int(raw.get("total_pending_lines"), 0), 0),
            "overlay_adjusted": bool(effective.get("overlay_adjusted", False)),
            "sparse_large_line_active": bool((raw.get("line_estimation") or {}).get("sparse_large_line_active", False)) if isinstance(raw.get("line_estimation"), dict) else False,
            "hot_file_count": len(hot_rows),
        },
        "sections": sections,
        "top_hot_files": hot_rows[:10],
        "recommended_commands": recommended_commands,
        "stop_conditions": [
            "raw_total_pending_lines <= 15000",
            "raw_core_pending_lines <= 10000",
            "raw_deferred_pending_lines <= 5000",
            "raw_oldest_pending_age_seconds <= 900",
            "overlay_gap_lines trends down without raw backlog rising",
            "no sparse_large_line_active raw pending-byte warning unless raw backlog is green and byte-window controls are only in watch mode",
        ],
        "top_actions": top_actions,
        "steps": steps,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate raw backlog measurement, hot-file mapping, drain refinement, intake relief, and safe cleanup.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-drain", action="store_true")
    parser.add_argument("--skip-intake", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--allow-stale-reaper", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        skip_drain=bool(args.skip_drain),
        skip_intake=bool(args.skip_intake),
        skip_cleanup=bool(args.skip_cleanup),
        allow_stale_reaper=bool(args.allow_stale_reaper),
        wait_timeout_seconds=float(args.wait_timeout_seconds),
        command_timeout_seconds=float(args.command_timeout_seconds),
    )
    out_path = Path(args.out_file)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    _write_json(out_path, payload)
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload.get("ok") or args.json else 1


if __name__ == "__main__":
    raise SystemExit(main())
