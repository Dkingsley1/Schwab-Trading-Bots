#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from scripts import data_retention_policy as retention


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_quarantine_bot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "backlog_quarantine_bot.lock"
DEFAULT_STALE_STAGE_ROOT = PROJECT_ROOT / "data" / "stale_stage"
DEFAULT_STALE_STAGE_MANIFEST = DEFAULT_STALE_STAGE_ROOT / "backlog_quarantine_manifest.jsonl"
FILE_DAY_RE = re.compile(r"(20\d{6})")


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


def _extract_file_day(text: str) -> str:
    match = FILE_DAY_RE.search(str(text or ""))
    return match.group(1) if match else ""


def _candidate_label(source_rel: str) -> str:
    rel = str(source_rel or "").strip()
    if "/shadow_pnl_attribution_" in rel:
        return "backlog_quarantine_shadow_attribution"
    if "decision_explanations_" in rel:
        return "backlog_quarantine_explanations"
    return ""


def _resolve_candidate_path(project_root: Path, source_rel: str) -> Path:
    return project_root / str(source_rel or "").strip()


def _is_previous_day_candidate(file_day: str, today_utc: str) -> bool:
    return bool(file_day and file_day < today_utc)


def _candidate_rows(
    project_root: Path,
    *,
    now_utc: datetime,
    min_shadow_pending_lines: int,
    min_explanation_pending_lines: int,
    min_shadow_age_hours: float,
    min_explanation_age_hours: float,
) -> list[dict[str, Any]]:
    backpressure = _load_json(project_root / "governance" / "health" / "ingestion_backpressure_latest.json")
    candidate_map: dict[str, dict[str, Any]] = {}
    today_utc = now_utc.strftime("%Y%m%d")

    for lane_key, lane_name in (("top_cold_pending_files", "cold"), ("top_deferred_pending_files", "deferred")):
        for raw in backpressure.get(lane_key) or []:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            label = _candidate_label(source_rel)
            if not label:
                continue
            path = _resolve_candidate_path(project_root, source_rel)
            if not path.exists():
                continue

            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            oldest_pending_age_seconds = max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0)
            file_day = _extract_file_day(source_rel)
            try:
                file_age_hours = max((now_utc - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0, 0.0)
            except OSError:
                file_age_hours = 0.0
            effective_age_hours = max(file_age_hours, oldest_pending_age_seconds / 3600.0)

            is_shadow = label == "backlog_quarantine_shadow_attribution"
            min_pending_lines = min_shadow_pending_lines if is_shadow else min_explanation_pending_lines
            min_age_hours = min_shadow_age_hours if is_shadow else min_explanation_age_hours
            if pending_lines < max(int(min_pending_lines), 1):
                continue
            if effective_age_hours < float(min_age_hours):
                continue
            if not _is_previous_day_candidate(file_day, today_utc):
                continue

            row = {
                "label": label,
                "lane": lane_name,
                "source_rel": source_rel,
                "path": str(path),
                "file_day": file_day,
                "pending_lines": pending_lines,
                "age_hours": round(effective_age_hours, 3),
                "candidate_reason": "previous_day_stale_backlog",
            }
            existing = candidate_map.get(str(path))
            if existing is None:
                candidate_map[str(path)] = row
                continue
            existing_is_cold = str(existing.get("lane") or "") == "cold"
            incoming_is_cold = lane_name == "cold"
            if incoming_is_cold and not existing_is_cold:
                candidate_map[str(path)] = row
                continue
            if pending_lines > int(existing.get("pending_lines", 0) or 0):
                candidate_map[str(path)] = row
                continue
            if effective_age_hours > float(existing.get("age_hours", 0.0) or 0.0):
                candidate_map[str(path)] = row

    candidate_rows = list(candidate_map.values())
    candidate_rows.sort(
        key=lambda row: (
            int(row.get("pending_lines", 0) or 0),
            float(row.get("age_hours", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return candidate_rows


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    max_move_files: int = 4,
    min_shadow_pending_lines: int = 50000,
    min_explanation_pending_lines: int = 25000,
    min_shadow_age_hours: float = 6.0,
    min_explanation_age_hours: float = 12.0,
    stale_stage_root: Path = DEFAULT_STALE_STAGE_ROOT,
    stale_stage_manifest: Path = DEFAULT_STALE_STAGE_MANIFEST,
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    candidates = _candidate_rows(
        project_root,
        now_utc=now_utc,
        min_shadow_pending_lines=min_shadow_pending_lines,
        min_explanation_pending_lines=min_explanation_pending_lines,
        min_shadow_age_hours=min_shadow_age_hours,
        min_explanation_age_hours=min_explanation_age_hours,
    )
    selected = candidates[: max(int(max_move_files), 0)]
    moved_files = 0
    moved_pending_lines = 0
    moved_bytes = 0
    moved_paths: list[str] = []
    move_errors: list[str] = []
    moved_by_label: dict[str, dict[str, int]] = {}

    if apply and selected:
        selected_map = {str(row.get("path") or ""): row for row in selected}
        grouped_paths: dict[str, list[Path]] = {}
        for row in selected:
            grouped_paths.setdefault(str(row.get("label") or ""), []).append(Path(str(row.get("path") or "")))

        external_root = retention._resolve_external_project_root()
        for label, rows in grouped_paths.items():
            result = retention._move_paths_to_stale_stage(
                paths=rows,
                label=label,
                project_root=project_root,
                external_root=external_root,
                stale_root=stale_stage_root,
                manifest_path=stale_stage_manifest,
            )
            moved_files += int(result.get("moved", 0) or 0)
            moved_bytes += int(result.get("moved_bytes", 0) or 0)
            moved_paths.extend(list(result.get("moved_paths") or []))
            move_errors.extend(list(result.get("errors") or []))
            moved_by_label[label] = {
                "moved_files": int(result.get("moved", 0) or 0),
                "moved_bytes": int(result.get("moved_bytes", 0) or 0),
            }
            for row_path in rows:
                row = selected_map.get(str(row_path))
                if row:
                    moved_pending_lines += int(row.get("pending_lines", 0) or 0)

    if not selected:
        overall_status = "no_candidates"
        ok = True
    elif not apply:
        overall_status = "ready"
        ok = True
    elif move_errors:
        overall_status = "partial"
        ok = False
    else:
        overall_status = "applied"
        ok = True

    recommended_actions: list[str] = []
    if selected and not apply:
        recommended_actions.append("stage prior-day shadow attribution and explanation backlog into stale_stage so the hot ingestion path stops carrying cold debt")
    if moved_files > 0:
        recommended_actions.append("refresh ingestion backpressure after backlog quarantine so the retry bot and storage controller see the lower cold backlog immediately")
    if move_errors:
        recommended_actions.append("inspect backlog quarantine move errors before widening deferred or cold lane budgets")
    if not recommended_actions:
        recommended_actions.append("keep backlog quarantine idle until stale prior-day attribution or explanation files exceed the quarantine threshold again")

    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply": bool(apply),
        "candidate_files": int(len(selected)),
        "candidate_pending_lines": int(sum(int(row.get("pending_lines", 0) or 0) for row in selected)),
        "moved_files": int(moved_files),
        "moved_pending_lines": int(moved_pending_lines),
        "moved_bytes": int(moved_bytes),
        "move_errors": move_errors,
        "stale_stage_root": str(stale_stage_root),
        "stale_stage_manifest": str(stale_stage_manifest),
        "candidate_rows": selected,
        "moved_paths": moved_paths[:10],
        "moved_by_label": moved_by_label,
        "recommended_actions": recommended_actions[:5],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage stale prior-day attribution and explanation backlog out of the hot ingestion path.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--stale-stage-root", default=str(DEFAULT_STALE_STAGE_ROOT))
    parser.add_argument("--stale-stage-manifest", default=str(DEFAULT_STALE_STAGE_MANIFEST))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-move-files", type=int, default=4)
    parser.add_argument("--min-shadow-pending-lines", type=int, default=50000)
    parser.add_argument("--min-explanation-pending-lines", type=int, default=25000)
    parser.add_argument("--min-shadow-age-hours", type=float, default=6.0)
    parser.add_argument("--min-explanation-age-hours", type=float, default=12.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    stale_stage_root = Path(args.stale_stage_root).expanduser()
    stale_stage_manifest = Path(args.stale_stage_manifest).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any]
    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "schema_version": 1,
                "ok": True,
                "overall_status": "already_running",
                "apply": bool(args.apply),
                "candidate_files": 0,
                "candidate_pending_lines": 0,
                "moved_files": 0,
                "moved_pending_lines": 0,
                "moved_bytes": 0,
                "move_errors": [],
                "recommended_actions": ["keep a single backlog quarantine worker active so stale-stage moves stay serialized"],
            }
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("backlog_quarantine_bot overall_status=already_running")
            return 0

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            max_move_files=int(args.max_move_files),
            min_shadow_pending_lines=int(args.min_shadow_pending_lines),
            min_explanation_pending_lines=int(args.min_explanation_pending_lines),
            min_shadow_age_hours=float(args.min_shadow_age_hours),
            min_explanation_age_hours=float(args.min_explanation_age_hours),
            stale_stage_root=stale_stage_root,
            stale_stage_manifest=stale_stage_manifest,
        )
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "backlog_quarantine_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"candidate_files={int(payload.get('candidate_files', 0) or 0)} "
            f"moved_files={int(payload.get('moved_files', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
