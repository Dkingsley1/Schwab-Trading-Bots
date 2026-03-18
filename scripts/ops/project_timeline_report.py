#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "project_timeline"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "project_timeline_state.json"

TIMELINE_MD_PDF_RE = re.compile(r"^project_timeline_(\d{8}_\d{6})\.(?:md|pdf)$")
TIMELINE_PRINT_RE = re.compile(r"^project_timeline_print_(\d{8}_\d{6})\.html$")
RECENT_ACTIVITY_GLOBS = [
    "master_bot_registry.json",
    "README.md",
    "COMMANDS.md",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "core/**/*.py",
    "config/**/*",
    "tests/**/*.py",
    "logs/**/*.log",
    "governance/health/*.json",
    "governance/walk_forward/*.json",
    "governance/events/*.jsonl",
    "governance/watchdog/*.jsonl",
    "data/**/*.json",
    "data/**/*.jsonl",
    "exports/sql_reports/*.json",
    "exports/sql_reports/*.md",
    "exports/reports/**/*.json",
    "exports/reports/**/*.md",
    "exports/reports/project_timeline/*.html",
    "exports/reports/project_timeline/*.pdf",
]


RECENT_ACTIVITY_EXCLUDE_PREFIXES = [
    "exports/reports/project_timeline/",
]
RECENT_ACTIVITY_EXCLUDE_FILES = {
    "governance/health/project_timeline_state.json",
}

MILESTONE_SUBJECT_WEIGHTS = (
    ("main control room", 18),
    ("project timeline", 16),
    ("autoupdating", 14),
    ("launchd", 12),
    ("retrain", 12),
    ("dividend", 10),
    ("compound", 10),
    ("canar", 9),
    ("promotion", 9),
    ("walk forward", 8),
    ("gate", 7),
    ("report", 7),
)

SIGNIFICANT_CODE_PREFIXES = (
    "core/",
    "scripts/",
    "config/",
    "tests/",
)

SIGNIFICANT_CODE_FILES = {
    ".gitignore",
    "COMMANDS.md",
    "README.md",
    "master_bot_registry.json",
}

SIGNIFICANT_ARTIFACT_PREFIXES = (
    "governance/health/shadow_loop_",
    "governance/health/data_ingress_latest_",
    "governance/health/retrain_",
    "governance/health/training_success_",
    "governance/health/model_card_",
    "governance/health/preflight_autofix_",
    "governance/health/storage_failback_sync_",
    "governance/health/jsonl_sql_ingestion_health_",
    "governance/health/sql_link_service_",
    "governance/walk_forward/",
    "logs/coinbase_live_",
    "logs/coinbase_futures_live_",
    "logs/all_sleeves_",
)

TOKEN_DISPLAY_MAP = {
    "api": "API",
    "coinbase": "Coinbase",
    "etf": "ETF",
    "jsonl": "JSONL",
    "mlx": "MLX",
    "schwab": "Schwab",
    "sql": "SQL",
}


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _run(cmd: List[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return 1, "", str(exc)


def _git_ok() -> bool:
    rc, out, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    return rc == 0 and out.strip().lower() == "true"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_git() -> Dict[str, Any]:
    if not _git_ok():
        return {
            "available": False,
            "branch": "unknown",
            "head": "unknown",
            "commits": [],
            "commit_count": 0,
            "status_porcelain": "",
            "status_lines": [],
            "status_branch_line": "",
        }

    _, branch, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    _, head, _ = _run(["git", "rev-parse", "HEAD"])
    _, log_out, _ = _run(["git", "log", "--reverse", "--date=iso", "--pretty=format:%ad|%h|%s"])
    _, status_porcelain, _ = _run(["git", "status", "--short", "--branch"])

    commits: List[Dict[str, str]] = []
    for line in log_out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        commits.append(
            {
                "date": parts[0].strip(),
                "sha": parts[1].strip(),
                "subject": parts[2].strip(),
            }
        )

    status_lines = status_porcelain.splitlines()
    status_branch_line = status_lines[0] if status_lines and status_lines[0].startswith("## ") else ""
    working_tree = [line for line in status_lines if not line.startswith("## ")]
    recent_working_tree_changes = _collect_recent_working_tree_changes(working_tree, limit=40)

    return {
        "available": True,
        "branch": branch.strip() or "unknown",
        "head": head.strip() or "unknown",
        "commits": commits,
        "commit_count": len(commits),
        "status_porcelain": status_porcelain,
        "status_lines": working_tree,
        "status_branch_line": status_branch_line,
        "recent_working_tree_changes": recent_working_tree_changes,
    }


def _classify_status(lines: List[str]) -> Dict[str, int]:
    counts = {
        "modified": 0,
        "added": 0,
        "deleted": 0,
        "renamed": 0,
        "untracked": 0,
        "other": 0,
    }
    for line in lines:
        code = line[:2]
        if code == "??":
            counts["untracked"] += 1
            continue
        if "M" in code:
            counts["modified"] += 1
        elif "A" in code:
            counts["added"] += 1
        elif "D" in code:
            counts["deleted"] += 1
        elif "R" in code:
            counts["renamed"] += 1
        else:
            counts["other"] += 1
    return counts


def _status_line_to_path(line: str) -> str:
    raw = str(line or "")
    if len(raw) >= 3:
        path_part = raw[3:].strip()
    else:
        path_part = raw.strip()

    if " -> " in path_part:
        path_part = path_part.split(" -> ", 1)[1].strip()

    if (path_part.startswith('"') and path_part.endswith('"')) or (path_part.startswith("'") and path_part.endswith("'")):
        path_part = path_part[1:-1]

    return path_part


def _collect_recent_working_tree_changes(status_lines: List[str], limit: int = 40) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for line in status_lines:
        path_part = _status_line_to_path(line)
        if not path_part or path_part in seen:
            continue
        seen.add(path_part)

        abs_path = (PROJECT_ROOT / path_part)
        exists = abs_path.exists()
        is_dir = abs_path.is_dir() if exists else False

        mtime_epoch = 0.0
        mtime_utc = ""
        mtime_local = ""
        size_bytes = 0

        if exists:
            try:
                st = abs_path.stat()
                mtime_epoch = float(st.st_mtime)
                mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
                mtime_local = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone().isoformat()
                if abs_path.is_file():
                    size_bytes = int(st.st_size)
            except Exception:
                pass

        status_code = line[:2].strip() if len(line) >= 2 else ""
        rows.append(
            {
                "status": status_code,
                "line": line,
                "path": path_part,
                "exists": bool(exists),
                "is_dir": bool(is_dir),
                "size_bytes": int(size_bytes),
                "mtime_epoch": float(mtime_epoch),
                "mtime_utc": mtime_utc,
                "mtime_local": mtime_local,
            }
        )

    rows.sort(key=lambda r: (float(r.get("mtime_epoch", 0.0)), str(r.get("path", ""))), reverse=True)
    return rows[: max(int(limit), 0)]


def _latest_file_name(glob_pat: str) -> str:
    files = sorted(PROJECT_ROOT.glob(glob_pat), key=lambda p: p.name)
    return files[-1].name if files else ""


def _parse_timeline_stamp(stamp: str) -> datetime | None:
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _prune_timeline_snapshots(out_dir: Path, keep_runs: int, older_than_days: int) -> Dict[str, Any]:
    keep_runs_eff = max(int(keep_runs), 0)
    older_days_eff = max(int(older_than_days), 0)

    run_files: Dict[str, List[Path]] = {}
    for p in out_dir.glob("project_timeline*"):
        name = p.name
        if name in {
            "project_timeline_latest.md",
            "project_timeline_latest.pdf",
            "project_timeline_print_latest.html",
        }:
            continue
        match = TIMELINE_MD_PDF_RE.match(name) or TIMELINE_PRINT_RE.match(name)
        if not match:
            continue
        stamp = match.group(1)
        run_files.setdefault(stamp, []).append(p)

    runs: List[tuple[str, datetime, List[Path]]] = []
    for stamp, files in run_files.items():
        dt = _parse_timeline_stamp(stamp)
        if dt is None:
            continue
        runs.append((stamp, dt, files))

    runs.sort(key=lambda row: row[1], reverse=True)
    keep_stamps = {row[0] for row in runs[:keep_runs_eff]} if keep_runs_eff > 0 else set()
    cutoff_ts = datetime.now(timezone.utc).timestamp() - float(older_days_eff * 86400)

    deleted_files = 0
    deleted_runs: List[str] = []
    delete_errors = 0
    for stamp, dt, files in runs:
        if stamp in keep_stamps:
            continue
        delete_due_to_keep_cap = bool(keep_runs_eff > 0)
        delete_due_to_age = bool(dt.timestamp() < cutoff_ts)
        if not delete_due_to_keep_cap and not delete_due_to_age:
            continue
        deleted_runs.append(stamp)
        for fp in files:
            try:
                fp.unlink(missing_ok=True)
                deleted_files += 1
            except Exception:
                delete_errors += 1

    return {
        "enabled": True,
        "total_runs_seen": len(runs),
        "keep_runs": keep_runs_eff,
        "older_than_days": older_days_eff,
        "deleted_runs": len(deleted_runs),
        "deleted_files": int(deleted_files),
        "delete_errors": int(delete_errors),
        "deleted_run_stamps": deleted_runs[:25],
    }


def _activity_local_date(row: Dict[str, Any]) -> str:
    txt = str(row.get("mtime_local", ""))
    if "T" in txt:
        return txt.split("T", 1)[0]
    return ""


def _collect_recent_project_activity(hours: int, limit: int = 200) -> List[Dict[str, Any]]:
    hours_eff = max(int(hours), 1)
    limit_eff = max(int(limit), 1)
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=hours_eff)).timestamp()

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in RECENT_ACTIVITY_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            try:
                if not path.exists() or not path.is_file():
                    continue
                rel = str(path.relative_to(PROJECT_ROOT))
                if rel in RECENT_ACTIVITY_EXCLUDE_FILES:
                    continue
                if any(rel.startswith(prefix) for prefix in RECENT_ACTIVITY_EXCLUDE_PREFIXES):
                    continue
                if rel in seen:
                    continue
                seen.add(rel)
                st = path.stat()
                mtime_epoch = float(st.st_mtime)
                if mtime_epoch < cutoff_ts:
                    continue
                rows.append(
                    {
                        "path": rel,
                        "mtime_epoch": mtime_epoch,
                        "mtime_utc": datetime.fromtimestamp(mtime_epoch, tz=timezone.utc).isoformat(),
                        "mtime_local": datetime.fromtimestamp(mtime_epoch, tz=timezone.utc).astimezone().isoformat(),
                        "size_bytes": int(st.st_size),
                    }
                )
            except Exception:
                continue

    rows.sort(key=lambda r: (float(r.get("mtime_epoch", 0.0)), str(r.get("path", ""))), reverse=True)
    if len(rows) <= limit_eff:
        return rows

    # Keep a guaranteed slice for the two newest local dates (today + yesterday when available),
    # then fill the remainder with newest activity.
    distinct_dates: List[str] = []
    for row in rows:
        d = _activity_local_date(row)
        if d and d not in distinct_dates:
            distinct_dates.append(d)
        if len(distinct_dates) >= 2:
            break

    guaranteed_per_date = max(1, min(80, limit_eff // 3))
    selected: List[Dict[str, Any]] = []
    used_paths: set[str] = set()

    for d in distinct_dates:
        taken = 0
        for row in rows:
            if _activity_local_date(row) != d:
                continue
            path_key = str(row.get("path", ""))
            if not path_key or path_key in used_paths:
                continue
            selected.append(row)
            used_paths.add(path_key)
            taken += 1
            if taken >= guaranteed_per_date:
                break

    for row in rows:
        path_key = str(row.get("path", ""))
        if not path_key or path_key in used_paths:
            continue
        selected.append(row)
        used_paths.add(path_key)
        if len(selected) >= limit_eff:
            break

    selected.sort(key=lambda r: (float(r.get("mtime_epoch", 0.0)), str(r.get("path", ""))), reverse=True)
    return selected[:limit_eff]


def _recent_activity_daily_counts(rows: List[Dict[str, Any]], max_days: int = 7) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in rows:
        d = _activity_local_date(row)
        if not d:
            continue
        counts[d] = int(counts.get(d, 0)) + 1

    out: List[Dict[str, Any]] = []
    for d in sorted(counts.keys(), reverse=True):
        out.append({"date": d, "count": counts[d]})
    return out[: max(int(max_days), 1)]


def _recent_project_activity_probe(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0, "latest_path": "", "latest_mtime_utc": "", "latest_mtime_epoch": 0.0}
    top = rows[0] if isinstance(rows[0], dict) else {}
    return {
        "count": len(rows),
        "latest_path": str(top.get("path", "")),
        "latest_mtime_utc": str(top.get("mtime_utc", "")),
        "latest_mtime_epoch": float(top.get("mtime_epoch", 0.0) or 0.0),
    }


def _build_live_timeline_events(context: Dict[str, Any], limit: int = 80) -> List[Dict[str, Any]]:
    git_data = context.get("git") if isinstance(context.get("git"), dict) else {}
    ops_data = context.get("ops") if isinstance(context.get("ops"), dict) else {}
    events: List[Dict[str, Any]] = []

    for row in git_data.get("commits", []):
        if not isinstance(row, dict):
            continue
        raw_date = str(row.get("date", "")).strip()
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_local = dt.astimezone()
            sort_epoch = float(dt.timestamp())
            date_local = dt_local.isoformat()
        except Exception:
            sort_epoch = 0.0
            date_local = raw_date
        events.append(
            {
                "kind": "commit",
                "sort_epoch": sort_epoch,
                "date_local": date_local,
                "label": str(row.get("sha", "")),
                "detail": str(row.get("subject", "")),
                "path": "",
                "status": "",
            }
        )

    for row in git_data.get("recent_working_tree_changes", []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip() or "WT"
        path = str(row.get("path", "")).strip()
        size_bytes = int(row.get("size_bytes", 0) or 0)
        exists = bool(row.get("exists"))
        detail = f"{status} working tree change"
        if path:
            detail += f" | path={path}"
        detail += f" | exists={str(exists).lower()} | size_bytes={size_bytes}"
        events.append(
            {
                "kind": "working",
                "sort_epoch": float(row.get("mtime_epoch", 0.0) or 0.0),
                "date_local": str(row.get("mtime_local", "")) or "missing",
                "label": path or "n/a",
                "detail": detail,
                "path": path,
                "status": status,
            }
        )

    for row in ops_data.get("recent_project_activity", []):
        if not isinstance(row, dict):
            continue
        path = str(row.get("path", "")).strip()
        size_bytes = int(row.get("size_bytes", 0) or 0)
        events.append(
            {
                "kind": "artifact",
                "sort_epoch": float(row.get("mtime_epoch", 0.0) or 0.0),
                "date_local": str(row.get("mtime_local", "")) or "missing",
                "label": path or "n/a",
                "detail": f"project artifact update | path={path} | size_bytes={size_bytes}",
                "path": path,
                "status": "",
            }
        )

    sorted_events = sorted(
        events,
        key=lambda r: (float(r.get("sort_epoch", 0.0)), str(r.get("kind", "")), str(r.get("label", ""))),
        reverse=True,
    )

    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for event in sorted_events:
        key = (
            str(event.get("kind", "")),
            str(event.get("date_local", "")),
            str(event.get("label", "")),
            str(event.get("detail", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)

    if len(deduped) <= max(int(limit), 1):
        return deduped

    def _timeline_local_date(row: Dict[str, Any]) -> str:
        txt = str(row.get("date_local", "")).strip()
        if "T" in txt:
            return txt.split("T", 1)[0]
        if " " in txt:
            return txt.split(" ", 1)[0]
        return txt

    distinct_dates: List[str] = []
    for row in deduped:
        d = _timeline_local_date(row)
        if d and d not in distinct_dates:
            distinct_dates.append(d)
        if len(distinct_dates) >= 3:
            break

    if not distinct_dates:
        return deduped[: max(int(limit), 1)]

    guaranteed_per_date = max(5, min(20, max(int(limit), 1) // len(distinct_dates)))
    selected: List[Dict[str, Any]] = []
    used: set[tuple[str, str, str, str]] = set()

    for d in distinct_dates:
        taken = 0
        for row in deduped:
            if _timeline_local_date(row) != d:
                continue
            key = (
                str(row.get("kind", "")),
                str(row.get("date_local", "")),
                str(row.get("label", "")),
                str(row.get("detail", "")),
            )
            if key in used:
                continue
            selected.append(row)
            used.add(key)
            taken += 1
            if taken >= guaranteed_per_date:
                break

    for row in deduped:
        key = (
            str(row.get("kind", "")),
            str(row.get("date_local", "")),
            str(row.get("label", "")),
            str(row.get("detail", "")),
        )
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= max(int(limit), 1):
            break

    return selected[: max(int(limit), 1)]


def _parse_datetime_to_local(raw: Any) -> tuple[float, str]:
    txt = str(raw or "").strip()
    if not txt:
        return 0.0, "missing"

    dt: datetime | None = None
    for candidate in (txt.replace("Z", "+00:00"), txt):
        try:
            dt = datetime.fromisoformat(candidate)
            break
        except Exception:
            continue

    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(txt, fmt)
                break
            except Exception:
                continue

    if dt is None:
        return 0.0, txt

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt_local = dt.astimezone()
    return float(dt.timestamp()), dt_local.isoformat()


def _timeline_local_date_value(txt: Any) -> str:
    raw = str(txt or "").strip()
    if "T" in raw:
        return raw.split("T", 1)[0]
    if " " in raw:
        return raw.split(" ", 1)[0]
    return raw


def _path_area(path: str) -> str:
    txt = str(path or "").strip()
    if txt.startswith("core/"):
        return "Core"
    if txt.startswith("scripts/ops/"):
        return "Ops"
    if txt.startswith("scripts/"):
        return "Scripts"
    if txt.startswith("config/"):
        return "Config"
    if txt.startswith("tests/"):
        return "Tests"
    if txt.startswith("governance/health/"):
        return "Governance"
    if txt.startswith("governance/walk_forward/"):
        return "Training"
    if txt.startswith("governance/events/"):
        return "Runtime"
    if txt.startswith("logs/"):
        return "Logs"
    if txt.startswith("exports/reports/"):
        return "Reports"
    if txt.startswith("data/"):
        return "Data"
    if txt in {"README.md", "COMMANDS.md"}:
        return "Docs"
    if txt == "master_bot_registry.json":
        return "Registry"
    return "Project"


def _humanize_token(token: str) -> str:
    raw = str(token or "").strip().lower()
    if not raw:
        return ""
    if raw in TOKEN_DISPLAY_MAP:
        return TOKEN_DISPLAY_MAP[raw]
    if raw.isdigit():
        return raw
    return raw.replace("-", " ").title()


def _humanize_slug(slug: str) -> str:
    parts = [part for part in str(slug or "").strip("_").split("_") if part]
    if not parts:
        return "Unknown"
    return " ".join(_humanize_token(part) for part in parts)


def _working_change_action(status: str) -> str:
    code = str(status or "").strip()
    if code == "??":
        return "added"
    if "R" in code:
        return "renamed"
    if "D" in code:
        return "deleted"
    if "A" in code:
        return "added"
    if "M" in code:
        return "modified"
    return "updated"


def _is_significant_code_path(path: str, is_dir: bool = False) -> bool:
    txt = str(path or "").strip()
    if not txt or is_dir:
        return False
    return txt in SIGNIFICANT_CODE_FILES or txt.startswith(SIGNIFICANT_CODE_PREFIXES)


def _is_significant_artifact_path(path: str) -> bool:
    txt = str(path or "").strip()
    if not txt:
        return False
    return txt.startswith(SIGNIFICANT_ARTIFACT_PREFIXES)


def _describe_working_change(path: str, status: str) -> Dict[str, str]:
    txt = str(path or "").strip()
    action = _working_change_action(status)
    area = _path_area(txt)

    title = f"{area} change"
    if txt == "master_bot_registry.json":
        title = "Bot registry update"
    elif txt == "scripts/ops/project_timeline_report.py":
        title = "Timeline report generator"
    elif txt == "scripts/ops/opsctl.sh":
        title = "Ops control entrypoint"
    elif txt in {"README.md", "COMMANDS.md"}:
        title = "Project documentation"
    elif txt.startswith("scripts/run_long_term_core_etf_shadow.py"):
        title = "Long-term core ETF sleeve"
    elif txt.startswith("scripts/run_long_term_sector_rotation_shadow.py"):
        title = "Long-term sector rotation sleeve"
    elif txt.startswith("scripts/run_long_term_dividend_compound_shadow.py"):
        title = "Long-term dividend compound sleeve"
    elif txt.startswith("core/"):
        title = "Core trading engine"
    elif txt.startswith("scripts/ops/"):
        title = "Ops automation"
    elif txt.startswith("scripts/"):
        title = "Trading workflow script"
    elif txt.startswith("tests/"):
        title = "Test coverage"
    elif txt.startswith("config/"):
        title = "Configuration"

    return {
        "area": area,
        "title": title,
        "detail": f"{txt} {action} in working tree",
    }


def _describe_artifact_change(path: str) -> Dict[str, str]:
    txt = str(path or "").strip()
    name = Path(txt).name

    if txt.startswith("governance/health/shadow_loop_"):
        slug = name.removeprefix("shadow_loop_").removesuffix(".json")
        slug = re.sub(r"_\d+$", "", slug)
        return {
            "area": "Runtime",
            "title": "Bot loop heartbeat",
            "detail": f"{_humanize_slug(slug)} loop health refreshed",
        }

    if txt.startswith("governance/health/data_ingress_latest_"):
        slug = name.removeprefix("data_ingress_latest_").removesuffix(".json")
        return {
            "area": "Runtime",
            "title": "Data ingress snapshot",
            "detail": f"{_humanize_slug(slug)} ingress snapshot refreshed",
        }

    if name == "retrain_scorecard_latest.json":
        return {
            "area": "Training",
            "title": "Retrain scorecard",
            "detail": "latest retrain scorecard refreshed",
        }

    if name == "training_success_latest.json":
        return {
            "area": "Training",
            "title": "Training success verdict",
            "detail": "latest training success snapshot refreshed",
        }

    if name == "model_card_latest.json":
        return {
            "area": "Training",
            "title": "Model card export",
            "detail": "latest model card refreshed",
        }

    if name == "preflight_autofix_latest.json":
        return {
            "area": "Operations",
            "title": "Preflight autofix snapshot",
            "detail": "latest preflight autofix status refreshed",
        }

    if name == "storage_failback_sync_latest.json":
        return {
            "area": "Operations",
            "title": "Storage failback sync",
            "detail": "latest storage route status refreshed",
        }

    if name == "jsonl_sql_ingestion_health_latest.json":
        return {
            "area": "Data",
            "title": "JSONL-to-SQL ingestion health",
            "detail": "ingestion health snapshot refreshed",
        }

    if name == "sql_link_service_latest.json":
        return {
            "area": "Data",
            "title": "SQL link service",
            "detail": "SQL link service health refreshed",
        }

    if txt.startswith("governance/walk_forward/"):
        return {
            "area": "Training",
            "title": "Walk-forward gate snapshot",
            "detail": f"{name} refreshed",
        }

    if txt.startswith("logs/coinbase_futures_live_"):
        return {
            "area": "Runtime",
            "title": "Coinbase futures live session",
            "detail": f"{name} advanced",
        }

    if txt.startswith("logs/coinbase_live_"):
        return {
            "area": "Runtime",
            "title": "Coinbase spot live session",
            "detail": f"{name} advanced",
        }

    if txt.startswith("logs/all_sleeves_"):
        return {
            "area": "Runtime",
            "title": "All-sleeves orchestration",
            "detail": f"{name} advanced",
        }

    return {
        "area": _path_area(txt),
        "title": "Project artifact refresh",
        "detail": f"{txt} refreshed",
    }


def _commit_milestone_score(subject: str, idx: int, total: int) -> int:
    text = str(subject or "").strip().lower()
    score = 0

    if idx == 0:
        score += 50
    if total > 0 and idx == total - 1:
        score += 35
    if text.startswith("feat:"):
        score += 10

    for needle, weight in MILESTONE_SUBJECT_WEIGHTS:
        if needle in text:
            score += weight

    if re.search(r"\b(create|add|improve|introduce|activate|install|build)\b", text):
        score += 4
    if len(text) >= 72:
        score += 2

    return score


def _working_change_score(path: str, status: str) -> int:
    txt = str(path or "").strip()
    action = _working_change_action(status)
    score = 0

    if txt.startswith("scripts/run_long_term_"):
        score += 18
    elif txt == "master_bot_registry.json":
        score += 16
    elif txt == "scripts/ops/project_timeline_report.py":
        score += 16
    elif txt == "scripts/ops/opsctl.sh":
        score += 14
    elif txt in {"README.md", "COMMANDS.md"}:
        score += 12
    elif txt.startswith("core/"):
        score += 12
    elif txt.startswith("scripts/ops/"):
        score += 11
    elif txt.startswith("scripts/"):
        score += 10
    elif txt.startswith("config/"):
        score += 9
    elif txt.startswith("tests/"):
        score += 8

    if action == "added":
        score += 3

    if any(token in txt for token in ("timeline", "retrain", "dividend", "long_term", "coinbase", "schwab")):
        score += 3

    return score


def _artifact_change_score(path: str) -> int:
    txt = str(path or "").strip()
    name = Path(txt).name
    score = 0

    if txt.startswith("governance/health/shadow_loop_"):
        score += 16
    elif txt.startswith("governance/health/data_ingress_latest_"):
        score += 14
    elif txt.startswith("governance/walk_forward/"):
        score += 15
    elif txt.startswith("logs/coinbase_futures_live_"):
        score += 13
    elif txt.startswith("logs/coinbase_live_"):
        score += 12
    elif txt.startswith("logs/all_sleeves_"):
        score += 11

    if name == "retrain_scorecard_latest.json":
        score += 18
    elif name == "training_success_latest.json":
        score += 17
    elif name == "model_card_latest.json":
        score += 16
    elif name in {
        "jsonl_sql_ingestion_health_latest.json",
        "sql_link_service_latest.json",
        "preflight_autofix_latest.json",
        "storage_failback_sync_latest.json",
    }:
        score += 14

    if any(token in txt for token in ("long_term_core_etf", "crypto_futures", "default_crypto_coinbase")):
        score += 3

    return score


def _reference_group_key(source: str, reference: str, title: str = "") -> tuple[str, str, str]:
    src = str(source or "").strip()
    ref = str(reference or "").strip()
    ttl = str(title or "").strip()
    name = Path(ref).name

    if src == "artifact" and ref.startswith("governance/health/shadow_loop_"):
        slug = name.removeprefix("shadow_loop_").removesuffix(".json")
        slug = re.sub(r"_\d+$", "", slug)
        return src, "shadow_loop", slug

    if src == "artifact" and ref.startswith("governance/health/data_ingress_latest_"):
        return src, "data_ingress", name

    if src == "artifact" and ref.startswith("logs/coinbase_futures_live_"):
        return src, "runtime_log", "coinbase_futures_live"

    if src == "artifact" and ref.startswith("logs/coinbase_live_"):
        return src, "runtime_log", "coinbase_live"

    if src == "artifact" and ref.startswith("logs/all_sleeves_"):
        return src, "runtime_log", "all_sleeves"

    if src == "commit":
        return src, "commit", ref

    return src, ttl, ref


def _collapse_ranked_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    collapsed: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        key = _reference_group_key(
            str(row.get("source", "")),
            str(row.get("reference", "")),
            str(row.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        collapsed.append(row)

    return collapsed


def _select_ranked_rows_with_date_coverage(
    rows: List[Dict[str, Any]],
    limit: int,
    guarantee_dates: int = 3,
    guarantee_per_date: int = 1,
) -> List[Dict[str, Any]]:
    limit_eff = max(int(limit), 1)
    if len(rows) <= limit_eff:
        return rows[:limit_eff]

    distinct_dates: List[str] = []
    rows_by_recency = sorted(
        rows,
        key=lambda row: (float(row.get("sort_epoch", 0.0) or 0.0), int(row.get("score", 0) or 0)),
        reverse=True,
    )
    for row in rows_by_recency:
        d = _timeline_local_date_value(row.get("date_local", ""))
        if d and d not in distinct_dates:
            distinct_dates.append(d)
        if len(distinct_dates) >= max(int(guarantee_dates), 0):
            break

    selected: List[Dict[str, Any]] = []
    used: set[tuple[str, str, str, str]] = set()

    for d in distinct_dates:
        taken = 0
        for row in rows:
            if _timeline_local_date_value(row.get("date_local", "")) != d:
                continue
            key = (
                str(row.get("source", "")),
                str(row.get("reference", "")),
                str(row.get("title", "")),
                str(row.get("date_local", "")),
            )
            if key in used:
                continue
            selected.append(row)
            used.add(key)
            taken += 1
            if taken >= max(int(guarantee_per_date), 1):
                break

    for row in rows:
        key = (
            str(row.get("source", "")),
            str(row.get("reference", "")),
            str(row.get("title", "")),
            str(row.get("date_local", "")),
        )
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= limit_eff:
            break

    return selected[:limit_eff]


def _build_major_milestones(context: Dict[str, Any], limit: int = 14) -> List[Dict[str, Any]]:
    git_data = context.get("git") if isinstance(context.get("git"), dict) else {}
    ops_data = context.get("ops") if isinstance(context.get("ops"), dict) else {}
    candidates: List[Dict[str, Any]] = []
    commits = git_data.get("commits", []) if isinstance(git_data.get("commits"), list) else []

    total_commits = len(commits)
    for idx, row in enumerate(commits):
        if not isinstance(row, dict):
            continue
        score = _commit_milestone_score(str(row.get("subject", "")), idx, total_commits)
        if idx not in {0, max(total_commits - 1, 0)} and score < 16:
            continue
        sort_epoch, date_local = _parse_datetime_to_local(row.get("date"))
        candidates.append(
            {
                "date_local": date_local,
                "sort_epoch": sort_epoch,
                "area": "Git",
                "source": "commit",
                "title": "Milestone commit",
                "detail": str(row.get("subject", "")),
                "reference": str(row.get("sha", "")),
                "score": score,
            }
        )

    for row in git_data.get("recent_working_tree_changes", []):
        if not isinstance(row, dict):
            continue
        path = str(row.get("path", "")).strip()
        if not _is_significant_code_path(path, bool(row.get("is_dir"))):
            continue
        score = _working_change_score(path, str(row.get("status", "")))
        if score < 14:
            continue
        desc = _describe_working_change(path, str(row.get("status", "")))
        candidates.append(
            {
                "date_local": str(row.get("mtime_local", "")) or "missing",
                "sort_epoch": float(row.get("mtime_epoch", 0.0) or 0.0),
                "area": desc["area"],
                "source": "working",
                "title": desc["title"],
                "detail": desc["detail"],
                "reference": path,
                "score": score,
            }
        )

    for row in ops_data.get("recent_project_activity", []):
        if not isinstance(row, dict):
            continue
        path = str(row.get("path", "")).strip()
        if not _is_significant_artifact_path(path):
            continue
        score = _artifact_change_score(path)
        if score < 15:
            continue
        desc = _describe_artifact_change(path)
        candidates.append(
            {
                "date_local": str(row.get("mtime_local", "")) or "missing",
                "sort_epoch": float(row.get("mtime_epoch", 0.0) or 0.0),
                "area": desc["area"],
                "source": "artifact",
                "title": desc["title"],
                "detail": desc["detail"],
                "reference": path,
                "score": score,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda row: (
            int(row.get("score", 0) or 0),
            float(row.get("sort_epoch", 0.0) or 0.0),
            str(row.get("reference", "")),
        ),
        reverse=True,
    )
    ranked = _collapse_ranked_rows(ranked)
    selected = _select_ranked_rows_with_date_coverage(ranked, limit=limit, guarantee_dates=4, guarantee_per_date=1)
    return sorted(selected, key=lambda row: (float(row.get("sort_epoch", 0.0) or 0.0), str(row.get("reference", ""))))


def _build_significant_changes(context: Dict[str, Any], limit: int = 18) -> List[Dict[str, Any]]:
    git_data = context.get("git") if isinstance(context.get("git"), dict) else {}
    ops_data = context.get("ops") if isinstance(context.get("ops"), dict) else {}
    recent_hours = int(ops_data.get("recent_project_activity_hours", 72) or 72)
    cutoff_ts = datetime.now(timezone.utc).timestamp() - float(recent_hours * 3600)
    candidates: List[Dict[str, Any]] = []
    commits = git_data.get("commits", []) if isinstance(git_data.get("commits"), list) else []

    total_commits = len(commits)
    for idx, row in enumerate(commits):
        if not isinstance(row, dict):
            continue
        sort_epoch, date_local = _parse_datetime_to_local(row.get("date"))
        if sort_epoch and sort_epoch < cutoff_ts:
            continue
        score = _commit_milestone_score(str(row.get("subject", "")), idx, total_commits)
        if score < 12:
            continue
        candidates.append(
            {
                "date_local": date_local,
                "sort_epoch": sort_epoch,
                "area": "Git",
                "source": "commit",
                "title": "Recent commit",
                "detail": str(row.get("subject", "")),
                "reference": str(row.get("sha", "")),
                "score": score,
            }
        )

    for row in git_data.get("recent_working_tree_changes", []):
        if not isinstance(row, dict):
            continue
        path = str(row.get("path", "")).strip()
        if not _is_significant_code_path(path, bool(row.get("is_dir"))):
            continue
        score = _working_change_score(path, str(row.get("status", "")))
        if score < 10:
            continue
        desc = _describe_working_change(path, str(row.get("status", "")))
        candidates.append(
            {
                "date_local": str(row.get("mtime_local", "")) or "missing",
                "sort_epoch": float(row.get("mtime_epoch", 0.0) or 0.0),
                "area": desc["area"],
                "source": "working",
                "title": desc["title"],
                "detail": desc["detail"],
                "reference": path,
                "score": score,
            }
        )

    for row in ops_data.get("recent_project_activity", []):
        if not isinstance(row, dict):
            continue
        path = str(row.get("path", "")).strip()
        if not _is_significant_artifact_path(path):
            continue
        score = _artifact_change_score(path)
        if score < 12:
            continue
        desc = _describe_artifact_change(path)
        candidates.append(
            {
                "date_local": str(row.get("mtime_local", "")) or "missing",
                "sort_epoch": float(row.get("mtime_epoch", 0.0) or 0.0),
                "area": desc["area"],
                "source": "artifact",
                "title": desc["title"],
                "detail": desc["detail"],
                "reference": path,
                "score": score,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda row: (
            int(row.get("score", 0) or 0),
            float(row.get("sort_epoch", 0.0) or 0.0),
            str(row.get("reference", "")),
        ),
        reverse=True,
    )
    ranked = _collapse_ranked_rows(ranked)
    selected = _select_ranked_rows_with_date_coverage(ranked, limit=limit, guarantee_dates=4, guarantee_per_date=3)
    return sorted(
        selected,
        key=lambda row: (float(row.get("sort_epoch", 0.0) or 0.0), int(row.get("score", 0) or 0)),
        reverse=True,
    )


def _collect_ops_snapshot() -> Dict[str, Any]:
    promotion = _load_json(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json")
    graduation = _load_json(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json")
    retrain = _load_json(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json")
    leak = _load_json(PROJECT_ROOT / "governance" / "health" / "leak_overfit_guard_latest.json")
    preflight = _load_json(PROJECT_ROOT / "governance" / "health" / "preflight_autofix_latest.json")
    storage = _load_json(PROJECT_ROOT / "governance" / "health" / "storage_failback_sync_latest.json")

    preflight_events: List[Dict[str, str]] = []
    for path in sorted(PROJECT_ROOT.glob("logs/all_sleeves_*.log"), key=lambda p: p.name):
        stamp = path.stem.replace("all_sleeves_", "")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
        except Exception:
            continue
        preflight_line = next((line for line in lines if line.startswith("PREFLIGHT ")), "")
        if not preflight_line:
            continue
        fail_line = next((line for line in lines if line.startswith(" - FAIL ")), "")
        preflight_events.append(
            {
                "stamp": stamp,
                "result": preflight_line.replace("PREFLIGHT ", "", 1).strip(),
                "detail": fail_line.replace(" - FAIL ", "", 1).strip() if fail_line else "",
            }
        )
    preflight_events = preflight_events[-25:]

    return {
        "promotion": promotion,
        "graduation": graduation,
        "retrain": retrain,
        "leak": leak,
        "preflight": preflight,
        "storage": storage,
        "latest_all_sleeves_log": _latest_file_name("logs/all_sleeves_*.log"),
        "latest_coinbase_log": _latest_file_name("logs/coinbase_live_*.log"),
        "preflight_events": preflight_events,
    }


def _build_signature(git_data: Dict[str, Any], ops_data: Dict[str, Any]) -> str:
    activity_probe = ops_data.get("recent_project_activity_probe") if isinstance(ops_data.get("recent_project_activity_probe"), dict) else {}
    token = {
        "head": git_data.get("head", ""),
        "branch": git_data.get("branch", ""),
        "status_porcelain": git_data.get("status_porcelain", ""),
        "latest_all_sleeves_log": ops_data.get("latest_all_sleeves_log", ""),
        "latest_coinbase_log": ops_data.get("latest_coinbase_log", ""),
        "promotion_ts": (ops_data.get("promotion") or {}).get("timestamp_utc", ""),
        "graduation_ts": (ops_data.get("graduation") or {}).get("timestamp_utc", ""),
        "retrain_ts": (ops_data.get("retrain") or {}).get("timestamp_utc", ""),
        "recent_activity_latest_path": activity_probe.get("latest_path", ""),
        "recent_activity_latest_mtime_utc": activity_probe.get("latest_mtime_utc", ""),
        "recent_activity_count": activity_probe.get("count", 0),
        "include_detailed_timeline": bool(ops_data.get("include_detailed_timeline")),
    }
    raw = json.dumps(token, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fmt(val: Any, default: str = "n/a") -> str:
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.6f}".rstrip("0").rstrip(".")
    txt = str(val).strip()
    return txt if txt else default


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = os.getenv("PROJECT_TIMELINE_PDF_BIN", "").strip()
    if env_override:
        env_bin = Path(env_override).expanduser()
        if env_bin.exists():
            kind = "wkhtmltopdf" if env_bin.name == "wkhtmltopdf" else "browser"
            return str(env_bin), kind

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if wkhtmltopdf:
        return wkhtmltopdf, "wkhtmltopdf"

    browser_bins = [
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("microsoft-edge"),
        shutil.which("msedge"),
    ]
    for candidate in browser_bins:
        if candidate:
            return candidate, "browser"

    if allow_gui_renderer:
        for candidate in (
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ):
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
    else:
        cmd = [
            renderer,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={pdf_path}",
            html_uri,
        ]
    rc, out, err = _run(cmd)
    if rc == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or "ok"
    detail = err or out or f"rc={rc}"
    return False, detail


def _render_markdown(context: Dict[str, Any]) -> str:
    git_data = context["git"]
    ops_data = context["ops"]
    counts = _classify_status(git_data["status_lines"])
    major_milestones = _build_major_milestones(context, limit=14)
    significant_changes = _build_significant_changes(context, limit=18)
    include_detailed_timeline = bool(context.get("include_detailed_timeline"))
    show_detailed_timeline = include_detailed_timeline or (not major_milestones and not significant_changes)
    live_timeline = _build_live_timeline_events(context, limit=80) if show_detailed_timeline else []

    promotion = ops_data.get("promotion") or {}
    graduation = ops_data.get("graduation") or {}
    retrain = ops_data.get("retrain") or {}
    leak = ops_data.get("leak") or {}
    preflight = ops_data.get("preflight") or {}
    storage = ops_data.get("storage") or {}
    recent_project_activity = ops_data.get("recent_project_activity") if isinstance(ops_data.get("recent_project_activity"), list) else []
    recent_project_hours = int(ops_data.get("recent_project_activity_hours", 48) or 48)

    md: List[str] = []
    md.append("# Project Timeline Report")
    md.append("")
    md.append(f"Generated (UTC): `{context['generated_utc']}`")
    md.append(f"Generated (Local): `{context['generated_local']}`")
    md.append(f"Project root: `{PROJECT_ROOT}`")
    md.append("")
    md.append("## Snapshot")
    md.append(f"- Git available: `{git_data['available']}`")
    md.append(f"- Branch: `{git_data['branch']}`")
    md.append(f"- HEAD: `{git_data['head']}`")
    md.append(f"- Total commits: `{git_data['commit_count']}`")
    if git_data["commits"]:
        md.append(
            f"- First commit: `{git_data['commits'][0]['date']}` `{git_data['commits'][0]['sha']}` "
            f"{git_data['commits'][0]['subject']}"
        )
        md.append(
            f"- Latest commit: `{git_data['commits'][-1]['date']}` `{git_data['commits'][-1]['sha']}` "
            f"{git_data['commits'][-1]['subject']}"
        )
    md.append("")
    md.append("## Major Milestones")
    if major_milestones:
        for idx, row in enumerate(major_milestones, start=1):
            md.append(
                f"{idx}. `{row.get('date_local', 'n/a')}` | `{row.get('area', 'n/a')}` | "
                f"`{row.get('title', 'n/a')}` | {row.get('detail', '')} "
                f"(ref: `{row.get('reference', 'n/a')}`)"
            )
    else:
        md.append("1. No milestone-level events identified.")
    md.append("")
    md.append("## Significant Recent Changes")
    if significant_changes:
        for idx, row in enumerate(significant_changes, start=1):
            md.append(
                f"{idx}. `{row.get('date_local', 'n/a')}` | `{row.get('area', 'n/a')}` | "
                f"`{row.get('title', 'n/a')}` | {row.get('detail', '')} "
                f"(ref: `{row.get('reference', 'n/a')}`)"
            )
    else:
        md.append("1. No significant recent changes identified.")
    if show_detailed_timeline:
        md.append("")
        md.append("## Detailed Recent Timeline")
        if live_timeline:
            for idx, row in enumerate(live_timeline, start=1):
                md.append(
                    f"{idx}. `{row.get('date_local', 'n/a')}` | `{row.get('kind', 'n/a')}` | "
                    f"`{row.get('label', 'n/a')}` | {row.get('detail', '')}"
                )
        else:
            md.append("1. No recent timeline events found.")
    md.append("")
    md.append("## Working Tree")
    md.append(f"- Modified: `{counts['modified']}`")
    md.append(f"- Added: `{counts['added']}`")
    md.append(f"- Deleted: `{counts['deleted']}`")
    md.append(f"- Renamed: `{counts['renamed']}`")
    md.append(f"- Untracked: `{counts['untracked']}`")
    md.append(f"- Other: `{counts['other']}`")
    if git_data["status_lines"]:
        md.append("")
        md.append("### Files")
        for line in git_data["status_lines"]:
            md.append(f"- `{line}`")
    else:
        md.append("- Working tree is clean.")

    recent_rows = git_data.get("recent_working_tree_changes") if isinstance(git_data, dict) else []
    if isinstance(recent_rows, list) and recent_rows:
        md.append("")
        md.append("### Recent Working File Activity (mtime local)")
        for row in recent_rows:
            status = _fmt(row.get("status"), "")
            path_txt = _fmt(row.get("path"), "n/a")
            mtime_local = _fmt(row.get("mtime_local"), "missing")
            size_txt = _fmt(row.get("size_bytes"), "0")
            md.append(f"- `{status}` `{path_txt}` | mtime=`{mtime_local}` | size_bytes=`{size_txt}`")

    md.append("")
    md.append(f"## Recent Project Activity (Last {recent_project_hours} Hours)")
    daily_counts = _recent_activity_daily_counts(recent_project_activity, max_days=7)
    if daily_counts:
        summary = ", ".join([f"`{row['date']}`=`{row['count']}`" for row in daily_counts])
        md.append(f"- Daily file-change counts (local): {summary}")
    if recent_project_activity:
        for row in recent_project_activity:
            md.append(
                f"- `{_fmt(row.get('mtime_local'), 'missing')}` | "
                f"`{_fmt(row.get('path'), 'n/a')}` | "
                f"size_bytes=`{_fmt(row.get('size_bytes'), '0')}`"
            )
    else:
        md.append("- No project artifacts changed in this time window.")

    md.append("")
    md.append("## Runtime and Gates")
    md.append(
        f"- Promotion gate: `promote_ok={_fmt(promotion.get('promote_ok'))}` "
        f"`fail_share={_fmt(promotion.get('fail_share'))}` "
        f"`failed/considered={_fmt(promotion.get('failed_bots'))}/{_fmt(promotion.get('considered_bots'))}` "
        f"`timestamp={_fmt(promotion.get('timestamp_utc'))}`"
    )
    maturity = graduation.get("maturity") if isinstance(graduation.get("maturity"), dict) else {}
    md.append(
        f"- Graduation gate: `ok={_fmt(graduation.get('ok'))}` "
        f"`mature_pass_rate={_fmt(maturity.get('mature_pass_rate'))}` "
        f"`immature_active_count={_fmt(graduation.get('immature_active_count'))}` "
        f"`timestamp={_fmt(graduation.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Retrain scorecard: `status_counts={_fmt(retrain.get('status_counts'))}` "
        f"`master_update_status={_fmt(retrain.get('master_update_status'))}` "
        f"`failure_count={_fmt(retrain.get('failure_count'))}` "
        f"`timestamp={_fmt(retrain.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Leak/overfit: `ok={_fmt(leak.get('ok'))}` "
        f"`counts={_fmt(leak.get('counts'))}` "
        f"`timestamp={_fmt(leak.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Preflight autofix: `preflight_ok={_fmt(preflight.get('preflight_ok'))}` "
        f"`broker={_fmt(preflight.get('broker'))}` "
        f"`simulate={_fmt(preflight.get('simulate'))}` "
        f"`timestamp={_fmt(preflight.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Storage route: `mode={_fmt(storage.get('mode'))}` "
        f"`active_root={_fmt(storage.get('active_root'))}`"
    )
    md.append(f"- Latest all_sleeves log: `{_fmt(ops_data.get('latest_all_sleeves_log'))}`")
    md.append(f"- Latest coinbase log: `{_fmt(ops_data.get('latest_coinbase_log'))}`")
    md.append("")
    md.append("## Preflight Milestones (from logs)")
    if ops_data["preflight_events"]:
        for event in ops_data["preflight_events"]:
            detail = f" | fail=`{event['detail']}`" if event.get("detail") else ""
            md.append(f"- `{event['stamp']}` | `{event['result']}`{detail}")
    else:
        md.append("- No preflight events found.")
    md.append("")
    md.append("## Git Commit History")
    if git_data["commits"]:
        for idx, row in enumerate(git_data["commits"], start=1):
            md.append(f"{idx}. `{row['date']}` | `{row['sha']}` | {row['subject']}")
    else:
        md.append("1. No git history available.")
    md.append("")
    md.append("## Auto-Update")
    md.append("- This file is generated by `scripts/ops/project_timeline_report.py`.")
    md.append(
        "- Auto mode compares git + runtime signatures and only refreshes when something changes "
        "(commits, working tree, key gate artifacts, loop logs, or recent project artifact mtimes)."
    )
    return "\n".join(md).strip() + "\n"


def _render_html(context: Dict[str, Any]) -> str:
    git_data = context["git"]
    ops_data = context["ops"]
    counts = _classify_status(git_data["status_lines"])
    major_milestones = _build_major_milestones(context, limit=14)
    significant_changes = _build_significant_changes(context, limit=18)
    include_detailed_timeline = bool(context.get("include_detailed_timeline"))
    show_detailed_timeline = include_detailed_timeline or (not major_milestones and not significant_changes)
    live_timeline = _build_live_timeline_events(context, limit=80) if show_detailed_timeline else []

    promotion = ops_data.get("promotion") or {}
    graduation = ops_data.get("graduation") or {}
    retrain = ops_data.get("retrain") or {}
    leak = ops_data.get("leak") or {}
    preflight = ops_data.get("preflight") or {}
    storage = ops_data.get("storage") or {}
    maturity = graduation.get("maturity") if isinstance(graduation.get("maturity"), dict) else {}
    recent_project_activity = ops_data.get("recent_project_activity") if isinstance(ops_data.get("recent_project_activity"), list) else []
    recent_project_hours = int(ops_data.get("recent_project_activity_hours", 48) or 48)

    commit_rows = []
    for idx, row in enumerate(git_data["commits"], start=1):
        commit_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(row['date'])}</td>"
            f"<td><code>{html.escape(row['sha'])}</code></td>"
            f"<td>{html.escape(row['subject'])}</td>"
            "</tr>"
        )
    if not commit_rows:
        commit_rows.append("<tr><td colspan='4'>No git history available.</td></tr>")

    status_rows = []
    for line in git_data["status_lines"]:
        status_rows.append(f"<tr><td><code>{html.escape(line)}</code></td></tr>")
    if not status_rows:
        status_rows.append("<tr><td>Working tree is clean.</td></tr>")

    activity_rows = []
    recent_rows = git_data.get("recent_working_tree_changes") if isinstance(git_data, dict) else []
    if isinstance(recent_rows, list):
        for row in recent_rows:
            activity_rows.append(
                "<tr>"
                f"<td><code>{html.escape(_fmt(row.get('status'), ''))}</code></td>"
                f"<td><code>{html.escape(_fmt(row.get('path')))}</code></td>"
                f"<td><code>{html.escape(_fmt(row.get('mtime_local'), 'missing'))}</code></td>"
                f"<td><code>{html.escape(_fmt(row.get('size_bytes'), '0'))}</code></td>"
                "</tr>"
            )
    if not activity_rows:
        activity_rows.append("<tr><td colspan='4'>No recent working file activity found.</td></tr>")

    recent_project_activity_rows = []
    daily_counts = _recent_activity_daily_counts(recent_project_activity, max_days=7)
    for row in recent_project_activity:
        recent_project_activity_rows.append(
            "<tr>"
            f"<td><code>{html.escape(_fmt(row.get('mtime_local'), 'missing'))}</code></td>"
            f"<td><code>{html.escape(_fmt(row.get('path')))}</code></td>"
            f"<td><code>{html.escape(_fmt(row.get('size_bytes'), '0'))}</code></td>"
            "</tr>"
        )
    if not recent_project_activity_rows:
        recent_project_activity_rows.append("<tr><td colspan='3'>No project artifacts changed in this time window.</td></tr>")

    preflight_rows = []
    for event in ops_data["preflight_events"]:
        preflight_rows.append(
            "<tr>"
            f"<td><code>{html.escape(event.get('stamp', ''))}</code></td>"
            f"<td>{html.escape(event.get('result', ''))}</td>"
            f"<td>{html.escape(event.get('detail', ''))}</td>"
            "</tr>"
        )
    if not preflight_rows:
        preflight_rows.append("<tr><td colspan='3'>No preflight events found.</td></tr>")

    live_timeline_rows = []
    for idx, row in enumerate(live_timeline, start=1):
        live_timeline_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{html.escape(_fmt(row.get('date_local')))}</code></td>"
            f"<td>{html.escape(_fmt(row.get('kind')))}</td>"
            f"<td><code>{html.escape(_fmt(row.get('label')))}</code></td>"
            f"<td>{html.escape(_fmt(row.get('detail')))}</td>"
            "</tr>"
        )
    if not live_timeline_rows:
        live_timeline_rows.append("<tr><td colspan='5'>No live timeline events found.</td></tr>")

    milestone_rows = []
    for idx, row in enumerate(major_milestones, start=1):
        milestone_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{html.escape(_fmt(row.get('date_local')))}</code></td>"
            f"<td>{html.escape(_fmt(row.get('area')))}</td>"
            f"<td>{html.escape(_fmt(row.get('title')))}</td>"
            f"<td>{html.escape(_fmt(row.get('detail')))}</td>"
            f"<td><code>{html.escape(_fmt(row.get('reference')))}</code></td>"
            "</tr>"
        )
    if not milestone_rows:
        milestone_rows.append("<tr><td colspan='6'>No milestone-level events identified.</td></tr>")

    significant_change_rows = []
    for idx, row in enumerate(significant_changes, start=1):
        significant_change_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{html.escape(_fmt(row.get('date_local')))}</code></td>"
            f"<td>{html.escape(_fmt(row.get('area')))}</td>"
            f"<td>{html.escape(_fmt(row.get('title')))}</td>"
            f"<td>{html.escape(_fmt(row.get('detail')))}</td>"
            f"<td><code>{html.escape(_fmt(row.get('reference')))}</code></td>"
            "</tr>"
        )
    if not significant_change_rows:
        significant_change_rows.append("<tr><td colspan='6'>No significant recent changes identified.</td></tr>")

    def li(label: str, value: Any) -> str:
        return f"<li><b>{html.escape(label)}:</b> <code>{html.escape(_fmt(value))}</code></li>"

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Project Timeline Report</title>
  <style>
    :root {{
      --text: #111;
      --muted: #555;
      --line: #ddd;
      --bg: #fff;
    }}
    body {{
      color: var(--text);
      background: var(--bg);
      font-family: \"Georgia\", \"Times New Roman\", serif;
      margin: 24px;
      line-height: 1.4;
    }}
    h1, h2, h3 {{
      margin: 0.6em 0 0.35em;
      page-break-after: avoid;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 16px;
    }}
    code {{
      font-family: \"Menlo\", \"Consolas\", monospace;
      font-size: 0.92em;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0 18px;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      font-size: 0.95rem;
    }}
    th {{
      text-align: left;
      background: #f7f7f7;
    }}
    ul {{
      margin-top: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .panel {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      page-break-inside: avoid;
    }}
    @media print {{
      body {{
        margin: 0.5in;
      }}
      a {{
        color: inherit;
        text-decoration: none;
      }}
    }}
  </style>
</head>
<body>
  <h1>Project Timeline Report</h1>
  <div class=\"meta\">
    Generated (UTC): <code>{html.escape(context['generated_utc'])}</code><br>
    Generated (Local): <code>{html.escape(context['generated_local'])}</code><br>
    Project root: <code>{html.escape(str(PROJECT_ROOT))}</code>
  </div>

  <h2>Snapshot</h2>
  <ul>
    {li("Git available", git_data["available"])}
    {li("Branch", git_data["branch"])}
    {li("HEAD", git_data["head"])}
    {li("Total commits", git_data["commit_count"])}
    {li("First commit", f"{git_data['commits'][0]['date']} {git_data['commits'][0]['sha']} {git_data['commits'][0]['subject']}" if git_data["commits"] else "n/a")}
    {li("Latest commit", f"{git_data['commits'][-1]['date']} {git_data['commits'][-1]['sha']} {git_data['commits'][-1]['subject']}" if git_data["commits"] else "n/a")}
  </ul>

  <h2>Major Milestones</h2>
  <table>
    <thead><tr><th>#</th><th>Date (Local)</th><th>Area</th><th>Milestone</th><th>Detail</th><th>Reference</th></tr></thead>
    <tbody>
      {"".join(milestone_rows)}
    </tbody>
  </table>

  <h2>Significant Recent Changes</h2>
  <table>
    <thead><tr><th>#</th><th>Date (Local)</th><th>Area</th><th>Change</th><th>Detail</th><th>Reference</th></tr></thead>
    <tbody>
      {"".join(significant_change_rows)}
    </tbody>
  </table>

  {"<h2>Detailed Recent Timeline</h2><table><thead><tr><th>#</th><th>Date (Local)</th><th>Type</th><th>Reference</th><th>Detail</th></tr></thead><tbody>" + "".join(live_timeline_rows) + "</tbody></table>" if show_detailed_timeline else ""}

  <h2>Working Tree</h2>
  <div class=\"grid\">
    <div class=\"panel\">
      <ul>
        {li("Modified", counts["modified"])}
        {li("Added", counts["added"])}
        {li("Deleted", counts["deleted"])}
        {li("Renamed", counts["renamed"])}
        {li("Untracked", counts["untracked"])}
        {li("Other", counts["other"])}
      </ul>
    </div>
    <div class=\"panel\">
      <div><b>Branch status</b></div>
      <div><code>{html.escape(git_data["status_branch_line"] or "n/a")}</code></div>
    </div>
  </div>
  <table>
    <thead><tr><th>Files</th></tr></thead>
    <tbody>
      {"".join(status_rows)}
    </tbody>
  </table>

  <h3>Recent Working File Activity (mtime local)</h3>
  <table>
    <thead><tr><th>Status</th><th>Path</th><th>Modified (Local)</th><th>Size Bytes</th></tr></thead>
    <tbody>
      {"".join(activity_rows)}
    </tbody>
  </table>

  <h2>Recent Project Activity (Last {recent_project_hours} Hours)</h2>
  <div><b>Daily file-change counts (local):</b> {html.escape(", ".join([f"{row['date']}={row['count']}" for row in daily_counts]) if daily_counts else "n/a")}</div>
  <table>
    <thead><tr><th>Modified (Local)</th><th>Path</th><th>Size Bytes</th></tr></thead>
    <tbody>
      {"".join(recent_project_activity_rows)}
    </tbody>
  </table>

  <h2>Runtime and Gates</h2>
  <ul>
    {li("Promotion gate promote_ok", promotion.get("promote_ok"))}
    {li("Promotion fail_share", promotion.get("fail_share"))}
    {li("Promotion failed/considered", f"{_fmt(promotion.get('failed_bots'))}/{_fmt(promotion.get('considered_bots'))}")}
    {li("Graduation ok", graduation.get("ok"))}
    {li("Graduation mature_pass_rate", maturity.get("mature_pass_rate"))}
    {li("Graduation immature_active_count", graduation.get("immature_active_count"))}
    {li("Retrain status_counts", retrain.get("status_counts"))}
    {li("Retrain master_update_status", retrain.get("master_update_status"))}
    {li("Leak/overfit ok", leak.get("ok"))}
    {li("Leak/overfit counts", leak.get("counts"))}
    {li("Preflight preflight_ok", preflight.get("preflight_ok"))}
    {li("Storage mode", storage.get("mode"))}
    {li("Storage active_root", storage.get("active_root"))}
    {li("Latest all_sleeves log", ops_data.get("latest_all_sleeves_log"))}
    {li("Latest coinbase log", ops_data.get("latest_coinbase_log"))}
  </ul>

  <h2>Preflight Milestones (from logs)</h2>
  <table>
    <thead><tr><th>Stamp</th><th>Result</th><th>First Fail Detail</th></tr></thead>
    <tbody>
      {"".join(preflight_rows)}
    </tbody>
  </table>

  <h2>Git Commit History</h2>
  <table>
    <thead><tr><th>#</th><th>Date</th><th>SHA</th><th>Subject</th></tr></thead>
    <tbody>
      {"".join(commit_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    return html_doc


def _load_state(path: Path) -> Dict[str, Any]:
    return _load_json(path)


def _save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full project timeline reports (markdown + printable HTML).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--auto", action="store_true", help="Skip regeneration when signature is unchanged.")
    parser.add_argument("--force", action="store_true", help="Regenerate even when signature is unchanged.")
    parser.add_argument(
        "--prune-auto",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PROJECT_TIMELINE_PRUNE_AUTO", "1").strip() == "1",
        help="Prune old timestamped timeline snapshots after report generation.",
    )
    parser.add_argument(
        "--prune-older-days",
        type=int,
        default=int(os.getenv("PROJECT_TIMELINE_PRUNE_OLDER_DAYS", os.getenv("RETENTION_PROJECT_TIMELINE_DAYS", "30"))),
        help="Only prune timestamped runs older than this many days.",
    )
    parser.add_argument(
        "--prune-keep-runs",
        type=int,
        default=int(os.getenv("PROJECT_TIMELINE_PRUNE_KEEP_RUNS", os.getenv("RETENTION_PROJECT_TIMELINE_KEEP_RUNS", "40"))),
        help="Always keep at least this many most-recent timestamped timeline runs.",
    )
    parser.add_argument(
        "--activity-hours",
        type=int,
        default=int(os.getenv("PROJECT_TIMELINE_ACTIVITY_HOURS", "72")),
        help="Include project activity rows modified within this local-time window.",
    )
    parser.add_argument(
        "--activity-limit",
        type=int,
        default=int(os.getenv("PROJECT_TIMELINE_ACTIVITY_LIMIT", "600")),
        help="Maximum number of recent project activity rows to include.",
    )
    parser.add_argument(
        "--include-detailed-timeline",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PROJECT_TIMELINE_INCLUDE_DETAILED_TIMELINE", "0").strip() == "1",
        help="Include the raw detailed recent timeline section in the report output.",
    )
    parser.add_argument(
        "--render-pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render a PDF alongside the markdown/html outputs. Auto mode defaults to off unless explicitly enabled.",
    )
    parser.add_argument(
        "--allow-gui-pdf-renderer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow GUI browser app bundles for PDF rendering when no CLI renderer is available.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.render_pdf is None:
        render_pdf = _env_flag("PROJECT_TIMELINE_AUTO_RENDER_PDF", "0") if args.auto else _env_flag("PROJECT_TIMELINE_RENDER_PDF", "1")
    else:
        render_pdf = bool(args.render_pdf)

    if args.allow_gui_pdf_renderer is None:
        allow_gui_pdf_renderer = _env_flag("PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER", "0")
    else:
        allow_gui_pdf_renderer = bool(args.allow_gui_pdf_renderer)

    generated_utc = datetime.now(timezone.utc).isoformat()
    generated_local = datetime.now().astimezone().isoformat()

    git_data = _collect_git()
    ops_data = _collect_ops_snapshot()
    recent_project_activity = _collect_recent_project_activity(
        hours=int(args.activity_hours),
        limit=int(args.activity_limit),
    )
    ops_data["recent_project_activity"] = recent_project_activity
    ops_data["recent_project_activity_probe"] = _recent_project_activity_probe(recent_project_activity)
    ops_data["recent_project_activity_hours"] = int(args.activity_hours)
    ops_data["include_detailed_timeline"] = bool(args.include_detailed_timeline)
    signature = _build_signature(git_data, ops_data)

    context = {
        "generated_utc": generated_utc,
        "generated_local": generated_local,
        "signature": signature,
        "git": git_data,
        "ops": ops_data,
        "include_detailed_timeline": bool(args.include_detailed_timeline),
        "render_pdf_enabled": bool(render_pdf),
        "allow_gui_pdf_renderer": bool(allow_gui_pdf_renderer),
    }

    out_dir = Path(args.output_dir)
    state_file = Path(args.state_file)
    latest_md = out_dir / "project_timeline_latest.md"
    latest_html = out_dir / "project_timeline_print_latest.html"
    latest_pdf = out_dir / "project_timeline_latest.pdf"

    state = _load_state(state_file)
    unchanged = (
        args.auto
        and not args.force
        and state.get("signature") == signature
        and latest_md.exists()
        and latest_html.exists()
        and ((not render_pdf) or latest_pdf.exists())
    )

    if unchanged:
        prune_summary = (
            _prune_timeline_snapshots(out_dir, keep_runs=int(args.prune_keep_runs), older_than_days=int(args.prune_older_days))
            if args.prune_auto
            else {"enabled": False}
        )
        payload = {
            "changed": False,
            "signature": signature,
            "latest_markdown": str(latest_md),
            "latest_printable_html": str(latest_html),
            "latest_pdf": str(latest_pdf) if latest_pdf.exists() else "",
            "generated_utc": generated_utc,
            "prune": prune_summary,
            "activity_hours": int(args.activity_hours),
            "activity_limit": int(args.activity_limit),
            "recent_activity_rows": len(recent_project_activity),
            "include_detailed_timeline": bool(args.include_detailed_timeline),
            "render_pdf_enabled": bool(render_pdf),
            "allow_gui_pdf_renderer": bool(allow_gui_pdf_renderer),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                "project_timeline_report unchanged "
                f"latest_markdown={latest_md} latest_printable_html={latest_html}"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ts_md = out_dir / f"project_timeline_{stamp}.md"
    ts_html = out_dir / f"project_timeline_print_{stamp}.html"
    ts_pdf = out_dir / f"project_timeline_{stamp}.pdf"

    md_text = _render_markdown(context)
    html_text = _render_html(context)

    latest_md.write_text(md_text, encoding="utf-8")
    latest_html.write_text(html_text, encoding="utf-8")
    ts_md.write_text(md_text, encoding="utf-8")
    ts_html.write_text(html_text, encoding="utf-8")

    if latest_pdf.exists():
        latest_pdf.unlink()
    if ts_pdf.exists():
        ts_pdf.unlink()

    pdf_ok = False
    pdf_detail = "pdf_render_disabled"
    if render_pdf:
        pdf_ok, pdf_detail = _render_pdf_from_html(
            latest_html,
            latest_pdf,
            allow_gui_renderer=allow_gui_pdf_renderer,
        )
        if pdf_ok:
            try:
                shutil.copy2(latest_pdf, ts_pdf)
            except Exception as exc:
                pdf_ok = False
                pdf_detail = f"timestamp_pdf_copy_failed:{exc}"
                ts_pdf = None
        else:
            ts_pdf = None
    else:
        ts_pdf = None

    prune_summary = (
        _prune_timeline_snapshots(out_dir, keep_runs=int(args.prune_keep_runs), older_than_days=int(args.prune_older_days))
        if args.prune_auto
        else {"enabled": False}
    )

    state_payload = {
        "signature": signature,
        "generated_utc": generated_utc,
        "latest_markdown": str(latest_md),
        "latest_printable_html": str(latest_html),
        "latest_pdf": str(latest_pdf) if latest_pdf.exists() else "",
        "timestamped_markdown": str(ts_md),
        "timestamped_printable_html": str(ts_html),
        "timestamped_pdf": str(ts_pdf) if ts_pdf is not None else "",
        "pdf_ok": bool(pdf_ok),
        "pdf_detail": str(pdf_detail),
        "head": git_data.get("head"),
        "branch": git_data.get("branch"),
        "commit_count": git_data.get("commit_count"),
        "prune": prune_summary,
        "activity_hours": int(args.activity_hours),
        "activity_limit": int(args.activity_limit),
        "recent_activity_rows": len(recent_project_activity),
        "include_detailed_timeline": bool(args.include_detailed_timeline),
        "render_pdf_enabled": bool(render_pdf),
        "allow_gui_pdf_renderer": bool(allow_gui_pdf_renderer),
    }
    _save_state(state_file, state_payload)

    payload = {
        "changed": True,
        "signature": signature,
        "latest_markdown": str(latest_md),
        "latest_printable_html": str(latest_html),
        "latest_pdf": str(latest_pdf) if latest_pdf.exists() else "",
        "timestamped_markdown": str(ts_md),
        "timestamped_printable_html": str(ts_html),
        "timestamped_pdf": str(ts_pdf) if ts_pdf is not None else "",
        "pdf_ok": bool(pdf_ok),
        "pdf_detail": str(pdf_detail),
        "generated_utc": generated_utc,
        "prune": prune_summary,
        "activity_hours": int(args.activity_hours),
        "activity_limit": int(args.activity_limit),
        "recent_activity_rows": len(recent_project_activity),
        "include_detailed_timeline": bool(args.include_detailed_timeline),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"Wrote: {ts_md}")
        print(f"Wrote: {ts_html}")
        if ts_pdf is not None:
            print(f"Wrote: {ts_pdf}")
        print(f"Latest MD: {latest_md}")
        print(f"Latest HTML: {latest_html}")
        if latest_pdf.exists():
            print(f"Latest PDF: {latest_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
