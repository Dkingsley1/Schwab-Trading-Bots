#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import iso_now, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "one_numbers_regression_guard_latest.json"
SUMMARY_PATH = PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json"
LATEST_CSV_PATH = PROJECT_ROOT / "exports" / "one_numbers" / "latest.csv"
LATEST_METRICS_CSV_PATH = PROJECT_ROOT / "exports" / "one_numbers" / "latest_metrics.csv"
HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "one_numbers_rollup_history.json"
RUNTIME_THROTTLE_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"
ASSIGNED_DRIFT_BOT = "system_drift_autopilot"
OWNER_BOT = "infrastructure_autofix_bot"
KEY_COLLAPSE_METRICS = (
    "decision_total_rows",
    "blocked_total",
    "data_blocked_total",
    "risk_blocked_total",
)
START_DAY_ENV_NAMES = (
    "ONE_NUMBERS_ORIGINAL_START_DAY",
    "ONE_NUMBERS_EXPECTED_START_DAY",
    "INFRA_SUPERVISOR_ONE_NUMBERS_START_DAY",
)
SOURCE_DAY_RE = re.compile(r"(20\d{6})")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _normalize_day(raw: Any) -> str:
    text = str(raw or "").strip().replace("-", "")
    if not re.fullmatch(r"20\d{6}", text):
        return ""
    try:
        datetime.strptime(text, "%Y%m%d")
    except Exception:
        return ""
    return text


def _session_open(now_utc: datetime) -> bool:
    tz_name = str(os.getenv("ONE_NUMBERS_SESSION_TIMEZONE", "America/New_York") or "America/New_York")
    start_text = str(os.getenv("ONE_NUMBERS_SESSION_START", "09:30") or "09:30")
    end_text = str(os.getenv("ONE_NUMBERS_SESSION_END", "16:00") or "16:00")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    local_now = now_utc.astimezone(tz)
    if local_now.weekday() >= 5:
        return False
    try:
        start_hour, start_minute = [int(part) for part in start_text.split(":", 1)]
        end_hour, end_minute = [int(part) for part in end_text.split(":", 1)]
    except Exception:
        start_hour, start_minute, end_hour, end_minute = 9, 30, 16, 0
    current_minutes = (local_now.hour * 60) + local_now.minute
    start_minutes = (start_hour * 60) + start_minute
    end_minutes = (end_hour * 60) + end_minute
    return bool(start_minutes <= current_minutes < end_minutes)


def _timeframe_bucket(summary: dict[str, Any], prefix: str) -> dict[str, int]:
    return {
        "decision_total_rows": _safe_int(summary.get(f"{prefix}decision_total_rows"), 0),
        "governance_total_rows": _safe_int(summary.get(f"{prefix}governance_total_rows"), 0),
        "blocked_total": _safe_int(summary.get(f"{prefix}blocked_total"), 0),
        "data_blocked_total": _safe_int(summary.get(f"{prefix}data_blocked_total"), 0),
        "risk_blocked_total": _safe_int(summary.get(f"{prefix}risk_blocked_total"), 0),
        "paper_executed_total": _safe_int(summary.get(f"{prefix}paper_executed_total"), 0),
        "watchdog_restarts": _safe_int(summary.get(f"{prefix}watchdog_restarts"), 0),
    }


def _current_bucket(summary: dict[str, Any]) -> dict[str, int]:
    return {
        "decision_total_rows": _safe_int(summary.get("combined_decision_total_rows"), 0),
        "governance_total_rows": _safe_int(summary.get("combined_governance_total_rows"), 0),
        "blocked_total": _safe_int(summary.get("combined_blocked_total"), 0),
        "data_blocked_total": _safe_int(summary.get("data_blocked_total"), 0),
        "risk_blocked_total": _safe_int(summary.get("risk_blocked_total"), 0),
        "paper_executed_total": _safe_int(summary.get("paper_executed_total"), 0),
        "watchdog_restarts": _safe_int(summary.get("watchdog_restarts"), 0),
    }


def _history_by_day(history_payload: dict[str, Any]) -> dict[str, Any]:
    history = history_payload.get("history_by_day") if isinstance(history_payload.get("history_by_day"), dict) else history_payload
    if not isinstance(history, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_day, row in history.items():
        day = _normalize_day(raw_day)
        if day:
            out[day] = row
    return out


def _history_days(history_payload: dict[str, Any]) -> int:
    return len(_history_by_day(history_payload))


def _expected_start_day(project_root: Path) -> tuple[str, str]:
    for env_name in START_DAY_ENV_NAMES:
        day = _normalize_day(os.getenv(env_name))
        if day:
            return day, f"env:{env_name}"
    config_path = project_root / "config" / "one_numbers_start_day.txt"
    try:
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            day = _normalize_day(line)
            if day:
                return day, str(config_path)
    except Exception:
        pass
    return "", ""


def _storage_active_root(project_root: Path) -> Path | None:
    route_status = _load_json(project_root / "governance" / "health" / "storage_route_status_latest.json")
    raw_root = str(route_status.get("active_root") or "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).expanduser()
    return root if root.exists() else None


def _source_scan_roots(project_root: Path) -> list[Path]:
    raw_roots = [
        project_root / "decision_explanations",
        project_root / "decisions",
        project_root / "governance",
        project_root / "local_fallback_storage" / "decision_explanations",
        project_root / "local_fallback_storage" / "decisions",
        project_root / "local_fallback_storage" / "governance",
    ]
    active_root = _storage_active_root(project_root)
    if active_root is not None:
        raw_roots.extend([active_root / "decision_explanations", active_root / "decisions", active_root / "governance"])
    roots: list[Path] = []
    seen: set[str] = set()
    for root in raw_roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen or not root.exists():
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _source_day_set(project_root: Path) -> set[str]:
    limit = max(_safe_int(os.getenv("ONE_NUMBERS_SOURCE_SCAN_FILE_LIMIT"), 50000), 1)
    days: set[str] = set()
    scanned = 0
    for root in _source_scan_roots(project_root):
        try:
            iterator = root.rglob("*")
        except Exception:
            continue
        for path in iterator:
            if scanned >= limit:
                return days
            try:
                if not path.is_file():
                    continue
            except Exception:
                continue
            scanned += 1
            name = path.name
            if ".jsonl" not in name:
                continue
            match = SOURCE_DAY_RE.search(name)
            if not match:
                continue
            day = _normalize_day(match.group(1))
            if day:
                days.add(day)
    return days


def _bounded_days(days: set[str], *, limit: int = 12) -> list[str]:
    return sorted(days)[: max(int(limit), 0)]


def _csv_alias_ok(path: Path) -> bool:
    return bool(path.exists() and path.is_file())


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    summary = _load_json(project_root / "exports" / "one_numbers" / "one_numbers_summary.json")
    history_payload = _load_json(project_root / "governance" / "health" / "one_numbers_rollup_history.json")
    runtime_throttle = _load_json(project_root / "governance" / "health" / "runtime_throttle_control_latest.json")
    history_by_day = _history_by_day(history_payload)
    history_days_available = _history_days(history_payload)
    history_day_set = set(history_by_day)
    source_day_set = _source_day_set(project_root)
    expected_start_day, expected_start_source = _expected_start_day(project_root)
    requested_day = str(summary.get("requested_day") or "").strip()
    resolved_day = str(summary.get("resolved_day") or summary.get("day_utc") or "").strip()
    report_mode = str(summary.get("report_mode") or "").strip()
    day_fallback_applied = bool(requested_day and resolved_day and requested_day != resolved_day)
    month_days = _safe_int(summary.get("month_to_date_days_covered"), 0)
    all_time_days = _safe_int(summary.get("all_time_days_covered"), 0)

    current_bucket = _current_bucket(summary)
    month_bucket = _timeframe_bucket(summary, "month_to_date_")
    all_time_bucket = _timeframe_bucket(summary, "all_time_")
    identical_collapse_metrics = [
        key
        for key in KEY_COLLAPSE_METRICS
        if current_bucket.get(key) == month_bucket.get(key) == all_time_bucket.get(key)
    ]
    timeframe_collapse_detected = bool(
        not day_fallback_applied
        and month_days > 1
        and all_time_days > 1
        and len(identical_collapse_metrics) == len(KEY_COLLAPSE_METRICS)
    )

    weaknesses: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []
    if not summary:
        weaknesses.append({"name": "summary_missing", "summary": "one_numbers_summary.json is missing or unreadable"})
    if not _csv_alias_ok(project_root / "exports" / "one_numbers" / "latest.csv"):
        weaknesses.append({"name": "latest_csv_alias_missing", "summary": "latest.csv is missing or broken"})
    if not _csv_alias_ok(project_root / "exports" / "one_numbers" / "latest_metrics.csv"):
        weaknesses.append({"name": "latest_metrics_alias_missing", "summary": "latest_metrics.csv is missing or broken"})
    if report_mode.startswith("lightweight"):
        weaknesses.append(
            {
                "name": "lightweight_report_mode",
                "summary": f"report_mode={report_mode} is no longer allowed for One Numbers CSV output",
            }
        )
    if current_bucket["decision_total_rows"] <= 0 and current_bucket["governance_total_rows"] > 0:
        weaknesses.append(
            {
                "name": "decision_rows_missing_with_governance_activity",
                "summary": (
                    "current-day decision rows are zero while governance rows are present; "
                    "run a full rebuild and inspect the decision ingestion path"
                ),
            }
        )
    if history_days_available <= 0:
        weaknesses.append({"name": "durable_rollup_history_missing", "summary": "durable one_numbers rollup history is missing"})
    elif history_days_available < max(all_time_days, month_days):
        weaknesses.append(
            {
                "name": "thin_rollup_history",
                "summary": f"history_days_available={history_days_available} below reported rollup days={max(all_time_days, month_days)}",
            }
        )
    if requested_day and resolved_day and requested_day != resolved_day and report_mode == "lightweight_cached":
        weaknesses.append(
            {
                "name": "lightweight_fallback_day",
                "summary": f"requested_day={requested_day} resolved_day={resolved_day} under lightweight cached mode",
            }
        )
    elif day_fallback_applied:
        advisories.append(
            {
                "name": "full_fallback_day",
                "summary": f"requested_day={requested_day} resolved_day={resolved_day} under full rebuild mode",
            }
        )
    if timeframe_collapse_detected:
        weaknesses.append(
            {
                "name": "timeframe_collapse_detected",
                "summary": f"current/month/all identical for {','.join(identical_collapse_metrics)} with month_days={month_days} all_time_days={all_time_days}",
            }
        )

    earliest_history_day = min(history_day_set) if history_day_set else ""
    earliest_source_day = min(source_day_set) if source_day_set else ""
    latest_observed_day = max({day for day in [requested_day, resolved_day, *history_day_set, *source_day_set] if _normalize_day(day)}, default="")
    source_days_missing_from_history = {day for day in source_day_set if day not in history_day_set}
    if not expected_start_day:
        weaknesses.append(
            {
                "name": "one_numbers_original_start_unpinned",
                "summary": (
                    "One Numbers original coverage has no pinned start day; set ONE_NUMBERS_ORIGINAL_START_DAY "
                    "or config/one_numbers_start_day.txt so the guard can prove all-time coverage."
                ),
            }
        )
    elif earliest_history_day and earliest_history_day > expected_start_day:
        weaknesses.append(
            {
                "name": "one_numbers_history_starts_after_expected",
                "summary": f"rollup history starts at {earliest_history_day}, after expected original start {expected_start_day}",
            }
        )
    elif expected_start_day and not history_day_set:
        weaknesses.append(
            {
                "name": "one_numbers_history_missing_since_expected_start",
                "summary": f"no rollup history is present for expected original start {expected_start_day}",
            }
        )
    if expected_start_day:
        missing_source_days_since_start = {day for day in source_days_missing_from_history if day >= expected_start_day}
    else:
        missing_source_days_since_start = source_days_missing_from_history
    if missing_source_days_since_start:
        weaknesses.append(
            {
                "name": "one_numbers_source_days_missing_from_rollup",
                "summary": (
                    "raw One Numbers source days are not present in durable rollup history: "
                    f"{','.join(_bounded_days(missing_source_days_since_start))}"
                ),
                "missing_days_sample": _bounded_days(missing_source_days_since_start),
                "missing_day_count": len(missing_source_days_since_start),
            }
        )

    session_open = _session_open(now_utc)
    throttle_profile = str(runtime_throttle.get("throttle_profile") or "").strip()
    throttle_status = str(runtime_throttle.get("overall_status") or "").strip()
    host_saturation_score = _safe_int(runtime_throttle.get("host_saturation_score"), 0)
    full_refresh_blocked_by_throttle = throttle_profile == "protect_live" or throttle_status == "blocked"
    preferred_repair_mode = "full"
    if full_refresh_blocked_by_throttle:
        advisories.append(
            {
                "name": "full_refresh_throttle_guarded",
                "summary": (
                    f"runtime throttle is {throttle_status or 'unknown'} profile={throttle_profile or 'unknown'} "
                    f"host_saturation_score={host_saturation_score}"
                ),
            }
        )
    overall_status = "ready" if not weaknesses else "degraded"
    backfill_limit = max(_safe_int(os.getenv("ONE_NUMBERS_BACKFILL_COMMAND_LIMIT"), 5), 0)
    backfill_commands = [
        [
            str(PY),
            str(project_root / "scripts" / "build_one_numbers_report.py"),
            "--day",
            day,
        ]
        for day in sorted(missing_source_days_since_start)[:backfill_limit]
    ]
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "requested_day": requested_day,
        "resolved_day": resolved_day,
        "report_mode": report_mode,
        "history_days_available": history_days_available,
        "original_coverage_contract": {
            "expected_start_day": expected_start_day,
            "expected_start_source": expected_start_source,
            "config_path": str(project_root / "config" / "one_numbers_start_day.txt"),
            "earliest_history_day": earliest_history_day,
            "earliest_source_day": earliest_source_day,
            "latest_observed_day": latest_observed_day,
            "history_day_count": len(history_day_set),
            "source_day_count": len(source_day_set),
            "source_days_missing_from_history_count": len(source_days_missing_from_history),
            "source_days_missing_from_history_sample": _bounded_days(source_days_missing_from_history),
        },
        "month_to_date_days_covered": month_days,
        "all_time_days_covered": all_time_days,
        "timeframe_collapse_detected": timeframe_collapse_detected,
        "identical_collapse_metrics": identical_collapse_metrics,
        "csv_alias_status": {
            "latest_csv_ok": _csv_alias_ok(project_root / "exports" / "one_numbers" / "latest.csv"),
            "latest_metrics_csv_ok": _csv_alias_ok(project_root / "exports" / "one_numbers" / "latest_metrics.csv"),
        },
        "assigned_infrastructure_drift_bot": {
            "bot": ASSIGNED_DRIFT_BOT,
            "owner_bot": OWNER_BOT,
            "surface": "one_numbers_regression_guard",
            "mode": "full_rebuild_only",
        },
        "repair_plan": {
            "session_open": session_open,
            "full_refresh_blocked_by_throttle": full_refresh_blocked_by_throttle,
            "preferred_mode": preferred_repair_mode,
            "recommended_command": [
                str(PY),
                str(project_root / "scripts" / "build_one_numbers_report.py"),
            ],
            "backfill_commands": backfill_commands,
        },
        "runtime_throttle": {
            "overall_status": throttle_status,
            "throttle_profile": throttle_profile,
            "host_saturation_score": host_saturation_score,
        },
        "advisories": advisories,
        "weaknesses": weaknesses,
    }


def _builder_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", "scripts/build_one_numbers_report.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and bool(str(proc.stdout or "").strip())


def apply_repairs(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload.get("weaknesses"):
        return {"attempted": False, "status": "no_op"}
    if _builder_running():
        return {"attempted": False, "status": "skipped_builder_running"}
    plan = payload.get("repair_plan") if isinstance(payload.get("repair_plan"), dict) else {}
    cmd = [str(part) for part in list(plan.get("recommended_command") or []) if str(part).strip()]
    if not cmd:
        return {"attempted": False, "status": "no_command"}
    commands = [
        [str(part) for part in list(row or []) if str(part).strip()]
        for row in list(plan.get("backfill_commands") or [])
        if isinstance(row, list)
    ]
    commands.append(cmd)
    timeout_sec = max(_safe_int(os.getenv("ONE_NUMBERS_REPAIR_TIMEOUT_SECONDS"), 1800), 1)
    attempts: list[dict[str, Any]] = []
    for command in commands:
        if not command:
            continue
        try:
            proc = subprocess.run(
                command,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            timed_out = False
            rc = int(proc.returncode)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        attempts.append(
            {
                "rc": rc,
                "timed_out": timed_out,
                "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
                "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
                "command": command,
            }
        )
        if timed_out or rc not in {0, 2}:
            break
    failed = [row for row in attempts if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2}]
    degraded = [row for row in attempts if int(row.get("rc", 1)) == 2 and not bool(row.get("timed_out", False))]
    return {
        "attempted": True,
        "status": "failed" if failed else ("degraded" if degraded else "applied"),
        "attempts": attempts,
        "attempt_count": len(attempts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard the One Numbers CSV against timeframe-collapse regressions and alias drift.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    if args.apply:
        payload["repair_result"] = apply_repairs(project_root, payload)
        payload = build_payload(project_root) | {"repair_result": payload.get("repair_result", {})}
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "one_numbers_regression_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"weakness_count={len(payload.get('weaknesses') or [])}"
        )
    return 0 if payload.get("overall_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
