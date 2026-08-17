#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "profitability_evidence_firewall_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "profitability_benchmark_capture_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _parse_local_time(raw: Any) -> time:
    text = str(raw or "16:05").strip()
    try:
        hour, minute = text.split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except Exception:
        return time(hour=16, minute=5)


def _source_files(project_root: Path, patterns: Iterable[Any]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for raw in patterns:
        pattern = str(raw or "").strip()
        if not pattern:
            continue
        for path in project_root.glob(pattern):
            if not path.is_file():
                continue
            identity = str(path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            files.append(path)
    return sorted(files, key=lambda path: str(path))


def _iter_tail_lines(path: Path, *, tail_bytes: int) -> Iterable[str]:
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield from handle
        except OSError:
            return
        return
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if tail_bytes > 0 and size > tail_bytes:
                handle.seek(max(size - tail_bytes, 0))
                handle.readline()
            for raw in handle:
                yield raw.decode("utf-8", errors="replace")
    except OSError:
        return


def _candidate_binding(project_root: Path) -> tuple[dict[str, Any], Any]:
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    performance = load_json(project_root / "governance" / "health" / "paper_performance_latest.json")
    window = _as_dict(performance.get("profitability_evidence_window"))
    cutoff = parse_iso_utc(window.get("candidate_cutoff_utc"))
    if cutoff is None:
        windows = _as_dict(state.get("scope_windows_started_utc"))
        values = [parse_iso_utc(value) for value in windows.values()]
        cutoff = max((value for value in values if value is not None), default=None)
    return {
        "candidate_id": str(state.get("candidate_id") or "").strip(),
        "generation": int(_safe_float(state.get("generation"), 0.0)),
        "cutoff_utc": cutoff.isoformat() if cutoff is not None else "",
        "bound": bool(state.get("candidate_id") and cutoff is not None),
    }, cutoff


def _existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _capture_candidates(
    project_root: Path,
    policy: dict[str, Any],
    *,
    cutoff: Any,
    candidate: dict[str, Any],
    now: datetime,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    capture = _as_dict(policy.get("capture"))
    symbol = str(capture.get("symbol") or "SPY").strip().upper()
    try:
        local_zone = ZoneInfo(str(capture.get("local_timezone") or "America/New_York"))
    except Exception:
        local_zone = ZoneInfo("America/New_York")
    boundary_time = _parse_local_time(capture.get("capture_after_local_time"))
    session_open_time = _parse_local_time(capture.get("session_open_local_time") or "09:30")
    now_local = now.astimezone(local_zone)
    cutoff_local = cutoff.astimezone(local_zone) if cutoff is not None else None
    minimum_quality = _safe_float(capture.get("minimum_source_quality_score"), 0.9)
    require_broker_native = bool(capture.get("require_broker_native", True))
    tail_bytes = max(int(_safe_float(capture.get("tail_bytes_per_file"), 64 * 1024 * 1024)), 0)
    files = _source_files(project_root, _as_list(capture.get("source_globs")))
    selected: dict[str, dict[str, Any]] = {}
    rows_parsed = 0
    rows_rejected = 0
    partial_candidate_days_rejected = 0
    for path in files:
        for raw in _iter_tail_lines(path, tail_bytes=tail_bytes):
            if f'"symbol": "{symbol}"' not in raw and f'"symbol":"{symbol}"' not in raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                rows_rejected += 1
                continue
            if not isinstance(row, dict) or str(row.get("symbol") or "").strip().upper() != symbol:
                continue
            rows_parsed += 1
            timestamp = parse_iso_utc(row.get("timestamp_utc") or row.get("timestamp"))
            if timestamp is None or cutoff is None or timestamp < cutoff or timestamp > now:
                rows_rejected += 1
                continue
            local_timestamp = timestamp.astimezone(local_zone)
            boundary = datetime.combine(local_timestamp.date(), boundary_time, tzinfo=local_zone)
            if local_timestamp < boundary or now_local < boundary:
                continue
            if (
                cutoff_local is not None
                and local_timestamp.date() == cutoff_local.date()
                and cutoff_local.time() > session_open_time
            ):
                partial_candidate_days_rejected += 1
                continue
            market = _as_dict(row.get("market"))
            last_price = _safe_float(market.get("last_price"), float("nan"))
            previous_close = _safe_float(market.get("prev_close"), float("nan"))
            quality = _safe_float(row.get("source_quality_score"), 0.0)
            quality_label = str(row.get("source_quality_label") or "").strip().lower()
            if (
                not math.isfinite(last_price)
                or not math.isfinite(previous_close)
                or last_price <= 0.0
                or previous_close <= 0.0
                or quality < minimum_quality
                or (require_broker_native and quality_label != "broker_native")
            ):
                rows_rejected += 1
                continue
            day = local_timestamp.date().isoformat()
            current = selected.get(day)
            if current is not None and timestamp <= current["_timestamp"]:
                continue
            cash_rate = _safe_float(policy.get("cash_annual_rate"), 0.04)
            selected[day] = {
                "schema_version": 1,
                "day_utc": day,
                "candidate_id": candidate["candidate_id"],
                "candidate_generation": candidate["generation"],
                "candidate_cutoff_utc": candidate["cutoff_utc"],
                "candidate_full_session": True,
                "symbol": symbol,
                "passive_return_bps": round((last_price / previous_close - 1.0) * 10_000.0, 8),
                "cash_return_bps": round(((1.0 + cash_rate) ** (1.0 / 252.0) - 1.0) * 10_000.0, 8),
                "benchmark_price": last_price,
                "previous_close": previous_close,
                "source_timestamp_utc": timestamp.isoformat(),
                "source_broker": str(row.get("source_broker") or row.get("broker") or ""),
                "source_provider": str(row.get("source_provider") or ""),
                "source_quality_label": quality_label,
                "source_quality_score": quality,
                "source_record_sha256": hashlib.sha256(raw.strip().encode("utf-8")).hexdigest(),
                "captured_at_utc": now.isoformat(),
                "point_in_time_immutable": True,
                "_timestamp": timestamp,
            }
    for row in selected.values():
        row.pop("_timestamp", None)
    return selected, {
        "source_file_count": len(files),
        "rows_parsed": rows_parsed,
        "rows_rejected": rows_rejected,
        "eligible_day_count": len(selected),
        "partial_candidate_days_rejected": partial_candidate_days_rejected,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    config = load_json(config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name)
    policy = _as_dict(config.get("benchmark_hurdle"))
    capture_policy = _as_dict(policy.get("capture"))
    candidate, cutoff = _candidate_binding(project_root)
    series_path = project_root / str(policy.get("series") or "governance/research/profitability_benchmark_returns.jsonl")
    selected, scan = _capture_candidates(
        project_root,
        policy,
        cutoff=cutoff,
        candidate=candidate,
        now=current_time,
    )
    appended: list[str] = []
    conflicts: list[str] = []
    existing = _existing_rows(series_path)
    existing_keys = {
        (str(row.get("candidate_id") or ""), str(row.get("day_utc") or "")): row
        for row in existing
    }
    if apply and candidate["bound"] and selected:
        series_path.parent.mkdir(parents=True, exist_ok=True)
        with series_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            locked_rows: list[dict[str, Any]] = []
            for raw in handle:
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                if isinstance(row, dict):
                    locked_rows.append(row)
            locked_keys = {
                (str(row.get("candidate_id") or ""), str(row.get("day_utc") or "")): row
                for row in locked_rows
            }
            handle.seek(0, 2)
            for day, row in sorted(selected.items()):
                key = (candidate["candidate_id"], day)
                if key in locked_keys:
                    existing_hash = str(locked_keys[key].get("source_record_sha256") or "")
                    if not existing_hash:
                        conflicts.append(day)
                    continue
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                appended.append(day)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        existing = _existing_rows(series_path)
        existing_keys = {
            (str(row.get("candidate_id") or ""), str(row.get("day_utc") or "")): row
            for row in existing
        }
    candidate_days = sorted(
        day
        for (candidate_id, day), row in existing_keys.items()
        if candidate_id == candidate["candidate_id"]
        and day
        and bool(row.get("candidate_full_session", False))
        and str(row.get("candidate_cutoff_utc") or "") == candidate["cutoff_utc"]
    )
    implementation_ready = bool(policy and capture_policy and policy.get("series"))
    blockers = []
    if not candidate["bound"]:
        blockers.append("candidate_binding_pending")
    if not candidate_days:
        blockers.append("completed_point_in_time_benchmark_day_pending")
    if conflicts:
        blockers.append("immutable_benchmark_series_conflict")
    payload = {
        "timestamp_utc": current_time.isoformat(),
        "schema_version": 1,
        "ok": implementation_ready,
        "overall_status": "ready" if candidate_days and not conflicts else "evidence_pending",
        "implementation_ready": implementation_ready,
        "candidate_binding": candidate,
        "apply": bool(apply),
        "series_path": str(series_path),
        "candidate_day_count": len(candidate_days),
        "candidate_days": candidate_days,
        "appended_days": appended,
        "conflict_days": conflicts,
        "scan": scan,
        "blockers": blockers,
        "control_contract": {
            "captures_only_after_configured_market_close_boundary": True,
            "candidate_cutoff_enforced": True,
            "mid_session_candidate_freeze_day_excluded": True,
            "broker_native_source_quality_required": True,
            "one_immutable_row_per_candidate_day": True,
            "source_record_hash_preserved": True,
            "live_execution_authority": False,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture immutable candidate-bound cash and passive benchmark evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config") / DEFAULT_CONFIG_PATH.name)
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("governance/research") / DEFAULT_OUT_PATH.name,
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    payload = build_payload(project_root, config_path=config_path, apply=args.apply)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "profitability_benchmark_capture "
            f"status={payload['overall_status']} days={payload['candidate_day_count']} "
            f"appended={len(payload['appended_days'])}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
