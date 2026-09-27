from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .evaluator import POLICY_PATH, build_report, load_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "research"
    / "institutional_decision_flow"
    / "latest.json"
)
DEFAULT_LOCK_PATH = (
    PROJECT_ROOT
    / "governance"
    / "locks"
    / "institutional_decision_flow_shadow.lock"
)


def _parse_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tail_json_rows(path: Path, *, tail_bytes: int) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            offset = max(size - max(int(tail_bytes), 4096), 0)
            handle.seek(offset)
            lines = handle.read().splitlines()
    except OSError:
        return []
    if offset > 0 and lines:
        lines = lines[1:]
    rows: list[dict[str, Any]] = []
    for raw in lines:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, UnicodeError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_recent_decisions(
    project_root: Path,
    *,
    lookback_hours: float = 24.0,
    tail_bytes_per_file: int = 4 * 1024 * 1024,
    max_rows: int = 2500,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(float(lookback_hours), 0.25))
    paths = sorted(
        project_root.glob("governance/shadow_*/master_control_*.jsonl"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[:96]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _tail_json_rows(path, tail_bytes=tail_bytes_per_file):
            timestamp = _parse_utc(row.get("timestamp_utc"))
            if timestamp is None or timestamp < cutoff:
                continue
            rows.append(row)
    rows.sort(key=lambda row: _parse_utc(row.get("timestamp_utc")) or datetime.min.replace(tzinfo=timezone.utc))
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("message_id") or "").strip()
        if not key:
            key = "|".join(
                str(row.get(field) or "")
                for field in (
                    "timestamp_utc",
                    "run_id",
                    "snapshot_id",
                    "shadow_profile",
                    "routing_lane",
                    "symbol",
                )
            )
        deduplicated[key] = row
    bounded = sorted(
        deduplicated.values(),
        key=lambda row: _parse_utc(row.get("timestamp_utc")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    return bounded[-max(int(max_rows), 1) :]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _append_compact_history(project_root: Path, payload: dict[str, Any]) -> None:
    timestamp = _parse_utc(payload.get("timestamp_utc")) or datetime.now(timezone.utc)
    path = (
        project_root
        / "governance"
        / "research"
        / "institutional_decision_flow"
        / f"history_{timestamp:%Y%m%d}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "report_id": payload.get("report_id"),
        "overall_status": payload.get("overall_status"),
        "input_contract": payload.get("input_contract"),
        "decision_efficiency": payload.get("decision_efficiency"),
        "capital_scale_readiness": payload.get("capital_scale_readiness"),
        "soak_contract": payload.get("soak_contract"),
    }
    previous_tail = ""
    if path.exists():
        try:
            with path.open("rb") as handle:
                handle.seek(max(path.stat().st_size - 8192, 0))
                previous_tail = handle.read().decode("utf-8", errors="ignore")
        except OSError:
            previous_tail = ""
    if str(payload.get("report_id") or "") and str(payload.get("report_id")) in previous_tail:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(compact, ensure_ascii=True, sort_keys=True) + "\n")


@contextmanager
def _single_flight(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("institutional decision-flow shadow evaluator already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_once(
    *,
    project_root: Path = PROJECT_ROOT,
    policy_path: Path | None = None,
    out_path: Path | None = None,
    lookback_hours: float = 24.0,
    tail_bytes_per_file: int = 4 * 1024 * 1024,
    max_rows: int = 2500,
) -> dict[str, Any]:
    root = project_root.resolve()
    policy = load_policy(policy_path or POLICY_PATH)
    rows = load_recent_decisions(
        root,
        lookback_hours=lookback_hours,
        tail_bytes_per_file=tail_bytes_per_file,
        max_rows=max_rows,
    )
    report = build_report(rows, policy)
    destination = out_path or (
        root / "governance" / "research" / "institutional_decision_flow" / "latest.json"
    )
    _write_json_atomic(destination, report)
    _append_compact_history(root, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only institutional decision-flow evidence sidecar."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--tail-bytes-per-file", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-rows", type=int, default=2500)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    lock_path = root / "governance" / "locks" / "institutional_decision_flow_shadow.lock"
    started = time.perf_counter()
    try:
        with _single_flight(lock_path):
            payload = run_once(
                project_root=root,
                policy_path=args.policy,
                out_path=args.out_file,
                lookback_hours=args.lookback_hours,
                tail_bytes_per_file=args.tail_bytes_per_file,
                max_rows=args.max_rows,
            )
    except RuntimeError as exc:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "overall_status": "already_running",
            "error": str(exc),
        }
    payload["runtime_seconds"] = round(time.perf_counter() - started, 6)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        efficiency = payload.get("decision_efficiency") or {}
        print(
            "institutional_decision_flow_shadow "
            f"status={payload.get('overall_status')} "
            f"rows={(payload.get('input_contract') or {}).get('latest_profile_lane_symbol_rows', 0)} "
            f"directional={efficiency.get('directional_intent_count', 0)} "
            f"qualified={efficiency.get('qualified_shadow_candidate_count', 0)}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
