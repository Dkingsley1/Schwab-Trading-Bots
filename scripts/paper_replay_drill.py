import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.storage_router import inspect_storage_path
from scripts.paper_performance_report import _paper_source_files
from scripts.ops.long_runtime_common import write_payload

MAX_REPLAY_BYTES = 128 * 1024**2
MAX_REPLAY_LINE_BYTES = 2 * 1024**2
MAX_REPLAY_ROWS = 20000


class ReplayScan:
    def __init__(self):
        self.deadline = time.monotonic() + 45
        self.bytes_read = 0
        self.rows_read = 0
        self.errors: set[str] = set()
        self.duplicates = 0

    def rows(self, path):
        route = inspect_storage_path(path)
        if route["status"] != "present":
            self.errors.add("source_route_" + str(route["status"]))
            return
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as stream:
                while True:
                    remaining = MAX_REPLAY_BYTES - self.bytes_read
                    if time.monotonic() >= self.deadline or remaining <= 0:
                        self.errors.add("source_scan_budget_exceeded")
                        return
                    line = stream.readline(min(MAX_REPLAY_LINE_BYTES, remaining))
                    self.bytes_read += len(line)
                    if not line:
                        return
                    if not line.endswith(b"\n"):
                        self.errors.add("incomplete_or_oversized_source_row")
                        return
                    if not line.strip():
                        continue
                    self.rows_read += 1
                    if self.rows_read > MAX_REPLAY_ROWS:
                        self.errors.add("source_row_budget_exceeded")
                        return
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        self.errors.add("source_row_not_object")
                        continue
                    yield row
        except (OSError, EOFError, ValueError):
            self.errors.add("source_read_or_decode_failed")


def _external_root() -> Path | None:
    configured = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    if PROJECT_ROOT != Path(__file__).resolve().parents[1]:
        return None
    return Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")) / os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot")


def _parse_iso_utc(value: str) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _candidate_paths(in_file: str, profile: str, domain: str, *, audit=None) -> list[Path]:
    if in_file:
        return [Path(in_file).expanduser()]
    paths, _, _ = _paper_source_files(PROJECT_ROOT, audit=audit)
    profile_l = str(profile or "").strip().lower()
    domain_l = str(domain or "").strip().lower()
    return [path for path in paths
        if (path.match("paper_trades_*.jsonl") or path.match("paper_trades_*.jsonl.gz"))
        and "independent_fills" not in path.parts
        and (not profile_l or f"shadow_{profile_l}" in str(path).lower())
        and (not domain_l or f"_{domain_l}" in str(path).lower())]


def _candidate_execution_result_paths(*, audit=None) -> list[Path]:
    return _execution_paths("execution_results_", audit=audit)


def _candidate_execution_intent_paths(*, audit=None) -> list[Path]:
    return _execution_paths("execution_intents_", audit=audit)


def _execution_paths(prefix: str, *, audit=None) -> list[Path]:
    audit = audit if audit is not None else {}

    def inspect(path):
        observed = inspect_storage_path(path)
        if observed["status"] == "missing":
            audit["unmaterialized_source_count"] = audit.get("unmaterialized_source_count", 0) + 1
        elif observed["status"] != "present":
            failed(path, observed["status"])
        return observed

    def failed(path, reason):
        audit["discovery_error_count"] = audit.get("discovery_error_count", 0) + 1
        errors = audit.setdefault("discovery_errors", [])
        if len(errors) < 10:
            errors.append({"path": str(path), "reason": reason})

    roots = [PROJECT_ROOT, PROJECT_ROOT / "local_fallback_storage"]
    external = _external_root()
    if external is not None:
        roots.append(external)
    paths = {}
    for root in roots:
        folder = root / "governance/execution_lanes"
        if inspect(folder)["status"] != "present":
            continue
        try:
            for path in folder.iterdir():
                if not path.name.startswith(prefix) or not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
                    continue
                route = inspect(path)
                if route["status"] == "present":
                    paths[str(route["resolved_path"])] = path
        except OSError as exc:
            failed(folder, type(exc).__name__)
    return sorted(paths.values())


def _path_date(path: Path) -> datetime | None:
    match = re.search(r"(20\d{6})", path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _recent_paths(paths: Iterable[Path], since: datetime) -> list[Path]:
    floor = (since - timedelta(days=1)).date()
    out: list[Path] = []
    for path in paths:
        path_dt = _path_date(path)
        if path_dt is not None and path_dt.date() < floor:
            continue
        out.append(path)
    return out


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "timestamp_utc": str(row.get("timestamp_utc", "")),
        "symbol": str(row.get("symbol", "")).upper(),
        "action": str(row.get("action", "")).upper(),
        "quantity": float(row.get("quantity", 0.0) or 0.0),
        "model_score": round(float(row.get("model_score", 0.0) or 0.0), 8),
        "threshold": round(float(row.get("threshold", 0.0) or 0.0), 8),
        "strategy": str(row.get("strategy", "")),
        "fill_price": round(float(row.get("fill_price", 0.0) or 0.0), 8),
        "expected_fill_price": round(float(row.get("expected_fill_price", 0.0) or 0.0), 8),
        "realized_pnl": round(float(row.get("realized_pnl", 0.0) or 0.0), 8),
        "unrealized_pnl": round(float(row.get("unrealized_pnl", 0.0) or 0.0), 8),
        "decision_id": str(row.get("decision_id", "")),
        "parent_decision_id": str(row.get("parent_decision_id", "")),
        "run_id": str(row.get("run_id", "")),
        "iter_id": str(row.get("iter_id", "")),
        "mode": str(row.get("mode", "")),
        "metadata_bot_id": str(meta.get("bot_id", "")),
    }


def _normalize_execution_result(row: dict[str, Any]) -> dict[str, Any] | None:
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    payload = decision or intent
    if not payload:
        return None

    mode = str(row.get("mode") or intent.get("target_mode") or payload.get("mode") or "").strip().lower()
    if mode != "paper":
        return None

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if not metadata and isinstance(intent.get("metadata"), dict):
        metadata = intent.get("metadata") or {}

    timestamp = (
        str(payload.get("timestamp_utc") or "").strip()
        or str(row.get("timestamp_utc") or "").strip()
        or str(intent.get("timestamp_utc") or "").strip()
        or str(row.get("intent_created_at") or "").strip()
    )
    if not timestamp:
        return None

    return {
        "timestamp_utc": timestamp,
        "symbol": str(payload.get("symbol") or intent.get("symbol") or "").upper(),
        "action": str(payload.get("action") or intent.get("action") or "").upper(),
        "quantity": float(payload.get("quantity", intent.get("quantity", 0.0)) or 0.0),
        "model_score": round(float(payload.get("model_score", intent.get("model_score", 0.0)) or 0.0), 8),
        "threshold": round(float(payload.get("threshold", intent.get("threshold", 0.0)) or 0.0), 8),
        "strategy": str(payload.get("strategy") or intent.get("strategy") or ""),
        "fill_price": round(float(payload.get("fill_price", 0.0) or 0.0), 8),
        "expected_fill_price": round(float(payload.get("expected_fill_price", 0.0) or 0.0), 8),
        "realized_pnl": round(float(payload.get("realized_pnl", 0.0) or 0.0), 8),
        "unrealized_pnl": round(float(payload.get("unrealized_pnl", 0.0) or 0.0), 8),
        "decision_id": str(
            payload.get("decision_id")
            or row.get("message_id")
            or row.get("intent_message_id")
            or intent.get("message_id")
            or ""
        ),
        "parent_decision_id": str(
            payload.get("parent_decision_id")
            or payload.get("parent_message_id")
            or intent.get("parent_message_id")
            or ""
        ),
        "run_id": str(payload.get("run_id") or intent.get("run_id") or ""),
        "iter_id": str(payload.get("iter_id") or intent.get("iter_id") or ""),
        "mode": "paper",
        "metadata_bot_id": str(metadata.get("bot_id", "")),
    }


def _is_stale_execution_result(row: dict[str, Any]) -> bool:
    if str(row.get("mode") or "").strip().lower() != "paper":
        return False
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    return bool(
        str(row.get("result_status") or "").strip().upper() == "STALE_INTENT_SKIPPED"
        or str(result.get("reason") or "").strip().lower() == "stale_execution_intent"
    )


def _normalize_execution_intent(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("target_mode") or "").strip().lower() != "paper":
        return None
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    timestamp = str(row.get("timestamp_utc") or "").strip()
    if not timestamp:
        return None
    return {
        "timestamp_utc": timestamp,
        "symbol": str(row.get("symbol") or "").upper(),
        "action": str(row.get("action") or "").upper(),
        "quantity": float(row.get("quantity", 0.0) or 0.0),
        "model_score": round(float(row.get("model_score", 0.0) or 0.0), 8),
        "threshold": round(float(row.get("threshold", 0.0) or 0.0), 8),
        "strategy": str(row.get("strategy") or ""),
        "fill_price": 0.0,
        "expected_fill_price": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "decision_id": str(row.get("message_id") or ""),
        "parent_decision_id": str(row.get("parent_message_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "iter_id": str(row.get("iter_id") or ""),
        "mode": "paper",
        "metadata_bot_id": str(metadata.get("bot_id", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic replay drill over paper-trade logs.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--min-rows", type=int, default=int(os.getenv("PAPER_REPLAY_DRILL_MIN_ROWS", "20")))
    parser.add_argument("--profile", default="")
    parser.add_argument("--domain", default="")
    parser.add_argument("--in-file", default="")
    parser.add_argument("--strict-exit", action="store_true", default=os.getenv("PAPER_REPLAY_DRILL_STRICT_EXIT", "0").strip() == "1")
    parser.add_argument("--expected-hash", default="")
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_replay_drill_latest.json"))
    parser.add_argument("--max-fallback-rows", type=int, default=int(os.getenv("PAPER_REPLAY_DRILL_MAX_FALLBACK_ROWS", "5000")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=max(int(args.hours), 1))
    scan = ReplayScan()
    audit: dict[str, Any] = {}
    paths = _recent_paths(_candidate_paths(args.in_file, args.profile, args.domain, audit=audit), since)

    normalized: list[dict[str, Any]] = []
    seen_rows: set[str] = set()

    def append_row(row: dict[str, Any]) -> bool:
        key = json.dumps(row, sort_keys=True, allow_nan=False)
        if key in seen_rows:
            scan.duplicates += 1
            return False
        seen_rows.add(key)
        normalized.append(row)
        return True

    files_scanned = 0
    for path in paths:
        files_scanned += 1
        try:
            for row in scan.rows(path):
                ts = _parse_iso_utc(str(row.get("timestamp_utc", "")))
                if ts is None or ts > now:
                    scan.errors.add("invalid_source_timestamp")
                    continue
                if ts < since:
                    continue
                append_row(_normalize_row(row))
        except (ValueError, TypeError, OverflowError):
            scan.errors.add("invalid_source_values")

    execution_result_files_scanned = 0
    execution_result_rows = 0
    execution_result_stale_skip_rows = 0
    latest_stale_skip_ts: datetime | None = None
    execution_intent_files_scanned = 0
    execution_intent_rows = 0
    fallback_row_cap = max(int(args.max_fallback_rows), max(int(args.min_rows), 1))
    fallback_allowed = not (args.in_file or args.profile or args.domain)
    if not normalized and fallback_allowed:
        for path in _recent_paths(_candidate_execution_result_paths(audit=audit), since):
            execution_result_files_scanned += 1
            try:
                for row in scan.rows(path):
                    if _is_stale_execution_result(row):
                        execution_result_stale_skip_rows += 1
                        stale_ts = _parse_iso_utc(str(row.get("timestamp_utc") or ""))
                        if stale_ts is not None and stale_ts <= now and (latest_stale_skip_ts is None or stale_ts > latest_stale_skip_ts):
                            latest_stale_skip_ts = stale_ts
                        continue
                    replay_row = _normalize_execution_result(row)
                    if not replay_row:
                        continue
                    ts = _parse_iso_utc(str(replay_row.get("timestamp_utc", "")))
                    if ts is None or ts > now:
                        scan.errors.add("invalid_source_timestamp")
                        continue
                    if ts < since:
                        continue
                    if append_row(replay_row):
                        execution_result_rows += 1
                    if execution_result_rows >= fallback_row_cap:
                        break
            except (ValueError, TypeError, OverflowError):
                scan.errors.add("invalid_source_values")
            if execution_result_rows >= fallback_row_cap:
                break
    if not normalized and fallback_allowed:
        for path in _recent_paths(_candidate_execution_intent_paths(audit=audit), since):
            execution_intent_files_scanned += 1
            try:
                for row in scan.rows(path):
                    replay_row = _normalize_execution_intent(row)
                    if not replay_row:
                        continue
                    ts = _parse_iso_utc(str(replay_row.get("timestamp_utc", "")))
                    if ts is None or ts > now:
                        scan.errors.add("invalid_source_timestamp")
                        continue
                    if ts < since:
                        continue
                    if append_row(replay_row):
                        execution_intent_rows += 1
                    if execution_intent_rows >= fallback_row_cap:
                        break
            except (ValueError, TypeError, OverflowError):
                scan.errors.add("invalid_source_values")
            if execution_intent_rows >= fallback_row_cap:
                break

    normalized.sort(key=lambda r: (r.get("timestamp_utc", ""), r.get("decision_id", ""), r.get("symbol", "")))

    canonical = {
        "rows": normalized,
        "window_hours": int(args.hours),
        "profile": args.profile or "all",
        "domain": args.domain or "all",
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    replay_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    failed: list[str] = sorted(scan.errors)
    if audit.get("discovery_error_count", 0):
        failed.append("source_discovery_incomplete")
    if execution_intent_rows:
        failed.append("execution_intents_only_not_paper_replay")
    active_stale_skip_age_seconds = (
        max((now - latest_stale_skip_ts).total_seconds(), 0.0)
        if latest_stale_skip_ts is not None
        else None
    )
    active_stale_skip_seconds = max(float(os.getenv("PAPER_REPLAY_ACTIVE_STALE_SKIP_SECONDS", "900") or 900.0), 60.0)
    stale_skips_active = bool(
        active_stale_skip_age_seconds is not None
        and float(active_stale_skip_age_seconds) <= active_stale_skip_seconds
    )
    if len(normalized) < max(int(args.min_rows), 0):
        failed.append("paper_rows_low")
        if execution_result_stale_skip_rows > 0 and execution_result_rows == 0 and stale_skips_active:
            failed.append("stale_execution_skips_only")

    expected = str(args.expected_hash or "").strip().lower()
    hash_match = True
    if expected:
        hash_match = (replay_hash == expected)
        if not hash_match:
            failed.append("expected_hash_mismatch")

    ok = len(failed) == 0
    out = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(ok),
        "failed_checks": failed,
        "source": {
            "files_scanned": int(files_scanned),
            "execution_result_files_scanned": int(execution_result_files_scanned),
            "execution_result_rows": int(execution_result_rows),
            "execution_result_stale_skip_rows": int(execution_result_stale_skip_rows),
            "execution_result_latest_stale_skip_age_seconds": (
                round(float(active_stale_skip_age_seconds), 3)
                if active_stale_skip_age_seconds is not None
                else None
            ),
            "execution_result_stale_skips_active": bool(stale_skips_active),
            "execution_result_active_stale_skip_seconds": float(active_stale_skip_seconds),
            "execution_intent_files_scanned": int(execution_intent_files_scanned),
            "execution_intent_rows": int(execution_intent_rows),
            "fallback_row_cap": int(fallback_row_cap),
            "source_mode": (
                "execution_result_fallback"
                if execution_result_rows
                else "execution_intent_fallback" if execution_intent_rows else "paper_trades"
            ),
            "window_hours": int(args.hours),
            "since_utc": since.isoformat(),
            "bytes_read": scan.bytes_read,
            "maximum_bytes": MAX_REPLAY_BYTES,
            "duplicate_rows_excluded": scan.duplicates,
            "scan_errors": sorted(scan.errors),
            "discovery": audit,
        },
        "profile": args.profile or "all",
        "domain": args.domain or "all",
        "rows": int(len(normalized)),
        "replay_hash": replay_hash,
        "expected_hash": expected,
        "hash_match": bool(hash_match),
        "thresholds": {
            "min_rows": int(args.min_rows),
        },
    }

    out_path = Path(args.out_file)
    if inspect_storage_path(out_path)["status"] not in {"present", "missing"}:
        raise ValueError("unsafe_replay_publication_route")
    write_payload(out_path, out)

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "paper_replay_drill "
            f"ok={int(bool(out['ok']))} rows={int(out['rows'])}/{int(args.min_rows)} "
            f"hash={replay_hash}"
        )

    if expected and not hash_match:
        return 2
    if out["ok"]:
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
