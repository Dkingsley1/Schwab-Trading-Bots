import argparse
import gzip
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _candidate_paths(in_file: str, profile: str, domain: str) -> list[Path]:
    if in_file:
        p = Path(in_file).expanduser().resolve()
        return [p]

    out: list[Path] = []
    patterns = [
        str(PROJECT_ROOT / "exports" / "trade_logs" / "**" / "paper_trades_*.jsonl"),
        str(PROJECT_ROOT / "paper_trades_*.jsonl"),
    ]
    profile_l = str(profile or "").strip().lower()
    domain_l = str(domain or "").strip().lower()
    for pat in patterns:
        for raw in sorted(glob.glob(pat, recursive=True)):
            rel = raw.lower()
            if profile_l and (f"shadow_{profile_l}" not in rel):
                continue
            if domain_l and (f"_{domain_l}" not in rel):
                continue
            out.append(Path(raw))
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _candidate_execution_result_paths() -> list[Path]:
    patterns = [
        str(PROJECT_ROOT / "governance" / "execution_lanes" / "execution_results_*.jsonl"),
        str(PROJECT_ROOT / "governance" / "execution_lanes" / "execution_results_*.jsonl.gz"),
        str(PROJECT_ROOT / "local_fallback_storage" / "governance" / "execution_lanes" / "execution_results_*.jsonl"),
        str(PROJECT_ROOT / "local_fallback_storage" / "governance" / "execution_lanes" / "execution_results_*.jsonl.gz"),
        str(Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/execution_lanes/execution_results_*.jsonl")),
        str(Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/execution_lanes/execution_results_*.jsonl.gz")),
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for raw in sorted(glob.glob(pat)):
            path = Path(raw)
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def _candidate_execution_intent_paths() -> list[Path]:
    patterns = [
        str(PROJECT_ROOT / "governance" / "execution_lanes" / "execution_intents_*.jsonl"),
        str(PROJECT_ROOT / "governance" / "execution_lanes" / "execution_intents_*.jsonl.gz"),
        str(PROJECT_ROOT / "local_fallback_storage" / "governance" / "execution_lanes" / "execution_intents_*.jsonl"),
        str(PROJECT_ROOT / "local_fallback_storage" / "governance" / "execution_lanes" / "execution_intents_*.jsonl.gz"),
        str(Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/execution_lanes/execution_intents_*.jsonl")),
        str(Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/execution_lanes/execution_intents_*.jsonl.gz")),
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for raw in sorted(glob.glob(pat)):
            path = Path(raw)
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


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


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


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
    paths = _candidate_paths(args.in_file, args.profile, args.domain)

    normalized: list[dict[str, Any]] = []
    files_scanned = 0
    for path in paths:
        files_scanned += 1
        try:
            with _open_text(path) as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    ts = _parse_iso_utc(str(row.get("timestamp_utc", "")))
                    if ts is None or ts < since:
                        continue
                    normalized.append(_normalize_row(row))
        except Exception:
            continue

    execution_result_files_scanned = 0
    execution_result_rows = 0
    execution_result_stale_skip_rows = 0
    latest_stale_skip_ts: datetime | None = None
    execution_intent_files_scanned = 0
    execution_intent_rows = 0
    fallback_row_cap = max(int(args.max_fallback_rows), max(int(args.min_rows), 1))
    if not normalized and not args.in_file:
        for path in _recent_paths(_candidate_execution_result_paths(), since):
            execution_result_files_scanned += 1
            try:
                with _open_text(path) as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            row = json.loads(s)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        if _is_stale_execution_result(row):
                            execution_result_stale_skip_rows += 1
                            stale_ts = _parse_iso_utc(str(row.get("timestamp_utc") or ""))
                            if stale_ts is not None and (latest_stale_skip_ts is None or stale_ts > latest_stale_skip_ts):
                                latest_stale_skip_ts = stale_ts
                            continue
                        replay_row = _normalize_execution_result(row)
                        if not replay_row:
                            continue
                        ts = _parse_iso_utc(str(replay_row.get("timestamp_utc", "")))
                        if ts is None or ts < since:
                            continue
                        normalized.append(replay_row)
                        execution_result_rows += 1
                        if execution_result_rows >= fallback_row_cap:
                            break
            except Exception:
                continue
            if execution_result_rows >= fallback_row_cap:
                break
    if not normalized and not args.in_file:
        for path in _recent_paths(_candidate_execution_intent_paths(), since):
            execution_intent_files_scanned += 1
            try:
                with _open_text(path) as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            row = json.loads(s)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        replay_row = _normalize_execution_intent(row)
                        if not replay_row:
                            continue
                        ts = _parse_iso_utc(str(replay_row.get("timestamp_utc", "")))
                        if ts is None or ts < since:
                            continue
                        normalized.append(replay_row)
                        execution_intent_rows += 1
                        if execution_intent_rows >= fallback_row_cap:
                            break
            except Exception:
                continue
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

    failed: list[str] = []
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

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
