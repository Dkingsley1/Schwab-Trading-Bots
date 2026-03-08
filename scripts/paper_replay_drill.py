import argparse
import glob
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
            with path.open("r", encoding="utf-8") as f:
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
    if len(normalized) < max(int(args.min_rows), 0):
        failed.append("paper_rows_low")

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
