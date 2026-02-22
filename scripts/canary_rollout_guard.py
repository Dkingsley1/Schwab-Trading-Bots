import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


def _safe_float(v: object, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return d


def main() -> int:
    parser = argparse.ArgumentParser(description="Canary rollout guard for staged promotion.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--canary-profiles", default="intraday_aggressive,swing_aggressive")
    parser.add_argument("--baseline-profiles", default="conservative,aggressive")
    parser.add_argument("--min-samples", type=int, default=400)
    parser.add_argument("--min-edge-delta", type=float, default=0.0005)
    parser.add_argument("--apply-env", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    like = f"governance/%/shadow_pnl_attribution_{args.day}.jsonl"
    rows = conn.execute(
        """
        SELECT
          COALESCE(json_extract(payload_json, '$.shadow_profile'), 'unknown') AS profile,
          COUNT(*) AS n,
          AVG(CAST(COALESCE(json_extract(payload_json, '$.pnl_proxy'), 0.0) AS REAL)) AS avg_pnl
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY profile
        """,
        (like,),
    ).fetchall()
    conn.close()

    profile_stats = {str(p): {"n": int(n), "avg_pnl": _safe_float(avg)} for p, n, avg in rows}
    canary_profiles = [x.strip() for x in args.canary_profiles.split(",") if x.strip()]
    baseline_profiles = [x.strip() for x in args.baseline_profiles.split(",") if x.strip()]

    canary_n = sum(profile_stats.get(p, {}).get("n", 0) for p in canary_profiles)
    base_n = sum(profile_stats.get(p, {}).get("n", 0) for p in baseline_profiles)

    canary_avg = sum(profile_stats.get(p, {}).get("avg_pnl", 0.0) * profile_stats.get(p, {}).get("n", 0) for p in canary_profiles) / max(canary_n, 1)
    base_avg = sum(profile_stats.get(p, {}).get("avg_pnl", 0.0) * profile_stats.get(p, {}).get("n", 0) for p in baseline_profiles) / max(base_n, 1)

    edge_delta = canary_avg - base_avg
    eligible = canary_n >= args.min_samples and base_n >= args.min_samples
    promote = eligible and edge_delta >= args.min_edge_delta

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "day": args.day,
        "canary_profiles": canary_profiles,
        "baseline_profiles": baseline_profiles,
        "canary_samples": canary_n,
        "baseline_samples": base_n,
        "canary_avg_pnl_proxy": round(canary_avg, 8),
        "baseline_avg_pnl_proxy": round(base_avg, 8),
        "edge_delta": round(edge_delta, 8),
        "eligible": eligible,
        "promote_canary": promote,
    }

    out = PROJECT_ROOT / "governance" / "health" / "canary_rollout_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.apply_env:
        env_file = PROJECT_ROOT / "governance" / "health" / "canary_rollout.env"
        weight = "0.12" if promote else "0.08"
        env_file.write_text(f"CANARY_MAX_WEIGHT={weight}\n", encoding="utf-8")

    print(f"canary_eligible={eligible} promote_canary={promote} edge_delta={edge_delta:.6f} samples_canary={canary_n} samples_baseline={base_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
