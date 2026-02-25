import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate canary diagnostics from promotion readiness history.")
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--latest-out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "canary_diagnostics_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(int(args.lookback_days), 1))
    rows = _read_history(Path(args.history_jsonl))

    counter: Counter[str] = Counter()
    sample_count = 0
    for row in rows:
        ts = _parse_ts(str(row.get("timestamp_utc", "")))
        if ts is None or ts < cutoff:
            continue
        sample_count += 1
        bots = row.get("failed_bots_list") if isinstance(row.get("failed_bots_list"), list) else []
        for b in bots:
            bot_id = str(b).strip().lower()
            if bot_id:
                counter[bot_id] += 1

    top = [{"bot_id": bot_id, "fail_days": n} for bot_id, n in counter.most_common(max(int(args.top_n), 1))]

    payload = {
        "timestamp_utc": now.isoformat(),
        "lookback_days": max(int(args.lookback_days), 1),
        "samples_considered": sample_count,
        "top_failing_bots": top,
        "targeted_retrain_bots": [r["bot_id"] for r in top],
    }

    out = Path(args.latest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "canary_diagnostics "
            f"samples={payload['samples_considered']} "
            f"top_targets={','.join(payload['targeted_retrain_bots'][:5]) if payload['targeted_retrain_bots'] else 'none'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
