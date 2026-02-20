import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS_RE = re.compile(r"_(\d{8})_(\d{6})$")


def bot_id_from_log_name(name: str) -> str:
    base = name[:-5] if name.endswith(".json") else name
    m = TS_RE.search(base)
    if not m:
        return base
    return base[: m.start()]


def timestamp_from_log_name(name: str) -> datetime:
    base = name[:-5] if name.endswith(".json") else name
    m = TS_RE.search(base)
    if not m:
        return datetime.min.replace(tzinfo=timezone.utc)
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return dt.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward style validation over historical bot training logs.")
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    args = parser.parse_args()

    logs_dir = PROJECT_ROOT / "logs"
    groups = defaultdict(list)

    for p in logs_dir.glob("brain_refinery_*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        bot_id = bot_id_from_log_name(p.name)
        ts = timestamp_from_log_name(p.name)
        acc = obj.get("metrics", {}).get("test_accuracy")
        if acc is None:
            continue
        groups[bot_id].append((ts, float(acc)))

    report = {}
    for bot_id, vals in groups.items():
        vals.sort(key=lambda x: x[0])
        if len(vals) < args.min_runs:
            report[bot_id] = {"runs": len(vals), "status": "insufficient_runs"}
            continue

        split = max(int(len(vals) * 0.7), 1)
        train_part = [x[1] for x in vals[:split]]
        fwd_part = [x[1] for x in vals[split:]]

        train_mean = mean(train_part)
        fwd_mean = mean(fwd_part)
        delta = fwd_mean - train_mean
        status = "pass" if fwd_mean >= 0.52 and delta >= -0.02 else "fail"

        report[bot_id] = {
            "runs": len(vals),
            "train_mean": round(train_mean, 6),
            "forward_mean": round(fwd_mean, 6),
            "delta": round(delta, 6),
            "status": status,
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "min_runs": args.min_runs,
        "bots": report,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
