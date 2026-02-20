import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect_old_files(base: Path, older_than_days: int) -> list[Path]:
    if not base.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    out = []
    for root, _, files in os.walk(base):
        for name in files:
            p = Path(root) / name
            try:
                mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mt < cutoff:
                out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old data artifacts by retention policy.")
    parser.add_argument("--decisions-days", type=int, default=45)
    parser.add_argument("--decision-explanations-days", type=int, default=45)
    parser.add_argument("--governance-days", type=int, default=60)
    parser.add_argument("--exports-days", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    targets = {
        "decisions": (PROJECT_ROOT / "decisions", args.decisions_days),
        "decision_explanations": (PROJECT_ROOT / "decision_explanations", args.decision_explanations_days),
        "governance": (PROJECT_ROOT / "governance", args.governance_days),
        "exports": (PROJECT_ROOT / "exports", args.exports_days),
    }

    to_delete = []
    for label, (base, days) in targets.items():
        rows = _collect_old_files(base, days)
        to_delete.extend(rows)
        print(f"[{label}] candidates={len(rows)} older_than_days={days}")

    deleted = 0
    if args.apply:
        for p in to_delete:
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass

    print(f"total_candidates={len(to_delete)} deleted={deleted} apply={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
