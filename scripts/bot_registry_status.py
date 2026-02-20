import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "master_bot_registry.json"


def _streak_bucket(streak: int) -> str:
    if streak <= 0:
        return "0"
    if streak == 1:
        return "1"
    if streak == 2:
        return "2"
    return "3+"


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick status view for master bot registry")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--top-risk", type=int, default=10, help="How many highest-streak non-deleted bots to print")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}")
        return 2

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    sub_bots = data.get("sub_bots", [])

    total = len(sub_bots)
    active = sum(1 for b in sub_bots if bool(b.get("active")))
    deleted = sum(1 for b in sub_bots if bool(b.get("deleted_from_rotation")))
    inactive = total - active

    buckets = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for row in sub_bots:
        streak = int(row.get("no_improvement_streak", 0) or 0)
        buckets[_streak_bucket(streak)] += 1

    print(
        "REGISTRY STATUS | "
        f"total={total} | active={active} | inactive={inactive} | deleted={deleted} | "
        f"streaks:0={buckets["0"]},1={buckets["1"]},2={buckets["2"]},3+={buckets["3+"]}"
    )

    risk_rows = [
        r for r in sub_bots
        if not bool(r.get("deleted_from_rotation")) and int(r.get("no_improvement_streak", 0) or 0) > 0
    ]
    risk_rows.sort(key=lambda r: int(r.get("no_improvement_streak", 0) or 0), reverse=True)

    top_n = max(int(args.top_risk), 0)
    if top_n and risk_rows:
        print("Top at-risk bots (non-deleted):")
        for row in risk_rows[:top_n]:
            print(
                f"- {row.get("bot_id")} "
                f"streak={int(row.get("no_improvement_streak", 0) or 0)} "
                f"active={bool(row.get("active"))} "
                f"acc={row.get("test_accuracy")} "
                f"reason={row.get("reason")}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
