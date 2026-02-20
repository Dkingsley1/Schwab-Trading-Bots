import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.master_bot import MasterBot


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/update master bot policy from sub-bot outcomes")
    parser.add_argument("--preferred-low", type=float, default=0.55)
    parser.add_argument("--preferred-high", type=float, default=0.65)
    parser.add_argument("--deactivate-below", type=float, default=0.50)
    parser.add_argument("--quality-floor", type=float, default=0.52)
    parser.add_argument("--promotion-margin", type=float, default=0.005)
    parser.add_argument("--no-improvement-retire-streak", type=int, default=3)
    parser.add_argument("--min-active-bots", type=int, default=int(os.getenv("MIN_ACTIVE_BOTS", "20")))
    parser.add_argument("--correlation-prune-threshold", type=float, default=float(os.getenv("CORRELATION_PRUNE_THRESHOLD", "0.92")))
    parser.add_argument("--print-full", action="store_true")
    args = parser.parse_args()

    master = MasterBot(
        project_root=PROJECT_ROOT,
        preferred_low=args.preferred_low,
        preferred_high=args.preferred_high,
        deactivate_below=args.deactivate_below,
        quality_floor=args.quality_floor,
        promotion_margin=args.promotion_margin,
        no_improvement_retire_streak=args.no_improvement_retire_streak,
        min_active_bots=args.min_active_bots,
        correlation_prune_threshold=args.correlation_prune_threshold,
    )

    payload = master.train_from_outcomes()

    print("Master bot registry updated")
    print(f"Registry: {os.path.join(PROJECT_ROOT, 'master_bot_registry.json')}")
    print(json.dumps(payload.get("summary", {}), indent=2))

    if args.print_full:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
