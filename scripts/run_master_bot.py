import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.master_bot import MasterBot


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _canary_gate_ok(promotion_gate_file: Path, stability_file: Path) -> tuple[bool, str]:
    if not promotion_gate_file.exists():
        return False, f"missing_promotion_gate:{promotion_gate_file}"
    gate = _read_json(promotion_gate_file)
    if not bool(gate.get("promote_ok", False)):
        return False, "promotion_gate_blocked"

    if not stability_file.exists():
        return False, f"missing_stability_file:{stability_file}"
    stability = _read_json(stability_file)
    if not bool(stability.get("ok", False)):
        failed = stability.get("failed_checks") or []
        return False, f"stability_failed:{','.join(failed) if isinstance(failed, list) else 'unknown'}"

    return True, "ok"


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
    parser.add_argument("--require-canary-gate", action="store_true", default=os.getenv("REQUIRE_CANARY_PROMOTION_GATE", "1") == "1")
    parser.add_argument("--no-require-canary-gate", dest="require_canary_gate", action="store_false")
    parser.add_argument("--promotion-gate-file", default=str(Path(PROJECT_ROOT) / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--stability-file", default=str(Path(PROJECT_ROOT) / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--print-full", action="store_true")
    args = parser.parse_args()

    if args.require_canary_gate:
        ok, reason = _canary_gate_ok(Path(args.promotion_gate_file), Path(args.stability_file))
        if not ok:
            print(f"Master bot update blocked by canary gate: {reason}")
            raise SystemExit(2)

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
