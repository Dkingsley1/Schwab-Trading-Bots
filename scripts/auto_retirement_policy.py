import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-retire underperforming bots using registry + walk-forward signals.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--walk-forward", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-accuracy", type=float, default=0.515)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    reg_path = Path(args.registry)
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    wf = {}
    wf_path = Path(args.walk_forward)
    if wf_path.exists():
        wf = json.loads(wf_path.read_text(encoding="utf-8")).get("bots", {})

    changed = []
    for row in reg.get("sub_bots", []):
        if not row.get("active"):
            continue
        acc = row.get("test_accuracy")
        if acc is None:
            continue

        retire_reason = None
        if float(acc) < args.min_accuracy:
            retire_reason = f"auto_retire_accuracy_below_{args.min_accuracy:.3f}"

        wf_row = wf.get(str(row.get("bot_id")), {})
        if wf_row.get("status") == "fail":
            retire_reason = "auto_retire_walk_forward_fail"

        if retire_reason:
            row["active"] = False
            row["reason"] = retire_reason
            row["weight"] = 0.0
            changed.append({"bot_id": row.get("bot_id"), "reason": retire_reason})

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "changed": changed,
        "count": len(changed),
        "applied": bool(args.apply),
    }

    if args.apply:
        backup = reg_path.with_name(f"master_bot_registry.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        backup.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
