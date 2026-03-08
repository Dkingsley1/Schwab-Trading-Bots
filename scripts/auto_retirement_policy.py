import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_guard import check_confirmed_training_success, check_registry_row_state_before_deletion
from core.accountability import write_registry_mutation_journal


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-retire underperforming bots using registry + walk-forward signals.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--walk-forward", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-accuracy", type=float, default=0.515)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--require-training-success",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("REQUIRE_CONFIRMED_TRAINING_SUCCESS", "1").strip() == "1",
    )
    parser.add_argument(
        "--max-training-age-hours",
        type=float,
        default=float(os.getenv("CONFIRMED_TRAINING_SUCCESS_MAX_AGE_HOURS", "72")),
    )
    parser.add_argument(
        "--training-success-file",
        default=str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"),
    )
    parser.add_argument(
        "--training-scorecard-file",
        default=str(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json"),
    )
    args = parser.parse_args()

    reg_path = Path(args.registry)
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    original_reg = json.loads(json.dumps(reg))
    wf = {}
    wf_path = Path(args.walk_forward)
    if wf_path.exists():
        wf = json.loads(wf_path.read_text(encoding="utf-8")).get("bots", {})

    guard_ok = True
    guard_reason = "disabled"
    guard_details = {}
    if args.require_training_success:
        guard_ok, guard_reason, guard_details = check_confirmed_training_success(
            project_root=str(PROJECT_ROOT),
            marker_path=str(args.training_success_file),
            scorecard_path=str(args.training_scorecard_file),
            max_age_hours=float(args.max_training_age_hours),
            require_master_update=True,
            min_trained_bots=1,
        )

    changed = []
    blocked = []
    blocked_state = []
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
            state_ok, state_reason, state_details = check_registry_row_state_before_deletion(row)
            if not state_ok:
                blocked_state.append(
                    {
                        "bot_id": row.get("bot_id"),
                        "reason": retire_reason,
                        "state_reason": state_reason,
                        "state_details": state_details,
                    }
                )
                continue
            if args.require_training_success and (not guard_ok):
                blocked.append({"bot_id": row.get("bot_id"), "reason": retire_reason, "guard_reason": guard_reason})
                continue
            row["active"] = False
            row["reason"] = retire_reason
            row["weight"] = 0.0
            changed.append({"bot_id": row.get("bot_id"), "reason": retire_reason})

    apply_blocked = bool(args.apply and args.require_training_success and (not guard_ok) and bool(blocked))
    if args.apply and (not apply_blocked):
        backup = reg_path.with_name(f"master_bot_registry.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        backup.write_text(json.dumps(original_reg, indent=2), encoding="utf-8")
        reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        try:
            write_registry_mutation_journal(
                project_root=str(PROJECT_ROOT),
                actor="auto_retirement_policy",
                reason=f"apply_min_accuracy_{float(args.min_accuracy):.3f}",
                before=original_reg if isinstance(original_reg, dict) else {},
                after=reg if isinstance(reg, dict) else {},
                extra={
                    "apply": bool(args.apply),
                    "apply_blocked": bool(apply_blocked),
                    "changed_count": len(changed),
                    "blocked_count": len(blocked),
                    "blocked_state_count": len(blocked_state),
                    "backup_path": str(backup),
                },
            )
        except Exception:
            pass

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "changed": changed,
        "count": len(changed),
        "blocked": blocked,
        "blocked_count": len(blocked),
        "blocked_state": blocked_state,
        "blocked_state_count": len(blocked_state),
        "applied": bool(args.apply and (not apply_blocked)),
        "apply_blocked": apply_blocked,
        "deletion_guard": {
            "required": bool(args.require_training_success),
            "ok": bool(guard_ok),
            "reason": guard_reason,
            "details": guard_details,
        },
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
