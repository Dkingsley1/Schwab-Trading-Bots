import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_unlink(path_str: str) -> bool:
    p = Path(path_str)
    if p.exists() and p.is_file():
        p.unlink()
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete/retire bots after repeated no-improvement retrains.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--min-streak", type=int, default=3)
    parser.add_argument("--delete-artifacts", action="store_true", help="Also delete model/log artifacts for retired bots.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    reg_path = Path(args.registry)
    reg = _load_registry(reg_path)

    now = datetime.now(timezone.utc)
    changed = []
    deleted_files = []

    for row in reg.get("sub_bots", []):
        streak = int(row.get("no_improvement_streak", 0) or 0)
        already_deleted = bool(row.get("deleted_from_rotation", False))

        if streak < args.min_streak and not already_deleted:
            continue

        reason = row.get("delete_reason") or f"deleted_no_improvement_{args.min_streak}_retrainings"
        row["deleted_from_rotation"] = True
        row["delete_reason"] = reason
        row["active"] = False
        row["weight"] = 0.0
        row["reason"] = reason

        changed.append(
            {
                "bot_id": row.get("bot_id"),
                "streak": streak,
                "reason": reason,
            }
        )

        if args.delete_artifacts:
            for key in ("model_path", "log_file"):
                p = row.get(key)
                if isinstance(p, str) and p.strip():
                    if _safe_unlink(p):
                        deleted_files.append(p)

    out = {
        "timestamp_utc": now.isoformat(),
        "apply": bool(args.apply),
        "min_streak": int(args.min_streak),
        "changed_count": len(changed),
        "changed": changed,
        "deleted_files_count": len(deleted_files),
        "deleted_files": deleted_files,
    }

    audit_dir = PROJECT_ROOT / "governance" / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    latest_path = audit_dir / "underperformer_prune_latest.json"

    if args.apply:
        backup = reg_path.with_name(f"master_bot_registry.backup_{now.strftime("%Y%m%d_%H%M%S")}.json")
        backup.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        latest_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
