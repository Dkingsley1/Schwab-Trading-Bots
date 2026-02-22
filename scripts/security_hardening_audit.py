import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Security/production hygiene audit.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "health" / "security_audit_latest.json"))
    args = parser.parse_args()

    checks = []

    pre_commit = PROJECT_ROOT / ".githooks" / "pre-commit"
    checks.append({"name": "pre_commit_hook_exists", "ok": pre_commit.exists()})

    hook_text = pre_commit.read_text(encoding="utf-8") if pre_commit.exists() else ""
    checks.append({"name": "pre_commit_secret_scan_enabled", "ok": "secret_scan.py --staged" in hook_text})

    gitignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8") if (PROJECT_ROOT / ".gitignore").exists() else ""
    checks.append({"name": "token_json_ignored", "ok": "token.json" in gitignore_text})

    approval = PROJECT_ROOT / "governance" / "champion_challenger" / "PROMOTION_APPROVED.flag"
    approval_ok = False
    if approval.exists():
        try:
            obj = json.loads(approval.read_text(encoding="utf-8"))
            approval_ok = bool(obj.get("approved_by")) and bool(obj.get("approved_at_utc")) and bool(obj.get("ticket"))
        except Exception:
            approval_ok = False
    checks.append({"name": "promotion_approval_signed_json", "ok": approval_ok or not approval.exists()})

    backup_dir = PROJECT_ROOT / "exports" / "env_snapshots"
    checks.append({"name": "backup_snapshot_exists", "ok": backup_dir.exists() and any(backup_dir.iterdir())})

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
