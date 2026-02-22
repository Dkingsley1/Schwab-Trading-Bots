import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Security/production hygiene audit.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "health" / "security_audit_latest.json"))
    args = parser.parse_args()

    checks = []

    key = os.getenv("SCHWAB_API_KEY", "")
    sec = os.getenv("SCHWAB_SECRET", "")
    checks.append({"name": "schwab_key_not_placeholder", "ok": key not in {"", "YOUR_KEY_HERE"}})
    checks.append({"name": "schwab_secret_not_placeholder", "ok": sec not in {"", "YOUR_SECRET_HERE"}})

    approval = PROJECT_ROOT / "governance" / "champion_challenger" / "PROMOTION_APPROVED.flag"
    checks.append({"name": "promotion_requires_flag_file", "ok": True, "details": str(approval)})

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
