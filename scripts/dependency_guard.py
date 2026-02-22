import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "config" / "requirements.lock.txt"
PIP_BIN = PROJECT_ROOT / ".venv312" / "bin" / "pip"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return p.returncode, p.stdout or "", p.stderr or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Dependency integrity guard.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "health" / "dependency_guard_latest.json"))
    parser.add_argument("--update-lock", action="store_true")
    args = parser.parse_args()

    checks = []
    ok = True

    checks.append({"name": "lock_exists", "ok": LOCK_PATH.exists(), "details": str(LOCK_PATH)})
    ok = ok and LOCK_PATH.exists()

    rc_check, _, err_check = _run([str(PIP_BIN), "check"])
    checks.append({"name": "pip_check", "ok": rc_check == 0, "details": (err_check.strip() or "ok")[:400]})
    ok = ok and (rc_check == 0)

    rc_freeze, out_freeze, err_freeze = _run([str(PIP_BIN), "freeze"])
    freeze_sorted = "\n".join(sorted([x.strip() for x in out_freeze.splitlines() if x.strip()])) + "\n"
    checks.append({"name": "pip_freeze_ok", "ok": rc_freeze == 0, "details": (err_freeze.strip() or "ok")[:400]})
    ok = ok and (rc_freeze == 0)

    lock_text = LOCK_PATH.read_text(encoding="utf-8") if LOCK_PATH.exists() else ""
    lock_sorted = "\n".join(sorted([x.strip() for x in lock_text.splitlines() if x.strip()])) + ("\n" if lock_text else "")

    drift = freeze_sorted != lock_sorted
    if args.update_lock and rc_freeze == 0:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(freeze_sorted, encoding="utf-8")
        drift = False

    checks.append(
        {
            "name": "lock_drift_free",
            "ok": not drift,
            "details": "match" if not drift else "requirements.lock.txt differs from current freeze",
        }
    )
    ok = ok and (not drift)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "checks": checks,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
