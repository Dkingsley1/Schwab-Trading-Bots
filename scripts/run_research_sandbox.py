import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = PROJECT_ROOT / ".venv312" / "bin" / "python"


def _run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return {"cmd": cmd, "rc": p.returncode, "stdout": (p.stdout or "")[-4000:], "stderr": (p.stderr or "")[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Research sandbox pipeline (dataset + walk-forward + gate).")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "exports" / "research_sandbox" / "latest.json"))
    args = parser.parse_args()

    steps = []
    steps.append(_run([str(PY), str(PROJECT_ROOT / "scripts" / "build_trade_learning_dataset.py")]))
    steps.append(_run([str(PY), str(PROJECT_ROOT / "scripts" / "walk_forward_validate.py")]))
    steps.append(_run([str(PY), str(PROJECT_ROOT / "scripts" / "walk_forward_promotion_gate.py")]))

    ok = all(s.get("rc", 1) in {0, 2} for s in steps)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "steps": steps,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
