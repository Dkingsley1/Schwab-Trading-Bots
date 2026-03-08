import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _proc_count(pattern: str) -> int:
    rc, out, _ = _run(["ps", "-axo", "command"])
    if rc != 0:
        return 0
    return sum(1 for line in out.splitlines() if pattern in line)


def _latest_age_seconds(glob_pat: str) -> float | None:
    paths = sorted(PROJECT_ROOT.glob(glob_pat), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    if not paths:
        return None
    try:
        return max(time.time() - paths[0].stat().st_mtime, 0.0)
    except Exception:
        return None


def _latest_path(glob_pat: str) -> str:
    paths = sorted(PROJECT_ROOT.glob(glob_pat), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return str(paths[0]) if paths else ""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _paper_mirror_status(latest_coinbase_log: str) -> str:
    if not latest_coinbase_log:
        return "unknown"
    p = Path(latest_coinbase_log)
    if not p.exists():
        return "unknown"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[:300]
        for line in lines:
            if "[PaperMirror] enabled" in line:
                return "enabled"
        return "disabled_or_not_logged"
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick runtime doctor for loops, feeds, storage, and alerts.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    generated = datetime.now(timezone.utc).isoformat()

    latest_schwab_log = _latest_path("logs/schwab_live_*.log")
    latest_coinbase_log = _latest_path("logs/coinbase_live_*.log")

    storage = _load_json(PROJECT_ROOT / "governance" / "health" / "storage_failback_sync_latest.json")
    preflight_alert = _load_json(PROJECT_ROOT / "governance" / "alerts" / "preflight_critical_latest.json")

    payload: Dict[str, Any] = {
        "generated_utc": generated,
        "processes": {
            "all_sleeves": _proc_count("scripts/run_all_sleeves.py"),
            "parallel_shadows": _proc_count("scripts/run_parallel_shadows.py"),
            "aggressive_modes": _proc_count("scripts/run_parallel_aggressive_modes.py"),
            "coinbase_loop": _proc_count("scripts/run_shadow_training_loop.py --broker coinbase"),
            "sql_link_writer": _proc_count("scripts/ops/sql_link_writer_service.py"),
        },
        "heartbeats": {
            "schwab_age_seconds": _latest_age_seconds("governance/health/shadow_loop_*schwab*.json"),
            "coinbase_age_seconds": _latest_age_seconds("governance/health/shadow_loop_*coinbase*.json"),
        },
        "logs": {
            "schwab_latest": latest_schwab_log,
            "coinbase_latest": latest_coinbase_log,
        },
        "storage": {
            "mode": storage.get("mode", "unknown"),
            "active_root": storage.get("active_root", ""),
            "autosync": (storage.get("autosync") or {}),
        },
        "paper_mirror": {
            "status": _paper_mirror_status(latest_coinbase_log),
        },
        "alerts": {
            "preflight_critical_latest": preflight_alert,
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        p = payload["processes"]
        hb = payload["heartbeats"]
        st = payload["storage"]
        print(f"ops_doctor generated_utc={generated}")
        print(
            "processes all_sleeves={all_sleeves} parallel_shadows={parallel_shadows} "
            "aggressive_modes={aggressive_modes} coinbase_loop={coinbase_loop} sql_link_writer={sql_link_writer}".format(**p)
        )
        print(
            f"heartbeats schwab_age_s={hb['schwab_age_seconds']} coinbase_age_s={hb['coinbase_age_seconds']}"
        )
        print(
            f"storage mode={st['mode']} active_root={st['active_root']} autosync={st['autosync']}"
        )
        print(
            f"logs schwab_latest={payload['logs']['schwab_latest'] or 'none'} "
            f"coinbase_latest={payload['logs']['coinbase_latest'] or 'none'}"
        )
        print(f"paper_mirror status={payload['paper_mirror']['status']}")
        if payload["alerts"]["preflight_critical_latest"]:
            print("alerts preflight_critical_latest=present")
        else:
            print("alerts preflight_critical_latest=none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
