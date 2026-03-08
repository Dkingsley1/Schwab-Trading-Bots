#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_step(name: str, cmd: List[str], cwd: Path, env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    payload: Dict[str, Any] = {
        "name": name,
        "cmd": cmd,
        "rc": int(proc.returncode),
        "ok": proc.returncode == 0,
        "stdout": out[-12000:],
        "stderr": err[-6000:],
        "parsed_json": None,
    }
    try:
        if out.startswith("{") and out.endswith("}"):
            payload["parsed_json"] = json.loads(out)
    except Exception:
        payload["parsed_json"] = None
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Automated retrain orchestration with stale cleanup + gate snapshots.")
    ap.add_argument("--max-age-minutes", type=float, default=float(os.getenv("RETRAIN_ARTIFACT_MAX_AGE_MINUTES", "360")))
    ap.add_argument("--keep-backups", type=int, default=int(os.getenv("LIFECYCLE_KEEP_BACKUPS", "25")))
    ap.add_argument("--min-free-gb", type=float, default=float(os.getenv("LIFECYCLE_MIN_FREE_GB", "10.0")))
    ap.add_argument("--bypass-market-guard", action="store_true", default=os.getenv("RETRAIN_BYPASS_MARKET_GUARD", "0").strip() == "1")
    ap.add_argument("--skip-retain-prune", action="store_true")
    ap.add_argument("--skip-retrain", action="store_true")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    py = str(VENV_PY if VENV_PY.exists() else Path(sys.executable))

    env = dict(os.environ)
    if args.bypass_market_guard:
        env["RETRAIN_BYPASS_MARKET_GUARD"] = "1"

    steps: List[Dict[str, Any]] = []

    steps.append(
        _run_step(
            "sync_snapshot_health_to_sql",
            [py, str(PROJECT_ROOT / "scripts" / "sync_snapshot_health_to_sql.py"), "--json"],
            PROJECT_ROOT,
            env,
        )
    )

    if not args.skip_retain_prune:
        steps.append(
            _run_step(
                "data_retention_policy",
                [py, str(PROJECT_ROOT / "scripts" / "data_retention_policy.py"), "--json", "--apply"],
                PROJECT_ROOT,
                env,
            )
        )

    steps.append(
        _run_step(
            "model_lifecycle_hygiene",
            [
                py,
                str(PROJECT_ROOT / "scripts" / "model_lifecycle_hygiene.py"),
                "--keep-backups",
                str(int(args.keep_backups)),
                "--min-free-gb",
                f"{float(args.min_free_gb):.3f}",
                "--json",
                "--apply-prune",
                "--repair-stale-artifacts",
                "--apply-repair",
            ],
            PROJECT_ROOT,
            env,
        )
    )

    steps.append(
        _run_step(
            "retrain_artifact_freshness_guard",
            [
                py,
                str(PROJECT_ROOT / "scripts" / "retrain_artifact_freshness_guard.py"),
                "--max-age-minutes",
                f"{float(args.max_age_minutes):.3f}",
                "--json",
            ],
            PROJECT_ROOT,
            env,
        )
    )

    freshness_ok = bool(steps[-1].get("ok", False))
    if (not args.skip_retrain) and freshness_ok:
        steps.append(
            _run_step(
                "weekly_retrain",
                [py, str(PROJECT_ROOT / "scripts" / "weekly_retrain.py"), "--continue-on-error"],
                PROJECT_ROOT,
                env,
            )
        )
    elif not args.skip_retrain:
        steps.append(
            {
                "name": "weekly_retrain",
                "cmd": [py, str(PROJECT_ROOT / "scripts" / "weekly_retrain.py"), "--continue-on-error"],
                "rc": 2,
                "ok": False,
                "stdout": "",
                "stderr": "skipped: freshness_guard_failed",
                "parsed_json": None,
            }
        )

    for name, cmd in [
        ("weekly_gate_blocker_report", [py, str(PROJECT_ROOT / "scripts" / "weekly_gate_blocker_report.py"), "--json"]),
        ("walk_forward_promotion_gate", [py, str(PROJECT_ROOT / "scripts" / "walk_forward_promotion_gate.py")]),
        ("lane_promotion_gate", [py, str(PROJECT_ROOT / "scripts" / "lane_promotion_gate.py"), "--json"]),
        ("new_bot_graduation_gate", [py, str(PROJECT_ROOT / "scripts" / "new_bot_graduation_gate.py"), "--json"]),
        ("unified_lane_scorecard", [py, str(PROJECT_ROOT / "scripts" / "unified_lane_scorecard.py"), "--json"]),
    ]:
        steps.append(_run_step(name, cmd, PROJECT_ROOT, env))

    ok = all(bool(s.get("ok", False)) for s in steps if s.get("name") != "weekly_retrain")
    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(ok),
        "bypass_market_guard": bool(args.bypass_market_guard),
        "freshness_required": True,
        "steps": steps,
    }

    out_path = PROJECT_ROOT / "governance" / "health" / "retrain_orchestrator_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload))
    else:
        print(f"retrain_orchestrator ok={payload['ok']} steps={len(steps)}")
        print(out_path)

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
