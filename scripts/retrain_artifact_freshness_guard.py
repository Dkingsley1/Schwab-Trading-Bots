import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
PAPER_REPLAY_SCRIPT = PROJECT_ROOT / "scripts" / "paper_replay_drill.py"
PAPER_RECON_SCRIPT = PROJECT_ROOT / "scripts" / "paper_reconciliation_slo_guard.py"


def _load(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_ts(value: Any) -> datetime | None:
    s = str(value or "").strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _python_bin() -> str:
    if VENV_PY.exists():
        return str(VENV_PY)
    return str(Path(sys.executable))


def _run_step(name: str, cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return {
        "name": str(name),
        "cmd": [str(x) for x in cmd],
        "rc": int(proc.returncode),
        "ok": proc.returncode == 0,
        "stdout": out[-6000:],
        "stderr": err[-3000:],
    }


def _check(path: Path, max_age_min: float, require_ok: bool) -> dict[str, Any]:
    payload = _load(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    now = datetime.now(timezone.utc)
    age_min = ((now - ts).total_seconds() / 60.0) if ts else 1e9
    ok_field = bool(payload.get("ok", True))
    ok = path.exists() and (age_min <= max_age_min) and ((not require_ok) or ok_field)
    return {
        "path": str(path),
        "exists": path.exists(),
        "timestamp_utc": ts.isoformat() if ts else "",
        "age_minutes": round(float(age_min), 4),
        "ok_field": bool(ok_field),
        "ok": bool(ok),
    }


def _prune_stale(path: Path, stale_dir: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    stale_dir.mkdir(parents=True, exist_ok=True)
    now_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = stale_dir / f"{path.stem}_{now_tag}{path.suffix}"
    try:
        path.replace(archived)
    except Exception as exc:
        return {
            "path": str(path),
            "archived_to": str(archived),
            "ok": False,
            "error": str(exc),
        }
    return {
        "path": str(path),
        "archived_to": str(archived),
        "ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-fast retrain guard for stale replay/reconciliation artifacts.")
    ap.add_argument("--paper-replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_replay_drill_latest.json"))
    ap.add_argument("--paper-recon-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_reconciliation_slo_latest.json"))
    ap.add_argument("--max-age-minutes", type=float, default=180.0)
    ap.add_argument("--auto-prune-stale", action=argparse.BooleanOptionalAction, default=os.getenv("RETRAIN_FRESHNESS_AUTO_PRUNE_STALE", "1").strip() == "1")
    ap.add_argument("--auto-refresh", action=argparse.BooleanOptionalAction, default=os.getenv("RETRAIN_FRESHNESS_AUTO_REFRESH", "1").strip() == "1")
    ap.add_argument("--paper-replay-refresh-hours", type=int, default=int(os.getenv("RETRAIN_FRESHNESS_PAPER_REPLAY_HOURS", "24")))
    ap.add_argument("--stale-archive-dir", default=str(PROJECT_ROOT / "governance" / "health" / "stale_artifacts"))
    ap.add_argument("--require-ok", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "retrain_artifact_freshness_latest.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    max_age_minutes = float(args.max_age_minutes)
    require_ok = bool(args.require_ok)
    stale_archive_dir = Path(args.stale_archive_dir)
    paper_replay_file = Path(args.paper_replay_file)
    paper_recon_file = Path(args.paper_recon_file)

    checks_initial = {
        "paper_replay": _check(paper_replay_file, max_age_minutes, require_ok),
        "paper_reconciliation": _check(paper_recon_file, max_age_minutes, require_ok),
    }
    failed_initial = [k for k, v in checks_initial.items() if not bool(v.get("ok", False))]

    prune_actions: list[dict[str, Any]] = []
    if args.auto_prune_stale:
        for key, path in [("paper_replay", paper_replay_file), ("paper_reconciliation", paper_recon_file)]:
            check = checks_initial.get(key, {})
            if (not bool(check.get("ok", False))) and bool(check.get("exists", False)):
                try:
                    age_minutes = float(check.get("age_minutes", 1e9))
                except Exception:
                    age_minutes = 1e9
                if age_minutes > max_age_minutes:
                    action = _prune_stale(path, stale_archive_dir)
                    if action:
                        action["check"] = key
                        prune_actions.append(action)

    refresh_steps: list[dict[str, Any]] = []
    if args.auto_refresh and failed_initial:
        py = _python_bin()
        if "paper_replay" in failed_initial and PAPER_REPLAY_SCRIPT.exists():
            refresh_steps.append(
                _run_step(
                    "refresh_paper_replay",
                    [
                        py,
                        str(PAPER_REPLAY_SCRIPT),
                        "--hours",
                        str(max(int(args.paper_replay_refresh_hours), 1)),
                        "--json",
                    ],
                )
            )
        if "paper_reconciliation" in failed_initial and PAPER_RECON_SCRIPT.exists():
            refresh_steps.append(
                _run_step(
                    "refresh_paper_reconciliation",
                    [
                        py,
                        str(PAPER_RECON_SCRIPT),
                        "--json",
                    ],
                )
            )

    checks = {
        "paper_replay": _check(paper_replay_file, max_age_minutes, require_ok),
        "paper_reconciliation": _check(paper_recon_file, max_age_minutes, require_ok),
    }
    failed = [k for k, v in checks.items() if not bool(v.get("ok", False))]

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "max_age_minutes": max_age_minutes,
        "require_ok": require_ok,
        "auto_prune_stale": bool(args.auto_prune_stale),
        "auto_refresh": bool(args.auto_refresh),
        "paper_replay_refresh_hours": int(args.paper_replay_refresh_hours),
        "initial_failed_checks": failed_initial,
        "checks_initial": checks_initial,
        "checks": checks,
        "prune_actions": prune_actions,
        "refresh_steps": refresh_steps,
    }
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"retrain_artifact_freshness_ok={int(out['ok'])} failed_checks={','.join(failed) if failed else 'none'}")
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
