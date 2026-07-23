#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "daily_verify_auto_remediation_bot_latest.json"
DEFAULT_TIMEOUT_SEC = 120


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run(cmd: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        return 124, str(exc.stdout or "").strip(), str(exc.stderr or "").strip()


def _remediation_map() -> dict[str, list[str]]:
    return {
        "new_bot_graduation_gate": [str(PY), str(PROJECT_ROOT / "scripts" / "new_bot_graduation_gate.py"), "--json"],
        "replay_hash_registry_guard": [str(PY), str(PROJECT_ROOT / "scripts" / "replay_hash_registry_guard.py"), "--json"],
        "paper_reconciliation_slo_guard": [str(PY), str(PROJECT_ROOT / "scripts" / "paper_reconciliation_slo_guard.py"), "--json"],
        "paper_execution_calibration_report": [str(PY), str(PROJECT_ROOT / "scripts" / "paper_execution_calibration_report.py"), "--hours", "24", "--json"],
        "promotion_quality_gate": [str(PY), str(PROJECT_ROOT / "scripts" / "promotion_quality_gate.py"), "--json"],
        "state_snapshot_drill": [str(PY), str(PROJECT_ROOT / "scripts" / "daily_state_snapshot_drill.py"), "--json"],
        "db_integrity": [str(PY), str(PROJECT_ROOT / "scripts" / "sqlite_performance_maintenance.py"), "--checkpoint-only", "--json"],
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    daily_verify = _load_json(project_root / "governance" / "health" / "daily_auto_verify_latest.json")
    failed = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    remediations = _remediation_map()
    attempts: list[dict[str, Any]] = []
    resolved: list[str] = []
    unresolved: list[str] = []

    for check in failed:
        name = str(check or "").strip()
        cmd = remediations.get(name)
        if not cmd:
            unresolved.append(name)
            attempts.append({"check": name, "actionable": False, "applied": False})
            continue
        row = {"check": name, "actionable": True, "applied": bool(apply), "cmd": cmd}
        if apply:
            rc, stdout, stderr = _run(cmd, timeout_sec=timeout_sec)
            row.update({"rc": rc, "stdout": stdout[:4000], "stderr": stderr[:2000], "ok": rc == 0})
            if rc == 0:
                resolved.append(name)
            else:
                unresolved.append(name)
        else:
            row["ok"] = False
            unresolved.append(name)
        attempts.append(row)

    if apply:
        _run([str(PY), str(project_root / "scripts" / "ops" / "runtime_gate_dashboard.py"), "--json"], timeout_sec=min(timeout_sec, 60))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": len(unresolved) == 0,
        "overall_status": "ready" if len(unresolved) == 0 else ("applied_with_followups" if apply else "pending"),
        "apply": bool(apply),
        "failed_checks_seen": failed,
        "resolved_checks": resolved,
        "unresolved_checks": unresolved,
        "attempts": attempts,
        "recommended_actions": [
            "rerun the bot after daily_auto_verify failures so deterministic repairable checks are refreshed immediately",
            "keep non-mapped failures operator-reviewed until they have a safe remediation command",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Infrastructure bot that auto-remediates safe daily_auto_verify failures.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "daily_verify_auto_remediation_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"resolved_checks={len(payload.get('resolved_checks') or [])}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
