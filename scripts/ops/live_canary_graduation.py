#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, write_payload

from core.live_canary_graduation import evaluate_live_canary_graduation
from core.live_order_ledger import LiveOrderLedger

DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "live_canary_graduation_v1.json"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "config" / "live_canary_micro_policy_v1.json"
DEFAULT_LEDGER_PATH = (
    PROJECT_ROOT / "governance" / "runtime" / "live_order_ledger.sqlite3"
)
DEFAULT_RECEIPTS_PATH = (
    PROJECT_ROOT / "governance" / "evidence" / "live_canary_closeout_receipts.jsonl"
)
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "live_canary_graduation_latest.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _project_path(project_root: Path, value: Any, default: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return default
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _load_receipts(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    receipts: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"closeout_receipts_unreadable:{type(exc).__name__}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"closeout_receipt_invalid_json:line={line_number}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"closeout_receipt_not_object:line={line_number}")
            continue
        receipts.append(payload)
    return receipts, errors


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    policy_path: Path | None = None,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
    receipts_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    policy_file = policy_path or root / "config" / DEFAULT_POLICY_PATH.name
    plan_file = plan_path or root / "config" / DEFAULT_PLAN_PATH.name
    policy = _load_json(policy_file)
    plan = _load_json(plan_file)
    evidence_policy = (
        policy.get("evidence") if isinstance(policy.get("evidence"), dict) else {}
    )
    ledger_file = ledger_path or _project_path(
        root,
        evidence_policy.get("live_order_ledger_path"),
        root / "governance" / "runtime" / DEFAULT_LEDGER_PATH.name,
    )
    receipts_file = receipts_path or _project_path(
        root,
        evidence_policy.get("closeout_receipts_path"),
        root / "governance" / "evidence" / DEFAULT_RECEIPTS_PATH.name,
    )
    candidate_file = root / "governance" / "runtime" / "production_candidate_state.json"
    candidate = _load_json(candidate_file)
    source_errors: list[str] = []
    if not policy:
        source_errors.append("live_canary_graduation_policy_missing_or_invalid")
    if not plan:
        source_errors.append("live_canary_plan_missing_or_invalid")
    receipts, receipt_errors = _load_receipts(receipts_file)
    source_errors.extend(receipt_errors)

    ledger_preexisting = ledger_file.exists()
    ledger = LiveOrderLedger(ledger_file)
    integrity = ledger.verify_integrity()
    payload = evaluate_live_canary_graduation(
        canary_plan=plan,
        graduation_policy=policy,
        order_intents=ledger.intents(),
        order_events=ledger.events(),
        closeout_receipts=receipts,
        ledger_integrity=integrity,
        source_errors=source_errors,
        current_candidate_id=str(candidate.get("candidate_id") or "").strip(),
    )
    payload.update(
        {
            "timestamp_utc": iso_now(),
            "ok": bool(payload.get("control_ok", False)),
            "sources": {
                "graduation_policy_path": str(policy_file),
                "graduation_policy_sha256": _file_sha256(policy_file),
                "canary_plan_path": str(plan_file),
                "canary_plan_sha256": _file_sha256(plan_file),
                "live_order_ledger_path": str(ledger_file),
                "live_order_ledger_preexisting": ledger_preexisting,
                "closeout_receipts_path": str(receipts_file),
                "closeout_receipts_present": receipts_file.exists(),
                "closeout_receipts_sha256": _file_sha256(receipts_file),
                "production_candidate_state_path": str(candidate_file),
            },
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fail-closed post-first-canary milestones and bounded, "
            "operator-only stage or capital review eligibility."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()

    def resolved(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else project_root / path

    payload = build_payload(
        project_root,
        policy_path=resolved(args.policy),
        plan_path=resolved(args.plan),
        ledger_path=resolved(args.ledger),
        receipts_path=resolved(args.receipts),
    )
    out_path = (
        resolved(args.out)
        or project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    )
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        stage = payload.get("stage_progression", {})
        print(
            "live_canary_graduation "
            f"status={payload.get('overall_status', 'unknown')} "
            f"phase={payload.get('phase', 'unknown')} "
            f"round_trips={payload.get('metrics', {}).get('reconciled_round_trip_count', 0)} "
            f"completed_stage={stage.get('highest_completed_stage', 0)}"
        )
    return 0 if payload.get("control_ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
