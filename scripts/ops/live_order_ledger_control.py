#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.live_order_ledger import LiveOrderLedger
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    from core.live_order_ledger import LiveOrderLedger
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_LEDGER_PATH = PROJECT_ROOT / "governance" / "runtime" / "live_order_ledger.sqlite3"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_order_ledger_control_latest.json"


def build_payload(project_root: Path = PROJECT_ROOT, *, ledger_path: Path | None = None) -> dict[str, Any]:
    path = ledger_path or project_root / "governance" / "runtime" / DEFAULT_LEDGER_PATH.name
    ledger_preexisting = path.exists()
    ledger = LiveOrderLedger(path)
    integrity = ledger.verify_integrity()
    unresolved = ledger.unresolved()
    submit_unknown = [row for row in unresolved if str(row.get("state") or "") == "submit_unknown"]
    cancel_unknown = [row for row in unresolved if str(row.get("state") or "") == "cancel_unknown"]
    cancel_pending = [row for row in unresolved if str(row.get("state") or "") == "cancel_pending"]
    stale_submitting = [row for row in unresolved if str(row.get("state") or "") == "submitting"]
    blockers = []
    if not integrity.get("ok", False):
        blockers.append("order_ledger_integrity_invalid")
        if not (integrity.get("event_chain") or {}).get("ok", False):
            blockers.append("order_event_chain_invalid")
    if submit_unknown:
        blockers.append("broker_submit_outcome_unknown")
    if stale_submitting:
        blockers.append("broker_submit_interrupted_before_outcome_recorded")
    if cancel_unknown:
        blockers.append("broker_cancel_outcome_unknown")
    if cancel_pending:
        blockers.append("broker_cancel_pending_reconciliation")
    status = "ready_idle" if not ledger_preexisting and not blockers else "ready" if not blockers else "blocked"
    return {
        "schema_version": 1,
        "timestamp_utc": iso_now(),
        "ok": not blockers,
        "overall_status": status,
        "ledger_path": str(path),
        "ledger_preexisting": ledger_preexisting,
        "integrity": integrity,
        "event_chain": integrity.get("event_chain", {}),
        "unresolved_intent_count": len(unresolved),
        "submit_unknown_count": len(submit_unknown),
        "submitting_count": len(stale_submitting),
        "cancel_unknown_count": len(cancel_unknown),
        "cancel_pending_count": len(cancel_pending),
        "unresolved_intents": unresolved[:100],
        "blockers": blockers,
        "live_execution_authority": False,
        "contract": {
            "stable_intent_id_required": True,
            "transactional_reservation_before_submit": True,
            "ambiguous_submit_never_auto_retried": True,
            "broker_reconciliation_required_for_unknown_submit_or_cancel": True,
            "hash_chained_order_events": True,
            "sqlite_quick_check_required": True,
            "foreign_key_integrity_required": True,
            "wal_full_sync_required": True,
            "event_state_materialization_must_match": True,
            "payload_hash_integrity_required": True,
        },
        "recommended_actions": [
            "reconcile every unknown or interrupted broker operation against broker truth before any new live order"
        ] if blockers else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the durable live-order intent and state ledger.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resolve-intent", default="")
    parser.add_argument(
        "--resolution",
        choices=("not_submitted", "open", "partially_filled", "filled", "canceled", "rejected", "expired"),
    )
    parser.add_argument("--broker-order-id", default="")
    parser.add_argument("--filled-quantity", type=float, default=0.0)
    parser.add_argument("--average-fill-price", type=float, default=0.0)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    ledger_path = args.ledger if args.ledger and args.ledger.is_absolute() else (project_root / args.ledger if args.ledger else None)
    out_path = args.out or Path("governance/health/live_order_ledger_control_latest.json")
    out_path = out_path if out_path.is_absolute() else project_root / out_path
    if str(args.resolve_intent or "").strip():
        if not args.resolution:
            parser.error("--resolve-intent requires --resolution")
        if len(str(args.evidence or "").strip()) < 12:
            parser.error("--resolve-intent requires --evidence with at least 12 characters")
        ledger = LiveOrderLedger(ledger_path or project_root / "governance" / "runtime" / DEFAULT_LEDGER_PATH.name)
        try:
            ledger.reconcile_ambiguous(
                intent_id=str(args.resolve_intent),
                resolution=str(args.resolution),
                evidence=str(args.evidence),
                broker_order_id=str(args.broker_order_id),
                filled_quantity=float(args.filled_quantity),
                average_fill_price=float(args.average_fill_price),
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
    payload = build_payload(project_root, ledger_path=ledger_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_order_ledger_control "
            f"status={payload['overall_status']} unresolved={payload['unresolved_intent_count']}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
