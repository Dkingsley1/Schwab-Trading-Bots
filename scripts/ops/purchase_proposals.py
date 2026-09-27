#!/usr/bin/env python3
"""Native bounded observer. There is intentionally no execution command."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from core.live_canary_preflight import _equity_session_state
from core.order_intent import canonical_payload_sha256
from core.purchase_proposals import (
    AUTHORITY,
    evaluate,
    validate_policy,
    validation_progress,
)
from core.supervised_broker_test import fresh, timestamp
from scripts.ops import supervised_broker_test as broker_test
from scripts.ops.long_runtime_common import run_bounded_process_group

POLICY_PATH = "config/purchase_proposal_policy_v1.json"
REPORT_PATH = "governance/health/purchase_proposals_latest.json"
HISTORY_PATH = "governance/runtime/purchase_proposal_validation.json"
REVOKE_PATH = "governance/runtime/purchase_proposal_revocation.json"


def read(root: Path, path: str, *, optional: bool = False) -> dict:
    target = broker_test.local_path(root, path)
    if optional and not target.exists():
        return {}
    if target.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("purchase_evidence_size_limit_exceeded")
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict):
        raise ValueError("purchase_evidence_invalid")
    return payload


def write(root: Path, path: str, payload: dict) -> None:
    target = broker_test.local_path(root, path)
    if (
        safe_write_json_atomic(
            str(target), payload, project_root=str(root), source="purchase_proposals"
        )
        is False
    ):
        raise ValueError("purchase_evidence_persistence_failed")
    target.chmod(0o600)


@contextlib.contextmanager
def lock(root: Path):
    path = broker_test.local_path(root, "governance/locks/purchase_proposals.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+"
    ) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def capture_observation(root: Path) -> dict:
    started = datetime.now(timezone.utc)
    result = run_bounded_process_group(
        [
            sys.executable,
            str(root / "scripts/ops/supervised_broker_test.py"),
            "observe",
            "--json",
        ],
        cwd=root,
        timeout_seconds=90,
        env={
            **os.environ,
            "MARKET_DATA_ONLY": "1",
            "ALLOW_ORDER_EXECUTION": "0",
            "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
            "EXECUTION_LANE_LIVE_ENABLED": "0",
            "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
        },
    )
    if result.get("rc") != 0:
        raise ValueError("broker_observation_failed_no_stale_fallback")
    observation = read(root, broker_test.REPORT_PATH)
    if timestamp(observation.get("timestamp_utc")) < started:
        raise ValueError("broker_observation_not_published_this_pass")
    return observation


def run(root: Path, command: str, *, scheduled: bool = False) -> dict:
    policy = read(root, POLICY_PATH)
    test = read(root, broker_test.POLICY_PATH)
    validate_policy(policy, test)
    now = datetime.now(timezone.utc)
    revoked = broker_test.local_path(root, REVOKE_PATH).exists()
    active = (
        timestamp(policy["valid_from_utc"]) <= now < timestamp(policy["expires_at_utc"])
    )
    if command == "status":
        prior = read(root, REPORT_PATH, optional=True)
        try:
            proposal_current = now < timestamp(prior.get("proposal_expires_at_utc"))
        except (TypeError, ValueError):
            proposal_current = False
        return {
            "timestamp_utc": now.isoformat(),
            "state": (
                "revoked"
                if revoked
                else "expired" if not active else "proposal_only_not_armed"
            ),
            "policy": policy,
            "revoked": revoked,
            "last_evaluation": prior,
            "last_evaluation_fresh": fresh(prior.get("timestamp_utc"), now, 1800),
            "last_proposal_fresh": bool(prior.get("proposal"))
            and proposal_current
            and not revoked
            and active
            and prior.get("policy_sha256") == canonical_payload_sha256(policy),
            "connected": False,
            **AUTHORITY,
        }
    if command == "revoke":
        # The durable stop must not wait behind a potentially slow broker reader.
        receipt = {
            "timestamp_utc": now.isoformat(),
            "policy_id": policy["policy_id"],
            "revoked": True,
            **AUTHORITY,
        }
        write(root, REVOKE_PATH, receipt)
        return receipt
    with lock(root):
        prior = read(root, REPORT_PATH, optional=True)
        revoked = broker_test.local_path(root, REVOKE_PATH).exists()
        active = (
            timestamp(policy["valid_from_utc"])
            <= now
            < timestamp(policy["expires_at_utc"])
        )
        if revoked or not active:
            report = {
                "timestamp_utc": now.isoformat(),
                "purpose": "purchase_proposals",
                "overall_status": "needs_attention",
                "state": "revoked" if revoked else "expired",
                "proposal": {},
                "connected": False,
                **AUTHORITY,
            }
            write(root, REPORT_PATH, report)
            return report
        if (
            scheduled
            and prior.get("policy_sha256") == canonical_payload_sha256(policy)
            and fresh(prior.get("timestamp_utc"), now, 899)
        ):
            return {
                **prior,
                "refresh_skipped": True,
                "skip_reason": "native_cadence_cooldown",
            }
        observation = capture_observation(root)
        candidate = read(root, "governance/runtime/production_candidate_state.json")
        blockers = broker_test.current_source_blockers(root, candidate)
        now = datetime.now(timezone.utc)
        report = evaluate(
            policy=policy,
            test=test,
            observation=observation,
            source={
                "ready": not blockers,
                "candidate_id": candidate.get("candidate_id"),
            },
            session=_equity_session_state(
                now=now, calendar_id="XNYS", buffer_minutes=5
            ),
            revoked=broker_test.local_path(root, REVOKE_PATH).exists(),
            now=now,
        )
        report["source_blockers"] = blockers
        history = read(root, HISTORY_PATH, optional=True)
        progress, updated = validation_progress(policy, report, history, now=now)
        report["validation"] = progress
        report["cadence"] = {
            "owner": "readiness_evidence_refresh:accrual",
            "target_interval_seconds": 900,
            "broker_read_deadline_seconds": 90,
            "scheduler_delays_possible": True,
        }
        if updated != history:
            write(root, HISTORY_PATH, updated)
        if broker_test.local_path(root, REVOKE_PATH).exists():
            report.update(
                state="revoked",
                proposal={},
                proposal_id=None,
                proposal_expires_at_utc=None,
            )
            report["reasons"] = sorted(
                set(report["reasons"]) | {"purchase_policy_revoked"}
            )
        write(root, REPORT_PATH, report)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("status", "evaluate", "revoke"), nargs="?", default="status"
    )
    parser.add_argument("--scheduled", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(PROJECT_ROOT, args.command, scheduled=args.scheduled)
        code = 0
    except BlockingIOError:
        result = {
            "state": "deferred",
            "reason": "purchase_observer_already_running",
            **AUTHORITY,
        }
        code = 4
    except Exception as exc:
        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "purchase_proposals",
            "overall_status": "blocked",
            "state": "blocked",
            "error": type(exc).__name__,
            "reason": "purchase_observation_failed_no_execution_or_stale_fallback",
            **AUTHORITY,
        }
        code = 2
        if args.command == "evaluate":
            try:
                with lock(PROJECT_ROOT):
                    write(PROJECT_ROOT, REPORT_PATH, result)
            except Exception:
                result["report_persistence_failed"] = True
    print(json.dumps(result, indent=None if args.json else 2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
