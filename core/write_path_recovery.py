"""Bounded owner receipts and checkpoint verification; never replay business actions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.status_label_contract import read_label_source
from core.storage_router import inspect_storage_path

STATE = "governance/health/write_path_recovery_state.json"
RECEIPTS = "governance/health/write_path_receipts"
MAX_BYTES = 1024 * 1024
MAX_DOMAINS = 256
MAX_RECORDS = 128
_REQUEST_CACHE: dict = {}
_LAST_RECEIPT: dict = {}


def timestamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def domain_id(source: str, target: str, day: str) -> str:
    return digest([source, target, day])


def failed_records(payloads: list[dict]) -> dict:
    try:
        records = [
            {"message_id": row.get("message_id"), "payload_sha256": digest(row)}
            for row in payloads[:MAX_RECORDS]
        ]
    except (TypeError, ValueError):
        records = []
    complete = (
        bool(payloads)
        and bool(records)
        and len(payloads) <= MAX_RECORDS
        and all(
            isinstance(row["message_id"], str) and row["message_id"] for row in records
        )
    )
    return {
        "failed_records": records if complete else [],
        "failed_record_count": len(payloads),
        "record_checkpoint_complete": complete,
    }


def local_path(root: Path, path: Path) -> Path:
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route["status"] not in {"present", "missing"}:
        raise ValueError(f"route_rejected:{route['status']}")
    return Path(route["resolved_path"])


def durable_json(root: Path, path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=True)
    if len(encoded.encode()) > 2 * MAX_BYTES:
        raise ValueError("recovery_receipt_size_budget_exceeded")
    target = local_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=".recovery-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        fd = os.open(target.parent, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _requests(root: Path, source: str, target: str, now: datetime) -> list[dict]:
    key = str(root)
    cached = _REQUEST_CACHE.get(key)
    if not cached or time.monotonic() - cached[0] >= 5:
        cached = (time.monotonic(), read_label_source(root, STATE))
        _REQUEST_CACHE[key] = cached
    return [
        request
        for request in cached[1].get("requests", [])[:4]
        if isinstance(request, dict)
        and re.fullmatch(r"[0-9a-f]{64}", str(request.get("id", "")))
        and request.get("source") == source
        and request.get("target_path") == target
        and (timestamp(request.get("expires_utc")) or now) > now
    ]


def record_success(
    root: str, source: str, target: str, data: bytes, *, kind: str
) -> None:
    """Verify only requested, real owner writes. Telemetry never changes write ACKs."""
    if not root or not data or len(data) > MAX_BYTES:
        return
    root_path = Path(root)
    now = datetime.now(timezone.utc)
    try:
        requests = _requests(root_path, source, target, now)
        if not requests:
            return
        selected = [
            r
            for r in requests
            if time.monotonic()
            - _LAST_RECEIPT.get((str(root_path), r["id"], r["generation"]), -1000)
            >= 60
        ]
        if not selected:
            return
        path = local_path(root_path, Path(target))
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size < len(data):
                return
            start = before.st_size - len(data) if kind == "jsonl" else 0
            if kind != "jsonl" and before.st_size != len(data):
                return
            handle.seek(start)
            if handle.read(len(data)) != data:
                return
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        for request in selected:
            proof = {
                "timestamp_utc": now.isoformat(),
                "source": source,
                "target_path": target,
                "generation": request["generation"],
                "kind": kind,
                "device": before.st_dev,
                "inode": before.st_ino,
                "offset": start,
                "length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "fsync_readback": True,
            }
            durable_json(
                root_path, root_path / RECEIPTS / f"{request['id']}.json", proof
            )
            _LAST_RECEIPT[(str(root_path), request["id"], request["generation"])] = (
                time.monotonic()
            )
    except Exception:
        # A failed receipt is missing proof, never a reason to replay an acknowledged write.
        return


def verify_receipt(root: Path, domain: dict, now: datetime) -> dict:
    proof = read_label_source(root, f"{RECEIPTS}/{domain['id']}.json")
    observed = timestamp(proof.get("timestamp_utc"))
    failure = timestamp(domain.get("latest_failure_utc"))
    if (
        not observed
        or not failure
        or not failure < observed <= now
        or (now - observed).total_seconds() > 300
    ):
        return {"verified": False, "reason": "fresh_post_failure_owner_write_required"}
    if (
        any(
            proof.get(key) != domain.get(key)
            for key in ("source", "target_path", "generation")
        )
        or proof.get("fsync_readback") is not True
    ):
        return {"verified": False, "reason": "owner_or_generation_mismatch"}
    length, offset = proof.get("length"), proof.get("offset")
    if (
        type(length) is not int
        or type(offset) is not int
        or not 0 < length <= MAX_BYTES
        or offset < 0
    ):
        return {"verified": False, "reason": "invalid_checkpoint_bounds"}
    path = local_path(root, Path(domain["target_path"]))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (proof.get("device"), proof.get("inode"))
            or before.st_size < offset + length
        ):
            return {"verified": False, "reason": "source_generation_changed"}
        handle.seek(offset)
        if hashlib.sha256(handle.read(length)).hexdigest() != proof.get("sha256"):
            return {"verified": False, "reason": "checkpoint_readback_mismatch"}
        matched = set()
        duplicate_or_conflicting_ids = False
        if proof.get("kind") == "jsonl":
            start = max(0, before.st_size - MAX_BYTES)
            handle.seek(start)
            raw = handle.read(MAX_BYTES)
            lines = raw.splitlines()
            if start:
                lines = lines[1:]
            if raw and not raw.endswith(b"\n"):
                raise ValueError("partial_jsonl_record")
            expected = {
                (r["message_id"], r["payload_sha256"])
                for r in domain.get("failed_records", [])
            }
            expected_ids = {key[0] for key in expected}
            seen_ids = set()
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("invalid_jsonl_record")
                key = (row.get("message_id"), digest(row))
                if key[0] in expected_ids:
                    duplicate_or_conflicting_ids |= (
                        key[0] in seen_ids or key not in expected
                    )
                    seen_ids.add(key[0])
                if key in expected:
                    matched.add(key)
        elif proof.get("kind") == "json":
            handle.seek(0)
            if before.st_size != length or not isinstance(
                json.loads(handle.read(length)), dict
            ):
                raise ValueError("invalid_atomic_snapshot")
        else:
            raise ValueError("unsupported_receipt_kind")
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("target_changed_during_verification")
    expected_count = len(
        {
            (r["message_id"], r["payload_sha256"])
            for r in domain.get("failed_records", [])
        }
    )
    reconciled = bool(
        domain.get("record_checkpoint_complete")
        and expected_count
        and len(matched) == expected_count
        and not duplicate_or_conflicting_ids
    )
    return {
        "verified": True,
        "observation_timestamp_utc": observed.isoformat(),
        "checkpoint_reconciled": reconciled,
        "matched_record_count": len(matched),
        "expected_record_count": expected_count,
        "duplicate_or_conflicting_ids": duplicate_or_conflicting_ids,
        "reason": (
            "checkpoint_records_verified"
            if reconciled
            else "historical_payload_checkpoint_unavailable_or_outside_bounded_tail"
        ),
    }


def recovery_pass(
    root: Path, history: dict, *, apply: bool, now: datetime | None = None
) -> dict:
    """Serialize native recovery cycles; max four verifications, no business replay."""
    now = now or datetime.now(timezone.utc)
    if not apply:
        state = read_label_source(root, STATE)
        return {
            "apply_requested": False,
            "state": state,
            "authority": "observation_only",
        }
    lock_path = local_path(root, root / "governance/health/write_path_recovery.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "r+"
    ) as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "apply_requested": True,
                "overall_status": "deferred",
                "reason": "recovery_owner_busy",
            }
        previous = read_label_source(root, STATE)
        if previous.get("_label_source_error"):
            return {
                "apply_requested": True,
                "overall_status": "blocked",
                "reason": "recovery_state_unreadable",
            }
        if previous and (
            previous.get("schema_version") != 1
            or not isinstance(previous.get("domains"), list)
            or len(previous["domains"]) > MAX_DOMAINS
            or any(
                not isinstance(d, dict)
                or not isinstance(d.get("id"), str)
                or type(d.get("count")) is not int
                or d["count"] < 0
                for d in previous["domains"]
            )
        ):
            return {
                "apply_requested": True,
                "overall_status": "blocked",
                "reason": "invalid_recovery_state",
            }
        domains = {row["id"]: row for row in previous.get("domains", [])}
        for incoming in history.get("domains", []):
            old = domains.get(incoming["id"], {})
            if old.get("generation") != incoming["generation"]:
                if old.get("count", 0) > incoming["count"]:
                    continue
                domains[incoming["id"]] = {
                    **incoming,
                    "phase": "awaiting_owner_write",
                    "attempts": 0,
                }
        if len(domains) > MAX_DOMAINS or not history.get("complete"):
            return {
                "apply_requested": True,
                "overall_status": "blocked",
                "reason": "domain_budget_or_history_incomplete",
                "retained_domain_count": len(domains),
                "state": previous,
            }
        requests = []
        checked = 0
        deadline = time.monotonic() + 3
        for domain in sorted(
            domains.values(), key=lambda d: d.get("last_checked_utc", "")
        ):
            if checked >= 4 or time.monotonic() >= deadline:
                break
            if domain.get("phase") == "escalated" or domain.get(
                "historical_reconciled"
            ):
                continue
            next_check = timestamp(domain.get("next_check_utc"))
            if next_check and now < next_check:
                continue
            checked += 1
            try:
                local_path(root, Path(domain["target_path"]))
                result = verify_receipt(root, domain, now)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                result = {"verified": False, "reason": str(exc)}
            domain.update(last_checked_utc=now.isoformat(), verification=result)
            if result["verified"]:
                observation = timestamp(result["observation_timestamp_utc"])
                first = timestamp(domain.get("probation_started_utc"))
                if not first:
                    domain["probation_started_utc"] = observation.isoformat()
                    first = observation
                qualified = (observation - first).total_seconds() >= 60
                domain["phase"] = "path_recovered" if qualified else "probation"
                domain["historical_reconciled"] = (
                    qualified and result["checkpoint_reconciled"]
                )
                domain["attempts"] = 0
                delay = 60
            else:
                domain.pop("probation_started_utc", None)
                domain["historical_reconciled"] = False
                domain["attempts"] = domain.get("attempts", 0) + 1
                domain["phase"] = (
                    "escalated" if domain["attempts"] >= 6 else "awaiting_owner_write"
                )
                delay = min(900, 60 * 2 ** min(domain["attempts"] - 1, 4))
            domain["next_check_utc"] = (now + timedelta(seconds=delay)).isoformat()
            if domain["phase"] != "escalated":
                requests.append(
                    {
                        key: domain[key]
                        for key in ("id", "source", "target_path", "generation")
                    }
                )
                requests[-1]["expires_utc"] = (
                    now + timedelta(seconds=1200)
                ).isoformat()
        # Retain unexpired requests for paths waiting through backoff, within one shared cap.
        ids = {r["id"] for r in requests}
        requests.extend(
            r
            for r in previous.get("requests", [])
            if r["id"] not in ids
            and r["id"] in domains
            and domains[r["id"]].get("phase") != "escalated"
            and r.get("generation") == domains[r["id"]].get("generation")
            and (timestamp(r.get("expires_utc")) or now) > now
        )
        unmapped = max(
            previous.get("unmapped_prior_failure_count", 0),
            history.get("unmapped_prior_failure_count", 0),
        )
        state = {
            "schema_version": 1,
            "timestamp_utc": now.isoformat(),
            "domains": list(domains.values()),
            "requests": requests[:4],
            "unmapped_prior_failure_count": unmapped,
            "count_basis": (
                "conservative_includes_unmapped_migration_debt"
                if unmapped
                else "retained_daily_owner_target_domains"
            ),
            "unreconciled_failure_count": unmapped
            + sum(
                d["count"]
                for d in domains.values()
                if not d.get("historical_reconciled")
            ),
            "path_recovered_count": sum(
                d.get("phase") == "path_recovered" for d in domains.values()
            ),
            "escalated_count": sum(
                d.get("phase") == "escalated" for d in domains.values()
            ),
            "automatic_replay_allowed": False,
            "release_authority": "write_path_evidence_only_no_trading_or_storage_gate_override",
        }
        durable_json(root, root / STATE, state)
        return {
            "apply_requested": True,
            "overall_status": (
                "degraded" if state["unreconciled_failure_count"] else "ready"
            ),
            "checked_count": checked,
            "state": state,
            "repair_owner": "existing_guarded_writer_and_storage_self_healing",
            "probation_seconds": 60,
            "max_attempts_per_generation": 6,
        }
