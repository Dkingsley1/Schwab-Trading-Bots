"""Local, fail-closed operator interlock; never a substitute for order authority."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.accountability import safe_write_json_atomic
from core.storage_router import inspect_storage_path

STATE = "governance/runtime/live_execution_switch_state.json"
LOCK = "governance/runtime/live_execution_switch.lock"
CANDIDATE = "governance/runtime/production_candidate_state.json"
PURPOSES = ("supervised_broker_test", "production_canary")
WARNING = (
    "WARNING: REAL MONEY. ON permits only the selected scope through existing "
    "checks; it does not authorize an order. OFF blocks new orders and replacements, "
    "including exits. It does not cancel pending orders, recall in-flight requests, "
    "or sell holdings. Keep independent broker access."
)


def local(root: Path, relative: str) -> Path:
    path = root / relative
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("live_switch_requires_local_owned_paths")
    return path


def _read(root: Path) -> dict:
    try:
        path = local(root, STATE)
        if path.stat().st_size > 16384:
            raise ValueError("live_switch_state_oversized")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("live_switch_state_invalid")
        return value
    except FileNotFoundError:
        return {}


def _write(root: Path, payload: dict) -> None:
    if (
        safe_write_json_atomic(
            str(local(root, STATE)),
            payload,
            project_root=str(root),
            source="live_execution_switch",
        )
        is False
    ):
        raise RuntimeError("live_switch_state_not_persisted")
    if _read(root) != payload:
        raise RuntimeError("live_switch_state_not_verified")


@contextmanager
def _lock(root: Path):
    path = local(root, LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as handle:
        # Checks and network calls never run under this short state-write lock.
        deadline = time.monotonic() + 2
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("live_switch_busy_retry_off_and_check_broker")
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _bindings(root: Path, purpose: str, symbol: str) -> dict:
    if purpose not in PURPOSES or not re.fullmatch(r"[A-Z0-9.\-]{1,16}", symbol):
        raise ValueError("live_switch_scope_invalid")
    if purpose == "supervised_broker_test":
        from core.supervised_broker_test import policy_path

        plan = policy_path(symbol)
    else:
        plan = "config/live_canary_micro_policy_v1.json"
    paths = [
        CANDIDATE,
        plan,
        "config/production_readiness_control_v1.json",
        "config/account_policy_registry.json",
    ]
    bindings = {}
    for relative in paths:
        path = local(root, relative)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("live_switch_binding_oversized")
        bindings[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = json.loads(local(root, CANDIDATE).read_text())
    if not candidate.get("candidate_id"):
        raise ValueError("live_switch_candidate_missing")
    return {"files": bindings, "candidate_id": candidate["candidate_id"]}


def switch_status(root: str | Path, *, now: float | None = None) -> dict:
    root = Path(root)
    current = time.time() if now is None else now
    try:
        state = _read(root)
        blockers = []
        if state.get("requested_on") is not True:
            blockers.append("live_execution_switch_off")
        else:
            issued, expires = state["issued_at"], state["expires_at"]
            if not (
                isinstance(issued, (int, float))
                and isinstance(expires, (int, float))
                and issued <= current < expires
                and 0 < expires - issued <= 3600
            ):
                blockers.append("live_execution_switch_expired_or_invalid_time")
            if state.get("schema_version") != 1 or state.get("session") not in {
                "NORMAL",
                "AM",
                "PM",
            }:
                blockers.append("live_execution_switch_invalid_state")
            if state.get("bindings") != _bindings(
                root, state["purpose"], state["symbol"]
            ):
                blockers.append("live_execution_switch_candidate_or_policy_changed")
        return {
            "ok": True,
            "switch": "OFF" if blockers else "ON",
            "requested_on": state.get("requested_on") is True,
            "purpose": state.get("purpose"),
            "symbol": state.get("symbol"),
            "session": state.get("session"),
            "expires_at_utc": state.get("expires_at_utc"),
            "blockers": blockers,
            "order_authorized": False,
            "other_gates": "checked_by_order_owner_at_submission",
            "runtime_adoption": "requires_processes_loaded_with_switch_support",
            "warning": WARNING,
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {
            "ok": False,
            "switch": "OFF",
            "blockers": ["live_execution_switch_unreadable"],
            "error": type(exc).__name__,
            "order_authorized": False,
            "warning": WARNING,
        }


def check_live_execution_switch(
    root: str | Path, *, broker: str, operation: str, context: dict | None = None
) -> dict:
    if operation not in {"place_order", "replace_order"} or broker == "mock":
        return {"allowed": True, "applicable": False}
    status = switch_status(root)
    context = context or {}
    blockers = list(status["blockers"])
    if status["switch"] == "ON" and (
        broker != "schwab"
        or status["purpose"] != context.get("purpose", "production_canary")
        or status["symbol"] != str(context.get("symbol", "")).upper()
        or status["session"] != context.get("session", "NORMAL")
    ):
        blockers.append("live_execution_switch_scope_mismatch")
    return {
        "allowed": not blockers,
        "applicable": True,
        "blockers": blockers,
        "switch": status["switch"],
    }


def switch_off(root: str | Path) -> dict:
    root = Path(root)
    with _lock(root):
        _write(
            root,
            {
                "schema_version": 1,
                "requested_on": False,
                "generation": uuid.uuid4().hex,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    return switch_status(root)


def _readiness(root: Path, *, purpose: str, symbol: str, session: str) -> dict:
    from core.live_canary_preflight import evaluate_live_canary_preflight
    from core.live_canary_allowlist import evaluate_live_canary_allowlist

    receipt = evaluate_live_canary_preflight(
        root,
        symbol=symbol,
        action="BUY",
        purpose=purpose,
        session=session,
    )
    blockers = list(receipt["blockers"])
    if purpose == "supervised_broker_test":
        # The native test collects fresh attestation inside its submit dialog.
        # Switch permission must not deadlock that dialog or replace its checks.
        deferred = set(receipt.get("operator_attestation_blockers", []))
        blockers = [item for item in blockers if item not in deferred]
    else:
        blockers.extend(evaluate_live_canary_allowlist(root)["blockers"])
    for name in ("OPERATOR_STOP", "GLOBAL_TRADING_HALT", "SYSTEM_POWER_OFF"):
        if local(root, f"governance/health/{name}.flag").exists() or os.environ.get(
            name, ""
        ).lower() in {"1", "true", "yes", "on"}:
            blockers.append(name.lower() + "_active")
    return {"blockers": sorted(set(blockers)), "candidate_id": receipt["candidate_id"]}


def switch_on(
    root: str | Path, *, purpose: str, symbol: str, session: str, minutes: int = 30
) -> dict:
    """Called only after interactive operator confirmation; performs no broker I/O."""
    root = Path(root)
    if (
        purpose not in PURPOSES
        or session not in {"NORMAL", "AM", "PM"}
        or not 1 <= minutes <= 60
    ):
        raise ValueError("live_switch_invalid_activation_scope")
    symbol = symbol.upper().strip()
    # Persist a fresh OFF generation before checks; a competing OFF invalidates it.
    generation = uuid.uuid4().hex
    with _lock(root):
        _write(
            root, {"schema_version": 1, "requested_on": False, "generation": generation}
        )
    bindings = _bindings(root, purpose, symbol)
    readiness = _readiness(root, purpose=purpose, symbol=symbol, session=session)
    if readiness["blockers"]:
        return {
            **switch_status(root),
            "ok": False,
            "activation_blockers": readiness["blockers"],
        }
    with _lock(root):
        if _read(root).get("generation") != generation:
            return {
                **switch_status(root),
                "ok": False,
                "activation_blockers": ["live_switch_request_superseded"],
            }
        if (
            bindings != _bindings(root, purpose, symbol)
            or bindings["candidate_id"] != readiness["candidate_id"]
        ):
            raise ValueError("live_switch_candidate_changed_during_checks")
        issued = time.time()
        _write(
            root,
            {
                "schema_version": 1,
                "requested_on": True,
                "generation": generation,
                "purpose": purpose,
                "symbol": symbol,
                "session": session,
                "issued_at": issued,
                "expires_at": issued + minutes * 60,
                "expires_at_utc": datetime.fromtimestamp(
                    issued + minutes * 60, timezone.utc
                ).isoformat(),
                "bindings": bindings,
            },
        )
    return switch_status(root)
