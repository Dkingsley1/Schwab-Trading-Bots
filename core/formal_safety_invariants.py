from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

TERMINAL_STATES = {"FILLED", "CANCELED", "REJECTED"}


@dataclass(frozen=True)
class SafetyState:
    order_state: str = "NEW"
    halted: bool = False
    submit_count: int = 0
    filled_quantity: int = 0
    requested_quantity: int = 2
    writer: str = "ledger"


def _successors(state: SafetyState) -> Iterable[tuple[str, SafetyState]]:
    yield "halt", replace(state, halted=True)
    yield "resume", replace(state, halted=False)
    if state.order_state in TERMINAL_STATES:
        return
    if state.order_state == "NEW" and not state.halted:
        yield "reserve", replace(state, order_state="RESERVED")
    if state.order_state == "RESERVED" and not state.halted and state.submit_count == 0:
        yield "submit", replace(state, order_state="SUBMITTED", submit_count=1)
    if state.order_state == "SUBMITTED":
        yield "ack", replace(state, order_state="ACKNOWLEDGED")
        yield "reject", replace(state, order_state="REJECTED")
    if state.order_state in {
        "RESERVED",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "PARTIALLY_FILLED",
    }:
        yield "cancel", replace(state, order_state="CANCELED")
    if state.order_state in {"ACKNOWLEDGED", "PARTIALLY_FILLED"}:
        next_fill = state.filled_quantity + 1
        next_state = (
            "FILLED" if next_fill == state.requested_quantity else "PARTIALLY_FILLED"
        )
        yield "fill_increment", replace(
            state, order_state=next_state, filled_quantity=next_fill
        )


def _violations(state: SafetyState) -> list[str]:
    errors: list[str] = []
    if state.submit_count not in {0, 1}:
        errors.append("intent_submitted_more_than_once")
    if not 0 <= state.filled_quantity <= state.requested_quantity:
        errors.append("fill_quantity_out_of_bounds")
    if state.writer != "ledger":
        errors.append("single_writer_violated")
    if state.order_state == "NEW" and (state.submit_count or state.filled_quantity):
        errors.append("new_state_contains_execution_progress")
    if (
        state.order_state == "FILLED"
        and state.filled_quantity != state.requested_quantity
    ):
        errors.append("filled_state_quantity_mismatch")
    return errors


def bounded_verify_order_safety(max_depth: int = 10) -> dict[str, Any]:
    initial = SafetyState()
    queue = deque([(initial, 0)])
    visited = {initial}
    violations: list[dict[str, Any]] = []
    transition_count = 0
    while queue:
        state, depth = queue.popleft()
        for error in _violations(state):
            violations.append({"error": error, "state": state.__dict__})
        if depth >= max_depth:
            continue
        for action, successor in _successors(state):
            transition_count += 1
            if state.halted and action in {"reserve", "submit"}:
                violations.append(
                    {
                        "error": "submission_transition_while_halted",
                        "state": state.__dict__,
                    }
                )
            if (
                state.order_state in TERMINAL_STATES
                and successor.order_state != state.order_state
            ):
                violations.append(
                    {"error": "terminal_state_mutated", "state": state.__dict__}
                )
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, depth + 1))
    return {
        "ok": not violations,
        "bounded_depth": max_depth,
        "state_count": len(visited),
        "transition_count": transition_count,
        "violations": violations,
        "proof_scope": "bounded_python_state_model",
        "execution_authority": False,
    }


def inspect_tla_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    required_tokens = (
        "---- MODULE OrderSafety ----",
        "Init ==",
        "Next ==",
        "TypeInvariant ==",
        "NoOverfill ==",
        "SingleSubmit ==",
        "Spec ==",
    )
    missing = [token for token in required_tokens if token not in text]
    return {
        "ok": spec_path.is_file() and not missing,
        "path": str(spec_path),
        "missing_tokens": missing,
        "tlc_executed": False,
        "formal_proof_ready": False,
        "external_evidence_debt": [
            "tlc_model_check_receipt",
            "independent_spec_review",
        ],
    }
