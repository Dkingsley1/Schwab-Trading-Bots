#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import (
        iso_now,
        load_json,
        payload_age_minutes,
        write_payload,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import (
        iso_now,
        load_json,
        payload_age_minutes,
        write_payload,
    )


DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "soak_reliability_sentinel_latest.json"
)
DEFAULT_STATE_PATH = (
    PROJECT_ROOT / "governance" / "runtime" / "soak_reliability_sentinel_state.json"
)
DEFAULT_LOCK_PATH = (
    PROJECT_ROOT / "governance" / "locks" / "soak_reliability_sentinel.lock"
)
DEFAULT_REQUEST_PATH = (
    PROJECT_ROOT / "governance" / "runtime" / "soak_self_healing_request.json"
)
DEFAULT_TRIGGER_PATH = (
    PROJECT_ROOT / "governance" / "runtime" / "soak_self_healing.trigger"
)
DEFAULT_AUDIT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "soak_reliability_sentinel_actions.jsonl"
)
SAFE_ENV = {
    "MARKET_DATA_ONLY": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
    "BOT_UNATTENDED_SOAK_ACTIVE": "1",
}

Runner = Callable[[list[str], Path, int, dict[str, str]], dict[str, Any]]


def _surface_contract(project_root: Path) -> dict[str, dict[str, Any]]:
    py = resolve_runtime_python(project_root)
    health = project_root / "governance" / "health"
    return {
        "health_gates": {
            "path": health / "health_gates_latest.json",
            "max_age_minutes": 15.0,
            "ready_statuses": {""},
            "command": [
                str(py),
                str(project_root / "scripts" / "health_gates.py"),
                "--json",
            ],
        },
        "ingestion_storage_control": {
            "path": health / "ingestion_storage_control_latest.json",
            "max_age_minutes": 5.0,
            "ready_statuses": {"ready"},
            "command": [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "storage-backpressure-autopilot",
                "--apply",
                "--quick-bounded",
                "--json",
            ],
            "timeout_seconds": 180,
        },
        "halt_trigger_control_plane": {
            "path": health / "halt_trigger_control_plane_latest.json",
            "max_age_minutes": 15.0,
            "ready_statuses": {"ready"},
            "command": [
                str(py),
                str(
                    project_root
                    / "scripts"
                    / "ops"
                    / "halt_trigger_control_plane.py"
                ),
                "--json",
            ],
        },
        "coordination_state": {
            "path": health / "coordination_state_latest.json",
            "max_age_minutes": 4.0,
            "refresh_lead_minutes": 1.0,
            "ready_statuses": {"ready", "guarded"},
            "command": [
                str(py),
                str(
                    project_root
                    / "scripts"
                    / "ops"
                    / "coordination_state_control.py"
                ),
                "--json",
            ],
        },
        "session_ready": {
            "path": health / "session_ready_latest.json",
            "max_age_minutes": 15.0,
            "ready_statuses": {"ready", ""},
            "command": [
                str(py),
                str(project_root / "scripts" / "session_ready_check.py"),
                "--json",
            ],
        },
        "process_watchdog": {
            "path": health / "process_watchdog_latest.json",
            "max_age_minutes": 4.0,
            "refresh_lead_minutes": 1.0,
            "ready_statuses": {"ready"},
            "command": [
                str(py),
                str(project_root / "scripts" / "ops" / "process_watchdog.py"),
                "--json",
            ],
        },
        "runtime_paper_regression_guard": {
            "path": health / "runtime_paper_regression_guard_latest.json",
            "max_age_minutes": 15.0,
            "ready_statuses": {"ready"},
            "command": [
                str(py),
                str(
                    project_root
                    / "scripts"
                    / "ops"
                    / "runtime_paper_regression_guard.py"
                ),
                "--json",
            ],
        },
        "live_order_ledger_control": {
            "path": health / "live_order_ledger_control_latest.json",
            "max_age_minutes": 15.0,
            "ready_statuses": {"ready", "ready_idle"},
            "command": [
                str(py),
                str(project_root / "scripts" / "ops" / "live_order_ledger_control.py"),
                "--json",
            ],
        },
        "local_storage_reserve_guard": {
            "path": health / "local_storage_reserve_guard_latest.json",
            "max_age_minutes": 5.0,
            "ready_statuses": {"ready"},
            "command": [
                str(py),
                str(
                    project_root / "scripts" / "ops" / "local_storage_reserve_guard.py"
                ),
                "--apply",
                "--skip-governor-reconcile",
                "--json",
            ],
        },
        "schwab_auth_supervisor": {
            "path": health / "schwab_auth_supervisor_latest.json",
            "max_age_minutes": 30.0,
            "ready_statuses": {"ready"},
            "command": [
                str(py),
                str(project_root / "scripts" / "ops" / "schwab_auth_supervisor.py"),
                "--json",
            ],
        },
        "system_role_contract": {
            "path": health / "system_role_contract_latest.json",
            "max_age_minutes": 30.0,
            "ready_statuses": {"ready"},
            "command": [
                str(py),
                str(
                    project_root / "scripts" / "ops" / "system_role_contract_control.py"
                ),
                "--json",
            ],
        },
        "authoritative_systems_control": {
            "path": health / "authoritative_systems_control_latest.json",
            "max_age_minutes": 20.0 * 60.0,
            "ready_statuses": {"ready"},
            "blocking": False,
            "command": [
                str(py),
                str(
                    project_root
                    / "scripts"
                    / "ops"
                    / "authoritative_systems_control.py"
                ),
                "--json",
            ],
        },
        "paper_live_equivalence": {
            "path": health / "paper_live_equivalence_latest.json",
            "max_age_minutes": 20.0 * 60.0,
            "ready_statuses": {
                "ready",
                "awaiting_live_shadow_samples",
                "equivalent",
                "partial_shadow_coverage",
            },
            "blocking": False,
            "command": [
                str(py),
                str(project_root / "scripts" / "paper_live_equivalence_report.py"),
                "--json",
            ],
        },
        "research_data_platform_control": {
            "path": health / "research_data_platform_control_latest.json",
            "max_age_minutes": 180.0,
            "ready_statuses": {"ready", "ready_with_evidence_debt"},
            "blocking": False,
            "command": [
                str(py),
                str(
                    project_root
                    / "scripts"
                    / "ops"
                    / "research_data_platform_control.py"
                ),
                "--json",
            ],
        },
        "institutional_research_extensions_control": {
            "path": (
                health / "institutional_research_extensions_control_latest.json"
            ),
            "max_age_minutes": 180.0,
            "ready_statuses": {"ready", "ready_with_evidence_debt"},
            "blocking": False,
            "command": [
                str(py),
                str(
                    project_root
                    / "scripts"
                    / "ops"
                    / "institutional_research_extensions_control.py"
                ),
                "--json",
            ],
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _surface_row(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["path"])
    payload = load_json(path)
    exists = bool(path.exists() and payload)
    age = payload_age_minutes(payload, path) if exists else None
    status = (
        str(payload.get("overall_status") or payload.get("status") or "")
        .strip()
        .lower()
    )
    ready_statuses = set(cfg.get("ready_statuses") or {"ready"})
    payload_ok = bool(payload.get("ok", status in ready_statuses))
    if name == "health_gates":
        payload_ok = not bool(payload.get("hard_gate_triggered", True))
    if name == "ingestion_storage_control":
        integrity = (
            payload.get("data_integrity")
            if isinstance(payload.get("data_integrity"), dict)
            else {}
        )
        payload_ok = bool(
            status == "ready"
            and payload.get("ok", True) is not False
            and str(payload.get("severity") or "stable") in {"stable", "elevated"}
            and int(integrity.get("sql_invalid_lines", 0) or 0) <= 0
            and int(integrity.get("sql_overlay_invalid_lines", 0) or 0) <= 0
            and int(integrity.get("sql_overlay_ops_write_failures", 0) or 0) <= 0
        )
    reserve = (
        payload.get("local_storage_reserve")
        if isinstance(payload.get("local_storage_reserve"), dict)
        else {}
    )
    recovery = (
        payload.get("recovery_request")
        if isinstance(payload.get("recovery_request"), dict)
        else {}
    )
    managed_recovery = bool(
        name == "local_storage_reserve_guard"
        and payload_ok
        and status == "watch"
        and bool(reserve.get("disk", {}).get("known", False))
        and not bool(reserve.get("pressure_active", False))
        and not bool(reserve.get("hard_block", False))
        and not bool(reserve.get("emergency_active", False))
        and bool(recovery.get("active", False))
        and not bool(recovery.get("paper_pause_required", False))
        and bool(recovery.get("collection_may_continue", False))
    )
    execution_policy = (
        payload.get("execution_policy")
        if isinstance(payload.get("execution_policy"), dict)
        else {}
    )
    managed_protective_state = bool(
        name == "halt_trigger_control_plane"
        and status == "blocked"
        and str(payload.get("effective_state") or "").strip().lower()
        == "live_read_only"
        and bool(execution_policy.get("paper_trade_lock_active", False))
        and not bool(execution_policy.get("operator_stop_active", False))
        and not bool(execution_policy.get("global_halt_active", False))
        and not bool(
            execution_policy.get("effective_live_order_execution_allowed", False)
        )
    )
    ready = bool(
        exists
        and (
            (payload_ok and status in ready_statuses)
            or managed_recovery
            or managed_protective_state
        )
    )
    stale = bool(
        not exists or age is None or float(age) > float(cfg["max_age_minutes"])
    )
    refresh_lead_minutes = max(float(cfg.get("refresh_lead_minutes", 0.0)), 0.0)
    refresh_due = bool(
        stale
        or (
            exists
            and age is not None
            and ready
            and refresh_lead_minutes > 0
            and float(age)
            >= max(float(cfg["max_age_minutes"]) - refresh_lead_minutes, 0.0)
        )
    )
    return {
        "name": name,
        "path": str(path),
        "impact_scope": (
            "operational" if cfg.get("blocking", True) else "evidence_only"
        ),
        "exists": exists,
        "age_minutes": round(float(age), 4) if age is not None else None,
        "max_age_minutes": float(cfg["max_age_minutes"]),
        "refresh_lead_minutes": refresh_lead_minutes,
        "status": status,
        "payload_ok": payload_ok,
        "ready": ready,
        "managed_recovery": managed_recovery,
        "managed_protective_state": managed_protective_state,
        "stale": stale,
        "refresh_due": refresh_due,
        "needs_repair": bool(refresh_due or not ready),
        "source_sha256": _sha256_file(path) if exists else "",
        "refresh_command": list(cfg.get("command") or []),
    }


def _load_state(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "schema_version": 1,
        "actions": (
            payload.get("actions") if isinstance(payload.get("actions"), dict) else {}
        ),
    }


def _action_gate(
    state: dict[str, Any],
    name: str,
    *,
    now_epoch: float,
    cooldown_seconds: float,
) -> dict[str, Any]:
    row = (state.get("actions") or {}).get(name)
    row = row if isinstance(row, dict) else {}
    circuit_until = float(row.get("circuit_until_epoch", 0.0) or 0.0)
    last_attempt = float(row.get("last_attempt_epoch", 0.0) or 0.0)
    if circuit_until > now_epoch:
        return {
            "allowed": False,
            "reason": "repair_circuit_open",
            "retry_after_seconds": round(circuit_until - now_epoch, 3),
        }
    if last_attempt > 0 and now_epoch - last_attempt < max(
        float(cooldown_seconds), 0.0
    ):
        return {
            "allowed": False,
            "reason": "repair_cooldown_active",
            "retry_after_seconds": round(
                max(float(cooldown_seconds), 0.0) - (now_epoch - last_attempt), 3
            ),
        }
    return {"allowed": True, "reason": "repair_allowed", "retry_after_seconds": 0.0}


def _record_action(
    state: dict[str, Any],
    name: str,
    *,
    result: dict[str, Any],
    now_epoch: float,
    max_failures: int,
    circuit_open_seconds: float,
) -> dict[str, Any]:
    actions = state.setdefault("actions", {})
    previous = actions.get(name) if isinstance(actions.get(name), dict) else {}
    ok = bool(result.get("ok", False))
    failures = 0 if ok else int(previous.get("consecutive_failures", 0) or 0) + 1
    circuit_until = (
        now_epoch + max(float(circuit_open_seconds), 1.0)
        if failures >= max(int(max_failures), 1)
        else 0.0
    )
    actions[name] = {
        "last_attempt_epoch": now_epoch,
        "last_attempt_utc": iso_now(),
        "last_ok": ok,
        "last_rc": int(result.get("rc", 1) or 0),
        "consecutive_failures": failures,
        "circuit_until_epoch": circuit_until,
        "last_error": str(result.get("error") or result.get("stderr_tail") or "")[
            -800:
        ],
    }
    return actions[name]


def _default_runner(
    command: list[str], project_root: Path, timeout_seconds: int, env: dict[str, str]
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_seconds), 1),
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        error = ""
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "timeout"
        rc = 124
        error = "timeout"
    parsed: dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    status = str(parsed.get("overall_status") or parsed.get("status") or "").lower()
    ok = bool(
        rc == 0
        and parsed.get("ok") is not False
        and status not in {"blocked", "critical", "failed"}
    )
    return {
        "ok": ok,
        "rc": rc,
        "duration_seconds": round(time.monotonic() - started, 4),
        "parsed": parsed,
        "stdout_tail": stdout[-1200:],
        "stderr_tail": stderr[-1200:],
        "error": error,
    }


def _append_audit(
    path: Path, rows: list[dict[str, Any]], *, max_lines: int = 1024
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    # SQL ingestion tracks append-only byte offsets. Replacing this bounded audit
    # made every retained row appear new on each sentinel pass; retention owns
    # archival and compaction after the hot cursor has consumed the stream.
    _ = max(int(max_lines), 1)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    state_path: Path | None = None,
    request_path: Path | None = None,
    trigger_path: Path | None = None,
    audit_path: Path | None = None,
    max_actions: int = 6,
    cooldown_seconds: float = 300.0,
    max_failures: int = 2,
    circuit_open_seconds: float = 3600.0,
    action_timeout_seconds: int = 60,
    repair_grace_seconds: float = 600.0,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    state_path = (
        state_path or project_root / "governance" / "runtime" / DEFAULT_STATE_PATH.name
    )
    request_path = (
        request_path
        or project_root / "governance" / "runtime" / DEFAULT_REQUEST_PATH.name
    )
    trigger_path = (
        trigger_path
        or project_root / "governance" / "runtime" / DEFAULT_TRIGGER_PATH.name
    )
    audit_path = (
        audit_path or project_root / "governance" / "health" / DEFAULT_AUDIT_PATH.name
    )
    contract = _surface_contract(project_root)
    before = {name: _surface_row(name, cfg) for name, cfg in contract.items()}
    state = _load_state(state_path)
    actions: list[dict[str, Any]] = []
    now_epoch = time.time()
    env = os.environ.copy()
    env.update(SAFE_ENV)

    if apply:
        executed_actions = 0
        for name, row in before.items():
            if executed_actions >= max(int(max_actions), 0) or not bool(
                row.get("needs_repair", False)
            ):
                continue
            command = list(contract[name].get("command") or [])
            if not command:
                continue
            action_started_epoch = time.time()
            gate = _action_gate(
                state,
                name,
                now_epoch=action_started_epoch,
                cooldown_seconds=cooldown_seconds,
            )
            if not gate["allowed"]:
                actions.append(
                    {"name": name, "executed": False, "ok": True, "gate": gate}
                )
                continue
            timeout_seconds = int(
                contract[name].get("timeout_seconds", action_timeout_seconds)
            )
            result = runner(command, project_root, timeout_seconds, env)
            if (
                name == "halt_trigger_control_plane"
                and int(result.get("rc", 1) or 0) == 0
                and isinstance(result.get("parsed"), dict)
            ):
                # This producer reports the intentional live-money lock as
                # "blocked". A successful publication is still a successful
                # repair; the post-run surface projection decides whether the
                # protective state itself is acceptable.
                result["ok"] = True
            executed_actions += 1
            action_state = _record_action(
                state,
                name,
                result=result,
                now_epoch=time.time(),
                max_failures=max_failures,
                circuit_open_seconds=circuit_open_seconds,
            )
            actions.append(
                {
                    "name": name,
                    "executed": True,
                    "command": command,
                    "ok": bool(result.get("ok", False)),
                    "rc": result.get("rc"),
                    "duration_seconds": result.get("duration_seconds"),
                    "stderr_tail": str(result.get("stderr_tail") or "")[-800:],
                    "circuit": action_state,
                }
            )
        _append_audit(
            audit_path,
            [
                {"timestamp_utc": iso_now(), **row}
                for row in actions
                if row.get("executed")
            ],
        )

    after = {name: _surface_row(name, cfg) for name, cfg in contract.items()}
    recovered_circuits: list[str] = []
    if apply:
        action_state = (
            state.get("actions") if isinstance(state.get("actions"), dict) else {}
        )
        for name, row in after.items():
            prior = (
                action_state.get(name)
                if isinstance(action_state.get(name), dict)
                else None
            )
            if (
                prior is None
                or not bool(row.get("ready", False))
                or bool(row.get("stale", True))
            ):
                continue
            if (
                int(prior.get("consecutive_failures", 0) or 0) > 0
                or float(prior.get("circuit_until_epoch", 0.0) or 0.0) > 0
            ):
                prior["consecutive_failures"] = 0
                prior["circuit_until_epoch"] = 0.0
                prior["last_ok"] = True
                prior["last_error"] = ""
                prior["recovered_at_utc"] = iso_now()
                recovered_circuits.append(name)
        state["timestamp_utc"] = iso_now()
        write_payload(state_path, state)
    blockers: list[str] = []
    warnings: list[str] = []
    evidence_advisories: list[str] = []
    deferred_refresh_reasons: list[str] = []
    actions_by_name = {
        str(row.get("name") or ""): row
        for row in actions
        if str(row.get("name") or "")
    }
    for name, row in after.items():
        issue_target = (
            blockers
            if contract[name].get("blocking", True)
            else evidence_advisories
        )
        if not row["exists"]:
            issue_target.append(f"{name}_missing")
        elif row["stale"]:
            refresh_reason = f"{name}_refresh_due"
            action = actions_by_name.get(name, {})
            action_failed = bool(action.get("executed", False)) and not bool(
                action.get("ok", False)
            )
            action_state = (
                (state.get("actions") or {}).get(name)
                if isinstance((state.get("actions") or {}).get(name), dict)
                else {}
            )
            circuit_open = bool(
                float(action_state.get("circuit_until_epoch", 0.0) or 0.0)
                > now_epoch
            )
            age_minutes = float(row.get("age_minutes") or 0.0)
            refresh_grace_minutes = max(float(repair_grace_seconds), 0.0) / 60.0
            bounded_refresh_queued = bool(
                apply
                and contract[name].get("blocking", True)
                and row.get("ready", False)
                and contract[name].get("command")
                and age_minutes
                <= float(row.get("max_age_minutes") or 0.0)
                + refresh_grace_minutes
                and not action_failed
                and not circuit_open
            )
            if bounded_refresh_queued:
                warnings.append(f"{name}_refresh_queued")
                deferred_refresh_reasons.append(refresh_reason)
            else:
                issue_target.append(refresh_reason)
        elif not row["ready"]:
            issue_target.append(f"{name}_not_ready")
        elif row.get("managed_recovery"):
            warnings.append(f"{name}_managed_recovery_pending")

    health = project_root / "governance" / "health"
    auth_payload = load_json(Path(contract["schwab_auth_supervisor"]["path"]))
    auth_operator_actions = [
        str(action)
        for action in auth_payload.get("operator_followups", [])
        if "token-refresh-interactive" in str(action)
    ]
    auth_operator_required = bool(
        auth_operator_actions
        and any(item.startswith("schwab_auth_supervisor_") for item in blockers)
    )
    machine_repairable_blockers = [
        item
        for item in blockers
        if not (auth_operator_required and item.startswith("schwab_auth_supervisor_"))
    ]
    managed_recovery_reasons = [
        f"{name}_managed_recovery_pending"
        for name, row in after.items()
        if bool(row.get("managed_recovery", False))
    ]
    machine_repairable_reasons = (
        machine_repairable_blockers
        + managed_recovery_reasons
        + deferred_refresh_reasons
    )
    heavy_path = health / "soak_self_healing_control_latest.json"
    heavy_payload = load_json(heavy_path)
    heavy_age = (
        payload_age_minutes(heavy_payload, heavy_path) if heavy_payload else None
    )
    heavy_required = bool(machine_repairable_reasons)
    heavy_fresh = bool(heavy_age is not None and float(heavy_age) <= 60.0)
    repair_signature = hashlib.sha256(
        json.dumps(
            sorted(machine_repairable_reasons), ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:16]
    trigger_payload = load_json(trigger_path)
    trigger_age = (
        payload_age_minutes(trigger_payload, trigger_path) if trigger_payload else None
    )
    trigger_matches = bool(
        trigger_payload
        and str(trigger_payload.get("repair_signature") or "") == repair_signature
    )
    if heavy_required and not heavy_fresh:
        grace_expired = bool(
            float(repair_grace_seconds) <= 0
            or (
                trigger_matches
                and trigger_age is not None
                and float(trigger_age) * 60.0 >= float(repair_grace_seconds)
            )
        )
        if grace_expired:
            blockers.append("heavy_self_healing_starved_while_repair_required")
        else:
            warnings.append("heavy_self_healing_wakeup_pending")

    open_circuits = [
        name
        for name, row in (state.get("actions") or {}).items()
        if isinstance(row, dict)
        and float(row.get("circuit_until_epoch", 0.0) or 0.0) > time.time()
    ]
    if open_circuits:
        warnings.append("bounded_repair_circuit_open")

    receipt_input = {
        name: {
            "source_sha256": row["source_sha256"],
            "age_minutes": row["age_minutes"],
            "ready": row["ready"],
            "stale": row["stale"],
        }
        for name, row in sorted(after.items())
    }
    receipt_sha = hashlib.sha256(
        json.dumps(
            receipt_input, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    request = {
        "timestamp_utc": iso_now(),
        "active": bool(blockers or heavy_required),
        "severity": (
            "critical" if blockers else ("proactive" if heavy_required else "none")
        ),
        "reasons": blockers or machine_repairable_reasons,
        "repair_signature": repair_signature,
        "heavy_repair_required": heavy_required,
        "machine_repairable_reasons": machine_repairable_reasons,
        "operator_intervention_required": auth_operator_required,
        "operator_actions": auth_operator_actions,
        "live_execution_authority": False,
        "paper_only": True,
    }
    if apply:
        write_payload(request_path, request)
        trigger_due = bool(
            heavy_required
            and (
                not trigger_matches
                or trigger_age is None
                or float(trigger_age) * 60.0 >= max(float(repair_grace_seconds), 60.0)
            )
        )
        if trigger_due:
            write_payload(
                trigger_path,
                {
                    "timestamp_utc": iso_now(),
                    "repair_signature": repair_signature,
                    "reasons": machine_repairable_reasons,
                    "paper_only": True,
                    "live_execution_authority": False,
                },
            )

    overall_status = "blocked" if blockers else ("watch" if warnings else "ready")
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": overall_status,
        "grade": "F" if blockers else ("A" if warnings else "A+"),
        "apply": bool(apply),
        "surfaces_before": before,
        "surfaces": after,
        "blockers": blockers,
        "warnings": warnings,
        "evidence_advisories": evidence_advisories,
        "repair_actions": actions,
        "repair_request": request,
        "heavy_controller": {
            "path": str(heavy_path),
            "present": bool(heavy_payload),
            "age_minutes": (
                round(float(heavy_age), 4) if heavy_age is not None else None
            ),
            "fresh": heavy_fresh,
            "freshness_required": heavy_required,
            "wakeup_trigger_path": str(trigger_path),
            "wakeup_trigger_age_minutes": (
                round(float(trigger_age), 4) if trigger_age is not None else None
            ),
            "wakeup_grace_seconds": max(float(repair_grace_seconds), 0.0),
            "policy": "heavy self-healing may remain dormant while all always-on safety surfaces are ready",
        },
        "bounded_repair": {
            "max_actions_per_cycle": max(int(max_actions), 0),
            "deferred_refresh_reasons": deferred_refresh_reasons,
            "cooldown_seconds": max(float(cooldown_seconds), 0.0),
            "max_failures_before_circuit": max(int(max_failures), 1),
            "circuit_open_seconds": max(float(circuit_open_seconds), 1.0),
            "open_circuits": sorted(open_circuits),
            "recovered_circuits": sorted(recovered_circuits),
            "state_path": str(state_path),
            "audit_path": str(audit_path),
        },
        "evidence_epoch": {
            "id": f"soak-sentinel:{receipt_sha[:16]}",
            "receipt_sha256": receipt_sha,
            "source_count": len(after),
        },
        "safety_contract": {
            "always_on_observation": True,
            "heavy_maintenance_separated": True,
            "proactive_nonblocking_evidence_refresh": True,
            "exact_refresh_allowlist_only": True,
            "automatic_source_code_changes": False,
            "automatic_live_orders": False,
            "market_data_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Always-on, bounded reliability sentinel for the unattended paper soak."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--trigger-file", type=Path)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--cooldown-seconds", type=float, default=300.0)
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--circuit-open-seconds", type=float, default=3600.0)
    parser.add_argument("--action-timeout-seconds", type=int, default=60)
    parser.add_argument("--repair-grace-seconds", type=float, default=600.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_path = args.out_file or Path(
        "governance/health/soak_reliability_sentinel_latest.json"
    )
    state_path = args.state_file or Path(
        "governance/runtime/soak_reliability_sentinel_state.json"
    )
    request_path = args.request_file or Path(
        "governance/runtime/soak_self_healing_request.json"
    )
    trigger_path = args.trigger_file or Path(
        "governance/runtime/soak_self_healing.trigger"
    )
    audit_path = args.audit_file or Path(
        "governance/health/soak_reliability_sentinel_actions.jsonl"
    )
    lock_path = args.lock_file or Path(
        "governance/locks/soak_reliability_sentinel.lock"
    )
    out_path = out_path if out_path.is_absolute() else project_root / out_path
    state_path = state_path if state_path.is_absolute() else project_root / state_path
    request_path = (
        request_path if request_path.is_absolute() else project_root / request_path
    )
    trigger_path = (
        trigger_path if trigger_path.is_absolute() else project_root / trigger_path
    )
    audit_path = audit_path if audit_path.is_absolute() else project_root / audit_path
    lock_path = lock_path if lock_path.is_absolute() else project_root / lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = {
                "ok": True,
                "overall_status": "already_running",
                "busy": True,
                "lock_path": str(lock_path),
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            return 0
        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            state_path=state_path,
            request_path=request_path,
            trigger_path=trigger_path,
            audit_path=audit_path,
            max_actions=int(args.max_actions),
            cooldown_seconds=float(args.cooldown_seconds),
            max_failures=int(args.max_failures),
            circuit_open_seconds=float(args.circuit_open_seconds),
            action_timeout_seconds=int(args.action_timeout_seconds),
            repair_grace_seconds=float(args.repair_grace_seconds),
        )
        write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "soak_reliability_sentinel "
            f"status={payload['overall_status']} blockers={len(payload['blockers'])} actions={len(payload['repair_actions'])}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
