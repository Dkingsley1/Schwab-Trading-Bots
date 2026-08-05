#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, parse_iso_utc, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, parse_iso_utc, payload_age_minutes, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "production_excellence_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "production_excellence_control_latest.json"
BAD_STATUSES = {"blocked", "critical", "degraded", "error", "failed", "missing", "not_ready", "stale"}
GRADE_RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4, "A+": 5}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _grade(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    return "A+" if text == "A++" else text


def _grade_at_least(raw: Any, floor: str) -> bool:
    return GRADE_RANK.get(_grade(raw), -1) >= GRADE_RANK.get(_grade(floor), 99)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _git_head(project_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def _scope_files(project_root: Path, patterns: Iterable[Any]) -> list[Path]:
    paths: set[Path] = set()
    for raw in patterns:
        pattern = str(raw or "").strip()
        if not pattern:
            continue
        for path in project_root.glob(pattern):
            if path.is_file() or path.is_symlink():
                paths.add(path)
    return sorted(paths, key=lambda item: str(item.relative_to(project_root)))


def _hash_scope(project_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        rel = str(path.relative_to(project_root)).replace(os.sep, "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(f"symlink:{os.readlink(path)}".encode("utf-8"))
        else:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def candidate_fingerprints(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    scopes = _as_dict(_as_dict(config.get("candidate")).get("scope_globs"))
    rows: dict[str, dict[str, Any]] = {}
    for scope, patterns in sorted(scopes.items()):
        files = _scope_files(project_root, _as_list(patterns))
        rows[str(scope)] = {
            "sha256": _hash_scope(project_root, files),
            "file_count": len(files),
        }
    combined = {scope: row["sha256"] for scope, row in rows.items()}
    return {
        "overall_sha256": _canonical_hash(combined),
        "scopes": rows,
        "scope_count": len(rows),
        "file_count": sum(_safe_int(row.get("file_count"), 0) for row in rows.values()),
    }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, errors
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [], [f"event_log_read_failed:{type(exc).__name__}"]
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            errors.append(f"invalid_json_line={index}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"non_object_line={index}")
            continue
        rows.append(payload)
    return rows, errors


def verify_candidate_event_chain(path: Path) -> dict[str, Any]:
    rows, errors = _read_jsonl(path)
    expected_previous = ""
    for index, row in enumerate(rows, start=1):
        actual_hash = str(row.get("event_hash") or "")
        previous_hash = str(row.get("previous_event_hash") or "")
        unsigned = dict(row)
        unsigned.pop("event_hash", None)
        expected_hash = _canonical_hash(unsigned)
        if previous_hash != expected_previous:
            errors.append(f"previous_hash_mismatch_line={index}")
        if actual_hash != expected_hash:
            errors.append(f"event_hash_mismatch_line={index}")
        expected_previous = actual_hash
    return {
        "ok": not errors,
        "event_count": len(rows),
        "chain_head": expected_previous,
        "errors": errors,
        "path": str(path),
    }


def _append_candidate_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _profitability_baseline(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / "paper_profitability_control_latest.json")
    summary = _as_dict(payload.get("paper_summary"))
    return {
        "artifact_timestamp_utc": payload.get("timestamp_utc", ""),
        "historical_raw_grade": payload.get("raw_profitability_grade", ""),
        "historical_net_pnl": _safe_float(summary.get("ending_net_pnl_total"), 0.0),
        "historical_realized_pnl": _safe_float(summary.get("ending_realized_pnl_total"), 0.0),
        "historical_unrealized_pnl": _safe_float(summary.get("ending_unrealized_pnl_total"), 0.0),
        "policy": "preserve_historical_ledger_and_measure_post_candidate_results_separately",
    }


def _candidate_paths(project_root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    candidate = _as_dict(config.get("candidate"))
    return (
        _project_path(
            project_root,
            candidate.get("state_path") or "governance/runtime/production_candidate_state.json",
        ),
        _project_path(
            project_root,
            candidate.get("event_log_path") or "governance/evidence/production_candidate_events.jsonl",
        ),
    )


def _changed_scopes(state: dict[str, Any], current: dict[str, Any]) -> list[str]:
    accepted = _as_dict(state.get("scope_fingerprints"))
    current_scopes = _as_dict(current.get("scopes"))
    names = sorted(set(accepted) | set(current_scopes))
    return [
        name
        for name in names
        if str(_as_dict(accepted.get(name)).get("sha256") or "")
        != str(_as_dict(current_scopes.get(name)).get("sha256") or "")
    ]


def manage_candidate(
    project_root: Path,
    config: dict[str, Any],
    *,
    initialize: bool = False,
    accept_change: bool = False,
    change_reason: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    now_text = current_time.isoformat()
    state_path, event_path = _candidate_paths(project_root, config)
    state = load_json(state_path)
    current = candidate_fingerprints(project_root, config)
    chain_before = verify_candidate_event_chain(event_path)
    changed_before = _changed_scopes(state, current) if state else []
    operation = "inspect"
    operation_error = ""

    if initialize and accept_change:
        operation_error = "initialize_and_accept_change_are_mutually_exclusive"
    elif initialize:
        operation = "initialize"
        if state:
            operation_error = "candidate_already_initialized"
        elif not chain_before.get("ok", False) or _safe_int(chain_before.get("event_count"), 0) > 0:
            operation_error = "candidate_event_log_not_empty_or_invalid"
    elif accept_change:
        operation = "accept_change"
        minimum_reason = _safe_int(_as_dict(config.get("candidate")).get("minimum_change_reason_chars"), 12)
        if not state:
            operation_error = "candidate_not_initialized"
        elif not chain_before.get("ok", False):
            operation_error = "candidate_event_chain_invalid"
        elif not changed_before:
            operation_error = "no_candidate_drift_to_accept"
        elif len(str(change_reason or "").strip()) < minimum_reason:
            operation_error = f"change_reason_shorter_than_{minimum_reason}_characters"

    if operation in {"initialize", "accept_change"} and not operation_error:
        previous_head = str(chain_before.get("chain_head") or "")
        previous_windows = _as_dict(state.get("scope_windows_started_utc")) if state else {}
        changed = sorted(_as_dict(current.get("scopes")).keys()) if operation == "initialize" else changed_before
        windows = dict(previous_windows)
        for scope in changed:
            windows[scope] = now_text
        generation = 1 if operation == "initialize" else _safe_int(state.get("generation"), 1) + 1
        candidate_id = f"pc-{str(current.get('overall_sha256') or '')[:12]}-g{generation}"
        event_unsigned = {
            "schema_version": 1,
            "timestamp_utc": now_text,
            "event_type": "candidate_initialized" if operation == "initialize" else "candidate_change_accepted",
            "candidate_id": candidate_id,
            "generation": generation,
            "git_head": _git_head(project_root),
            "overall_sha256": current.get("overall_sha256"),
            "changed_scopes": changed,
            "change_reason": str(change_reason or "initial production-excellence candidate freeze").strip(),
            "previous_event_hash": previous_head,
        }
        event = {**event_unsigned, "event_hash": _canonical_hash(event_unsigned)}
        new_state = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "generation": generation,
            "initialized_at_utc": state.get("initialized_at_utc", now_text) if state else now_text,
            "accepted_at_utc": now_text,
            "accepted_git_head": event_unsigned["git_head"],
            "overall_sha256": current.get("overall_sha256"),
            "scope_fingerprints": current.get("scopes", {}),
            "scope_windows_started_utc": windows,
            "profitability_baseline": state.get("profitability_baseline") if state else _profitability_baseline(project_root),
            "last_change": {
                "timestamp_utc": now_text,
                "changed_scopes": changed,
                "change_reason": event_unsigned["change_reason"],
                "event_hash": event["event_hash"],
            },
            "event_chain_head": event["event_hash"],
            "live_execution_authority": False,
        }
        _append_candidate_event(event_path, event)
        _atomic_write_json(state_path, new_state)
        state = new_state

    chain_after = verify_candidate_event_chain(event_path)
    changed_after = _changed_scopes(state, current) if state else []
    return {
        "state": state,
        "current": current,
        "changed_scopes": changed_after,
        "candidate_drift": bool(changed_after),
        "event_chain": chain_after,
        "operation": operation,
        "operation_error": operation_error,
        "state_path": str(state_path),
        "event_path": str(event_path),
    }


def _artifact(project_root: Path, raw_path: Any, max_age_hours: float, now: datetime) -> dict[str, Any]:
    path = _project_path(project_root, raw_path)
    payload = load_json(path)
    age = payload_age_minutes(payload, path, now=now) if payload else None
    fresh = bool(age is not None and age <= max(float(max_age_hours), 0.0) * 60.0)
    return {
        "path": str(path),
        "payload": payload,
        "present": bool(payload),
        "age_minutes": round(float(age), 4) if age is not None else None,
        "fresh": fresh,
        "status": _status(payload.get("overall_status") or payload.get("status")) or "missing",
    }


def _check(check_id: str, title: str, passed: bool, *, evidence: Any = None, action: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "title": title,
        "passed": bool(passed),
        "status": "ready" if passed else "blocked",
        "evidence": evidence,
        "action": "" if passed else action,
    }


def _score_grade(score: float, all_passed: bool) -> str:
    if all_passed and score >= 100.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _pillar(pillar_id: str, title: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for row in checks if row.get("passed", False))
    total = len(checks)
    score = round((100.0 * passed / total), 2) if total else 0.0
    all_passed = bool(total and passed == total)
    return {
        "pillar_id": pillar_id,
        "title": title,
        "ready": all_passed,
        "status": "ready" if all_passed else "blocked",
        "score": score,
        "grade": _score_grade(score, all_passed),
        "passed_check_count": passed,
        "check_count": total,
        "failed_checks": [str(row.get("check_id")) for row in checks if not row.get("passed", False)],
        "checks": checks,
        "next_actions": [str(row.get("action")) for row in checks if not row.get("passed", False) and row.get("action")],
    }


def _window_start(state: dict[str, Any], scopes: Iterable[Any]) -> datetime | None:
    windows = _as_dict(state.get("scope_windows_started_utc"))
    values = [parse_iso_utc(windows.get(str(scope))) for scope in scopes]
    parsed = [value for value in values if value is not None]
    return max(parsed) if parsed else None


def _window_age_hours(start: datetime | None, now: datetime) -> float:
    return max((now - start).total_seconds() / 3600.0, 0.0) if start is not None else 0.0


def _find_domain(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for row in _as_list(payload.get("domains")):
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return {}


def _find_section(payload: dict[str, Any], section_id: str) -> dict[str, Any]:
    for row in _as_list(payload.get("sections")):
        if isinstance(row, dict) and str(row.get("section_id") or "") == section_id:
            return row
    return {}


def _drill_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("drill") or ""): row
        for row in _as_list(payload.get("drills"))
        if isinstance(row, dict) and str(row.get("drill") or "")
    }


def _profitable_sleeves(performance: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
    rows: list[dict[str, Any]] = []
    for sleeve in _as_list(performance.get("sleeve_latest")):
        if not isinstance(sleeve, dict):
            continue
        expectancy = _as_dict(sleeve.get("post_cost_expectancy"))
        total = _safe_float(expectancy.get("total_post_cost_pnl_delta"), 0.0)
        if bool(expectancy.get("positive_lower_confidence_bound_95", False)) and total > 0.0:
            rows.append({"profile": sleeve.get("profile"), "total_post_cost_pnl_delta": total})
    positive_total = sum(_safe_float(row.get("total_post_cost_pnl_delta"), 0.0) for row in rows)
    concentration = None
    if positive_total > 0.0 and rows:
        concentration = max(_safe_float(row.get("total_post_cost_pnl_delta"), 0.0) for row in rows) / positive_total
    return rows, concentration


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    initialize_candidate: bool = False,
    accept_candidate_change: bool = False,
    change_reason: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    effective_config_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    if not effective_config_path.is_absolute():
        effective_config_path = project_root / effective_config_path
    config = load_json(effective_config_path)
    candidate_policy = _as_dict(config.get("candidate"))
    candidate = manage_candidate(
        project_root,
        config,
        initialize=initialize_candidate,
        accept_change=accept_candidate_change,
        change_reason=change_reason,
        now=current_time,
    )
    state = _as_dict(candidate.get("state"))
    chain = _as_dict(candidate.get("event_chain"))
    candidate_ready = bool(
        state
        and not candidate.get("candidate_drift", False)
        and chain.get("ok", False)
        and _safe_int(chain.get("event_count"), 0) >= 1
        and str(chain.get("chain_head") or "") == str(state.get("event_chain_head") or "")
        and not candidate.get("operation_error")
    )

    pillar_1 = _pillar(
        "p01_frozen_candidate",
        "Frozen Production Candidate",
        [
            _check("candidate_initialized", "Candidate state exists", bool(state), evidence=candidate.get("state_path"), action="initialize the production candidate with an explicit freeze command"),
            _check("candidate_has_no_drift", "Candidate fingerprint has no unaccepted drift", bool(state) and not candidate.get("candidate_drift", False), evidence=candidate.get("changed_scopes", []), action="review the changed scopes and explicitly accept the change with a reason"),
            _check("candidate_event_chain_valid", "Candidate events form a valid hash chain", bool(chain.get("ok", False)) and _safe_int(chain.get("event_count"), 0) >= 1, evidence=chain, action="restore the candidate event log from verified evidence and investigate tampering or corruption"),
            _check("candidate_chain_head_matches_state", "State references the verified event-chain head", bool(state) and str(chain.get("chain_head") or "") == str(state.get("event_chain_head") or ""), evidence={"state": state.get("event_chain_head"), "log": chain.get("chain_head")}, action="reconcile candidate state to the verified event chain before continuing"),
            _check("candidate_fingerprint_nonempty", "Candidate fingerprint covers source files", _safe_int(_as_dict(candidate.get("current")).get("file_count"), 0) > 0, evidence=_as_dict(candidate.get("current")), action="repair production-excellence scope globs so critical source files are fingerprinted"),
        ],
    )

    soak_policy = _as_dict(config.get("soak"))
    soak = _artifact(project_root, soak_policy.get("artifact"), _safe_float(soak_policy.get("max_artifact_age_hours"), 2.0), current_time)
    soak_payload = _as_dict(soak.get("payload"))
    soak_start = _window_start(state, _as_list(candidate_policy.get("soak_scopes")))
    soak_age = _window_age_hours(soak_start, current_time)
    required_soak = _safe_float(soak_policy.get("required_hours"), 720.0)
    checkpoint = _safe_float(soak_policy.get("checkpoint_hours"), 168.0)
    pillar_2 = _pillar(
        "p02_clean_30_day_soak",
        "Clean 30-Day Soak",
        [
            _check("soak_candidate_frozen", "Soak runs against an unchanged candidate", candidate_ready, evidence={"candidate_id": state.get("candidate_id"), "changed_scopes": candidate.get("changed_scopes", [])}, action="freeze or reconcile the candidate before counting soak time"),
            _check("soak_artifact_fresh", "Soak evidence is fresh", bool(soak.get("fresh", False)), evidence={k: soak.get(k) for k in ("path", "age_minutes", "status")}, action="refresh unattended-soak readiness from authoritative runtime evidence"),
            _check("soak_runtime_ready", "Unattended paper runtime is A+ and safe", bool(soak_payload.get("ok", False) and soak_payload.get("safe_to_leave_unattended", False) and _status(soak_payload.get("overall_status")) == "ready" and _grade(soak_payload.get("overall_grade")) == "A+"), evidence={"status": soak_payload.get("overall_status"), "grade": soak_payload.get("overall_grade"), "blockers": soak_payload.get("blockers", [])}, action="clear true unattended-soak blockers while keeping live execution locked"),
            _check("seven_day_checkpoint", "Candidate has seven clean days", bool(candidate_ready and soak_age >= checkpoint), evidence={"window_start_utc": soak_start.isoformat() if soak_start else "", "age_hours": round(soak_age, 4), "required_hours": checkpoint}, action="continue the unchanged soak through the seven-day checkpoint"),
            _check("thirty_day_window", "Candidate has 720 clean hours", bool(candidate_ready and soak_age >= required_soak), evidence={"window_start_utc": soak_start.isoformat() if soak_start else "", "age_hours": round(soak_age, 4), "required_hours": required_soak}, action="continue the unchanged candidate until the full 30-day window is complete"),
            _check("soak_has_no_blockers", "Soak has no unresolved blockers", not _as_list(soak_payload.get("blockers")), evidence=soak_payload.get("blockers", []), action="resolve and record every soak blocker before the clean window can count"),
        ],
    )

    recovery_policy = _as_dict(config.get("recovery"))
    recovery = _artifact(project_root, recovery_policy.get("artifact"), _safe_float(recovery_policy.get("max_artifact_age_hours"), 24.0), current_time)
    recovery_payload = _as_dict(recovery.get("payload"))
    drill_index = _drill_rows(recovery_payload)
    recovery_checks = [
        _check("recovery_artifact_fresh", "Recovery drill evidence is fresh", bool(recovery.get("fresh", False)), evidence={k: recovery.get(k) for k in ("path", "age_minutes", "status")}, action="refresh the chaos-drill coordinator"),
    ]
    for drill_name in _as_list(recovery_policy.get("required_drills")):
        name = str(drill_name)
        row = _as_dict(drill_index.get(name))
        passed = bool(
            row
            and row.get("recorded_drill", False)
            and _status(row.get("result")) == "pass"
            and row.get("containment_verified", False)
            and row.get("no_duplicate_orders", False)
            and not row.get("overdue", True)
            and _safe_float(row.get("recovery_seconds"), 10**9) <= _safe_float(recovery_policy.get("max_recovery_seconds"), 300.0)
        )
        recovery_checks.append(
            _check(
                f"drill_{name}",
                f"Verified recovery drill: {name}",
                passed,
                evidence=row,
                action=f"run and record the isolated {name} drill with containment, recovery time, and duplicate-order proof",
            )
        )
    pillar_3 = _pillar("p03_infrastructure_recovery", "Infrastructure Recovery Proof", recovery_checks)

    live_policy = _as_dict(config.get("live_execution"))
    max_live_age = _safe_float(live_policy.get("max_artifact_age_hours"), 12.0)
    live_money = _artifact(project_root, "governance/health/live_money_readiness_contract_latest.json", max_live_age, current_time)
    live_smoke = _artifact(project_root, "governance/health/live_readiness_smoke_latest.json", max_live_age, current_time)
    live_canary = _artifact(project_root, "governance/health/live_canary_control_latest.json", max_live_age, current_time)
    order_ledger_control = _artifact(project_root, "governance/health/live_order_ledger_control_latest.json", max_live_age, current_time)
    production = _artifact(project_root, "governance/health/production_readiness_control_latest.json", max_live_age, current_time)
    live_money_payload = _as_dict(live_money.get("payload"))
    live_smoke_payload = _as_dict(live_smoke.get("payload"))
    live_canary_payload = _as_dict(live_canary.get("payload"))
    order_ledger_payload = _as_dict(order_ledger_control.get("payload"))
    production_payload = _as_dict(production.get("payload"))
    risk_section = _find_section(live_money_payload, "risk_controls")
    firewall_domain = _find_domain(production_payload, "live_execution_risk_firewall")
    readiness_config = load_json(project_root / "config" / "production_readiness_control_v1.json")
    firewall_policy = _as_dict(readiness_config.get("live_execution_risk_firewall"))
    required_sources = [_project_path(project_root, path) for path in _as_list(live_policy.get("required_source_paths"))]
    allowed_assets = {str(item).upper() for item in _as_list(firewall_policy.get("allowed_asset_types"))}
    allowed_instructions = {str(item).upper() for item in _as_list(firewall_policy.get("allowed_instructions"))}
    live_locked = bool(
        not live_canary_payload.get("supervised_canary_ready", False)
        and live_canary_payload.get("live_lane_should_be_read_only", False)
    )
    pillar_4 = _pillar(
        "p04_live_execution_engineering",
        "Live Execution Engineering",
        [
            _check("live_sources_present", "Durable execution-control sources exist", all(path.exists() for path in required_sources), evidence=[str(path) for path in required_sources], action="restore the firewall, durable order ledger, and broker execution integration"),
            _check("durable_order_ledger_ready", "Durable order-intent ledger is fresh and reconciled", bool(order_ledger_control.get("fresh", False) and order_ledger_payload.get("ok", False) and _safe_int(order_ledger_payload.get("submit_unknown_count"), 0) == 0), evidence={"age_minutes": order_ledger_control.get("age_minutes"), "status": order_ledger_payload.get("overall_status"), "blockers": order_ledger_payload.get("blockers", [])}, action="refresh and reconcile the durable live-order ledger"),
            _check("risk_controls_A_ready", "Pre-trade risk controls are fresh and A-grade", bool(live_money.get("fresh", False) and risk_section.get("ready", False) and _grade_at_least(risk_section.get("grade"), "A")), evidence=risk_section, action="refresh and prove the risk-service boundary, limits, and kill switches"),
            _check("live_readiness_smoke", "Validate-only live wiring passes", bool(live_smoke.get("fresh", False) and _status(live_smoke_payload.get("overall_status")) == "ready" and _safe_float(live_smoke_payload.get("readiness_score"), 0.0) >= 100.0), evidence={"status": live_smoke_payload.get("overall_status"), "score": live_smoke_payload.get("readiness_score"), "hard_blocks": live_smoke_payload.get("hard_blocks", [])}, action="repair the validate-only broker and execution smoke path"),
            _check("production_firewall_guarded", "Production order firewall is fail-closed", bool(production.get("fresh", False) and firewall_domain.get("ok", False) and _status(firewall_domain.get("status")) == "ready_guarded"), evidence=firewall_domain, action="refresh production readiness and restore the guarded firewall"),
            _check("microscopic_notional_cap", "Firewall uses the microscopic order cap", 0.0 < _safe_float(firewall_policy.get("max_single_order_notional"), 0.0) <= _safe_float(live_policy.get("max_single_order_notional"), 100.0), evidence=firewall_policy.get("max_single_order_notional"), action="reduce the initial live order-notional cap to the production-excellence ceiling"),
            _check("microscopic_daily_loss_cap", "Firewall uses the microscopic daily loss cap", 0.0 < _safe_float(firewall_policy.get("max_daily_loss"), 0.0) <= _safe_float(live_policy.get("max_daily_loss"), 25.0), evidence=firewall_policy.get("max_daily_loss"), action="reduce the initial live daily-loss cap to the production-excellence ceiling"),
            _check("cash_equity_only", "Initial canary permits cash equities only", allowed_assets == {str(item).upper() for item in _as_list(live_policy.get("allowed_asset_types"))}, evidence=sorted(allowed_assets), action="restrict initial live orders to the configured cash-equity asset set"),
            _check("long_only_instructions", "Initial canary permits BUY and SELL only", allowed_instructions == {str(item).upper() for item in _as_list(live_policy.get("allowed_instructions"))}, evidence=sorted(allowed_instructions), action="block short, option, futures, roll, and leveraged instructions from the first canary"),
            _check("live_read_only_lock", "Live submit remains read-only until promotion", live_locked, evidence={"mode": live_canary_payload.get("recommended_mode"), "blocking_reasons": live_canary_payload.get("blocking_reasons", [])}, action="restore the read-only live lane until every promotion gate is complete"),
        ],
    )

    fill_policy = _as_dict(config.get("fill_evidence"))
    fill = _artifact(project_root, fill_policy.get("artifact"), _safe_float(fill_policy.get("max_artifact_age_hours"), 12.0), current_time)
    fill_payload = _as_dict(fill.get("payload"))
    fill_window = _as_dict(fill_payload.get("calibration_window"))
    fill_cutoff = parse_iso_utc(fill_window.get("cutoff_utc"))
    fill_required_start = _window_start(state, ["execution", "data", "dependencies"])
    independent_samples = _safe_int(fill_payload.get("independent_samples"), 0)
    pillar_5 = _pillar(
        "p05_independent_fill_evidence",
        "Independent Fill Evidence",
        [
            _check("fill_artifact_fresh", "Fill-calibration evidence is fresh", bool(fill.get("fresh", False)), evidence={k: fill.get(k) for k in ("path", "age_minutes", "status")}, action="refresh paper execution calibration"),
            _check("fill_window_matches_candidate", "Calibration excludes pre-candidate evidence", bool(fill_required_start and fill_cutoff and fill_cutoff >= fill_required_start), evidence={"candidate_window_start_utc": fill_required_start.isoformat() if fill_required_start else "", "calibration_cutoff_utc": fill_cutoff.isoformat() if fill_cutoff else ""}, action="refresh calibration after the candidate freeze so old fill samples are excluded"),
            _check("independent_fill_minimum", "Independent fills meet the calibration minimum", independent_samples >= _safe_int(fill_policy.get("minimum_independent_samples"), 30), evidence=independent_samples, action="continue collecting broker-paper or explicit market-replay fill observations"),
            _check("independent_fill_promotion_strength", "Independent fills meet the stronger promotion sample floor", independent_samples >= _safe_int(fill_policy.get("minimum_promotion_samples"), 100), evidence=independent_samples, action="continue independent fill collection beyond the basic calibration minimum"),
            _check("independent_evidence_ready", "Calibration marks independent evidence ready", bool(fill_payload.get("independent_evidence_ready", False)), evidence=fill_payload.get("failed_checks", []), action="resolve fill-model bias, MAE, p95, or sample-quality failures"),
            _check("model_fills_not_substituted", "Model-derived fills do not count as independent samples", independent_samples >= 0 and _safe_int(fill_payload.get("samples"), independent_samples) == independent_samples, evidence={"independent_samples": independent_samples, "model_derived_samples": fill_payload.get("model_derived_samples", 0), "reported_samples": fill_payload.get("samples")}, action="separate model-derived diagnostics from independent calibration evidence"),
        ],
    )

    promotion_policy = _as_dict(config.get("promotion"))
    promotion = _artifact(project_root, promotion_policy.get("artifact"), _safe_float(promotion_policy.get("max_artifact_age_hours"), 24.0), current_time)
    packet = _artifact(project_root, promotion_policy.get("packet_artifact"), _safe_float(promotion_policy.get("max_artifact_age_hours"), 24.0), current_time)
    promotion_payload = _as_dict(promotion.get("payload"))
    packet_payload = _as_dict(packet.get("payload"))
    promotion_details = _as_dict(promotion_payload.get("details"))
    promotion_summary = _as_dict(promotion_details.get("promotion"))
    candidate_ids = [str(item) for item in _as_list(promotion_details.get("promotion_candidate_ids")) if str(item)]
    pillar_6 = _pillar(
        "p06_real_promotion_candidates",
        "Real Promotion Candidates",
        [
            _check("promotion_artifacts_fresh", "Promotion gate and packet are fresh", bool(promotion.get("fresh", False) and packet.get("fresh", False)), evidence={"gate_age_minutes": promotion.get("age_minutes"), "packet_age_minutes": packet.get("age_minutes")}, action="refresh the promotion pipeline and packet builder"),
            _check("considered_bot_floor", "At least four bots are independently considered", _safe_int(promotion_summary.get("considered_bots"), 0) >= _safe_int(promotion_policy.get("minimum_considered_bots"), 4), evidence=promotion_summary, action="produce enough qualified, independently evaluated bots for a real cohort"),
            _check("candidate_bot_floor", "At least four promotion candidates qualify", len(candidate_ids) >= _safe_int(promotion_policy.get("minimum_candidate_bots"), 4), evidence=candidate_ids, action="keep weak bots collect-only and graduate at least four evidence-backed candidates"),
            _check("promotion_quality_ready", "Promotion quality gate passes", bool(promotion_payload.get("ok", False) and not _as_list(promotion_payload.get("failed_checks"))), evidence=promotion_payload.get("failed_checks", []), action="clear replay, leakage, graduation, admission, and paper-evidence blockers"),
            _check("promotion_packet_ready", "Promotion packet is complete and reviewable", bool(packet_payload.get("ok", False) and packet_payload.get("packet_complete", packet_payload.get("ready_for_committee", False))), evidence={"ok": packet_payload.get("ok"), "packet_complete": packet_payload.get("packet_complete"), "ready_for_committee": packet_payload.get("ready_for_committee")}, action="build a complete, hash-backed promotion packet for the qualified cohort"),
            _check("leak_overfit_replay_ready", "Leakage, overfit, replay, and schema gates pass", all(bool(promotion_details.get(key, False)) for key in ("leak_overfit_ok", "replay_ok", "golden_replay_regression_ok", "retrain_schema_compatibility_ok")), evidence={key: promotion_details.get(key) for key in ("leak_overfit_ok", "replay_ok", "golden_replay_regression_ok", "retrain_schema_compatibility_ok")}, action="repair any leakage, replay, schema, or overfit evidence failure"),
        ],
    )

    profit_policy = _as_dict(config.get("profitability"))
    performance = _artifact(project_root, profit_policy.get("performance_artifact"), _safe_float(profit_policy.get("max_artifact_age_hours"), 12.0), current_time)
    profit_control = _artifact(project_root, profit_policy.get("control_artifact"), _safe_float(profit_policy.get("max_artifact_age_hours"), 12.0), current_time)
    performance_payload = _as_dict(performance.get("payload"))
    control_payload = _as_dict(profit_control.get("payload"))
    expectancy = _as_dict(performance_payload.get("post_cost_expectancy"))
    profitable_sleeves, positive_concentration = _profitable_sleeves(performance_payload)
    profit_start = _window_start(state, _as_list(candidate_policy.get("profitability_scopes")))
    first_profit_sample = parse_iso_utc(expectancy.get("first_sample_timestamp_utc"))
    baseline = _as_dict(state.get("profitability_baseline"))
    current_summary = _as_dict(control_payload.get("paper_summary"))
    forward_raw_delta = _safe_float(current_summary.get("ending_net_pnl_total"), 0.0) - _safe_float(baseline.get("historical_net_pnl"), 0.0)
    drawdown = _safe_float(expectancy.get("max_cumulative_drawdown_post_cost_pnl"), float("inf"))
    post_cost_total = _safe_float(expectancy.get("total_post_cost_pnl_delta"), 0.0)
    drawdown_ratio = drawdown / post_cost_total if post_cost_total > 0.0 and drawdown != float("inf") else None
    raw_grade = _grade(control_payload.get("raw_profitability_grade"))
    controlled_grade = _grade(control_payload.get("controlled_profitability_grade"))
    grade_separated = bool(raw_grade and controlled_grade and raw_grade != controlled_grade and f"{raw_grade} raw" in str(control_payload.get("profitability_display_grade") or ""))
    pillar_7 = _pillar(
        "p07_profitability_evidence",
        "Post-Cost Profitability Evidence",
        [
            _check("profit_artifacts_fresh", "Profitability evidence is fresh", bool(performance.get("fresh", False) and profit_control.get("fresh", False)), evidence={"performance_age_minutes": performance.get("age_minutes"), "control_age_minutes": profit_control.get("age_minutes")}, action="refresh paper performance and profitability control from current trade deltas"),
            _check("profit_window_matches_candidate", "Post-cost samples belong to the frozen candidate", bool(profit_start and first_profit_sample and first_profit_sample >= profit_start), evidence={"candidate_window_start_utc": profit_start.isoformat() if profit_start else "", "first_sample_timestamp_utc": first_profit_sample.isoformat() if first_profit_sample else ""}, action="exclude pre-candidate trade deltas from the forward profitability cohort"),
            _check("post_cost_sample_floor", "Post-cost sample floor is met", _safe_int(expectancy.get("sample_count"), 0) >= _safe_int(profit_policy.get("minimum_post_cost_samples"), 100), evidence=expectancy.get("sample_count", 0), action="continue collecting schema-v2 post-cost trade deltas"),
            _check("positive_post_cost_lcb", "The 95% lower confidence bound is positive", bool(expectancy.get("positive_lower_confidence_bound_95", False)), evidence=expectancy, action="do not promote until post-cost expectancy is positive with confidence"),
            _check("positive_forward_pnl", "Forward cohort P&L is positive", post_cost_total > 0.0 and forward_raw_delta > 0.0, evidence={"post_cost_pnl_delta": post_cost_total, "forward_raw_ledger_delta": round(forward_raw_delta, 6), "historical_raw_ledger_preserved": baseline}, action="allow the frozen cohort to earn positive raw and post-cost results without rewriting history"),
            _check("profitable_sleeve_diversity", "At least three sleeves have positive confident expectancy", len(profitable_sleeves) >= _safe_int(profit_policy.get("minimum_profitable_sleeves"), 3), evidence=profitable_sleeves, action="promote only after profitability is distributed across multiple qualified sleeves"),
            _check("profit_concentration_bounded", "No sleeve dominates positive P&L", positive_concentration is not None and positive_concentration <= _safe_float(profit_policy.get("maximum_single_sleeve_positive_pnl_share"), 0.5), evidence=positive_concentration, action="reduce dependence on a single sleeve or isolated winning trade"),
            _check("drawdown_bounded", "Post-cost gain exceeds maximum cumulative drawdown", drawdown_ratio is not None and drawdown_ratio <= 1.0, evidence={"max_cumulative_drawdown": None if drawdown == float("inf") else drawdown, "drawdown_to_profit_ratio": drawdown_ratio}, action="collect drawdown evidence and keep risk-adjusted losses within the cohort's earned profit"),
            _check("raw_controlled_grades_separate", "Raw results remain separate from controlled safety", grade_separated, evidence={"raw_grade": raw_grade, "controlled_grade": controlled_grade, "display": control_payload.get("profitability_display_grade")}, action="restore explicit raw-versus-controlled profitability labeling"),
        ],
    )

    canary_policy = _as_dict(config.get("canary"))
    canary_control = _artifact(project_root, canary_policy.get("control_artifact"), max_live_age, current_time)
    canary_rollout = _artifact(project_root, canary_policy.get("rollout_artifact"), max_live_age, current_time)
    canary_control_payload = _as_dict(canary_control.get("payload"))
    rollout_payload = _as_dict(canary_rollout.get("payload"))
    effective_weight = max(_safe_float(canary_control_payload.get("target_canary_weight"), 0.0), _safe_float(canary_control_payload.get("applied_canary_weight"), 0.0))
    canary_envelope_safe = bool(allowed_assets == {"EQUITY"} and allowed_instructions == {"BUY", "SELL"})
    pillar_8 = _pillar(
        "p08_controlled_canary_graduation",
        "Controlled Canary Graduation",
        [
            _check("canary_artifacts_fresh", "Canary control and rollout evidence are fresh", bool(canary_control.get("fresh", False) and canary_rollout.get("fresh", False)), evidence={"control_age_minutes": canary_control.get("age_minutes"), "rollout_age_minutes": canary_rollout.get("age_minutes")}, action="refresh canary rollout and live canary control"),
            _check("canary_baseline_samples", "Baseline cohort has sufficient samples", _safe_int(rollout_payload.get("baseline_samples"), 0) >= _safe_int(canary_policy.get("minimum_baseline_samples"), 400), evidence=rollout_payload.get("baseline_samples", 0), action="continue baseline paper collection"),
            _check("canary_candidate_samples", "Canary cohort has sufficient samples", _safe_int(rollout_payload.get("canary_samples"), 0) >= _safe_int(canary_policy.get("minimum_canary_samples"), 400), evidence=rollout_payload.get("canary_samples", 0), action="continue candidate paper-canary collection"),
            _check("canary_edge_ready", "Canary beats its baseline and is eligible", bool(rollout_payload.get("eligible", False) and rollout_payload.get("promote_canary", False)), evidence={"eligible": rollout_payload.get("eligible"), "promote_canary": rollout_payload.get("promote_canary"), "edge_delta": rollout_payload.get("edge_delta")}, action="keep the canary in paper until it beats the baseline with enough samples"),
            _check("microscopic_canary_weight", "Initial canary weight is at most 1%", 0.0 < effective_weight <= _safe_float(canary_policy.get("max_initial_weight"), 0.01), evidence=effective_weight, action="cap the first live canary at or below 1%"),
            _check("unleveraged_equity_envelope", "Initial canary is long-only cash equity", canary_envelope_safe, evidence={"allowed_asset_types": sorted(allowed_assets), "allowed_instructions": sorted(allowed_instructions)}, action="remove leverage, options, futures, and short-sale authority from the initial canary"),
            _check("promotion_packet_required", "Canary depends on a complete promotion packet", bool(packet_payload.get("ok", False)), evidence={"packet_ok": packet_payload.get("ok"), "packet_complete": packet_payload.get("packet_complete")}, action="complete and review the promotion packet before live canary consideration"),
            _check("read_only_until_release", "Canary remains read-only before explicit release", bool(canary_control_payload.get("live_lane_should_be_read_only", False)), evidence=canary_control_payload.get("blocking_reasons", []), action="restore the read-only release boundary"),
        ],
    )

    grading_policy = _as_dict(config.get("grading_integrity"))
    content_store = _artifact(project_root, "governance/content_store/latest.json", _safe_float(grading_policy.get("max_artifact_age_hours"), 12.0), current_time)
    source_verification = _artifact(project_root, "governance/health/source_verification_latest.json", _safe_float(grading_policy.get("max_artifact_age_hours"), 12.0), current_time)
    content_payload = _as_dict(content_store.get("payload"))
    source_payload = _as_dict(source_verification.get("payload"))
    pillar_9 = _pillar(
        "p09_grading_integrity",
        "Non-Gameable Grading",
        [
            _check("a_plus_requires_all_checks", "A+ requires every check to pass", bool(grading_policy.get("a_plus_requires_all_checks", False)), evidence=grading_policy, action="restore fail-closed A+ grading policy"),
            _check("missing_evidence_scores_zero", "Missing evidence scores zero", _safe_int(grading_policy.get("missing_evidence_score"), -1) == 0, evidence=grading_policy.get("missing_evidence_score"), action="set missing evidence to a zero score"),
            _check("candidate_chain_tamper_evident", "Candidate evidence is hash-chained", bool(chain.get("ok", False) and _safe_int(chain.get("event_count"), 0) >= 1), evidence=chain, action="repair the tamper-evident candidate event chain"),
            _check("raw_controlled_labels_honest", "Raw and controlled profitability labels are separate", grade_separated, evidence={"raw_grade": raw_grade, "controlled_grade": controlled_grade, "display": control_payload.get("profitability_display_grade")}, action="do not allow controlled safety grades to overwrite raw financial results"),
            _check("content_store_fresh_and_clean", "Immutable content-store evidence is fresh and clean", bool(content_store.get("fresh", False) and content_payload.get("ok", False) and _safe_int(content_payload.get("unsafe_skipped_blob_count"), 0) == 0), evidence={"age_minutes": content_store.get("age_minutes"), "ok": content_payload.get("ok"), "unsafe_skipped_blob_count": content_payload.get("unsafe_skipped_blob_count")}, action="refresh and verify the content-addressed evidence store"),
            _check("source_verification_ready", "Source provenance is verified", bool(source_verification.get("fresh", False) and source_payload.get("ok", False) and _status(source_payload.get("overall_status")) == "ready"), evidence={"status": source_payload.get("overall_status"), "unverified_sources": source_payload.get("unverified_sources", [])}, action="verify or quarantine every unverified decision input source"),
        ],
    )

    institutional = _as_dict(config.get("institutional_operations"))
    institutional_max_age = _safe_float(institutional.get("max_artifact_age_hours"), 36.0)
    institutional_specs = [
        ("production_readiness", institutional.get("production_readiness_artifact")),
        ("security", institutional.get("security_artifact")),
        ("remote_alert", institutional.get("remote_alert_artifact")),
        ("backup_restore", institutional.get("backup_restore_artifact")),
        ("blackstart", institutional.get("blackstart_artifact")),
    ]
    institutional_rows = {name: _artifact(project_root, path, institutional_max_age, current_time) for name, path in institutional_specs}
    institutional_checks: list[dict[str, Any]] = []
    for name, row in institutional_rows.items():
        payload = _as_dict(row.get("payload"))
        if name == "production_readiness":
            passed = bool(row.get("fresh", False) and payload.get("live_money_production_bar_ready", False) and payload.get("live_money_canary_consideration_ready", False))
        elif name == "blackstart":
            passed = bool(row.get("fresh", False) and payload.get("ok", False) and payload.get("production_grade_ready", False))
        else:
            passed = bool(row.get("fresh", False) and payload.get("ok", False) and _status(payload.get("overall_status") or payload.get("status")) not in BAD_STATUSES)
        institutional_checks.append(
            _check(
                f"{name}_ready",
                f"{name.replace('_', ' ').title()} is fresh and ready",
                passed,
                evidence={"path": row.get("path"), "age_minutes": row.get("age_minutes"), "status": row.get("status"), "ok": payload.get("ok")},
                action=f"refresh and prove the {name.replace('_', ' ')} operating control",
            )
        )
    rollback_manifest = project_root / "governance" / "rollback" / "production_rollback_manifest_latest.json"
    institutional_checks.append(_check("rollback_manifest_present", "A rollback manifest is available", bool(load_json(rollback_manifest)), evidence=str(rollback_manifest), action="build and verify the production rollback manifest"))
    pillar_10 = _pillar("p10_institutional_operations", "Institutional Operating Proof", institutional_checks)

    pillars = [pillar_1, pillar_2, pillar_3, pillar_4, pillar_5, pillar_6, pillar_7, pillar_8, pillar_9, pillar_10]
    ready_count = sum(1 for pillar in pillars if pillar.get("ready", False))
    all_ready = ready_count == len(pillars)
    average_score = round(sum(_safe_float(pillar.get("score"), 0.0) for pillar in pillars) / max(len(pillars), 1), 2)
    overall_grade = "A+" if all_ready else _score_grade(average_score, False)
    blockers = [str(pillar.get("pillar_id")) for pillar in pillars if not pillar.get("ready", False)]
    next_actions: list[str] = []
    for pillar in pillars:
        for action in _as_list(pillar.get("next_actions")):
            if str(action) and str(action) not in next_actions:
                next_actions.append(str(action))

    return {
        "schema_version": 1,
        "timestamp_utc": current_time.isoformat(),
        "policy_id": str(config.get("policy_id") or "production_excellence_10_pillar_v1"),
        "source": "production_excellence_control",
        "ok": all_ready,
        "overall_status": "ready" if all_ready else "blocked",
        "overall_grade": overall_grade,
        "overall_score": average_score,
        "ten_out_of_ten_ready": all_ready,
        "live_money_consideration_ready": all_ready,
        "live_execution_authority": False,
        "live_orders_must_remain_disabled": not all_ready,
        "ready_pillar_count": ready_count,
        "pillar_count": len(pillars),
        "blocked_pillars": blockers,
        "candidate": {
            "candidate_id": state.get("candidate_id", ""),
            "generation": state.get("generation", 0),
            "candidate_ready": candidate_ready,
            "candidate_drift": bool(candidate.get("candidate_drift", False)),
            "changed_scopes": candidate.get("changed_scopes", []),
            "operation": candidate.get("operation"),
            "operation_error": candidate.get("operation_error"),
            "state_path": candidate.get("state_path"),
            "event_path": candidate.get("event_path"),
            "event_chain": chain,
            "scope_windows_started_utc": state.get("scope_windows_started_utc", {}),
            "historical_profitability_baseline": baseline,
        },
        "pillars": pillars,
        "next_actions": next_actions,
        "grading_contract": {
            "a_plus_requires_all_ten_pillars": True,
            "missing_evidence_is_failure": True,
            "controlled_safety_is_not_profitability_proof": True,
            "historical_raw_ledger_is_never_rewritten": True,
            "candidate_event_log_is_tamper_evident_not_claimed_immutable": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ten-pillar production-excellence contract.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--apply", action="store_true", help="Write the latest health artifact.")
    parser.add_argument("--initialize-candidate", action="store_true", help="Freeze the first versioned production candidate.")
    parser.add_argument("--accept-candidate-change", action="store_true", help="Accept detected source drift and selectively reset affected evidence windows.")
    parser.add_argument("--change-reason", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if (args.initialize_candidate or args.accept_candidate_change) and not args.apply:
        parser.error("candidate mutations require --apply")
    if args.initialize_candidate and args.accept_candidate_change:
        parser.error("candidate mutation flags are mutually exclusive")

    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    out_path = args.out if args.out.is_absolute() else project_root / args.out
    payload = build_payload(
        project_root,
        config_path=config_path,
        initialize_candidate=args.initialize_candidate,
        accept_candidate_change=args.accept_candidate_change,
        change_reason=args.change_reason,
    )
    if args.apply:
        write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "production_excellence_control "
            f"status={payload['overall_status']} grade={payload['overall_grade']} "
            f"pillars={payload['ready_pillar_count']}/{payload['pillar_count']} "
            f"live_consideration={int(bool(payload['live_money_consideration_ready']))}"
        )
    return 0 if not payload.get("candidate", {}).get("operation_error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
