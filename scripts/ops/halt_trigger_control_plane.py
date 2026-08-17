import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from core.halt_flags import inspect_halt_flag

THAW_SAFE_RUNTIME_STATES = {
    "ready",
    "guarded_live_read_only",
    "coverage_cycles_ready",
    "scheduled_off_hours_launch",
    "off_hours_cold_lane_launch_ready",
}
PROTECTIVE_THROTTLE_PROFILES = {
    "protect",
    "protect_live",
    "critical",
    "halt",
    "panic",
}
PROTECTIVE_FLUIDITY_BANDS = {
    "protect",
    "protect_live",
    "critical",
    "halt",
}
BAD_ARTIFACT_STATES = {"missing", "invalid_json", "stale"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _health_root() -> Path:
    return PROJECT_ROOT / "governance" / "health"


def _watchdog_root() -> Path:
    return PROJECT_ROOT / "governance" / "watchdog"


def _alerts_root() -> Path:
    return PROJECT_ROOT / "governance" / "alerts"


def _latest_path() -> Path:
    return _health_root() / "halt_trigger_control_plane_latest.json"


def _events_path() -> Path:
    return _watchdog_root() / "halt_trigger_control_plane_events.jsonl"


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "missing"
    except json.JSONDecodeError:
        return {}, "invalid_json"
    except Exception as exc:
        return {}, f"read_error:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return {}, f"payload_not_object:{type(payload).__name__}"
    return payload, ""


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_age_seconds(path: Path) -> float | None:
    try:
        return max((_now() - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)).total_seconds(), 0.0)
    except Exception:
        return None


def _artifact(name: str, path: Path, *, max_age_seconds: int, required: bool) -> dict[str, Any]:
    payload, error = _load_json(path)
    exists = path.exists()
    timestamp = _parse_timestamp(payload.get("timestamp_utc")) if payload else None
    timestamp_source = "timestamp_utc" if timestamp else "mtime"
    if timestamp:
        age = max((_now() - timestamp).total_seconds(), 0.0)
    else:
        age = _file_age_seconds(path) if exists else None

    state = "fresh"
    if not exists:
        state = "missing"
    elif error:
        state = error
    elif age is not None and age > max_age_seconds:
        state = "stale"

    return {
        "name": name,
        "path": str(path),
        "exists": bool(exists),
        "required": bool(required),
        "state": state,
        "error": error,
        "timestamp_utc": timestamp.isoformat() if timestamp else str(payload.get("timestamp_utc") or ""),
        "timestamp_source": timestamp_source,
        "age_seconds": round(age, 3) if age is not None else None,
        "max_age_seconds": int(max_age_seconds),
        "payload": payload,
    }


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _execution_expected() -> bool:
    return _as_bool(os.getenv("ALLOW_ORDER_EXECUTION"), False) and not _as_bool(os.getenv("MARKET_DATA_ONLY"), True)


def _active_names(payload: dict[str, Any]) -> list[str]:
    return sorted(str(name) for name, active in payload.items() if _as_bool(active, False))


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    name: str,
    severity: str,
    source: str,
    summary: str,
    detail: dict[str, Any] | None = None,
    blocks_live_execution: bool = True,
    blocks_halt_clear: bool = True,
    blocks_heavy_viewer: bool = False,
) -> None:
    row = {
        "name": name,
        "severity": severity,
        "source": source,
        "summary": summary,
        "blocks_live_execution": bool(blocks_live_execution),
        "blocks_halt_clear": bool(blocks_halt_clear),
        "blocks_heavy_viewer": bool(blocks_heavy_viewer),
    }
    if detail:
        row["detail"] = detail
    if not any(existing["name"] == name and existing["source"] == source for existing in issues):
        issues.append(row)


def _inspect_strict_flag(path: Path, *, name: str) -> dict[str, Any]:
    inspected = inspect_halt_flag(path)
    return {
        "name": name,
        "path": inspected["path"],
        "active": bool(inspected["exists"]),
        "valid": (not inspected["exists"]) or bool(inspected["valid"]),
        "reason": str(inspected.get("reason") or ""),
        "error": str(inspected.get("error") or ""),
        "size_bytes": int(inspected.get("size_bytes") or 0),
        "payload": inspected.get("payload") if isinstance(inspected.get("payload"), dict) else {},
        "format": "json_reason" if bool(inspected.get("valid")) else "",
    }


def _parse_key_value_flag(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            parsed[key] = value.strip()
    return parsed


def _inspect_paper_trade_lock(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": "paper_trade_lock",
        "path": str(path),
        "active": path.exists(),
        "valid": not path.exists(),
        "reason": "",
        "error": "",
        "size_bytes": 0,
        "payload": {},
        "format": "",
    }
    if not path.exists():
        return out
    try:
        out["size_bytes"] = int(path.stat().st_size)
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        out["error"] = f"read_error:{type(exc).__name__}"
        return out
    if not raw.strip():
        out["error"] = "empty_payload"
        return out
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = _parse_key_value_flag(raw)
        if payload:
            out["payload"] = payload
            out["format"] = "key_value"
            out["reason"] = str(payload.get("reason") or payload.get("policy") or "paper_trade_lock")
            out["valid"] = True
            return out
        out["error"] = "invalid_json"
        return out
    if not isinstance(payload, dict):
        out["error"] = f"payload_not_object:{type(payload).__name__}"
        return out
    out["payload"] = payload
    out["format"] = "json"
    out["reason"] = str(payload.get("reason") or payload.get("policy") or "paper_trade_lock")
    out["valid"] = True
    return out


def _manual_flags() -> dict[str, Any]:
    health = _health_root()
    return {
        "operator_stop": _inspect_strict_flag(health / "OPERATOR_STOP.flag", name="operator_stop"),
        "global_halt": _inspect_strict_flag(health / "GLOBAL_TRADING_HALT.flag", name="global_halt"),
        "paper_trade_lock": _inspect_paper_trade_lock(health / "PAPER_TRADE_LOCK.flag"),
    }


def _artifact_set() -> dict[str, dict[str, Any]]:
    health = _health_root()
    return {
        "global_killswitch": _artifact(
            "global_killswitch",
            health / "global_killswitch_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_GLOBAL_KILLSWITCH_MAX_AGE_SECONDS", "900")),
            required=True,
        ),
        "health_gates": _artifact(
            "health_gates",
            health / "health_gates_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_HEALTH_GATES_MAX_AGE_SECONDS", "900")),
            required=True,
        ),
        "runtime_throttle": _artifact(
            "runtime_throttle",
            health / "runtime_throttle_control_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_RUNTIME_THROTTLE_MAX_AGE_SECONDS", "900")),
            required=True,
        ),
        "live_runtime_separation": _artifact(
            "live_runtime_separation",
            health / "live_runtime_separation_control_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_LIVE_RUNTIME_MAX_AGE_SECONDS", "1800")),
            required=True,
        ),
        "incident_auto_halt": _artifact(
            "incident_auto_halt",
            _alerts_root() / "incident_auto_halt_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_INCIDENT_AUTO_HALT_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
        "paper_execution_truth": _artifact(
            "paper_execution_truth",
            health / "paper_execution_truth_layer_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_PAPER_TRUTH_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
        "paper_400_ramp": _artifact(
            "paper_400_ramp",
            health / "paper_400_ramp_latest.json",
            max_age_seconds=int(os.getenv("HALT_TRIGGER_PAPER_400_RAMP_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
    }


def _evaluate_manual_flags(flags: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    operator = flags["operator_stop"]
    global_halt = flags["global_halt"]
    paper_lock = flags["paper_trade_lock"]
    if operator["active"]:
        _add_issue(
            issues,
            name="manual_operator_stop_active",
            severity="critical",
            source="OPERATOR_STOP.flag",
            summary="Operator stop is engaged.",
            detail={"reason": operator.get("reason"), "valid": operator.get("valid")},
        )
    if global_halt["active"]:
        _add_issue(
            issues,
            name="manual_global_halt_active",
            severity="critical",
            source="GLOBAL_TRADING_HALT.flag",
            summary="Global trading halt is engaged.",
            detail={"reason": global_halt.get("reason"), "valid": global_halt.get("valid")},
        )
    if paper_lock["active"]:
        _add_issue(
            issues,
            name="paper_trade_lock_active",
            severity="critical",
            source="PAPER_TRADE_LOCK.flag",
            summary="Paper trade lock keeps live order execution disabled.",
            detail={"reason": paper_lock.get("reason"), "valid": paper_lock.get("valid"), "format": paper_lock.get("format")},
            blocks_halt_clear=False,
        )
    for flag_name in ("operator_stop", "global_halt"):
        flag = flags[flag_name]
        if flag["active"] and not flag["valid"]:
            _add_issue(
                issues,
                name=f"invalid_{flag_name}_payload",
                severity="critical",
                source=str(Path(flag["path"]).name),
                summary="Active halt flag payload is malformed or missing a reason; fail closed until it is handled by an operator.",
                detail={"error": flag.get("error"), "size_bytes": flag.get("size_bytes")},
                blocks_heavy_viewer=False,
            )
    if paper_lock["active"] and not paper_lock["valid"]:
        _add_issue(
            issues,
            name="invalid_paper_trade_lock_payload",
            severity="warning",
            source="PAPER_TRADE_LOCK.flag",
            summary="Paper trade lock payload is unreadable; keep live orders disabled and repair the lock metadata.",
            detail={"error": paper_lock.get("error"), "size_bytes": paper_lock.get("size_bytes")},
            blocks_halt_clear=False,
        )


def _evaluate_artifact_health(artifacts: dict[str, dict[str, Any]], issues: list[dict[str, Any]]) -> None:
    for name, artifact in artifacts.items():
        state = str(artifact.get("state") or "")
        required = bool(artifact.get("required", False))
        if state in BAD_ARTIFACT_STATES and required:
            _add_issue(
                issues,
                name=f"critical_artifact_{state}:{name}",
                severity="critical",
                source=str(artifact.get("path") or name),
                summary=f"Required safety artifact {name} is {state}; fail closed until it refreshes.",
                detail={
                    "age_seconds": artifact.get("age_seconds"),
                    "max_age_seconds": artifact.get("max_age_seconds"),
                    "error": artifact.get("error"),
                },
                blocks_heavy_viewer=state == "stale" and name in {"runtime_throttle", "live_runtime_separation"},
            )
        elif state in BAD_ARTIFACT_STATES:
            _add_issue(
                issues,
                name=f"advisory_artifact_{state}:{name}",
                severity="warning",
                source=str(artifact.get("path") or name),
                summary=f"Optional safety artifact {name} is {state}.",
                detail={"age_seconds": artifact.get("age_seconds"), "max_age_seconds": artifact.get("max_age_seconds")},
                blocks_live_execution=False,
                blocks_halt_clear=False,
            )


def _evaluate_global_killswitch(payload: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    halt_state = _lower(payload.get("halt_state"))
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
    clear_blockers = payload.get("clear_blockers") if isinstance(payload.get("clear_blockers"), list) else []
    if _as_bool(payload.get("halt"), False) or halt_state == "active":
        _add_issue(
            issues,
            name="global_killswitch_active",
            severity="critical",
            source="global_killswitch_latest.json",
            summary="Global risk killswitch reports an active halt.",
            detail={"halt_state": halt_state or "", "reason_count": len(reasons)},
        )
    if reasons:
        _add_issue(
            issues,
            name="global_killswitch_reasons_present",
            severity="critical",
            source="global_killswitch_latest.json",
            summary="Global risk killswitch has active reasons.",
            detail={"reasons": reasons},
        )
    if clear_blockers:
        _add_issue(
            issues,
            name="global_killswitch_clear_blocked",
            severity="critical",
            source="global_killswitch_latest.json",
            summary="Global halt auto-clear has blockers.",
            detail={"clear_blockers": clear_blockers},
        )


def _evaluate_health_gates(payload: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    hard_gates = payload.get("hard_gates") if isinstance(payload.get("hard_gates"), dict) else {}
    active = _active_names(hard_gates)
    critical_priority = []
    ingestion = payload.get("ingestion_pressure")
    if isinstance(ingestion, dict):
        critical_priority = [
            str(item)
            for item in ingestion.get("critical_priority_failures", [])
            if str(item).strip()
        ]
    if _as_bool(payload.get("hard_gate_triggered"), False) and active:
        _add_issue(
            issues,
            name="health_hard_gates_active",
            severity="critical",
            source="health_gates_latest.json",
            summary="Health gates report active hard stops.",
            detail={"hard_gates": active, "critical_priority_failures": critical_priority},
            blocks_heavy_viewer=True,
        )


def _evaluate_runtime_throttle(payload: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    profile = _lower(payload.get("throttle_profile"))
    overall_status = _lower(payload.get("overall_status"))
    release = payload.get("release_contract") if isinstance(payload.get("release_contract"), dict) else {}
    storage = payload.get("storage_stabilization") if isinstance(payload.get("storage_stabilization"), dict) else {}
    mac = payload.get("mac_fluidity_contract") if isinstance(payload.get("mac_fluidity_contract"), dict) else {}
    mac_band = _lower(mac.get("fluidity_band"))
    storage_mode = _lower(storage.get("recommended_operating_mode"))
    storage_backlog = _lower(storage.get("backlog_drain_status"))
    heavy_blocks: list[str] = []

    if profile in PROTECTIVE_THROTTLE_PROFILES or overall_status in {"blocked", "critical"}:
        heavy_blocks.append("runtime_throttle_protective_profile")
        _add_issue(
            issues,
            name="runtime_throttle_protective_profile",
            severity="critical",
            source="runtime_throttle_control_latest.json",
            summary="Runtime throttle is in a protective profile.",
            detail={"overall_status": overall_status, "throttle_profile": profile},
            blocks_heavy_viewer=True,
        )
    if mac_band in PROTECTIVE_FLUIDITY_BANDS:
        heavy_blocks.append("mac_fluidity_protective_band")
        _add_issue(
            issues,
            name="mac_fluidity_protective_band",
            severity="critical",
            source="runtime_throttle_control_latest.json",
            summary="Mac fluidity contract says to protect the foreground system.",
            detail={"fluidity_band": mac_band},
            blocks_heavy_viewer=True,
        )
    if _as_bool(release.get("live_lane_should_be_read_only"), False):
        _add_issue(
            issues,
            name="runtime_release_live_read_only",
            severity="critical",
            source="runtime_throttle_control_latest.json",
            summary="Runtime release contract keeps live lane read-only.",
            detail={
                "effective_live_read_only_reason": release.get("effective_live_read_only_reason"),
                "paper_trade_lock_active": _as_bool(release.get("paper_trade_lock_active"), False),
            },
            blocks_halt_clear=False,
        )
    if storage_mode in {"protect", "protect_live", "critical"}:
        heavy_blocks.append("storage_protective_mode")
        _add_issue(
            issues,
            name="storage_protective_mode",
            severity="critical",
            source="runtime_throttle_control_latest.json",
            summary="Storage stabilization recommends a protective operating mode.",
            detail={"recommended_operating_mode": storage_mode, "backlog_drain_status": storage_backlog},
            blocks_heavy_viewer=True,
        )

    return {
        "overall_status": overall_status,
        "throttle_profile": profile,
        "mac_fluidity_band": mac_band,
        "storage_recommended_operating_mode": storage_mode,
        "storage_backlog_drain_status": storage_backlog,
        "heavy_viewer_wait_reasons": heavy_blocks,
    }


def _evaluate_live_runtime(payload: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    clearance = payload.get("clearance_plan") if isinstance(payload.get("clearance_plan"), dict) else {}
    release = payload.get("release_contract") if isinstance(payload.get("release_contract"), dict) else {}
    live_plane = payload.get("live_plane") if isinstance(payload.get("live_plane"), dict) else {}
    clearance_state = _lower(clearance.get("clearance_state"))
    if clearance_state and clearance_state not in THAW_SAFE_RUNTIME_STATES:
        _add_issue(
            issues,
            name="runtime_clearance_not_thaw_safe",
            severity="critical",
            source="live_runtime_separation_control_latest.json",
            summary="Live runtime clearance is not in a thaw-safe state.",
            detail={"clearance_state": clearance_state},
            blocks_heavy_viewer=True,
        )
    if _as_bool(release.get("live_lane_should_be_read_only"), False):
        _add_issue(
            issues,
            name="live_runtime_release_read_only",
            severity="critical",
            source="live_runtime_separation_control_latest.json",
            summary="Live runtime separation keeps live lane read-only.",
            detail={"release_contract": release},
            blocks_halt_clear=False,
        )
    if _as_bool(release.get("heavy_research_must_stay_cold_lane"), False):
        _add_issue(
            issues,
            name="heavy_research_must_stay_cold_lane",
            severity="warning",
            source="live_runtime_separation_control_latest.json",
            summary="Heavy research must stay on the cold lane.",
            detail={"release_contract": release},
            blocks_live_execution=False,
            blocks_halt_clear=False,
            blocks_heavy_viewer=True,
        )
    return {
        "clearance_state": clearance_state,
        "live_lane_running": _as_bool(live_plane.get("live_lane_running"), False),
        "live_ready": _as_bool(live_plane.get("ready"), False),
    }


def _evaluate_incident(payload: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
    failed = payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else []
    suppressed = _as_bool(detail.get("enforcement_suppressed"), False)
    if _as_bool(payload.get("halt"), False):
        _add_issue(
            issues,
            name="incident_auto_halt_active",
            severity="critical",
            source="incident_auto_halt_latest.json",
            summary="Incident auto-halt reports an active halt.",
            detail={"event": payload.get("event"), "failed_checks": failed},
        )
    elif failed and not suppressed:
        _add_issue(
            issues,
            name="incident_auto_halt_failed_checks",
            severity="critical",
            source="incident_auto_halt_latest.json",
            summary="Incident auto-halt has failed checks that are not suppressed.",
            detail={"failed_checks": failed},
        )
    elif failed and suppressed:
        _add_issue(
            issues,
            name="incident_auto_halt_suppressed_failures",
            severity="warning",
            source="incident_auto_halt_latest.json",
            summary="Incident checks are failing but enforcement is suppressed because live execution is not expected.",
            detail={"failed_checks": failed},
            blocks_live_execution=False,
            blocks_halt_clear=False,
        )
    return {"failed_checks": failed, "enforcement_suppressed": suppressed}


def _evaluate_paper(payload: dict[str, Any], issues: list[dict[str, Any]], source: str) -> None:
    if _as_bool(payload.get("pause_paper_execution"), False) or _as_bool(payload.get("blocked"), False):
        _add_issue(
            issues,
            name=f"{source}_paper_execution_paused",
            severity="warning",
            source=f"{source}.json",
            summary="Paper execution is paused or blocked by its control artifact.",
            detail={"reason": payload.get("reason"), "blockers": payload.get("blockers")},
            blocks_live_execution=False,
            blocks_halt_clear=False,
        )


def _state_from_issues(issues: list[dict[str, Any]]) -> str:
    names = {issue["name"] for issue in issues}
    if "manual_operator_stop_active" in names:
        return "operator_stop"
    if "manual_global_halt_active" in names or "global_killswitch_active" in names:
        return "global_halt_active"
    if "health_hard_gates_active" in names or "global_killswitch_reasons_present" in names:
        return "hard_gate_blocked"
    if any(name.startswith("critical_artifact_") for name in names):
        return "safety_artifact_uncertain"
    if "paper_trade_lock_active" in names or "runtime_release_live_read_only" in names or "live_runtime_release_read_only" in names:
        return "live_read_only"
    if any(issue["severity"] == "warning" for issue in issues):
        return "degraded"
    return "clear"


def _commands(flags: dict[str, Any], issues: list[dict[str, Any]], safe_to_attempt_auto_clear: bool) -> dict[str, Any]:
    recommended: list[list[str]] = [["./scripts/ops/opsctl.sh", "halt-trigger-status", "--json"]]
    if flags["operator_stop"]["active"]:
        recommended.append(["./scripts/ops/opsctl.sh", "operator-release", "--json"])
    if any(issue["name"].startswith("critical_artifact_") for issue in issues):
        recommended.append(["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"])
    if any(issue["name"] == "health_hard_gates_active" for issue in issues):
        recommended.append(["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"])
        recommended.append(["./scripts/ops/opsctl.sh", "collector-contracts", "--json"])
    if safe_to_attempt_auto_clear:
        recommended.append(["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"])
    elif flags["global_halt"]["active"]:
        recommended.append(["./scripts/ops/opsctl.sh", "global-halt-status", "--json"])

    deduped: list[list[str]] = []
    for command in recommended:
        if command not in deduped:
            deduped.append(command)

    return {
        "status": ["./scripts/ops/opsctl.sh", "halt-trigger-status", "--json"],
        "global_halt_status": ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
        "refresh_global_halt_blockers": ["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"],
        "safe_global_halt_auto_clear": ["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"],
        "operator_release": ["./scripts/ops/opsctl.sh", "operator-release", "--json"],
        "manual_clear_all_halts": ["./scripts/ops/opsctl.sh", "clear-all-halts", "--json"],
        "recommended": deduped,
    }


def evaluate() -> dict[str, Any]:
    artifacts = _artifact_set()
    flags = _manual_flags()
    issues: list[dict[str, Any]] = []
    _evaluate_manual_flags(flags, issues)
    _evaluate_artifact_health(artifacts, issues)

    global_killswitch = artifacts["global_killswitch"]["payload"]
    health_gates = artifacts["health_gates"]["payload"]
    runtime_throttle = artifacts["runtime_throttle"]["payload"]
    live_runtime = artifacts["live_runtime_separation"]["payload"]
    incident = artifacts["incident_auto_halt"]["payload"]
    paper_truth = artifacts["paper_execution_truth"]["payload"]
    paper_ramp = artifacts["paper_400_ramp"]["payload"]

    if global_killswitch:
        _evaluate_global_killswitch(global_killswitch, issues)
    if health_gates:
        _evaluate_health_gates(health_gates, issues)
    runtime_summary = _evaluate_runtime_throttle(runtime_throttle, issues) if runtime_throttle else {}
    live_runtime_summary = _evaluate_live_runtime(live_runtime, issues) if live_runtime else {}
    incident_summary = _evaluate_incident(incident, issues) if incident else {}
    if paper_truth:
        _evaluate_paper(paper_truth, issues, "paper_execution_truth")
    if paper_ramp:
        _evaluate_paper(paper_ramp, issues, "paper_400_ramp")

    halt_clear_blockers = [issue for issue in issues if bool(issue.get("blocks_halt_clear", False))]
    live_execution_blockers = [issue for issue in issues if bool(issue.get("blocks_live_execution", False))]
    heavy_viewer_blockers = [issue for issue in issues if bool(issue.get("blocks_heavy_viewer", False))]

    global_halt_only_clear_blockers = [
        issue
        for issue in halt_clear_blockers
        if issue["name"] not in {"manual_global_halt_active", "global_killswitch_active"}
    ]
    killswitch_clear_ready = _as_bool(global_killswitch.get("clear_ready"), False) if global_killswitch else False
    safe_to_attempt_auto_clear = bool(
        flags["global_halt"]["active"]
        and flags["global_halt"]["valid"]
        and killswitch_clear_ready
        and not global_halt_only_clear_blockers
    )
    control_plane_allows_live_orders = not live_execution_blockers
    env_execution_expected = _execution_expected()
    effective_live_order_execution_allowed = bool(control_plane_allows_live_orders and env_execution_expected)
    heavy_livefeed_allowed = not heavy_viewer_blockers

    payload = {
        "timestamp_utc": _now_iso(),
        "schema_version": 1,
        "overall_status": "ready" if not issues else "blocked" if live_execution_blockers else "degraded",
        "effective_state": _state_from_issues(issues),
        "manual_flags": flags,
        "artifacts": {
            name: {key: value for key, value in artifact.items() if key != "payload"}
            for name, artifact in artifacts.items()
        },
        "issues": issues,
        "blockers": {
            "halt_clear": [issue["name"] for issue in halt_clear_blockers],
            "live_execution": [issue["name"] for issue in live_execution_blockers],
            "heavy_viewer": [issue["name"] for issue in heavy_viewer_blockers],
        },
        "execution_policy": {
            "control_plane_allows_live_orders": control_plane_allows_live_orders,
            "environment_expects_live_orders": env_execution_expected,
            "effective_live_order_execution_allowed": effective_live_order_execution_allowed,
            "safe_to_attempt_global_halt_auto_clear": safe_to_attempt_auto_clear,
            "paper_trade_lock_active": bool(flags["paper_trade_lock"]["active"]),
            "operator_stop_active": bool(flags["operator_stop"]["active"]),
            "global_halt_active": bool(flags["global_halt"]["active"]),
        },
        "viewer_policy": {
            "light_livefeed_allowed": True,
            "heavy_livefeed_allowed": heavy_livefeed_allowed,
            "heavy_livefeed_wait_reasons": [issue["name"] for issue in heavy_viewer_blockers],
            "policy": "read_only_viewers_may_run_during_halts_but_heavy_viewer_waits_for_runtime_or_storage_protective_safeguards",
        },
        "signals": {
            "global_killswitch": {
                "halt": _as_bool(global_killswitch.get("halt"), False) if global_killswitch else False,
                "halt_state": str(global_killswitch.get("halt_state") or "") if global_killswitch else "",
                "clear_ready": killswitch_clear_ready,
                "clear_blockers": global_killswitch.get("clear_blockers", []) if isinstance(global_killswitch.get("clear_blockers"), list) else [],
                "reasons": global_killswitch.get("reasons", []) if isinstance(global_killswitch.get("reasons"), list) else [],
            },
            "health_gates": {
                "hard_gate_triggered": _as_bool(health_gates.get("hard_gate_triggered"), False) if health_gates else False,
                "active_hard_gates": _active_names(health_gates.get("hard_gates", {})) if isinstance(health_gates.get("hard_gates"), dict) else [],
                "recommended_operating_mode": str(health_gates.get("recommended_operating_mode") or "") if health_gates else "",
            },
            "runtime_throttle": runtime_summary,
            "live_runtime_separation": live_runtime_summary,
            "incident_auto_halt": incident_summary,
        },
    }
    payload["control_commands"] = _commands(flags, issues, safe_to_attempt_auto_clear)
    return payload


def _write_outputs(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not safe_write_json_atomic(
        str(_latest_path()),
        payload,
        project_root=str(PROJECT_ROOT),
        source="halt_trigger_control_plane",
    ):
        errors.append(f"write_latest_failed:{_latest_path()}")
    try:
        _events_path().parent.mkdir(parents=True, exist_ok=True)
        with _events_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as exc:
        errors.append(f"append_events_failed:{type(exc).__name__}:{exc}")
    return errors


def _print_human(payload: dict[str, Any]) -> None:
    print(f"halt_trigger_state={payload['effective_state']}")
    print(f"overall_status={payload['overall_status']}")
    print(f"live_order_execution_allowed={int(payload['execution_policy']['effective_live_order_execution_allowed'])}")
    print(f"heavy_livefeed_allowed={int(payload['viewer_policy']['heavy_livefeed_allowed'])}")
    blockers = payload.get("blockers", {})
    print("halt_clear_blockers=" + ",".join(blockers.get("halt_clear") or []))
    print("live_execution_blockers=" + ",".join(blockers.get("live_execution") or []))
    print(f"latest={_latest_path()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize kill switches, halt triggers, and live execution blockers.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--no-write", action="store_true", help="Do not write latest/event artifacts.")
    parser.add_argument("--assert-clear", action="store_true", help="Return non-zero when halt/live-execution blockers exist.")
    parser.add_argument("--exit-zero", action="store_true", help="Always return zero after emitting the snapshot.")
    args = parser.parse_args()

    payload = evaluate()
    if not args.no_write:
        errors = _write_outputs(payload)
        if errors:
            payload["io_errors"] = errors
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        _print_human(payload)

    if args.exit_zero:
        return 0
    if args.assert_clear and (payload["blockers"]["halt_clear"] or payload["blockers"]["live_execution"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
