#!/usr/bin/env python3
from __future__ import annotations

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


BAD_ARTIFACT_STATES = {"missing", "invalid_json", "stale", "payload_not_object"}
PROTECTIVE_FLUIDITY_BANDS = {"protect", "protect_live", "critical", "halt"}
PROTECTIVE_THROTTLE_PROFILES = {"protect", "protect_live", "critical", "halt", "panic"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _health_root() -> Path:
    return PROJECT_ROOT / "governance" / "health"


def _alerts_root() -> Path:
    return PROJECT_ROOT / "governance" / "alerts"


def _watchdog_root() -> Path:
    return PROJECT_ROOT / "governance" / "watchdog"


def _latest_path() -> Path:
    return _health_root() / "coordination_state_latest.json"


def _events_path() -> Path:
    return _watchdog_root() / "coordination_state_events.jsonl"


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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
        return {}, "payload_not_object"
    return payload, ""


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


def _artifact_set() -> dict[str, dict[str, Any]]:
    health = _health_root()
    alerts = _alerts_root()
    return {
        "halt_trigger_control_plane": _artifact(
            "halt_trigger_control_plane",
            health / "halt_trigger_control_plane_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_HALT_TRIGGER_MAX_AGE_SECONDS", "900")),
            required=True,
        ),
        "runtime_throttle": _artifact(
            "runtime_throttle",
            health / "runtime_throttle_control_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_RUNTIME_THROTTLE_MAX_AGE_SECONDS", "900")),
            required=True,
        ),
        "process_watchdog": _artifact(
            "process_watchdog",
            health / "process_watchdog_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_PROCESS_WATCHDOG_MAX_AGE_SECONDS", "900")),
            required=True,
        ),
        "shadow_watchdog_tripwire": _artifact(
            "shadow_watchdog_tripwire",
            health / "shadow_watchdog_tripwire_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_TRIPWIRE_MAX_AGE_SECONDS", "900")),
            required=False,
        ),
        "guardrail_triprate": _artifact(
            "guardrail_triprate",
            health / "guardrail_triprate_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_TRIPRATE_MAX_AGE_SECONDS", "21600")),
            required=False,
        ),
        "remote_alert_control": _artifact(
            "remote_alert_control",
            health / "remote_alert_control_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_REMOTE_ALERT_MAX_AGE_SECONDS", "3600")),
            required=False,
        ),
        "training_runtime_control": _artifact(
            "training_runtime_control",
            health / "training_runtime_control_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_TRAINING_RUNTIME_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
        "livefeed_local": _artifact(
            "livefeed_local",
            health / "livefeed_local_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_LIVEFEED_LOCAL_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
        "heavy_livefeed": _artifact(
            "heavy_livefeed",
            health / "live_feed_heavy_guarded_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_HEAVY_LIVEFEED_MAX_AGE_SECONDS", "10800")),
            required=False,
        ),
        "operator_control": _artifact(
            "operator_control",
            health / "operator_control_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_OPERATOR_CONTROL_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
        "incident_auto_halt": _artifact(
            "incident_auto_halt",
            alerts / "incident_auto_halt_latest.json",
            max_age_seconds=int(os.getenv("COORDINATION_INCIDENT_AUTO_HALT_MAX_AGE_SECONDS", "7200")),
            required=False,
        ),
    }


def _source_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        out[name] = {key: value for key, value in artifact.items() if key != "payload"}
    return out


def _artifact_issues(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for name, artifact in artifacts.items():
        state = str(artifact.get("state") or "")
        if state not in BAD_ARTIFACT_STATES:
            continue
        required = bool(artifact.get("required", False))
        issues.append(
            {
                "name": f"{'required' if required else 'optional'}_artifact_{state}:{name}",
                "severity": "critical" if required else "warning",
                "source": str(artifact.get("path") or name),
                "required": required,
                "age_seconds": artifact.get("age_seconds"),
                "max_age_seconds": artifact.get("max_age_seconds"),
                "error": artifact.get("error"),
            }
        )
    return issues


def _tripwire_guard_tags(note: str) -> list[str]:
    tags: list[str] = []
    for raw in note.split(","):
        token = raw.strip()
        if token.endswith("_guard_active") or token.endswith("_pause_guard_active"):
            tags.append(token)
    return tags


def _triage_tripwire(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
    incidents = payload.get("active_incidents") if isinstance(payload.get("active_incidents"), list) else []
    rows: list[dict[str, Any]] = []
    counts = {
        "actionable": 0,
        "suppressed_by_guard": 0,
        "expected_offline": 0,
        "stale": 0,
        "needs_operator": 0,
    }

    artifact_stale = str(artifact.get("state") or "") == "stale"
    for incident in incidents:
        if not isinstance(incident, dict):
            continue
        note = str(incident.get("note") or "")
        action = _lower(incident.get("action"))
        guard_tags = _tripwire_guard_tags(note)
        secondary: list[str] = []
        if guard_tags:
            secondary.append("guard_suppressed")
        if "operator_or_computer_task_guard_active" in note or "creative_audio_pause_guard_active" in note:
            secondary.append("expected_offline")
            counts["expected_offline"] += 1

        if artifact_stale:
            classification = "stale"
        elif action == "suppressed" and guard_tags:
            classification = "suppressed_by_guard"
        elif not _as_bool(incident.get("process_live"), True) and _as_bool(incident.get("heartbeat_lost"), False):
            classification = "needs_operator"
        elif action in {"halt", "restart", "notify", "page"}:
            classification = "actionable"
        else:
            classification = "actionable" if _as_bool(incident.get("required"), False) else "suppressed_by_guard"

        counts[classification] = counts.get(classification, 0) + 1
        rows.append(
            {
                "target": str(incident.get("target") or ""),
                "classification": classification,
                "secondary_tags": sorted(set(secondary)),
                "required": _as_bool(incident.get("required"), False),
                "process_live": _as_bool(incident.get("process_live"), False),
                "heartbeat_lost": _as_bool(incident.get("heartbeat_lost"), False),
                "consecutive_unhealthy_cycles": _as_int(incident.get("consecutive_unhealthy_cycles"), 0),
                "action": action,
                "guard_tags": guard_tags,
                "note": note,
            }
        )

    active = _as_bool(payload.get("active"), False)
    actionable = counts["actionable"] + counts["needs_operator"]
    return {
        "overall_status": "active_actionable" if actionable else "active_suppressed" if active else "clear",
        "enabled": _as_bool(payload.get("enabled"), False),
        "active": active,
        "artifact_state": str(artifact.get("state") or ""),
        "counts": counts,
        "incidents": rows,
    }


def _classify_crash_cause(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "target",
            "event",
            "message",
            "reason",
            "status",
            "note",
            "last_error",
            "summary",
            "category",
        )
    ).lower()
    if "memory" in text or "swap" in text or "pressure" in text:
        return "memory_pressure"
    if "auth" in text or "token" in text or "oauth" in text or "credential" in text:
        return "auth_or_token_failure"
    if "mount" in text or "volume" in text or "bot_logs" in text or "storage" in text:
        return "storage_mount_or_backpressure"
    if "network" in text or "dns" in text or "outage" in text or "timeout" in text:
        return "network_or_timeout"
    if "restart storm" in text or "budget" in text or "crashloop" in text:
        return "restart_storm_or_budget"
    if "terminal" in text or "tty" in text or "session" in text:
        return "terminal_session_loss"
    if "heartbeat" in text or "process_missing" in text or "missing" in text:
        return "heartbeat_or_missing_process"
    return "unknown_restart"


def _crashloop_classifier(process_watchdog: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in ("restarts", "restart_storms", "recent_restart_storms"):
        value = process_watchdog.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                row = dict(item)
            else:
                row = {"message": str(item)}
            row["source_collection"] = key
            row["classified_cause"] = _classify_crash_cause(row)
            rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        cause = str(row.get("classified_cause") or "unknown_restart")
        counts[cause] = counts.get(cause, 0) + 1

    return {
        "overall_status": "clear" if not rows else "restart_pressure_present",
        "event_count": len(rows),
        "cause_counts": counts,
        "events": rows[:25],
    }


def _training_policy(training: dict[str, Any]) -> dict[str, Any]:
    contract = training.get("training_launch_contract") if isinstance(training.get("training_launch_contract"), dict) else {}
    launch_blockers = contract.get("launch_blockers") if isinstance(contract.get("launch_blockers"), list) else []
    prep_blockers = contract.get("prep_blockers") if isinstance(contract.get("prep_blockers"), list) else []
    prep_allowed = _as_bool(contract.get("prep_allowed"), bool(contract) and not prep_blockers)
    launch_allowed = _as_bool(contract.get("launch_allowed"), False)
    return {
        "prep_allowed": prep_allowed,
        "launch_allowed": launch_allowed,
        "mode": str(contract.get("mode") or training.get("operating_mode") or ""),
        "recommended_batch_size": _as_int(contract.get("recommended_batch_size"), 0),
        "launch_blockers": [str(item) for item in launch_blockers],
        "prep_blockers": [str(item) for item in prep_blockers],
        "recommended_prep_commands": contract.get("recommended_prep_commands", [])
        if isinstance(contract.get("recommended_prep_commands"), list)
        else [],
        "recommended_retrain_command": contract.get("recommended_retrain_command", [])
        if isinstance(contract.get("recommended_retrain_command"), list)
        else [],
    }


def _operator_intent(runtime_throttle: dict[str, Any], operator_control: dict[str, Any]) -> dict[str, Any]:
    mac = runtime_throttle.get("mac_fluidity_contract") if isinstance(runtime_throttle.get("mac_fluidity_contract"), dict) else {}
    fluidity_band = _lower(mac.get("fluidity_band"))
    foreground_first = _as_bool(mac.get("foreground_first"), True)
    foreground_active = _as_bool(mac.get("foreground_active"), False)
    operator_stop = _as_bool(operator_control.get("operator_stop"), False)
    return {
        "mode": "operator_stop" if operator_stop else "foreground_first" if foreground_first else "balanced",
        "foreground_first": foreground_first,
        "foreground_active": foreground_active,
        "fluidity_band": fluidity_band,
        "protected": bool(operator_stop or foreground_first or fluidity_band in PROTECTIVE_FLUIDITY_BANDS),
        "policy": "operator_ui_email_browser_remains_above_heavy_viewers_training_and_support_jobs",
    }


def _build_policy(
    artifacts: dict[str, dict[str, Any]],
    tripwire: dict[str, Any],
    crashloops: dict[str, Any],
    training: dict[str, Any],
    operator: dict[str, Any],
) -> dict[str, Any]:
    halt = artifacts["halt_trigger_control_plane"]["payload"]
    runtime = artifacts["runtime_throttle"]["payload"]
    process = artifacts["process_watchdog"]["payload"]
    heavy = artifacts["heavy_livefeed"]["payload"]
    livefeed = artifacts["livefeed_local"]["payload"]

    halt_execution = halt.get("execution_policy") if isinstance(halt.get("execution_policy"), dict) else {}
    viewer = halt.get("viewer_policy") if isinstance(halt.get("viewer_policy"), dict) else {}
    halt_blockers = halt.get("blockers") if isinstance(halt.get("blockers"), dict) else {}
    manual_flags = halt.get("manual_flags") if isinstance(halt.get("manual_flags"), dict) else {}
    paper_lock = manual_flags.get("paper_trade_lock") if isinstance(manual_flags.get("paper_trade_lock"), dict) else {}
    paper_payload = paper_lock.get("payload") if isinstance(paper_lock.get("payload"), dict) else {}

    mac = runtime.get("mac_fluidity_contract") if isinstance(runtime.get("mac_fluidity_contract"), dict) else {}
    throttle_profile = _lower(runtime.get("throttle_profile"))
    fluidity_band = _lower(mac.get("fluidity_band"))
    protective_runtime = throttle_profile in PROTECTIVE_THROTTLE_PROFILES or fluidity_band in PROTECTIVE_FLUIDITY_BANDS

    process_intel = process.get("watchdog_intelligence") if isinstance(process.get("watchdog_intelligence"), dict) else {}
    process_ready = _lower(process.get("overall_status")) in {"ready", "ok", "healthy"} and _as_int(process_intel.get("active_issue_count"), 0) == 0
    tripwire_actionable = _as_int(tripwire["counts"].get("actionable"), 0) + _as_int(tripwire["counts"].get("needs_operator"), 0)

    heavy_allowed_by_halt = _as_bool(viewer.get("heavy_livefeed_allowed"), True)
    heavy_guard_allowed = _as_bool(heavy.get("allowed"), True)
    heavy_allowed = bool(heavy_allowed_by_halt and heavy_guard_allowed and not protective_runtime)

    paper_allowed = _as_bool(paper_payload.get("paper_execution_allowed"), True)
    if _as_bool(halt_execution.get("operator_stop_active"), False):
        paper_allowed = False

    livefeed_alive = _as_bool(livefeed.get("alive"), False) or _lower(livefeed.get("status")) == "running"
    terminal_restart_safe = bool(
        process_ready
        and livefeed_alive
        and crashloops.get("overall_status") == "clear"
        and tripwire_actionable == 0
    )

    live_blockers = [str(item) for item in halt_blockers.get("live_execution", [])] if isinstance(halt_blockers.get("live_execution"), list) else []
    heavy_blockers = [str(item) for item in viewer.get("heavy_livefeed_wait_reasons", [])] if isinstance(viewer.get("heavy_livefeed_wait_reasons"), list) else []
    if protective_runtime:
        heavy_blockers.append("foreground_or_runtime_protective_band")
    if not heavy_guard_allowed:
        heavy_blockers.append(str(heavy.get("reason") or "heavy_livefeed_guard_not_allowed"))

    return {
        "live_orders": {
            "allowed": _as_bool(halt_execution.get("effective_live_order_execution_allowed"), False),
            "control_plane_allows": _as_bool(halt_execution.get("control_plane_allows_live_orders"), False),
            "environment_expects_live_orders": _as_bool(halt_execution.get("environment_expects_live_orders"), False),
            "blockers": live_blockers,
        },
        "paper_execution": {
            "allowed": paper_allowed,
            "paper_trade_lock_active": _as_bool(halt_execution.get("paper_trade_lock_active"), bool(paper_lock.get("active"))),
            "reason": str(paper_lock.get("reason") or paper_payload.get("policy") or ""),
        },
        "light_livefeed": {
            "allowed": True,
            "alive": livefeed_alive,
            "contract": str(livefeed.get("contract") or ""),
            "reattach_command": ["./scripts/ops/opsctl.sh", "livefeed-refresh", "--mirror-only"],
        },
        "heavy_viewer": {
            "allowed": heavy_allowed,
            "guard_allowed": heavy_guard_allowed,
            "running": _as_int(heavy.get("heavy_pid"), 0) > 0,
            "blockers": sorted(set(item for item in heavy_blockers if item)),
            "policy": "heavy_viewer_runs_when_read_only_and_runtime_fluidity_is_not_protective",
        },
        "training_prep": {
            "allowed": bool(training.get("prep_allowed")),
            "blockers": training.get("prep_blockers", []),
        },
        "training_launch": {
            "allowed": bool(training.get("launch_allowed")),
            "mode": str(training.get("mode") or ""),
            "recommended_batch_size": _as_int(training.get("recommended_batch_size"), 0),
            "blockers": training.get("launch_blockers", []),
        },
        "terminal_restart": {
            "safe": terminal_restart_safe,
            "reattach_required": True,
            "blockers": []
            if terminal_restart_safe
            else [
                reason
                for reason, active in (
                    ("process_watchdog_not_clear", not process_ready),
                    ("livefeed_local_not_alive", not livefeed_alive),
                    ("restart_pressure_present", crashloops.get("overall_status") != "clear"),
                    ("actionable_tripwire_present", tripwire_actionable > 0),
                )
                if active
            ],
            "reattach_commands": [
                ["./scripts/ops/opsctl.sh", "livefeed-refresh", "--mirror-only"],
                ["./scripts/ops/opsctl.sh", "coordination-status", "--json"],
            ],
        },
        "operator_foreground": {
            "protected": bool(operator.get("protected")),
            "mode": str(operator.get("mode") or ""),
            "fluidity_band": str(operator.get("fluidity_band") or ""),
        },
    }


def _priority_arbiter(policies: dict[str, Any], operator: dict[str, Any]) -> dict[str, Any]:
    live_orders_allowed = _as_bool(policies["live_orders"].get("allowed"), False)
    specs = [
        ("operator_ui_email", 100, True, "protected" if bool(operator.get("protected")) else "available"),
        ("safety_watchdogs", 95, True, "required"),
        ("livefeed_light", 85, _as_bool(policies["light_livefeed"].get("allowed"), True), "allowed"),
        ("market_data_ingestion", 80, True, "allowed"),
        ("livefeed_heavy", 70, _as_bool(policies["heavy_viewer"].get("allowed"), False), "allowed"),
        ("shadow_loops", 60, True, "read_only" if not live_orders_allowed else "live_support"),
        ("training_prep", 50, _as_bool(policies["training_prep"].get("allowed"), False), "allowed"),
        ("training_launch", 30, _as_bool(policies["training_launch"].get("allowed"), False), "allowed"),
    ]
    lanes = [
        {
            "lane": lane,
            "priority": priority,
            "state": ready_state if ready else "deferred",
            "allowed": bool(ready),
        }
        for lane, priority, ready, ready_state in specs
    ]
    return {
        "policy": "operator_ui_then_safety_then_visibility_then_ingestion_then_heavy_viewers_then_training",
        "lanes": lanes,
        "deferred_lanes": [lane["lane"] for lane in lanes if not lane["allowed"]],
    }


def _process_lease_supervisor(process_watchdog: dict[str, Any], policies: dict[str, Any]) -> dict[str, Any]:
    intel = process_watchdog.get("watchdog_intelligence") if isinstance(process_watchdog.get("watchdog_intelligence"), dict) else {}
    target_count = _as_int(intel.get("target_count"), 0)
    healthy_count = _as_int(intel.get("healthy_target_count"), 0)
    missing = intel.get("missing_targets") if isinstance(intel.get("missing_targets"), list) else []
    stale = intel.get("stale_targets") if isinstance(intel.get("stale_targets"), list) else []
    return {
        "overall_status": "ready" if target_count == healthy_count and not missing and not stale else "attention",
        "target_count": target_count,
        "healthy_target_count": healthy_count,
        "missing_targets": missing,
        "stale_targets": stale,
        "terminal_restart_safe": _as_bool(policies["terminal_restart"].get("safe"), False),
        "lease_model": "watchdog_heartbeat_plus_launchd_reattach",
        "reattach_commands": policies["terminal_restart"].get("reattach_commands", []),
    }


def _coordination_timeline(
    artifacts: dict[str, dict[str, Any]],
    policies: dict[str, Any],
    tripwire: dict[str, Any],
    crashloops: dict[str, Any],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for name, artifact in artifacts.items():
        if artifact.get("timestamp_utc"):
            events.append(
                {
                    "timestamp_utc": artifact.get("timestamp_utc"),
                    "source": name,
                    "state": artifact.get("state"),
                }
            )
    events.append(
        {
            "timestamp_utc": _now_iso(),
            "source": "coordination_state",
            "state": "sampled",
            "live_orders_allowed": policies["live_orders"]["allowed"],
            "heavy_viewer_allowed": policies["heavy_viewer"]["allowed"],
            "training_launch_allowed": policies["training_launch"]["allowed"],
            "tripwire_status": tripwire.get("overall_status"),
            "crashloop_status": crashloops.get("overall_status"),
        }
    )
    events.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    return {
        "event_stream_path": str(_events_path()),
        "latest_events": events[-20:],
    }


def _overall_status(
    artifact_issues: list[dict[str, Any]],
    policies: dict[str, Any],
    tripwire: dict[str, Any],
    crashloops: dict[str, Any],
) -> str:
    if any(issue.get("required") for issue in artifact_issues):
        return "blocked"
    if _as_int(tripwire["counts"].get("actionable"), 0) or _as_int(tripwire["counts"].get("needs_operator"), 0):
        return "blocked"
    if crashloops.get("overall_status") != "clear":
        return "degraded"
    if not policies["live_orders"]["allowed"] or not policies["training_launch"]["allowed"]:
        return "guarded"
    if not policies["heavy_viewer"]["allowed"]:
        return "degraded"
    return "ready"


def _recommended_commands(policies: dict[str, Any], training: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = [["./scripts/ops/opsctl.sh", "coordination-status", "--json"]]
    if not policies["terminal_restart"]["safe"]:
        commands.append(["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"])
    if not policies["light_livefeed"]["alive"]:
        commands.append(["./scripts/ops/opsctl.sh", "livefeed-refresh", "--mirror-only"])
    if not policies["training_launch"]["allowed"]:
        for command in training.get("recommended_prep_commands") or []:
            if isinstance(command, list) and command:
                commands.append([str(part) for part in command])
    deduped: list[list[str]] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped


def evaluate() -> dict[str, Any]:
    artifacts = _artifact_set()
    artifact_issues = _artifact_issues(artifacts)
    runtime = artifacts["runtime_throttle"]["payload"]
    operator_control = artifacts["operator_control"]["payload"]
    process_watchdog = artifacts["process_watchdog"]["payload"]

    tripwire = _triage_tripwire(artifacts["shadow_watchdog_tripwire"])
    crashloops = _crashloop_classifier(process_watchdog)
    training = _training_policy(artifacts["training_runtime_control"]["payload"])
    operator = _operator_intent(runtime, operator_control)
    policies = _build_policy(artifacts, tripwire, crashloops, training, operator)
    priority = _priority_arbiter(policies, operator)
    lease_supervisor = _process_lease_supervisor(process_watchdog, policies)
    timeline = _coordination_timeline(artifacts, policies, tripwire, crashloops)
    overall = _overall_status(artifact_issues, policies, tripwire, crashloops)

    payload = {
        "timestamp_utc": _now_iso(),
        "schema_version": 1,
        "overall_status": overall,
        "coordination_mode": operator["mode"],
        "source_artifacts": _source_summary(artifacts),
        "artifact_issues": artifact_issues,
        "policies": policies,
        "priority_arbiter": priority,
        "operator_intent": operator,
        "process_lease_supervisor": lease_supervisor,
        "tripwire_triage": tripwire,
        "crashloop_cause_classifier": crashloops,
        "coordination_timeline": timeline,
        "recommended_commands": _recommended_commands(policies, training),
    }
    return payload


def _write_outputs(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not safe_write_json_atomic(
        str(_latest_path()),
        payload,
        project_root=str(PROJECT_ROOT),
        source="coordination_state_control",
    ):
        errors.append(f"write_latest_failed:{_latest_path()}")
    try:
        _events_path().parent.mkdir(parents=True, exist_ok=True)
        event_row = {
            "timestamp_utc": payload.get("timestamp_utc"),
            "overall_status": payload.get("overall_status"),
            "coordination_mode": payload.get("coordination_mode"),
            "policies": payload.get("policies", {}),
            "tripwire_triage": payload.get("tripwire_triage", {}),
            "crashloop_cause_classifier": payload.get("crashloop_cause_classifier", {}),
        }
        with _events_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(event_row, ensure_ascii=True) + "\n")
    except Exception as exc:
        errors.append(f"append_events_failed:{type(exc).__name__}:{exc}")
    return errors


def _print_human(payload: dict[str, Any]) -> None:
    policies = payload["policies"]
    print(f"coordination_status={payload['overall_status']}")
    print(f"coordination_mode={payload['coordination_mode']}")
    print(f"live_orders_allowed={int(policies['live_orders']['allowed'])}")
    print(f"paper_execution_allowed={int(policies['paper_execution']['allowed'])}")
    print(f"light_livefeed_alive={int(policies['light_livefeed']['alive'])}")
    print(f"heavy_viewer_allowed={int(policies['heavy_viewer']['allowed'])}")
    print(f"training_launch_allowed={int(policies['training_launch']['allowed'])}")
    print(f"terminal_restart_safe={int(policies['terminal_restart']['safe'])}")
    print("deferred_lanes=" + ",".join(payload["priority_arbiter"].get("deferred_lanes") or []))
    print(f"latest={_latest_path()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate runtime, livefeed, tripwire, restart, and training policies.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--no-write", action="store_true", help="Do not write latest/event artifacts.")
    parser.add_argument("--assert-ready", action="store_true", help="Return non-zero unless coordination is fully ready.")
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
    if args.assert_ready and payload.get("overall_status") != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
