#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "infrastructure_autofix_bot_latest.json"
PYTHON_BIN = Path(sys.executable)
COMMANDS_HYGIENE_SCRIPT = Path(__file__).resolve().with_name("commands_hygiene_bot.py")
COMMAND_VALIDITY_SCRIPT = Path(__file__).resolve().with_name("command_validity_bot.py")
OPTIONS_FLOW_EXPORT_HYGIENE_SCRIPT = Path(__file__).resolve().with_name("options_flow_export_hygiene_bot.py")
OPTIONS_FLOW_EFFICIENCY_SCRIPT = Path(__file__).resolve().with_name("options_flow_efficiency_bot.py")
STORAGE_PRESSURE_CLEARANCE_SCRIPT = Path(__file__).resolve().with_name("storage_pressure_clearance_bot.py")
STORAGE_RECONNECT_INFRABOT_SCRIPT = Path(__file__).resolve().with_name("storage_reconnect_infrabot.py")
CORE_BOT_MATERIALIZATION_INFRABOT_SCRIPT = Path(__file__).resolve().with_name("core_bot_materialization_infrabot.py")
CORE_BOT_TIER_ORGANIZER_SCRIPT = Path(__file__).resolve().with_name("organize_core_bot_tiers.py")
ONE_NUMBERS_REGRESSION_GUARD_SCRIPT = Path(__file__).resolve().with_name("one_numbers_regression_guard.py")
STATEFUL_STORAGE_REGRESSION_GUARD_SCRIPT = Path(__file__).resolve().with_name("stateful_storage_regression_guard.py")
MASTER_INFRA_SUPERVISOR_SCRIPT = Path(__file__).resolve().with_name("master_infrastructure_supervisor.py")


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


def _artifact_candidates(project_root: Path, raw_path: str | Path) -> list[Path]:
    path = Path(raw_path)
    if path.is_absolute():
        return [path]
    candidates = [project_root / path]
    parts = path.parts
    if parts and parts[0] in {"data", "decisions", "decision_explanations", "exports", "governance", "logs", "models"}:
        candidates.append(project_root / "local_fallback_storage" / path)
        candidates.append(Path("/Volumes/BOT_LOGS/schwab_trading_bot") / path)
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _load_freshest_json(project_root: Path, raw_path: str | Path) -> dict[str, Any]:
    best_payload: dict[str, Any] = {}
    best_mtime = -1.0
    for candidate in _artifact_candidates(project_root, raw_path):
        payload = load_json(candidate)
        if not payload:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except Exception:
            mtime = 0.0
        if mtime >= best_mtime:
            best_payload = payload
            best_mtime = mtime
    return best_payload


def _collector_contract_row(payload: dict[str, Any], name: str) -> dict[str, Any]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("name") or "").strip() == name:
            return row
    return {}


def _source_verification_row(payload: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("source_id") or "").strip() == source_id:
            return row
    return {}


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 1200,
) -> dict[str, Any]:
    daily_verify = _load_freshest_json(project_root, "governance/health/daily_auto_verify_latest.json")
    storage_control = _load_freshest_json(project_root, "governance/health/ingestion_storage_control_latest.json")
    collector_contracts = _load_freshest_json(project_root, "governance/health/collector_contracts_latest.json")
    source_verification = _load_freshest_json(project_root, "governance/health/source_verification_latest.json")
    options_flow_context = _load_freshest_json(project_root, "governance/health/options_flow_context_sync_latest.json")
    if not options_flow_context:
        options_flow_context = _load_freshest_json(project_root, "governance/health/tastytrade_context_sync_latest.json")
    schwab_education_context = _load_freshest_json(project_root, "governance/health/schwab_education_context_sync_latest.json")
    auth_lease = _load_freshest_json(project_root, "governance/health/auth_lease_manager_latest.json")
    schwab_auth_supervisor = _load_freshest_json(project_root, "governance/health/schwab_auth_supervisor_latest.json")
    blackstart = _load_freshest_json(project_root, "governance/health/blackstart_recovery_latest.json")
    freshness = _load_freshest_json(project_root, "governance/health/artifact_freshness_slo_latest.json")
    snapshot_cache = _load_freshest_json(project_root, "governance/health/runtime_snapshot_cache_control_latest.json")
    remote_alert = _load_freshest_json(project_root, "governance/health/remote_alert_control_latest.json")
    training_quality = _load_freshest_json(project_root, "governance/health/training_quality_control_latest.json")
    supportability = _load_freshest_json(project_root, "governance/health/supportability_control_latest.json")
    bot_quality = _load_freshest_json(project_root, "governance/health/bot_quality_autopilot_latest.json")
    system_drift = _load_freshest_json(project_root, "governance/health/system_drift_guard_latest.json")
    one_numbers_guard = _load_freshest_json(project_root, "governance/health/one_numbers_regression_guard_latest.json")
    stateful_storage_guard = _load_freshest_json(project_root, "governance/health/stateful_storage_regression_guard_latest.json")
    storage_reconnect = _load_freshest_json(project_root, "governance/health/storage_reconnect_infrabot_latest.json")
    storage_reconnect_guard = _load_freshest_json(project_root, "governance/health/storage_reconnect_regression_guard_latest.json")
    core_bot_materialization = _load_freshest_json(project_root, "governance/health/core_bot_materialization_infrabot_latest.json")
    core_bot_materialization_guard = _load_freshest_json(project_root, "governance/health/core_bot_materialization_guard_latest.json")
    core_bot_tier_organizer = _load_freshest_json(project_root, "governance/health/core_bot_tier_organizer_latest.json")
    master_infra = _load_freshest_json(project_root, "governance/health/master_infrastructure_supervisor_latest.json")

    failed_checks = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    repair_plan: list[dict[str, Any]] = []
    operator_followups: list[str] = []

    def add_plan(name: str, reason: str, cmd: list[str]) -> None:
        repair_plan.append({"name": name, "reason": reason, "cmd": cmd})

    if failed_checks:
        add_plan(
            "daily_verify_auto_remediation",
            f"failed_checks={len(failed_checks)}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "daily_verify_auto_remediation_bot.py"), "--apply", "--json"],
        )

    storage_status = str(storage_control.get("overall_status") or "")
    storage_backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    retention_debt_gb = _safe_float(((storage_control.get("storage") or {}).get("retention_debt_gb")), 0.0)
    total_pending_lines = _safe_int(storage_backpressure.get("total_pending_lines"), 0)
    total_drain_minutes = _safe_float(storage_backpressure.get("estimated_total_drain_minutes"), 0.0)
    if storage_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "storage_pressure_clearance",
            f"ingestion_storage_status={storage_status} retention_debt_gb={retention_debt_gb:.3f} pending_lines={total_pending_lines} total_drain_minutes={total_drain_minutes:.3f}",
            [str(PYTHON_BIN), str(STORAGE_PRESSURE_CLEARANCE_SCRIPT), "--apply", "--force-clear-stale-gate", "--json"],
        )

    stateful_storage_status = str(stateful_storage_guard.get("overall_status") or "")
    if not stateful_storage_guard or stateful_storage_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "stateful_storage_regression_guard",
            f"stateful_storage_status={stateful_storage_status or 'missing'}",
            [str(PYTHON_BIN), str(STATEFUL_STORAGE_REGRESSION_GUARD_SCRIPT), "--apply", "--json"],
        )

    storage_reconnect_status = str(storage_reconnect.get("overall_status") or "")
    storage_reconnect_guard_status = str(storage_reconnect_guard.get("overall_status") or "")
    if (
        not storage_reconnect
        or storage_reconnect_status in {"blocked", "degraded", "needs_work"}
        or storage_reconnect_guard_status in {"blocked", "degraded", "needs_work"}
    ):
        add_plan(
            "storage_reconnect_infrabot",
            f"reconnect_status={storage_reconnect_status or 'missing'} guard_status={storage_reconnect_guard_status or 'missing'}",
            [str(PYTHON_BIN), str(STORAGE_RECONNECT_INFRABOT_SCRIPT), "--apply", "--json"],
        )

    core_materialization_status = str(core_bot_materialization.get("overall_status") or "")
    core_materialization_guard_status = str(core_bot_materialization_guard.get("overall_status") or "")
    if (
        not core_bot_materialization
        or core_materialization_status in {"blocked", "degraded", "needs_work"}
        or core_materialization_guard_status in {"blocked", "degraded", "needs_work"}
    ):
        add_plan(
            "core_bot_materialization_infrabot",
            f"materialization_status={core_materialization_status or 'missing'} guard_status={core_materialization_guard_status or 'missing'}",
            [str(PYTHON_BIN), str(CORE_BOT_MATERIALIZATION_INFRABOT_SCRIPT), "--apply", "--json"],
        )
    core_tier_status = str(core_bot_tier_organizer.get("overall_status") or "")
    if not core_bot_tier_organizer or core_tier_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "core_bot_tier_organizer",
            f"tier_status={core_tier_status or 'missing'}",
            [str(PYTHON_BIN), str(CORE_BOT_TIER_ORGANIZER_SCRIPT), "--json"],
        )

    one_numbers_status = str(one_numbers_guard.get("overall_status") or "")
    one_numbers_weaknesses = one_numbers_guard.get("weaknesses") if isinstance(one_numbers_guard.get("weaknesses"), list) else []
    if not one_numbers_guard or one_numbers_status in {"blocked", "degraded"} or one_numbers_weaknesses:
        add_plan(
            "one_numbers_regression_guard",
            f"one_numbers_status={one_numbers_status or 'missing'} weaknesses={len(one_numbers_weaknesses)}",
            [str(PYTHON_BIN), str(ONE_NUMBERS_REGRESSION_GUARD_SCRIPT), "--apply", "--json"],
        )

    schwab_contract = _collector_contract_row(collector_contracts, "schwab_education_context")
    if (
        (schwab_contract and not bool(schwab_contract.get("contract_ok", False)))
        or (schwab_education_context and not bool(schwab_education_context.get("ok", False)))
    ):
        add_plan(
            "schwab_education_refresh",
            "schwab_education_contract_not_ok",
            [str(PYTHON_BIN), str(project_root / "scripts" / "collect_schwab_education_context.py"), "--json"],
        )

    options_flow_row = _source_verification_row(source_verification, "polygon_unusual_whales_options_context")
    options_flow_sources = options_flow_context.get("sources") if isinstance(options_flow_context.get("sources"), dict) else {}
    options_flow_export = (
        options_flow_sources.get("unusual_whales_export")
        if isinstance(options_flow_sources.get("unusual_whales_export"), dict)
        else {}
    )
    if bool(options_flow_export.get("configured", False)) and (
        not bool(options_flow_export.get("ok", False))
        or bool(options_flow_export.get("errors"))
        or int(options_flow_export.get("rejected_row_count", 0) or 0) > 0
    ):
        add_plan(
            "options_flow_export_hygiene",
            "options_flow_export_handoff_needs_attention",
            [str(PYTHON_BIN), str(OPTIONS_FLOW_EXPORT_HYGIENE_SCRIPT), "--apply", "--json"],
        )
    if (
        (options_flow_row and str(options_flow_row.get("verification_status") or "") == "single_source_unverified")
        or (options_flow_context and str(options_flow_context.get("overall_status") or "") in {"blocked", "degraded"})
        or (options_flow_context and not bool(options_flow_context.get("ok", False)))
    ):
        add_plan(
            "options_flow_efficiency",
            "options_flow_context_not_ready",
            [str(PYTHON_BIN), str(OPTIONS_FLOW_EFFICIENCY_SCRIPT), "--apply", "--json"],
        )
    if bool(options_flow_context.get("operator_action_required", False)):
        auth_issue = str(options_flow_context.get("auth_issue") or "").strip() or "options_flow_auth_issue"
        operator_followups.append(
            f"review Polygon and Unusual Whales access because the collector reported {auth_issue} and the autofix bot cannot rotate API keys or invent export feeds for you"
        )

    commands_hygiene = _run_json(
        [str(PYTHON_BIN), str(COMMANDS_HYGIENE_SCRIPT), "--project-root", str(project_root), "--json"],
        cwd=project_root,
        timeout_sec=min(int(timeout_sec), 180),
    )
    commands_hygiene_payload = commands_hygiene.get("payload") if isinstance(commands_hygiene.get("payload"), dict) else {}
    command_validity = _run_json(
        [str(PYTHON_BIN), str(COMMAND_VALIDITY_SCRIPT), "--project-root", str(project_root), "--json"],
        cwd=project_root,
        timeout_sec=min(int(timeout_sec), 180),
    )
    command_validity_payload = command_validity.get("payload") if isinstance(command_validity.get("payload"), dict) else {}
    if (
        str(commands_hygiene_payload.get("overall_status") or "") in {"degraded", "blocked"}
        or bool(commands_hygiene_payload.get("commands_changed", False))
        or bool(commands_hygiene_payload.get("runbook_changed", False))
    ):
        add_plan(
            "commands_hygiene",
            "commands_md_or_runbook_drift",
            [
                str(PYTHON_BIN),
                str(COMMANDS_HYGIENE_SCRIPT),
                "--project-root",
                str(project_root),
                "--apply",
                "--json",
            ],
        )
    if (
        str(command_validity_payload.get("overall_status") or "") == "blocked"
        or _safe_int((((command_validity_payload.get("metrics") or {}).get("blocked_entry_count"))), 0) > 0
    ):
        add_plan(
            "command_validity",
            "commands_md_contains_invalid_or_unresolved_entries",
            [
                str(PYTHON_BIN),
                str(COMMAND_VALIDITY_SCRIPT),
                "--project-root",
                str(project_root),
                "--apply",
                "--json",
            ],
        )
    drift_status = str(system_drift.get("overall_status") or "")
    if drift_status in {"blocked", "degraded"}:
        add_plan(
            "system_drift_autopilot",
            f"system_drift_status={drift_status}",
            [
                str(PYTHON_BIN),
                str(project_root / "scripts" / "ops" / "system_drift_autopilot.py"),
                "--apply",
                "--json",
            ],
        )

    master_status = str(master_infra.get("overall_status") or "")
    if not master_infra or master_status in {"blocked", "degraded"}:
        add_plan(
            "master_infrastructure_supervisor_refresh",
            f"master_infra_status={master_status or 'missing'}",
            [str(PYTHON_BIN), str(MASTER_INFRA_SUPERVISOR_SCRIPT), "--json"],
        )
        for check in master_infra.get("checks") or []:
            if not isinstance(check, dict) or check.get("name") != "launchd_job_health":
                continue
            if check.get("status") in {"blocked", "degraded"}:
                add_plan(
                    "ops_automation_launchd_install",
                    "launchd_job_health_not_ready",
                    [str(project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh")],
                )
            break

    auth_status = str(auth_lease.get("overall_status") or "")
    if auth_status in {"blocked", "degraded"}:
        add_plan(
            "schwab_auth_supervisor",
            f"auth_lease_status={auth_status}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "schwab_auth_supervisor.py"), "--apply", "--json"],
        )

    schwab_auth_status = str(schwab_auth_supervisor.get("overall_status") or "")
    if schwab_auth_status in {"blocked", "degraded"}:
        add_plan(
            "schwab_auth_supervisor",
            f"schwab_auth_supervisor_status={schwab_auth_status}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "schwab_auth_supervisor.py"), "--apply", "--json"],
        )

    if str(freshness.get("overall_status") or "") in {"blocked", "degraded"}:
        add_plan(
            "artifact_freshness_refresh",
            f"artifact_freshness_status={str(freshness.get('overall_status') or '')}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "artifact_freshness_slo.py"), "--json"],
        )

    if str(snapshot_cache.get("overall_status") or "") in {"blocked", "degraded"}:
        add_plan(
            "runtime_snapshot_refresh",
            f"snapshot_cache_status={str(snapshot_cache.get('overall_status') or '')}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "build_runtime_training_snapshot.py"), "--json"],
        )
        add_plan(
            "runtime_snapshot_recheck",
            "refresh_runtime_snapshot_controls",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "runtime_snapshot_cache_control.py"), "--json"],
        )

    if str(blackstart.get("overall_status") or "") in {"blocked", "degraded"}:
        add_plan(
            "restart_sanity_bundle",
            f"blackstart_status={str(blackstart.get('overall_status') or '')}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "restart_sanity_bundle.py"), "--json"],
        )
        add_plan(
            "blackstart_recheck",
            "refresh_blackstart_status",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "blackstart_recovery.py"), "--json"],
        )

    if (
        str(training_quality.get("overall_status") or "") in {"blocked", "degraded"}
        or str(supportability.get("overall_status") or "") in {"blocked", "degraded"}
        or str(bot_quality.get("overall_status") or "") in {"blocked", "degraded"}
    ):
        add_plan(
            "bot_quality_autopilot",
            "bot_quality_or_supportability_degraded",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "bot_quality_autopilot.py"), "--apply", "--json"],
        )

    channels = remote_alert.get("channels") if isinstance(remote_alert.get("channels"), dict) else {}
    if not bool(channels.get("any_configured", False)):
        operator_followups.append("configure at least one remote alert channel because the autofix bot cannot invent webhook or pushover credentials")
    if str(remote_alert.get("overall_status") or "") == "blocked" and _safe_int(((remote_alert.get("critical_backlog") or {}).get("unsent_count")), 0) > 0:
        operator_followups.append("review unsent critical alerts after configuring remote alert channels so the pager backlog drains cleanly")

    attempts: list[dict[str, Any]] = []
    if apply:
        for row in repair_plan:
            attempts.append(_run_json(list(row.get("cmd") or []), cwd=project_root, timeout_sec=timeout_sec))
        for cmd in (
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "runtime_gate_dashboard.py"), "--json"],
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "operator_cockpit.py"), "--json"],
            [str(PYTHON_BIN), str(project_root / "scripts" / "collector_contracts.py"), "--json"],
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "source_verification_report.py"), "--json"],
        ):
            attempts.append(_run_json(cmd, cwd=project_root, timeout_sec=min(int(timeout_sec), 300)))

    hard_failed_attempts = [
        row
        for row in attempts
        if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2}
    ]
    degraded_attempts = [row for row in attempts if int(row.get("rc", 1)) == 2 and not bool(row.get("timed_out", False))]
    overall_status = "ready"
    if operator_followups or hard_failed_attempts:
        overall_status = "blocked"
    elif repair_plan or degraded_attempts:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "install the infrastructure autofix bot on a timer so safe repairs happen before small degradations stack into outages",
            "let the bot run in apply mode for safe fixes, but keep destructive retention or credential changes operator-gated",
            "configure remote alert delivery so degraded states page you instead of silently accumulating" if operator_followups else "",
            "use the bot-quality autopilot alongside the infrastructure autofix bot so system health and bot quality improve together" if repair_plan else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "repair_plan": repair_plan,
        "attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in attempts
        ],
        "applyable_repair_count": len(repair_plan),
        "operator_followups": operator_followups,
        "metrics": {
            "daily_verify_failed_checks": len(failed_checks),
            "retention_debt_gb": retention_debt_gb,
            "auth_expires_in_seconds": _safe_float(((auth_lease.get("lease_budget") or {}).get("expires_in_seconds")), 0.0),
            "snapshot_ready": bool(((snapshot_cache.get("cache_health") or {}).get("snapshot_ready", False))),
            "unsent_critical_alerts": _safe_int(((remote_alert.get("critical_backlog") or {}).get("unsent_count")), 0),
            "storage_total_pending_lines": total_pending_lines,
            "storage_total_drain_minutes": total_drain_minutes,
            "stateful_storage_local_gb": _safe_float(((stateful_storage_guard.get("metrics") or {}).get("local_stateful_gb")), 0.0),
            "storage_reconnect_repair_plan_count": _safe_int(((storage_reconnect.get("metrics") or {}).get("repair_plan_count")), 0),
            "storage_reconnect_missing_contract_count": _safe_int(((storage_reconnect_guard.get("metrics") or {}).get("missing_contract_count")), 0),
            "commands_duplicate_entries": _safe_int((((commands_hygiene_payload.get("metrics") or {}).get("duplicate_entry_count"))), 0),
            "commands_runbook_changed": bool(commands_hygiene_payload.get("runbook_changed", False)),
            "commands_blocked_entries": _safe_int((((command_validity_payload.get("metrics") or {}).get("blocked_entry_count"))), 0),
        },
        "infra_bots": [
            "infrastructure_autofix_bot",
            "commands_hygiene_bot",
            "command_validity_bot",
            "system_drift_guard",
            "system_drift_autopilot",
            "daily_verify_auto_remediation_bot",
            "storage_pressure_clearance_bot",
            "storage_reconnect_infrabot",
            "storage_reconnect_regression_guard",
            "core_bot_materialization_infrabot",
            "core_bot_materialization_guard",
            "core_bot_tier_organizer",
            "storage_backpressure_autopilot",
            "stateful_storage_regression_guard",
            "one_numbers_regression_guard",
            "master_infrastructure_supervisor",
            "premarket_token_guard",
            "bot_quality_autopilot",
            "restart_sanity_bundle",
            "writer_lock_handoff_infrabot",
            "health_truth_reconciler_infrabot",
            "provider_cross_verification_infrabot",
            "paper_feedback_repair_infrabot",
            "promotion_replay_gate_infrabot",
            "bot_data_labeling_targeter_infrabot",
            "recovery_drill_infrabot",
            "self_audit_freshness_infrabot",
            "cotenant_headroom_guard_infrabot",
            "protected_volume_boundary_infrabot",
            "training_batch_readiness_infrabot",
            "paper_execution_queue_reconciler_infrabot",
            "duplicate_alpha_compression_infrabot",
            "livefeed_mirror_continuity_infrabot",
            "auth_lease_preflight_infrabot",
            "market_explanation_evidence_infrabot",
        ],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely orchestrate automated fixes for degraded infrastructure surfaces.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "infrastructure_autofix_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_plan={int(payload.get('applyable_repair_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
