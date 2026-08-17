#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_runtime_control_latest.json"
TRAINING_REPAIR_ACTIONS = {
    "rebuild_model_artifact",
    "calibrate_abstention_before_retry",
    "collect_more_data_before_retry",
    "quality_guard_repair_before_retry",
    "recover_training_log",
    "repair_runtime_inputs",
    "refresh_training_diagnostics",
    "targeted_retrain",
}
TRAINING_BATCH_PROFILES = {
    "coverage_micro_canary",
    "coverage_small_canary",
    "coverage_canary",
    "coverage_batch10_canary",
    "coverage_batch20_canary",
    "coverage_batch30_canary",
}
TRAINING_BATCH_MAX = 30
TRAINING_TIMEOUT_FALLBACK_WINDOW_MINUTES = 24 * 60
TRAINING_TARGET_COOLDOWN_MINUTES = 24 * 60
STORAGE_OVERRIDE_MAX_AGE_SECONDS = 900.0
SUPPORT_MAINTENANCE_FREEZE_REASON = "support_maintenance_frozen_for_mac_fluidity"
CREATIVE_SESSION_ACTIVE_REASON = "creative_session_active"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml_backend_contract import resolve_backend_contract
from core.runtime_python import resolve_runtime_python


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_minutes(raw: Any) -> float | None:
    ts = _parse_ts(raw)
    if ts is None:
        return None
    return max((datetime.now(timezone.utc) - ts).total_seconds() / 60.0, 0.0)


def _build_training_evidence_gate(project_root: Path, *, max_age_minutes: int) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    paths = {
        "feature_store": project_root / "governance" / "feature_store" / "latest.json",
        "schema_compatibility": health / "retrain_schema_compatibility_latest.json",
        "golden_replay": health / "golden_replay_regression_latest.json",
        "bot_needs": health / "bot_needs_intelligence_latest.json",
    }
    artifacts = {name: _load_json(path) for name, path in paths.items()}
    production_registry_present = (project_root / "master_bot_registry.json").is_file()
    expected = bool(
        production_registry_present
        or paths["feature_store"].is_file()
        or paths["schema_compatibility"].is_file()
        or paths["golden_replay"].is_file()
    )
    if not expected:
        return {
            "active": False,
            "ready": True,
            "mode": "not_configured",
            "blockers": [],
            "policy": "production repositories enforce strict lineage, schema, replay, and lifecycle evidence",
        }

    ages = {name: _age_minutes(payload.get("timestamp_utc")) for name, payload in artifacts.items()}
    fresh = {
        name: bool(age is not None and age <= max(int(max_age_minutes), 1))
        for name, age in ages.items()
    }
    feature = artifacts["feature_store"]
    schema = artifacts["schema_compatibility"]
    replay = artifacts["golden_replay"]
    bot_needs = artifacts["bot_needs"]
    stage_board = bot_needs.get("training_stage_board") if isinstance(bot_needs.get("training_stage_board"), dict) else {}
    checks = {
        "feature_store_strict_ready": bool(feature.get("strict_ok", False) and fresh["feature_store"]),
        "schema_compatibility_ready": bool(
            schema.get("ok", False)
            and not list(schema.get("failed_checks") or [])
            and fresh["schema_compatibility"]
        ),
        "golden_replay_strict_ready": bool(
            replay.get("strict_ready", replay.get("overall_status") == "ready")
            and replay.get("ok", False)
            and fresh["golden_replay"]
        ),
        "training_lifecycle_invariants_ready": bool(stage_board.get("ready", False) and fresh["bot_needs"]),
    }
    epoch_ids = {
        str((payload.get("evidence_epoch") or {}).get("id") or "")
        for payload in artifacts.values()
        if isinstance(payload.get("evidence_epoch"), dict)
        and str((payload.get("evidence_epoch") or {}).get("id") or "").strip()
    }
    epoch_declared_count = sum(
        1 for payload in artifacts.values() if str((payload.get("evidence_epoch") or {}).get("id") or "").strip()
    )
    epoch_consistent = bool(
        (not production_registry_present and epoch_declared_count == 0)
        or (epoch_declared_count == len(artifacts) and len(epoch_ids) == 1)
    )
    checks["single_evidence_epoch"] = epoch_consistent
    blocker_by_check = {
        "feature_store_strict_ready": "feature_store_not_strict_ready",
        "schema_compatibility_ready": "retrain_schema_compatibility_not_ready",
        "golden_replay_strict_ready": "golden_replay_not_strict_ready",
        "training_lifecycle_invariants_ready": "training_lifecycle_invariant_failed",
        "single_evidence_epoch": "training_evidence_epoch_mismatch",
    }
    blockers = [blocker_by_check[key] for key, value in checks.items() if not value]
    return {
        "active": True,
        "ready": not blockers,
        "mode": "strict_production_training_evidence",
        "checks": checks,
        "blockers": blockers,
        "artifact_ages_minutes": {name: round(age, 3) if age is not None else None for name, age in ages.items()},
        "maximum_age_minutes": max(int(max_age_minutes), 1),
        "evidence_epoch_ids": sorted(epoch_ids),
        "evidence_epoch_declared_count": epoch_declared_count,
        "source_artifacts": {name: str(path) for name, path in paths.items()},
        "policy": "training launches fail closed unless feature lineage, schema compatibility, deterministic replay, bot lifecycle invariants, and evidence epoch agree",
    }


def _training_candidate_selector_contract(
    bot_needs: dict[str, Any],
    *,
    max_age_minutes: int,
) -> dict[str, Any]:
    selector = bot_needs.get("training_candidate_selector") if isinstance(bot_needs.get("training_candidate_selector"), dict) else {}
    active = bool(selector.get("active", False))
    age_minutes = _age_minutes(bot_needs.get("timestamp_utc"))
    fresh = bool(age_minutes is not None and age_minutes <= max(int(max_age_minutes), 1))
    selected_candidates = [
        row
        for row in selector.get("selected_candidates") or []
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    ]
    selected_bot_ids = _ordered_unique(
        [str(row.get("bot_id") or "").strip().lower() for row in selected_candidates]
    )
    if not selector:
        status = "unavailable"
        reason = "training_candidate_selector_missing"
    elif not active:
        status = "inactive"
        reason = "training_candidate_selector_inactive"
    elif not fresh:
        status = "stale"
        reason = "training_candidate_selector_not_fresh"
    else:
        status = "ready"
        reason = "fresh_training_candidate_selector_authoritative"
    return {
        "status": status,
        "reason": reason,
        "active": active,
        "fresh": fresh,
        "authoritative": bool(active and fresh),
        "age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
        "maximum_age_minutes": max(int(max_age_minutes), 1),
        "selected_count": len(selected_bot_ids),
        "candidate_count": _safe_int(selector.get("candidate_count"), len(selected_bot_ids)),
        "selected_bot_ids": selected_bot_ids,
        "selected_candidates": selected_candidates,
        "mode": str(selector.get("mode") or ""),
        "policy": "launch_only_the_intersection_of_runtime_safe_and_fresh_bot_needs_selected_candidates",
    }


def _apply_training_target_cooldown(
    selector: dict[str, Any],
    retrain_scorecard: dict[str, Any],
    *,
    cooldown_minutes: int,
) -> dict[str, Any]:
    out = dict(selector)
    selected_candidates = [
        dict(row)
        for row in selector.get("selected_candidates") or []
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    ]
    scorecard_timestamp = str(
        retrain_scorecard.get("ended_utc")
        or retrain_scorecard.get("timestamp_utc")
        or retrain_scorecard.get("started_utc")
        or ""
    ).strip()
    age_minutes = _age_minutes(scorecard_timestamp)
    window_minutes = max(int(cooldown_minutes), 0)
    successful_bot_ids = {
        str(row.get("bot_id") or "").strip().lower()
        for row in retrain_scorecard.get("target_outcomes") or []
        if isinstance(row, dict)
        and str(row.get("bot_id") or "").strip()
        and str(row.get("status") or "").strip().lower() in {"trained", "success", "ok"}
    }
    active = bool(
        window_minutes > 0
        and age_minutes is not None
        and age_minutes <= window_minutes
        and successful_bot_ids
    )
    available: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    remaining_minutes = max(float(window_minutes) - float(age_minutes or 0.0), 0.0) if active else 0.0
    for row in selected_candidates:
        bot_id = str(row.get("bot_id") or "").strip()
        if active and bot_id.lower() in successful_bot_ids:
            blocked.append(
                {
                    **row,
                    "cooldown_reason": "recent_successful_training_run",
                    "cooldown_remaining_minutes": round(remaining_minutes, 3),
                }
            )
        else:
            available.append(row)

    scorecard_dt = _parse_ts(scorecard_timestamp)
    cooldown_until_utc = (
        (scorecard_dt + timedelta(minutes=window_minutes)).isoformat()
        if active and scorecard_dt is not None
        else ""
    )
    out["upstream_selected_count"] = len(selected_candidates)
    out["selected_candidates"] = available
    out["selected_bot_ids"] = _ordered_unique(
        [str(row.get("bot_id") or "").strip().lower() for row in available]
    )
    out["selected_count"] = len(out["selected_bot_ids"])
    out["cooldown"] = {
        "active": bool(active and blocked),
        "window_minutes": window_minutes,
        "scorecard_timestamp_utc": scorecard_timestamp,
        "scorecard_age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
        "cooldown_until_utc": cooldown_until_utc,
        "blocked_count": len(blocked),
        "blocked_bot_ids": [str(row.get("bot_id") or "") for row in blocked],
        "blocked_candidates": blocked,
        "policy": "a successful target may run at most once per cooldown window while evidence artifacts catch up",
    }
    if bool(out.get("authoritative", False)) and blocked and not available:
        out["status"] = "cooldown"
        out["reason"] = "recent_successful_training_target_cooldown"
    out["policy"] = (
        "launch_only_the_intersection_of_runtime_safe, fresh bot-needs candidates, "
        "and targets outside the successful-run cooldown"
    )
    return out


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _command_option(command: list[Any], flag: str) -> str:
    parts = [str(part) for part in command]
    try:
        index = parts.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(parts):
        return ""
    return parts[index + 1].strip()


def _recent_small_canary_timeout(autopilot: dict[str, Any]) -> dict[str, Any]:
    launch_result = autopilot.get("last_launch_result") if isinstance(autopilot.get("last_launch_result"), dict) else {}
    gate = autopilot.get("last_training_gate") if isinstance(autopilot.get("last_training_gate"), dict) else {}
    command = gate.get("recommended_command") if isinstance(gate.get("recommended_command"), list) else []
    profile = _command_option(command, "--retrain-profile")
    bot_arg = _command_option(command, "--include-bot-ids")
    bot_ids = [part.strip() for part in bot_arg.split(",") if part.strip()]
    age_minutes = _age_minutes(autopilot.get("timestamp_utc"))
    fresh = bool(
        age_minutes is not None
        and age_minutes <= TRAINING_TIMEOUT_FALLBACK_WINDOW_MINUTES
    )
    timed_out = bool(launch_result.get("timed_out", False))
    timeout_profiles = {"coverage_micro_canary", "coverage_small_canary"}
    active = bool(timed_out and fresh and profile in timeout_profiles and len(bot_ids) >= 1)
    return {
        "active": active,
        "fresh": fresh,
        "timed_out": timed_out,
        "age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
        "window_minutes": TRAINING_TIMEOUT_FALLBACK_WINDOW_MINUTES,
        "profile": profile,
        "bot_ids": bot_ids,
        "returncode": launch_result.get("returncode"),
        "command": [str(part) for part in command],
        "source": "training_drain_autopilot",
    }


def _recent_retrain_launch_timeout(launch: dict[str, Any]) -> dict[str, Any]:
    profile = str(launch.get("retrain_profile") or "").strip()
    selector = launch.get("selector_summary") if isinstance(launch.get("selector_summary"), dict) else {}
    bot_ids = [str(item).strip() for item in selector.get("include_bot_ids") or [] if str(item).strip()]
    final_status = str(launch.get("final_status") or "").strip().lower()
    exit_code = _safe_int(launch.get("exit_code"), _safe_int(launch.get("returncode"), 0))
    age_minutes = (
        _age_minutes(launch.get("ended_utc"))
        or _age_minutes(launch.get("finished_utc"))
        or _age_minutes(launch.get("timestamp_utc"))
        or _age_minutes(launch.get("started_utc"))
    )
    fresh = bool(
        age_minutes is not None
        and age_minutes <= TRAINING_TIMEOUT_FALLBACK_WINDOW_MINUTES
    )
    timeout_profiles = {"coverage_micro_canary", "coverage_small_canary"}
    timed_out = bool(exit_code == 124 or "timed_out" in final_status or "timeout" in final_status)
    active = bool(timed_out and fresh and profile in timeout_profiles and len(bot_ids) >= 1)
    return {
        "active": active,
        "fresh": fresh,
        "timed_out": timed_out,
        "age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
        "window_minutes": TRAINING_TIMEOUT_FALLBACK_WINDOW_MINUTES,
        "profile": profile,
        "bot_ids": bot_ids,
        "returncode": exit_code,
        "command": [str(part) for part in launch.get("argv") or []],
        "source": "retrain_launch_latest",
        "final_status": final_status,
        "timeout_phase": str(launch.get("timeout_phase") or launch.get("phase") or ""),
        "timeout_progress": launch.get("timeout_progress") if isinstance(launch.get("timeout_progress"), dict) else {},
    }


def _select_timeout_fallback(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in candidates if bool(row.get("active", False))]
    if active:
        return sorted(active, key=lambda row: _safe_float(row.get("age_minutes"), 999999.0))[0]
    return candidates[0] if candidates else {
        "active": False,
        "fresh": False,
        "timed_out": False,
        "age_minutes": None,
        "window_minutes": TRAINING_TIMEOUT_FALLBACK_WINDOW_MINUTES,
        "profile": "",
        "bot_ids": [],
        "returncode": None,
        "command": [],
        "source": "",
    }


def _support_maintenance_freeze_only(reasons: list[str]) -> bool:
    normalized = {str(item or "").strip().lower() for item in reasons if str(item or "").strip()}
    return bool(normalized) and normalized.issubset({SUPPORT_MAINTENANCE_FREEZE_REASON})


def _training_advisory_resource_guard_only(reasons: list[str]) -> bool:
    normalized = {str(item or "").strip().lower() for item in reasons if str(item or "").strip()}
    return bool(normalized) and normalized.issubset(
        {
            SUPPORT_MAINTENANCE_FREEZE_REASON,
            CREATIVE_SESSION_ACTIVE_REASON,
        }
    )


def _runtime_backend_probe(project_root: Path) -> dict[str, Any]:
    runtime_python = resolve_runtime_python(project_root)
    current_python = Path(sys.executable).resolve()
    probe = {
        "runtime_python_path": str(runtime_python),
        "current_python_path": str(current_python),
        "runtime_python_exists": runtime_python.exists(),
        "runtime_matches_current": runtime_python.exists() and runtime_python.resolve() == current_python,
        "installed_backends": {},
        "native_contract": {},
        "portable_contract": {},
        "probe_rc": 127,
        "probe_error": "",
        "parity_state": "missing_runtime_python",
    }
    if not runtime_python.exists():
        return probe

    cmd = [
        str(runtime_python),
        "-c",
        (
            "import importlib.util, json, platform, sys; "
            "mods={name:(importlib.util.find_spec(name) is not None) for name in ('mlx','torch','onnxruntime','tensorflow','jax')}; "
            "print(json.dumps({'python': sys.version.split()[0], 'platform': platform.platform(), 'modules': mods}, ensure_ascii=True))"
        ),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        probe["probe_rc"] = int(proc.returncode)
        if proc.returncode == 0:
            parsed = {}
            for raw in reversed([line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]):
                try:
                    candidate = json.loads(raw)
                except Exception:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            modules = parsed.get("modules") if isinstance(parsed.get("modules"), dict) else {}
            installed = {
                "mlx": bool(modules.get("mlx", False)),
                "pytorch": bool(modules.get("torch", False)),
                "onnx": bool(modules.get("onnxruntime", False)),
                "tensorflow": bool(modules.get("tensorflow", False)),
                "jax": bool(modules.get("jax", False)),
            }
            probe["installed_backends"] = installed
            probe["runtime_python_version"] = str(parsed.get("python") or "")
            probe["runtime_platform"] = str(parsed.get("platform") or "")
            probe["native_contract"] = resolve_backend_contract("native_default", mode="native", installed=installed)
            probe["portable_contract"] = resolve_backend_contract("portable_auto", mode="portable", installed=installed)
        else:
            probe["probe_error"] = "\n".join((proc.stderr or "").splitlines()[-8:]) or "\n".join((proc.stdout or "").splitlines()[-8:])
    except Exception as exc:
        probe["probe_error"] = str(exc)

    native_contract = probe.get("native_contract") if isinstance(probe.get("native_contract"), dict) else {}
    if int(probe.get("probe_rc", 127)) != 0:
        probe["parity_state"] = "runtime_probe_failed"
    elif bool(native_contract.get("runtime_training_supported", False)):
        probe["parity_state"] = "ready"
    elif bool((probe.get("portable_contract") or {}).get("shadow_replay_supported", False)):
        probe["parity_state"] = "portable_only"
    else:
        probe["parity_state"] = "native_backend_missing"
    return probe


def _bot_family(bot_id: str) -> str:
    lowered = str(bot_id or "").strip().lower()
    for token, family in (
        ("intraday", "intraday"),
        ("swing", "swing"),
        ("crypto", "crypto"),
        ("bond", "bond"),
        ("fx", "fx"),
        ("dividend", "dividend"),
        ("futures", "futures"),
    ):
        if token in lowered:
            return family
    return "general"


def _sequence_timeout_reason(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("reason") or ""),
            str(row.get("stdout_tail") or ""),
            str(row.get("stderr_tail") or ""),
        ]
    ).lower()
    if "loading_sequences" in text:
        return "loading_sequences_timeout"
    if "memory_guard" in text:
        return "memory_guard"
    if "timeout" in text:
        return "runtime_timeout"
    return ""


def _build_precompute_targets(
    *,
    training_quality: dict[str, Any],
    retrain_scorecard: dict[str, Any],
    coverage_seed: dict[str, Any],
    coverage_gap_closer: dict[str, Any],
    training_requalification: dict[str, Any],
    bot_needs: dict[str, Any],
    candidate_selector: dict[str, Any],
    candidate_advancement: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidate_advancement = candidate_advancement or {}
    targets: dict[str, dict[str, Any]] = {}
    selector_authoritative = bool(candidate_selector.get("authoritative", False))
    selector_selected_ids = {
        str(bot_id or "").strip().lower()
        for bot_id in candidate_selector.get("selected_bot_ids") or []
        if str(bot_id or "").strip()
    }

    def ensure(bot_id: str) -> dict[str, Any]:
        row = targets.setdefault(
            bot_id,
            {
                "bot_id": bot_id,
                "family": _bot_family(bot_id),
                "priority": 0.0,
                "reasons": [],
                "actions": [],
                "candidate_actions": [],
                "current_runs": 0,
                "runs_remaining": 0,
                "needs_runtime_input_repair": False,
                "bot_needs_selector_authoritative": selector_authoritative,
                "bot_needs_can_train_now": bool(
                    selector_authoritative and bot_id.strip().lower() in selector_selected_ids
                ),
            },
        )
        return row

    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    for bot_id in targeted_actions.get("targeted_retrain_bot_ids") or []:
        bot = str(bot_id or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + 20.0
        row["reasons"].append("targeted_retrain")
        row["actions"].append("precompute_or_refresh_shared_snapshot")

    for bot_id in targeted_actions.get("quality_probation_bot_ids") or []:
        bot = str(bot_id or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + 12.0
        row["reasons"].append("quality_probation")
        row["actions"].append("retry_after_runtime_cache_refresh")

    for failure in retrain_scorecard.get("failure_details") or []:
        if not isinstance(failure, dict):
            continue
        bot = str(failure.get("bot_id") or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + 30.0
        timeout_reason = _sequence_timeout_reason(failure)
        row["reasons"].append(timeout_reason or "retrain_failure")
        row["actions"].append("pin_sequence_cache_before_retry")

    for outcome in retrain_scorecard.get("target_outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        bot = str(outcome.get("bot_id") or "").strip()
        status = str(outcome.get("status") or "").strip().lower()
        if not bot or status in {"", "trained", "success", "ok"}:
            continue
        row = ensure(bot)
        reason_text = " ".join(
            str(outcome.get(key) or "")
            for key in ("reason", "failure_reason", "message")
        ).lower()
        if status == "deferred_sample_starved" or "defer_runtime_training_until_more_data" in reason_text:
            row["priority"] = float(row["priority"]) + 34.0
            row["reasons"].append("latest_deferred_sample_starved")
            row["actions"].append("collect_more_data_before_retry")
            row["candidate_actions"].append("route_to_data_first_requalification")
            row["needs_runtime_input_repair"] = True
        elif status == "failed" and "synthetic_training_quality_guard_failed" in reason_text:
            row["priority"] = float(row["priority"]) + 32.0
            row["reasons"].append("latest_quality_guard_failed")
            row["actions"].append("quality_guard_repair_before_retry")
            row["actions"].append("calibrate_abstention_before_retry")
            row["candidate_actions"].append("repair_long_short_precision_and_acted_coverage")
        else:
            row["priority"] = float(row["priority"]) + 24.0
            row["reasons"].append(f"latest_retrain_{status}")
            row["actions"].append("refresh_training_diagnostics")

    for seed_row in coverage_seed.get("seed_queue") or []:
        if not isinstance(seed_row, dict):
            continue
        bot = str(seed_row.get("bot_id") or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + min(_safe_float(seed_row.get("priority"), 0.0), 15.0)
        row["reasons"].append("coverage_seed")
        row["actions"].append("reuse_shared_snapshot_for_walk_forward")
        row["current_runs"] = max(_safe_int(row.get("current_runs"), 0), _safe_int(seed_row.get("current_runs"), 0))
        row["runs_remaining"] = max(_safe_int(row.get("runs_remaining"), 0), _safe_int(seed_row.get("runs_remaining"), 0))
        row["needs_runtime_input_repair"] = bool(row.get("needs_runtime_input_repair", False) or seed_row.get("needs_runtime_input_repair", False))
        for action in seed_row.get("actions") or []:
            if str(action or "").strip():
                row["candidate_actions"].append(str(action).strip())

    def add_ranked_candidate(candidate_row: dict[str, Any], *, reason: str, priority_cap: float, default_action: str) -> None:
        bot = str(candidate_row.get("bot_id") or "").strip()
        if not bot:
            return
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + min(_safe_float(candidate_row.get("priority"), 0.0), float(priority_cap))
        row["reasons"].append(reason)
        row["actions"].append(default_action)
        row["current_runs"] = max(
            _safe_int(row.get("current_runs"), 0),
            _safe_int(candidate_row.get("current_runs"), _safe_int(candidate_row.get("walk_forward_runs"), 0)),
        )
        row["runs_remaining"] = max(_safe_int(row.get("runs_remaining"), 0), _safe_int(candidate_row.get("runs_remaining"), 0))
        row["needs_runtime_input_repair"] = bool(
            row.get("needs_runtime_input_repair", False)
            or candidate_row.get("needs_runtime_input_repair", False)
            or "repair_runtime_inputs" in [str(action or "").strip() for action in candidate_row.get("actions") or []]
        )
        for action in candidate_row.get("actions") or []:
            if str(action or "").strip():
                row["candidate_actions"].append(str(action).strip())

    for candidate_row in coverage_gap_closer.get("active_stage_candidates") or []:
        if isinstance(candidate_row, dict):
            add_ranked_candidate(
                candidate_row,
                reason="coverage_gap_stage",
                priority_cap=18.0,
                default_action="reuse_shared_snapshot_for_staged_candidate",
            )

    for candidate_row in coverage_gap_closer.get("backup_candidates") or []:
        if isinstance(candidate_row, dict):
            add_ranked_candidate(
                candidate_row,
                reason="coverage_gap_backup",
                priority_cap=12.0,
                default_action="prepare_backup_candidate_for_walk_forward",
            )

    ranked_requalification: list[dict[str, Any]] = []
    seen_requalification: set[str] = set()
    for key in ("top_reactivation_ready", "top_candidates"):
        for candidate_row in training_requalification.get(key) or []:
            if isinstance(candidate_row, dict):
                bot = str(candidate_row.get("bot_id") or "").strip().lower()
                if not bot or bot in seen_requalification:
                    continue
                seen_requalification.add(bot)
                ranked_requalification.append(candidate_row)
    for candidate_row in ranked_requalification:
        add_ranked_candidate(
            candidate_row,
            reason="training_requalification_candidate",
            priority_cap=10.0,
            default_action="prepare_requalification_candidate_for_batch_training",
        )

    for candidate_row in candidate_advancement.get("training_queue") or []:
        if not isinstance(candidate_row, dict):
            continue
        add_ranked_candidate(
            candidate_row,
            reason="promotion_candidate_advancement",
            priority_cap=25.0,
            default_action="prepare_candidate_bound_walk_forward_training",
        )

    bot_needs_records: dict[str, dict[str, Any]] = {}
    for need_row in bot_needs.get("bot_needs") or []:
        if not isinstance(need_row, dict):
            continue
        bot = str(need_row.get("bot_id") or "").strip()
        if bot:
            bot_needs_records[bot] = need_row
    next_batches = bot_needs.get("next_batches") if isinstance(bot_needs.get("next_batches"), dict) else {}
    for rank, raw_bot in enumerate(next_batches.get("training_topoff") or []):
        bot = str(raw_bot or "").strip()
        if not bot:
            continue
        need_row = bot_needs_records.get(bot, {})
        evidence = need_row.get("evidence") if isinstance(need_row.get("evidence"), dict) else {}
        row = ensure(bot)
        fallback_priority = max(52.0 - (rank * 0.1), 40.0)
        row["priority"] = float(row["priority"]) + min(_safe_float(need_row.get("priority"), fallback_priority), 14.0)
        row["reasons"].append("bot_needs_training_topoff")
        row["actions"].append("reuse_shared_snapshot_for_walk_forward")
        row["actions"].append("prepare_memory_guarded_training_topoff")
        row["candidate_actions"].extend(
            [
                "generate_walk_forward_runs",
                "refresh_promotion_gate",
                "recheck_promotion_quality_gate",
            ]
        )
        row["current_runs"] = max(
            _safe_int(row.get("current_runs"), 0),
            _safe_int(evidence.get("walk_forward_runs"), 0),
        )
        row["runs_remaining"] = max(
            _safe_int(row.get("runs_remaining"), 0),
            _safe_int(evidence.get("walk_forward_runs_remaining"), 0),
        )

    for bucket, action, candidate_action in (
        ("repair_first", "repair_runtime_inputs", "repair_before_batch_training"),
        ("collect_more_data", "collect_more_data_before_retry", "route_to_data_first_requalification"),
        ("calibration", "calibrate_abstention_before_retry", "repair_calibration_before_batch_training"),
        ("overfitting", "refresh_training_diagnostics", "clear_overfit_guard_before_batch_training"),
    ):
        for rank, raw_bot in enumerate(next_batches.get(bucket) or []):
            bot = str(raw_bot or "").strip()
            if not bot:
                continue
            need_row = bot_needs_records.get(bot, {})
            evidence = need_row.get("evidence") if isinstance(need_row.get("evidence"), dict) else {}
            row = ensure(bot)
            fallback_priority = max(60.0 - (rank * 0.1), 44.0)
            row["priority"] = float(row["priority"]) + min(_safe_float(need_row.get("priority"), fallback_priority), 16.0)
            row["reasons"].append(f"bot_needs_{bucket}")
            row["actions"].append(action)
            row["candidate_actions"].append(candidate_action)
            row["current_runs"] = max(
                _safe_int(row.get("current_runs"), 0),
                _safe_int(evidence.get("walk_forward_runs"), 0),
            )
            row["runs_remaining"] = max(
                _safe_int(row.get("runs_remaining"), 0),
                _safe_int(evidence.get("walk_forward_runs_remaining"), 0),
            )
            if bucket in {"repair_first", "collect_more_data"}:
                row["needs_runtime_input_repair"] = True

    for rank, selected in enumerate(candidate_selector.get("selected_candidates") or []):
        if not isinstance(selected, dict):
            continue
        bot = str(selected.get("bot_id") or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + max(30.0 - (rank * 0.1), 20.0)
        row["reasons"].append("bot_needs_training_candidate_selector")
        row["actions"].append("prepare_authorized_micro_canary")
        row["current_runs"] = max(
            _safe_int(row.get("current_runs"), 0),
            _safe_int(selected.get("walk_forward_runs"), 0),
        )
        row["runs_remaining"] = max(
            _safe_int(row.get("runs_remaining"), 0),
            _safe_int(selected.get("walk_forward_runs_remaining"), 0),
        )
        row["bot_needs_selector_authoritative"] = selector_authoritative
        row["bot_needs_can_train_now"] = bool(selector_authoritative and bot.lower() in selector_selected_ids)

    out: list[dict[str, Any]] = []
    for row in targets.values():
        row["reasons"] = _ordered_unique(list(row.get("reasons") or []))
        row["actions"] = _ordered_unique(list(row.get("actions") or []))
        row["candidate_actions"] = _ordered_unique(list(row.get("candidate_actions") or []))
        all_actions = {str(action or "").strip() for action in list(row["actions"]) + list(row["candidate_actions"])}
        repair_first = bool(row.get("needs_runtime_input_repair", False) or all_actions.intersection(TRAINING_REPAIR_ACTIONS))
        if selector_authoritative and bool(row.get("bot_needs_can_train_now", False)):
            row["training_stage"] = "selector_approved_canary"
        elif repair_first:
            row["training_stage"] = "repair_first"
        elif _safe_int(row.get("runs_remaining"), 0) > 0:
            row["training_stage"] = "coverage_topoff"
        elif "coverage_seed" in row["reasons"]:
            row["training_stage"] = "promotion_confirmation"
        elif "loading_sequences_timeout" in row["reasons"]:
            row["training_stage"] = "cache_retry"
        else:
            row["training_stage"] = "precompute"
        out.append(row)
    out.sort(key=lambda row: (-_safe_float(row.get("priority"), 0.0), str(row.get("bot_id") or "")))
    return out


def _public_target(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bot_id": str(row.get("bot_id") or ""),
        "family": str(row.get("family") or ""),
        "priority": round(_safe_float(row.get("priority"), 0.0), 3),
        "training_stage": str(row.get("training_stage") or "precompute"),
        "current_runs": _safe_int(row.get("current_runs"), 0),
        "runs_remaining": _safe_int(row.get("runs_remaining"), 0),
        "needs_runtime_input_repair": bool(row.get("needs_runtime_input_repair", False)),
        "bot_needs_selector_authoritative": bool(row.get("bot_needs_selector_authoritative", False)),
        "bot_needs_can_train_now": bool(row.get("bot_needs_can_train_now", False)),
        "reasons": list(row.get("reasons") or [])[:6],
        "actions": list(row.get("actions") or [])[:6],
        "candidate_actions": list(row.get("candidate_actions") or [])[:8],
    }


def _build_backpressure_training_gate(
    *,
    health_gates: dict[str, Any],
    storage_control: dict[str, Any],
    ingestion_backpressure: dict[str, Any],
    super_drainer: dict[str, Any],
) -> dict[str, Any]:
    inputs = health_gates.get("inputs") if isinstance(health_gates.get("inputs"), dict) else {}
    storage_bp = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage_overlay = (
        storage_control.get("sql_ingestion_pending_overlay")
        if isinstance(storage_control.get("sql_ingestion_pending_overlay"), dict)
        else {}
    )
    storage_control_override = (
        inputs.get("backpressure_storage_control_override")
        if isinstance(inputs.get("backpressure_storage_control_override"), dict)
        else {}
    )
    health_gates_age_minutes = _age_minutes(
        health_gates.get("timestamp_utc") or health_gates.get("generated_utc")
    )
    health_gates_age_seconds = (
        max(float(health_gates_age_minutes), 0.0) * 60.0
        if health_gates_age_minutes is not None
        else None
    )
    health_gates_fresh_for_override = bool(
        health_gates_age_seconds is not None
        and health_gates_age_seconds <= STORAGE_OVERRIDE_MAX_AGE_SECONDS
    )
    super_summary = super_drainer.get("summary") if isinstance(super_drainer.get("summary"), dict) else {}
    super_drainer_age_minutes = _age_minutes(super_drainer.get("timestamp_utc")) if super_drainer else None
    super_drainer_fresh = bool(super_drainer and super_drainer_age_minutes is not None and super_drainer_age_minutes <= 30.0)
    storage_overlay_reconciled_downward = bool(
        storage_bp.get("overlay_adjusted", False)
        and storage_overlay.get("used_for_pressure", False)
        and storage_overlay.get("reconciled_downward_for_pressure", False)
    )
    pending_threshold = max(
        _safe_int(storage_bp.get("pending_lines_threshold"), 0),
        15_000,
    )
    oldest_threshold = max(
        _safe_float(storage_bp.get("oldest_age_threshold_seconds"), 0.0),
        240.0,
    )
    storage_status = str(storage_control.get("overall_status") or "").strip().lower()
    storage_severity = str(storage_control.get("severity") or "").strip().lower()
    storage_shedding = (
        storage_control.get("writer_shedding")
        if isinstance(storage_control.get("writer_shedding"), dict)
        else {}
    )
    storage_hard_breaches = [str(item or "").strip() for item in storage_shedding.get("hard_breaches") or [] if str(item or "").strip()]
    pressure_index = _safe_float(storage_control.get("pressure_index"), 0.0)
    override_pending_lines = max(
        _safe_int(storage_control_override.get("pending_lines"), 0),
        _safe_int(storage_control_override.get("pending_lines_total"), 0),
    )
    override_oldest_age = _safe_float(storage_control_override.get("oldest_pending_age_seconds"), 0.0)
    embedded_override_age_seconds = _safe_float(storage_control_override.get("age_seconds"), 0.0)
    effective_override_age_seconds = max(
        embedded_override_age_seconds,
        float(health_gates_age_seconds or 0.0),
    )
    override_clear = bool(
        storage_control_override.get("active", False)
        and health_gates_fresh_for_override
        and override_pending_lines <= pending_threshold
        and override_oldest_age <= oldest_threshold
        and not bool(storage_control_override.get("overload", False))
        and (
            bool(storage_control_override.get("queue_clear", False))
            or bool(storage_control_override.get("overlay_clear", False))
            or str(storage_control_override.get("source") or "").strip() == "fresh_empty_sql_ingestion_overlay"
        )
        and effective_override_age_seconds <= STORAGE_OVERRIDE_MAX_AGE_SECONDS
    )
    effective_pressure_index = 0.0 if override_clear else pressure_index
    storage_pending_lines = _safe_int(storage_bp.get("total_pending_lines"), 0)
    storage_oldest_age = _safe_float(storage_bp.get("oldest_pending_age_seconds"), 0.0)
    storage_backpressure_numeric_clear = bool(
        storage_bp
        and storage_pending_lines <= pending_threshold
        and storage_oldest_age <= oldest_threshold
        and pressure_index < 1.0
    )
    storage_status_authoritative = bool(
        storage_status in {"ready", "advisory", "ok", "stable", ""}
        or (
            storage_status == "needs_work"
            and storage_severity in {"ready", "advisory", "ok", "stable", "watch", ""}
            and not storage_hard_breaches
        )
        or storage_backpressure_numeric_clear
    )
    storage_live_authoritative = bool(
        storage_bp
        and storage_pending_lines <= pending_threshold
        and storage_oldest_age <= oldest_threshold
        and pressure_index < 1.0
        and (
            storage_status_authoritative
            or storage_status in {"blocked", "critical"}
            or storage_severity in {"blocked", "critical"}
        )
    )
    if override_clear:
        pending_lines = override_pending_lines
        oldest_age = override_oldest_age
    elif storage_overlay_reconciled_downward or storage_live_authoritative:
        pending_lines = _safe_int(storage_bp.get("total_pending_lines"), 0)
        oldest_age = _safe_float(storage_bp.get("oldest_pending_age_seconds"), 0.0)
    else:
        pending_lines = max(
            _safe_int(inputs.get("backpressure_pending_lines"), 0),
            _safe_int(storage_bp.get("total_pending_lines"), 0),
            _safe_int(ingestion_backpressure.get("pending_lines_total"), 0),
            _safe_int(ingestion_backpressure.get("pending_lines"), 0),
            _safe_int(super_drainer.get("final_pending_lines"), 0) if super_drainer_fresh else 0,
            _safe_int(super_summary.get("final_pending_lines"), 0) if super_drainer_fresh else 0,
        )
        oldest_age = max(
            _safe_float(inputs.get("backpressure_oldest_pending_age_seconds"), 0.0),
            _safe_float(storage_bp.get("oldest_pending_age_seconds"), 0.0),
            _safe_float(ingestion_backpressure.get("oldest_pending_age_seconds_total"), 0.0),
            _safe_float(ingestion_backpressure.get("oldest_pending_age_seconds"), 0.0),
        )
    storage_numeric_clear = bool(
        (storage_bp or override_clear)
        and pending_lines <= pending_threshold
        and oldest_age <= oldest_threshold
        and effective_pressure_index < 1.0
        and (storage_live_authoritative or override_clear)
    )
    health_severe = bool(inputs.get("backpressure_overload_severe", False))
    direct_overload = bool(ingestion_backpressure.get("overload", False))
    # Health-gate and ingestion artifacts can lag behind the SQL-backed storage
    # truth after a super-drain. Let fresh below-threshold storage measurements
    # clear stale severe flags, while still honoring true hard storage states.
    effective_health_severe = bool(health_severe and not storage_numeric_clear)
    effective_direct_overload = bool(direct_overload and not storage_numeric_clear)
    severe = bool(
        effective_health_severe
        or (storage_status in {"blocked", "critical"} and not storage_numeric_clear)
        or (storage_severity in {"blocked", "critical"} and not storage_numeric_clear)
        or effective_pressure_index >= 1.0
        or pending_lines > pending_threshold
        or oldest_age > oldest_threshold
        or (effective_direct_overload and (pending_lines > 0 or oldest_age > 0.0))
    )
    sql_status = str(inputs.get("sql_progress_status") or "").strip().lower()
    cooling_down = bool(
        severe
        and pending_lines <= 1000
        and oldest_age <= 1800.0
        and storage_status not in {"blocked", "critical"}
        and storage_severity not in {"blocked", "critical"}
        and effective_pressure_index < 1.0
        and sql_status in {"", "ok", "complete", "idle"}
    )
    sources = []
    if health_severe or _safe_int(inputs.get("backpressure_pending_lines"), 0) > 0:
        sources.append("health_gates")
    if storage_control:
        sources.append("ingestion_storage_control")
    if storage_overlay_reconciled_downward:
        sources.append("sql_overlay_reconciled_downward")
    if override_clear:
        sources.append("health_gate_storage_control_override")
    elif storage_control_override and not health_gates_fresh_for_override:
        sources.append("stale_health_gate_storage_control_override_ignored")
    if ingestion_backpressure:
        sources.append("ingestion_backpressure")
    if super_drainer_fresh:
        sources.append("backpressure_super_drainer")
    elif super_drainer:
        sources.append("stale_backpressure_super_drainer_ignored")
    return {
        "severe": severe,
        "cooling_down": cooling_down,
        "pending_lines": int(pending_lines),
        "oldest_pending_age_seconds": round(float(oldest_age), 3),
        "pending_lines_threshold": int(pending_threshold),
        "oldest_age_threshold_seconds": round(float(oldest_threshold), 3),
        "storage_status": storage_status,
        "storage_severity": storage_severity,
        "pressure_index": round(float(effective_pressure_index), 6),
        "raw_pressure_index": round(float(pressure_index), 6),
        "sql_progress_status": sql_status,
        "storage_numeric_clear": storage_numeric_clear,
        "storage_live_authoritative": storage_live_authoritative,
        "storage_backpressure_numeric_clear": storage_backpressure_numeric_clear,
        "storage_status_authoritative": storage_status_authoritative,
        "storage_hard_breaches": storage_hard_breaches,
        "stale_health_severe_ignored": bool(health_severe and storage_numeric_clear),
        "stale_ingestion_overload_ignored": bool(direct_overload and storage_numeric_clear),
        "storage_status_backpressure_ignored": bool(storage_status in {"blocked", "critical"} and storage_numeric_clear),
        "storage_severity_backpressure_ignored": bool(storage_severity in {"blocked", "critical"} and storage_numeric_clear),
        "storage_control_override_clear": bool(override_clear),
        "storage_control_override_embedded_age_seconds": round(float(embedded_override_age_seconds), 3),
        "storage_control_override_effective_age_seconds": round(float(effective_override_age_seconds), 3),
        "health_gates_age_minutes": round(float(health_gates_age_minutes), 3) if health_gates_age_minutes is not None else None,
        "health_gates_fresh_for_override": bool(health_gates_fresh_for_override),
        "stale_storage_control_override_ignored": bool(storage_control_override and not health_gates_fresh_for_override),
        "super_drainer_fresh": bool(super_drainer_fresh),
        "super_drainer_age_minutes": round(float(super_drainer_age_minutes), 3) if super_drainer_age_minutes is not None else None,
        "stale_super_drainer_ignored": bool(super_drainer and not super_drainer_fresh),
        "sources": _ordered_unique(sources),
    }


def _writer_cycle_state(writer_cycle: dict[str, Any]) -> dict[str, Any]:
    for key in ("writer_state_after_wait", "writer_state_after_remediation", "writer_state_before"):
        state = writer_cycle.get(key)
        if isinstance(state, dict) and state:
            return state
    return {}


def _completed_writer_handoff_needed(state: dict[str, Any], writer_cycle: dict[str, Any]) -> bool:
    summary = writer_cycle.get("summary") if isinstance(writer_cycle.get("summary"), dict) else {}
    current_step = str(state.get("current_step") or state.get("effective_current_step") or "").strip().lower()
    writer_status = str(state.get("status") or writer_cycle.get("overall_status") or "").strip().lower()
    running = bool(state.get("running", False))
    if "complete_lock_handoff_needed" in state:
        return bool(state.get("complete_lock_handoff_needed"))
    if bool(state.get("active_source") == "completed_lock_handoff_needed"):
        return True
    inferred_from_current_state = bool(
        current_step == "complete"
        and writer_status in {"ok", "complete", "idle"}
        and not running
        and bool(state.get("writer_lock_held", False))
        and not bool(state.get("child_writer_active", False))
    )
    if state:
        return inferred_from_current_state
    return bool(summary.get("completed_writer_lock_handoff_needed"))


def _build_pretraining_drain_buffer(
    *,
    project_root: Path,
    backpressure_gate: dict[str, Any],
    writer_cycle: dict[str, Any],
    backlog_accelerator: dict[str, Any],
) -> dict[str, Any]:
    state = _writer_cycle_state(writer_cycle)
    completed = _safe_int(state.get("completed_shard_count"), 0)
    planned = _safe_int(state.get("planned_shard_count"), 0)
    current_step = str(state.get("current_step") or "").strip().lower()
    writer_status = str(state.get("status") or writer_cycle.get("overall_status") or "").strip().lower()
    progress_age = _safe_float(state.get("progress_age_minutes"), 0.0)
    cycle_age = _safe_float(state.get("cycle_age_minutes"), 0.0)
    running = bool(state.get("running", False))
    handoff_needed = _completed_writer_handoff_needed(state, writer_cycle)
    active = bool(state.get("active", False) or (running and current_step not in {"", "complete"}))
    if current_step == "complete" and writer_status in {"ok", "complete", "idle"} and not running:
        active = False
    pending_lines = _safe_int(backpressure_gate.get("pending_lines"), 0)
    oldest_age = _safe_float(backpressure_gate.get("oldest_pending_age_seconds"), 0.0)
    pending_threshold = max(_safe_int(backpressure_gate.get("pending_lines_threshold"), 15_000), 1)
    age_threshold = max(_safe_float(backpressure_gate.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    warm_pending = min(1_000, max(int(pending_threshold * 0.2), 250))
    hold_pending = min(2_500, max(int(pending_threshold * 0.45), warm_pending + 1))
    warm_age = min(120.0, max(age_threshold * 0.5, 60.0))
    hold_age = min(300.0, max(age_threshold * 1.25, warm_age + 1.0))
    storage_numeric_clear = bool(backpressure_gate.get("storage_numeric_clear", False))
    pressure_index = _safe_float(backpressure_gate.get("pressure_index"), 0.0)
    if storage_numeric_clear and pressure_index < 0.75:
        # Once storage is genuinely green, do not keep training in a no-op
        # micro-drain loop just because a small live tail is above the very
        # conservative warm buffer.
        warm_pending = max(warm_pending, min(5_000, int(pending_threshold * 0.35)))
        warm_age = max(warm_age, min(180.0, age_threshold))
        hold_pending = max(hold_pending, warm_pending)
        hold_age = max(hold_age, warm_age)
    accel_score = _safe_int((backlog_accelerator.get("bulletproof_score") or {}).get("score"), 100)
    accel_status = str(backlog_accelerator.get("overall_status") or "").strip().lower()
    reasons: list[str] = []
    status = "clear"
    batch_cap = TRAINING_BATCH_MAX
    launch_blocker = ""
    recommended_command: list[str] = []
    safe_to_launch = True

    writer_command = [str(project_root / "scripts" / "ops" / "opsctl.sh"), "writer-cycle-coordinator", "--json"]
    writer_apply_command = [str(project_root / "scripts" / "ops" / "opsctl.sh"), "writer-cycle-coordinator", "--apply", "--skip-maintenance", "--json"]
    writer_handoff_command = [str(project_root / "scripts" / "ops" / "opsctl.sh"), "writer-cycle-coordinator", "--apply", "--handoff-only", "--json"]
    if bool(backpressure_gate.get("severe", False)):
        status = "blocked_by_backpressure_gate"
        safe_to_launch = False
        batch_cap = 0
        recommended_command = writer_handoff_command if handoff_needed else writer_command
        reasons.append("backpressure gate is already severe")
        if handoff_needed:
            reasons.append("completed writer lock handoff can be cleared with the fast handoff-only path")
    elif handoff_needed:
        status = "clear_completed_writer_handoff"
        safe_to_launch = False
        batch_cap = 0
        launch_blocker = "completed_writer_lock_handoff_pending"
        recommended_command = writer_handoff_command
        reasons.append("completed writer lock handoff should be cleared before training")
    elif active and progress_age >= 12.0 and (planned <= 0 or completed < planned) and current_step not in {"complete"}:
        status = "writer_attention_required"
        safe_to_launch = False
        batch_cap = 0
        launch_blocker = "writer_progress_stale_before_training"
        recommended_command = writer_apply_command
        reasons.append("active writer progress is stale before training launch")
    elif active and (pending_lines > hold_pending or oldest_age > hold_age):
        status = "hold_for_active_writer_catchup"
        safe_to_launch = False
        batch_cap = 0
        launch_blocker = "pretraining_drain_buffer_active"
        recommended_command = writer_command
        reasons.append("active writer should reduce warm backlog before training starts")
    elif (
        not active
        and storage_numeric_clear
        and pressure_index < 0.10
        and oldest_age <= 1.0
        and pending_lines <= pending_threshold
    ):
        status = "clear_low_pressure_live_tail"
        batch_cap = min(batch_cap, 2)
        reasons.append("pending tail is fresh and below threshold, so micro-drain is not required before a small canary")
    elif not active and (pending_lines > warm_pending or oldest_age > warm_age):
        status = "run_micro_drain_first"
        safe_to_launch = False
        batch_cap = 0
        launch_blocker = "pretraining_micro_drain_recommended"
        recommended_command = writer_handoff_command if handoff_needed else writer_apply_command
        reasons.append("one bounded writer pass should run before training")
    elif active and (pending_lines > warm_pending or oldest_age > warm_age):
        status = "trim_batch_during_writer_catchup"
        batch_cap = 2
        recommended_command = writer_command
        reasons.append("writer is healthy but still carrying warm pending work")
    elif active:
        status = "writer_active_backlog_green"
        recommended_command = writer_command
        reasons.append("writer is active, but backlog is inside the training buffer")

    if safe_to_launch and accel_score < 80:
        status = "accelerator_grade_caution" if status == "clear" else status
        batch_cap = min(batch_cap, 2)
        reasons.append("backlog P-core accelerator grade is below B")
    elif safe_to_launch and accel_status == "advisory" and status == "clear":
        reasons.append("backlog accelerator has advisory context but no launch blocker")

    return {
        "status": status,
        "safe_to_launch_now": bool(safe_to_launch and batch_cap > 0),
        "launch_blocker": launch_blocker,
        "batch_cap": int(max(batch_cap, 0)),
        "pending_lines": int(pending_lines),
        "oldest_pending_age_seconds": round(float(oldest_age), 3),
        "warm_pending_line_buffer": int(warm_pending),
        "hold_pending_line_buffer": int(hold_pending),
        "warm_oldest_age_seconds": round(float(warm_age), 3),
        "hold_oldest_age_seconds": round(float(hold_age), 3),
        "writer": {
            "active": bool(active),
            "running": bool(running),
            "completed_lock_handoff_needed": bool(handoff_needed),
            "status": writer_status,
            "current_step": current_step,
            "completed_shard_count": int(completed),
            "planned_shard_count": int(planned),
            "progress_age_minutes": round(float(progress_age), 3),
            "cycle_age_minutes": round(float(cycle_age), 3),
        },
        "accelerator": {
            "status": accel_status,
            "score": int(accel_score),
            "letter": str((backlog_accelerator.get("bulletproof_score") or {}).get("letter") or ""),
        },
        "recommended_command": recommended_command,
        "reasons": _ordered_unique(reasons),
    }


def _build_host_training_headroom_gate(
    *,
    project_root: Path,
    memory_intelligence: dict[str, Any],
    autonomic_governor: dict[str, Any],
) -> dict[str, Any]:
    classification = memory_intelligence.get("classification") if isinstance(memory_intelligence.get("classification"), dict) else {}
    reopen_gate = memory_intelligence.get("reopen_gate") if isinstance(memory_intelligence.get("reopen_gate"), dict) else {}
    multitasking = memory_intelligence.get("multitasking_headroom") if isinstance(memory_intelligence.get("multitasking_headroom"), dict) else {}
    budgets = autonomic_governor.get("budgets") if isinstance(autonomic_governor.get("budgets"), dict) else {}
    training_budget = budgets.get("training") if isinstance(budgets.get("training"), dict) else {}
    memory_status = str(classification.get("status") or "unknown").strip().lower()
    multitasking_level = str(multitasking.get("level") or "").strip().lower()
    memory_safe = bool(reopen_gate.get("safe_for_training", True)) if memory_intelligence else True
    small_canary_safe = bool(reopen_gate.get("small_canary_training_safe", False))
    small_batch_safe = bool(reopen_gate.get("small_batch_training_safe", False))
    batch10_safe = bool(reopen_gate.get("batch10_training_safe", False))
    batch20_safe = bool(reopen_gate.get("batch20_training_safe", False))
    batch20_execution_mode = str(reopen_gate.get("batch20_execution_mode") or "").strip()
    batch30_safe = bool(reopen_gate.get("batch30_training_safe", False))
    batch30_execution_mode = str(reopen_gate.get("batch30_execution_mode") or "").strip()
    memory_batch_cap = _safe_int(
        reopen_gate.get("training_batch_cap"),
        30
        if batch30_safe
        else 20
        if batch20_safe
        else 10
        if batch10_safe
        else 4
        if memory_safe
        else 2
        if small_batch_safe
        else 1
        if small_canary_safe
        else 0,
    )
    multitasking_blocks = bool(reopen_gate.get("training_blocked_by_multitasking", False))
    if memory_intelligence and multitasking.get("training_allowed_by_multitasking") is False and bool(multitasking.get("active", False)):
        multitasking_blocks = True
    reentry_gate = training_budget.get("reentry_gate") if isinstance(training_budget.get("reentry_gate"), dict) else {}
    reentry_gate_allows = bool(reentry_gate.get("allowed", False)) and not bool(reentry_gate.get("blockers"))
    watchdog_training_blocked = bool(training_budget.get("watchdog_training_blocked", False))
    governor_allows = bool(training_budget.get("allowed", reentry_gate_allows if training_budget else True)) if autonomic_governor else True
    if reentry_gate_allows and not watchdog_training_blocked:
        governor_allows = True
    governor_profile = str(training_budget.get("profile") or reentry_gate.get("profile") or "").strip()
    governor_micro_allows = bool(governor_allows and governor_profile == "coverage_micro_canary")
    governor_small_allows = bool(governor_allows and governor_profile == "coverage_small_canary")
    governor_batch10_allows = bool(governor_allows and governor_profile == "coverage_batch10_canary")
    governor_batch20_allows = bool(governor_allows and governor_profile == "coverage_batch20_canary")
    governor_batch30_allows = bool(governor_allows and governor_profile == "coverage_batch30_canary")
    governor_batch_cap = _safe_int(
        reentry_gate.get("max_parallel_trainings"),
        30
        if governor_batch30_allows
        else 20
        if governor_batch20_allows
        else 10
        if governor_batch10_allows
        else 4
        if governor_allows and governor_profile == "coverage_canary"
        else 2
        if governor_small_allows
        else 1
        if governor_micro_allows
        else 0
        if not governor_allows
        else 4,
    )
    blockers: list[str] = []
    reasons: list[str] = []
    batch_cap = TRAINING_BATCH_MAX
    if memory_status in {"hard_relief", "swap_relief"} or (memory_status == "compression_relief" and not (small_canary_safe or batch20_safe or batch30_safe)):
        blockers.append("host_memory_relief_active")
        reasons.append("memory pressure is in relief mode")
        batch_cap = 0
    elif batch30_safe:
        batch_cap = min(batch_cap, 30)
        if batch30_execution_mode == "sequential_memory_guarded_waves":
            reasons.append("memory pressure intelligence has cleared batch-30 as sequential memory-guarded waves")
        else:
            reasons.append("memory pressure intelligence has cleared the batch-30 training lane")
    elif batch20_safe:
        batch_cap = min(batch_cap, 20)
        if batch20_execution_mode == "sequential_memory_guarded_waves":
            reasons.append("memory pressure intelligence has cleared batch-20 as sequential memory-guarded waves")
        else:
            reasons.append("memory pressure intelligence has cleared the batch-20 training lane")
    elif batch10_safe:
        batch_cap = min(batch_cap, 10)
        reasons.append("memory pressure intelligence has cleared the batch-10 training lane")
    elif memory_status == "compression_relief" and small_canary_safe:
        batch_cap = min(batch_cap, 1)
        reasons.append("memory is still in compression relief, but the one-bot micro-canary lane is explicitly cleared")
    elif memory_safe:
        batch_cap = min(batch_cap, 4)
    elif not memory_safe and not (small_batch_safe or small_canary_safe):
        blockers.append("host_training_headroom_not_clear")
        reasons.append("memory pressure intelligence has not cleared training headroom")
        batch_cap = 0
    elif not memory_safe and small_batch_safe:
        batch_cap = min(batch_cap, 2)
        reasons.append("full training headroom is still soaking, but a two-bot small canary is memory-safe")
    elif not memory_safe and small_canary_safe:
        batch_cap = min(batch_cap, 1)
        reasons.append("full training headroom is still soaking, but a one-bot micro-canary is memory-safe")
    batch_cap = min(batch_cap, max(memory_batch_cap, 0))
    if multitasking_blocks:
        blockers.append("host_multitasking_reserve_active")
        reasons.append("foreground app reserve is active")
        batch_cap = 0
    if not governor_allows:
        blockers.append("autonomic_training_budget_closed")
        reasons.append("autonomic governor training budget is closed")
        batch_cap = 0
    elif governor_batch30_allows:
        batch_cap = min(batch_cap, 30, max(governor_batch_cap, 0))
        reasons.append("autonomic governor is allowing the batch-30 canary lane")
    elif governor_batch20_allows:
        batch_cap = min(batch_cap, 20, max(governor_batch_cap, 0))
        reasons.append("autonomic governor is allowing the batch-20 canary lane")
    elif governor_batch10_allows:
        batch_cap = min(batch_cap, 10, max(governor_batch_cap, 0))
        reasons.append("autonomic governor is allowing the batch-10 canary lane")
    elif governor_small_allows:
        batch_cap = min(batch_cap, 2, max(governor_batch_cap, 0))
        reasons.append("autonomic governor is allowing the two-bot small canary lane")
    elif governor_micro_allows:
        batch_cap = min(batch_cap, 1)
        reasons.append("autonomic governor is only allowing the micro-canary lane")
    elif governor_allows:
        batch_cap = min(batch_cap, 4, max(governor_batch_cap, 1))
    status = "ready" if not blockers else "blocked"
    selected_profile = "none"
    if status == "ready" and batch_cap > 0:
        if batch_cap >= 30 and governor_batch30_allows:
            selected_profile = "coverage_batch30_canary"
        elif batch_cap >= 20 and (governor_batch20_allows or governor_batch30_allows):
            selected_profile = "coverage_batch20_canary"
        elif batch_cap >= 10 and (governor_batch10_allows or governor_batch20_allows or governor_batch30_allows):
            selected_profile = "coverage_batch10_canary"
        elif batch_cap >= 4:
            selected_profile = "coverage_canary"
        elif batch_cap >= 2:
            selected_profile = "coverage_small_canary"
        else:
            selected_profile = "coverage_micro_canary"
    command: list[str] = []
    if blockers:
        command = [str(project_root / "scripts" / "ops" / "opsctl.sh"), "memory-pressure-intelligence", "--apply", "--json"]
    reentry_stages = [
        {
            "stage": "micro_canary",
            "profile": "coverage_micro_canary",
            "max_parallel_trainings": 1,
            "memory_safe": small_canary_safe,
            "governor_allows": governor_micro_allows or governor_small_allows or governor_batch10_allows or governor_batch20_allows or governor_batch30_allows,
            "required_clear_samples": 1,
        },
        {
            "stage": "small_canary",
            "profile": "coverage_small_canary",
            "max_parallel_trainings": 2,
            "memory_safe": small_batch_safe,
            "governor_allows": governor_small_allows or governor_batch10_allows or governor_batch20_allows or governor_batch30_allows,
            "required_clear_samples": 2,
        },
        {
            "stage": "batch10_canary",
            "profile": "coverage_batch10_canary",
            "max_parallel_trainings": 10,
            "memory_safe": batch10_safe,
            "governor_allows": governor_batch10_allows or governor_batch20_allows or governor_batch30_allows,
            "required_clear_samples": 3,
        },
        {
            "stage": "batch20_canary",
            "profile": "coverage_batch20_canary",
            "max_parallel_trainings": 20,
            "memory_safe": batch20_safe,
            "governor_allows": governor_batch20_allows or governor_batch30_allows,
            "required_clear_samples": 4,
            "execution_mode": batch20_execution_mode,
            "wave_size": _safe_int(reopen_gate.get("batch20_wave_size"), 0),
            "between_wave_memory_recheck": bool(reopen_gate.get("batch20_requires_between_target_memory_recheck", False)),
        },
        {
            "stage": "batch30_canary",
            "profile": "coverage_batch30_canary",
            "max_parallel_trainings": 30,
            "memory_safe": batch30_safe,
            "governor_allows": governor_batch30_allows,
            "required_clear_samples": 4,
            "execution_mode": batch30_execution_mode,
            "wave_size": _safe_int(reopen_gate.get("batch30_wave_size"), 0),
            "between_wave_memory_recheck": bool(reopen_gate.get("batch30_requires_between_target_memory_recheck", False)),
        },
    ]
    for stage in reentry_stages:
        stage["allowed_now"] = bool(
            status == "ready"
            and batch_cap >= _safe_int(stage.get("max_parallel_trainings"), 0)
            and bool(stage.get("memory_safe", False))
            and bool(stage.get("governor_allows", False))
        )
    next_stage = next(
        (
            stage
            for stage in reentry_stages
            if bool(stage.get("memory_safe", False)) and bool(stage.get("governor_allows", False))
        ),
        reentry_stages[0],
    )
    return {
        "status": status,
        "safe_for_training": not blockers,
        "launch_blockers": _ordered_unique(blockers),
        "batch_cap": int(batch_cap),
        "memory_status": memory_status,
        "memory_decision": str(classification.get("decision") or ""),
        "recommended_p_core_worker_cap": _safe_int(classification.get("recommended_p_core_worker_cap"), 0),
        "multitasking_level": multitasking_level,
        "open_apps": list(multitasking.get("open_apps") or [])[:10],
        "training_blocked_by_multitasking": bool(multitasking_blocks),
        "governor_training_allowed": bool(governor_allows),
        "governor_reentry_gate_allowed": bool(reentry_gate_allows),
        "governor_watchdog_training_blocked": bool(watchdog_training_blocked),
        "governor_profile": governor_profile,
        "selected_training_profile": selected_profile,
        "small_canary_training_safe": small_canary_safe,
        "small_batch_training_safe": small_batch_safe,
        "batch10_training_safe": batch10_safe,
        "batch20_training_safe": batch20_safe,
        "batch20_execution_mode": batch20_execution_mode,
        "batch20_wave_size": _safe_int(reopen_gate.get("batch20_wave_size"), 0),
        "batch20_requires_between_target_memory_recheck": bool(reopen_gate.get("batch20_requires_between_target_memory_recheck", False)),
        "batch30_training_safe": batch30_safe,
        "batch30_execution_mode": batch30_execution_mode,
        "batch30_wave_size": _safe_int(reopen_gate.get("batch30_wave_size"), 0),
        "batch30_requires_between_target_memory_recheck": bool(reopen_gate.get("batch30_requires_between_target_memory_recheck", False)),
        "memory_training_batch_cap": int(memory_batch_cap),
        "governor_batch_cap": int(governor_batch_cap),
        "runtime_clear_reentry_ladder": reentry_stages,
        "next_reentry_stage": next_stage,
        "recommended_command": command,
        "reasons": _ordered_unique(reasons),
    }


def _build_resource_guard_training_gate(project_root: Path, resource_guard: dict[str, Any]) -> dict[str, Any]:
    raw_ok = bool(resource_guard.get("resource_guard_ok", resource_guard.get("ok", True)))
    memory_state = str(resource_guard.get("memory_pressure_state") or "unknown").strip().lower()
    reasons = [str(item).strip() for item in resource_guard.get("resource_guard_reasons") or [] if str(item).strip()]
    advisory_freeze = bool((not raw_ok) and memory_state == "green" and _training_advisory_resource_guard_only(reasons))
    training_ok = bool(raw_ok or advisory_freeze)
    blockers: list[str] = []
    if not training_ok or memory_state not in {"green", "unknown"}:
        blockers.append("resource_guard_not_green")
    command = []
    if blockers:
        command = [
            str(resolve_runtime_python(project_root)),
            str(project_root / "scripts" / "resource_guard.py"),
            "--profile",
            "refresh",
            "--json",
        ]
    return {
        "status": "ready" if not blockers else "blocked",
        "ok": training_ok,
        "raw_ok": raw_ok,
        "training_ok": training_ok,
        "advisory_only": advisory_freeze,
        "launch_blockers": blockers,
        "memory_pressure_state": memory_state,
        "memory_pressure_kind": str(resource_guard.get("memory_pressure_kind") or ""),
        "swap_used_gb": round(_safe_float(resource_guard.get("swap_used_gb"), 0.0), 3),
        "creative_session_level": str(resource_guard.get("creative_session_level") or ""),
        "reasons": _ordered_unique(reasons),
        "recommended_command": command,
    }


def _build_storage_quota_training_gate(project_root: Path, storage_quota: dict[str, Any]) -> dict[str, Any]:
    summary = storage_quota.get("quota_summary") if isinstance(storage_quota.get("quota_summary"), dict) else {}
    hard_breaches = _safe_int(summary.get("hard_breaches"), 0)
    soft_breaches = _safe_int(summary.get("soft_breaches"), 0)
    blocked_families = [str(item) for item in summary.get("blocked_families") or [] if str(item or "").strip()]
    degraded_families = [str(item) for item in summary.get("degraded_families") or [] if str(item or "").strip()]
    blockers: list[str] = []
    reasons: list[str] = []
    if hard_breaches > 0:
        blockers.append("storage_quota_hard_breach")
        reasons.append("storage quota guard has hard-breached lanes; heavy training stays gated until quota is below hard limits")
    elif soft_breaches > 0:
        reasons.append("storage quota guard has soft-breached lanes; keep training canary-sized")
    command = [str(project_root / "scripts" / "ops" / "opsctl.sh"), "storage-quota-guard", "--json"] if blockers else []
    return {
        "status": "ready" if not blockers else "blocked",
        "launch_blockers": blockers,
        "hard_breaches": hard_breaches,
        "soft_breaches": soft_breaches,
        "blocked_families": blocked_families,
        "degraded_families": degraded_families,
        "worst_over_hard_gb": round(_safe_float(summary.get("worst_over_hard_gb"), 0.0), 3),
        "worst_hard_ratio": round(_safe_float(summary.get("worst_hard_ratio"), 0.0), 3),
        "recommended_actions": [str(item) for item in storage_quota.get("recommended_actions") or [] if str(item or "").strip()][:6],
        "recommended_command": command,
        "reasons": _ordered_unique(reasons),
    }


def _build_training_launch_contract(
    *,
    project_root: Path,
    snapshot_fresh: bool,
    resource_guard_ok: bool,
    memory_pressure_state: str,
    resource_guard_gate: dict[str, Any],
    storage_quota_gate: dict[str, Any],
    parity_state: str,
    mlx_failure_active: bool,
    backpressure_gate: dict[str, Any],
    pretraining_drain_buffer: dict[str, Any],
    host_headroom_gate: dict[str, Any],
    training_quality_blocked: bool,
    training_quality_score: float,
    precompute_targets: list[dict[str, Any]],
    candidate_selector: dict[str, Any],
    fresh_minutes: int,
    batch_limit: int,
    training_evidence_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backpressure_severe = bool(backpressure_gate.get("severe", False))
    backpressure_cooling_down = bool(backpressure_gate.get("cooling_down", False))
    timeout_fallback = _select_timeout_fallback(
        [
            _recent_small_canary_timeout(
                _load_json(project_root / "governance" / "health" / "training_drain_autopilot_latest.json")
            ),
            _recent_retrain_launch_timeout(
                _load_json(project_root / "governance" / "health" / "retrain_launch_latest.json")
            ),
        ]
    )
    repair_first = [_public_target(row) for row in precompute_targets if str(row.get("training_stage") or "") == "repair_first"]
    unfiltered_canary_pool = [
        _public_target(row)
        for row in precompute_targets
        if str(row.get("training_stage") or "") != "repair_first"
    ]
    selector_active = bool(candidate_selector.get("active", False))
    selector_authoritative = bool(candidate_selector.get("authoritative", False))
    if selector_active:
        canary_pool = [
            row
            for row in unfiltered_canary_pool
            if selector_authoritative and bool(row.get("bot_needs_can_train_now", False))
        ]
        eligibility_blocked_targets = [
            row
            for row in unfiltered_canary_pool
            if not bool(row.get("bot_needs_can_train_now", False))
        ]
    else:
        canary_pool = unfiltered_canary_pool
        eligibility_blocked_targets = []
    requested_batch = min(max(int(batch_limit), 1), TRAINING_BATCH_MAX)
    selected_profile = str(
        host_headroom_gate.get("selected_training_profile")
        or host_headroom_gate.get("governor_profile")
        or ""
    ).strip()
    recovery_pool = canary_pool if selector_active else (canary_pool if canary_pool else repair_first)
    recovery_min_pool = 1 if selected_profile in {"coverage_micro_canary", "coverage_small_canary"} else min(requested_batch, 10)
    quality_recovery_canary = bool(
        training_quality_blocked
        and selected_profile in TRAINING_BATCH_PROFILES
        and _safe_float(training_quality_score, 0.0) >= 50.0
        and len(recovery_pool) >= recovery_min_pool
    )

    launch_blockers: list[str] = []
    prep_blockers: list[str] = []
    if not snapshot_fresh:
        launch_blockers.append("runtime_snapshot_not_fresh")
        prep_blockers.append("runtime_snapshot_not_fresh")
    if selector_active and not selector_authoritative:
        launch_blockers.append("training_candidate_selector_not_fresh")
    elif selector_authoritative and _safe_int(candidate_selector.get("selected_count"), 0) <= 0:
        launch_blockers.append("no_bot_needs_training_candidates")
    elif selector_authoritative and not canary_pool:
        launch_blockers.append("bot_needs_training_candidates_not_runtime_ready")
    if not resource_guard_ok or memory_pressure_state not in {"green", "unknown"}:
        launch_blockers.append("resource_guard_not_green")
        prep_blockers.append("resource_guard_not_green")
    for blocker in storage_quota_gate.get("launch_blockers") or []:
        text = str(blocker or "").strip()
        if text:
            launch_blockers.append(text)
    if backpressure_severe:
        launch_blockers.append("backpressure_cooling_down" if backpressure_cooling_down else "backpressure_overload_severe")
    drain_blocker = str(pretraining_drain_buffer.get("launch_blocker") or "").strip()
    if drain_blocker:
        launch_blockers.append(drain_blocker)
    for blocker in host_headroom_gate.get("launch_blockers") or []:
        text = str(blocker or "").strip()
        if text:
            launch_blockers.append(text)
    if training_quality_blocked and not quality_recovery_canary:
        launch_blockers.append("training_quality_blocked")
    if parity_state in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"}:
        launch_blockers.append(f"runtime_backend_{parity_state}")
        prep_blockers.append(f"runtime_backend_{parity_state}")
    if mlx_failure_active:
        launch_blockers.append("mlx_failure_active")
        prep_blockers.append("mlx_failure_active")
    evidence_gate = training_evidence_gate or {"active": False, "ready": True, "blockers": []}
    for blocker in evidence_gate.get("blockers") or []:
        text = str(blocker or "").strip()
        if text:
            launch_blockers.append(text)

    drain_batch_cap = min(max(_safe_int(pretraining_drain_buffer.get("batch_cap"), TRAINING_BATCH_MAX), 0), TRAINING_BATCH_MAX)
    host_batch_cap = min(max(_safe_int(host_headroom_gate.get("batch_cap"), 4), 0), TRAINING_BATCH_MAX)
    launch_pool = recovery_pool if quality_recovery_canary and not canary_pool else canary_pool
    batch_size = min(requested_batch, drain_batch_cap, host_batch_cap, len(launch_pool))
    original_selected_profile = selected_profile
    original_batch_size = batch_size
    timeout_fallback_active = bool(
        timeout_fallback.get("active", False)
        and batch_size > 1
        and selected_profile in TRAINING_BATCH_PROFILES
    )
    if timeout_fallback_active:
        selected_profile = "coverage_micro_canary"
        batch_size = 1
    canary_batch = launch_pool[:batch_size]
    prep_allowed = bool(snapshot_fresh and resource_guard_ok and parity_state not in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"} and not mlx_failure_active and precompute_targets)
    launch_allowed = bool(not launch_blockers and canary_batch)
    mode = "canary_training_allowed" if launch_allowed else ("prep_only" if prep_allowed else "blocked")

    retrain_command: list[str] = []
    if launch_allowed:
        profile = selected_profile
        if profile not in TRAINING_BATCH_PROFILES:
            profile = "coverage_canary"
        retrain_command = [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "retrain-force-targeted",
            "--include-bot-ids",
            ",".join(row["bot_id"] for row in canary_batch),
            "--retrain-profile",
            profile,
            "--skip-master-update",
        ]

    drain_command = pretraining_drain_buffer.get("recommended_command") if isinstance(pretraining_drain_buffer.get("recommended_command"), list) else []
    host_command = host_headroom_gate.get("recommended_command") if isinstance(host_headroom_gate.get("recommended_command"), list) else []
    resource_command = resource_guard_gate.get("recommended_command") if isinstance(resource_guard_gate.get("recommended_command"), list) else []
    quota_command = storage_quota_gate.get("recommended_command") if isinstance(storage_quota_gate.get("recommended_command"), list) else []
    recommended_prep_commands = []
    if drain_command:
        recommended_prep_commands.append(list(drain_command))
    if host_command and host_command not in recommended_prep_commands:
        recommended_prep_commands.append(list(host_command))
    if resource_command and resource_command not in recommended_prep_commands:
        recommended_prep_commands.append(list(resource_command))
    if quota_command and quota_command not in recommended_prep_commands:
        recommended_prep_commands.append(list(quota_command))
    recommended_prep_commands.extend(
        [
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "runtime-training-snapshot",
                "--reuse-if-fresh-minutes",
                str(max(int(fresh_minutes), 1)),
                "--json",
            ],
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "training-runtime-control",
                "--json",
            ],
        ]
    )

    return {
        "mode": mode,
        "launch_allowed": launch_allowed,
        "prep_allowed": prep_allowed,
        "launch_blockers": _ordered_unique(launch_blockers),
        "prep_blockers": _ordered_unique(prep_blockers),
        "backpressure_gate": {
            "severe": bool(backpressure_severe),
            "cooling_down": backpressure_cooling_down,
            "pending_lines": _safe_int(backpressure_gate.get("pending_lines"), 0),
            "oldest_pending_age_seconds": round(_safe_float(backpressure_gate.get("oldest_pending_age_seconds"), 0.0), 3),
            "pending_lines_threshold": _safe_int(backpressure_gate.get("pending_lines_threshold"), 0),
            "oldest_age_threshold_seconds": round(_safe_float(backpressure_gate.get("oldest_age_threshold_seconds"), 0.0), 3),
            "storage_status": str(backpressure_gate.get("storage_status") or ""),
            "storage_severity": str(backpressure_gate.get("storage_severity") or ""),
            "pressure_index": round(_safe_float(backpressure_gate.get("pressure_index"), 0.0), 6),
            "sql_progress_status": str(backpressure_gate.get("sql_progress_status") or ""),
            "sources": list(backpressure_gate.get("sources") or []),
        },
        "pretraining_drain_buffer": pretraining_drain_buffer,
        "host_training_headroom_gate": host_headroom_gate,
        "resource_guard_gate": resource_guard_gate,
        "storage_quota_training_gate": storage_quota_gate,
        "training_quality_recovery_canary": quality_recovery_canary,
        "training_quality_score": round(_safe_float(training_quality_score, 0.0), 3),
        "canary_batch": canary_batch,
        "repair_first_targets": repair_first[: max(int(batch_limit), 1)],
        "prep_targets": [_public_target(row) for row in precompute_targets[: max(int(batch_limit), 1)]],
        "requested_batch_size": int(requested_batch),
        "available_canary_pool_size": len(canary_pool),
        "unfiltered_canary_pool_size": len(unfiltered_canary_pool),
        "eligibility_blocked_target_count": len(eligibility_blocked_targets),
        "eligibility_blocked_targets": eligibility_blocked_targets[: max(int(batch_limit), 1)],
        "training_candidate_selector": candidate_selector,
        "training_evidence_gate": evidence_gate,
        "available_repair_first_pool_size": len(repair_first),
        "effective_launch_pool_size": len(launch_pool),
        "drain_batch_cap": int(drain_batch_cap),
        "host_batch_cap": int(host_batch_cap),
        "recommended_batch_size": len(canary_batch) if launch_allowed else 0,
        "recommended_retrain_profile": selected_profile if launch_allowed else "",
        "timeout_fallback": {
            **timeout_fallback,
            "applied": timeout_fallback_active,
            "original_recommended_profile": original_selected_profile,
            "original_recommended_batch_size": int(original_batch_size),
            "fallback_profile": "coverage_micro_canary" if timeout_fallback_active else "",
            "fallback_batch_size": 1 if timeout_fallback_active else 0,
            "reason": "recent_coverage_canary_timeout" if timeout_fallback_active else "",
        },
        "recommended_retrain_command": retrain_command,
        "recommended_prep_commands": recommended_prep_commands,
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, fresh_minutes: int = 360, limit: int = 8) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    snapshot = _load_json(health_root / "runtime_training_snapshot_latest.json")
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    retrain_scorecard = _load_json(health_root / "retrain_scorecard_latest.json")
    training_success = _load_json(health_root / "training_success_latest.json")
    resource_guard = _load_json(health_root / "resource_guard_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")
    storage_control = _load_json(health_root / "ingestion_storage_control_latest.json")
    storage_quota = _load_json(health_root / "storage_quota_guard_latest.json")
    ingestion_backpressure = _load_json(health_root / "ingestion_backpressure_latest.json")
    super_drainer = _load_json(health_root / "backpressure_super_drainer_latest.json")
    writer_cycle = _load_json(health_root / "writer_cycle_coordinator_latest.json")
    backlog_accelerator = _load_json(health_root / "backlog_pcore_accelerator_latest.json")
    memory_intelligence = _load_json(health_root / "memory_pressure_intelligence_latest.json")
    autonomic_governor = _load_json(health_root / "autonomic_resource_governor_latest.json")
    training_requalification = _load_json(health_root / "training_requalification_latest.json")
    candidate_advancement = _load_json(health_root / "promotion_candidate_advancement_latest.json")
    bot_needs = _load_json(health_root / "bot_needs_intelligence_latest.json")
    training_stage_board = (
        bot_needs.get("training_stage_board") if isinstance(bot_needs.get("training_stage_board"), dict) else {}
    )
    training_stage_counts = (
        training_stage_board.get("counts") if isinstance(training_stage_board.get("counts"), dict) else {}
    )
    collection_rollup = _load_json(health_root / "data_collection_observation_rollup_latest.json")
    coverage_seed = _load_json(walk_root / "coverage_seed_latest.json")
    coverage_gap_closer = _load_json(walk_root / "coverage_gap_closer_latest.json")
    runtime_probe = _runtime_backend_probe(project_root)

    snapshot_age_minutes = _age_minutes(snapshot.get("timestamp_utc"))
    snapshot_content_fresh = bool(
        snapshot.get("content_fresh", True)
        if _safe_int(snapshot.get("schema_version"), 1) >= 2
        else True
    )
    snapshot_fresh = bool(
        snapshot
        and _safe_int(snapshot.get("sequence_count"), 0) > 0
        and _safe_int(snapshot.get("row_count"), 0) > 0
        and snapshot_age_minutes is not None
        and snapshot_age_minutes <= max(int(fresh_minutes), 1)
        and snapshot_content_fresh
    )
    sequence_count = _safe_int(snapshot.get("sequence_count"), 0)
    row_count = _safe_int(snapshot.get("row_count"), 0)
    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    training_candidate_selector = _training_candidate_selector_contract(
        bot_needs,
        max_age_minutes=max(int(fresh_minutes), 1),
    )
    training_candidate_selector = _apply_training_target_cooldown(
        training_candidate_selector,
        retrain_scorecard,
        cooldown_minutes=max(
            _safe_int(
                os.getenv("BOT_TRAINING_TARGET_COOLDOWN_MINUTES"),
                TRAINING_TARGET_COOLDOWN_MINUTES,
            ),
            0,
        ),
    )
    training_evidence_gate = _build_training_evidence_gate(
        project_root,
        max_age_minutes=max(int(fresh_minutes), 1),
    )
    precompute_targets = _build_precompute_targets(
        training_quality=training_quality,
        retrain_scorecard=retrain_scorecard,
        coverage_seed=coverage_seed,
        coverage_gap_closer=coverage_gap_closer,
        training_requalification=training_requalification,
        candidate_advancement=candidate_advancement,
        bot_needs=bot_needs,
        candidate_selector=training_candidate_selector,
    )
    resource_guard_raw_ok = bool(resource_guard.get("resource_guard_ok", True))
    memory_pressure_state = str(resource_guard.get("memory_pressure_state") or "").strip().lower() or "unknown"
    resource_guard_gate = _build_resource_guard_training_gate(project_root, resource_guard)
    resource_guard_training_ok = bool(resource_guard_gate.get("training_ok", resource_guard_raw_ok))
    storage_quota_gate = _build_storage_quota_training_gate(project_root, storage_quota)
    operating_mode = str(health_gates.get("recommended_operating_mode") or "").strip() or "unknown"
    backpressure_gate = _build_backpressure_training_gate(
        health_gates=health_gates,
        storage_control=storage_control,
        ingestion_backpressure=ingestion_backpressure,
        super_drainer=super_drainer,
    )
    pretraining_drain_buffer = _build_pretraining_drain_buffer(
        project_root=project_root,
        backpressure_gate=backpressure_gate,
        writer_cycle=writer_cycle,
        backlog_accelerator=backlog_accelerator,
    )
    host_headroom_gate = _build_host_training_headroom_gate(
        project_root=project_root,
        memory_intelligence=memory_intelligence,
        autonomic_governor=autonomic_governor,
    )
    backpressure_severe = bool(backpressure_gate.get("severe", False))

    parity_state = str(runtime_probe.get("parity_state") or "")
    training_failure_details = training_success.get("failure_details") if isinstance(training_success.get("failure_details"), list) else []
    training_quality_blocked = str(training_quality.get("overall_status") or "").strip().lower() == "blocked"
    mlx_failure_detected = any(
        "no module named 'mlx'" in " ".join(
            str((row or {}).get(field) or "")
            for field in ("reason", "stdout_tail", "stderr_tail")
        ).lower()
        for row in training_failure_details
        if isinstance(row, dict)
    )
    mlx_runtime_available = bool(((runtime_probe.get("installed_backends") or {}).get("mlx", False)))
    mlx_failure_active = bool(mlx_failure_detected and not mlx_runtime_available)
    core_runtime_ready = bool(
        snapshot_fresh
        and resource_guard_training_ok
        and parity_state not in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"}
        and not mlx_failure_active
    )
    coverage_repair_ready = bool(
        core_runtime_ready
        and (
            _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0) > 0
            or len(precompute_targets) > 0
        )
    )
    training_launch_contract = _build_training_launch_contract(
        project_root=project_root,
        snapshot_fresh=snapshot_fresh,
        resource_guard_ok=resource_guard_training_ok,
        memory_pressure_state=memory_pressure_state,
        resource_guard_gate=resource_guard_gate,
        storage_quota_gate=storage_quota_gate,
        parity_state=parity_state,
        mlx_failure_active=mlx_failure_active,
        backpressure_gate=backpressure_gate,
        pretraining_drain_buffer=pretraining_drain_buffer,
        host_headroom_gate=host_headroom_gate,
        training_quality_blocked=training_quality_blocked,
        training_quality_score=_safe_float(training_quality.get("training_quality_score"), 0.0),
        precompute_targets=precompute_targets,
        candidate_selector=training_candidate_selector,
        fresh_minutes=fresh_minutes,
        batch_limit=limit,
        training_evidence_gate=training_evidence_gate,
    )

    training_launch_allowed = bool(training_launch_contract.get("launch_allowed", False))
    training_prep_allowed = bool(training_launch_contract.get("prep_allowed", False))
    training_launch_blockers = [
        str(item)
        for item in (training_launch_contract.get("launch_blockers") if isinstance(training_launch_contract.get("launch_blockers"), list) else [])
        if str(item or "").strip()
    ]

    overall_status = "ready"
    if not snapshot_fresh or not resource_guard_training_ok:
        overall_status = "constrained"
    if backpressure_severe:
        overall_status = "blocked"
    elif parity_state in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"} or mlx_failure_active:
        overall_status = "blocked"
    elif training_launch_allowed and not training_launch_blockers:
        overall_status = "ready"
    elif training_quality_blocked:
        overall_status = "degraded" if coverage_repair_ready else "blocked"
    elif not bool(host_headroom_gate.get("safe_for_training", True)):
        overall_status = "constrained"
    if bool(training_candidate_selector.get("active", False)) and not bool(training_candidate_selector.get("authoritative", False)):
        overall_status = "blocked"
    if bool(training_evidence_gate.get("active", False)) and not bool(training_evidence_gate.get("ready", False)):
        overall_status = "blocked"
    elif (
        bool(training_candidate_selector.get("authoritative", False))
        and _safe_int(training_candidate_selector.get("selected_count"), 0) > 0
        and "bot_needs_training_candidates_not_runtime_ready" in training_launch_blockers
        and overall_status == "ready"
    ):
        overall_status = "constrained"

    controlled_idle_no_candidates = bool(
        overall_status == "constrained"
        and training_prep_allowed
        and bool(training_candidate_selector.get("active", False))
        and bool(training_candidate_selector.get("fresh", False))
        and bool(training_candidate_selector.get("authoritative", False))
        and _safe_int(training_candidate_selector.get("selected_count"), 0) == 0
        and "no_bot_needs_training_candidates" in training_launch_blockers
        and set(training_launch_blockers).issubset(
            {"no_bot_needs_training_candidates", "autonomic_training_budget_closed"}
        )
        and bool(training_evidence_gate.get("ready", False))
        and snapshot_fresh
        and resource_guard_training_ok
        and not backpressure_severe
        and parity_state not in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"}
        and not mlx_failure_active
    )
    operational_status = "ready_idle" if controlled_idle_no_candidates else overall_status
    operational_ok = bool(overall_status == "ready" or controlled_idle_no_candidates)

    recommended_actions: list[str] = []
    if not snapshot_fresh:
        recommended_actions.append("refresh the shared runtime training snapshot before retrying targeted retrains")
    if training_evidence_gate.get("blockers"):
        recommended_actions.append(
            "refresh the ordered training evidence chain before launching a canary: "
            + ", ".join(str(item) for item in training_evidence_gate.get("blockers") or [])
        )
    selector_status = str(training_candidate_selector.get("status") or "")
    if selector_status == "stale":
        recommended_actions.append("refresh bot-needs intelligence before launching any training candidate")
    elif selector_status == "cooldown":
        recommended_actions.append("keep recently trained targets idle until their per-bot evidence cooldown expires")
    elif bool(training_candidate_selector.get("authoritative", False)) and _safe_int(training_candidate_selector.get("selected_count"), 0) <= 0:
        recommended_actions.append("keep training idle because the fresh bot-needs selector has no eligible candidates")
    elif "bot_needs_training_candidates_not_runtime_ready" in training_launch_blockers:
        recommended_actions.append("repair the selector-approved candidates until they also clear runtime launch requirements")
    if any("loading_sequences_timeout" in list(row.get("reasons") or []) for row in precompute_targets[: max(int(limit), 1)]):
        recommended_actions.append("precompute or reuse shared sequence caches for bots that timed out in loading_sequences")
    if not resource_guard_training_ok or memory_pressure_state not in {"green", "unknown"}:
        recommended_actions.append("wait for green memory pressure before forcing targeted retrains that expand sequence windows")
    if resource_guard_gate.get("launch_blockers") and resource_guard_gate.get("reasons"):
        recommended_actions.append("clear the resource guard reason before training: " + "; ".join(list(resource_guard_gate.get("reasons") or [])[:3]))
    elif bool(resource_guard_gate.get("advisory_only", False)):
        recommended_actions.append("resource guard is advisory-only for guarded training while memory stays green")
    if storage_quota_gate.get("launch_blockers"):
        recommended_actions.append(
            "clear storage quota hard breaches before launching batch training: "
            + ", ".join(list(storage_quota_gate.get("blocked_families") or [])[:4])
        )
    if backpressure_severe and bool(backpressure_gate.get("cooling_down", False)):
        recommended_actions.append("keep retrain workers parked while the backpressure gate cools below the training threshold")
    elif backpressure_severe:
        recommended_actions.append("treat retrain workers as background-only until ingestion backpressure exits the severe state")
    drain_buffer_status = str(pretraining_drain_buffer.get("status") or "")
    if drain_buffer_status == "clear_completed_writer_handoff":
        recommended_actions.append("clear the completed writer handoff with the fast handoff-only coordinator path before training")
    elif drain_buffer_status == "writer_attention_required":
        recommended_actions.append("run the writer-cycle coordinator recovery check before training because writer progress is stale")
    elif drain_buffer_status == "hold_for_active_writer_catchup":
        recommended_actions.append("let the active writer reduce warm pending work before starting the coverage canary")
    elif drain_buffer_status == "run_micro_drain_first":
        recommended_actions.append("run one bounded P-core writer catch-up pass before launching the coverage canary")
    elif drain_buffer_status == "trim_batch_during_writer_catchup":
        recommended_actions.append("trim the canary batch while the active writer catches up")
    elif drain_buffer_status == "writer_active_backlog_green":
        recommended_actions.append("the writer is active but inside the training buffer; training can stay canary-sized")
    if not bool(host_headroom_gate.get("safe_for_training", True)):
        if bool(host_headroom_gate.get("training_blocked_by_multitasking", False)):
            recommended_actions.append("hold training until foreground app reserve clears or the operator explicitly wants to trade responsiveness for training")
        else:
            recommended_actions.append("refresh memory-pressure intelligence and wait for training headroom before launching canary retrains")
    if training_launch_contract["mode"] == "prep_only":
        recommended_actions.append("keep training in prep-only mode until the launch blockers clear")
    elif training_launch_contract["mode"] == "canary_training_allowed":
        timeout_fallback = training_launch_contract.get("timeout_fallback") if isinstance(training_launch_contract.get("timeout_fallback"), dict) else {}
        if bool(timeout_fallback.get("applied", False)):
            recommended_actions.append("retry the timed-out coverage canary as a one-bot coverage_micro_canary before widening training again")
        elif bool(training_launch_contract.get("training_quality_recovery_canary", False)):
            recommended_actions.append("run only the guarded recovery canary with --skip-master-update while quality control is blocked")
        else:
            recommended_actions.append("run only the recommended coverage_canary batch before widening training")
    if _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0) > 0:
        recommended_actions.append("reuse the shared snapshot when seeding walk-forward coverage so promotion coverage improves without rebuilding runtime inputs")
    if parity_state in {"missing_runtime_python", "runtime_probe_failed"}:
        recommended_actions.append("repair the runtime python selection before retrying MLX-backed retrains")
    elif parity_state == "native_backend_missing" or mlx_failure_active:
        recommended_actions.append("install or repair MLX inside the runtime interpreter so native retrains stop failing before model code loads")
    elif parity_state == "portable_only":
        recommended_actions.append("keep non-MLX backends in replay and sidecar duty until the native runtime regains MLX support")

    retry_pack = retrain_scorecard.get("retry_pack") if isinstance(retrain_scorecard.get("retry_pack"), dict) else {}
    snapshot_coverage = snapshot.get("coverage") if isinstance(snapshot.get("coverage"), dict) else {}
    repair_contract = {
        "parity_state": parity_state,
        "runtime_python_path": str(runtime_probe.get("runtime_python_path") or ""),
        "runtime_matches_current": bool(runtime_probe.get("runtime_matches_current", False)),
        "probe_rc": int(runtime_probe.get("probe_rc", 0) or 0),
        "verify_runtime_command": [
            str(runtime_probe.get("runtime_python_path") or ""),
            "-c",
            "import mlx, sys; print(sys.executable)",
        ]
        if str(runtime_probe.get("runtime_python_path") or "")
        else [],
        "retry_pack_command": list(retry_pack.get("command") or []),
        "portable_contract_roles": list(((runtime_probe.get("portable_contract") or {}).get("roles_supported") or [])),
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "overall_status": overall_status,
        "ok": overall_status == "ready",
        "operational_status": operational_status,
        "operational_ok": operational_ok,
        "operational_training": {
            "status": operational_status,
            "ok": operational_ok,
            "state": "idle_no_eligible_candidates" if controlled_idle_no_candidates else "active_contract",
            "controlled_idle_no_candidates": controlled_idle_no_candidates,
            "raw_status": overall_status,
            "raw_ok": overall_status == "ready",
            "raw_state_preserved": True,
        },
        "snapshot_ready": snapshot_fresh,
        "launch_allowed": training_launch_allowed,
        "prep_allowed": training_prep_allowed,
        "launch_blockers": training_launch_blockers,
        "prep_blockers": [
            str(item)
            for item in (training_launch_contract.get("prep_blockers") if isinstance(training_launch_contract.get("prep_blockers"), list) else [])
            if str(item or "").strip()
        ],
        "training_quality_score": round(_safe_float(training_quality.get("training_quality_score"), 0.0), 3),
        "training_evidence_gate": training_evidence_gate,
        "training_stage_reconciliation": {
            "collection_threshold_ready_count": _safe_int(
                collection_rollup.get("collection_threshold_ready_count"),
                _safe_int(collection_rollup.get("training_ready_count"), 0),
            ),
            "collection_rollup_gate_ready_count": _safe_int(
                collection_rollup.get("collection_threshold_ready_count"),
                _safe_int(collection_rollup.get("training_ready_count"), 0),
            ),
            "registry_bot_collection_floor_ready_count": _safe_int(
                training_stage_counts.get("collection_floor_ready"),
                0,
            ),
            "selector_candidate_count": _safe_int(training_candidate_selector.get("candidate_count"), 0),
            "selector_selected_count": _safe_int(training_candidate_selector.get("selected_count"), 0),
            "counts_share_denominator": False,
            "count_populations": {
                "collection_rollup_gate_ready_count": "collector rollup units with configured collection gates",
                "registry_bot_collection_floor_ready_count": "active registry bots with configured and satisfied observation floors",
                "selector_candidate_count": "registry bots that also pass label, diagnostic, balance, and overfit gates",
            },
            "counts_are_expected_to_differ": True,
            "explanation": "collection rollup and registry-stage counts use different populations; only label-safe, diagnostic-fresh, balanced, overfit-clear registry bots enter the retrain selector",
        },
        "recommended_batch_size": _safe_int(training_launch_contract.get("recommended_batch_size"), 0),
        "recommended_retrain_profile": str(training_launch_contract.get("recommended_retrain_profile") or ""),
        "recommended_retrain_command": list(training_launch_contract.get("recommended_retrain_command") or []),
        "snapshot_age_minutes": round(float(snapshot_age_minutes), 3) if snapshot_age_minutes is not None else None,
        "snapshot_content_fresh": snapshot_content_fresh,
        "snapshot_content_age_minutes": snapshot.get("content_age_minutes"),
        "fresh_window_minutes": int(max(int(fresh_minutes), 1)),
        "snapshot": {
            "sequence_count": sequence_count,
            "row_count": row_count,
            "rows_path": str(snapshot.get("rows_path") or ""),
            "lookback_days": _safe_int(snapshot.get("lookback_days"), 0),
            "latest_row_timestamp_utc": str(snapshot.get("latest_row_timestamp_utc") or ""),
            "content_fresh": snapshot_content_fresh,
            "top_modes": list(snapshot_coverage.get("top_modes") or [])[:5],
            "top_sequences": list(snapshot_coverage.get("top_sequences") or [])[:5],
        },
        "resource_guard": {
            "ok": resource_guard_raw_ok,
            "training_ok": resource_guard_training_ok,
            "memory_pressure_state": memory_pressure_state,
            "swap_used_gb": round(_safe_float(resource_guard.get("swap_used_gb"), 0.0), 3),
        },
        "resource_guard_training_gate": resource_guard_gate,
        "storage_quota_training_gate": storage_quota_gate,
        "runtime_backend_parity": {
            "parity_state": parity_state,
            "runtime_python_path": str(runtime_probe.get("runtime_python_path") or ""),
            "current_python_path": str(runtime_probe.get("current_python_path") or ""),
            "runtime_python_exists": bool(runtime_probe.get("runtime_python_exists", False)),
            "runtime_matches_current": bool(runtime_probe.get("runtime_matches_current", False)),
            "runtime_python_version": str(runtime_probe.get("runtime_python_version") or ""),
            "runtime_platform": str(runtime_probe.get("runtime_platform") or ""),
            "installed_backends": runtime_probe.get("installed_backends") if isinstance(runtime_probe.get("installed_backends"), dict) else {},
            "native_contract": runtime_probe.get("native_contract") if isinstance(runtime_probe.get("native_contract"), dict) else {},
            "portable_contract": runtime_probe.get("portable_contract") if isinstance(runtime_probe.get("portable_contract"), dict) else {},
            "probe_error": str(runtime_probe.get("probe_error") or ""),
            "mlx_failure_detected": mlx_failure_detected,
            "mlx_failure_active": mlx_failure_active,
        },
        "training_quality": {
            "overall_status": str(training_quality.get("overall_status") or ""),
            "training_quality_score": round(_safe_float(training_quality.get("training_quality_score"), 0.0), 3),
            "top_priorities": list(training_quality.get("top_priorities") or [])[:6],
            "targeted_retrain_bot_ids": list(targeted_actions.get("targeted_retrain_bot_ids") or []),
        },
        "coverage_seed": {
            "coverage_shortfall_bots": _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0),
            "seed_queue_size": len(coverage_seed.get("seed_queue") or []),
        },
        "bot_needs": {
            "overall_status": str(bot_needs.get("overall_status") or ""),
            "training_topoff_candidates": len(
                ((bot_needs.get("next_batches") if isinstance(bot_needs.get("next_batches"), dict) else {}).get("training_topoff") or [])
            ),
            "need_counts": bot_needs.get("need_counts") if isinstance(bot_needs.get("need_counts"), dict) else {},
            "training_candidate_selector": training_candidate_selector,
        },
        "backpressure_training_gate": backpressure_gate,
        "pretraining_drain_buffer": pretraining_drain_buffer,
        "host_training_headroom_gate": host_headroom_gate,
        "coverage_repair_ready": bool(coverage_repair_ready),
        "retry_pack": {
            "command": list(retry_pack.get("command") or []),
            "include_bot_ids": list(retry_pack.get("include_bot_ids") or []),
            "skip_master_update": bool(retry_pack.get("skip_master_update", False)),
        },
        "operating_mode": operating_mode,
        "precompute_targets": precompute_targets[: max(int(limit), 1)],
        "repair_contract": repair_contract,
        "training_launch_contract": training_launch_contract,
        "recommended_actions": _ordered_unique(recommended_actions),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a training-runtime control plane for snapshot reuse, cache posture, and targeted precompute retries.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--fresh-minutes", type=int, default=360)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), fresh_minutes=int(args.fresh_minutes), limit=int(args.limit))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_runtime_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"snapshot_ready={int(bool(payload.get('snapshot_ready', False)))} "
            f"precompute_targets={len(payload.get('precompute_targets') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
