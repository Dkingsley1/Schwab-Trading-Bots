#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import process_fanout_guard
    from scripts.ops.long_runtime_common import iso_now, load_json, payload_age_minutes, status_rank, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from . import process_fanout_guard
    from .long_runtime_common import iso_now, load_json, payload_age_minutes, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "guard_intelligence_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "guard_intelligence_state.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.guard_intelligence_override"

KEY_ARTIFACTS = {
    "process_fanout": "process_fanout_guard_latest.json",
    "resource_guard": "resource_guard_latest.json",
    "memory_efficiency": "memory_efficiency_control_latest.json",
    "pressure_relief": "pressure_relief_control_latest.json",
    "swap_pressure": "swap_pressure_governor_latest.json",
    "storage_backpressure": "storage_backpressure_autopilot_latest.json",
    "ingestion_backpressure": "ingestion_backpressure_latest.json",
    "storage_mount": "storage_mount_guard_latest.json",
    "system_drift": "system_drift_guard_latest.json",
    "runtime_throttle": "runtime_throttle_control_latest.json",
    "paper_400_ramp": "paper_400_ramp_latest.json",
}

NEGATIVE_STATUSES = {"blocked", "critical", "degraded", "red", "halted", "failed"}
WARNING_STATUSES = {"warn", "warning", "needs_work", "needs_coverage", "yellow", "active"}
RUNTIME_BLOCKER_ARTIFACTS = {"process_fanout", "resource_guard", "pressure_relief", "swap_pressure", "storage_mount"}
CORE_STALENESS_ARTIFACTS = {"process_fanout", "resource_guard"}


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


def _env_float(name: str, default: float) -> float:
    return _safe_float(os.getenv(name), default)


def _env_int(name: str, default: int) -> int:
    return _safe_int(os.getenv(name), default)


def _status_from_payload(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status", "severity", "memory_pressure_state", "state"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return "missing"


def _load_guard_artifacts(project_root: Path, *, now: datetime) -> dict[str, dict[str, Any]]:
    health_root = project_root / "governance" / "health"
    artifacts: dict[str, dict[str, Any]] = {}
    max_age_minutes = _env_float("GUARD_INTELLIGENCE_ARTIFACT_MAX_AGE_MINUTES", 60.0)
    for name, filename in KEY_ARTIFACTS.items():
        path = health_root / filename
        payload = load_json(path)
        exists = bool(path.exists())
        age = payload_age_minutes(payload, path, now=now) if exists else None
        status = _status_from_payload(payload) if payload else "missing"
        stale = bool(age is not None and age > max_age_minutes)
        artifacts[name] = {
            "path": str(path),
            "exists": exists,
            "age_minutes": round(float(age), 3) if age is not None else None,
            "stale": stale,
            "status": status,
            "ok": bool(payload.get("ok", status_rank(status) <= 1)) if payload else False,
            "payload": payload,
        }
    return artifacts


def _paper_soak_execution_allowed(artifacts: dict[str, dict[str, Any]]) -> bool:
    runtime = artifacts.get("runtime_throttle", {}).get("payload")
    runtime = runtime if isinstance(runtime, dict) else {}
    policy = runtime.get("paper_execution_policy") if isinstance(runtime.get("paper_execution_policy"), dict) else {}
    capacity = runtime.get("paper_capacity_contract") if isinstance(runtime.get("paper_capacity_contract"), dict) else {}
    runtime_policy = capacity.get("runtime_policy") if isinstance(capacity.get("runtime_policy"), dict) else {}
    ramp = artifacts.get("paper_400_ramp", {}).get("payload")
    ramp = ramp if isinstance(ramp, dict) else {}
    ramp_armed = bool(ramp.get("armed", False)) or str(ramp.get("stage") or "").strip().lower() == "armed"
    policy_armed = bool(policy.get("armed", False)) or str(policy.get("stage") or "").strip().lower() == "armed"
    paper_allowed = bool(policy.get("paper_execution_allowed", False)) and not bool(policy.get("pause_paper_execution", False))
    pressure_bypassed = bool(policy.get("pressure_pause_bypassed", False)) or "paper_ramp" in str(
        policy.get("reason") or policy.get("pressure_pause_bypass_reason") or ""
    ).lower()
    capacity_ready = bool(capacity.get("ready_for_700_bot_paper", False)) or bool(capacity.get("capacity_limited_paper_execution", False))
    live_locked = bool(runtime_policy.get("live_execution_blocked", True))
    return bool(paper_allowed and (ramp_armed or policy_armed or pressure_bypassed) and capacity_ready and live_locked)


def _paper_soak_pressure_bypass_active(artifacts: dict[str, dict[str, Any]]) -> bool:
    runtime = artifacts.get("runtime_throttle", {}).get("payload")
    runtime = runtime if isinstance(runtime, dict) else {}
    policy = runtime.get("paper_execution_policy") if isinstance(runtime.get("paper_execution_policy"), dict) else {}
    capacity = runtime.get("paper_capacity_contract") if isinstance(runtime.get("paper_capacity_contract"), dict) else {}
    reason = " ".join(
        str(policy.get(key) or "")
        for key in ("reason", "pressure_pause_bypass_reason", "capacity_limit_reason")
    ).lower()
    return bool(
        _paper_soak_execution_allowed(artifacts)
        and (
            bool(policy.get("pressure_pause_bypassed", False))
            or bool(capacity.get("attribution_capacity_advisory", False))
            or "pressure_only" in reason
            or "full_force" in reason
        )
    )


def _pressure_relief_advisory_for_paper_soak(artifacts: dict[str, dict[str, Any]]) -> bool:
    row = artifacts.get("pressure_relief", {})
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    pressure_bypass = _paper_soak_pressure_bypass_active(artifacts)
    if not payload or not (bool(payload.get("ok", False)) or pressure_bypass) or not _paper_soak_execution_allowed(artifacts):
        return False
    tier = str(payload.get("tier") or "").strip().lower()
    compute = str(payload.get("compute_pressure_level") or "").strip().lower()
    memory = str(payload.get("memory_pressure_level") or "").strip().lower()
    storage = payload.get("storage_pressure") if isinstance(payload.get("storage_pressure"), dict) else {}
    storage_severity = str(storage.get("severity") or "").strip().lower()
    swap = payload.get("swap_pressure") if isinstance(payload.get("swap_pressure"), dict) else {}
    swap_tier = str(swap.get("tier") or swap.get("raw_tier") or "").strip().lower()
    severe_compute = compute in {"high", "critical", "red"} and not pressure_bypass
    severe_memory = memory in {"high", "critical", "red"}
    severe_storage = storage_severity in {"critical", "red", "blocked"}
    severe_swap = swap_tier in {"survival", "pause_research", "constrained"}
    return bool(
        (tier in {"observe", "calm", "guarded_relief"} or (tier == "deep_relief" and pressure_bypass))
        and not severe_compute
        and not severe_memory
        and not severe_storage
        and not severe_swap
    )


def _artifact_status_counts(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    stale: list[str] = []
    for name, row in artifacts.items():
        status = str(row.get("status") or "").lower()
        if status == "missing":
            if name in {"process_fanout", "resource_guard"}:
                warnings.append(name)
            continue
        stale_artifact = bool(row.get("stale"))
        if stale_artifact and name in CORE_STALENESS_ARTIFACTS:
            stale.append(name)
        if status in NEGATIVE_STATUSES or status_rank(status) >= 3:
            if name == "pressure_relief" and _pressure_relief_advisory_for_paper_soak(artifacts):
                warnings.append(name)
                continue
            if stale_artifact or name not in RUNTIME_BLOCKER_ARTIFACTS:
                warnings.append(name)
            else:
                blockers.append(name)
        elif status in WARNING_STATUSES or status_rank(status) == 2:
            warnings.append(name)
    return {
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "stale_core_count": len(stale),
        "blockers": blockers,
        "warnings": warnings,
        "stale_core_artifacts": stale,
    }


def _fanout_from_live_processes(project_root: Path, artifact: dict[str, Any], *, collect_live: bool) -> dict[str, Any]:
    payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    max_count = _env_int("PROCESS_FANOUT_GUARD_MAX_COUNT", _safe_int(thresholds.get("max_count"), 120))
    target_count = _env_int("PROCESS_FANOUT_GUARD_TARGET_COUNT", _safe_int(thresholds.get("target_count"), 80))
    scoring_max_count = _env_int(
        "GUARD_INTELLIGENCE_SCORING_FANOUT_MAX_COUNT",
        _env_int("GUARD_INTELLIGENCE_FULL_FANOUT_MAX_COUNT", 180),
    )
    max_count = max(max_count, scoring_max_count)
    max_rss_mb = _env_float("PROCESS_FANOUT_GUARD_MAX_RSS_MB", _safe_float(thresholds.get("max_rss_mb"), 5120.0))
    target_rss_mb = _env_float("PROCESS_FANOUT_GUARD_TARGET_RSS_MB", _safe_float(thresholds.get("target_rss_mb"), 4096.0))
    scoring_max_rss_mb = _env_float(
        "GUARD_INTELLIGENCE_SCORING_FANOUT_MAX_RSS_MB",
        _env_float("GUARD_INTELLIGENCE_FULL_FANOUT_MAX_RSS_MB", 12288.0),
    )
    max_rss_mb = max(max_rss_mb, scoring_max_rss_mb)

    rows = []
    if collect_live:
        try:
            rows = process_fanout_guard.collect_processes(project_marker=str(project_root))
        except Exception:
            rows = []
    artifact_triggered = bool(payload.get("triggered", False))
    if rows:
        process_count = len(rows)
        total_rss_mb = round(sum(row.rss_mb for row in rows), 3)
        source = "live_process_table"
    else:
        fanout = payload.get("fanout") if isinstance(payload.get("fanout"), dict) else {}
        process_count = _safe_int(fanout.get("process_count"), 0)
        total_rss_mb = _safe_float(fanout.get("total_rss_mb"), 0.0)
        source = "process_fanout_artifact"

    count_ratio = process_count / max(max_count, 1)
    rss_ratio = total_rss_mb / max(max_rss_mb, 1.0)
    pressure_score = round(max(count_ratio, rss_ratio), 4)
    triggered = bool(count_ratio >= 1.0 or rss_ratio >= 1.0 or (source != "live_process_table" and artifact_triggered))
    return {
        "source": source,
        "process_count": process_count,
        "total_rss_mb": round(total_rss_mb, 3),
        "max_count": max_count,
        "target_count": target_count,
        "max_rss_mb": round(max_rss_mb, 3),
        "target_rss_mb": round(target_rss_mb, 3),
        "count_ratio": round(count_ratio, 4),
        "rss_ratio": round(rss_ratio, 4),
        "pressure_score": pressure_score,
        "triggered": triggered,
    }


def _resource_pressure_score(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resource = artifacts.get("resource_guard", {}).get("payload")
    payload = resource if isinstance(resource, dict) else {}
    state = str(payload.get("memory_pressure_state") or payload.get("memory_pressure_kind") or "").strip().lower()
    free_pct = _safe_float(payload.get("memory_free_pct"), 100.0)
    swap_gb = _safe_float(payload.get("swap_used_gb"), 0.0)
    compressor_gb = _safe_float(payload.get("compressor_gb"), 0.0)
    score = 0.15
    if state == "red":
        score = 1.15
    elif state == "yellow":
        score = 0.82
    elif state == "green":
        score = 0.2
    if free_pct <= 8.0:
        score = max(score, 1.1)
    elif free_pct <= 14.0:
        score = max(score, 0.88)
    if swap_gb >= 24.0:
        score = max(score, 1.12)
    elif swap_gb >= 12.0:
        score = max(score, 0.84)
    if compressor_gb >= 18.0:
        score = max(score, 0.9)
    return {
        "score": round(score, 4),
        "memory_pressure_state": state or "unknown",
        "memory_free_pct": round(free_pct, 3),
        "swap_used_gb": round(swap_gb, 3),
        "compressor_gb": round(compressor_gb, 3),
    }


def _storage_pressure_score(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    details: dict[str, Any] = {}
    pressure_relief_advisory = _pressure_relief_advisory_for_paper_soak(artifacts)
    for name in ("ingestion_backpressure", "storage_backpressure", "pressure_relief", "swap_pressure"):
        payload = artifacts.get(name, {}).get("payload")
        if not isinstance(payload, dict):
            continue
        pressure_index = _safe_float(payload.get("pressure_index"), 0.0)
        status = _status_from_payload(payload)
        score = min(max(pressure_index / 3.0, 0.0), 1.2) if pressure_index else 0.0
        stale = bool(artifacts.get(name, {}).get("stale"))
        advisory = bool(name == "pressure_relief" and pressure_relief_advisory)
        if advisory:
            pressure_bypass = _paper_soak_pressure_bypass_active(artifacts)
            score = max(score, 0.65 if pressure_bypass else 0.35)
        elif status in NEGATIVE_STATUSES and not stale:
            score = max(score, 1.05)
        elif status in WARNING_STATUSES and not stale:
            score = max(score, 0.72)
        scores.append(score)
        details[name] = {
            "status": status,
            "pressure_index": round(pressure_index, 3),
            "score": round(score, 4),
            "paper_soak_advisory": advisory,
        }
    score = max(scores) if scores else 0.0
    return {"score": round(score, 4), "details": details}


def _mode_env(mode: str, pressure_score: float) -> dict[str, str]:
    full_max = _env_float("GUARD_INTELLIGENCE_FULL_FANOUT_MAX_RSS_MB", 12288.0)
    full_target = _env_float("GUARD_INTELLIGENCE_FULL_FANOUT_TARGET_RSS_MB", 8192.0)
    full_max_count = _env_int("GUARD_INTELLIGENCE_FULL_FANOUT_MAX_COUNT", 180)
    full_target_count = _env_int("GUARD_INTELLIGENCE_FULL_FANOUT_TARGET_COUNT", 140)
    balanced_max = _env_float("GUARD_INTELLIGENCE_BALANCED_FANOUT_MAX_RSS_MB", 9216.0)
    balanced_target = _env_float("GUARD_INTELLIGENCE_BALANCED_FANOUT_TARGET_RSS_MB", 6144.0)
    balanced_max_count = _env_int("GUARD_INTELLIGENCE_BALANCED_FANOUT_MAX_COUNT", 130)
    balanced_target_count = _env_int("GUARD_INTELLIGENCE_BALANCED_FANOUT_TARGET_COUNT", 100)
    protective_max = _env_float("GUARD_INTELLIGENCE_PROTECTIVE_FANOUT_MAX_RSS_MB", 6144.0)
    protective_target = _env_float("GUARD_INTELLIGENCE_PROTECTIVE_FANOUT_TARGET_RSS_MB", 4096.0)
    protective_max_count = _env_int("GUARD_INTELLIGENCE_PROTECTIVE_FANOUT_MAX_COUNT", 90)
    protective_target_count = _env_int("GUARD_INTELLIGENCE_PROTECTIVE_FANOUT_TARGET_COUNT", 70)

    base = {
        "GUARD_INTELLIGENCE_ENABLED": "1",
        "GUARD_INTELLIGENCE_POLICY_MODE": mode,
        "GUARD_INTELLIGENCE_PRESSURE_SCORE": f"{pressure_score:.4f}",
        "PROCESS_FANOUT_GUARD_PRESERVE_CLEAR_COOLDOWN": "1",
        "SHADOW_WATCHDOG_DIRECT_CHILD_SLEEVES": "0",
    }
    if mode == "protective_throttle":
        return {
            **base,
            "PROCESS_FANOUT_GUARD_ACTIVE": "1",
            "PROCESS_FANOUT_GUARD_MAX_COUNT": str(protective_max_count),
            "PROCESS_FANOUT_GUARD_TARGET_COUNT": str(protective_target_count),
            "PROCESS_FANOUT_GUARD_MAX_RSS_MB": f"{protective_max:.1f}",
            "PROCESS_FANOUT_GUARD_TARGET_RSS_MB": f"{protective_target:.1f}",
            "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE": "0",
            "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE": "0",
            "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "0",
            "TRAINING_RUNTIME_PAUSED_FOR_FANOUT": "1",
            "SHADOW_RESEARCH_PAUSED_FOR_FANOUT": "1",
            "SHADOW_LOOP_INTERVAL": "60",
            "DIVIDEND_SHADOW_INTERVAL": "120",
            "BOND_SHADOW_INTERVAL": "180",
            "SPECIALIZED_SLEEVE_INTERVAL": "300",
            "SLEEVE_WORKERS_BASELINE": "1",
            "SLEEVE_WORKERS_DIVIDEND": "1",
            "SLEEVE_WORKERS_BOND": "1",
        }
    if mode == "balanced_guarded":
        return {
            **base,
            "PROCESS_FANOUT_GUARD_ACTIVE": "0",
            "PROCESS_FANOUT_GUARD_MAX_COUNT": str(balanced_max_count),
            "PROCESS_FANOUT_GUARD_TARGET_COUNT": str(balanced_target_count),
            "PROCESS_FANOUT_GUARD_MAX_RSS_MB": f"{balanced_max:.1f}",
            "PROCESS_FANOUT_GUARD_TARGET_RSS_MB": f"{balanced_target:.1f}",
            "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE": "1",
            "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE": "1",
            "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "0",
            "TRAINING_RUNTIME_PAUSED_FOR_FANOUT": "0",
            "SHADOW_RESEARCH_PAUSED_FOR_FANOUT": "0",
            "SHADOW_LOOP_INTERVAL": "30",
            "DIVIDEND_SHADOW_INTERVAL": "90",
            "BOND_SHADOW_INTERVAL": "120",
            "SPECIALIZED_SLEEVE_INTERVAL": "240",
            "SLEEVE_WORKERS_BASELINE": "2",
            "SLEEVE_WORKERS_DIVIDEND": "1",
            "SLEEVE_WORKERS_BOND": "1",
        }
    return {
        **base,
        "PROCESS_FANOUT_GUARD_ACTIVE": "0",
        "PROCESS_FANOUT_GUARD_MAX_COUNT": str(full_max_count),
        "PROCESS_FANOUT_GUARD_TARGET_COUNT": str(full_target_count),
        "PROCESS_FANOUT_GUARD_MAX_RSS_MB": f"{full_max:.1f}",
        "PROCESS_FANOUT_GUARD_TARGET_RSS_MB": f"{full_target:.1f}",
        "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE": "1",
        "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE": "1",
        "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "1",
        "TRAINING_RUNTIME_PAUSED_FOR_FANOUT": "0",
        "SHADOW_RESEARCH_PAUSED_FOR_FANOUT": "0",
        "SHADOW_LOOP_INTERVAL": "15",
        "DIVIDEND_SHADOW_INTERVAL": "60",
        "BOND_SHADOW_INTERVAL": "120",
        "SPECIALIZED_SLEEVE_INTERVAL": "180",
        "SLEEVE_WORKERS_BASELINE": os.getenv("SLEEVE_WORKERS_BASELINE", os.getenv("ASYNC_PIPELINE_WORKERS", "4")),
        "SLEEVE_WORKERS_DIVIDEND": os.getenv("SLEEVE_WORKERS_DIVIDEND", "2"),
        "SLEEVE_WORKERS_BOND": os.getenv("SLEEVE_WORKERS_BOND", "2"),
    }


def _write_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/guard_intelligence_layer.py"]
    lines.extend(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _select_mode(
    *,
    fanout: dict[str, Any],
    resource: dict[str, Any],
    storage: dict[str, Any],
    counts: dict[str, Any],
    state: dict[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    calm_threshold = _env_float("GUARD_INTELLIGENCE_CALM_THRESHOLD", 0.72)
    warm_threshold = _env_float("GUARD_INTELLIGENCE_WARM_THRESHOLD", 0.84)
    hot_threshold = _env_float("GUARD_INTELLIGENCE_HOT_THRESHOLD", 1.0)
    stable_required = _env_int("GUARD_INTELLIGENCE_MIN_STABLE_SAMPLES", 1)

    raw_pressure = max(
        _safe_float(fanout.get("pressure_score"), 0.0),
        _safe_float(resource.get("score"), 0.0),
        _safe_float(storage.get("score"), 0.0),
    )
    previous_ewma = _safe_float(state.get("pressure_ewma"), raw_pressure)
    ewma = round((previous_ewma * 0.65) + (raw_pressure * 0.35), 4)
    has_blocker = _safe_int(counts.get("blocker_count"), 0) > 0
    hot = bool(fanout.get("triggered")) or raw_pressure >= hot_threshold or has_blocker
    warm = raw_pressure >= warm_threshold or ewma >= warm_threshold
    calm = raw_pressure <= calm_threshold and not has_blocker and _safe_int(counts.get("stale_core_count"), 0) == 0

    stable_samples = _safe_int(state.get("stable_samples"), 0)
    pressure_samples = _safe_int(state.get("pressure_samples"), 0)
    stable_samples = stable_samples + 1 if calm else 0
    pressure_samples = pressure_samples + 1 if hot else 0

    if hot:
        mode = "protective_throttle"
    elif stable_samples >= stable_required:
        mode = "full_schwab_observe"
    elif warm:
        mode = "balanced_guarded"
    else:
        mode = "balanced_guarded"

    next_state = {
        "pressure_ewma": ewma,
        "stable_samples": stable_samples,
        "pressure_samples": pressure_samples,
        "last_policy_mode": mode,
        "last_pressure_score": round(raw_pressure, 4),
        "last_updated_utc": iso_now(),
    }
    return mode, round(raw_pressure, 4), next_state


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    collect_live: bool = True,
    out_path: Path = DEFAULT_OUT_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    artifacts = _load_guard_artifacts(project_root, now=now)
    counts = _artifact_status_counts(artifacts)
    fanout = _fanout_from_live_processes(project_root, artifacts["process_fanout"], collect_live=collect_live)
    resource = _resource_pressure_score(artifacts)
    storage = _storage_pressure_score(artifacts)
    state = load_json(state_path)
    mode, pressure_score, next_state = _select_mode(
        fanout=fanout,
        resource=resource,
        storage=storage,
        counts=counts,
        state=state,
    )
    env = _mode_env(mode, pressure_score)
    override_changed = False
    if apply:
        override_changed = _write_override(override_path, env)
    write_payload(state_path, next_state)

    status = "ready" if mode == "full_schwab_observe" else "warn" if mode == "balanced_guarded" else "active"
    safe_artifacts = {
        key: {sub_key: value for sub_key, value in row.items() if sub_key != "payload"}
        for key, row in artifacts.items()
    }
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": status,
        "ok": True,
        "apply": bool(apply),
        "policy_mode": mode,
        "pressure_score": pressure_score,
        "decision_memory": next_state,
        "signals": {
            "fanout": fanout,
            "resource_pressure": resource,
            "storage_pressure": storage,
            "guard_status_counts": counts,
        },
        "recommended_env_overrides": env,
        "override": {
            "path": str(override_path),
            "changed": override_changed,
            "written": bool(apply),
        },
        "artifacts": safe_artifacts,
        "self_update": {
            "state_path": str(state_path),
            "state_updated": True,
            "override_changed": override_changed,
            "reason": "guard policy changed" if override_changed else "guard policy already current",
        },
        "codex_handoff": {
            "summary": "Guard intelligence reconciles process fanout, memory pressure, storage pressure, and guard health before allowing full Schwab sleeves.",
            "next_best_commands": [
                "./scripts/ops/opsctl.sh guard-intelligence --apply --json",
                "./scripts/ops/opsctl.sh process-fanout-guard --json",
                "./scripts/ops/opsctl.sh health-fast --json",
            ],
        },
    }
    write_payload(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-updating intelligence layer for runtime guards.")
    parser.add_argument("--apply", action="store_true", help="Write the guard intelligence runtime override when policy changes.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    parser.add_argument("--no-live-processes", action="store_true", help="Use existing guard artifacts instead of collecting the process table.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        apply=bool(args.apply),
        collect_live=not bool(args.no_live_processes),
        out_path=Path(args.out).expanduser(),
        state_path=Path(args.state).expanduser(),
        override_path=Path(args.override).expanduser(),
    )
    if args.json:
        import json

        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            f"guard_intelligence status={payload['overall_status']} "
            f"mode={payload['policy_mode']} pressure={payload['pressure_score']:.3f}"
        )


if __name__ == "__main__":
    main()
