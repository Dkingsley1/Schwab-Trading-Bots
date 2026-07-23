import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import fcntl
import hashlib
import gc
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.training_quality_thresholds import TARGET_TEST_ACCURACY_FLOOR

from core.runtime_python import resolve_runtime_python, resolve_training_python
from scripts import retrain_lane_scheduler as retrain_lane_scheduler_src

CORE_DIR = os.path.join(PROJECT_ROOT, "core")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "master_bot_registry.json")
VENV_PY = str(resolve_training_python(PROJECT_ROOT))
MASTER_RUNNER = os.path.join(PROJECT_ROOT, "scripts", "run_master_bot.py")
TRADE_DATASET_BUILDER = os.path.join(PROJECT_ROOT, "scripts", "build_behavior_dataset_from_decisions.py")
TRADE_DATASET_BUILDER_LEGACY = os.path.join(PROJECT_ROOT, "scripts", "build_trade_learning_dataset.py")
TRADE_BEHAVIOR_TRAINER = os.path.join(PROJECT_ROOT, "scripts", "train_trade_behavior_bot.py")
TRADE_BEHAVIOR_DATASET = os.path.join(PROJECT_ROOT, "data", "trade_history", "trade_learning_dataset.json")
SNAPSHOT_HEALTH_SYNC_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "sync_snapshot_health_to_sql.py")
PRUNE_UNDERPERFORMERS = os.path.join(PROJECT_ROOT, "scripts", "prune_underperformers.py")
PRUNE_REDUNDANT = os.path.join(PROJECT_ROOT, "scripts", "prune_redundant_bots.py")
ARCHIVE_OLD_MODELS = os.path.join(PROJECT_ROOT, "scripts", "archive_old_models.py")
CANARY_DIAGNOSTICS = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "canary_diagnostics_latest.json")
RETIRE_PERSISTENT_LOSERS = os.path.join(PROJECT_ROOT, "scripts", "retire_persistent_losers.py")
PROMOTION_READINESS_PATH = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "promotion_readiness_latest.json")
PROMOTION_BOTTLENECK_PATH = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "promotion_bottleneck_latest.json")
HEALTH_GATES_PATH = os.path.join(PROJECT_ROOT, "governance", "health", "health_gates_latest.json")
WALK_FORWARD_VALIDATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "walk_forward_validate.py")
WALK_FORWARD_PROMOTION_GATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "walk_forward_promotion_gate.py")
LANE_PROMOTION_GATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "lane_promotion_gate.py")
PROMOTION_READINESS_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "promotion_readiness_summary.py")
PROMOTION_BOTTLENECK_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "promotion_bottleneck_focus.py")
SCHEMA_MIGRATION_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "schema_migration_guard.py")
FEATURE_STORE_MANIFEST_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "feature_store_manifest.py")
BOT_SUPPORT_OWNER_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "bot_support_owner_guard.py")
NEW_BOT_GRADUATION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "new_bot_graduation_gate.py")
NEW_BOT_ADMISSION_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "new_bot_admission_guard.py")
RETRAIN_SCHEMA_COMPATIBILITY_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "retrain_schema_compatibility_guard.py")
LEAK_OVERFIT_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "leak_overfit_guard.py")
GOLDEN_REPLAY_REGRESSION_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "golden_replay_regression_guard.py")
COHORT_DRIFT_BASELINE_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "cohort_drift_baseline_guard.py")
CHAMPION_CHALLENGER_PROBATION_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "champion_challenger_probation_guard.py")
CHAMPION_CHALLENGER_PROBATION_ACTION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "champion_challenger_probation_action.py")
RETRAIN_LANE_SCHEDULER_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "retrain_lane_scheduler.py")
PROMOTION_PACKET_BUILDER_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "promotion_packet_builder.py")
MODEL_LIFECYCLE_HYGIENE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "model_lifecycle_hygiene.py")
WEEKLY_GATE_BLOCKER_REPORT_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "weekly_gate_blocker_report.py")
RETRAIN_ARTIFACT_FRESHNESS_GUARD = os.path.join(PROJECT_ROOT, "scripts", "retrain_artifact_freshness_guard.py")
TRAINING_SAMPLE_QUOTA_GUARD = os.path.join(PROJECT_ROOT, "scripts", "training_sample_quota_guard.py")
REPLAY_FEATURE_ABLATION_REPORT = os.path.join(PROJECT_ROOT, "scripts", "replay_feature_ablation_report.py")
EXPORT_MODEL_CARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "export_model_card.py")
COUNTERFACTUAL_REPLAY_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "counterfactual_replay_harness.py")
COUNTERFACTUAL_REPLAY_LATEST = os.path.join(PROJECT_ROOT, "governance", "health", "counterfactual_replay_latest.json")
PLATFORM_CONTROL_PLANE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "platform_control_plane_report.py")
POINT_IN_TIME_EVENT_STORE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "point_in_time_event_store.py")
LIVE_READINESS_SMOKE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "live_readiness_smoke.py")
DATA_RETENTION_POLICY = os.path.join(PROJECT_ROOT, "scripts", "data_retention_policy.py")
DATA_DIVERGENCE_GLOBAL_FILE = os.path.join(PROJECT_ROOT, "governance", "health", "data_source_divergence_latest.json")
DATA_DIVERGENCE_BOND_FILE = os.path.join(PROJECT_ROOT, "governance", "health", "data_source_divergence_bond_latest.json")
DATA_DIVERGENCE_NON_BOND_FILE = os.path.join(PROJECT_ROOT, "governance", "health", "data_source_divergence_non_bond_latest.json")
RETRAIN_OPERATOR_NOTES_PATH = os.path.join(PROJECT_ROOT, "governance", "health", "retrain_operator_notes_latest.json")
PAPER_PERFORMANCE_PATH = os.path.join(PROJECT_ROOT, "governance", "health", "paper_performance_latest.json")
PAPER_HARD_EXAMPLES_PATH = os.path.join(PROJECT_ROOT, "governance", "training_diagnostics", "paper_hard_examples_latest.json")
TRAINING_DIAGNOSTICS_DIR = os.path.join(PROJECT_ROOT, "governance", "training_diagnostics")
TRAINING_SAMPLE_STARVED_QUEUE_LATEST = os.path.join(TRAINING_DIAGNOSTICS_DIR, "training_sample_starved_queue_latest.json")
TRAINING_QUALITY_REPAIR_QUEUE_LATEST = os.path.join(TRAINING_DIAGNOSTICS_DIR, "training_quality_repair_queue_latest.json")
TRAINING_TIMEOUT_QUEUE_LATEST = os.path.join(TRAINING_DIAGNOSTICS_DIR, "training_timeout_queue_latest.json")
RETRAIN_INPUT_FEATURE_DIAGNOSTICS_LATEST = os.path.join(TRAINING_DIAGNOSTICS_DIR, "retrain_input_feature_diagnostics_latest.json")
RETRAIN_REPLAY_SUMMARY_LATEST = os.path.join(TRAINING_DIAGNOSTICS_DIR, "retrain_replay_summary_latest.json")
RUNTIME_TRAINING_SNAPSHOT_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "build_runtime_training_snapshot.py")
RUNTIME_TRAINING_SNAPSHOT_LATEST = os.path.join(PROJECT_ROOT, "governance", "health", "runtime_training_snapshot_latest.json")
COLLECTOR_CONTRACTS_LATEST = os.path.join(PROJECT_ROOT, "governance", "health", "collector_contracts_latest.json")
STORAGE_TIER_POLICY_LATEST = os.path.join(PROJECT_ROOT, "governance", "health", "storage_tier_policy_latest.json")
JSONL_DISCOVERY_MANIFEST_LATEST = os.path.join(PROJECT_ROOT, "governance", "health", "jsonl_discovery_manifest_latest.json")
RETRAIN_RETRY_PACK_LATEST = os.path.join(PROJECT_ROOT, "governance", "health", "retrain_retry_pack_latest.json")
WALK_FORWARD_LATEST = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "walk_forward_latest.json")

_BOT_NEEDS_EVIDENCE_CACHE: dict[str, dict[str, Any]] | None = None
_BOT_NEEDS_EVIDENCE_CACHE_SOURCE = ""

ADVANCED_RETRAIN_DIAGNOSTIC_FEATURES = (
    "core_cross_sectional_rank_norm",
    "core_regime_specialist_blend_norm",
    "core_event_reaction_norm",
    "core_cross_asset_confirmation_norm",
    "day_failed_breakout_risk_norm",
    "day_closing_squeeze_norm",
    "swing_weekly_pullback_quality_norm",
    "dividend_payout_stress_gate_norm",
    "long_term_factor_exposure_control_norm",
    "long_term_overlap_rebalance_norm",
    "options_skew_dislocation_norm",
    "options_gamma_wall_reaction_norm",
    "futures_basis_dislocation_norm",
    "futures_overnight_inventory_norm",
)

SEGMENT_TO_REPLAY_PROFILES = {
    "trend": ["aggressive", "swing_aggressive", "default"],
    "mean_revert": ["conservative", "dividend", "bond"],
    "shock": ["intraday_aggressive", "aggressive", "schwab_futures", "crypto_futures"],
    "liquidity": ["intraday_aggressive", "aggressive", "fx"],
    "other": ["default", "conservative"],
}

_MLX_LOCK_HANDLE = None


def _acquire_mlx_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = "unknown"
        fh.close()
        print(f"[MLXLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
        return None

    fh.seek(0)
    fh.truncate(0)
    fh.write(json.dumps({
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cmd": sys.argv,
    }, ensure_ascii=True))
    fh.flush()
    print(f"[MLXLock] acquired lock_path={lock_path} pid={os.getpid()}")
    return fh


def _normalized_bot_id_from_script(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".py"):
        name = name[:-3]
    return name.lower()


def _safe_json_load(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _effective_min_considered(gate: Mapping[str, Any], thresholds: Mapping[str, Any]) -> int:
    effective_thresholds = gate.get("effective_thresholds") if isinstance(gate.get("effective_thresholds"), Mapping) else {}
    raw_value = effective_thresholds.get("min_considered_bots", thresholds.get("min_considered_bots", 4))
    try:
        return max(int(float(raw_value or 4)), 1)
    except Exception:
        return 4


def _parse_json_output(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    for line in reversed(raw.splitlines()):
        candidate = line.strip()
        if not (candidate.startswith("{") and candidate.endswith("}")):
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        return payload if isinstance(payload, dict) else {}

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _retrain_launch_artifact_dir() -> str:
    return os.path.join(PROJECT_ROOT, "governance", "training_diagnostics", "retrain_launches")


def _retrain_launch_latest_path(*, dry_run: bool) -> str:
    latest_name = "retrain_launch_dry_run_latest.json" if dry_run else "retrain_launch_latest.json"
    return os.path.join(PROJECT_ROOT, "governance", "health", latest_name)


def _retrain_launch_source_latest_path(*, dry_run: bool, source: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(source or "").strip().lower()).strip("_")
    if not slug:
        slug = "unknown"
    latest_name = (
        f"retrain_launch_dry_run_latest_{slug}.json"
        if dry_run
        else f"retrain_launch_latest_{slug}.json"
    )
    return os.path.join(PROJECT_ROOT, "governance", "health", latest_name)


def _safe_parent_command(pid: int) -> str:
    if int(pid) <= 0:
        return ""
    try:
        proc = subprocess.run(
            ["/bin/ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    return ((proc.stdout or "").strip().splitlines() or [""])[0].strip()


def _csv_tokens(raw: Any) -> list[str]:
    out: list[str] = []
    for chunk in str(raw or "").split(","):
        text = chunk.strip()
        if text:
            out.append(text)
    return out


def _infer_retrain_trigger_source(*, parent_command: str) -> str:
    explicit = str(os.getenv("RETRAIN_TRIGGER_SOURCE", "") or "").strip()
    if explicit:
        return explicit

    launch_label = " ".join(
        [
            str(os.getenv("RETRAIN_TRIGGER_LABEL", "") or "").strip(),
            str(os.getenv("LAUNCH_JOB_LABEL", "") or "").strip(),
            str(os.getenv("XPC_SERVICE_NAME", "") or "").strip(),
        ]
    ).lower()
    parent_lower = str(parent_command or "").strip().lower()

    if "com.dankingsley.retrain.daily_small" in launch_label:
        return "launchd_daily_small"
    if "com.dankingsley.retrain.weekly_full" in launch_label:
        return "launchd_weekly_full"
    if "run_shadow_training_loop.py" in parent_lower:
        return "shadow_training_loop_auto_retrain"
    if "manual_retrain_with_pause.py" in parent_lower:
        return "manual_retrain_with_pause"
    if "retrain_orchestrator.py" in parent_lower:
        return "retrain_orchestrator"
    if "opsctl.sh" in parent_lower:
        return "opsctl"
    return "direct_weekly_retrain"


def _build_retrain_launch_record(args: argparse.Namespace, retrain_profile: str) -> dict[str, Any]:
    started_utc = datetime.now(timezone.utc).isoformat()
    launch_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parent_pid = os.getppid()
    parent_command = _safe_parent_command(parent_pid)
    include_bot_ids = _csv_tokens(getattr(args, "include_bot_ids", ""))
    exclude_bot_ids = _csv_tokens(getattr(args, "exclude_bot_ids", ""))
    regime_focus = str(getattr(args, "regime_focus", "") or "").strip()
    source = _infer_retrain_trigger_source(parent_command=parent_command)
    source_label = (
        str(os.getenv("RETRAIN_TRIGGER_LABEL", "") or "").strip()
        or str(os.getenv("LAUNCH_JOB_LABEL", "") or "").strip()
        or str(os.getenv("XPC_SERVICE_NAME", "") or "").strip()
    )
    run_mode = "targeted" if include_bot_ids or regime_focus or bool(getattr(args, "skip_master_update", False)) else "full"
    if getattr(args, "dry_run", False):
        run_mode = f"{run_mode}_dry_run"

    return {
        "timestamp_utc": started_utc,
        "started_utc": started_utc,
        "state": "running",
        "launch_slug": launch_slug,
        "pid": int(os.getpid()),
        "parent_pid": int(parent_pid),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "argv": [str(part) for part in sys.argv],
        "source": source,
        "source_label": source_label,
        "source_context": str(os.getenv("RETRAIN_TRIGGER_CONTEXT", "") or "").strip(),
        "source_broker": str(os.getenv("RETRAIN_TRIGGER_BROKER", "") or "").strip(),
        "source_profile": str(os.getenv("RETRAIN_TRIGGER_PROFILE", "") or "").strip(),
        "launch_log_path": str(os.getenv("RETRAIN_LAUNCH_LOG_PATH", "") or "").strip(),
        "xpc_service_name": str(os.getenv("XPC_SERVICE_NAME", "") or "").strip(),
        "launch_job_label": str(os.getenv("LAUNCH_JOB_LABEL", "") or "").strip(),
        "correlation_run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "correlation_iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
        "parent_command": parent_command,
        "retrain_profile": str(retrain_profile or "").strip() or "default",
        "run_mode": run_mode,
        "selector_summary": {
            "include_bot_ids": include_bot_ids,
            "exclude_bot_ids": exclude_bot_ids,
            "regime_focus": regime_focus,
            "active_only": bool(getattr(args, "active_only", False)),
            "include_deleted": bool(getattr(args, "include_deleted", False)),
            "force_all_targets": bool(getattr(args, "force_all_targets", False)),
            "max_targets": int(getattr(args, "max_targets", 0) or 0),
            "min_model_age_hours": float(getattr(args, "min_model_age_hours", 0.0) or 0.0),
            "skip_master_update": bool(getattr(args, "skip_master_update", False)),
            "continue_on_error": bool(getattr(args, "continue_on_error", False)),
        },
    }


def _persist_retrain_launch_record(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = dict(record or {})
    launch_slug = str(payload.get("launch_slug") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    payload["launch_slug"] = launch_slug
    artifact_path = str(payload.get("artifact_path") or "").strip()
    if not artifact_path:
        artifact_path = os.path.join(
            _retrain_launch_artifact_dir(),
            f"retrain_launch_{launch_slug}_{int(payload.get('pid', os.getpid()) or os.getpid())}.json",
        )
        payload["artifact_path"] = artifact_path
    latest_path = _retrain_launch_latest_path(dry_run=dry_run)
    payload["latest_path"] = latest_path
    source_latest_path = ""
    source = str(payload.get("source") or "").strip()
    if source:
        source_latest_path = _retrain_launch_source_latest_path(dry_run=dry_run, source=source)
    payload["source_latest_path"] = source_latest_path
    payload["latest_alias_paths"] = [path for path in (latest_path, source_latest_path) if str(path or "").strip()]

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    if source_latest_path:
        os.makedirs(os.path.dirname(source_latest_path), exist_ok=True)
        with open(source_latest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    return payload


def _finalize_retrain_launch_record(
    record: dict[str, Any],
    *,
    dry_run: bool,
    final_status: str,
    exit_code: int,
    scorecard_path: str = "",
    retry_pack_path: str = "",
    master_update_status: str = "",
    failure_count: int | None = None,
) -> dict[str, Any]:
    payload = dict(record or {})
    payload["state"] = "completed"
    payload["ended_utc"] = datetime.now(timezone.utc).isoformat()
    payload["final_status"] = str(final_status or "").strip() or "completed"
    payload["exit_code"] = int(exit_code)
    if scorecard_path:
        payload["scorecard_path"] = str(scorecard_path)
    if retry_pack_path:
        payload["retry_pack_path"] = str(retry_pack_path)
    if master_update_status:
        payload["master_update_status"] = str(master_update_status)
    if failure_count is not None:
        payload["failure_count"] = int(failure_count)
    return _persist_retrain_launch_record(payload, dry_run=dry_run)


def _paper_feedback_summary(path: str) -> dict[str, Any]:
    payload = _safe_json_load(path)
    sleeves = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    total_executions = 0
    active_sleeves = 0
    non_flat_strategies = 0
    for row in sleeves:
        if not isinstance(row, dict):
            continue
        executions = int(float(row.get("executions", 0) or 0))
        total_executions += max(executions, 0)
        if executions > 0:
            active_sleeves += 1
        non_flat_strategies += max(int(float(row.get("non_flat_strategy_count", 0) or 0)), 0)
    return {
        "total_executions": total_executions,
        "active_sleeves": active_sleeves,
        "non_flat_strategies": non_flat_strategies,
    }


def _promotion_state_precheck_failures(
    *,
    promotion_readiness_path: str = PROMOTION_READINESS_PATH,
    health_gates_path: str = HEALTH_GATES_PATH,
    paper_performance_path: str = PAPER_PERFORMANCE_PATH,
) -> list[str]:
    failures: list[str] = []

    require_health_gate_clear = os.getenv("MASTER_PROMOTION_REQUIRE_HEALTH_GATE_CLEAR", "1").strip() == "1"
    require_readiness_coverage = os.getenv("MASTER_PROMOTION_REQUIRE_READINESS_COVERAGE", "1").strip() == "1"
    require_paper_feedback_floor = os.getenv("MASTER_PROMOTION_REQUIRE_PAPER_FEEDBACK_FLOOR", "1").strip() == "1"

    if require_health_gate_clear:
        health = _safe_json_load(health_gates_path)
        if not health:
            failures.append("health_gates:missing")
        elif bool(health.get("hard_gate_triggered", False)):
            failures.append("health_gates:hard_gate_triggered")

    if require_readiness_coverage:
        readiness = _safe_json_load(promotion_readiness_path)
        if not readiness:
            failures.append("promotion_readiness:missing")
        else:
            thresholds = readiness.get("thresholds") if isinstance(readiness.get("thresholds"), dict) else {}
            promote_ok = bool(readiness.get("promote_ok", False))
            coverage_ok = bool(readiness.get("coverage_ok", promote_ok))
            considered_bots = int(float(readiness.get("considered_bots", 0) or 0))
            min_considered_bots = _effective_min_considered(readiness, thresholds)
            if not promote_ok:
                failures.append("promotion_readiness:promote_ok=false")
            if not coverage_ok:
                failures.append("promotion_readiness:coverage_ok=false")
            if considered_bots < min_considered_bots:
                failures.append(f"promotion_readiness:considered_bots={considered_bots}<{min_considered_bots}")

    if require_paper_feedback_floor:
        paper_feedback = _paper_feedback_summary(paper_performance_path)
        min_executions = max(int(float(os.getenv("MASTER_PROMOTION_MIN_PAPER_EXECUTIONS", "24"))), 0)
        min_sleeves = max(int(float(os.getenv("MASTER_PROMOTION_MIN_PAPER_SLEEVES", "3"))), 0)
        if paper_feedback["total_executions"] < min_executions:
            failures.append(
                f"paper_feedback:executions={paper_feedback['total_executions']}<{min_executions}"
            )
        if paper_feedback["active_sleeves"] < min_sleeves:
            failures.append(
                f"paper_feedback:active_sleeves={paper_feedback['active_sleeves']}<{min_sleeves}"
            )

    return failures


SEGMENT_KEYWORDS = {
    "trend": ["trend", "breakout", "donchian", "momentum", "dmi", "relative_strength", "gap_open", "seasonal"],
    "mean_revert": ["mean_revert", "vwap", "bollinger", "keltner", "bond", "dividend", "yield", "income", "defensive", "drip", "compound", "quality", "allocator", "risk_budget"],
    "shock": ["flash", "shock", "event", "crash", "anomaly", "macro", "inflation", "pmi", "rates", "credit", "futures", "term_structure", "vol", "news"],
    "liquidity": ["liquidity", "spread", "order_flow", "microstructure", "execution", "latency", "position_1m_3m"],
}

_OPERATOR_NOTE_SEGMENT_HINTS = {
    "guard_heavy_regime": "shock",
    "defensive_dividend_repeat": "mean_revert",
    "futures_event_risk": "shock",
    "crypto_throttle_repeat": "liquidity",
    "stock_crypto_overlap": "other",
}


def _segment_bot_id(bot_id: str) -> str:
    b = (bot_id or "").lower()
    for seg, keys in SEGMENT_KEYWORDS.items():
        if any(k in b for k in keys):
            return seg
    return "other"


def _apply_regime_focus(targets: list[str], regime_focus: str) -> list[str]:
    focus = {x.strip().lower() for x in str(regime_focus or "").split(",") if x.strip()}
    if not focus:
        return targets
    return [t for t in targets if _segment_bot_id(_normalized_bot_id_from_script(t)) in focus]


def _apply_regime_balanced_order(targets: list[str]) -> list[str]:
    if not targets:
        return targets
    buckets: dict[str, list[str]] = {}
    for t in targets:
        seg = _segment_bot_id(_normalized_bot_id_from_script(t))
        buckets.setdefault(seg, []).append(t)

    for k in buckets:
        buckets[k] = sorted(buckets[k], key=lambda x: _normalized_bot_id_from_script(x))

    ordered: list[str] = []
    seg_order = ["trend", "mean_revert", "shock", "liquidity", "other"]
    while True:
        moved = False
        for seg in seg_order:
            rows = buckets.get(seg, [])
            if rows:
                ordered.append(rows.pop(0))
                moved = True
        if not moved:
            break
    return ordered


def _apply_excluded_bot_ids(targets: list[str], excluded_bot_ids: str) -> list[str]:
    excluded = {x.strip().lower() for x in str(excluded_bot_ids or "").split(",") if x.strip()}
    if not excluded:
        return targets
    return [t for t in targets if _normalized_bot_id_from_script(t) not in excluded]


def _apply_included_bot_ids(targets: list[str], included_bot_ids: str) -> list[str]:
    wanted = [x.strip().lower() for x in str(included_bot_ids or "").split(",") if x.strip()]
    if not wanted:
        return targets
    target_map = {_normalized_bot_id_from_script(t): t for t in targets}
    out: list[str] = []
    seen: set[str] = set()
    for bot_id in wanted:
        target = target_map.get(bot_id)
        if target and bot_id not in seen:
            out.append(target)
            seen.add(bot_id)
    return out


def _should_include_deleted_targets(args: argparse.Namespace, explicit_include_requested: bool) -> bool:
    return bool(getattr(args, "force_all_targets", False) or args.include_deleted or explicit_include_requested)


def _dedupe_targets_preserve_order(targets: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for target in targets:
        bot_id = _normalized_bot_id_from_script(target)
        if bot_id in seen:
            continue
        seen.add(bot_id)
        out.append(target)
    return out


def _reshape_target_queue(
    targets: list[str],
    *,
    allow_auto_queue_reshaping: bool,
    regime_focus: str,
    regime_balance: bool,
    exclude_bot_ids: str,
    canary_priority_file: str,
    canary_priority_top_n: int,
    distillation_priority: bool,
    distill_assign_map: dict[str, dict],
    distillation_extra_pass: int,
    new_bot_boost: bool,
    new_bot_targets: list[str],
    new_bot_extra_pass: int,
) -> tuple[list[str], int, int]:
    canary_priority_selected = 0
    distill_selected = 0

    if allow_auto_queue_reshaping:
        if regime_focus:
            focused = _apply_regime_focus(targets, str(regime_focus))
            if focused:
                targets = focused

        if regime_balance:
            targets = _apply_regime_balanced_order(targets)

        targets = _apply_excluded_bot_ids(targets, str(exclude_bot_ids or ""))

        targets, canary_priority_selected = _apply_canary_priority(
            targets,
            diagnostics_file=str(canary_priority_file),
            top_n=int(canary_priority_top_n),
        )

        if distillation_priority and distill_assign_map:
            targets, distill_selected = _prioritize_targets_for_distillation(targets, distill_assign_map)

        if distillation_priority and distill_assign_map and int(distillation_extra_pass) > 0:
            student_targets = [t for t in targets if _normalized_bot_id_from_script(t) in distill_assign_map]
            extra_n = min(max(int(distillation_extra_pass), 0), len(student_targets))
            if extra_n > 0:
                targets = targets + student_targets[:extra_n]

        if new_bot_boost and new_bot_targets and int(new_bot_extra_pass) > 0:
            extra_new_n = min(max(int(new_bot_extra_pass), 0), len(new_bot_targets))
            if extra_new_n > 0:
                targets = targets + new_bot_targets[:extra_new_n]
    else:
        targets = _apply_excluded_bot_ids(targets, str(exclude_bot_ids or ""))
    targets = _dedupe_targets_preserve_order(targets)

    return targets, int(canary_priority_selected), int(distill_selected)


def _resolve_data_divergence_file(scope: str, fallback_file: str) -> tuple[str, str]:
    token = str(scope or "").strip().lower()
    if token in {"", "all", "global", "all_profiles"}:
        return fallback_file or DATA_DIVERGENCE_GLOBAL_FILE, "all_profiles"
    if token in {"bond", "bond_profile", "bond-only", "bond_only"}:
        return DATA_DIVERGENCE_BOND_FILE, "bond_profile"
    if token in {"non_bond", "non-bond", "nonbond", "non_bond_profiles"}:
        return DATA_DIVERGENCE_NON_BOND_FILE, "non_bond_profiles"
    return fallback_file or DATA_DIVERGENCE_GLOBAL_FILE, token


def _load_json_file(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _latest_training_diagnostic_path(bot_id: str) -> str:
    safe_bot_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(bot_id or "").strip())
    return os.path.join(TRAINING_DIAGNOSTICS_DIR, f"{safe_bot_id}_latest.json")


def _training_diagnostic_state(bot_id: str) -> dict[str, Any]:
    path = _latest_training_diagnostic_path(bot_id)
    payload = _load_json_file(path)
    if not payload:
        return {}
    attempts = payload.get("autofix_attempts") if isinstance(payload.get("autofix_attempts"), list) else []
    attempt_rows = [row for row in attempts if isinstance(row, dict)]
    sample_count = _coerce_int(payload.get("sample_count"), 0)
    eligible_sequences = _coerce_int(payload.get("eligible_sequences"), 0)
    observation_count = _coerce_int(payload.get("observation_count"), 0)
    sequence_count = _coerce_int(payload.get("sequence_count"), 0)
    insufficiency_reason = str(payload.get("insufficiency_reason") or "").strip()
    for row in attempt_rows:
        sample_count = max(sample_count, _coerce_int(row.get("samples"), 0), _coerce_int(row.get("sample_count"), 0))
        eligible_sequences = max(eligible_sequences, _coerce_int(row.get("eligible_sequences"), 0))
        observation_count = max(observation_count, _coerce_int(row.get("observation_count"), 0))
        sequence_count = max(sequence_count, _coerce_int(row.get("sequence_count"), 0))
        if not insufficiency_reason:
            insufficiency_reason = str(row.get("insufficiency_reason") or "").strip()

    ts = _parse_ts(str(payload.get("timestamp_utc") or ""))
    age_minutes = None
    if ts is not None:
        age_minutes = max((datetime.now(timezone.utc) - ts).total_seconds() / 60.0, 0.0)
    return {
        "bot_id": str(bot_id or "").strip().lower(),
        "diagnostics_path": path,
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "status": str(payload.get("status") or ""),
        "quality_deferred": bool(payload.get("quality_deferred", False)),
        "age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
        "sample_count": int(sample_count),
        "eligible_sequences": int(eligible_sequences),
        "observation_count": int(observation_count),
        "sequence_count": int(sequence_count),
        "positive_rate": _coerce_float(payload.get("positive_rate"), 0.0),
        "insufficiency_reason": insufficiency_reason,
        "quality_failures": payload.get("quality_failures") if isinstance(payload.get("quality_failures"), list) else [],
        "failure_categories": payload.get("failure_categories") if isinstance(payload.get("failure_categories"), list) else [],
    }


def _batch_readiness_prefilter_enabled(retrain_profile: str) -> bool:
    raw = str(os.getenv("RETRAIN_BATCH_READINESS_PREFILTER", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return str(retrain_profile or "").strip().lower() in {
        "coverage_batch10_canary",
        "coverage_batch20_canary",
        "coverage_batch30_canary",
    }


def _latest_bot_needs_training_evidence(bot_id: str) -> dict[str, Any]:
    global _BOT_NEEDS_EVIDENCE_CACHE, _BOT_NEEDS_EVIDENCE_CACHE_SOURCE
    clean_bot_id = str(bot_id or "").strip().lower()
    if not clean_bot_id:
        return {}
    path = os.path.join(PROJECT_ROOT, "governance", "health", "bot_needs_intelligence_latest.json")
    if _BOT_NEEDS_EVIDENCE_CACHE is None or _BOT_NEEDS_EVIDENCE_CACHE_SOURCE != path:
        _BOT_NEEDS_EVIDENCE_CACHE = {}
        _BOT_NEEDS_EVIDENCE_CACHE_SOURCE = path
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh) or {}
        except Exception:
            payload = {}
        ts = _parse_ts(str((payload or {}).get("timestamp_utc") or ""))
        age_minutes = None
        if ts is not None:
            age_minutes = max((datetime.now(timezone.utc) - ts).total_seconds() / 60.0, 0.0)
        max_age_minutes = max(
            _coerce_float(os.getenv("RETRAIN_BATCH_PREFILTER_BOT_NEEDS_MAX_AGE_MINUTES", "720"), 720.0),
            1.0,
        )
        if age_minutes is not None and age_minutes <= max_age_minutes:
            for row in payload.get("bot_needs") or []:
                if not isinstance(row, dict):
                    continue
                row_bot_id = str(row.get("bot_id") or "").strip().lower()
                if not row_bot_id:
                    continue
                evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
                _BOT_NEEDS_EVIDENCE_CACHE[row_bot_id] = {
                    "bot_id": row_bot_id,
                    "primary_need": str(row.get("primary_need") or ""),
                    "sample_count": _coerce_int(evidence.get("sample_count"), 0),
                    "eligible_sequences": _coerce_int(evidence.get("eligible_sequences"), 0),
                    "observation_count": _coerce_int(evidence.get("observation_count"), 0),
                    "diagnostic_age_hours": _coerce_float(evidence.get("diagnostic_age_hours"), 0.0),
                    "source_path": path,
                    "source_age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
                }
    return dict((_BOT_NEEDS_EVIDENCE_CACHE or {}).get(clean_bot_id) or {})


def _sample_starved_prefilter_decision(target: str, retrain_profile: str) -> dict[str, Any] | None:
    if not _batch_readiness_prefilter_enabled(retrain_profile):
        return None
    if os.getenv("RETRAIN_BATCH_PREFILTER_ALLOW_SNAPSHOT_REPAIR", "").strip().lower() in {"1", "true", "yes", "on"}:
        snapshot_file = str(os.getenv("RUNTIME_TRAIN_SNAPSHOT_FILE", "") or "").strip()
        if snapshot_file and os.path.exists(snapshot_file):
            return None
    bot_id = _normalized_bot_id_from_script(target)
    try:
        with open(target, "r", encoding="utf-8") as fh:
            target_source = fh.read()
        if "custom_label_builder=" in target_source or "custom_sample_filter=" in target_source:
            return None
    except Exception:
        pass
    state = _training_diagnostic_state(bot_id)
    if not state:
        return None
    max_age_minutes = max(_coerce_float(os.getenv("RETRAIN_BATCH_PREFILTER_MAX_AGE_MINUTES", "360"), 360.0), 1.0)
    age = state.get("age_minutes")
    if age is None or float(age) > max_age_minutes:
        return None
    sample_count = _coerce_int(state.get("sample_count"), 0)
    eligible_sequences = _coerce_int(state.get("eligible_sequences"), 0)
    insufficiency_reason = str(state.get("insufficiency_reason") or "").strip().lower()
    if sample_count <= 0 and eligible_sequences <= 0 and insufficiency_reason in {"", "sample_count", "sample_count_fast_fail_zero_sample"}:
        needs_evidence = _latest_bot_needs_training_evidence(bot_id)
        if (
            _coerce_int(needs_evidence.get("sample_count"), 0) > 0
            or _coerce_int(needs_evidence.get("eligible_sequences"), 0) > 0
        ):
            return None
        return {
            "bot_id": bot_id,
            "target": target,
            "status": "prefiltered_sample_starved",
            "reason": "fresh_zero_sample_diagnostic",
            "sample_count": int(sample_count),
            "eligible_sequences": int(eligible_sequences),
            "observation_count": _coerce_int(state.get("observation_count"), 0),
            "sequence_count": _coerce_int(state.get("sequence_count"), 0),
            "age_minutes": age,
            "diagnostics_path": state.get("diagnostics_path"),
            "recommended_next_step": "repair labels/sample eligibility or collect more targeted observations before retraining this bot",
        }
    return None


def _fresh_health_payload(payload: dict, *, max_age_hours: float) -> tuple[dict, bool]:
    if not isinstance(payload, dict) or not payload:
        return {}, False
    ts = _parse_ts(str(payload.get("timestamp_utc") or ""))
    if ts is None:
        return payload, False
    age_hours = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 3600.0
    return payload, age_hours <= max(max_age_hours, 0.0)


def _safe_write_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path


def _write_training_diagnostic_artifact(base_name: str, payload: dict, *, dry_run: bool) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(TRAINING_DIAGNOSTICS_DIR, exist_ok=True)
    out_path = os.path.join(TRAINING_DIAGNOSTICS_DIR, f"{base_name}_{ts}.json")
    _safe_write_json(out_path, payload)
    latest_name = f"{base_name}_dry_run_latest.json" if dry_run else f"{base_name}_latest.json"
    _safe_write_json(os.path.join(TRAINING_DIAGNOSTICS_DIR, latest_name), payload)
    return out_path


def _build_retrain_input_feature_diagnostics(dataset_path: str) -> dict:
    dataset_obj = _load_json_file(dataset_path)
    feature_names_raw = dataset_obj.get("feature_names") if isinstance(dataset_obj.get("feature_names"), list) else []
    feature_names = [str(name) for name in feature_names_raw if str(name)]
    rows = dataset_obj.get("data") if isinstance(dataset_obj.get("data"), list) else []
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    tracked: dict[str, dict[str, Any]] = {}
    for feature_name in ADVANCED_RETRAIN_DIAGNOSTIC_FEATURES:
        idx = feature_index.get(feature_name)
        tracked[feature_name] = {
            "present_in_schema": idx is not None,
            "schema_index": int(idx) if idx is not None else -1,
            "sample_count": 0,
            "nonzero_count": 0,
            "high_count": 0,
            "mean_norm": 0.0,
            "mean_abs_norm": 0.0,
            "high_signal_label_counts": {"negative": 0, "neutral": 0, "positive": 0},
            "nonzero_label_counts": {"negative": 0, "neutral": 0, "positive": 0},
        }

    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "neutral").strip().lower()
        if label not in {"negative", "neutral", "positive"}:
            label = "neutral"
        values = row.get("features") or []
        if not isinstance(values, list):
            continue
        for feature_name, stats in tracked.items():
            idx = stats.get("schema_index", -1)
            if not isinstance(idx, int) or idx < 0 or idx >= len(values):
                continue
            try:
                value = float(values[idx] or 0.0)
            except Exception:
                continue
            stats["sample_count"] += 1
            stats["mean_norm"] += value
            stats["mean_abs_norm"] += abs(value)
            if abs(value) > 1e-9:
                stats["nonzero_count"] += 1
                stats["nonzero_label_counts"][label] += 1
            if abs(value) >= 0.67:
                stats["high_count"] += 1
                stats["high_signal_label_counts"][label] += 1

    dataset_rows = int(dataset_obj.get("rows", 0) or 0)
    for stats in tracked.values():
        sample_count = max(int(stats.get("sample_count", 0) or 0), 1)
        mean_norm = float(stats.get("mean_norm", 0.0) or 0.0) / sample_count
        mean_abs_norm = float(stats.get("mean_abs_norm", 0.0) or 0.0) / sample_count
        nonzero_count = int(stats.get("nonzero_count", 0) or 0)
        high_count = int(stats.get("high_count", 0) or 0)
        stats["mean_norm"] = round(mean_norm, 6)
        stats["mean_abs_norm"] = round(mean_abs_norm, 6)
        stats["coverage_ratio"] = round(nonzero_count / sample_count, 6)
        stats["high_signal_ratio"] = round(high_count / sample_count, 6)
        stats["dataset_row_ratio"] = round(sample_count / max(dataset_rows, 1), 6)
        stats.pop("schema_index", None)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path if os.path.exists(dataset_path) else "",
        "dataset_rows": dataset_rows,
        "feature_dim": int(dataset_obj.get("feature_dim", 0) or 0),
        "feature_schema_version": str(dataset_obj.get("feature_schema_version") or ""),
        "tracked_feature_count": len(ADVANCED_RETRAIN_DIAGNOSTIC_FEATURES),
        "tracked_features": tracked,
    }


def _build_failed_bot_replay_summary(
    *,
    failure_details: list[dict],
    counterfactual_summary: dict | None = None,
    paper_performance: dict | None = None,
) -> dict:
    counterfactual = counterfactual_summary if isinstance(counterfactual_summary, dict) else _load_json_file(COUNTERFACTUAL_REPLAY_LATEST)
    paper_payload = paper_performance if isinstance(paper_performance, dict) else _load_json_file(PAPER_PERFORMANCE_PATH)

    candidate_rows = counterfactual.get("top_candidates") if isinstance(counterfactual.get("top_candidates"), list) else []
    candidates_by_profile = {
        str((row or {}).get("profile") or "").strip().lower(): row
        for row in candidate_rows
        if isinstance(row, dict) and str((row or {}).get("profile") or "").strip()
    }
    sleeve_rows = {
        str((row or {}).get("profile") or "").strip().lower(): row
        for row in (paper_payload.get("sleeve_latest") if isinstance(paper_payload.get("sleeve_latest"), list) else [])
        if isinstance(row, dict) and str((row or {}).get("profile") or "").strip()
    }

    profile_pressure: dict[str, dict[str, Any]] = {}
    bot_summaries: list[dict[str, Any]] = []
    for row in failure_details:
        bot_id = str((row or {}).get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        segment = _segment_bot_id(bot_id)
        replay_profiles = list(SEGMENT_TO_REPLAY_PROFILES.get(segment, SEGMENT_TO_REPLAY_PROFILES["other"]))
        profile_rows: list[dict[str, Any]] = []
        for profile in replay_profiles:
            sleeve = sleeve_rows.get(profile, {})
            candidate = candidates_by_profile.get(profile, {})
            profile_rows.append(
                {
                    "profile": profile,
                    "current_end_net": round(float((sleeve.get("ending_net_pnl_total", 0.0) if isinstance(sleeve, dict) else 0.0) or 0.0), 6),
                    "current_win_rate": sleeve.get("win_rate") if isinstance(sleeve, dict) else None,
                    "counterfactual_threshold_delta": float((candidate.get("threshold_delta", 0.0) if isinstance(candidate, dict) else 0.0) or 0.0),
                    "counterfactual_tradeability_floor": float((candidate.get("tradeability_floor", 0.0) if isinstance(candidate, dict) else 0.0) or 0.0),
                    "counterfactual_aggregate_net_pnl_total": round(float((candidate.get("aggregate_net_pnl_total", 0.0) if isinstance(candidate, dict) else 0.0) or 0.0), 6),
                }
            )
            bucket = profile_pressure.setdefault(
                profile,
                {
                    "profile": profile,
                    "failed_bot_count": 0,
                    "current_end_net": round(float((sleeve.get("ending_net_pnl_total", 0.0) if isinstance(sleeve, dict) else 0.0) or 0.0), 6),
                    "current_win_rate": sleeve.get("win_rate") if isinstance(sleeve, dict) else None,
                    "best_counterfactual_threshold_delta": float((candidate.get("threshold_delta", 0.0) if isinstance(candidate, dict) else 0.0) or 0.0),
                    "best_counterfactual_tradeability_floor": float((candidate.get("tradeability_floor", 0.0) if isinstance(candidate, dict) else 0.0) or 0.0),
                    "best_counterfactual_aggregate_net_pnl_total": round(float((candidate.get("aggregate_net_pnl_total", 0.0) if isinstance(candidate, dict) else 0.0) or 0.0), 6),
                },
            )
            bucket["failed_bot_count"] += 1
        bot_summaries.append(
            {
                "bot_id": bot_id,
                "segment": segment,
                "reason": str((row or {}).get("reason") or "").strip(),
                "recommended_profiles": replay_profiles,
                "profile_summaries": profile_rows,
            }
        )

    ranked_profiles = sorted(
        profile_pressure.values(),
        key=lambda item: (-int(item.get("failed_bot_count", 0) or 0), float(item.get("current_end_net", 0.0) or 0.0)),
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "failed_bot_count": len(bot_summaries),
        "profiles_reviewed": [str(row.get("profile") or "") for row in ranked_profiles],
        "profile_pressure": ranked_profiles,
        "bot_summaries": bot_summaries,
    }


def _failure_is_insufficient_data(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return "insufficient_runtime_training_data" in text or "insufficient_runtime_training_side_samples" in text


def _failure_is_deferred_sample_starved(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return "defer_runtime_training_until_more_data" in text


def _failure_is_deferred_quality_guard(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return (
        "defer_training_quality_guard" in text
        or "deferred_quality_guard" in text
        or "runtime_training_quality_guard_failed" in text
        or "synthetic_training_quality_guard_failed" in text
    )


def _failure_is_target_timeout(rc: int, reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return int(rc) == 124 or "[timeout] command exceeded" in text or "timeoutexpired" in text


def _diagnostic_state_is_deferred_quality_guard(state: dict[str, Any]) -> bool:
    status = str((state or {}).get("status") or "").strip().lower()
    categories = (state or {}).get("failure_categories")
    category_set = {str(item or "").strip().lower() for item in categories} if isinstance(categories, list) else set()
    return (
        status == "deferred_quality_guard"
        or bool((state or {}).get("quality_deferred", False))
        or "quality_guard_failure" in category_set
    )


def _insufficient_data_retry_overrides(target: str, attempt_index: int) -> dict[str, str]:
    bot_id = _normalized_bot_id_from_script(target)
    base_lookback = 28 if any(token in bot_id for token in ("intraday", "proxy", "simple", "dmi", "choppy")) else 45
    lookback_days = base_lookback if attempt_index <= 0 else max(base_lookback, 60)
    overrides = {
        "RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA": "1",
        "RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE": str(int(lookback_days)),
        "RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE": "1",
        "RUNTIME_TRAIN_AUTOFIX_ALLOW_SYMBOL_SCOPE_BROADEN": "1",
        "RUNTIME_TRAIN_AUTOFIX_MAX_LOOKBACK_DAYS": str(max(int(lookback_days), 60)),
        "RUNTIME_TRAIN_MIN_CONFIDENCE_OVERRIDE": "0.0",
    }
    if attempt_index >= 1:
        overrides["RUNTIME_TRAIN_AUTOFIX_MIN_CONFIDENCE_FLOOR"] = "0.0"
    return overrides


def _runtime_snapshot_family_floor(bot_id: str) -> int:
    text = str(bot_id or "").strip().lower()
    if any(tok in text for tok in ("dividend", "yield_trap", "compounder")):
        return 90
    if any(tok in text for tok in ("long_interval", "long_term", "core_etf", "quality_compound")):
        return 120
    if any(tok in text for tok in ("bond", "rates", "treasury", "duration")):
        return 90
    if any(tok in text for tok in ("futures", "order_book", "followthrough", "curve", "basis")):
        return 75
    if any(tok in text for tok in ("options", "iv_", "put_call", "vol_surface", "gamma")):
        return 60
    if any(tok in text for tok in ("intraday", "ultrafast", "proxy", "simple", "dmi", "choppy", "news_shocks", "flash")):
        return 60
    return 75


def _registry_bot_role(bot_id: str) -> str:
    payload = _load_json_file(REGISTRY_PATH)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    wanted = str(bot_id or "").strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("bot_id") or "").strip().lower() == wanted:
            return str(row.get("bot_role") or "").strip().lower()
    return ""


def _runtime_snapshot_role_bonus(bot_role: str) -> int:
    role = str(bot_role or "").strip().lower()
    if role == "infrastructure_sub_bot":
        return 14
    if role in {"options_sub_bot", "futures_sub_bot"}:
        return 7
    return 0


def _target_runtime_snapshot_lookback_days(target: str) -> int:
    try:
        source = open(target, "r", encoding="utf-8").read()
    except Exception:
        return 0
    if "train_runtime_indicator_bot" not in source and "train_crypto_runtime_bot" not in source:
        return 0
    bot_id = _normalized_bot_id_from_script(target)
    declared = 14
    match = re.search(r"\blookback_days\s*=\s*(\d+)", source)
    if match:
        declared = int(match.group(1))
    role = _registry_bot_role(bot_id)
    profile_floor = _runtime_snapshot_family_floor(bot_id) + _runtime_snapshot_role_bonus(role)
    return max(int(declared), int(profile_floor), 1)


def _required_runtime_snapshot_lookback_days(targets: list[str], configured_lookback_days: int) -> int:
    required = max(int(configured_lookback_days), 1)
    for target in targets:
        required = max(required, _target_runtime_snapshot_lookback_days(target))
    return int(required)


def _write_paper_hard_example_pack(
    *,
    paper_performance_file: str,
    out_file: str,
    top_strategies: int = 24,
) -> tuple[str, dict]:
    payload = _load_json_file(paper_performance_file)
    sleeve_rows = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    strategies: list[dict] = []
    weak_profiles: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sleeve in sleeve_rows:
        if not isinstance(sleeve, dict):
            continue
        profile = str(sleeve.get("profile") or "").strip().lower()
        if not profile:
            continue
        ending_net = float(sleeve.get("ending_net_pnl_total", 0.0) or 0.0)
        win_rate = sleeve.get("win_rate")
        losing_count = int(sleeve.get("losing_strategy_count", 0) or 0)
        winning_count = int(sleeve.get("winning_strategy_count", 0) or 0)
        is_weak = ending_net < 0.0 or (win_rate is not None and float(win_rate) < 0.45) or losing_count > winning_count
        if not is_weak:
            continue
        weak_profiles.append(
            {
                "profile": profile,
                "ending_net_pnl_total": round(ending_net, 6),
                "win_rate": (round(float(win_rate), 6) if win_rate is not None else None),
                "losing_strategy_count": losing_count,
                "winning_strategy_count": winning_count,
            }
        )
        for row in sleeve.get("top_losing_strategies") or []:
            if not isinstance(row, dict):
                continue
            strategy = str(row.get("strategy") or "").strip()
            if not strategy:
                continue
            key = (profile, strategy)
            if key in seen:
                continue
            seen.add(key)
            strategies.append(
                {
                    "profile": profile,
                    "strategy": strategy,
                    "ending_net_pnl_total": round(float(row.get("ending_net_pnl_total", 0.0) or 0.0), 6),
                    "trade_count": int(abs(float(row.get("ending_net_pnl_total", 0.0) or 0.0)) // 1) + 1,
                    "source": "paper_performance_latest",
                }
            )
    strategies.sort(key=lambda row: (float(row.get("ending_net_pnl_total", 0.0) or 0.0), row.get("strategy", "")))
    strategies = strategies[: max(int(top_strategies), 1)]
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "weak_profile_count": int(len(weak_profiles)),
        "strategy_count": int(len(strategies)),
        "weak_profiles": weak_profiles,
        "strategies": strategies,
    }
    return _safe_write_json(out_file, out), out


def _run_optional_json_artifact(
    *,
    script_path: str,
    extra_args: list[str] | None,
    dry_run: bool,
    env: dict[str, str],
    extra_nice: int = 0,
) -> tuple[int, dict]:
    if not script_path or (not os.path.exists(script_path)):
        return 0, {}
    cmd = [VENV_PY, script_path]
    if extra_args:
        cmd.extend(extra_args)
    if "--json" not in cmd:
        cmd.append("--json")
    rc, stdout_text, _stderr_text = run_cmd_capture(cmd, dry_run, env, extra_nice=extra_nice)
    try:
        payload = json.loads(stdout_text.strip()) if stdout_text.strip() else {}
    except Exception:
        payload = {}
    return rc, payload if isinstance(payload, dict) else {}


def _log_schema_version() -> int:
    try:
        return max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1)
    except Exception:
        return 2


def _load_retrain_operator_notes(path: str) -> dict:
    payload = _load_json_file(path)
    if not payload:
        return {}
    observations = [str(item).strip() for item in (payload.get("observations") or []) if str(item).strip()]
    training_guidance = [str(item).strip() for item in (payload.get("training_guidance") or []) if str(item).strip()]
    tags = [str(item).strip() for item in (payload.get("tags") or []) if str(item).strip()]
    out = {
        "title": str(payload.get("title", "") or "").strip(),
        "timestamp_utc": str(payload.get("timestamp_utc", "") or "").strip(),
        "timestamp_local": str(payload.get("timestamp_local", "") or "").strip(),
        "requested_by": str(payload.get("requested_by", "") or "").strip(),
        "source": str(payload.get("source", "") or "").strip(),
        "summary": str(payload.get("summary", "") or "").strip(),
        "tags": tags,
        "observations": observations,
        "training_guidance": training_guidance,
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
    }
    return {k: v for k, v in out.items() if v not in ("", [], {}, None)}


def _derive_regime_focus_from_operator_notes(path: str, top_n: int = 2) -> str:
    notes = _load_retrain_operator_notes(path)
    if not notes:
        return ""
    picks: list[str] = []
    for tag in notes.get("tags", []) or []:
        seg = _OPERATOR_NOTE_SEGMENT_HINTS.get(str(tag).strip().lower())
        if seg and seg not in picks:
            picks.append(seg)
    text_blobs = [
        str(notes.get("summary", "") or ""),
        *[str(item) for item in (notes.get("observations") or [])],
        *[str(item) for item in (notes.get("training_guidance") or [])],
    ]
    lowered = " ".join(text_blobs).lower()
    keyword_hints = [
        ("guard-heavy", "shock"),
        ("event-risk", "shock"),
        ("futures", "shock"),
        ("defensive", "mean_revert"),
        ("dividend", "mean_revert"),
        ("crypto throttle", "liquidity"),
        ("risk-control", "liquidity"),
    ]
    for needle, seg in keyword_hints:
        if needle in lowered and seg not in picks:
            picks.append(seg)
    filtered = [seg for seg in picks if seg in {"trend", "mean_revert", "shock", "liquidity", "other"}]
    return ",".join(filtered[: max(int(top_n), 1)])


def _sha256_file(path: str) -> str:
    if (not path) or (not os.path.exists(path)):
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _sha256_json_obj(obj: dict) -> str:
    try:
        encoded = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        return ""


def _git_commit(project_root: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return str(proc.stdout or "").strip()
        return ""
    except Exception:
        return ""


def _latest_file(pattern: str) -> str:
    rows = sorted(glob.glob(pattern))
    return rows[-1] if rows else ""


def _load_retry_pack_priority_map(path: str) -> dict[str, float]:
    obj = _load_json_file(path)
    out: dict[str, float] = {}
    include_rows = obj.get("include_bot_ids") if isinstance(obj.get("include_bot_ids"), list) else []
    for idx, bot_id in enumerate(include_rows):
        key = str(bot_id).strip().lower()
        if not key:
            continue
        out[key] = max(out.get(key, 0.0), 20.0 - float(idx))
    failed_rows = obj.get("failed_bots") if isinstance(obj.get("failed_bots"), list) else []
    for row in failed_rows:
        if isinstance(row, dict):
            key = str(row.get("bot_id") or "").strip().lower()
        else:
            key = str(row).strip().lower()
        if not key:
            continue
        out[key] = max(out.get(key, 0.0), 12.0)
    return out


def _target_priority_score(
    *,
    bot_id: str,
    is_active: bool,
    age_h: float,
    retry_priority_map: dict[str, float],
    walk_forward_runs: dict[str, int],
) -> float:
    score = 0.0
    if is_active:
        score += 100.0
    if float(age_h) >= 1e8:
        score += 80.0
    score += min(max(float(age_h), 0.0) / 24.0, 60.0)
    score += float(retry_priority_map.get(bot_id, 0.0) or 0.0)
    runs = int(walk_forward_runs.get(bot_id, 0) or 0)
    score += float(max(12 - min(runs, 12), 0)) * 2.0
    return score


def _csv_token_count(raw: str) -> int:
    return len([part.strip() for part in str(raw or "").split(",") if part.strip()])


def _apply_retrain_profile_defaults(args: argparse.Namespace) -> str:
    profile = str(getattr(args, "retrain_profile", "") or "").strip().lower()
    if not profile:
        profile = "default"
    if profile == "default":
        explicit_target_count = _csv_token_count(str(getattr(args, "include_bot_ids", "") or ""))
        if 0 < explicit_target_count <= 5:
            profile = "canary"

    coverage_canary_snapshot_available = bool(
        str(
            os.getenv("RETRAIN_COVERAGE_CANARY_SNAPSHOT_FILE", "")
            or os.getenv("RUNTIME_TRAIN_SNAPSHOT_FILE", "")
        ).strip()
        or os.path.exists(RUNTIME_TRAINING_SNAPSHOT_LATEST)
    )

    coverage_profiles = {
        "coverage_micro_canary",
        "coverage_small_canary",
        "coverage_canary",
        "coverage_batch10_canary",
        "coverage_batch20_canary",
        "coverage_batch30_canary",
    }

    if profile in coverage_profiles:
        args.counterfactual_replay = False
        args.paper_hard_example_pack = False
        args.require_sample_quotas = False
        args.new_bot_boost = False
        args.build_runtime_training_snapshot = False
        args.runtime_training_snapshot_prefer_sqlite = False
        args.runtime_train_use_snapshot = bool(coverage_canary_snapshot_available)
        args.runtime_train_prefer_sqlite = bool(coverage_canary_snapshot_available)
        args.runtime_train_fast_fail_zero_sample_attempts = max(int(args.runtime_train_fast_fail_zero_sample_attempts), 2)
        if profile in {"coverage_batch20_canary", "coverage_batch30_canary"}:
            args.runtime_train_fast_fail_zero_sample_attempts = 1
        target_timeout = (
            600
            if profile == "coverage_micro_canary"
            else 720
            if profile == "coverage_small_canary"
            else 900
            if profile == "coverage_canary"
            else 900
            if profile == "coverage_batch10_canary"
            else 1200
            if profile == "coverage_batch20_canary"
            else 900
        )
        if int(args.target_timeout_seconds) <= 0 or int(args.target_timeout_seconds) > target_timeout:
            args.target_timeout_seconds = target_timeout
        args.cold_lane_retrain_extras = False
        args.auto_insufficient_data_retry = False

    if profile in {"canary", "fast_canary"}:
        args.counterfactual_replay = False
        args.paper_hard_example_pack = False
        args.require_sample_quotas = False
        args.new_bot_boost = False
        args.build_runtime_training_snapshot = False
        args.runtime_training_snapshot_prefer_sqlite = False
        args.runtime_train_use_snapshot = False
        args.runtime_train_prefer_sqlite = False
        args.runtime_train_fast_fail_zero_sample_attempts = max(int(args.runtime_train_fast_fail_zero_sample_attempts), 2)
        args.target_timeout_seconds = max(int(args.target_timeout_seconds), 900)
        args.cold_lane_retrain_extras = False

    if profile in {"fast", "fast_daytime", "daytime"}:
        args.counterfactual_replay = False
        args.paper_hard_example_pack = False
        args.require_sample_quotas = False
        args.new_bot_boost = False
        args.build_runtime_training_snapshot = True
        args.runtime_training_snapshot_prefer_sqlite = True
        args.runtime_train_use_snapshot = True
        args.runtime_train_prefer_sqlite = True
        args.runtime_train_fast_fail_zero_sample_attempts = max(int(args.runtime_train_fast_fail_zero_sample_attempts), 2)
        args.cold_lane_retrain_extras = False
    elif profile in {"full", "full_overnight", "overnight"}:
        args.counterfactual_replay = True
        args.paper_hard_example_pack = True
        args.require_sample_quotas = True
        args.build_runtime_training_snapshot = True
        args.runtime_training_snapshot_prefer_sqlite = True
        args.runtime_train_use_snapshot = True
        args.runtime_train_prefer_sqlite = True
        args.cold_lane_retrain_extras = True
    return profile


def _runtime_training_snapshot_preflight_failure(
    summary: dict[str, Any],
    *,
    min_sequences: int,
    min_rows: int,
) -> str:
    if not isinstance(summary, dict) or not summary:
        return "snapshot_missing"
    sequence_count = int(summary.get("sequence_count", 0) or 0)
    row_count = int(summary.get("row_count", 0) or 0)
    if sequence_count < max(int(min_sequences), 0):
        return f"snapshot_sequence_count_below_floor:{sequence_count}<{int(min_sequences)}"
    if row_count < max(int(min_rows), 0):
        return f"snapshot_row_count_below_floor:{row_count}<{int(min_rows)}"
    return ""


def _build_retrain_lineage(
    *,
    stage: str,
    registry_path: str,
    registry_backup_path: str,
    target_count: int,
    retrain_profile: str = "",
) -> dict:
    dataset_obj = _load_json_file(TRADE_BEHAVIOR_DATASET)
    dataset_lineage = dataset_obj.get("lineage") if isinstance(dataset_obj.get("lineage"), dict) else {}
    hard_examples = _load_json_file(PAPER_HARD_EXAMPLES_PATH)
    counterfactual = _load_json_file(COUNTERFACTUAL_REPLAY_LATEST)
    runtime_snapshot = _load_json_file(RUNTIME_TRAINING_SNAPSHOT_LATEST)

    latest_behavior_model = _latest_file(os.path.join(PROJECT_ROOT, "models", "trade_behavior_policy_*.npz"))
    latest_behavior_log = _latest_file(os.path.join(PROJECT_ROOT, "logs", "trade_behavior_policy_*.json"))

    registry_obj = _load_json_file(registry_path)

    return {
        "lineage_schema_version": 1,
        "stage": str(stage),
        "retrain_profile": str(retrain_profile or ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_count": int(target_count),
        "git_commit": _git_commit(PROJECT_ROOT),
        "weekly_retrain_script": os.path.abspath(__file__),
        "weekly_retrain_script_sha256": _sha256_file(os.path.abspath(__file__)),
        "registry_path": registry_path,
        "registry_sha256": _sha256_file(registry_path),
        "registry_payload_sha256": _sha256_json_obj(registry_obj),
        "registry_backup_before_retrain": registry_backup_path if os.path.exists(registry_backup_path) else "",
        "registry_backup_before_retrain_sha256": _sha256_file(registry_backup_path),
        "trade_behavior_dataset": TRADE_BEHAVIOR_DATASET if os.path.exists(TRADE_BEHAVIOR_DATASET) else "",
        "trade_behavior_dataset_sha256": _sha256_file(TRADE_BEHAVIOR_DATASET),
        "trade_behavior_feature_schema_version": str(dataset_lineage.get("feature_schema_version") or ""),
        "trade_behavior_dataset_payload_sha256": str(dataset_lineage.get("output_payload_sha256") or ""),
        "trade_behavior_dataset_builder_script": str(dataset_lineage.get("builder_script") or ""),
        "trade_behavior_dataset_builder_script_sha256": str(dataset_lineage.get("builder_script_sha256") or ""),
        "trade_behavior_dataset_builder_git_commit": str(dataset_lineage.get("git_commit") or ""),
        "trade_behavior_model_latest": latest_behavior_model,
        "trade_behavior_model_latest_sha256": _sha256_file(latest_behavior_model),
        "trade_behavior_log_latest": latest_behavior_log,
        "trade_behavior_log_latest_sha256": _sha256_file(latest_behavior_log),
        "paper_hard_examples_latest": PAPER_HARD_EXAMPLES_PATH if os.path.exists(PAPER_HARD_EXAMPLES_PATH) else "",
        "paper_hard_examples_sha256": _sha256_file(PAPER_HARD_EXAMPLES_PATH),
        "paper_hard_example_strategy_count": int(hard_examples.get("strategy_count", 0) or 0),
        "counterfactual_replay_latest": COUNTERFACTUAL_REPLAY_LATEST if os.path.exists(COUNTERFACTUAL_REPLAY_LATEST) else "",
        "counterfactual_replay_sha256": _sha256_file(COUNTERFACTUAL_REPLAY_LATEST),
        "counterfactual_replay_profiles": list(counterfactual.get("profiles_reviewed") or []),
        "retrain_input_feature_diagnostics_latest": RETRAIN_INPUT_FEATURE_DIAGNOSTICS_LATEST if os.path.exists(RETRAIN_INPUT_FEATURE_DIAGNOSTICS_LATEST) else "",
        "retrain_input_feature_diagnostics_sha256": _sha256_file(RETRAIN_INPUT_FEATURE_DIAGNOSTICS_LATEST),
        "retrain_replay_summary_latest": RETRAIN_REPLAY_SUMMARY_LATEST if os.path.exists(RETRAIN_REPLAY_SUMMARY_LATEST) else "",
        "retrain_replay_summary_sha256": _sha256_file(RETRAIN_REPLAY_SUMMARY_LATEST),
        "runtime_training_snapshot_latest": RUNTIME_TRAINING_SNAPSHOT_LATEST if os.path.exists(RUNTIME_TRAINING_SNAPSHOT_LATEST) else "",
        "runtime_training_snapshot_sha256": _sha256_file(RUNTIME_TRAINING_SNAPSHOT_LATEST),
        "runtime_training_snapshot_rows_path": str(runtime_snapshot.get("rows_path") or ""),
        "runtime_training_snapshot_rows_sha256": _sha256_file(str(runtime_snapshot.get("rows_path") or "")),
        "runtime_training_snapshot_row_count": int(runtime_snapshot.get("row_count", 0) or 0),
        "runtime_training_snapshot_sequence_count": int(runtime_snapshot.get("sequence_count", 0) or 0),
        "collector_contracts_latest": COLLECTOR_CONTRACTS_LATEST if os.path.exists(COLLECTOR_CONTRACTS_LATEST) else "",
        "collector_contracts_sha256": _sha256_file(COLLECTOR_CONTRACTS_LATEST),
        "storage_tier_policy_latest": STORAGE_TIER_POLICY_LATEST if os.path.exists(STORAGE_TIER_POLICY_LATEST) else "",
        "storage_tier_policy_sha256": _sha256_file(STORAGE_TIER_POLICY_LATEST),
        "jsonl_discovery_manifest_latest": JSONL_DISCOVERY_MANIFEST_LATEST if os.path.exists(JSONL_DISCOVERY_MANIFEST_LATEST) else "",
        "jsonl_discovery_manifest_sha256": _sha256_file(JSONL_DISCOVERY_MANIFEST_LATEST),
    }


def _check_data_quality_floor(
    *,
    coverage_file: str,
    divergence_file: str,
    min_coverage_ratio: float,
    max_divergence_spread: float,
) -> tuple[bool, str, dict]:
    coverage = _load_json_file(coverage_file)
    divergence, divergence_fresh = _fresh_health_payload(
        _load_json_file(divergence_file),
        max_age_hours=float(os.getenv("RETRAIN_DATA_DIVERGENCE_MAX_AGE_HOURS", "8")),
    )

    coverage_ratio = float(coverage.get("coverage_ratio", 0.0) or 0.0)
    worst_spread = float(divergence.get("worst_relative_spread", 0.0) or 0.0)

    if coverage and (coverage_ratio < float(min_coverage_ratio)):
        return False, f"snapshot_coverage_ratio={coverage_ratio:.4f} < min_coverage_ratio={float(min_coverage_ratio):.4f}", {
            "coverage_ratio": coverage_ratio,
            "min_coverage_ratio": float(min_coverage_ratio),
            "worst_relative_spread": worst_spread,
            "max_divergence_spread": float(max_divergence_spread),
        }

    if divergence and (worst_spread > float(max_divergence_spread)):
        return False, f"worst_relative_spread={worst_spread:.4f} > max_divergence_spread={float(max_divergence_spread):.4f}", {
            "coverage_ratio": coverage_ratio,
            "min_coverage_ratio": float(min_coverage_ratio),
            "worst_relative_spread": worst_spread,
            "max_divergence_spread": float(max_divergence_spread),
            "divergence_timestamp_utc": divergence.get("timestamp_utc", ""),
            "divergence_fresh": divergence_fresh,
        }

    return True, "ok", {
        "coverage_ratio": coverage_ratio,
        "min_coverage_ratio": float(min_coverage_ratio),
        "worst_relative_spread": worst_spread,
        "max_divergence_spread": float(max_divergence_spread),
        "divergence_timestamp_utc": divergence.get("timestamp_utc", ""),
        "divergence_fresh": divergence_fresh,
    }


def _apply_canary_priority(targets: list[str], diagnostics_file: str, top_n: int) -> tuple[list[str], int]:
    if not targets or top_n <= 0:
        return targets, 0
    diag = _load_json_file(diagnostics_file)
    rows = diag.get("top_failing_bots") if isinstance(diag.get("top_failing_bots"), list) else []
    ids = []
    for row in rows[:max(int(top_n), 0)]:
        bot_id = str((row or {}).get("bot_id", "")).strip().lower()
        if bot_id:
            ids.append(bot_id)
    if not ids:
        return targets, 0

    wanted = set(ids)
    front = [t for t in targets if _normalized_bot_id_from_script(t) in wanted]
    rest = [t for t in targets if _normalized_bot_id_from_script(t) not in wanted]
    return front + rest, len(front)


def _registry_accuracy_map(path: str) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    out: dict[str, float] = {}
    for row in obj.get("sub_bots", []) if isinstance(obj, dict) else []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        try:
            out[bot_id] = float(row.get("test_accuracy", 0.0) or 0.0)
        except Exception:
            out[bot_id] = 0.0
    return out


def _write_retrain_scorecard(
    *,
    started_utc: str,
    ended_utc: str,
    target_count: int,
    failures: list[str],
    failure_details: list[dict],
    skipped_by_memory: list[str],
    target_outcomes: list[dict],
    prev_registry_snapshot: dict[str, float],
    curr_registry_snapshot: dict[str, float],
    prev_acc: dict[str, float],
    curr_acc: dict[str, float],
    master_update_status: str,
    data_quality_summary: dict,
    canary_priority_selected: int,
    distill_selected: int,
    retry_pack: dict | None = None,
    operator_notes: dict | None = None,
    retrain_input_diagnostics: dict | None = None,
    replay_summary: dict | None = None,
    lineage: dict | None = None,
    launch_context: dict | None = None,
    dry_run: bool = False,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "exports", "sql_reports")
    os.makedirs(out_dir, exist_ok=True)

    improved = 0
    degraded = 0
    unchanged = 0
    for bot_id, old_acc in prev_acc.items():
        if bot_id not in curr_acc:
            continue
        new_acc = curr_acc.get(bot_id, old_acc)
        if new_acc > old_acc + 1e-9:
            improved += 1
        elif new_acc < old_acc - 1e-9:
            degraded += 1
        else:
            unchanged += 1

    status_counts: dict[str, int] = {}
    for row in target_outcomes:
        s = str((row or {}).get("status", "unknown"))
        status_counts[s] = status_counts.get(s, 0) + 1

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "log_schema_version": _log_schema_version(),
        "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "target_count": int(target_count),
        "status_counts": status_counts,
        "failure_count": len(failures),
        "skipped_by_memory_count": len(skipped_by_memory),
        "master_update_status": master_update_status,
        "canary_priority_selected": int(canary_priority_selected),
        "distillation_priority_selected": int(distill_selected),
        "data_quality": data_quality_summary,
        "registry_before": prev_registry_snapshot,
        "registry_after": curr_registry_snapshot,
        "accuracy_delta": {
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
        },
        "target_outcomes": target_outcomes,
        "failures": failures,
        "failure_details": failure_details,
        "skipped_by_memory": skipped_by_memory,
    }
    if retry_pack is not None:
        payload["retry_pack"] = retry_pack
    if operator_notes:
        payload["operator_notes"] = operator_notes
    if retrain_input_diagnostics:
        payload["retrain_input_diagnostics"] = retrain_input_diagnostics
    if replay_summary:
        payload["replay_summary"] = replay_summary
    if lineage:
        payload["lineage"] = lineage
    if launch_context:
        payload["launch_context"] = launch_context

    json_path = os.path.join(out_dir, f"retrain_scorecard_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    latest_name = "retrain_scorecard_dry_run_latest.json" if dry_run else "retrain_scorecard_latest.json"
    latest_json = os.path.join(PROJECT_ROOT, "governance", "health", latest_name)
    os.makedirs(os.path.dirname(latest_json), exist_ok=True)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    md_path = os.path.join(out_dir, f"retrain_scorecard_{ts}.md")
    lines = [
        f"# Retrain Scorecard ({payload['timestamp_utc']})",
        f"- Window: {started_utc} -> {ended_utc}",
        f"- Targets: {target_count}",
        f"- Master update: {master_update_status}",
        f"- Failures: {len(failures)}",
        f"- Skipped by memory/thermal: {len(skipped_by_memory)}",
        f"- Accuracy delta: improved={improved} degraded={degraded} unchanged={unchanged}",
        f"- Canary-priority selected: {int(canary_priority_selected)}",
        f"- Distillation-priority selected: {int(distill_selected)}",
    ]
    if failure_details:
        lines.append("")
        lines.append("## Failure Details")
        for row in failure_details:
            bot_id = str((row or {}).get("bot_id", "") or "").strip() or "unknown_bot"
            rc = (row or {}).get("rc", "n/a")
            reason = str((row or {}).get("reason", "") or "").strip() or "command_failed_without_output"
            lines.append(f"- {bot_id}: rc={rc} reason={reason}")
    if retry_pack:
        lines.append("")
        lines.append("## Retry Pack")
        lines.append(f"- Include bot ids: {', '.join(retry_pack.get('include_bot_ids') or []) or 'none'}")
        lines.append(f"- Skip master update: {bool(retry_pack.get('skip_master_update', False))}")
        lines.append(f"- Distillation priority: {bool(retry_pack.get('distillation_priority', False))}")
        retry_cmd = retry_pack.get("command") or []
        if retry_cmd:
            lines.append(f"- Command: {' '.join(str(item) for item in retry_cmd)}")
    if operator_notes:
        title = str(operator_notes.get("title", "") or "").strip() or "Operator Notes"
        lines.append("")
        lines.append(f"## {title}")
        summary = str(operator_notes.get("summary", "") or "").strip()
        if summary:
            lines.append(f"- Summary: {summary}")
        tags = [str(item) for item in (operator_notes.get("tags") or []) if str(item).strip()]
        if tags:
            lines.append(f"- Tags: {', '.join(tags)}")
        observations = [str(item) for item in (operator_notes.get("observations") or []) if str(item).strip()]
        for item in observations[:8]:
            lines.append(f"- Observation: {item}")
        training_guidance = [str(item) for item in (operator_notes.get("training_guidance") or []) if str(item).strip()]
        for item in training_guidance[:8]:
            lines.append(f"- Training guidance: {item}")
    if retrain_input_diagnostics:
        tracked = retrain_input_diagnostics.get("tracked_features") if isinstance(retrain_input_diagnostics.get("tracked_features"), dict) else {}
        top_features = []
        for feature_name, stats in tracked.items():
            if not isinstance(stats, dict) or not bool(stats.get("present_in_schema", False)):
                continue
            top_features.append(
                (
                    str(feature_name),
                    float(stats.get("coverage_ratio", 0.0) or 0.0),
                    float(stats.get("high_signal_ratio", 0.0) or 0.0),
                )
            )
        top_features.sort(key=lambda item: (-item[1], -item[2], item[0]))
        lines.append("")
        lines.append("## Retrain Input Diagnostics")
        lines.append(f"- Dataset rows: {int(retrain_input_diagnostics.get('dataset_rows', 0) or 0)}")
        for feature_name, coverage_ratio, high_signal_ratio in top_features[:6]:
            lines.append(
                f"- {feature_name}: coverage={coverage_ratio:.4f} high_signal={high_signal_ratio:.4f}"
            )
    if replay_summary:
        lines.append("")
        lines.append("## Failed-Bot Replay Summary")
        for row in (replay_summary.get("profile_pressure") if isinstance(replay_summary.get("profile_pressure"), list) else [])[:6]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {str(row.get('profile') or '')}: "
                f"failed_bot_count={int(row.get('failed_bot_count', 0) or 0)} "
                f"current_end_net={float(row.get('current_end_net', 0.0) or 0.0):.6f} "
                f"counterfactual_threshold_delta={float(row.get('best_counterfactual_threshold_delta', 0.0) or 0.0):.4f}"
            )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path

def _write_training_success_marker(
    *,
    target_outcomes: list[dict],
    failures: list[str],
    failure_details: list[dict],
    skipped_by_memory: list[str],
    master_update_status: str,
    data_quality_summary: dict,
    operator_notes: dict | None = None,
    lineage: dict | None = None,
    dry_run: bool = False,
) -> str:
    trained_count = sum(1 for row in target_outcomes if str((row or {}).get("status", "")) == "trained")
    failure_count = len(failures)
    precheck_ok = str(master_update_status).startswith("updated")
    data_quality_present = bool(data_quality_summary)
    data_quality_ok = True if not data_quality_present else bool((data_quality_summary or {}).get("ok", False))
    training_completed_ok = (failure_count == 0) and (trained_count > 0)
    trained_ok_but_not_promotable = bool(training_completed_ok and (not precheck_ok))

    if failure_count > 0:
        reason = f"training_failures_present:{failure_count}"
    elif trained_count <= 0:
        reason = "no_trained_targets"
    elif not precheck_ok:
        reason = f"trained_ok_but_not_promotable:{master_update_status}"
    elif data_quality_present and not data_quality_ok:
        reason = "data_quality_not_ok"
    else:
        reason = "ok"

    confirmed = training_completed_ok and precheck_ok and data_quality_ok

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "log_schema_version": _log_schema_version(),
        "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
        "confirmed_training_success": bool(confirmed),
        "training_completed_ok": bool(training_completed_ok),
        "promotion_applied": bool(precheck_ok),
        "trained_ok_but_not_promotable": bool(trained_ok_but_not_promotable),
        "promotion_status": ("promoted" if precheck_ok else "held_out"),
        "data_quality_present": bool(data_quality_present),
        "data_quality_ok": bool(data_quality_ok),
        "reason": reason,
        "trained_count": int(trained_count),
        "failure_count": int(failure_count),
        "failure_details": list(failure_details or []),
        "skipped_by_memory_count": int(len(skipped_by_memory)),
        "master_update_status": str(master_update_status),
    }
    if operator_notes:
        payload["operator_notes"] = operator_notes
    if lineage:
        payload["lineage"] = lineage

    out_name = "training_success_dry_run_latest.json" if dry_run else "training_success_latest.json"
    out_path = os.path.join(PROJECT_ROOT, "governance", "health", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return out_path


def _categorize_failure_reason(reason: str) -> list[str]:
    text = str(reason or "").strip().lower()
    categories: list[str] = []
    if _failure_is_insufficient_data(text):
        categories.append("sample_starved")
    if "label_balance_score" in text and "label_cleanup" not in categories:
        categories.append("label_cleanup")
    if any(token in text for token in ("acted_accuracy", "long_precision", "short_precision", "precision_balance_score", "acted_coverage")):
        categories.append("threshold_calibration")
    if any(token in text for token in ("best_val_loss", "final_val_loss")):
        categories.append("symbol_narrowing")
    if "accuracy_lift_over_majority" in text:
        categories.append("family_guard_review")
    return categories


def _load_chronic_failure_bot_ids() -> set[str]:
    payload = {}
    try:
        with open(PROMOTION_BOTTLENECK_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception:
        payload = {}
    chronic: set[str] = set()
    for row in payload.get("top_failing_bots") or []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        fail_days = int(row.get("fail_days", 0) or 0)
        categories = [str(item or "").strip().lower() for item in (row.get("recommended_categories") or []) if str(item or "").strip()]
        if fail_days >= 3 or ("distillation_candidate" in categories):
            chronic.add(bot_id)
    return chronic


def _write_retry_pack(
    *,
    failures: list[str],
    failure_details: list[dict],
    master_update_status: str,
    dry_run: bool = False,
) -> dict | None:
    bot_ids = []
    for row in failure_details:
        bot_id = str((row or {}).get("bot_id", "") or "").strip().lower()
        if bot_id and bot_id not in bot_ids:
            bot_ids.append(bot_id)
    if not bot_ids:
        return None
    chronic_bot_ids = sorted([bot_id for bot_id in bot_ids if bot_id in _load_chronic_failure_bot_ids()])
    recommendation_categories: set[str] = set()
    for row in failure_details:
        for item in _categorize_failure_reason(str((row or {}).get("reason", "") or "")):
            recommendation_categories.add(item)
    if chronic_bot_ids:
        recommendation_categories.add("distillation_candidate")
    command = [
        "./scripts/ops/opsctl.sh",
        "retrain-force-targeted",
        "--include-bot-ids",
        ",".join(bot_ids),
        "--skip-master-update",
    ]
    if chronic_bot_ids:
        command.append("--distillation-priority")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "failure_count": int(len(failures)),
        "include_bot_ids": bot_ids,
        "chronic_bot_ids": chronic_bot_ids,
        "skip_master_update": True,
        "distillation_priority": bool(chronic_bot_ids),
        "master_update_status": str(master_update_status),
        "recommendation_categories": sorted(recommendation_categories),
        "command": command,
    }
    out_dir = os.path.join(PROJECT_ROOT, "exports", "sql_reports")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    retry_path = os.path.join(out_dir, f"retrain_retry_pack_{ts}.json")
    with open(retry_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    latest_name = "retrain_retry_pack_dry_run_latest.json" if dry_run else "retrain_retry_pack_latest.json"
    latest_path = os.path.join(PROJECT_ROOT, "governance", "health", latest_name)
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    payload["path"] = retry_path
    return payload


def _write_sample_starved_queue(*, target_outcomes: list[dict], dry_run: bool = False) -> dict | None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in target_outcomes or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip()
        if status not in {"deferred_sample_starved", "prefiltered_sample_starved"}:
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id or bot_id in seen:
            continue
        seen.add(bot_id)
        rows.append(
            {
                "bot_id": bot_id,
                "status": status,
                "sample_count": _coerce_int(row.get("sample_count"), 0),
                "eligible_sequences": _coerce_int(row.get("eligible_sequences"), 0),
                "observation_count": _coerce_int(row.get("observation_count"), 0),
                "sequence_count": _coerce_int(row.get("sequence_count"), 0),
                "diagnostics_path": str(row.get("diagnostics_path") or ""),
                "reason": str(row.get("reason") or ""),
                "recommended_next_step": str(
                    row.get("recommended_next_step")
                    or "repair labels/sample eligibility or collect more targeted observations before retraining this bot"
                ),
            }
        )
    bot_ids = [row["bot_id"] for row in rows]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "queue_type": "sample_starved_training_repair",
        "sample_starved_count": len(rows),
        "bot_ids": bot_ids,
        "rows": rows,
        "recommended_commands": {
            "inspect_needs": [
                "./scripts/ops/opsctl.sh",
                "bot-needs",
                "--include-bot-ids",
                ",".join(bot_ids),
                "--json",
            ]
            if bot_ids
            else [],
            "refresh_labeling_intelligence": [
                "./scripts/ops/opsctl.sh",
                "training-labeling-intelligence",
                "--apply",
                "--json",
            ],
        },
        "scaling_contract": {
            "batch20_skips_known_zero_sample_bots": True,
            "zero_sample_fast_fail_attempts": 1,
            "keeps_quality_guard_intact": True,
        },
    }
    if dry_run:
        latest_path = os.path.join(TRAINING_DIAGNOSTICS_DIR, "training_sample_starved_queue_dry_run_latest.json")
    else:
        latest_path = TRAINING_SAMPLE_STARVED_QUEUE_LATEST
    _safe_write_json(latest_path, payload)
    payload["path"] = latest_path
    return payload


def _quality_repair_axes(row: dict[str, Any]) -> list[str]:
    quality_failures = row.get("quality_failures") if isinstance(row.get("quality_failures"), list) else []
    text = " ".join([str(row.get("reason") or ""), *(str(item or "") for item in quality_failures)]).lower()
    categories = {str(item or "").strip().lower() for item in row.get("failure_categories") or [] if str(item or "").strip()}
    axes: list[str] = []
    if any(token in text for token in ("long_precision", "long_acted_count", "short_precision", "short_acted_count", "require_both_sides_precision", "precision_balance_score")):
        axes.append("side_balanced_label_depth")
    if any(token in text for token in ("acted_accuracy", "accuracy_lift_over_majority")):
        axes.append("threshold_calibration")
    if "acted_coverage" in text or "acted_coverage_tuning" in categories:
        axes.append("acted_coverage_tuning")
    if any(token in text for token in ("best_val_loss", "final_val_loss")) or "symbol_narrowing" in categories:
        axes.append("symbol_scope_or_feature_narrowing")
    if "defer_runtime_training_until_more_data" in text or "defer_until_more_data" in categories:
        axes.append("collect_representative_observations")
    if "family_guard_review" in categories:
        axes.append("family_guard_review")
    if not axes:
        axes.append("quality_guard_review")
    return list(dict.fromkeys(axes))


def _write_quality_repair_queue(*, target_outcomes: list[dict], dry_run: bool = False) -> dict | None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in target_outcomes or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip()
        failure_categories = {
            str(item or "").strip().lower()
            for item in row.get("failure_categories") or []
            if str(item or "").strip()
        }
        reason_text = str(row.get("reason") or "")
        if status not in {"deferred_quality_guard", "failed_quality_guard"} and not _failure_is_deferred_quality_guard(reason_text):
            if "quality_guard_failure" not in failure_categories:
                continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id or bot_id in seen:
            continue
        seen.add(bot_id)
        rows.append(
            {
                "bot_id": bot_id,
                "target": str(row.get("target") or ""),
                "status": status or "deferred_quality_guard",
                "sample_count": _coerce_int(row.get("sample_count"), 0),
                "eligible_sequences": _coerce_int(row.get("eligible_sequences"), 0),
                "observation_count": _coerce_int(row.get("observation_count"), 0),
                "sequence_count": _coerce_int(row.get("sequence_count"), 0),
                "repair_axes": _quality_repair_axes(row),
                "quality_failures": row.get("quality_failures") if isinstance(row.get("quality_failures"), list) else [],
                "failure_categories": sorted(failure_categories),
                "diagnostics_path": str(row.get("diagnostics_path") or ""),
                "reason": reason_text,
                "recommended_next_step": str(
                    row.get("recommended_next_step")
                    or "refresh label/depth focus, calibrate thresholds, then retry as a guarded targeted canary"
                ),
            }
        )
    if not rows:
        return None
    bot_ids = [row["bot_id"] for row in rows]
    axis_counts = Counter(axis for row in rows for axis in row.get("repair_axes", []))
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "queue_type": "quality_guard_training_repair",
        "quality_repair_count": len(rows),
        "bot_ids": bot_ids,
        "axis_counts": dict(sorted(axis_counts.items())),
        "rows": rows,
        "recommended_commands": {
            "inspect_needs": [
                "./scripts/ops/opsctl.sh",
                "bot-needs",
                "--include-bot-ids",
                ",".join(bot_ids),
                "--json",
            ],
            "targeted_data_intake": [
                "./scripts/ops/opsctl.sh",
                "training-data-intake",
                "--apply",
                "--include-bot-ids",
                ",".join(bot_ids),
                "--json",
            ],
            "refresh_labeling_intelligence": [
                "./scripts/ops/opsctl.sh",
                "training-labeling-intelligence",
                "--apply",
                "--json",
            ],
            "guarded_retry": [
                "./scripts/ops/opsctl.sh",
                "retrain-force-targeted",
                "--include-bot-ids",
                ",".join(bot_ids),
                "--retrain-profile",
                "coverage_batch30_canary",
                "--skip-master-update",
                "--runtime-train-use-snapshot",
                "--thread-cap",
                "1",
                "--memory-guard",
            ],
        },
        "scaling_contract": {
            "quality_gates_remain_authoritative": True,
            "repair_before_retry": True,
            "retry_is_targeted": True,
            "side_balance_and_coverage_failures_are_explicit": True,
        },
    }
    if dry_run:
        latest_path = os.path.join(TRAINING_DIAGNOSTICS_DIR, "training_quality_repair_queue_dry_run_latest.json")
    else:
        latest_path = TRAINING_QUALITY_REPAIR_QUEUE_LATEST
    _safe_write_json(latest_path, payload)
    payload["path"] = latest_path
    return payload


def _write_training_timeout_queue(*, target_outcomes: list[dict], dry_run: bool = False) -> dict | None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in target_outcomes or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip() != "deferred_timeout":
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id or bot_id in seen:
            continue
        seen.add(bot_id)
        rows.append(
            {
                "bot_id": bot_id,
                "target": str(row.get("target") or ""),
                "status": "deferred_timeout",
                "rc": _coerce_int(row.get("rc"), 124),
                "reason": str(row.get("reason") or ""),
                "recommended_next_step": str(
                    row.get("recommended_next_step")
                    or "rerun targeted with a wider timeout or optimize this bot's runtime training path"
                ),
            }
        )
    if not rows:
        return None
    bot_ids = [row["bot_id"] for row in rows]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "queue_type": "runtime_training_timeout_retry",
        "timeout_count": len(rows),
        "bot_ids": bot_ids,
        "rows": rows,
        "recommended_commands": {
            "targeted_retry_wider_timeout": [
                "./scripts/ops/opsctl.sh",
                "retrain-force-targeted",
                "--include-bot-ids",
                ",".join(bot_ids),
                "--skip-master-update",
                "--retrain-profile",
                "full_overnight",
                "--target-timeout-seconds",
                "3600",
                "--thread-cap",
                "1",
                "--target-workers",
                "2",
            ]
        },
        "scaling_contract": {
            "timeouts_are_deferred_not_fleet_failures": True,
            "keeps_quality_guard_intact": True,
            "retry_is_targeted": True,
        },
    }
    if dry_run:
        latest_path = os.path.join(TRAINING_DIAGNOSTICS_DIR, "training_timeout_queue_dry_run_latest.json")
    else:
        latest_path = TRAINING_TIMEOUT_QUEUE_LATEST
    _safe_write_json(latest_path, payload)
    payload["path"] = latest_path
    return payload


def _market_open_now_et(start_hour: int, end_hour: int) -> bool:
    if ZoneInfo is None:
        return False
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    wd = now_et.weekday()
    if wd >= 5:
        return False
    h = now_et.hour
    return start_hour <= h < end_hour


def _monthly_stamp_path() -> str:
    return os.path.join(PROJECT_ROOT, "governance", "monthly_prune_stamp.json")


def _monthly_prune_due() -> bool:
    stamp = _monthly_stamp_path()
    now = datetime.now(timezone.utc)
    if not os.path.exists(stamp):
        return True
    try:
        with open(stamp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        y = int(obj.get("year", 0))
        m = int(obj.get("month", 0))
        return (y, m) != (now.year, now.month)
    except Exception:
        return True


def _write_monthly_prune_stamp() -> None:
    now = datetime.now(timezone.utc)
    path = _monthly_stamp_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"year": now.year, "month": now.month, "timestamp_utc": now.isoformat()}, f, ensure_ascii=True)


def _weekly_archive_stamp_path() -> str:
    return os.path.join(PROJECT_ROOT, "governance", "weekly_model_archive_stamp.json")


def _weekly_archive_due() -> bool:
    now = datetime.now(timezone.utc)
    yw = now.isocalendar()
    year = int(yw[0])
    week = int(yw[1])

    path = _weekly_archive_stamp_path()
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        y = int(obj.get("year", 0))
        w = int(obj.get("week", 0))
        return (y, w) != (year, week)
    except Exception:
        return True


def _write_weekly_archive_stamp() -> None:
    now = datetime.now(timezone.utc)
    yw = now.isocalendar()
    path = _weekly_archive_stamp_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"year": int(yw[0]), "week": int(yw[1]), "timestamp_utc": now.isoformat()}, f, ensure_ascii=True)


def _load_deleted_bot_ids(registry_path: str) -> set[str]:
    if not os.path.exists(registry_path):
        return set()
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return set()

    out: set[str] = set()
    for row in reg.get("sub_bots", []):
        if bool(row.get("deleted_from_rotation", False)):
            bot_id = str(row.get("bot_id", "")).strip().lower()
            if bot_id:
                out.add(bot_id)
    return out


def build_targets(include_deleted: bool = False) -> list[str]:
    deleted_ids = set()
    if not include_deleted:
        deleted_ids = _load_deleted_bot_ids(REGISTRY_PATH)

    targets: list[str] = []

    v3 = os.path.join(CORE_DIR, "brain_refinery_V3.py")
    if os.path.exists(v3):
        if include_deleted or _normalized_bot_id_from_script(v3) not in deleted_ids:
            targets.append(v3)

    versioned = sorted(glob.glob(os.path.join(CORE_DIR, "brain_refinery_v*.py")))
    for script in versioned:
        if include_deleted or _normalized_bot_id_from_script(script) not in deleted_ids:
            targets.append(script)

    return targets


def _parse_size_to_gb(value: str) -> float:
    try:
        s = value.strip().upper()
        if s.endswith("G"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1024.0
        if s.endswith("K"):
            return float(s[:-1]) / (1024.0 * 1024.0)
        return float(s)
    except Exception:
        return 0.0


def _memory_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}

    try:
        proc = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False)
        out = proc.stdout or ""
        for raw in out.splitlines():
            line = raw.strip()
            lower = line.lower()
            if "free percentage" in lower:
                rhs = line.split(":", 1)[-1].strip().replace("%", "")
                snapshot["free_pct"] = float(rhs)
            elif "available percentage" in lower:
                rhs = line.split(":", 1)[-1].strip().replace("%", "")
                snapshot["available_pct"] = float(rhs)
    except Exception:
        pass

    try:
        proc = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)
        out = (proc.stdout or "").strip()
        if "used =" in out:
            used_part = out.split("used =", 1)[1].strip().split()[0]
            snapshot["swap_used_gb"] = _parse_size_to_gb(used_part)
    except Exception:
        pass

    return snapshot


def _memory_ready(min_free_pct: float, max_swap_gb: float) -> tuple[bool, str, dict[str, float]]:
    snap = _memory_snapshot()

    free_pct = snap.get("free_pct")
    available_pct = snap.get("available_pct")
    effective_free_pct = available_pct if available_pct is not None else free_pct
    if effective_free_pct is not None and effective_free_pct < min_free_pct:
        free_metric = "available_pct" if available_pct is not None else "free_pct"
        return False, f"{free_metric}={effective_free_pct:.1f} < min_free_pct={min_free_pct:.1f}", snap

    swap_gb = snap.get("swap_used_gb")
    if swap_gb is not None and swap_gb > max_swap_gb:
        # macOS can keep swap allocated long after pressure clears; allow progress when free memory is healthy.
        swap_relax_free_pct = float(os.getenv("RETRAIN_SWAP_RELAX_FREE_PCT", "38"))
        swap_relax_available_pct = float(os.getenv("RETRAIN_SWAP_RELAX_AVAILABLE_PCT", "55"))
        relax_floor = max(float(min_free_pct), float(swap_relax_free_pct))
        free_ok = free_pct is not None and free_pct >= relax_floor
        available_ok = available_pct is not None and available_pct >= swap_relax_available_pct
        if free_ok or available_ok:
            relaxed_on = (
                f"free_pct={free_pct:.1f} >= relax_floor={relax_floor:.1f}"
                if free_ok
                else f"available_pct={available_pct:.1f} >= swap_relax_available_pct={swap_relax_available_pct:.1f}"
            )
            return True, (
                f"swap_relaxed swap_used_gb={swap_gb:.2f} > max_swap_gb={max_swap_gb:.2f} "
                f"but {relaxed_on}"
            ), snap
        return False, f"swap_used_gb={swap_gb:.2f} > max_swap_gb={max_swap_gb:.2f}", snap

    return True, "ok", snap


def _thermal_snapshot() -> dict[str, float]:
    snap: dict[str, float] = {}
    try:
        proc = subprocess.run(["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for raw in out.splitlines():
            line = raw.strip()
            if "CPU_Speed_Limit" in line and "=" in line:
                snap["cpu_speed_limit"] = float(line.split("=", 1)[1].strip())
            if "Scheduler_Limit" in line and "=" in line:
                snap["scheduler_limit"] = float(line.split("=", 1)[1].strip())
    except Exception:
        pass
    return snap


def _thermal_ready(min_cpu_speed_limit: float, min_scheduler_limit: float) -> tuple[bool, str, dict[str, float]]:
    snap = _thermal_snapshot()
    csl = snap.get("cpu_speed_limit")
    if csl is not None and csl < min_cpu_speed_limit:
        return False, f"cpu_speed_limit={csl:.0f} < min_cpu_speed_limit={min_cpu_speed_limit:.0f}", snap
    sl = snap.get("scheduler_limit")
    if sl is not None and sl < min_scheduler_limit:
        return False, f"scheduler_limit={sl:.0f} < min_scheduler_limit={min_scheduler_limit:.0f}", snap
    return True, "ok", snap


def _wait_for_thermal_gate(
    *,
    enabled: bool,
    min_cpu_speed_limit: float,
    min_scheduler_limit: float,
    poll_seconds: int,
    max_wait_seconds: int,
    label: str,
    dry_run: bool,
) -> bool:
    if dry_run or not enabled:
        return True

    start = time.time()
    while True:
        ok, reason, snap = _thermal_ready(
            min_cpu_speed_limit=min_cpu_speed_limit,
            min_scheduler_limit=min_scheduler_limit,
        )
        if ok:
            return True

        waited = int(time.time() - start)
        if max_wait_seconds > 0 and waited >= max_wait_seconds:
            print(f"[ThermalGate] skip label={label} waited={waited}s reason={reason}")
            return False

        print(f"[ThermalGate] wait label={label} waited={waited}s reason={reason} snapshot={snap}")
        time.sleep(max(poll_seconds, 1))


def _wait_for_memory_gate(
    *,
    enabled: bool,
    min_free_pct: float,
    max_swap_gb: float,
    poll_seconds: int,
    max_wait_seconds: int,
    label: str,
    dry_run: bool,
) -> bool:
    if dry_run or not enabled:
        return True

    start = time.time()
    while True:
        ok, reason, snap = _memory_ready(min_free_pct=min_free_pct, max_swap_gb=max_swap_gb)
        if ok:
            return True

        waited = int(time.time() - start)
        if max_wait_seconds > 0 and waited >= max_wait_seconds:
            print(f"[MemoryGate] skip label={label} waited={waited}s reason={reason}")
            return False

        print(
            f"[MemoryGate] wait label={label} waited={waited}s reason={reason} "
            f"snapshot={snap}"
        )
        time.sleep(max(poll_seconds, 1))


def _build_child_env(thread_cap: int) -> dict[str, str]:
    env = os.environ.copy()
    cap = str(max(int(thread_cap), 1))

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(key, cap)

    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    existing_pythonpath = str(env.get("PYTHONPATH", "") or "").strip()
    pythonpath_parts = [str(PROJECT_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.extend([part for part in existing_pythonpath.split(os.pathsep) if str(part).strip()])
    deduped_pythonpath: list[str] = []
    seen_pythonpath: set[str] = set()
    for part in pythonpath_parts:
        normalized = str(part).strip()
        if not normalized or normalized in seen_pythonpath:
            continue
        seen_pythonpath.add(normalized)
        deduped_pythonpath.append(normalized)
    env["PYTHONPATH"] = os.pathsep.join(deduped_pythonpath)
    return env


def _configured_runtime_snapshot_path(env: Mapping[str, str]) -> str:
    explicit = str(env.get("RUNTIME_TRAIN_SNAPSHOT_FILE", "") or "").strip()
    if explicit:
        return explicit
    return str(env.get("RETRAIN_COVERAGE_CANARY_SNAPSHOT_FILE", "") or "").strip()


def _configured_runtime_snapshot_summary(env: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    path = _configured_runtime_snapshot_path(env)
    if path and os.path.exists(path):
        return path, _safe_json_load(path)
    return "", {}


def _mapping_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _apply_env_cap(env: dict[str, str], key: str, desired: int, *, minimum: int = 0) -> int:
    target = max(int(desired), int(minimum))
    current = _mapping_int(env.get(key, ""), 0)
    if current <= 0 or current > target:
        env[key] = str(target)
        return target
    return current


def _apply_env_floor(env: dict[str, str], key: str, desired: int, *, minimum: int = 1) -> int:
    target = max(int(desired), int(minimum))
    current = _mapping_int(env.get(key, ""), 0)
    if current < target:
        env[key] = str(target)
        return target
    return current


def _apply_retrain_profile_env_overrides(env: dict[str, str], retrain_profile: str) -> dict[str, str]:
    profile = str(retrain_profile or "").strip().lower()
    if profile not in {
        "coverage_micro_canary",
        "coverage_small_canary",
        "coverage_canary",
        "coverage_batch10_canary",
        "coverage_batch20_canary",
        "coverage_batch30_canary",
    }:
        return {}

    requested_overrides: dict[str, str] = {}
    if profile == "coverage_micro_canary":
        default_lookback = 14
        default_stride = 2
        default_samples = 2000
        default_batch_cap = 32
    elif profile == "coverage_small_canary":
        default_lookback = 30
        default_stride = 1
        default_samples = 4000
        default_batch_cap = 48
    elif profile == "coverage_batch10_canary":
        default_lookback = 45
        default_stride = 1
        default_samples = 6000
        default_batch_cap = 64
    elif profile == "coverage_batch20_canary":
        default_lookback = 60
        default_stride = 1
        default_samples = 8000
        default_batch_cap = 64
    elif profile == "coverage_batch30_canary":
        default_lookback = 60
        default_stride = 1
        default_samples = 9000
        default_batch_cap = 64
    else:
        default_lookback = 45
        default_stride = 1
        default_samples = 6000
        default_batch_cap = 64
    lookback_cap = max(_mapping_int(os.getenv("RETRAIN_COVERAGE_CANARY_LOOKBACK_CAP_DAYS", str(default_lookback)), default_lookback), 1)
    stride_floor = max(_mapping_int(os.getenv("RETRAIN_COVERAGE_CANARY_SAMPLE_STRIDE", str(default_stride)), default_stride), 1)
    max_samples = max(_mapping_int(os.getenv("RETRAIN_COVERAGE_CANARY_MAX_SAMPLES", str(default_samples)), default_samples), 0)
    batch_size_cap = max(_mapping_int(os.getenv("RETRAIN_COVERAGE_CANARY_BATCH_SIZE_CAP", str(default_batch_cap)), default_batch_cap), 32)

    requested_overrides["RUNTIME_TRAIN_LOOKBACK_DAYS_CAP"] = str(
        _apply_env_cap(env, "RUNTIME_TRAIN_LOOKBACK_DAYS_CAP", lookback_cap, minimum=1)
    )
    requested_overrides["RUNTIME_TRAIN_AUTOFIX_MAX_LOOKBACK_DAYS"] = str(
        _apply_env_cap(env, "RUNTIME_TRAIN_AUTOFIX_MAX_LOOKBACK_DAYS", lookback_cap, minimum=1)
    )
    env["RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR"] = str(int(stride_floor))
    env["RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE"] = str(int(stride_floor))
    requested_overrides["RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR"] = str(int(stride_floor))
    requested_overrides["RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE"] = str(int(stride_floor))
    requested_overrides["RUNTIME_TRAIN_MAX_SAMPLES"] = str(
        _apply_env_cap(env, "RUNTIME_TRAIN_MAX_SAMPLES", max_samples, minimum=0)
    )
    requested_overrides["RUNTIME_TRAIN_BATCH_SIZE_CAP"] = str(
        _apply_env_cap(env, "RUNTIME_TRAIN_BATCH_SIZE_CAP", batch_size_cap, minimum=32)
    )
    env["RUNTIME_TRAIN_AUTOFIX_ALLOW_SYMBOL_SCOPE_BROADEN"] = "1"
    env["RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA"] = "1"
    requested_overrides["RUNTIME_TRAIN_AUTOFIX_ALLOW_SYMBOL_SCOPE_BROADEN"] = "1"
    requested_overrides["RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA"] = "1"
    return requested_overrides


def _apply_nice(nice_value: int) -> None:
    if nice_value <= 0:
        return
    try:
        os.nice(nice_value)
        print(f"Applied process nice={nice_value} for retrain wrapper and child jobs")
    except Exception as exc:
        print(f"WARN: could not apply nice={nice_value}: {exc}")


def _child_output_quiet(env: dict[str, str]) -> bool:
    return str((env or {}).get("RETRAIN_QUIET_CHILD_OUTPUT", os.getenv("RETRAIN_QUIET_CHILD_OUTPUT", "0"))).strip() == "1"


def _quiet_tail(text: str, max_lines: int = 40) -> str:
    lines = str(text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def run_cmd(cmd: list[str], dry_run: bool, env: dict[str, str], extra_nice: int = 0) -> int:
    full_cmd = cmd
    if extra_nice > 0:
        full_cmd = ["/usr/bin/nice", "-n", str(extra_nice)] + cmd
    print("$ " + " ".join(full_cmd))
    if dry_run:
        return 0
    if _child_output_quiet(env):
        proc = subprocess.run(full_cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            stdout_text = _quiet_tail(proc.stdout)
            stderr_text = _quiet_tail(proc.stderr)
            if stdout_text:
                print(stdout_text, end="\n")
            if stderr_text:
                print(stderr_text, end="\n", file=sys.stderr)
        return proc.returncode
    proc = subprocess.run(full_cmd, cwd=PROJECT_ROOT, env=env)
    return proc.returncode


def run_cmd_capture(
    cmd: list[str],
    dry_run: bool,
    env: dict[str, str],
    extra_nice: int = 0,
    timeout_seconds: int = 0,
) -> tuple[int, str, str]:
    full_cmd = cmd
    if extra_nice > 0:
        full_cmd = ["/usr/bin/nice", "-n", str(extra_nice)] + cmd
    print("$ " + " ".join(full_cmd))
    if dry_run:
        return 0, "", ""
    try:
        proc = subprocess.run(
            full_cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_seconds), 0) or None,
        )
        stdout_text = str(proc.stdout or "")
        stderr_text = str(proc.stderr or "")
        if _child_output_quiet(env):
            if proc.returncode != 0:
                stdout_tail = _quiet_tail(stdout_text)
                stderr_tail = _quiet_tail(stderr_text)
                if stdout_tail:
                    print(stdout_tail, end="\n")
                if stderr_tail:
                    print(stderr_tail, end="\n", file=sys.stderr)
        else:
            if stdout_text:
                print(stdout_text, end="")
            if stderr_text:
                print(stderr_text, end="", file=sys.stderr)
        return proc.returncode, stdout_text, stderr_text
    except subprocess.TimeoutExpired as exc:
        stdout_text = str(exc.stdout or "")
        stderr_text = str(exc.stderr or "")
        if stdout_text:
            print(stdout_text, end="")
        timeout_message = f"[Timeout] command exceeded {max(int(timeout_seconds), 0)}s: {' '.join(full_cmd)}"
        if stderr_text:
            print(stderr_text, end="", file=sys.stderr)
        print(timeout_message, file=sys.stderr)
        stderr_combined = "\n".join([part for part in [stderr_text.strip(), timeout_message] if part]).strip()
        return 124, stdout_text, stderr_combined


def _launch_optional_json_artifact(
    *,
    script_path: str,
    extra_args: list[str] | None,
    dry_run: bool,
    env: dict[str, str],
    extra_nice: int = 0,
) -> dict[str, Any] | None:
    if not script_path or (not os.path.exists(script_path)):
        return None
    cmd = [VENV_PY, script_path]
    if extra_args:
        cmd.extend(extra_args)
    if "--json" not in cmd:
        cmd.append("--json")
    full_cmd = cmd
    if extra_nice > 0:
        full_cmd = ["/usr/bin/nice", "-n", str(extra_nice)] + cmd
    print("$ " + " ".join(full_cmd))
    if dry_run:
        return {"cmd": list(full_cmd), "dry_run": True}
    proc = subprocess.Popen(
        full_cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {"cmd": list(full_cmd), "proc": proc, "quiet": _child_output_quiet(env)}


def _finish_optional_json_artifact(handle: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    if not handle:
        return 0, {}
    if handle.get("dry_run"):
        return 0, {}
    proc = handle.get("proc")
    if proc is None:
        return 0, {}
    stdout_text, stderr_text = proc.communicate()
    stdout_text = str(stdout_text or "")
    stderr_text = str(stderr_text or "")
    if bool(handle.get("quiet")):
        if int(proc.returncode or 0) != 0:
            stdout_tail = _quiet_tail(stdout_text)
            stderr_tail = _quiet_tail(stderr_text)
            if stdout_tail:
                print(stdout_tail, end="\n")
            if stderr_tail:
                print(stderr_tail, end="\n", file=sys.stderr)
    else:
        if stdout_text:
            print(stdout_text, end="")
        if stderr_text:
            print(stderr_text, end="", file=sys.stderr)
    try:
        payload = json.loads(stdout_text.strip()) if stdout_text.strip() else {}
    except Exception:
        payload = {}
    return int(proc.returncode or 0), payload if isinstance(payload, dict) else {}


def _current_rss_gb() -> float:
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return 0.0
        rss_kb = float(str(proc.stdout or "").strip() or 0.0)
        return rss_kb / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _snapshot_is_reusable(
    summary: dict[str, Any],
    *,
    lookback_days: int,
    min_sequences: int,
    min_rows: int,
    prefer_sqlite: bool,
    reuse_if_fresh_minutes: int,
) -> bool:
    if max(int(reuse_if_fresh_minutes), 0) <= 0:
        return False
    if not isinstance(summary, dict) or not summary:
        return False
    if bool(summary.get("prefer_sqlite", False)) != bool(prefer_sqlite):
        return False
    if int(summary.get("lookback_days", 0) or 0) < max(int(lookback_days), 1):
        return False
    if _runtime_training_snapshot_preflight_failure(
        summary,
        min_sequences=min_sequences,
        min_rows=min_rows,
    ):
        return False
    rows_path = str(summary.get("rows_path") or "").strip()
    if rows_path and (not os.path.exists(rows_path)):
        return False
    _, fresh = _fresh_health_payload(summary, max_age_hours=float(max(int(reuse_if_fresh_minutes), 0)) / 60.0)
    return bool(fresh)


def _tail_text(text: str, max_lines: int = 40) -> str:
    lines = [str(line).rstrip() for line in str(text or "").splitlines() if str(line).strip()]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _extract_failure_reason(stdout_text: str, stderr_text: str) -> str:
    for blob in (stderr_text, stdout_text):
        lines = [str(line).strip() for line in str(blob or "").splitlines() if str(line).strip()]
        if not lines:
            continue
        for line in reversed(lines):
            if line.startswith("Traceback"):
                continue
            return line
    return "command_failed_without_output"


def _target_training_env(
    *,
    args: argparse.Namespace,
    target: str,
    child_env: dict[str, str],
    new_bot_ids: set[str],
    distill_assign_map: dict[str, dict],
) -> tuple[str, dict[str, str]]:
    target_env = dict(child_env)
    bot_id = _normalized_bot_id_from_script(target)
    is_new_bot = bool(getattr(args, "new_bot_boost", False)) and bot_id in new_bot_ids
    if is_new_bot:
        target_env["FEATURE_FRESHNESS_GUARD_ENABLED"] = "1"
        target_env["FEATURE_FRESHNESS_MAX_AGE_SECONDS"] = f"{float(args.new_bot_feature_freshness_max_age_seconds):.4f}"
        target_env["RETRAIN_NEW_BOT_MODE"] = "1"

    dist_row = distill_assign_map.get(bot_id, {}) if getattr(args, "distillation_priority", False) else {}
    if dist_row:
        teacher_ids = [
            str((t or {}).get("bot_id", "")).strip()
            for t in (dist_row.get("teachers", []) or [])
            if str((t or {}).get("bot_id", "")).strip()
        ]
        target_env["DISTILLATION_STUDENT"] = "1"
        target_env["DISTILLATION_TEACHERS"] = ",".join(teacher_ids)
        base_tw = float(dist_row.get("teacher_blend_weight", 0.30) or 0.30)
        if is_new_bot:
            base_tw = max(base_tw, float(args.new_bot_distillation_weight))
        target_env["DISTILLATION_TEACHER_WEIGHT"] = str(base_tw)
    else:
        target_env["DISTILLATION_STUDENT"] = "0"

    return bot_id, target_env


def _run_retrain_target(
    *,
    target: str,
    args: argparse.Namespace,
    child_env: dict[str, str],
    effective_retrain_profile: str,
    new_bot_ids: set[str],
    distill_assign_map: dict[str, dict],
    dynamic_max_swap_gb: float,
) -> dict[str, Any]:
    target_name = os.path.basename(target)
    bot_id = _normalized_bot_id_from_script(target)
    base_result: dict[str, Any] = {
        "bot_id": bot_id,
        "target": target,
        "failure": "",
        "failure_detail": None,
        "skipped_by_memory": "",
        "stop": False,
        "dynamic_max_swap_gb": float(dynamic_max_swap_gb),
    }

    readiness_skip = _sample_starved_prefilter_decision(target, effective_retrain_profile)
    if readiness_skip:
        print(
            "[BatchReadinessPrefilter] skipped "
            f"bot_id={readiness_skip.get('bot_id')} "
            f"status={readiness_skip.get('status')} "
            f"samples={readiness_skip.get('sample_count')} "
            f"eligible_sequences={readiness_skip.get('eligible_sequences')}"
        )
        return {**base_result, "status": readiness_skip.get("status"), "outcome": readiness_skip}

    allowed = _wait_for_memory_gate(
        enabled=args.memory_guard,
        min_free_pct=args.min_free_pct,
        max_swap_gb=dynamic_max_swap_gb,
        poll_seconds=args.memory_poll_seconds,
        max_wait_seconds=args.memory_max_wait_seconds,
        label=target_name,
        dry_run=args.dry_run,
    )
    if (not allowed) and args.adaptive_swap_gate and (not args.dry_run):
        ok_now, reason_now, snap_now = _memory_ready(min_free_pct=args.min_free_pct, max_swap_gb=dynamic_max_swap_gb)
        swap_now = float(snap_now.get("swap_used_gb", 0.0) or 0.0)
        free_now = float(snap_now.get("free_pct", 0.0) or 0.0)
        if (not ok_now) and ("swap" in reason_now) and (swap_now > dynamic_max_swap_gb) and (free_now >= args.min_free_pct):
            next_swap = min(
                float(args.adaptive_swap_max_gb),
                max(dynamic_max_swap_gb + float(args.adaptive_swap_step_gb), swap_now + 0.10),
            )
            if next_swap > dynamic_max_swap_gb:
                print(
                    f"[AdaptiveSwapGate] raise label={target_name} "
                    f"from={dynamic_max_swap_gb:.2f} to={next_swap:.2f} "
                    f"reason={reason_now}"
                )
                dynamic_max_swap_gb = next_swap
                base_result["dynamic_max_swap_gb"] = float(dynamic_max_swap_gb)
                allowed = _wait_for_memory_gate(
                    enabled=args.memory_guard,
                    min_free_pct=args.min_free_pct,
                    max_swap_gb=dynamic_max_swap_gb,
                    poll_seconds=args.memory_poll_seconds,
                    max_wait_seconds=max(int(args.memory_max_wait_seconds / 2), 120),
                    label=target_name,
                    dry_run=args.dry_run,
                )
    if not allowed:
        outcome = {"bot_id": bot_id, "target": target, "status": "skipped_memory"}
        return {**base_result, "status": "skipped_memory", "outcome": outcome, "skipped_by_memory": target}

    thermal_ok = _wait_for_thermal_gate(
        enabled=args.thermal_guard,
        min_cpu_speed_limit=args.thermal_min_cpu_speed_limit,
        min_scheduler_limit=args.thermal_min_scheduler_limit,
        poll_seconds=args.memory_poll_seconds,
        max_wait_seconds=args.memory_max_wait_seconds,
        label=target_name,
        dry_run=args.dry_run,
    )
    if not thermal_ok:
        outcome = {"bot_id": bot_id, "target": target, "status": "skipped_thermal"}
        return {**base_result, "status": "skipped_thermal", "outcome": outcome, "skipped_by_memory": target}

    bot_id, target_env = _target_training_env(
        args=args,
        target=target,
        child_env=child_env,
        new_bot_ids=new_bot_ids,
        distill_assign_map=distill_assign_map,
    )

    rc, captured_stdout, captured_stderr = run_cmd_capture(
        [VENV_PY, target],
        args.dry_run,
        target_env,
        timeout_seconds=max(int(args.target_timeout_seconds), 0),
    )
    retry_attempts: list[dict[str, object]] = []
    failure_reason = _extract_failure_reason(captured_stdout, captured_stderr)
    if rc != 0 and args.auto_insufficient_data_retry and _failure_is_insufficient_data(failure_reason):
        for retry_index in range(2):
            retry_env = dict(target_env)
            overrides = _insufficient_data_retry_overrides(target, retry_index)
            retry_env.update(overrides)
            retry_attempts.append(
                {
                    "attempt_index": int(retry_index),
                    "reason": "insufficient_data_retry",
                    "overrides": dict(overrides),
                }
            )
            print(
                "[InsufficientDataRetry] "
                f"bot_id={bot_id} attempt={retry_index} "
                f"lookback_override={overrides.get('RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE')} "
                f"stride_override={overrides.get('RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE')}"
            )
            rc, captured_stdout, captured_stderr = run_cmd_capture(
                [VENV_PY, target],
                args.dry_run,
                retry_env,
                timeout_seconds=max(int(args.target_timeout_seconds), 0),
            )
            failure_reason = _extract_failure_reason(captured_stdout, captured_stderr)
            if rc == 0 or (not _failure_is_insufficient_data(failure_reason)):
                break

    if rc != 0 and _failure_is_deferred_sample_starved(failure_reason):
        deferred_state = _training_diagnostic_state(bot_id)
        if _diagnostic_state_is_deferred_quality_guard(deferred_state):
            outcome = {
                "bot_id": bot_id,
                "target": target,
                "status": "deferred_quality_guard",
                "reason": failure_reason,
                "sample_count": _coerce_int(deferred_state.get("sample_count"), 0),
                "eligible_sequences": _coerce_int(deferred_state.get("eligible_sequences"), 0),
                "observation_count": _coerce_int(deferred_state.get("observation_count"), 0),
                "sequence_count": _coerce_int(deferred_state.get("sequence_count"), 0),
                "quality_failures": deferred_state.get("quality_failures", []),
                "failure_categories": deferred_state.get("failure_categories", []),
                "diagnostics_path": str(deferred_state.get("diagnostics_path") or ""),
                "recommended_next_step": "calibrate thresholds or collect more representative observations before retraining this bot",
                "retry_attempts": retry_attempts,
            }
            print(f"DEFERRED: {target} (quality-guard)")
            return {**base_result, "status": "deferred_quality_guard", "outcome": outcome}
        outcome = {
            "bot_id": bot_id,
            "target": target,
            "status": "deferred_sample_starved",
            "reason": failure_reason,
            "sample_count": _coerce_int(deferred_state.get("sample_count"), 0),
            "eligible_sequences": _coerce_int(deferred_state.get("eligible_sequences"), 0),
            "observation_count": _coerce_int(deferred_state.get("observation_count"), 0),
            "sequence_count": _coerce_int(deferred_state.get("sequence_count"), 0),
            "diagnostics_path": str(deferred_state.get("diagnostics_path") or ""),
            "recommended_next_step": "repair labels/sample eligibility or collect more targeted observations before retraining this bot",
            "retry_attempts": retry_attempts,
        }
        print(f"DEFERRED: {target} (sample-starved)")
        return {**base_result, "status": "deferred_sample_starved", "outcome": outcome}

    if rc != 0 and _failure_is_deferred_quality_guard(failure_reason):
        deferred_state = _training_diagnostic_state(bot_id)
        outcome = {
            "bot_id": bot_id,
            "target": target,
            "status": "deferred_quality_guard",
            "reason": failure_reason,
            "quality_failures": deferred_state.get("quality_failures", []),
            "failure_categories": deferred_state.get("failure_categories", []),
            "diagnostics_path": str(deferred_state.get("diagnostics_path") or ""),
            "recommended_next_step": "calibrate thresholds or family guard before retraining this bot",
            "retry_attempts": retry_attempts,
        }
        print(f"DEFERRED: {target} (quality-guard)")
        return {**base_result, "status": "deferred_quality_guard", "outcome": outcome}

    if rc != 0 and _failure_is_target_timeout(rc, failure_reason):
        outcome = {
            "bot_id": bot_id,
            "target": target,
            "status": "deferred_timeout",
            "rc": rc,
            "reason": failure_reason,
            "stdout_tail": _tail_text(captured_stdout),
            "stderr_tail": _tail_text(captured_stderr),
            "recommended_next_step": "rerun targeted with a wider timeout or optimize this bot's runtime training path",
            "retry_attempts": retry_attempts,
        }
        print(f"DEFERRED: {target} (timeout)")
        return {**base_result, "status": "deferred_timeout", "outcome": outcome}

    if rc != 0:
        failure_detail = {
            "bot_id": bot_id,
            "target": target,
            "status": "failed",
            "rc": rc,
            "reason": failure_reason,
            "stdout_tail": _tail_text(captured_stdout),
            "stderr_tail": _tail_text(captured_stderr),
            "retry_attempts": retry_attempts,
        }
        print(f"FAIL: {target} (exit={rc})")
        return {
            **base_result,
            "status": "failed",
            "outcome": dict(failure_detail),
            "failure": target,
            "failure_detail": failure_detail,
            "stop": not bool(args.continue_on_error),
        }

    outcome = {
        "bot_id": bot_id,
        "target": target,
        "status": "trained",
        "retry_attempts": retry_attempts,
    }
    return {**base_result, "status": "trained", "outcome": outcome}



def _registry_snapshot(path: str) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    s = obj.get("summary", {}) if isinstance(obj, dict) else {}
    top = s.get("top_active", []) if isinstance(s, dict) else []
    top_quality = 0.0
    if top and isinstance(top, list):
        first = top[0] if isinstance(top[0], dict) else {}
        top_quality = float(first.get("quality_score", first.get("test_accuracy", 0.0)) or 0.0)

    return {
        "active_bots": float(s.get("active_bots", 0) or 0),
        "deleted_from_rotation": float(s.get("deleted_from_rotation", 0) or 0),
        "top_quality": top_quality,
    }


def _should_rollback_registry(prev: dict[str, float], curr: dict[str, float]) -> tuple[bool, str]:
    if not prev or not curr:
        return False, "snapshot_missing"

    prev_active = prev.get("active_bots", 0.0)
    curr_active = curr.get("active_bots", 0.0)
    prev_deleted = prev.get("deleted_from_rotation", 0.0)
    curr_deleted = curr.get("deleted_from_rotation", 0.0)
    curr_top_quality = curr.get("top_quality", 0.0)

    min_active = float(os.getenv("ROLLBACK_MIN_ACTIVE_BOTS", os.getenv("MIN_ACTIVE_BOTS", "150")))
    max_active_drop_pct = float(os.getenv("ROLLBACK_MAX_ACTIVE_DROP_PCT", "0.55"))
    max_deleted_jump = float(os.getenv("ROLLBACK_MAX_DELETED_JUMP", "20"))
    min_top_quality = float(os.getenv("ROLLBACK_MIN_TOP_QUALITY", "0.28"))

    if curr_active < min_active:
        return True, f"active_bots_below_floor curr={curr_active:.0f} min={min_active:.0f}"

    if prev_active > 0:
        drop_pct = (prev_active - curr_active) / prev_active
        if drop_pct > max_active_drop_pct:
            return True, f"active_drop_pct={drop_pct:.2f} > max_active_drop_pct={max_active_drop_pct:.2f}"

    if (curr_deleted - prev_deleted) > max_deleted_jump:
        return True, f"deleted_jump={curr_deleted - prev_deleted:.0f} > max_deleted_jump={max_deleted_jump:.0f}"

    if curr_top_quality < min_top_quality:
        return True, f"top_quality={curr_top_quality:.3f} < min_top_quality={min_top_quality:.3f}"

    return False, "healthy"




def _load_active_bot_map(registry_path: str) -> dict[str, bool]:
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return {}

    out: dict[str, bool] = {}
    for row in reg.get("sub_bots", []):
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        out[bot_id] = bool(row.get("active", False))
    return out


def _load_registry_rows(registry_path: str) -> dict[str, dict]:
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for row in reg.get("sub_bots", []):
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        out[bot_id] = dict(row)
    return out


def _low_readiness_retrain_reason(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ""
    reason = str(row.get("reason", "") or "").strip()
    promotion_reason = str(row.get("promotion_reason", "") or "").strip()
    reason_text = " ".join(tok for tok in [reason, promotion_reason] if tok).lower()
    quality = float(row.get("quality_score", row.get("test_accuracy", 0.0)) or 0.0)
    test_accuracy = float(row.get("test_accuracy", 0.0) or 0.0)
    streak = int(row.get("no_improvement_streak", 0) or 0)
    if reason_text.startswith("manual_canary_restore") and (quality < 0.42 or streak >= 3):
        return "manual_canary_restore_low_readiness"
    if reason.startswith("min_active_floor_override_") and (quality < 0.50 or test_accuracy < TARGET_TEST_ACCURACY_FLOOR):
        return "min_active_floor_low_readiness"
    if reason.startswith("protected_collection_floor_") and (quality < 0.30 or test_accuracy < TARGET_TEST_ACCURACY_FLOOR or streak >= 2):
        return "protected_collection_floor_low_readiness"
    if reason.startswith("bucket_diversity_") and (quality < 0.25 and test_accuracy < TARGET_TEST_ACCURACY_FLOOR):
        return "bucket_diversity_low_readiness"
    return ""


def _latest_model_age_hours(bot_id: str) -> float | None:
    model_glob = os.path.join(PROJECT_ROOT, "models", f"{bot_id}_*.npz")
    paths = sorted(glob.glob(model_glob))
    if not paths:
        return None
    latest = paths[-1]
    try:
        age_sec = max(time.time() - os.path.getmtime(latest), 0.0)
        return age_sec / 3600.0
    except Exception:
        return None


def _filter_targets_for_efficiency(
    targets: list[str],
    *,
    active_only: bool,
    max_targets: int,
    min_model_age_hours: float,
    skip_low_readiness: bool,
) -> tuple[list[str], dict[str, int]]:
    active_map = _load_active_bot_map(REGISTRY_PATH)
    registry_rows = _load_registry_rows(REGISTRY_PATH) if skip_low_readiness else {}
    retry_priority_map = _load_retry_pack_priority_map(RETRAIN_RETRY_PACK_LATEST)
    walk_forward_runs = _load_walk_forward_runs(WALK_FORWARD_LATEST)

    rows: list[tuple[str, str, bool, float, float]] = []
    low_readiness_skipped = 0
    for t in targets:
        bot_id = _normalized_bot_id_from_script(t)
        is_active = bool(active_map.get(bot_id, False))
        low_readiness_reason = _low_readiness_retrain_reason(registry_rows.get(bot_id)) if registry_rows else ""
        if low_readiness_reason:
            low_readiness_skipped += 1
            continue
        age_h = _latest_model_age_hours(bot_id)
        if age_h is None:
            age_h = 1e9  # prioritize bots without prior model artifact
        priority_score = _target_priority_score(
            bot_id=bot_id,
            is_active=is_active,
            age_h=float(age_h),
            retry_priority_map=retry_priority_map,
            walk_forward_runs=walk_forward_runs,
        )
        rows.append((t, bot_id, is_active, age_h, priority_score))

    pre = len(rows)
    if active_only:
        rows = [r for r in rows if r[2]]

    if min_model_age_hours > 0:
        rows = [r for r in rows if r[3] >= min_model_age_hours]

    # Prioritize active first, then urgency score, then stalest models first.
    rows.sort(key=lambda r: (0 if r[2] else 1, -float(r[4]), -float(r[3]), r[1]))

    if max_targets > 0:
        rows = rows[:max_targets]

    filtered = [r[0] for r in rows]
    stats = {
        "pre": pre,
        "post": len(filtered),
        "active_selected": sum(1 for r in rows if r[2]),
        "low_readiness_skipped": int(low_readiness_skipped),
        "retry_priority_selected": sum(1 for r in rows if retry_priority_map.get(r[1], 0.0) > 0.0),
    }
    return filtered, stats



def _load_walk_forward_runs(path: str) -> dict[str, int]:
    obj = _load_json_file(path)
    bots = obj.get("bots") if isinstance(obj.get("bots"), dict) else {}
    out: dict[str, int] = {}
    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        key = str(bot_id).strip().lower()
        if not key:
            continue
        out[key] = int(row.get("runs", 0) or 0)
    return out


def _select_new_bot_targets(targets: list[str], runs_map: dict[str, int], max_runs: int) -> list[str]:
    out: list[str] = []
    for t in targets:
        bid = _normalized_bot_id_from_script(t)
        runs = int(runs_map.get(bid, 0) or 0)
        if runs <= max(int(max_runs), 0):
            out.append(t)
    return out


def _derive_regime_focus_from_readiness(path: str, top_n: int = 2) -> str:
    obj = _load_json_file(path)
    rows = obj.get("failed_by_segment") if isinstance(obj.get("failed_by_segment"), dict) else {}
    if not rows:
        return ""
    ranked = sorted(rows.items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))
    picks = [k for k, _ in ranked if str(k).strip().lower() in {"trend", "mean_revert", "shock", "liquidity", "other"}]
    return ",".join(picks[: max(int(top_n), 1)])


def _effective_int(base: int, floor_value: int) -> int:
    return int(max(int(base), int(floor_value)))


def _load_accuracy_map(registry_path: str) -> dict[str, float]:
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return {}

    out: dict[str, float] = {}
    for row in reg.get("sub_bots", []):
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        try:
            out[bot_id] = float(row.get("test_accuracy", 0.0) or 0.0)
        except Exception:
            out[bot_id] = 0.0
    return out


def _apply_retrain_curriculum(targets: list[str], registry_path: str) -> list[str]:
    if os.getenv("RETRAIN_CURRICULUM_ENABLED", "1").strip() != "1":
        return targets

    acc = _load_accuracy_map(registry_path)

    def rank(path: str) -> tuple[int, float, str]:
        bot_id = _normalized_bot_id_from_script(path)
        a = acc.get(bot_id, 0.0)
        # Train stronger anchors first, then weak/missing models.
        band = 0
        if a >= 0.58:
            band = 0
        elif a >= 0.50:
            band = 1
        else:
            band = 2
        return (band, -a, bot_id)

    return sorted(targets, key=rank)


def _load_distillation_plan(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _distillation_assignment_map(plan: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in plan.get("assignments", []) if isinstance(plan, dict) else []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("student_bot_id", "")).strip().lower()
        if not bot_id:
            continue
        out[bot_id] = row
    return out


def _prioritize_targets_for_distillation(targets: list[str], assign_map: dict[str, dict]) -> tuple[list[str], int]:
    if not targets or not assign_map:
        return targets, 0

    student_targets: list[str] = []
    other_targets: list[str] = []
    for t in targets:
        bot_id = _normalized_bot_id_from_script(t)
        if bot_id in assign_map:
            student_targets.append(t)
        else:
            other_targets.append(t)

    return student_targets + other_targets, len(student_targets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all brain_refinery training scripts and refresh master bot registry.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining scripts when one fails.")
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Also retrain bots marked deleted_from_rotation in registry.",
    )
    parser.add_argument(
        "--force-all-targets",
        action="store_true",
        default=os.getenv("RETRAIN_FORCE_ALL_TARGETS", "0").strip() == "1",
        help="Retrain the complete target roster and bypass target caps/readiness narrowing.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        default=os.getenv("RETRAIN_ACTIVE_ONLY", "1").strip() == "1",
        help="Retrain only currently active bots for faster runs (default on).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=int(os.getenv("RETRAIN_MAX_TARGETS", "30")),
        help="Maximum number of bot scripts to retrain per run (0 = no cap).",
    )
    parser.add_argument(
        "--min-model-age-hours",
        type=float,
        default=float(os.getenv("RETRAIN_MIN_MODEL_AGE_HOURS", "0")),
        help="Skip bots retrained more recently than this many hours.",
    )
    parser.add_argument(
        "--thread-cap",
        type=int,
        default=int(os.getenv("RETRAIN_THREAD_CAP", "2")),
        help="Cap BLAS/OpenMP threads per training process (default: RETRAIN_THREAD_CAP or 2).",
    )
    parser.add_argument(
        "--target-workers",
        type=int,
        default=int(os.getenv("RETRAIN_TARGET_WORKERS", "1")),
        help="Number of bot target training subprocesses to run concurrently (default: RETRAIN_TARGET_WORKERS or 1).",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=int(os.getenv("RETRAIN_NICE", "10")),
        help="Niceness to apply to retrain process (default: RETRAIN_NICE or 10).",
    )
    parser.add_argument(
        "--memory-guard",
        action="store_true",
        default=os.getenv("RETRAIN_MEMORY_GUARD", "1").strip() == "1",
        help="Enable memory gate before each target run.",
    )
    parser.add_argument(
        "--min-free-pct",
        type=float,
        default=float(os.getenv("RETRAIN_MIN_FREE_PCT", "18")),
        help="Minimum free memory percentage required before launching next model.",
    )
    parser.add_argument(
        "--max-swap-gb",
        type=float,
        default=float(os.getenv("RETRAIN_MAX_SWAP_GB", "2.5")),
        help="Maximum allowed swap usage (GB) before launching next model.",
    )
    parser.add_argument(
        "--adaptive-swap-gate",
        action="store_true",
        default=os.getenv("RETRAIN_ADAPTIVE_SWAP_GATE", "1").strip() == "1",
        help="Auto-relax swap gate when swap is persistently above threshold.",
    )
    parser.add_argument(
        "--adaptive-swap-step-gb",
        type=float,
        default=float(os.getenv("RETRAIN_ADAPTIVE_SWAP_STEP_GB", "0.6")),
        help="Step increase for adaptive swap gate.",
    )
    parser.add_argument(
        "--adaptive-swap-max-gb",
        type=float,
        default=float(os.getenv("RETRAIN_ADAPTIVE_SWAP_MAX_GB", "6.0")),
        help="Upper cap for adaptive swap gate relaxation.",
    )
    parser.add_argument(
        "--memory-poll-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_MEMORY_POLL_SECONDS", "20")),
        help="How often to re-check memory gate while waiting.",
    )
    parser.add_argument(
        "--memory-max-wait-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_MEMORY_MAX_WAIT_SECONDS", "1800")),
        help="Max wait per target before skipping due to memory pressure.",
    )
    parser.add_argument(
        "--between-target-sleep-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_BETWEEN_TARGET_SLEEP_SECONDS", "4")),
        help="Cooldown sleep between targets to smooth memory pressure.",
    )
    parser.add_argument(
        "--thermal-guard",
        action="store_true",
        default=os.getenv("RETRAIN_THERMAL_GUARD", "1").strip() == "1",
        help="Enable thermal gate checks before each target run.",
    )
    parser.add_argument(
        "--thermal-min-cpu-speed-limit",
        type=float,
        default=float(os.getenv("RETRAIN_THERMAL_MIN_CPU_SPEED_LIMIT", "75")),
        help="Minimum pmset CPU_Speed_Limit required to launch next model.",
    )
    parser.add_argument(
        "--thermal-min-scheduler-limit",
        type=float,
        default=float(os.getenv("RETRAIN_THERMAL_MIN_SCHEDULER_LIMIT", "75")),
        help="Minimum pmset Scheduler_Limit required to launch next model.",
    )
    parser.add_argument(
        "--ops-extra-nice",
        type=int,
        default=int(os.getenv("RETRAIN_OPS_EXTRA_NICE", "6")),
        help="Extra nice offset for ops tasks (master registry update + behavior jobs).",
    )
    parser.add_argument(
        "--after-hours-only",
        action="store_true",
        default=os.getenv("RETRAIN_AFTER_HOURS_ONLY", "1").strip() == "1",
        help="Skip retrain during market hours (ET) unless explicitly disabled.",
    )
    parser.add_argument(
        "--session-start-hour",
        type=int,
        default=int(os.getenv("MARKET_SESSION_START_HOUR", "8")),
    )
    parser.add_argument(
        "--session-end-hour",
        type=int,
        default=int(os.getenv("MARKET_SESSION_END_HOUR", "20")),
    )
    parser.add_argument(
        "--monthly-prune",
        action="store_true",
        default=os.getenv("MONTHLY_PRUNE_ENABLED", "1").strip() == "1",
        help="Run monthly underperformer/redundancy prune once per month.",
    )
    parser.add_argument(
        "--weekly-model-archive",
        action="store_true",
        default=os.getenv("WEEKLY_MODEL_ARCHIVE_ENABLED", "1").strip() == "1",
        help="Archive old model artifacts once per ISO week.",
    )
    parser.add_argument(
        "--archive-keep-per-bot",
        type=int,
        default=int(os.getenv("MODEL_ARCHIVE_KEEP_PER_BOT", "8")),
    )
    parser.add_argument(
        "--archive-min-age-hours",
        type=float,
        default=float(os.getenv("MODEL_ARCHIVE_MIN_AGE_HOURS", "24")),
    )
    parser.add_argument(
        "--distillation-priority",
        action="store_true",
        default=os.getenv("RETRAIN_DISTILLATION_PRIORITY", "1").strip() == "1",
        help="Prioritize student bots from distillation plan in retrain order.",
    )
    parser.add_argument(
        "--distillation-plan",
        default=os.getenv("DISTILLATION_PLAN_PATH", os.path.join(PROJECT_ROOT, "governance", "distillation", "teacher_student_plan_latest.json")),
        help="Path to teacher-student distillation plan JSON.",
    )
    parser.add_argument(
        "--distillation-student-extra-pass",
        type=int,
        default=int(os.getenv("RETRAIN_DISTILLATION_STUDENT_EXTRA_PASS", "0")),
        help="Optional extra retrain passes for prioritized student bots (count).",
    )
    parser.add_argument(
        "--require-data-quality-floor",
        action="store_true",
        default=os.getenv("RETRAIN_REQUIRE_DATA_QUALITY_FLOOR", "1").strip() == "1",
        help="Block retrain start when snapshot coverage/divergence quality floor is not met.",
    )
    parser.add_argument(
        "--min-snapshot-coverage-ratio",
        type=float,
        default=float(os.getenv("RETRAIN_MIN_SNAPSHOT_COVERAGE_RATIO", "0.85")),
    )
    parser.add_argument(
        "--max-data-divergence-spread",
        type=float,
        default=float(os.getenv("RETRAIN_MAX_DATA_DIVERGENCE_SPREAD", "0.03")),
    )
    parser.add_argument(
        "--data-divergence-scope",
        default=os.getenv("RETRAIN_DATA_DIVERGENCE_SCOPE", "all"),
        help="Select divergence artifact scope: all, bond, or non_bond.",
    )
    parser.add_argument(
        "--snapshot-coverage-file",
        default=os.getenv("SNAPSHOT_COVERAGE_FILE", os.path.join(PROJECT_ROOT, "governance", "health", "snapshot_coverage_latest.json")),
    )
    parser.add_argument(
        "--data-divergence-file",
        default=os.getenv("DATA_DIVERGENCE_FILE", DATA_DIVERGENCE_GLOBAL_FILE),
    )
    parser.add_argument(
        "--require-full-snapshot-sync",
        action="store_true",
        default=os.getenv("RETRAIN_REQUIRE_FULL_SNAPSHOT_SYNC", "1").strip() == "1",
        help="Require all retained debug snapshot dirs to be ingested into SQLite before trade-behavior retrain.",
    )
    parser.add_argument(
        "--purge-incorporated-snapshots",
        action="store_true",
        default=os.getenv("RETRAIN_PURGE_INGESTED_SNAPSHOTS", "1").strip() == "1",
        help="After confirmed successful retrain, purge fully ingested debug snapshot dirs.",
    )
    parser.add_argument(
        "--purge-debug-snapshots-days",
        type=int,
        default=int(os.getenv("RETRAIN_PURGE_DEBUG_SNAPSHOTS_DAYS", "0")),
        help="Retention window to use for post-retrain debug snapshot purge.",
    )
    parser.add_argument(
        "--purge-debug-snapshots-keep",
        type=int,
        default=int(os.getenv("RETRAIN_PURGE_DEBUG_SNAPSHOTS_KEEP", "0")),
        help="Number of newest debug snapshot dirs to keep after post-retrain purge.",
    )
    parser.add_argument(
        "--regime-balance",
        action="store_true",
        default=os.getenv("RETRAIN_REGIME_BALANCE", "1").strip() == "1",
        help="Distribute retrain targets across regime buckets instead of clustered order.",
    )
    parser.add_argument(
        "--regime-focus",
        default=os.getenv("RETRAIN_REGIME_FOCUS", ""),
        help="Optional comma-separated regime focus list (trend,mean_revert,shock,liquidity,other).",
    )
    parser.add_argument(
        "--exclude-bot-ids",
        default=os.getenv("RETRAIN_EXCLUDE_BOT_IDS", ""),
        help="Optional comma-separated bot ids to exclude from retrain target queue.",
    )
    parser.add_argument(
        "--include-bot-ids",
        default=os.getenv("RETRAIN_INCLUDE_BOT_IDS", ""),
        help="Optional comma-separated bot ids to exclusively retrain.",
    )
    parser.add_argument(
        "--canary-priority-file",
        default=os.getenv("RETRAIN_CANARY_PRIORITY_FILE", CANARY_DIAGNOSTICS),
    )
    parser.add_argument(
        "--canary-priority-top-n",
        type=int,
        default=int(os.getenv("RETRAIN_CANARY_PRIORITY_TOP_N", "10")),
        help="Prioritize top recurring canary-failing bots at the front of retrain queue.",
    )
    parser.add_argument(
        "--retire-persistent-losers",
        action="store_true",
        default=os.getenv("RETRAIN_RETIRE_PERSISTENT_LOSERS", "1").strip() == "1",
        help="Run persistent-loser retirement automation after retrain summary.",
    )
    parser.add_argument(
        "--retire-apply",
        action="store_true",
        default=os.getenv("RETRAIN_RETIRE_APPLY", "1").strip() == "1",
        help="Apply retirement changes to registry (otherwise report-only).",
    )
    parser.add_argument(
        "--retire-lookback-days",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_LOOKBACK_DAYS", "14")),
    )
    parser.add_argument(
        "--retire-min-fail-days",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_MIN_FAIL_DAYS", "5")),
    )
    parser.add_argument(
        "--retire-min-no-improvement-streak",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_MIN_NO_IMPROVEMENT_STREAK", "2")),
    )
    parser.add_argument(
        "--retire-max-per-run",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_MAX_PER_RUN", "6")),
    )
    parser.add_argument(
        "--new-bot-boost",
        action="store_true",
        default=os.getenv("RETRAIN_NEW_BOT_BOOST", "1").strip() == "1",
        help="Accelerate learning for newer bots with stronger teacher pressure and extra passes.",
    )
    parser.add_argument(
        "--new-bot-max-runs",
        type=int,
        default=int(os.getenv("RETRAIN_NEW_BOT_MAX_RUNS", "24")),
        help="Bots at or below this walk-forward run count are treated as newer bots.",
    )
    parser.add_argument(
        "--new-bot-extra-pass",
        type=int,
        default=int(os.getenv("RETRAIN_NEW_BOT_EXTRA_PASS", "2")),
        help="Extra retrain passes to apply to newer bots.",
    )
    parser.add_argument(
        "--new-bot-distillation-weight",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_DISTILLATION_WEIGHT", "0.45")),
        help="Minimum teacher blend weight for newer bots when distillation metadata exists.",
    )
    parser.add_argument(
        "--new-bot-feature-freshness-max-age-seconds",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_FEATURE_FRESHNESS_MAX_AGE_SECONDS", "12")),
        help="Tighter feature freshness age budget for newer bots.",
    )
    parser.add_argument(
        "--new-bot-neutral-hold-min",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_NEUTRAL_HOLD_MIN", "0.68")),
    )
    parser.add_argument(
        "--new-bot-neutral-hold-margin-min",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_NEUTRAL_HOLD_MARGIN_MIN", "0.08")),
    )
    parser.add_argument(
        "--new-bot-regime-auto-focus",
        action="store_true",
        default=os.getenv("RETRAIN_NEW_BOT_REGIME_AUTO_FOCUS", "1").strip() == "1",
        help="Auto-focus retrain queue on worst failing regime segments when boost mode is enabled.",
    )
    parser.add_argument(
        "--walk-forward-file",
        default=os.getenv("RETRAIN_WALK_FORWARD_FILE", os.path.join(PROJECT_ROOT, "governance", "walk_forward", "walk_forward_latest.json")),
    )
    parser.add_argument(
        "--promotion-bottleneck-priority",
        action="store_true",
        default=os.getenv("RETRAIN_PROMOTION_BOTTLENECK_PRIORITY", "1").strip() == "1",
        help="Use promotion bottleneck profile to bias regime focus and priority queue.",
    )
    parser.add_argument(
        "--promotion-bottleneck-file",
        default=os.getenv("RETRAIN_PROMOTION_BOTTLENECK_FILE", PROMOTION_BOTTLENECK_PATH),
    )
    parser.add_argument(
        "--refresh-promotion-artifacts",
        action="store_true",
        default=os.getenv("RETRAIN_REFRESH_PROMOTION_ARTIFACTS", "1").strip() == "1",
        help="Refresh walk-forward, promotion gate, graduation gate, and leak/overfit artifacts before master update.",
    )
    parser.add_argument(
        "--allow-precheck-failures",
        action="store_true",
        default=os.getenv("RETRAIN_ALLOW_PRECHECK_FAILURES", "0").strip() == "1",
        help="Allow master update even when promotion prechecks fail; marks run as precheck override.",
    )
    parser.add_argument(
        "--skip-master-update",
        action="store_true",
        default=os.getenv("RETRAIN_SKIP_MASTER_UPDATE", "0").strip() == "1",
        help="Train targets without refreshing promotion artifacts or updating master registry.",
    )
    parser.add_argument(
        "--weekly-gate-blocker-report",
        action="store_true",
        default=os.getenv("RETRAIN_WEEKLY_GATE_BLOCKER_REPORT", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-hygiene",
        action="store_true",
        default=os.getenv("RETRAIN_LIFECYCLE_HYGIENE", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-apply-prune",
        action="store_true",
        default=os.getenv("RETRAIN_LIFECYCLE_APPLY_PRUNE", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-keep-backups",
        type=int,
        default=int(os.getenv("RETRAIN_LIFECYCLE_KEEP_BACKUPS", "25")),
    )
    parser.add_argument(
        "--lifecycle-min-free-gb",
        type=float,
        default=float(os.getenv("RETRAIN_LIFECYCLE_MIN_FREE_GB", "10")),
    )
    parser.add_argument(
        "--lifecycle-repair-stale-artifacts",
        action="store_true",
        default=os.getenv("RETRAIN_LIFECYCLE_REPAIR_STALE_ARTIFACTS", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-apply-repair",
        action="store_true",
        default=os.getenv("RETRAIN_LIFECYCLE_APPLY_REPAIR", "1").strip() == "1",
    )
    parser.add_argument(
        "--require-artifact-freshness",
        action="store_true",
        default=os.getenv("RETRAIN_REQUIRE_ARTIFACT_FRESHNESS", "1").strip() == "1",
        help="Fail fast if replay/reconciliation artifacts are stale or unhealthy.",
    )
    parser.add_argument(
        "--artifact-freshness-max-age-minutes",
        type=float,
        default=float(os.getenv("RETRAIN_ARTIFACT_FRESHNESS_MAX_AGE_MINUTES", "180")),
    )
    parser.add_argument(
        "--require-sample-quotas",
        action="store_true",
        default=os.getenv("RETRAIN_REQUIRE_SAMPLE_QUOTAS", "1").strip() == "1",
        help="Enforce minimum regime/symbol sample quotas before behavior training.",
    )
    parser.add_argument(
        "--sample-min-per-regime",
        type=int,
        default=int(os.getenv("RETRAIN_SAMPLE_MIN_PER_REGIME", "120")),
    )
    parser.add_argument(
        "--sample-min-per-symbol",
        type=int,
        default=int(os.getenv("RETRAIN_SAMPLE_MIN_PER_SYMBOL", "25")),
    )
    parser.add_argument(
        "--sample-max-top-symbol-share",
        type=float,
        default=float(os.getenv("RETRAIN_SAMPLE_MAX_TOP_SYMBOL_SHARE", "0.25")),
    )
    parser.add_argument(
        "--auto-insufficient-data-retry",
        action="store_true",
        default=os.getenv("RETRAIN_AUTO_INSUFFICIENT_DATA_RETRY", "1").strip() == "1",
        help="Automatically retry sample-starved bots with wider runtime-training overrides before marking them failed.",
    )
    parser.add_argument(
        "--counterfactual-replay",
        action="store_true",
        default=os.getenv("RETRAIN_COUNTERFACTUAL_REPLAY", "1").strip() == "1",
        help="Run a fast counterfactual replay summary before retrain.",
    )
    parser.add_argument(
        "--paper-hard-example-pack",
        action="store_true",
        default=os.getenv("RETRAIN_PAPER_HARD_EXAMPLE_PACK", "1").strip() == "1",
        help="Build a hard-example pack from weak paper sleeves before retrain.",
    )
    parser.add_argument(
        "--cold-lane-retrain-extras",
        action="store_true",
        default=os.getenv("RETRAIN_COLD_LANE_EXTRAS", "1").strip() == "1",
        help="Enable slower research/maintenance sidecars such as replay and hard-example packs during retrain.",
    )
    parser.add_argument(
        "--parallel-sidecars",
        action="store_true",
        default=os.getenv("RETRAIN_PARALLEL_SIDECARS", "1").strip() == "1",
        help="Run safe optional sidecar artifacts concurrently where possible.",
    )
    parser.add_argument(
        "--retrain-profile",
        default=os.getenv("RETRAIN_PROFILE", "default"),
        help="Named retrain profile for future-run defaults (default, canary, coverage_micro_canary, coverage_small_canary, coverage_canary, coverage_batch10_canary, coverage_batch20_canary, coverage_batch30_canary, fast_daytime, full_overnight).",
    )
    parser.add_argument(
        "--build-runtime-training-snapshot",
        action="store_true",
        default=os.getenv("RETRAIN_BUILD_RUNTIME_TRAINING_SNAPSHOT", "1").strip() == "1",
    )
    parser.add_argument(
        "--runtime-training-snapshot-lookback-days",
        type=int,
        default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_LOOKBACK_DAYS", "14")),
    )
    parser.add_argument(
        "--runtime-training-snapshot-reuse-if-fresh-minutes",
        type=int,
        default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_REUSE_IF_FRESH_MINUTES", "360")),
    )
    parser.add_argument(
        "--runtime-training-snapshot-prefer-sqlite",
        action="store_true",
        default=os.getenv("RUNTIME_TRAIN_SNAPSHOT_PREFER_SQLITE", "1").strip() == "1",
    )
    parser.add_argument(
        "--runtime-train-use-snapshot",
        action="store_true",
        default=os.getenv("RUNTIME_TRAIN_USE_SNAPSHOT", "1").strip() == "1",
    )
    parser.add_argument(
        "--runtime-train-prefer-sqlite",
        action="store_true",
        default=os.getenv("RUNTIME_TRAIN_PREFER_SQLITE", "1").strip() == "1",
    )
    parser.add_argument(
        "--runtime-train-fast-fail-zero-sample-attempts",
        type=int,
        default=int(os.getenv("RUNTIME_TRAIN_FAST_FAIL_ZERO_SAMPLE_ATTEMPTS", "0")),
    )
    parser.add_argument(
        "--target-timeout-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_TARGET_TIMEOUT_SECONDS", "0")),
        help="Per-target timeout for individual bot trainers; 0 disables the timeout.",
    )
    parser.add_argument(
        "--runtime-training-snapshot-min-sequences",
        type=int,
        default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_MIN_SEQUENCES", "1")),
    )
    parser.add_argument(
        "--runtime-training-snapshot-min-rows",
        type=int,
        default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_MIN_ROWS", "64")),
    )
    args = parser.parse_args()
    effective_retrain_profile = _apply_retrain_profile_defaults(args)
    if args.force_all_targets:
        args.include_deleted = True
        args.active_only = False
        args.max_targets = 0
        args.min_model_age_hours = 0.0
        args.promotion_bottleneck_priority = False
    launch_record = _persist_retrain_launch_record(
        _build_retrain_launch_record(args, effective_retrain_profile),
        dry_run=args.dry_run,
    )
    retry_pack_path = ""
    sample_starved_queue_path = ""
    timeout_queue_path = ""
    master_update_status = ""

    def finish(code: int, final_status: str, *, scorecard_path: str = "", failure_count: int | None = None) -> int:
        nonlocal launch_record
        launch_record = _finalize_retrain_launch_record(
            launch_record,
            dry_run=args.dry_run,
            final_status=final_status,
            exit_code=code,
            scorecard_path=scorecard_path,
            retry_pack_path=retry_pack_path,
            master_update_status=master_update_status,
            failure_count=failure_count,
        )
        return int(code)

    lock_path = os.getenv("MLX_RETRAIN_LOCK_PATH", os.path.join(PROJECT_ROOT, "governance", "mlx_retrain.lock"))
    global _MLX_LOCK_HANDLE
    _MLX_LOCK_HANDLE = _acquire_mlx_lock(lock_path)
    if _MLX_LOCK_HANDLE is None:
        print("Another MLX retrain is already active. Skipping this retrain run.")
        return finish(0, "skipped_lock_busy")

    if args.after_hours_only and _market_open_now_et(args.session_start_hour, args.session_end_hour):
        print("Retrain skipped: market session is open (after-hours-only enabled).")
        return finish(0, "skipped_market_open")

    if not os.path.exists(VENV_PY):
        print(f"ERROR: venv python not found at {VENV_PY}")
        return finish(2, "failed_missing_venv")

    effective_divergence_file, effective_divergence_scope = _resolve_data_divergence_file(
        str(args.data_divergence_scope or ""),
        str(args.data_divergence_file or DATA_DIVERGENCE_GLOBAL_FILE),
    )

    data_quality_summary: dict = {}
    if args.require_data_quality_floor:
        dq_ok, dq_reason, dq_detail = _check_data_quality_floor(
            coverage_file=str(args.snapshot_coverage_file),
            divergence_file=effective_divergence_file,
            min_coverage_ratio=float(args.min_snapshot_coverage_ratio),
            max_divergence_spread=float(args.max_data_divergence_spread),
        )
        data_quality_summary = {
            "ok": dq_ok,
            "reason": dq_reason,
            "data_divergence_scope": effective_divergence_scope,
            "data_divergence_file": effective_divergence_file,
            **dq_detail,
        }
        print(
            "Data quality floor: "
            f"ok={dq_ok} "
            f"coverage={dq_detail.get('coverage_ratio', 0.0):.4f}/{dq_detail.get('min_coverage_ratio', 0.0):.4f} "
            f"divergence={dq_detail.get('worst_relative_spread', 0.0):.4f}/{dq_detail.get('max_divergence_spread', 0.0):.4f} "
            f"scope={effective_divergence_scope}"
        )
        if not dq_ok:
            print(f"Retrain blocked by data quality floor: {dq_reason}")
            return finish(1, "blocked_data_quality_floor")

    if args.require_artifact_freshness and os.path.exists(RETRAIN_ARTIFACT_FRESHNESS_GUARD):
        rc_fresh = run_cmd(
            [
                VENV_PY,
                RETRAIN_ARTIFACT_FRESHNESS_GUARD,
                "--max-age-minutes",
                str(float(args.artifact_freshness_max_age_minutes)),
                "--json",
            ],
            args.dry_run,
            os.environ.copy(),
            extra_nice=max(args.ops_extra_nice, 0),
        )
        if rc_fresh != 0:
            print("Retrain blocked by artifact freshness guard.")
            return finish(1, "blocked_artifact_freshness")

    explicit_include_requested = bool(str(args.include_bot_ids or "").strip())
    for script_path, label in [
        (SCHEMA_MIGRATION_GUARD_SCRIPT, "schema migration guard"),
        (BOT_SUPPORT_OWNER_GUARD_SCRIPT, "bot support owner guard"),
        (FEATURE_STORE_MANIFEST_SCRIPT, "feature store manifest"),
        (NEW_BOT_ADMISSION_GUARD_SCRIPT, "new bot admission guard"),
        (RETRAIN_SCHEMA_COMPATIBILITY_GUARD_SCRIPT, "retrain schema compatibility guard"),
        (GOLDEN_REPLAY_REGRESSION_GUARD_SCRIPT, "golden replay regression guard"),
        (COHORT_DRIFT_BASELINE_GUARD_SCRIPT, "cohort drift baseline guard"),
    ]:
        if not os.path.exists(script_path):
            continue
        guard_cmd = [VENV_PY, script_path, "--json"]
        if script_path == NEW_BOT_ADMISSION_GUARD_SCRIPT and explicit_include_requested:
            guard_cmd += ["--include-bot-ids", str(args.include_bot_ids or "")]
            if args.skip_master_update or args.allow_precheck_failures:
                guard_cmd.append("--advisory-only")
        rc_guard = run_cmd(
            guard_cmd,
            args.dry_run,
            os.environ.copy(),
            extra_nice=max(args.ops_extra_nice, 0),
        )
        if rc_guard != 0:
            print(f"Retrain blocked by {label}.")
            return finish(1, f"blocked_{label.replace(' ', '_')}")

    promotion_bottleneck_handle: dict[str, Any] | None = None
    if args.promotion_bottleneck_priority and os.path.exists(PROMOTION_BOTTLENECK_SCRIPT):
        if args.parallel_sidecars:
            promotion_bottleneck_handle = _launch_optional_json_artifact(
                script_path=PROMOTION_BOTTLENECK_SCRIPT,
                extra_args=None,
                dry_run=args.dry_run,
                env=os.environ.copy(),
                extra_nice=max(args.ops_extra_nice, 0),
            )
        else:
            _ = run_cmd([VENV_PY, PROMOTION_BOTTLENECK_SCRIPT, "--json"], args.dry_run, os.environ.copy(), extra_nice=max(args.ops_extra_nice, 0))

    effective_canary_priority_top_n = int(args.canary_priority_top_n)
    effective_distillation_extra_pass = int(args.distillation_student_extra_pass)
    effective_regime_focus = str(args.regime_focus or "")
    operator_note_focus = _derive_regime_focus_from_operator_notes(RETRAIN_OPERATOR_NOTES_PATH, top_n=2)

    if args.new_bot_boost:
        effective_canary_priority_top_n = _effective_int(effective_canary_priority_top_n, 30)
        effective_distillation_extra_pass = _effective_int(effective_distillation_extra_pass, int(args.new_bot_extra_pass))
        args.distillation_priority = True
        args.regime_balance = True
        if args.new_bot_regime_auto_focus and not effective_regime_focus:
            auto_focus = _derive_regime_focus_from_readiness(PROMOTION_READINESS_PATH, top_n=2)
            if auto_focus:
                effective_regime_focus = auto_focus

    bottleneck_profile = _load_json_file(str(args.promotion_bottleneck_file)) if args.promotion_bottleneck_priority else {}
    if bottleneck_profile:
        rec = bottleneck_profile.get("recommended_retrain_profile") if isinstance(bottleneck_profile.get("recommended_retrain_profile"), dict) else {}
        if (not effective_regime_focus) and str(rec.get("RETRAIN_REGIME_FOCUS", "")).strip():
            effective_regime_focus = str(rec.get("RETRAIN_REGIME_FOCUS", "")).strip()
        try:
            effective_canary_priority_top_n = max(
                effective_canary_priority_top_n,
                int(rec.get("RETRAIN_CANARY_PRIORITY_TOP_N", 0) or 0),
            )
        except Exception:
            pass
        try:
            rec_targets = int(rec.get("RETRAIN_MAX_TARGETS", 0) or 0)
            if rec_targets > 0 and not explicit_include_requested:
                args.max_targets = min(int(args.max_targets), rec_targets) if int(args.max_targets) > 0 else rec_targets
        except Exception:
            pass
    if (not str(args.regime_focus or "").strip()) and operator_note_focus:
        if (not effective_regime_focus) or str(effective_regime_focus).strip().lower() == "other":
            effective_regime_focus = operator_note_focus

    include_deleted_targets = _should_include_deleted_targets(args, explicit_include_requested)
    deleted_ids = _load_deleted_bot_ids(REGISTRY_PATH)
    targets = build_targets(include_deleted=include_deleted_targets)
    if not targets:
        print("ERROR: no brain_refinery targets found")
        return finish(2, "failed_no_targets")

    if explicit_include_requested:
        if include_deleted_targets and not args.include_deleted:
            print("Include-bot-ids override: allowing explicitly requested deleted targets.")
        targets = _apply_included_bot_ids(targets, str(args.include_bot_ids or ""))
        if not targets:
            print(f"ERROR: include_bot_ids selected zero targets: {args.include_bot_ids}")
            return finish(2, "failed_include_filter_zero_targets")

    base_targets = list(targets)
    lane_schedule_summary: dict[str, Any] = {}
    if not explicit_include_requested and not args.force_all_targets and base_targets:
        lane_schedule_summary = retrain_lane_scheduler_src.build_payload(
            registry=_load_json_file(REGISTRY_PATH),
            walk_forward=_load_json_file(str(args.walk_forward_file)),
            new_bot_admission_guard=_load_json_file(os.path.join(PROJECT_ROOT, "governance", "health", "new_bot_admission_guard_latest.json")),
            probation_guard=_load_json_file(os.path.join(PROJECT_ROOT, "governance", "health", "champion_challenger_probation_latest.json")),
            target_bot_ids=[_normalized_bot_id_from_script(item) for item in base_targets],
            max_targets=0,
            new_bot_max_runs=int(args.new_bot_max_runs),
        )
        lane_out_path = os.path.join(PROJECT_ROOT, "governance", "health", "retrain_lane_scheduler_latest.json")
        try:
            os.makedirs(os.path.dirname(lane_out_path), exist_ok=True)
            with open(lane_out_path, "w", encoding="utf-8") as handle:
                json.dump(lane_schedule_summary, handle, ensure_ascii=True, indent=2)
        except Exception as exc:
            print(f"WARN: could not write lane scheduler artifact: {exc}")
        scheduled_ids = lane_schedule_summary.get("selected_bot_ids") if isinstance(lane_schedule_summary.get("selected_bot_ids"), list) else []
        if scheduled_ids:
            target_by_id = {_normalized_bot_id_from_script(item): item for item in base_targets}
            base_targets = [target_by_id[bot_id] for bot_id in scheduled_ids if bot_id in target_by_id]
        lane_rows = lane_schedule_summary.get("lanes") if isinstance(lane_schedule_summary.get("lanes"), dict) else {}
        print(
            "Lane scheduler: "
            f"mature={int(((lane_rows.get('mature') or {}).get('candidate_count', 0) or 0))} "
            f"probation={int(((lane_rows.get('probation') or {}).get('candidate_count', 0) or 0))} "
            f"new={int(((lane_rows.get('new') or {}).get('candidate_count', 0) or 0))} "
            f"selected={int(((lane_schedule_summary.get('summary') or {}).get('selected_count', 0) or 0))}"
        )
        targets = list(base_targets)
    effective_active_only = bool(args.active_only)
    if explicit_include_requested and effective_active_only:
        effective_active_only = False
        print("Include-bot-ids override: bypassing active_only filter.")

    min_age = max(float(args.min_model_age_hours), 0.0)
    force_all_targets = bool(args.force_all_targets)
    if force_all_targets:
        print("Force-all-targets override: bypassing active_only, target caps, readiness skips, and queue narrowing.")
    effective_max_targets = 0 if (explicit_include_requested or force_all_targets) else max(int(args.max_targets), 0)
    effective_min_model_age_hours = 0.0 if (explicit_include_requested or force_all_targets) else min_age
    if explicit_include_requested:
        print("Include-bot-ids override: bypassing max_targets/min_model_age_hours and preserving explicit target set.")

    targets, target_stats = _filter_targets_for_efficiency(
        targets,
        active_only=effective_active_only,
        max_targets=effective_max_targets,
        min_model_age_hours=effective_min_model_age_hours,
        skip_low_readiness=not (explicit_include_requested or force_all_targets),
    )
    if not targets and min_age > 0:
        print("WARN: age filter selected zero targets; retrying with min_model_age_hours=0")
        targets, target_stats = _filter_targets_for_efficiency(
            base_targets,
            active_only=effective_active_only,
            max_targets=effective_max_targets,
            min_model_age_hours=0.0,
            skip_low_readiness=not (explicit_include_requested or force_all_targets),
        )
    if not targets:
        print("WARN: efficiency filter selected zero targets; falling back to full target set")
        targets = base_targets
        target_stats = {"pre": len(base_targets), "post": len(base_targets), "active_selected": 0}

    if not explicit_include_requested:
        targets = _apply_retrain_curriculum(targets, REGISTRY_PATH)
    elif effective_regime_focus:
        print(f"Include-bot-ids override: ignoring regime_focus filter for explicit targets: {effective_regime_focus}")

    wf_runs = _load_walk_forward_runs(str(args.walk_forward_file))
    new_bot_targets = _select_new_bot_targets(targets, wf_runs, int(args.new_bot_max_runs)) if args.new_bot_boost else []
    new_bot_ids = {_normalized_bot_id_from_script(x) for x in new_bot_targets} if args.new_bot_boost else set()

    distill_plan = _load_distillation_plan(args.distillation_plan) if args.distillation_priority else {}
    distill_assign_map = _distillation_assignment_map(distill_plan)
    targets, canary_priority_selected, distill_selected = _reshape_target_queue(
        targets,
        allow_auto_queue_reshaping=not (explicit_include_requested or force_all_targets),
        regime_focus=str(effective_regime_focus or ""),
        regime_balance=bool(args.regime_balance),
        exclude_bot_ids=str(args.exclude_bot_ids or ""),
        canary_priority_file=str(args.canary_priority_file),
        canary_priority_top_n=int(effective_canary_priority_top_n),
        distillation_priority=bool(args.distillation_priority),
        distill_assign_map=distill_assign_map,
        distillation_extra_pass=int(effective_distillation_extra_pass),
        new_bot_boost=bool(args.new_bot_boost),
        new_bot_targets=new_bot_targets,
        new_bot_extra_pass=int(args.new_bot_extra_pass),
    )
    runtime_snapshot_target_count = sum(1 for target in targets if _target_runtime_snapshot_lookback_days(target) > 0)
    required_runtime_snapshot_lookback_days = _required_runtime_snapshot_lookback_days(
        targets,
        int(args.runtime_training_snapshot_lookback_days),
    )
    direct_runtime_without_snapshot = str(
        os.getenv("RETRAIN_DIRECT_RUNTIME_WITHOUT_SNAPSHOT", "")
        or os.getenv("RETRAIN_DIRECT_RUNTIME_NO_SNAPSHOT", "")
        or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    runtime_snapshot_no_build = str(
        os.getenv("RETRAIN_RUNTIME_SNAPSHOT_NO_BUILD", "")
        or os.getenv("RETRAIN_SKIP_RUNTIME_SNAPSHOT_BUILD", "")
        or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if explicit_include_requested and runtime_snapshot_target_count > 0 and (
        not bool(args.runtime_train_use_snapshot)
        or not bool(args.build_runtime_training_snapshot)
        or int(args.runtime_training_snapshot_lookback_days) < int(required_runtime_snapshot_lookback_days)
    ) and not direct_runtime_without_snapshot and not runtime_snapshot_no_build:
        args.runtime_training_snapshot_lookback_days = int(required_runtime_snapshot_lookback_days)
        args.build_runtime_training_snapshot = True
        args.runtime_training_snapshot_prefer_sqlite = True
        args.runtime_train_use_snapshot = True
        args.runtime_train_prefer_sqlite = True
        print(
            "Runtime snapshot scope: "
            f"explicit_runtime_targets={runtime_snapshot_target_count} "
            f"lookback_days={int(required_runtime_snapshot_lookback_days)} "
            "snapshot_reuse_enabled=1"
        )
    elif explicit_include_requested and runtime_snapshot_target_count > 0 and direct_runtime_without_snapshot:
        args.build_runtime_training_snapshot = False
        args.runtime_train_use_snapshot = False
        print(
            "Runtime snapshot scope: "
            f"explicit_runtime_targets={runtime_snapshot_target_count} "
            "direct_runtime_without_snapshot=1"
        )
    elif explicit_include_requested and runtime_snapshot_target_count > 0 and runtime_snapshot_no_build:
        args.build_runtime_training_snapshot = False
        args.runtime_train_use_snapshot = True
        args.runtime_train_prefer_sqlite = True
        print(
            "Runtime snapshot scope: "
            f"explicit_runtime_targets={runtime_snapshot_target_count} "
            "reuse_existing_snapshot_without_build=1"
        )

    child_env = _build_child_env(args.thread_cap)
    child_env["DISTILLATION_ENABLED"] = "1" if args.distillation_priority else "0"
    child_env["DISTILLATION_PLAN_PATH"] = str(args.distillation_plan)
    child_env["REQUIRE_CANARY_PROMOTION_GATE"] = "1"
    child_env["RETRAIN_PROFILE"] = str(effective_retrain_profile)
    child_env["RETRAIN_COLD_LANE_EXTRAS"] = "1" if args.cold_lane_retrain_extras else "0"
    child_env["RUNTIME_TRAIN_PREFER_SQLITE"] = "1" if args.runtime_train_prefer_sqlite else "0"
    child_env["RUNTIME_TRAIN_USE_SNAPSHOT"] = "1" if args.runtime_train_use_snapshot else "0"
    if runtime_snapshot_no_build and args.runtime_train_use_snapshot:
        child_env["RUNTIME_TRAIN_SNAPSHOT_ONLY"] = "1"
    child_env["RUNTIME_TRAIN_FAST_FAIL_ZERO_SAMPLE_ATTEMPTS"] = str(
        max(int(args.runtime_train_fast_fail_zero_sample_attempts), 0)
    )
    profile_env_overrides = _apply_retrain_profile_env_overrides(child_env, effective_retrain_profile)
    hard_example_pack_summary: dict = {}
    retrain_input_diagnostics: dict = {}
    replay_summary: dict = {}
    runtime_training_snapshot_summary: dict = {}
    existing_runtime_snapshot_summary = _load_json_file(RUNTIME_TRAINING_SNAPSHOT_LATEST)
    configured_snapshot_path, configured_snapshot_summary = _configured_runtime_snapshot_summary(child_env)
    if args.build_runtime_training_snapshot and os.path.exists(RUNTIME_TRAINING_SNAPSHOT_SCRIPT):
        if _snapshot_is_reusable(
            existing_runtime_snapshot_summary,
            lookback_days=max(int(args.runtime_training_snapshot_lookback_days), 1),
            min_sequences=max(int(args.runtime_training_snapshot_min_sequences), 0),
            min_rows=max(int(args.runtime_training_snapshot_min_rows), 0),
            prefer_sqlite=bool(args.runtime_training_snapshot_prefer_sqlite),
            reuse_if_fresh_minutes=max(int(args.runtime_training_snapshot_reuse_if_fresh_minutes), 0),
        ):
            runtime_training_snapshot_summary = dict(existing_runtime_snapshot_summary)
            runtime_training_snapshot_summary["reused"] = True
            runtime_training_snapshot_summary["reuse_reason"] = "fresh_compatible_snapshot"
            child_env["RUNTIME_TRAIN_SNAPSHOT_FILE"] = str(
                runtime_training_snapshot_summary.get("health_path") or RUNTIME_TRAINING_SNAPSHOT_LATEST
            )
            print(
                "Runtime training snapshot: "
                f"reused=1 sequences={int(runtime_training_snapshot_summary.get('sequence_count', 0) or 0)} "
                f"rows={int(runtime_training_snapshot_summary.get('row_count', 0) or 0)}"
            )
        else:
            snapshot_cmd = [
                VENV_PY,
                RUNTIME_TRAINING_SNAPSHOT_SCRIPT,
                "--lookback-days",
                str(max(int(args.runtime_training_snapshot_lookback_days), 1)),
                "--reuse-if-fresh-minutes",
                str(max(int(args.runtime_training_snapshot_reuse_if_fresh_minutes), 0)),
                "--json",
            ]
            if args.runtime_training_snapshot_prefer_sqlite:
                snapshot_cmd.append("--prefer-sqlite")
            else:
                snapshot_cmd.append("--no-prefer-sqlite")
            rc_snapshot, out_snapshot, _err_snapshot = run_cmd_capture(
                snapshot_cmd,
                args.dry_run,
                child_env,
                extra_nice=max(args.ops_extra_nice, 0),
            )
            runtime_training_snapshot_summary = _parse_json_output(out_snapshot) if rc_snapshot == 0 else {}
            if rc_snapshot != 0:
                print(f"WARN: runtime training snapshot build failed rc={rc_snapshot}")
            elif runtime_training_snapshot_summary:
                child_env["RUNTIME_TRAIN_SNAPSHOT_FILE"] = str(
                    runtime_training_snapshot_summary.get("health_path") or RUNTIME_TRAINING_SNAPSHOT_LATEST
                )
                print(
                    "Runtime training snapshot: "
                    f"sequences={int(runtime_training_snapshot_summary.get('sequence_count', 0) or 0)} "
                    f"rows={int(runtime_training_snapshot_summary.get('row_count', 0) or 0)}"
                )
    elif (configured_snapshot_path and os.path.exists(configured_snapshot_path)) or os.path.exists(RUNTIME_TRAINING_SNAPSHOT_LATEST):
        if configured_snapshot_path:
            child_env["RUNTIME_TRAIN_SNAPSHOT_FILE"] = configured_snapshot_path
            runtime_training_snapshot_summary = configured_snapshot_summary
        else:
            child_env["RUNTIME_TRAIN_SNAPSHOT_FILE"] = RUNTIME_TRAINING_SNAPSHOT_LATEST
            runtime_training_snapshot_summary = _safe_json_load(RUNTIME_TRAINING_SNAPSHOT_LATEST)
    if args.runtime_train_use_snapshot and runtime_snapshot_target_count > 0:
        snapshot_failure = _runtime_training_snapshot_preflight_failure(
            runtime_training_snapshot_summary,
            min_sequences=max(int(args.runtime_training_snapshot_min_sequences), 0),
            min_rows=max(int(args.runtime_training_snapshot_min_rows), 0),
        )
        if snapshot_failure:
            print(f"ERROR: runtime training snapshot preflight failed: {snapshot_failure}")
            return finish(3, "failed_runtime_snapshot_preflight")
    if args.paper_hard_example_pack and args.cold_lane_retrain_extras:
        pack_path, hard_example_pack_summary = _write_paper_hard_example_pack(
            paper_performance_file=PAPER_PERFORMANCE_PATH,
            out_file=PAPER_HARD_EXAMPLES_PATH,
        )
        child_env["RUNTIME_TRAIN_HARD_EXAMPLE_PACK_FILE"] = str(pack_path)
    elif args.paper_hard_example_pack and not args.cold_lane_retrain_extras:
        print("Skipping paper hard-example pack because cold-lane retrain extras are disabled.")
    counterfactual_summary: dict = {}
    counterfactual_handle: dict[str, Any] | None = None
    if args.counterfactual_replay and args.cold_lane_retrain_extras:
        if args.parallel_sidecars and (not args.build_runtime_training_snapshot or runtime_training_snapshot_summary.get("reused")):
            counterfactual_handle = _launch_optional_json_artifact(
                script_path=COUNTERFACTUAL_REPLAY_SCRIPT,
                extra_args=None,
                dry_run=args.dry_run,
                env=child_env,
                extra_nice=1,
            )
        else:
            rc_counterfactual, counterfactual_summary = _run_optional_json_artifact(
                script_path=COUNTERFACTUAL_REPLAY_SCRIPT,
                extra_args=None,
                dry_run=args.dry_run,
                env=child_env,
                extra_nice=1,
            )
            if rc_counterfactual != 0:
                print(f"WARN: counterfactual replay failed rc={rc_counterfactual}")
    elif args.counterfactual_replay and not args.cold_lane_retrain_extras:
        print("Skipping counterfactual replay because cold-lane retrain extras are disabled.")
    if args.new_bot_boost:
        child_env["TRADE_BEHAVIOR_STRICT_NEUTRAL_GATE"] = "1"
        child_env["TRADE_BEHAVIOR_HOLD_NEUTRAL_MIN"] = f"{float(args.new_bot_neutral_hold_min):.4f}"
        child_env["TRADE_BEHAVIOR_HOLD_MARGIN_MIN"] = f"{float(args.new_bot_neutral_hold_margin_min):.4f}"
    _apply_nice(args.nice)

    if counterfactual_handle is not None:
        rc_counterfactual, counterfactual_summary = _finish_optional_json_artifact(counterfactual_handle)
        if rc_counterfactual != 0:
            print(f"WARN: counterfactual replay failed rc={rc_counterfactual}")
    if promotion_bottleneck_handle is not None:
        _rc_bottleneck, refreshed_bottleneck = _finish_optional_json_artifact(promotion_bottleneck_handle)
        if refreshed_bottleneck:
            bottleneck_profile = refreshed_bottleneck

    target_workers = max(int(getattr(args, "target_workers", 1) or 1), 1)
    if target_workers > 1 and not args.continue_on_error:
        print("WARN: target_workers>1 requires --continue-on-error; falling back to serial target execution.")
        target_workers = 1

    print(
        "Resource limits: "
        f"thread_cap={args.thread_cap} "
        f"target_workers={target_workers} "
        f"OMP={child_env.get('OMP_NUM_THREADS')} "
        f"OPENBLAS={child_env.get('OPENBLAS_NUM_THREADS')} "
        f"VECLIB={child_env.get('VECLIB_MAXIMUM_THREADS')}"
    )
    if hard_example_pack_summary:
        print(
            "Paper hard-example pack: "
            f"profiles={int(hard_example_pack_summary.get('weak_profile_count', 0) or 0)} "
            f"strategies={int(hard_example_pack_summary.get('strategy_count', 0) or 0)}"
        )
    if counterfactual_summary:
        print(
            "Counterfactual replay: "
            f"profiles={','.join(counterfactual_summary.get('profiles_reviewed') or []) or 'none'} "
            f"candidates={int(counterfactual_summary.get('candidate_count', 0) or 0)}"
        )
    swap_relax_free_pct = float(os.getenv("RETRAIN_SWAP_RELAX_FREE_PCT", "38"))
    swap_relax_available_pct = float(os.getenv("RETRAIN_SWAP_RELAX_AVAILABLE_PCT", "55"))
    print(
        "Memory gate: "
        f"enabled={args.memory_guard} "
        f"min_free_pct={args.min_free_pct:.1f} "
        f"max_swap_gb={args.max_swap_gb:.2f} "
        f"swap_relax_free_pct={swap_relax_free_pct:.1f} "
        f"swap_relax_available_pct={swap_relax_available_pct:.1f} "
        f"adaptive={args.adaptive_swap_gate} "
        f"adaptive_step_gb={args.adaptive_swap_step_gb:.2f} "
        f"adaptive_cap_gb={args.adaptive_swap_max_gb:.2f} "
        f"poll={args.memory_poll_seconds}s "
        f"max_wait={args.memory_max_wait_seconds}s "
        f"cooldown={args.between_target_sleep_seconds}s"
    )
    print(
        "Thermal gate: "
        f"enabled={args.thermal_guard} "
        f"min_cpu_speed_limit={args.thermal_min_cpu_speed_limit:.0f} "
        f"min_scheduler_limit={args.thermal_min_scheduler_limit:.0f}"
    )
    print(
        "Retrain lanes: "
        f"profile={effective_retrain_profile} "
        f"cold_lane_extras={args.cold_lane_retrain_extras} "
        f"parallel_sidecars={args.parallel_sidecars} "
        f"target_workers={target_workers} "
        f"target_timeout_seconds={int(args.target_timeout_seconds)} "
        f"snapshot_reuse_minutes={int(args.runtime_training_snapshot_reuse_if_fresh_minutes)}"
    )
    if profile_env_overrides:
        print(
            "Profile runtime caps: "
            + " ".join(f"{key}={value}" for key, value in sorted(profile_env_overrides.items()))
        )

    if not args.include_deleted and deleted_ids:
        print(f"Skipping deleted bots from rotation: {len(deleted_ids)}")

    print(
        "Efficiency filter: "
        f"active_only={effective_active_only} "
        f"max_targets={args.max_targets} "
        f"min_model_age_hours={args.min_model_age_hours:.1f} "
        f"selected={target_stats.get('post', 0)}/{target_stats.get('pre', 0)} "
        f"low_readiness_skipped={target_stats.get('low_readiness_skipped', 0)}"
    )
    print(
        "Queue strategy: "
        f"regime_balance={args.regime_balance} "
        f"regime_focus={effective_regime_focus or 'all'} "
        f"include_bot_ids={str(args.include_bot_ids or 'none')} "
        f"exclude_bot_ids={str(args.exclude_bot_ids or 'none')} "
        f"canary_priority_selected={canary_priority_selected} "
        f"new_bot_boost={args.new_bot_boost} "
        f"bottleneck_profile_used={bool(bottleneck_profile)}"
    )

    operator_notes = _load_retrain_operator_notes(RETRAIN_OPERATOR_NOTES_PATH)
    if operator_notes:
        print(
            "Operator note loaded: "
            f"title={operator_notes.get('title', 'operator_note')} "
            f"observations={len(operator_notes.get('observations', []) or [])} "
            f"guidance={len(operator_notes.get('training_guidance', []) or [])}"
        )
        summary = str(operator_notes.get("summary", "") or "").strip()
        if summary:
            print(f"Operator note summary: {summary}")

    started = datetime.now(timezone.utc).isoformat()
    print(f"Weekly retrain start (UTC): {started}")
    print(f"Targets: {len(targets)}")
    if args.new_bot_boost:
        print(
            "New-bot boost: "
            f"new_targets={len(new_bot_targets)} "
            f"extra_pass={args.new_bot_extra_pass} "
            f"teacher_weight_floor={float(args.new_bot_distillation_weight):.2f} "
            f"feature_freshness_max_age_s={float(args.new_bot_feature_freshness_max_age_seconds):.1f}"
        )
    print("Efficiency tip: keep streaming/video/browser load low during retrain windows.")

    failures: list[str] = []
    failure_details: list[dict] = []
    skipped_by_memory: list[str] = []
    target_outcomes: list[dict] = []
    dynamic_max_swap_gb = float(args.max_swap_gb)

    def _apply_target_result(result: dict[str, Any]) -> bool:
        nonlocal dynamic_max_swap_gb
        try:
            dynamic_max_swap_gb = max(dynamic_max_swap_gb, float(result.get("dynamic_max_swap_gb", dynamic_max_swap_gb)))
        except (TypeError, ValueError):
            pass

        outcome = result.get("outcome")
        if isinstance(outcome, dict):
            target_outcomes.append(outcome)

        skipped_target = str(result.get("skipped_by_memory") or "").strip()
        if skipped_target:
            skipped_by_memory.append(skipped_target)

        failure = str(result.get("failure") or "").strip()
        if failure:
            failures.append(failure)

        failure_detail = result.get("failure_detail")
        if isinstance(failure_detail, dict):
            failure_details.append(failure_detail)

        return bool(result.get("stop"))

    if target_workers > 1:
        print(
            "Target execution: "
            f"workers={target_workers} "
            f"per_process_thread_cap={args.thread_cap} "
            f"planned_targets={len(targets)}"
        )
        indexed_results: list[tuple[int, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=target_workers) as executor:
            future_map = {
                executor.submit(
                    _run_retrain_target,
                    target=target,
                    args=args,
                    child_env=child_env,
                    effective_retrain_profile=effective_retrain_profile,
                    new_bot_ids=new_bot_ids,
                    distill_assign_map=distill_assign_map,
                    dynamic_max_swap_gb=dynamic_max_swap_gb,
                ): (idx, target)
                for idx, target in enumerate(targets)
            }
            for future in as_completed(future_map):
                idx, target = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    bot_id = _normalized_bot_id_from_script(target)
                    failure_detail = {
                        "bot_id": bot_id,
                        "target": target,
                        "status": "failed",
                        "rc": 1,
                        "reason": f"target_worker_exception:{type(exc).__name__}:{exc}",
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "retry_attempts": [],
                    }
                    result = {
                        "bot_id": bot_id,
                        "target": target,
                        "status": "failed",
                        "outcome": dict(failure_detail),
                        "failure": target,
                        "failure_detail": failure_detail,
                        "skipped_by_memory": "",
                        "stop": False,
                        "dynamic_max_swap_gb": dynamic_max_swap_gb,
                    }
                indexed_results.append((idx, result))
                print(
                    "[TargetWorker] done "
                    f"index={idx + 1}/{len(targets)} "
                    f"bot_id={result.get('bot_id')} "
                    f"status={result.get('status')}"
                )
        for _idx, result in sorted(indexed_results, key=lambda item: item[0]):
            _apply_target_result(result)
        if not args.dry_run:
            gc.collect()

    for target in ([] if target_workers > 1 else targets):
        target_name = os.path.basename(target)
        readiness_skip = _sample_starved_prefilter_decision(target, effective_retrain_profile)
        if readiness_skip:
            target_outcomes.append(readiness_skip)
            print(
                "[BatchReadinessPrefilter] skipped "
                f"bot_id={readiness_skip.get('bot_id')} "
                f"status={readiness_skip.get('status')} "
                f"samples={readiness_skip.get('sample_count')} "
                f"eligible_sequences={readiness_skip.get('eligible_sequences')}"
            )
            continue

        allowed = _wait_for_memory_gate(
            enabled=args.memory_guard,
            min_free_pct=args.min_free_pct,
            max_swap_gb=dynamic_max_swap_gb,
            poll_seconds=args.memory_poll_seconds,
            max_wait_seconds=args.memory_max_wait_seconds,
            label=target_name,
            dry_run=args.dry_run,
        )
        if (not allowed) and args.adaptive_swap_gate and (not args.dry_run):
            ok_now, reason_now, snap_now = _memory_ready(min_free_pct=args.min_free_pct, max_swap_gb=dynamic_max_swap_gb)
            swap_now = float(snap_now.get("swap_used_gb", 0.0) or 0.0)
            free_now = float(snap_now.get("free_pct", 0.0) or 0.0)
            if (not ok_now) and ("swap" in reason_now) and (swap_now > dynamic_max_swap_gb) and (free_now >= args.min_free_pct):
                next_swap = min(float(args.adaptive_swap_max_gb), max(dynamic_max_swap_gb + float(args.adaptive_swap_step_gb), swap_now + 0.10))
                if next_swap > dynamic_max_swap_gb:
                    print(
                        f"[AdaptiveSwapGate] raise label={target_name} "
                        f"from={dynamic_max_swap_gb:.2f} to={next_swap:.2f} "
                        f"reason={reason_now}"
                    )
                    dynamic_max_swap_gb = next_swap
                    allowed = _wait_for_memory_gate(
                        enabled=args.memory_guard,
                        min_free_pct=args.min_free_pct,
                        max_swap_gb=dynamic_max_swap_gb,
                        poll_seconds=args.memory_poll_seconds,
                        max_wait_seconds=max(int(args.memory_max_wait_seconds / 2), 120),
                        label=target_name,
                        dry_run=args.dry_run,
                    )
        if not allowed:
            skipped_by_memory.append(target)
            target_outcomes.append({"bot_id": _normalized_bot_id_from_script(target), "target": target, "status": "skipped_memory"})
            continue

        thermal_ok = _wait_for_thermal_gate(
            enabled=args.thermal_guard,
            min_cpu_speed_limit=args.thermal_min_cpu_speed_limit,
            min_scheduler_limit=args.thermal_min_scheduler_limit,
            poll_seconds=args.memory_poll_seconds,
            max_wait_seconds=args.memory_max_wait_seconds,
            label=target_name,
            dry_run=args.dry_run,
        )
        if not thermal_ok:
            skipped_by_memory.append(target)
            target_outcomes.append({"bot_id": _normalized_bot_id_from_script(target), "target": target, "status": "skipped_thermal"})
            continue

        target_env = dict(child_env)
        bot_id = _normalized_bot_id_from_script(target)
        is_new_bot = bot_id in new_bot_ids if args.new_bot_boost else False
        if is_new_bot:
            target_env["FEATURE_FRESHNESS_GUARD_ENABLED"] = "1"
            target_env["FEATURE_FRESHNESS_MAX_AGE_SECONDS"] = f"{float(args.new_bot_feature_freshness_max_age_seconds):.4f}"
            target_env["RETRAIN_NEW_BOT_MODE"] = "1"
        dist_row = distill_assign_map.get(bot_id, {}) if args.distillation_priority else {}
        if dist_row:
            teacher_ids = [str((t or {}).get("bot_id", "")).strip() for t in (dist_row.get("teachers", []) or []) if str((t or {}).get("bot_id", "")).strip()]
            target_env["DISTILLATION_STUDENT"] = "1"
            target_env["DISTILLATION_TEACHERS"] = ",".join(teacher_ids)
            base_tw = float(dist_row.get("teacher_blend_weight", 0.30) or 0.30)
            if is_new_bot:
                base_tw = max(base_tw, float(args.new_bot_distillation_weight))
            target_env["DISTILLATION_TEACHER_WEIGHT"] = str(base_tw)
        else:
            target_env["DISTILLATION_STUDENT"] = "0"

        rc, captured_stdout, captured_stderr = run_cmd_capture(
            [VENV_PY, target],
            args.dry_run,
            target_env,
            timeout_seconds=max(int(args.target_timeout_seconds), 0),
        )
        retry_attempts: list[dict[str, object]] = []
        failure_reason = _extract_failure_reason(captured_stdout, captured_stderr)
        if rc != 0 and args.auto_insufficient_data_retry and _failure_is_insufficient_data(failure_reason):
            for retry_index in range(2):
                retry_env = dict(target_env)
                overrides = _insufficient_data_retry_overrides(target, retry_index)
                retry_env.update(overrides)
                retry_attempts.append(
                    {
                        "attempt_index": int(retry_index),
                        "reason": "insufficient_data_retry",
                        "overrides": dict(overrides),
                    }
                )
                print(
                    "[InsufficientDataRetry] "
                    f"bot_id={bot_id} attempt={retry_index} "
                    f"lookback_override={overrides.get('RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE')} "
                    f"stride_override={overrides.get('RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE')}"
                )
                rc, captured_stdout, captured_stderr = run_cmd_capture(
                    [VENV_PY, target],
                    args.dry_run,
                    retry_env,
                    timeout_seconds=max(int(args.target_timeout_seconds), 0),
                )
                failure_reason = _extract_failure_reason(captured_stdout, captured_stderr)
                if rc == 0 or (not _failure_is_insufficient_data(failure_reason)):
                    break
        if rc != 0 and _failure_is_deferred_sample_starved(failure_reason):
            deferred_bot_id = _normalized_bot_id_from_script(target)
            deferred_state = _training_diagnostic_state(deferred_bot_id)
            if _diagnostic_state_is_deferred_quality_guard(deferred_state):
                target_outcomes.append(
                    {
                        "bot_id": deferred_bot_id,
                        "target": target,
                        "status": "deferred_quality_guard",
                        "reason": failure_reason,
                        "sample_count": _coerce_int(deferred_state.get("sample_count"), 0),
                        "eligible_sequences": _coerce_int(deferred_state.get("eligible_sequences"), 0),
                        "observation_count": _coerce_int(deferred_state.get("observation_count"), 0),
                        "sequence_count": _coerce_int(deferred_state.get("sequence_count"), 0),
                        "quality_failures": deferred_state.get("quality_failures", []),
                        "failure_categories": deferred_state.get("failure_categories", []),
                        "diagnostics_path": str(deferred_state.get("diagnostics_path") or ""),
                        "recommended_next_step": "calibrate thresholds or collect more representative observations before retraining this bot",
                        "retry_attempts": retry_attempts,
                    }
                )
                print(f"DEFERRED: {target} (quality-guard)")
                if not args.dry_run:
                    gc.collect()
                    time.sleep(max(args.between_target_sleep_seconds, 0))
                continue
            target_outcomes.append(
                {
                    "bot_id": deferred_bot_id,
                    "target": target,
                    "status": "deferred_sample_starved",
                    "reason": failure_reason,
                    "sample_count": _coerce_int(deferred_state.get("sample_count"), 0),
                    "eligible_sequences": _coerce_int(deferred_state.get("eligible_sequences"), 0),
                    "observation_count": _coerce_int(deferred_state.get("observation_count"), 0),
                    "sequence_count": _coerce_int(deferred_state.get("sequence_count"), 0),
                    "diagnostics_path": str(deferred_state.get("diagnostics_path") or ""),
                    "recommended_next_step": "repair labels/sample eligibility or collect more targeted observations before retraining this bot",
                    "retry_attempts": retry_attempts,
                }
            )
            print(f"DEFERRED: {target} (sample-starved)")
            if not args.dry_run:
                gc.collect()
                time.sleep(max(args.between_target_sleep_seconds, 0))
            continue
        if rc != 0 and _failure_is_deferred_quality_guard(failure_reason):
            deferred_bot_id = _normalized_bot_id_from_script(target)
            deferred_state = _training_diagnostic_state(deferred_bot_id)
            target_outcomes.append(
                {
                    "bot_id": deferred_bot_id,
                    "target": target,
                    "status": "deferred_quality_guard",
                    "reason": failure_reason,
                    "quality_failures": deferred_state.get("quality_failures", []),
                    "failure_categories": deferred_state.get("failure_categories", []),
                    "diagnostics_path": str(deferred_state.get("diagnostics_path") or ""),
                    "recommended_next_step": "calibrate thresholds or family guard before retraining this bot",
                    "retry_attempts": retry_attempts,
                }
            )
            print(f"DEFERRED: {target} (quality-guard)")
            if not args.dry_run:
                gc.collect()
                time.sleep(max(args.between_target_sleep_seconds, 0))
            continue
        if rc != 0 and _failure_is_target_timeout(rc, failure_reason):
            target_outcomes.append(
                {
                    "bot_id": _normalized_bot_id_from_script(target),
                    "target": target,
                    "status": "deferred_timeout",
                    "rc": rc,
                    "reason": failure_reason,
                    "stdout_tail": _tail_text(captured_stdout),
                    "stderr_tail": _tail_text(captured_stderr),
                    "recommended_next_step": "rerun targeted with a wider timeout or optimize this bot's runtime training path",
                    "retry_attempts": retry_attempts,
                }
            )
            print(f"DEFERRED: {target} (timeout)")
            if not args.dry_run:
                gc.collect()
                time.sleep(max(args.between_target_sleep_seconds, 0))
            continue
        if rc != 0:
            failures.append(target)
            failure_detail = {
                "bot_id": _normalized_bot_id_from_script(target),
                "target": target,
                "status": "failed",
                "rc": rc,
                "reason": failure_reason,
                "stdout_tail": _tail_text(captured_stdout),
                "stderr_tail": _tail_text(captured_stderr),
                "retry_attempts": retry_attempts,
            }
            failure_details.append(failure_detail)
            target_outcomes.append(dict(failure_detail))
            print(f"FAIL: {target} (exit={rc})")
            if not args.continue_on_error:
                break
        else:
            target_outcomes.append(
                {
                    "bot_id": _normalized_bot_id_from_script(target),
                    "target": target,
                    "status": "trained",
                    "retry_attempts": retry_attempts,
                }
            )

        if not args.dry_run:
            gc.collect()
            time.sleep(max(args.between_target_sleep_seconds, 0))

    if failures and not args.continue_on_error:
        print("Stopped early due to failure.")

    prev_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
    prev_acc_map = _registry_accuracy_map(REGISTRY_PATH)
    curr_registry_snapshot = dict(prev_registry_snapshot)
    curr_acc_map = dict(prev_acc_map)
    master_update_status = "skipped"
    registry_backup_path = os.path.join(PROJECT_ROOT, "governance", "registry_backup_before_retrain.json")
    try:
        if os.path.exists(REGISTRY_PATH):
            os.makedirs(os.path.dirname(registry_backup_path), exist_ok=True)
            shutil.copy2(REGISTRY_PATH, registry_backup_path)
    except Exception as exc:
        print(f"WARN: could not backup registry before update: {exc}")

    if args.skip_master_update:
        master_update_status = "skipped_by_flag"
        print("Master registry update skipped by flag.")
    elif _wait_for_memory_gate(
        enabled=args.memory_guard,
        min_free_pct=args.min_free_pct,
        max_swap_gb=dynamic_max_swap_gb,
        poll_seconds=args.memory_poll_seconds,
        max_wait_seconds=args.memory_max_wait_seconds,
        label="run_master_bot.py",
        dry_run=args.dry_run,
    ) and _wait_for_thermal_gate(
        enabled=args.thermal_guard,
        min_cpu_speed_limit=args.thermal_min_cpu_speed_limit,
        min_scheduler_limit=args.thermal_min_scheduler_limit,
        poll_seconds=args.memory_poll_seconds,
        max_wait_seconds=args.memory_max_wait_seconds,
        label="run_master_bot.py",
        dry_run=args.dry_run,
    ):
        precheck_failures: list[str] = []
        if args.refresh_promotion_artifacts:
            artifact_steps = [
                (WALK_FORWARD_VALIDATE_SCRIPT, False),
                (WALK_FORWARD_PROMOTION_GATE_SCRIPT, True),
                (LANE_PROMOTION_GATE_SCRIPT, True),
                (PROMOTION_READINESS_SCRIPT, False),
                (PROMOTION_BOTTLENECK_SCRIPT, False),
                (SCHEMA_MIGRATION_GUARD_SCRIPT, True),
                (BOT_SUPPORT_OWNER_GUARD_SCRIPT, True),
                (FEATURE_STORE_MANIFEST_SCRIPT, True),
                (NEW_BOT_GRADUATION_SCRIPT, True),
                (NEW_BOT_ADMISSION_GUARD_SCRIPT, True),
                (RETRAIN_SCHEMA_COMPATIBILITY_GUARD_SCRIPT, True),
                (LEAK_OVERFIT_GUARD_SCRIPT, True),
                (GOLDEN_REPLAY_REGRESSION_GUARD_SCRIPT, True),
                (COHORT_DRIFT_BASELINE_GUARD_SCRIPT, True),
                (CHAMPION_CHALLENGER_PROBATION_GUARD_SCRIPT, True),
                (CHAMPION_CHALLENGER_PROBATION_ACTION_SCRIPT, True),
            ]
            for script_path, required_ok in artifact_steps:
                if not os.path.exists(script_path):
                    if required_ok:
                        precheck_failures.append(f"missing:{os.path.basename(script_path)}")
                    continue
                cmd = [VENV_PY, script_path]
                if script_path in {
                    PROMOTION_READINESS_SCRIPT,
                    PROMOTION_BOTTLENECK_SCRIPT,
                    SCHEMA_MIGRATION_GUARD_SCRIPT,
                    BOT_SUPPORT_OWNER_GUARD_SCRIPT,
                    FEATURE_STORE_MANIFEST_SCRIPT,
                    NEW_BOT_GRADUATION_SCRIPT,
                    NEW_BOT_ADMISSION_GUARD_SCRIPT,
                    RETRAIN_SCHEMA_COMPATIBILITY_GUARD_SCRIPT,
                    LEAK_OVERFIT_GUARD_SCRIPT,
                    GOLDEN_REPLAY_REGRESSION_GUARD_SCRIPT,
                    COHORT_DRIFT_BASELINE_GUARD_SCRIPT,
                    CHAMPION_CHALLENGER_PROBATION_GUARD_SCRIPT,
                    CHAMPION_CHALLENGER_PROBATION_ACTION_SCRIPT,
                    LANE_PROMOTION_GATE_SCRIPT,
                }:
                    cmd.append("--json")
                rc_art = run_cmd(cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
                if rc_art != 0 and required_ok:
                    precheck_failures.append(f"{os.path.basename(script_path)}:exit_{rc_art}")
        for item in _promotion_state_precheck_failures():
            if item not in precheck_failures:
                precheck_failures.append(item)

        if precheck_failures and (not args.allow_precheck_failures):
            master_update_status = "precheck_failed"
            print("FAIL: promotion prechecks failed")
            for item in precheck_failures:
                print(f" - {item}")
        else:
            if precheck_failures and args.allow_precheck_failures:
                print("WARN: promotion prechecks failed but override is enabled")
                for item in precheck_failures:
                    print(f" - {item}")
            master_cmd = [sys.executable, MASTER_RUNNER]
            if precheck_failures and args.allow_precheck_failures:
                master_cmd.extend([
                    "--no-require-canary-gate",
                    "--no-require-graduation-gate",
                    "--no-require-leak-overfit-gate",
                    "--no-require-lane-promotion-gate",
                    "--no-require-promotion-quality-gate",
                    "--no-require-health-gate-clear",
                    "--no-require-promotion-readiness",
                    "--no-require-paper-feedback-floor",
                ])
            rc = run_cmd(master_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            if rc != 0:
                master_update_status = f"failed_exit_{rc}"
                print(f"FAIL: master bot update (exit={rc})")
            else:
                master_update_status = "updated_precheck_override" if precheck_failures else "updated"

            if not args.dry_run and rc == 0:
                curr_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
                curr_acc_map = _registry_accuracy_map(REGISTRY_PATH)
                rollback_bad, rollback_reason = _should_rollback_registry(prev_registry_snapshot, curr_registry_snapshot)
                if rollback_bad and os.path.exists(registry_backup_path):
                    shutil.copy2(registry_backup_path, REGISTRY_PATH)
                    curr_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
                    curr_acc_map = _registry_accuracy_map(REGISTRY_PATH)
                    master_update_status = f"rolled_back:{rollback_reason}"
                    print(f"[Rollback] restored previous master registry reason={rollback_reason}")
                else:
                    print(f"[Rollback] registry check status={rollback_reason}")
    else:
        master_update_status = "skipped_memory_or_thermal_gate"
        print("WARN: skipped master registry update due to memory gate timeout")

    ended = datetime.now(timezone.utc).isoformat()
    print(f"Weekly retrain end (UTC): {ended}")

    marker_lineage = _build_retrain_lineage(
        stage="post_master_update",
        registry_path=REGISTRY_PATH,
        registry_backup_path=registry_backup_path,
        target_count=len(targets),
        retrain_profile=effective_retrain_profile,
    )
    marker_path = _write_training_success_marker(
        target_outcomes=target_outcomes,
        failures=failures,
        failure_details=failure_details,
        skipped_by_memory=skipped_by_memory,
        master_update_status=master_update_status,
        data_quality_summary=data_quality_summary,
        operator_notes=operator_notes,
        lineage=marker_lineage,
        dry_run=args.dry_run,
    )
    print(f"Training success marker written: {marker_path}")

    confirmed_training_success = False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            confirmed_training_success = bool((json.load(f) or {}).get("confirmed_training_success", False))
    except Exception:
        confirmed_training_success = False

    if args.purge_incorporated_snapshots and (not args.dry_run):
        if confirmed_training_success:
            if os.path.exists(DATA_RETENTION_POLICY):
                print("Purging fully ingested debug snapshots after confirmed retrain success...")
                retention_cmd = [
                    VENV_PY,
                    DATA_RETENTION_POLICY,
                    "--apply",
                    "--json",
                    "--debug-snapshots-days",
                    str(max(int(args.purge_debug_snapshots_days), 0)),
                    "--debug-snapshots-keep",
                    str(max(int(args.purge_debug_snapshots_keep), 0)),
                    "--require-training-success",
                ]
                _ = run_cmd(retention_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            else:
                print(f"WARN: data retention script missing: {DATA_RETENTION_POLICY}")
        else:
            print("Skipping debug snapshot purge because confirmed training success was not achieved.")

    if skipped_by_memory:
        print(f"Skipped by memory gate: {len(skipped_by_memory)}")
        for s in skipped_by_memory:
            print(f" - {s}")

    retry_pack = None
    if failures:
        replay_summary = _build_failed_bot_replay_summary(
            failure_details=failure_details,
            counterfactual_summary=counterfactual_summary,
            paper_performance=_load_json_file(PAPER_PERFORMANCE_PATH),
        )
        if replay_summary:
            replay_summary["artifact_path"] = _write_training_diagnostic_artifact(
                "retrain_replay_summary",
                replay_summary,
                dry_run=args.dry_run,
            )
        retry_pack = _write_retry_pack(
            failures=failures,
            failure_details=failure_details,
            master_update_status=master_update_status,
            dry_run=args.dry_run,
        )
        if retry_pack:
            retry_pack_path = str(retry_pack.get("path") or "")
            print(f"Retry pack written: {retry_pack.get('path')}")

    sample_starved_queue = _write_sample_starved_queue(
        target_outcomes=target_outcomes,
        dry_run=args.dry_run,
    )
    if sample_starved_queue:
        sample_starved_queue_path = str(sample_starved_queue.get("path") or "")
        print(f"Sample-starved training queue written: {sample_starved_queue_path}")

    quality_repair_queue_path = ""
    quality_repair_queue = _write_quality_repair_queue(
        target_outcomes=target_outcomes,
        dry_run=args.dry_run,
    )
    if quality_repair_queue:
        quality_repair_queue_path = str(quality_repair_queue.get("path") or "")
        print(f"Quality repair training queue written: {quality_repair_queue_path}")

    timeout_queue = _write_training_timeout_queue(
        target_outcomes=target_outcomes,
        dry_run=args.dry_run,
    )
    if timeout_queue:
        timeout_queue_path = str(timeout_queue.get("path") or "")
        print(f"Training timeout queue written: {timeout_queue_path}")

    if failures:
        print(f"Completed with {len(failures)} failures.")
        for f in failures:
            print(f" - {f}")
        scorecard_lineage = _build_retrain_lineage(
            stage="final_scorecard",
            registry_path=REGISTRY_PATH,
            registry_backup_path=registry_backup_path,
            target_count=len(targets),
            retrain_profile=effective_retrain_profile,
        )
        scorecard_path = _write_retrain_scorecard(
            started_utc=started,
            ended_utc=ended,
            target_count=len(targets),
            failures=failures,
            failure_details=failure_details,
            skipped_by_memory=skipped_by_memory,
            target_outcomes=target_outcomes,
            prev_registry_snapshot=prev_registry_snapshot,
            curr_registry_snapshot=curr_registry_snapshot,
            prev_acc=prev_acc_map,
            curr_acc=curr_acc_map,
            master_update_status=master_update_status,
            data_quality_summary=data_quality_summary,
            canary_priority_selected=canary_priority_selected,
            distill_selected=distill_selected,
            retry_pack=retry_pack,
            operator_notes=operator_notes,
            retrain_input_diagnostics=retrain_input_diagnostics,
            replay_summary=replay_summary,
            lineage=scorecard_lineage,
            launch_context={
                **launch_record,
                "master_update_status": master_update_status,
                "retry_pack_path": retry_pack_path,
                "sample_starved_queue_path": sample_starved_queue_path,
                "quality_repair_queue_path": quality_repair_queue_path,
                "timeout_queue_path": timeout_queue_path,
            },
            dry_run=args.dry_run,
        )
        print(f"Retrain scorecard written: {scorecard_path}")
        return finish(1, "completed_with_failures", scorecard_path=scorecard_path, failure_count=len(failures))

    enable_trade_behavior_retrain = os.getenv("ENABLE_TRADE_BEHAVIOR_RETRAIN", "1").strip() == "1"
    trade_behavior_strict = os.getenv("TRADE_BEHAVIOR_STRICT", "0").strip() == "1"

    if enable_trade_behavior_retrain:
        print("Running trade history behavior learning step...")
        trade_behavior_trained_ok = False

        if os.path.exists(SNAPSHOT_HEALTH_SYNC_SCRIPT):
            snapshot_sync_cmd = [VENV_PY, SNAPSHOT_HEALTH_SYNC_SCRIPT, "--json"]
            if args.require_full_snapshot_sync:
                snapshot_sync_cmd.append("--require-full-debug-sync")
            snapshot_sync_rc = run_cmd(snapshot_sync_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            if snapshot_sync_rc != 0:
                print(f"FAIL: snapshot SQL sync coverage gate (exit={snapshot_sync_rc})")
                return finish(1, "failed_snapshot_health_sync")
        else:
            print(f"WARN: snapshot health SQL sync script missing: {SNAPSHOT_HEALTH_SYNC_SCRIPT}")

        dataset_builder_override = os.getenv("TRADE_BEHAVIOR_DATASET_BUILDER", "").strip()
        allow_legacy_trade_dataset_builder = os.getenv("TRADE_BEHAVIOR_ALLOW_LEGACY_DATASET_BUILDER", "0").strip() == "1"
        dataset_builder_candidates: list[str] = []
        if dataset_builder_override:
            dataset_builder_candidates.append(dataset_builder_override)
        else:
            if os.path.exists(TRADE_DATASET_BUILDER):
                dataset_builder_candidates.append(TRADE_DATASET_BUILDER)
            if (
                allow_legacy_trade_dataset_builder
                and os.path.exists(TRADE_DATASET_BUILDER_LEGACY)
                and TRADE_DATASET_BUILDER_LEGACY not in dataset_builder_candidates
            ):
                dataset_builder_candidates.append(TRADE_DATASET_BUILDER_LEGACY)

        if (not dataset_builder_override) and (not allow_legacy_trade_dataset_builder) and os.path.exists(TRADE_DATASET_BUILDER_LEGACY):
            print("Trade dataset builder legacy fallback disabled; curated builder only")

        dataset_build_rc = 0
        if dataset_builder_candidates:
            for idx, builder in enumerate(dataset_builder_candidates):
                print(f"Trade dataset builder: {builder}")
                dataset_build_rc = run_cmd([VENV_PY, builder], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
                if dataset_build_rc == 0:
                    break
                if idx < (len(dataset_builder_candidates) - 1):
                    print(
                        f"WARN: trade dataset builder failed (exit={dataset_build_rc}); "
                        f"falling back to {dataset_builder_candidates[idx + 1]}"
                    )

            if dataset_build_rc != 0 and trade_behavior_strict:
                print("FAIL: trade dataset build")
                return finish(1, "failed_trade_dataset_build")
        else:
            print(
                f"WARN: trade dataset builder missing: {TRADE_DATASET_BUILDER} "
                f"(legacy: {TRADE_DATASET_BUILDER_LEGACY})"
            )

        if args.require_sample_quotas and os.path.exists(TRAINING_SAMPLE_QUOTA_GUARD):
            quota_cmd = [
                VENV_PY,
                TRAINING_SAMPLE_QUOTA_GUARD,
                "--dataset",
                TRADE_BEHAVIOR_DATASET,
                "--min-per-regime",
                str(int(args.sample_min_per_regime)),
                "--min-per-symbol",
                str(int(args.sample_min_per_symbol)),
                "--max-top-symbol-share",
                str(float(args.sample_max_top_symbol_share)),
                "--json",
            ]
            rc_quota = run_cmd(quota_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            if rc_quota != 0:
                print("FAIL: training sample quota guard")
                return finish(1, "failed_training_sample_quota_guard")

        retrain_input_diagnostics = _build_retrain_input_feature_diagnostics(TRADE_BEHAVIOR_DATASET)
        if retrain_input_diagnostics:
            retrain_input_diagnostics["artifact_path"] = _write_training_diagnostic_artifact(
                "retrain_input_feature_diagnostics",
                retrain_input_diagnostics,
                dry_run=args.dry_run,
            )
            tracked = retrain_input_diagnostics.get("tracked_features") if isinstance(retrain_input_diagnostics.get("tracked_features"), dict) else {}
            present_count = sum(1 for row in tracked.values() if isinstance(row, dict) and bool(row.get("present_in_schema", False)))
            print(
                "Retrain input diagnostics: "
                f"dataset_rows={int(retrain_input_diagnostics.get('dataset_rows', 0) or 0)} "
                f"tracked_present={present_count}/{len(ADVANCED_RETRAIN_DIAGNOSTIC_FEATURES)}"
            )

        if os.path.exists(TRADE_BEHAVIOR_TRAINER):
            rc = run_cmd([VENV_PY, TRADE_BEHAVIOR_TRAINER], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            trade_behavior_trained_ok = (rc == 0)
            if rc != 0 and trade_behavior_strict:
                print("FAIL: trade behavior trainer")
                return finish(1, "failed_trade_behavior_trainer")
        else:
            print(f"WARN: trade behavior trainer missing: {TRADE_BEHAVIOR_TRAINER}")

        if os.path.exists(REPLAY_FEATURE_ABLATION_REPORT):
            rc_ablation = run_cmd(
                [VENV_PY, REPLAY_FEATURE_ABLATION_REPORT, "--json"],
                args.dry_run,
                child_env,
                extra_nice=max(args.ops_extra_nice, 0),
            )
            if rc_ablation != 0:
                if trade_behavior_strict:
                    print("FAIL: replay feature ablation report")
                    return finish(1, "failed_replay_feature_ablation")
                print("WARN: replay feature ablation report failed")

        if trade_behavior_trained_ok and args.purge_incorporated_snapshots and (not args.dry_run):
            if os.path.exists(DATA_RETENTION_POLICY):
                print("Purging fully incorporated debug snapshots after successful trade-behavior training...")
                retention_cmd = [
                    VENV_PY,
                    DATA_RETENTION_POLICY,
                    "--apply",
                    "--json",
                    "--debug-snapshots-days",
                    str(max(int(args.purge_debug_snapshots_days), 0)),
                    "--debug-snapshots-keep",
                    str(max(int(args.purge_debug_snapshots_keep), 0)),
                    "--require-snapshot-training-coverage",
                ]
                _ = run_cmd(retention_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            else:
                print(f"WARN: data retention script missing: {DATA_RETENTION_POLICY}")

    if args.monthly_prune and (not args.dry_run) and _monthly_prune_due():
        print("Running monthly prune pass...")
        if os.path.exists(PRUNE_UNDERPERFORMERS):
            _ = run_cmd([VENV_PY, PRUNE_UNDERPERFORMERS, "--min-streak", os.getenv("MONTHLY_PRUNE_MIN_STREAK", "3")], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
        if os.path.exists(PRUNE_REDUNDANT):
            _ = run_cmd([VENV_PY, PRUNE_REDUNDANT], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
        _write_monthly_prune_stamp()

    if args.weekly_model_archive and (not args.dry_run) and _weekly_archive_due():
        print("Running weekly model archive pass...")
        if os.path.exists(ARCHIVE_OLD_MODELS):
            _ = run_cmd(
                [
                    VENV_PY,
                    ARCHIVE_OLD_MODELS,
                    "--keep-per-bot",
                    str(max(args.archive_keep_per_bot, 1)),
                    "--min-age-hours",
                    str(max(args.archive_min_age_hours, 0.0)),
                ],
                args.dry_run,
                child_env,
                extra_nice=max(args.ops_extra_nice, 0),
            )
            _write_weekly_archive_stamp()
        else:
            print(f"WARN: archive script missing: {ARCHIVE_OLD_MODELS}")

    if args.retire_persistent_losers and (not args.dry_run) and os.path.exists(RETIRE_PERSISTENT_LOSERS):
        print("Running persistent-loser retirement scan...")
        retire_cmd = [
            VENV_PY,
            RETIRE_PERSISTENT_LOSERS,
            "--lookback-days",
            str(max(args.retire_lookback_days, 1)),
            "--min-fail-days",
            str(max(args.retire_min_fail_days, 1)),
            "--min-no-improvement-streak",
            str(max(args.retire_min_no_improvement_streak, 1)),
            "--max-retire-per-run",
            str(max(args.retire_max_per_run, 0)),
            "--json",
        ]
        if args.retire_apply:
            retire_cmd.append("--apply")
        _ = run_cmd(retire_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    if args.weekly_gate_blocker_report and os.path.exists(WEEKLY_GATE_BLOCKER_REPORT_SCRIPT):
        _ = run_cmd([VENV_PY, WEEKLY_GATE_BLOCKER_REPORT_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    scorecard_lineage = _build_retrain_lineage(
        stage="final_scorecard",
        registry_path=REGISTRY_PATH,
        registry_backup_path=registry_backup_path,
        target_count=len(targets),
        retrain_profile=effective_retrain_profile,
    )
    scorecard_path = _write_retrain_scorecard(
        started_utc=started,
        ended_utc=ended,
        target_count=len(targets),
        failures=failures,
        failure_details=failure_details,
        skipped_by_memory=skipped_by_memory,
        target_outcomes=target_outcomes,
        prev_registry_snapshot=prev_registry_snapshot,
        curr_registry_snapshot=curr_registry_snapshot,
        prev_acc=prev_acc_map,
        curr_acc=curr_acc_map,
        master_update_status=master_update_status,
        data_quality_summary=data_quality_summary,
        canary_priority_selected=canary_priority_selected,
        distill_selected=distill_selected,
        retry_pack=retry_pack,
        operator_notes=operator_notes,
        retrain_input_diagnostics=retrain_input_diagnostics,
        replay_summary=replay_summary,
        lineage=scorecard_lineage,
        launch_context={
            **launch_record,
            "master_update_status": master_update_status,
            "retry_pack_path": retry_pack_path,
            "sample_starved_queue_path": sample_starved_queue_path,
            "quality_repair_queue_path": quality_repair_queue_path,
            "timeout_queue_path": timeout_queue_path,
        },
        dry_run=args.dry_run,
    )
    print(f"Retrain scorecard written: {scorecard_path}")

    if os.path.exists(EXPORT_MODEL_CARD_SCRIPT):
        _ = run_cmd([VENV_PY, EXPORT_MODEL_CARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(SCHEMA_MIGRATION_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, SCHEMA_MIGRATION_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(BOT_SUPPORT_OWNER_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, BOT_SUPPORT_OWNER_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(FEATURE_STORE_MANIFEST_SCRIPT):
        _ = run_cmd([VENV_PY, FEATURE_STORE_MANIFEST_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(NEW_BOT_ADMISSION_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, NEW_BOT_ADMISSION_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(RETRAIN_SCHEMA_COMPATIBILITY_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, RETRAIN_SCHEMA_COMPATIBILITY_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(GOLDEN_REPLAY_REGRESSION_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, GOLDEN_REPLAY_REGRESSION_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(COHORT_DRIFT_BASELINE_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, COHORT_DRIFT_BASELINE_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(CHAMPION_CHALLENGER_PROBATION_GUARD_SCRIPT):
        _ = run_cmd([VENV_PY, CHAMPION_CHALLENGER_PROBATION_GUARD_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(CHAMPION_CHALLENGER_PROBATION_ACTION_SCRIPT):
        _ = run_cmd([VENV_PY, CHAMPION_CHALLENGER_PROBATION_ACTION_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(PROMOTION_PACKET_BUILDER_SCRIPT):
        _ = run_cmd([VENV_PY, PROMOTION_PACKET_BUILDER_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    if args.lifecycle_hygiene and os.path.exists(MODEL_LIFECYCLE_HYGIENE_SCRIPT):
        lifecycle_cmd = [
            VENV_PY,
            MODEL_LIFECYCLE_HYGIENE_SCRIPT,
            "--keep-backups",
            str(max(int(args.lifecycle_keep_backups), 1)),
            "--min-free-gb",
            str(max(float(args.lifecycle_min_free_gb), 0.0)),
            "--json",
        ]
        if args.lifecycle_apply_prune:
            lifecycle_cmd.append("--apply-prune")
        if args.lifecycle_repair_stale_artifacts:
            lifecycle_cmd.append("--repair-stale-artifacts")
        if args.lifecycle_apply_repair and args.lifecycle_apply_prune:
            lifecycle_cmd.append("--apply-repair")
        if str(master_update_status).startswith("updated") or str(master_update_status).startswith("rolled_back"):
            lifecycle_cmd.append("--update-last-known-good")
        _ = run_cmd(lifecycle_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    if os.path.exists(POINT_IN_TIME_EVENT_STORE_SCRIPT):
        _ = run_cmd([VENV_PY, POINT_IN_TIME_EVENT_STORE_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(LIVE_READINESS_SMOKE_SCRIPT):
        _ = run_cmd([VENV_PY, LIVE_READINESS_SMOKE_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
    if os.path.exists(PLATFORM_CONTROL_PLANE_SCRIPT):
        _ = run_cmd([VENV_PY, PLATFORM_CONTROL_PLANE_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    if skipped_by_memory:
        print("Completed with memory-gate skips.")
        return finish(1, "completed_with_memory_gate_skips", scorecard_path=scorecard_path, failure_count=len(failures))

    print("Completed successfully.")
    return finish(0, "completed_successfully", scorecard_path=scorecard_path, failure_count=len(failures))


if __name__ == "__main__":
    raise SystemExit(main())
