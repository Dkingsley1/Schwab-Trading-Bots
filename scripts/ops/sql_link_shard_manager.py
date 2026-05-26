import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.channel_queue import default_queue_db_path
from core.sqlite_runtime import connect_sqlite, resolve_sqlite_runtime_settings
from scripts import ops_data_plane

PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
LINK_SCRIPT = PROJECT_ROOT / "scripts" / "link_jsonl_to_sql.py"
HOT_RETENTION_SCRIPT = PROJECT_ROOT / "scripts" / "sql_hot_retention.py"
QUEUE_RETENTION_SCRIPT = PROJECT_ROOT / "scripts" / "sql_queue_retention.py"
SQLITE_MAINTENANCE_SCRIPT = PROJECT_ROOT / "scripts" / "sqlite_performance_maintenance.py"
PRIMARY_DB_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
QUEUE_DB_PATH = Path(
    str(
        os.getenv(
            "SQL_LINK_SERVICE_QUEUE_DB",
            os.getenv(
                "BOT_CHANNEL_QUEUE_DB",
                default_queue_db_path(PROJECT_ROOT),
            ),
        )
    )
).expanduser()
SHARD_DB_ROOT = PROJECT_ROOT / "data" / "sql_link_shards"
SHARD_STATE_ROOT = PROJECT_ROOT / "governance" / "sql_link_shards"
HEALTH_ROOT = PROJECT_ROOT / "governance" / "health"
EVENT_ROOT = PROJECT_ROOT / "governance" / "events"
LATEST_HEALTH = HEALTH_ROOT / "sql_link_service_latest.json"
PROGRESS_HEALTH = HEALTH_ROOT / "sql_link_service_progress_latest.json"
REQUEST_PATH = HEALTH_ROOT / "sql_link_service_request_latest.json"
MAINTENANCE_STATE_PATH = HEALTH_ROOT / "sql_link_service_maintenance_state.json"
INTEGRITY_MARKER_ROOT = HEALTH_ROOT / "sql_link_integrity"
SWAP_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.swap_pressure_override"
RUNTIME_RESOURCE_GUARD_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.runtime_resource_guard_override"
PRESSURE_RELIEF_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.pressure_relief_override"


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        if path.is_dir():
            return
        raise


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in raw.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            out[key] = value
    return out


def _retention_maintenance_paused_for_swap(*, override_path: Path | None = None) -> tuple[bool, dict[str, str]]:
    override = override_path or SWAP_OVERRIDE_PATH
    effective = dict(os.environ)
    effective.update(_load_env_file(override))
    tier = str(effective.get("SWAP_PRESSURE_TIER", "")).strip()
    paused = str(effective.get("RETENTION_MAINTENANCE_PAUSED_FOR_SWAP", "0")).strip() == "1"
    heavy_paused = str(effective.get("SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED", "0")).strip() == "1"
    return bool(paused or heavy_paused or tier in {"pause_research", "survival"}), effective


def _swap_pause_details(env: dict[str, str]) -> dict[str, object]:
    return {
        "skipped": True,
        "reason": "swap_pressure_pause",
        "swap_pressure_tier": str(env.get("SWAP_PRESSURE_TIER", "")),
        "swap_used_gb": str(env.get("SWAP_PRESSURE_SWAP_USED_GB", "")),
    }


def _queue_retention_inline_vacuum_enabled() -> bool:
    return str(os.getenv("SQL_LINK_SERVICE_QUEUE_VACUUM_INLINE_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _queue_retention_inline_max_rows(default: int) -> int:
    configured = int(os.getenv("SQL_LINK_SERVICE_QUEUE_INLINE_MAX_ROWS", "50000"))
    return max(min(int(default), max(configured, 0)), 0)


def _queue_retention_timeout_seconds() -> int:
    return max(int(os.getenv("SQL_LINK_SERVICE_QUEUE_RETENTION_TIMEOUT_SECONDS", "45")), 5)


JSONL_COLUMNS = [
    "source_file",
    "source_rel",
    "line_no",
    "ingested_at",
    "payload_sha1",
    "payload_json",
    "run_id",
    "iter_id",
    "decision_id",
    "parent_decision_id",
    "log_schema_version",
]
JSON_FILE_COLUMNS = [
    "source_file",
    "source_rel",
    "stream",
    "modified_at",
    "ingested_at",
    "payload_sha1",
    "payload_json",
    "payload_size_bytes",
    "log_schema_version",
]
DEFAULT_SHARD_DEFS = {
    "health_fast": {
        "path_contains": (
            "governance/health/data_ingress_latest_,"
            "governance/health/ingestion_backpressure_latest.json,"
            "governance/health/data_source_divergence_latest.json,"
            "governance/health/data_source_divergence_bond_latest.json,"
            "governance/health/data_source_divergence_non_bond_latest.json,"
            "governance/health/jsonl_sql_ingestion_health_,"
            "governance/health/sql_link_service_,"
            "governance/health/paper_performance_latest.json,"
            "governance/health/one_numbers_latest.json,"
            "governance/health/daily_runtime_summary_latest.json"
        ),
        "skip_json_files": False,
        "max_files": 12,
        "state_checkpoint_lines": 500,
        "merge_max_json_file_rows": 64,
    },
    "crypto_governance": {
        "include_streams": "governance_events,governance_watchdog,governance,governance_walk_forward,governance_distillation,governance_canary",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "path_not_contains": (
            "governance/channels/api/default_crypto_schwab/,"
            "governance/channels/ingress/default_crypto_schwab/,"
            "governance/channels/api/crypto_futures_crypto_schwab/,"
            "governance/channels/ingress/crypto_futures_crypto_schwab/,"
            "governance/channels/risk/"
        ),
        "skip_json_files": False,
        "max_files": 12,
        "max_lines_per_file": 8000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 6000,
        "merge_max_json_file_rows": 128,
    },
    "crypto_api_ingress": {
        "include_streams": "governance",
        "path_contains": (
            "governance/channels/api/default_crypto_schwab/,"
            "governance/channels/ingress/default_crypto_schwab/,"
            "governance/channels/api/crypto_futures_crypto_schwab/,"
            "governance/channels/ingress/crypto_futures_crypto_schwab/"
        ),
        "skip_json_files": True,
        "max_files": 6,
        "max_lines_per_file": 16000,
        "state_checkpoint_lines": 2000,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "crypto_runtime": {
        "include_streams": "governance",
        "path_contains": (
            "governance/channels/runtime/default_crypto_coinbase/,"
            "governance/channels/runtime/crypto_futures_crypto_coinbase/,"
            "governance/channels/runtime/default_crypto_schwab/,"
            "governance/channels/runtime/crypto_futures_crypto_schwab/"
        ),
        "skip_json_files": True,
        "max_files": 6,
        "max_lines_per_file": 12000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 8000,
    },
    "crypto_trading_fast": {
        "include_streams": "paper_broker_bridge,top_level_trade_links",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "crypto_explanations": {
        "include_streams": "decision_explanations",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
        "merge_priority": "low",
        "merge_hot_days": 3,
        "hot_retention_enabled": True,
        "hot_retention_max_db_gb": 8.0,
        "hot_retention_trigger_growth_gb": 1.5,
        "hot_retention_trigger_rows": 150000,
        "hot_retention_hot_days": 3,
        "hot_retention_hot_hours": 0,
        "hot_retention_archive_period": "month",
        "hot_retention_archive_retention_days": 365,
        "hot_retention_vacuum_threshold_gb": 4.0,
        "hot_retention_batch_size": 90000,
        "hot_retention_max_rows": 250000,
    },
    "crypto_shadow_attribution": {
        "include_streams": "governance",
        "path_contains": "shadow_pnl_attribution_,shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "path_not_contains": "governance/channels/risk/",
        "skip_json_files": True,
        "max_files": 8,
        "merge_to_primary": False,
        "hot_retention_enabled": True,
        "hot_retention_max_db_gb": 1.5,
        "hot_retention_trigger_growth_gb": 0.5,
        "hot_retention_trigger_rows": 100000,
        "hot_retention_hot_days": 1,
        "hot_retention_hot_hours": 0,
        "hot_retention_archive_period": "day",
        "hot_retention_archive_retention_days": 90,
        "hot_retention_vacuum_threshold_gb": 1.0,
        "hot_retention_batch_size": 120000,
        "hot_retention_max_rows": 250000,
    },
    "crypto_trading": {
        "include_streams": "decisions,trade_logs",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 10,
        "max_lines_per_file": 12000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 8000,
    },
    "governance": {
        "include_streams": "governance_events,governance_watchdog,governance,governance_walk_forward,governance_distillation,governance_canary",
        "path_not_contains": (
            "shadow_pnl_attribution_,"
            "shadow_crypto/,shadow_crypto_futures_crypto/,"
            "default_crypto_coinbase,crypto_futures_crypto_coinbase,"
            "default_crypto_schwab,crypto_futures_crypto_schwab,"
            "governance/watchdog/,"
            "governance/channels/risk/,"
            "governance/channels/runtime/,"
            "governance/events/channel_schema_violations_"
        ),
        "skip_json_files": False,
        "max_lines_per_file": 8000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 6000,
        "merge_max_json_file_rows": 128,
    },
    "support_watchdog": {
        "include_streams": "governance_watchdog",
        "path_contains": "governance/watchdog/",
        "skip_json_files": True,
        "max_files": 4,
        "max_lines_per_file": 96000,
        "state_checkpoint_lines": 4000,
        "merge_max_jsonl_rows": 64000,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "risk_support": {
        "include_streams": "governance",
        "path_contains": "governance/channels/risk/",
        "skip_json_files": True,
        "max_files": 8,
        "max_lines_per_file": 400000,
        "state_checkpoint_lines": 16000,
        "merge_max_jsonl_rows": 0,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "schema_violations": {
        "include_streams": "schema_violations",
        "path_contains": "governance/events/channel_schema_violations_",
        "skip_json_files": True,
        "max_files": 2,
        "max_lines_per_file": 4000,
        "state_checkpoint_lines": 1000,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "writer_progress": {
        "include_streams": "governance_events,governance_watchdog,governance",
        "path_contains": (
            "governance/health/sql_link_service_,"
            "governance/health/jsonl_sql_ingestion_health_,"
            "governance/health/writer_cycle_coordinator_,"
            "governance/health/writer_process_intelligence_,"
            "governance/health/backpressure_drainer_fleet_,"
            "governance/health/backpressure_super_drainer_,"
            "governance/health/drainer_intelligence_layer_"
        ),
        "skip_json_files": False,
        "max_files": 10,
        "max_lines_per_file": 6000,
        "state_checkpoint_lines": 1000,
        "merge_max_jsonl_rows": 4000,
        "merge_max_json_file_rows": 96,
    },
    "predictive_stability": {
        "include_streams": "governance,governance_events",
        "path_contains": (
            "predictive_stability,"
            "pressure_trajectory,"
            "stability_forecast,"
            "halt_forecast,"
            "pressure_memory,"
            "trajectory_memory,"
            "runtime_forecast,"
            "stability_oracle"
        ),
        "skip_json_files": False,
        "max_files": 8,
        "max_lines_per_file": 6000,
        "state_checkpoint_lines": 1000,
        "merge_max_jsonl_rows": 3000,
        "merge_max_json_file_rows": 64,
    },
    "self_healing": {
        "include_streams": "governance,governance_events,governance_watchdog",
        "path_contains": (
            "self_healing,"
            "blocked_surface,"
            "recovery_router,"
            "blackstart,"
            "safe_recovery,"
            "autofix,"
            "incident_closeout,"
            "recovery_plan"
        ),
        "skip_json_files": False,
        "max_files": 8,
        "max_lines_per_file": 6000,
        "state_checkpoint_lines": 1000,
        "merge_max_jsonl_rows": 3000,
        "merge_max_json_file_rows": 64,
    },
    "collector_utility": {
        "include_streams": "governance,data,external_context,external_feeds",
        "path_contains": (
            "collector_utility,"
            "collector_budget,"
            "collection_value,"
            "collector_overlap,"
            "observation_rollup,"
            "collection_maturity,"
            "freshness_value,"
            "collector_thin"
        ),
        "skip_json_files": False,
        "max_files": 8,
        "max_lines_per_file": 6000,
        "state_checkpoint_lines": 1000,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "hot_path_storage": {
        "include_streams": "governance,governance_events",
        "path_contains": (
            "hot_path_storage,"
            "storage_budget,"
            "hot_lane_budget,"
            "warm_lane_budget,"
            "cold_lane_budget,"
            "storage_tier_policy,"
            "queue_watermark,"
            "write_budget"
        ),
        "skip_json_files": False,
        "max_files": 8,
        "max_lines_per_file": 6000,
        "state_checkpoint_lines": 1000,
        "merge_max_jsonl_rows": 3000,
        "merge_max_json_file_rows": 64,
    },
    "admission_evidence": {
        "include_streams": "governance,governance_walk_forward,governance_events",
        "path_contains": (
            "new_bot_admission,"
            "admission_evidence,"
            "sample_depth,"
            "walk_forward_evidence,"
            "promotion_evidence,"
            "replay_hash_evidence,"
            "feature_store_evidence,"
            "teacher_lineage"
        ),
        "skip_json_files": False,
        "max_files": 8,
        "max_lines_per_file": 6000,
        "state_checkpoint_lines": 1000,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "reports": {
        "include_streams": "governance,governance_events",
        "path_contains": (
            "exports/reports/,"
            "docs/showcase/generated/,"
            "operator_cockpit,"
            "system_self_brief,"
            "system_self_model,"
            "showcase,"
            "report_quality"
        ),
        "skip_json_files": False,
        "max_files": 6,
        "max_lines_per_file": 4000,
        "state_checkpoint_lines": 1000,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
    "runtime": {
        "include_streams": "governance",
        "path_contains": "governance/channels/runtime/",
        "path_not_contains": (
            "governance/channels/runtime/default_crypto_coinbase/,"
            "governance/channels/runtime/crypto_futures_crypto_coinbase/,"
            "governance/channels/runtime/default_crypto_schwab/,"
            "governance/channels/runtime/crypto_futures_crypto_schwab/"
        ),
        "skip_json_files": True,
        "max_files": 10,
        "max_lines_per_file": 12000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 8000,
    },
    "trading_fast": {
        "include_streams": "paper_broker_bridge,top_level_trade_links",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "explanations": {
        "include_streams": "decision_explanations",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
        "merge_priority": "low",
        "merge_hot_days": 3,
        "hot_retention_enabled": True,
        "hot_retention_max_db_gb": 12.0,
        "hot_retention_trigger_growth_gb": 2.0,
        "hot_retention_trigger_rows": 200000,
        "hot_retention_hot_days": 3,
        "hot_retention_hot_hours": 0,
        "hot_retention_archive_period": "month",
        "hot_retention_archive_retention_days": 365,
        "hot_retention_vacuum_threshold_gb": 6.0,
        "hot_retention_batch_size": 90000,
        "hot_retention_max_rows": 300000,
    },
    "shadow_attribution": {
        "include_streams": "governance",
        "path_contains": "shadow_pnl_attribution_",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
        "merge_to_primary": False,
        "hot_retention_enabled": True,
        "hot_retention_max_db_gb": 2.0,
        "hot_retention_trigger_growth_gb": 0.5,
        "hot_retention_trigger_rows": 100000,
        "hot_retention_hot_days": 1,
        "hot_retention_hot_hours": 0,
        "hot_retention_archive_period": "day",
        "hot_retention_archive_retention_days": 90,
        "hot_retention_vacuum_threshold_gb": 1.0,
        "hot_retention_batch_size": 120000,
        "hot_retention_max_rows": 250000,
    },
    "aggressive_trading": {
        "include_streams": "decisions,trade_logs",
        "path_contains": "shadow_aggressive_,shadow_intraday_aggressive_,shadow_swing_aggressive_",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 12,
        "max_lines_per_file": 20000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 16000,
    },
    "trading": {
        "include_streams": "decisions,trade_logs",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab,shadow_aggressive_,shadow_intraday_aggressive_,shadow_swing_aggressive_",
        "skip_json_files": True,
        "max_files": 14,
        "max_lines_per_file": 16000,
        "state_checkpoint_lines": 2000,
        "merge_max_jsonl_rows": 12000,
    },
    "data": {
        "include_streams": "data,external_context,external_feeds,feature_store,event_store",
        "path_contains": (
            "data/external_context/,"
            "exports/external_context/,"
            "exports/external_feeds/,"
            "governance/feature_store/,"
            "governance/health/point_in_time_event_store_latest.json,"
            "governance/health/collector_contracts_latest.json,"
            "governance/health/source_verification_latest.json,"
            "governance/health/data_source_divergence_"
        ),
        "skip_json_files": False,
        "max_files": 18,
        "merge_priority": "low",
        "merge_to_primary": False,
    },
}
ARCHIVE_MAINTENANCE_GLOBS = ("*.compact.sqlite3", "*.precompact.bak.sqlite3")
LEGACY_DEFAULT_SHARDS = "trading,governance,data"
PRE_FAST_DEFAULT_SHARDS = "crypto_governance,crypto_trading,governance,trading,data"
PRE_BACKLOG_SPLIT_DEFAULT_SHARDS = "health_fast,crypto_trading_fast,trading_fast,crypto_governance,crypto_trading,governance,trading,data"
CURRENT_DEFAULT_SHARDS = "health_fast,trading_fast,crypto_trading_fast,writer_progress,runtime,crypto_runtime,crypto_api_ingress,aggressive_trading,trading,crypto_trading,predictive_stability,self_healing,hot_path_storage,risk_support,governance,support_watchdog,crypto_governance,schema_violations,collector_utility,admission_evidence,data,reports,explanations,crypto_explanations,shadow_attribution,crypto_shadow_attribution"
SENTINEL_SHARDS = {"health_fast", "writer_progress"}
HOT_SHARDS = {
    "trading_fast",
    "crypto_trading_fast",
    "aggressive_trading",
    "trading",
    "crypto_trading",
    "runtime",
    "crypto_runtime",
    "crypto_api_ingress",
}
WARM_SHARDS = {
    "predictive_stability",
    "self_healing",
    "hot_path_storage",
    "governance",
    "crypto_governance",
    "risk_support",
    "support_watchdog",
    "schema_violations",
}
COLD_SHARDS = {
    "collector_utility",
    "admission_evidence",
    "data",
    "reports",
    "explanations",
    "crypto_explanations",
    "shadow_attribution",
    "crypto_shadow_attribution",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_size_gb(path: Path) -> float:
    try:
        logical_bytes = float(path.stat().st_size)
    except Exception:
        return 0.0

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
        try:
            page_size_row = conn.execute("PRAGMA page_size").fetchone()
            page_count_row = conn.execute("PRAGMA page_count").fetchone()
            freelist_row = conn.execute("PRAGMA freelist_count").fetchone()
        finally:
            conn.close()
        page_size = int(page_size_row[0] if page_size_row and page_size_row[0] is not None else 0)
        page_count = int(page_count_row[0] if page_count_row and page_count_row[0] is not None else 0)
        freelist_count = int(freelist_row[0] if freelist_row and freelist_row[0] is not None else 0)
        live_page_bytes = max(page_count - freelist_count, 0) * max(page_size, 0)
        if live_page_bytes > 0:
            logical_bytes = min(logical_bytes, float(live_page_bytes))
    except Exception:
        pass
    return logical_bytes / (1024.0 ** 3)


def _wal_size_gb(path: Path) -> float:
    return _db_size_gb(Path(f"{path}-wal"))


def _parse_json_output(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    lines = [line for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _busy_progress_summary() -> dict:
    payload = _load_json(PROGRESS_HEALTH)
    if not isinstance(payload, dict):
        return {}
    summary = {
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "status": str(payload.get("status") or ""),
        "current_step": str(payload.get("current_step") or ""),
        "completed_shard_count": _as_int(payload.get("completed_shard_count"), 0),
        "completed_merge_count": _as_int(payload.get("completed_merge_count"), 0),
        "merged_rows_this_cycle": _as_int(payload.get("merged_rows_this_cycle"), 0),
        "primary_db": str(payload.get("primary_db") or ""),
        "primary_db_realpath": str(payload.get("primary_db_realpath") or ""),
        "primary_db_role": str(payload.get("primary_db_role") or ""),
    }
    return {k: v for k, v in summary.items() if v not in ("", 0)}


def _write_json(path: Path, payload: dict) -> None:
    _ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_iso_utc(raw: object) -> datetime | None:
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


def _sanitize_request_env_overrides(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    allowed_prefixes = ("SQL_LINK_SERVICE_",)
    allowed_exact = {
        "INGEST_MAX_DEFERRED_FILES",
        "INGEST_MAX_BYTES_PER_FILE",
        "INGEST_OVERSIZE_PAYLOAD_BYTES",
        "INGEST_TOP_PENDING_FILES",
        "JSONL_SQL_MAX_COLD_LANE_FILES",
        "SQLITE_BATCH_MAX_BYTES",
        "SQLITE_CACHE_SIZE_KB",
        "SQLITE_MMAP_SIZE_MB",
        "SQLITE_TEMP_STORE_MODE",
        "SQLITE_CACHE_SPILL",
        "SQLITE_WAL_AUTOCHECKPOINT_PAGES",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB",
        "BOT_OPS_SQLITE_TEMP_STORE_MODE",
        "BOT_OPS_SQLITE_BUSY_TIMEOUT_MS",
        "BOT_OPS_SQLITE_CACHE_SPILL",
        "BOT_OPS_SQLITE_WAL_AUTOCHECKPOINT_PAGES",
        "LOG_DATA_INGRESS",
        "LOG_API_CALLS",
        "LOG_LOOP_STATE",
        "LOG_SHADOW_PNL_ATTRIBUTION",
        "INGEST_JOURNAL_DAILY_ENABLED",
        "INGEST_JOURNAL_FILE_START_ENABLED",
        "INGEST_JOURNAL_CHECKPOINT_ENABLED",
        "INGEST_JOURNAL_ZERO_PENDING_ENABLED",
        "INGEST_JOURNAL_ERRORS_ALWAYS",
        "BACKLOG_PCORE_ALLOCATION_ACTIVE",
        "BACKLOG_DRAIN_SINGLE_WRITER_ONLY",
        "BACKLOG_PCORE_PREPROCESS_WORKERS",
        "BACKLOG_PCORE_BURST_MODE",
        "BACKLOG_PCORE_BURST_REASON",
        "SQL_LINK_SERVICE_ADAPTIVE_SHARD_ORDER",
        "SQL_LINK_SERVICE_SHARD_ORDER_MODE",
        "SQL_LINK_SERVICE_SENTINEL_SHARDS_FIRST",
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO",
        "RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND",
        "RUNTIME_THROTTLE_RESEARCH_NICE",
        "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG",
        "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN",
        "TRAINING_PCORE_MAX_WORKERS",
        "TRAINING_PCORE_NICE",
    }
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key or "").strip()
        if not name:
            continue
        if not (name in allowed_exact or any(name.startswith(prefix) for prefix in allowed_prefixes)):
            continue
        cleaned[name] = str(value)
    return cleaned


def _load_active_request(path: Path = REQUEST_PATH) -> dict[str, object]:
    payload = _load_json(path)
    if not payload:
        return {}
    if payload.get("active") is False:
        return {}
    expires_utc = _parse_iso_utc(payload.get("expires_utc"))
    if expires_utc is not None and expires_utc <= datetime.now(timezone.utc):
        try:
            path.unlink()
        except Exception:
            pass
        return {}
    overrides = _sanitize_request_env_overrides(payload.get("env_overrides"))
    if not overrides:
        return {}
    return {
        "request_kind": str(payload.get("request_kind") or ""),
        "requested_at": str(payload.get("requested_at") or ""),
        "expires_utc": str(payload.get("expires_utc") or ""),
        "reason": str(payload.get("reason") or ""),
        "p_core_backlog_allocation_contract": (
            payload.get("p_core_backlog_allocation_contract")
            if isinstance(payload.get("p_core_backlog_allocation_contract"), dict)
            else {}
        ),
        "env_overrides": overrides,
    }


def _live_runtime_control_overrides() -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in (RUNTIME_RESOURCE_GUARD_OVERRIDE_PATH, PRESSURE_RELIEF_OVERRIDE_PATH):
        merged.update(_sanitize_request_env_overrides(_load_env_file(path)))
    return merged


def _cycle_runtime_overrides(active_request: dict[str, object]) -> dict[str, str]:
    overrides = _live_runtime_control_overrides()
    request_overrides = active_request.get("env_overrides") if isinstance(active_request.get("env_overrides"), dict) else {}
    overrides.update({str(key): str(value) for key, value in request_overrides.items()})
    return overrides


def _p_core_drain_contract(active_request: dict[str, object]) -> dict[str, Any]:
    overrides = active_request.get("env_overrides") if isinstance(active_request.get("env_overrides"), dict) else {}
    request_contract = (
        active_request.get("p_core_backlog_allocation_contract")
        if isinstance(active_request.get("p_core_backlog_allocation_contract"), dict)
        else {}
    )
    workers = _as_int(overrides.get("BACKLOG_PCORE_PREPROCESS_WORKERS"), _as_int(overrides.get("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 0))
    active = str(overrides.get("BACKLOG_PCORE_ALLOCATION_ACTIVE") or request_contract.get("active") or "").lower() in {"1", "true", "yes"}
    return {
        "active": bool(active),
        "policy": str(request_contract.get("policy") or "p_core_preprocess_single_sql_writer"),
        "single_writer_only": str(overrides.get("BACKLOG_DRAIN_SINGLE_WRITER_ONLY") or overrides.get("SQL_LINK_SERVICE_SINGLE_WRITER_ONLY") or "0") == "1",
        "sqlite_writer_count": 1,
        "preprocess_worker_budget": int(max(workers, 0)),
        "p_core_burst_intelligence": {
            "mode": str(overrides.get("BACKLOG_PCORE_BURST_MODE") or ""),
            "selected_workers": int(max(workers, 0)),
            "reason": str(overrides.get("BACKLOG_PCORE_BURST_REASON") or ""),
        },
        "avoid_background_taskpolicy": str(overrides.get("RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND") or "0") != "1",
        "training_pcore_gate": {
            "allowed_when_backlog_green": str(overrides.get("TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN") or "0") == "1",
            "max_workers": _as_int(overrides.get("TRAINING_PCORE_MAX_WORKERS"), 0),
            "nice_target": _as_int(overrides.get("TRAINING_PCORE_NICE"), _as_int(overrides.get("RUNTIME_THROTTLE_RESEARCH_NICE"), 0)),
        },
    }


@contextmanager
def _temporary_env_overrides(overrides: dict[str, str]):
    if not overrides:
        yield
        return
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dynamic_env_value(overrides: dict[str, str], name: str, default: object) -> str:
    if name in overrides:
        return str(overrides[name])
    return str(os.getenv(name, str(default)))


def _dynamic_env_int(overrides: dict[str, str], name: str, default: int) -> int:
    try:
        return int(str(_dynamic_env_value(overrides, name, default)).strip())
    except Exception:
        return int(default)


def _dynamic_env_float(overrides: dict[str, str], name: str, default: float) -> float:
    try:
        return float(str(_dynamic_env_value(overrides, name, default)).strip())
    except Exception:
        return float(default)


def _dynamic_env_flag(overrides: dict[str, str], name: str, default: bool) -> bool:
    raw = str(_dynamic_env_value(overrides, name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _shard_lane_tier(name: str) -> str:
    safe_name = str(name or "").strip().lower()
    if safe_name in SENTINEL_SHARDS:
        return "sentinel"
    if safe_name in HOT_SHARDS:
        return "hot"
    if safe_name in COLD_SHARDS:
        return "cold"
    if safe_name in WARM_SHARDS:
        return "warm"
    return "warm"


def _shard_health_snapshot(shard: dict[str, object]) -> dict[str, object]:
    health_path = Path(str(shard.get("health_file") or ""))
    health = _load_json(health_path)
    sqlite_bucket = health.get("sqlite", {}) if isinstance(health.get("sqlite"), dict) else {}
    sqlite_json_bucket = health.get("sqlite_json_files", {}) if isinstance(health.get("sqlite_json_files"), dict) else {}
    pending_lines = max(
        _as_int(sqlite_bucket.get("pending_lines"), 0),
        _as_int(health.get("pending_lines"), 0),
        _as_int(health.get("pending_lines_total"), 0),
    )
    pending_json_files = max(
        _as_int(sqlite_json_bucket.get("pending_files"), 0),
        _as_int(sqlite_json_bucket.get("pending"), 0),
        _as_int(health.get("pending_json_files"), 0),
    )
    inserted_rows = max(
        _as_int(sqlite_bucket.get("inserted"), 0) + _as_int(sqlite_json_bucket.get("inserted"), 0),
        _as_int(health.get("inserted"), 0),
    )
    return {
        "path": str(health_path),
        "exists": health_path.exists(),
        "timestamp_utc": str(health.get("timestamp_utc") or ""),
        "pending_lines": int(pending_lines),
        "pending_json_files": int(pending_json_files),
        "inserted_rows": int(inserted_rows),
        "last_rc": _as_int(health.get("rc"), 0),
        "last_status": str(health.get("overall_status") or health.get("status") or ""),
    }


def _shard_link_priority_score(shard: dict[str, object], *, original_index: int) -> tuple[float, dict[str, object]]:
    name = str(shard.get("name") or "").strip()
    tier = _shard_lane_tier(name)
    health = _shard_health_snapshot(shard)
    tier_base = {
        "sentinel": 10000.0,
        "hot": 8500.0,
        "warm": 5200.0,
        "cold": 1800.0,
    }.get(tier, 4000.0)
    pending_lines = _as_int(health.get("pending_lines"), 0)
    pending_json_files = _as_int(health.get("pending_json_files"), 0)
    heat_score = _as_float(shard.get("last_heat_score"), 0.0)
    score = tier_base
    score += min(float(pending_lines) / 8.0, 1800.0)
    score += min(float(pending_json_files) * 45.0, 900.0)
    score += min(max(float(heat_score), 0.0) * 120.0, 720.0)
    if bool(shard.get("heat_promotion_candidate", False)):
        score += 350.0
    if not bool(shard.get("merge_to_primary", True)):
        score -= 120.0
    if str(shard.get("merge_priority") or "").strip().lower() == "low":
        score -= 80.0
    score -= float(original_index) / 1000.0
    return score, {
        "shard": name,
        "tier": tier,
        "score": round(score, 3),
        "original_index": int(original_index),
        "pending_lines": int(pending_lines),
        "pending_json_files": int(pending_json_files),
        "last_heat_score": round(float(heat_score), 3),
        "heat_promotion_candidate": bool(shard.get("heat_promotion_candidate", False)),
        "merge_to_primary": bool(shard.get("merge_to_primary", True)),
        "merge_priority": str(shard.get("merge_priority") or "normal"),
    }


def _adaptive_shard_order_enabled() -> bool:
    mode = str(os.getenv("SQL_LINK_SERVICE_SHARD_ORDER_MODE", "") or "").strip().lower()
    if mode in {"stable", "original", "off"}:
        return False
    return _env_flag("SQL_LINK_SERVICE_ADAPTIVE_SHARD_ORDER", True)


def _prioritize_shards_for_linking(shards: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[tuple[float, int, dict[str, object], dict[str, object]]] = []
    for idx, shard in enumerate(shards):
        score, metadata = _shard_link_priority_score(shard, original_index=idx)
        rows.append((score, idx, shard, metadata))
    stable_order = not _adaptive_shard_order_enabled()
    if stable_order:
        ordered_rows = rows
        policy = "stable_config_order"
    else:
        sentinel_first = _env_flag("SQL_LINK_SERVICE_SENTINEL_SHARDS_FIRST", True)
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                0 if sentinel_first and str(row[2].get("name") or "") in SENTINEL_SHARDS else 1,
                -float(row[0]),
                int(row[1]),
            ),
        )
        policy = "adaptive_hot_pending_sentinel_first"
    ordered_shards = [row[2] for row in ordered_rows]
    priority_rows = [row[3] for row in ordered_rows]
    return ordered_shards, {
        "enabled": not stable_order,
        "policy": policy,
        "planned_order": [str(row.get("name") or "") for row in ordered_shards],
        "priority_rows": priority_rows,
        "sentinel_shards": sorted(SENTINEL_SHARDS),
        "hot_shards": sorted(HOT_SHARDS),
    }


def _connect_primary_db(primary_db: Path, sqlite_timeout_seconds: int) -> sqlite3.Connection:
    return connect_sqlite(
        primary_db,
        project_root=PROJECT_ROOT,
        timeout_seconds=max(float(sqlite_timeout_seconds), 1.0),
    )


def _write_service_progress(
    *,
    cycle_started_utc: str,
    current_step: str,
    lock_path: Path,
    primary_db: Path,
    shards: list[dict[str, object]] | None = None,
    shard_results: list[dict[str, object]] | None = None,
    merge_results: list[dict[str, object]] | None = None,
    merged_rows_this_cycle: int = 0,
    running: bool = True,
    ok: bool = True,
    note: str = "",
    active_request: dict[str, object] | None = None,
    shard_link_plan: dict[str, object] | None = None,
) -> None:
    primary_db_realpath = str(primary_db.resolve(strict=False))
    planned_names = [str((row or {}).get("name", "")) for row in (shards or []) if str((row or {}).get("name", "")).strip()]
    completed_names = [str((row or {}).get("shard", "")) for row in (shard_results or []) if str((row or {}).get("shard", "")).strip()]
    timed_out_names = [
        str((row or {}).get("shard", ""))
        for row in (shard_results or [])
        if isinstance(row, dict) and bool(row.get("timed_out", False)) and str(row.get("shard") or "").strip()
    ]
    completed_set = set(completed_names)
    pending_names = [name for name in planned_names if name not in completed_set]
    payload = {
        "timestamp_utc": _now_utc(),
        "status": ("running" if running else ("ok" if ok else "error")),
        "ok": bool(ok),
        "running": bool(running),
        "cycle_started_utc": str(cycle_started_utc or ""),
        "current_step": str(current_step or ""),
        "lock_path": str(lock_path),
        "primary_db": str(primary_db),
        "primary_db_realpath": primary_db_realpath,
        "primary_db_role": _primary_db_role(primary_db, Path(primary_db_realpath)),
        "maintenance_state_path": str(MAINTENANCE_STATE_PATH),
        "shards": shard_results if isinstance(shard_results, list) else [],
        "merge_results": merge_results if isinstance(merge_results, list) else [],
        "planned_shards": planned_names,
        "completed_shards": completed_names,
        "pending_shards": pending_names,
        "timed_out_shards": timed_out_names,
        "planned_shard_count": len(planned_names),
        "completed_shard_count": len(shard_results or []),
        "pending_shard_count": len(pending_names),
        "timed_out_shard_count": len(timed_out_names),
        "completed_merge_count": len(merge_results or []),
        "merged_rows_this_cycle": int(merged_rows_this_cycle),
        "note": str(note or ""),
        "active_request": active_request if isinstance(active_request, dict) else {},
        "shard_link_plan": shard_link_plan if isinstance(shard_link_plan, dict) else {},
    }
    _write_json(PROGRESS_HEALTH, payload)


def _as_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_int(raw: object, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _configured_primary_db_path(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        return PRIMARY_DB_PATH
    return Path(text).expanduser()


def _primary_db_role(primary_db: Path, primary_db_realpath: Path | None = None) -> str:
    raw_path = str(primary_db)
    real_path = str((primary_db_realpath or primary_db).resolve(strict=False))
    if "/local_fallback_storage/" in raw_path or "/local_fallback_storage/" in real_path:
        return "compatibility_cache"
    if primary_db == PRIMARY_DB_PATH:
        return "routed_primary"
    return "override_primary"


def _as_reason_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _load_maintenance_state(path: Path, *, db_size_gb: float, wal_size_gb: float) -> dict[str, object]:
    payload = _load_json(path)
    wal = payload.get("wal_checkpoint", {}) if isinstance(payload.get("wal_checkpoint"), dict) else {}
    hot = payload.get("hot_retention", {}) if isinstance(payload.get("hot_retention"), dict) else {}
    return {
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "wal_checkpoint": {
            "last_run_utc": str(wal.get("last_run_utc") or ""),
            "baseline_db_size_gb": _as_float(wal.get("baseline_db_size_gb"), db_size_gb),
            "baseline_wal_size_gb": _as_float(wal.get("baseline_wal_size_gb"), wal_size_gb),
            "rows_since_last_run": _as_int(wal.get("rows_since_last_run"), 0),
            "last_trigger_reasons": _as_reason_list(wal.get("last_trigger_reasons")),
        },
        "hot_retention": {
            "last_run_utc": str(hot.get("last_run_utc") or ""),
            "baseline_db_size_gb": _as_float(hot.get("baseline_db_size_gb"), db_size_gb),
            "baseline_wal_size_gb": _as_float(hot.get("baseline_wal_size_gb"), wal_size_gb),
            "rows_since_last_run": _as_int(hot.get("rows_since_last_run"), 0),
            "last_trigger_reasons": _as_reason_list(hot.get("last_trigger_reasons")),
        },
    }


def _load_shard_hot_state(
    maintenance_state: dict[str, object],
    *,
    shard_name: str,
    db_size_gb: float,
) -> dict[str, object]:
    buckets = maintenance_state.setdefault("shard_hot_retention", {})
    if not isinstance(buckets, dict):
        buckets = {}
        maintenance_state["shard_hot_retention"] = buckets
    raw = buckets.get(shard_name, {}) if isinstance(buckets.get(shard_name), dict) else {}
    state = {
        "last_run_utc": str(raw.get("last_run_utc") or ""),
        "last_run_epoch": _as_float(raw.get("last_run_epoch"), 0.0),
        "baseline_db_size_gb": _as_float(raw.get("baseline_db_size_gb"), db_size_gb),
        "rows_since_last_run": _as_int(raw.get("rows_since_last_run"), 0),
        "last_trigger_reasons": _as_reason_list(raw.get("last_trigger_reasons")),
    }
    buckets[shard_name] = state
    return state


def _merged_rows_inserted(merge_results: list[dict[str, object]]) -> int:
    total = 0
    for row in merge_results:
        if not isinstance(row, dict):
            continue
        total += _as_int(row.get("jsonl_rows_inserted"), 0)
        total += _as_int(row.get("json_file_rows_inserted"), 0)
    return max(total, 0)


def _should_skip_low_priority_merge(
    *,
    shard: dict[str, object],
    primary_db_size_gb: float,
    skip_threshold_gb: float,
) -> tuple[bool, str]:
    merge_priority = str(shard.get("merge_priority", "normal") or "normal").strip().lower()
    if merge_priority != "low":
        return False, ""
    if float(skip_threshold_gb) <= 0.0:
        return False, ""
    if float(primary_db_size_gb) < float(skip_threshold_gb):
        return False, ""
    return True, f"primary_db_size_gb>={float(skip_threshold_gb):g}"


def _wal_checkpoint_trigger_reasons(
    *,
    wal_size_gb: float,
    wal_threshold_gb: float,
    wal_growth_gb: float,
    wal_growth_trigger_gb: float,
    rows_since_last_run: int,
    row_trigger: int,
) -> list[str]:
    reasons: list[str] = []
    if wal_size_gb <= 0.0:
        return reasons
    if wal_threshold_gb > 0.0 and wal_size_gb >= wal_threshold_gb:
        reasons.append(f"wal_size_gb>={wal_threshold_gb:g}")
    if wal_growth_trigger_gb > 0.0 and wal_growth_gb >= wal_growth_trigger_gb:
        reasons.append(f"wal_growth_gb>={wal_growth_trigger_gb:g}")
    if row_trigger > 0 and rows_since_last_run >= row_trigger:
        reasons.append(f"rows_since_last_run>={row_trigger}")
    return reasons


def _hot_retention_trigger_reasons(
    *,
    db_size_gb: float,
    max_db_gb: float,
    db_growth_gb: float,
    growth_trigger_gb: float,
    rows_since_last_run: int,
    row_trigger: int,
    has_successful_run: bool,
) -> list[str]:
    reasons: list[str] = []
    if max_db_gb > 0.0 and db_size_gb >= max_db_gb:
        reasons.append(f"{'bootstrap_' if not has_successful_run else ''}db_size_gb>={max_db_gb:g}")
    if not has_successful_run:
        return reasons
    if growth_trigger_gb <= 0.0 and row_trigger <= 0:
        return reasons
    if growth_trigger_gb > 0.0 and db_growth_gb >= growth_trigger_gb:
        reasons.append(f"db_growth_gb>={growth_trigger_gb:g}")
    if row_trigger > 0 and rows_since_last_run >= row_trigger:
        reasons.append(f"rows_since_last_run>={row_trigger}")
    return reasons


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _normalized_shard_config(raw: str) -> str:
    cleaned = ",".join(_parse_csv(raw))
    if not cleaned or cleaned in {LEGACY_DEFAULT_SHARDS, PRE_FAST_DEFAULT_SHARDS, PRE_BACKLOG_SPLIT_DEFAULT_SHARDS}:
        return CURRENT_DEFAULT_SHARDS
    return cleaned


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return float(raw)
    except Exception:
        return float(default)


def _effective_cycle_args(args: argparse.Namespace, overrides: dict[str, str]) -> argparse.Namespace:
    values = vars(args).copy()
    values["interval_seconds"] = max(_dynamic_env_int(overrides, "SQL_LINK_SERVICE_INTERVAL_SECONDS", int(args.interval_seconds)), 10)
    values["link_mode"] = str(_dynamic_env_value(overrides, "SQL_LINK_SERVICE_LINK_MODE", str(args.link_mode or "sqlite")) or "sqlite")
    values["shards"] = str(_dynamic_env_value(overrides, "SQL_LINK_SERVICE_SHARDS", str(args.shards or "")))
    values["preprocess_workers"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_PREPROCESS_WORKERS", int(getattr(args, "preprocess_workers", 1))),
        1,
    )
    values["low_priority_merge_skip_gb"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_LOW_PRIORITY_MERGE_SKIP_GB", float(args.low_priority_merge_skip_gb)),
        0.0,
    )
    values["merge_max_seconds_per_cycle"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE", float(args.merge_max_seconds_per_cycle)),
        0.0,
    )
    values["shard_link_timeout_seconds"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS", int(args.shard_link_timeout_seconds)),
        1,
    )
    values["auto_wal_checkpoint"] = _dynamic_env_flag(overrides, "SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT", bool(args.auto_wal_checkpoint))
    values["wal_checkpoint_threshold_gb"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB", float(args.wal_checkpoint_threshold_gb)),
        0.0,
    )
    values["wal_checkpoint_trigger_growth_gb"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB", float(args.wal_checkpoint_trigger_growth_gb)),
        0.0,
    )
    values["wal_checkpoint_trigger_rows"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS", int(args.wal_checkpoint_trigger_rows)),
        0,
    )
    values["wal_checkpoint_min_interval_seconds"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS", int(args.wal_checkpoint_min_interval_seconds)),
        60,
    )
    values["wal_truncate_max_gb"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB", float(args.wal_truncate_max_gb)),
        0.0,
    )
    values["wal_checkpoint_mode"] = str(
        _dynamic_env_value(overrides, "SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE", str(args.wal_checkpoint_mode or "auto")) or "auto"
    )
    values["auto_hot_retention"] = _dynamic_env_flag(overrides, "SQL_LINK_SERVICE_AUTO_HOT_RETENTION", bool(args.auto_hot_retention))
    values["auto_queue_retention"] = _dynamic_env_flag(
        overrides,
        "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION",
        bool(getattr(args, "auto_queue_retention", True)),
    )
    values["auto_local_fallback_prune"] = _dynamic_env_flag(
        overrides,
        "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE",
        bool(getattr(args, "auto_local_fallback_prune", True)),
    )
    values["hot_retention_max_db_gb"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_HOT_MAX_DB_GB", float(args.hot_retention_max_db_gb)),
        0.0,
    )
    values["hot_retention_trigger_growth_gb"] = max(
        _dynamic_env_float(overrides, "SQL_LINK_SERVICE_HOT_TRIGGER_GROWTH_GB", float(args.hot_retention_trigger_growth_gb)),
        0.0,
    )
    values["hot_retention_trigger_rows"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_HOT_TRIGGER_ROWS", int(args.hot_retention_trigger_rows)),
        0,
    )
    values["hot_retention_hot_days"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_HOT_DAYS", int(args.hot_retention_hot_days)),
        0,
    )
    values["hot_retention_hot_hours"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_HOT_HOURS", int(args.hot_retention_hot_hours)),
        0,
    )
    values["hot_retention_batch_size"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_HOT_BATCH_SIZE", int(args.hot_retention_batch_size)),
        1000,
    )
    values["hot_retention_max_rows"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_HOT_MAX_ROWS", int(args.hot_retention_max_rows)),
        0,
    )
    values["hot_retention_min_interval_seconds"] = max(
        _dynamic_env_int(overrides, "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS", int(args.hot_retention_min_interval_seconds)),
        60,
    )
    return argparse.Namespace(**values)


def _merge_hot_cutoff_utc(days: int) -> str:
    if int(days) <= 0:
        return ""
    return (datetime.now(timezone.utc) - timedelta(days=max(int(days), 1))).isoformat()


def _archive_maintenance_blockers(archive_root: str) -> list[str]:
    root = Path(str(archive_root or "")).expanduser()
    if not root.exists():
        return []
    blockers: list[str] = []
    for pattern in ARCHIVE_MAINTENANCE_GLOBS:
        for path in sorted(root.glob(pattern)):
            blockers.append(str(path))
    return blockers


def _stale_local_fallback_paths(*roots: Path, older_than_seconds: int) -> list[Path]:
    cutoff_epoch = time.time() - max(int(older_than_seconds), 0)
    rows: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            base = Path(root)
        except Exception:
            continue
        if not base.exists():
            continue
        for path in base.rglob("*.local_fallback*"):
            if not path.is_file():
                continue
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            try:
                if float(path.stat().st_mtime) > cutoff_epoch:
                    continue
            except Exception:
                continue
            rows.append(path)
    rows.sort(key=lambda item: float(item.stat().st_mtime) if item.exists() else 0.0)
    return rows


def _prune_stale_local_fallback_artifacts(
    *,
    roots: list[Path],
    older_than_seconds: int,
    max_files: int,
) -> dict[str, object]:
    candidates = _stale_local_fallback_paths(*roots, older_than_seconds=max(int(older_than_seconds), 0))
    deleted_files = 0
    deleted_bytes = 0
    delete_errors = 0
    deleted_paths: list[str] = []
    for path in candidates[: max(int(max_files), 0) or len(candidates)]:
        try:
            size = int(path.stat().st_size)
        except Exception:
            size = 0
        try:
            path.unlink()
            deleted_files += 1
            deleted_bytes += max(size, 0)
            deleted_paths.append(str(path))
        except OSError:
            delete_errors += 1
    return {
        "enabled": True,
        "roots": [str(root) for root in roots],
        "older_than_seconds": int(older_than_seconds),
        "candidate_files": int(len(candidates)),
        "deleted_files": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "delete_errors": int(delete_errors),
        "deleted_paths": deleted_paths[:20],
    }


def _integrity_marker_path(shard_name: str) -> Path:
    safe_name = str(shard_name or "unknown").strip().replace("/", "_") or "unknown"
    return INTEGRITY_MARKER_ROOT / f"{safe_name}.json"


def _load_integrity_marker(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_integrity_marker(path: Path, payload: dict[str, object]) -> None:
    _ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _should_run_deep_integrity_check(*, shard_name: str, sqlite_db: Path) -> tuple[bool, Path, dict[str, object]]:
    marker_path = _integrity_marker_path(shard_name)
    marker = _load_integrity_marker(marker_path)
    min_interval_seconds = max(int(os.getenv("SQL_LINK_SERVICE_DEEP_INTEGRITY_MIN_INTERVAL_SECONDS", "21600")), 300)
    max_inline_db_gb = max(float(os.getenv("SQL_LINK_SERVICE_DEEP_INTEGRITY_MAX_INLINE_DB_GB", "1.5")), 0.0)
    db_size_gb = _db_size_gb(sqlite_db)
    checked_at_epoch = float(marker.get("checked_at_epoch", 0.0) or 0.0)
    marker_ok = bool(marker.get("ok", False))
    if max_inline_db_gb > 0.0 and db_size_gb > max_inline_db_gb:
        return False, marker_path, marker
    if checked_at_epoch <= 0.0 or not marker_ok:
        return True, marker_path, marker
    age_seconds = max(time.time() - checked_at_epoch, 0.0)
    if age_seconds >= float(min_interval_seconds):
        return True, marker_path, marker
    return False, marker_path, marker


def _sqlite_integrity_status(path: Path, *, deep: bool) -> tuple[bool, str]:
    if not path.exists():
        return True, "missing"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(path))
        if deep:
            row = conn.execute("PRAGMA quick_check").fetchone()
            status = str((row or ("unknown",))[0] or "unknown").strip() or "unknown"
            ok = status.lower() == "ok"
        else:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            status = "opened"
            ok = True
        return ok, status
    except sqlite3.DatabaseError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _quarantine_shard_artifacts(
    *,
    shard_name: str,
    sqlite_db: Path,
    state_file: Path,
    health_file: Path,
) -> dict[str, object]:
    deep_check, marker_path, marker = _should_run_deep_integrity_check(shard_name=shard_name, sqlite_db=sqlite_db)
    marker_age_seconds = max(time.time() - _as_float(marker.get("checked_at_epoch"), 0.0), 0.0)
    db_size_gb = _db_size_gb(sqlite_db)
    skip_probe_min_interval_seconds = max(
        int(os.getenv("SQL_LINK_SERVICE_OPEN_PROBE_MIN_INTERVAL_SECONDS", "900")),
        60,
    )
    skip_probe_min_db_gb = max(
        float(os.getenv("SQL_LINK_SERVICE_OPEN_PROBE_SKIP_MIN_DB_GB", "4.0")),
        0.0,
    )
    payload = {
        "triggered": False,
        "reason": "",
        "quarantine_root": "",
        "moved_paths": [],
        "health_file_reset": False,
        "integrity_probe_mode": "quick_check" if deep_check else "open_probe",
        "integrity_marker": str(marker_path),
    }
    if (
        not deep_check
        and bool(marker.get("ok", False))
        and db_size_gb >= float(skip_probe_min_db_gb)
        and marker_age_seconds < float(skip_probe_min_interval_seconds)
    ):
        payload.update(
            {
                "reason": "recent_ok_marker_skip",
                "integrity_probe_mode": "recent_marker_skip",
            }
        )
        return payload

    ok, detail = _sqlite_integrity_status(sqlite_db, deep=deep_check)
    payload["reason"] = str(detail)
    try:
        stat = sqlite_db.stat()
        _save_integrity_marker(
            marker_path,
            {
                "shard_name": str(shard_name),
                "checked_at": _now_utc(),
                "checked_at_epoch": float(time.time()),
                "db_size_bytes": int(stat.st_size),
                "db_mtime": float(stat.st_mtime),
                "ok": bool(ok),
                "detail": str(detail),
                "probe_mode": str(payload["integrity_probe_mode"]),
            },
        )
    except Exception:
        pass
    if ok:
        return payload

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_root = SHARD_DB_ROOT / "corrupt_quarantine" / f"{shard_name}_{stamp}"
    quarantine_root.mkdir(parents=True, exist_ok=True)

    moved_paths: list[str] = []
    for path in (sqlite_db, Path(f"{sqlite_db}-wal"), Path(f"{sqlite_db}-shm"), state_file):
        if not path.exists():
            continue
        dest = quarantine_root / path.name
        path.replace(dest)
        moved_paths.append(str(dest))

    if health_file.exists():
        try:
            health_file.unlink()
            payload["health_file_reset"] = True
        except Exception:
            payload["health_file_reset"] = False

    payload.update(
        {
            "triggered": True,
            "quarantine_root": str(quarantine_root),
            "moved_paths": moved_paths,
        }
    )
    return payload


def _shard_env(name: str, suffix: str) -> str:
    return f"SQL_LINK_SERVICE_SHARD_{name.upper()}_{suffix}"


def _table_exists(conn: sqlite3.Connection, db_alias: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {db_alias}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def _ensure_primary_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            UNIQUE(source_file, line_no)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS json_file_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            stream TEXT NOT NULL,
            modified_at TEXT,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_size_bytes INTEGER NOT NULL DEFAULT 0,
            log_schema_version INTEGER,
            UNIQUE(source_rel, payload_sha1)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shard_merge_state (
            shard_name TEXT PRIMARY KEY,
            last_jsonl_id INTEGER NOT NULL DEFAULT 0,
            last_json_file_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jsonl_records_source_rel_line ON jsonl_records(source_rel, line_no)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jsonl_records_ingested_at ON jsonl_records(ingested_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_json_file_records_source_rel ON json_file_records(source_rel)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_json_file_records_ingested_at ON json_file_records(ingested_at)")


def _read_shard_cursor(conn: sqlite3.Connection, shard_name: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT last_jsonl_id, last_json_file_id FROM shard_merge_state WHERE shard_name = ?",
        (shard_name,),
    ).fetchone()
    if not row:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def _merge_upper_bound_id(
    conn: sqlite3.Connection,
    *,
    table: str,
    last_id: int,
    merge_hot_cutoff_utc: str,
    row_limit: int,
) -> int:
    params: list[object] = [int(last_id)]
    where_sql = "WHERE id > ?"
    if str(merge_hot_cutoff_utc or "").strip():
        where_sql += " AND ingested_at >= ?"
        params.append(str(merge_hot_cutoff_utc))
    if int(row_limit) > 0:
        params.append(int(row_limit))
        row = conn.execute(
            f"""
            SELECT COALESCE(MAX(id), 0)
            FROM (
                SELECT id
                FROM sharddb.{table}
                {where_sql}
                ORDER BY id
                LIMIT ?
            )
            """,
            tuple(params),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT COALESCE(MAX(id), 0) FROM sharddb.{table} {where_sql}",
            tuple(params),
        ).fetchone()
    return int((row or (0,))[0] or 0)


def _probe_shard_merge_state(
    *,
    shard_name: str,
    shard_db: Path,
    primary_db: Path,
    sqlite_timeout_seconds: int,
) -> dict[str, object]:
    result = {
        "shard": shard_name,
        "shard_db": str(shard_db),
        "primary_db": str(primary_db),
        "ok": True,
        "merge_required": False,
        "last_jsonl_id": 0,
        "last_json_file_id": 0,
        "max_jsonl_id": 0,
        "max_json_file_id": 0,
    }
    if not shard_db.exists():
        result["ok"] = False
        result["error"] = "shard_db_missing"
        return result

    conn = _connect_primary_db(primary_db, sqlite_timeout_seconds)
    try:
        conn.execute(f"PRAGMA busy_timeout={int(max(float(sqlite_timeout_seconds), 1.0) * 1000)}")
        _ensure_primary_schema(conn)
        last_jsonl_id, last_json_file_id = _read_shard_cursor(conn, shard_name)
        result["last_jsonl_id"] = last_jsonl_id
        result["last_json_file_id"] = last_json_file_id
        conn.execute("ATTACH DATABASE ? AS sharddb", (str(shard_db),))
        if _table_exists(conn, "sharddb", "jsonl_records"):
            result["max_jsonl_id"] = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.jsonl_records").fetchone()[0] or 0
            )
        if _table_exists(conn, "sharddb", "json_file_records"):
            result["max_json_file_id"] = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.json_file_records").fetchone()[0] or 0
            )
        conn.execute("DETACH DATABASE sharddb")
        result["merge_required"] = bool(
            int(result["max_jsonl_id"]) > int(last_jsonl_id)
            or int(result["max_json_file_id"]) > int(last_json_file_id)
            or (int(result["max_jsonl_id"]) > 0 and int(last_jsonl_id) > int(result["max_jsonl_id"]))
            or (int(result["max_json_file_id"]) > 0 and int(last_json_file_id) > int(result["max_json_file_id"]))
        )
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        return result
    finally:
        conn.close()


def _merge_shard_into_primary(
    *,
    shard_name: str,
    shard_db: Path,
    primary_db: Path,
    sqlite_timeout_seconds: int,
    merge_hot_cutoff_utc: str = "",
    merge_max_jsonl_rows: int = 0,
    merge_max_json_file_rows: int = 0,
) -> dict[str, object]:
    result = {
        "shard": shard_name,
        "shard_db": str(shard_db),
        "primary_db": str(primary_db),
        "ok": True,
        "jsonl_rows_inserted": 0,
        "json_file_rows_inserted": 0,
        "last_jsonl_id": 0,
        "last_json_file_id": 0,
        "jsonl_cursor_reset": False,
        "json_file_cursor_reset": False,
        "merge_hot_cutoff_utc": str(merge_hot_cutoff_utc or ""),
        "max_jsonl_id": 0,
        "max_json_file_id": 0,
        "merge_target_jsonl_id": 0,
        "merge_target_json_file_id": 0,
        "merge_capped": False,
    }
    if not shard_db.exists():
        result["ok"] = False
        result["error"] = "shard_db_missing"
        return result

    conn = _connect_primary_db(primary_db, sqlite_timeout_seconds)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={int(max(float(sqlite_timeout_seconds), 1.0) * 1000)}")
        _ensure_primary_schema(conn)
        last_jsonl_id, last_json_file_id = _read_shard_cursor(conn, shard_name)
        conn.execute("ATTACH DATABASE ? AS sharddb", (str(shard_db),))

        if _table_exists(conn, "sharddb", "jsonl_records"):
            max_jsonl_id = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.jsonl_records").fetchone()[0] or 0
            )
            result["max_jsonl_id"] = max_jsonl_id
            if max_jsonl_id > 0 and last_jsonl_id > max_jsonl_id:
                last_jsonl_id = 0
                result["jsonl_cursor_reset"] = True
            target_jsonl_id = _merge_upper_bound_id(
                conn,
                table="jsonl_records",
                last_id=last_jsonl_id,
                merge_hot_cutoff_utc=str(merge_hot_cutoff_utc or ""),
                row_limit=max(int(merge_max_jsonl_rows), 0),
            )
            result["merge_target_jsonl_id"] = int(target_jsonl_id)
            if int(target_jsonl_id) > int(last_jsonl_id):
                col_list = ",".join(JSONL_COLUMNS)
                params: tuple[object, ...]
                where_sql = "WHERE id > ? AND id <= ?"
                params = (last_jsonl_id, target_jsonl_id)
                if str(merge_hot_cutoff_utc or "").strip():
                    where_sql += " AND ingested_at >= ?"
                    params = (last_jsonl_id, target_jsonl_id, str(merge_hot_cutoff_utc))
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO main.jsonl_records ({col_list})
                    SELECT {col_list}
                    FROM sharddb.jsonl_records
                    {where_sql}
                    ORDER BY id
                    """,
                    params,
                )
                result["jsonl_rows_inserted"] = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
                result["merge_capped"] = bool(
                    int(merge_max_jsonl_rows) > 0 and int(target_jsonl_id) < int(max_jsonl_id)
                )
                last_jsonl_id = target_jsonl_id

        if _table_exists(conn, "sharddb", "json_file_records"):
            max_json_file_id = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.json_file_records").fetchone()[0] or 0
            )
            result["max_json_file_id"] = max_json_file_id
            if max_json_file_id > 0 and last_json_file_id > max_json_file_id:
                last_json_file_id = 0
                result["json_file_cursor_reset"] = True
            target_json_file_id = _merge_upper_bound_id(
                conn,
                table="json_file_records",
                last_id=last_json_file_id,
                merge_hot_cutoff_utc=str(merge_hot_cutoff_utc or ""),
                row_limit=max(int(merge_max_json_file_rows), 0),
            )
            result["merge_target_json_file_id"] = int(target_json_file_id)
            if int(target_json_file_id) > int(last_json_file_id):
                col_list = ",".join(JSON_FILE_COLUMNS)
                params = (last_json_file_id, target_json_file_id)
                where_sql = "WHERE id > ? AND id <= ?"
                if str(merge_hot_cutoff_utc or "").strip():
                    where_sql += " AND ingested_at >= ?"
                    params = (last_json_file_id, target_json_file_id, str(merge_hot_cutoff_utc))
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO main.json_file_records ({col_list})
                    SELECT {col_list}
                    FROM sharddb.json_file_records
                    {where_sql}
                    ORDER BY id
                    """,
                    params,
                )
                result["json_file_rows_inserted"] = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
                result["merge_capped"] = bool(
                    result["merge_capped"]
                    or (int(merge_max_json_file_rows) > 0 and int(target_json_file_id) < int(max_json_file_id))
                )
                last_json_file_id = target_json_file_id

        conn.execute(
            """
            INSERT INTO shard_merge_state (shard_name, last_jsonl_id, last_json_file_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(shard_name) DO UPDATE SET
                last_jsonl_id = excluded.last_jsonl_id,
                last_json_file_id = excluded.last_json_file_id,
                updated_at = excluded.updated_at
            """,
            (shard_name, last_jsonl_id, last_json_file_id, _now_utc()),
        )
        conn.commit()
        conn.execute("DETACH DATABASE sharddb")
        result["last_jsonl_id"] = last_jsonl_id
        result["last_json_file_id"] = last_json_file_id
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        return result
    finally:
        conn.close()


def _run_hot_retention(
    *,
    db_path: Path,
    hot_days: int,
    hot_hours: int,
    batch_size: int,
    max_rows: int,
    archive_db: str,
    archive_root: str,
    archive_period: str,
    archive_retention_days: int,
    archive_prune_vacuum: bool,
    cold_export_root: str,
    cold_export_format: str,
    cold_export_batch_size: int,
    cold_export_compression: str,
    vacuum: bool,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(HOT_RETENTION_SCRIPT),
        "--db",
        str(db_path),
        "--archive-db",
        str(archive_db),
        "--batch-size",
        str(max(batch_size, 1000)),
        "--max-rows",
        str(max(max_rows, 0)),
        "--archive-period",
        str(archive_period or "single"),
        "--archive-retention-days",
        str(max(archive_retention_days, 0)),
        "--json",
    ]
    if int(hot_hours) > 0:
        cmd.extend(["--hot-hours", str(max(hot_hours, 1))])
    else:
        cmd.extend(["--hot-days", str(max(hot_days, 1))])
    if archive_prune_vacuum:
        cmd.append("--archive-prune-vacuum")
    if str(archive_root or "").strip():
        cmd.extend(["--archive-root", str(archive_root)])
    if str(cold_export_root or "").strip():
        cmd.extend(
            [
                "--cold-export-root",
                str(cold_export_root),
                "--cold-export-format",
                str(cold_export_format or "parquet"),
                "--cold-export-batch-size",
                str(max(cold_export_batch_size, 1000)),
                "--cold-export-compression",
                str(cold_export_compression or "zstd"),
            ]
        )
    if vacuum:
        cmd.append("--vacuum")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _shard_inserted_rows(shard_result: dict[str, object]) -> int:
    health = shard_result.get("health", {}) if isinstance(shard_result.get("health"), dict) else {}
    sqlite_bucket = health.get("sqlite", {}) if isinstance(health.get("sqlite"), dict) else {}
    sqlite_json_bucket = health.get("sqlite_json_files", {}) if isinstance(health.get("sqlite_json_files"), dict) else {}
    return _as_int(sqlite_bucket.get("inserted"), 0) + _as_int(sqlite_json_bucket.get("inserted"), 0)


def _shard_pending_lines(shard_result: dict[str, object]) -> int:
    health = shard_result.get("health", {}) if isinstance(shard_result.get("health"), dict) else {}
    sqlite_bucket = health.get("sqlite", {}) if isinstance(health.get("sqlite"), dict) else {}
    return max(_as_int(sqlite_bucket.get("pending_lines"), 0), 0)


def _shard_link_merge_eligible(shard_result: dict[str, object] | None) -> bool:
    if not isinstance(shard_result, dict):
        return False
    if int(shard_result.get("rc", 1)) == 0:
        return True
    if not bool(shard_result.get("timed_out", False)):
        return False
    return bool(_shard_inserted_rows(shard_result) > 0 or _shard_pending_lines(shard_result) <= 0)


def _shard_link_hard_failed(shard_result: dict[str, object]) -> bool:
    return not _shard_link_merge_eligible(shard_result)


def _merge_followup_summary(
    *,
    merge_results: list[dict[str, object]],
    shard_results: list[dict[str, object]],
) -> dict[str, object]:
    budget_exhausted = [
        row
        for row in merge_results
        if isinstance(row, dict)
        and str(row.get("reason") or "").startswith("merge_cycle_budget_exhausted")
    ]
    capped = [row for row in merge_results if isinstance(row, dict) and bool(row.get("merge_capped", False))]
    hard_failed = [row for row in shard_results if isinstance(row, dict) and _shard_link_hard_failed(row)]
    partial_timeout = [
        row
        for row in shard_results
        if isinstance(row, dict) and bool(row.get("timed_out", False)) and _shard_link_merge_eligible(row)
    ]
    skipped_budget_shards = [str(row.get("shard") or "") for row in budget_exhausted if str(row.get("shard") or "").strip()]
    capped_shards = [str(row.get("shard") or "") for row in capped if str(row.get("shard") or "").strip()]
    hard_failed_shards = [str(row.get("shard") or "") for row in hard_failed if str(row.get("shard") or "").strip()]
    followup_reasons = []
    if capped_shards:
        followup_reasons.append("merge_row_cap_remaining")
    if skipped_budget_shards:
        followup_reasons.append("merge_cycle_budget_exhausted")
    if partial_timeout:
        followup_reasons.append("partial_timeout_shards_merge_eligible")
    if hard_failed_shards:
        followup_reasons.append("hard_failed_shards_need_replay")
    return {
        "followup_needed": bool(followup_reasons),
        "catch_up_recommended": bool(capped_shards or skipped_budget_shards or partial_timeout),
        "followup_reasons": followup_reasons,
        "merge_capped_count": len(capped_shards),
        "merge_budget_exhausted_count": len(skipped_budget_shards),
        "partial_timeout_shard_count": len(partial_timeout),
        "hard_failed_shard_count": len(hard_failed_shards),
        "capped_shards": capped_shards[:16],
        "budget_exhausted_shards": skipped_budget_shards[:16],
        "hard_failed_shards": hard_failed_shards[:16],
        "recommended_next_wave": (
            "run another focused writer-cycle coordinator wave after refreshing backpressure"
            if bool(capped_shards or skipped_budget_shards or partial_timeout)
            else ""
        ),
    }


def _run_queue_retention(
    *,
    db_path: str,
    acked_days: int,
    batch_size: int,
    max_rows: int,
    cleanup_consumer_state_days: int,
    prune_orphans: bool,
    orphan_days: int,
    vacuum: bool,
) -> tuple[int, str, str]:
    effective_max_rows = _queue_retention_inline_max_rows(max_rows)
    cmd = [
        str(PY),
        str(QUEUE_RETENTION_SCRIPT),
        "--db",
        str(db_path),
        "--acked-days",
        str(max(acked_days, 1)),
        "--batch-size",
        str(max(batch_size, 1000)),
        "--max-rows",
        str(effective_max_rows),
        "--cleanup-consumer-state-days",
        str(max(cleanup_consumer_state_days, 1)),
        "--json",
    ]
    if prune_orphans:
        cmd.extend(["--prune-orphans", "--orphan-days", str(max(orphan_days, 1))])
    if vacuum:
        cmd.append("--vacuum")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=_queue_retention_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "", "queue_retention_timeout"
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _run_wal_checkpoint(
    *,
    db_path: Path,
    checkpoint_threshold_gb: float,
    truncate_max_gb: float,
    checkpoint_mode: str,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(SQLITE_MAINTENANCE_SCRIPT),
        "--db",
        str(db_path),
        "--checkpoint-only",
        "--wal-checkpoint-threshold-gb",
        str(max(checkpoint_threshold_gb, 0.0)),
        "--wal-truncate-max-gb",
        str(max(truncate_max_gb, 0.0)),
        "--wal-checkpoint-mode",
        str(checkpoint_mode or "auto"),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _build_shards(shard_names: list[str]) -> list[dict[str, object]]:
    day_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    specs: list[dict[str, object]] = []
    try:
        heat_map = ops_data_plane.load_shard_heat_map(PROJECT_ROOT)
    except Exception:
        heat_map = {}
    for name in shard_names:
        safe_name = str(name).strip().lower().replace("-", "_")
        if not safe_name:
            continue
        defaults = DEFAULT_SHARD_DEFS.get(safe_name, {})
        include_streams = os.getenv(_shard_env(safe_name, "INCLUDE_STREAMS"), defaults.get("include_streams", ""))
        exclude_streams = os.getenv(_shard_env(safe_name, "EXCLUDE_STREAMS"), defaults.get("exclude_streams", ""))
        path_contains = os.getenv(_shard_env(safe_name, "PATH_CONTAINS"), defaults.get("path_contains", ""))
        path_not_contains = os.getenv(_shard_env(safe_name, "PATH_NOT_CONTAINS"), defaults.get("path_not_contains", ""))
        skip_json_files = _env_flag(_shard_env(safe_name, "SKIP_JSON_FILES"), bool(defaults.get("skip_json_files", True)))
        merge_to_primary = _env_flag(_shard_env(safe_name, "MERGE_TO_PRIMARY"), bool(defaults.get("merge_to_primary", True)))
        merge_hot_days = max(_env_int(_shard_env(safe_name, "MERGE_HOT_DAYS"), int(defaults.get("merge_hot_days", 0) or 0)), 0)
        hot_retention_enabled = _env_flag(_shard_env(safe_name, "HOT_RETENTION_ENABLED"), bool(defaults.get("hot_retention_enabled", False)))
        max_lines_per_file = max(
            _env_int(_shard_env(safe_name, "MAX_LINES_PER_FILE"), int(defaults.get("max_lines_per_file", 0) or 0)),
            0,
        )
        state_checkpoint_lines = max(
            _env_int(_shard_env(safe_name, "STATE_CHECKPOINT_LINES"), int(defaults.get("state_checkpoint_lines", 10000) or 10000)),
            0,
        )
        merge_max_jsonl_rows = max(
            _env_int(_shard_env(safe_name, "MERGE_MAX_JSONL_ROWS"), int(defaults.get("merge_max_jsonl_rows", 0) or 0)),
            0,
        )
        merge_max_json_file_rows = max(
            _env_int(_shard_env(safe_name, "MERGE_MAX_JSON_FILE_ROWS"), int(defaults.get("merge_max_json_file_rows", 0) or 0)),
            0,
        )
        archive_root_default = SHARD_DB_ROOT / "archives" / safe_name
        cold_export_root_default = SHARD_DB_ROOT / "cold_archives" / safe_name
        max_files = max(_env_int(_shard_env(safe_name, "MAX_FILES"), int(defaults.get("max_files", 0) or 0)), 0)
        heat_row = heat_map.get(safe_name, {})
        if bool(heat_row.get("promotion_candidate", False)):
            max_files += 2
        specs.append(
            {
                "name": safe_name,
                "include_streams": include_streams,
                "exclude_streams": exclude_streams,
                "path_contains": path_contains,
                "path_not_contains": path_not_contains,
                "skip_json_files": skip_json_files,
                "merge_to_primary": merge_to_primary,
                "merge_priority": str(os.getenv(_shard_env(safe_name, "MERGE_PRIORITY"), str(defaults.get("merge_priority", "normal") or "normal"))),
                "merge_hot_days": merge_hot_days,
                "max_files": max_files,
                "max_lines_per_file": max_lines_per_file,
                "state_checkpoint_lines": state_checkpoint_lines,
                "merge_max_jsonl_rows": merge_max_jsonl_rows,
                "merge_max_json_file_rows": merge_max_json_file_rows,
                "sqlite_db": SHARD_DB_ROOT / f"jsonl_link_{safe_name}.sqlite3",
                "state_file": SHARD_STATE_ROOT / f"jsonl_sql_link_state_{safe_name}.json",
                "health_file": HEALTH_ROOT / f"jsonl_sql_ingestion_health_{safe_name}_latest.json",
                "journal_file": HEALTH_ROOT / f"jsonl_ingest_batch_journal_{safe_name}_latest.jsonl",
                "journal_events_file": EVENT_ROOT / f"jsonl_ingest_batches_{safe_name}_{day_utc}.jsonl",
                "invalid_log_file": EVENT_ROOT / f"jsonl_ingestion_invalid_{safe_name}_{day_utc}.jsonl",
                "hot_retention_enabled": hot_retention_enabled,
                "hot_retention_max_db_gb": _env_float(_shard_env(safe_name, "HOT_RETENTION_MAX_DB_GB"), float(defaults.get("hot_retention_max_db_gb", 0.0) or 0.0)),
                "hot_retention_trigger_growth_gb": _env_float(_shard_env(safe_name, "HOT_RETENTION_TRIGGER_GROWTH_GB"), float(defaults.get("hot_retention_trigger_growth_gb", 0.0) or 0.0)),
                "hot_retention_trigger_rows": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_TRIGGER_ROWS"), int(defaults.get("hot_retention_trigger_rows", 0) or 0)), 0),
                "hot_retention_hot_days": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_HOT_DAYS"), int(defaults.get("hot_retention_hot_days", 0) or 0)), 0),
                "hot_retention_hot_hours": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_HOT_HOURS"), int(defaults.get("hot_retention_hot_hours", 0) or 0)), 0),
                "hot_retention_batch_size": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_BATCH_SIZE"), int(defaults.get("hot_retention_batch_size", 50000) or 50000)), 1000),
                "hot_retention_max_rows": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_MAX_ROWS"), int(defaults.get("hot_retention_max_rows", 0) or 0)), 0),
                "hot_retention_archive_period": str(os.getenv(_shard_env(safe_name, "HOT_RETENTION_ARCHIVE_PERIOD"), str(defaults.get("hot_retention_archive_period", "day") or "day"))),
                "hot_retention_archive_retention_days": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_ARCHIVE_RETENTION_DAYS"), int(defaults.get("hot_retention_archive_retention_days", 365) or 365)), 0),
                "hot_retention_vacuum_threshold_gb": _env_float(_shard_env(safe_name, "HOT_RETENTION_VACUUM_THRESHOLD_GB"), float(defaults.get("hot_retention_vacuum_threshold_gb", 0.0) or 0.0)),
                "hot_retention_min_interval_seconds": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_MIN_INTERVAL_SECONDS"), int(defaults.get("hot_retention_min_interval_seconds", 300) or 300)), 60),
                "hot_retention_archive_root": os.getenv(_shard_env(safe_name, "HOT_RETENTION_ARCHIVE_ROOT"), str(archive_root_default)),
                "hot_retention_cold_export_root": os.getenv(_shard_env(safe_name, "HOT_RETENTION_COLD_EXPORT_ROOT"), str(cold_export_root_default if hot_retention_enabled else "")),
                "hot_retention_cold_export_format": str(os.getenv(_shard_env(safe_name, "HOT_RETENTION_COLD_EXPORT_FORMAT"), str(defaults.get("hot_retention_cold_export_format", "parquet") or "parquet"))),
                "hot_retention_cold_export_batch_size": max(_env_int(_shard_env(safe_name, "HOT_RETENTION_COLD_EXPORT_BATCH_SIZE"), int(defaults.get("hot_retention_cold_export_batch_size", 50000) or 50000)), 1000),
                "hot_retention_cold_export_compression": str(os.getenv(_shard_env(safe_name, "HOT_RETENTION_COLD_EXPORT_COMPRESSION"), str(defaults.get("hot_retention_cold_export_compression", "zstd") or "zstd"))),
                "heat_promotion_candidate": bool(heat_row.get("promotion_candidate", False)),
                "last_heat_score": float(heat_row.get("last_heat_score", 0.0) or 0.0),
            }
        )
    return specs


def _run_shard_links(
    *,
    shards: list[dict[str, object]],
    link_mode: str,
    sqlite_timeout_seconds: int,
    sqlite_lock_retries: int,
    sqlite_lock_retry_delay_seconds: float,
    shard_link_timeout_seconds: int,
    preprocess_workers: int = 1,
    progress_callback=None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    worker_count = max(1, min(int(preprocess_workers), max(len(shards), 1), 8))

    def _link_one_shard(shard: dict[str, object]) -> dict[str, object]:
        started = time.monotonic()
        recovery = _quarantine_shard_artifacts(
            shard_name=str(shard["name"]),
            sqlite_db=Path(str(shard["sqlite_db"])),
            state_file=Path(str(shard["state_file"])),
            health_file=Path(str(shard["health_file"])),
        )
        cmd = [
            str(PY),
            str(LINK_SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--mode",
            str(link_mode or "sqlite"),
            "--sqlite-db",
            str(shard["sqlite_db"]),
            "--state-file",
            str(shard["state_file"]),
            "--health-file",
            str(shard["health_file"]),
            "--journal-file",
            str(shard["journal_file"]),
            "--journal-events-file",
            str(shard["journal_events_file"]),
            "--invalid-log-file",
            str(shard["invalid_log_file"]),
            "--sqlite-timeout-seconds",
            str(max(sqlite_timeout_seconds, 30)),
            "--sqlite-lock-retries",
            str(max(sqlite_lock_retries, 0)),
            "--sqlite-lock-retry-delay-seconds",
            str(max(sqlite_lock_retry_delay_seconds, 0.1)),
        ]
        for key, flag in [
            ("include_streams", "--include-streams"),
            ("exclude_streams", "--exclude-streams"),
            ("path_contains", "--path-contains"),
            ("path_not_contains", "--path-not-contains"),
        ]:
            raw = str(shard.get(key, "") or "").strip()
            if raw:
                cmd.extend([flag, raw])
        if int(shard.get("max_files", 0) or 0) > 0:
            cmd.extend(["--max-files", str(int(shard["max_files"]))])
        if int(shard.get("max_lines_per_file", 0) or 0) > 0:
            cmd.extend(["--max-lines-per-file", str(int(shard["max_lines_per_file"]))])
        if int(shard.get("state_checkpoint_lines", 0) or 0) > 0:
            cmd.extend(["--sqlite-state-checkpoint-lines", str(int(shard["state_checkpoint_lines"]))])
        if bool(shard.get("skip_json_files")):
            cmd.append("--skip-json-files")
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(int(shard_link_timeout_seconds), 1),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
            returncode = 124
        duration_ms = round(max(time.monotonic() - started, 0.0) * 1000.0, 3)
        health = {}
        health_path = Path(str(shard["health_file"]))
        if health_path.exists():
            try:
                health = json.loads(health_path.read_text(encoding="utf-8"))
            except Exception:
                health = {}
        return {
            "shard": str(shard["name"]),
            "sqlite_db": str(shard["sqlite_db"]),
            "state_file": str(shard["state_file"]),
            "health_file": str(shard["health_file"]),
            "filters": {
                "include_streams": _parse_csv(str(shard.get("include_streams", "") or "")),
                "exclude_streams": _parse_csv(str(shard.get("exclude_streams", "") or "")),
                "path_contains": _parse_csv(str(shard.get("path_contains", "") or "")),
                "path_not_contains": _parse_csv(str(shard.get("path_not_contains", "") or "")),
                "max_files": int(shard.get("max_files", 0) or 0),
                "max_lines_per_file": int(shard.get("max_lines_per_file", 0) or 0),
                "state_checkpoint_lines": int(shard.get("state_checkpoint_lines", 0) or 0),
                "skip_json_files": bool(shard.get("skip_json_files")),
            },
            "recovery": recovery,
            "rc": returncode,
            "timed_out": timed_out,
            "timeout_seconds": int(shard_link_timeout_seconds),
            "duration_ms": duration_ms,
            "preprocess_worker_count": int(worker_count),
            "parallel_preprocess": bool(worker_count > 1),
            "stdout_tail": "\n".join((stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((stderr or "").splitlines()[-20:]),
            "health": health,
        }

    def _failure_result(shard: dict[str, object], exc: Exception) -> dict[str, object]:
        return {
            "shard": str(shard.get("name") or ""),
            "sqlite_db": str(shard.get("sqlite_db") or ""),
            "state_file": str(shard.get("state_file") or ""),
            "health_file": str(shard.get("health_file") or ""),
            "filters": {},
            "recovery": {},
            "rc": 1,
            "timed_out": False,
            "timeout_seconds": int(shard_link_timeout_seconds),
            "duration_ms": 0.0,
            "preprocess_worker_count": int(worker_count),
            "parallel_preprocess": bool(worker_count > 1),
            "stdout_tail": "",
            "stderr_tail": f"shard_link_worker_failed:{exc}",
            "health": {},
        }

    if worker_count <= 1:
        for shard in shards:
            results.append(_link_one_shard(shard))
            if callable(progress_callback):
                try:
                    progress_callback(list(results))
                except Exception:
                    pass
        return results

    completed_by_index: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {executor.submit(_link_one_shard, shard): idx for idx, shard in enumerate(shards)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            shard = shards[idx]
            try:
                completed_by_index[idx] = future.result()
            except Exception as exc:
                completed_by_index[idx] = _failure_result(shard, exc)
            ordered_partial = [completed_by_index[key] for key in sorted(completed_by_index)]
            if callable(progress_callback):
                try:
                    progress_callback(ordered_partial)
                except Exception:
                    pass
    results = [completed_by_index[idx] for idx in range(len(shards)) if idx in completed_by_index]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Sharded SQL linker manager with incremental merge back into the primary SQLite DB.")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_INTERVAL_SECONDS", "20")))
    parser.add_argument("--sqlite-timeout-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_SQLITE_TIMEOUT", "300")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQL_LINK_SERVICE_LOCK_RETRIES", "200")))
    parser.add_argument("--sqlite-lock-retry-delay-seconds", type=float, default=float(os.getenv("SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS", "0.5")))
    parser.add_argument("--shard-link-timeout-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--link-mode", choices=("sqlite", "both"), default=os.getenv("SQL_LINK_SERVICE_LINK_MODE", "sqlite"))
    parser.add_argument("--lock-path", default=str(PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"))
    parser.add_argument("--primary-db", default=os.getenv("SQL_LINK_SERVICE_PRIMARY_DB", str(PRIMARY_DB_PATH)))
    parser.add_argument("--shards", default=os.getenv("SQL_LINK_SERVICE_SHARDS", ""))
    parser.add_argument("--preprocess-workers", type=int, default=int(os.getenv("SQL_LINK_SERVICE_PREPROCESS_WORKERS", "1")))
    parser.add_argument("--low-priority-merge-skip-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_LOW_PRIORITY_MERGE_SKIP_GB", "120")))
    parser.add_argument("--merge-max-seconds-per-cycle", type=float, default=float(os.getenv("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE", "60")))
    parser.add_argument("--auto-wal-checkpoint", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT", "1") == "1")
    parser.add_argument("--wal-checkpoint-threshold-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB", "2")))
    parser.add_argument("--wal-checkpoint-trigger-growth-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB", "1.5")))
    parser.add_argument("--wal-checkpoint-trigger-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS", "750000")))
    parser.add_argument("--wal-checkpoint-min-interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS", "900")))
    parser.add_argument("--wal-truncate-max-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB", "8")))
    parser.add_argument("--wal-checkpoint-mode", choices=("auto", "passive", "truncate", "restart"), default=os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE", "auto"))
    parser.add_argument("--auto-hot-retention", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_HOT_RETENTION", "1") == "1")
    parser.add_argument("--hot-retention-max-db-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_HOT_MAX_DB_GB", "12")))
    parser.add_argument("--hot-retention-trigger-growth-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_HOT_TRIGGER_GROWTH_GB", "2")))
    parser.add_argument("--hot-retention-trigger-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_TRIGGER_ROWS", "500000")))
    parser.add_argument("--hot-retention-hot-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_DAYS", "3")))
    parser.add_argument("--hot-retention-hot-hours", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_HOURS", "0")))
    parser.add_argument("--hot-retention-batch-size", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_BATCH_SIZE", "120000")))
    parser.add_argument("--hot-retention-max-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_MAX_ROWS", "1000000")))
    parser.add_argument("--hot-retention-min-interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS", "180")))
    parser.add_argument("--hot-retention-vacuum-threshold-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_HOT_VACUUM_THRESHOLD_GB", "20")))
    parser.add_argument("--hot-retention-archive-db", default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_DB", str(PROJECT_ROOT / "data" / "jsonl_link_archive.sqlite3")))
    parser.add_argument("--hot-retention-archive-root", default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_ROOT", str(PROJECT_ROOT / "data" / "jsonl_link_archives")))
    parser.add_argument("--hot-retention-archive-period", choices=("single", "day", "month"), default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_PERIOD", "day"))
    parser.add_argument("--hot-retention-archive-retention-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_RETENTION_DAYS", "365")))
    parser.add_argument("--hot-retention-archive-prune-vacuum", action="store_true", default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_PRUNE_VACUUM", "1") == "1")
    parser.add_argument("--hot-retention-cold-export-root", default=os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_ROOT", str(PROJECT_ROOT / "data" / "cold_archives" / "jsonl_link_primary")))
    parser.add_argument("--hot-retention-cold-export-format", choices=("parquet",), default=os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_FORMAT", "parquet"))
    parser.add_argument("--hot-retention-cold-export-batch-size", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_BATCH_SIZE", "50000")))
    parser.add_argument("--hot-retention-cold-export-compression", default=os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_COMPRESSION", "zstd"))
    parser.add_argument("--auto-queue-retention", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION", "1") == "1")
    parser.add_argument("--queue-retention-db", default=os.getenv("SQL_LINK_SERVICE_QUEUE_DB", str(QUEUE_DB_PATH)))
    parser.add_argument("--queue-retention-max-db-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_QUEUE_MAX_DB_GB", "10")))
    parser.add_argument("--queue-retention-acked-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_ACKED_DAYS", "7")))
    parser.add_argument("--queue-retention-batch-size", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_BATCH_SIZE", "80000")))
    parser.add_argument("--queue-retention-max-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_MAX_ROWS", "240000")))
    parser.add_argument("--queue-retention-min-interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS", "900")))
    parser.add_argument("--queue-retention-vacuum-threshold-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_QUEUE_VACUUM_THRESHOLD_GB", "20")))
    parser.add_argument("--queue-retention-cleanup-consumer-state-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_CLEANUP_CONSUMER_STATE_DAYS", "30")))
    parser.add_argument("--queue-retention-prune-orphans", action="store_true", default=os.getenv("SQL_LINK_SERVICE_QUEUE_PRUNE_ORPHANS", "0") == "1")
    parser.add_argument("--queue-retention-orphan-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS", "45")))
    parser.add_argument("--auto-local-fallback-prune", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE", "1") == "1")
    parser.add_argument("--local-fallback-prune-older-than-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_LOCAL_FALLBACK_PRUNE_OLDER_THAN_SECONDS", "43200")))
    parser.add_argument("--local-fallback-prune-max-files", type=int, default=int(os.getenv("SQL_LINK_SERVICE_LOCAL_FALLBACK_PRUNE_MAX_FILES", "200")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    shard_names = _parse_csv(_normalized_shard_config(args.shards))
    if not shard_names:
        msg = {"ok": False, "reason": "no_shards_configured"}
        print(json.dumps(msg, ensure_ascii=True) if args.json else "sql_link_shard_manager no shards configured")
        return 2

    lock_path = Path(args.lock_path)
    _ensure_directory(lock_path.parent)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        owner = fh.read().strip()
        msg = {"ok": False, "reason": "writer_lock_busy", "lock_path": str(lock_path), "owner": owner}
        progress = _busy_progress_summary()
        if progress:
            msg["service_progress"] = progress
        if args.json:
            print(json.dumps(msg, ensure_ascii=True))
        else:
            detail = ""
            if progress:
                detail = (
                    f" current_step={progress.get('current_step', 'unknown')}"
                    f" completed_shards={progress.get('completed_shard_count', 0)}"
                    f" completed_merges={progress.get('completed_merge_count', 0)}"
                    f" merged_rows={progress.get('merged_rows_this_cycle', 0)}"
                )
            print(f"sql_link_shard_manager busy owner={owner or 'unknown'}{detail}")
        return 0

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={_now_utc()} cmd=sql_link_shard_manager")
    fh.flush()

    primary_db = _configured_primary_db_path(args.primary_db)
    _ensure_directory(primary_db.parent)
    _ensure_directory(SHARD_DB_ROOT)
    _ensure_directory(SHARD_STATE_ROOT)
    _ensure_directory(HEALTH_ROOT)
    _ensure_directory(EVENT_ROOT)

    maintenance_state = _load_maintenance_state(
        MAINTENANCE_STATE_PATH,
        db_size_gb=_db_size_gb(primary_db),
        wal_size_gb=_wal_size_gb(primary_db),
    )

    last_wal_checkpoint_ts = 0.0
    last_hot_retention_ts = 0.0
    last_queue_retention_ts = 0.0

    while True:
        ts = _now_utc()
        cycle_ts = time.time()
        active_request = _load_active_request(REQUEST_PATH)
        request_overrides = _cycle_runtime_overrides(active_request)
        cycle_args = _effective_cycle_args(args, request_overrides)
        cycle_shard_names = _parse_csv(_normalized_shard_config(str(cycle_args.shards or "")))
        if not cycle_shard_names:
            cycle_shard_names = list(shard_names)
        with _temporary_env_overrides(request_overrides):
            shards = _build_shards(cycle_shard_names)
            shards, shard_link_plan = _prioritize_shards_for_linking(shards)
        merge_results: list[dict[str, object]] = []
        _write_service_progress(
            cycle_started_utc=ts,
            current_step="shard_linking",
            lock_path=lock_path,
            primary_db=primary_db,
            shards=shards,
            shard_results=[],
            merge_results=merge_results,
            merged_rows_this_cycle=0,
            running=True,
            ok=True,
            active_request=active_request,
            shard_link_plan=shard_link_plan,
        )
        with _temporary_env_overrides(request_overrides):
            shard_results = _run_shard_links(
                shards=shards,
                link_mode=str(cycle_args.link_mode or "sqlite"),
                sqlite_timeout_seconds=int(cycle_args.sqlite_timeout_seconds),
                sqlite_lock_retries=int(cycle_args.sqlite_lock_retries),
                sqlite_lock_retry_delay_seconds=float(cycle_args.sqlite_lock_retry_delay_seconds),
                shard_link_timeout_seconds=int(cycle_args.shard_link_timeout_seconds),
                preprocess_workers=int(getattr(cycle_args, "preprocess_workers", 1)),
                progress_callback=lambda rows: _write_service_progress(
                    cycle_started_utc=ts,
                    current_step="shard_linking",
                    lock_path=lock_path,
                    primary_db=primary_db,
                    shards=shards,
                    shard_results=rows,
                    merge_results=merge_results,
                    merged_rows_this_cycle=0,
                    running=True,
                    ok=True,
                    active_request=active_request,
                    shard_link_plan=shard_link_plan,
                ),
            )

        _write_service_progress(
            cycle_started_utc=ts,
            current_step="merge_primary",
            lock_path=lock_path,
            primary_db=primary_db,
            shards=shards,
            shard_results=shard_results,
            merge_results=merge_results,
            merged_rows_this_cycle=0,
            running=True,
            ok=True,
            active_request=active_request,
            shard_link_plan=shard_link_plan,
        )
        merge_started_ts = time.time()
        for shard in shards:
            result = next((row for row in shard_results if row["shard"] == shard["name"]), None)
            if _shard_link_merge_eligible(result):
                merge_budget_seconds = max(float(cycle_args.merge_max_seconds_per_cycle), 0.0)
                if (
                    merge_budget_seconds > 0.0
                    and str(shard.get("name") or "") != "health_fast"
                    and (time.time() - merge_started_ts) >= merge_budget_seconds
                ):
                    merge_results.append(
                        {
                            "shard": str(shard["name"]),
                            "shard_db": str(shard["sqlite_db"]),
                            "primary_db": str(primary_db),
                            "ok": True,
                            "skipped": True,
                            "reason": f"merge_cycle_budget_exhausted:{round(time.time() - merge_started_ts, 3)}s",
                            "jsonl_rows_inserted": 0,
                            "json_file_rows_inserted": 0,
                            "last_jsonl_id": 0,
                            "last_json_file_id": 0,
                            "max_jsonl_id": 0,
                            "max_json_file_id": 0,
                        }
                    )
                    _write_service_progress(
                        cycle_started_utc=ts,
                        current_step="merge_primary",
                        lock_path=lock_path,
                        primary_db=primary_db,
                        shards=shards,
                        shard_results=shard_results,
                        merge_results=merge_results,
                        merged_rows_this_cycle=_merged_rows_inserted(merge_results),
                        running=True,
                        ok=True,
                        active_request=active_request,
                        shard_link_plan=shard_link_plan,
                    )
                    continue
                if str(cycle_args.link_mode or "sqlite") != "sqlite":
                    merge_results.append(
                        {
                            "shard": str(shard["name"]),
                            "shard_db": str(shard["sqlite_db"]),
                            "primary_db": str(primary_db),
                            "ok": True,
                            "skipped": True,
                            "reason": f"merge_disabled_for_link_mode_{str(cycle_args.link_mode or 'sqlite')}",
                            "jsonl_rows_inserted": 0,
                            "json_file_rows_inserted": 0,
                            "last_jsonl_id": 0,
                            "last_json_file_id": 0,
                            "max_jsonl_id": 0,
                            "max_json_file_id": 0,
                        }
                    )
                    _write_service_progress(
                        cycle_started_utc=ts,
                        current_step="merge_primary",
                        lock_path=lock_path,
                        primary_db=primary_db,
                        shards=shards,
                        shard_results=shard_results,
                        merge_results=merge_results,
                        merged_rows_this_cycle=_merged_rows_inserted(merge_results),
                        running=True,
                        ok=True,
                        active_request=active_request,
                        shard_link_plan=shard_link_plan,
                    )
                    continue
                skip_merge, skip_reason = _should_skip_low_priority_merge(
                    shard=shard,
                    primary_db_size_gb=_db_size_gb(primary_db),
                    skip_threshold_gb=float(cycle_args.low_priority_merge_skip_gb),
                )
                if skip_merge:
                    merge_results.append(
                        {
                            "shard": str(shard["name"]),
                            "shard_db": str(shard["sqlite_db"]),
                            "primary_db": str(primary_db),
                            "ok": True,
                            "skipped": True,
                            "reason": f"primary_merge_deferred:{skip_reason}",
                            "jsonl_rows_inserted": 0,
                            "json_file_rows_inserted": 0,
                            "last_jsonl_id": 0,
                            "last_json_file_id": 0,
                            "max_jsonl_id": 0,
                            "max_json_file_id": 0,
                        }
                    )
                    _write_service_progress(
                        cycle_started_utc=ts,
                        current_step="merge_primary",
                        lock_path=lock_path,
                        primary_db=primary_db,
                        shards=shards,
                        shard_results=shard_results,
                        merge_results=merge_results,
                        merged_rows_this_cycle=_merged_rows_inserted(merge_results),
                        running=True,
                        ok=True,
                        active_request=active_request,
                        shard_link_plan=shard_link_plan,
                    )
                    continue
                if not bool(shard.get("merge_to_primary", True)):
                    merge_results.append(
                        {
                            "shard": str(shard["name"]),
                            "shard_db": str(shard["sqlite_db"]),
                            "primary_db": str(primary_db),
                            "ok": True,
                            "skipped": True,
                            "reason": "cold_lane_no_primary_merge",
                            "jsonl_rows_inserted": 0,
                            "json_file_rows_inserted": 0,
                            "last_jsonl_id": 0,
                            "last_json_file_id": 0,
                            "max_jsonl_id": 0,
                            "max_json_file_id": 0,
                        }
                    )
                else:
                    merge_probe = _probe_shard_merge_state(
                                shard_name=str(shard["name"]),
                                shard_db=Path(str(shard["sqlite_db"])),
                                primary_db=primary_db,
                                sqlite_timeout_seconds=int(cycle_args.sqlite_timeout_seconds),
                            )
                    if bool(merge_probe.get("ok", False)) and not bool(merge_probe.get("merge_required", True)):
                        merge_results.append(
                            {
                                "shard": str(shard["name"]),
                                "shard_db": str(shard["sqlite_db"]),
                                "primary_db": str(primary_db),
                                "ok": True,
                                "skipped": True,
                                "reason": "merge_up_to_date",
                                "jsonl_rows_inserted": 0,
                                "json_file_rows_inserted": 0,
                                "last_jsonl_id": int(merge_probe.get("last_jsonl_id", 0) or 0),
                                "last_json_file_id": int(merge_probe.get("last_json_file_id", 0) or 0),
                                "max_jsonl_id": int(merge_probe.get("max_jsonl_id", 0) or 0),
                                "max_json_file_id": int(merge_probe.get("max_json_file_id", 0) or 0),
                                "merge_hot_cutoff_utc": _merge_hot_cutoff_utc(int(shard.get("merge_hot_days", 0) or 0)),
                            }
                        )
                    else:
                        merge_results.append(
                            _merge_shard_into_primary(
                                shard_name=str(shard["name"]),
                                shard_db=Path(str(shard["sqlite_db"])),
                                primary_db=primary_db,
                                sqlite_timeout_seconds=int(cycle_args.sqlite_timeout_seconds),
                                merge_hot_cutoff_utc=_merge_hot_cutoff_utc(int(shard.get("merge_hot_days", 0) or 0)),
                                merge_max_jsonl_rows=int(shard.get("merge_max_jsonl_rows", 0) or 0),
                                merge_max_json_file_rows=int(shard.get("merge_max_json_file_rows", 0) or 0),
                            )
                        )
                _write_service_progress(
                    cycle_started_utc=ts,
                    current_step="merge_primary",
                    lock_path=lock_path,
                    primary_db=primary_db,
                    shards=shards,
                    shard_results=shard_results,
                    merge_results=merge_results,
                    merged_rows_this_cycle=_merged_rows_inserted(merge_results),
                    running=True,
                    ok=True,
                    active_request=active_request,
                    shard_link_plan=shard_link_plan,
                )

        shard_hot_retention_results: list[dict[str, object]] = []
        for shard in shards:
            shard_name = str(shard["name"])
            result = next((row for row in shard_results if row["shard"] == shard_name), None)
            if (
                not bool(cycle_args.auto_hot_retention)
                or not result
                or int(result.get("rc", 1)) != 0
                or not bool(shard.get("hot_retention_enabled", False))
            ):
                continue
            db_path = Path(str(shard["sqlite_db"]))
            db_size = _db_size_gb(db_path)
            shard_state = _load_shard_hot_state(maintenance_state, shard_name=shard_name, db_size_gb=db_size)
            shard_state["rows_since_last_run"] = _as_int(shard_state.get("rows_since_last_run"), 0) + _shard_inserted_rows(result)
            growth_gb = max(db_size - _as_float(shard_state.get("baseline_db_size_gb"), db_size), 0.0)
            trigger_reasons = _hot_retention_trigger_reasons(
                db_size_gb=db_size,
                max_db_gb=float(shard.get("hot_retention_max_db_gb", 0.0) or 0.0),
                db_growth_gb=growth_gb,
                growth_trigger_gb=float(shard.get("hot_retention_trigger_growth_gb", 0.0) or 0.0),
                rows_since_last_run=_as_int(shard_state.get("rows_since_last_run"), 0),
                row_trigger=int(shard.get("hot_retention_trigger_rows", 0) or 0),
                has_successful_run=bool(str(shard_state.get("last_run_utc") or "").strip()),
            )
            shard_retention = {
                "shard": shard_name,
                "enabled": True,
                "db_path": str(db_path),
                "db_size_gb_before": round(db_size, 3),
                "max_db_gb": float(shard.get("hot_retention_max_db_gb", 0.0) or 0.0),
                "trigger_growth_gb": float(shard.get("hot_retention_trigger_growth_gb", 0.0) or 0.0),
                "trigger_rows": int(shard.get("hot_retention_trigger_rows", 0) or 0),
                "rows_since_last_run": _as_int(shard_state.get("rows_since_last_run"), 0),
                "db_growth_gb_since_last_run": round(growth_gb, 3),
                "hot_days": int(shard.get("hot_retention_hot_days", 0) or 0),
                "hot_hours": int(shard.get("hot_retention_hot_hours", 0) or 0),
                "archive_root": str(shard.get("hot_retention_archive_root") or ""),
                "cold_export_root": str(shard.get("hot_retention_cold_export_root") or ""),
                "trigger_reasons": list(trigger_reasons),
                "ran": False,
                "rc": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "details": {},
                "skipped_reason": "",
            }
            if trigger_reasons:
                since_last = cycle_ts - float(_as_float(shard_state.get("last_run_epoch"), 0.0))
                if since_last >= max(int(shard.get("hot_retention_min_interval_seconds", 300) or 300), 60):
                    swap_pause, swap_env = _retention_maintenance_paused_for_swap()
                    if swap_pause:
                        shard_retention["skipped_reason"] = "swap_pressure_pause"
                        shard_retention["details"] = _swap_pause_details(swap_env)
                    else:
                        do_vacuum = db_size >= float(shard.get("hot_retention_vacuum_threshold_gb", 0.0) or 0.0)
                        rc, out, err = _run_hot_retention(
                            db_path=db_path,
                            hot_days=int(shard.get("hot_retention_hot_days", 1) or 1),
                            hot_hours=int(shard.get("hot_retention_hot_hours", 0) or 0),
                            batch_size=int(shard.get("hot_retention_batch_size", 50000) or 50000),
                            max_rows=int(shard.get("hot_retention_max_rows", 0) or 0),
                            archive_db=str(Path(str(shard.get("hot_retention_archive_root") or "")) / "latest.sqlite3"),
                            archive_root=str(shard.get("hot_retention_archive_root") or ""),
                            archive_period=str(shard.get("hot_retention_archive_period", "day") or "day"),
                            archive_retention_days=int(shard.get("hot_retention_archive_retention_days", 365) or 365),
                            archive_prune_vacuum=True,
                            cold_export_root=str(shard.get("hot_retention_cold_export_root") or ""),
                            cold_export_format=str(shard.get("hot_retention_cold_export_format", "parquet") or "parquet"),
                            cold_export_batch_size=int(shard.get("hot_retention_cold_export_batch_size", 50000) or 50000),
                            cold_export_compression=str(shard.get("hot_retention_cold_export_compression", "zstd") or "zstd"),
                            vacuum=do_vacuum,
                        )
                        shard_retention.update(
                            {
                                "ran": True,
                                "rc": int(rc),
                                "stdout_tail": "\n".join(out.splitlines()[-12:]),
                                "stderr_tail": "\n".join(err.splitlines()[-12:]),
                                "details": _parse_json_output(out),
                                "vacuum": bool(do_vacuum),
                            }
                        )
                        if int(rc) == 0:
                            shard_state.update(
                                {
                                    "last_run_utc": ts,
                                    "last_run_epoch": float(cycle_ts),
                                    "baseline_db_size_gb": round(_db_size_gb(db_path), 3),
                                    "rows_since_last_run": 0,
                                    "last_trigger_reasons": list(trigger_reasons),
                                }
                            )
                else:
                    shard_retention["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
            else:
                shard_retention["skipped_reason"] = "below_data_trigger"
            shard_retention["db_size_gb_after"] = round(_db_size_gb(db_path), 3)
            shard_hot_retention_results.append(shard_retention)

        local_fallback_prune = {
            "enabled": bool(cycle_args.auto_local_fallback_prune),
            "candidate_files": 0,
            "deleted_files": 0,
            "deleted_bytes": 0,
            "delete_errors": 0,
            "deleted_paths": [],
            "roots": [],
            "older_than_seconds": int(args.local_fallback_prune_older_than_seconds),
        }
        if cycle_args.auto_local_fallback_prune:
            local_fallback_prune = _prune_stale_local_fallback_artifacts(
                roots=[HEALTH_ROOT, PROJECT_ROOT / "data"],
                older_than_seconds=int(args.local_fallback_prune_older_than_seconds),
                max_files=int(args.local_fallback_prune_max_files),
            )

        partial_timeout_shard_count = sum(
            1
            for row in shard_results
            if isinstance(row, dict) and bool(row.get("timed_out", False)) and _shard_link_merge_eligible(row)
        )
        hard_failed_shard_count = sum(1 for row in shard_results if isinstance(row, dict) and _shard_link_hard_failed(row))
        overall_rc = 0 if hard_failed_shard_count <= 0 and all(bool(row.get("ok", False)) for row in merge_results) and all(int(row.get("rc", 0)) == 0 for row in shard_hot_retention_results if bool(row.get("ran", False))) else 1
        merged_rows = _merged_rows_inserted(merge_results)
        merge_followup = _merge_followup_summary(
            merge_results=merge_results,
            shard_results=shard_results,
        )
        for key in ("wal_checkpoint", "hot_retention"):
            bucket = maintenance_state.get(key, {})
            if isinstance(bucket, dict):
                bucket["rows_since_last_run"] = _as_int(bucket.get("rows_since_last_run"), 0) + int(merged_rows)

        wal_size = _wal_size_gb(primary_db)
        checkpoint_state = maintenance_state.get("wal_checkpoint", {}) if isinstance(maintenance_state.get("wal_checkpoint"), dict) else {}
        checkpoint_rows_since_last = _as_int(checkpoint_state.get("rows_since_last_run"), 0)
        checkpoint_wal_growth_gb = max(wal_size - _as_float(checkpoint_state.get("baseline_wal_size_gb"), wal_size), 0.0)
        checkpoint_trigger_reasons = _wal_checkpoint_trigger_reasons(
            wal_size_gb=wal_size,
            wal_threshold_gb=float(cycle_args.wal_checkpoint_threshold_gb),
            wal_growth_gb=checkpoint_wal_growth_gb,
            wal_growth_trigger_gb=float(cycle_args.wal_checkpoint_trigger_growth_gb),
            rows_since_last_run=checkpoint_rows_since_last,
            row_trigger=int(cycle_args.wal_checkpoint_trigger_rows),
        )
        wal_checkpoint = {
            "enabled": bool(cycle_args.auto_wal_checkpoint),
            "wal_size_gb_before": round(wal_size, 3),
            "threshold_gb": float(cycle_args.wal_checkpoint_threshold_gb),
            "trigger_growth_gb": float(cycle_args.wal_checkpoint_trigger_growth_gb),
            "trigger_rows": int(cycle_args.wal_checkpoint_trigger_rows),
            "truncate_max_gb": float(cycle_args.wal_truncate_max_gb),
            "mode": str(cycle_args.wal_checkpoint_mode),
            "rows_since_last_run": int(checkpoint_rows_since_last),
            "wal_growth_gb_since_last_run": round(checkpoint_wal_growth_gb, 3),
            "trigger_reasons": list(checkpoint_trigger_reasons),
            "ran": False,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "details": {},
            "skipped_reason": "",
        }
        if cycle_args.auto_wal_checkpoint and overall_rc == 0 and wal_size <= 0.0:
            wal_checkpoint["skipped_reason"] = "no_wal"
        elif cycle_args.auto_wal_checkpoint and overall_rc == 0 and checkpoint_trigger_reasons:
            since_last = cycle_ts - float(last_wal_checkpoint_ts)
            if since_last >= max(int(cycle_args.wal_checkpoint_min_interval_seconds), 60):
                rc, out, err = _run_wal_checkpoint(
                    db_path=primary_db,
                    checkpoint_threshold_gb=float(cycle_args.wal_checkpoint_threshold_gb),
                    truncate_max_gb=float(cycle_args.wal_truncate_max_gb),
                    checkpoint_mode=str(cycle_args.wal_checkpoint_mode),
                )
                wal_checkpoint.update(
                    {
                        "ran": True,
                        "rc": int(rc),
                        "stdout_tail": "\n".join(out.splitlines()[-12:]),
                        "stderr_tail": "\n".join(err.splitlines()[-12:]),
                        "details": _parse_json_output(out),
                    }
                )
                last_wal_checkpoint_ts = cycle_ts
                if int(rc) == 0:
                    checkpoint_state.update(
                        {
                            "last_run_utc": ts,
                            "baseline_db_size_gb": round(_db_size_gb(primary_db), 3),
                            "baseline_wal_size_gb": round(_wal_size_gb(primary_db), 3),
                            "rows_since_last_run": 0,
                            "last_trigger_reasons": list(checkpoint_trigger_reasons),
                        }
                    )
            else:
                wal_checkpoint["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
        elif cycle_args.auto_wal_checkpoint:
            wal_checkpoint["skipped_reason"] = "link_failed" if overall_rc != 0 else "below_data_trigger"
        wal_checkpoint["wal_size_gb_after"] = round(_wal_size_gb(primary_db), 3)

        archive_blockers = _archive_maintenance_blockers(str(cycle_args.hot_retention_archive_root or ""))
        db_size = _db_size_gb(primary_db)
        hot_state = maintenance_state.get("hot_retention", {}) if isinstance(maintenance_state.get("hot_retention"), dict) else {}
        hot_rows_since_last = _as_int(hot_state.get("rows_since_last_run"), 0)
        hot_db_growth_gb = max(db_size - _as_float(hot_state.get("baseline_db_size_gb"), db_size), 0.0)
        hot_trigger_reasons = _hot_retention_trigger_reasons(
            db_size_gb=db_size,
            max_db_gb=float(cycle_args.hot_retention_max_db_gb),
            db_growth_gb=hot_db_growth_gb,
            growth_trigger_gb=float(cycle_args.hot_retention_trigger_growth_gb),
            rows_since_last_run=hot_rows_since_last,
            row_trigger=int(cycle_args.hot_retention_trigger_rows),
            has_successful_run=bool(str(hot_state.get("last_run_utc") or "").strip()),
        )
        hot_retention = {
            "enabled": bool(cycle_args.auto_hot_retention),
            "db_size_gb_before": round(db_size, 3),
            "max_db_gb": float(cycle_args.hot_retention_max_db_gb),
            "trigger_growth_gb": float(cycle_args.hot_retention_trigger_growth_gb),
            "trigger_rows": int(cycle_args.hot_retention_trigger_rows),
            "hot_days": int(cycle_args.hot_retention_hot_days),
            "hot_hours": int(cycle_args.hot_retention_hot_hours),
            "archive_db": str(cycle_args.hot_retention_archive_db),
            "archive_root": str(cycle_args.hot_retention_archive_root or ""),
            "archive_period": str(cycle_args.hot_retention_archive_period),
            "archive_retention_days": int(cycle_args.hot_retention_archive_retention_days),
            "archive_prune_vacuum": bool(cycle_args.hot_retention_archive_prune_vacuum),
            "cold_export_root": str(cycle_args.hot_retention_cold_export_root or ""),
            "cold_export_format": str(cycle_args.hot_retention_cold_export_format),
            "batch_size": int(cycle_args.hot_retention_batch_size),
            "max_rows": int(cycle_args.hot_retention_max_rows),
            "rows_since_last_run": int(hot_rows_since_last),
            "db_growth_gb_since_last_run": round(hot_db_growth_gb, 3),
            "trigger_reasons": list(hot_trigger_reasons),
            "ran": False,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "details": {},
            "skipped_reason": "",
            "maintenance_blockers": archive_blockers,
        }
        if archive_blockers:
            hot_retention["skipped_reason"] = "archive_maintenance_blocked"
        elif cycle_args.auto_hot_retention and overall_rc == 0 and hot_trigger_reasons:
            since_last = cycle_ts - float(last_hot_retention_ts)
            if since_last >= max(int(cycle_args.hot_retention_min_interval_seconds), 60):
                swap_pause, swap_env = _retention_maintenance_paused_for_swap()
                if swap_pause:
                    hot_retention["skipped_reason"] = "swap_pressure_pause"
                    hot_retention["details"] = _swap_pause_details(swap_env)
                else:
                    do_vacuum = db_size >= float(cycle_args.hot_retention_vacuum_threshold_gb)
                    rc, out, err = _run_hot_retention(
                        db_path=primary_db,
                        hot_days=int(cycle_args.hot_retention_hot_days),
                        hot_hours=int(cycle_args.hot_retention_hot_hours),
                        batch_size=int(cycle_args.hot_retention_batch_size),
                        max_rows=int(cycle_args.hot_retention_max_rows),
                        archive_db=str(cycle_args.hot_retention_archive_db),
                        archive_root=str(cycle_args.hot_retention_archive_root or ""),
                        archive_period=str(cycle_args.hot_retention_archive_period),
                        archive_retention_days=int(cycle_args.hot_retention_archive_retention_days),
                        archive_prune_vacuum=bool(cycle_args.hot_retention_archive_prune_vacuum),
                        cold_export_root=str(cycle_args.hot_retention_cold_export_root or ""),
                        cold_export_format=str(cycle_args.hot_retention_cold_export_format),
                        cold_export_batch_size=int(cycle_args.hot_retention_cold_export_batch_size),
                        cold_export_compression=str(cycle_args.hot_retention_cold_export_compression),
                        vacuum=do_vacuum,
                    )
                    hot_retention.update(
                        {
                            "ran": True,
                            "rc": int(rc),
                            "stdout_tail": "\n".join(out.splitlines()[-12:]),
                            "stderr_tail": "\n".join(err.splitlines()[-12:]),
                            "details": _parse_json_output(out),
                            "vacuum": bool(do_vacuum),
                        }
                    )
                    last_hot_retention_ts = cycle_ts
                    if int(rc) == 0:
                        hot_state.update(
                            {
                                "last_run_utc": ts,
                                "baseline_db_size_gb": round(_db_size_gb(primary_db), 3),
                                "baseline_wal_size_gb": round(_wal_size_gb(primary_db), 3),
                                "rows_since_last_run": 0,
                                "last_trigger_reasons": list(hot_trigger_reasons),
                            }
                        )
            else:
                hot_retention["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
        elif cycle_args.auto_hot_retention:
            hot_retention["skipped_reason"] = "below_data_trigger" if overall_rc == 0 else "link_failed"
        hot_retention["db_size_gb_after"] = round(_db_size_gb(primary_db), 3)

        queue_db_path = Path(str(args.queue_retention_db))
        queue_db_size = _db_size_gb(queue_db_path)
        queue_retention = {
            "enabled": bool(cycle_args.auto_queue_retention),
            "db_path": str(queue_db_path),
            "db_size_gb_before": round(queue_db_size, 3),
            "max_db_gb": float(args.queue_retention_max_db_gb),
            "inline_vacuum_enabled": _queue_retention_inline_vacuum_enabled(),
            "acked_days": int(args.queue_retention_acked_days),
            "batch_size": int(args.queue_retention_batch_size),
            "max_rows": int(args.queue_retention_max_rows),
            "prune_orphans": bool(args.queue_retention_prune_orphans),
            "orphan_days": int(args.queue_retention_orphan_days),
            "cleanup_consumer_state_days": int(args.queue_retention_cleanup_consumer_state_days),
            "ran": False,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "details": {},
            "skipped_reason": "",
        }
        if cycle_args.auto_queue_retention and overall_rc == 0 and queue_db_path.exists() and queue_db_size >= float(args.queue_retention_max_db_gb):
            since_last = cycle_ts - float(last_queue_retention_ts)
            if since_last >= max(int(args.queue_retention_min_interval_seconds), 60):
                swap_pause, swap_env = _retention_maintenance_paused_for_swap()
                if swap_pause:
                    queue_retention["skipped_reason"] = "swap_pressure_pause"
                    queue_retention["details"] = _swap_pause_details(swap_env)
                else:
                    do_vacuum = bool(
                        _queue_retention_inline_vacuum_enabled()
                        and queue_db_size >= float(args.queue_retention_vacuum_threshold_gb)
                    )
                    rc, out, err = _run_queue_retention(
                        db_path=str(queue_db_path),
                        acked_days=int(args.queue_retention_acked_days),
                        batch_size=int(args.queue_retention_batch_size),
                        max_rows=int(args.queue_retention_max_rows),
                        cleanup_consumer_state_days=int(args.queue_retention_cleanup_consumer_state_days),
                        prune_orphans=bool(args.queue_retention_prune_orphans),
                        orphan_days=int(args.queue_retention_orphan_days),
                        vacuum=do_vacuum,
                    )
                    queue_retention.update(
                        {
                            "ran": True,
                            "rc": int(rc),
                            "stdout_tail": "\n".join(out.splitlines()[-12:]),
                            "stderr_tail": "\n".join(err.splitlines()[-12:]),
                            "details": _parse_json_output(out),
                            "vacuum": bool(do_vacuum),
                        }
                    )
                    last_queue_retention_ts = cycle_ts
            else:
                queue_retention["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
        elif cycle_args.auto_queue_retention:
            if not queue_db_path.exists():
                queue_retention["skipped_reason"] = "db_missing"
            else:
                queue_retention["skipped_reason"] = "db_below_threshold" if overall_rc == 0 else "link_failed"
        queue_retention["db_size_gb_after"] = round(_db_size_gb(queue_db_path), 3)

        payload = {
            "timestamp_utc": ts,
            "ok": overall_rc == 0,
            "rc": int(overall_rc),
            "mode": "sharded_merge",
            "link_mode": str(cycle_args.link_mode or "sqlite"),
            "sinks": {
                "sqlite": {
                    "enabled": True,
                    "status": "active",
                },
                "mysql": {
                    "enabled": str(cycle_args.link_mode or "sqlite") in {"mysql", "both"},
                    "status": (
                        "active"
                        if str(cycle_args.link_mode or "sqlite") in {"mysql", "both"}
                        else "disabled_by_link_mode"
                    ),
                },
            },
            "low_priority_merge_skip_gb": float(cycle_args.low_priority_merge_skip_gb),
            "merge_max_seconds_per_cycle": float(cycle_args.merge_max_seconds_per_cycle),
            "preprocess_workers": int(getattr(cycle_args, "preprocess_workers", 1)),
            "parallel_shard_linking": int(getattr(cycle_args, "preprocess_workers", 1)) > 1,
            "primary_db_role": _primary_db_role(primary_db),
            "lock_path": str(lock_path),
            "primary_db": str(primary_db),
            "primary_db_realpath": str(primary_db.resolve(strict=False)),
            "sqlite_db_size_gb": round(_db_size_gb(primary_db), 3),
            "sqlite_wal_size_gb": round(_wal_size_gb(primary_db), 3),
            "sqlite_runtime_settings": resolve_sqlite_runtime_settings(PROJECT_ROOT),
            "queue_db_size_gb": round(_db_size_gb(queue_db_path), 3),
            "maintenance_state_path": str(MAINTENANCE_STATE_PATH),
            "merged_rows_this_cycle": int(merged_rows),
            "partial_timeout_shard_count": int(partial_timeout_shard_count),
            "hard_failed_shard_count": int(hard_failed_shard_count),
            "merge_followup": merge_followup,
            "planned_shard_count": len(shards),
            "completed_shard_count": len(shard_results),
            "timed_out_shard_count": sum(1 for row in shard_results if isinstance(row, dict) and bool(row.get("timed_out", False))),
            "shard_link_plan": shard_link_plan,
            "shards": shard_results,
            "merge_results": merge_results,
            "wal_checkpoint": wal_checkpoint,
            "hot_retention": hot_retention,
            "shard_hot_retention": shard_hot_retention_results,
            "queue_retention": queue_retention,
            "local_fallback_prune": local_fallback_prune,
            "archive_maintenance_blockers": archive_blockers,
            "active_request": active_request,
            "p_core_drain_contract": _p_core_drain_contract(active_request),
        }
        _ensure_directory(LATEST_HEALTH.parent)
        LATEST_HEALTH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        _write_service_progress(
            cycle_started_utc=ts,
            current_step="complete",
            lock_path=lock_path,
            primary_db=primary_db,
            shards=shards,
            shard_results=shard_results,
            merge_results=merge_results,
            merged_rows_this_cycle=int(merged_rows),
            running=False,
            ok=overall_rc == 0,
            active_request=active_request,
            shard_link_plan=shard_link_plan,
        )
        maintenance_state["timestamp_utc"] = ts
        _write_json(MAINTENANCE_STATE_PATH, maintenance_state)

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"sql_link_shard_manager rc={overall_rc} ok={overall_rc == 0} ts={ts}")

        if args.once:
            break
        time.sleep(max(int(cycle_args.interval_seconds), 10))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
