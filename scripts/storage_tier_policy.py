#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from link_jsonl_to_sql import (
    _ingestion_lane_label,
    _storage_temperature_label,
    _storage_tier_label,
    discover_jsonl_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_ROOTS = (
    "decisions",
    "decision_explanations",
    "exports/paper_broker_bridge",
    "governance/events",
    "governance/channels",
    "data/sql_link_shards",
    "governance/content_store",
    "data/stale_stage",
)
DEFAULT_STORAGE_SUFFIXES = {
    ".json",
    ".jsonl",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".wal",
    ".shm",
    ".log",
    ".txt",
}


def _path_size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _economic_value(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("decisions/") or rel.startswith("exports/paper_broker_bridge/"):
        return "critical"
    if rel.startswith("decision_explanations/"):
        return "high"
    if rel.startswith("governance/"):
        return "medium"
    if rel.startswith("data/"):
        return "medium"
    return "low"


def _path_family(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("decisions/"):
        return "decisions"
    if rel.startswith("decision_explanations/"):
        return "decision_explanations"
    if rel.startswith("data/sql_link_shards/"):
        return "sql_link_shards"
    if rel.startswith("governance/content_store/"):
        return "content_store"
    if rel.startswith("governance/events/"):
        return "governance_events"
    if rel.startswith("governance/channels/"):
        return "governance_channels"
    if rel.startswith("exports/paper_broker_bridge/"):
        return "paper_bridge"
    if rel.startswith("data/stale_stage/"):
        return "stale_stage"
    if rel.startswith("data/"):
        return "data"
    if rel.startswith("governance/"):
        return "governance"
    if rel.startswith("exports/"):
        return "exports"
    return "other"


def _service_role(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("decisions/") or rel.startswith("exports/paper_broker_bridge/"):
        return "live_decisioning"
    if rel.startswith("decision_explanations/"):
        return "explainability"
    if rel.startswith("data/sql_link_shards/"):
        return "stateful_sql"
    if rel.startswith("governance/content_store/"):
        return "artifact_store"
    if rel.startswith("governance/events/") or rel.startswith("governance/channels/"):
        return "governance_telemetry"
    if rel.startswith("data/stale_stage/"):
        return "staging_reaper"
    return "analytics"


def _recommended_action(*, role: str, value: str, lane: str) -> str:
    if role == "stateful_sql":
        return "compact_or_mirror_sqlite_shards"
    if role == "explainability":
        return "offload_explanation_history_to_cold_tier"
    if role == "artifact_store":
        return "garbage_collect_or_externalize_artifact_blobs"
    if role == "governance_telemetry":
        return "archive_governance_telemetry"
    if role == "staging_reaper":
        return "reap_stale_stage_artifacts"
    if value == "critical" or lane == "hot_lane":
        return "keep_on_hot_path"
    return "move_to_async_cold_path"


def _is_storage_file(path: Path, *, project_root: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rel = str(path.relative_to(project_root))
    except Exception:
        rel = str(path)
    if rel.startswith("governance/content_store/"):
        return True
    lowered_name = path.name.lower()
    if lowered_name.endswith((".sqlite3-wal", ".sqlite3-shm", ".db-wal", ".db-shm")):
        return True
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return bool(suffixes & DEFAULT_STORAGE_SUFFIXES)


def discover_storage_files(project_root: Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in discover_jsonl_files(project_root):
        seen[str(path.resolve())] = path
    for rel_root in DEFAULT_SCAN_ROOTS:
        root = project_root / rel_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not _is_storage_file(path, project_root=project_root):
                continue
            seen[str(path.resolve())] = path
    return sorted(seen.values(), key=lambda path: str(path))


def _candidate_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    action = str(row.get("recommended_action") or "")
    role = str(row.get("service_role") or "")
    priority = {
        "garbage_collect_or_externalize_artifact_blobs": 5,
        "offload_explanation_history_to_cold_tier": 4,
        "archive_governance_telemetry": 3,
        "compact_or_mirror_sqlite_shards": 2,
        "reap_stale_stage_artifacts": 1,
    }.get(action, 0)
    role_bonus = {
        "artifact_store": 3,
        "explainability": 2,
        "governance_telemetry": 1,
        "stateful_sql": 1,
    }.get(role, 0)
    return (priority, role_bonus, str(row.get("relative_path") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize hot-path debt and cold-tier candidates across active storage lanes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--hot-budget-gb", type=float, default=25.0)
    parser.add_argument("--cold-candidate-min-mb", type=float, default=128.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    files = discover_storage_files(project_root)
    by_temperature: dict[str, dict[str, int]] = {}
    by_storage_tier: dict[str, dict[str, int]] = {}
    by_lane: dict[str, dict[str, int]] = {}
    by_value: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    by_service_role: dict[str, dict[str, int]] = {}
    top_files: list[dict[str, Any]] = []
    cold_path_candidates: list[dict[str, Any]] = []
    async_offload_bytes = 0
    live_hot_path_bytes = 0
    candidate_min_bytes = max(int(float(args.cold_candidate_min_mb) * 1024 * 1024), 1)

    for path in files:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        size_bytes = _path_size_bytes(path)
        temperature = _storage_temperature_label(rel)
        tier = _storage_tier_label(rel)
        lane = _ingestion_lane_label(rel)
        value = _economic_value(rel)
        family = _path_family(rel)
        service_role = _service_role(rel)
        row = {
            "relative_path": rel,
            "size_bytes": int(size_bytes),
            "temperature": temperature,
            "storage_tier": tier,
            "ingestion_lane": lane,
            "economic_value": value,
            "family": family,
            "service_role": service_role,
            "recommended_action": _recommended_action(role=service_role, value=value, lane=lane),
        }
        for bucket, key in (
            (by_temperature, temperature),
            (by_storage_tier, tier),
            (by_lane, lane),
            (by_value, value),
            (by_family, family),
            (by_service_role, service_role),
        ):
            entry = bucket.setdefault(key, {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += int(size_bytes)
        top_files.append(row)
        if service_role in {"live_decisioning", "stateful_sql", "explainability"}:
            live_hot_path_bytes += int(size_bytes)
        if (
            int(size_bytes) >= candidate_min_bytes
            and value != "critical"
            and row["recommended_action"] != "keep_on_hot_path"
        ):
            cold_path_candidates.append(row)
            async_offload_bytes += int(size_bytes)

    top_files.sort(key=lambda row: (-int(row.get("size_bytes", 0) or 0), str(row.get("relative_path") or "")))
    cold_path_candidates.sort(
        key=lambda row: (
            -_candidate_priority(row)[0],
            -_candidate_priority(row)[1],
            -int(row.get("size_bytes", 0) or 0),
            str(row.get("relative_path") or ""),
        )
    )
    hot_budget_bytes = max(int(float(args.hot_budget_gb) * 1024 * 1024 * 1024), 1)
    hot_path_over_budget_bytes = max(int(live_hot_path_bytes) - hot_budget_bytes, 0)
    overall_status = "ready"
    if hot_path_over_budget_bytes > 0:
        overall_status = "degraded"
    if live_hot_path_bytes > hot_budget_bytes * 2:
        overall_status = "blocked"

    recommended_actions: list[str] = []
    if hot_path_over_budget_bytes > 0:
        recommended_actions.append(
            "trim live hot-path storage by offloading explanation, telemetry, and artifact-store payloads before they compete with decisions and SQLite writers"
        )
    if any(str(row.get("service_role") or "") == "stateful_sql" for row in top_files[: max(int(args.top_n), 1)]):
        recommended_actions.append(
            "treat SQL link shards as stateful hot-path debt: compact, checkpoint, or mirror them instead of leaving the writer tier to absorb full historical growth"
        )
    if any(str(row.get("service_role") or "") == "artifact_store" for row in cold_path_candidates[: max(int(args.top_n), 1)]):
        recommended_actions.append(
            "run content-store GC aggressively so large immutable blobs move off the trading path first"
        )
    if any(str(row.get("service_role") or "") == "governance_telemetry" for row in cold_path_candidates[: max(int(args.top_n), 1)]):
        recommended_actions.append(
            "archive governance telemetry on an async cadence rather than keeping nearline event history on the same device as live writes"
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "overall_status": overall_status,
        "file_count": len(files),
        "by_temperature": by_temperature,
        "by_storage_tier": by_storage_tier,
        "by_lane": by_lane,
        "by_economic_value": by_value,
        "by_family": by_family,
        "by_service_role": by_service_role,
        "top_files": top_files[: max(int(args.top_n), 1)],
        "cold_path_candidates": cold_path_candidates[: max(int(args.top_n), 1)],
        "pressure": {
            "hot_files": int((by_temperature.get("hot") or {}).get("files", 0)),
            "hot_bytes": int((by_temperature.get("hot") or {}).get("bytes", 0)),
            "warm_bytes": int((by_temperature.get("warm") or {}).get("bytes", 0)),
            "cold_lane_bytes": int((by_lane.get("cold_lane") or {}).get("bytes", 0)),
            "live_hot_path_bytes": int(live_hot_path_bytes),
            "hot_budget_bytes": int(hot_budget_bytes),
            "hot_path_over_budget_bytes": int(hot_path_over_budget_bytes),
            "async_offload_candidate_bytes": int(async_offload_bytes),
        },
        "upgrade_plan": {
            "storage_split_target": "keep decisions and active SQL state on the hot path while explanations, telemetry, and artifact blobs drain asynchronously",
            "top_hot_path_families": sorted(
                [
                    {"family": key, **value}
                    for key, value in by_family.items()
                    if key in {"decisions", "decision_explanations", "sql_link_shards", "paper_bridge"}
                ],
                key=lambda row: (-int(row.get("bytes", 0) or 0), str(row.get("family") or "")),
            )[:5],
            "recommended_actions": recommended_actions,
        },
    }
    out = project_root / "governance" / "health" / "storage_tier_policy_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_tier_policy file_count={files} hot_bytes={hot} warm_bytes={warm} cold_lane_bytes={cold}".format(
                files=payload["file_count"],
                hot=payload["pressure"]["hot_bytes"],
                warm=payload["pressure"]["warm_bytes"],
                cold=payload["pressure"]["cold_lane_bytes"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
