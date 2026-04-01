import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_gate_dashboard_latest.json"


def _hours_to_minutes(hours: float) -> float:
    return max(float(hours), 0.0) * 60.0


def _days_to_minutes(days: float) -> float:
    return _hours_to_minutes(float(days) * 24.0)


def _artifact_config(project_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "session_ready": {
            "paths": [project_root / "governance" / "health" / "session_ready_latest.json"],
            "max_age_minutes": 15.0,
            "required": True,
        },
        "daily_auto_verify": {
            "paths": [project_root / "governance" / "health" / "daily_auto_verify_latest.json"],
            "max_age_minutes": _hours_to_minutes(36.0),
            "required": False,
        },
        "health_gates": {
            "paths": [project_root / "governance" / "health" / "health_gates_latest.json"],
            "max_age_minutes": 240.0,
            "required": True,
        },
        "sql_link_service": {
            "paths": [
                project_root / "governance" / "health" / "sql_link_service_progress_latest.json",
                project_root / "governance" / "health" / "sql_link_service_latest.json",
            ],
            "max_age_minutes": 30.0,
            "required": True,
        },
        "sql_ingestion": {
            "paths": [
                project_root / "governance" / "health" / "jsonl_sql_ingestion_health_trading_latest.json",
                project_root / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json",
                project_root / "governance" / "health" / "jsonl_sql_ingestion_health_data_latest.json",
                project_root / "governance" / "health" / "jsonl_sql_ingestion_health_governance_latest.json",
            ],
            "max_age_minutes": 30.0,
            "required": True,
        },
        "promotion_readiness": {
            "paths": [project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "new_bot_graduation": {
            "paths": [project_root / "governance" / "walk_forward" / "new_bot_graduation_latest.json"],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "replay_hash_registry_guard": {
            "paths": [project_root / "governance" / "health" / "replay_hash_registry_guard_latest.json"],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "promotion_quality_gate": {
            "paths": [project_root / "governance" / "health" / "promotion_quality_gate_latest.json"],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "retrain_artifact_freshness": {
            "paths": [project_root / "governance" / "health" / "retrain_artifact_freshness_latest.json"],
            "max_age_minutes": _hours_to_minutes(24.0),
            "required": False,
        },
        "retrain_scorecard": {
            "paths": [project_root / "governance" / "health" / "retrain_scorecard_latest.json"],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
        "official_macro_context_sync": {
            "paths": [project_root / "governance" / "health" / "official_macro_context_sync_latest.json"],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
        "live_macro_media": {
            "paths": [project_root / "governance" / "health" / "live_macro_media_status.json"],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _payload_timestamp(payload: Dict[str, Any], path: Path) -> datetime | None:
    for key in ("timestamp_utc", "updated_at_utc", "updated_at", "created_at", "ended_utc", "started_utc"):
        ts = _parse_iso_utc(payload.get(key))
        if ts is not None:
            return ts
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _pick_latest_artifact(paths: Iterable[Path]) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    fallback: Path | None = None
    for path in paths:
        fallback = fallback or path
        payload = _load_json(path)
        if not payload:
            continue
        ts = _payload_timestamp(payload, path)
        score = float(ts.timestamp()) if ts is not None else 0.0
        candidates.append((score, path, payload))
    if not candidates:
        return fallback or Path(""), {}
    candidates.sort(key=lambda row: row[0])
    _, path, payload = candidates[-1]
    return path, payload


def _infer_ok(payload: Dict[str, Any]) -> bool | None:
    if not isinstance(payload, dict) or not payload:
        return None
    raw_ok = payload.get("ok")
    if isinstance(raw_ok, bool):
        return raw_ok
    if "hard_gate_triggered" in payload:
        return not bool(payload.get("hard_gate_triggered"))
    if "promote_ok" in payload:
        return bool(payload.get("promote_ok"))
    if "learning_ready" in payload:
        return bool(payload.get("learning_ready"))
    status = str(payload.get("status", "") or "").strip().lower()
    if status in {"ok", "healthy", "ready", "success", "pass"}:
        return True
    if status in {"error", "failed", "fail", "degraded", "stale"}:
        return False
    return None


def _infer_status(payload: Dict[str, Any], ok_value: bool | None) -> str:
    status = str(payload.get("status", "") or "").strip().lower()
    if status:
        return status
    if ok_value is True:
        return "ok"
    if ok_value is False:
        return "error"
    return "unknown"


def _artifact_summary(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if name == "session_ready":
        checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
        return {
            "expected_profiles": payload.get("expected_profiles") if isinstance(payload.get("expected_profiles"), list) else [],
            "check_count": len(checks),
        }
    if name == "daily_auto_verify":
        return {
            "failed_checks": payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [],
            "completed_checks": int(payload.get("completed_checks", 0) or 0),
        }
    if name == "health_gates":
        return {
            "data_quality_score": float(payload.get("data_quality_score", 0.0) or 0.0),
            "hard_gate_triggered": bool(payload.get("hard_gate_triggered", False)),
            "inputs": payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {},
        }
    if name == "sql_ingestion":
        sqlite = payload.get("sqlite") if isinstance(payload.get("sqlite"), dict) else {}
        return {
            "pending_lines": int(sqlite.get("pending_lines", 0) or 0),
            "oldest_uningested_age_seconds": float(sqlite.get("oldest_uningested_age_seconds", 0.0) or 0.0),
            "invalid_lines": int(sqlite.get("invalid", 0) or 0),
            "files_discovered": int(payload.get("files_discovered", 0) or 0),
        }
    if name == "promotion_readiness":
        return {
            "promote_ok": bool(payload.get("promote_ok", False)),
            "considered_bots": int(payload.get("considered_bots", 0) or 0),
            "failed_bots": int(payload.get("failed_bots", 0) or 0),
            "fail_share": float(payload.get("fail_share", 0.0) or 0.0),
        }
    if name == "new_bot_graduation":
        return {
            "ok": bool(payload.get("ok", False)),
            "mature_bots": int(((payload.get("maturity") or {}).get("mature_bots", 0)) or 0),
            "immature_active_count": int(payload.get("immature_active_count", 0) or 0),
        }
    if name == "replay_hash_registry_guard":
        return {
            "ok": bool(payload.get("ok", False)),
            "failed_checks": payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [],
        }
    if name == "promotion_quality_gate":
        return {
            "ok": bool(payload.get("ok", False)),
            "failed_checks": payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [],
        }
    if name == "sql_link_service":
        return {
            "running": bool(payload.get("running", False)),
            "current_step": str(payload.get("current_step", "") or ""),
            "completed_shard_count": int(payload.get("completed_shard_count", 0) or 0),
            "completed_merge_count": int(payload.get("completed_merge_count", 0) or 0),
            "merged_rows_this_cycle": int(payload.get("merged_rows_this_cycle", 0) or 0),
        }
    if name == "retrain_artifact_freshness":
        return {
            "failed_checks": payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [],
            "max_age_minutes": float(payload.get("max_age_minutes", 0.0) or 0.0),
        }
    if name == "retrain_scorecard":
        outcomes = payload.get("target_outcomes") if isinstance(payload.get("target_outcomes"), list) else []
        status_counts = payload.get("status_counts") if isinstance(payload.get("status_counts"), dict) else {}
        trained_count = int(status_counts.get("trained", 0) or 0)
        return {
            "target_count": int(payload.get("target_count", 0) or 0),
            "trained_count": trained_count,
            "failure_count": int(payload.get("failure_count", 0) or 0),
            "master_update_status": str(payload.get("master_update_status", "") or ""),
            "outcome_count": len(outcomes),
        }
    if name == "official_macro_context_sync":
        sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
        return {
            "source_count": len(sources),
            "ok_sources": sum(1 for row in sources.values() if isinstance(row, dict) and bool(row.get("ok", False))),
        }
    if name == "live_macro_media":
        return {
            "source": str(payload.get("source", "") or ""),
            "speaker": str(payload.get("speaker", "") or ""),
            "learning_ready": bool(payload.get("learning_ready", False)),
            "training_feature_count": int(payload.get("training_feature_count", 0) or 0),
        }
    return {}


def _latest_shadow_loop_timestamp(project_root: Path) -> datetime | None:
    latest: datetime | None = None
    for path in (project_root / "governance" / "health").glob("shadow_loop_*.json"):
        payload = _load_json(path)
        ts = _payload_timestamp(payload, path)
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _lock_owner_pid(lock_path_text: str) -> int | None:
    lock_path = Path(str(lock_path_text or "").strip()).expanduser()
    if not lock_path.exists():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    for token in raw.split():
        if not token.startswith("pid="):
            continue
        try:
            return int(token.split("=", 1)[1])
        except Exception:
            return None
    return None


def _live_sql_writer_pid(artifact: Dict[str, Any]) -> int | None:
    if not bool(artifact.get("exists")):
        return None
    summary = artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    if not bool(summary.get("running")):
        return None
    age_minutes = artifact.get("age_minutes")
    max_age_minutes = float(artifact.get("max_age_minutes", 30.0) or 30.0)
    if age_minutes is not None and float(age_minutes) > max_age_minutes * 8.0:
        return None
    lock_path_text = str(artifact.get("path") or "")
    payload_path = Path(lock_path_text) if lock_path_text else Path("")
    payload = _load_json(payload_path) if payload_path.exists() else {}
    pid = _lock_owner_pid(str(payload.get("lock_path") or ""))
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def _severity_from_attention(attention: list[str]) -> int:
    if any(item.endswith("_missing") for item in attention):
        return 3
    degraded_markers = (
        "_stale",
        "_not_ok",
        "health_gates_hard_gate_triggered",
        "daily_auto_verify_not_ok",
    )
    if any(any(marker in item for marker in degraded_markers) for item in attention):
        return 2
    if attention:
        return 1
    return 0


def _registry_summary(project_root: Path) -> Dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    active_rows = [row for row in sub_bots if isinstance(row, dict) and bool(row.get("active"))]
    deleted_rows = [row for row in sub_bots if isinstance(row, dict) and bool(row.get("deleted_from_rotation"))]
    return {
        "updated_at_utc": str(registry.get("updated_at_utc", "") or ""),
        "total_bots": int(summary.get("total_bots", len(sub_bots)) or len(sub_bots)),
        "active_bots": int(summary.get("active_bots", len(active_rows)) or len(active_rows)),
        "deleted_from_rotation": int(summary.get("deleted_from_rotation", len(deleted_rows)) or len(deleted_rows)),
        "top_active": summary.get("top_active") if isinstance(summary.get("top_active"), list) else [],
        "deletion_guard_ok": bool(summary.get("deletion_guard_ok", False)),
        "deletion_guard_reason": str(summary.get("deletion_guard_reason", "") or ""),
    }


def _resolved_daily_auto_verify_failures(
    daily_verify_payload: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
) -> tuple[list[str], list[str]]:
    failed = daily_verify_payload.get("failed_checks") if isinstance(daily_verify_payload.get("failed_checks"), list) else []
    unresolved: list[str] = []
    resolved: list[str] = []
    for name in failed:
        key = str(name or "").strip()
        if key == "new_bot_graduation_gate" and artifacts.get("new_bot_graduation", {}).get("ok") is True:
            resolved.append(key)
            continue
        if key == "replay_hash_registry_guard" and artifacts.get("replay_hash_registry_guard", {}).get("ok") is True:
            resolved.append(key)
            continue
        if key == "promotion_quality_gate":
            promo_ok = artifacts.get("promotion_quality_gate", {}).get("ok")
            if promo_ok is True:
                resolved.append(key)
                continue
        unresolved.append(key)
    return unresolved, resolved


def build_dashboard(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    artifacts: Dict[str, Dict[str, Any]] = {}
    attention: List[str] = []
    severity = 0

    for name, cfg in _artifact_config(project_root).items():
        path, payload = _pick_latest_artifact(cfg["paths"])
        exists = bool(path and path.exists() and payload)
        ts = _payload_timestamp(payload, path) if exists else None
        age_minutes = max((now - ts).total_seconds() / 60.0, 0.0) if ts is not None else None
        ok_value = _infer_ok(payload) if exists else None
        status = _infer_status(payload, ok_value) if exists else "missing"
        stale = bool(age_minutes is not None and age_minutes > float(cfg["max_age_minutes"]))
        summary = _artifact_summary(name, payload) if exists else {}
        if name == "sql_link_service" and stale:
            live_pid = _live_sql_writer_pid(
                {
                    "path": str(path) if path else "",
                    "exists": exists,
                    "age_minutes": age_minutes,
                    "max_age_minutes": float(cfg["max_age_minutes"]),
                    "summary": summary,
                }
            )
            if live_pid is not None:
                stale = False
                summary = {
                    **summary,
                    "freshness_inferred_from_live_lock": True,
                    "lock_owner_pid": live_pid,
                }
        if name == "sql_ingestion" and stale:
            sql_service = artifacts.get("sql_link_service", {})
            sql_service_summary = sql_service.get("summary") if isinstance(sql_service.get("summary"), dict) else {}
            sql_service_fresh = bool(sql_service.get("exists")) and (not bool(sql_service.get("stale")))
            sql_service_live = bool(sql_service_summary.get("running")) or (sql_service.get("ok") is True)
            pending_lines = int(summary.get("pending_lines", 0) or 0)
            invalid_lines = int(summary.get("invalid_lines", 0) or 0)
            if sql_service_fresh and sql_service_live and pending_lines == 0 and invalid_lines == 0:
                stale = False
                summary = {
                    **summary,
                    "freshness_via_service_heartbeat": True,
                    "service_heartbeat_age_minutes": sql_service.get("age_minutes"),
                }
        artifacts[name] = {
            "path": str(path) if path else "",
            "exists": exists,
            "timestamp_utc": ts.isoformat() if ts is not None else "",
            "age_minutes": round(age_minutes, 4) if age_minutes is not None else None,
            "max_age_minutes": float(cfg["max_age_minutes"]),
            "required": bool(cfg["required"]),
            "ok": ok_value,
            "stale": stale,
            "status": status,
            "summary": summary,
        }
        if bool(cfg["required"]) and not exists:
            attention.append(f"{name}_missing")
            severity = max(severity, 3)
        elif bool(cfg["required"]) and stale:
            attention.append(f"{name}_stale")
            severity = max(severity, 2)
        elif ok_value is False and name in {"session_ready", "health_gates", "retrain_artifact_freshness", "sql_link_service"}:
            attention.append(f"{name}_not_ok")
            severity = max(severity, 2)

    health_inputs = artifacts.get("health_gates", {}).get("summary", {}).get("inputs", {})
    if artifacts.get("health_gates", {}).get("summary", {}).get("hard_gate_triggered"):
        attention.append("health_gates_hard_gate_triggered")
        severity = max(severity, 2)
    daily_verify_payload = _load_json(Path(str(artifacts.get("daily_auto_verify", {}).get("path", "") or "")))
    unresolved_daily_verify, resolved_daily_verify = _resolved_daily_auto_verify_failures(daily_verify_payload, artifacts)
    if artifacts.get("daily_auto_verify", {}).get("exists"):
        artifacts["daily_auto_verify"]["summary"]["resolved_failed_checks"] = resolved_daily_verify
        artifacts["daily_auto_verify"]["summary"]["effective_failed_checks"] = unresolved_daily_verify
    if artifacts.get("daily_auto_verify", {}).get("ok") is False and unresolved_daily_verify:
        if not (
            set(unresolved_daily_verify) == {"promotion_quality_gate"}
            and artifacts.get("promotion_readiness", {}).get("summary", {}).get("promote_ok") is False
        ):
            attention.append("daily_auto_verify_not_ok")
            severity = max(severity, 2)
    if artifacts.get("promotion_readiness", {}).get("summary", {}).get("promote_ok") is False:
        attention.append("promotion_not_ready")
        severity = max(severity, 1)
    if float(health_inputs.get("blocked_rate", 0.0) or 0.0) >= 0.50:
        attention.append("blocked_rate_elevated")
        severity = max(severity, 1)

    sql_ingestion_artifact = artifacts.get("sql_ingestion", {})
    sql_service_artifact = artifacts.get("sql_link_service", {})
    sql_ingestion_summary = sql_ingestion_artifact.get("summary") if isinstance(sql_ingestion_artifact.get("summary"), dict) else {}
    if (
        "sql_link_service_stale" in attention
        and bool(sql_ingestion_artifact.get("exists"))
        and not bool(sql_ingestion_artifact.get("stale"))
        and int(sql_ingestion_summary.get("pending_lines", 0) or 0) == 0
        and int(sql_ingestion_summary.get("invalid_lines", 0) or 0) == 0
    ):
        attention = [item for item in attention if item != "sql_link_service_stale"]
        sql_service_summary = sql_service_artifact.get("summary") if isinstance(sql_service_artifact.get("summary"), dict) else {}
        sql_service_artifact["summary"] = {
            **sql_service_summary,
            "freshness_inferred_from_sql_ingestion": True,
        }
        artifacts["sql_link_service"] = sql_service_artifact

    if "session_ready_stale" in attention:
        latest_shadow_ts = _latest_shadow_loop_timestamp(project_root)
        if latest_shadow_ts is not None:
            shadow_age_minutes = max((now - latest_shadow_ts).total_seconds() / 60.0, 0.0)
            if shadow_age_minutes <= float(artifacts.get("session_ready", {}).get("max_age_minutes", 15.0) or 15.0):
                attention = [item for item in attention if item != "session_ready_stale"]
                session_ready_artifact = artifacts.get("session_ready", {})
                session_ready_summary = session_ready_artifact.get("summary") if isinstance(session_ready_artifact.get("summary"), dict) else {}
                session_ready_artifact["stale"] = False
                session_ready_artifact["summary"] = {
                    **session_ready_summary,
                    "freshness_inferred_from_shadow_loop": True,
                    "latest_shadow_activity_age_minutes": round(shadow_age_minutes, 4),
                }
                artifacts["session_ready"] = session_ready_artifact

    severity = _severity_from_attention(attention)

    status_map = {
        0: "ok",
        1: "warn",
        2: "degraded",
        3: "critical",
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "overall": {
            "status": status_map.get(severity, "unknown"),
            "ok": severity == 0,
            "attention": attention,
        },
        "artifacts": artifacts,
        "registry": _registry_summary(project_root),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a normalized runtime monitoring dashboard snapshot.")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_dashboard(PROJECT_ROOT)
    out_path = Path(args.out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"runtime_gate_dashboard status={payload['overall']['status']} "
            f"attention={','.join(payload['overall']['attention']) if payload['overall']['attention'] else 'none'}"
        )
    return 0 if payload["overall"]["status"] in {"ok", "warn"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
