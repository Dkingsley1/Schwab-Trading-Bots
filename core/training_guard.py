import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _parse_utc_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        return None


def _load_json(path: str) -> Dict[str, Any]:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _snapshot_training_coverage_guard(
    *,
    project_root: str,
    reference_ts: Optional[datetime],
    max_age_hours: float,
    required: bool,
    coverage_path: Optional[str],
    required_ratio: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    coverage_file = coverage_path or os.path.join(
        project_root,
        "governance",
        "health",
        "snapshot_training_coverage_latest.json",
    )

    ratio_epsilon = max(min(_as_float(os.getenv("SNAPSHOT_TRAINING_RATIO_EPSILON", "1e-6"), 1e-6), 0.01), 0.0)
    required_ratio = max(min(float(required_ratio), 1.0), 0.0)
    snapshot_max_age_hours = max(
        _as_float(os.getenv("SNAPSHOT_TRAINING_MAX_AGE_HOURS", str(max_age_hours)), max_age_hours),
        0.0,
    )

    details: Dict[str, Any] = {
        "required": bool(required),
        "coverage_file": coverage_file,
        "required_ratio": float(required_ratio),
        "ratio_epsilon": float(ratio_epsilon),
        "max_age_hours": float(snapshot_max_age_hours),
    }

    if not required:
        details["reason"] = "disabled"
        return True, "disabled", details

    payload = _load_json(coverage_file)
    if not payload:
        details["reason"] = "missing_snapshot_training_coverage_artifact"
        return False, "missing_snapshot_training_coverage_artifact", details

    ts = _parse_utc_timestamp(payload.get("timestamp_utc"))
    if ts is None:
        details["reason"] = "invalid_snapshot_training_coverage_timestamp"
        return False, "invalid_snapshot_training_coverage_timestamp", details

    now = datetime.now(timezone.utc)
    age_hours = max((now - ts).total_seconds() / 3600.0, 0.0)

    raw_ratio = _as_float(payload.get("snapshot_raw_sql_ingest_ratio"), 0.0)
    fill_ratio = _as_float(payload.get("snapshot_cov_fill_ratio"), 0.0)
    feature_ratio = _as_float(payload.get("snapshot_feature_coverage_ratio"), 0.0)
    all_snapshot_data_incorporated = bool(payload.get("all_snapshot_data_incorporated", False))

    details.update(
        {
            "timestamp_utc": ts.isoformat(),
            "age_hours": round(age_hours, 4),
            "snapshot_raw_sql_ingest_ratio": float(raw_ratio),
            "snapshot_cov_fill_ratio": float(fill_ratio),
            "snapshot_feature_coverage_ratio": float(feature_ratio),
            "all_snapshot_data_incorporated": bool(all_snapshot_data_incorporated),
            "reason": str(payload.get("reason") or ""),
        }
    )

    if age_hours > snapshot_max_age_hours:
        return (
            False,
            f"stale_snapshot_training_coverage age_hours={age_hours:.2f} max_age_hours={snapshot_max_age_hours:.2f}",
            details,
        )

    if reference_ts is not None and ts < reference_ts:
        details["reference_timestamp_utc"] = reference_ts.isoformat()
        return False, "snapshot_training_coverage_older_than_training_success", details

    if not all_snapshot_data_incorporated:
        return False, "snapshot_training_not_fully_incorporated", details

    if raw_ratio + ratio_epsilon < required_ratio:
        return False, "snapshot_raw_sql_ingest_ratio_below_required", details

    if fill_ratio + ratio_epsilon < required_ratio:
        return False, "snapshot_cov_fill_ratio_below_required", details

    if feature_ratio + ratio_epsilon < required_ratio:
        return False, "snapshot_feature_coverage_ratio_below_required", details

    return True, "ok", details


def check_registry_row_state_before_deletion(
    row: Dict[str, Any],
    *,
    min_streak: Optional[int] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    bot_id = str((row or {}).get("bot_id") or "").strip()
    if not bot_id:
        return False, "missing_bot_id", {}

    active = bool((row or {}).get("active", False))
    deleted = bool((row or {}).get("deleted_from_rotation", False))
    lifecycle_state = str((row or {}).get("lifecycle_state") or "").strip().lower()

    if deleted and active:
        return False, "invalid_state_deleted_and_active", {
            "bot_id": bot_id,
            "active": active,
            "deleted_from_rotation": deleted,
            "lifecycle_state": lifecycle_state,
        }

    if lifecycle_state and lifecycle_state not in {"active", "inactive", "deleted"}:
        return False, f"invalid_lifecycle_state:{lifecycle_state}", {
            "bot_id": bot_id,
            "active": active,
            "deleted_from_rotation": deleted,
            "lifecycle_state": lifecycle_state,
        }

    if min_streak is not None:
        streak = _as_int((row or {}).get("no_improvement_streak"), 0)
        if streak < int(min_streak):
            return False, f"streak_below_threshold:{streak}<{int(min_streak)}", {
                "bot_id": bot_id,
                "no_improvement_streak": streak,
            }

    return True, "ok", {
        "bot_id": bot_id,
        "active": active,
        "deleted_from_rotation": deleted,
        "lifecycle_state": lifecycle_state or ("deleted" if deleted else ("active" if active else "inactive")),
    }


def check_confirmed_training_success(
    *,
    project_root: str,
    marker_path: Optional[str] = None,
    scorecard_path: Optional[str] = None,
    max_age_hours: float = 72.0,
    require_master_update: bool = True,
    min_trained_bots: int = 1,
    require_snapshot_training_complete: Optional[bool] = None,
    snapshot_coverage_path: Optional[str] = None,
    min_snapshot_training_ratio: float = 1.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Returns (ok, reason, details) for deletion safety checks."""
    marker_file = marker_path or os.path.join(project_root, "governance", "health", "training_success_latest.json")
    score_file = scorecard_path or os.path.join(project_root, "governance", "health", "retrain_scorecard_latest.json")
    max_age_hours = max(float(max_age_hours), 0.0)
    min_trained_bots = max(int(min_trained_bots), 0)

    if require_snapshot_training_complete is None:
        require_snapshot_training_complete = _env_flag("REQUIRE_SNAPSHOT_TRAINING_COMPLETE_FOR_DELETION", True)

    required_snapshot_ratio = max(min(float(min_snapshot_training_ratio), 1.0), 0.0)
    required_snapshot_ratio = max(
        min(
            _as_float(
                os.getenv("SNAPSHOT_TRAINING_REQUIRED_RATIO", str(required_snapshot_ratio)),
                required_snapshot_ratio,
            ),
            1.0,
        ),
        0.0,
    )

    now = datetime.now(timezone.utc)

    marker = _load_json(marker_file)
    if marker:
        ts = _parse_utc_timestamp(marker.get("timestamp_utc"))
        age_hours = (now - ts).total_seconds() / 3600.0 if ts is not None else None
        confirmed = bool(marker.get("confirmed_training_success", False))
        trained_count = _as_int(marker.get("trained_count"), 0)
        failure_count = _as_int(marker.get("failure_count"), 0)
        master_update_status = str(marker.get("master_update_status") or "")

        reason = str(marker.get("reason") or ("ok" if confirmed else "not_confirmed"))
        if ts is None:
            return False, "invalid_marker_timestamp", {"source": "training_success_marker", "marker_file": marker_file}
        if age_hours is not None and age_hours > max_age_hours:
            return (
                False,
                f"stale_training_success_marker age_hours={age_hours:.2f} max_age_hours={max_age_hours:.2f}",
                {
                    "source": "training_success_marker",
                    "marker_file": marker_file,
                    "timestamp_utc": ts.isoformat(),
                    "age_hours": round(age_hours, 4),
                    "max_age_hours": max_age_hours,
                    "confirmed_training_success": confirmed,
                },
            )
        if not confirmed:
            return (
                False,
                f"training_success_not_confirmed:{reason}",
                {
                    "source": "training_success_marker",
                    "marker_file": marker_file,
                    "timestamp_utc": ts.isoformat(),
                    "age_hours": round(age_hours or 0.0, 4),
                    "trained_count": trained_count,
                    "failure_count": failure_count,
                    "master_update_status": master_update_status,
                    "reason": reason,
                },
            )

        base_details = {
            "source": "training_success_marker",
            "marker_file": marker_file,
            "timestamp_utc": ts.isoformat(),
            "age_hours": round(age_hours or 0.0, 4),
            "trained_count": trained_count,
            "failure_count": failure_count,
            "master_update_status": master_update_status,
            "reason": reason,
        }

        snap_ok, snap_reason, snap_details = _snapshot_training_coverage_guard(
            project_root=project_root,
            reference_ts=ts,
            max_age_hours=max_age_hours,
            required=bool(require_snapshot_training_complete),
            coverage_path=snapshot_coverage_path,
            required_ratio=required_snapshot_ratio,
        )
        base_details["snapshot_training_guard"] = snap_details
        if not snap_ok:
            return False, f"snapshot_training_guard_failed:{snap_reason}", base_details

        return True, "ok", base_details

    score = _load_json(score_file)
    if not score:
        return False, "missing_training_success_artifact", {"source": "none", "marker_file": marker_file, "scorecard_file": score_file}

    ts = _parse_utc_timestamp(score.get("timestamp_utc"))
    if ts is None:
        return False, "invalid_scorecard_timestamp", {"source": "retrain_scorecard", "scorecard_file": score_file}

    age_hours = (now - ts).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return (
            False,
            f"stale_retrain_scorecard age_hours={age_hours:.2f} max_age_hours={max_age_hours:.2f}",
            {
                "source": "retrain_scorecard",
                "scorecard_file": score_file,
                "timestamp_utc": ts.isoformat(),
                "age_hours": round(age_hours, 4),
                "max_age_hours": max_age_hours,
            },
        )

    status_counts = score.get("status_counts") if isinstance(score.get("status_counts"), dict) else {}
    trained_count = _as_int(status_counts.get("trained"), 0)
    failure_count = _as_int(score.get("failure_count"), 0)
    master_update_status = str(score.get("master_update_status") or "")
    master_updated = master_update_status.startswith("updated")

    if trained_count < min_trained_bots:
        return (
            False,
            f"trained_count_below_min trained_count={trained_count} min_trained_bots={min_trained_bots}",
            {
                "source": "retrain_scorecard",
                "scorecard_file": score_file,
                "timestamp_utc": ts.isoformat(),
                "age_hours": round(age_hours, 4),
                "trained_count": trained_count,
                "failure_count": failure_count,
                "master_update_status": master_update_status,
            },
        )

    if failure_count > 0:
        return (
            False,
            f"training_failures_present failure_count={failure_count}",
            {
                "source": "retrain_scorecard",
                "scorecard_file": score_file,
                "timestamp_utc": ts.isoformat(),
                "age_hours": round(age_hours, 4),
                "trained_count": trained_count,
                "failure_count": failure_count,
                "master_update_status": master_update_status,
            },
        )

    if require_master_update and (not master_updated):
        return (
            False,
            f"master_registry_not_updated status={master_update_status or 'unknown'}",
            {
                "source": "retrain_scorecard",
                "scorecard_file": score_file,
                "timestamp_utc": ts.isoformat(),
                "age_hours": round(age_hours, 4),
                "trained_count": trained_count,
                "failure_count": failure_count,
                "master_update_status": master_update_status,
                "require_master_update": True,
            },
        )

    base_details = {
        "source": "retrain_scorecard",
        "scorecard_file": score_file,
        "timestamp_utc": ts.isoformat(),
        "age_hours": round(age_hours, 4),
        "trained_count": trained_count,
        "failure_count": failure_count,
        "master_update_status": master_update_status,
        "require_master_update": bool(require_master_update),
    }

    snap_ok, snap_reason, snap_details = _snapshot_training_coverage_guard(
        project_root=project_root,
        reference_ts=ts,
        max_age_hours=max_age_hours,
        required=bool(require_snapshot_training_complete),
        coverage_path=snapshot_coverage_path,
        required_ratio=required_snapshot_ratio,
    )
    base_details["snapshot_training_guard"] = snap_details
    if not snap_ok:
        return False, f"snapshot_training_guard_failed:{snap_reason}", base_details

    return True, "ok", base_details
