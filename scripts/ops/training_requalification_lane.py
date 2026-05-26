#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_quality_thresholds import TARGET_QUALITY_SCORE_FLOOR, TARGET_TEST_ACCURACY_FLOOR

DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_requalification_latest.json"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "governance" / "training_diagnostics" / "requalification_queue_latest.jsonl"
DEFAULT_REPAIR_OUT_PATH = PROJECT_ROOT / "governance" / "training_diagnostics" / "requalification_repairs_latest.json"
DEFAULT_WALK_FORWARD_PATH = PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_registry(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _load_registry_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_registry(path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


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


def _parse_csv(raw: Any) -> list[str]:
    return [str(part).strip().lower() for part in str(raw or "").split(",") if str(part).strip()]


def _label_contract_from_registry_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("label_contract") or row.get("training_label_contract") or row.get("universal_label_contract")
    if not isinstance(raw, dict):
        return {}
    label_family = str(raw.get("label_family") or raw.get("family") or "").strip()
    primary_horizon = str(raw.get("primary_horizon") or raw.get("primary_label_horizon") or "").strip()
    if not label_family or not primary_horizon:
        return {}
    return {
        "label_family": label_family,
        "primary_horizon": primary_horizon,
        "aux_horizons": list(raw.get("aux_horizons") or raw.get("aux_label_horizons") or []),
        "required_context": list(raw.get("required_context") or raw.get("required_label_context") or []),
        "contract_version": str(raw.get("contract_version") or raw.get("version") or row.get("data_label_contract_version") or ""),
        "source": "registry_requalification_repair",
    }


def _age_hours(path: Path) -> float | None:
    try:
        return max((datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0, 0.0)
    except Exception:
        return None


def _latest_artifact(base: Path, bot_id: str, suffixes: tuple[str, ...]) -> Path | None:
    matches: list[Path] = []
    for suffix in suffixes:
        matches.extend(sorted(base.glob(f"{bot_id}_*{suffix}")))
        matches.extend(sorted(base.glob(f"{bot_id}{suffix}")))
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return matches[-1]


def _latest_log_artifact_for_bot(root: Path, bot_id: str) -> Path | None:
    search_bases = [
        root / "logs",
        root / "governance" / "training_diagnostics",
        root / "governance" / "walk_forward",
    ]
    for base in search_bases:
        artifact = _latest_artifact(base, bot_id, (".json", ".jsonl", ".log", ".txt"))
        if artifact is not None:
            return artifact
    return None


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
        ("risk_budget", "infrastructure"),
        ("allocator", "infrastructure"),
        ("seasonal", "signal"),
    ):
        if token in lowered:
            return family
    return "general"


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


def _current_regime_priority_index(payload: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    current_regime = payload.get("current_regime") if isinstance(payload.get("current_regime"), dict) else {}
    live_regime = str(current_regime.get("live_regime") or "").strip().lower()
    rows = current_regime.get("regime_fit_replacements") if isinstance(current_regime.get("regime_fit_replacements"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = row
    return live_regime, out


def _is_bootstrap_runtime_candidate(row: dict[str, Any]) -> bool:
    reasons = {
        str(row.get("reason") or "").strip().lower(),
        str(row.get("promotion_reason") or "").strip().lower(),
    }
    return bool(reasons & {"new_runtime_candidate", "planned_roster_expansion_slot"})


def _data_collection_threshold(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    started_text = str(row.get("data_collection_started_utc") or "").strip()
    started_age_days: float | None = None
    if started_text:
        try:
            parsed = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            started_age_days = max((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0, 0.0)
        except Exception:
            started_age_days = None
    observations = max(
        _safe_int(row.get("data_collection_observations"), 0),
        _safe_int(row.get("collected_observation_count"), 0),
        _safe_int(row.get("observation_count"), 0),
    )
    paper_standard = row.get("paper_promotion_standard") if isinstance(row.get("paper_promotion_standard"), dict) else {}
    min_observations = max(
        _safe_int(row.get("minimum_training_observations"), 0),
        _safe_int(paper_standard.get("minimum_observations"), 0),
        0,
    )
    min_days = max(
        _safe_int(row.get("minimum_data_collection_days"), 0),
        _safe_int(paper_standard.get("minimum_collection_days"), 0),
        0,
    )
    observations_ready = bool(min_observations <= 0 or observations >= min_observations)
    days_ready = bool(min_days <= 0 or (started_age_days is not None and started_age_days >= float(min_days)))
    ready = bool(observations_ready and days_ready)
    return ready, {
        "observations": observations,
        "minimum_training_observations": min_observations,
        "observations_ready": observations_ready,
        "collection_age_days": round(float(started_age_days), 3) if started_age_days is not None else None,
        "minimum_data_collection_days": min_days,
        "days_ready": days_ready,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_walk_forward_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("bots") if isinstance(payload.get("bots"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for bot_id, row in rows.items():
        text = str(bot_id or "").strip().lower()
        if text and isinstance(row, dict):
            out[text] = row
    return out


def _recovered_diagnostic(
    *,
    bot_id: str,
    log_payload: dict[str, Any],
    log_path: Path,
    model_path: Path | None,
    label_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = log_payload.get("metrics") if isinstance(log_payload.get("metrics"), dict) else {}
    acted_count = max(
        _safe_int(metrics.get("acted_count"), 0),
        _safe_int(metrics.get("long_acted_count"), 0) + _safe_int(metrics.get("short_acted_count"), 0),
    )
    sample_count = max(acted_count, 1 if metrics else 0)
    acted_accuracy = _safe_float(metrics.get("acted_accuracy"), -1.0)
    accuracy_lift = _safe_float(metrics.get("accuracy_lift_over_majority"), 0.0)
    failures: list[str] = []
    if acted_accuracy >= 0.0 and acted_accuracy < 0.60:
        failures.append(f"acted_accuracy={acted_accuracy:.4f} < recovered_min_acted_accuracy=0.6000")
    if accuracy_lift < 0.02:
        failures.append(
            "accuracy_lift_over_majority="
            f"{accuracy_lift:.4f} < recovered_min_accuracy_lift_over_majority=0.0200"
        )
    status = "failed" if failures else "passed"
    diag_timestamp = str(log_payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
    positive_rate = _safe_float(metrics.get("positive_rate"), 0.5)
    runtime_meta = {
        "sample_count": int(sample_count),
        "eligible_sequences": int(1 if sample_count > 0 else 0),
        "sequence_count": int(1 if sample_count > 0 else 0),
        "observation_count": int(sample_count),
        "positive_rate": round(float(positive_rate), 6),
        "skipped_filtered": 0,
        "skipped_low_confidence": 0,
        "skipped_labels": 0,
        "recovered_from_training_log": True,
        "recovery_source_log_path": str(log_path),
        "recovery_source_model_path": str(model_path) if model_path else "",
    }
    if label_contract:
        runtime_meta["label_contract"] = dict(label_contract)
    payload = {
        "timestamp_utc": diag_timestamp,
        "run_tag": bot_id,
        "status": status,
        "family": _bot_family(bot_id),
        "sample_count": int(sample_count),
        "eligible_sequences": int(runtime_meta["eligible_sequences"]),
        "sequence_count": int(runtime_meta["sequence_count"]),
        "observation_count": int(runtime_meta["observation_count"]),
        "positive_rate": round(float(positive_rate), 6),
        "skipped_filtered": 0,
        "skipped_low_confidence": 0,
        "skipped_labels": 0,
        "metrics": metrics,
        "runtime_meta": runtime_meta,
        "quality_failures": failures,
        "failure_categories": (["quality_guard_failure"] if failures else []),
        "repaired_from_log": True,
    }
    return payload


def _diagnostic_from_source_payload(
    *,
    bot_id: str,
    source_payload: dict[str, Any],
    source_path: Path,
    model_path: Path | None,
    label_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(source_payload.get("metrics"), dict):
        return _recovered_diagnostic(
            bot_id=bot_id,
            log_payload=source_payload,
            log_path=source_path,
            model_path=model_path,
            label_contract=label_contract,
        )
    payload = dict(source_payload)
    payload.setdefault("timestamp_utc", str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()))
    payload.setdefault("run_tag", bot_id)
    runtime_meta = payload.get("runtime_meta") if isinstance(payload.get("runtime_meta"), dict) else {}
    payload["runtime_meta"] = {
        **runtime_meta,
        "recovered_from_existing_diagnostic": True,
        "recovery_source_log_path": str(source_path),
        "recovery_source_model_path": str(model_path) if model_path else "",
    }
    if label_contract:
        payload["runtime_meta"]["label_contract"] = dict(label_contract)
    payload["repair_source_path"] = str(source_path)
    return payload


def apply_repairs(
    project_root: Path = PROJECT_ROOT,
    *,
    include_bot_ids: list[str] | None = None,
    repair_out_path: Path = DEFAULT_REPAIR_OUT_PATH,
) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry_payload = _load_registry(registry_path)
    rows = registry_payload.get("sub_bots") if isinstance(registry_payload.get("sub_bots"), list) else []
    registry_changed = False
    include = {str(bot_id or "").strip().lower() for bot_id in (include_bot_ids or []) if str(bot_id or "").strip()}
    diagnostics_dir = project_root / "governance" / "training_diagnostics"
    snapshot = _load_json(project_root / "governance" / "health" / "runtime_training_snapshot_latest.json")
    snapshot_ready = bool(_safe_int(snapshot.get("row_count"), 0) > 0 and _safe_int(snapshot.get("sequence_count"), 0) > 0)
    repaired_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    registry_backup_path = ""

    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        if include and bot_id not in include:
            continue

        latest_model = _latest_artifact(project_root / "models", bot_id, (".npz",))
        latest_log = _latest_log_artifact_for_bot(project_root, bot_id)
        diag_path = diagnostics_dir / f"{bot_id}_latest.json"
        diag_exists_before = diag_path.exists()
        diag_rebuilt = False
        registry_row_changed = False
        diag_age_hours = _age_hours(diag_path) if diag_path.exists() else None
        latest_log_newer_than_diag = bool(
            latest_log and diag_path.exists() and latest_log.stat().st_mtime > diag_path.stat().st_mtime
        )
        diag_payload = _load_json(diag_path) if diag_path.exists() else {}
        diag_runtime_meta = diag_payload.get("runtime_meta") if isinstance(diag_payload.get("runtime_meta"), dict) else {}
        diag_missing_label_contract = bool(
            _label_contract_from_registry_row(row)
            and not isinstance(diag_runtime_meta.get("label_contract"), dict)
            and not isinstance(diag_runtime_meta.get("training_label_contract"), dict)
        )

        if latest_model and str(row.get("model_path") or "") != str(latest_model):
            row["model_path"] = str(latest_model)
            registry_row_changed = True
        if latest_log and str(row.get("log_file") or "") != str(latest_log):
            row["log_file"] = str(latest_log)
            registry_row_changed = True

        if latest_log and (
            (not diag_exists_before)
            or latest_log_newer_than_diag
            or (diag_age_hours is not None and diag_age_hours > 72.0)
            or diag_missing_label_contract
        ):
            log_payload = _load_json(latest_log)
            recovered = _diagnostic_from_source_payload(
                bot_id=bot_id,
                source_payload=log_payload,
                source_path=latest_log,
                model_path=latest_model,
                label_contract=_label_contract_from_registry_row(row),
            )
            _write_json(diag_path, recovered)
            diag_rebuilt = True

        if registry_row_changed:
            if not registry_backup_path and registry_path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_path = project_root / "governance" / "lifecycle" / f"master_bot_registry.requalification_backup_{stamp}.json"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
                registry_backup_path = str(backup_path)
            registry_changed = True

        repaired_rows.append(
            {
                "bot_id": bot_id,
                "model_path": str(latest_model) if latest_model else "",
                "log_path": str(latest_log) if latest_log else "",
                "diagnostic_path": str(diag_path) if diag_path.exists() else "",
                "registry_row_updated": bool(registry_row_changed),
                "diagnostic_rebuilt": bool(diag_rebuilt),
                "runtime_snapshot_ready": snapshot_ready,
            }
        )
        if not latest_log or not diag_path.exists():
            unresolved_rows.append(
                {
                    "bot_id": bot_id,
                    "missing_log": not bool(latest_log),
                    "missing_diagnostic": not diag_path.exists(),
                    "runtime_snapshot_ready": snapshot_ready,
                }
            )

    if registry_changed:
        _write_json(registry_path, registry_payload)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(unresolved_rows) == 0,
        "include_bot_ids": sorted(include),
        "registry_path": str(registry_path),
        "registry_updated": bool(registry_changed),
        "registry_backup_path": registry_backup_path,
        "runtime_snapshot_ready": snapshot_ready,
        "repaired_count": len(repaired_rows),
        "repaired_rows": repaired_rows,
        "unresolved_count": len(unresolved_rows),
        "unresolved_rows": unresolved_rows,
    }
    _write_json(repair_out_path, result)
    return result


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    diagnostics_dir = project_root / "governance" / "training_diagnostics"
    walk_forward_rows = _load_walk_forward_rows(project_root / "governance" / "walk_forward" / "walk_forward_latest.json")
    roster_resilience = _load_json(project_root / "governance" / "health" / "roster_resilience_planner_latest.json")
    rows = _load_registry_rows(registry_path)
    live_regime, regime_priority_index = _current_regime_priority_index(roster_resilience)
    candidates: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip().lower() if isinstance(row, dict) else ""
        threshold_ready, threshold_meta = _data_collection_threshold(row)
        if bool(row.get("training_excluded", False)) and not threshold_ready:
            excluded_rows.append(
                {
                    "bot_id": bot_id,
                    "reason": str(row.get("training_exclusion_reason") or "training_excluded"),
                    "until": str(row.get("training_exclusion_until") or ""),
                    **threshold_meta,
                }
            )
            continue
        if bool(row.get("active", False)) and not (bool(row.get("data_collection_active", False)) and threshold_ready):
            continue
        lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
        if lifecycle_state in {"retired", "deleted", "deactivated"} or bool(row.get("deleted_from_rotation", False)):
            continue
        if not bot_id:
            continue
        diag_path = diagnostics_dir / f"{bot_id}_latest.json"
        diag = _load_json(diag_path) if diag_path.exists() else {}
        model_path = _latest_artifact(project_root / "models", bot_id, (".npz",))
        log_path = _latest_log_artifact_for_bot(project_root, bot_id)
        quality_score = _safe_float(row.get("quality_score"), 0.0)
        test_accuracy = _safe_float(row.get("test_accuracy"), 0.0)
        diag_age_hours = _age_hours(diag_path) if diag_path.exists() else None
        inferred_status = str(diag.get("status") or "missing_diagnostic").strip().lower()
        bootstrap_candidate = _is_bootstrap_runtime_candidate(row)
        regime_priority = regime_priority_index.get(bot_id, {})
        regime_fit_score = _safe_int(regime_priority.get("regime_fit_score"), 0)
        current_regime_priority = regime_fit_score > 0
        walk_forward = walk_forward_rows.get(bot_id, {})
        walk_forward_runs = _safe_int(walk_forward.get("runs"), 0)
        walk_forward_status = str(walk_forward.get("status") or "").strip().lower()
        actions: list[str] = []
        if not model_path:
            actions.append("rebuild_model_artifact")
        if not log_path and not bootstrap_candidate:
            actions.append("recover_training_log")
        if not diag and not bootstrap_candidate:
            actions.append("refresh_training_diagnostics")
        elif diag_age_hours is not None and diag_age_hours > 72.0:
            actions.append("refresh_training_diagnostics")
        sample_count = _safe_int(diag.get("sample_count"), 0)
        if inferred_status == "deferred_sample_starved" or (bool(diag) and sample_count == 0) or (bootstrap_candidate and not diag):
            actions.append("repair_runtime_inputs")
        has_repair_pressure = "repair_runtime_inputs" in actions
        coverage_ready = quality_score >= TARGET_QUALITY_SCORE_FLOOR and bool(model_path) and (bool(log_path) or bootstrap_candidate)
        bootstrap_stage_candidate = bootstrap_candidate and current_regime_priority and str(row.get("bot_role") or "") != "infrastructure_sub_bot"
        coverage_stage_candidate = bootstrap_stage_candidate or (coverage_ready and (
            not has_repair_pressure or quality_score >= 0.80 or test_accuracy >= TARGET_TEST_ACCURACY_FLOOR
        ))
        if coverage_stage_candidate:
            actions.append("seed_walk_forward_coverage")
        elif quality_score >= TARGET_QUALITY_SCORE_FLOOR and "repair_runtime_inputs" not in actions:
            actions.append("targeted_retrain")
        priority = round(
            (quality_score * 100.0)
            + (15.0 if model_path else 0.0)
            + (10.0 if log_path else 0.0)
            + (8.0 if diag and (diag_age_hours or 0.0) <= 72.0 else 0.0)
            - (25.0 if "repair_runtime_inputs" in actions else 0.0),
            3,
        )
        priority += float(min(max(walk_forward_runs, 0), 12) * 3.0)
        if str(row.get("bot_role") or "") != "infrastructure_sub_bot":
            priority += 5.0
        if current_regime_priority:
            priority += float(35.0 + (regime_fit_score * 5.0))
        if bootstrap_candidate:
            priority += 12.0
        if lifecycle_state == "probation":
            priority -= 4.0
        candidates.append(
            {
                "bot_id": bot_id,
                "bot_role": str(row.get("bot_role") or ""),
                "lifecycle_state": lifecycle_state or "inactive_backlog",
                "quality_score": round(quality_score, 6),
                "test_accuracy": round(test_accuracy, 6),
                "diagnostic_present": bool(diag),
                "diagnostic_age_hours": round(float(diag_age_hours), 3) if diag_age_hours is not None else None,
                "sample_count": sample_count,
                "walk_forward_runs": int(walk_forward_runs),
                "walk_forward_status": walk_forward_status or ("insufficient_runs" if walk_forward_runs <= 0 else ""),
                "model_path": str(model_path) if model_path else "",
                "log_path": str(log_path) if log_path else "",
                "actions": actions,
                "priority": priority,
                "bootstrap_candidate": bootstrap_candidate,
                "current_regime_priority": current_regime_priority,
                "regime_fit_score": regime_fit_score,
                "live_regime": live_regime,
                "data_collection_threshold": threshold_meta if bool(row.get("data_collection_active", False)) else {},
            }
        )
    candidates.sort(key=lambda row: (-float(row.get("priority", 0.0) or 0.0), str(row.get("bot_id") or "")))
    ready_candidates = [row for row in candidates if "seed_walk_forward_coverage" in list(row.get("actions") or [])]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "registry_path": str(registry_path),
        "candidate_count": len(candidates),
        "training_excluded_count": len(excluded_rows),
        "training_excluded": excluded_rows[:80],
        "reactivation_ready_count": len(ready_candidates),
        "top_candidates": candidates[:20],
        "top_reactivation_ready": ready_candidates[:10],
        "current_regime": {
            "live_regime": live_regime,
            "priority_candidate_count": sum(1 for row in candidates if bool(row.get("current_regime_priority", False))),
        },
        "recommended_actions": [
            "refresh diagnostics before reactivating any stale candidate",
            "repair runtime inputs before retraining sample-starved bots",
            "seed walk-forward coverage only for candidates with intact model/log artifacts and acceptable quality score",
            "allow regime-fit bootstrap candidates to enter the coverage lane even before they have a historical log artifact",
        ],
    }
    return payload


def _write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a requalification lane for inactive bots that can graduate back into coverage.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--queue-out", default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--repair-out", default=str(DEFAULT_REPAIR_OUT_PATH))
    parser.add_argument("--include-bot-ids", default="")
    parser.add_argument("--apply-repair", action="store_true")
    parser.add_argument("--write-queue", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    repair_result: dict[str, Any] = {}
    include_bot_ids = _parse_csv(args.include_bot_ids)
    if args.apply_repair:
        repair_result = apply_repairs(
            project_root,
            include_bot_ids=include_bot_ids,
            repair_out_path=Path(args.repair_out).expanduser(),
        )

    payload = build_payload(project_root)
    if repair_result:
        payload["repair_result"] = {
            "ok": bool(repair_result.get("ok", False)),
            "repaired_count": int(repair_result.get("repaired_count", 0) or 0),
            "unresolved_count": int(repair_result.get("unresolved_count", 0) or 0),
            "repaired_bot_ids": [
                str(row.get("bot_id") or "")
                for row in repair_result.get("repaired_rows") or []
                if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
            ],
            "unresolved_bot_ids": [
                str(row.get("bot_id") or "")
                for row in repair_result.get("unresolved_rows") or []
                if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
            ],
            "runtime_snapshot_ready": bool(repair_result.get("runtime_snapshot_ready", False)),
            "repair_artifact": str(Path(args.repair_out).expanduser()),
        }
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.write_queue:
        _write_queue(Path(args.queue_out).expanduser(), list(payload.get("top_candidates") or []))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_requalification_lane "
            f"candidate_count={int(payload.get('candidate_count', 0) or 0)} "
            f"reactivation_ready_count={int(payload.get('reactivation_ready_count', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
