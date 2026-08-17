#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "strategy_generation_v1.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "strategy_generations" / "strategy_generation_state.json"
DEFAULT_EVENT_PATH = PROJECT_ROOT / "governance" / "strategy_generations" / "strategy_generation_events.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "strategy_generation_control_latest.json"
ACTIVE_STATES = {
    "proposed_collection_only",
    "training",
    "trained_collection_only",
    "paper_evaluation_pending",
    "paper_challenger_qualified",
}
PARENT_OFFSPRING_STATES = {"paper_challenger_qualified"}
TERMINAL_STATES = {
    "archived",
    "rejected_evidence",
    "retired",
    "training_failed_quarantined",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_ID_PATTERN = re.compile(r"^strategy_g[0-9]{4}_[0-9a-f]{12}$")


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(raw: Any) -> bool:
    return bool(HEX_SHA256.fullmatch(str(raw or "").strip().lower()))


def _resolve_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _state_digest(state: dict[str, Any]) -> str:
    return _canonical_hash({key: value for key, value in state.items() if key != "state_hash"})


def _seal_state(state: dict[str, Any]) -> dict[str, Any]:
    state["schema_version"] = 2
    state["state_hash"] = _state_digest(state)
    return state


def _verify_state(state: dict[str, Any], *, require_hash: bool) -> dict[str, Any]:
    errors: list[str] = []
    offspring = state.get("offspring")
    if _safe_int(state.get("schema_version"), 0) < 2:
        errors.append("state_schema_version_unsupported")
    if not isinstance(offspring, list):
        errors.append("state_offspring_not_list")
    expected = str(state.get("state_hash") or "")
    if require_hash and not expected:
        errors.append("state_hash_missing")
    if expected and not hmac.compare_digest(expected, _state_digest(state)):
        errors.append("state_hash_mismatch")
    return {"ok": not errors, "state_hash": expected, "errors": errors}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    write_payload(path, _seal_state(state))


def _integrity_context(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    policy = _as_dict(config.get("integrity"))
    key_path = _resolve_path(project_root, policy.get("event_signing_key_path"))
    secret = ""
    try:
        secret = key_path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        mode = key_path.stat().st_mode & 0o777
    except OSError:
        mode = 0
    minimum_bytes = max(_safe_int(policy.get("minimum_signing_key_bytes"), 32), 32)
    require_private = bool(policy.get("require_private_key_permissions", True))
    private_permissions = bool(mode and not (mode & 0o077))
    errors = ordered_unique(
        [
            "event_signing_key_outside_experiment_governance"
            if not _is_within(key_path, project_root / "governance" / "experiments")
            else "",
            "event_signing_key_missing" if not key_path.is_file() else "",
            "event_signing_key_too_short" if len(secret.encode("utf-8")) < minimum_bytes else "",
            "event_signing_key_permissions_not_private"
            if require_private and not private_permissions
            else "",
        ]
    )
    return {
        "ok": not errors,
        "key_path": str(key_path),
        "key_id": str(policy.get("event_signing_key_id") or "strategy-generation-local"),
        "secret": secret,
        "require_signed_events": bool(policy.get("require_signed_events", True)),
        "require_state_hash": bool(policy.get("require_state_hash", True)),
        "private_permissions": private_permissions,
        "key_mode": f"{mode:03o}" if mode else "",
        "errors": errors,
    }


def _policy_validation(project_root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = _as_dict(config.get("resource_limits"))
    evaluation = _as_dict(config.get("evaluation"))
    safety = _as_dict(config.get("safety_contract"))
    integrity = _integrity_context(project_root, config)
    errors: list[str] = []
    if _safe_int(config.get("schema_version"), 0) < 2:
        errors.append("generation_policy_schema_version_unsupported")
    if str(config.get("operating_mode") or "") != "collection_only_research":
        errors.append("generation_operating_mode_not_collection_only")
    maximum_per_generation = _safe_int(limits.get("max_offspring_per_generation"), 0)
    maximum_active = _safe_int(limits.get("max_active_offspring"), 0)
    maximum_retained = _safe_int(limits.get("max_retained_offspring"), 0)
    if not (1 <= maximum_per_generation <= maximum_active <= maximum_retained):
        errors.append("generation_population_limits_invalid")
    if _safe_int(limits.get("max_concurrent_training_jobs"), 0) != 1:
        errors.append("generation_training_must_be_single_flight")
    if _safe_int(limits.get("max_active_offspring_per_parent"), 0) != 1:
        errors.append("generation_parent_active_child_limit_must_be_one")
    if not (1 <= _safe_int(limits.get("max_lineage_depth"), 0) <= 5):
        errors.append("generation_lineage_depth_limit_invalid")
    if _safe_int(limits.get("max_candidate_artifact_bytes"), 0) <= 0:
        errors.append("candidate_artifact_limit_missing")
    if _safe_int(limits.get("max_total_candidate_artifact_bytes"), 0) < _safe_int(
        limits.get("max_candidate_artifact_bytes"), 0
    ):
        errors.append("total_candidate_artifact_limit_invalid")
    authority_must_be_false = (
        not bool(safety.get("execution_authority", True))
        and not bool(safety.get("paper_execution_authority", True))
        and not bool(safety.get("live_money_promotion_allowed", True))
    )
    if not authority_must_be_false:
        errors.append("generation_execution_authority_policy_unsafe")
    if not bool(safety.get("candidate_models_are_never_loaded_by_serving", False)):
        errors.append("candidate_serving_isolation_not_required")
    if not bool(safety.get("human_release_required_for_registry_admission", False)):
        errors.append("human_registry_release_not_required")
    for key in (
        "require_candidate_model_hash",
        "require_generation_manifest_hash",
        "require_dataset_hash",
        "require_holdout_hash",
        "require_replay_hash",
        "require_evaluator_identity",
        "require_signed_attestation",
        "require_unique_evaluation_run",
    ):
        if not bool(evaluation.get(key, False)):
            errors.append(f"evaluation_policy_{key}_disabled")
    if not str(evaluation.get("required_evaluator_role") or "").strip():
        errors.append("evaluation_required_evaluator_role_missing")
    evaluation_root = _resolve_path(project_root, evaluation.get("allowed_evaluation_root"))
    if not _is_within(evaluation_root, project_root / "governance" / "strategy_generations"):
        errors.append("evaluation_root_outside_strategy_governance")
    errors.extend(_as_list(integrity.get("errors")))
    errors = ordered_unique([str(item) for item in errors])
    public_integrity = {key: value for key, value in integrity.items() if key != "secret"}
    return (
        {
            "ok": not errors,
            "policy_sha256": _canonical_hash(config),
            "errors": errors,
            "event_integrity": public_integrity,
            "evaluation_root": str(evaluation_root),
            "authority_must_be_false": authority_must_be_false,
        },
        integrity,
    )


def _default_state(policy_id: str) -> dict[str, Any]:
    return _seal_state({
        "schema_version": 2,
        "policy_id": policy_id,
        "generation": 0,
        "last_generation_utc": "",
        "event_chain_head": "",
        "offspring": [],
    })


def _load_state(path: Path, policy_id: str) -> dict[str, Any]:
    state = load_json(path)
    if not state:
        return _default_state(policy_id)
    return state


def _event_signature(*, event_hash: str, previous_event_hash: str, key_id: str, secret: str) -> str:
    payload = _canonical_hash(
        {
            "event_hash": event_hash,
            "previous_event_hash": previous_event_hash,
            "signature_key_id": key_id,
        }
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest() if secret else ""


def verify_event_chain(
    path: Path,
    *,
    signing_secret: str = "",
    signing_key_id: str = "",
    require_signatures: bool = False,
) -> dict[str, Any]:
    previous = ""
    errors: list[str] = []
    count = 0
    if path.exists():
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            count += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                errors.append(f"line_{line_number}_invalid_json")
                continue
            if not isinstance(row, dict):
                errors.append(f"line_{line_number}_not_object")
                continue
            event_hash = str(row.get("event_hash") or "")
            signature = str(row.get("event_signature") or "")
            row_key_id = str(row.get("signature_key_id") or "")
            unsigned = {
                key: value
                for key, value in row.items()
                if key not in {"event_hash", "event_signature", "signature_key_id"}
            }
            if str(unsigned.get("previous_event_hash") or "") != previous:
                errors.append(f"line_{line_number}_previous_hash_mismatch")
            if _canonical_hash(unsigned) != event_hash:
                errors.append(f"line_{line_number}_event_hash_mismatch")
            if require_signatures:
                if not signature:
                    errors.append(f"line_{line_number}_event_signature_missing")
                if row_key_id != signing_key_id:
                    errors.append(f"line_{line_number}_signature_key_id_mismatch")
                expected_signature = _event_signature(
                    event_hash=event_hash,
                    previous_event_hash=str(unsigned.get("previous_event_hash") or ""),
                    key_id=row_key_id,
                    secret=signing_secret,
                )
                if not expected_signature or not hmac.compare_digest(signature, expected_signature):
                    errors.append(f"line_{line_number}_event_signature_mismatch")
            previous = event_hash
    return {
        "ok": not errors,
        "event_count": count,
        "chain_head": previous,
        "signed_event_count": count if require_signatures and not errors else 0,
        "signatures_required": bool(require_signatures),
        "signature_key_id": signing_key_id if require_signatures else "",
        "errors": errors,
    }


def _last_event(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return {}
    for raw in reversed(rows):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            return row
    return {}


def _recover_state_from_signed_tail(state: dict[str, Any], event_path: Path) -> dict[str, Any]:
    event = _last_event(event_path)
    prior_head = str(state.get("event_chain_head") or "")
    if not event or str(event.get("previous_event_hash") or "") != prior_head:
        return {"recovered": False, "reason": "event_tail_is_not_one_step_ahead"}
    snapshots = [row for row in _as_list(event.get("offspring_snapshots")) if isinstance(row, dict)]
    if not snapshots:
        return {"recovered": False, "reason": "signed_tail_has_no_redo_snapshot"}
    for snapshot in snapshots:
        candidate_id = str(snapshot.get("offspring_id") or "")
        if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
            return {"recovered": False, "reason": "signed_tail_candidate_id_invalid"}
        if any(
            bool(snapshot.get(key, False))
            for key in ("execution_authority", "paper_execution_authority", "serving_eligible")
        ):
            return {"recovered": False, "reason": "signed_tail_authority_contract_violated"}
    current_rows = [row for row in _as_list(state.get("offspring")) if isinstance(row, dict)]
    positions = {str(row.get("offspring_id") or ""): index for index, row in enumerate(current_rows)}
    for snapshot in snapshots:
        candidate_id = str(snapshot.get("offspring_id") or "")
        if candidate_id in positions:
            current_rows[positions[candidate_id]] = snapshot
        else:
            positions[candidate_id] = len(current_rows)
            current_rows.append(snapshot)
    state["offspring"] = current_rows
    state["generation"] = max(
        _safe_int(state.get("generation"), 0),
        _safe_int(event.get("strategy_generation"), 0),
        max((_safe_int(row.get("strategy_generation"), 0) for row in snapshots), default=0),
    )
    if str(event.get("event_type") or "") == "strategy_generation_proposed":
        state["last_generation_utc"] = str(event.get("timestamp_utc") or state.get("last_generation_utc") or "")
    state["event_chain_head"] = str(event.get("event_hash") or "")
    _seal_state(state)
    return {
        "recovered": True,
        "event_type": str(event.get("event_type") or ""),
        "event_hash": str(event.get("event_hash") or ""),
        "offspring_ids": [str(row.get("offspring_id") or "") for row in snapshots],
    }


def _append_event(
    path: Path,
    state: dict[str, Any],
    event_type: str,
    details: dict[str, Any],
    *,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 2,
        "timestamp_utc": iso_now(),
        "event_type": event_type,
        **details,
        "previous_event_hash": str(state.get("event_chain_head") or ""),
    }
    event_hash = _canonical_hash(unsigned)
    key_id = str(integrity.get("key_id") or "")
    event = {
        **unsigned,
        "event_hash": event_hash,
        "signature_key_id": key_id,
        "event_signature": _event_signature(
            event_hash=event_hash,
            previous_event_hash=str(unsigned.get("previous_event_hash") or ""),
            key_id=key_id,
            secret=str(integrity.get("secret") or ""),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    state["event_chain_head"] = event["event_hash"]
    return event


def _registry_map(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return {
        str(row.get("bot_id") or "").strip().lower(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    }


def _parent_rejections(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules = _as_dict(config.get("parent_eligibility"))
    limits = _as_dict(config.get("resource_limits"))
    allowed_roles = {str(item) for item in _as_list(rules.get("allowed_bot_roles"))}
    allowed_grades = {str(item).lower() for item in _as_list(rules.get("allowed_teacher_grades"))}
    registry = _registry_map(project_root)
    registry_path = project_root / "master_bot_registry.json"
    quality_path = project_root / "governance" / "distillation" / "teacher_quality_latest.json"
    quality = load_json(quality_path)
    current = now or datetime.now(timezone.utc)
    quality_timestamp = _parse_timestamp(quality.get("timestamp_utc"))
    if quality_timestamp is None and quality_path.is_file():
        quality_timestamp = datetime.fromtimestamp(quality_path.stat().st_mtime, tz=timezone.utc)
    quality_age_hours = (
        max((current - quality_timestamp).total_seconds() / 3600.0, 0.0)
        if quality_timestamp is not None
        else None
    )
    maximum_quality_age = max(_safe_float(rules.get("maximum_teacher_evidence_age_hours"), 6.0), 0.1)
    quality_fresh = bool(quality_age_hours is not None and quality_age_hours <= maximum_quality_age)
    quality_sha256 = _file_hash(quality_path) if quality_path.is_file() else ""
    registry_sha256 = _file_hash(registry_path) if registry_path.is_file() else ""
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for teacher in _as_list(quality.get("qualified_teachers")):
        if not isinstance(teacher, dict):
            continue
        bot_id = str(teacher.get("bot_id") or "").strip().lower()
        role = str(teacher.get("bot_role") or "unknown").strip()
        reg = registry.get(bot_id, {})
        reasons: list[str] = []
        if not quality_fresh:
            reasons.append("teacher_quality_evidence_stale_or_undated")
        if allowed_roles and role not in allowed_roles:
            reasons.append("role_not_strategy_eligible")
        if str(teacher.get("teacher_grade") or "").strip().lower() not in allowed_grades:
            reasons.append("teacher_grade_below_generation_floor")
        if _safe_float(teacher.get("teacher_score"), 0.0) < _safe_float(rules.get("minimum_teacher_score"), 0.0):
            reasons.append("teacher_score_below_generation_floor")
        if _safe_int(teacher.get("walk_forward_runs"), 0) < _safe_int(rules.get("minimum_walk_forward_runs"), 0):
            reasons.append("walk_forward_runs_below_generation_floor")
        if _safe_float(teacher.get("walk_forward_forward_mean"), 0.0) < _safe_float(
            rules.get("minimum_walk_forward_forward_mean"), 0.0
        ):
            reasons.append("walk_forward_mean_below_generation_floor")
        if bool(rules.get("require_positive_paper_bonus", True)) and _safe_float(teacher.get("paper_bonus"), 0.0) <= 0.0:
            reasons.append("positive_paper_evidence_missing")
        overfit_policy = _as_dict(teacher.get("overfit_policy"))
        if bool(rules.get("require_overfit_may_teach", True)) and not bool(overfit_policy.get("may_teach", False)):
            reasons.append("overfit_policy_forbids_reproduction")
        if bool(rules.get("require_active_parent", True)) and not bool(reg.get("active", teacher.get("active", False))):
            reasons.append("parent_not_active")
        if bool(rules.get("require_training_allowed", True)) and bool(
            reg.get("training_excluded", reg.get("exclude_from_training", False))
        ):
            reasons.append("parent_training_excluded")
        module_path = project_root / "core" / f"{bot_id}.py"
        if not module_path.is_file():
            reasons.append("parent_training_module_missing")
        module_sha256 = _file_hash(module_path) if module_path.is_file() else ""
        if bool(rules.get("require_source_module_hash", True)) and not module_sha256:
            reasons.append("parent_training_module_hash_missing")
        row = {
            "parent_id": bot_id,
            "source_module_bot_id": bot_id,
            "bot_role": role,
            "teacher_grade": str(teacher.get("teacher_grade") or ""),
            "teacher_score": round(_safe_float(teacher.get("teacher_score"), 0.0), 6),
            "walk_forward_runs": _safe_int(teacher.get("walk_forward_runs"), 0),
            "walk_forward_forward_mean": round(_safe_float(teacher.get("walk_forward_forward_mean"), 0.0), 6),
            "paper_bonus": round(_safe_float(teacher.get("paper_bonus"), 0.0), 6),
            "lineage_depth": 0,
            "training_module": str(module_path),
            "source_module_sha256": module_sha256,
            "teacher_evidence_sha256": quality_sha256,
            "teacher_evidence_age_hours": round(quality_age_hours, 6) if quality_age_hours is not None else None,
            "registry_sha256": registry_sha256,
            "rejection_reasons": ordered_unique(reasons),
        }
        (rejected if reasons else eligible).append(row)

    for offspring in _as_list(state.get("offspring")):
        if not isinstance(offspring, dict) or str(offspring.get("lifecycle_state") or "") not in PARENT_OFFSPRING_STATES:
            continue
        evaluation = _as_dict(offspring.get("evaluation"))
        reasons: list[str] = []
        if not bool(evaluation.get("qualified", False)):
            reasons.append("offspring_evaluation_not_qualified")
        if bool(rules.get("require_human_lineage_parent_approval_for_offspring", True)) and not bool(
            offspring.get("lineage_parent_approved", False)
        ):
            reasons.append("offspring_lineage_parent_human_approval_missing")
        if _safe_int(offspring.get("lineage_depth"), 1) >= _safe_int(limits.get("max_lineage_depth"), 3):
            reasons.append("maximum_lineage_depth_reached")
        module_bot_id = str(offspring.get("source_module_bot_id") or "").strip().lower()
        module_path = project_root / "core" / f"{module_bot_id}.py"
        if not module_path.is_file():
            reasons.append("offspring_source_module_missing")
        row = {
            "parent_id": str(offspring.get("offspring_id") or ""),
            "source_module_bot_id": module_bot_id,
            "bot_role": str(offspring.get("bot_role") or "unknown"),
            "teacher_grade": "paper_challenger",
            "teacher_score": round(_safe_float(evaluation.get("composite_score"), 0.0), 6),
            "walk_forward_runs": _safe_int(evaluation.get("walk_forward_runs"), 0),
            "walk_forward_forward_mean": round(_safe_float(evaluation.get("forward_mean"), 0.0), 6),
            "paper_bonus": round(max(_safe_float(evaluation.get("out_of_sample_net_pnl"), 0.0), 0.0), 6),
            "lineage_depth": _safe_int(offspring.get("lineage_depth"), 1),
            "training_module": str(module_path),
            "source_module_sha256": _file_hash(module_path) if module_path.is_file() else "",
            "teacher_evidence_sha256": str(evaluation.get("evaluation_sha256") or ""),
            "teacher_evidence_age_hours": None,
            "registry_sha256": registry_sha256,
            "rejection_reasons": ordered_unique(reasons),
        }
        (rejected if reasons else eligible).append(row)

    eligible.sort(key=lambda row: (row["teacher_score"], row["paper_bonus"], row["parent_id"]), reverse=True)
    rejected.sort(key=lambda row: (len(row["rejection_reasons"]), row["parent_id"]))
    return eligible, rejected


def _bounded_values(bounds: dict[str, Any]) -> list[float]:
    minimum = _safe_float(bounds.get("minimum"), 0.0)
    maximum = _safe_float(bounds.get("maximum"), minimum)
    step = abs(_safe_float(bounds.get("step"), 1.0)) or 1.0
    count = max(int(round((maximum - minimum) / step)), 0)
    return [round(minimum + (idx * step), 8) for idx in range(count + 1)]


def _genome(config: dict[str, Any], *, generation: int, parent_id: str) -> dict[str, Any]:
    bounds = _as_dict(config.get("genome_bounds"))
    seed = hashlib.sha256(f"{config.get('policy_id')}|{generation}|{parent_id}".encode("utf-8")).digest()
    genome: dict[str, Any] = {}
    for index, name in enumerate(
        ("teacher_weight", "lookback_days_delta", "minimum_confidence_delta", "sample_stride_delta")
    ):
        values = _bounded_values(_as_dict(bounds.get(name)))
        value = values[seed[index] % len(values)] if values else 0.0
        genome[name] = int(value) if name.endswith("days_delta") or name.endswith("stride_delta") else float(value)
    genome["warm_start_from_parent"] = True
    genome["mutation_count"] = sum(
        1
        for key in ("lookback_days_delta", "minimum_confidence_delta", "sample_stride_delta")
        if _safe_float(genome.get(key), 0.0) != 0.0
    )
    return genome


def _active_offspring(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _as_list(state.get("offspring"))
        if isinstance(row, dict) and str(row.get("lifecycle_state") or "") in ACTIVE_STATES
    ]


def _retained_offspring(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _as_list(state.get("offspring"))
        if isinstance(row, dict) and str(row.get("lifecycle_state") or "") not in {"archived", "retired"}
    ]


def _candidate_artifact_usage(state: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    by_candidate: dict[str, int] = {}
    for row in _as_list(state.get("offspring")):
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("offspring_id") or "")
        candidate_bytes = 0
        for key in ("model_path", "log_path", "diagnostics_path"):
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            path = Path(raw).expanduser().resolve()
            paths[str(path)] = path
            try:
                candidate_bytes += path.stat().st_size if path.is_file() else 0
            except OSError:
                pass
        by_candidate[candidate_id] = candidate_bytes
    total_bytes = 0
    for path in paths.values():
        try:
            total_bytes += path.stat().st_size if path.is_file() else 0
        except OSError:
            pass
    return {
        "total_bytes": total_bytes,
        "referenced_file_count": len(paths),
        "by_candidate_bytes": by_candidate,
    }


def _stale_training_candidates(
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    limits = _as_dict(config.get("resource_limits"))
    stale_after = max(
        _safe_int(limits.get("training_timeout_seconds"), 7200)
        + _safe_int(limits.get("training_stale_grace_seconds"), 900),
        60,
    )
    current = now or datetime.now(timezone.utc)
    stale: list[dict[str, Any]] = []
    for row in _as_list(state.get("offspring")):
        if not isinstance(row, dict) or str(row.get("lifecycle_state") or "") != "training":
            continue
        started = _parse_timestamp(row.get("training_started_at_utc"))
        age_seconds = (current - started).total_seconds() if started is not None else float("inf")
        if age_seconds > stale_after:
            stale.append(
                {
                    "offspring_id": str(row.get("offspring_id") or ""),
                    "training_age_seconds": round(age_seconds, 3) if age_seconds != float("inf") else None,
                    "stale_after_seconds": stale_after,
                }
            )
    return stale


def _proposal_blockers(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    eligible_parents: list[dict[str, Any]],
    *,
    now: datetime,
) -> list[str]:
    limits = _as_dict(config.get("resource_limits"))
    blockers: list[str] = []
    if not bool(config.get("enabled", False)):
        blockers.append("strategy_generation_disabled")
    if not eligible_parents:
        blockers.append("no_parent_has_reproduction_grade_evidence")
    if len(_active_offspring(state)) >= _safe_int(limits.get("max_active_offspring"), 4):
        blockers.append("active_offspring_cap_reached")
    if len(_retained_offspring(state)) >= _safe_int(limits.get("max_retained_offspring"), 24):
        blockers.append("retained_offspring_cap_reached")
    artifact_usage = _candidate_artifact_usage(state)
    if _safe_int(artifact_usage.get("total_bytes"), 0) >= _safe_int(
        limits.get("max_total_candidate_artifact_bytes"), 4 * 1024**3
    ):
        blockers.append("candidate_artifact_storage_cap_reached")
    if _stale_training_candidates(state, config, now=now):
        blockers.append("stale_offspring_training_requires_reconciliation")
    last_generation = _parse_timestamp(state.get("last_generation_utc"))
    cooldown_hours = max(_safe_int(limits.get("generation_cooldown_hours"), 168), 0)
    if last_generation is not None and now < last_generation + timedelta(hours=cooldown_hours):
        blockers.append("generation_cooldown_active")
    free_disk_gb = shutil.disk_usage(project_root).free / (1024.0**3)
    if free_disk_gb < _safe_float(limits.get("minimum_free_disk_gb"), 25.0):
        blockers.append("generation_disk_reserve_not_met")
    return ordered_unique(blockers)


def _new_offspring(
    config: dict[str, Any],
    *,
    generation: int,
    parent: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    genome = _genome(config, generation=generation, parent_id=str(parent["parent_id"]))
    identity = {
        "policy_id": config.get("policy_id"),
        "strategy_generation": generation,
        "parent_ids": [parent["parent_id"]],
        "source_module_bot_id": parent["source_module_bot_id"],
        "genome": genome,
    }
    offspring_id = f"strategy_g{generation:04d}_{_canonical_hash(identity)[:12]}"
    safety = _as_dict(config.get("safety_contract"))
    return {
        "offspring_id": offspring_id,
        "strategy_generation": generation,
        "lineage_depth": _safe_int(parent.get("lineage_depth"), 0) + 1,
        "parent_bot_ids": [parent["parent_id"]],
        "source_module_bot_id": parent["source_module_bot_id"],
        "training_module": parent["training_module"],
        "bot_role": parent["bot_role"],
        "created_at_utc": now.isoformat(),
        "lifecycle_state": str(safety.get("initial_lifecycle_state") or "proposed_collection_only"),
        "genome": genome,
        "execution_authority": False,
        "paper_execution_authority": False,
        "serving_eligible": False,
        "registry_admission_eligible": False,
        "lineage_parent_approved": False,
        "inherits_parent_grade": False,
        "paper_allocation_limit": 0.0,
        "live_order_budget": 0.0,
        "policy_sha256": _canonical_hash(config),
        "parent_source_module_sha256": str(parent.get("source_module_sha256") or ""),
        "parent_teacher_evidence_sha256": str(parent.get("teacher_evidence_sha256") or ""),
        "parent_registry_sha256": str(parent.get("registry_sha256") or ""),
        "model_path": "",
        "log_path": "",
        "diagnostics_path": "",
        "evaluation": {},
        "admission_state": "human_review_required",
    }


def propose_generation(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    event_path: Path,
    *,
    now: datetime,
    integrity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    eligible, _ = _parent_rejections(project_root, config, state, now=now)
    blockers = _proposal_blockers(project_root, config, state, eligible, now=now)
    if blockers:
        return [], blockers
    limits = _as_dict(config.get("resource_limits"))
    maximum = min(
        _safe_int(limits.get("max_offspring_per_generation"), 2),
        max(_safe_int(limits.get("max_active_offspring"), 4) - len(_active_offspring(state)), 0),
    )
    role_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}
    active_parent_counts: dict[str, int] = {}
    for row in _active_offspring(state):
        for parent_id in _as_list(row.get("parent_bot_ids")):
            normalized = str(parent_id or "")
            active_parent_counts[normalized] = active_parent_counts.get(normalized, 0) + 1
    selected: list[dict[str, Any]] = []
    generation = _safe_int(state.get("generation"), 0) + 1
    for parent in eligible:
        parent_id = str(parent["parent_id"])
        role = str(parent.get("bot_role") or "unknown")
        if active_parent_counts.get(parent_id, 0) >= _safe_int(
            limits.get("max_active_offspring_per_parent"), 1
        ):
            continue
        if parent_counts.get(parent_id, 0) >= _safe_int(limits.get("max_offspring_per_parent_per_generation"), 1):
            continue
        if role_counts.get(role, 0) >= _safe_int(limits.get("max_offspring_per_role_per_generation"), 1):
            continue
        child = _new_offspring(config, generation=generation, parent=parent, now=now)
        if any(str(row.get("offspring_id")) == child["offspring_id"] for row in _as_list(state.get("offspring"))):
            continue
        selected.append(child)
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        if len(selected) >= maximum:
            break
    if not selected:
        return [], ["resource_diversity_caps_produced_no_offspring"]
    state["generation"] = generation
    state["last_generation_utc"] = now.isoformat()
    state.setdefault("offspring", []).extend(selected)
    generation_path = (
        project_root
        / "governance"
        / "strategy_generations"
        / "generations"
        / f"strategy_generation_{generation:04d}.json"
    )
    write_payload(
        generation_path,
        {
            "schema_version": 2,
            "policy_id": config.get("policy_id"),
            "policy_sha256": _canonical_hash(config),
            "timestamp_utc": now.isoformat(),
            "strategy_generation": generation,
            "resource_limits": limits,
            "offspring": selected,
        },
    )
    generation_manifest_sha256 = _file_hash(generation_path)
    for child in selected:
        child["generation_manifest_path"] = str(generation_path)
        child["generation_manifest_sha256"] = generation_manifest_sha256
    _append_event(
        event_path,
        state,
        "strategy_generation_proposed",
        {
            "strategy_generation": generation,
            "offspring_ids": [row["offspring_id"] for row in selected],
            "parent_ids": [row["parent_bot_ids"][0] for row in selected],
            "offspring_snapshots": selected,
            "generation_manifest_sha256": generation_manifest_sha256,
            "policy_sha256": _canonical_hash(config),
            "execution_authority": False,
        },
        integrity=integrity,
    )
    return selected, []


def _training_gate(project_root: Path, config: dict[str, Any], state: dict[str, Any]) -> list[str]:
    limits = _as_dict(config.get("resource_limits"))
    runtime = load_json(project_root / "governance" / "health" / "training_runtime_control_latest.json")
    contract = _as_dict(runtime.get("training_launch_contract"))
    throttle = load_json(project_root / "governance" / "health" / "runtime_throttle_control_latest.json")
    release = _as_dict(throttle.get("release_contract"))
    snapshot = _as_dict(throttle.get("runtime_snapshot"))
    thermal = _as_dict(snapshot.get("thermal"))
    blockers: list[str] = []
    launch_blockers = {
        str(item)
        for item in _as_list(contract.get("launch_blockers"))
        if str(item).strip()
    }
    generation_relevant_launch_blockers = launch_blockers - {"no_bot_needs_training_candidates"}
    training_count = sum(
        1 for row in _as_list(state.get("offspring")) if isinstance(row, dict) and row.get("lifecycle_state") == "training"
    )
    if training_count >= _safe_int(limits.get("max_concurrent_training_jobs"), 1):
        blockers.append("offspring_training_concurrency_cap_reached")
    generation_queue_is_only_missing_candidate = bool(
        not generation_relevant_launch_blockers
        and launch_blockers == {"no_bot_needs_training_candidates"}
        and contract.get("prep_allowed", False)
    )
    if not bool(contract.get("launch_allowed", False)) and not generation_queue_is_only_missing_candidate:
        blockers.append("training_runtime_launch_not_allowed")
    if str(throttle.get("overall_status") or "").lower() != "ready":
        blockers.append("runtime_throttle_not_ready")
    if str(throttle.get("compute_pressure_level") or "").lower() not in {
        str(item).lower() for item in _as_list(limits.get("allowed_compute_pressure_levels"))
    }:
        blockers.append("compute_pressure_not_generation_safe")
    if str(throttle.get("memory_pressure_level") or "").lower() not in {
        str(item).lower() for item in _as_list(limits.get("allowed_memory_pressure_levels"))
    }:
        blockers.append("memory_pressure_not_generation_safe")
    if _safe_float(throttle.get("host_saturation_score"), 100.0) > _safe_float(
        limits.get("maximum_host_saturation_score"), 55.0
    ):
        blockers.append("host_saturation_above_generation_cap")
    if any(bool(thermal.get(key, False)) for key in ("thermal_warning_active", "performance_warning_active", "cpu_power_warning_active")):
        blockers.append("thermal_or_performance_warning_active")
    if not bool(release.get("shared_host_training_resume_allowed", False)):
        blockers.append("shared_host_training_not_released")
    if shutil.disk_usage(project_root).free / (1024.0**3) < _safe_float(limits.get("minimum_free_disk_gb"), 25.0):
        blockers.append("generation_disk_reserve_not_met")
    return ordered_unique(blockers)


def _latest_parent_log(project_root: Path, parent_id: str) -> dict[str, Any]:
    matches = sorted(
        (project_root / "logs").glob(f"{parent_id}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return load_json(matches[0]) if matches else {}


def _candidate_by_id(state: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    for row in _as_list(state.get("offspring")):
        if isinstance(row, dict) and str(row.get("offspring_id") or "") == candidate_id:
            return row
    return None


def _candidate_pretraining_integrity(
    project_root: Path,
    config: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    candidate_id = str(candidate.get("offspring_id") or "")
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        blockers.append("offspring_id_contract_invalid")
    if str(candidate.get("policy_sha256") or "") != _canonical_hash(config):
        blockers.append("offspring_policy_hash_mismatch")
    if any(
        bool(candidate.get(key, False))
        for key in ("execution_authority", "paper_execution_authority", "serving_eligible")
    ):
        blockers.append("offspring_authority_contract_violated")
    source_module_id = str(candidate.get("source_module_bot_id") or "").strip().lower()
    source_module = (project_root / "core" / f"{source_module_id}.py").resolve()
    if not _is_within(source_module, project_root / "core") or not source_module.is_file():
        blockers.append("offspring_source_module_missing_or_outside_core")
    elif not hmac.compare_digest(
        str(candidate.get("parent_source_module_sha256") or ""),
        _file_hash(source_module),
    ):
        blockers.append("offspring_parent_source_module_changed")
    manifest_path = _resolve_path(project_root, candidate.get("generation_manifest_path"))
    if not _is_within(manifest_path, project_root / "governance" / "strategy_generations" / "generations"):
        blockers.append("offspring_generation_manifest_outside_governance")
    elif not manifest_path.is_file():
        blockers.append("offspring_generation_manifest_missing")
    elif not hmac.compare_digest(
        str(candidate.get("generation_manifest_sha256") or ""),
        _file_hash(manifest_path),
    ):
        blockers.append("offspring_generation_manifest_hash_mismatch")
    return ordered_unique(blockers)


def train_next_offspring(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    event_path: Path,
    integrity: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    candidate = next(
        (
            row
            for row in _as_list(state.get("offspring"))
            if isinstance(row, dict) and row.get("lifecycle_state") == "proposed_collection_only"
        ),
        None,
    )
    blockers = _training_gate(project_root, config, state)
    if candidate is None:
        blockers.append("no_proposed_offspring_waiting_for_training")
    elif not blockers:
        blockers.extend(_candidate_pretraining_integrity(project_root, config, candidate))
    if blockers:
        return candidate, ordered_unique(blockers)
    assert candidate is not None
    candidate["lifecycle_state"] = "training"
    candidate["training_started_at_utc"] = iso_now()
    _append_event(
        event_path,
        state,
        "offspring_training_started",
        {
            "offspring_id": candidate["offspring_id"],
            "strategy_generation": candidate["strategy_generation"],
            "offspring_snapshots": [candidate],
        },
        integrity=integrity,
    )
    _write_state(state_path, state)

    parent_id = str(candidate["parent_bot_ids"][0])
    source_module_id = str(candidate["source_module_bot_id"])
    parent_log = _latest_parent_log(project_root, parent_id)
    if not parent_log and parent_id.startswith("strategy_g"):
        parent_candidate = _candidate_by_id(state, parent_id)
        if parent_candidate:
            parent_log = load_json(Path(str(parent_candidate.get("log_path") or "")))
    runtime_config = _as_dict(_as_dict(parent_log.get("config")).get("runtime"))
    genome = _as_dict(candidate.get("genome"))
    lookback = max(_safe_int(runtime_config.get("lookback_days"), 60) + _safe_int(genome.get("lookback_days_delta"), 0), 1)
    confidence = min(
        max(
            _safe_float(runtime_config.get("min_confidence"), 0.0)
            + _safe_float(genome.get("minimum_confidence_delta"), 0.0),
            0.0,
        ),
        1.0,
    )
    stride = max(_safe_int(runtime_config.get("sample_stride"), 1) + _safe_int(genome.get("sample_stride_delta"), 0), 1)
    manifest_path = (
        project_root
        / "governance"
        / "strategy_generations"
        / "generations"
        / f"strategy_generation_{_safe_int(candidate.get('strategy_generation'), 0):04d}.json"
    )
    env = dict(os.environ)
    env.update(
        {
            "STRATEGY_GENERATION_CANDIDATE_ID": str(candidate["offspring_id"]),
            "STRATEGY_GENERATION_PARENT_ID": parent_id,
            "STRATEGY_GENERATION_SOURCE_MODULE_BOT_ID": source_module_id,
            "STRATEGY_GENERATION_MANIFEST": str(manifest_path),
            "STRATEGY_GENERATION_WARM_START": "1",
            "DISTILLATION_ENABLED": "1",
            "DISTILLATION_STUDENT": "1",
            "DISTILLATION_TEACHERS": parent_id,
            "DISTILLATION_TEACHER_WEIGHT": str(_safe_float(genome.get("teacher_weight"), 0.25)),
            "RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE": str(lookback),
            "RUNTIME_TRAIN_MIN_CONFIDENCE_OVERRIDE": str(confidence),
            "RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE": str(stride),
            "RUNTIME_TRAIN_DEFER_SAMPLE_STARVED": "1",
        }
    )
    command = [sys.executable, str(project_root / "core" / f"{source_module_id}.py")]
    timeout = max(_safe_int(_as_dict(config.get("resource_limits")).get("training_timeout_seconds"), 7200), 60)
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return_code = int(result.returncode)
        stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stderr_tail = f"training_timeout_after_{timeout}_seconds: {exc}"
    except OSError as exc:
        return_code = 125
        stderr_tail = f"training_process_launch_failed: {exc}"

    model_matches = sorted((project_root / "models").glob(f"{candidate['offspring_id']}_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_matches = sorted((project_root / "logs").glob(f"{candidate['offspring_id']}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    diagnostics_path = project_root / "governance" / "training_diagnostics" / f"{candidate['offspring_id']}_latest.json"
    artifact_paths = [model_matches[0] if model_matches else None, log_matches[0] if log_matches else None, diagnostics_path]
    artifact_bytes = sum(path.stat().st_size for path in artifact_paths if path is not None and path.is_file())
    maximum_artifact_bytes = _safe_int(
        _as_dict(config.get("resource_limits")).get("max_candidate_artifact_bytes"),
        512 * 1024**2,
    )
    artifact_namespace_ready = bool(
        model_matches
        and log_matches
        and model_matches[0].name.startswith(f"{candidate['offspring_id']}_")
        and log_matches[0].name.startswith(f"{candidate['offspring_id']}_")
        and diagnostics_path.name.startswith(f"{candidate['offspring_id']}_")
        and _is_within(model_matches[0], project_root / "models")
        and _is_within(log_matches[0], project_root / "logs")
        and _is_within(diagnostics_path, project_root / "governance" / "training_diagnostics")
    )
    success = bool(
        return_code == 0
        and artifact_namespace_ready
        and diagnostics_path.is_file()
        and artifact_bytes <= maximum_artifact_bytes
    )
    artifact_failures = ordered_unique(
        [
            "training_process_nonzero" if return_code != 0 else "",
            "candidate_artifact_namespace_invalid" if not artifact_namespace_ready else "",
            "candidate_diagnostics_missing" if not diagnostics_path.is_file() else "",
            "candidate_artifact_limit_exceeded" if artifact_bytes > maximum_artifact_bytes else "",
        ]
    )
    candidate["training_completed_at_utc"] = iso_now()
    candidate["training_return_code"] = return_code
    candidate["training_error_tail"] = (
        ""
        if success
        else " | ".join([*artifact_failures, stderr_tail[-4000:]]).strip(" |")
    )
    candidate["training_artifact_bytes"] = artifact_bytes
    candidate["training_artifact_limit_bytes"] = maximum_artifact_bytes
    candidate["training_artifact_namespace_ready"] = artifact_namespace_ready
    candidate["lifecycle_state"] = "trained_collection_only" if success else "training_failed_quarantined"
    if model_matches:
        candidate["model_path"] = str(model_matches[0])
        candidate["model_sha256"] = _file_hash(model_matches[0])
    if log_matches:
        candidate["log_path"] = str(log_matches[0])
        candidate["log_sha256"] = _file_hash(log_matches[0])
    if diagnostics_path.is_file():
        candidate["diagnostics_path"] = str(diagnostics_path)
        candidate["diagnostics_sha256"] = _file_hash(diagnostics_path)
    _append_event(
        event_path,
        state,
        "offspring_training_completed" if success else "offspring_training_failed",
        {
            "offspring_id": candidate["offspring_id"],
            "return_code": return_code,
            "model_sha256": candidate.get("model_sha256", ""),
            "offspring_snapshots": [candidate],
            "execution_authority": False,
        },
        integrity=integrity,
    )
    _write_state(state_path, state)
    return candidate, ([] if success else ["offspring_training_failed"])


def _evaluation_signature(evaluation: dict[str, Any], *, key_id: str, secret: str) -> str:
    body = {key: value for key, value in evaluation.items() if key != "attestation"}
    raw = json.dumps(
        {"evaluation": body, "signature_key_id": key_id},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest() if secret else ""


def _evaluation_run_is_unique(state: dict[str, Any], *, candidate_id: str, evaluation_run_id: str) -> bool:
    if not evaluation_run_id:
        return False
    for row in _as_list(state.get("offspring")):
        if not isinstance(row, dict) or str(row.get("offspring_id") or "") == candidate_id:
            continue
        prior = _as_dict(row.get("evaluation"))
        if str(prior.get("evaluation_run_id") or "") == evaluation_run_id:
            return False
    return True


def evaluate_offspring(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    event_path: Path,
    *,
    candidate_id: str,
    evaluation_path: Path,
    integrity: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    candidate = _candidate_by_id(state, candidate_id)
    if candidate is None:
        return None, ["offspring_not_found"]
    if str(candidate.get("lifecycle_state") or "") not in {"trained_collection_only", "paper_evaluation_pending"}:
        return candidate, ["offspring_not_ready_for_evaluation"]
    rules = _as_dict(config.get("evaluation"))
    resolved_evaluation_path = evaluation_path.expanduser().resolve()
    allowed_evaluation_root = _resolve_path(project_root, rules.get("allowed_evaluation_root"))
    if not _is_within(resolved_evaluation_path, allowed_evaluation_root):
        return candidate, ["evaluation_artifact_outside_locked_generation_root"]
    evaluation = load_json(resolved_evaluation_path)
    if not evaluation:
        return candidate, ["evaluation_artifact_missing_or_invalid"]
    if str(evaluation.get("candidate_id") or "") != candidate_id:
        return candidate, ["evaluation_candidate_id_mismatch"]
    evaluated_at = _parse_timestamp(evaluation.get("evaluated_at_utc"))
    training_completed = _parse_timestamp(candidate.get("training_completed_at_utc"))
    now = datetime.now(timezone.utc)
    evaluation_age_hours = (
        max((now - evaluated_at).total_seconds() / 3600.0, 0.0)
        if evaluated_at is not None
        else None
    )
    model_path = _resolve_path(project_root, candidate.get("model_path"))
    candidate_model_hash = str(candidate.get("model_sha256") or "")
    current_model_hash = _file_hash(model_path) if model_path.is_file() else ""
    attestation = _as_dict(evaluation.get("attestation"))
    attestation_key_id = str(attestation.get("key_id") or "")
    attestation_signature = str(attestation.get("signature") or "")
    expected_signature = _evaluation_signature(
        evaluation,
        key_id=attestation_key_id,
        secret=str(integrity.get("secret") or ""),
    )
    evaluation_run_id = str(evaluation.get("evaluation_run_id") or "").strip()
    checks = {
        "independent_evaluation": (not bool(rules.get("require_independent_evaluation", True))) or bool(evaluation.get("independent_evaluation", False)),
        "locked_holdout": (not bool(rules.get("require_locked_holdout", True))) or bool(evaluation.get("locked_holdout", False)),
        "exact_replay": (not bool(rules.get("require_exact_replay", True))) or bool(evaluation.get("exact_replay", False)),
        "evaluation_timestamp": bool(evaluated_at is not None and evaluated_at <= now + timedelta(minutes=5)),
        "evaluation_after_training": bool(
            evaluated_at is not None and training_completed is not None and evaluated_at >= training_completed
        ),
        "evaluation_fresh": bool(
            evaluation_age_hours is not None
            and evaluation_age_hours <= _safe_float(rules.get("maximum_evaluation_age_hours"), 24.0)
        ),
        "candidate_model_artifact": bool(
            model_path.is_file()
            and _is_within(model_path, project_root / "models")
            and _is_sha256(candidate_model_hash)
            and hmac.compare_digest(current_model_hash, candidate_model_hash)
        ),
        "candidate_model_hash_binding": (not bool(rules.get("require_candidate_model_hash", True)))
        or bool(
            _is_sha256(candidate_model_hash)
            and hmac.compare_digest(str(evaluation.get("model_sha256") or ""), candidate_model_hash)
        ),
        "generation_manifest_hash_binding": (not bool(rules.get("require_generation_manifest_hash", True)))
        or bool(
            _is_sha256(candidate.get("generation_manifest_sha256"))
            and hmac.compare_digest(
                str(evaluation.get("generation_manifest_sha256") or ""),
                str(candidate.get("generation_manifest_sha256") or ""),
            )
        ),
        "dataset_hash": (not bool(rules.get("require_dataset_hash", True)))
        or _is_sha256(evaluation.get("dataset_sha256")),
        "holdout_hash": (not bool(rules.get("require_holdout_hash", True)))
        or _is_sha256(evaluation.get("holdout_sha256")),
        "replay_hash": (not bool(rules.get("require_replay_hash", True)))
        or _is_sha256(evaluation.get("replay_sha256")),
        "evaluator_identity": (not bool(rules.get("require_evaluator_identity", True)))
        or bool(
            str(evaluation.get("evaluator_id") or "").strip()
            and str(evaluation.get("evaluator_version") or "").strip()
            and str(evaluation.get("evaluator_role") or "")
            == str(rules.get("required_evaluator_role") or "")
        ),
        "signed_attestation": (not bool(rules.get("require_signed_attestation", True)))
        or bool(
            attestation_key_id == str(integrity.get("key_id") or "")
            and attestation_signature
            and expected_signature
            and hmac.compare_digest(attestation_signature, expected_signature)
        ),
        "unique_evaluation_run": (not bool(rules.get("require_unique_evaluation_run", True)))
        or _evaluation_run_is_unique(
            state,
            candidate_id=candidate_id,
            evaluation_run_id=evaluation_run_id,
        ),
        "out_of_sample_trades": _safe_int(evaluation.get("out_of_sample_trades"), 0) >= _safe_int(rules.get("minimum_out_of_sample_trades"), 100),
        "out_of_sample_net_pnl": _safe_float(evaluation.get("out_of_sample_net_pnl"), 0.0) >= _safe_float(rules.get("minimum_out_of_sample_net_pnl"), 0.01),
        "net_expectancy": _safe_float(evaluation.get("net_expectancy"), -1.0) > _safe_float(rules.get("minimum_net_expectancy"), 0.0),
        "stressed_post_cost_expectancy": _safe_float(evaluation.get("stressed_post_cost_expectancy"), -1.0) > _safe_float(rules.get("minimum_stressed_post_cost_expectancy"), 0.0),
        "lower_confidence_bound": _safe_float(evaluation.get("lower_confidence_bound"), -1.0) > _safe_float(rules.get("minimum_lower_confidence_bound"), 0.0),
        "maximum_drawdown": _safe_float(evaluation.get("maximum_drawdown"), 1.0) <= _safe_float(rules.get("maximum_drawdown"), 0.12),
        "parent_return_correlation": abs(_safe_float(evaluation.get("parent_return_correlation"), 1.0)) <= _safe_float(rules.get("maximum_parent_return_correlation"), 0.92),
        "multiple_testing_adjusted_p_value": _safe_float(evaluation.get("multiple_testing_adjusted_p_value"), 1.0) <= _safe_float(rules.get("maximum_multiple_testing_adjusted_p_value"), 0.05),
    }
    failed = [name for name, passed in checks.items() if not passed]
    qualified = not failed
    candidate["evaluation"] = {
        **evaluation,
        "qualified": qualified,
        "checks": checks,
        "failed_checks": failed,
        "evaluation_path": str(resolved_evaluation_path),
        "evaluation_sha256": _file_hash(resolved_evaluation_path),
        "evaluation_age_hours": round(evaluation_age_hours, 6) if evaluation_age_hours is not None else None,
        "attestation_verified": bool(checks["signed_attestation"]),
    }
    candidate["lifecycle_state"] = (
        str(_as_dict(config.get("safety_contract")).get("qualified_lifecycle_state") or "paper_challenger_qualified")
        if qualified
        else "rejected_evidence"
    )
    candidate["execution_authority"] = False
    candidate["paper_execution_authority"] = False
    candidate["serving_eligible"] = False
    candidate["registry_admission_eligible"] = False
    candidate["lineage_parent_approved"] = False
    candidate["paper_allocation_limit"] = 0.0
    candidate["live_order_budget"] = 0.0
    candidate["admission_state"] = "human_review_required" if qualified else "rejected"
    _append_event(
        event_path,
        state,
        "offspring_paper_challenger_qualified" if qualified else "offspring_evidence_rejected",
        {
            "offspring_id": candidate_id,
            "qualified": qualified,
            "failed_checks": failed,
            "evaluation_sha256": candidate["evaluation"]["evaluation_sha256"],
            "offspring_snapshots": [candidate],
            "execution_authority": False,
        },
        integrity=integrity,
    )
    return candidate, failed


def retire_offspring(
    state: dict[str, Any],
    event_path: Path,
    *,
    candidate_id: str,
    reason: str,
    integrity: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    candidate = _candidate_by_id(state, candidate_id)
    if candidate is None:
        return None, ["offspring_not_found"]
    candidate["lifecycle_state"] = "retired"
    candidate["retired_at_utc"] = iso_now()
    candidate["retirement_reason"] = reason
    candidate["execution_authority"] = False
    candidate["paper_execution_authority"] = False
    candidate["serving_eligible"] = False
    candidate["registry_admission_eligible"] = False
    candidate["lineage_parent_approved"] = False
    candidate["paper_allocation_limit"] = 0.0
    candidate["live_order_budget"] = 0.0
    _append_event(
        event_path,
        state,
        "offspring_retired",
        {
            "offspring_id": candidate_id,
            "reason": reason,
            "offspring_snapshots": [candidate],
            "execution_authority": False,
        },
        integrity=integrity,
    )
    return candidate, []


def reconcile_stale_training(
    config: dict[str, Any],
    state: dict[str, Any],
    event_path: Path,
    *,
    integrity: dict[str, Any],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current = now or datetime.now(timezone.utc)
    stale = _stale_training_candidates(state, config, now=current)
    reconciled: list[dict[str, Any]] = []
    for stale_row in stale:
        candidate = _candidate_by_id(state, str(stale_row.get("offspring_id") or ""))
        if candidate is None:
            continue
        candidate["lifecycle_state"] = "training_failed_quarantined"
        candidate["training_completed_at_utc"] = current.isoformat()
        candidate["training_return_code"] = 126
        candidate["training_error_tail"] = "stale training state reconciled after controller interruption"
        candidate["execution_authority"] = False
        candidate["paper_execution_authority"] = False
        candidate["serving_eligible"] = False
        candidate["registry_admission_eligible"] = False
        candidate["lineage_parent_approved"] = False
        candidate["paper_allocation_limit"] = 0.0
        candidate["live_order_budget"] = 0.0
        _append_event(
            event_path,
            state,
            "offspring_stale_training_reconciled",
            {
                **stale_row,
                "offspring_snapshots": [candidate],
                "execution_authority": False,
                "recovery_action": "quarantined",
            },
            integrity=integrity,
        )
        reconciled.append({**stale_row, "recovery_action": "quarantined"})
    return reconciled


def _candidate_integrity_audit(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    limits = _as_dict(config.get("resource_limits"))
    violations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    active_parent_counts: dict[str, int] = {}
    artifact_usage = _candidate_artifact_usage(state)
    for row in _as_list(state.get("offspring")):
        if not isinstance(row, dict):
            violations.append({"offspring_id": "", "violations": ["offspring_state_row_not_object"]})
            continue
        candidate_id = str(row.get("offspring_id") or "")
        reasons: list[str] = []
        if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
            reasons.append("offspring_id_contract_invalid")
        if candidate_id in seen_ids:
            reasons.append("duplicate_offspring_id")
        seen_ids.add(candidate_id)
        if str(row.get("policy_sha256") or "") != _canonical_hash(config):
            reasons.append("offspring_policy_hash_mismatch")
        if _safe_int(row.get("lineage_depth"), 0) > _safe_int(limits.get("max_lineage_depth"), 3):
            reasons.append("offspring_lineage_depth_exceeded")
        if any(
            bool(row.get(key, False))
            for key in ("execution_authority", "paper_execution_authority", "serving_eligible")
        ):
            reasons.append("offspring_authority_contract_violated")
        if _safe_float(row.get("paper_allocation_limit"), 0.0) != 0.0:
            reasons.append("offspring_paper_allocation_nonzero")
        if _safe_float(row.get("live_order_budget"), 0.0) != 0.0:
            reasons.append("offspring_live_order_budget_nonzero")
        lifecycle = str(row.get("lifecycle_state") or "")
        if lifecycle in ACTIVE_STATES:
            for parent_id in _as_list(row.get("parent_bot_ids")):
                normalized = str(parent_id or "")
                active_parent_counts[normalized] = active_parent_counts.get(normalized, 0) + 1
        manifest_path = _resolve_path(project_root, row.get("generation_manifest_path"))
        if not _is_within(manifest_path, project_root / "governance" / "strategy_generations" / "generations"):
            reasons.append("offspring_generation_manifest_outside_governance")
        elif not manifest_path.is_file():
            reasons.append("offspring_generation_manifest_missing")
        elif not hmac.compare_digest(
            str(row.get("generation_manifest_sha256") or ""),
            _file_hash(manifest_path),
        ):
            reasons.append("offspring_generation_manifest_hash_mismatch")
        if lifecycle in {
            "trained_collection_only",
            "paper_evaluation_pending",
            "paper_challenger_qualified",
            "rejected_evidence",
        }:
            for path_key, hash_key, expected_root in (
                ("model_path", "model_sha256", project_root / "models"),
                ("log_path", "log_sha256", project_root / "logs"),
                ("diagnostics_path", "diagnostics_sha256", project_root / "governance" / "training_diagnostics"),
            ):
                artifact_path = _resolve_path(project_root, row.get(path_key))
                if not artifact_path.is_file() or not _is_within(artifact_path, expected_root):
                    reasons.append(f"offspring_{path_key}_missing_or_outside_namespace")
                elif not hmac.compare_digest(str(row.get(hash_key) or ""), _file_hash(artifact_path)):
                    reasons.append(f"offspring_{path_key}_hash_mismatch")
        evaluation = _as_dict(row.get("evaluation"))
        if evaluation:
            evaluation_path = _resolve_path(project_root, evaluation.get("evaluation_path"))
            if not evaluation_path.is_file():
                reasons.append("offspring_evaluation_artifact_missing")
            elif not hmac.compare_digest(
                str(evaluation.get("evaluation_sha256") or ""),
                _file_hash(evaluation_path),
            ):
                reasons.append("offspring_evaluation_artifact_hash_mismatch")
            else:
                source_evaluation = load_json(evaluation_path)
                attestation = _as_dict(source_evaluation.get("attestation"))
                expected = _evaluation_signature(
                    source_evaluation,
                    key_id=str(attestation.get("key_id") or ""),
                    secret=str(integrity.get("secret") or ""),
                )
                if not expected or not hmac.compare_digest(str(attestation.get("signature") or ""), expected):
                    reasons.append("offspring_evaluation_attestation_invalid")
        candidate_bytes = _safe_int(_as_dict(artifact_usage.get("by_candidate_bytes")).get(candidate_id), 0)
        if candidate_bytes > _safe_int(limits.get("max_candidate_artifact_bytes"), 512 * 1024**2):
            reasons.append("offspring_candidate_artifact_limit_exceeded")
        if reasons:
            violations.append({"offspring_id": candidate_id, "violations": ordered_unique(reasons)})
    maximum_active_per_parent = _safe_int(limits.get("max_active_offspring_per_parent"), 1)
    for parent_id, count in active_parent_counts.items():
        if count > maximum_active_per_parent:
            violations.append(
                {
                    "offspring_id": "",
                    "violations": [f"active_offspring_per_parent_exceeded:{parent_id}:{count}"],
                }
            )
    total_artifact_limit = _safe_int(limits.get("max_total_candidate_artifact_bytes"), 4 * 1024**3)
    if _safe_int(artifact_usage.get("total_bytes"), 0) > total_artifact_limit:
        violations.append({"offspring_id": "", "violations": ["total_candidate_artifact_limit_exceeded"]})
    return {
        "ok": not violations,
        "audited_candidate_count": len(seen_ids),
        "violation_count": sum(len(_as_list(row.get("violations"))) for row in violations),
        "violations": violations[:25],
        "active_parent_counts": active_parent_counts,
        "artifact_usage": artifact_usage,
    }


def build_payload(
    project_root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    event_path: Path,
    *,
    action: str,
    action_blockers: list[str] | None = None,
) -> dict[str, Any]:
    policy_validation, integrity = _policy_validation(project_root, config)
    state_validation = _verify_state(
        state,
        require_hash=bool(integrity.get("require_state_hash", True)),
    )
    eligible, rejected = _parent_rejections(project_root, config, state)
    teacher_quality = load_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json"
    )
    chain = verify_event_chain(
        event_path,
        signing_secret=str(integrity.get("secret") or ""),
        signing_key_id=str(integrity.get("key_id") or ""),
        require_signatures=bool(integrity.get("require_signed_events", True)),
    )
    candidate_integrity = _candidate_integrity_audit(
        project_root,
        config,
        state,
        integrity=integrity,
    )
    active = _active_offspring(state)
    limits = _as_dict(config.get("resource_limits"))
    now = datetime.now(timezone.utc)
    proposal_blockers = _proposal_blockers(project_root, config, state, eligible, now=now)
    resource_blockers = [
        blocker
        for blocker in proposal_blockers
        if blocker != "no_parent_has_reproduction_grade_evidence"
    ]
    free_disk_gb = shutil.disk_usage(project_root).free / (1024.0**3)
    retained = _retained_offspring(state)
    stale_training = _stale_training_candidates(state, config, now=now)
    artifact_usage = _as_dict(candidate_integrity.get("artifact_usage"))
    action_blockers = ordered_unique(action_blockers or [])
    status = "ready_to_propose"
    if not policy_validation.get("ok", False):
        status = "blocked_policy_integrity"
    elif not state_validation.get("ok", False):
        status = "blocked_state_integrity"
    elif not chain.get("ok", False):
        status = "blocked_event_chain"
    elif not candidate_integrity.get("ok", False):
        status = "blocked_candidate_integrity"
    elif action_blockers:
        status = "action_blocked"
    elif not eligible:
        status = "collecting_parent_evidence"
    elif active:
        status = "offspring_active"
    elif proposal_blockers:
        status = "bounded_idle"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "policy_id": config.get("policy_id"),
        "ok": bool(
            policy_validation.get("ok", False)
            and state_validation.get("ok", False)
            and chain.get("ok", False)
            and candidate_integrity.get("ok", False)
        ),
        "overall_status": status,
        "action": action,
        "action_blockers": action_blockers,
        "strategy_generation": _safe_int(state.get("generation"), 0),
        "release_candidate_generation_is_separate": True,
        "generation_semantics": "strategy_generation_counts_bounded_offspring_waves; production_candidate_generation_counts_accepted_code_and_evidence_freezes",
        "summary": {
            "eligible_parent_count": len(eligible),
            "rejected_parent_count": len(rejected),
            "total_offspring_count": len(_as_list(state.get("offspring"))),
            "active_offspring_count": len(active),
            "retained_offspring_count": len(retained),
            "stale_training_count": len(stale_training),
            "trained_offspring_count": sum(1 for row in _as_list(state.get("offspring")) if isinstance(row, dict) and row.get("lifecycle_state") == "trained_collection_only"),
            "qualified_paper_challenger_count": sum(1 for row in _as_list(state.get("offspring")) if isinstance(row, dict) and row.get("lifecycle_state") == "paper_challenger_qualified"),
            "execution_authority_count": sum(1 for row in _as_list(state.get("offspring")) if isinstance(row, dict) and bool(row.get("execution_authority", False))),
            "candidate_integrity_violation_count": _safe_int(candidate_integrity.get("violation_count"), 0),
        },
        "resource_envelope": {
            **limits,
            "free_disk_gb": round(free_disk_gb, 3),
            "retained_offspring_count": len(retained),
            "candidate_artifact_bytes": _safe_int(artifact_usage.get("total_bytes"), 0),
            "candidate_artifact_gb": round(_safe_int(artifact_usage.get("total_bytes"), 0) / (1024.0**3), 6),
            "proposal_blockers": proposal_blockers,
            "offspring_are_persistent_processes": False,
            "training_is_serial": _safe_int(limits.get("max_concurrent_training_jobs"), 1) == 1,
        },
        "safety_contract": _as_dict(config.get("safety_contract")),
        "hardening": {
            "grade": "A+"
            if bool(
                policy_validation.get("ok", False)
                and state_validation.get("ok", False)
                and chain.get("ok", False)
                and candidate_integrity.get("ok", False)
            )
            else "F",
            "policy_validation": policy_validation,
            "state_integrity": state_validation,
            "candidate_integrity": candidate_integrity,
            "stale_training_candidates": stale_training,
            "signed_event_chain": bool(chain.get("signatures_required", False)),
            "crash_reconciliation_available": True,
            "evaluation_attestation_required": bool(
                _as_dict(config.get("evaluation")).get("require_signed_attestation", True)
            ),
            "recursive_lineage_requires_human_approval": bool(
                _as_dict(config.get("parent_eligibility")).get(
                    "require_human_lineage_parent_approval_for_offspring", True
                )
            ),
        },
        "parent_evidence": {
            "teacher_quality_status": teacher_quality.get("overall_status"),
            "teacher_quality_summary": _as_dict(teacher_quality.get("summary")),
            "overfitting_awareness": _as_dict(teacher_quality.get("overfitting_awareness")),
            "excluded_reasons": _as_list(teacher_quality.get("excluded_reasons")),
        },
        "event_chain": chain,
        "eligible_parents": eligible,
        "rejected_parents": rejected[:25],
        "active_offspring": active,
        "recommended_actions": ordered_unique(
            [
                "collect independent walk-forward and positive paper evidence before allowing any parent to reproduce" if not eligible else "",
                "wait for the generation cooldown or retire weak offspring before proposing another bounded wave" if resource_blockers else "",
                "keep offspring collection-only until locked holdout, exact replay, post-cost, drawdown, diversity, and multiple-testing checks all pass" if active else "",
                "repair the signed append-only strategy-generation event chain before any mutation" if not chain.get("ok", False) else "",
                "repair the sealed strategy-generation state before any mutation" if not state_validation.get("ok", False) else "",
                "repair candidate artifact or authority integrity before any lifecycle action" if not candidate_integrity.get("ok", False) else "",
                "run strategy-generation --reconcile-stale to quarantine interrupted training state" if stale_training else "",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded, collection-only strategy offspring generation and lineage control.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--state-file", default="")
    parser.add_argument("--event-file", default="")
    parser.add_argument("--out-file", default="")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--propose", action="store_true")
    actions.add_argument("--train-next", action="store_true")
    actions.add_argument("--reconcile-stale", action="store_true")
    actions.add_argument("--evaluate-offspring", default="")
    actions.add_argument("--retire-offspring", default="")
    parser.add_argument("--evaluation-file", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_json(config_path)
    if not config:
        raise SystemExit(f"strategy_generation_config_missing path={config_path}")
    policy_validation, integrity = _policy_validation(project_root, config)
    state_path = Path(args.state_file).expanduser() if args.state_file else project_root / DEFAULT_STATE_PATH.relative_to(PROJECT_ROOT)
    event_path = Path(args.event_file).expanduser() if args.event_file else project_root / DEFAULT_EVENT_PATH.relative_to(PROJECT_ROOT)
    out_path = Path(args.out_file).expanduser() if args.out_file else project_root / DEFAULT_OUT_PATH.relative_to(PROJECT_ROOT)
    policy_id = str(config.get("policy_id") or "bounded_strategy_lineage_v1")
    action = (
        "propose"
        if args.propose
        else "train_next"
        if args.train_next
        else "reconcile_stale"
        if args.reconcile_stale
        else "evaluate_offspring"
        if args.evaluate_offspring
        else "retire_offspring"
        if args.retire_offspring
        else "inspect"
    )
    mutation_requested = action != "inspect"
    blockers: list[str] = []
    mutated = False
    lock_handle = None
    if mutation_requested:
        lock_path = state_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            blockers.append("strategy_generation_mutation_lock_busy")

    try:
        state = _load_state(state_path, policy_id)
        state_validation = _verify_state(
            state,
            require_hash=bool(integrity.get("require_state_hash", True)),
        )
        chain = verify_event_chain(
            event_path,
            signing_secret=str(integrity.get("secret") or ""),
            signing_key_id=str(integrity.get("key_id") or ""),
            require_signatures=bool(integrity.get("require_signed_events", True)),
        )
        if str(state.get("event_chain_head") or "") != str(chain.get("chain_head") or ""):
            recovery = (
                _recover_state_from_signed_tail(state, event_path)
                if args.reconcile_stale and bool(chain.get("ok", False))
                else {"recovered": False, "reason": "explicit_reconcile_required"}
            )
            if bool(recovery.get("recovered", False)):
                _write_state(state_path, state)
                state_validation = _verify_state(
                    state,
                    require_hash=bool(integrity.get("require_state_hash", True)),
                )
                mutated = True
            else:
                chain.setdefault("errors", []).append("state_event_chain_head_mismatch")
                chain.setdefault("errors", []).append(str(recovery.get("reason") or "state_recovery_failed"))
                chain["ok"] = False
        if mutation_requested and not blockers and not bool(policy_validation.get("ok", False)):
            blockers.extend(_as_list(policy_validation.get("errors")) or ["strategy_generation_policy_invalid"])
        if mutation_requested and not blockers and not bool(state_validation.get("ok", False)):
            blockers.extend(_as_list(state_validation.get("errors")) or ["strategy_generation_state_invalid"])
        if not blockers and not bool(chain.get("ok", False)):
            blockers.extend(chain.get("errors") or ["strategy_generation_event_chain_invalid"])
        elif not blockers and args.propose:
            proposed, blockers = propose_generation(
                project_root,
                config,
                state,
                event_path,
                now=datetime.now(timezone.utc),
                integrity=integrity,
            )
            mutated = bool(proposed)
        elif not blockers and args.train_next:
            candidate, blockers = train_next_offspring(
                project_root,
                config,
                state,
                state_path,
                event_path,
                integrity,
            )
            mutated = bool(candidate and not blockers)
        elif not blockers and args.reconcile_stale:
            reconciled = reconcile_stale_training(
                config,
                state,
                event_path,
                integrity=integrity,
            )
            mutated = bool(mutated or reconciled)
        elif not blockers and args.evaluate_offspring:
            if not args.evaluation_file:
                blockers = ["evaluation_file_required"]
            else:
                candidate, blockers = evaluate_offspring(
                    project_root,
                    config,
                    state,
                    event_path,
                    candidate_id=str(args.evaluate_offspring),
                    evaluation_path=Path(args.evaluation_file).expanduser(),
                    integrity=integrity,
                )
                mutated = candidate is not None and "offspring_not_found" not in blockers
        elif not blockers and args.retire_offspring:
            if len(str(args.reason or "").strip()) < 12:
                blockers = ["retirement_reason_must_be_at_least_12_characters"]
            else:
                candidate, blockers = retire_offspring(
                    state,
                    event_path,
                    candidate_id=str(args.retire_offspring),
                    reason=str(args.reason).strip(),
                    integrity=integrity,
                )
                mutated = candidate is not None and not blockers

        if mutated:
            _write_state(state_path, state)
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()
    payload = build_payload(project_root, config, state, event_path, action=action, action_blockers=blockers)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "strategy_generation_control "
            f"status={payload.get('overall_status')} "
            f"generation={payload.get('strategy_generation')} "
            f"eligible_parents={_safe_int(_as_dict(payload.get('summary')).get('eligible_parent_count'), 0)} "
            f"active_offspring={_safe_int(_as_dict(payload.get('summary')).get('active_offspring_count'), 0)}"
        )
    return 0 if bool(payload.get("ok", False)) and not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
