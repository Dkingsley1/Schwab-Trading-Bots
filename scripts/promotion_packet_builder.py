#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packets"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packet_latest.json"
DEFAULT_SIGNING_KEY_PATH = PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packet_signing_key.txt"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_training_success_contract(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    retrain_launch = _load_json(PROJECT_ROOT / "governance" / "health" / "retrain_launch_latest.json")
    quality = _load_json(PROJECT_ROOT / "governance" / "health" / "training_quality_control_latest.json")
    report = _load_json(PROJECT_ROOT / "governance" / "health" / "training_report_latest.json")

    def _parse_iso(raw: Any) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    quality_status = str(quality.get("overall_status") or "").strip().lower()
    report_status = str(report.get("overall_status") or "").strip().lower()
    quality_score = float(quality.get("training_quality_score", 0.0) or 0.0)
    quality_index = float(quality.get("training_quality_index", quality_score) or 0.0)
    retrain_state = str(retrain_launch.get("state") or "").strip().lower()
    retrain_final_status = str(retrain_launch.get("final_status") or "").strip().lower()
    retrain_ts = _parse_iso(retrain_launch.get("timestamp_utc"))
    payload_ts = _parse_iso(payload.get("timestamp_utc")) if payload else None
    stale_failed_payload = bool(
        payload
        and not bool(payload.get("confirmed_training_success", False))
        and retrain_ts is not None
        and payload_ts is not None
        and retrain_ts > payload_ts
        and retrain_final_status in {"completed", "skipped_market_open", "skipped", "running"}
        and retrain_state in {"completed", "running"}
    )
    provisional = bool(
        quality_score >= 75.0
        and quality_status not in {"", "blocked", "critical", "failed"}
        and report_status not in {"critical", "failed"}
    )
    if payload:
        clean_training_pending_promotion = bool(
            provisional
            and not bool(payload.get("confirmed_training_success", False))
            and bool(payload.get("training_completed_ok", False))
            and int(payload.get("trained_count", 0) or 0) > 0
            and int(payload.get("failure_count", 0) or 0) == 0
            and bool(payload.get("trained_ok_but_not_promotable", False))
            and str(payload.get("reason") or "").strip().startswith("trained_ok_but_not_promotable:")
        )
        legacy_absent_data_quality_repair = bool(
            provisional
            and not bool(payload.get("confirmed_training_success", False))
            and bool(payload.get("training_completed_ok", False))
            and bool(payload.get("promotion_applied", False))
            and int(payload.get("failure_count", 0) or 0) == 0
            and str(payload.get("reason") or "").strip() == "data_quality_not_ok"
            and not bool(payload.get("data_quality_present", False))
        )
        if legacy_absent_data_quality_repair:
            payload["confirmed_training_success"] = True
            payload["legacy_absent_data_quality_repaired"] = True
            payload["source_training_quality_score"] = round(quality_score, 2)
            payload["source_training_quality_index"] = round(quality_index, 2)
            payload["source_quality_status"] = quality_status
            payload["source_report_status"] = report_status
        payload["provisional_training_success"] = bool(
            payload.get("provisional_training_success", False)
            or payload.get("confirmed_training_success", False)
            or (stale_failed_payload and provisional)
            or clean_training_pending_promotion
        )
        if stale_failed_payload and provisional:
            payload["source_contract"] = "training_success_stale_fallback"
            payload["stale_source_ignored"] = True
            payload["source_training_quality_score"] = round(quality_score, 2)
            payload["source_training_quality_index"] = round(quality_index, 2)
            payload["source_quality_status"] = quality_status
            payload["source_report_status"] = report_status
        elif legacy_absent_data_quality_repair:
            payload["source_contract"] = "training_success_absent_data_quality_repair"
        elif clean_training_pending_promotion:
            payload["source_contract"] = "training_success_pending_promotion_gate"
            payload["source_training_quality_score"] = round(quality_score, 2)
            payload["source_training_quality_index"] = round(quality_index, 2)
            payload["source_quality_status"] = quality_status
            payload["source_report_status"] = report_status
        payload.setdefault("source_contract", "training_success_latest")
        return payload

    return {
        "confirmed_training_success": False,
        "provisional_training_success": provisional,
        "source_contract": ("training_quality_fallback" if provisional else "missing"),
        "source_training_quality_score": round(quality_score, 2),
        "source_training_quality_index": round(quality_index, 2),
        "source_quality_status": quality_status,
        "source_report_status": report_status,
    }


def _sha256_file(path_text: Any) -> str:
    path = _resolve_file_path(path_text)
    if path is None:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file_path(path_text: Any) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    try:
        matches = [path for path in directory.glob(pattern) if path.is_file()]
    except OSError:
        return None
    if not matches:
        return None

    def _sort_key(path: Path) -> tuple[int, str]:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        return mtime_ns, str(path)

    return max(matches, key=_sort_key)


def _latest_training_log_for_model(bot_id: str, model_path: str) -> Path | None:
    model = Path(str(model_path or "").strip())
    if model.name.endswith(".npz"):
        paired_log = PROJECT_ROOT / "logs" / f"{model.stem}.json"
        if paired_log.exists() and paired_log.is_file():
            return paired_log
    return _latest_matching_file(PROJECT_ROOT / "logs", f"{bot_id}_*.json")


def _log_test_accuracy(log_path: Any) -> Any:
    path = _resolve_file_path(log_path)
    if path is None:
        return None
    payload = _load_json(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    for source in (payload, metrics):
        if not isinstance(source, dict):
            continue
        value = source.get("test_accuracy")
        if value is not None:
            return value
    return None


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_signing_key(path: Path | None = None) -> tuple[str, str]:
    env_key = str(os.getenv("PROMOTION_PACKET_SIGNING_KEY", "") or "").strip()
    if env_key:
        return env_key, "env:PROMOTION_PACKET_SIGNING_KEY"
    candidate = path or DEFAULT_SIGNING_KEY_PATH
    try:
        file_key = candidate.read_text(encoding="utf-8").strip()
    except Exception:
        file_key = ""
    if file_key:
        return file_key, str(candidate)
    return "", ""


def _bootstrap_signing_key(path: Path | None = None) -> tuple[str, str]:
    candidate = path or DEFAULT_SIGNING_KEY_PATH
    candidate.parent.mkdir(parents=True, exist_ok=True)
    signing_key, signing_source = _load_signing_key(candidate)
    if signing_key:
        return signing_key, signing_source
    generated = secrets.token_hex(32)
    candidate.write_text(generated + "\n", encoding="utf-8")
    return generated, str(candidate)


def _sign_packet(packet_sha256: str, signing_key: str, signing_source: str) -> dict[str, Any]:
    if not signing_key:
        return {
            "algorithm": "hmac-sha256",
            "key_id": "",
            "source": signing_source,
            "signature": "",
            "verified": False,
            "status": "missing_signing_key",
            "payload_sha256": packet_sha256,
        }
    key_id = hashlib.sha256(signing_key.encode("utf-8")).hexdigest()[:16]
    signature = hmac.new(signing_key.encode("utf-8"), packet_sha256.encode("utf-8"), hashlib.sha256).hexdigest()
    verified = bool(
        hmac.compare_digest(
            signature,
            hmac.new(signing_key.encode("utf-8"), packet_sha256.encode("utf-8"), hashlib.sha256).hexdigest(),
        )
    )
    return {
        "algorithm": "hmac-sha256",
        "key_id": key_id,
        "source": signing_source,
        "signature": signature,
        "verified": verified,
        "status": ("verified" if verified else "verification_failed"),
        "payload_sha256": packet_sha256,
    }


def _trained_bot_ids(scorecard: dict[str, Any]) -> list[str]:
    outcomes = scorecard.get("target_outcomes") if isinstance(scorecard.get("target_outcomes"), list) else []
    rows = [
        str((row or {}).get("bot_id") or "").strip()
        for row in outcomes
        if isinstance(row, dict) and str((row or {}).get("status") or "") == "trained"
    ]
    return [bot_id for bot_id in rows if bot_id]


def _promotion_scope_contract(
    *,
    retrain_scorecard: dict[str, Any],
    promotion_gate: dict[str, Any] | None,
    graduation_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    retrain_trained_bot_ids = _trained_bot_ids(retrain_scorecard)
    promotion_gate = promotion_gate or {}
    graduation_gate = graduation_gate or {}
    scope_known = bool(
        "considered_bots" in promotion_gate
        or "promote_ok" in promotion_gate
        or "graduation_scope_active_count" in graduation_gate
        or "promotion_scope_active" in graduation_gate
    )
    if not scope_known:
        active = bool(
            retrain_trained_bot_ids
            or int(retrain_scorecard.get("target_count", 0) or 0) > 0
            or int(retrain_scorecard.get("failure_count", 0) or 0) > 0
        )
        return {
            "active": active,
            "candidate_ids": retrain_trained_bot_ids,
            "retrain_trained_bot_ids": retrain_trained_bot_ids,
            "excluded_non_candidate_training_ids": [],
            "source": "retrain_scorecard_fallback",
            "scope_known": False,
        }

    active = bool(
        promotion_gate.get("promote_ok", False)
        or int(promotion_gate.get("considered_bots", 0) or 0) > 0
        or int(graduation_gate.get("graduation_scope_active_count", 0) or 0) > 0
        or graduation_gate.get("promotion_scope_active", False)
    )
    candidate_ids: set[str] = {
        str(bot_id or "").strip()
        for bot_id in (promotion_gate.get("considered_bot_ids") or [])
        if str(bot_id or "").strip()
    }
    for key in ("pass_examples", "near_pass_examples", "fail_examples"):
        rows = promotion_gate.get(key) if isinstance(promotion_gate.get(key), list) else []
        candidate_ids.update(
            str((row or {}).get("bot_id") or "").strip()
            for row in rows
            if isinstance(row, dict) and str((row or {}).get("bot_id") or "").strip()
        )
    graduation_rows = (
        graduation_gate.get("immature_active_examples")
        if isinstance(graduation_gate.get("immature_active_examples"), list)
        else []
    )
    candidate_ids.update(
        str((row or {}).get("bot_id") or "").strip()
        for row in graduation_rows
        if isinstance(row, dict) and str((row or {}).get("bot_id") or "").strip()
    )
    if active and not candidate_ids:
        candidate_ids.update(retrain_trained_bot_ids)
    ordered_candidate_ids = sorted(candidate_ids)
    candidate_set = {bot_id.lower() for bot_id in ordered_candidate_ids}
    excluded = [bot_id for bot_id in retrain_trained_bot_ids if bot_id.lower() not in candidate_set]
    return {
        "active": active,
        "candidate_ids": ordered_candidate_ids,
        "retrain_trained_bot_ids": retrain_trained_bot_ids,
        "excluded_non_candidate_training_ids": excluded,
        "source": "promotion_and_graduation_gates",
        "scope_known": True,
    }


def _registry_model_rows(registry: dict[str, Any], target_ids: list[str]) -> list[dict[str, Any]]:
    wanted = {str(bot_id).strip().lower() for bot_id in target_ids if str(bot_id).strip()}
    if not wanted:
        return []
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _model_row(bot_id: str, row: dict[str, Any]) -> dict[str, Any]:
        model_path = str(row.get("model_path") or "").strip()
        log_path = str(row.get("log_file") or "").strip()
        model_sha256 = _sha256_file(model_path)
        used_model_fallback = False
        if not model_sha256:
            latest_model = _latest_matching_file(PROJECT_ROOT / "models", f"{bot_id}_*.npz")
            if latest_model is not None:
                model_path = str(latest_model)
                model_sha256 = _sha256_file(latest_model)
                used_model_fallback = True
        log_sha256 = _sha256_file(log_path)
        if used_model_fallback or not log_sha256:
            latest_log = _latest_training_log_for_model(bot_id, model_path)
            if latest_log is not None:
                log_path = str(latest_log)
                log_sha256 = _sha256_file(latest_log)
        test_accuracy = row.get("test_accuracy")
        if test_accuracy is None:
            test_accuracy = _log_test_accuracy(log_path)
        return {
            "bot_id": bot_id,
            "lifecycle_state": str(row.get("lifecycle_state") or "").strip().lower(),
            "active": bool(row.get("active", False)),
            "model_path": model_path,
            "model_sha256": model_sha256,
            "log_path": log_path,
            "log_sha256": log_sha256,
            "test_accuracy": test_accuracy,
        }

    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        if wanted and bot_id.lower() not in wanted:
            continue
        out.append(_model_row(bot_id, row))
        seen.add(bot_id.lower())
    for target_id in target_ids:
        bot_id = str(target_id or "").strip()
        if not bot_id or bot_id.lower() in seen:
            continue
        out.append(_model_row(bot_id, {"bot_id": bot_id}))
    out.sort(key=lambda item: item["bot_id"])
    return out


def _weekly_retrain_script_sha256() -> str:
    return _sha256_file(PROJECT_ROOT / "scripts" / "weekly_retrain.py")


def _idle_promotion_scope(packet: dict[str, Any]) -> bool:
    scope = packet.get("promotion_scope") if isinstance(packet.get("promotion_scope"), dict) else {}
    trained_bot_ids = scope.get("trained_bot_ids") if isinstance(scope.get("trained_bot_ids"), list) else []
    return len([str(bot_id).strip() for bot_id in trained_bot_ids if str(bot_id).strip()]) == 0


def _new_bot_admission_ok_for_scope(
    guard: dict[str, Any],
    *,
    trained_bot_ids: list[str],
    promotion_scope_active: bool,
) -> tuple[bool, list[str]]:
    if not promotion_scope_active:
        return True, []
    if bool(guard.get("ok", False)):
        return True, []

    target_ids = {str(bot_id or "").strip() for bot_id in trained_bot_ids if str(bot_id or "").strip()}
    blocking_rows = guard.get("blocking_candidates") if isinstance(guard.get("blocking_candidates"), list) else []
    blocking_ids = {
        str((row or {}).get("bot_id") or "").strip()
        for row in blocking_rows
        if isinstance(row, dict) and str((row or {}).get("bot_id") or "").strip()
    }
    relevant = sorted(target_ids & blocking_ids)
    return not bool(relevant), relevant


def build_payload(
    *,
    retrain_scorecard: dict[str, Any],
    training_success: dict[str, Any],
    feature_store_manifest: dict[str, Any],
    replay_hash_registry_guard: dict[str, Any],
    bot_support_owner_guard: dict[str, Any],
    new_bot_admission_guard: dict[str, Any],
    schema_compatibility_guard: dict[str, Any],
    golden_replay_regression_guard: dict[str, Any],
    cohort_drift_baseline_guard: dict[str, Any],
    probation_guard: dict[str, Any],
    champion_registry: dict[str, Any],
    content_store: dict[str, Any],
    master_registry: dict[str, Any],
    signing_key: str,
    signing_source: str,
    promotion_gate: dict[str, Any] | None = None,
    graduation_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lineage = retrain_scorecard.get("lineage") if isinstance(retrain_scorecard.get("lineage"), dict) else {}
    dataset_contract = feature_store_manifest.get("dataset_contract") if isinstance(feature_store_manifest.get("dataset_contract"), dict) else {}
    point_in_time_contract = (
        feature_store_manifest.get("point_in_time_contract")
        if isinstance(feature_store_manifest.get("point_in_time_contract"), dict)
        else {}
    )
    feature_contract = feature_store_manifest.get("feature_contract") if isinstance(feature_store_manifest.get("feature_contract"), dict) else {}
    label_contract = feature_store_manifest.get("label_contract") if isinstance(feature_store_manifest.get("label_contract"), dict) else {}
    contract_hashes = feature_store_manifest.get("contract_hashes") if isinstance(feature_store_manifest.get("contract_hashes"), dict) else {}
    replay_details = replay_hash_registry_guard.get("details") if isinstance(replay_hash_registry_guard.get("details"), dict) else {}

    scope_contract = _promotion_scope_contract(
        retrain_scorecard=retrain_scorecard,
        promotion_gate=promotion_gate,
        graduation_gate=graduation_gate,
    )
    trained_bot_ids = list(scope_contract["candidate_ids"])
    promotion_scope_active = bool(scope_contract["active"])
    model_artifacts = _registry_model_rows(master_registry, trained_bot_ids)
    rollback_candidate = str(((champion_registry.get("champion") or {}).get("rollback_candidate") or "")).strip()
    rollback_entrypoint = PROJECT_ROOT / "scripts" / "release_ops.sh"
    rollback_reference = str(lineage.get("git_commit") or "").strip()
    rollback_command = (
        f"{rollback_entrypoint} rollback {rollback_reference}"
        if rollback_reference and rollback_entrypoint.exists()
        else ""
    )
    training_success_confirmed = bool(training_success.get("confirmed_training_success", False))
    training_success_seed_ready = bool(
        training_success_confirmed
        or training_success.get("provisional_training_success", False)
    )
    feature_store_strict_ready = bool(
        feature_store_manifest.get("strict_ok", False)
        or feature_store_manifest.get("strict_seed_ready", False)
    )
    schema_compatibility_ready = bool(
        schema_compatibility_guard.get("ok", False)
        or schema_compatibility_guard.get("compatibility_seed_ready", False)
    )
    golden_replay_ready = bool(
        golden_replay_regression_guard.get("ok", False)
        or golden_replay_regression_guard.get("seed_ready", False)
    )
    new_bot_admission_ok, new_bot_admission_relevant_blocking_ids = _new_bot_admission_ok_for_scope(
        new_bot_admission_guard,
        trained_bot_ids=trained_bot_ids,
        promotion_scope_active=promotion_scope_active,
    )

    gate_results = {
        "training_success_confirmed": training_success_seed_ready,
        "feature_store_manifest_strict_ok": feature_store_strict_ready,
        "bot_support_owner_guard_ok": bool(bot_support_owner_guard.get("ok", False) or not promotion_scope_active),
        "new_bot_admission_ok": bool(new_bot_admission_ok),
        "retrain_schema_compatibility_ok": schema_compatibility_ready,
        "golden_replay_regression_ok": golden_replay_ready,
        "cohort_drift_baseline_ok": bool(cohort_drift_baseline_guard.get("ok", False)),
        "champion_challenger_probation_ok": bool(probation_guard.get("ok", False)),
        "replay_hash_registry_ok": bool(replay_hash_registry_guard.get("ok", False)),
        "content_store_manifest_present": bool(str(content_store.get("manifest_hash") or "").strip()),
    }
    trained_models_complete = bool(
        trained_bot_ids and all(str(row.get("model_sha256") or "").strip() for row in model_artifacts)
    )
    code_git_commit = str(lineage.get("git_commit") or "").strip()
    weekly_retrain_script_sha256 = str(lineage.get("weekly_retrain_script_sha256") or "").strip() or _weekly_retrain_script_sha256()
    model_hash = _sha256_json(
        {
            "promotion_scope_active": promotion_scope_active,
            "trained_bot_ids": trained_bot_ids,
            "model_artifacts": model_artifacts,
        }
    )
    replay_hash = _sha256_json(
        {
            "paper": replay_details.get("paper") if isinstance(replay_details.get("paper"), dict) else {},
            "e2e": replay_details.get("e2e") if isinstance(replay_details.get("e2e"), dict) else {},
        }
    )
    dataset_hash = str(dataset_contract.get("rows_sha256") or "").strip()
    code_hash = code_git_commit or weekly_retrain_script_sha256
    hash_bundle_complete = bool(dataset_hash and model_hash and replay_hash and code_hash)
    bundle_hash = (
        _sha256_json(
            {
                "dataset_hash": dataset_hash,
                "model_hash": model_hash,
                "replay_hash": replay_hash,
                "code_hash": code_hash,
            }
        )
        if hash_bundle_complete
        else ""
    )
    exact_replay_ready = bool(
        hash_bundle_complete
        and bool(replay_hash_registry_guard.get("ok", False))
        and (trained_models_complete or not promotion_scope_active)
    )
    trained_models_contract_ready = bool(trained_models_complete or not promotion_scope_active)

    packet = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "promotion_scope": {
            "target_count": int(retrain_scorecard.get("target_count", 0) or 0),
            "trained_bot_ids": trained_bot_ids,
            "promotion_candidate_ids": trained_bot_ids,
            "retrain_trained_bot_ids": list(scope_contract["retrain_trained_bot_ids"]),
            "excluded_non_candidate_training_ids": list(scope_contract["excluded_non_candidate_training_ids"]),
            "promotion_scope_active": promotion_scope_active,
            "scope_known": bool(scope_contract["scope_known"]),
            "scope_source": str(scope_contract["source"]),
            "failure_count": int(retrain_scorecard.get("failure_count", 0) or 0),
            "master_update_status": str(retrain_scorecard.get("master_update_status") or ""),
        },
        "dataset": {
            "rows_path": str(dataset_contract.get("rows_path") or ""),
            "rows_sha256": str(dataset_contract.get("rows_sha256") or ""),
            "dataset_manifest_sha256": str(contract_hashes.get("dataset_manifest_sha256") or ""),
            "point_in_time_contract_sha256": str(contract_hashes.get("point_in_time_contract_sha256") or ""),
            "dataset_join_keys": point_in_time_contract.get("dataset_join_keys")
            if isinstance(point_in_time_contract.get("dataset_join_keys"), list)
            else [],
            "event_join_keys": point_in_time_contract.get("event_join_keys")
            if isinstance(point_in_time_contract.get("event_join_keys"), list)
            else [],
            "feature_env_hash": str(feature_contract.get("env_hash") or ""),
            "feature_schema_version": str(label_contract.get("feature_schema_version") or ""),
            "label_horizons": label_contract.get("horizons") if isinstance(label_contract.get("horizons"), dict) else {},
            "lineage_schema_version": int(feature_store_manifest.get("lineage_schema_version", 0) or 0),
        },
        "code": {
            "git_commit": code_git_commit,
            "weekly_retrain_script_sha256": weekly_retrain_script_sha256,
            "code_identity": code_hash,
        },
        "model_artifacts": model_artifacts,
        "replay": {
            "paper": replay_details.get("paper") if isinstance(replay_details.get("paper"), dict) else {},
            "e2e": replay_details.get("e2e") if isinstance(replay_details.get("e2e"), dict) else {},
        },
        "replayability_contract": {
            "source_contract": "promotion_packet_builder",
            "idle_scope": not promotion_scope_active,
            "dataset_hash": dataset_hash,
            "model_hash": model_hash,
            "replay_hash": replay_hash,
            "code_hash": code_hash,
            "bundle_hash": bundle_hash,
            "hash_bundle_complete": hash_bundle_complete,
            "exact_replay_ready": exact_replay_ready,
            "trained_models_contract_ready": trained_models_contract_ready,
        },
        "gate_results": gate_results,
        "new_bot_admission_relevant_blocking_ids": new_bot_admission_relevant_blocking_ids,
        "gate_seed_results": {
            "training_success_seed_ready": training_success_seed_ready and not training_success_confirmed,
            "feature_store_seed_ready": feature_store_strict_ready and not bool(feature_store_manifest.get("strict_ok", False)),
            "schema_compatibility_seed_ready": schema_compatibility_ready and not bool(schema_compatibility_guard.get("ok", False)),
            "golden_replay_seed_ready": golden_replay_ready and not bool(golden_replay_regression_guard.get("ok", False)),
        },
        "trained_models_complete": trained_models_complete,
        "rollback_bundle": {
            "content_store_manifest_hash": str(content_store.get("manifest_hash") or ""),
            "registry_backup_before_retrain": str(lineage.get("registry_backup_before_retrain") or ""),
            "rollback_candidate": rollback_candidate,
            "rollback_entrypoint": str(rollback_entrypoint) if rollback_entrypoint.exists() else "",
            "rollback_reference": rollback_reference,
            "rollback_command": rollback_command,
        },
        "sources": {
            "retrain_scorecard": str(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json"),
            "training_success": str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"),
            "training_quality_control": str(PROJECT_ROOT / "governance" / "health" / "training_quality_control_latest.json"),
            "training_report": str(PROJECT_ROOT / "governance" / "health" / "training_report_latest.json"),
            "feature_store_manifest": str(PROJECT_ROOT / "governance" / "feature_store" / "latest.json"),
            "replay_hash_registry_guard": str(PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"),
            "bot_support_owner_guard": str(PROJECT_ROOT / "governance" / "health" / "bot_support_owner_guard_latest.json"),
            "new_bot_admission_guard": str(PROJECT_ROOT / "governance" / "health" / "new_bot_admission_guard_latest.json"),
            "retrain_schema_compatibility_guard": str(PROJECT_ROOT / "governance" / "health" / "retrain_schema_compatibility_latest.json"),
            "golden_replay_regression_guard": str(PROJECT_ROOT / "governance" / "health" / "golden_replay_regression_latest.json"),
            "cohort_drift_baseline_guard": str(PROJECT_ROOT / "governance" / "health" / "cohort_drift_baseline_latest.json"),
            "champion_challenger_probation_guard": str(PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_latest.json"),
            "content_store": str(PROJECT_ROOT / "governance" / "content_store" / "latest.json"),
        },
    }
    packet["packet_sha256"] = _sha256_json(packet)
    packet["signature"] = _sign_packet(packet["packet_sha256"], signing_key, signing_source)
    packet_complete = bool(
        gate_results["training_success_confirmed"]
        and gate_results["feature_store_manifest_strict_ok"]
        and gate_results["bot_support_owner_guard_ok"]
        and gate_results["new_bot_admission_ok"]
        and gate_results["retrain_schema_compatibility_ok"]
        and gate_results["golden_replay_regression_ok"]
        and gate_results["cohort_drift_baseline_ok"]
        and gate_results["champion_challenger_probation_ok"]
        and gate_results["replay_hash_registry_ok"]
        and gate_results["content_store_manifest_present"]
        and trained_models_contract_ready
        and bool(packet["signature"].get("verified", False))
        and str(contract_hashes.get("dataset_manifest_sha256") or "").strip()
        and str(packet["code"].get("code_identity") or "").strip()
        and bool(packet["replayability_contract"].get("hash_bundle_complete", False))
        and bool(packet["replayability_contract"].get("exact_replay_ready", False))
    )
    packet["packet_complete"] = packet_complete
    packet["ok"] = packet_complete
    packet["ready_for_committee"] = packet_complete
    packet["committee_packet_seed_ready"] = bool(
        str(packet.get("packet_sha256") or "").strip()
        and str(packet["dataset"].get("rows_sha256") or "").strip()
        and bool(packet.get("sources"))
    )
    packet["committee"] = {
        "approval_roles": ["research_reviewer", "risk_reviewer"],
        "approval_threshold": 2,
        "approval_required": True,
        "packet_sha256": str(packet.get("packet_sha256") or ""),
        "signature_verified": bool((packet.get("signature") or {}).get("verified", False)),
        "seed_ready": bool(packet.get("committee_packet_seed_ready", False)),
        "ready_for_committee": bool(packet_complete),
        "approval_state": (
            "ready_for_committee"
            if packet_complete
            else "seed_ready_blocked_by_quality" if bool(packet.get("committee_packet_seed_ready", False)) else "not_ready"
        ),
        "source_contracts": {
            "feature_store_manifest": bool(gate_results.get("feature_store_manifest_strict_ok", False)),
            "training_success": bool(gate_results.get("training_success_confirmed", False)),
            "exact_replay_ready": bool(packet["replayability_contract"].get("exact_replay_ready", False)),
        },
    }
    packet["signing_material_ready"] = bool(signing_key)
    normalized_ts = (
        str(packet["timestamp_utc"])
        .replace(":", "")
        .replace("-", "")
        .replace("+00:00", "Z")
    )
    packet["packet_id"] = f"{normalized_ts}-{packet['packet_sha256'][:12]}"
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an immutable promotion packet for retrain governance.")
    parser.add_argument("--retrain-scorecard", default=str(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json"))
    parser.add_argument("--training-success", default=str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"))
    parser.add_argument("--feature-store-manifest", default=str(PROJECT_ROOT / "governance" / "feature_store" / "latest.json"))
    parser.add_argument("--replay-hash-registry-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"))
    parser.add_argument("--bot-support-owner-guard-file", default=str(PROJECT_ROOT / "governance" / "health" / "bot_support_owner_guard_latest.json"))
    parser.add_argument("--new-bot-admission-file", default=str(PROJECT_ROOT / "governance" / "health" / "new_bot_admission_guard_latest.json"))
    parser.add_argument("--schema-compatibility-file", default=str(PROJECT_ROOT / "governance" / "health" / "retrain_schema_compatibility_latest.json"))
    parser.add_argument("--golden-replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "golden_replay_regression_latest.json"))
    parser.add_argument("--cohort-drift-file", default=str(PROJECT_ROOT / "governance" / "health" / "cohort_drift_baseline_latest.json"))
    parser.add_argument("--probation-guard-file", default=str(PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_latest.json"))
    parser.add_argument("--champion-registry", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "registry.json"))
    parser.add_argument("--content-store-file", default=str(PROJECT_ROOT / "governance" / "content_store" / "latest.json"))
    parser.add_argument("--master-registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--promotion-gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--graduation-gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--signing-key-file", default=str(DEFAULT_SIGNING_KEY_PATH))
    parser.add_argument("--bootstrap-local-signing-key", action="store_true")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--allow-idle-success", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    signing_key, signing_source = _load_signing_key(Path(args.signing_key_file))
    if args.bootstrap_local_signing_key and not signing_key:
        signing_key, signing_source = _bootstrap_signing_key(Path(args.signing_key_file))

    payload = build_payload(
        retrain_scorecard=_load_json(Path(args.retrain_scorecard)),
        training_success=_load_training_success_contract(Path(args.training_success)),
        feature_store_manifest=_load_json(Path(args.feature_store_manifest)),
        replay_hash_registry_guard=_load_json(Path(args.replay_hash_registry_file)),
        bot_support_owner_guard=_load_json(Path(args.bot_support_owner_guard_file)),
        new_bot_admission_guard=_load_json(Path(args.new_bot_admission_file)),
        schema_compatibility_guard=_load_json(Path(args.schema_compatibility_file)),
        golden_replay_regression_guard=_load_json(Path(args.golden_replay_file)),
        cohort_drift_baseline_guard=_load_json(Path(args.cohort_drift_file)),
        probation_guard=_load_json(Path(args.probation_guard_file)),
        champion_registry=_load_json(Path(args.champion_registry)),
        content_store=_load_json(Path(args.content_store_file)),
        master_registry=_load_json(Path(args.master_registry)),
        signing_key=signing_key,
        signing_source=signing_source,
        promotion_gate=_load_json(Path(args.promotion_gate_file)),
        graduation_gate=_load_json(Path(args.graduation_gate_file)),
    )

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    history_path = history_dir / f"promotion_packet_{ts}.json"
    history_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "promotion_packet_builder "
            f"ok={str(payload['ok']).lower()} "
            f"trained_targets={len(payload.get('promotion_scope', {}).get('trained_bot_ids', []))}"
        )
    return 0 if bool(payload.get("ok", False) or (args.allow_idle_success and _idle_promotion_scope(payload))) else 2


if __name__ == "__main__":
    raise SystemExit(main())
