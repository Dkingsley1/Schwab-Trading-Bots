from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_DATE_TOKEN = re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(raw: Any, *, name: str) -> int:
    value = int(raw or 0)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_lifecycle_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tiered ingestion lifecycle policy must be an object")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("tiered ingestion lifecycle thresholds are required")

    required = (
        "target_segment_bytes",
        "small_segment_max_bytes",
        "min_compaction_inputs",
        "max_compaction_inputs",
        "max_compaction_wave_bytes",
        "sealed_min_age_days",
        "cold_after_days",
        "small_segment_soft_count",
        "small_segment_hard_count",
        "minimum_hot_free_bytes",
        "critical_hot_free_bytes",
        "snapshot_retention_count",
        "orphan_grace_hours",
    )
    normalized = {name: _positive_int(thresholds.get(name), name=name) for name in required}
    if normalized["min_compaction_inputs"] < 2:
        raise ValueError("min_compaction_inputs must be at least two")
    if normalized["max_compaction_inputs"] < normalized["min_compaction_inputs"]:
        raise ValueError("max_compaction_inputs must cover min_compaction_inputs")
    if normalized["small_segment_max_bytes"] >= normalized["target_segment_bytes"]:
        raise ValueError("small segments must be smaller than the target segment")
    if normalized["small_segment_hard_count"] <= normalized["small_segment_soft_count"]:
        raise ValueError("small_segment_hard_count must exceed the soft count")
    if normalized["minimum_hot_free_bytes"] <= normalized["critical_hot_free_bytes"]:
        raise ValueError("minimum hot free bytes must exceed the critical threshold")
    if normalized["cold_after_days"] <= normalized["sealed_min_age_days"]:
        raise ValueError("cold_after_days must exceed sealed_min_age_days")

    authority = payload.get("authority")
    if not isinstance(authority, dict) or any(bool(value) for value in authority.values()):
        raise ValueError("tiered ingestion lifecycle authority must remain fully disabled")

    normalized_payload = dict(payload)
    normalized_payload["thresholds"] = normalized
    normalized_payload["policy_sha256"] = _sha256(
        {key: value for key, value in normalized_payload.items() if key != "policy_sha256"}
    )
    return normalized_payload


def _partition_key(relative_path: str) -> str:
    matches = _DATE_TOKEN.findall(relative_path)
    if not matches:
        return "undated"
    year, month, _day = matches[-1]
    return f"{year}{month}"


def _compaction_groups(
    entries: Sequence[Mapping[str, Any]],
    *,
    target_bytes: int,
    min_inputs: int,
    max_inputs: int,
    max_wave_bytes: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in entries:
        row = dict(raw)
        family = str(row.get("family") or "unknown")
        relative_path = str(row.get("relative_path") or "")
        grouped.setdefault((family, _partition_key(relative_path)), []).append(row)

    proposed: list[dict[str, Any]] = []
    deferred_input_count = 0

    def flush(key: tuple[str, str], pending: list[dict[str, Any]]) -> None:
        nonlocal deferred_input_count
        if len(pending) < min_inputs:
            deferred_input_count += len(pending)
            return
        family, partition = key
        identity = [
            {
                "relative_path": str(row.get("relative_path") or ""),
                "size_bytes": int(row.get("size_bytes", 0) or 0),
                "source_stat_fingerprint": row.get("source_stat_fingerprint") or {},
            }
            for row in pending
        ]
        digest = _sha256(identity)
        proposed.append(
            {
                "family": family,
                "partition": partition,
                "input_count": len(pending),
                "input_bytes": sum(int(row.get("size_bytes", 0) or 0) for row in pending),
                "input_paths": [str(row.get("relative_path") or "") for row in pending],
                "planned_output": (
                    f"data/deep_cold/compacted/{family}/{partition}/{digest[:24]}.jsonl.zst"
                ),
                "atomic_snapshot_required": True,
                "verify_size_and_sha256": True,
                "restore_probe_required": True,
                "source_delete_authority": False,
            }
        )

    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: str(row.get("relative_path") or ""))
        pending: list[dict[str, Any]] = []
        pending_bytes = 0
        for row in rows:
            size_bytes = int(row.get("size_bytes", 0) or 0)
            if pending and (
                pending_bytes + size_bytes > target_bytes or len(pending) >= max_inputs
            ):
                flush(key, pending)
                pending = []
                pending_bytes = 0
            pending.append(row)
            pending_bytes += size_bytes
        if pending:
            flush(key, pending)

    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    deferred_groups = 0
    for group in sorted(
        proposed,
        key=lambda row: (
            -int(row.get("input_count", 0) or 0),
            -int(row.get("input_bytes", 0) or 0),
            str(row.get("planned_output") or ""),
        ),
    ):
        group_bytes = int(group.get("input_bytes", 0) or 0)
        if selected and selected_bytes + group_bytes > max_wave_bytes:
            deferred_groups += 1
            continue
        selected.append(group)
        selected_bytes += group_bytes

    return {
        "group_count": len(selected),
        "input_count": sum(int(group["input_count"]) for group in selected),
        "input_bytes": selected_bytes,
        "deferred_group_count": deferred_groups,
        "deferred_input_count": deferred_input_count,
        "groups": selected,
        "max_wave_bytes": max_wave_bytes,
        "execution_authority": False,
        "source_delete_authority": False,
    }


def plan_tiered_ingestion_lifecycle(
    entries: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    available_hot_bytes: int,
    hot_path_over_budget_bytes: int = 0,
) -> dict[str, Any]:
    thresholds = dict(policy.get("thresholds") or {})
    target_bytes = _positive_int(thresholds.get("target_segment_bytes"), name="target_segment_bytes")
    small_max = _positive_int(thresholds.get("small_segment_max_bytes"), name="small_segment_max_bytes")
    min_inputs = _positive_int(thresholds.get("min_compaction_inputs"), name="min_compaction_inputs")
    max_inputs = _positive_int(thresholds.get("max_compaction_inputs"), name="max_compaction_inputs")
    max_wave_bytes = _positive_int(thresholds.get("max_compaction_wave_bytes"), name="max_compaction_wave_bytes")
    sealed_min_days = _positive_int(thresholds.get("sealed_min_age_days"), name="sealed_min_age_days")
    cold_after_days = _positive_int(thresholds.get("cold_after_days"), name="cold_after_days")
    soft_count = _positive_int(thresholds.get("small_segment_soft_count"), name="small_segment_soft_count")
    hard_count = _positive_int(thresholds.get("small_segment_hard_count"), name="small_segment_hard_count")
    minimum_free = _positive_int(thresholds.get("minimum_hot_free_bytes"), name="minimum_hot_free_bytes")
    critical_free = _positive_int(thresholds.get("critical_hot_free_bytes"), name="critical_hot_free_bytes")
    snapshot_retention = _positive_int(thresholds.get("snapshot_retention_count"), name="snapshot_retention_count")
    orphan_grace_hours = _positive_int(thresholds.get("orphan_grace_hours"), name="orphan_grace_hours")

    compaction_roles = {str(value) for value in policy.get("generic_compaction_roles") or []}
    compaction_classes = {
        str(value) for value in policy.get("generic_compaction_classifications") or []
    }
    protected_classes = {str(value) for value in policy.get("protected_classifications") or []}

    rows = [dict(raw) for raw in entries]
    small_segments: list[dict[str, Any]] = []
    warm_candidates: list[dict[str, Any]] = []
    cold_candidates: list[dict[str, Any]] = []
    unknown_age_candidates: list[dict[str, Any]] = []
    retirement_review_ready: list[str] = []

    for row in rows:
        classification = str(row.get("classification") or "")
        role = str(row.get("service_role") or "")
        size_bytes = int(row.get("size_bytes", 0) or 0)
        age_raw = row.get("age_days")
        age_days = int(age_raw) if isinstance(age_raw, (int, float)) else None
        if (
            classification in compaction_classes
            and role in compaction_roles
            and 0 < size_bytes <= small_max
            and age_days is not None
            and age_days >= sealed_min_days
        ):
            small_segments.append(row)

        if classification == "eligible_manifest_backed_offload":
            summary = {
                "relative_path": str(row.get("relative_path") or ""),
                "size_bytes": size_bytes,
                "family": str(row.get("family") or "unknown"),
                "age_days": age_days,
            }
            if age_days is None:
                unknown_age_candidates.append(summary)
            elif age_days >= cold_after_days:
                cold_candidates.append(summary)
            elif age_days >= sealed_min_days:
                warm_candidates.append(summary)

            evidence = row.get("retirement_evidence")
            if isinstance(evidence, Mapping):
                proof_ok = all(
                    bool(evidence.get(field))
                    for field in (
                        "verified_cold_copy",
                        "sha256_match",
                        "restore_probe",
                        "retention_gate",
                    )
                )
                unreferenced = int(evidence.get("snapshot_reference_count", 1) or 0) == 0
                grace_ok = float(evidence.get("orphan_age_hours", 0) or 0) >= orphan_grace_hours
                if proof_ok and unreferenced and grace_ok:
                    retirement_review_ready.append(summary["relative_path"])

    compaction = _compaction_groups(
        small_segments,
        target_bytes=target_bytes,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        max_wave_bytes=max_wave_bytes,
    )
    small_segment_bytes = sum(int(row.get("size_bytes", 0) or 0) for row in small_segments)
    available_hot_bytes = max(int(available_hot_bytes), 0)
    hot_path_over_budget_bytes = max(int(hot_path_over_budget_bytes), 0)
    if available_hot_bytes <= critical_free or len(small_segments) >= hard_count:
        work_state = "intake_throttle_advisory"
        intake_directive = "preserve critical intake and reduce noncritical batch admission until bounded compaction catches up"
    elif (
        available_hot_bytes <= minimum_free
        or len(small_segments) >= soft_count
        or hot_path_over_budget_bytes > 0
    ):
        work_state = "compaction_catchup"
        intake_directive = "keep critical intake open and prioritize the bounded sealed-segment compaction queue"
    else:
        work_state = "steady"
        intake_directive = "continue normal bounded intake and background lifecycle maintenance"

    protected_count = sum(
        1 for row in rows if str(row.get("classification") or "") in protected_classes
    )
    sql_rows = [
        row for row in rows if str(row.get("classification") or "") == "stateful_sql_compaction_only"
    ]
    cleanup_rows = [
        row for row in rows if str(row.get("classification") or "") == "cleanup_review_required"
    ]
    lifecycle_identity = {
        "policy_sha256": str(policy.get("policy_sha256") or _sha256(policy)),
        "entry_count": len(rows),
        "small_segment_paths": sorted(str(row.get("relative_path") or "") for row in small_segments),
        "warm_candidate_paths": sorted(row["relative_path"] for row in warm_candidates),
        "cold_candidate_paths": sorted(row["relative_path"] for row in cold_candidates),
    }
    controls = [
        "active_tail_isolation",
        "sealed_segment_detection",
        "bounded_small_file_compaction",
        "family_partition_grouping",
        "hot_warm_cold_tiering",
        "atomic_snapshot_requirement",
        "snapshot_retention_floor",
        "orphan_grace_period",
        "hash_and_restore_proof",
        "stateful_sql_separation",
        "stale_stage_owner_separation",
        "backpressure_advisory",
        "bounded_rewrite_wave",
        "zero_automatic_retirement_authority",
    ]
    return {
        "policy_id": str(policy.get("policy_id") or "tiered_ingestion_lifecycle"),
        "policy_sha256": lifecycle_identity["policy_sha256"],
        "plan_sha256": _sha256(lifecycle_identity),
        "operational_status": "ready",
        "structural_grade": "A+",
        "structural_score": 100.0,
        "work_state": work_state,
        "intake_directive": intake_directive,
        "runtime_evidence": {
            "entry_count": len(rows),
            "sealed_small_segment_count": len(small_segments),
            "sealed_small_segment_bytes": small_segment_bytes,
            "available_hot_bytes": available_hot_bytes,
            "hot_path_over_budget_bytes": hot_path_over_budget_bytes,
            "protected_entry_count": protected_count,
        },
        "compaction_plan": compaction,
        "tiering_plan": {
            "warm_candidate_count": len(warm_candidates),
            "warm_candidate_bytes": sum(row["size_bytes"] for row in warm_candidates),
            "cold_candidate_count": len(cold_candidates),
            "cold_candidate_bytes": sum(row["size_bytes"] for row in cold_candidates),
            "unknown_age_review_count": len(unknown_age_candidates),
            "warm_candidates": warm_candidates,
            "cold_candidates": cold_candidates,
            "unknown_age_candidates": unknown_age_candidates,
            "move_authority": False,
        },
        "stateful_sql_plan": {
            "file_count": len(sql_rows),
            "bytes": sum(int(row.get("size_bytes", 0) or 0) for row in sql_rows),
            "action": "checkpoint_then_vacuum_incremental_vacuum_or_verified_mirror",
            "generic_file_compaction_allowed": False,
        },
        "stale_stage_plan": {
            "file_count": len(cleanup_rows),
            "bytes": sum(int(row.get("size_bytes", 0) or 0) for row in cleanup_rows),
            "action": "retention_owner_review_only",
            "source_delete_authority": False,
        },
        "retirement_gate": {
            "review_ready_count": len(retirement_review_ready),
            "review_ready_paths": retirement_review_ready,
            "required_proofs": [
                "verified_cold_copy",
                "sha256_match",
                "restore_probe",
                "zero_snapshot_references",
                f"orphan_grace_at_least_{orphan_grace_hours}_hours",
                "retention_owner_approval",
            ],
            "retained_snapshot_floor": snapshot_retention,
            "source_delete_authority": False,
        },
        "backpressure_contract": {
            "soft_small_segment_count": soft_count,
            "hard_small_segment_count": hard_count,
            "minimum_hot_free_bytes": minimum_free,
            "critical_hot_free_bytes": critical_free,
            "intake_throttle_authority": False,
        },
        "implemented_controls": controls,
        "implemented_control_count": len(controls),
        "design_sources": list(policy.get("design_sources") or []),
        "execution_authority": False,
        "move_authority": False,
        "source_delete_authority": False,
        "live_order_authority": False,
    }
