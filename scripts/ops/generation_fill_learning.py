#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import scripts.paper_performance_report as paper_report
    from scripts.ops.generation_behavior_attribution import (
        _candidate_windows,
        _decision_files,
        _read_events,
        _row_candidate_identity,
    )
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    import scripts.paper_performance_report as paper_report
    from .generation_behavior_attribution import (
        _candidate_windows,
        _decision_files,
        _read_events,
        _row_candidate_identity,
    )
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_CONFIG_PATH = Path("config/generation_fill_learning_v1.json")
DEFAULT_EVENT_PATH = Path("governance/evidence/production_candidate_events.jsonl")
DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_LINEAGE_PATH = Path("governance/health/training_lineage_manifest_latest.json")
DEFAULT_QUALITY_PATH = Path("governance/health/training_quality_control_latest.json")
DEFAULT_OUT_PATH = Path("governance/research/generation_fill_learning_latest.json")
DEFAULT_MARKDOWN_PATH = Path(
    "exports/reports/operator/generation_fill_learning_latest.md"
)
DEFAULT_DATASET_DIR = Path("governance/training/generation_fill_learning")

DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "contract_id": "generation_fill_learning_v1",
    "source_generation_policy": {
        "include_current_target_generation": True,
        "timestamp_only_generation_inference_allowed": False,
    },
    "fill_source_quality_multipliers": {
        "broker_paper_fill": 1.0,
        "observed_touch": 1.0,
        "paper_broker_fill": 1.0,
        "schwab_paper": 1.0,
        "market_replay_fill": 0.65,
        "expected_fill_model": 0.35,
        "unknown": 0.0,
    },
    "outcome_weight_policy": {
        "negative": 2.85,
        "positive": 1.0,
        "neutral": 0.5,
        "neutral_pnl_epsilon": 1e-9,
        "maximum_row_weight": 3.5,
        "receipt_recovered_provenance_multiplier": 0.9,
    },
    "recency_policy": {"half_life_days": 30.0, "minimum_multiplier": 0.15},
    "generation_balance_policy": {"maximum_generation_weight_share": 0.35},
    "validation_policy": {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "embargo_hours": 6.0,
        "minimum_rows_per_partition": 5,
        "minimum_rows_per_generation_fold": 5,
        "minimum_generation_folds": 2,
    },
    "challenger_policy": {
        "training_lineage_contract_required": True,
        "training_quality_ready_required": True,
    },
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _utc(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _merged_policy(raw: Mapping[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_POLICY))
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key].update(dict(value))
        else:
            merged[key] = value
    return merged


def _event_manifest(
    events: Sequence[Mapping[str, Any]], *, target_generation: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_candidate: dict[str, dict[str, Any]] = {}
    previous_candidate_id = ""
    for raw in events:
        generation = _safe_int(raw.get("generation"), 0)
        if generation <= 0 or generation > target_generation:
            continue
        scopes = sorted({str(item) for item in _as_list(raw.get("changed_scopes"))})
        row = {
            "candidate_id": str(raw.get("candidate_id") or ""),
            "generation": generation,
            "accepted_at_utc": str(raw.get("timestamp_utc") or ""),
            "event_type": str(raw.get("event_type") or ""),
            "change_reason": str(raw.get("change_reason") or ""),
            "changed_scopes": scopes,
            "changed_scope_count": len(scopes),
            "single_scope_change": len(scopes) == 1,
            "previous_candidate_id": previous_candidate_id,
            "accepted_git_head": str(raw.get("git_head") or ""),
            "candidate_sha256": str(raw.get("overall_sha256") or ""),
            "event_hash": str(raw.get("event_hash") or ""),
        }
        rows.append(row)
        if row["candidate_id"]:
            by_candidate[row["candidate_id"]] = row
            previous_candidate_id = row["candidate_id"]
    return rows, by_candidate


def _candidate_identity(row: Mapping[str, Any]) -> tuple[str, int, str]:
    metadata = _as_dict(row.get("metadata"))
    binding = _as_dict(row.get("candidate_binding"))
    candidate_id = str(
        metadata.get("production_candidate_id")
        or row.get("production_candidate_id")
        or binding.get("observed_candidate_id")
        or ""
    ).strip()
    generation = _safe_int(
        metadata.get("production_candidate_generation"),
        _safe_int(binding.get("generation"), 0),
    )
    receipt = str(
        metadata.get("production_candidate_receipt_sha256")
        or row.get("production_candidate_receipt_sha256")
        or ""
    ).strip()
    return candidate_id, generation, receipt


def _profile(row: Mapping[str, Any]) -> str:
    metadata = _as_dict(row.get("metadata"))
    return (
        str(
            metadata.get("source_profile")
            or row.get("paper_profile")
            or row.get("profile")
            or "default"
        )
        .strip()
        .lower()
        or "default"
    )


def _recovery_keys(row: Mapping[str, Any]) -> dict[str, set[str]]:
    metadata = _as_dict(row.get("metadata"))
    intent = _as_dict(row.get("order_intent_evidence"))
    semantic = _as_dict(intent.get("semantic_order"))
    decision_ids = {
        str(value).strip()
        for value in (
            row.get("decision_id"),
            metadata.get("decision_id"),
            semantic.get("decision_id"),
        )
        if str(value or "").strip()
    }
    parent_ids = {
        str(value).strip()
        for value in (
            row.get("parent_decision_id"),
            metadata.get("parent_decision_id"),
            row.get("parent_message_id"),
        )
        if str(value or "").strip()
    }
    message_ids = {
        str(value).strip()
        for value in (row.get("message_id"), metadata.get("message_id"))
        if str(value or "").strip()
    }
    snapshot_id = str(
        row.get("snapshot_id") or metadata.get("snapshot_id") or ""
    ).strip()
    snapshot_keys: set[str] = set()
    if snapshot_id:
        snapshot_keys.add(
            "|".join(
                (
                    snapshot_id,
                    str(row.get("symbol") or "").strip().upper(),
                    _profile(row),
                    str(row.get("action") or row.get("master_action") or "")
                    .strip()
                    .upper(),
                )
            )
        )
    return {
        "decision_id": decision_ids,
        "parent_decision_id": parent_ids,
        "message_id": message_ids,
        "snapshot_id": {snapshot_id} if snapshot_id else set(),
        "snapshot_composite": snapshot_keys,
    }


def _iter_paper_rows(
    files: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = Counter()
    seen: set[str] = set()
    for path in files:
        opener = gzip.open if ".jsonl.gz" in path.name else Path.open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    try:
                        row = json.loads(raw)
                    except (TypeError, ValueError):
                        counters["invalid_json_rows"] += 1
                        continue
                    if not isinstance(row, dict):
                        continue
                    counters["records_read"] += 1
                    if paper_report._calibration_only_paper_row_reason(row):
                        counters["calibration_only_rows_excluded"] += 1
                        continue
                    if paper_report._pnl_schema_version(row) < 2:
                        counters["pre_flow_schema_rows_excluded"] += 1
                        continue
                    if "post_cost_pnl_delta" not in row:
                        counters["missing_post_cost_outcome_rows_excluded"] += 1
                        continue
                    identity = paper_report._paper_execution_identity(row)
                    if identity in seen:
                        counters["mirrored_rows_suppressed"] += 1
                        continue
                    seen.add(identity)
                    rows.append(
                        {
                            "row": row,
                            "source_path": str(path),
                            "source_record_id": identity,
                            "source_row_sha256": _canonical_hash(row),
                        }
                    )
                    counters["flow_rows_accepted"] += 1
        except (OSError, EOFError, gzip.BadGzipFile):
            counters["unreadable_files"] += 1
    return rows, dict(sorted(counters.items()))


def _window_valid(
    candidate_id: str,
    generation: int,
    timestamp: datetime | None,
    windows: Mapping[str, Mapping[str, Any]],
) -> bool:
    window = _as_dict(windows.get(candidate_id))
    started = _utc(window.get("started_utc"))
    ended = _utc(window.get("ended_utc"))
    return bool(
        candidate_id
        and generation > 0
        and timestamp is not None
        and _safe_int(window.get("generation"), 0) == generation
        and started is not None
        and ended is not None
        and started <= timestamp < ended
    )


def _decision_receipt_index(
    files: Sequence[Path],
    *,
    needed: Mapping[str, set[str]],
    windows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, int]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counters = Counter()
    needed_snapshot_ids = needed.get("snapshot_id", set())
    snapshot_pattern = re.compile(r'"snapshot_id"\s*:\s*"([^"]+)"')
    timestamp_pattern = re.compile(r'"timestamp_utc"\s*:\s*"([^"]+)"')
    candidate_pattern = re.compile(r'"production_candidate_id"\s*:\s*"([^"]+)"')
    generation_pattern = re.compile(r'"production_candidate_generation"\s*:\s*(\d+)')
    for path in files:
        opener = gzip.open if path.name.endswith(".gz") else Path.open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    counters["decision_records_read"] += 1
                    snapshot_match = snapshot_pattern.search(raw, 0, 4096)
                    snapshot_id = (
                        snapshot_match.group(1) if snapshot_match is not None else ""
                    )
                    if len(raw) > 1_000_000:
                        if snapshot_id not in needed_snapshot_ids:
                            counters["large_decision_rows_fast_skipped"] += 1
                            continue
                        tail = raw[-262_144:]
                        candidate_ids = set(candidate_pattern.findall(tail))
                        generations = {
                            _safe_int(value, 0)
                            for value in generation_pattern.findall(tail)
                            if _safe_int(value, 0) > 0
                        }
                        timestamp_match = timestamp_pattern.search(raw, 0, 1024)
                        timestamp = (
                            _utc(timestamp_match.group(1))
                            if timestamp_match is not None
                            else None
                        )
                        if len(candidate_ids) == 1 and len(generations) == 1:
                            candidate_id = next(iter(candidate_ids))
                            generation = next(iter(generations))
                            if _window_valid(
                                candidate_id, generation, timestamp, windows
                            ):
                                index[("snapshot_id", snapshot_id)].append(
                                    {
                                        "candidate_id": candidate_id,
                                        "generation": generation,
                                        "receipt_kind": "snapshot_id",
                                        "decision_row_sha256": hashlib.sha256(
                                            raw.encode("utf-8")
                                        ).hexdigest(),
                                    }
                                )
                                counters[
                                    "large_decision_receipt_rows_fast_matched"
                                ] += 1
                                continue
                        counters["large_matched_rows_without_unique_tail_identity"] += 1
                        continue
                    try:
                        row = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(row, dict):
                        continue
                    candidate_id, generation = _row_candidate_identity(row)
                    timestamp = _utc(row.get("timestamp_utc"))
                    if not _window_valid(candidate_id, generation, timestamp, windows):
                        counters["decision_records_without_verified_identity"] += 1
                        continue
                    matched = False
                    for kind, values in _recovery_keys(row).items():
                        for value in values.intersection(needed.get(kind, set())):
                            index[(kind, value)].append(
                                {
                                    "candidate_id": candidate_id,
                                    "generation": generation,
                                    "receipt_kind": kind,
                                    "decision_row_sha256": _canonical_hash(row),
                                }
                            )
                            matched = True
                    if matched:
                        counters["decision_receipt_rows_matched"] += 1
        except (OSError, EOFError, gzip.BadGzipFile):
            counters["decision_files_unreadable"] += 1
    counters["decision_receipt_key_count"] = len(index)
    return index, dict(sorted(counters.items()))


def _resolve_identity(
    row: Mapping[str, Any],
    *,
    receipt_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    windows: Mapping[str, Mapping[str, Any]],
    event_by_candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    timestamp = _utc(row.get("timestamp_utc") or row.get("timestamp"))
    candidate_id, generation, receipt = _candidate_identity(row)
    if candidate_id:
        event = _as_dict(event_by_candidate.get(candidate_id))
        event_generation = _safe_int(event.get("generation"), 0)
        method = "direct_candidate_metadata"
        if generation <= 0 and event_generation > 0:
            generation = event_generation
            method = "candidate_id_event_chain_recovered"
        if event_generation and generation != event_generation:
            return {
                "status": "identity_conflict",
                "reason": "candidate_generation_mismatch",
                "candidate_id": candidate_id,
                "generation": generation,
                "binding_method": method,
            }
        if not _window_valid(candidate_id, generation, timestamp, windows):
            return {
                "status": "identity_conflict",
                "reason": "candidate_identity_outside_accepted_window",
                "candidate_id": candidate_id,
                "generation": generation,
                "binding_method": method,
            }
        return {
            "status": "verified",
            "reason": "",
            "candidate_id": candidate_id,
            "generation": generation,
            "binding_method": method,
            "candidate_receipt_sha256": receipt
            or str(event.get("candidate_sha256") or ""),
            "decision_receipt_sha256": "",
        }

    matches: list[dict[str, Any]] = []
    matched_kinds: set[str] = set()
    for kind, values in _recovery_keys(row).items():
        for value in values:
            for raw in receipt_index.get((kind, value), []):
                matches.append(dict(raw))
                matched_kinds.add(kind)
    identities = {
        (str(match.get("candidate_id") or ""), _safe_int(match.get("generation"), 0))
        for match in matches
    }
    if len(identities) > 1:
        return {
            "status": "identity_conflict",
            "reason": "exact_receipts_disagree_on_candidate_identity",
            "candidate_id": "",
            "generation": 0,
            "binding_method": "conflicting_exact_receipts",
        }
    if len(identities) == 1:
        recovered_id, recovered_generation = next(iter(identities))
        if _window_valid(recovered_id, recovered_generation, timestamp, windows):
            hashes = sorted(
                {
                    str(match.get("decision_row_sha256") or "")
                    for match in matches
                    if str(match.get("decision_row_sha256") or "")
                }
            )
            return {
                "status": "verified",
                "reason": "",
                "candidate_id": recovered_id,
                "generation": recovered_generation,
                "binding_method": "receipt_recovered:"
                + ",".join(sorted(matched_kinds)),
                "candidate_receipt_sha256": str(
                    _as_dict(event_by_candidate.get(recovered_id)).get(
                        "candidate_sha256"
                    )
                    or ""
                ),
                "decision_receipt_sha256": _canonical_hash(hashes),
            }
        return {
            "status": "identity_conflict",
            "reason": "receipt_identity_outside_accepted_window",
            "candidate_id": recovered_id,
            "generation": recovered_generation,
            "binding_method": "receipt_recovered_outside_window",
        }
    return {
        "status": "legacy_unbound",
        "reason": "no_exact_candidate_or_decision_receipt;timestamp_only_binding_forbidden",
        "candidate_id": "",
        "generation": 0,
        "binding_method": "none",
    }


def _fill_source(row: Mapping[str, Any]) -> str:
    metadata = _as_dict(row.get("metadata"))
    return (
        str(
            row.get("paper_fill_source")
            or metadata.get("paper_fill_source")
            or "unknown"
        )
        .strip()
        .lower()
        or "unknown"
    )


def _outcome_label(pnl: float, *, epsilon: float) -> str:
    if pnl > epsilon:
        return "positive"
    if pnl < -epsilon:
        return "negative"
    return "neutral"


def _compact_learning_row(
    source: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    target: Mapping[str, Any],
    event: Mapping[str, Any],
    policy: Mapping[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    row = _as_dict(source.get("row"))
    metadata = _as_dict(row.get("metadata"))
    entry_policy = _as_dict(metadata.get("entry_policy"))
    intent = _as_dict(row.get("order_intent_evidence"))
    component_hashes = _as_dict(intent.get("component_hashes"))
    outcome_policy = _as_dict(policy.get("outcome_weight_policy"))
    recency_policy = _as_dict(policy.get("recency_policy"))
    source_multipliers = _as_dict(policy.get("fill_source_quality_multipliers"))
    timestamp = _utc(row.get("timestamp_utc") or row.get("timestamp"))
    pnl = _safe_float(row.get("post_cost_pnl_delta"), 0.0)
    label = _outcome_label(
        pnl,
        epsilon=max(_safe_float(outcome_policy.get("neutral_pnl_epsilon"), 1e-9), 0.0),
    )
    age_days = (
        max((generated_at - timestamp).total_seconds() / 86400.0, 0.0)
        if timestamp
        else 0.0
    )
    half_life = max(_safe_float(recency_policy.get("half_life_days"), 30.0), 0.001)
    recency_multiplier = max(
        _safe_float(recency_policy.get("minimum_multiplier"), 0.15),
        math.pow(0.5, age_days / half_life),
    )
    fill_source = _fill_source(row)
    source_multiplier = max(
        _safe_float(
            source_multipliers.get(fill_source, source_multipliers.get("unknown", 0.0)),
            0.0,
        ),
        0.0,
    )
    recovered = str(identity.get("binding_method") or "").startswith(
        "receipt_recovered"
    )
    provenance_multiplier = (
        _safe_float(outcome_policy.get("receipt_recovered_provenance_multiplier"), 0.9)
        if recovered
        else 1.0
    )
    base_weight = max(_safe_float(outcome_policy.get(label), 1.0), 0.0)
    raw_weight = min(
        base_weight * source_multiplier * recency_multiplier * provenance_multiplier,
        max(_safe_float(outcome_policy.get("maximum_row_weight"), 3.5), 0.0),
    )
    empirical = fill_source in {
        "broker_paper_fill",
        "observed_touch",
        "paper_broker_fill",
        "schwab_paper",
    }
    regime = (
        str(
            row.get("institutional_decision_flow_regime_state")
            or metadata.get("regime")
            or entry_policy.get("profile_family")
            or row.get("spread_regime")
            or "unknown"
        )
        .strip()
        .lower()
    )
    keys = _recovery_keys(row)
    return {
        "schema_version": 1,
        "learning_target": {
            "candidate_id": str(target.get("candidate_id") or ""),
            "generation": _safe_int(target.get("generation"), 0),
            "accepted_at_utc": str(target.get("accepted_at_utc") or ""),
            "role": "offline_challenger_learning_owner",
        },
        "source_provenance": {
            "candidate_id": str(identity.get("candidate_id") or ""),
            "generation": _safe_int(identity.get("generation"), 0),
            "binding_method": str(identity.get("binding_method") or ""),
            "source_record_id": str(source.get("source_record_id") or ""),
            "source_row_sha256": str(source.get("source_row_sha256") or ""),
            "candidate_receipt_sha256": str(
                identity.get("candidate_receipt_sha256") or ""
            ),
            "decision_receipt_sha256": str(
                identity.get("decision_receipt_sha256") or ""
            ),
            "source_change_event_hash": str(event.get("event_hash") or ""),
            "source_change_scopes": list(event.get("changed_scopes") or []),
            "source_path_sha256": _canonical_hash(str(source.get("source_path") or "")),
        },
        "identity": {
            "timestamp_utc": timestamp.isoformat() if timestamp else "",
            "decision_ids": sorted(keys.get("decision_id", set())),
            "parent_decision_ids": sorted(keys.get("parent_decision_id", set())),
            "message_ids": sorted(keys.get("message_id", set())),
            "execution_id": str(row.get("execution_id") or row.get("fill_id") or ""),
            "entry_id": str(row.get("entry_id") or row.get("parent_decision_id") or ""),
            "exit_id": str(row.get("exit_id") or row.get("close_execution_id") or ""),
        },
        "routing": {
            "profile": _profile(row),
            "sleeve": str(metadata.get("sleeve") or _profile(row)).strip().lower(),
            "strategy": paper_report._strategy_of(row),
            "symbol": str(row.get("symbol") or "").strip().upper(),
            "action": str(row.get("action") or "").strip().upper(),
            "asset_class": str(row.get("asset_class") or "").strip().lower(),
            "routing_lane": str(row.get("routing_lane") or "").strip().lower(),
            "regime": regime or "unknown",
        },
        "point_in_time_features": {
            "model_score": row.get("model_score"),
            "decision_threshold": row.get("threshold"),
            "regime_fit_norm": entry_policy.get("regime_fit_norm"),
            "evidence_quality_norm": entry_policy.get("evidence_quality_norm"),
            "risk_multiplier_norm": entry_policy.get("risk_multiplier_norm"),
            "overlap_pressure_norm": entry_policy.get("overlap_pressure_norm"),
            "conflict_pressure_norm": entry_policy.get("conflict_pressure_norm"),
            "model_spread_bps": row.get("model_spread_bps"),
            "model_latency_ms": row.get("model_latency_ms"),
            "source_quality_score": row.get("source_quality_score"),
            "event_proximity_norm": row.get("event_proximity_norm"),
            "allocation_conflict_norm": row.get("allocation_conflict_norm"),
        },
        "model_and_policy_receipts": {
            "intent_sha256": str(intent.get("intent_sha256") or ""),
            "semantic_order_sha256": str(
                component_hashes.get("semantic_order_sha256") or ""
            ),
            "quote_snapshot_sha256": str(
                component_hashes.get("quote_snapshot_sha256") or ""
            ),
            "expected_fill_sha256": str(
                component_hashes.get("expected_fill_sha256") or ""
            ),
            "risk_decision_sha256": str(
                component_hashes.get("risk_decision_sha256") or ""
            ),
            "decision_playbook_sha256": str(
                row.get("institutional_decision_flow_playbook_sha256") or ""
            ),
            "paper_valuation_policy_version": str(
                row.get("paper_valuation_policy_version") or ""
            ),
        },
        "label": {
            "outcome": label,
            "post_cost_pnl_delta": round(pnl, 10),
            "post_cost_return_bps": row.get("post_cost_return_bps"),
            "realized_pnl_delta": row.get("realized_pnl_delta"),
            "execution_cost_amount": row.get("expected_execution_cost_amount"),
            "maximum_adverse_excursion": row.get("maximum_adverse_excursion")
            or row.get("mae"),
            "maximum_favorable_excursion": row.get("maximum_favorable_excursion")
            or row.get("mfe"),
            "fill_source": fill_source,
            "empirical_fill": empirical,
            "simulation_pretraining_only": not empirical,
        },
        "weighting": {
            "base_outcome_multiplier": round(base_weight, 8),
            "fill_source_quality_multiplier": round(source_multiplier, 8),
            "recency_multiplier": round(recency_multiplier, 8),
            "provenance_multiplier": round(provenance_multiplier, 8),
            "raw_weight": round(raw_weight, 8),
            "generation_balance_multiplier": 1.0,
            "sample_weight": round(raw_weight, 8),
        },
        "eligibility": {
            "developmental_pretraining_eligible": raw_weight > 0.0,
            "empirical_outcome_training_eligible": bool(empirical and raw_weight > 0.0),
            "promotion_grade_eligible": False,
            "current_generation_forward_evidence": False,
            "live_execution_authority": False,
        },
        "validation_partition": "unassigned",
    }


def _balanced_generation_weights(
    rows: list[dict[str, Any]], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _as_dict(row.get("eligibility")).get(
            "developmental_pretraining_eligible", False
        ):
            continue
        generation = _safe_int(
            _as_dict(row.get("source_provenance")).get("generation"), 0
        )
        if generation > 0:
            grouped[generation].append(row)
    if not grouped:
        return {
            "generation_count": 0,
            "maximum_generation_weight_share": 0.0,
            "generation_budgets": [],
        }
    balance = _as_dict(policy.get("generation_balance_policy"))
    configured_cap = min(
        max(
            _safe_float(balance.get("maximum_generation_weight_share"), 0.35),
            0.01,
        ),
        1.0,
    )
    effective_cap = max(configured_cap, 1.0 / len(grouped))
    scores = {}
    for generation, items in grouped.items():
        mean_recency = sum(
            _safe_float(_as_dict(item.get("weighting")).get("recency_multiplier"), 0.0)
            for item in items
        ) / max(len(items), 1)
        scores[generation] = math.sqrt(len(items)) * mean_recency
    remaining = set(scores)
    shares: dict[int, float] = {}
    remaining_share = 1.0
    while remaining:
        score_total = sum(scores[generation] for generation in remaining)
        provisional = {
            generation: (
                remaining_share * scores[generation] / score_total
                if score_total > 0.0
                else remaining_share / len(remaining)
            )
            for generation in remaining
        }
        capped = [
            generation
            for generation, share in provisional.items()
            if share > effective_cap + 1e-12
        ]
        if not capped:
            shares.update(provisional)
            break
        for generation in capped:
            shares[generation] = effective_cap
            remaining.remove(generation)
            remaining_share -= effective_cap
        if remaining_share <= 1e-12:
            shares.update({generation: 0.0 for generation in remaining})
            break
    eligible_count = sum(len(items) for items in grouped.values())
    budget_rows: list[dict[str, Any]] = []
    for generation, items in sorted(grouped.items()):
        raw_total = sum(
            _safe_float(_as_dict(item.get("weighting")).get("raw_weight"), 0.0)
            for item in items
        )
        target_total = shares.get(generation, 0.0) * eligible_count
        multiplier = target_total / raw_total if raw_total > 0.0 else 0.0
        for item in items:
            weighting = _as_dict(item.get("weighting"))
            sample_weight = _safe_float(weighting.get("raw_weight"), 0.0) * multiplier
            weighting["generation_balance_multiplier"] = round(multiplier, 8)
            weighting["sample_weight"] = round(sample_weight, 8)
            item["weighting"] = weighting
        budget_rows.append(
            {
                "generation": generation,
                "sample_count": len(items),
                "raw_weight_total": round(raw_total, 8),
                "normalized_weight_share": round(shares.get(generation, 0.0), 8),
                "generation_balance_multiplier": round(multiplier, 8),
            }
        )
    return {
        "generation_count": len(grouped),
        "configured_maximum_generation_weight_share": configured_cap,
        "effective_maximum_generation_weight_share": round(effective_cap, 8),
        "maximum_generation_weight_share": round(max(shares.values()), 8),
        "generation_budgets": budget_rows,
    }


def _validation_plan(
    rows: list[dict[str, Any]], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    validation = _as_dict(policy.get("validation_policy"))
    eligible = [
        row
        for row in rows
        if _as_dict(row.get("eligibility")).get(
            "developmental_pretraining_eligible", False
        )
        and _utc(_as_dict(row.get("identity")).get("timestamp_utc")) is not None
    ]
    eligible.sort(
        key=lambda row: _utc(_as_dict(row.get("identity")).get("timestamp_utc"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )
    n = len(eligible)
    train_fraction = min(
        max(_safe_float(validation.get("train_fraction"), 0.6), 0.0), 1.0
    )
    validation_fraction = min(
        max(_safe_float(validation.get("validation_fraction"), 0.2), 0.0), 1.0
    )
    train_end = min(max(int(n * train_fraction), 0), n)
    validation_end = min(
        max(int(n * (train_fraction + validation_fraction)), train_end), n
    )
    for index, row in enumerate(eligible):
        row["validation_partition"] = (
            "chronological_train"
            if index < train_end
            else (
                "chronological_validation"
                if index < validation_end
                else "chronological_test"
            )
        )
    embargo_hours = max(_safe_float(validation.get("embargo_hours"), 6.0), 0.0)
    boundaries: list[datetime] = []
    for index in (train_end, validation_end):
        if 0 < index < n:
            timestamp = _utc(
                _as_dict(eligible[index].get("identity")).get("timestamp_utc")
            )
            if timestamp:
                boundaries.append(timestamp)
    for row in eligible:
        timestamp = _utc(_as_dict(row.get("identity")).get("timestamp_utc"))
        if timestamp and any(
            abs((timestamp - boundary).total_seconds()) <= embargo_hours * 3600.0
            for boundary in boundaries
        ):
            row["validation_partition"] = "purged_embargo"
    partition_counts = Counter(
        str(row.get("validation_partition") or "") for row in eligible
    )
    generation_counts = Counter(
        _safe_int(_as_dict(row.get("source_provenance")).get("generation"), 0)
        for row in eligible
    )
    minimum_fold_rows = max(
        _safe_int(validation.get("minimum_rows_per_generation_fold"), 5), 1
    )
    fold_generations = sorted(
        generation
        for generation, count in generation_counts.items()
        if generation > 0 and count >= minimum_fold_rows
    )
    folds = [
        {
            "held_out_generation": generation,
            "held_out_sample_count": generation_counts[generation],
            "training_generations": [
                other for other in fold_generations if other != generation
            ],
            "status": "ready" if len(fold_generations) > 1 else "collecting",
        }
        for generation in fold_generations
    ]
    minimum_partition = max(
        _safe_int(validation.get("minimum_rows_per_partition"), 5), 1
    )
    minimum_folds = max(_safe_int(validation.get("minimum_generation_folds"), 2), 1)
    chronological_ready = all(
        partition_counts.get(partition, 0) >= minimum_partition
        for partition in (
            "chronological_train",
            "chronological_validation",
            "chronological_test",
        )
    )
    generation_holdout_ready = len(folds) >= minimum_folds and all(
        fold.get("training_generations") for fold in folds
    )
    return {
        "status": (
            "ready"
            if chronological_ready and generation_holdout_ready
            else "collecting"
        ),
        "eligible_sample_count": n,
        "partition_counts": dict(sorted(partition_counts.items())),
        "chronological_split_ready": chronological_ready,
        "purged_embargo_hours": embargo_hours,
        "leave_one_generation_out_ready": generation_holdout_ready,
        "leave_one_generation_out_folds": folds,
        "regime_counts": dict(
            sorted(
                Counter(
                    str(_as_dict(row.get("routing")).get("regime") or "unknown")
                    for row in eligible
                ).items()
            )
        ),
        "point_in_time_split_only": True,
        "random_shuffle_split_allowed": False,
    }


def _training_gate(
    project_root: Path,
    *,
    target: Mapping[str, Any],
    validation: Mapping[str, Any],
    eligible_count: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = load_json(project_root / DEFAULT_LINEAGE_PATH)
    quality = load_json(project_root / DEFAULT_QUALITY_PATH)
    challenger = _as_dict(policy.get("challenger_policy"))
    lineage_ready = bool(
        lineage.get("lineage_contract_ready", False)
        and lineage.get("feature_store_lineage_ok", False)
        and lineage.get("exact_replay_ready", False)
        and lineage.get("hash_bundle_complete", False)
    )
    quality_ready = bool(
        quality.get("ok", False)
        and str(quality.get("overall_status") or "") not in {"blocked", "critical"}
    )
    blockers: list[str] = []
    if eligible_count <= 0:
        blockers.append("no_verified_weighted_historical_fill_rows")
    if str(validation.get("status") or "") != "ready":
        blockers.append("chronological_or_generation_holdout_validation_collecting")
    if challenger.get("training_lineage_contract_required", True) and not lineage_ready:
        blockers.append("training_lineage_contract_not_ready")
    if challenger.get("training_quality_ready_required", True) and not quality_ready:
        blockers.append("training_quality_control_not_ready")
    allowed = not blockers
    return {
        "status": "ready" if allowed else "blocked",
        "learning_target_candidate_id": str(target.get("candidate_id") or ""),
        "learning_target_generation": _safe_int(target.get("generation"), 0),
        "training_lineage_ready": lineage_ready,
        "training_lineage_score": lineage.get("lineage_score"),
        "training_lineage_missing_contracts": list(
            lineage.get("missing_contracts") or []
        ),
        "training_quality_ready": quality_ready,
        "training_quality_score": quality.get("training_quality_score"),
        "training_quality_status": str(quality.get("overall_status") or "missing"),
        "challenger_training_launch_allowed": allowed,
        "blockers": blockers,
        "offline_challenger_only": True,
        "automatic_runtime_swap_allowed": False,
        "automatic_promotion_allowed": False,
        "paper_order_authority": False,
        "live_execution_authority": False,
    }


def build_learning_bundle(
    project_root: Path = PROJECT_ROOT,
    *,
    target_generation: int | None = None,
    paper_files: Sequence[Path] | None = None,
    decision_files: Sequence[Path] | None = None,
    config_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = project_root.expanduser().resolve()
    now = _utc(generated_at_utc) or datetime.now(timezone.utc)
    policy = _merged_policy(load_json(config_path or root / DEFAULT_CONFIG_PATH))
    candidate = load_json(root / DEFAULT_CANDIDATE_PATH)
    current_generation = _safe_int(candidate.get("generation"), 0)
    requested_generation = target_generation or current_generation
    target_matches_current = bool(
        current_generation > 0 and requested_generation == current_generation
    )
    target = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "generation": current_generation,
        "accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
        "candidate_sha256": str(candidate.get("overall_sha256") or ""),
        "requested_generation": requested_generation,
        "requested_generation_matches_current_candidate": target_matches_current,
    }
    events, chain = _read_events(root / DEFAULT_EVENT_PATH)
    windows, by_candidate_window = _candidate_windows(events, now=now)
    manifest, event_by_candidate = _event_manifest(
        events, target_generation=requested_generation
    )
    chain_valid = bool(chain.get("ok", False) and events)

    if paper_files is None:
        (
            discovered_paper_files,
            source_kind,
            calibration_excluded,
        ) = paper_report._paper_source_files(root)
    else:
        discovered_paper_files = [Path(path) for path in paper_files]
        source_kind = "explicit_test_or_operator_source"
        calibration_excluded = 0
    present_paper_files = [path for path in discovered_paper_files if path.is_file()]
    missing_paper_paths = len(discovered_paper_files) - len(present_paper_files)
    source_rows, paper_scan = _iter_paper_rows(present_paper_files)

    needed: dict[str, set[str]] = defaultdict(set)
    for source in source_rows:
        row = _as_dict(source.get("row"))
        candidate_id, _, _ = _candidate_identity(row)
        if candidate_id:
            continue
        for kind, values in _recovery_keys(row).items():
            needed[kind].update(values)
    if decision_files is None:
        unbound_days = {
            timestamp.strftime("%Y%m%d")
            for source in source_rows
            if not _candidate_identity(_as_dict(source.get("row")))[0]
            and (
                timestamp := _utc(
                    _as_dict(source.get("row")).get("timestamp_utc")
                    or _as_dict(source.get("row")).get("timestamp")
                )
            )
            is not None
        }
        unbound_profiles = {
            _profile(_as_dict(source.get("row")))
            for source in source_rows
            if not _candidate_identity(_as_dict(source.get("row")))[0]
        }
        first_started = min(
            (
                timestamp
                for window in windows
                if (timestamp := _utc(window.get("started_utc"))) is not None
            ),
            default=now,
        )
        discovered_rows = _decision_files(root, start=first_started, end=now)
        discovered_decision_files = []
        for file_row in discovered_rows:
            path = Path(file_row["path"])
            if unbound_days and str(file_row.get("day") or "") not in unbound_days:
                continue
            path_text = str(path).lower()
            if unbound_profiles and not any(
                f"shadow_{profile}_" in path_text or f"shadow_{profile}/" in path_text
                for profile in unbound_profiles
            ):
                continue
            discovered_decision_files.append(path)
    else:
        discovered_decision_files = [Path(path) for path in decision_files]
    receipt_index, decision_scan = _decision_receipt_index(
        discovered_decision_files,
        needed=needed,
        windows=by_candidate_window,
    )

    learning_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    identity_counts = Counter()
    source_counts = Counter()
    include_current = bool(
        _as_dict(policy.get("source_generation_policy")).get(
            "include_current_target_generation", True
        )
    )
    for source in source_rows:
        row = _as_dict(source.get("row"))
        identity = _resolve_identity(
            row,
            receipt_index=receipt_index,
            windows=by_candidate_window,
            event_by_candidate=event_by_candidate,
        )
        identity_counts[str(identity.get("status") or "unknown")] += 1
        source_counts[_fill_source(row)] += 1
        generation = _safe_int(identity.get("generation"), 0)
        generation_allowed = bool(
            identity.get("status") == "verified"
            and generation > 0
            and (
                generation < requested_generation
                or (include_current and generation == requested_generation)
            )
        )
        if not generation_allowed:
            quarantine_rows.append(
                {
                    "schema_version": 1,
                    "learning_target_generation": requested_generation,
                    "source_record_id": str(source.get("source_record_id") or ""),
                    "source_row_sha256": str(source.get("source_row_sha256") or ""),
                    "source_path_sha256": _canonical_hash(
                        str(source.get("source_path") or "")
                    ),
                    "timestamp_utc": str(
                        row.get("timestamp_utc") or row.get("timestamp") or ""
                    ),
                    "candidate_id": str(identity.get("candidate_id") or ""),
                    "generation": generation,
                    "status": str(identity.get("status") or "legacy_unbound"),
                    "reason": str(identity.get("reason") or "generation_not_allowed"),
                    "binding_method": str(identity.get("binding_method") or "none"),
                    "timestamp_only_generation_inference_used": False,
                    "developmental_pretraining_eligible": False,
                    "promotion_grade_eligible": False,
                    "live_execution_authority": False,
                }
            )
            continue
        event = _as_dict(
            event_by_candidate.get(str(identity.get("candidate_id") or ""))
        )
        learning_rows.append(
            _compact_learning_row(
                source,
                identity=identity,
                target=target,
                event=event,
                policy=policy,
                generated_at=now,
            )
        )

    balance = _balanced_generation_weights(learning_rows, policy=policy)
    validation = _validation_plan(learning_rows, policy=policy)
    eligible_count = sum(
        1
        for row in learning_rows
        if _as_dict(row.get("eligibility")).get(
            "developmental_pretraining_eligible", False
        )
    )
    empirical_count = sum(
        1
        for row in learning_rows
        if _as_dict(row.get("eligibility")).get(
            "empirical_outcome_training_eligible", False
        )
    )
    gate = _training_gate(
        root,
        target=target,
        validation=validation,
        eligible_count=eligible_count,
        policy=policy,
    )
    if not chain_valid:
        gate["blockers"] = sorted(
            set(gate.get("blockers", [])) | {"candidate_event_chain_invalid"}
        )
        gate["challenger_training_launch_allowed"] = False
        gate["status"] = "blocked"
    if not target_matches_current:
        gate["blockers"] = sorted(
            set(gate.get("blockers", []))
            | {"requested_learning_target_is_not_current_candidate"}
        )
        gate["challenger_training_launch_allowed"] = False
        gate["status"] = "blocked"

    dataset_sha256 = _canonical_hash(learning_rows)
    policy_sha256 = _canonical_hash(policy)
    learning_run_id = f"g{requested_generation}-{dataset_sha256[:16]}"
    label_counts = Counter(
        str(_as_dict(row.get("label")).get("outcome") or "unknown")
        for row in learning_rows
    )
    generation_counts = Counter(
        _safe_int(_as_dict(row.get("source_provenance")).get("generation"), 0)
        for row in learning_rows
    )
    weighted_labels: dict[str, float] = defaultdict(float)
    for row in learning_rows:
        label = str(_as_dict(row.get("label")).get("outcome") or "unknown")
        weighted_labels[label] += _safe_float(
            _as_dict(row.get("weighting")).get("sample_weight"), 0.0
        )
    status = (
        "blocked"
        if not chain_valid or not target_matches_current
        else (
            "challenger_training_ready"
            if gate.get("challenger_training_launch_allowed", False)
            else (
                "dataset_ready_challenger_training_blocked"
                if eligible_count > 0
                else "collecting_verified_historical_fills"
            )
        )
    )
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": bool(chain_valid and target_matches_current),
        "overall_status": status,
        "learning_run_id": learning_run_id,
        "learning_target": target,
        "candidate_event_chain": {
            "valid": chain_valid,
            "event_count": _safe_int(chain.get("event_count"), 0),
            "chain_head": str(chain.get("chain_head") or ""),
            "errors": list(chain.get("errors") or []),
        },
        "dataset": {
            "dataset_sha256": dataset_sha256,
            "policy_sha256": policy_sha256,
            "verified_row_count": len(learning_rows),
            "developmental_pretraining_eligible_row_count": eligible_count,
            "empirical_outcome_training_eligible_row_count": empirical_count,
            "simulation_pretraining_only_row_count": (eligible_count - empirical_count),
            "quarantined_row_count": len(quarantine_rows),
            "source_generation_counts": {
                str(key): value for key, value in sorted(generation_counts.items())
            },
            "outcome_label_counts": dict(sorted(label_counts.items())),
            "weighted_outcome_label_totals": {
                key: round(value, 8) for key, value in sorted(weighted_labels.items())
            },
            "fill_source_counts": dict(sorted(source_counts.items())),
            "identity_status_counts": dict(sorted(identity_counts.items())),
            "compact_rows_contain_raw_future_pnl_as_predictive_feature": False,
        },
        "source_scan": {
            "paper_source_kind": source_kind,
            "paper_source_candidate_path_count": len(discovered_paper_files),
            "paper_file_count": len(present_paper_files),
            "paper_source_paths_not_present": missing_paper_paths,
            "decision_file_count": len(discovered_decision_files),
            "calibration_source_file_count_excluded": calibration_excluded,
            "paper_counters": paper_scan,
            "decision_receipt_counters": decision_scan,
        },
        "accepted_generation_change_manifest": {
            "generation_count": len(manifest),
            "first_generation": (
                _safe_int(manifest[0].get("generation"), 0) if manifest else 0
            ),
            "last_generation": (
                _safe_int(manifest[-1].get("generation"), 0) if manifest else 0
            ),
            "manifest_sha256": _canonical_hash(manifest),
            "single_scope_change_count": sum(
                1 for row in manifest if row.get("single_scope_change", False)
            ),
            "multi_scope_change_count": sum(
                1 for row in manifest if not row.get("single_scope_change", False)
            ),
            "rows": manifest,
            "causal_treatment_claim_requires_isolated_single_change": True,
            "multi_change_generation_comparisons_are_associational_only": True,
        },
        "generation_weight_balance": balance,
        "validation": validation,
        "challenger_training_gate": gate,
        "implementation_controls": [
            "recover generation identity only from candidate metadata or exact decision receipts",
            "retain unmatched history in a zero-authority legacy_unbound quarantine ledger",
            "preserve immutable fill, candidate, decision, model, policy, cost, and outcome receipts",
            "join every verified source generation to its accepted change manifest",
            "weight losses as hard negatives and profitable outcomes as positives with bounded source quality",
            "apply recency decay and generation balance so stale or dominant history cannot overwhelm G104",
            "use purged chronological and leave-one-generation-out validation without random shuffling",
            "route only to an offline G104 challenger behind lineage and training-quality gates",
        ],
        "policy": {
            "learning_outputs_attributed_to_generation": requested_generation,
            "historical_source_fills_reattributed_to_generation_104": False,
            "source_generations_preserved": True,
            "timestamp_only_generation_inference_allowed": False,
            "legacy_unbound_rows_train_models": False,
            "simulation_rows_establish_empirical_profitability": False,
            "historical_rows_grade_g104_forward_profitability": False,
            "historical_rows_earn_g104_clean_soak_credit": False,
            "source_generation_is_predictive_feature": False,
            "association_is_causal_proof": False,
            "profitability_guaranteed": False,
            "automatic_runtime_swap_allowed": False,
            "automatic_promotion_allowed": False,
            "paper_order_authority": False,
            "live_execution_authority": False,
        },
        "materialization": {
            "dataset_relative_path": str(
                DEFAULT_DATASET_DIR / f"g{requested_generation}_learning_rows.jsonl"
            ),
            "quarantine_relative_path": str(
                DEFAULT_DATASET_DIR / f"g{requested_generation}_quarantine_rows.jsonl"
            ),
            "apply_required": True,
        },
    }
    return payload, learning_rows, quarantine_rows


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def render_markdown(payload: Mapping[str, Any]) -> str:
    target = _as_dict(payload.get("learning_target"))
    dataset = _as_dict(payload.get("dataset"))
    gate = _as_dict(payload.get("challenger_training_gate"))
    validation = _as_dict(payload.get("validation"))
    lines = [
        "# Generation Fill Learning",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        f"Learning owner: `G{target.get('generation', 0)}` / `{target.get('candidate_id', '')}`",
        "",
        "## Dataset",
        "",
        f"- Verified rows: `{dataset.get('verified_row_count', 0)}`",
        f"- Developmental pretraining rows: `{dataset.get('developmental_pretraining_eligible_row_count', 0)}`",
        f"- Empirical outcome rows: `{dataset.get('empirical_outcome_training_eligible_row_count', 0)}`",
        f"- Simulation-only rows: `{dataset.get('simulation_pretraining_only_row_count', 0)}`",
        f"- Quarantined rows: `{dataset.get('quarantined_row_count', 0)}`",
        f"- Dataset SHA-256: `{dataset.get('dataset_sha256', '')}`",
        "",
        "## Validation And Launch",
        "",
        f"- Chronological split ready: `{validation.get('chronological_split_ready', False)}`",
        f"- Leave-one-generation-out ready: `{validation.get('leave_one_generation_out_ready', False)}`",
        f"- Training lineage ready: `{gate.get('training_lineage_ready', False)}`",
        f"- Training quality ready: `{gate.get('training_quality_ready', False)}`",
        f"- G104 challenger launch allowed: `{gate.get('challenger_training_launch_allowed', False)}`",
    ]
    blockers = list(gate.get("blockers") or [])
    if blockers:
        lines.append(f"- Blockers: `{', '.join(str(item) for item in blockers)}`")
    lines.extend(
        [
            "",
            "## Provenance Boundary",
            "",
            "The learning run belongs to the current candidate generation. Every historical fill retains its verified source generation. Unbound rows are quarantined, timestamp-only generation inference is forbidden, simulated fills cannot establish empirical profitability, and no historical row grades G104 forward performance or clean-soak credit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a provenance-preserving historical paper-fill learning lane for the current candidate."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--target-generation", type=int)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    payload, learning_rows, quarantine_rows = build_learning_bundle(
        root,
        target_generation=args.target_generation,
        config_path=config_path,
    )
    out_path = Path(args.out_file).expanduser()
    markdown_path = Path(args.markdown_file).expanduser()
    dataset_dir = Path(args.dataset_dir).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    if not markdown_path.is_absolute():
        markdown_path = root / markdown_path
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    if args.apply:
        generation = _safe_int(
            _as_dict(payload.get("learning_target")).get("requested_generation"),
            _safe_int(_as_dict(payload.get("learning_target")).get("generation"), 0),
        )
        dataset_path = dataset_dir / f"g{generation}_learning_rows.jsonl"
        quarantine_path = dataset_dir / f"g{generation}_quarantine_rows.jsonl"
        _write_jsonl_atomic(dataset_path, learning_rows)
        _write_jsonl_atomic(quarantine_path, quarantine_rows)
        payload["materialization"] = {
            **_as_dict(payload.get("materialization")),
            "applied": True,
            "dataset_path": str(dataset_path),
            "quarantine_path": str(quarantine_path),
            "dataset_row_count": len(learning_rows),
            "quarantine_row_count": len(quarantine_rows),
        }
    else:
        payload["materialization"] = {
            **_as_dict(payload.get("materialization")),
            "applied": False,
        }
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        dataset = _as_dict(payload.get("dataset"))
        gate = _as_dict(payload.get("challenger_training_gate"))
        print(
            "generation_fill_learning "
            f"status={payload.get('overall_status', '')} "
            f"target=G{_as_dict(payload.get('learning_target')).get('generation', 0)} "
            f"verified={dataset.get('verified_row_count', 0)} "
            f"quarantined={dataset.get('quarantined_row_count', 0)} "
            f"challenger_ready={gate.get('challenger_training_launch_allowed', False)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
