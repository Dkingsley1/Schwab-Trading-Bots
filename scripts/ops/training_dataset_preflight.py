#!/usr/bin/env python3
"""Materialize bounded research datasets without entering a model trainer."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

if __name__ == "__main__":
    os.environ["BOT_MLX_DISABLE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "core"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.crypto_runtime_bot_common import (
    _runtime_confidence, _runtime_feature_vector, _runtime_label, _runtime_sample_filter,
)
from core.runtime_training_common import _load_runtime_snapshot_rows, make_runtime_windowed_dataset
from core.training_diagnostic_contract import diagnostic_age_hours, materialization_contract_valid
from scripts.ops.long_runtime_common import load_json, write_payload

SUPPORTED_MODULES = (
    "brain_refinery_v257_crypto_spot_momentum_regime_bot",
    "brain_refinery_v258_crypto_perp_funding_squeeze_detector",
    "brain_refinery_v259_crypto_etf_tradfi_flow_bridge",
    "brain_refinery_v260_crypto_stablecoin_liquidity_impulse_bot",
    "brain_refinery_v261_crypto_eth_gas_defi_activity_guard",
    "brain_refinery_v262_crypto_solana_high_beta_rotation_bot",
    "brain_refinery_v263_crypto_exchange_status_latency_guard",
    "brain_refinery_v264_crypto_cross_exchange_divergence_arbitrage_bot",
    "brain_refinery_v265_crypto_risk_off_contagion_shock_guard",
    "brain_refinery_v266_crypto_weekend_gap_liquidity_bot",
)
EXPLICIT_RUNTIME_MODULES = (
    "brain_refinery_v35_dmi_state_machine",
    "brain_refinery_v100_stock_crypto_overlap_context",
    "brain_refinery_v103_crypto_throttle_relief_momentum",
)
AUTHORITY = {
    "training_launch_authority": False,
    "promotion_authority": False,
    "paper_execution_authority": False,
    "live_execution_authority": False,
}


def preparation_spec(bot_id: str):
    if bot_id not in (*SUPPORTED_MODULES, *EXPLICIT_RUNTIME_MODULES):
        raise ValueError("unsupported_research_bot")
    module = importlib.import_module("core." + bot_id)
    if bot_id in SUPPORTED_MODULES:
        return module.SPEC
    options = module.runtime_training_options()
    if options["run_tag"] != bot_id or options.get("allow_fallback_on_insufficient_data") is not False:
        raise ValueError("incompatible_research_training_options")
    return SimpleNamespace(
        bot_id=bot_id, feature_names=options["feature_names"],
        feature_fields=options["feature_names"], mode_allowlist=options["mode_allowlist"],
        symbol_allowlist=options.get("symbol_allowlist"), sample_stride=options.get("sample_stride", 1),
        window=options["window"], horizon=options["horizon"],
        lookback_days=options["lookback_days"], min_confidence=options["min_confidence"],
        min_samples=options["min_samples"], min_sequences=options["min_sequences"],
        min_positive_samples=options.get("min_positive_samples", 0),
        min_negative_samples=options.get("min_negative_samples", 0),
        feature_builder=options["runtime_feature_builder"], label_builder=options["runtime_label_builder"],
        sample_filter=options["sample_filter"], confidence_builder=options["confidence_builder"],
    )


def snapshot_feature_fields(bot_ids=None) -> list[str]:
    fields = {
        "data_quality_market_data_latency_norm",
        "data_quality_quote_deviation_norm",
        "fx_corr_confidence_norm",
        "market_crypto_corr_confidence_norm",
        "market_micro_trend_persistence_norm",
    }
    selected = list(SUPPORTED_MODULES if bot_ids is None else bot_ids)
    for module_name in selected:
        spec = preparation_spec(module_name)
        fields.update(spec.feature_fields)
    # Keep gate/label dependencies too, including fields absent from X.
    for module_name in (*selected, "crypto_runtime_bot_common", "runtime_requested_bot_common"):
        source = PROJECT_ROOT / "core" / f"{module_name}.py"
        for node in ast.walk(ast.parse(source.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"observation_feature", "feature_std", "feature_ema"}):
                fields.update(arg.value for arg in node.args
                              if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    return sorted(fields)


def _epoch(raw: str) -> float:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("missing_timestamp_timezone")
    return parsed.timestamp()


def purged_split_evidence(evidence: list[dict], embargo_minutes: int) -> list[dict]:
    """Keep the entire feature-to-outcome interval inside one time partition."""
    if not evidence:
        return []
    times = sorted(_epoch(row["feature_timestamp_utc"]) for row in evidence)
    boundary_one = times[min(int(len(times) * 0.60), len(times) - 1)]
    boundary_two = times[min(int(len(times) * 0.80), len(times) - 1)]
    embargo_seconds = max(embargo_minutes, 0) * 60
    result = []
    for row in evidence:
        start = _epoch(row["feature_window_started_at_utc"])
        anchor = _epoch(row["feature_timestamp_utc"])
        end = _epoch(row["label_matured_at_utc"])
        split = "purged"
        if start <= anchor < end:
            if end < boundary_one - embargo_seconds:
                split = "train"
            elif start >= boundary_one and end < boundary_two - embargo_seconds:
                split = "validation"
            elif start >= boundary_two:
                split = "test"
        result.append({
            **row, "train_validation_test_split": split, "embargo_minutes": embargo_minutes,
            "sample_eligibility_reason": "feature_or_outcome_interval_crosses_split_boundary" if split == "purged" else "point_in_time_interval_inside_partition",
        })
    return result


def prepare_dataset(spec, sequences: dict, contract: dict, split_policy: dict) -> tuple:
    if contract.get("objective_class") != "market_outcome" or not materialization_contract_valid(contract, spec.bot_id):
        raise ValueError("missing_or_incompatible_materialization_contract")
    symbols = getattr(spec, "symbol_allowlist", None)
    selected = {key: rows for key, rows in sequences.items()
                if key[0] in spec.mode_allowlist and (not symbols or key[1] in symbols)}
    X, y, meta = make_runtime_windowed_dataset(
        sequences=selected,
        feature_builder=getattr(spec, "feature_builder", None) or (lambda rows, idx: _runtime_feature_vector(spec, rows, idx)),
        label_builder=getattr(spec, "label_builder", None) or (lambda rows, idx, horizon: _runtime_label(spec, rows, idx, horizon)),
        label_contract=contract,
        label_owner_id=spec.bot_id,
        sample_filter=getattr(spec, "sample_filter", None) or (lambda rows, idx, horizon: _runtime_sample_filter(spec, rows, idx, horizon)),
        confidence_builder=getattr(spec, "confidence_builder", None) or (lambda rows, idx, horizon: _runtime_confidence(spec, rows, idx, horizon)),
        min_confidence=spec.min_confidence,
        window=spec.window,
        horizon=spec.horizon,
        sample_stride=getattr(spec, "sample_stride", 1),
        max_samples=2000,
        include_sample_evidence=True,
        max_rejection_evidence=100,
        balance_samples=False,
    )
    evidence = purged_split_evidence(meta.pop("sample_evidence"), int(split_policy.get("embargo_minutes", 390)))
    meta.pop("_sample_confidence", None)
    split_counts = Counter(row["train_validation_test_split"] for row in evidence)
    eligible_indices = [i for i, row in enumerate(evidence) if row["train_validation_test_split"] != "purged"]
    eligible_y = y[eligible_indices]
    eligible_sequence_count = len({(evidence[i]["mode"], evidence[i]["symbol"]) for i in eligible_indices})
    eligible_positive_rate = float(np.mean(eligible_y >= 0.5)) if len(eligible_y) else None
    checks = {
        "sample_floor": len(eligible_y) >= max(spec.min_samples, int(split_policy.get("minimum_total_samples", 240))),
        "sequence_floor": eligible_sequence_count >= max(spec.min_sequences, int(split_policy.get("minimum_eligible_sequences", 8))),
        "label_balance": eligible_positive_rate is not None and 0.35 <= eligible_positive_rate <= 0.65,
        "positive_examples": int(np.sum(eligible_y >= 0.5)) >= spec.min_positive_samples,
        "negative_examples": int(np.sum(eligible_y < 0.5)) >= spec.min_negative_samples,
    }
    for split, default in (("train", 144), ("validation", 48), ("test", 48)):
        checks[f"{split}_floor"] = split_counts[split] >= int(split_policy.get(f"minimum_{split}_samples", default))
        values = [row["label_value"] for row in evidence if row["train_validation_test_split"] == split]
        checks[f"{split}_both_classes"] = bool(values and min(values) < 0.5 <= max(values))
    row = {
        "bot_id": spec.bot_id,
        "observation_count": sum(map(len, selected.values())),
        "requested_lookback_days": spec.lookback_days,
        "runtime_meta": meta,
        "split_counts": dict(split_counts),
        "eligible_after_purge_sample_count": len(eligible_y),
        "eligible_after_purge_sequence_count": eligible_sequence_count,
        "eligible_after_purge_positive_rate": eligible_positive_rate,
        "purged_samples_are_training_evidence": False,
        "pre_split_label_balancing": False,
        "checks": checks,
        "blockers": [key for key, ready in checks.items() if not ready],
        "data_checks_passed": all(checks.values()),
        "model_validation_performed": False,
        "authority_contract": dict(AUTHORITY),
    }
    return X, y, evidence, row


def _materialize(directory: Path, bot_id: str, X, y, evidence: list[dict], receipt: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle, features=X, labels=y,
                split=np.asarray([row.get("train_validation_test_split", "unassigned") for row in evidence]),
                eligible_mask=np.asarray([row.get("train_validation_test_split") in {"train", "validation", "test"} for row in evidence], dtype=bool),
                evidence_json=np.asarray(json.dumps(evidence, sort_keys=True)),
                receipt_json=np.asarray(json.dumps(receipt, sort_keys=True)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        output = directory / f"{bot_id}_{digest}.npz"
        os.replace(temporary, output)
        temporary = None
        return {
            "path": str(output), "sha256": digest, "sample_count": len(y),
            "eligible_sample_count": sum(row.get("train_validation_test_split") in {"train", "validation", "test"} for row in evidence),
            "purged_sample_count": sum(row.get("train_validation_test_split") == "purged" for row in evidence),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_payload(root: Path, *, bot_ids: list[str], materialize: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    deadline = time.monotonic() + 60.0
    registry = load_json(root / "master_bot_registry.json")
    active = {row["bot_id"]: row for row in registry.get("sub_bots", []) if row.get("active") and not row.get("deleted_from_rotation")}
    selected_ids = bot_ids or [bot_id for bot_id in SUPPORTED_MODULES if bot_id in active]
    snapshot_path = root / "governance/health/runtime_training_snapshot_latest.json"
    snapshot = load_json(snapshot_path)
    age = diagnostic_age_hours(snapshot, now)
    payload = {
        "timestamp_utc": now.isoformat(), "schema_version": 1,
        "overall_status": "blocked", "active_bot_count": len(active),
        "requested_bot_count": len(selected_ids), "results": [], "blockers": [],
        "authority_contract": dict(AUTHORITY),
    }
    if len(selected_ids) > 10:
        payload["blockers"] = ["preflight_bot_budget_exceeded"]
        return payload
    content_age = diagnostic_age_hours({"timestamp_utc": snapshot.get("latest_row_timestamp_utc")}, now)
    if age is None or age > 24 or content_age is None or content_age > 24 or snapshot.get("schema_version") != 2 or not snapshot.get("rows_sha256"):
        payload["blockers"] = ["snapshot_missing_stale_or_unverified"]
        return payload
    # The explicit snapshot loader verifies its digest and never falls back to a live database.
    rows_path = Path(str(snapshot.get("rows_path") or ""))
    from core.storage_router import inspect_storage_path

    if inspect_storage_path(rows_path)["status"] != "present" or rows_path.stat().st_size > 1024 * 1024 * 1024:
        payload["blockers"] = ["snapshot_unavailable_or_exceeds_preflight_byte_budget"]
        return payload
    read_audit = {}
    specs = {bot_id: preparation_spec(bot_id) for bot_id in selected_ids
             if bot_id in (*SUPPORTED_MODULES, *EXPLICIT_RUNTIME_MODULES) and bot_id in active}
    feature_fields = snapshot_feature_fields(specs)
    sequences = _load_runtime_snapshot_rows(
        root, lookback_days=min(max(int(snapshot.get("lookback_days", 1)), 1), 14),
        mode_allowlist=sorted({mode for spec in specs.values() for mode in spec.mode_allowlist}), symbol_allowlist=None,
        max_source_bytes=1024 * 1024 * 1024, max_retained_bytes=32 * 1024 * 1024,
        max_retained_rows=30000, max_line_bytes=2 * 1024 * 1024,
        deadline_monotonic=deadline - 15.0, strict_snapshot=True,
        read_audit=read_audit,
        feature_allowlist=feature_fields,
    )
    payload["snapshot_read_audit"] = read_audit
    payload["snapshot_read_budget"] = {
        "source_bytes": 1024 * 1024 * 1024, "retained_encoded_row_bytes": 32 * 1024 * 1024,
        "line_bytes": 2 * 1024 * 1024, "retained_rows": 30000,
        "read_deadline_seconds": 45, "full_digest_required": True,
        "retained_byte_limit_is_not_total_process_rss": True,
        "retained_feature_fields": feature_fields,
        "projection_preserves_all_row_identity_fields": True,
    }
    if not sequences or sum(map(len, sequences.values())) > 30000 or load_json(snapshot_path) != snapshot:
        payload["blockers"] = [read_audit.get("reason") or "snapshot_empty_changed_or_exceeds_preflight_read_budget"]
        return payload
    if any(_epoch(row["timestamp_utc"]) > now.timestamp() for rows in sequences.values() for row in rows):
        payload["blockers"] = ["snapshot_contains_future_observations"]
        return payload
    intake = load_json(root / "governance/training_labeling_intelligence/label_depth_training_dataset_latest.json")
    work_items = {row["bot_id"]: row for row in intake.get("work_items", [])}
    source_files = [root / "core" / name for name in (
        "runtime_training_common.py", "crypto_runtime_bot_common.py", "runtime_requested_bot_common.py", "training_diagnostic_contract.py", "indicator_bot_common.py",
    )]
    source_files.append(root / "scripts/ops/training_dataset_preflight.py")
    source_receipt = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    for bot_id in selected_ids:
        if bot_id not in specs:
            payload["results"].append({"bot_id": bot_id, "blockers": ["unsupported_or_inactive_bot"], "data_checks_passed": False})
            continue
        if time.monotonic() >= deadline:
            payload["results"].append({"bot_id": bot_id, "blockers": ["preflight_time_budget"], "data_checks_passed": False})
            continue
        try:
            spec = specs[bot_id]
            contract = active[bot_id].get("training_label_materialization_contract", {})
            if contract.get("objective_class") not in (None, "market_outcome"):
                payload["results"].append({
                    "bot_id": bot_id, "objective_class": contract["objective_class"],
                    "blockers": ["requires_objective_specific_materializer"], "data_checks_passed": False,
                    "authority_contract": dict(AUTHORITY),
                })
                continue
            split_policy = work_items.get(bot_id, {}).get("label_quality_contract", {}).get("split_policy", {})
            X, y, evidence, result = prepare_dataset(spec, sequences, contract, split_policy)
            receipt = {
                "snapshot_rows_sha256": snapshot["rows_sha256"],
                "snapshot_timestamp_utc": snapshot["timestamp_utc"],
                "label_contract_sha256": contract["contract_sha256"],
                "materialization_contract": contract,
                "source_sha256": {**source_receipt, f"core/{bot_id}.py": hashlib.sha256((root / "core" / f"{bot_id}.py").read_bytes()).hexdigest()},
                "feature_names": list(spec.feature_names),
                "preparation_settings": {
                    "window": spec.window, "horizon": spec.horizon,
                    "min_confidence": spec.min_confidence, "max_samples": 2000,
                    "mode_allowlist": list(spec.mode_allowlist),
                    "symbol_allowlist": getattr(spec, "symbol_allowlist", None),
                    "sample_stride": getattr(spec, "sample_stride", 1),
                    "split_policy": split_policy, "runtime_meta": result["runtime_meta"],
                },
                "authority_contract": dict(AUTHORITY),
            }
            result["receipt"] = receipt
            if materialize:
                result["dataset"] = _materialize(root / "exports/training/preflight", bot_id, X, y, evidence, receipt)
            payload["results"].append(result)
        except Exception as exc:
            payload["results"].append({"bot_id": bot_id, "blockers": [f"preflight_failed:{type(exc).__name__}:{exc}"], "data_checks_passed": False})
    payload.update({
        "overall_status": "assessed",
        "snapshot_timestamp_utc": snapshot["timestamp_utc"],
        "snapshot_latest_row_timestamp_utc": snapshot["latest_row_timestamp_utc"],
        "snapshot_partial": bool(snapshot.get("incremental_partial")),
        "snapshot_lookback_days": snapshot["lookback_days"],
        "audited_bot_count": sum("runtime_meta" in row for row in payload["results"]),
        "data_checks_passed_count": sum(row["data_checks_passed"] for row in payload["results"]),
        "materialized_sample_count": sum(row.get("dataset", {}).get("sample_count", 0) for row in payload["results"]),
        "materialized_eligible_sample_count": sum(row.get("dataset", {}).get("eligible_sample_count", 0) for row in payload["results"]),
    })
    payload["unassessed_active_bot_count"] = len(active) - payload["audited_bot_count"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-bot-ids", default="")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT, bot_ids=list(dict.fromkeys(filter(None, (item.strip() for item in args.include_bot_ids.split(","))))), materialize=args.materialize)
    write_payload(PROJECT_ROOT / "governance/health/training_dataset_preflight_latest.json", payload)
    print(json.dumps(payload, indent=2) if args.json else f"Training data preflight: {payload['overall_status']}; audited={payload.get('audited_bot_count', 0)}; data_checks_passed={payload.get('data_checks_passed_count', 0)}")
    return 0 if payload["overall_status"] == "assessed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
