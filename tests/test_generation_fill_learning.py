from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops.generation_fill_learning import build_learning_bundle


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_event_chain(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = ""
    rows: list[str] = []
    for raw in events:
        event = {**raw, "previous_event_hash": previous}
        event["event_hash"] = _canonical_hash(event)
        previous = event["event_hash"]
        rows.append(json.dumps(event, ensure_ascii=True, sort_keys=True))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _initialize_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    _write_event_chain(
        project / "governance/evidence/production_candidate_events.jsonl",
        [
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g103",
                "generation": 103,
                "timestamp_utc": "2026-08-25T00:00:00+00:00",
                "change_reason": "baseline",
                "changed_scopes": ["strategy"],
                "git_head": "head-103",
                "overall_sha256": "candidate-hash-103",
            },
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g104",
                "generation": 104,
                "timestamp_utc": "2026-08-26T00:00:00+00:00",
                "change_reason": "crisis drill hardening",
                "changed_scopes": ["operations", "promotion"],
                "git_head": "head-104",
                "overall_sha256": "candidate-hash-104",
            },
        ],
    )
    runtime = project / "governance/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "production_candidate_state.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-g104",
                "generation": 104,
                "accepted_at_utc": "2026-08-26T00:00:00+00:00",
                "overall_sha256": "candidate-hash-104",
            }
        ),
        encoding="utf-8",
    )
    health = project / "governance/health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "training_lineage_manifest_latest.json").write_text(
        json.dumps(
            {
                "lineage_contract_ready": True,
                "feature_store_lineage_ok": True,
                "exact_replay_ready": True,
                "hash_bundle_complete": True,
                "lineage_score": 100.0,
                "missing_contracts": [],
            }
        ),
        encoding="utf-8",
    )
    (health / "training_quality_control_latest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "overall_status": "ready",
                "training_quality_score": 100.0,
            }
        ),
        encoding="utf-8",
    )
    return project


def _fill(
    *,
    timestamp: str,
    decision_id: str,
    pnl: float,
    candidate_id: str = "",
    generation: int = 0,
    fill_source: str = "expected_fill_model",
) -> dict:
    metadata = {
        "source_profile": "dividend",
        "snapshot_id": f"snapshot-{decision_id}",
        "entry_policy": {
            "profile_family": "dividend",
            "regime_fit_norm": 0.8,
            "evidence_quality_norm": 0.9,
        },
    }
    if candidate_id:
        metadata["production_candidate_id"] = candidate_id
        metadata["production_candidate_generation"] = generation
    return {
        "timestamp_utc": timestamp,
        "mode": "paper",
        "symbol": "SCHD",
        "action": "BUY" if pnl >= 0.0 else "SELL",
        "decision_id": decision_id,
        "message_id": f"fill-{decision_id}",
        "paper_pnl_schema_version": 3,
        "post_cost_pnl_delta": pnl,
        "post_cost_return_bps": pnl * 10.0,
        "expected_execution_cost_amount": 0.05,
        "paper_fill_source": fill_source,
        "strategy": "sleeve::dividend_income::quality_value::v1",
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_g104_learning_owner_preserves_source_generation_and_quarantines_unbound(
    tmp_path: Path,
) -> None:
    project = _initialize_project(tmp_path)
    paper_path = tmp_path / "paper_trades_20260825.jsonl"
    _write_jsonl(
        paper_path,
        [
            _fill(
                timestamp="2026-08-25T10:00:00+00:00",
                decision_id="direct-positive",
                pnl=1.0,
                candidate_id="candidate-g103",
                generation=103,
            ),
            _fill(
                timestamp="2026-08-25T11:00:00+00:00",
                decision_id="receipt-negative",
                pnl=-1.0,
            ),
            _fill(
                timestamp="2026-08-25T12:00:00+00:00",
                decision_id="legacy-unbound",
                pnl=-2.0,
            ),
            _fill(
                timestamp="2026-08-26T01:00:00+00:00",
                decision_id="current-positive",
                pnl=2.0,
                candidate_id="candidate-g104",
                generation=104,
                fill_source="observed_touch",
            ),
        ],
    )
    decision_path = tmp_path / "master_control_20260825.jsonl"
    _write_jsonl(
        decision_path,
        [
            {
                "timestamp_utc": "2026-08-25T10:59:30+00:00",
                "decision_id": "receipt-negative",
                "symbol": "SCHD",
                "action": "SELL",
                "shadow_profile": "dividend",
                "metadata": {
                    "production_candidate_id": "candidate-g103",
                    "production_candidate_generation": 103,
                },
            }
        ],
    )

    payload, rows, quarantine = build_learning_bundle(
        project,
        target_generation=104,
        paper_files=[paper_path],
        decision_files=[decision_path],
        generated_at_utc="2026-08-27T00:00:00+00:00",
    )

    assert payload["ok"] is True
    assert payload["learning_target"]["generation"] == 104
    assert payload["policy"]["learning_outputs_attributed_to_generation"] == 104
    assert (
        payload["policy"]["historical_source_fills_reattributed_to_generation_104"]
        is False
    )
    assert payload["dataset"]["verified_row_count"] == 3
    assert payload["dataset"]["quarantined_row_count"] == 1
    assert payload["source_scan"]["paper_source_candidate_path_count"] == 1
    assert payload["source_scan"]["paper_file_count"] == 1
    assert payload["source_scan"]["paper_source_paths_not_present"] == 0
    assert {row["source_provenance"]["generation"] for row in rows} == {103, 104}
    recovered = next(
        row for row in rows if row["identity"]["decision_ids"] == ["receipt-negative"]
    )
    assert recovered["source_provenance"]["generation"] == 103
    assert recovered["source_provenance"]["binding_method"].startswith(
        "receipt_recovered"
    )
    negative = next(row for row in rows if row["label"]["outcome"] == "negative")
    positive = next(
        row
        for row in rows
        if row["label"]["outcome"] == "positive"
        and row["source_provenance"]["generation"] == 103
    )
    assert (
        negative["weighting"]["base_outcome_multiplier"]
        > positive["weighting"]["base_outcome_multiplier"]
    )
    assert quarantine[0]["status"] == "legacy_unbound"
    assert quarantine[0]["timestamp_only_generation_inference_used"] is False
    assert quarantine[0]["developmental_pretraining_eligible"] is False


def test_generation_learning_builds_purged_chronological_and_generation_holdouts(
    tmp_path: Path,
) -> None:
    project = _initialize_project(tmp_path)
    config = project / "config/generation_fill_learning_v1.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "validation_policy": {
                    "train_fraction": 0.6,
                    "validation_fraction": 0.2,
                    "embargo_hours": 0.0,
                    "minimum_rows_per_partition": 1,
                    "minimum_rows_per_generation_fold": 2,
                    "minimum_generation_folds": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    start = datetime(2026, 8, 25, 1, tzinfo=timezone.utc)
    fills: list[dict] = []
    for index in range(10):
        timestamp = start + timedelta(minutes=index)
        fills.append(
            _fill(
                timestamp=timestamp.isoformat(),
                decision_id=f"g103-{index}",
                pnl=-1.0 if index % 2 else 1.0,
                candidate_id="candidate-g103",
                generation=103,
                fill_source="observed_touch",
            )
        )
    start = datetime(2026, 8, 26, 1, tzinfo=timezone.utc)
    for index in range(10):
        timestamp = start + timedelta(minutes=index)
        fills.append(
            _fill(
                timestamp=timestamp.isoformat(),
                decision_id=f"g104-{index}",
                pnl=-1.0 if index % 2 else 1.0,
                candidate_id="candidate-g104",
                generation=104,
                fill_source="observed_touch",
            )
        )
    paper_path = tmp_path / "paper_trades_20260826.jsonl"
    _write_jsonl(paper_path, fills)

    payload, rows, quarantine = build_learning_bundle(
        project,
        target_generation=104,
        paper_files=[paper_path],
        decision_files=[],
        config_path=config,
        generated_at_utc="2026-08-27T00:00:00+00:00",
    )

    assert not quarantine
    assert len(rows) == 20
    assert payload["validation"]["chronological_split_ready"] is True
    assert payload["validation"]["leave_one_generation_out_ready"] is True
    assert payload["validation"]["random_shuffle_split_allowed"] is False
    assert (
        payload["challenger_training_gate"]["challenger_training_launch_allowed"]
        is True
    )
    assert payload["overall_status"] == "challenger_training_ready"
    assert (
        payload["generation_weight_balance"]["maximum_generation_weight_share"]
        <= payload["generation_weight_balance"][
            "effective_maximum_generation_weight_share"
        ]
    )


def test_generation_learning_refuses_to_assign_output_to_noncurrent_generation(
    tmp_path: Path,
) -> None:
    project = _initialize_project(tmp_path)
    payload, _, _ = build_learning_bundle(
        project,
        target_generation=103,
        paper_files=[],
        decision_files=[],
        generated_at_utc="2026-08-27T00:00:00+00:00",
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert (
        "requested_learning_target_is_not_current_candidate"
        in payload["challenger_training_gate"]["blockers"]
    )
