from __future__ import annotations

import hashlib
import gzip
import json
from itertools import permutations
from pathlib import Path

import pytest

from scripts.ops.generation_behavior_attribution import build_payload, _read_events
from scripts.ops.production_excellence_control import read_candidate_event_chain


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


def test_candidate_history_survives_split_archives_and_active_tail(tmp_path):
    path = tmp_path / "production_candidate_events.jsonl"
    _write_event_chain(path, [
        {"event_type": "candidate_change_accepted", "generation": i,
         "candidate_id": f"candidate-g{i}", "timestamp_utc": f"2026-09-0{i}T12:00:00+00:00"}
        for i in range(1, 4)
    ])
    lines = path.read_bytes().splitlines(keepends=True)
    path.with_name(path.name + ".gz").write_bytes(gzip.compress(lines[0]))
    path.with_name(path.name + ".raw-training.gz").write_bytes(gzip.compress(lines[1]))
    path.write_bytes(lines[2])
    events, chain = _read_events(path)
    assert chain["ok"] and chain["event_count"] == 3
    assert [row["generation"] for row in events] == [1, 2, 3]
    path.with_name(path.name + ".raw-training.gz").write_bytes(gzip.compress(b"".join(lines[1:])))
    assert _read_events(path)[1]["event_count"] == 3
    path.unlink()
    assert _read_events(path)[1]["event_count"] == 3


def test_candidate_archive_corruption_or_missing_prefix_fails_closed(tmp_path):
    path = tmp_path / "production_candidate_events.jsonl"
    _write_event_chain(path, [
        {"event_type": "candidate_change_accepted", "generation": i,
         "candidate_id": f"candidate-g{i}", "timestamp_utc": f"2026-09-0{i}T12:00:00+00:00"}
        for i in range(1, 3)
    ])
    lines = path.read_bytes().splitlines(keepends=True)
    archive = path.with_name(path.name + ".gz")
    archive.write_bytes(gzip.compress(lines[0]))
    path.write_bytes(lines[1])
    archive.unlink()
    events, chain = read_candidate_event_chain(path)
    assert not events and not chain["ok"]
    assert "previous_hash_mismatch_line=1" in chain["errors"]
    archive.write_bytes(b"not gzip")
    assert not _read_events(path)[1]["ok"]
    archive.write_bytes(gzip.compress(lines[0].replace(b"candidate-g1", b"candidate-g9")))
    assert not _read_events(path)[1]["ok"]


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_candidate_partitions_use_hash_order_not_archive_suffix(tmp_path, order):
    path = tmp_path / "events.jsonl"
    _write_event_chain(path, [{"generation": i} for i in range(1, 7)])
    lines = path.read_bytes().splitlines(keepends=True)
    partitions = [b"".join(lines[i:i + 2]) for i in (0, 2, 4)]
    sources = [path.with_name(path.name + ".gz"),
               path.with_name(path.name + ".raw-training.gz"), path]
    for target, index in zip(sources, order):
        content = partitions[index]
        target.write_bytes(gzip.compress(content) if target.suffix == ".gz" else content)
    original = [source.read_bytes() for source in sources]
    rows, chain = read_candidate_event_chain(path)
    assert chain["ok"] and chain["event_count"] == 6
    assert [row["generation"] for row in rows] == list(range(1, 7))
    assert [source.read_bytes() for source in sources] == original
    # Reversing events inside any partition is still invalid.
    target = sources[0]
    target.write_bytes(gzip.compress(b"".join(partitions[order[0]].splitlines(keepends=True)[::-1])))
    assert not read_candidate_event_chain(path)[1]["ok"]


def test_reverse_archive_order_preserves_missing_tail_and_fork_failures(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_event_chain(path, [{"generation": i} for i in range(1, 4)])
    lines = path.read_bytes().splitlines(keepends=True)
    path.with_name(path.name + ".raw-training.gz").write_bytes(gzip.compress(lines[0]))
    path.with_name(path.name + ".gz").write_bytes(gzip.compress(lines[1]))
    path.unlink()
    rows, chain = read_candidate_event_chain(path)
    assert chain["ok"] and [r["generation"] for r in rows] == [1, 2]
    assert chain["chain_head"] != json.loads(lines[2])["event_hash"]
    fork = {**json.loads(lines[1]), "generation": 99}
    fork.pop("event_hash")
    fork["event_hash"] = _canonical_hash(fork)
    path.write_text(json.dumps(fork) + "\n")
    assert not read_candidate_event_chain(path)[1]["ok"]


def test_candidate_reader_does_not_hide_replayed_rows_within_partition(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_event_chain(path, [{"generation": 1}])
    path.write_bytes(path.read_bytes() * 2)
    assert not read_candidate_event_chain(path)[1]["ok"]
    path.with_name(path.name + ".gz").write_bytes(gzip.compress(path.read_bytes().splitlines(keepends=True)[0]))
    assert not read_candidate_event_chain(path)[1]["ok"]


def test_archived_recovery_anchor_must_prove_prior_head_and_window_reset(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_event_chain(path, [{"generation": 8}])
    prior = json.loads(path.read_text())["event_hash"]
    path.with_name(path.name + ".gz").write_bytes(gzip.compress(path.read_bytes()))
    anchor = {"generation": 9, "event_type": "candidate_chain_recovery_anchor",
              "recovery_evidence": {"prior_state_event_chain_head": prior, "all_evidence_windows_reset": True}}
    _write_event_chain(path, [anchor, {"generation": 10}])
    rows, chain = read_candidate_event_chain(path)
    assert chain["ok"] and [row["generation"] for row in rows] == [8, 9, 10]
    assert chain["verified_recovery_boundary_lines"] == [2]
    anchor["recovery_evidence"]["all_evidence_windows_reset"] = False
    _write_event_chain(path, [anchor])
    assert not read_candidate_event_chain(path)[1]["ok"]
    anchor["recovery_evidence"].update(all_evidence_windows_reset=True, prior_state_event_chain_head="wrong")
    _write_event_chain(path, [anchor])
    assert not read_candidate_event_chain(path)[1]["ok"]


def _decision(
    *,
    candidate_id: str,
    generation: int,
    timestamp: str,
    message_id: str,
    action: str,
    intent: str,
    profile: str,
    route_valid: bool,
) -> dict:
    return {
        "timestamp_utc": timestamp,
        "message_id": message_id,
        "channel": "decision",
        "production_candidate_id": candidate_id,
        "metadata": {
            "production_candidate_id": candidate_id,
            "production_candidate_generation": generation,
        },
        "shadow_profile": profile,
        "symbol": "SPY",
        "snapshot_id": message_id,
        "action": action,
        "master_action": action,
        "master_intent_action": intent,
        "decision_disposition": (
            "qualified_shadow_candidate"
            if action in {"BUY", "SELL"}
            else "no_edge_hold"
        ),
        "decision_blocking_stage": (
            "none" if action in {"BUY", "SELL"} else "signal_selection"
        ),
        "decision_guard_categories": [],
        "decision_guard_reasons": [],
        "institutional_decision_flow_ingestion_route_receipt_valid": route_valid,
        "institutional_decision_flow_ingestion_route_quality_norm": 0.9,
        "institutional_decision_flow_utility_norm": 0.7,
        "source_quality_score": 0.95,
        "feature_freshness": {"ok": True},
        "master_latency_slo": {"ok": True, "elapsed_ms": 100.0},
        "circuit_breakers": {
            "lane_kill_switch_active": False,
            "kill_switch_active": False,
        },
    }


def test_generation_behavior_comparison_preserves_segmented_soak_context(
    tmp_path: Path,
) -> None:
    event_path = (
        tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl"
    )
    _write_event_chain(
        event_path,
        [
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g65",
                "generation": 65,
                "timestamp_utc": "2026-08-18T00:00:00+00:00",
                "change_reason": "baseline",
                "changed_scopes": ["strategy"],
            },
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g99",
                "generation": 99,
                "timestamp_utc": "2026-08-19T00:00:00+00:00",
                "change_reason": "new decision controls",
                "changed_scopes": ["operations", "strategy"],
            },
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g100",
                "generation": 100,
                "timestamp_utc": "2026-08-20T00:00:00+00:00",
                "change_reason": "next candidate",
                "changed_scopes": ["data"],
            },
        ],
    )
    decision_path = tmp_path / "master_control_20260818.jsonl"
    decisions = [
        _decision(
            candidate_id="candidate-g65",
            generation=65,
            timestamp="2026-08-18T01:00:00+00:00",
            message_id="g65-1",
            action="HOLD",
            intent="HOLD",
            profile="dividend",
            route_valid=False,
        ),
        _decision(
            candidate_id="candidate-g65",
            generation=65,
            timestamp="2026-08-18T01:01:00+00:00",
            message_id="g65-2",
            action="BUY",
            intent="BUY",
            profile="dividend",
            route_valid=False,
        ),
        _decision(
            candidate_id="candidate-g99",
            generation=99,
            timestamp="2026-08-19T01:00:00+00:00",
            message_id="g99-1",
            action="BUY",
            intent="BUY",
            profile="dividend",
            route_valid=True,
        ),
        _decision(
            candidate_id="candidate-g99",
            generation=99,
            timestamp="2026-08-19T01:01:00+00:00",
            message_id="g99-2",
            action="SELL",
            intent="SELL",
            profile="dividend",
            route_valid=True,
        ),
    ]
    decisions.append(dict(decisions[-1]))
    decision_path.write_text(
        "\n".join(json.dumps(row) for row in decisions) + "\n", encoding="utf-8"
    )
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "continuous_soak_integrity_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "control_grade": "A+",
                "main_soak_counting_mode": "cumulative_segmented_candidate_wall_clock",
                "main_soak_elapsed_hours": 300.0,
                "main_soak_active_runtime_evidence_hours": 295.0,
                "main_soak_planned_maintenance_excluded_hours": 5.0,
                "main_soak_includes_pre_reset_time": True,
                "clean_window_started_utc": "2026-08-20T00:00:00+00:00",
                "clean_window_elapsed_hours": 1.0,
                "historical_soak_evidence": {"segment_count": 3},
            }
        ),
        encoding="utf-8",
    )
    (health / "paper_performance_latest.json").write_text(
        json.dumps(
            {
                "developmental_generation_flows": {
                    "generation_flows": [
                        {
                            "candidate_id": "candidate-g65",
                            "candidate_generation": 65,
                            "candidate_generation_consistent": True,
                            "sample_count": 2,
                            "observed_days": 1,
                            "post_cost_pnl_delta_total": -1.0,
                            "execution_cost_total": 0.2,
                        },
                        {
                            "candidate_id": "candidate-g99",
                            "candidate_generation": 99,
                            "candidate_generation_consistent": True,
                            "sample_count": 2,
                            "observed_days": 1,
                            "post_cost_pnl_delta_total": 1.0,
                            "execution_cost_total": 0.2,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "governance" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "production_candidate_state.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-g100",
                "generation": 100,
                "accepted_at_utc": "2026-08-20T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    payload = build_payload(
        tmp_path,
        from_generation=65,
        to_generation=99,
        minimum_decisions=2,
        minimum_observation_hours=0.0,
        source_files=[decision_path],
        generated_at_utc="2026-08-20T01:00:00+00:00",
    )

    assert payload["ok"] is True
    assert payload["cumulative_soak_context"]["main_soak_elapsed_hours"] == 300.0
    assert (
        payload["cumulative_soak_context"][
            "historical_segments_grade_current_candidate"
        ]
        is False
    )
    comparison = payload["comparison"]
    assert comparison["behavior_comparison_ready"] is True
    assert comparison["identity_bound_behavior_comparison_ready"] is True
    assert comparison["legacy_window_association_involved"] is False
    assert comparison["economic_comparison_ready"] is True
    assert comparison["metric_deltas"]["final_directional_rate"]["before"] == 0.5
    assert comparison["metric_deltas"]["final_directional_rate"]["after"] == 1.0
    assert (
        comparison["metric_deltas"]["post_cost_pnl_delta_total"]["absolute_delta"]
        == 2.0
    )
    assert payload["scan_window"]["counters"]["duplicate_records_suppressed"] == 1
    assert payload["policy"]["live_execution_authority"] is False


def test_generation_behavior_keeps_legacy_window_association_descriptive(
    tmp_path: Path,
) -> None:
    event_path = (
        tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl"
    )
    _write_event_chain(
        event_path,
        [
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g65",
                "generation": 65,
                "timestamp_utc": "2026-08-18T00:00:00+00:00",
                "change_reason": "legacy baseline",
                "changed_scopes": ["strategy"],
            },
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g99",
                "generation": 99,
                "timestamp_utc": "2026-08-19T00:00:00+00:00",
                "change_reason": "identity stamping",
                "changed_scopes": ["operations"],
            },
            {
                "schema_version": 1,
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g100",
                "generation": 100,
                "timestamp_utc": "2026-08-20T00:00:00+00:00",
                "change_reason": "next candidate",
                "changed_scopes": ["operations"],
            },
        ],
    )
    legacy = _decision(
        candidate_id="candidate-g65",
        generation=65,
        timestamp="2026-08-18T01:00:00+00:00",
        message_id="legacy-g65",
        action="HOLD",
        intent="HOLD",
        profile="dividend",
        route_valid=False,
    )
    legacy.pop("production_candidate_id")
    legacy["metadata"].pop("production_candidate_id")
    legacy["metadata"].pop("production_candidate_generation")
    bound = _decision(
        candidate_id="candidate-g99",
        generation=99,
        timestamp="2026-08-19T01:00:00+00:00",
        message_id="bound-g99",
        action="BUY",
        intent="BUY",
        profile="dividend",
        route_valid=True,
    )
    decision_path = tmp_path / "master_control_20260818.jsonl"
    decision_path.write_text(
        json.dumps(legacy) + "\n" + json.dumps(bound) + "\n",
        encoding="utf-8",
    )

    payload = build_payload(
        tmp_path,
        from_generation=65,
        to_generation=99,
        minimum_decisions=1,
        minimum_observation_hours=0.0,
        source_files=[decision_path],
        generated_at_utc="2026-08-20T01:00:00+00:00",
    )

    comparison = payload["comparison"]
    assert comparison["behavior_comparison_ready"] is True
    assert comparison["identity_bound_behavior_comparison_ready"] is False
    assert comparison["legacy_window_association_involved"] is True
    assert (
        comparison["status"]
        == "behavior_comparison_ready_with_legacy_window_association"
    )
    assert comparison["metric_deltas"]["final_directional_rate"]["after"] == 1.0
    assert payload["policy"]["legacy_window_association_never_promotion_grade"] is True


def test_generation_behavior_blocks_invalid_candidate_event_chain(
    tmp_path: Path,
) -> None:
    event_path = (
        tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl"
    )
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text(
        json.dumps(
            {
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-g65",
                "generation": 65,
                "timestamp_utc": "2026-08-18T00:00:00+00:00",
                "previous_event_hash": "wrong",
                "event_hash": "wrong",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_payload(
        tmp_path,
        source_files=[],
        generated_at_utc="2026-08-20T01:00:00+00:00",
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "candidate_event_chain_blocked"
    assert payload["candidate_event_chain"]["valid"] is False
