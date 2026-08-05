import gzip
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import data_collection_observation_rollup as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _registry(bot_id: str) -> dict:
    return {
        "summary": {},
        "sub_bots": [
            {
                "bot_id": bot_id,
                "bot_role": "signal_sub_bot",
                "active": True,
                "lifecycle_state": "data_collection_only",
                "data_collection_active": True,
                "data_collection_started_utc": "2026-04-20T00:00:00+00:00",
                "data_collection_observations": 0,
                "minimum_training_observations": 2,
                "minimum_data_collection_days": 1,
                "training_excluded": True,
                "exclude_from_training": True,
            }
        ],
    }


def test_observation_rollup_bootstraps_and_updates_registry(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v167_intraday_opening_range_momentum_burst"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_intraday_aggressive_equities" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            [
                json.dumps({"status": "DATA_ONLY_BLOCKED", "reasons": [f"bot_id={bot_id}"]}),
                json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": bot_id}}),
                json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": "brain_refinery_v1_other"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["mode"] == "bootstrap_tail"
    assert payload["bots_with_observations"] == 1
    assert payload["total_observations"] == 2
    assert row["data_collection_observations"] == 2
    assert row["collected_observation_count"] == 2
    assert row["data_collection_training_ready"] is True
    assert row["training_excluded"] is False
    assert registry["summary"]["data_collection_training_ready_bots"] == 1


def test_observation_rollup_counts_only_new_lines_after_bootstrap(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v171_intraday_relative_volume_surge_chaser"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_intraday_aggressive_equities" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(json.dumps({"status": "DATA_ONLY_BLOCKED", "reasons": [f"bot_id={bot_id}"]}) + "\n", encoding="utf-8")

    src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    with decision_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": bot_id}}) + "\n")

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "incremental"
    assert payload["new_rows_counted"] == 1
    assert registry["sub_bots"][0]["data_collection_observations"] == 2


def test_observation_rollup_reads_compressed_decision_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v188_crypto_breakout_liquidity_rotation"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_crypto" / f"decision_explanations_{stamp}.jsonl.gz"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(decision_file, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"status": "DATA_ONLY_BLOCKED", "strategy": bot_id}) + "\n")
        handle.write(json.dumps({"status": "SHADOW_ONLY", "reasons": [f"bot_id={bot_id}"]}) + "\n")

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["files_scanned"] == 1
    assert payload["bots_with_observations"] == 1
    assert payload["total_observations"] == 2
    assert registry["sub_bots"][0]["data_collection_observations"] == 2
    assert state["file_line_counts"][str(decision_file.relative_to(project_root))] == 2


def test_observation_rollup_credits_governance_artifact_references_once(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v1010_recursive_awareness_causal_incident_root_cause_builder_bot"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    artifact_path = project_root / "governance" / "health" / "deep_recursive_awareness_latest.json"
    _write_json(
        artifact_path,
        {
            "ok": True,
            "generated_at_utc": "2026-05-04T10:00:00+00:00",
            "pack": {"bot_ids": [bot_id]},
        },
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    second_payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["artifact_files_scanned"] == 1
    assert payload["new_artifact_observations_counted"] == 1
    assert registry["sub_bots"][0]["data_collection_observations"] == 1
    assert second_payload["new_artifact_observations_counted"] == 0
    assert second_payload["total_observations"] == 1


def test_observation_rollup_uses_training_diagnostics_as_observation_floor(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v1614_training_labeling_label_contract_normalizer_telemetry_collector_bot"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    _write_json(
        project_root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json",
        {
            "status": "deferred_sample_starved",
            "sample_count": 1,
            "eligible_sequences": 1,
        },
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["overall_status"] == "ready"
    assert payload["bots_with_observations"] == 1
    assert payload["zero_observation_count"] == 0
    assert payload["diagnostic_files_scanned"] == 1
    assert payload["diagnostic_observations_counted"] == 1
    assert row["data_collection_observations"] == 1
    assert row["data_collection_training_ready"] is False
    assert row["training_excluded"] is True


def test_observation_rollup_counts_governance_channel_events(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v261_crypto_eth_gas_defi_activity_guard"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    stamp = src._day_stamps(1)[0]
    channel_file = project_root / "governance" / "channels" / "risk" / "crypto_futures_basis" / f"risk_{stamp}.jsonl"
    channel_file.parent.mkdir(parents=True, exist_ok=True)
    channel_file.write_text(
        "\n".join(
            [
                json.dumps({"bot_id": bot_id, "channel": "risk", "action": "HOLD"}),
                json.dumps({"bot_id": bot_id, "channel": "risk", "action": "HOLD"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["overall_status"] == "ready"
    assert payload["bots_with_observations"] == 1
    assert payload["channel_files_scanned"] == 1
    assert payload["channel_observations_counted"] == 2
    assert registry["sub_bots"][0]["data_collection_observations"] == 2


def test_iter_tail_lines_bounds_sparse_large_line(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(src, "DEFAULT_CHANNEL_TAIL_BYTES", 128)
    path = tmp_path / "sparse_channel.jsonl"
    path.write_bytes(
        b"x" * 4096
        + b"\n"
        + json.dumps({"bot_id": "brain_refinery_v171_intraday_relative_volume_surge_chaser"}).encode("utf-8")
        + b"\n"
        + json.dumps({"bot_id": "brain_refinery_v172_intraday_breakout_retest_quality"}).encode("utf-8")
        + b"\n"
    )

    lines = src._iter_tail_lines(path, limit=2)

    assert len(lines) == 2
    assert all("bot_id" in line for line in lines)
    assert sum(len(line) for line in lines) < 256


def test_observation_rollup_manages_training_labeling_observer_zero_debt(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v1661_training_labeling_label_contract_normalizer_telemetry_collector_bot"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry(bot_id)
    row = registry["sub_bots"][0]
    row["data_collection_mode"] = "active_observer"
    row["data_collection_reason"] = "training_labeling_intelligence_collect_only_until_label_and_training_effect_gates_clear"
    row["minimum_training_observations"] = 70000
    row["minimum_data_collection_days"] = 180
    row["trading_enabled"] = False
    row["labeling_tags"] = ["collection_guard:training_labeling_intelligence_v1"]
    _write_json(registry_path, registry)

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["overall_status"] == "ready"
    assert payload["bots_with_observations"] == 0
    assert payload["effective_bots_with_observations"] == 1
    assert payload["zero_observation_count"] == 0
    assert payload["managed_zero_observation_count"] == 1
    assert payload["raw_zero_observation_count"] == 1
    assert payload["zero_observation_repair_lane"]["active"] is False
    assert payload["managed_zero_observation_lane"]["active"] is True
    assert row["data_collection_training_ready"] is False
    assert row["training_exclusion_reason"] == "collecting_training_labeling_effect_evidence_before_training"
    assert row["training_exclusion_until"] == "training_labeling_collection_threshold_met"


def test_observation_rollup_includes_training_excluded_paper_live_data(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v56_meta_ranker"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry(bot_id)
    registry["sub_bots"][0]["lifecycle_state"] = "paper_live_data"
    registry["sub_bots"][0]["minimum_training_observations"] = 2
    _write_json(registry_path, registry)
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_infra" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            [
                json.dumps({"status": "DATA_ONLY_BLOCKED", "metadata": {"bot_id": bot_id}}),
                json.dumps({"status": "SHADOW_ONLY", "reasons": [f"bot_id={bot_id}"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["collector_count"] == 1
    assert payload["bots_with_observations"] == 1
    assert row["data_collection_observations"] == 2
    assert row["data_collection_training_ready"] is True
    assert row["training_excluded"] is False
    assert registry["summary"]["data_collection_training_ready_bots"] == 1


def test_observation_rollup_keeps_paper_live_data_blocked_without_observation_floor(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v99_defensive_dividend_concentration"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry(bot_id)
    row = registry["sub_bots"][0]
    row["lifecycle_state"] = "paper_live_data"
    row.pop("minimum_training_observations", None)
    row["training_excluded"] = False
    row["exclude_from_training"] = False
    _write_json(registry_path, registry)
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_dividend" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            [
                json.dumps({"status": "DATA_ONLY_BLOCKED", "metadata": {"bot_id": bot_id}}),
                json.dumps({"status": "SHADOW_ONLY", "reasons": [f"bot_id={bot_id}"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["collector_count"] == 1
    assert payload["bots_with_observations"] == 1
    assert row["data_collection_observations"] == 2
    assert row["data_collection_threshold_progress"]["training_ready"] is True
    assert row["data_collection_training_ready"] is False
    assert row["training_excluded"] is True
    assert row["exclude_from_training"] is True
    assert row["training_exclusion_reason"] == "paper_live_data_requires_minimum_training_observations"
    assert row["promotion_block_reason"] == "awaiting_data_collection_quality_gate"
    assert payload["training_ready_count"] == 0
    assert payload["training_ready_bot_ids"] == []
    assert registry["summary"]["data_collection_training_ready_bots"] == 0


def test_observation_rollup_uses_nested_paper_promotion_observation_floor(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v99_defensive_dividend_concentration"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry(bot_id)
    row = registry["sub_bots"][0]
    row["lifecycle_state"] = "paper_live_data"
    row.pop("minimum_training_observations", None)
    row["paper_promotion_standard"] = {"minimum_observations": 1000, "minimum_collection_days": 1}
    _write_json(registry_path, registry)
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_dividend" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            [
                json.dumps({"status": "DATA_ONLY_BLOCKED", "metadata": {"bot_id": bot_id}}),
                json.dumps({"status": "SHADOW_ONLY", "reasons": [f"bot_id={bot_id}"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["bots_with_observations"] == 1
    assert row["data_collection_threshold_progress"]["minimum_training_observations"] == 1000
    assert row["data_collection_threshold_progress"]["training_ready"] is False
    assert row["data_collection_training_ready"] is False
    assert row["training_exclusion_reason"] == "minimum_data_collection_threshold_not_met"
    assert payload["training_ready_count"] == 0


def test_observation_rollup_excludes_bare_alias_when_canonical_collector_is_active(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    alias_id = "brain_refinery_v1"
    canonical_id = "brain_refinery_v1_price_forecaster_baseline"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    alias_row = _registry(alias_id)["sub_bots"][0]
    alias_row.pop("minimum_training_observations", None)
    canonical_row = _registry(canonical_id)["sub_bots"][0]
    canonical_row["core_module_path"] = "core/brain_refinery_v1_price_forecaster_baseline.py"
    canonical_row["minimum_training_observations"] = 1
    _write_json(registry_path, {"summary": {}, "sub_bots": [alias_row, canonical_row]})
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_signal" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": canonical_id}}) + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )

    assert payload["collector_count"] == 1
    assert payload["bots_with_observations"] == 1
    assert payload["zero_observation_bot_ids"] == []
    assert payload["top_collectors"][0]["bot_id"] == canonical_id


def test_observation_rollup_blocks_zero_observation_collection_bot_without_floor(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v2"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry(bot_id)
    row = registry["sub_bots"][0]
    row.pop("minimum_training_observations", None)
    row["training_excluded"] = False
    row["exclude_from_training"] = False
    _write_json(registry_path, registry)

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["bots_with_observations"] == 0
    assert payload["training_ready_count"] == 0
    assert row["data_collection_training_ready"] is False
    assert row["training_excluded"] is True
    assert row["exclude_from_training"] is True
    assert row["training_exclusion_reason"] == "data_collection_requires_observations"
    assert row["promotion_block_reason"] == "awaiting_data_collection_quality_gate"
    assert payload["zero_observation_bot_ids"] == [bot_id]
    assert payload["zero_observation_repair_lane"]["active"] is True
    assert payload["zero_observation_repair_lane"]["target_bot_ids"] == [bot_id]
