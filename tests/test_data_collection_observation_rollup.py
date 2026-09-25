import gzip
import fcntl
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import data_collection_observation_rollup as src


def test_incremental_reads_resume_only_after_complete_rows(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    row = (json.dumps({"value": "caf\u00e9"}, ensure_ascii=False) + "\n").encode()
    path.write_bytes(row * 3 + row[:8])
    monkeypatch.setattr(src, "MAX_SOURCE_READ_BYTES", len(row) + 5)
    offset = 0
    for _ in range(3):
        budget = src.ObservationScanBudget(max_bytes=1000)
        lines, end, _ = src._iter_new_lines(path, offset=offset, budget=budget)
        assert lines == [row.decode()]
        assert end == offset + len(row)
        assert budget.bytes_read <= len(row) + 5
        offset = end
    lines, end, _ = src._iter_new_lines(path, offset=offset)
    assert lines == [] and end == offset
    with path.open("ab") as handle:
        handle.write(row[8:])
    lines, end, _ = src._iter_new_lines(path, offset=offset)
    assert lines == [row.decode()] and end == 4 * len(row)


def test_bootstrap_cursor_comes_from_same_bounded_tail_read(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a":1}\n{"a":2}\n{"a":')
    budget = src.ObservationScanBudget(max_bytes=100)
    audit = {}
    lines = src._iter_tail_lines(path, limit=10, budget=budget, read_audit=audit)
    assert lines == ['{"a":1}\n', '{"a":2}\n']
    assert audit["complete_end"] == 16
    assert budget.bytes_read == path.stat().st_size


def test_gzip_expansion_cap_never_advances_cursor_or_counts_partial_decode(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b'{"a":1}\n' * 10000)
    monkeypatch.setattr(src, "MAX_SOURCE_READ_BYTES", 1024)
    budget = src.ObservationScanBudget(max_bytes=2048)
    lines, offset, line_offset = src._iter_new_lines(path, offset=0, budget=budget)
    assert (lines, offset, line_offset) == ([], 0, 0)
    assert budget.bytes_read == 1024
    assert budget.receipt()["limited_source_count"] == 1
    assert not budget.receipt()["scan_complete"]


def test_shared_scan_budget_bounds_json_and_decisions(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_bytes(b'{"ok":true}')
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"a":1}\n' * 100)
    budget = src.ObservationScanBudget(max_bytes=27)
    assert budget.load_json(meta) == {"ok": True}
    assert src._iter_new_lines(rows, offset=0, budget=budget)[1] == 16
    assert budget.bytes_read == 27
    assert budget.load_json(meta) == {}
    assert budget.bytes_read == 27


def test_expired_scan_does_not_read_source(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a":1}\n')
    budget = src.ObservationScanBudget(seconds=-1)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("read after deadline")))
    assert src._iter_new_lines(path, offset=0, budget=budget) == ([], 0, 0)
    assert budget.bytes_read == 0


def test_disappearing_source_does_not_advance_cursor_or_certify_empty_tail(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    budget = src.ObservationScanBudget()
    monkeypatch.setattr(budget, "present", lambda path: True)
    assert src._iter_new_lines(path, offset=14, line_offset=2, budget=budget) == ([], 14, 2)
    assert str(path) in budget.failed_sources


def test_discovery_rejects_protected_intermediate_alias_before_traversal(tmp_path, monkeypatch):
    root = tmp_path / "decision_explanations"
    root.mkdir()
    (root / "blocked").symlink_to("/Volumes/VIDEO/unavailable")
    original = Path.lstat

    def checked_lstat(path, *args, **kwargs):
        assert not str(path).startswith("/Volumes/VIDEO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked_lstat)
    assert src._decision_files(tmp_path, days=1) == []


def test_state_only_partial_scan_keeps_registry_and_exclusions_unchanged(tmp_path, monkeypatch):
    bot_id = "brain_refinery_v167_test_collector"
    registry_path = tmp_path / "master_bot_registry.json"
    state_path = tmp_path / "state.json"
    _write_json(registry_path, _registry(bot_id))
    _write_json(state_path, {"initialized": True})
    before = registry_path.read_bytes()
    stamp = src._day_stamps(1)[0]
    rows = tmp_path / "decision_explanations" / "lane" / f"decision_explanations_{stamp}.jsonl"
    rows.parent.mkdir(parents=True)
    line = json.dumps({"bot_id": bot_id}) + "\n"
    rows.write_text(line * 10)
    monkeypatch.setattr(src, "MAX_SOURCE_READ_BYTES", len(line) * 2 + 1)
    payload = src.build_payload(project_root=tmp_path, registry_path=registry_path,
        state_path=state_path, days=1, bootstrap_tail_lines=20, apply=True, state_only=True)
    assert registry_path.read_bytes() == before
    assert payload["registry_written"] is False
    assert payload["new_rows_counted"] == 2
    assert payload["overall_status"] == "degraded"
    assert payload["operational_ok"] is True
    assert payload["training_exclusion_releasable_count"] == 0
    state = json.loads(state_path.read_text())
    assert state["file_offsets"][str(rows.relative_to(tmp_path))] == len(line) * 2


def test_failed_gzip_cannot_certify_operational_readiness(tmp_path):
    bot_id = "brain_refinery_v167_test_collector"
    registry_path = tmp_path / "master_bot_registry.json"
    registry = _registry(bot_id)
    registry["sub_bots"][0]["data_collection_observations"] = 100
    _write_json(registry_path, registry)
    stamp = src._day_stamps(1)[0]
    rows = tmp_path / "decision_explanations" / "lane" / f"decision_explanations_{stamp}.jsonl.gz"
    rows.parent.mkdir(parents=True)
    rows.write_bytes(b"not gzip")
    payload = src.build_payload(project_root=tmp_path, registry_path=registry_path,
        state_path=tmp_path / "state.json", days=1, bootstrap_tail_lines=20, apply=False)
    assert payload["overall_status"] == "blocked"
    assert not payload["operational_ok"]
    assert payload["training_exclusion_releasable_count"] == 0


def test_main_singleflight_defers_without_overwriting_receipt(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    out = tmp_path / "out.json"
    out.write_text('{"previous":true}')
    lock_path = state.with_suffix(".json.lock")
    monkeypatch.setattr(sys, "argv", ["rollup", "--project-root", str(tmp_path),
        "--registry", str(tmp_path / "registry.json"), "--state-file", str(state),
        "--out-file", str(out), "--apply", "--state-only", "--json"])
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert src.main() == 75
    assert out.read_text() == '{"previous":true}'
    assert not state.exists()


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
    assert payload["training_readiness_definition"] == "collection_gate_only_not_retrain_launch"
    assert payload["training_ready_bot_ids_truncated"] is False
    assert payload["training_stage_contract"]["authorizes_training_launch"] is False
    assert "bot_needs_candidate_selection" in payload["training_stage_contract"]["next_required_stages"]
    assert payload["registry_write_contract"]["atomic"] is True
    assert payload["registry_write_contract"]["candidate_fingerprint_normalized"] is True


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
    assert row["data_collection_threshold_progress"]["observation_floor_configured"] is False
    assert row["data_collection_threshold_progress"]["training_ready"] is False
    assert row["data_collection_training_ready"] is False
    assert row["training_excluded"] is True
    assert row["exclude_from_training"] is True
    assert row["training_exclusion_reason"] == "paper_live_data_requires_minimum_training_observations"
    assert row["promotion_block_reason"] == "awaiting_data_collection_quality_gate"
    assert payload["training_ready_count"] == 0
    assert payload["training_ready_bot_ids"] == []
    assert registry["summary"]["data_collection_training_ready_bots"] == 0


def test_observation_rollup_keeps_collect_only_bot_blocked_without_observation_floor(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v98_crypto_execution_throttle_reentry"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry(bot_id)
    registry["sub_bots"][0].pop("minimum_training_observations", None)
    _write_json(registry_path, registry)
    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow_crypto" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            [
                json.dumps({"status": "DATA_ONLY_BLOCKED", "metadata": {"bot_id": bot_id}}),
                json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": bot_id}}),
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
    row = json.loads(registry_path.read_text(encoding="utf-8"))["sub_bots"][0]

    assert row["data_collection_observations"] == 2
    assert row["data_collection_threshold_progress"]["observation_floor_configured"] is False
    assert row["data_collection_threshold_progress"]["training_ready"] is False
    assert row["data_collection_training_ready"] is False
    assert row["training_excluded"] is True
    assert row["training_exclusion_reason"] == "data_collection_requires_minimum_training_observations"
    assert payload["training_ready_count"] == 0


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


def test_observation_rollup_projects_broad_collection_ready_with_fail_closed_sample_debt(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    registry = _registry("brain_refinery_v101_observed")
    for index in range(102, 105):
        row = dict(registry["sub_bots"][0])
        row["bot_id"] = f"brain_refinery_v{index}_observed"
        registry["sub_bots"].append(row)
    debt = registry["sub_bots"][-1]
    debt["bot_id"] = "brain_refinery_v11_stoch_vol"
    debt["training_exclusion_reason"] = "sample_starved_requalification_collect_only"
    debt["promotion_block_reason"] = "sample_starved_requalification_no_promotion"
    debt["trading_enabled"] = False
    debt["paper_trading_enabled"] = False
    debt["live_trading_enabled"] = False
    debt["execution_enabled"] = False
    _write_json(registry_path, registry)

    stamp = src._day_stamps(1)[0]
    decision_file = project_root / "decision_explanations" / "shadow" / f"decision_explanations_{stamp}.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            json.dumps({"status": "HOLD", "bot_id": row["bot_id"]})
            for row in registry["sub_bots"][:3]
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

    assert payload["overall_status"] == "degraded"
    assert payload["unmanaged_zero_observation_count"] == 1
    assert payload["fail_closed_zero_observation_count"] == 1
    assert payload["operational_status"] == "ready"
    assert payload["operational_ok"] is True
    assert payload["operational_collection"]["raw_status_preserved"] is True
