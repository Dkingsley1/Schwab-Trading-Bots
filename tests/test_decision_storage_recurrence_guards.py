from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from core import accountability, base_trader
from core.decision_logger import compact_decision_features
from scripts import run_shadow_training_loop
from scripts.ops import decision_log_compactor
from scripts.ops import hot_lane_retention_control
from scripts.ops import ingestion_storage_governor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sub_bot_essential_features_reference_lossless_snapshot() -> None:
    features = {f"feature_{index:04d}_risk_norm": float(index) for index in range(983)}
    features.update({"last_price": 101.5, "mom_5m": 0.01})

    compacted, contract = compact_decision_features(
        features,
        metadata={"layer": "sub_bot", "snapshot_id": "snap-123"},
        mode="essential",
    )

    assert len(compacted) <= 128
    assert compacted["last_price"] == 101.5
    assert contract["effective_mode"] == "essential"
    assert contract["omitted_feature_count"] > 800
    assert contract["feature_snapshot_id"] == "snap-123"
    assert contract["lossless_source"] == "primary_decision_with_matching_snapshot_id"


def test_primary_decision_keeps_lossless_features_in_essential_mode() -> None:
    features = {f"feature_{index}": float(index) for index in range(200)}

    for layer in (
        "grand_master",
        "master_bot",
        "options_master",
        "master_options",
        "futures_master",
        "master_futures",
    ):
        compacted, contract = compact_decision_features(
            features,
            metadata={"layer": layer, "snapshot_id": "snap-primary"},
            mode="essential",
        )

        assert compacted == features
        assert contract["effective_mode"] == "full"
        assert contract["lossless"] is True


def test_runtime_profiles_keep_payload_bounds_enabled() -> None:
    critical = ingestion_storage_governor._profile_env("critical_backpressure", Path("/tmp/project"))
    elevated = ingestion_storage_governor._profile_env("elevated_backpressure", Path("/tmp/project"))
    normal = ingestion_storage_governor._profile_env("steady_state", Path("/tmp/project"))
    full_audit = hot_lane_retention_control._env_for_mode(
        "full_decision_evidence",
        reasons=[],
        top_lanes=[],
    )

    assert critical["DECISION_LOG_FEATURE_MODE"] == "minimal"
    assert elevated["DECISION_LOG_FEATURE_MODE"] == "essential"
    assert normal["DECISION_LOG_FEATURE_MODE"] == "essential"
    assert full_audit["DECISION_LOG_FEATURE_MODE"] == "essential"
    assert full_audit["DECISION_EXPLANATION_FEATURE_MODE"] == "minimal"


def test_hot_lane_override_wins_shared_logging_controls(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    config.mkdir(parents=True)
    pressure = config / ".env.storage_pressure_override"
    hot_lane = config / ".env.hot_lane_retention_override"
    pressure.write_text(
        "DECISION_LOG_FEATURE_MODE=essential\nLOG_DATA_INGRESS=1\n",
        encoding="utf-8",
    )
    hot_lane.write_text(
        "DECISION_LOG_FEATURE_MODE=minimal\nLOG_DATA_INGRESS=0\n",
        encoding="utf-8",
    )

    base_trader._DYNAMIC_STORAGE_OVERRIDE_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    base_values = base_trader._dynamic_storage_overrides(str(tmp_path))
    assert base_values["DECISION_LOG_FEATURE_MODE"] == "minimal"
    assert base_values["LOG_DATA_INGRESS"] == "0"

    monkeypatch.setattr(
        run_shadow_training_loop,
        "DYNAMIC_STORAGE_OVERRIDE_PATHS",
        (pressure, hot_lane),
    )
    run_shadow_training_loop._DYNAMIC_STORAGE_OVERRIDE_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    loop_values = run_shadow_training_loop._dynamic_storage_overrides()
    assert loop_values["DECISION_LOG_FEATURE_MODE"] == "minimal"
    assert loop_values["LOG_DATA_INGRESS"] == "0"


def test_active_backpressure_logging_controls_win_hot_lane_restore(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    config.mkdir(parents=True)
    pressure = config / ".env.storage_pressure_override"
    hot_lane = config / ".env.hot_lane_retention_override"
    pressure.write_text(
        "RAW_LIVE_EXPANSION_GUARD_ACTIVE=1\n"
        "LOG_SUB_BOT_DECISIONS=0\n"
        "DECISION_LOG_FEATURE_MODE=minimal\n"
        "SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS=8\n",
        encoding="utf-8",
    )
    hot_lane.write_text(
        "LOG_SUB_BOT_DECISIONS=1\n"
        "DECISION_LOG_FEATURE_MODE=essential\n"
        "SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS=1\n",
        encoding="utf-8",
    )

    base_trader._DYNAMIC_STORAGE_OVERRIDE_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    base_values = base_trader._dynamic_storage_overrides(str(tmp_path))
    assert base_values["LOG_SUB_BOT_DECISIONS"] == "0"
    assert base_values["DECISION_LOG_FEATURE_MODE"] == "minimal"
    assert base_values["SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS"] == "8"

    monkeypatch.setattr(run_shadow_training_loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (pressure, hot_lane))
    run_shadow_training_loop._DYNAMIC_STORAGE_OVERRIDE_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    loop_values = run_shadow_training_loop._dynamic_storage_overrides()
    assert loop_values["LOG_SUB_BOT_DECISIONS"] == "0"
    assert loop_values["DECISION_LOG_FEATURE_MODE"] == "minimal"
    assert loop_values["SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS"] == "8"

    accountability._DYNAMIC_RUNTIME_CONTROL_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    assert accountability._dynamic_runtime_control(
        str(tmp_path), "LOG_SUB_BOT_DECISIONS", "1"
    ) == "0"
    assert accountability._dynamic_runtime_control(
        str(tmp_path), "SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS", "1"
    ) == "8"


def test_pressure_sub_bot_signal_sampling_is_deterministic(monkeypatch) -> None:
    controls = {
        "SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS": "8",
        "SIGNAL_GENERATION_SUB_BOT_WINDOW_SECONDS": "3600",
    }
    monkeypatch.setattr(
        accountability,
        "_dynamic_runtime_control",
        lambda _root, name, default: controls.get(name, default),
    )
    monkeypatch.setattr(accountability.time, "time", lambda: 7_200.0)
    payloads = [
        {
            "symbol": "SPY",
            "action": "HOLD",
            "status": "HOLD",
            "strategy": f"brain_refinery_v{index}",
            "metadata": {"layer": "sub_bot"},
        }
        for index in range(128)
    ]

    accountability._LOW_SIGNAL_RECENT.clear()
    first = [
        accountability._should_emit_signal_generation_event(
            payload,
            classification="bad_signal",
            reason="hold_or_no_trade_signal",
            project_root="/tmp/project",
        )
        for payload in payloads
    ]
    accountability._LOW_SIGNAL_RECENT.clear()
    second = [
        accountability._should_emit_signal_generation_event(
            payload,
            classification="bad_signal",
            reason="hold_or_no_trade_signal",
            project_root="/tmp/project",
        )
        for payload in payloads
    ]

    assert first == second
    assert 4 <= sum(first) <= 32
    assert accountability._should_emit_signal_generation_event(
        {
            "symbol": "SPY",
            "action": "BUY",
            "status": "PAPER_EXECUTED",
            "strategy": "always_keep_execution_outcome",
            "metadata": {"layer": "sub_bot"},
        },
        classification="good_signal",
        reason="trade_intent_generated",
        project_root="/tmp/project",
    ) is True


def test_hot_lane_reports_managed_state_without_hiding_raw_pressure(tmp_path: Path, monkeypatch) -> None:
    external_root = tmp_path / "external"
    external_root.mkdir()
    storage_tier_path = tmp_path / "governance" / "health" / "storage_tier_policy_latest.json"
    _write_json(
        storage_tier_path,
        {
            "overall_status": "ready",
            "timestamp_utc": "2026-08-04T17:00:00+00:00",
            "pressure": {
                "live_hot_path_bytes": 30 * 1024**3,
                "hot_path_over_budget_bytes": 0,
            },
        },
    )
    monkeypatch.setattr(
        hot_lane_retention_control,
        "resolve_external_storage",
        lambda: SimpleNamespace(external_root=external_root),
    )
    monkeypatch.setattr(
        hot_lane_retention_control,
        "_disk_snapshot",
        lambda path: {
            "path": str(path),
            "checked_path": str(path),
            "protected": False,
            "total_gb": 500.0,
            "used_gb": 250.0,
            "free_gb": 250.0,
            "used_percent": 50.0,
        },
    )
    monkeypatch.setattr(
        hot_lane_retention_control,
        "_scan_decision_logs",
        lambda *args, **kwargs: [
            {
                "relative_path": "decisions/crypto/trade_decisions_20260804.jsonl",
                "size_bytes": 25 * 1024**3,
                "size_gb": 25.0,
                "is_current_day": True,
                "protected": False,
            }
        ],
    )
    monkeypatch.setattr(hot_lane_retention_control, "_scan_sql_shards", lambda *args, **kwargs: [])

    payload = hot_lane_retention_control.build_payload(
        tmp_path,
        apply=True,
        target_free_gb=125.0,
        pressure_free_gb=64.0,
        hot_total_thin_gb=10.0,
        hot_file_thin_gb=10.0,
        restore_total_gb=4.0,
        restore_file_gb=2.0,
        out_path=tmp_path / "health.json",
        override_path=tmp_path / "config" / ".env.hot_lane_retention_override",
        storage_tier_path=storage_tier_path,
    )

    assert payload["raw_pressure_status"] == "critical"
    assert payload["raw_pressure_grade"] == "C"
    assert payload["overall_status"] == "active"
    assert payload["overall_grade"] == "A"
    assert payload["containment_status"] == "ready"
    assert payload["containment"]["raw_incident_debt_visible"] is True
    assert payload["override_applied"] is True


def test_hot_lane_noop_apply_preserves_control_epoch(tmp_path: Path) -> None:
    path = tmp_path / ".env.hot_lane_retention_override"
    env = {"HOT_LANE_RETENTION_ACTIVE": "1", "DECISION_LOG_FEATURE_MODE": "minimal"}

    assert hot_lane_retention_control._write_override(
        path,
        env,
        payload={"timestamp_utc": "2026-08-04T17:44:47+00:00"},
    ) is True
    first_content = path.read_text(encoding="utf-8")
    assert hot_lane_retention_control._write_override(
        path,
        env,
        payload={"timestamp_utc": "2026-08-04T18:44:47+00:00"},
    ) is False
    assert path.read_text(encoding="utf-8") == first_content


def test_sub_bot_signal_generation_is_time_window_sampled(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "config" / ".env.storage_pressure_override"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("SIGNAL_GENERATION_SUB_BOT_WINDOW_SECONDS=3600\n", encoding="utf-8")
    accountability._DYNAMIC_RUNTIME_CONTROL_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    accountability._LOW_SIGNAL_RECENT.clear()
    monkeypatch.setattr(accountability.time, "time", lambda: 1000.0)
    payload = {
        "symbol": "BTC-USD",
        "action": "BUY",
        "strategy": "bot-1",
        "metadata": {"layer": "sub_bot"},
    }

    first = accountability._should_emit_signal_generation_event(
        payload,
        classification="good_signal",
        reason="trade_intent_generated",
        project_root=str(tmp_path),
    )
    second = accountability._should_emit_signal_generation_event(
        payload,
        classification="good_signal",
        reason="trade_intent_generated",
        project_root=str(tmp_path),
    )

    assert first is True
    assert second is False


def test_derived_master_signal_generation_is_sampled_but_execution_is_lossless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    override = tmp_path / "config" / ".env.hot_lane_retention_override"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("SIGNAL_GENERATION_DERIVED_WINDOW_SECONDS=900\n", encoding="utf-8")
    accountability._DYNAMIC_RUNTIME_CONTROL_CACHE.update(
        {"checked_at_monotonic": 0.0, "fingerprint": (), "values": {}}
    )
    accountability._LOW_SIGNAL_RECENT.clear()
    monkeypatch.setattr(accountability.time, "time", lambda: 1000.0)
    shadow = {
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "grand_master_bot",
        "metadata": {"layer": "grand_master"},
    }
    executed = {**shadow, "status": "PAPER_EXECUTED"}

    assert accountability._should_emit_signal_generation_event(
        shadow,
        classification="good_signal",
        reason="trade_intent_generated",
        project_root=str(tmp_path),
    ) is True
    assert accountability._should_emit_signal_generation_event(
        shadow,
        classification="good_signal",
        reason="trade_intent_generated",
        project_root=str(tmp_path),
    ) is False
    assert accountability._should_emit_signal_generation_event(
        executed,
        classification="good_signal",
        reason="trade_intent_generated",
        project_root=str(tmp_path),
    ) is True
    assert accountability._should_emit_signal_generation_event(
        executed,
        classification="good_signal",
        reason="trade_intent_generated",
        project_root=str(tmp_path),
    ) is True


def test_master_control_infrastructure_rows_are_bounded_after_full_evaluation() -> None:
    rows = [
        {
            "bot_id": f"infra_{index:04d}",
            "action": "SELL" if index % 2 else "HOLD",
            "score": 0.2 + (index % 10) / 100.0,
            "threshold": 0.55,
            "eligible_for_master_vote": index < 6,
            "lifecycle_state": "active",
            "observer_meta": {f"metric_{offset}": float(offset) for offset in range(24)},
        }
        for index in range(942)
    ]

    retained, contract = run_shadow_training_loop._compact_infrastructure_governance_rows(
        rows,
        max_rows=16,
    )

    assert len(retained) == 16
    assert sum(bool(row["eligible_for_master_vote"]) for row in retained) == 6
    assert contract["source_row_count"] == 942
    assert contract["omitted_row_count"] == 926
    assert contract["all_vote_eligible_rows_retained"] is True
    assert contract["decision_evaluation_used_full_rows"] is True
    assert contract["paper_execution_behavior_changed"] is False
    assert len(json.dumps({"rows": retained, "contract": contract})) < 20_000


def _current_explanation_fixture(tmp_path: Path, *, fully_ingested: bool) -> Path:
    day = "20260804"
    source = (
        tmp_path
        / "decision_explanations"
        / "shadow_crypto"
        / f"decision_explanations_{day}.jsonl"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('{"symbol":"BTC-USD","action":"HOLD"}\n' * 32, encoding="utf-8")
    stat = source.stat()
    rel = str(source.relative_to(tmp_path))
    progress = {
        "last_line": 32 if fully_ingested else 16,
        "last_offset_bytes": stat.st_size if fully_ingested else stat.st_size // 2,
        "mtime": stat.st_mtime,
        "file_inode": stat.st_ino,
        "file_size_bytes": stat.st_size,
    }
    _write_json(
        tmp_path / "governance" / "sql_link_shards" / "jsonl_sql_link_state_crypto_explanations.json",
        {"sqlite": {rel: progress}},
    )
    override = tmp_path / "config" / ".env.storage_pressure_override"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("LOG_DECISION_EXPLANATIONS=0\n", encoding="utf-8")
    return source


def test_current_day_compaction_requires_exact_sql_eof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(decision_log_compactor, "_today_stamp", lambda: "20260804")
    _current_explanation_fixture(tmp_path, fully_ingested=False)

    payload = decision_log_compactor.build_payload(
        project_root=tmp_path,
        apply=False,
        include_current_day=True,
        families=["decision_explanations"],
        min_file_mb=0.000001,
        min_age_minutes=0,
        target_free_gb=0,
    )

    assert payload["summary"]["candidate_count"] == 0


def test_current_day_compaction_rotates_only_inert_fully_ingested_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(decision_log_compactor, "_today_stamp", lambda: "20260804")
    source = _current_explanation_fixture(tmp_path, fully_ingested=True)

    payload = decision_log_compactor.build_payload(
        project_root=tmp_path,
        apply=True,
        include_current_day=True,
        families=["decision_explanations"],
        min_file_mb=0.000001,
        min_age_minutes=0,
        target_free_gb=0,
        compression_level=1,
    )

    archive = source.with_name(source.name + ".gz")
    assert payload["overall_status"] == "applied"
    assert payload["records"][0]["source_fingerprint_verified"] is True
    assert not source.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        assert handle.read().count("\n") == 32
