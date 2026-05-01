import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import scripts.data_source_divergence_bot as divergence_bot
import scripts.run_shadow_training_loop as loop
import scripts.run_master_bot as run_master_bot
import scripts.weekly_retrain as weekly_retrain


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_divergence_payloads_split_bond_and_non_bond_scopes(tmp_path) -> None:
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    bond_file = tmp_path / "governance" / "shadow_bond_equities" / f"master_control_{day}.jsonl"
    aggressive_file = tmp_path / "governance" / "shadow_aggressive_equities" / f"master_control_{day}.jsonl"
    conservative_file = tmp_path / "governance" / "shadow_conservative_equities" / f"master_control_{day}.jsonl"

    _write_jsonl(
        bond_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.00}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.08}},
        ],
    )
    _write_jsonl(
        aggressive_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 1111.00}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 1112.00}},
        ],
    )
    _write_jsonl(
        conservative_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.10}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.20}},
        ],
    )

    payload, scopes = divergence_bot.build_divergence_payloads(tmp_path, hours=2, max_relative_spread=0.03)

    assert payload["ok"] is True
    assert payload["compared_buckets"] > 0
    assert payload["worst_relative_spread"] > 0.03
    assert payload["cross_profile"]["ok"] is False
    assert payload["cross_profile"]["worst_relative_spread"] > 0.03
    assert scopes["bond_profile"]["ok"] is True
    assert scopes["non_bond_profiles"]["ok"] is True
    assert payload["cross_profile"]["compared_buckets"] > 0


def test_divergence_payloads_ignore_simulated_rows(tmp_path) -> None:
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    aggressive_file = tmp_path / "governance" / "shadow_aggressive_equities" / f"master_control_{day}.jsonl"
    conservative_file = tmp_path / "governance" / "shadow_conservative_equities" / f"master_control_{day}.jsonl"

    _write_jsonl(
        aggressive_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 1111.00, "spread_bps": 8.0, "bid_size": 1000.0, "ask_size": 1000.0}},
            {"timestamp_utc": ts, "symbol": "TLT", "simulate": True, "market": {"last_price": 1112.00}},
        ],
    )
    _write_jsonl(
        conservative_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.10}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.20}},
        ],
    )

    payload, scopes = divergence_bot.build_divergence_payloads(tmp_path, hours=2, max_relative_spread=0.03)

    assert payload["ok"] is True
    assert payload["skipped_simulated_rows"] == 2
    assert scopes["non_bond_profiles"]["ok"] is True


def test_weekly_retrain_include_targets_preserves_requested_order() -> None:
    targets = [
        "/tmp/core/brain_refinery_v96_credit_spread_rotation_bot.py",
        "/tmp/core/brain_refinery_v95_rates_regime_bond_bot.py",
        "/tmp/core/brain_refinery_v92_macro_rates_curve_regime.py",
    ]

    selected = weekly_retrain._apply_included_bot_ids(
        targets,
        "brain_refinery_v92_macro_rates_curve_regime,brain_refinery_v95_rates_regime_bond_bot",
    )

    assert selected == [
        "/tmp/core/brain_refinery_v92_macro_rates_curve_regime.py",
        "/tmp/core/brain_refinery_v95_rates_regime_bond_bot.py",
    ]


def test_weekly_retrain_explicit_include_allows_deleted_targets(tmp_path) -> None:
    core_dir = tmp_path / "core"
    registry_path = tmp_path / "master_bot_registry.json"
    requested = "brain_refinery_v44_intraday_scalp_1m_5m"

    core_dir.mkdir(parents=True, exist_ok=True)
    (core_dir / f"{requested}.py").write_text("BOT_ID='brain_refinery_v44_intraday_scalp_1m_5m'\n", encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": requested,
                        "deleted_from_rotation": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    original_core_dir = weekly_retrain.CORE_DIR
    original_registry_path = weekly_retrain.REGISTRY_PATH
    try:
        weekly_retrain.CORE_DIR = str(core_dir)
        weekly_retrain.REGISTRY_PATH = str(registry_path)

        default_targets = weekly_retrain.build_targets(include_deleted=False)
        explicit_targets = weekly_retrain.build_targets(
            include_deleted=weekly_retrain._should_include_deleted_targets(
                weekly_retrain.argparse.Namespace(include_deleted=False),
                explicit_include_requested=True,
            )
        )

        assert default_targets == []
        assert weekly_retrain._apply_included_bot_ids(explicit_targets, requested) == [
            str(core_dir / f"{requested}.py")
        ]
    finally:
        weekly_retrain.CORE_DIR = original_core_dir
        weekly_retrain.REGISTRY_PATH = original_registry_path


def test_weekly_retrain_force_all_targets_allows_deleted_targets() -> None:
    args = weekly_retrain.argparse.Namespace(include_deleted=False, force_all_targets=True)

    assert weekly_retrain._should_include_deleted_targets(args, explicit_include_requested=False) is True


def test_weekly_retrain_resolves_bond_divergence_scope() -> None:
    path, scope = weekly_retrain._resolve_data_divergence_file("bond", "/tmp/fallback.json")
    assert scope == "bond_profile"
    assert path.endswith("data_source_divergence_bond_latest.json")

    path2, scope2 = weekly_retrain._resolve_data_divergence_file("non_bond", "/tmp/fallback.json")
    assert scope2 == "non_bond_profiles"
    assert path2.endswith("data_source_divergence_non_bond_latest.json")


def test_weekly_retrain_segment_keywords_cover_defensive_event_and_liquidity_bots() -> None:
    assert weekly_retrain._segment_bot_id("brain_refinery_v31_defensive_rotation") == "mean_revert"
    assert weekly_retrain._segment_bot_id("brain_refinery_v27_term_structure_vol") == "shock"
    assert weekly_retrain._segment_bot_id("brain_refinery_v48_position_1m_3m") == "liquidity"


def test_weekly_retrain_operator_notes_can_drive_regime_focus(tmp_path) -> None:
    note_path = tmp_path / "retrain_operator_notes_latest.json"
    note_path.write_text(
        json.dumps(
            {
                "title": "Operator note",
                "summary": "Current regime is guard-heavy with defensive dividend repeat and crypto throttle behavior.",
                "tags": ["guard_heavy_regime", "defensive_dividend_repeat", "crypto_throttle_repeat"],
                "observations": ["Futures event-risk keeps repeating."],
                "training_guidance": ["Treat repeated crypto throttle blocks as a risk-control pattern worth learning from."],
            }
        ),
        encoding="utf-8",
    )

    assert weekly_retrain._derive_regime_focus_from_operator_notes(str(note_path), top_n=3) == "shock,mean_revert,liquidity"


def test_weekly_retrain_parses_json_from_noisy_runtime_snapshot_output() -> None:
    payload = weekly_retrain._parse_json_output(
        "Runtime training snapshot: starting\n"
        '{"ok": true, "sequence_count": 4, "row_count": 512}\n'
    )

    assert payload["ok"] is True
    assert payload["sequence_count"] == 4
    assert payload["row_count"] == 512


def test_weekly_retrain_targeted_queue_preserves_explicit_targets_without_auto_reshaping() -> None:
    v26 = "/tmp/core/brain_refinery_v26_relative_strength_cross_section.py"
    v48 = "/tmp/core/brain_refinery_v48_position_1m_3m.py"
    v12 = "/tmp/core/brain_refinery_v12_news_shocks.py"

    reshaped, canary_selected, distill_selected = weekly_retrain._reshape_target_queue(
        [v26, v48, v12, v26],
        allow_auto_queue_reshaping=False,
        regime_focus="trend",
        regime_balance=True,
        exclude_bot_ids="",
        canary_priority_file="/tmp/missing_canary.json",
        canary_priority_top_n=10,
        distillation_priority=True,
        distill_assign_map={"brain_refinery_v26_relative_strength_cross_section": {"student_bot_id": "brain_refinery_v26_relative_strength_cross_section"}},
        distillation_extra_pass=2,
        new_bot_boost=True,
        new_bot_targets=[v26, v48],
        new_bot_extra_pass=2,
    )

    assert reshaped == [v26, v48, v12]
    assert canary_selected == 0
    assert distill_selected == 0


def test_weekly_retrain_insufficient_data_retry_overrides_expand_scope() -> None:
    first = weekly_retrain._insufficient_data_retry_overrides(
        "/tmp/core/brain_refinery_v43_intraday_ultrafast_proxy.py",
        0,
    )
    second = weekly_retrain._insufficient_data_retry_overrides(
        "/tmp/core/brain_refinery_v43_intraday_ultrafast_proxy.py",
        1,
    )

    assert int(first["RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE"]) == 1
    assert int(first["RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE"]) >= 28
    assert int(second["RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE"]) >= int(first["RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE"])


def test_weekly_retrain_auto_queue_dedupes_extra_pass_targets() -> None:
    v35 = "/tmp/core/brain_refinery_v35_dmi_state_machine.py"
    v56 = "/tmp/core/brain_refinery_v56_meta_ranker.py"

    reshaped, canary_selected, distill_selected = weekly_retrain._reshape_target_queue(
        [v35, v56],
        allow_auto_queue_reshaping=True,
        regime_focus="",
        regime_balance=False,
        exclude_bot_ids="",
        canary_priority_file="/tmp/missing_canary.json",
        canary_priority_top_n=10,
        distillation_priority=True,
        distill_assign_map={"brain_refinery_v35_dmi_state_machine": {"student_bot_id": "brain_refinery_v35_dmi_state_machine"}},
        distillation_extra_pass=2,
        new_bot_boost=True,
        new_bot_targets=[v35],
        new_bot_extra_pass=2,
    )

    assert reshaped == [v35, v56]
    assert canary_selected == 0
    assert distill_selected == 1


def test_weekly_retrain_efficiency_filter_skips_low_readiness_restores(tmp_path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v44_intraday_scalp_1m_5m",
                        "active": True,
                        "reason": "manual_canary_restore:day_swing_lane_expand",
                        "quality_score": 0.21,
                        "test_accuracy": 0.50,
                        "no_improvement_streak": 3,
                    },
                    {
                        "bot_id": "brain_refinery_v69_cost_aware_execution_filter",
                        "active": True,
                        "reason": "min_active_floor_override_30:target_30_active",
                        "quality_score": 0.46,
                        "test_accuracy": 0.52,
                        "no_improvement_streak": 1,
                    },
                    {
                        "bot_id": "brain_refinery_v35_dmi_state_machine",
                        "active": True,
                        "reason": "bucket_diversity_trend",
                        "quality_score": 0.99,
                        "test_accuracy": 0.85,
                        "no_improvement_streak": 0,
                    },
                    {
                        "bot_id": "brain_refinery_v12_news_shocks",
                        "active": True,
                        "reason": "bucket_diversity_shock",
                        "quality_score": 0.20,
                        "test_accuracy": 0.47,
                        "no_improvement_streak": 1,
                    },
                    {
                        "bot_id": "brain_refinery_v27_term_structure_vol",
                        "active": True,
                        "reason": "protected_collection_floor_options",
                        "quality_score": 0.22,
                        "test_accuracy": 0.50,
                        "no_improvement_streak": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    original_registry_path = weekly_retrain.REGISTRY_PATH
    try:
        weekly_retrain.REGISTRY_PATH = str(registry_path)
        filtered, stats = weekly_retrain._filter_targets_for_efficiency(
            [
                "/tmp/core/brain_refinery_v44_intraday_scalp_1m_5m.py",
                "/tmp/core/brain_refinery_v69_cost_aware_execution_filter.py",
                "/tmp/core/brain_refinery_v35_dmi_state_machine.py",
                "/tmp/core/brain_refinery_v12_news_shocks.py",
                "/tmp/core/brain_refinery_v27_term_structure_vol.py",
            ],
            active_only=True,
            max_targets=10,
            min_model_age_hours=0.0,
            skip_low_readiness=True,
        )
        force_all_filtered, force_all_stats = weekly_retrain._filter_targets_for_efficiency(
            [
                "/tmp/core/brain_refinery_v44_intraday_scalp_1m_5m.py",
                "/tmp/core/brain_refinery_v69_cost_aware_execution_filter.py",
                "/tmp/core/brain_refinery_v35_dmi_state_machine.py",
                "/tmp/core/brain_refinery_v12_news_shocks.py",
                "/tmp/core/brain_refinery_v27_term_structure_vol.py",
            ],
            active_only=False,
            max_targets=0,
            min_model_age_hours=0.0,
            skip_low_readiness=False,
        )
    finally:
        weekly_retrain.REGISTRY_PATH = original_registry_path

    assert filtered == ["/tmp/core/brain_refinery_v35_dmi_state_machine.py"]
    assert stats["low_readiness_skipped"] == 4
    assert set(force_all_filtered) == {
        "/tmp/core/brain_refinery_v44_intraday_scalp_1m_5m.py",
        "/tmp/core/brain_refinery_v69_cost_aware_execution_filter.py",
        "/tmp/core/brain_refinery_v35_dmi_state_machine.py",
        "/tmp/core/brain_refinery_v12_news_shocks.py",
        "/tmp/core/brain_refinery_v27_term_structure_vol.py",
    }
    assert force_all_stats["post"] == 5
    assert force_all_stats["low_readiness_skipped"] == 0


def test_weekly_retrain_retry_pack_priority_biases_queue(tmp_path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "sub_bots": [
                    {"bot_id": "brain_refinery_v35_dmi_state_machine", "active": True},
                    {"bot_id": "brain_refinery_v75_model_drift_guard", "active": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    retry_pack_path = tmp_path / "governance" / "health" / "retrain_retry_pack_latest.json"
    retry_pack_path.parent.mkdir(parents=True, exist_ok=True)
    retry_pack_path.write_text(
        json.dumps({"include_bot_ids": ["brain_refinery_v75_model_drift_guard"]}),
        encoding="utf-8",
    )
    walk_forward_path = tmp_path / "governance" / "walk_forward" / "walk_forward_latest.json"
    walk_forward_path.parent.mkdir(parents=True, exist_ok=True)
    walk_forward_path.write_text(json.dumps({"bots": {}}), encoding="utf-8")

    original_registry_path = weekly_retrain.REGISTRY_PATH
    original_retry_pack = weekly_retrain.RETRAIN_RETRY_PACK_LATEST
    original_walk_forward = weekly_retrain.WALK_FORWARD_LATEST
    original_latest_model_age = weekly_retrain._latest_model_age_hours
    try:
        weekly_retrain.REGISTRY_PATH = str(registry_path)
        weekly_retrain.RETRAIN_RETRY_PACK_LATEST = str(retry_pack_path)
        weekly_retrain.WALK_FORWARD_LATEST = str(walk_forward_path)
        weekly_retrain._latest_model_age_hours = lambda _bot_id: 24.0
        filtered, stats = weekly_retrain._filter_targets_for_efficiency(
            [
                "/tmp/core/brain_refinery_v35_dmi_state_machine.py",
                "/tmp/core/brain_refinery_v75_model_drift_guard.py",
            ],
            active_only=True,
            max_targets=2,
            min_model_age_hours=0.0,
            skip_low_readiness=False,
        )
    finally:
        weekly_retrain.REGISTRY_PATH = original_registry_path
        weekly_retrain.RETRAIN_RETRY_PACK_LATEST = original_retry_pack
        weekly_retrain.WALK_FORWARD_LATEST = original_walk_forward
        weekly_retrain._latest_model_age_hours = original_latest_model_age

    assert filtered[0] == "/tmp/core/brain_refinery_v75_model_drift_guard.py"
    assert stats["retry_priority_selected"] == 1


def test_weekly_retrain_fast_daytime_profile_disables_heavy_extras() -> None:
    args = weekly_retrain.argparse.Namespace(
        retrain_profile="fast_daytime",
        counterfactual_replay=True,
        paper_hard_example_pack=True,
        cold_lane_retrain_extras=True,
        require_sample_quotas=True,
        new_bot_boost=True,
        build_runtime_training_snapshot=False,
        runtime_training_snapshot_prefer_sqlite=False,
        runtime_train_use_snapshot=False,
        runtime_train_prefer_sqlite=False,
        runtime_train_fast_fail_zero_sample_attempts=0,
        target_timeout_seconds=0,
    )

    profile = weekly_retrain._apply_retrain_profile_defaults(args)

    assert profile == "fast_daytime"
    assert args.counterfactual_replay is False
    assert args.paper_hard_example_pack is False
    assert args.require_sample_quotas is False
    assert args.new_bot_boost is False
    assert args.build_runtime_training_snapshot is True
    assert args.runtime_train_use_snapshot is True
    assert args.runtime_train_prefer_sqlite is True
    assert args.runtime_train_fast_fail_zero_sample_attempts == 2
    assert args.cold_lane_retrain_extras is False


def test_weekly_retrain_canary_profile_disables_snapshot_and_sets_timeout() -> None:
    args = weekly_retrain.argparse.Namespace(
        retrain_profile="canary",
        include_bot_ids="brain_refinery_v4_simple,brain_refinery_v13_choppy",
        counterfactual_replay=True,
        paper_hard_example_pack=True,
        cold_lane_retrain_extras=True,
        require_sample_quotas=True,
        new_bot_boost=True,
        build_runtime_training_snapshot=True,
        runtime_training_snapshot_prefer_sqlite=True,
        runtime_train_use_snapshot=True,
        runtime_train_prefer_sqlite=True,
        runtime_train_fast_fail_zero_sample_attempts=0,
        target_timeout_seconds=0,
    )

    profile = weekly_retrain._apply_retrain_profile_defaults(args)

    assert profile == "canary"
    assert args.counterfactual_replay is False
    assert args.paper_hard_example_pack is False
    assert args.build_runtime_training_snapshot is False
    assert args.runtime_train_use_snapshot is False
    assert args.runtime_train_prefer_sqlite is False
    assert args.runtime_train_fast_fail_zero_sample_attempts == 2
    assert args.target_timeout_seconds == 900
    assert args.cold_lane_retrain_extras is False


def test_weekly_retrain_coverage_canary_profile_clamps_runtime_inputs() -> None:
    args = weekly_retrain.argparse.Namespace(
        retrain_profile="coverage_canary",
        include_bot_ids="brain_refinery_v35_dmi_state_machine,brain_refinery_v4_simple",
        counterfactual_replay=True,
        paper_hard_example_pack=True,
        cold_lane_retrain_extras=True,
        require_sample_quotas=True,
        new_bot_boost=True,
        build_runtime_training_snapshot=True,
        runtime_training_snapshot_prefer_sqlite=True,
        runtime_train_use_snapshot=True,
        runtime_train_prefer_sqlite=True,
        runtime_train_fast_fail_zero_sample_attempts=0,
        target_timeout_seconds=0,
        auto_insufficient_data_retry=True,
    )

    profile = weekly_retrain._apply_retrain_profile_defaults(args)

    assert profile == "coverage_canary"
    assert args.counterfactual_replay is False
    assert args.paper_hard_example_pack is False
    assert args.build_runtime_training_snapshot is False
    assert args.runtime_train_use_snapshot is False
    assert args.runtime_train_prefer_sqlite is False
    assert args.runtime_train_fast_fail_zero_sample_attempts == 2
    assert args.target_timeout_seconds == 600
    assert args.cold_lane_retrain_extras is False
    assert args.auto_insufficient_data_retry is False


def test_weekly_retrain_coverage_canary_profile_can_use_external_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_TRAIN_SNAPSHOT_FILE", "/tmp/runtime_training_snapshot_v13_repair_latest.json")
    args = weekly_retrain.argparse.Namespace(
        retrain_profile="coverage_canary",
        include_bot_ids="brain_refinery_v13_choppy",
        counterfactual_replay=True,
        paper_hard_example_pack=True,
        cold_lane_retrain_extras=True,
        require_sample_quotas=True,
        new_bot_boost=True,
        build_runtime_training_snapshot=True,
        runtime_training_snapshot_prefer_sqlite=True,
        runtime_train_use_snapshot=False,
        runtime_train_prefer_sqlite=False,
        runtime_train_fast_fail_zero_sample_attempts=0,
        target_timeout_seconds=0,
        auto_insufficient_data_retry=True,
    )

    profile = weekly_retrain._apply_retrain_profile_defaults(args)

    assert profile == "coverage_canary"
    assert args.build_runtime_training_snapshot is False
    assert args.runtime_train_use_snapshot is True
    assert args.runtime_train_prefer_sqlite is True
    assert args.target_timeout_seconds == 600


def test_weekly_retrain_coverage_canary_profile_env_overrides_allow_stride_one_recovery(monkeypatch) -> None:
    env = {
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "3",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE": "3",
        "RUNTIME_TRAIN_AUTOFIX_ALLOW_SYMBOL_SCOPE_BROADEN": "0",
        "RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA": "0",
    }

    overrides = weekly_retrain._apply_retrain_profile_env_overrides(env, "coverage_canary")

    assert overrides["RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR"] == "1"
    assert overrides["RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE"] == "1"
    assert overrides["RUNTIME_TRAIN_AUTOFIX_ALLOW_SYMBOL_SCOPE_BROADEN"] == "1"
    assert overrides["RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA"] == "1"
    assert env["RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR"] == "1"
    assert env["RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE"] == "1"


def test_weekly_retrain_default_auto_promotes_small_explicit_target_set_to_canary() -> None:
    args = weekly_retrain.argparse.Namespace(
        retrain_profile="default",
        include_bot_ids="brain_refinery_v4_simple,brain_refinery_v13_choppy,brain_refinery_v35_dmi_state_machine",
        counterfactual_replay=True,
        paper_hard_example_pack=True,
        cold_lane_retrain_extras=True,
        require_sample_quotas=True,
        new_bot_boost=True,
        build_runtime_training_snapshot=True,
        runtime_training_snapshot_prefer_sqlite=True,
        runtime_train_use_snapshot=True,
        runtime_train_prefer_sqlite=True,
        runtime_train_fast_fail_zero_sample_attempts=0,
        target_timeout_seconds=0,
    )

    profile = weekly_retrain._apply_retrain_profile_defaults(args)

    assert profile == "canary"
    assert args.build_runtime_training_snapshot is False
    assert args.runtime_train_use_snapshot is False


def test_weekly_retrain_run_cmd_capture_returns_timeout() -> None:
    rc, stdout_text, stderr_text = weekly_retrain.run_cmd_capture(
        [
            sys.executable,
            "-c",
            "import time; print('start'); time.sleep(2); print('done')",
        ],
        dry_run=False,
        env=os.environ.copy(),
        timeout_seconds=1,
    )

    assert rc == 124
    assert stdout_text in {"", "start\n"}
    assert "command exceeded 1s" in stderr_text


def test_runtime_training_snapshot_preflight_failure_flags_empty_snapshot() -> None:
    reason = weekly_retrain._runtime_training_snapshot_preflight_failure(
        {"sequence_count": 0, "row_count": 0},
        min_sequences=1,
        min_rows=64,
    )

    assert reason == "snapshot_sequence_count_below_floor:0<1"


def test_runtime_training_snapshot_preflight_accepts_healthy_snapshot() -> None:
    reason = weekly_retrain._runtime_training_snapshot_preflight_failure(
        {"sequence_count": 24, "row_count": 1200},
        min_sequences=1,
        min_rows=64,
    )

    assert reason == ""


def test_bond_quote_quarantine_clamps_implausible_price() -> None:
    last_price, prev_close = loop._apply_bond_quote_quarantine(
        symbol="TLT",
        last_price=1111.0,
        prev_close=111.0,
        closes=[110.8, 111.1],
    )

    assert round(last_price, 4) == 111.1
    assert round(prev_close, 4) == 111.0


def test_weekly_retrain_run_cmd_capture_preserves_failure_output() -> None:
    rc, stdout_text, stderr_text = weekly_retrain.run_cmd_capture(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); print('boom', file=sys.stderr); raise SystemExit(3)",
        ],
        False,
        dict(os.environ),
    )

    assert rc == 3
    assert "hello" in stdout_text
    assert "boom" in stderr_text
    assert weekly_retrain._extract_failure_reason(stdout_text, stderr_text) == "boom"


def test_weekly_retrain_scorecard_records_failure_details(tmp_path) -> None:
    original_root = weekly_retrain.PROJECT_ROOT
    weekly_retrain.PROJECT_ROOT = str(tmp_path)
    try:
        failure_details = [
            {
                "bot_id": "brain_refinery_v15_liquidity_droughts",
                "target": "/tmp/brain_refinery_v15_liquidity_droughts.py",
                "status": "failed",
                "rc": 1,
                "reason": "insufficient_runtime_training_data",
                "stdout_tail": "dataset_ready samples=65",
                "stderr_tail": "RuntimeError: insufficient_runtime_training_data",
            }
        ]
        target_outcomes = list(failure_details)
        scorecard_path = weekly_retrain._write_retrain_scorecard(
            started_utc="2026-03-21T01:00:00+00:00",
            ended_utc="2026-03-21T01:05:00+00:00",
            target_count=1,
            failures=["/tmp/brain_refinery_v15_liquidity_droughts.py"],
            failure_details=failure_details,
            skipped_by_memory=[],
            target_outcomes=target_outcomes,
            prev_registry_snapshot={"active_bots": 10.0},
            curr_registry_snapshot={"active_bots": 10.0},
            prev_acc={"brain_refinery_v15_liquidity_droughts": 0.50},
            curr_acc={"brain_refinery_v15_liquidity_droughts": 0.50},
            master_update_status="skipped_by_flag",
            data_quality_summary={"ok": True},
            canary_priority_selected=0,
            distill_selected=0,
            lineage={"stage": "final_scorecard"},
            launch_context={"source": "opsctl_retrain_force_targeted", "run_mode": "targeted"},
        )

        assert Path(scorecard_path).exists()
        latest_payload = json.loads(
            (tmp_path / "governance" / "health" / "retrain_scorecard_latest.json").read_text(encoding="utf-8")
        )
        assert latest_payload["failure_count"] == 1
        assert latest_payload["target_outcomes"][0]["bot_id"] == "brain_refinery_v15_liquidity_droughts"
        assert latest_payload["failure_details"][0]["reason"] == "insufficient_runtime_training_data"
        assert latest_payload["launch_context"]["source"] == "opsctl_retrain_force_targeted"
    finally:
        weekly_retrain.PROJECT_ROOT = original_root


def test_weekly_retrain_training_success_marker_distinguishes_trained_ok_but_not_promotable(tmp_path) -> None:
    original_root = weekly_retrain.PROJECT_ROOT
    weekly_retrain.PROJECT_ROOT = str(tmp_path)
    try:
        marker_path = weekly_retrain._write_training_success_marker(
            target_outcomes=[
                {
                    "bot_id": "brain_refinery_v26_relative_strength_cross_section",
                    "target": "/tmp/brain_refinery_v26_relative_strength_cross_section.py",
                    "status": "trained",
                }
            ],
            failures=[],
            failure_details=[],
            skipped_by_memory=[],
            master_update_status="skipped_by_flag",
            data_quality_summary={"ok": True},
            operator_notes=None,
            lineage={"stage": "post_master_update"},
        )

        payload = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        assert payload["training_completed_ok"] is True
        assert payload["promotion_applied"] is False
        assert payload["confirmed_training_success"] is False
        assert payload["trained_ok_but_not_promotable"] is True
        assert payload["promotion_status"] == "held_out"
        assert payload["reason"] == "trained_ok_but_not_promotable:skipped_by_flag"
    finally:
        weekly_retrain.PROJECT_ROOT = original_root


def test_weekly_retrain_dry_run_scorecard_does_not_overwrite_real_latest(tmp_path) -> None:
    original_root = weekly_retrain.PROJECT_ROOT
    weekly_retrain.PROJECT_ROOT = str(tmp_path)
    try:
        scorecard_path = weekly_retrain._write_retrain_scorecard(
            started_utc="2026-03-21T01:00:00+00:00",
            ended_utc="2026-03-21T01:05:00+00:00",
            target_count=1,
            failures=[],
            failure_details=[],
            skipped_by_memory=[],
            target_outcomes=[{"bot_id": "brain_refinery_v4_simple", "status": "trained"}],
            prev_registry_snapshot={"active_bots": 10.0},
            curr_registry_snapshot={"active_bots": 10.0},
            prev_acc={"brain_refinery_v4_simple": 0.50},
            curr_acc={"brain_refinery_v4_simple": 0.51},
            master_update_status="skipped_by_flag",
            data_quality_summary={"ok": True},
            canary_priority_selected=0,
            distill_selected=0,
            lineage={"stage": "final_scorecard"},
            dry_run=True,
        )

        assert Path(scorecard_path).exists()
        assert (tmp_path / "governance" / "health" / "retrain_scorecard_dry_run_latest.json").exists()
        assert not (tmp_path / "governance" / "health" / "retrain_scorecard_latest.json").exists()
    finally:
        weekly_retrain.PROJECT_ROOT = original_root


def test_weekly_retrain_dry_run_training_success_uses_dry_run_latest(tmp_path) -> None:
    original_root = weekly_retrain.PROJECT_ROOT
    weekly_retrain.PROJECT_ROOT = str(tmp_path)
    try:
        marker_path = weekly_retrain._write_training_success_marker(
            target_outcomes=[{"bot_id": "brain_refinery_v4_simple", "status": "trained"}],
            failures=[],
            failure_details=[],
            skipped_by_memory=[],
            master_update_status="skipped_by_flag",
            data_quality_summary={"ok": True},
            operator_notes=None,
            lineage={"stage": "post_master_update"},
            dry_run=True,
        )

        assert Path(marker_path).exists()
        assert marker_path.endswith("training_success_dry_run_latest.json")
        assert not (tmp_path / "governance" / "health" / "training_success_latest.json").exists()
    finally:
        weekly_retrain.PROJECT_ROOT = original_root


def test_weekly_retrain_build_retrain_input_feature_diagnostics_tracks_advanced_features(tmp_path) -> None:
    dataset_path = tmp_path / "trade_learning_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rows": 2,
                "feature_dim": 3,
                "feature_schema_version": "trade_behavior_features_v4",
                "feature_names": [
                    "core_cross_sectional_rank_norm",
                    "day_failed_breakout_risk_norm",
                    "long_term_factor_exposure_control_norm",
                ],
                "data": [
                    {
                        "label": "positive",
                        "features": [0.8, 0.1, 0.0],
                    },
                    {
                        "label": "negative",
                        "features": [0.0, 0.9, 0.7],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = weekly_retrain._build_retrain_input_feature_diagnostics(str(dataset_path))

    tracked = payload["tracked_features"]
    assert payload["dataset_rows"] == 2
    assert tracked["core_cross_sectional_rank_norm"]["coverage_ratio"] == 0.5
    assert tracked["day_failed_breakout_risk_norm"]["high_signal_ratio"] == 0.5
    assert tracked["long_term_factor_exposure_control_norm"]["high_signal_label_counts"]["negative"] == 1


def test_weekly_retrain_build_failed_bot_replay_summary_maps_failed_bots_to_profiles() -> None:
    payload = weekly_retrain._build_failed_bot_replay_summary(
        failure_details=[
            {
                "bot_id": "brain_refinery_v12_news_shocks",
                "reason": "precision_guard_failed",
            }
        ],
        counterfactual_summary={
            "top_candidates": [
                {
                    "profile": "intraday_aggressive",
                    "threshold_delta": -0.03,
                    "tradeability_floor": 0.55,
                    "aggregate_net_pnl_total": 42.0,
                },
                {
                    "profile": "aggressive",
                    "threshold_delta": -0.01,
                    "tradeability_floor": 0.50,
                    "aggregate_net_pnl_total": 10.0,
                },
            ]
        },
        paper_performance={
            "sleeve_latest": [
                {
                    "profile": "intraday_aggressive",
                    "ending_net_pnl_total": -12.5,
                    "win_rate": 0.4,
                },
                {
                    "profile": "aggressive",
                    "ending_net_pnl_total": 3.0,
                    "win_rate": 0.55,
                },
            ]
        },
    )

    assert payload["failed_bot_count"] == 1
    assert payload["bot_summaries"][0]["segment"] == "shock"
    assert "intraday_aggressive" in payload["bot_summaries"][0]["recommended_profiles"]
    assert payload["profile_pressure"][0]["profile"] == "intraday_aggressive"
    assert payload["profile_pressure"][0]["current_end_net"] == -12.5


def test_weekly_retrain_retry_pack_marks_chronic_failures_for_distillation(tmp_path) -> None:
    original_root = weekly_retrain.PROJECT_ROOT
    original_bottleneck_path = weekly_retrain.PROMOTION_BOTTLENECK_PATH
    weekly_retrain.PROJECT_ROOT = str(tmp_path)
    weekly_retrain.PROMOTION_BOTTLENECK_PATH = str(tmp_path / "governance" / "walk_forward" / "promotion_bottleneck_latest.json")
    Path(weekly_retrain.PROMOTION_BOTTLENECK_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(weekly_retrain.PROMOTION_BOTTLENECK_PATH).write_text(
        json.dumps(
            {
                "top_failing_bots": [
                    {
                        "bot_id": "brain_refinery_v35_dmi_state_machine",
                        "fail_days": 6,
                        "recommended_categories": ["distillation_candidate"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        retry_pack = weekly_retrain._write_retry_pack(
            failures=["/tmp/brain_refinery_v35_dmi_state_machine.py"],
            failure_details=[
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "reason": "runtime_training_quality_guard_failed long_precision=0.20",
                }
            ],
            master_update_status="failed_exit_2",
        )

        assert retry_pack is not None
        assert retry_pack["distillation_priority"] is True
        assert retry_pack["chronic_bot_ids"] == ["brain_refinery_v35_dmi_state_machine"]
        assert "--distillation-priority" in retry_pack["command"]
        latest = json.loads((tmp_path / "governance" / "health" / "retrain_retry_pack_latest.json").read_text(encoding="utf-8"))
        assert latest["include_bot_ids"] == ["brain_refinery_v35_dmi_state_machine"]
    finally:
        weekly_retrain.PROJECT_ROOT = original_root
        weekly_retrain.PROMOTION_BOTTLENECK_PATH = original_bottleneck_path


def test_sub_bot_runtime_scope_blocks_bond_specialists_outside_trained_sleeve() -> None:
    bond_bot = loop.SubBot(
        bot_id="brain_refinery_v95_rates_regime_bond_bot",
        weight=0.9,
        active=True,
        reason="probation",
        test_accuracy=0.56,
    )
    credit_bot = loop.SubBot(
        bot_id="brain_refinery_v96_credit_spread_rotation_bot",
        weight=0.8,
        active=True,
        reason="probation",
        test_accuracy=0.55,
    )

    ok_bond, reasons_bond = loop._sub_bot_runtime_scope_ok(bond_bot, symbol="BTC-USD", profile="crypto")
    ok_credit, reasons_credit = loop._sub_bot_runtime_scope_ok(credit_bot, symbol="/ES", profile="schwab_futures")
    ok_allowed, _ = loop._sub_bot_runtime_scope_ok(bond_bot, symbol="TLT", profile="bond")

    assert ok_bond is False
    assert "scope_blocked_profile=crypto" in reasons_bond
    assert ok_credit is False
    assert "scope_blocked_profile=schwab_futures" in reasons_credit
    assert ok_allowed is True


def test_weekly_retrain_promotion_precheck_failures_capture_health_readiness_and_paper(tmp_path) -> None:
    health_path = tmp_path / "governance" / "health" / "health_gates_latest.json"
    readiness_path = tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json"
    paper_path = tmp_path / "governance" / "health" / "paper_performance_latest.json"

    health_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)

    health_path.write_text(json.dumps({"hard_gate_triggered": True}), encoding="utf-8")
    readiness_path.write_text(
        json.dumps(
            {
                "promote_ok": False,
                "coverage_ok": False,
                "considered_bots": 0,
                "thresholds": {"min_considered_bots": 4},
            }
        ),
        encoding="utf-8",
    )
    paper_path.write_text(json.dumps({"sleeve_latest": [{"profile": "default", "executions": 2}]}), encoding="utf-8")

    failures = weekly_retrain._promotion_state_precheck_failures(
        promotion_readiness_path=str(readiness_path),
        health_gates_path=str(health_path),
        paper_performance_path=str(paper_path),
    )

    assert "health_gates:hard_gate_triggered" in failures
    assert "promotion_readiness:promote_ok=false" in failures
    assert "promotion_readiness:coverage_ok=false" in failures
    assert "promotion_readiness:considered_bots=0<4" in failures
    assert "paper_feedback:executions=2<24" in failures
    assert "paper_feedback:active_sleeves=1<3" in failures


def test_run_master_bot_guard_helpers_block_unhealthy_master_update(tmp_path) -> None:
    health_path = tmp_path / "health_gates_latest.json"
    readiness_path = tmp_path / "promotion_readiness_latest.json"
    paper_path = tmp_path / "paper_performance_latest.json"

    health_path.write_text(
        json.dumps({"hard_gate_triggered": True, "recommended_operating_mode": "shadow_only"}),
        encoding="utf-8",
    )
    readiness_path.write_text(
        json.dumps(
            {
                "promote_ok": False,
                "coverage_ok": False,
                "considered_bots": 1,
                "thresholds": {"min_considered_bots": 4},
            }
        ),
        encoding="utf-8",
    )
    paper_path.write_text(
        json.dumps({"sleeve_latest": [{"profile": "default", "executions": 8, "non_flat_strategy_count": 1}]}),
        encoding="utf-8",
    )

    health_ok, health_reason, _health_detail = run_master_bot._health_gate_ok(health_path)
    readiness_ok, readiness_reason, readiness_detail = run_master_bot._promotion_readiness_ok(readiness_path)
    paper_ok, paper_reason, paper_detail = run_master_bot._paper_feedback_ok(paper_path)

    assert health_ok is False
    assert health_reason == "health_gate_triggered"
    assert readiness_ok is False
    assert readiness_reason == "promotion_readiness_blocked"
    assert readiness_detail["min_considered_bots"] == 4
    assert paper_ok is False
    assert paper_reason == "paper_feedback_floor_blocked"
    assert paper_detail["min_executions"] == 24
