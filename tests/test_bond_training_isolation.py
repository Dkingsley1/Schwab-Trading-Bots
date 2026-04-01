import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import scripts.data_source_divergence_bot as divergence_bot
import scripts.run_shadow_training_loop as loop
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
    finally:
        weekly_retrain.REGISTRY_PATH = original_registry_path

    assert filtered == ["/tmp/core/brain_refinery_v35_dmi_state_machine.py"]
    assert stats["low_readiness_skipped"] == 4


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
        )

        assert Path(scorecard_path).exists()
        latest_payload = json.loads(
            (tmp_path / "governance" / "health" / "retrain_scorecard_latest.json").read_text(encoding="utf-8")
        )
        assert latest_payload["failure_count"] == 1
        assert latest_payload["target_outcomes"][0]["bot_id"] == "brain_refinery_v15_liquidity_droughts"
        assert latest_payload["failure_details"][0]["reason"] == "insufficient_runtime_training_data"
    finally:
        weekly_retrain.PROJECT_ROOT = original_root


def test_weekly_retrain_training_success_marker_distinguishes_trained_not_promoted(tmp_path) -> None:
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
        assert payload["reason"] == "trained_not_promoted:skipped_by_flag"
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
