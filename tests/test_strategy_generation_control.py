from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import strategy_generation_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _config() -> dict:
    return json.loads((src.DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))


def _integrity(project_root: Path, config: dict) -> dict:
    key_path = project_root / config["integrity"]["event_signing_key_path"]
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("a" * 64, encoding="utf-8")
    os.chmod(key_path, 0o600)
    policy, integrity = src._policy_validation(project_root, config)
    assert policy["ok"] is True
    return integrity


def _teacher(bot_id: str, role: str, score: float) -> dict:
    return {
        "bot_id": bot_id,
        "bot_role": role,
        "teacher_grade": "elite",
        "teacher_score": score,
        "walk_forward_runs": 18,
        "walk_forward_forward_mean": 0.59,
        "paper_bonus": 0.02,
        "active": True,
        "overfit_policy": {"may_teach": True, "may_promote": False},
    }


def _seed_parents(project_root: Path) -> None:
    teachers = [
        _teacher("brain_refinery_v10_seasonal", "signal_sub_bot", 0.82),
        _teacher("brain_refinery_v17_mixed_regime", "signal_sub_bot", 0.79),
        _teacher("brain_refinery_v58_ensemble_diversity_controller", "options_sub_bot", 0.77),
    ]
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {"qualified_teachers": teachers},
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": row["bot_id"],
                    "bot_role": row["bot_role"],
                    "active": True,
                    "training_excluded": False,
                }
                for row in teachers
            ]
        },
    )
    for row in teachers:
        module = project_root / "core" / f"{row['bot_id']}.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("# test training module\n", encoding="utf-8")


def test_generation_proposes_only_bounded_dormant_offspring(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_parents(project_root)
    config = _config()
    config["resource_limits"]["generation_cooldown_hours"] = 0
    integrity = _integrity(project_root, config)
    state = src._default_state(config["policy_id"])
    event_path = project_root / "governance" / "strategy_generations" / "events.jsonl"

    offspring, blockers = src.propose_generation(
        project_root,
        config,
        state,
        event_path,
        now=datetime(2026, 8, 7, tzinfo=timezone.utc),
        integrity=integrity,
    )

    assert blockers == []
    assert len(offspring) == 2
    assert len({row["bot_role"] for row in offspring}) == 2
    assert all(row["lifecycle_state"] == "proposed_collection_only" for row in offspring)
    assert all(row["execution_authority"] is False for row in offspring)
    assert all(row["paper_execution_authority"] is False for row in offspring)
    assert all(row["serving_eligible"] is False for row in offspring)
    assert all(row["inherits_parent_grade"] is False for row in offspring)
    chain = src.verify_event_chain(
        event_path,
        signing_secret=integrity["secret"],
        signing_key_id=integrity["key_id"],
        require_signatures=True,
    )
    assert chain["ok"] is True
    assert chain["signed_event_count"] == 1


def test_generation_waits_when_no_parent_has_reproduction_evidence(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {
            "qualified_teachers": [
                {
                    **_teacher("brain_refinery_v10_seasonal", "signal_sub_bot", 0.82),
                    "walk_forward_runs": 0,
                    "paper_bonus": 0.0,
                    "overfit_policy": {"may_teach": False},
                }
            ]
        },
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {"sub_bots": [{"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot", "active": True}]},
    )
    module = project_root / "core" / "brain_refinery_v10_seasonal.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("# test\n", encoding="utf-8")
    config = _config()
    integrity = _integrity(project_root, config)
    state = src._default_state(config["policy_id"])

    eligible, rejected = src._parent_rejections(project_root, config, state)
    offspring, blockers = src.propose_generation(
        project_root,
        config,
        state,
        project_root / "events.jsonl",
        now=datetime(2026, 8, 7, tzinfo=timezone.utc),
        integrity=integrity,
    )

    assert eligible == []
    assert offspring == []
    assert "no_parent_has_reproduction_grade_evidence" in blockers
    assert "overfit_policy_forbids_reproduction" in rejected[0]["rejection_reasons"]


def test_qualified_offspring_still_has_no_execution_authority(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = _config()
    integrity = _integrity(project_root, config)
    state = src._default_state(config["policy_id"])
    model_path = project_root / "models" / "strategy_g0001_abcdef123456_model.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"candidate model")
    manifest_path = project_root / "governance" / "strategy_generations" / "generations" / "strategy_generation_0001.json"
    _write_json(manifest_path, {"strategy_generation": 1, "offspring_ids": ["strategy_g0001_abcdef123456"]})
    completed_at = datetime.now(timezone.utc) - timedelta(minutes=2)
    candidate = {
        "offspring_id": "strategy_g0001_abcdef123456",
        "strategy_generation": 1,
        "lineage_depth": 1,
        "parent_bot_ids": ["brain_refinery_v10_seasonal"],
        "source_module_bot_id": "brain_refinery_v10_seasonal",
        "bot_role": "signal_sub_bot",
        "lifecycle_state": "trained_collection_only",
        "execution_authority": False,
        "paper_execution_authority": False,
        "serving_eligible": False,
        "model_path": str(model_path),
        "model_sha256": src._file_hash(model_path),
        "generation_manifest_path": str(manifest_path),
        "generation_manifest_sha256": src._file_hash(manifest_path),
        "training_completed_at_utc": completed_at.isoformat(),
    }
    state["offspring"] = [candidate]
    evaluation_path = project_root / config["evaluation"]["allowed_evaluation_root"] / "evaluation.json"
    evaluation = {
        "candidate_id": candidate["offspring_id"],
        "evaluation_run_id": "eval-run-0001",
        "evaluated_at_utc": (completed_at + timedelta(minutes=1)).isoformat(),
        "evaluator_id": "independent-evaluator-test",
        "evaluator_version": "1.0",
        "evaluator_role": "independent_strategy_evaluator",
        "model_sha256": candidate["model_sha256"],
        "generation_manifest_sha256": candidate["generation_manifest_sha256"],
        "dataset_sha256": "b" * 64,
        "holdout_sha256": "c" * 64,
        "replay_sha256": "d" * 64,
        "independent_evaluation": True,
        "locked_holdout": True,
        "exact_replay": True,
        "out_of_sample_trades": 150,
        "out_of_sample_net_pnl": 25.0,
        "net_expectancy": 0.03,
        "stressed_post_cost_expectancy": 0.01,
        "lower_confidence_bound": 0.002,
        "maximum_drawdown": 0.06,
        "parent_return_correlation": 0.70,
        "multiple_testing_adjusted_p_value": 0.02,
        "composite_score": 0.81,
    }
    evaluation["attestation"] = {
        "key_id": integrity["key_id"],
        "signature": src._evaluation_signature(
            evaluation,
            key_id=integrity["key_id"],
            secret=integrity["secret"],
        ),
    }
    _write_json(evaluation_path, evaluation)

    result, failed = src.evaluate_offspring(
        project_root,
        config,
        state,
        project_root / "events.jsonl",
        candidate_id=candidate["offspring_id"],
        evaluation_path=evaluation_path,
        integrity=integrity,
    )

    assert failed == []
    assert result is not None
    assert result["lifecycle_state"] == "paper_challenger_qualified"
    assert result["evaluation"]["qualified"] is True
    assert result["execution_authority"] is False
    assert result["paper_execution_authority"] is False
    assert result["serving_eligible"] is False
    assert result["registry_admission_eligible"] is False
    assert result["lineage_parent_approved"] is False
    assert result["paper_allocation_limit"] == 0.0
    assert result["live_order_budget"] == 0.0


def test_offspring_queue_can_clear_normal_selector_idle_without_bypassing_resource_gates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = _config()
    config["resource_limits"]["minimum_free_disk_gb"] = 0
    state = src._default_state(config["policy_id"])
    state["offspring"] = [
        {"offspring_id": "strategy_g0001_abcdef123456", "lifecycle_state": "proposed_collection_only"}
    ]
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {
            "training_launch_contract": {
                "launch_allowed": False,
                "prep_allowed": True,
                "launch_blockers": ["no_bot_needs_training_candidates"],
            }
        },
    )
    throttle_path = project_root / "governance" / "health" / "runtime_throttle_control_latest.json"
    _write_json(
        throttle_path,
        {
            "overall_status": "ready",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "host_saturation_score": 10.0,
            "release_contract": {"shared_host_training_resume_allowed": True},
            "runtime_snapshot": {
                "thermal": {
                    "thermal_warning_active": False,
                    "performance_warning_active": False,
                    "cpu_power_warning_active": False,
                }
            },
        },
    )

    assert src._training_gate(project_root, config, state) == []

    throttle = json.loads(throttle_path.read_text(encoding="utf-8"))
    throttle["memory_pressure_level"] = "high"
    _write_json(throttle_path, throttle)

    assert "memory_pressure_not_generation_safe" in src._training_gate(project_root, config, state)


def test_signed_event_chain_detects_post_write_tampering(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_parents(project_root)
    config = _config()
    config["resource_limits"]["generation_cooldown_hours"] = 0
    integrity = _integrity(project_root, config)
    state = src._default_state(config["policy_id"])
    event_path = project_root / "governance" / "strategy_generations" / "events.jsonl"

    offspring, blockers = src.propose_generation(
        project_root,
        config,
        state,
        event_path,
        now=datetime.now(timezone.utc),
        integrity=integrity,
    )

    assert blockers == []
    assert offspring
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["offspring_ids"] = ["strategy_g9999_ffffffffffff"]
    unsigned = {
        key: value
        for key, value in event.items()
        if key not in {"event_hash", "event_signature", "signature_key_id"}
    }
    event["event_hash"] = src._canonical_hash(unsigned)
    event_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    chain = src.verify_event_chain(
        event_path,
        signing_secret=integrity["secret"],
        signing_key_id=integrity["key_id"],
        require_signatures=True,
    )

    assert chain["ok"] is False
    assert any("event_signature_mismatch" in reason for reason in chain["errors"])


def test_retained_population_cap_blocks_new_generation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_parents(project_root)
    config = _config()
    integrity = _integrity(project_root, config)
    state = src._default_state(config["policy_id"])
    state["offspring"] = [
        {
            "offspring_id": f"strategy_g0001_{index:012x}",
            "lifecycle_state": "training_failed_quarantined",
        }
        for index in range(config["resource_limits"]["max_retained_offspring"])
    ]

    offspring, blockers = src.propose_generation(
        project_root,
        config,
        state,
        project_root / "events.jsonl",
        now=datetime.now(timezone.utc),
        integrity=integrity,
    )

    assert offspring == []
    assert "retained_offspring_cap_reached" in blockers


def test_reconcile_quarantines_stale_training_without_authority(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = _config()
    integrity = _integrity(project_root, config)
    state = src._default_state(config["policy_id"])
    candidate = {
        "offspring_id": "strategy_g0001_abcdef123456",
        "lifecycle_state": "training",
        "training_started_at_utc": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "execution_authority": False,
        "paper_execution_authority": False,
        "serving_eligible": False,
    }
    state["offspring"] = [candidate]
    event_path = project_root / "governance" / "strategy_generations" / "events.jsonl"

    reconciled = src.reconcile_stale_training(
        config,
        state,
        event_path,
        integrity=integrity,
    )

    assert len(reconciled) == 1
    assert candidate["lifecycle_state"] == "training_failed_quarantined"
    assert candidate["execution_authority"] is False
    assert candidate["paper_execution_authority"] is False
    assert candidate["live_order_budget"] == 0.0
    assert src.verify_event_chain(
        event_path,
        signing_secret=integrity["secret"],
        signing_key_id=integrity["key_id"],
        require_signatures=True,
    )["ok"] is True


def test_evaluation_attestation_binds_metrics_and_candidate_hashes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = _config()
    integrity = _integrity(project_root, config)
    evaluation = {
        "candidate_id": "strategy_g0001_abcdef123456",
        "model_sha256": "b" * 64,
        "holdout_sha256": "c" * 64,
        "net_expectancy": 0.03,
    }
    signature = src._evaluation_signature(
        evaluation,
        key_id=integrity["key_id"],
        secret=integrity["secret"],
    )
    evaluation["net_expectancy"] = -0.03

    assert signature != src._evaluation_signature(
        evaluation,
        key_id=integrity["key_id"],
        secret=integrity["secret"],
    )


def test_qualified_offspring_cannot_reproduce_without_human_lineage_approval(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = _config()
    source_module = project_root / "core" / "brain_refinery_v10_seasonal.py"
    source_module.parent.mkdir(parents=True, exist_ok=True)
    source_module.write_text("# source\n", encoding="utf-8")
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "qualified_teachers": []},
    )
    _write_json(project_root / "master_bot_registry.json", {"sub_bots": []})
    state = src._default_state(config["policy_id"])
    state["offspring"] = [
        {
            "offspring_id": "strategy_g0001_abcdef123456",
            "source_module_bot_id": "brain_refinery_v10_seasonal",
            "bot_role": "signal_sub_bot",
            "lineage_depth": 1,
            "lifecycle_state": "paper_challenger_qualified",
            "lineage_parent_approved": False,
            "evaluation": {
                "qualified": True,
                "composite_score": 0.81,
                "out_of_sample_net_pnl": 25.0,
            },
        }
    ]

    eligible, rejected = src._parent_rejections(project_root, config, state)

    assert eligible == []
    assert "offspring_lineage_parent_human_approval_missing" in rejected[0]["rejection_reasons"]


def test_policy_and_state_integrity_fail_closed_on_unsafe_mutation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config = _config()
    _integrity(project_root, config)
    state = src._default_state(config["policy_id"])

    state["generation"] = 99
    state_check = src._verify_state(state, require_hash=True)
    assert state_check["ok"] is False
    assert "state_hash_mismatch" in state_check["errors"]

    config["safety_contract"]["paper_execution_authority"] = True
    policy, _ = src._policy_validation(project_root, config)
    assert policy["ok"] is False
    assert "generation_execution_authority_policy_unsafe" in policy["errors"]


def test_signed_tail_snapshot_recovers_one_interrupted_state_commit(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_parents(project_root)
    config = _config()
    config["resource_limits"]["generation_cooldown_hours"] = 0
    integrity = _integrity(project_root, config)
    persisted_state = src._default_state(config["policy_id"])
    in_memory_state = json.loads(json.dumps(persisted_state))
    event_path = project_root / "governance" / "strategy_generations" / "events.jsonl"

    offspring, blockers = src.propose_generation(
        project_root,
        config,
        in_memory_state,
        event_path,
        now=datetime.now(timezone.utc),
        integrity=integrity,
    )
    recovery = src._recover_state_from_signed_tail(persisted_state, event_path)

    assert blockers == []
    assert recovery["recovered"] is True
    assert persisted_state["event_chain_head"] == in_memory_state["event_chain_head"]
    assert {row["offspring_id"] for row in persisted_state["offspring"]} == {
        row["offspring_id"] for row in offspring
    }
    assert src._verify_state(persisted_state, require_hash=True)["ok"] is True
