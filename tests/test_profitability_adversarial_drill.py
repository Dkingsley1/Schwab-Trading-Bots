import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import profitability_adversarial_drill as drill
from core.system_role_contracts import evaluate_component_action

FIXED_TIME = "2026-08-26T12:00:00+00:00"


def _build(**kwargs):
    return drill.build_payload(
        project_root=PROJECT_ROOT,
        generated_at_utc=FIXED_TIME,
        **kwargs,
    )


def _scenario(payload, scenario_id):
    return next(
        row for row in payload["scenario_results"] if row["scenario_id"] == scenario_id
    )


def test_adversarial_drill_runs_all_fourteen_scenarios_at_a_plus() -> None:
    payload = _build()

    assert payload["ok"] is True
    assert payload["status"] == "ready"
    assert payload["control_grade"] == "A+"
    assert payload["control_score"] == 100.0
    assert payload["selection"]["scenario_count"] == 14
    assert payload["diagnostic_summary"]["passed_scenario_count"] == 14
    assert payload["diagnostic_summary"]["scenario_check_count"] >= 70
    assert payload["diagnostic_summary"]["scenario_failed_check_count"] == 0
    assert all(row["ok"] for row in payload["scenario_results"])


def test_capacity_drill_covers_full_fleet_and_both_scale_endpoints() -> None:
    payload = _build(requested_scenarios=["capacity"])
    scenario = _scenario(payload, "capital_capacity_scaling")
    metrics = scenario["metrics"]
    curves = scenario["details"]["sleeve_curves"]

    assert payload["ok"] is True
    assert metrics["sleeve_count"] == 111
    assert metrics["trading_sleeve_count"] == 86
    assert metrics["control_only_sleeve_count"] == 25
    assert metrics["objective_class_count"] == 12
    assert metrics["hot_strategy_count"] == 879
    assert metrics["research_strategy_count"] == 12000
    assert metrics["capital_tier_count"] == 22
    assert metrics["market_state_count"] == 6
    assert metrics["surface_point_count"] == 86 * 22 * 6

    trading = [row for row in curves if row["applicable"]]
    controls = [row for row in curves if not row["applicable"]]
    assert all(len(row["capital_tiers"]) == 22 for row in trading)
    assert all(row["canary_200"]["capital_usd"] == 200 for row in trading)
    assert all(
        row["target_scale_1000000000"]["capital_usd"] == 1000000000 for row in trading
    )
    assert all(row["organic_calibration_required"] for row in trading)
    assert all(not row["capital_tiers"] for row in controls)
    assert sum(row["hot_strategy_count"] for row in curves) == 879
    assert sum(row["research_strategy_count"] for row in curves) == 12000


def test_capacity_drill_distinguishes_fractional_canary_from_contract_products() -> (
    None
):
    payload = _build(requested_scenarios=["capital_scaling"])
    curves = _scenario(payload, "capital_capacity_scaling")["details"]["sleeve_curves"]
    by_sleeve = {row["sleeve_id"]: row for row in curves}

    dividend = by_sleeve["dividend_income"]
    options = by_sleeve["compound_options"]
    futures = by_sleeve["futures_index_intraday"]

    assert dividend["asset_class"] == "equity"
    assert dividend["canary_200"]["executable_at_configured_unit"] is True
    assert options["asset_class"] in {"options", "structured_derivative"}
    assert options["canary_200"]["executable_at_configured_unit"] is False
    assert futures["asset_class"] == "futures"
    assert futures["canary_200"]["executable_at_configured_unit"] is False
    assert futures["target_scale_1000000000"]["executable_at_configured_unit"] is True


def test_scenario_failure_modes_are_detected_without_relaxing_controls() -> None:
    payload = _build()

    whipsaw = _scenario(payload, "regime_transition_whipsaw")
    assert (
        whipsaw["metrics"]["guarded_action_flip_count"]
        < whipsaw["metrics"]["raw_action_flip_count"]
    )
    costs = _scenario(payload, "net_alpha_break_even_ladder")
    assert costs["metrics"]["first_blocked_round_trip_cost_bps"] == 20.0
    liquidity = _scenario(payload, "liquidity_evaporation_partial_fill")
    assert liquidity["metrics"]["additional_entry_quantity"] == 0.0
    assert liquidity["metrics"]["residual_inventory_quantity"] > 0.0
    decay = _scenario(payload, "gradual_strategy_decay")
    assert decay["metrics"]["stable_change_points"] == 0
    assert decay["metrics"]["decay_change_points"] > 0
    consensus = _scenario(payload, "false_model_consensus")
    assert (
        consensus["metrics"]["effective_independent_vote_count"]
        < consensus["metrics"]["raw_vote_count"]
    )


def test_drill_is_candidate_immutable_and_has_zero_execution_authority() -> None:
    candidate_path = PROJECT_ROOT / drill.DEFAULT_CANDIDATE_PATH
    before = candidate_path.read_bytes()
    payload = _build()
    after = candidate_path.read_bytes()

    assert before == after
    assert payload["candidate_mutation_guard"]["unchanged"] is True
    assert payload["candidate_binding"]["valid"] is True
    assert payload["candidate_binding"]["live_execution_authority"] is False
    assert payload["authority_contract"]
    assert not any(payload["authority_contract"].values())
    assert payload["resource_contract"]["persistent_processes_started"] == 0
    assert payload["resource_contract"]["network_requests"] == 0
    assert payload["resource_contract"]["broker_requests"] == 0
    assert payload["resource_contract"]["orders_submitted"] == 0
    assert payload["evidence_classification"]["profitability_proof"] is False
    assert (
        payload["evidence_classification"]["capacity_is_certified_deployable_capital"]
        is False
    )


def test_policy_declares_exact_scenarios_and_bounded_capacity_contract() -> None:
    policy = json.loads(
        (PROJECT_ROOT / drill.DEFAULT_POLICY_PATH).read_text(encoding="utf-8")
    )
    required = policy["scenario_contract"]["required_scenario_ids"]
    capacity = policy["capital_capacity_contract"]

    assert len(required) == len(set(required)) == 14
    assert set(required) == set(drill.SCENARIO_FUNCTIONS)
    assert capacity["capital_tiers_usd"][0] == 200
    assert capacity["capital_tiers_usd"][-1] == 1000000000
    assert len(capacity["capital_tiers_usd"]) == 22
    assert len(capacity["market_states"]) == 6
    assert {"flash_crash_dislocation", "crowded_exit"} <= set(capacity["market_states"])
    assert len(capacity["objective_profiles"]) == 12
    assert policy["evidence_contract"]["automatic_threshold_tuning_allowed"] is False
    assert not any(policy["authority_contract"].values())


def test_adversarial_artifact_has_one_declared_atomic_writer() -> None:
    ownership = json.loads(
        (PROJECT_ROOT / "config" / "control_surface_ownership_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        row
        for row in ownership["controls"]
        if row["resource_path"]
        == "governance/research/profitability_adversarial_drill_latest.json"
    ]

    assert len(rows) == 1
    assert rows[0]["owner_source"] == "scripts/ops/profitability_adversarial_drill.py"
    assert rows[0]["owner_command"] == [
        "./scripts/ops/opsctl.sh",
        "profitability-adversarial-drill",
        "--json",
    ]
    assert rows[0]["mutation_mode"] == "atomic_snapshot"
    assert rows[0]["coordination"] == "atomic_replace"


def test_adversarial_writer_has_diagnostic_only_role_authority() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "config" / "system_role_contracts_v1.json").read_text(
            encoding="utf-8"
        )
    )
    component = next(
        row
        for row in contract["components"]
        if row["component_id"] == "profitability_adversarial_drill_controller"
    )
    role = next(
        row for row in contract["roles"] if row["role_id"] == component["role_id"]
    )

    assert component["role_id"] == "evaluation_auditor"
    assert set(component["allowed_actions"]) == {
        "validate_evidence",
        "publish_evidence",
        "emit_health",
    }
    assert role["execution_authority"]["paper_trade"] is False
    assert role["execution_authority"]["live_submit"] is False
    assert role["execution_authority"]["automatic_promotion"] is False

    publication = evaluate_component_action(
        PROJECT_ROOT,
        component_id="profitability_adversarial_drill_controller",
        action="publish_evidence",
        state_domain="profitability_adversarial_drill_evidence",
        resource_path="governance/research/profitability_adversarial_drill_latest.json",
    )
    assert publication["ok"] is True

    for forbidden_action in (
        "paper_submit",
        "live_submit",
        "promote_candidate",
        "allocate_shadow_capital",
        "set_risk_limit",
        "automatic_threshold_tuning",
    ):
        decision = evaluate_component_action(
            PROJECT_ROOT,
            component_id="profitability_adversarial_drill_controller",
            action=forbidden_action,
        )
        assert decision["ok"] is False


def test_alias_selection_is_deterministic() -> None:
    first = _build(requested_scenarios=["path_dependency"])
    second = _build(requested_scenarios=["path_dependency"])

    assert first == second
    assert first["ok"] is True
    assert first["selection"]["scenario_count"] == 1
    assert first["scenario_results"][0]["scenario_id"] == "portfolio_path_dependency"
