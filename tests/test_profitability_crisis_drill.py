import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import profitability_crisis_drill as drill

FIXED_TIME = "2026-08-25T20:00:00+00:00"


def _build(**kwargs):
    return drill.build_payload(
        project_root=PROJECT_ROOT,
        generated_at_utc=FIXED_TIME,
        **kwargs,
    )


def test_crisis_drill_covers_three_historical_collapse_scenarios() -> None:
    payload = _build()

    assert payload["ok"] is True
    assert payload["control_grade"] == "A+"
    assert payload["selection"]["scenario_count"] == 3
    assert {row["scenario_id"] for row in payload["scenario_results"]} == {
        "gfc_2008_financial_crisis",
        "covid_2020_pandemic_crash",
        "us_regional_banking_2023",
    }
    assert payload["diagnostic_summary"]["phase_count"] == 15


def test_crisis_drill_blocks_severe_entries_but_reopens_recovery() -> None:
    payload = _build()
    phases = [phase for row in payload["scenario_results"] for phase in row["phases"]]
    severe = [phase for phase in phases if phase["is_severe_phase"]]
    recovery = [phase for phase in phases if phase["is_recovery_phase"]]

    assert severe
    assert all(phase["severe_new_long_entry_blocked"] for phase in severe)
    assert len(recovery) == 3
    assert all(phase["recovery_reentry_opportunity"] for phase in recovery)
    assert all(phase["best_cost_adjusted_action"] == "BUY" for phase in recovery)
    assert all(phase["existing_long_reduce_only_exit"]["available"] for phase in phases)


def test_crisis_drill_is_diagnostic_only_and_does_not_mutate_candidate() -> None:
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
    assert payload["evidence_classification"]["diagnostic_only"] is True
    assert payload["evidence_classification"]["promotion_evidence"] is False
    assert payload["evidence_classification"]["profitability_proof"] is False


def test_crisis_scenario_sources_and_parameter_labels_are_explicit() -> None:
    policy = json.loads(
        (PROJECT_ROOT / drill.DEFAULT_POLICY_PATH).read_text(encoding="utf-8")
    )
    for raw_path in policy["scenario_paths"]:
        scenario = json.loads((PROJECT_ROOT / raw_path).read_text(encoding="utf-8"))
        references = scenario["source"]["references"]
        assert references
        assert all(row["official_authoritative_source"] is True for row in references)
        assert all(row["url"].startswith("https://") for row in references)
        assert (
            "not historical tick reconstruction"
            in scenario["profitability_drill"]["parameter_notice"]
        )
        assert (
            scenario["replay_contract"]["label_policy"] == "research_only_no_execution"
        )


def test_crisis_drill_supports_scenario_alias_selection_deterministically() -> None:
    first = _build(requested_scenarios=["banking_crisis_2008"])
    second = _build(requested_scenarios=["banking_crisis_2008"])

    assert first == second
    assert first["selection"]["scenario_count"] == 1
    assert first["scenario_results"][0]["scenario_id"] == "gfc_2008_financial_crisis"
    assert first["ok"] is True
