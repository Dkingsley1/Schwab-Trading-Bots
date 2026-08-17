import json
from pathlib import Path

from scripts.ops import readiness_blocker_rollup as rollup


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_profitability_symptoms_roll_up_to_one_causal_root(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write(
        health / "production_excellence_control_latest.json",
        {
            "pillars": [
                {"pillar_id": "p07_profitability_evidence", "grade": "F", "score": 50, "failed_checks": ["post_cost_sample_floor", "positive_post_cost_lcb"]},
                {"pillar_id": "p10_institutional_operations", "grade": "B", "score": 83, "failed_checks": ["production_readiness_ready"]},
            ]
        },
    )
    _write(
        health / "production_quality_slo_guard_latest.json",
        {
            "breached_lanes": [
                {"lane_id": "raw_profitability_recovery", "status": "breach", "blocking_reasons": ["raw_profitability_grade_below_A"], "active_minutes": 120}
            ],
            "warning_lanes": [],
        },
    )
    _write(
        health / "live_money_readiness_contract_latest.json",
        {
            "grade_summary": {"below_floor_sections": ["paper_profitability_control"]},
            "transition_runway": {
                "pillars": [
                    {
                        "pillar_id": "paper_truth",
                        "ready": False,
                        "runway_status": "late_blocked",
                        "blocked_sections": ["paper_profitability_control"],
                        "blockers": ["paper_profitability_control_below_A"],
                    }
                ]
            },
        },
    )
    _write(
        health / "paper_profitability_control_latest.json",
        {"raw_profitability_grade": "C", "a_plus_target_contract": {"current": {"net_pnl": -1000}}},
    )

    payload = rollup.build_payload(tmp_path)

    assert payload["unique_root_cause_count"] == 1
    root = payload["root_causes"][0]
    assert root["root_id"] == "raw_profitability_evidence"
    assert root["priority"] == "critical"
    assert root["downstream_surface_count"] >= 5
    assert payload["duplicate_symptom_reduction_count"] > 0


def test_distinct_evidence_roots_remain_distinct(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write(
        health / "production_excellence_control_latest.json",
        {
            "pillars": [
                {"pillar_id": "p05_independent_fill_evidence", "failed_checks": ["independent_fill_minimum"]},
                {"pillar_id": "p06_real_promotion_candidates", "failed_checks": ["candidate_bot_floor"]},
                {"pillar_id": "p08_controlled_canary_graduation", "failed_checks": ["canary_edge_ready"]},
            ]
        },
    )
    _write(health / "paper_profitability_control_latest.json", {"raw_profitability_grade": "A"})

    payload = rollup.build_payload(tmp_path)

    assert {row["root_id"] for row in payload["root_causes"]} == {
        "independent_fill_evidence",
        "promotion_candidate_coverage",
        "canary_cohort_evidence",
    }
    assert payload["unique_root_cause_count"] == 3


def test_soak_runtime_grade_maps_to_freshness_root(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write(
        health / "production_excellence_control_latest.json",
        {
            "pillars": [
                {
                    "pillar_id": "p02_clean_30_day_soak",
                    "failed_checks": ["soak_runtime_ready"],
                }
            ]
        },
    )
    _write(health / "paper_profitability_control_latest.json", {"raw_profitability_grade": "A"})

    payload = rollup.build_payload(tmp_path)

    assert payload["unique_root_cause_count"] == 1
    assert payload["root_causes"][0]["root_id"] == "readiness_artifact_freshness"
