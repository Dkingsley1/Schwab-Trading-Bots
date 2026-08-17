from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import roster_expansion_slots as slots_src


ALPHA_RESEARCH_OS_BOT_IDS = {
    "brain_refinery_v754_alpha_hypothesis_knowledge_graph_bot",
    "brain_refinery_v755_bayesian_evidence_score_bot",
    "brain_refinery_v756_duplicate_alpha_similarity_detector_bot",
    "brain_refinery_v757_compute_capital_allocator_bot",
    "brain_refinery_v758_automatic_research_committee_bot",
    "brain_refinery_v759_active_learning_experiment_designer_bot",
    "brain_refinery_v760_causal_intervention_natural_experiment_bot",
    "brain_refinery_v761_semantic_feature_ontology_harmonizer_bot",
    "brain_refinery_v762_bayesian_model_averaging_ensemble_bot",
    "brain_refinery_v763_research_debt_sunset_policy_bot",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_alpha_research_os_slots_are_planned() -> None:
    specs = {
        str(row.get("bot_id") or ""): row
        for row in slots_src.DEFAULT_SLOT_SPECS
        if str(row.get("bot_id") or "") in ALPHA_RESEARCH_OS_BOT_IDS
    }

    assert set(specs) == ALPHA_RESEARCH_OS_BOT_IDS
    assert {row["sleeve_profile"] for row in specs.values()} == {
        "alpha_research_os",
        "research_meta_governance",
    }
    assert all(row["bot_role"] == "infrastructure_sub_bot" for row in specs.values())
    assert all(row["sleeve_family"] == "quant_models" for row in specs.values())
    assert "alpha_hypothesis_graph_trace" in specs["brain_refinery_v754_alpha_hypothesis_knowledge_graph_bot"]["data_intake_collections"]
    assert "bayesian_evidence_score_trace" in specs["brain_refinery_v755_bayesian_evidence_score_bot"]["data_intake_collections"]
    assert "compute_capital_allocation_trace" in specs["brain_refinery_v757_compute_capital_allocator_bot"]["data_intake_collections"]
    assert "active_learning_experiment_trace" in specs["brain_refinery_v759_active_learning_experiment_designer_bot"]["data_intake_collections"]
    assert "research_debt_sunset_trace" in specs["brain_refinery_v763_research_debt_sunset_policy_bot"]["data_intake_collections"]


def test_alpha_research_os_apply_keeps_collection_only_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    apply_result = slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("bot_id") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in ALPHA_RESEARCH_OS_BOT_IDS
    }

    assert set(rows) == ALPHA_RESEARCH_OS_BOT_IDS
    assert apply_result["added_slots"] >= len(ALPHA_RESEARCH_OS_BOT_IDS)
    for row in rows.values():
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["data_collection_active"] is True
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["allocation_enabled"] is False
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["direct_execution_allowed"] is False
        assert row["minimum_training_observations"] >= 3000
        assert "research_only" in row["labeling_tags"]


def test_alpha_research_os_provider_and_storage_contracts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("sleeve_profile") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in {
            "brain_refinery_v754_alpha_hypothesis_knowledge_graph_bot",
            "brain_refinery_v759_active_learning_experiment_designer_bot",
        }
    }

    assert rows["alpha_research_os"]["provider_capability_profile"] == "research_only_alpha_research_os_guard"
    assert rows["research_meta_governance"]["provider_capability_profile"] == "research_only_research_meta_governance_guard"
    assert "governance/quant_models/alpha_research_os" in rows["alpha_research_os"]["storage_targets"]
    assert "governance/research_os" in rows["alpha_research_os"]["storage_targets"]
    assert "governance/quant_models/research_meta_governance" in rows["research_meta_governance"]["storage_targets"]
    assert "governance/feature_store" in rows["research_meta_governance"]["storage_targets"]


def test_sleeve_strategy_manifest_includes_alpha_research_os_wave() -> None:
    manifest = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {str(row.get("name") or ""): row for row in manifest["sleeves"]}

    assert set(sleeves) >= {
        "alpha_research_os",
        "research_meta_governance",
    }
    assert len(sleeves["alpha_research_os"]["strategies"]) == 5
    assert len(sleeves["research_meta_governance"]["strategies"]) == 5
