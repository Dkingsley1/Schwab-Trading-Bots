import json
from pathlib import Path

from scripts.ops import roster_expansion_slots as slots
import scripts.run_all_sleeves as run_all_sleeves
import scripts.run_specialized_sleeve_shadow as specialized


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_new_quant_sleeves_are_registered_for_collection() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeve_names = {row["name"] for row in config["sleeves"]}

    for sleeve in {
        "signature_hawkes_generators",
        "crowd_physics_games",
        "lit_order_book_transformers",
        "critic_hmm_pinsde",
        "causal_omni_symbolic",
        "rlbf_dms_equivariant",
    }:
        assert sleeve in sleeve_names
        assert sleeve in config["ticker_universes"]
        assert sleeve in specialized.SLEEVE_DEFAULTS
        assert sleeve in run_all_sleeves.SPECIALIZED_SLEEVE_PROFILES
        assert specialized.SLEEVE_DEFAULTS[sleeve]["domain"] == "quant_models"


def test_quant_roster_slots_are_research_only_and_labeled() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}

    for bot_id in {
        "brain_refinery_v488_signature_hawkes_games_regression_guard_bot",
        "brain_refinery_v496_order_book_transformer_resource_guard_bot",
        "brain_refinery_v497_agentic_quant_memory_guard_bot",
        "brain_refinery_v504_causal_omni_symbolic_regression_guard_bot",
        "brain_refinery_v505_rlbf_dms_equivariant_resource_guard_bot",
    }:
        row = by_id[bot_id]
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["direct_execution_allowed"] is False
        assert row["execution_policy_label"] == "research_only_no_execution"
        assert row["eligible_for_master_vote"] is False
        assert row["label_contract"]["contract_version"] == "quant_research_labels_v1"
