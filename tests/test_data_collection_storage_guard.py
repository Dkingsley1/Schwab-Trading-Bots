from scripts.ops import data_collection_storage_guard as src


def test_quant_research_collectors_use_lighter_storage_profile() -> None:
    row = {
        "slot_kind": "graph_attention_cross_asset_spillover",
        "bot_role": "signal_sub_bot",
        "data_label_contract_version": "quant_research_labels_v1",
        "data_intake_collections": ["mlx_graph_library_profile"],
    }

    assert src._collector_kind(row) == "quant_research"

    profile = src._guard_profile("throttle", "quant_research")

    assert profile["capture_mode"] == "metadata_only"
    assert profile["sample_rate"] <= 0.08
    assert profile["max_daily_storage_mb"] <= 20
