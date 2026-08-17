from core.master_bot import MasterBot


def test_infer_bot_role_classifies_options_and_futures() -> None:
    assert MasterBot._infer_bot_role("brain_refinery_options_iv_surface") == "options_sub_bot"
    assert MasterBot._infer_bot_role("brain_refinery_futures_funding_basis") == "futures_sub_bot"
    assert MasterBot._infer_bot_role("brain_refinery_v71_champion_challenger_layer") == "infrastructure_sub_bot"


def test_normalize_registry_row_forces_deleted_inactive() -> None:
    row = MasterBot._normalize_registry_row(
        {
            "bot_id": "brain_refinery_v20_garch",
            "active": True,
            "deleted_from_rotation": True,
            "lifecycle_state": "active",
            "weight": 0.25,
        }
    )

    assert row["active"] is False
    assert row["deleted_from_rotation"] is True
    assert row["lifecycle_state"] == "deleted"
    assert row["weight"] == 0.0


def test_normalize_registry_row_preserves_data_collection_only() -> None:
    row = MasterBot._normalize_registry_row(
        {
            "bot_id": "brain_refinery_v498_double_machine_learning_causal_bot",
            "active": True,
            "deleted_from_rotation": False,
            "lifecycle_state": "data_collection_only",
            "weight": 0.0,
        }
    )

    assert row["active"] is True
    assert row["lifecycle_state"] == "data_collection_only"
