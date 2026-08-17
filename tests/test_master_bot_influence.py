import scripts.run_shadow_training_loop as loop
import pytest


def test_parse_sub_bots_applies_configured_weight_boost(monkeypatch) -> None:
    monkeypatch.setenv(
        "MASTER_BOT_WEIGHT_BOOSTS",
        "brain_refinery_v10_seasonal:1.40,brain_refinery_v35_dmi_state_machine:1.25",
    )

    bots = loop._parse_sub_bots(
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "weight": 0.20,
                    "active": True,
                    "reason": "ok",
                    "test_accuracy": 0.92,
                },
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "weight": 0.18,
                    "active": True,
                    "reason": "ok",
                    "test_accuracy": 0.83,
                },
                {
                    "bot_id": "brain_refinery_v24_vwap_deviation",
                    "weight": 0.17,
                    "active": True,
                    "reason": "ok",
                    "test_accuracy": 0.50,
                },
            ]
        }
    )

    by_id = {bot.bot_id: bot for bot in bots}
    assert by_id["brain_refinery_v10_seasonal"].weight == pytest.approx(0.28)
    assert by_id["brain_refinery_v35_dmi_state_machine"].weight == pytest.approx(0.225)
    assert by_id["brain_refinery_v24_vwap_deviation"].weight == pytest.approx(0.17)


def test_parse_bot_weight_boosts_clamps_invalid_values() -> None:
    boosts = loop._parse_bot_weight_boosts(
        "brain_refinery_v10_seasonal:5.5,brain_refinery_v35_dmi_state_machine:-1,bad,nope:abc"
    )

    assert boosts["brain_refinery_v10_seasonal"] == 3.0
    assert boosts["brain_refinery_v35_dmi_state_machine"] == 0.10
    assert "bad" not in boosts
    assert "nope" not in boosts
