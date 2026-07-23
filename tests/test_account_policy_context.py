import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "account_policy_context.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("account_policy_context", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load account_policy_context")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_account_policy_context_uses_safe_defaults_without_exposing_secrets(tmp_path: Path) -> None:
    module = _load_module()

    payload = module.build_payload(tmp_path, registry_path=tmp_path / "missing.json")

    assert payload["overall_status"] == "ready"
    assert payload["coverage"]["configured_account_slots"] == 3
    assert payload["coverage"]["margin_slots"] == 0
    assert payload["bot_contract"]["auto_order_enabled"] is False
    assert payload["bot_contract"]["day_trading_rule_awareness"] == "finra_intraday_margin_replaces_legacy_pdt"
    assert payload["bot_contract"]["day_trade_widening_allowed"] is False
    redaction = payload["account_policy_context"]["redaction_contract"]
    assert redaction["account_numbers_exposed_in_policy"] is False
    assert redaction["account_hashes_exposed_in_policy"] is False
    transition = payload["account_policy_context"]["pdt_intraday_margin_transition"]
    assert transition["finra_effective_date"] == "2026-06-04"
    assert transition["schwab_day_trade_count_retire_date"] == "2026-06-08"
    assert transition["phase_in_end_date"] == "2027-10-20"
    probe = payload["account_policy_context"]["intraday_margin_probe_contract"]
    assert probe["status"] == "scheduled_pre_schwab_cutover"
    assert payload["bot_contract"]["intraday_margin_probe_status"] == "scheduled_pre_schwab_cutover"
    assert (
        payload["bot_contract"]["broker_developer_platform_order_limit_policy"]
        == "operator_managed_external_throttle_not_internal_scalability_ceiling"
    )


def test_account_policy_context_reads_registry_and_blocks_auto_order(tmp_path: Path) -> None:
    module = _load_module()
    registry = tmp_path / "account_policy_registry.json"
    registry.write_text(
        json.dumps(
            {
                "account_slots": [
                    {
                        "account_policy_key": "paper_test",
                        "account_type": "cash",
                        "tax_treatment": "taxable",
                        "broker": "schwab",
                        "env_names": ["SCHWAB_TEST_ACCOUNT_HASH"],
                        "auto_order_enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(tmp_path, registry_path=registry)

    assert payload["overall_status"] == "blocked"
    assert payload["account_policy_context"]["registry_present"] is True
    assert payload["bot_contract"]["auto_order_enabled"] is True


def test_account_policy_context_tracks_schwab_cutover_dates(tmp_path: Path) -> None:
    module = _load_module()

    pre_cutover = module.build_payload(
        tmp_path,
        registry_path=tmp_path / "missing.json",
        as_of_date="2026-05-29",
    )
    assert pre_cutover["bot_contract"]["legacy_pdt_framework_active_for_schwab_policy"] is True
    assert pre_cutover["bot_contract"]["schwab_day_trade_count_retired"] is False
    assert pre_cutover["bot_contract"]["pdt_transition_phase"] == "legacy_pdt_until_finra_effective_date"

    schwab_cutover = module.build_payload(
        tmp_path,
        registry_path=tmp_path / "missing.json",
        as_of_date="2026-06-08",
    )
    assert schwab_cutover["bot_contract"]["legacy_pdt_framework_active_for_schwab_policy"] is False
    assert schwab_cutover["bot_contract"]["schwab_day_trade_count_retired"] is True
    assert schwab_cutover["bot_contract"]["day_trade_widening_allowed"] is False
    assert schwab_cutover["bot_contract"]["intraday_margin_probe_status"] == "needs_broker_intraday_margin_probe"
    assert schwab_cutover["account_policy_context"]["intraday_margin_probe_contract"]["probe_required_now"] is True
    assert (
        schwab_cutover["account_policy_context"]["pdt_intraday_margin_transition"]["phase"]
        == "schwab_day_trade_count_retired_intraday_margin_phase_in"
    )


def test_account_policy_context_requires_intraday_margin_context_for_margin_accounts(tmp_path: Path) -> None:
    module = _load_module()
    registry = tmp_path / "account_policy_registry.json"
    registry.write_text(
        json.dumps(
            {
                "account_slots": [
                    {
                        "account_policy_key": "schwab_margin_test",
                        "account_type": "margin",
                        "tax_treatment": "taxable",
                        "broker": "schwab",
                        "margin_enabled": True,
                        "env_names": ["SCHWAB_MARGIN_TEST_HASH"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(tmp_path, registry_path=registry, as_of_date="2026-06-10")

    policy = payload["account_policy_context"]["slot_margin_policies"][0]
    assert policy["margin_enabled"] is True
    assert policy["requires_intraday_margin_buying_power_confirmation"] is True
    assert policy["day_trade_widening_allowed"] is False
    assert policy["pdt_or_intraday_margin_applicability"] == (
        "schwab_intraday_margin_framework_pending_buying_power_confirmation"
    )


def test_account_policy_context_detects_broker_intraday_buying_power(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "broker_truth_equities_schwab_latest.json",
        {
            "overall_status": "ready",
            "account_metrics": {
                "intraday_buying_power": 2500.0,
                "available_funds": 1200.0,
                "buying_power": 1500.0,
                "equity": 3000.0,
            },
        },
    )

    payload = module.build_payload(tmp_path, registry_path=tmp_path / "missing.json", as_of_date="2026-06-08")
    probe = payload["account_policy_context"]["intraday_margin_probe_contract"]

    assert probe["status"] == "ready"
    assert probe["intraday_buying_power_observed"] is True
    assert probe["intraday_buying_power_source_key"] == "intraday_buying_power"
    assert payload["bot_contract"]["intraday_margin_buying_power_observed"] is True


def test_account_policy_context_reads_current_schwab_shared_snapshot_intraday_power(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "broker_truth_shared_snapshot_schwab_latest.json",
        {
            "overall_status": "ready",
            "fetched": {
                "payload": {
                    "securitiesAccount": {
                        "initialBalances": {
                            "dayTradingBuyingPower": 128772.0,
                            "buyingPower": 128772.0,
                            "equity": 41000.0,
                        },
                        "currentBalances": {
                            "dayTradingBuyingPower": 1288.38,
                            "buyingPower": 1288.38,
                            "availableFunds": 1288.38,
                            "cashBalance": 1288.38,
                            "liquidationValue": 41353.1,
                        },
                    }
                }
            },
        },
    )

    payload = module.build_payload(tmp_path, registry_path=tmp_path / "missing.json", as_of_date="2026-06-08")
    probe = payload["account_policy_context"]["intraday_margin_probe_contract"]

    assert probe["status"] == "ready"
    assert probe["intraday_buying_power_source_key"] == "intraday_buying_power"
    assert probe["intraday_buying_power"] == 1288.38
    assert probe["buying_power"] == 1288.38


def test_account_policy_context_simulates_paper_intraday_margin_deficit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "broker_truth_equities_schwab_latest.json",
        {
            "overall_status": "ready",
            "account_metrics": {
                "available_funds": 1000.0,
                "buying_power": 1000.0,
            },
        },
    )
    monkeypatch.setenv("PAPER_INTRADAY_MARGIN_SIM_EXPOSURE_USD", "1500")

    payload = module.build_payload(tmp_path, registry_path=tmp_path / "missing.json", as_of_date="2026-06-08")
    simulator = payload["account_policy_context"]["paper_intraday_margin_deficit_simulator"]

    assert simulator["status"] == "deficit_simulated"
    assert simulator["live_execution_allowed"] is False
    assert simulator["simulated_margin_deficit_usd"] == 500.0
