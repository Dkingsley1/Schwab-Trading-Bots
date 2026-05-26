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


def test_account_policy_context_uses_safe_defaults_without_exposing_secrets(tmp_path: Path) -> None:
    module = _load_module()

    payload = module.build_payload(tmp_path, registry_path=tmp_path / "missing.json")

    assert payload["overall_status"] == "ready"
    assert payload["coverage"]["configured_account_slots"] == 3
    assert payload["bot_contract"]["auto_order_enabled"] is False
    redaction = payload["account_policy_context"]["redaction_contract"]
    assert redaction["account_numbers_exposed_in_policy"] is False
    assert redaction["account_hashes_exposed_in_policy"] is False


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
