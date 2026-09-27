from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts.ops import schwab_account_hash_keychain_sync as sync_src


def _policy() -> tuple[dict, dict]:
    aliases = {
        "schwab_accounts": {
            "tail:2831": {
                "account_policy_key": "schwab_cash_account_1",
                "operator_verified": True,
            },
            "tail:5625": {
                "account_policy_key": "schwab_roth_ira_primary",
                "operator_verified": True,
            },
        }
    }
    registry = {
        "account_slots": [
            {
                "account_policy_key": "schwab_cash_account_1",
                "canary_candidate": True,
                "env_names": [
                    "SCHWAB_CASH_ACCOUNT_1_HASH",
                    "SCHWAB_TAXABLE_ACCOUNT_1_HASH",
                ],
            },
            {
                "account_policy_key": "schwab_roth_ira_primary",
                "canary_candidate": False,
                "env_names": [
                    "SCHWAB_ACCOUNT_HASH",
                    "SCHWAB_ROTH_ACCOUNT_HASH",
                ],
            },
        ]
    }
    return aliases, registry


def test_sync_plan_maps_only_operator_verified_policy_slots() -> None:
    aliases, registry = _policy()
    plan = sync_src._build_sync_plan(
        account_rows=[
            {"account_number": "12342831", "account_hash": "opaque-2831"},
            {"account_number": "****5625", "account_hash": "opaque-5625"},
        ],
        aliases=aliases,
        registry=registry,
    )

    assert plan["ready"] is True
    assert plan["candidate_binding_count"] == 1
    candidate = next(row for row in plan["bindings"] if row["canary_candidate"])
    assert candidate["account_number_tail"] == "2831"
    assert candidate["env_names"] == [
        "SCHWAB_CASH_ACCOUNT_1_HASH",
        "SCHWAB_TAXABLE_ACCOUNT_1_HASH",
    ]


def test_keychain_apply_verifies_storage_without_emitting_raw_hashes() -> None:
    aliases, registry = _policy()
    plan = sync_src._build_sync_plan(
        account_rows=[
            {"account_number": "12342831", "account_hash": "opaque-2831"},
            {"account_number": "12345625", "account_hash": "opaque-5625"},
        ],
        aliases=aliases,
        registry=registry,
    )
    stored: dict[tuple[str, str], str] = {}

    def write(service: str, account: str, value: str) -> tuple[bool, str]:
        stored[(service, account)] = value
        return True, "stored"

    def read(service: str, account: str) -> str:
        return stored.get((service, account), "")

    result = sync_src._apply_sync_plan(
        plan,
        account="test-user",
        write_keychain=write,
        read_keychain=read,
    )

    assert result["ok"] is True
    assert result["stored_binding_count"] == 3
    serialized = json.dumps(result)
    assert "opaque-2831" not in serialized
    assert "opaque-5625" not in serialized


def test_invalid_account_mapping_fails_before_any_keychain_write() -> None:
    aliases, registry = _policy()
    plan = sync_src._build_sync_plan(
        account_rows=[
            {"account_number": "12342831", "account_hash": "opaque-2831"},
            {"account_number": "12349999", "account_hash": "opaque-9999"},
        ],
        aliases=aliases,
        registry=registry,
    )
    writes: list[str] = []

    result = sync_src._apply_sync_plan(
        plan,
        account="test-user",
        write_keychain=lambda service, _account, _value: (
            writes.append(service) is None,
            "stored",
        ),
        read_keychain=lambda _service, _account: "",
    )

    assert plan["ready"] is False
    assert result["ok"] is False
    assert writes == []


def test_live_runtime_pins_global_hash_to_selected_policy_variable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "SCHWAB_KEYCHAIN_FALLBACK_ENABLED": "0",
            "SCHWAB_LIVE_ACCOUNT_POLICY_KEY": "schwab_cash_account_1",
            "SCHWAB_ACCOUNT_HASH": "stale-taxable-account-hash",
            "SCHWAB_CASH_ACCOUNT_1_HASH": "stale-taxable-account-hash",
            "SCHWAB_ROTH_ACCOUNT_HASH": "opaque-candidate-hash",
        }
    )
    proc = subprocess.run(
        [
            "zsh",
            "-lc",
            (
                f"source {project_root}/scripts/ops/load_runtime_env.sh live --quiet; "
                '[[ "$SCHWAB_ACCOUNT_HASH" == "opaque-candidate-hash" ]]; '
                '[[ "$SCHWAB_LIVE_ACCOUNT_POLICY_KEY" == "schwab_roth_ira_primary" ]]; '
                '[[ "$SCHWAB_LIVE_ACCOUNT_POLICY_SOURCE" == "canary_plan" ]]; '
                '[[ "$SCHWAB_ACCOUNT_HASH_SOURCE" == "policy_bound_keychain" ]]; '
                '[[ "$SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER" == "0" ]]'
            ),
        ],
        cwd=str(project_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
