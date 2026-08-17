import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.golden_replay_regression_guard as src


def test_golden_replay_guard_allows_seed_ready_registry_when_pack_is_missing() -> None:
    payload = src.build_payload(
        golden_pack={},
        replay_hash_registry={
            "ok": True,
            "details": {
                "paper": {"current_hash": "paper-hash"},
                "e2e": {"current_hash": "e2e-hash"},
            },
        },
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "degraded"
    assert payload["seed_ready"] is True
    assert payload["case_count"] == 0


def test_golden_replay_guard_is_ready_with_matching_pack_case() -> None:
    replay = src.replay_src.run_replay(src.replay_src._default_payload())
    actions = {
        row["symbol"]: row["action_out"]
        for row in replay["canonical"]["results"]
    }

    payload = src.build_payload(
        golden_pack={
            "schema_version": 1,
            "cases": [
                {
                    "name": "default_case",
                    "payload": src.replay_src._default_payload(),
                    "expected_hash": replay["replay_hash"],
                    "expected_actions": actions,
                }
            ],
        },
        replay_hash_registry={
            "ok": True,
            "details": {"paper": {"current_hash": "paper-hash"}},
        },
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["case_count"] == 1
    assert payload["failed_case_count"] == 0
    assert payload["strict_ready"] is True


def test_golden_replay_guard_requires_declared_coverage() -> None:
    replay = src.replay_src.run_replay(src.replay_src._default_payload())

    payload = src.build_payload(
        golden_pack={
            "schema_version": 2,
            "required_coverage": ["normal_buy_sell", "daily_loss_halt"],
            "cases": [
                {
                    "name": "default_case",
                    "coverage": ["normal_buy_sell"],
                    "payload": src.replay_src._default_payload(),
                    "expected_hash": replay["replay_hash"],
                }
            ],
        },
        replay_hash_registry={"ok": True, "details": {"paper": {"current_hash": "paper-hash"}}},
    )

    assert payload["ok"] is False
    assert payload["strict_ready"] is False
    assert payload["missing_coverage"] == ["daily_loss_halt"]
