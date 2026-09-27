import sys
import json
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.golden_replay_regression_guard as src


@pytest.fixture
def reference_pack(monkeypatch):
    for name in os.environ:
        if name.startswith(("RISK_MAX_", "SIZING_")):
            monkeypatch.delenv(name)
    return json.loads(src.DEFAULT_PACK_PATH.read_text())


def test_tracked_reference_has_independent_full_expectations(reference_pack):
    payload = src.build_payload(golden_pack=reference_pack)
    assert src.DEFAULT_PACK_PATH.parent.name == "config"
    assert payload["strict_ready"] is True
    assert payload["case_count"] == 9
    assert payload["missing_coverage"] == []
    assert payload["strategy_profitability_proven"] is False
    assert payload["order_authorized"] is False
    assert all(row["canonical_match"] for row in payload["cases"])
    normal = reference_pack["cases"][0]["expected_canonical"]
    assert normal["results"][0]["raw_qty"] == 3000.0
    assert normal["results"][0]["alloc_qty"] == 1200.0


def test_reference_rejects_changed_sizing_without_rebaselining(reference_pack, monkeypatch):
    monkeypatch.setattr(src.replay_src, "size_from_action", lambda **kwargs: 0.0)
    payload = src.build_payload(golden_pack=reference_pack)
    assert payload["strict_ready"] is False
    assert "normal_actions" in payload["failed_cases"]
    assert "allocation_cap" in payload["failed_cases"]


def test_reference_does_not_override_runtime_risk_policy(reference_pack, monkeypatch):
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS_PROXY", "0.10")
    payload = src.build_payload(golden_pack=reference_pack)
    assert payload["strict_ready"] is False
    assert "daily_loss_halt" in payload["failed_cases"]
    assert os.environ["RISK_MAX_DAILY_LOSS_PROXY"] == "0.10"


@pytest.mark.parametrize("reference", [None, {}, {"results": [], "exposure_state": {}}])
def test_invalid_reference_fails_closed(reference_pack, reference):
    reference_pack["cases"][0]["expected_canonical"] = reference
    payload = src.build_payload(golden_pack=reference_pack)
    assert payload["strict_ready"] is False
    assert payload["cases"][0]["reference_valid"] is False


def test_malformed_case_cannot_be_ignored(reference_pack):
    reference_pack["cases"].append(None)
    payload = src.build_payload(golden_pack=reference_pack)
    assert payload["strict_ready"] is False
    assert payload["invalid_cases"] == ["case_9"]


def test_engine_exception_is_blocked_evidence(reference_pack, monkeypatch):
    def broken_replay(_):
        raise ValueError("bad fixture")

    monkeypatch.setattr(src.replay_src, "run_replay", broken_replay)
    payload = src.build_payload(golden_pack=reference_pack)
    assert payload["strict_ready"] is False
    assert payload["cases"][0]["replay_error"] == "ValueError"


def test_missing_pack_without_registry_is_blocked():
    payload = src.build_payload(golden_pack={})
    assert payload["ok"] is False
    assert payload["strict_ready"] is False


def test_reference_hash_must_agree_when_both_are_declared(reference_pack):
    reference_pack["cases"][0]["expected_hash"] = "0" * 64
    payload = src.build_payload(golden_pack=reference_pack)
    assert payload["strict_ready"] is False
    assert payload["cases"][0]["reference_valid"] is False


def test_cli_records_actual_input_paths(reference_pack, tmp_path, monkeypatch):
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(reference_pack))
    registry = tmp_path / "registry.json"
    output = tmp_path / "output.json"
    monkeypatch.setattr(sys, "argv", ["guard", "--pack-file", str(pack),
        "--replay-hash-registry-file", str(registry), "--out-file", str(output)])
    assert src.main() == 0
    payload = json.loads(output.read_text())
    assert payload["source_artifacts"]["golden_replay_pack"] == str(pack)
    assert payload["source_artifacts"]["replay_hash_registry_guard"] == str(registry)


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
