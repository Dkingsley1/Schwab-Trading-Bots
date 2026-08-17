import json
from pathlib import Path

import pytest

from scripts.ops import profitability_holdout_vault as vault


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_holdout_vault_detects_mutation_and_limits_access(tmp_path: Path) -> None:
    config = json.loads(vault.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / vault.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-1", "generation": 1},
    )
    dataset = tmp_path / "holdout.jsonl"
    dataset.write_text('{"label": 1}\n', encoding="utf-8")

    manifest = vault.seal_dataset(tmp_path, config, dataset)
    ready = vault.build_payload(tmp_path, config_path=config_path)
    access = vault.record_evaluation_access(tmp_path, config, evidence="evaluation-report-sha256:abc")

    assert manifest["training_access_forbidden"] is True
    assert ready["evidence_ready"] is True
    assert access["purpose"] == "evaluation"
    with pytest.raises(RuntimeError, match="access limit"):
        vault.record_evaluation_access(tmp_path, config, evidence="second-evaluation")

    dataset.write_text('{"label": 0}\n', encoding="utf-8")
    tampered = vault.build_payload(tmp_path, config_path=config_path)

    assert tampered["evidence_ready"] is False
    assert "sealed_holdout_digest_mismatch" in tampered["blockers"]


def test_holdout_vault_requires_evidence_reference(tmp_path: Path) -> None:
    config = json.loads(vault.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / vault.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-1", "generation": 1},
    )
    dataset = tmp_path / "holdout.jsonl"
    dataset.write_text('{"label": 1}\n', encoding="utf-8")
    vault.seal_dataset(tmp_path, config, dataset)

    with pytest.raises(ValueError, match="evidence reference"):
        vault.record_evaluation_access(tmp_path, config, evidence="")
