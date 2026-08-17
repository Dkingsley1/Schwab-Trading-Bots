from pathlib import Path
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.model_lifecycle_hygiene as hygiene


def test_artifact_state_distinguishes_log_only_from_hard_missing() -> None:
    assert hygiene._artifact_state(model_ok=True, log_ok=True) == "ok"
    assert hygiene._artifact_state(model_ok=True, log_ok=False) == "missing_log_only"
    assert hygiene._artifact_state(model_ok=False, log_ok=True) == "missing_model_only"
    assert hygiene._artifact_state(model_ok=False, log_ok=False) == "missing_both"


def test_latest_log_artifact_for_bot_finds_non_json_logs(tmp_path) -> None:
    root = tmp_path / "project"
    log_dir = root / "governance" / "training_diagnostics"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact = log_dir / "brain_refinery_v9_gamma_failure.log"
    artifact.write_text("failure", encoding="utf-8")

    resolved = hygiene._latest_log_artifact_for_bot(root, "brain_refinery_v9_gamma")

    assert resolved == artifact


def test_latest_log_artifact_for_bot_does_not_cross_match_shorter_bot_ids(tmp_path) -> None:
    root = tmp_path / "project"
    diag_dir = root / "governance" / "training_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    wrong = diag_dir / "brain_refinery_v35_dmi_state_machine_latest.json"
    wrong.write_text("{}", encoding="utf-8")

    resolved = hygiene._latest_log_artifact_for_bot(root, "brain_refinery_v3")

    assert resolved is None


def test_main_downgrades_active_bot_with_stale_training_diagnostic(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(hygiene, "PROJECT_ROOT", root)
    model_path = root / "models" / "brain_refinery_v4_simple.npz"
    log_path = root / "logs" / "brain_refinery_v4_simple.log"
    diagnostic_path = root / "governance" / "training_diagnostics" / "brain_refinery_v4_simple_latest.json"
    registry_path = root / "master_bot_registry.json"
    out_file = root / "governance" / "lifecycle" / "model_lifecycle_latest.json"
    manifest_file = root / "governance" / "lifecycle" / "model_manifest_latest.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    diagnostic_path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")

    stale_epoch = 1_700_000_000
    os.utime(diagnostic_path, (stale_epoch, stale_epoch))

    registry_path.write_text(
        json.dumps(
            {
                "summary": {"active_bots": 1, "inactive_bots": 0},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v4_simple",
                        "active": True,
                        "lifecycle_state": "active",
                        "model_path": str(model_path),
                        "log_file": str(log_path),
                    }
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_lifecycle_hygiene.py",
            "--registry",
            str(registry_path),
            "--out-file",
            str(out_file),
            "--manifest-file",
            str(manifest_file),
            "--max-training-diagnostic-age-hours",
            "1",
            "--no-repair-stale-artifacts",
            "--apply-diagnostic-downgrade",
            "--json",
        ],
    )

    rc = hygiene.main()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = updated_registry["sub_bots"][0]

    assert rc == 2
    assert payload["stale_active_training_diagnostics"] == 1
    assert payload["repair"]["downgraded_for_stale_diagnostics"] == 1
    assert payload["repair"]["registry_updated"] is True
    assert row["active"] is False
    assert row["lifecycle_state"] == "probation"
    assert row["reason"] == "stale_training_diagnostic"


def test_main_downgrades_runtime_input_gap_bot(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(hygiene, "PROJECT_ROOT", root)
    model_path = root / "models" / "brain_refinery_v21_flash_crash.npz"
    log_path = root / "logs" / "brain_refinery_v21_flash_crash.log"
    diagnostic_path = root / "governance" / "training_diagnostics" / "brain_refinery_v21_flash_crash_latest.json"
    registry_path = root / "master_bot_registry.json"
    out_file = root / "governance" / "lifecycle" / "model_lifecycle_latest.json"
    manifest_file = root / "governance" / "lifecycle" / "model_manifest_latest.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    diagnostic_path.write_text(
        json.dumps(
            {
                "status": "deferred_sample_starved",
                "sample_count": 0,
                "eligible_sequences": 0,
                "sequence_count": 0,
            }
        ),
        encoding="utf-8",
    )

    registry_path.write_text(
        json.dumps(
            {
                "summary": {"active_bots": 1, "inactive_bots": 0},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v21_flash_crash",
                        "active": True,
                        "lifecycle_state": "active",
                        "model_path": str(model_path),
                        "log_file": str(log_path),
                    }
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_lifecycle_hygiene.py",
            "--registry",
            str(registry_path),
            "--out-file",
            str(out_file),
            "--manifest-file",
            str(manifest_file),
            "--apply-runtime-input-downgrade",
            "--json",
        ],
    )

    rc = hygiene.main()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = updated_registry["sub_bots"][0]

    assert rc == 0
    assert payload["repair"]["downgraded_for_runtime_input_gaps"] == 1
    assert payload["repair"]["registry_updated"] is True
    assert row["active"] is False
    assert row["lifecycle_state"] == "probation"
    assert row["reason"] == "unsupported_runtime_inputs"
    assert row["runtime_input_gap_cause"] == "shared_runtime_input_gap"


def test_main_excludes_collection_only_active_rows_from_artifact_hygiene(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(hygiene, "PROJECT_ROOT", root)
    registry_path = root / "master_bot_registry.json"
    out_file = root / "governance" / "lifecycle" / "model_lifecycle_latest.json"
    manifest_file = root / "governance" / "lifecycle" / "model_manifest_latest.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "summary": {"active_bots": 1, "inactive_bots": 0},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v900_collection_seed",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "model_path": "",
                        "log_file": "",
                    }
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_lifecycle_hygiene.py",
            "--registry",
            str(registry_path),
            "--out-file",
            str(out_file),
            "--manifest-file",
            str(manifest_file),
            "--json",
        ],
    )

    rc = hygiene.main()

    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["active_bots"] == 1
    assert payload["supportability_active_bots"] == 0
    assert payload["active_collection_only_bots"] == 1
    assert payload["missing_active_artifacts_total"] == 0
    assert payload["stale_active_training_diagnostics"] == 0


def test_main_repairs_stale_active_diagnostic_from_latest_training_artifact(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(hygiene, "PROJECT_ROOT", root)
    model_path = root / "models" / "brain_refinery_v10_seasonal.npz"
    log_path = root / "governance" / "training_diagnostics" / "brain_refinery_v10_seasonal_20260421.json"
    diagnostic_path = root / "governance" / "training_diagnostics" / "brain_refinery_v10_seasonal_latest.json"
    registry_path = root / "master_bot_registry.json"
    out_file = root / "governance" / "lifecycle" / "model_lifecycle_latest.json"
    manifest_file = root / "governance" / "lifecycle" / "model_manifest_latest.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model", encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-21T20:00:00+00:00",
                "metrics": {
                    "acted_count": 12,
                    "acted_accuracy": 0.71,
                    "accuracy_lift_over_majority": 0.08,
                    "positive_rate": 0.44,
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostic_path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")

    stale_epoch = 1_700_000_000
    fresh_epoch = stale_epoch + 7200
    os.utime(diagnostic_path, (stale_epoch, stale_epoch))
    os.utime(log_path, (fresh_epoch, fresh_epoch))

    registry_path.write_text(
        json.dumps(
            {
                "summary": {"active_bots": 1, "inactive_bots": 0},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v10_seasonal",
                        "active": True,
                        "lifecycle_state": "active",
                        "model_path": str(model_path),
                        "log_file": str(log_path),
                    }
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_lifecycle_hygiene.py",
            "--registry",
            str(registry_path),
            "--out-file",
            str(out_file),
            "--manifest-file",
            str(manifest_file),
            "--max-training-diagnostic-age-hours",
            "1",
            "--json",
        ],
    )

    rc = hygiene.main()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    repaired = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["stale_active_training_diagnostics"] == 0
    assert payload["repair"]["fixed_count"] == 1
    assert repaired["run_tag"] == "brain_refinery_v10_seasonal"
    assert repaired["repaired_from_log"] is True
    assert repaired["runtime_meta"]["recovery_source_log_path"] == str(log_path)
