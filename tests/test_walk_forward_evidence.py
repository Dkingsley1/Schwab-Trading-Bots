import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import walk_forward_validate as validator
from scripts import walk_forward_promotion_gate as gate

NOW = datetime(2026, 9, 14, 18, tzinfo=timezone.utc)


def log(
    root,
    name="brain_refinery_v35_dmi_20260912_120000.json",
    *,
    accuracy=0.7,
    payload=None,
):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    data = json.dumps(
        payload if payload is not None else {"metrics": {"test_accuracy": accuracy}}
    ).encode()
    path.write_bytes(gzip.compress(data) if name.endswith(".gz") else data)
    return path


def test_routed_gzip_and_plain_copies_count_once(tmp_path):
    local, external = tmp_path / "local", tmp_path / "external"
    first = log(local)
    log(external, first.name + ".gz")
    alias = tmp_path / "alias"
    alias.symlink_to(local, target_is_directory=True)
    groups, audit = validator._training_log_evidence([local, alias, external], now=NOW)
    assert audit["complete"]
    assert audit["duplicate_records"] == 1
    assert len(groups["brain_refinery_v35_dmi"]) == 1
    assert (
        validator.bot_id_from_log_name(first.name + ".gz") == "brain_refinery_v35_dmi"
    )


@pytest.mark.parametrize(
    "value", [None, True, "0.9", float("nan"), float("inf"), -0.1, 1.1]
)
def test_invalid_metrics_never_earn_runs(tmp_path, value):
    log(tmp_path, accuracy=value)
    groups, audit = validator._training_log_evidence([tmp_path], now=NOW)
    assert not groups
    assert audit["rejected_records"] == 1


@pytest.mark.parametrize("stamp", ["20261312_120000", "20260915_120000", "latest"])
def test_invalid_or_future_run_time_rejected(tmp_path, stamp):
    log(tmp_path, f"brain_refinery_v35_dmi_{stamp}.json")
    groups, audit = validator._training_log_evidence([tmp_path], now=NOW)
    assert not groups
    assert audit["rejected_records"] == 1


def test_diagnostic_only_does_not_earn_runs(tmp_path):
    log(tmp_path, payload={"diagnostic_only": True, "metrics": {"test_accuracy": 0.99}})
    assert not validator._training_log_evidence([tmp_path], now=NOW)[0]


def test_conflicting_run_cannot_be_selected_by_path_order(tmp_path):
    log(tmp_path / "a", accuracy=0.8)
    log(tmp_path / "b", accuracy=0.2)
    groups, audit = validator._training_log_evidence(
        [tmp_path / "a", tmp_path / "b"], now=NOW
    )
    assert not groups and not audit["complete"]
    assert "conflicting_training_run" in audit["errors"]


@pytest.mark.parametrize(
    "budget",
    [
        {"max_files": 1},
        {"max_entries": 1},
        {"max_file_bytes": 10},
        {"max_total_bytes": 10},
        {"timeout_seconds": 0},
    ],
)
def test_partial_budget_never_certifies_subset(tmp_path, budget):
    log(tmp_path)
    log(tmp_path, "brain_refinery_v35_dmi_20260913_120000.json", accuracy=0.8)
    groups, audit = validator._training_log_evidence([tmp_path], now=NOW, **budget)
    assert not groups and not audit["complete"]


def test_corrupt_gzip_and_missing_roots_are_distinct(tmp_path):
    path = log(tmp_path / "logs", "brain_refinery_v35_dmi_20260912_120000.json.gz")
    path.write_bytes(path.read_bytes()[:-5])
    groups, audit = validator._training_log_evidence(
        [path.parent, tmp_path / "missing"], now=NOW
    )
    assert not groups and not audit["complete"]
    assert audit["missing_roots"] == 1
    assert "log_read_failed" in audit["errors"]


def test_protected_root_and_alias_are_rejected_before_scandir(tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    alias.symlink_to("/Volumes/VIDEO", target_is_directory=True)
    monkeypatch.setattr(
        validator.os, "scandir", lambda _: pytest.fail("protected directory accessed")
    )
    groups, audit = validator._training_log_evidence(
        [alias, Path("/Volumes/VIDEO")], now=NOW
    )
    assert not groups and not audit["complete"]
    assert audit["error_count"] == 2


def test_protected_file_alias_is_never_opened(tmp_path, monkeypatch):
    alias = tmp_path / "brain_refinery_v35_dmi_20260912_120000.json"
    alias.symlink_to("/Volumes/VIDEO/private.json")
    monkeypatch.setattr(
        validator,
        "open",
        lambda *a, **k: pytest.fail("protected file opened"),
        raising=False,
    )
    groups, audit = validator._training_log_evidence([tmp_path], now=NOW)
    assert not groups and not audit["complete"]


def test_repair_boundary_excludes_every_old_run(tmp_path, monkeypatch, capsys):
    log(tmp_path / "logs")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v35_dmi",
                        "training_repair_status": "repaired",
                        "log_file": "brain_refinery_v35_dmi_20260913_120000.json.gz",
                    }
                ]
            }
        )
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(validator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        validator,
        "_configured_external_project_root_no_io",
        lambda: tmp_path / "external",
    )
    monkeypatch.setattr(
        validator.sys,
        "argv",
        ["validator", "--registry-file", str(registry), "--out", str(output)],
    )
    assert validator.main() == 0
    row = json.loads(output.read_text())["bots"]["brain_refinery_v35_dmi"]
    assert row["runs"] == 0
    assert row["lineage_reset_active"]
    assert row["pre_repair_runs_excluded"] == 1
    assert row["status"] == "insufficient_runs"


@pytest.mark.parametrize(
    "count,complete,expected",
    [(1, True, False), (3, True, False), (4, True, True), (4, False, False)],
)
def test_promotion_preserves_four_bot_floor_and_complete_scan(
    tmp_path, monkeypatch, capsys, count, complete, expected
):
    source, output = tmp_path / "source.json", tmp_path / "gate.json"
    source.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_evidence": {"complete": complete},
                "bots": {
                    f"bot{i}": {
                        "runs": 12,
                        "forward_mean": 0.7,
                        "delta": 0.02,
                        "trading_quality_score": 0.8,
                        "overfit_gap": 0,
                        "status": "pass",
                    }
                    for i in range(count)
                },
            }
        )
    )
    monkeypatch.setattr(gate, "_ops_thresholds", lambda: {})
    monkeypatch.setattr(
        validator.sys,
        "argv",
        [
            "gate",
            "--in-file",
            str(source),
            "--registry-file",
            str(tmp_path / "missing"),
            "--min-considered-bots",
            "4",
            "--out-file",
            str(output),
        ],
    )
    assert gate.main() == (0 if expected else 2)
    result = json.loads(output.read_text())
    assert result["promote_ok"] is expected
    assert result["effective_thresholds"]["min_considered_bots"] == 4
    assert result["coverage_shortfall_bots"] == max(4 - count, 0)


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "invalid",
        "2026-09-14T18:00:00",
        "2026-09-14T18:00:01+00:00",
        "2026-09-14T17:44:59+00:00",
    ],
)
def test_stale_or_invalid_scan_cannot_refresh_promotion(timestamp):
    ready, _ = gate._source_evidence_ready(
        {"timestamp_utc": timestamp, "source_evidence": {"complete": True}}, NOW
    )
    assert not ready


@pytest.mark.parametrize(
    "evidence",
    [None, [], {}, {"complete": "true"}, {"complete": 1}, {"complete": False}],
)
def test_scan_completeness_requires_typed_proof(evidence):
    ready, _ = gate._source_evidence_ready(
        {"timestamp_utc": NOW.isoformat(), "source_evidence": evidence}, NOW
    )
    assert not ready
