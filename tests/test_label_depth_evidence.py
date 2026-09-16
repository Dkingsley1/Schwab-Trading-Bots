from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core import runtime_training_common as runtime
from core.training_diagnostic_contract import materialization_contract_valid
from scripts import training_label_audit as audit
from scripts.ops import training_dataset_preflight as preflight


def observations(prices=(100.0, 110.0, 90.0, 105.0, 102.0)):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [{
        "timestamp_utc": (start + timedelta(minutes=i)).isoformat(),
        "mode": "shadow_crypto", "symbol": "BTC-USD", "snapshot_id": f"s-{i}",
        "price": price, "features": {"last_price": price, "marker": float(i)},
    } for i, price in enumerate(prices)]


def dataset(rows, **kwargs):
    return runtime.make_runtime_windowed_dataset(
        sequences={("shadow_crypto", "BTC-USD"): rows},
        feature_builder=kwargs.pop("feature_builder", lambda seq, i: [seq[i]["features"]["marker"]]),
        label_builder=kwargs.pop("label_builder", lambda seq, i, h: float(i % 2)),
        window=kwargs.pop("window", 1), horizon=kwargs.pop("horizon", 1),
        include_sample_evidence=True, max_rejection_evidence=2, **kwargs,
    )


def test_observed_path_labels_are_gross_not_trade_profit():
    rows = observations()
    result = runtime.runtime_label_evidence(rows, 0, 3)
    labels = result["outcome_labels"]
    assert labels["forward_return_gross"] == pytest.approx(0.05)
    assert labels["max_observed_return_gross"] == pytest.approx(0.10)
    assert labels["min_observed_return_gross"] == pytest.approx(-0.10)
    assert labels["observed_path_max_drawdown"] == pytest.approx(1 - 90 / 110)
    assert labels["outcome_path_max_gap_seconds"] == 60
    assert labels["cost_adjustment_status"] == "not_applied"
    assert not labels["is_trade_pnl"] and not labels["profitability_evidence"]


def test_lineage_binds_window_and_actual_price_path():
    rows = observations()
    first = runtime.runtime_label_evidence(rows, 2, 2, feature_window=2)
    wider = runtime.runtime_label_evidence(rows, 2, 2, feature_window=3)
    changed = deepcopy(rows)
    changed[3]["price"] = 106.0
    changed[3]["features"]["last_price"] = 106.0
    revised = runtime.runtime_label_evidence(changed, 2, 2, feature_window=2)
    assert len({x["lineage_sha256"] for x in (first, wider, revised)}) == 3


@pytest.mark.parametrize("field,value,reason", [
    ("symbol", "ETH-USD", "cross_symbol_evidence_window"),
    ("mode", "paper", "cross_mode_evidence_window"),
    ("snapshot_id", "s-0", "duplicate_window_snapshot_id"),
    ("timestamp_utc", "2025-01-01T00:00:00Z", "nonchronological_evidence_window"),
])
def test_interior_evidence_must_match_even_when_endpoints_are_valid(field, value, reason):
    rows = observations()
    rows[1][field] = value
    result = runtime.runtime_label_evidence(rows, 0, 3)
    assert not result["eligible"]
    assert reason in result["reasons"]
    assert result["outcome_labels"]["forward_return_gross"] is None


def test_bad_feature_window_is_not_admitted():
    rows = observations()
    rows[0]["timestamp_utc"] = rows[2]["timestamp_utc"]
    _, _, meta = dataset(rows, window=3)
    assert meta["sample_count"] == 1
    assert meta["label_disposition_audit"]["rejected_candidate_count"] == 1


def test_rejection_examples_are_bounded_without_losing_total_counts():
    _, y, meta = dataset(observations(), sample_filter=lambda *args: False)
    result = meta["label_disposition_audit"]
    assert len(y) == 0
    assert result["candidate_count"] == result["rejected_candidate_count"] == 4
    assert result["rejection_counts"] == {"strategy_sample_filter": 4}
    assert len(result["rejection_examples"]) == 2
    assert result["rejection_examples_truncated"]
    assert all(row["label_value"] is None and not row["training_eligible"] for row in result["rejection_examples"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_are_rejected_not_turned_into_zeros(value):
    _, y, meta = dataset(observations(), feature_builder=lambda *args: [value])
    assert not len(y)
    assert meta["label_disposition_audit"]["rejection_counts"] == {"invalid_or_inconsistent_feature_vector": 4}


def test_contract_cannot_be_bypassed_by_label_repair_arguments():
    _, y, meta = dataset(
        observations(), sample_filter=lambda *args: False, bypass_sample_filter=True,
        fallback_direction_label=True, label_builder=lambda *args: None,
        label_contract={"objective_class": "market_outcome", "sample_filter_bypass_allowed": False, "directional_fallback_allowed": False},
    )
    assert not len(y)
    assert not meta["label_repair_enabled"] and not meta["label_repair_bypassed_filter"]


def test_auxiliary_outcomes_never_enter_predictor_features():
    X, y, meta = dataset(observations(), window=2)
    for features, label, evidence in zip(X, y, meta["sample_evidence"]):
        i = int(evidence["feature_snapshot_id"].split("-")[1])
        np.testing.assert_array_equal(features, [i - 1, i])
        assert evidence["label_value"] == label[0]
        assert evidence["feature_values_sha256"] == hashlib.sha256(features.astype("<f4").tobytes()).hexdigest()
        assert evidence["sample_regime"] == "unknown"
        assert evidence["sample_session"] == "unknown"
    disposition = meta["label_disposition_audit"]
    assert disposition["candidate_count"] == disposition["rejected_candidate_count"] + disposition["accepted_before_selection_count"]


def test_pre_split_mode_does_not_rebalance_using_holdout_labels(monkeypatch):
    monkeypatch.setattr(runtime, "_rebalance_binary_runtime_dataset", lambda *args: pytest.fail("label balancing before split"))
    monkeypatch.setattr(runtime, "_apply_symbol_and_regime_balance", lambda *args: pytest.fail("context balancing before split"))
    X, y, meta = dataset(observations(), balance_samples=False)
    assert len(X) == len(y) == 4
    assert not meta["label_balance_applied"]


def test_all_bot_inventory_does_not_call_registry_contracts_measured_labels(tmp_path):
    rows = []
    for i, objective in enumerate(("market_outcome", "operational_effect", "research_validation")):
        bot_id = f"bot_{i}"
        contract = {"bot_id": bot_id, "objective_class": objective, "required_outputs": ["actual_outcome"]}
        contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows.append({"bot_id": bot_id, "active": True, "training_label_materialization_contract": contract})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"sub_bots": rows}))
    result = audit.build_label_audit_payload(registry_path=registry, diagnostics_dir=tmp_path / "diagnostics")
    assert result["label_evidence_coverage"]["audited_bot_count"] == 3
    assert not result["label_evidence_coverage"]["inventory_truncated"]
    for row in result["all_bot_label_inventory"]:
        assert row["contract_integrity_valid"]
        assert row["status"] == "outcome_evidence_pending"
        assert "missing_measured_label_evidence" in row["blockers"]
        assert not row["training_admission_authority"]
    for row in result["all_bot_label_inventory"][1:]:
        assert "authority_specific_outcome_materializer_required" in row["blockers"]


def test_purged_rows_cannot_satisfy_data_readiness(monkeypatch):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence = [{
        "mode": "shadow_crypto", "symbol": "BTC-USD", "label_value": float(i % 2),
        "feature_window_started_at_utc": (start - timedelta(days=1)).isoformat(),
        "feature_timestamp_utc": (start + timedelta(hours=i)).isoformat(),
        "label_matured_at_utc": (start + timedelta(days=2)).isoformat(),
    } for i in range(20)]
    monkeypatch.setattr(preflight, "make_runtime_windowed_dataset", lambda **kwargs: (
        np.ones((20, 2)), np.asarray([[i % 2] for i in range(20)]),
        {"sample_evidence": evidence, "contributing_sequences": 1, "positive_rate": 0.5},
    ))
    spec = SimpleNamespace(bot_id="test", mode_allowlist=[], min_confidence=0, window=1, horizon=1, min_samples=1, min_sequences=1, min_positive_samples=1, min_negative_samples=1, lookback_days=14)
    contract = {"objective_class": "market_outcome", "bot_id": "test"}
    contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _, _, _, result = preflight.prepare_dataset(spec, {}, contract, {"minimum_total_samples": 1, "minimum_eligible_sequences": 1})
    assert result["eligible_after_purge_sample_count"] == 0
    assert result["eligible_after_purge_sequence_count"] == 0
    assert not result["checks"]["sample_floor"]
    assert not result["checks"]["sequence_floor"]
    assert not result["checks"]["label_balance"]
    assert not result["data_checks_passed"]


def test_materialization_requires_identity_and_untampered_contract():
    contract = {"bot_id": "a", "objective_class": "market_outcome", "minimum_label_maturity_seconds": 60}
    contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert materialization_contract_valid(contract, "a")
    assert not materialization_contract_valid(contract, "b")
    assert not materialization_contract_valid({**contract, "minimum_label_maturity_seconds": 0}, "a")
    with pytest.raises(ValueError, match="materialization_contract"):
        preflight.prepare_dataset(SimpleNamespace(bot_id="b"), {}, contract, {})


def test_extreme_price_ratios_are_not_serialized_as_infinity():
    rows = observations((1e-300, 1e300))
    evidence = runtime.runtime_label_evidence(rows, 0, 1)
    assert not evidence["eligible"]
    assert "nonfinite_market_path_return" in evidence["reasons"]
    json.dumps(evidence, allow_nan=False)


def test_snapshot_streaming_limits_fail_closed_without_partial_labels(tmp_path):
    rows = observations()
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    for i, row in enumerate(rows):
        row["timestamp_utc"] = (start + timedelta(minutes=i)).isoformat()
    data = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path = tmp_path / "rows.jsonl"
    path.write_bytes(data)
    summary_path = tmp_path / "snapshot.json"
    summary = {"schema_version": 2, "lookback_days": 14, "rows_path": str(path), "row_count": len(rows), "rows_sha256": hashlib.sha256(data).hexdigest()}
    summary_path.write_text(json.dumps(summary))
    kwargs = {"lookback_days": 14, "mode_allowlist": ["shadow_crypto"], "symbol_allowlist": None, "snapshot_file": summary_path, "strict_snapshot": True}
    assert sum(map(len, runtime._load_runtime_snapshot_rows(tmp_path, **kwargs).values())) == len(rows)
    for limit in ({"max_source_bytes": 1}, {"max_retained_bytes": 1}, {"max_retained_rows": 1}, {"max_line_bytes": 10}, {"deadline_monotonic": time.monotonic() - 1}):
        receipt = {}
        assert runtime._load_runtime_snapshot_rows(tmp_path, **kwargs, **limit, read_audit=receipt) == {}
        assert receipt["status"] == "blocked" and not receipt["digest_verified"]
        assert "exceeded" in receipt["reason"]
    summary["row_count"] += 1
    summary_path.write_text(json.dumps(summary))
    assert runtime._load_runtime_snapshot_rows(tmp_path, **kwargs) == {}


def test_streaming_snapshot_rejects_malformed_rows_and_bad_digests(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"not-json\n")
    summary_path = tmp_path / "snapshot.json"
    summary = {"schema_version": 2, "lookback_days": 14, "rows_path": str(path), "row_count": 1, "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary_path.write_text(json.dumps(summary))
    assert runtime._load_runtime_snapshot_rows(tmp_path, lookback_days=14, mode_allowlist=None, symbol_allowlist=None, snapshot_file=summary_path, strict_snapshot=True) == {}


def test_dataset_exports_explicit_purged_mask_and_usable_count(tmp_path):
    evidence = [{"train_validation_test_split": split} for split in ("train", "purged", "test")]
    result = preflight._materialize(tmp_path, "example", np.ones((3, 2)), np.zeros((3, 1)), evidence, {})
    assert result["sample_count"] == 3 and result["eligible_sample_count"] == 2
    assert result["purged_sample_count"] == 1
    with np.load(result["path"], allow_pickle=False) as data:
        assert data["split"].tolist() == ["train", "purged", "test"]
        assert data["eligible_mask"].tolist() == [True, False, True]
