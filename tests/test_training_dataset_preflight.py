from datetime import datetime, timedelta, timezone
import ast
import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from core.training_diagnostic_contract import diagnostic_age_hours
from scripts.ops import training_dataset_preflight as preflight
from core import runtime_training_common as runtime


@pytest.mark.parametrize("timestamp", [None, "bad", "2099-01-01T00:00:00Z"])
def test_unknown_or_future_diagnostics_are_not_fresh(timestamp):
    assert diagnostic_age_hours({"timestamp_utc": timestamp}, datetime.now(timezone.utc)) is None


def test_preflight_rejects_missing_snapshot_without_loading_sequences(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_load_runtime_snapshot_rows", lambda *a, **k: pytest.fail("unexpected dataset read"))
    result = preflight.build_payload(tmp_path, bot_ids=[])
    assert result["blockers"] == ["snapshot_missing_stale_or_unverified"]
    assert not any(result["authority_contract"].values())


def test_preflight_batch_limit_is_checked_before_snapshot_read(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_load_runtime_snapshot_rows", lambda *a, **k: pytest.fail("unexpected read"))
    result = preflight.build_payload(tmp_path, bot_ids=[f"bot_{i}" for i in range(11)])
    assert result["blockers"] == ["preflight_bot_budget_exceeded"]


@pytest.mark.parametrize("bot_id", preflight.EXPLICIT_RUNTIME_MODULES)
def test_research_adapter_reuses_production_options_without_launching_training(bot_id, monkeypatch):
    monkeypatch.setenv("BOT_MLX_DISABLE", "1")
    module = importlib.import_module("core." + bot_id)
    options = module.runtime_training_options()
    spec = preflight.preparation_spec(bot_id)
    assert spec.feature_builder is options["runtime_feature_builder"]
    assert spec.label_builder is options["runtime_label_builder"]
    assert spec.sample_filter is options["sample_filter"]
    assert spec.confidence_builder is options["confidence_builder"]
    for key in ("feature_names", "mode_allowlist", "min_confidence", "window", "horizon",
                "min_samples", "min_sequences", "lookback_days"):
        assert getattr(spec, key) == options[key]
    assert spec.symbol_allowlist == options.get("symbol_allowlist")
    assert spec.sample_stride == options.get("sample_stride", 1)
    assert bot_id not in preflight.SUPPORTED_MODULES
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return "captured_without_fit"

    trainer = importlib.import_module("indicator_bot_common")
    monkeypatch.setattr(trainer, "train_runtime_indicator_bot", capture)
    monkeypatch.setattr(module, "train_runtime_indicator_bot", capture, raising=False)
    assert module.train_brain() == "captured_without_fit"
    assert calls == [options]


@pytest.mark.parametrize("bot_id", preflight.EXPLICIT_RUNTIME_MODULES)
def test_runtime_preparation_preserves_filters_stride_and_outcome_owner(bot_id, monkeypatch):
    monkeypatch.setenv("BOT_MLX_DISABLE", "1")
    spec = preflight.preparation_spec(bot_id)
    mode = spec.mode_allowlist[0]
    symbol = spec.symbol_allowlist[0] if spec.symbol_allowlist else "BTC-USD"
    sequences = {(mode, symbol): [{"marker": "included"}], ("unrelated_mode", symbol): [{}]}
    if spec.symbol_allowlist:
        sequences[(mode, "NOT_AN_ALLOWED_SYMBOL")] = [{}]
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return np.empty((0, 1)), np.empty((0, 1)), {"sample_evidence": []}

    monkeypatch.setattr(preflight, "make_runtime_windowed_dataset", build)
    contract = {"bot_id": bot_id, "objective_class": "market_outcome"}
    contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _, _, _, result = preflight.prepare_dataset(spec, sequences, contract, {})
    assert calls[0]["sequences"] == {(mode, symbol): [{"marker": "included"}]}
    assert calls[0]["sample_stride"] == spec.sample_stride
    assert calls[0]["feature_builder"] is spec.feature_builder
    assert calls[0]["label_builder"] is spec.label_builder
    assert calls[0]["label_owner_id"] == bot_id
    assert calls[0]["label_contract"] == contract
    assert calls[0]["balance_samples"] is False
    assert not result["data_checks_passed"]
    assert not any(result["authority_contract"].values())


def test_explicit_projection_includes_strategy_and_shared_gate_inputs(monkeypatch):
    monkeypatch.setenv("BOT_MLX_DISABLE", "1")
    for bot_id in (*preflight.EXPLICIT_RUNTIME_MODULES, preflight.SUPPORTED_MODULES[0]):
        fields = set(preflight.snapshot_feature_fields([bot_id]))
        for module in (bot_id, "crypto_runtime_bot_common", "runtime_requested_bot_common"):
            tree = ast.parse((preflight.PROJECT_ROOT / "core" / f"{module}.py").read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"observation_feature", "feature_std", "feature_ema"}:
                    assert {arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)} <= fields


def test_unknown_research_module_never_imports(monkeypatch):
    monkeypatch.setattr(preflight.importlib, "import_module", lambda *a: pytest.fail("unexpected import"))
    with pytest.raises(ValueError, match="unsupported_research_bot"):
        preflight.preparation_spec("untrusted_training_module")


def test_crypto_data_preparation_does_not_import_the_model_trainer():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import scripts.ops.training_dataset_preflight; "
        "assert 'indicator_bot_common' not in sys.modules; "
        "assert getattr(sys.modules.get('mlx.core'), '__file__', None) is None",
    ], cwd=preflight.PROJECT_ROOT, env={**os.environ, "BOT_MLX_DISABLE": "1"}, check=True, timeout=20)


def _sequences():
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    result = {}
    for symbol, count in (("AAA", 260), ("BBB", 80), ("CCC", 60)):
        result[("shadow_crypto", symbol)] = [
            {
                "timestamp_utc": (start + timedelta(hours=i)).isoformat(),
                "ts_epoch": (start + timedelta(hours=i)).timestamp(),
                "mode": "shadow_crypto", "symbol": symbol, "snapshot_id": f"{symbol}-{i}",
                "price": 100 + i,
                "features": {"last_price": 100 + i, "marker": i},
            }
            for i in range(count)
        ]
    return result


def test_sample_lineage_stays_aligned_through_balance_and_cap(monkeypatch):
    monkeypatch.setenv("RUNTIME_TRAIN_SYMBOL_MAX_SHARE", "0.35")
    X, y, meta = runtime.make_runtime_windowed_dataset(
        sequences=_sequences(), feature_builder=lambda rows, i: [rows[i]["features"]["marker"]],
        label_builder=lambda rows, i, h: float(i % 9 == 0), window=2, horizon=2,
        max_samples=37, include_sample_evidence=True,
    )
    evidence = meta["sample_evidence"]
    assert len(evidence) == len(X) == len(y) == 37
    assert meta["symbol_cap_applied"]
    assert meta["contributing_sequences"] == 3
    for features, label, row in zip(X, y, evidence):
        assert int(row["feature_snapshot_id"].split("-")[-1]) == int(features[-1])
        assert row["label_value"] == label[0]
    assert len({row["lineage_sha256"] for row in evidence}) == len(evidence)


def test_time_splits_purge_overlapping_feature_and_outcome_windows():
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [{
        "feature_window_started_at_utc": (start + timedelta(hours=i - 4)).isoformat(),
        "feature_timestamp_utc": (start + timedelta(hours=i)).isoformat(),
        "label_matured_at_utc": (start + timedelta(hours=i + 8)).isoformat(),
    } for i in range(100)]
    split = preflight.purged_split_evidence(rows, embargo_minutes=60)
    groups = {name: [row for row in split if row["train_validation_test_split"] == name] for name in ("train", "validation", "test")}
    for first, second in (("train", "validation"), ("validation", "test")):
        assert groups[first] and groups[second]
        latest = max(preflight._epoch(row["label_matured_at_utc"]) for row in groups[first])
        earliest = min(preflight._epoch(row["feature_window_started_at_utc"]) for row in groups[second])
        assert latest + 3600 < earliest
    assert any(row["train_validation_test_split"] == "purged" for row in split)


def test_materialized_data_roundtrips_with_receipt_without_pickle(tmp_path):
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.asarray([[0.0], [1.0]], dtype=np.float32)
    evidence = [{"lineage_sha256": "a"}, {"lineage_sha256": "b"}]
    result = preflight._materialize(tmp_path, "test", X, y, evidence, preflight.AUTHORITY)
    path = Path(result["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["sha256"]
    with np.load(path, allow_pickle=False) as dataset:
        np.testing.assert_array_equal(dataset["features"], X)
        np.testing.assert_array_equal(dataset["labels"], y)
        assert json.loads(str(dataset["evidence_json"])) == evidence
        assert not any(json.loads(str(dataset["receipt_json"])).values())


def test_preflight_projection_covers_strategy_gate_and_label_dependencies():
    fields = set(preflight.snapshot_feature_fields())
    for module in (*preflight.SUPPORTED_MODULES, "crypto_runtime_bot_common", "runtime_requested_bot_common"):
        tree = ast.parse((preflight.PROJECT_ROOT / "core" / f"{module}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"observation_feature", "feature_std", "feature_ema"}:
                dependencies = {arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
                assert dependencies <= fields, (module, dependencies - fields)
    for module in preflight.SUPPORTED_MODULES:
        assert set(importlib.import_module("core." + module).SPEC.feature_fields) <= fields


def test_projection_fits_budget_and_keeps_full_snapshot_digest_and_identity(tmp_path):
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [{
        "mode": "shadow_crypto", "symbol": "BTC-USD", "price": 100 + i,
        "timestamp_utc": (start + timedelta(minutes=i)).isoformat(),
        "snapshot_id": f"snapshot-{i}", "provider": "coinbase", "instrument_type": "spot",
        "features": {"mom_5m": 0.1, "unused": "x" * 10000},
    } for i in range(3)]
    data = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path = tmp_path / "rows.jsonl"
    path.write_bytes(data)
    summary = tmp_path / "snapshot.json"
    summary.write_text(json.dumps({"schema_version": 2, "lookback_days": 14, "rows_path": str(path), "row_count": 3, "rows_sha256": hashlib.sha256(data).hexdigest()}))
    kwargs = dict(lookback_days=14, mode_allowlist=None, symbol_allowlist=None, snapshot_file=summary, strict_snapshot=True, max_retained_bytes=2048)
    assert runtime._load_runtime_snapshot_rows(tmp_path, **kwargs) == {}
    audit = {}
    result = runtime._load_runtime_snapshot_rows(tmp_path, **kwargs, feature_allowlist=["mom_5m"], read_audit=audit)
    projected = result[("shadow_crypto", "BTC-USD")]
    assert audit["digest_verified"] and audit["retained_encoded_row_bytes"] < 2048
    assert audit["source_bytes_scanned"] == len(data)
    for before, after in zip(rows, projected):
        assert after == {**before, "features": {"mom_5m": 0.1}}
    # Tampering with an omitted feature still invalidates the source receipt.
    path.write_bytes(data.replace(b"xxxxxxxx", b"yyyyyyyy"))
    assert runtime._load_runtime_snapshot_rows(tmp_path, **kwargs, feature_allowlist=["mom_5m"]) == {}
