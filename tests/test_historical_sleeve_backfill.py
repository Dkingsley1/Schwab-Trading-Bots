import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import pytest

from core import historical_sleeve_labels as labels
from scripts.ops import historical_sleeve_backfill as full

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest():
    contracts, aliases = labels.label_contracts(ROOT)
    return {"contracts": contracts, "aliases": aliases}


@pytest.fixture
def admitted(monkeypatch):
    monkeypatch.setattr(full, "_check", lambda *args: None)


def row(second=0, price=100, **changes):
    ts = f"2026-09-01T14:{second // 60:02d}:{second % 60:02d}+00:00"
    return {
        "timestamp_utc": ts,
        "sleeve_id": "day_trading",
        "symbol": "SPY",
        "source_provider": "schwab",
        "instrument_type": "equity",
        "snapshot_id": f"snap-{second}",
        "production_candidate_id": "old-candidate",
        "market": {"last_price": price, "snapshot_ts_utc": ts},
        **changes,
    }


def source(path, kind="jsonl"):
    fp = full.common.fingerprint(path)
    return {
        "path": str(path),
        "kind": kind,
        "fingerprint": fp,
        "source_id": labels.digest([str(path), fp]),
        "status": "pending",
    }


def scan(tmp_path, manifest, path, kind="jsonl"):
    return full._scan_source(
        source(path, kind), manifest, tmp_path, tmp_path, time.monotonic() + 20, 0
    )


def test_all_111_research_horizons_are_defined_and_execution_unchanged(manifest):
    cs = manifest["contracts"]
    assert len(cs) == 111
    assert all(not c["unresolved_horizon"] for c in cs.values())
    assert all(c["execution_holding_horizon_unchanged"] for c in cs.values())
    assert cs["crypto_spot"]["research_horizon"]["primary_seconds"] == 4 * 3600
    assert cs["swing_aggressive"]["research_horizon"]["primary_seconds"] == 5 * 86400
    assert cs["dividend_income"]["research_horizon"]["primary_seconds"] == 90 * 86400
    assert "synchronized" in cs["pairs_correlation"]["research_horizon"]["endpoint"]
    assert "completed_oos" in cs["quant_pricing_models"]["research_horizon"]["endpoint"]
    assert cs["quant_pricing_models"]["holding_horizon"] == "research_defined"
    for c in cs.values():
        h = c["research_horizon"]
        assert h["primary_seconds"] > 0 and h["secondary_seconds"]
        assert h["horizon_sha256"] == labels.digest(
            {k: v for k, v in h.items() if k != "horizon_sha256"}
        )
        assert not any(c["authority"].values())


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, 367 * 86400])
def test_bad_horizon_rejected(bad):
    policy = json.loads(
        (ROOT / "config/historical_sleeve_research_horizons_v1.json").read_text()
    )
    policy["recipes"]["crypto_spot"]["primary_seconds"] = bad
    with pytest.raises(ValueError, match="duration"):
        labels.research_horizon(policy, "crypto_spot")


@pytest.mark.parametrize("compressed", [False, True])
def test_scan_preserves_source_and_hashes_every_annotation(
    tmp_path, manifest, admitted, compressed
):
    path = tmp_path / ("input.jsonl.gz" if compressed else "input.jsonl")
    raw = b"".join((json.dumps(r) + "\n").encode() for r in (row(), row(300, 103)))
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    before = path.read_bytes()
    receipt = scan(tmp_path, manifest, path, "gzip" if compressed else "jsonl")
    outputs = list(full._labels(receipt))
    assert receipt["rows"] == 2
    assert receipt["sleeve_counts"] == {"day_trading": 2}
    assert (
        outputs[0]["source_row_sha256"]
        == hashlib.sha256(raw.splitlines(keepends=True)[0]).hexdigest()
    )
    assert path.read_bytes() == before
    assert all(not r["authority"]["training_eligible"] for r in outputs)
    assert all(r["primary_label"]["value"] is None for r in outputs)


def test_sqlite_reads_committed_wal_and_both_payload_tables(
    tmp_path, manifest, admitted
):
    path = tmp_path / "live.sqlite3"
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        for name in ("jsonl_records", "json_file_records"):
            writer.execute(
                f"CREATE TABLE {name}(id INTEGER PRIMARY KEY,source_rel TEXT,payload_json TEXT)"
            )
            writer.execute(
                f"INSERT INTO {name} VALUES(1,?,?)",
                ("history.jsonl", json.dumps(row())),
            )
        writer.commit()
        assert Path(str(path) + "-wal").stat().st_size > 0
        receipt = scan(tmp_path, manifest, path, "sqlite")
        results = list(full._labels(receipt))
        assert {r["source_ordinal"] for r in results} == {
            "jsonl_records:1",
            "json_file_records:1",
        }
        assert (
            receipt["sqlite_read_policy"] == "per_source_read_transaction_including_wal"
        )
        assert writer.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0] == 1
        writer.execute(
            "INSERT INTO jsonl_records VALUES(2,?,?)",
            ("history.jsonl", json.dumps(row(300))),
        )


def test_changed_source_and_oversize_are_not_silently_accepted(
    tmp_path, manifest, admitted, monkeypatch
):
    path = tmp_path / "input.jsonl"
    path.write_text("x" * 300 + "\n{}\nnot-json\n")
    entry = source(path)
    monkeypatch.setattr(full.common, "MAX_LINE", 100)
    receipt = scan(tmp_path, manifest, path)
    outputs = list(full._labels(receipt))
    assert len(outputs) == 3
    assert outputs[0]["reasons"] == ["oversized_source_record"]
    assert outputs[2]["reasons"] == ["malformed_or_nonobject_record"]
    path.write_text("changed\n")
    with pytest.raises(full.Paused, match="source_changed"):
        list(full._raw_records(entry, time.monotonic() + 5))


def test_output_limit_leaves_no_completion_receipt(
    tmp_path, manifest, admitted, monkeypatch
):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    monkeypatch.setattr(full, "MAX_OUTPUT", 1)
    with pytest.raises(full.Paused, match="output_budget"):
        scan(tmp_path, manifest, path)
    assert not list(tmp_path.glob("*.receipt.json"))
    assert not list(tmp_path.glob("*.part"))


def test_corrupt_partition_cannot_be_resumed(tmp_path, manifest, admitted):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    receipt = scan(tmp_path, manifest, path)
    with Path(receipt["labels_file"]).open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(full.Paused, match="integrity"):
        list(full._labels(receipt))


def test_context_index_deduplicates_and_remains_nontraining(
    tmp_path, manifest, admitted
):
    receipts = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in (row(), row(300, 103))))
        receipts.append(scan(tmp_path, manifest, path))
    result = full._context_stage(
        receipts, manifest, tmp_path, tmp_path, time.monotonic() + 10
    )
    assert result["unique_market_marks"] == 2
    assert result["counts"]["observed_gross_price_context"] == 1
    with gzip.open(result["file"], "rt") as handle:
        outputs = [json.loads(r) for r in handle]
    assert all(not r["training_eligible"] for r in outputs)
    repeat = full._context_stage(
        receipts, manifest, tmp_path, tmp_path, time.monotonic() + 10
    )
    assert repeat == result


def test_conflicting_prices_are_quarantined(tmp_path, manifest, admitted):
    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in (row(), row(300, 103), row(300, 110)))
    )
    result = full._context_stage(
        [scan(tmp_path, manifest, path)],
        manifest,
        tmp_path,
        tmp_path,
        time.monotonic() + 10,
    )
    assert result["counts"]["quarantined"] == 1
    assert not result["counts"].get("observed_gross_price_context")


def test_parquet_export_reads_payload_not_metadata(tmp_path, manifest, admitted):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "export.parquet"
    pq.write_table(
        pa.table(
            {"source_rel": ["history.jsonl"], "payload_json": [json.dumps(row())]}
        ),
        path,
    )
    receipt = scan(tmp_path, manifest, path, "parquet")
    assert receipt["rows"] == 1
    assert list(full._labels(receipt))[0]["sleeve_id"] == "day_trading"


def test_publish_horizons_contains_every_sleeve(tmp_path, manifest):
    full.publish_horizons(tmp_path, manifest["contracts"])
    path = tmp_path / full.common.STATE_REL / "research_horizons.json"
    assert json.loads(path.read_text())["sleeve_count"] == 111
    assert (
        len(
            [
                line
                for line in path.with_suffix(".md").read_text().splitlines()
                if line.startswith("| ")
            ]
        )
        == 113
    )


def test_full_resume_and_denial_preserve_actual_evidence(
    tmp_path, manifest, admitted, monkeypatch
):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    monkeypatch.setattr(
        full,
        "label_contracts",
        lambda root: (manifest["contracts"], manifest["aliases"]),
    )
    monkeypatch.setattr(full.common, "source_roots", lambda root: [path])
    first = full.run_full(tmp_path)
    assert first["historical_source_scan_complete"]
    assert first["source_records_processed"] == 1
    assert not first["historical_backfill_complete"]
    monkeypatch.setattr(
        full, "_scan_source", lambda *args: pytest.fail("completed source rescanned")
    )
    second = full.run_full(tmp_path)
    assert second["source_records_processed"] == 1

    def blocked(*args):
        raise full.Paused("test_admission_withdrawn")

    monkeypatch.setattr(full, "_check", blocked)
    third = full.run_full(tmp_path)
    assert third["previous_evidence_not_recomputed"]
    assert third["source_records_processed"] == 1
    assert third["timestamp_utc"] == second["timestamp_utc"]
    assert third["blockers"] == ["test_admission_withdrawn"]


def test_protected_source_rejected_before_open():
    with pytest.raises(ValueError, match="protected"):
        list(
            full._raw_records(
                {"path": "/Volumes/VIDEO/not-read.jsonl", "kind": "jsonl"},
                time.monotonic() + 5,
            )
        )
