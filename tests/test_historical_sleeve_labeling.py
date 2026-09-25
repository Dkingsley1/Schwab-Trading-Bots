import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import pytest

from core import historical_sleeve_labels as labels
from scripts.ops import historical_sleeve_labeling as backfill

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_worker_memory(monkeypatch):
    # Unit tests model the bounded worker, not the entire pytest process history.
    monkeypatch.setattr(backfill, "_rss", lambda: 0)


@pytest.fixture
def contracts():
    return labels.label_contracts(ROOT)


def row(second=0, price=100, sleeve="day_trading", **changes):
    ts = f"2026-09-01T14:{second // 60:02d}:{second % 60:02d}+00:00"
    return {
        "timestamp_utc": ts,
        "sleeve_id": sleeve,
        "symbol": "SPY",
        "source_provider": "schwab",
        "instrument_type": "equity",
        "snapshot_id": f"snap-{second}",
        "production_candidate_id": "historic-candidate",
        "market": {"last_price": price, "snapshot_ts_utc": ts},
        **changes,
    }


def manifest(contracts):
    return {"contracts": contracts[0], "aliases": contracts[1]}


def source(path):
    kind = (
        "sqlite"
        if path.suffix == ".sqlite3"
        else "gzip" if path.suffix == ".gz" else "jsonl"
    )
    fp = backfill.fingerprint(path)
    return {
        "path": str(path),
        "kind": kind,
        "fingerprint": fp,
        "source_id": labels.digest([str(path), fp]),
        "status": "pending",
    }


def register(db, entry):
    db.execute(
        "INSERT INTO sources(source_id,status) VALUES(?,?)",
        (entry["source_id"], "pending"),
    )
    db.commit()


@pytest.mark.parametrize("rss", [backfill.MAX_RSS, backfill.MAX_RSS + 1])
def test_memory_limit_preserves_cursor_and_blocks_labels(
    tmp_path, contracts, monkeypatch, rss
):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (row(), row(300, 103))))
    original = path.read_bytes()
    entry = source(path)
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        monkeypatch.setattr(backfill, "_rss", lambda: rss)
        result = backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        assert result == {"bytes": 0, "rows": 0}
        state = db.execute("SELECT * FROM sources").fetchone()
        assert state["cursor"] == 0
        assert state["status"] == "pending"
        assert db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 0
        monkeypatch.setattr(backfill, "_rss", lambda: 0)
        assert (
            backfill.ingest_source(
                db, entry, manifest(contracts), time.monotonic() + 10, 100000
            )["rows"]
            == 2
        )
        monkeypatch.setattr(backfill, "_rss", lambda: rss)
        assert (
            backfill.label_contexts(db, manifest(contracts), time.monotonic() + 10) == 0
        )
        assert db.execute("SELECT COUNT(*) FROM contexts").fetchone()[0] == 0
    assert path.read_bytes() == original


def test_all_sleeves_have_objective_specific_nonexecuting_contracts(contracts):
    cs, _ = contracts
    assert len(cs) == 111
    assert sum(c["objective_class"] == "control_only" for c in cs.values()) == 25
    assert (
        cs["day_trading"]["required_outcome_evidence"]
        != cs["dividend_income"]["required_outcome_evidence"]
    )
    assert (
        cs["volatility"]["required_outcome_evidence"]
        != cs["short_bias_hedge"]["required_outcome_evidence"]
    )
    for contract in cs.values():
        assert not any(contract["authority"].values())
        assert (
            labels.digest({k: v for k, v in contract.items() if k != "contract_sha256"})
            == contract["contract_sha256"]
        )
        if contract["objective_class"] == "control_only":
            assert not contract["supplemental_price_horizons_seconds"]


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("crypto_futures", "crypto_futures"),
        ("default_crypto_coinbase", "crypto_spot"),
        ("swing_aggressive_equities_schwab", "swing_aggressive"),
        ("shadow_dividend_equities", "dividend_income"),
    ],
)
def test_profile_routing(contracts, profile, expected):
    cs, aliases = contracts
    assert labels._profile(profile, aliases, cs, crypto="crypto" in profile) == expected


def test_conflicting_sleeves_are_not_silently_remapped(contracts):
    result = labels.annotate(
        row(), *contracts, "decisions/shadow_dividend_equities/example.jsonl"
    )
    assert result["sleeve_id"] == ""
    assert result["reasons"] == ["conflicting_sleeve_identity"]


def test_no_forecasts_cash_yield_or_producer_pnl_become_primary_labels(contracts):
    result = labels.annotate(
        row(
            sleeve="dividend_income",
            dividend_amount=5,
            post_cost_pnl_delta=20,
            account_id="never-copy",
        ),
        *contracts,
    )
    assert result["primary_label"]["value"] is None
    assert (
        result["reported_outcomes"]["verification"] == "producer_claim_not_reverified"
    )
    assert "account_id" not in result
    assert not any(result["authority"].values())


def test_elapsed_time_context_is_separate_from_payoff(contracts):
    cs, aliases = contracts
    first, last = (labels.annotate(r, cs, aliases) for r in (row(), row(300, 103)))
    result = labels.price_context(first, last, 300, cs["day_trading"])
    assert result["value"] == pytest.approx(0.03)
    assert result["status"] == "observed_gross_price_context"
    assert not result["training_eligible"]
    assert first["primary_label"]["value"] is None


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            {"provider": "coinbase"},
            "missing_or_mismatched_instrument_or_candidate_lineage",
        ),
        (
            {"source_candidate_id": "next-generation"},
            "missing_or_mismatched_instrument_or_candidate_lineage",
        ),
        ({"quote_timestamp_utc": ""}, "missing_or_stale_quote_timestamp"),
        ({"snapshot_id": "snap-0"}, "reused_snapshot"),
        ({"epoch": None}, "horizon_not_observed_within_tolerance"),
        ({"price": float("inf")}, "invalid_price_return"),
    ],
)
def test_bad_endpoints_remain_null(contracts, change, reason):
    cs, aliases = contracts
    first, last = (labels.annotate(r, cs, aliases) for r in (row(), row(300, 103)))
    result = labels.price_context(first, {**last, **change}, 300, cs["day_trading"])
    assert result["value"] is None
    assert result["reason"] == reason


@pytest.mark.parametrize("compressed", [False, True])
def test_stream_resume_preserves_sources_and_receipts(tmp_path, contracts, compressed):
    path = tmp_path / ("history.jsonl.gz" if compressed else "history.jsonl")
    payload = "".join(json.dumps(r) + "\n" for r in (row(), row(300, 103)))
    path.write_bytes(
        gzip.compress(payload.encode()) if compressed else payload.encode()
    )
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = source(path)
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        one = backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 1
        )
        assert one["rows"] == 1
        assert db.execute("SELECT rows FROM sources").fetchone()[0] == 1
        backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        assert db.execute("SELECT status FROM sources").fetchone()[0] == "complete"
        assert db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
        backfill.label_contexts(db, manifest(contracts), time.monotonic() + 10)
        assert (
            db.execute(
                "SELECT COUNT(*) FROM contexts WHERE status='observed_gross_price_context'"
            ).fetchone()[0]
            == 1
        )
        assert (
            not db.execute("SELECT payload FROM observations")
            .fetchone()[0]
            .find("historic-candidate")
            == -1
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


def test_changed_source_is_not_silently_resumed(tmp_path, contracts):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    entry = source(path)
    path.write_text(json.dumps(row(300)) + "\n")
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        assert db.execute("SELECT status FROM sources").fetchone()[0] == "changed"
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_invalid_rows_are_accounted_and_oversize_is_not_split(
    tmp_path, contracts, monkeypatch
):
    path = tmp_path / "history.jsonl"
    path.write_text("not-json\n[]\n" + "x" * 2000 + "\n")
    entry = source(path)
    monkeypatch.setattr(backfill, "MAX_LINE", 1000)
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        state = db.execute("SELECT * FROM sources").fetchone()
        assert state["status"] == "blocked"
        assert state["rows"] == 2
        assert db.execute("SELECT n FROM dispositions").fetchone()[0] == 2


def test_sqlite_archive_is_readonly(tmp_path, contracts):
    path = tmp_path / "archive.sqlite3"
    with sqlite3.connect(path) as sql:
        sql.execute(
            "CREATE TABLE jsonl_records(id INTEGER PRIMARY KEY,source_rel TEXT,payload_json TEXT)"
        )
        sql.executemany(
            "INSERT INTO jsonl_records VALUES(?,?,?)",
            [
                (i + 1, "decisions/history.jsonl", json.dumps(r))
                for i, r in enumerate((row(), row(300, 104)))
            ],
        )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = source(path)
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        assert db.execute("SELECT status FROM sources").fetchone()[0] == "complete"
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not Path(str(path) + "-wal").exists()
    assert not Path(str(path) + "-shm").exists()


def test_late_conflicting_mark_invalidates_prior_context_revision(tmp_path, contracts):
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        for index, records in enumerate(((row(), row(300, 103)), (row(300, 107),))):
            path = tmp_path / f"history-{index}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            entry = source(path)
            register(db, entry)
            backfill.ingest_source(
                db, entry, manifest(contracts), time.monotonic() + 10, 100000
            )
            backfill.label_contexts(db, manifest(contracts), time.monotonic() + 10)
        result = db.execute(
            "SELECT status,payload,revision FROM contexts WHERE observation_id=1 AND horizon=300"
        ).fetchone()
        assert result["status"] == "quarantined"
        assert json.loads(result["payload"])["value"] is None
        assert result["revision"] == 3


def test_admission_fails_closed_without_fresh_reports(tmp_path):
    assert "preparation_not_admitted" in backfill.admission(tmp_path)
    assert "storage_writers_not_admitted" in backfill.admission(tmp_path)


def test_protected_alias_is_rejected_before_open(tmp_path):
    path = tmp_path / "protected-alias"
    path.symlink_to("/Volumes/VIDEO/do-not-open")
    with pytest.raises(ValueError, match="protected_or_unavailable"):
        backfill.safe_path(path)


def test_health_directory_hold_is_respected(tmp_path):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "RUNTIME_MAINTENANCE_HOLD.flag").touch()
    assert "maintenance_or_operator_hold" in backfill.admission(tmp_path)


def test_storage_budget_checkpoints_without_full_database_error(
    tmp_path, contracts, monkeypatch
):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    entry = source(path)
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        monkeypatch.setattr(backfill, "database_headroom", lambda _: False)
        backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        state = db.execute("SELECT * FROM sources").fetchone()
        assert state["status"] == "pending"
        assert state["reason"] == "sidecar_storage_budget_reached"
        assert state["rows"] == 0


def test_sqlite_wal_is_never_opened_immutable(tmp_path, contracts):
    path = tmp_path / "history.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE jsonl_records(id INTEGER,source_rel TEXT,payload_json TEXT)"
        )
    entry = source(path)
    Path(str(path) + "-wal").touch()
    with backfill.connect(tmp_path / "labels.sqlite3") as db:
        register(db, entry)
        backfill.ingest_source(
            db, entry, manifest(contracts), time.monotonic() + 10, 100000
        )
        assert (
            db.execute("SELECT reason FROM sources").fetchone()[0]
            == "sqlite_wal_appeared"
        )


def test_inventory_digest_tampering_is_detected(tmp_path, contracts, monkeypatch):
    monkeypatch.setattr(backfill, "label_contracts", lambda _: contracts)
    monkeypatch.setattr(backfill, "source_roots", lambda _: [])
    backfill.run(tmp_path, execute=False)
    base = tmp_path / backfill.STATE_REL
    pointer = json.loads((base / "latest_run.json").read_text())
    path = base / "runs" / pointer["run_id"] / "inventory.json"
    value = json.loads(path.read_text())
    value["created_at_utc"] = "tampered"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inventory_integrity_failure"):
        backfill.run(tmp_path, execute=False)


def test_blocked_run_preserves_evidence_time_without_materializing(
    tmp_path, monkeypatch
):
    health = tmp_path / backfill.REPORT_REL
    health.parent.mkdir(parents=True)
    health.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-09-01T00:00:00+00:00",
                "source_records_processed": 42,
            }
        )
    )
    monkeypatch.setattr(
        backfill, "admission", lambda _: ["storage_regression_gate_not_admitted"]
    )
    monkeypatch.setattr(
        backfill,
        "label_contracts",
        lambda _: pytest.fail("must not start another inventory"),
    )
    result = backfill.run(tmp_path, execute=True)
    assert result["evidence_timestamp_utc"] == "2026-09-01T00:00:00+00:00"
    assert result["source_records_processed"] == 42
    assert result["batch_fresh_payload_bytes_processed"] == 0
    assert result["previous_evidence_not_recomputed"]
    assert not any(result["authority"].values())
    assert not (tmp_path / backfill.STATE_REL / "runs").exists()
