import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import paper_replay_drill as src


def _rows(count=20, *, age_days=0):
    stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    return [{"timestamp_utc": stamp, "decision_id": str(i), "symbol": "SPY",
             "action": "BUY", "quantity": 1, "fill_price": 100, "mode": "paper"}
            for i in range(count)]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(data)
    else:
        path.write_text(data)


def _run(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    out = tmp_path / "replay.json"
    monkeypatch.setattr(sys, "argv", ["paper-replay", "--hours", "336", "--strict-exit",
        "--out-file", str(out), "--json", *extra])
    rc = src.main()
    return rc, json.loads(out.read_text())


def test_compressed_external_trade_logs_are_discovered(tmp_path, monkeypatch):
    external = tmp_path / "external"
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external))
    path = external / "exports/trade_logs/shadow_aggressive_equities/paper_trades_paper.jsonl.gz"
    _write(path, _rows())
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 0 and payload["ok"]
    assert payload["rows"] == 20
    assert payload["source"]["source_mode"] == "paper_trades"


def test_identical_raw_and_compressed_rows_do_not_inflate_floor(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    rows = _rows(1) * 20
    _write(tmp_path / "paper_trades_test.jsonl", rows)
    _write(tmp_path / "paper_trades_test.jsonl.gz", rows)
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2 and payload["rows"] == 1
    assert payload["source"]["duplicate_rows_excluded"] == 39
    assert "paper_rows_low" in payload["failed_checks"]


def test_old_and_future_rows_cannot_earn_window_coverage(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    _write(tmp_path / "paper_trades_test.jsonl", _rows(age_days=30) + _rows(age_days=-1))
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2 and payload["rows"] == 0
    assert "invalid_source_timestamp" in payload["failed_checks"]


def test_malformed_source_cannot_pass_using_valid_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    path = tmp_path / "paper_trades_test.jsonl"
    _write(path, _rows())
    with path.open("a") as handle:
        handle.write("invalid-json\n")
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2 and payload["rows"] == 20
    assert "source_read_or_decode_failed" in payload["failed_checks"]


def test_shared_byte_budget_cannot_publish_partial_success(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    _write(tmp_path / "paper_trades_test.jsonl", _rows())
    monkeypatch.setattr(src, "MAX_REPLAY_BYTES", 100)
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2
    assert payload["source"]["bytes_read"] == 100
    assert "incomplete_or_oversized_source_row" in payload["failed_checks"]


def test_gzip_crc_failure_preserves_failed_scan(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    path = tmp_path / "paper_trades_test.jsonl.gz"
    _write(path, _rows())
    path.write_bytes(path.read_bytes()[:-5])
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2
    assert "source_read_or_decode_failed" in payload["failed_checks"]


def test_execution_intents_are_diagnostic_not_completed_paper_replay(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    rows = [dict(row, target_mode="paper", message_id=row["decision_id"]) for row in _rows()]
    _write(tmp_path / "governance/execution_lanes/execution_intents_latest.jsonl", rows)
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2 and payload["rows"] == 20
    assert "execution_intents_only_not_paper_replay" in payload["failed_checks"]


def test_scoped_replay_does_not_fall_back_to_unscoped_execution_results(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    rows = [{"mode": "paper", "result": {"decision": row}} for row in _rows()]
    _write(tmp_path / "governance/execution_lanes/execution_results_latest.jsonl", rows)
    rc, payload = _run(tmp_path, monkeypatch, "--profile", "aggressive")
    assert rc == 2 and payload["rows"] == 0


def test_unreadable_fallback_route_cannot_pass_from_another_route(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    rows = [{"mode": "paper", "result": {"decision": row}} for row in _rows()]
    _write(tmp_path / "governance/execution_lanes/execution_results_latest.jsonl", rows)
    denied = tmp_path / "local_fallback_storage/governance/execution_lanes"
    inspect = src.inspect_storage_path

    def guarded(path):
        return {"status": "permission_denied"} if path == denied else inspect(path)

    monkeypatch.setattr(src, "inspect_storage_path", guarded)
    rc, payload = _run(tmp_path, monkeypatch)
    assert rc == 2 and payload["rows"] == 20
    assert "source_discovery_incomplete" in payload["failed_checks"]
    assert payload["source"]["discovery"]["discovery_error_count"] == 1


def test_protected_input_alias_is_rejected_before_target_stat(tmp_path, monkeypatch):
    path = tmp_path / "paper_trades_test.jsonl"
    path.symlink_to("/Volumes/VIDEO/never-read.jsonl")
    original = Path.lstat

    def checked(path, *args, **kwargs):
        assert not str(path).startswith("/Volumes/VIDEO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked)
    rc, payload = _run(tmp_path, monkeypatch, "--in-file", str(path))
    assert rc == 2
    assert "source_route_protected_path" in payload["failed_checks"]
