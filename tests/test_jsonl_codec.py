from __future__ import annotations

import gzip
import json
import math

import pytest

from scripts import jsonl_codec as codec
from scripts.ops.market_cycle_extraction_engine import _parse_jsonl_lines
from scripts.ops.paper_profitability_control import _iter_jsonl_records


@pytest.mark.parametrize(
    "text",
    [
        '{"n":18446744073709551616,"negative":-18446744073709551617}',
        '{"zero":-0.0,"x":[true,false,null,0.1,1e-300]}',
        '{"same":1,"same":2,"text":"\\u2603"}',
        '{"text":"\\ud800"}',
        '{"big":1e400,"negative":-Infinity}',
        "[]",
        "null",
        '"hello"',
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_compatibility(text, fallback, monkeypatch):
    if fallback:
        monkeypatch.setattr(codec, "_decode", None)
    result, expected = codec.loads(text), json.loads(text)
    assert result == expected
    if isinstance(result, dict) and "n" in result:
        assert type(result["n"]) is int
    if isinstance(result, dict) and "zero" in result:
        assert math.copysign(1, result["zero"]) == -1


def test_legacy_nan_and_backend_failure(monkeypatch):
    assert math.isnan(codec.loads('{"value":NaN}')["value"])

    def broken(_):
        raise RuntimeError("optional_backend_failed")

    monkeypatch.setattr(codec, "_decode", broken)
    assert codec.loads('{"v":3}') == {"v": 3}


@pytest.mark.parametrize(
    "text", ['{"a":}', '{"x":01}', "\ufeff{}", '{"n":' + "1" * 5000 + "}"]
)
def test_invalid_json_stays_invalid(text):
    with pytest.raises(ValueError):
        codec.loads(text)


def test_market_reader_retains_utf8_dict_and_tail_rules():
    rows = [
        b'{"v":1}',
        b"bad",
        b"[]",
        b'{"v":2}',
        b'{"v":3}',
        b"\xff",
        "{}".encode("utf-16"),
    ]
    assert _parse_jsonl_lines(rows, limit=2) == [{"v": 2}, {"v": 3}]


@pytest.mark.parametrize("compressed", [False, True])
def test_profitability_reader_retains_record_budget(tmp_path, compressed):
    path = tmp_path / ("rows.jsonl.gz" if compressed else "rows.jsonl")
    content = b'\n{"v":1}\nbad\n[]\n{"v":4}\n'
    path.write_bytes(gzip.compress(content) if compressed else content)
    assert list(_iter_jsonl_records(path, max_records=3)) == [{"v": 1}]
    assert list(_iter_jsonl_records(path, max_records=4)) == [{"v": 1}, {"v": 4}]
