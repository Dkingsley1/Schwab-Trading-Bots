import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import deep_cold_storage_layer as src


def test_explicit_age_only_changes_closed_gzip_movement(tmp_path, monkeypatch):
    root = tmp_path / "decisions" / "sleeve"
    root.mkdir(parents=True)
    old = root / "trade_decisions_20200101.jsonl.gz"
    paths = [
        old,
        root / "trade_decisions_20200101.jsonl",
        root / "trade_decisions_20200231.jsonl.gz",
        root / "trade_decisions_20990101.jsonl.gz",
        root / f"trade_decisions_{datetime.now(timezone.utc):%Y%m%d}.jsonl.gz",
        root / "latest.jsonl.gz",
    ]
    for path in paths:
        path.write_bytes(b"retained bytes")
        os.utime(path, (time.time() - 13 * 3600,) * 2)
    monkeypatch.setattr(
        src,
        "resolve_external_storage",
        lambda: SimpleNamespace(external_root=tmp_path / "external"),
    )
    options = dict(min_size_mb=0.000001, include_compressed_history=True)
    assert not src.build_payload(tmp_path, **options)["top_rows"]
    payload = src.build_payload(tmp_path, closed_history_min_age_hours=12, **options)
    assert [row["path"] for row in payload["top_rows"]] == [str(old)]
    assert payload["closed_history_min_age_hours"] == 12
    assert not src.build_payload(
        tmp_path, closed_history_min_age_hours=12, min_size_mb=0.000001
    )["top_rows"]
    os.utime(old, (time.time(),) * 2)
    assert not src.build_payload(tmp_path, closed_history_min_age_hours=12, **options)[
        "top_rows"
    ]


@pytest.mark.parametrize("age", [-1, 0, 0.99, float("nan"), float("inf")])
def test_invalid_age_rejected_before_inventory(tmp_path, monkeypatch, age):
    def forbidden():
        raise AssertionError("inventory started")

    monkeypatch.setattr(src, "resolve_external_storage", forbidden)
    with pytest.raises(ValueError, match="at least 1"):
        src.build_payload(tmp_path, closed_history_min_age_hours=age)
