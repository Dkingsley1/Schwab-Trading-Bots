from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.ops import artifact_freshness_slo as slo


@pytest.mark.parametrize("source", ["bad", None, "stale", "future"])
def test_missing_invalid_and_stale_observation_is_not_fresh(tmp_path, monkeypatch, source):
    now = datetime.now(timezone.utc)
    path = tmp_path / "test.json"
    payload = {"timestamp_utc": now.isoformat()}
    if source == "stale":
        payload["source_timestamp_utc"] = (now - timedelta(hours=2)).isoformat()
    elif source == "future":
        payload["source_timestamp_utc"] = (now + timedelta(hours=2)).isoformat()
    else:
        payload["source_timestamp_utc"] = source
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(slo, "_artifact_contract", lambda _: {
        "test": {"path": path, "required": True, "max_age_minutes": 15, "refresh_command": "test"}})
    result = slo.build_payload(tmp_path)
    assert result["sla_summary"]["stale_required"] == 1
    assert result["artifacts"][0]["stale"]
