from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import telemetry_redaction_canary as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_telemetry_redaction_canary_executes_current_config_without_persisting_raw_secrets(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    config_path = project_root / "config" / "production_readiness_control_v1.json"
    synthetic_token = "top-" + "secret-value"
    _write_json(
        config_path,
        {
            "observability_redaction": {
                "enabled_by_default": False,
                "allowed_export_modes": ["disabled", "local_only"],
                "redaction_patterns": [r"(?i)token=[^\s]+", r"(?i)account_id=[a-z0-9-]+"],
                "redaction_samples": [
                    {
                        "name": "token",
                        "input": f"token={synthetic_token} symbol=SPY",
                        "must_not_contain": [synthetic_token],
                    },
                    {
                        "name": "account",
                        "input": "account_id=ABC-123456 symbol=QQQ",
                        "must_not_contain": ["ABC-123456"],
                    },
                ],
                "policy": "test_redaction_policy",
            }
        },
    )

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "ready"
    assert payload["production_grade_ready"] is True
    assert payload["sample_count"] == 2
    assert payload["passed_sample_count"] == 2
    assert payload["leak_count"] == 0
    assert payload["control_contract"]["raw_canary_inputs_persisted"] is False
    serialized = json.dumps(payload, ensure_ascii=True)
    assert synthetic_token not in serialized
    assert "ABC-123456" not in serialized


def test_telemetry_redaction_canary_fails_closed_without_executable_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = project_root / "config" / "production_readiness_control_v1.json"
    _write_json(config_path, {"observability_redaction": {"redaction_patterns": [], "redaction_samples": []}})

    payload = src.build_payload(project_root, config_path=config_path)

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert "redaction_patterns_missing" in payload["blockers"]
    assert "redaction_canary_samples_missing" in payload["blockers"]
