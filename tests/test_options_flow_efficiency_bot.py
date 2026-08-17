from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import options_flow_efficiency_bot as efficiency_src
import options_flow_export_hygiene_bot as hygiene_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_options_flow_export_hygiene_bot_promotes_canonical_export(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    raw_export = export_dir / "uw_drop_20260415.json"
    raw_export.write_text(
        json.dumps({"symbols": {"SPY": {"iv_rank": {"iv_rank": 60}}}}),
        encoding="utf-8",
    )

    payload = hygiene_src.build_payload(
        tmp_path,
        export_path=str(export_dir),
        max_age_seconds=21600,
        min_stable_seconds=0,
        apply=True,
    )

    assert payload["overall_status"] == "ready"
    assert payload["promotion"]["promoted"] is True
    assert Path(payload["promotion"]["promoted_path"]).exists()
    promoted = json.loads(Path(payload["promotion"]["promoted_path"]).read_text(encoding="utf-8"))
    assert promoted["schema_version"] == "uw_options_flow_export.v2"


def test_options_flow_efficiency_bot_refreshes_when_context_is_stale(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    _write_json(
        health / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": stale_ts,
            "ok": False,
            "overall_status": "blocked",
            "sources": {"unusual_whales_export": {"selected_candidate": ""}},
        },
    )
    export_path = project_root / "uw_export.json"
    export_path.write_text(
        json.dumps({"symbols": {"SPY": {"iv_rank": {"iv_rank": 58}}}}),
        encoding="utf-8",
    )

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        _write_json(
            health / "options_flow_context_sync_latest.json",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "overall_status": "ready",
                "context_profile": "polygon_backbone_only",
                "sources": {"unusual_whales_export": {"selected_candidate": str(export_path), "ok": True}},
            },
        )
        return {"rc": 0, "timed_out": False, "stdout_tail": "", "stderr_tail": "", "payload": {"ok": True}}

    monkeypatch.setattr(efficiency_src, "_run_json", _fake_run_json)

    payload = efficiency_src.build_payload(
        project_root,
        apply=True,
        export_path=str(export_path),
        status_max_age_seconds=60,
        export_max_age_seconds=21600,
        export_min_stable_seconds=0,
        timeout_sec=30,
    )

    assert payload["refresh_needed"] is True
    assert payload["metrics"]["collector_executed"] is True
    assert "refreshed_context" in payload["actions_taken"]
    assert payload["latest_context"]["overall_status"] == "ready"


def test_options_flow_efficiency_bot_skips_refresh_when_context_is_current(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    health = project_root / "governance" / "health"
    export_path = project_root / "latest_options_flow_export.json"
    export_path.write_text(
        json.dumps({"symbols": {"SPY": {"iv_rank": {"iv_rank": 58}}}}),
        encoding="utf-8",
    )
    _write_json(
        health / "options_flow_context_sync_latest.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "overall_status": "ready",
            "context_profile": "polygon_backbone_only",
            "sources": {
                "unusual_whales_export": {
                    "selected_candidate": str(export_path),
                    "ok": True,
                }
            },
        },
    )

    payload = efficiency_src.build_payload(
        project_root,
        apply=False,
        export_path=str(export_path),
        status_max_age_seconds=14400,
        export_max_age_seconds=21600,
        export_min_stable_seconds=0,
        timeout_sec=30,
    )

    assert payload["refresh_needed"] is False
    assert payload["metrics"]["collector_executed"] is False
    assert payload["overall_status"] == "ready"
