from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import governance_telemetry_compactor as src


def _write_channel_file(project_root: Path, *, channel: str = "decision", profile: str = "default_crypto_schwab") -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = project_root / "governance" / "channels" / channel / profile / f"{channel}_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp_utc":"2026-05-22T00:00:00+00:00","symbol":"BTC-USD","action":"HOLD"}\n' * 8, encoding="utf-8")
    return path


def _write_master_control_file(project_root: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = project_root / "governance" / "shadow_crypto" / f"master_control_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp_utc":"2026-05-22T00:00:00+00:00","mode":"shadow_crypto"}\n' * 8, encoding="utf-8")
    return path


def _write_execution_lane_file(project_root: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = project_root / "governance" / "execution_lanes" / f"execution_intents_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp_utc":"2026-05-22T00:00:00+00:00","intent":"paper"}\n' * 8, encoding="utf-8")
    return path


def test_compactor_dry_run_selects_oversized_current_day_channel(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        channels=["decision"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["selected_count"] == 1
    assert payload["records"][0]["relative_path"] == "governance/channels/decision/default_crypto_schwab/" + source.name
    assert source.exists()
    assert source.read_text(encoding="utf-8").count("\n") == 8


def test_compactor_default_discovers_risk_channel(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path, channel="risk")

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
    )

    assert payload["overall_status"] == "planned"
    assert payload["policy"]["requested_channels"] == ["all"]
    assert "risk" in payload["policy"]["channels"]
    assert payload["records"][0]["relative_path"] == "governance/channels/risk/default_crypto_schwab/" + source.name


def test_compactor_apply_rotates_to_stale_stage_and_keeps_fresh_path(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        channels=["decision"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
        compression_level=1,
    )

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["archived_count"] == 1
    assert payload["summary"]["raw_archived_bytes"] > 0
    assert source.exists()
    assert source.read_text(encoding="utf-8") == ""

    archive_rel = payload["records"][0]["archive_path"]
    archive_path = tmp_path / archive_rel
    assert archive_path.exists()
    assert "data/stale_stage/governance_telemetry_compactor/" in archive_rel
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        archived = handle.read()
    assert archived.count("\n") == 8
    assert "BTC-USD" in archived


def test_compactor_can_skip_current_day_files(tmp_path: Path) -> None:
    _write_channel_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        channels=["decision"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=False,
    )

    assert payload["overall_status"] == "nothing_to_do"
    assert payload["summary"]["candidate_count"] == 0


def test_compactor_covers_master_control_and_execution_lanes(tmp_path: Path) -> None:
    master_control = _write_master_control_file(tmp_path)
    execution_lane = _write_execution_lane_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        families=["master_control", "execution_lanes"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
    )

    rels = {row["relative_path"]: row["family"] for row in payload["records"]}
    assert payload["overall_status"] == "planned"
    assert rels[f"governance/shadow_crypto/{master_control.name}"] == "master_control"
    assert rels[f"governance/execution_lanes/{execution_lane.name}"] == "execution_lanes"


def test_compactor_applies_to_master_control_without_losing_active_path(tmp_path: Path) -> None:
    source = _write_master_control_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        families=["master_control"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
        compression_level=1,
    )

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["archived_count"] == 1
    assert source.exists()
    assert source.read_text(encoding="utf-8") == ""
    archive_path = tmp_path / payload["records"][0]["archive_path"]
    assert archive_path.exists()
