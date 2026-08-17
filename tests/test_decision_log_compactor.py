from __future__ import annotations

import gzip
from pathlib import Path

from scripts.ops import decision_log_compactor as src


def _write_decision_file(project_root: Path, day: str = "20260519", profile: str = "paper") -> Path:
    path = project_root / "decisions" / profile / f"trade_decisions_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"timestamp_utc":"2026-05-19T00:00:00+00:00","symbol":"BTC-USD","action":"HOLD"}\n' * 16,
        encoding="utf-8",
    )
    return path


def _write_fallback_file(project_root: Path, day: str = "20260519") -> Path:
    path = project_root / "decision_explanations" / "shadow_crypto" / f"decision_explanations_{day}.jsonl.local_fallback.1"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"timestamp_utc":"2026-05-19T00:00:00+00:00","symbol":"BTC-USD","reason":"fallback"}\n' * 16,
        encoding="utf-8",
    )
    return path


def _write_shadow_pnl_file(project_root: Path, day: str = "20260519") -> Path:
    path = project_root / "governance" / "shadow_crypto" / f"shadow_pnl_attribution_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"timestamp_utc":"2026-05-19T00:00:00+00:00","symbol":"BTC-USD","pnl":0.0}\n' * 16,
        encoding="utf-8",
    )
    return path


def test_decision_log_compactor_dry_run_selects_old_large_files(tmp_path: Path) -> None:
    source = _write_decision_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["selected_count"] == 1
    assert payload["records"][0]["relative_path"] == "decisions/paper/" + source.name
    assert source.exists()


def test_decision_log_compactor_apply_gzips_in_place(tmp_path: Path) -> None:
    source = _write_decision_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
        compression_level=1,
    )

    archive = source.with_name(source.name + ".gz")
    assert payload["overall_status"] == "applied"
    assert payload["summary"]["compacted_count"] == 1
    assert not source.exists()
    assert archive.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert content.count("\n") == 16
    assert "BTC-USD" in content


def test_decision_log_compactor_replaces_existing_archive_after_success(tmp_path: Path) -> None:
    source = _write_decision_file(tmp_path)
    archive = source.with_name(source.name + ".gz")
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write('{"symbol":"LEGACY_SYMBOL"}\n')

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
        compression_level=1,
    )

    assert payload["overall_status"] == "applied"
    assert payload["records"][0]["archive_replaced"] is True
    assert not source.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert "LEGACY_SYMBOL" not in content
    assert "BTC-USD" in content


def test_decision_log_compactor_skips_current_day_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(src, "_today_stamp", lambda: "20260522")
    _write_decision_file(tmp_path, day="20260522")

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "nothing_to_do"
    assert payload["summary"]["candidate_count"] == 0


def test_decision_log_compactor_includes_local_fallback_spillover(tmp_path: Path) -> None:
    source = _write_fallback_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        families=["decision_explanations"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "planned"
    assert payload["records"][0]["relative_path"] == "decision_explanations/shadow_crypto/" + source.name
    assert payload["records"][0]["fallback_copy"] is True


def test_decision_log_compactor_applies_to_local_fallback_spillover(tmp_path: Path) -> None:
    source = _write_fallback_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        families=["decision_explanations"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
        compression_level=1,
    )

    archive = source.with_name(source.name + ".gz")
    assert payload["overall_status"] == "applied"
    assert not source.exists()
    assert archive.exists()


def test_decision_log_compactor_includes_shadow_pnl_attribution_family(tmp_path: Path) -> None:
    source = _write_shadow_pnl_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        families=["shadow_pnl_attribution"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert payload["records"][0]["relative_path"] == "governance/shadow_crypto/" + source.name
