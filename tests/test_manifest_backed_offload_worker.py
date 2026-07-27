from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import manifest_backed_offload_worker as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _manifest(project_root: Path, source_rel: str, size_bytes: int) -> dict:
    return {
        "timestamp_utc": "2026-06-13T00:00:00+00:00",
        "schema_version": 1,
        "project_root": str(project_root),
        "entries": [
            {
                "relative_path": source_rel,
                "planned_cold_relative_path": f"data/deep_cold/manifest_backed/{source_rel}",
                "size_bytes": size_bytes,
                "classification": "eligible_manifest_backed_offload",
                "allowed_actions": [
                    "copy_to_cold_tier",
                    "verify_size",
                    "verify_sha256",
                    "write_restore_proof",
                    "source_delete_requires_retention_gate",
                ],
                "proof_required": {
                    "post_copy_size_match": True,
                    "post_copy_sha256_match": True,
                    "restore_probe": True,
                    "retention_gate_before_source_delete": True,
                },
            }
        ],
    }


def test_manifest_backed_offload_blocks_without_cold_target(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "decision_explanations" / "shadow" / "decision_explanations_20260612.jsonl.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"explanation\n" * 64)
    manifest_path = project_root / "governance" / "health" / "storage_tier_offload_manifest_latest.json"
    _write_json(manifest_path, _manifest(project_root, str(source.relative_to(project_root)), source.stat().st_size))

    payload = src.build_payload(
        project_root,
        manifest_path=manifest_path,
        target_root=str(tmp_path / "missing_cold"),
        out_path=project_root / "out.json",
        proof_path=project_root / "proofs.jsonl",
    )

    assert payload["overall_status"] == "blocked_waiting_for_cold_target"
    assert payload["selected_count"] == 1
    assert payload["apply_result"]["source_delete_performed"] is False


def test_manifest_backed_offload_apply_copies_verifies_and_keeps_source(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    target_root = tmp_path / "BOT_COLD" / "schwab_trading_bot"
    target_root.mkdir(parents=True)
    source = project_root / "decision_explanations" / "shadow" / "decision_explanations_20260612.jsonl.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"explanation\n" * 128)
    source_rel = str(source.relative_to(project_root))
    manifest_path = project_root / "governance" / "health" / "storage_tier_offload_manifest_latest.json"
    proof_path = project_root / "governance" / "health" / "proofs.jsonl"
    _write_json(manifest_path, _manifest(project_root, source_rel, source.stat().st_size))

    payload = src.build_payload(
        project_root,
        manifest_path=manifest_path,
        target_root=str(target_root),
        apply=True,
        max_files=1,
        max_gb=1.0,
        out_path=project_root / "out.json",
        proof_path=proof_path,
    )

    target = target_root / "data" / "deep_cold" / "manifest_backed" / source_rel
    assert payload["overall_status"] == "applied"
    assert payload["apply_result"]["copied_count"] == 1
    assert payload["apply_result"]["source_delete_performed"] is False
    assert source.exists()
    assert target.exists()
    assert target.read_bytes() == source.read_bytes()
    proof = json.loads(proof_path.read_text(encoding="utf-8").strip())
    assert proof["status"] == "copied_verified"
    assert proof["source_sha256"] == proof["target_sha256"]
    assert proof["source_retained"] is True


def test_manifest_backed_offload_release_source_after_restore_proof(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    target_root = tmp_path / "BOT_COLD" / "schwab_trading_bot"
    target_root.mkdir(parents=True)
    source = project_root / "decision_explanations" / "shadow" / "decision_explanations_20260612.jsonl.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"explanation\n" * 128)
    source_rel = str(source.relative_to(project_root))
    manifest_path = project_root / "governance" / "health" / "storage_tier_offload_manifest_latest.json"
    proof_path = project_root / "governance" / "health" / "proofs.jsonl"
    _write_json(manifest_path, _manifest(project_root, source_rel, source.stat().st_size))

    payload = src.build_payload(
        project_root,
        manifest_path=manifest_path,
        target_root=str(target_root),
        apply=True,
        max_files=1,
        max_gb=1.0,
        release_source_after_verify=True,
        out_path=project_root / "out.json",
        proof_path=proof_path,
    )

    target = target_root / "data" / "deep_cold" / "manifest_backed" / source_rel
    assert payload["overall_status"] == "applied"
    assert payload["apply_result"]["copied_count"] == 1
    assert payload["apply_result"]["released_count"] == 1
    assert payload["apply_result"]["source_delete_performed"] is True
    assert not source.exists()
    assert target.exists()
    proof = json.loads(proof_path.read_text(encoding="utf-8").strip())
    assert proof["status"] == "copied_verified_source_released"
    assert proof["source_release_status"] == "released_after_restore_proof"
    assert proof["source_retained"] is False


def test_video_cold_archive_override_is_subtree_scoped(monkeypatch) -> None:
    monkeypatch.setenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", "1")
    monkeypatch.setenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", "/Volumes/VIDEO/schwab_trading_bot_cold")

    assert src._is_protected(Path("/Volumes/VIDEO")) is True
    assert src._is_protected(Path("/Volumes/VIDEO/schwab_trading_bot_cold")) is False
    assert src._is_protected(Path("/Volumes/VIDEO/schwab_trading_bot_cold/data/file.gz")) is False
