import json
from pathlib import Path

from scripts.ops import bot_founder_dna_lineage as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_project(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "summary": {"total_bots": 3, "active_bots": 2},
            "master_policy": {"paper_first_required": True},
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v1",
                    "bot_role": "signal_sub_bot",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "legacy_harmonization_version": "legacy_v1_107_expansion_bridge_v1",
                    "legacy_harmonization_scope": "brain_refinery_v1_to_v107",
                    "target_functions": ["legacy_bot_harmonization"],
                },
                {
                    "bot_id": "brain_refinery_v108_intraday_breakout",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "target_functions": ["runtime_throttle"],
                    "data_intake_collections": ["price_bars"],
                    "bootstrap_teacher_bot_ids": ["brain_refinery_v43_intraday_ultrafast_proxy"],
                },
                {
                    "bot_id": "brain_refinery_v240_options_guard",
                    "bot_role": "options_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                },
            ],
        },
    )


def test_founder_dna_lineage_reports_partial_coverage_before_apply(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)

    summary = payload["summary"]
    assert summary["overall_status"] == "needs_work"
    assert summary["founder_present"] is True
    assert summary["total_bots"] == 3
    assert summary["explicit_founder_dna_count"] == 0
    assert summary["legacy_bridge_count"] == 1
    assert summary["missing_founder_dna_count"] == 3
    assert payload["sections"]["infrastructure_bot"]["writes_runtime_loops"] is False


def test_apply_registry_stamps_every_bot_and_updates_policy(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    result = src.apply_registry_contract(tmp_path, timestamp_utc="2026-05-02T12:00:00+00:00")
    payload = src.build_payload(tmp_path, max_rows=10)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))

    assert result["applied"] is True
    assert result["changed_rows"] == 3
    assert Path(result["backup_path"]).exists()
    assert payload["summary"]["overall_status"] == "ready"
    assert payload["summary"]["explicit_founder_dna_count"] == 3
    assert payload["summary"]["coverage_ratio"] == 1.0
    assert registry["summary"]["founder_dna_covered_bots"] == 3
    assert registry["founder_dna_policy"]["founder_bot_id"] == src.FOUNDER_BOT_ID
    assert registry["master_policy"]["founder_dna_lineage_required"] is True
    for row in registry["sub_bots"]:
        assert row["founder_bot_id"] == src.FOUNDER_BOT_ID
        assert row["founder_dna_version"] == src.DNA_VERSION
        assert "bot_founder_dna_lineage" in row["target_functions"]
        assert "founder_dna_lineage_manifest" in row["data_intake_collections"]


def test_apply_registry_is_row_idempotent_after_initial_stamp(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    src.apply_registry_contract(tmp_path, timestamp_utc="2026-05-02T12:00:00+00:00")

    result = src.apply_registry_contract(tmp_path, timestamp_utc="2026-05-02T13:00:00+00:00")

    assert result["changed_rows"] == 0


def test_write_artifacts_uses_project_root_manifest(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    payload = src.build_payload(tmp_path, max_rows=2)

    paths = src.write_artifacts(tmp_path, payload, out_file=tmp_path / "governance" / "health" / "dna.json")

    assert Path(paths["health_artifact"]).exists()
    assert Path(paths["manifest_artifact"]).exists()
    manifest = json.loads(Path(paths["manifest_artifact"]).read_text(encoding="utf-8"))
    assert manifest["summary"]["total_bots"] == 3
    assert len(manifest["lineage_rows"]) == 2
