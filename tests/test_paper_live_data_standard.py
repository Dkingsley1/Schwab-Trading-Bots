import json
from pathlib import Path

from scripts.ops import paper_live_data_standard as src


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": {"active_bots": 0}, "sub_bots": rows}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_profitability_weak_profiles(project_root: Path, profiles: list[str]) -> None:
    _write_json(
        project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json",
        {
            "raw_profitability_a_recovery_contract": {
                "active": True,
                "weak_profiles": profiles,
                "runtime_enforcement": {
                    "block_new_entries_on_weak_profiles": True,
                    "keep_sells_and_reduce_only_paths_open": True,
                },
            },
            "raw_profitability_improvement_contract": {
                "active": True,
                "weak_sleeve_zero_entry_contract": {
                    "profiles": [
                        {"profile": profile, "block_new_entries": True}
                        for profile in profiles
                    ],
                },
            },
        },
    )


def test_paper_live_data_standard_keeps_legacy_paper_and_new_collecting(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    _write_registry(
        registry_path,
        [
            {
                "bot_id": "legacy_active",
                "active": True,
                "lifecycle_state": "active",
                "test_accuracy": 0.61,
                "direct_execution_allowed": False,
            },
            {
                "bot_id": "new_collector",
                "active": True,
                "lifecycle_state": "data_collection_only",
                "data_collection_active": True,
                "paper_runtime_stability_mode": "full_force_guarded",
            },
            {
                "bot_id": "brain_refinery_v26_restored_probation",
                "active": False,
                "lifecycle_state": "probation",
                "reason": "stale_training_diagnostic",
                "test_accuracy": 0.58,
            },
            {
                "bot_id": "ready_new_paper",
                "active": True,
                "lifecycle_state": "data_collection_only",
                "data_collection_training_ready": True,
                "data_collection_threshold_progress": {
                    "observations": 1250,
                    "minimum_training_observations": 1000,
                    "observations_ready": True,
                    "collection_age_days": 8,
                    "days_ready": True,
                    "training_ready": True,
                },
                "label_contract": {"version": "universal_training_label_contract_v1"},
                "quality_score": 0.57,
            },
            {
                "bot_id": "deleted_bot",
                "active": False,
                "deleted_from_rotation": True,
                "lifecycle_state": "inactive",
            },
        ],
    )

    payload = src.build_payload(tmp_path, registry_path=registry_path)
    projected = payload["projected_registry"]["sub_bots"]
    by_id = {row["bot_id"]: row for row in projected}

    assert payload["overall_status"] == "ready"
    assert payload["counts_after"]["active_bots"] == 4
    assert payload["counts_after"]["data_collection_active_bots"] == 4
    assert payload["counts_after"]["paper_live_data_enabled_bots"] == 3
    assert payload["counts_after"]["legacy_bootstrap_paper_bots"] == 1
    assert payload["counts_after"]["standard_promoted_paper_bots"] == 1
    assert payload["counts_after"]["collection_until_standard_bots"] == 1
    assert by_id["legacy_active"]["paper_live_data_enabled"] is True
    assert by_id["legacy_active"]["paper_execution_allowed"] is True
    assert by_id["ready_new_paper"]["paper_live_data_enabled"] is True
    assert by_id["ready_new_paper"]["paper_standard_cohort"] == "standard_promoted"
    assert by_id["new_collector"]["paper_live_data_enabled"] is False
    assert by_id["new_collector"]["promotion_blocked_until"] == "paper_live_data_standard_met"
    assert by_id["brain_refinery_v26_restored_probation"]["active"] is True
    assert by_id["brain_refinery_v26_restored_probation"]["lifecycle_state"] == "paper_live_data"
    assert by_id["brain_refinery_v26_restored_probation"]["paper_standard_cohort"] == "legacy_bootstrap"
    assert by_id["deleted_bot"]["active"] is False
    assert by_id["deleted_bot"]["paper_standard_status"] == "deleted_preserved"
    assert all(row.get("direct_execution_allowed") is False for row in projected)
    assert all(row.get("live_trading_enabled") is not True for row in projected)


def test_paper_live_data_standard_apply_updates_registry_summary_and_backup(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    out_path = tmp_path / "governance" / "health" / "paper_live_data_standard_latest.json"
    override_path = tmp_path / "config" / ".env.paper_live_data_standard_override"
    backup_dir = tmp_path / "governance" / "lifecycle"
    _write_registry(
        registry_path,
        [
            {"bot_id": "legacy_active", "active": True, "lifecycle_state": "active"},
            {"bot_id": "new_collector", "active": True, "lifecycle_state": "data_collection_only"},
        ],
    )

    payload = src.build_payload(tmp_path, registry_path=registry_path)
    applied = src.apply_payload(
        tmp_path,
        payload,
        registry_path=registry_path,
        out_path=out_path,
        override_path=override_path,
        backup_dir=backup_dir,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    health = json.loads(out_path.read_text(encoding="utf-8"))

    assert applied["apply_result"]["applied"] is True
    assert registry["summary"]["active_bots"] == 2
    assert registry["summary"]["data_collection_active_bots"] == 2
    assert registry["summary"]["paper_live_data_enabled_bots"] == 1
    assert registry["summary"]["collection_until_standard_bots"] == 1
    assert Path(applied["apply_result"]["backup_path"]).exists()
    assert override_path.exists()
    override_text = override_path.read_text(encoding="utf-8")
    assert "PAPER_LIVE_DATA_STANDARD_ENABLED=1" in override_text
    assert "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS=1" in override_text
    assert "ALLOW_ORDER_EXECUTION=0" in override_text
    assert health["counts_after"]["paper_live_data_enabled_bots"] == 1


def test_paper_live_data_standard_keeps_coinbase_paper_probation_when_profiles_are_weak(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    override_path = tmp_path / "config" / ".env.paper_live_data_standard_override"
    _write_registry(
        registry_path,
        [
            {"bot_id": "legacy_active", "active": True, "lifecycle_state": "active"},
            {"bot_id": "new_collector", "active": True, "lifecycle_state": "data_collection_only"},
        ],
    )
    _write_profitability_weak_profiles(
        tmp_path,
        ["default", "crypto_futures", "bond", "fx", "options_on_futures"],
    )

    payload = src.build_payload(tmp_path, registry_path=registry_path)
    src.apply_payload(
        tmp_path,
        payload,
        registry_path=registry_path,
        out_path=tmp_path / "governance" / "health" / "paper_live_data_standard_latest.json",
        override_path=override_path,
        backup_dir=tmp_path / "governance" / "lifecycle",
    )

    override_text = override_path.read_text(encoding="utf-8")
    schwab_line = next(line for line in override_text.splitlines() if line.startswith("SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES="))
    assert "volatility" in schwab_line
    assert "default" not in schwab_line
    assert "bond" not in schwab_line
    assert "fx" not in schwab_line
    assert "COINBASE_TOP_BOT_PAPER_TRADING_TOP_N=50" in override_text
    assert "COINBASE_TOP_BOT_PAPER_TRADING_PROFILES=default" in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N=30" in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES=crypto_futures" in override_text
    assert "COINBASE_PAPER_PROBATION_ENABLED=1" in override_text
    assert "COINBASE_PAPER_PROBATIONARY_PROFILES=default,crypto_futures" in override_text
    assert "PAPER_PROFITABILITY_WEAK_PROFILES=bond,crypto_futures,default,fx,options_on_futures" in override_text
    assert "PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT=1" in override_text
    assert "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=1" in override_text
    assert "RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST=volatility" in override_text


def test_paper_live_data_standard_preview_writes_health_without_registry_change(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    out_path = tmp_path / "governance" / "health" / "paper_live_data_standard_latest.json"
    override_path = tmp_path / "config" / ".env.paper_live_data_standard_override"
    _write_registry(
        registry_path,
        [
            {"bot_id": "legacy_active", "active": True, "lifecycle_state": "active"},
            {"bot_id": "new_collector", "active": True, "lifecycle_state": "data_collection_only"},
        ],
    )
    original = registry_path.read_text(encoding="utf-8")

    payload = src.build_payload(tmp_path, registry_path=registry_path)
    preview = src.preview_payload(
        tmp_path,
        payload,
        registry_path=registry_path,
        out_path=out_path,
        override_path=override_path,
    )
    health = json.loads(out_path.read_text(encoding="utf-8"))

    assert preview["apply_result"]["applied"] is False
    assert preview["apply_result"]["mode"] == "preview_no_registry_change"
    assert registry_path.read_text(encoding="utf-8") == original
    assert out_path.exists()
    assert not override_path.exists()
    assert health["overall_status"] == "ready"
    assert "projected_registry" not in health
