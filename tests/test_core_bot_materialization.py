from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from scripts.ops import materialize_core_bot_modules as materializer
import core_bot_materialization_guard as guard_src


def _write_registry(project_root: Path, bot_id: str) -> None:
    (project_root / "core").mkdir(parents=True, exist_ok=True)
    (project_root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": bot_id,
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "reason": "planned_roster_expansion_slot",
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Example Expansion Bot",
                        "slot_kind": "example_signal",
                        "slot_priority": "high",
                        "slot_objective": "Collect enough observations before promotion.",
                        "target_functions": ["example_sync"],
                        "preferred_regimes": ["mixed_transition"],
                        "data_collection_active": True,
                        "training_excluded": True,
                        "exclude_from_training": True,
                        "training_candidate_after_threshold": True,
                        "minimum_training_observations": 1000,
                        "minimum_data_collection_days": 7,
                        "trading_enabled": False,
                        "paper_trading_enabled": False,
                        "live_trading_enabled": False,
                    }
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_materialize_core_bot_modules_creates_registry_backed_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v999_example_collection_bot"
    _write_registry(project_root, bot_id)

    before = guard_src.build_payload(project_root)
    payload = materializer.materialize(project_root)
    after = guard_src.build_payload(project_root)

    generated = project_root / "core" / f"{bot_id}.py"
    assert before["summary"]["missing_core_module_count"] == 1
    assert payload["created_count"] == 1
    assert generated.exists()
    assert "BOT_SPEC" in generated.read_text(encoding="utf-8")
    assert after["overall_status"] == "ready"
    assert after["summary"]["missing_core_module_count"] == 0
    assert after["summary"]["duplicate_core_version_count"] == 0


def test_materializer_does_not_overwrite_hand_built_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v998_custom_collection_bot"
    _write_registry(project_root, bot_id)
    custom_file = project_root / "core" / f"{bot_id}.py"
    custom_file.write_text("# hand built\nBOT_ID = 'custom'\n", encoding="utf-8")

    payload = materializer.materialize(project_root)

    assert payload["created_count"] == 0
    assert payload["skipped_existing_count"] == 1
    assert custom_file.read_text(encoding="utf-8") == "# hand built\nBOT_ID = 'custom'\n"


def test_guard_flags_duplicate_core_versions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    core_dir = project_root / "core"
    core_dir.mkdir(parents=True)
    (project_root / "master_bot_registry.json").write_text(json.dumps({"sub_bots": []}), encoding="utf-8")
    (core_dir / "brain_refinery_v777_first.py").write_text("BOT_ID = 'first'\n", encoding="utf-8")
    (core_dir / "brain_refinery_v777_second.py").write_text("BOT_ID = 'second'\n", encoding="utf-8")

    payload = guard_src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["summary"]["duplicate_core_version_count"] == 1
    assert payload["duplicate_core_versions"]["777"] == [
        "brain_refinery_v777_first.py",
        "brain_refinery_v777_second.py",
    ]
