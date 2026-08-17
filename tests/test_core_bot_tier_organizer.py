from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import organize_core_bot_tiers as tier_src


def test_core_bot_tier_organizer_builds_nonbreaking_tier_view(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    core = project_root / "core"
    ops = project_root / "scripts" / "ops"
    core.mkdir(parents=True)
    ops.mkdir(parents=True)
    (core / "master_bot.py").write_text("BOT_ID = 'master_bot'\n", encoding="utf-8")
    (core / "brain_refinery_v1_signal.py").write_text("BOT_ID = 'signal'\n", encoding="utf-8")
    (core / "brain_refinery_v2_guard.py").write_text("BOT_ID = 'guard'\n", encoding="utf-8")
    (ops / "example_guard.py").write_text("BOT_ID = 'example_guard'\n", encoding="utf-8")
    (core / "bot_catalog.json").write_text(
        json.dumps(
            {
                "bots": [
                    {
                        "bot_id": "brain_refinery_v1_signal",
                        "bot_role": "signal_sub_bot",
                        "category": "general_signal",
                        "active": True,
                        "core_file": "core/brain_refinery_v1_signal.py",
                    },
                    {
                        "bot_id": "brain_refinery_v2_guard",
                        "bot_role": "infrastructure_sub_bot",
                        "category": "infrastructure",
                        "active": True,
                        "core_file": "core/brain_refinery_v2_guard.py",
                    },
                    {
                        "bot_id": "example_guard",
                        "bot_role": "ops_infrastructure_bot",
                        "category": "ops_infrastructure",
                        "active": None,
                        "source": "scripts/ops/example_guard.py",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = tier_src.build_tier_view(project_root)

    assert payload["overall_status"] == "ready"
    assert (core / "bot_tiers" / "01_master" / "master_bot.py").is_symlink()
    assert (core / "bot_tiers" / "02_infrastructure" / "brain_refinery_v2_guard.py").is_symlink()
    assert (core / "bot_tiers" / "03_sub_bots" / "brain_refinery_v1_signal.py").is_symlink()
    assert (core / "bot_tiers" / "04_ops_infrastructure" / "example_guard.py").is_symlink()
    assert "Generated PyCharm view" in (core / "bot_tiers" / "README.md").read_text(encoding="utf-8")
