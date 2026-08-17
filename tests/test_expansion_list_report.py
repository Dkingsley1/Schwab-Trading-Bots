from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "expansion_list_report.py"
spec = importlib.util.spec_from_file_location("expansion_list_report", MODULE_PATH)
expansion_report = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(expansion_report)


def test_build_report_groups_registry_expansion_packs(tmp_path: Path) -> None:
    registry = {
        "sub_bots": [
            {
                "bot_id": "brain_refinery_v1206_strategy_gap_convertible_bond_arbitrage_evidence_collector_bot",
                "bot_role": "infrastructure_sub_bot",
                "active": True,
                "data_collection_active": True,
                "training_excluded": True,
                "sleeve_profile": "convertible_bond_arbitrage",
                "capability_pack_slug": "quant_strategy_gap",
                "capability_pack_version": "quant_strategy_gap_v1",
                "capability_pack_display_name": "Quant Strategy Gap Pack",
                "capability_pack_contract": {
                    "display_name": "Quant Strategy Gap Pack",
                    "storage_retention_rule": {"retention_profile": "strategy_gap_hot_5d", "sample_rate": 0.03, "max_daily_mb_per_bot": 4},
                    "paper_only_floor": {"graduation_requires_minimum_observations": 45000, "graduation_requires_collection_days": 120},
                },
            },
            {
                "bot_id": "brain_refinery_v1207_strategy_gap_convertible_bond_arbitrage_signal_modeler_bot",
                "bot_role": "signal_sub_bot",
                "active": True,
                "data_collection_active": True,
                "training_excluded": True,
                "sleeve_profile": "convertible_bond_arbitrage",
                "capability_pack_slug": "quant_strategy_gap",
                "capability_pack_version": "quant_strategy_gap_v1",
                "capability_pack_display_name": "Quant Strategy Gap Pack",
            },
        ]
    }
    (tmp_path / "master_bot_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "quant_strategy_gap_v1.json").write_text(
        json.dumps(
            {
                "quant_strategy_gap_version": "quant_strategy_gap_v1",
                "pack": {
                    "display_name": "Quant Strategy Gap Pack",
                    "strategies": [{"display_name": "Convertible Bond Arbitrage"}],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = expansion_report.build_report(tmp_path)
    markdown = expansion_report._render_markdown(payload)

    assert payload["summary"]["registry_expansion_pack_count"] == 1
    assert payload["summary"]["registry_expansion_pack_bot_count"] == 2
    assert payload["summary"]["quant_strategy_gap_strategy_count"] == 1
    assert payload["registry_expansion_packs"][0]["slug"] == "quant_strategy_gap"
    assert "Quant Strategy Gap Pack" in markdown
    assert "Convertible Bond Arbitrage" in markdown


def test_write_report_creates_markdown_json_without_pdf(tmp_path: Path) -> None:
    (tmp_path / "master_bot_registry.json").write_text(json.dumps({"sub_bots": []}), encoding="utf-8")
    (tmp_path / "config").mkdir()

    payload = expansion_report.write_report(tmp_path, render_pdf=False)

    assert Path(payload["artifact_paths"]["markdown"]).exists()
    assert Path(payload["artifact_paths"]["json"]).exists()
    assert payload["pdf"]["ok"] is False
