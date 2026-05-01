from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import build_core_bot_catalog as catalog_src


def test_core_bot_catalog_indexes_registry_and_ops_bots() -> None:
    payload = catalog_src.build_catalog(PROJECT_ROOT)

    summary = payload["summary"]
    bot_ids = {row["bot_id"] for row in payload["bots"]}

    assert summary["registry_total_bots"] >= 240
    assert summary["registry_data_collection_active"] >= 1
    assert "brain_refinery_v256_cross_sleeve_teacher_anchor_bot" in bot_ids
    assert "brain_refinery_v257_crypto_spot_momentum_regime_bot" in bot_ids
    assert "brain_refinery_v266_crypto_weekend_gap_liquidity_bot" in bot_ids
    assert "storage_reconnect_infrabot" in bot_ids
    assert "storage_reconnect_regression_guard" in bot_ids
    assert summary["total_indexed_rows"] == len(payload["bots"])

    rows = {row["bot_id"]: row for row in payload["bots"]}
    assert rows["brain_refinery_v257_crypto_spot_momentum_regime_bot"]["core_file"] == "core/brain_refinery_v257_crypto_spot_momentum_regime_bot.py"
    assert rows["brain_refinery_v266_crypto_weekend_gap_liquidity_bot"]["core_file"] == "core/brain_refinery_v266_crypto_weekend_gap_liquidity_bot.py"
    assert rows["brain_refinery_v257_crypto_spot_momentum_regime_bot"]["notes"] == "Physical core module found."


def test_core_bot_catalog_writes_core_files(tmp_path: Path) -> None:
    md_out = tmp_path / "BOT_CATALOG.md"
    json_out = tmp_path / "bot_catalog.json"
    payload = catalog_src.build_catalog(PROJECT_ROOT)

    md_out.write_text(catalog_src.render_markdown(payload), encoding="utf-8")
    json_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    assert "Core Bot Catalog" in md_out.read_text(encoding="utf-8")
    saved = json.loads(json_out.read_text(encoding="utf-8"))
    assert saved["summary"]["total_indexed_rows"] == len(saved["bots"])
