from __future__ import annotations

import json
from pathlib import Path

from scripts import run_all_sleeves as src


def test_position_overlay_adds_valid_held_underlyings_without_duplicates(tmp_path: Path) -> None:
    study_path = tmp_path / "study.json"
    study_path.write_text(
        json.dumps(
            {
                "ok": True,
                "underlyings": [
                    {"underlying": "NVDA"},
                    {"underlying": "BRK.B"},
                    {"underlying": "NVDA"},
                    {"underlying": "/ES"},
                    {"underlying": "BAD SYMBOL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    overlay = src._position_overlay_symbols(study_path)
    merged = src._merge_symbol_csv("SPY,NVDA", overlay)

    assert overlay == ["NVDA", "BRK.B"]
    assert merged == "SPY,NVDA,BRK.B"
