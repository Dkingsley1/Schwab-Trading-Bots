from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import macro_crosscheck_report as mcr


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_build_macro_crosscheck_payload_passes_when_overlap_checks_match(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "exports" / "external_feeds" / "latest_status.json",
        {
            "timestamp_utc": "2026-03-20T17:58:23+00:00",
            "bls": {"ok": True},
            "bea": {"ok": True},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {
            "timestamp_utc": "2026-03-20T21:40:22+00:00",
            "sources": {
                "bls": {"ok": True, "fallback": "html_page_parse"},
                "bls_calendar": {"ok": True},
                "bea": {"ok": True},
                "treasury": {"ok": True, "rows": 14, "fallback": "html_page_parse"},
                "federal_reserve_calendar": {"ok": True, "error": "2026-04:HTTP Error 404: Not Found"},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {
            "timestamp_utc": "2026-03-20T17:59:02+00:00",
            "sources": {
                "treasury_auctions": {"ok": True, "rows": 12},
            },
        },
    )

    payload = mcr.build_macro_crosscheck_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["passed_checks"] == payload["total_checks"] == 4
    assert "official_treasury_fallback=html_page_parse" in payload["notes"]
    assert "official_bls_fallback=html_page_parse" in payload["notes"]
    assert "fed_calendar_partial_error" in payload["notes"]
    assert payload["checks"]["bls_dual_source"]["ok"] is True
    assert payload["checks"]["bea_dual_source"]["ok"] is True
    assert payload["checks"]["treasury_dual_source"]["ok"] is True


def test_build_macro_crosscheck_payload_fails_when_overlap_missing(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "exports" / "external_feeds" / "latest_status.json",
        {"timestamp_utc": "2026-03-20T17:58:23+00:00", "bls": {"ok": False}, "bea": {"ok": False}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "official_macro_context_sync_latest.json",
        {"timestamp_utc": "2026-03-20T21:40:22+00:00", "sources": {"treasury": {"ok": False}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "market_micro_sync_latest.json",
        {"timestamp_utc": "2026-03-20T17:59:02+00:00", "sources": {"treasury_auctions": {"ok": False}}},
    )

    payload = mcr.build_macro_crosscheck_payload(tmp_path)

    assert payload["ok"] is False
    assert payload["passed_checks"] < payload["total_checks"]
