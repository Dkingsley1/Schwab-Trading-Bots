from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ops" / "schwab_indicator_intelligence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("schwab_indicator_intelligence", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load schwab_indicator_intelligence")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_parse_group_links_filters_to_official_catalog_items() -> None:
    module = _load_module()
    html = """
    <html><body>
      <a href="/center/reference/Tech-Indicators/studies-library">Studies Library</a>
      <a href="/center/reference/Tech-Indicators/studies-library/A-B/ADX">ADX</a>
      <a href="/center/reference/Tech-Indicators/studies-library/A-B/ATR">ATR</a>
      <a href="/center/reference/Tech-Indicators/strategies/A-D/ADXTrend">ADXTrend</a>
    </body></html>
    """

    rows = module._parse_group_links("study", "A-B", html)

    assert [row["name"] for row in rows] == ["ADX", "ATR"]
    assert all(row["kind"] == "study" for row in rows)
    assert all("/studies-library/A-B/" in row["url"] for row in rows)


def test_offline_payload_builds_partial_catalog_with_sleeve_routes(tmp_path: Path) -> None:
    module = _load_module()
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "bot_intraday", "active": True, "sleeve_profile": "intraday_aggressive"},
                {"bot_id": "bot_options", "active": True, "sleeve_profile": "options_income"},
            ]
        },
    )
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(tmp_path / "governance" / "health" / "health_fast_latest.json", {"ok": True, "overall_status": "ready"})

    payload = module.build_payload(tmp_path, offline=True)

    assert payload["overall_status"] == "schwab_indicator_intelligence_partial_catalog"
    assert payload["coverage"]["used_fallback_seed"] is True
    assert payload["coverage"]["study_count"] >= 1
    assert payload["coverage"]["strategy_count"] >= 1
    assert payload["routing_contract"]["live_execution_authority"] is False
    sleeves = {row["sleeve"]: row for row in payload["sleeve_applicability_matrix"]}
    assert sleeves["intraday_aggressive"]["mapped_item_count"] > 0
    assert "advisory_feature_routing_no_execution_authority" == sleeves["intraday_aggressive"]["authority"]


def test_official_payload_classifies_strategy_and_circumstances(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    rows = [
        {"kind": "study", "name": f"ADX{i}", "group": "A-B", "url": f"https://example.test/ADX{i}"}
        for i in range(50)
    ] + [
        {"kind": "strategy", "name": f"BollingerBandsLE{i}", "group": "A-D", "url": f"https://example.test/BollingerBandsLE{i}"}
        for i in range(50)
    ]

    def fake_official_catalog(*, offline: bool = False, timeout: int = 20, retry_count: int = 1):
        return rows, {
            "mode": "official_fetch",
            "source_urls": ["https://toslc.thinkorswim.com/center/reference/Tech-Indicators/studies-library"],
            "groups_required": 14,
            "groups_fetched": 14,
            "groups_failed": 0,
            "failures": [],
        }

    monkeypatch.setattr(module, "_official_catalog", fake_official_catalog)
    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "schwab_indicator_intelligence_ready"
    assert payload["coverage"]["official_fetch_complete"] is True
    assert payload["coverage"]["catalog_item_count"] == 100
    strategy = next(item for item in payload["catalog_items"] if item["kind"] == "strategy")
    assert "strategy_signal" in strategy["families"]
    assert strategy["mechanism_summary"]
    assert "paper_validation_evidence" in strategy["required_inputs"]
    assert "never_promotes_live_authority_by_itself" in strategy["circumstance_triggers"]
    assert "strategy_template_not_execution_permission" in strategy["risk_notes"]
