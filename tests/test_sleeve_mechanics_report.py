from __future__ import annotations

from pathlib import Path

from scripts.ops import sleeve_mechanics_report as sleeve_mechanics


def test_sleeve_mechanics_report_explains_launch_and_candidates() -> None:
    payload = sleeve_mechanics.build_report()
    summary = payload["summary"]
    steps = {str(row["step"]) for row in payload["how_sleeves_work"]}
    sleeves = {str(row.get("name")): row for row in payload["sleeves"]}

    assert summary["manifest_sleeve_count"] >= 100
    assert summary["specialized_launcher_profile_count"] >= 90
    assert summary["all_sleeves_profile_count"] == summary["specialized_launcher_profile_count"]
    assert summary["launcher_ready_count"] >= 90
    assert {"manifest", "launcher defaults", "wrapper", "gates"} <= steps

    alpha_research_os = sleeves["alpha_research_os"]
    assert alpha_research_os["launcher_ready"] is True
    assert alpha_research_os["execution_posture"] == "market_data_only_no_order_execution"
    assert sleeves["collateral_margin_liquidity"]["source_profile"]

    candidate_names = {str(row.get("name")) for row in payload["expansion_candidates"]}
    assert "convertible_bond_arbitrage" in candidate_names
    assert "intraday_momentum_muscle" in candidate_names
    assert "volatility_risk_premium_harvesting" in candidate_names
    assert "cliquet_ratchet_options" in candidate_names
    assert "bermudan_exercise_monte_carlo_policy" in candidate_names


def test_write_sleeve_mechanics_report_creates_artifacts(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "sleeve_mechanics"
    health_path = tmp_path / "health" / "sleeve_mechanics_latest.json"
    monkeypatch.setattr(sleeve_mechanics, "OUT_DIR", out_dir)
    monkeypatch.setattr(sleeve_mechanics, "MD_PATH", out_dir / "sleeve_mechanics_latest.md")
    monkeypatch.setattr(sleeve_mechanics, "HEALTH_PATH", health_path)

    payload = sleeve_mechanics.write_report()

    assert health_path.exists()
    assert sleeve_mechanics.MD_PATH.exists()
    text = sleeve_mechanics.MD_PATH.read_text(encoding="utf-8")
    assert "# Sleeve Mechanics" in text
    assert "## How Sleeves Work" in text
    assert payload["artifact_paths"]["json"].endswith("sleeve_mechanics_latest.json")
