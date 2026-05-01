from scripts.ops import strategy_inventory_report as src


def test_strategy_inventory_report_includes_advanced_sleeves_and_pdf(monkeypatch, tmp_path) -> None:
    out_dir = tmp_path / "reports" / "strategy_inventory"
    health_path = tmp_path / "health" / "strategy_inventory_latest.json"
    monkeypatch.setattr(src, "OUT_DIR", out_dir)
    monkeypatch.setattr(src, "MD_PATH", out_dir / "strategy_inventory_latest.md")
    monkeypatch.setattr(src, "PDF_PATH", out_dir / "strategy_inventory_latest.pdf")
    monkeypatch.setattr(src, "HEALTH_PATH", health_path)

    payload = src.write_report(render_pdf=True)
    sleeves = {str(row.get("name")): row for row in payload["sleeves"]}

    assert payload["strategy_count"] >= 100
    assert "gamma_scalping" in sleeves
    assert "volatility_arbitrage" in sleeves
    assert "cdo_squared" in sleeves
    assert "strategy_inventory_latest.pdf" in str(payload["pdf"]["pdf_path"])
    assert src.MD_PATH.exists()
    assert src.PDF_PATH.exists()
    assert health_path.exists()
