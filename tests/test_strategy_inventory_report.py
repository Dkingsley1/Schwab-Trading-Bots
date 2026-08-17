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
    assert "institutional_data_plumbing" in sleeves
    assert "lobdif_crisis_microstructure" in sleeves
    assert "macro_crisis_scenario_lab" in sleeves
    assert "xva_counterparty_margin" in sleeves
    assert "credit_derivatives_cdx_cds" in sleeves
    assert "securitized_products_mbs_abs_clo" in sleeves
    assert "repo_securities_lending" in sleeves
    assert "market_data_tape_normalization" in sleeves
    assert "provider_adapter_verification" in sleeves
    assert "proof_quantum_formal_backends" in sleeves
    assert "model_risk_validation" in sleeves
    assert "transaction_cost_slippage_intelligence" in sleeves
    assert "portfolio_construction" in sleeves
    assert "event_intelligence" in sleeves
    assert "feature_quality_data_confidence" in sleeves
    assert "liquidity_regime" in sleeves
    assert "system_governor_expansion" in sleeves
    assert "institutional_data_plumbing" in payload["advanced_collection_sleeves"]
    assert "lobdif_crisis_microstructure" in payload["advanced_collection_sleeves"]
    assert "macro_crisis_scenario_lab" in payload["advanced_collection_sleeves"]
    assert "provider_adapter_verification" in payload["advanced_collection_sleeves"]
    assert "system_governor_expansion" in payload["advanced_collection_sleeves"]
    assert "strategy_inventory_latest.pdf" in str(payload["pdf"]["pdf_path"])
    assert src.MD_PATH.exists()
    assert src.PDF_PATH.exists()
    assert health_path.exists()
