from pathlib import Path

from scripts.ops.library_efficiency_deepening import EFFICIENCY_LAYERS, build_payload


def test_library_efficiency_deepening_dual_mode_contracts(tmp_path: Path):
    packages = sorted(
        {
            package
            for layer in EFFICIENCY_LAYERS
            for package in layer["required_packages"]
        }
    )
    lock_path = tmp_path / "config" / "requirements.lock.txt"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("\n".join(f"{package}==1.0.0" for package in packages) + "\n", encoding="utf-8")

    payload = build_payload(tmp_path, lock_path=lock_path)

    assert payload["ok"] is True
    assert payload["layer_count"] == 10
    assert payload["backend_family_scope"] == ["mlx", "non_mlx"]
    assert payload["mode_scope"] == ["paper", "live"]
    assert payload["paper_contract_count"] == 10
    assert payload["live_contract_count"] == 10
    assert payload["required_package_coverage"] == 1.0
    assert payload["missing_required_packages"] == []
    assert payload["paper_mode"]["paper_execution_authority_enabled"] is False
    assert payload["live_mode"]["live_execution_authority_enabled"] is False
    for layer in payload["layers"]:
        assert layer["paper_contract"]["enabled"] is True
        assert layer["live_contract"]["enabled"] is True
        assert layer["paper_contract"]["paper_execution_authority_enabled"] is False
        assert layer["live_contract"]["live_execution_authority_enabled"] is False
        assert layer["paper_contract"]["uses_same_feature_contract_as_live"] is True
        assert layer["live_contract"]["uses_same_feature_contract_as_paper"] is True
