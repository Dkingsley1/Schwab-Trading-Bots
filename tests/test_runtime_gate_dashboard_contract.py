import importlib.util
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/runtime_gate_dashboard.py")
spec = importlib.util.spec_from_file_location("runtime_gate_dashboard_contract", MODULE_PATH)
runtime_gate_dashboard = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runtime_gate_dashboard)


def test_runtime_gate_dashboard_marks_missing_sections_with_explicit_contract_state(tmp_path):
    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["runtime"]["artifact_status"] == "missing"
    assert payload["runtime"]["artifact_reason"] == "artifact_missing"
    assert payload["runtime"]["mode"] == "unknown"
    assert payload["apple_silicon"]["artifact_status"] == "missing"
    assert payload["memory"]["artifact_status"] == "missing"
    assert payload["training"]["artifact_status"] == "missing"
    assert payload["platform"]["artifact_status"] == "missing"
