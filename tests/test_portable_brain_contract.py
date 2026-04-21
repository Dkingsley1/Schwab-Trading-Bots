import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import portable_brain_contract as src


def test_portable_brain_contract_prefers_native_on_apple_silicon(monkeypatch, tmp_path: Path) -> None:
    hardware = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "processor": "arm",
        "chip": "Apple M5 Max",
        "memory_gb": 64.0,
        "cpu_count": 16,
        "is_apple_silicon": True,
        "accelerator_hint": "metal",
        "recognized_host_and_chip": True,
    }
    monkeypatch.setattr(src, "detect_installed_backends", lambda: {"mlx": True, "onnx": False, "pytorch": True, "tensorflow": False, "jax": False})

    payload = src.build_payload(
        hardware=hardware,
        profile=src.detect_host_profile(hardware),
        override_path=tmp_path / ".env.host_profile_override",
        changed=False,
        action="status",
    )

    assert payload["host_contract"]["host_profile"] == "max_throughput"
    assert payload["host_contract"]["recommended_runtime_access_mode"] == "native"
    assert payload["recommended_runtime_mode"] == "native"
    assert payload["recommended_backend"] == "native_default"
    assert payload["native_contract"]["live_trading_supported"] is True
    assert payload["cross_platform_proof_node"]["shadow_replay_supported"] is True
    assert payload["parity_contract"]["parity_focus"] == "mlx_vs_portable_replay"
    assert payload["nightly_proof_contract"]["ready"] is True


def test_portable_brain_contract_prefers_portable_profile_on_linux(monkeypatch, tmp_path: Path) -> None:
    hardware = {
        "system": "Linux",
        "release": "6.8.0",
        "machine": "x86_64",
        "processor": "x86_64",
        "chip": "AMD Ryzen Workstation",
        "memory_gb": 96.0,
        "cpu_count": 24,
        "is_apple_silicon": False,
        "accelerator_hint": "cuda",
        "recognized_host_and_chip": True,
    }
    monkeypatch.setattr(src, "detect_installed_backends", lambda: {"mlx": False, "onnx": True, "pytorch": True, "tensorflow": False, "jax": False})

    payload = src.build_payload(
        hardware=hardware,
        profile=src.detect_host_profile(hardware),
        override_path=tmp_path / ".env.host_profile_override",
        changed=False,
        action="status",
    )

    assert payload["host_contract"]["host_profile"] == "portable_throughput"
    assert payload["host_contract"]["recommended_runtime_access_mode"] == "portable"
    assert payload["recommended_runtime_mode"] == "portable"
    assert payload["recommended_backend"] == "portable_auto"
    assert payload["adaptation_contract"]["env_override_count"] > 0
    assert payload["cross_platform_proof_node"]["status"] in {"ready", "active_host_candidate"}
    assert payload["parity_contract"]["nightly_proof_supported"] is True
    assert "backend_parity_report" in payload["nightly_proof_contract"]["report_paths"]
