import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import pytorch_runtime_audit as src


def test_package_rows_detect_pytorch_lock_drift() -> None:
    rows, ok = src._package_rows(
        ("torch",),
        {"torch": "2.10.0"},
        {"torch": "2.9.1"},
    )

    assert ok is False
    assert rows == [
        {
            "package": "torch",
            "locked_version": "2.10.0",
            "installed_version": "2.9.1",
            "status": "version_mismatch",
        }
    ]


def test_recommendations_prefer_shadow_canary_when_mps_ready() -> None:
    recommendations = src._recommendations(
        [
            {
                "package": "torch",
                "locked_version": "2.10.0",
                "installed_version": "2.10.0",
                "status": "ok",
            }
        ],
        {
            "mps_built": True,
            "mps_available": True,
            "tensor_smoke_ok": True,
            "compile_available": True,
            "compile_smoke_ok": False,
            "selected_device": "mps",
        },
    )

    assert "candidate_pytorch_shadow_sidecar_on_mps" in recommendations
    assert "keep_torch_compile_off_for_canary" in recommendations
    assert "keep_mlx_default_live_backend_on_apple_silicon" in recommendations
    assert "pytorch_canary_is_sidecar_only_until_trading_brain_backend_exists" in recommendations
