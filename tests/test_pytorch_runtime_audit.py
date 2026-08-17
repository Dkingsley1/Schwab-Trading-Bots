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


def test_recommendations_keep_pytorch_manual_when_mps_ready() -> None:
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

    assert "pytorch_runtime_available_for_manual_offline_replay_only" in recommendations
    assert "keep_torch_compile_off_for_canary" in recommendations
    assert "keep_mlx_default_live_backend_on_apple_silicon" in recommendations
    assert "keep_pytorch_replay_canary_disabled_during_live_mlx_collection" in recommendations


def test_pip_check_tolerates_mlx_graphs_optional_pin_override() -> None:
    assert src._pip_check_effectively_ok(
        {
            "ok": False,
            "stdout_tail": "\n".join(
                [
                    "mlx-graphs 0.0.9 has requirement fsspec==2024.2.0, but you have fsspec 2026.2.0.",
                    "mlx-graphs 0.0.9 has requirement requests==2.31.0, but you have requests 2.32.5.",
                    "mlx-graphs 0.0.9 has requirement tqdm==4.66.1, but you have tqdm 4.67.3.",
                ]
            ),
            "stderr_tail": "",
        }
    )
