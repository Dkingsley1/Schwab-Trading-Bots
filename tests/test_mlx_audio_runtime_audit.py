from scripts.ops import mlx_audio_runtime_audit as src


def test_package_rows_flag_missing_audio_runtime_packages() -> None:
    rows, ok = src._package_rows(
        ("mlx-audio", "mlx", "mlx-lm", "transformers", "miniaudio"),
        {
            "mlx-audio": "0.4.0",
            "mlx": "0.31.1",
            "transformers": "5.0.0rc3",
        },
    )

    assert ok is False
    assert rows == [
        {"package": "mlx-audio", "installed_version": "0.4.0", "status": "ok"},
        {"package": "mlx", "installed_version": "0.31.1", "status": "ok"},
        {"package": "mlx-lm", "installed_version": None, "status": "missing_runtime"},
        {"package": "transformers", "installed_version": "5.0.0rc3", "status": "ok"},
        {"package": "miniaudio", "installed_version": None, "status": "missing_runtime"},
    ]
