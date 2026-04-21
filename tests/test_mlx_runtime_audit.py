from scripts.ops import mlx_runtime_audit as src


def test_package_rows_detect_drift_states() -> None:
    rows, ok = src._package_rows(
        ("mlx", "mlx-data", "mlx-vlm", "mlx-whisper", "transformers", "schwab-py", "duckdb"),
        {
            "mlx": "0.30.6",
            "mlx-data": "0.2.0",
            "mlx-vlm": "0.4.4",
            "transformers": "5.3.0",
            "schwab-py": "1.5.1",
        },
        {
            "mlx": "0.31.0",
            "mlx-data": "0.2.0",
            "mlx-vlm": "0.4.4",
            "mlx-whisper": "0.4.3",
            "transformers": "5.3.0",
            "duckdb": "1.5.0",
        },
    )

    assert ok is False
    assert rows == [
        {
            "package": "mlx",
            "locked_version": "0.30.6",
            "installed_version": "0.31.0",
            "status": "version_mismatch",
        },
        {
            "package": "mlx-data",
            "locked_version": "0.2.0",
            "installed_version": "0.2.0",
            "status": "ok",
        },
        {
            "package": "mlx-vlm",
            "locked_version": "0.4.4",
            "installed_version": "0.4.4",
            "status": "ok",
        },
        {
            "package": "mlx-whisper",
            "locked_version": None,
            "installed_version": "0.4.3",
            "status": "missing_lock",
        },
        {
            "package": "transformers",
            "locked_version": "5.3.0",
            "installed_version": "5.3.0",
            "status": "ok",
        },
        {
            "package": "schwab-py",
            "locked_version": "1.5.1",
            "installed_version": None,
            "status": "missing_runtime",
        },
        {
            "package": "duckdb",
            "locked_version": None,
            "installed_version": "1.5.0",
            "status": "missing_lock",
        },
    ]


def test_recommendations_highlight_compile_canary_when_safe() -> None:
    recommendations = src._recommendations(
        [
            {
                "package": "mlx",
                "locked_version": "0.31.0",
                "installed_version": "0.31.0",
                "status": "ok",
            }
        ],
        {
            "compile_available": True,
            "compile_smoke_ok": True,
            "metal_available": True,
            "jit_env": "0",
        },
    )

    assert "candidate_mlx_compile_canary_before_enabling_training_jit" in recommendations
    assert "mlx_metal_jit_default_off" in recommendations


def test_recommendations_hold_compile_rollout_on_failed_smoke() -> None:
    recommendations = src._recommendations(
        [],
        {
            "compile_available": True,
            "compile_smoke_ok": False,
            "metal_available": True,
            "jit_env": "0",
        },
    )

    assert recommendations == [
        "keep_mlx_compile_opt_in_until_compile_smoke_passes",
        "mlx_metal_jit_default_off",
    ]
