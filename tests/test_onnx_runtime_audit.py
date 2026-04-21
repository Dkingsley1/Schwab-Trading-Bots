from scripts.ops import onnx_runtime_audit as src


def test_package_rows_flag_missing_onnx_runtime_packages() -> None:
    rows, ok = src._package_rows(
        ("onnx", "onnxruntime", "ml-dtypes", "flatbuffers"),
        {
            "onnx": "1.20.1",
            "onnxruntime": "1.24.3",
            "flatbuffers": "25.12.19",
        },
    )

    assert ok is False
    assert rows == [
        {"package": "onnx", "installed_version": "1.20.1", "status": "ok"},
        {"package": "onnxruntime", "installed_version": "1.24.3", "status": "ok"},
        {"package": "ml-dtypes", "installed_version": None, "status": "missing_runtime"},
        {"package": "flatbuffers", "installed_version": "25.12.19", "status": "ok"},
    ]
