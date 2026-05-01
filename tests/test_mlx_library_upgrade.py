from pathlib import Path

from scripts.ops import mlx_library_upgrade as src


def test_build_payload_uses_only_mlx_lock_rows(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "\n".join(
            [
                "mlx==0.31.1",
                "mlx-lm==0.31.2",
                "numpy==2.4.0",
                "mlx-embedding-models==0.0.11",
                "parakeet-mlx==0.5.1",
            ]
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(lock_path=lock, python_bin=Path("/venv/bin/python"))

    assert payload["ok"] is True
    assert payload["install_command"] == [
        "/venv/bin/python",
        "-m",
        "pip",
        "install",
        "-U",
        "mlx==0.31.1",
        "mlx-embedding-models==0.0.11",
        "mlx-lm==0.31.2",
        "parakeet-mlx==0.5.1",
    ]
    assert {row["package"] for row in payload["packages"]} == {
        "mlx",
        "mlx-lm",
        "mlx-embedding-models",
        "parakeet-mlx",
    }
