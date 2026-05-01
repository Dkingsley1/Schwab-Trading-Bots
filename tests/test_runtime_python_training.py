from pathlib import Path

from core import runtime_python as src


def test_resolve_runtime_python_prefers_mlx_capable_env_in_auto_mode(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    py312 = root / ".venv312" / "bin" / "python"
    py314 = root / ".venv314" / "bin" / "python"
    py312.parent.mkdir(parents=True, exist_ok=True)
    py314.parent.mkdir(parents=True, exist_ok=True)
    py312.write_text("", encoding="utf-8")
    py314.write_text("", encoding="utf-8")

    monkeypatch.delenv("BOT_PYTHON_BIN", raising=False)
    monkeypatch.delenv("BOT_PYTHON_VERSION", raising=False)
    monkeypatch.delenv("BOT_RUNTIME_LANE", raising=False)
    monkeypatch.delenv("BOT_PYTHON_RUNTIME", raising=False)
    monkeypatch.delenv("BOT_PREFER_MLX_RUNTIME", raising=False)
    monkeypatch.setattr(src, "_python_supports_module", lambda path_text, module_name: path_text == str(py312) and module_name == "mlx")

    resolved = src.resolve_runtime_python(root)

    assert resolved == py312


def test_resolve_runtime_python_falls_back_to_portable_env_without_mlx(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    py312 = root / ".venv312" / "bin" / "python"
    py314 = root / ".venv314" / "bin" / "python"
    py312.parent.mkdir(parents=True, exist_ok=True)
    py314.parent.mkdir(parents=True, exist_ok=True)
    py312.write_text("", encoding="utf-8")
    py314.write_text("", encoding="utf-8")

    monkeypatch.delenv("BOT_PYTHON_BIN", raising=False)
    monkeypatch.delenv("BOT_PYTHON_VERSION", raising=False)
    monkeypatch.delenv("BOT_RUNTIME_LANE", raising=False)
    monkeypatch.delenv("BOT_PYTHON_RUNTIME", raising=False)
    monkeypatch.delenv("BOT_PREFER_MLX_RUNTIME", raising=False)
    monkeypatch.setattr(src, "_python_supports_module", lambda path_text, module_name: False)

    resolved = src.resolve_runtime_python(root)

    assert resolved == py314


def test_resolve_training_python_prefers_mlx_capable_env(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    py312 = root / ".venv312" / "bin" / "python"
    py314 = root / ".venv314" / "bin" / "python"
    py312.parent.mkdir(parents=True, exist_ok=True)
    py314.parent.mkdir(parents=True, exist_ok=True)
    py312.write_text("", encoding="utf-8")
    py314.write_text("", encoding="utf-8")

    monkeypatch.delenv("BOT_TRAINING_PYTHON_BIN", raising=False)
    monkeypatch.delenv("BOT_TRAINING_PYTHON_VERSION", raising=False)
    monkeypatch.setattr(src, "_python_supports_module", lambda path_text, module_name: path_text == str(py312))

    resolved = src.resolve_training_python(root)

    assert resolved == py312


def test_resolve_training_python_falls_back_when_no_env_has_mlx(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    py312 = root / ".venv312" / "bin" / "python"
    py314 = root / ".venv314" / "bin" / "python"
    py312.parent.mkdir(parents=True, exist_ok=True)
    py314.parent.mkdir(parents=True, exist_ok=True)
    py312.write_text("", encoding="utf-8")
    py314.write_text("", encoding="utf-8")

    monkeypatch.delenv("BOT_TRAINING_PYTHON_BIN", raising=False)
    monkeypatch.delenv("BOT_TRAINING_PYTHON_VERSION", raising=False)
    monkeypatch.setattr(src, "_python_supports_module", lambda path_text, module_name: False)

    resolved = src.resolve_training_python(root)

    assert resolved == py312
