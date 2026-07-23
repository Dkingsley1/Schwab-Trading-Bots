from __future__ import annotations

import errno
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ops" / "system_self_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("system_self_model", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load system_self_model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_report_write_falls_back_on_no_space(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "exports" / "reports" / "operator" / "system_self_model_latest.md"
    fallback = tmp_path / "local_fallback_storage" / "exports" / "reports" / "operator" / "system_self_model_latest.md"
    original_write_text = Path.write_text

    def fake_write_text(self: Path, text: str, *args, **kwargs):
        if self == target:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    result = module._write_text_with_local_fallback(target, "fallback works\n")

    assert result["storage_mode"] == "local_fallback"
    assert result["path"] == str(fallback)
    assert result["primary_path"] == str(target)
    assert fallback.read_text(encoding="utf-8") == "fallback works\n"
