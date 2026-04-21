import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    module_name = f"test_loader_{path.stem}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shadow_launchers_default_to_python314_lane(monkeypatch) -> None:
    monkeypatch.delenv("BOT_RUNTIME_LANE", raising=False)
    monkeypatch.delenv("BOT_SHADOW_RUNTIME_LANE", raising=False)

    launcher_paths = [
        PROJECT_ROOT / "scripts" / "run_dividend_shadow.py",
        PROJECT_ROOT / "scripts" / "run_dividend_capture_shadow.py",
        PROJECT_ROOT / "scripts" / "run_dividend_compound_shadow.py",
        PROJECT_ROOT / "scripts" / "run_long_term_core_etf_shadow.py",
        PROJECT_ROOT / "scripts" / "run_long_term_dividend_compound_shadow.py",
        PROJECT_ROOT / "scripts" / "run_long_term_sector_rotation_shadow.py",
    ]

    for path in launcher_paths:
        module = _load_module(path)
        assert ".venv314/bin/python" in str(module.VENV_PY)

    failover_module = _load_module(PROJECT_ROOT / "scripts" / "failover_hot_standby.py")
    assert ".venv314/bin/python" in str(failover_module.RUNTIME_PY)


def test_shadow_launchers_honor_explicit_runtime_override(monkeypatch) -> None:
    monkeypatch.setenv("BOT_RUNTIME_LANE", "production")
    monkeypatch.delenv("BOT_SHADOW_RUNTIME_LANE", raising=False)

    module = _load_module(PROJECT_ROOT / "scripts" / "run_dividend_shadow.py")

    assert ".venv312/bin/python" in str(module.VENV_PY)
