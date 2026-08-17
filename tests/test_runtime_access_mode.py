import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import runtime_access_mode as src


def test_override_lines_for_portable_contains_expected_flags() -> None:
    lines = src.override_lines_for_mode("portable")

    assert "BOT_RUNTIME_ACCESS_MODE=portable" in lines
    assert "BOT_LOGS_PREFER_EXTERNAL=0" in lines
    assert "BOT_SQL_ACCESS_PORTABLE=1" in lines
    assert "BOT_ML_BACKEND=portable_auto" in lines
    assert "BOT_ML_RUNTIME_OPTIONAL=1" in lines
    assert "BOT_MLX_OPTIONAL=1" in lines
    assert "RUNTIME_TRAIN_USE_SNAPSHOT=1" in lines
    assert "RUNTIME_TRAIN_PREFER_SQLITE=1" in lines


def test_override_lines_for_portable_can_pin_specific_backend() -> None:
    lines = src.override_lines_for_mode("portable", "onnx")

    assert "BOT_ML_BACKEND=onnx" in lines
    assert "BOT_ML_RUNTIME_OPTIONAL=1" in lines


def test_write_override_round_trip_portable(tmp_path) -> None:
    override_path = tmp_path / ".env.access_mode_override"

    changed = src._write_override(override_path, "portable")

    assert changed is True
    assert override_path.exists()
    assert src._parse_override_mode(override_path) == "portable"
    payload = src.build_payload("portable", override_path, changed=changed, action="set")
    assert payload["portable_enabled"] is True
    assert payload["ml_backend"] == "portable_auto"
    assert payload["runtime_flags"]["BOT_LOGS_PREFER_EXTERNAL"] == "0"
    assert payload["runtime_flags"]["BOT_ML_RUNTIME_OPTIONAL"] == "1"
    assert payload["runtime_flags"]["RUNTIME_TRAIN_USE_SNAPSHOT"] == "1"


def test_write_override_native_removes_file(tmp_path) -> None:
    override_path = tmp_path / ".env.access_mode_override"
    override_path.write_text("\n".join(src.override_lines_for_mode("portable")) + "\n", encoding="utf-8")

    changed = src._write_override(override_path, "native")

    assert changed is True
    assert not override_path.exists()
    payload = src.build_payload("native", override_path, changed=changed, action="set")
    assert payload["portable_enabled"] is False
    assert payload["override_exists"] is False


def test_effective_mode_prefers_environment_when_override_absent(monkeypatch, tmp_path) -> None:
    override_path = tmp_path / ".env.access_mode_override"

    monkeypatch.setenv("BOT_RUNTIME_ACCESS_MODE", "portable")
    monkeypatch.setenv("BOT_ML_BACKEND", "onnx")

    mode, mode_source = src._effective_mode(override_path)
    effective_mode, backend, settings_source = src._effective_settings(override_path)

    assert mode == "portable"
    assert mode_source == "environment"
    assert effective_mode == "portable"
    assert backend == "onnx"
    assert settings_source == "environment"


def test_build_payload_includes_backend_contract(monkeypatch, tmp_path) -> None:
    override_path = tmp_path / ".env.access_mode_override"

    monkeypatch.setattr(src, "detect_installed_backends", lambda: {"mlx": True, "pytorch": True, "onnx": False, "tensorflow": False, "jax": False})
    monkeypatch.setattr(
        src,
        "resolve_backend_contract",
        lambda backend, mode=None: {
            "mode": mode,
            "requested_backend": backend,
            "effective_backend": "pytorch",
            "roles_supported": ["shadow_replay", "sidecar_canary"],
            "observation_only": True,
        },
    )

    payload = src.build_payload("portable", override_path, ml_backend="portable_auto", changed=False, action="status")

    assert payload["detected_backends"]["pytorch"] is True
    assert payload["backend_contract"]["effective_backend"] == "pytorch"
    assert payload["backend_contract"]["observation_only"] is True
