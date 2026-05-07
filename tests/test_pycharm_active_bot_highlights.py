from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import pycharm_active_bot_highlights as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_pycharm_active_bot_highlights_writes_scope_and_file_color(tmp_path: Path) -> None:
    project_root = tmp_path / "schwab_trading_bot"
    core = project_root / "core"
    core.mkdir(parents=True)
    (core / "brain_refinery_v1_alpha.py").write_text("# active\n", encoding="utf-8")
    (core / "brain_refinery_v2_beta.py").write_text("# inactive\n", encoding="utf-8")
    (core / "brain_refinery_v3_gamma.py").write_text("# active\n", encoding="utf-8")
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v1_alpha", "active": True},
                {"bot_id": "brain_refinery_v2_beta", "active": False},
                {"bot_id": "brain_refinery_v3_gamma", "active": True},
                {"bot_id": "brain_refinery_v4_missing", "active": True},
            ]
        },
    )

    payload = src.build_payload(project_root, apply=True)

    assert payload["overall_status"] == "ready"
    assert payload["active_core_bot_file_count"] == 2
    assert payload["inactive_core_bot_file_count"] == 1
    assert payload["missing_active_core_file_count"] == 1
    assert payload["scope_strategy"] == "brain_refinery_family_with_inactive_exclusions"
    assert payload["project_view_style"] == "scope_background_color"
    assert payload["foreground_blue_source"] == "pycharm_vcs_modified_file_status"
    assert payload["foreground_blue_supported_without_dirtying_files"] is False
    assert payload["vcs_blue_requires_file_modification"] is True
    assert payload["vcs_blue_dirty_file_count_to_match"] == 2
    scope_text = (project_root / ".idea" / "scopes" / "Active_Bots.xml").read_text(encoding="utf-8")
    assert "Active Bots" in scope_text
    assert "file[schwab_trading_bot]:core/brain_refinery_v*.py" in scope_text
    assert "!file[schwab_trading_bot]:core/brain_refinery_v2_beta.py" in scope_text
    file_colors = (project_root / ".idea" / "fileColors.xml").read_text(encoding="utf-8")
    assert 'name="enabledForProjectView" value="true"' in file_colors
    assert 'name="fileColorsEnabledForProjectView" value="true"' in file_colors
    assert 'scope="Active Bots"' in file_colors
    assert 'color="Blue"' in file_colors
    workspace = (project_root / ".idea" / "workspace.xml").read_text(encoding="utf-8")
    assert '<component name="FileColors">' in workspace
    assert '<fileColor scope="Active Bots" color="Blue" />' in workspace
