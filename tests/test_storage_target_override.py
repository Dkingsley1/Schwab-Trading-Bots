import shutil
import subprocess

import pytest

from core.storage_target_override import (
    build_storage_target_override_text,
    write_storage_target_override,
)


@pytest.mark.parametrize("shell", ["sh", "zsh"])
@pytest.mark.parametrize(
    "name", ["BOT_LOGS", "Extreme SSD", "Dan's $HOME `false` $(false) SSD"]
)
def test_target_values_round_trip_without_shell_evaluation(tmp_path, shell, name):
    executable = shutil.which(shell)
    if not executable:
        pytest.skip(f"{shell} unavailable")
    mount = f"/Volumes/{name}"
    project = "platform data"
    target = tmp_path / "target.env"
    target.write_text(
        build_storage_target_override_text(
            mount_root=mount,
            project_dir=project,
            mount_candidates=[mount, "/Volumes/Alternate SSD"],
            volume_uuid="test-uuid",
            disk_identifier="disk5s1",
        )
    )
    keys = [
        "MOUNT",
        "MOUNT_CANDIDATES",
        "VOLUME_NAME",
        "PROJECT_DIR",
        "PROJECT_ROOT",
        "VOLUME_UUID",
        "DISK_IDENTIFIER",
    ]
    command = 'set -eu\n. "$1"\n' + "\n".join(
        f'printf "%s\\0" "$BOT_LOGS_EXTERNAL_{key}"' for key in keys
    )
    result = subprocess.run(
        [executable, "-c", command, "target-test", str(target)],
        capture_output=True,
        check=True,
        timeout=10,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.stdout.decode().split("\0")[:-1] == [
        mount,
        f"{mount},/Volumes/Alternate SSD",
        name,
        project,
        f"{mount}/{project}",
        "test-uuid",
        "disk5s1",
    ]
    assert not result.stderr


def test_legacy_simple_assignment_format_unchanged():
    text = build_storage_target_override_text(mount_root="/Volumes/BOT_LOGS")
    assert "BOT_LOGS_EXTERNAL_MOUNT=/Volumes/BOT_LOGS\n" in text
    assert (
        "BOT_LOGS_EXTERNAL_PROJECT_ROOT=/Volumes/BOT_LOGS/schwab_trading_bot\n" in text
    )


def test_spaced_target_write_is_idempotent(tmp_path):
    path = tmp_path / "target.env"
    args = dict(mount_root="/Volumes/Extreme SSD", override_path=path)
    assert write_storage_target_override(**args)["changed"]
    before = path.stat().st_mtime_ns
    assert not write_storage_target_override(**args)["changed"]
    assert path.stat().st_mtime_ns == before


def test_restricted_profile_is_explicit_and_persisted(tmp_path):
    path = tmp_path / "target.env"
    args = dict(
        mount_root="/Volumes/Extreme SSD",
        route_profile="sqlite_primary",
        override_path=path,
    )
    assert write_storage_target_override(**args)["changed"]
    assert "BOT_STORAGE_ROUTE_PROFILE=sqlite_primary\n" in path.read_text()
    assert not write_storage_target_override(**args)["changed"]
    assert "BOT_STORAGE_ROUTE_PROFILE" not in build_storage_target_override_text(
        mount_root="/Volumes/BOT_LOGS"
    )
    with pytest.raises(ValueError, match="unsupported"):
        build_storage_target_override_text(
            mount_root="/Volumes/Extreme SSD", route_profile="typo"
        )
