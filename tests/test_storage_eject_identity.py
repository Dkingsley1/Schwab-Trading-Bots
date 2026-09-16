from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/ops/storage_eject_guard.swift"


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS guard")
def test_native_volume_identity_contract(tmp_path):
    source = SOURCE.read_text().split("private let guardInstance =", 1)[0]
    harness = tmp_path / "identity.swift"
    harness.write_text(source + r"""
private func diskAppearedCallback(disk: DADisk, context: UnsafeMutableRawPointer?) {}
private func diskDisappearedCallback(disk: DADisk, context: UnsafeMutableRawPointer?) {}
private func diskUnmountApprovalCallback(disk: DADisk, context: UnsafeMutableRawPointer?) -> Unmanaged<DADissenter>? { nil }
private func diskEjectApprovalCallback(disk: DADisk, context: UnsafeMutableRawPointer?) -> Unmanaged<DADissenter>? { nil }
let expected = "E03167D1-54E8-4F8D-8039-9E5644F387BF"
func matches(_ uuid: String, _ name: String, _ path: String?, expectedUUID: String = expected) -> Bool {
    StorageEjectGuard.matchesVolumeIdentity(uuid: uuid, name: name, mountPath: path,
        expectedUUID: expectedUUID, expectedName: "BOT_LOGS", mountRoots: ["/Volumes/BOT_LOGS"])
}
precondition(matches(expected.lowercased(), "BOT_LOGS", "/Volumes/BOT_LOGS"))
precondition(matches(expected, "Renamed", "/Volumes/Renamed"))
precondition(matches(expected, "BOT_LOGS", nil))
precondition(!matches("wrong", "BOT_LOGS", "/Volumes/BOT_LOGS"))
precondition(!matches("", "BOT_LOGS", "/Volumes/BOT_LOGS"))
precondition(!matches("wrong", "LaCie", "/Volumes/LaCie"))
precondition(!matches("", "LaCie", "/Volumes/LaCie", expectedUUID: ""))
precondition(!matches("", "BOT_LOGS", "/Volumes/LaCie", expectedUUID: ""))
precondition(matches("", "BOT_LOGS", "/Volumes/BOT_LOGS", expectedUUID: ""))
let key = kDADiskDescriptionVolumeUUIDKey as String
let cfUUID = CFUUIDCreateFromString(kCFAllocatorDefault, expected as CFString)!
precondition(StorageEjectGuard.volumeUUID([key: cfUUID]) == expected)
precondition(StorageEjectGuard.volumeUUID([key: UUID(uuidString: expected)!]) == expected)
precondition(StorageEjectGuard.volumeUUID([key: "invalid"]) == "")
precondition(StorageEjectGuard.volumeUUID([:]) == "")
print("volume identity contracts passed")
""")
    binary = tmp_path / "identity"
    result = subprocess.run(
        [
            "/usr/bin/swiftc",
            "-module-cache-path",
            str(tmp_path / "cache"),
            str(harness),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_local_pin_veto_precedes_auto_failback_and_failed_switch_stops_cascade():
    source = SOURCE.read_text()
    decision = source.split("func shouldRestoreExternalOnAppear()", 1)[1].split(
        "func shouldRestartLocalOnDisappear", 1
    )[0]
    assert decision.index("guard externalPreferredByConfig()") < decision.index(
        "BOT_LOGS_AUTO_FAILBACK_ON_APPEAR"
    )
    preference = source.split("func externalPreferredByConfig()", 1)[1].split(
        "func startMountPollTimer", 1
    )[0]
    assert "if localOverrideActive()" in preference
    restore = source.split("func restoreExternalCollection()", 1)[1].split("func ", 1)[
        0
    ]
    failure = restore.index("guard switchRC == 0 else")
    assert failure < restore.index("feed-refresh")
    assert "return" in restore[failure : restore.index("feed-refresh")]
    assert "diskutil list -plist external" not in source
