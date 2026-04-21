#!/usr/bin/swift

import DiskArbitration
import Foundation

final class StorageEjectGuard {
    let projectRoot: URL
    let mountRoot: String
    let volumeName: String
    let logPath: URL
    let overridePath: URL
    let serial = DispatchQueue(label: "com.dankingsley.storage_eject_guard")
    var targetVolumeBSDName: String?
    var targetWholeBSDName: String?
    var lastEjectHandledAt = Date.distantPast
    var lastRestoreHandledAt = Date.distantPast

    init(projectRoot: URL, mountRoot: String) {
        self.projectRoot = projectRoot
        self.mountRoot = mountRoot
        self.volumeName = URL(fileURLWithPath: mountRoot).lastPathComponent
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logDir = home.appendingPathComponent("Library/Logs/schwab_trading_bot", isDirectory: true)
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        self.logPath = logDir.appendingPathComponent("storage_eject_guard.log")
        self.overridePath = projectRoot.appendingPathComponent("config/.env.storage_override")
    }

    func run() {
        log("starting mountRoot=\(mountRoot) projectRoot=\(projectRoot.path)")
        guard let session = DASessionCreate(kCFAllocatorDefault) else {
            log("failed to create DiskArbitration session")
            return
        }
        refreshTargetIdentity(session: session)
        DARegisterDiskAppearedCallback(session, nil, diskAppearedCallback, nil)
        DARegisterDiskDisappearedCallback(session, nil, diskDisappearedCallback, nil)
        DARegisterDiskUnmountApprovalCallback(session, nil, diskUnmountApprovalCallback, nil)
        DARegisterDiskEjectApprovalCallback(session, nil, diskEjectApprovalCallback, nil)
        DASessionSetDispatchQueue(session, DispatchQueue.main)
        dispatchMain()
    }

    func refreshTargetIdentity(session: DASession) {
        let url = URL(fileURLWithPath: mountRoot) as CFURL
        guard let disk = DADiskCreateFromVolumePath(kCFAllocatorDefault, session, url) else {
            targetVolumeBSDName = nil
            targetWholeBSDName = nil
            return
        }
        targetVolumeBSDName = StorageEjectGuard.bsdName(for: disk)
        if let whole = DADiskCopyWholeDisk(disk) {
            targetWholeBSDName = StorageEjectGuard.bsdName(for: whole)
        } else {
            targetWholeBSDName = targetVolumeBSDName
        }
        log("refreshed target volumeBSD=\(targetVolumeBSDName ?? "none") wholeBSD=\(targetWholeBSDName ?? "none")")
    }

    func handleAppeared(_ disk: DADisk) {
        guard matchesMountPath(disk) else { return }
        serial.sync {
            targetVolumeBSDName = StorageEjectGuard.bsdName(for: disk)
            if let whole = DADiskCopyWholeDisk(disk) {
                targetWholeBSDName = StorageEjectGuard.bsdName(for: whole)
            } else {
                targetWholeBSDName = targetVolumeBSDName
            }
            log("disk appeared volumeBSD=\(targetVolumeBSDName ?? "none") wholeBSD=\(targetWholeBSDName ?? "none")")

            guard localOverrideActive() else {
                return
            }

            let now = Date()
            if now.timeIntervalSince(lastRestoreHandledAt) < 5 {
                log("skipping duplicate restore event")
                return
            }
            lastRestoreHandledAt = now
            restoreExternalCollection()
        }
    }

    func handleDisappeared(_ disk: DADisk) {
        guard matchesTargetDisk(disk) else { return }
        let shouldRestartLocal = serial.sync { () -> Bool in
            log("disk disappeared volumeBSD=\(targetVolumeBSDName ?? "none") wholeBSD=\(targetWholeBSDName ?? "none")")
            targetVolumeBSDName = nil
            targetWholeBSDName = nil
            return localOverrideActive()
        }
        if shouldRestartLocal {
            serial.async {
                self.restartLocalCollectionAfterEject()
            }
        }
    }

    func handleApproval(_ disk: DADisk, action: String) -> Unmanaged<DADissenter>? {
        return serial.sync {
            guard matchesTargetDisk(disk) else {
                return nil
            }

            let now = Date()
            if now.timeIntervalSince(lastEjectHandledAt) < 5 {
                log("skipping duplicate \(action) event")
                return nil
            }
            lastEjectHandledAt = now

            let diskName = StorageEjectGuard.bsdName(for: disk) ?? "unknown"
            log("handling \(action) for disk=\(diskName) mountRoot=\(mountRoot)")

            let switchRC = prepareLocalFallbackForEject()
            log("prepare-local-for-eject rc=\(switchRC)")

            let released = releaseExternalMountBlockers(timeout: 12.0)
            log("external_mount_release ok=\(released)")
            return nil
        }
    }

    func prepareLocalFallbackForEject() -> Int32 {
        let opsctl = projectRoot.appendingPathComponent("scripts/ops/opsctl.sh").path
        let stopRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) stop",
            ],
            timeout: 45
        )
        log("opsctl stop before eject rc=\(stopRC)")
        let switchRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) storage-switch-local --no-refresh",
            ],
            timeout: 60
        )
        log("opsctl storage-switch-local --no-refresh rc=\(switchRC)")
        return switchRC != 0 ? switchRC : stopRC
    }

    func restartLocalCollectionAfterEject() {
        let opsctl = projectRoot.appendingPathComponent("scripts/ops/opsctl.sh").path
        log("restarting local collection after eject for mountRoot=\(mountRoot)")
        let refreshRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) feed-refresh --source all",
            ],
            timeout: 180
        )
        log("opsctl feed-refresh local-after-eject rc=\(refreshRC)")
        let coordinatorRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) storage-transition-coordinator --transition-mode local --apply --json >/dev/null 2>&1 || true",
            ],
            timeout: 120
        )
        log("opsctl storage-transition-coordinator local-after-eject rc=\(coordinatorRC)")
    }

    func restoreExternalCollection() {
        log("restoring external collection for mountRoot=\(mountRoot)")
        let switchRC = switchStorage(mode: "external")
        log("storage-switch-external rc=\(switchRC)")
    }

    func waitForExternalWritersToExit(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if !externalWritersStillRunning() {
                return true
            }
            Thread.sleep(forTimeInterval: 0.25)
        }
        return !externalWritersStillRunning()
    }

    func externalWritersStillRunning() -> Bool {
        let rc = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "ps -axo command | egrep 'run_all_sleeves.py|run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_dividend_shadow.py|run_bond_shadow.py|run_shadow_training_loop.py --broker (schwab|coinbase)|sql_link_shard_manager.py|sql_link_writer_service.py' | grep -v grep >/dev/null",
            ],
            timeout: 3
        )
        return rc == 0
    }

    func mountHasOpenHandles() -> Bool {
        let rc = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "lsof +D \(shellQuote(mountRoot)) >/dev/null 2>&1",
            ],
            timeout: 5
        )
        return rc == 0
    }

    func cleanupKnownMountBlockers(force: Bool) {
        let signal = force ? "-KILL" : "-TERM"
        let rc = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                """
                set +e
                mount_root=\(shellQuote(mountRoot))
                lsof +D "$mount_root" -Fpc 2>/dev/null | awk '
                  /^p/ { pid = substr($0, 2); next }
                  /^c/ { print pid "|" substr($0, 2) }
                ' | while IFS='|' read -r pid cmd; do
                  [[ -n "$pid" ]] || continue
                  if [[ "$cmd" == "tail" ]]; then
                    kill \(signal) "$pid" >/dev/null 2>&1 || true
                    continue
                  fi
                  full_cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
                  case "$full_cmd" in
                    *scripts/data_retention_policy.py*|\
                    *scripts/run_all_sleeves.py*|\
                    *scripts/run_parallel_shadows.py*|\
                    *scripts/run_parallel_aggressive_modes.py*|\
                    *scripts/run_dividend_shadow.py*|\
                    *scripts/run_bond_shadow.py*|\
                    *scripts/run_fx_shadow.py*|\
                    *scripts/run_shadow_training_loop.py*|\
                    *scripts/ops/sql_link_shard_manager.py*|\
                    *scripts/ops/sql_link_writer_service.py*|\
                    *scripts/link_jsonl_to_sql.py*|\
                    *scripts/ops/process_watchdog.py*|\
                    *scripts/ops/storage_maintenance_lane.py*|\
                    *scripts/ops/external_backlog_drain.py*)
                      kill \(signal) "$pid" >/dev/null 2>&1 || true
                      ;;
                  esac
                done
                exit 0
                """,
            ],
            timeout: 8
        )
        log("cleanup-known-mount-blockers force=\(force) rc=\(rc)")
    }

    func releaseExternalMountBlockers(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        cleanupKnownMountBlockers(force: false)
        while Date() < deadline {
            if !mountHasOpenHandles() {
                return true
            }
            if Date().addingTimeInterval(4.0) >= deadline {
                cleanupKnownMountBlockers(force: true)
            } else {
                cleanupKnownMountBlockers(force: false)
            }
            Thread.sleep(forTimeInterval: 0.35)
        }
        return !mountHasOpenHandles()
    }

    func writeLocalOverride() {
        let body = "# Auto-managed by storage_eject_guard.swift\nBOT_LOGS_PREFER_EXTERNAL=0\n"
        do {
            try FileManager.default.createDirectory(at: overridePath.deletingLastPathComponent(), withIntermediateDirectories: true)
            try body.write(to: overridePath, atomically: true, encoding: .utf8)
            log("wrote local storage override at \(overridePath.path)")
        } catch {
            log("failed to write local storage override: \(error)")
        }
    }

    func clearLocalOverride() {
        do {
            if FileManager.default.fileExists(atPath: overridePath.path) {
                try FileManager.default.removeItem(at: overridePath)
                log("cleared local storage override at \(overridePath.path)")
            }
        } catch {
            log("failed to clear local storage override: \(error)")
        }
    }

    func localOverrideActive() -> Bool {
        guard let body = try? String(contentsOf: overridePath, encoding: .utf8) else {
            return false
        }
        return body.contains("BOT_LOGS_PREFER_EXTERNAL=0")
    }

    func switchStorage(mode: String) -> Int32 {
        let opsctl = projectRoot.appendingPathComponent("scripts/ops/opsctl.sh").path
        let subcommand = (mode == "local") ? "storage-switch-local" : "storage-switch-external"
        return run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) \(subcommand)",
            ],
            timeout: 90
        )
    }

    func matchesMountPath(_ disk: DADisk) -> Bool {
        guard let description = DADiskCopyDescription(disk) as? [String: Any] else {
            return false
        }
        if let url = description[kDADiskDescriptionVolumePathKey as String] as? URL {
            return url.path == mountRoot
        }
        if let name = description[kDADiskDescriptionVolumeNameKey as String] as? String {
            return name == volumeName
        }
        return false
    }

    func matchesTargetDisk(_ disk: DADisk) -> Bool {
        if matchesMountPath(disk) {
            return true
        }

        if let bsd = StorageEjectGuard.bsdName(for: disk) {
            if bsd == targetVolumeBSDName || bsd == targetWholeBSDName {
                return true
            }
        }

        if let whole = DADiskCopyWholeDisk(disk), let bsd = StorageEjectGuard.bsdName(for: whole) {
            if bsd == targetWholeBSDName {
                return true
            }
        }

        return false
    }

    func log(_ message: String) {
        let line = "[\(StorageEjectGuard.iso8601Now())] \(message)\n"
        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logPath.path) {
                if let handle = try? FileHandle(forWritingTo: logPath) {
                    do {
                        try handle.seekToEnd()
                        try handle.write(contentsOf: data)
                        try handle.close()
                    } catch {
                        print(line, terminator: "")
                    }
                } else {
                    print(line, terminator: "")
                }
            } else {
                try? data.write(to: logPath)
            }
        }
        print(line, terminator: "")
    }

    func run(launchPath: String, arguments: [String], timeout: TimeInterval) -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        process.currentDirectoryURL = projectRoot
        process.standardOutput = Pipe()
        process.standardError = Pipe()

        do {
            try process.run()
        } catch {
            log("failed to run \(launchPath): \(error)")
            return -1
        }

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }

        if process.isRunning {
            process.terminate()
            return -2
        }

        return process.terminationStatus
    }

    static func bsdName(for disk: DADisk) -> String? {
        guard let ptr = DADiskGetBSDName(disk) else {
            return nil
        }
        return String(cString: ptr)
    }

    static func iso8601Now() -> String {
        ISO8601DateFormatter().string(from: Date())
    }
}

private func shellQuote(_ value: String) -> String {
    if value.isEmpty {
        return "''"
    }
    return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private let guardInstance = StorageEjectGuard(
    projectRoot: URL(fileURLWithPath: ProcessInfo.processInfo.environment["PROJECT_ROOT"] ?? FileManager.default.currentDirectoryPath),
    mountRoot: ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_MOUNT"] ?? "/Volumes/BOT_LOGS"
)

private func diskAppearedCallback(disk: DADisk, context: UnsafeMutableRawPointer?) {
    guardInstance.handleAppeared(disk)
}

private func diskDisappearedCallback(disk: DADisk, context: UnsafeMutableRawPointer?) {
    guardInstance.handleDisappeared(disk)
}

private func diskUnmountApprovalCallback(disk: DADisk, context: UnsafeMutableRawPointer?) -> Unmanaged<DADissenter>? {
    guardInstance.handleApproval(disk, action: "unmount")
}

private func diskEjectApprovalCallback(disk: DADisk, context: UnsafeMutableRawPointer?) -> Unmanaged<DADissenter>? {
    guardInstance.handleApproval(disk, action: "eject")
}

guardInstance.run()
