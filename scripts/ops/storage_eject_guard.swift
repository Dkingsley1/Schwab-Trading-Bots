#!/usr/bin/swift

import DiskArbitration
import Foundation

final class StorageEjectGuard {
    struct TargetVolume {
        let deviceIdentifier: String
        let volumeName: String
        let volumeUUID: String
        let mountPoint: String?

        var isMounted: Bool {
            guard let mountPoint else { return false }
            return !mountPoint.isEmpty
        }
    }

    let projectRoot: URL
    let configuredMountRoot: String
    let candidateMountRoots: [String]
    let candidateVolumeNames: Set<String>
    let expectedProjectDir: String
    let targetVolumeName: String
    let targetVolumeUUIDHint: String
    let targetDiskIdentifierHint: String
    let disappearanceGraceSeconds: TimeInterval
    let logPath: URL
    let overridePath: URL
    let serial = DispatchQueue(label: "com.dankingsley.storage_eject_guard")
    var mountRoot: String
    var targetVolumeBSDName: String?
    var targetWholeBSDName: String?
    var lastEjectHandledAt = Date.distantPast
    var lastRestoreHandledAt = Date.distantPast
    var lastMountAttemptAt = Date.distantPast
    var mountPollTimer: DispatchSourceTimer?
    var pendingDisappearWorkItem: DispatchWorkItem?

    init(projectRoot: URL, mountRoot: String) {
        self.projectRoot = projectRoot
        self.configuredMountRoot = mountRoot
        self.expectedProjectDir = ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_PROJECT_DIR"] ?? "schwab_trading_bot"
        self.candidateMountRoots = StorageEjectGuard.resolveCandidateMountRoots(primary: mountRoot)
        self.candidateVolumeNames = Set(self.candidateMountRoots.map { URL(fileURLWithPath: $0).lastPathComponent })
        self.targetVolumeName = ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_VOLUME_NAME"] ?? URL(fileURLWithPath: mountRoot).lastPathComponent
        self.targetVolumeUUIDHint = ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_VOLUME_UUID"] ?? ""
        self.targetDiskIdentifierHint = ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_DISK_IDENTIFIER"] ?? ""
        self.disappearanceGraceSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_DISAPPEAR_GRACE_SECONDS", defaultValue: 15.0)
        self.mountRoot = mountRoot
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logDir = home.appendingPathComponent("Library/Logs/schwab_trading_bot", isDirectory: true)
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        self.logPath = logDir.appendingPathComponent("storage_eject_guard.log")
        self.overridePath = projectRoot.appendingPathComponent("config/.env.storage_override")
    }

    func run() {
        log("starting configuredMountRoot=\(configuredMountRoot) targetVolumeName=\(targetVolumeName) candidateMountRoots=\(candidateMountRoots.joined(separator: ",")) projectRoot=\(projectRoot.path)")
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
        startMountPollTimer()
        serial.async {
            self.maybeMountTargetVolume(reason: "startup")
        }
        dispatchMain()
    }

    func refreshTargetIdentity(session: DASession) {
        let resolvedMountRoot = candidateMountRoots.first { FileManager.default.fileExists(atPath: $0) } ?? configuredMountRoot
        mountRoot = resolvedMountRoot
        let url = URL(fileURLWithPath: resolvedMountRoot) as CFURL
        guard let disk = DADiskCreateFromVolumePath(kCFAllocatorDefault, session, url) else {
            targetVolumeBSDName = nil
            targetWholeBSDName = nil
            log("target identity unavailable for mountRoot=\(resolvedMountRoot)")
            return
        }
        targetVolumeBSDName = StorageEjectGuard.bsdName(for: disk)
        if let whole = DADiskCopyWholeDisk(disk) {
            targetWholeBSDName = StorageEjectGuard.bsdName(for: whole)
        } else {
            targetWholeBSDName = targetVolumeBSDName
        }
        log("refreshed target mountRoot=\(mountRoot) volumeBSD=\(targetVolumeBSDName ?? "none") wholeBSD=\(targetWholeBSDName ?? "none")")
    }

    func handleAppeared(_ disk: DADisk) {
        guard matchesMountPath(disk) else { return }
        serial.sync {
            pendingDisappearWorkItem?.cancel()
            pendingDisappearWorkItem = nil
            if let volumeURL = StorageEjectGuard.volumeURL(for: disk) {
                mountRoot = volumeURL.path
            }
            targetVolumeBSDName = StorageEjectGuard.bsdName(for: disk)
            if let whole = DADiskCopyWholeDisk(disk) {
                targetWholeBSDName = StorageEjectGuard.bsdName(for: whole)
            } else {
                targetWholeBSDName = targetVolumeBSDName
            }
            log("disk appeared mountRoot=\(mountRoot) volumeBSD=\(targetVolumeBSDName ?? "none") wholeBSD=\(targetWholeBSDName ?? "none") mode=\(currentStorageMode())")

            guard shouldRestoreExternalOnAppear() else {
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

    func handleObservedDiskAppeared(_ disk: DADisk) {
        if matchesMountPath(disk) {
            handleAppeared(disk)
            return
        }
        serial.async {
            self.maybeMountTargetVolume(reason: "disk_appeared")
        }
    }

    func handleDisappeared(_ disk: DADisk) {
        guard matchesTargetDisk(disk) else { return }
        let disappearedBSD = StorageEjectGuard.bsdName(for: disk) ?? "unknown"
        serial.async {
            let mode = self.currentStorageMode()
            self.log("disk disappeared mountRoot=\(self.mountRoot) volumeBSD=\(self.targetVolumeBSDName ?? "none") wholeBSD=\(self.targetWholeBSDName ?? "none") disk=\(disappearedBSD) mode=\(mode)")
            self.targetVolumeBSDName = nil
            self.targetWholeBSDName = nil

            guard self.shouldRestartLocalOnDisappear(mode: mode) else {
                return
            }

            self.pendingDisappearWorkItem?.cancel()
            let workItem = DispatchWorkItem { [weak self] in
                self?.confirmDisappearAndRestartLocal(originalMode: mode, disappearedBSD: disappearedBSD)
            }
            self.pendingDisappearWorkItem = workItem
            self.log("scheduled disappearance verification grace_seconds=\(self.disappearanceGraceSeconds) disk=\(disappearedBSD) mode=\(mode)")
            self.serial.asyncAfter(deadline: .now() + self.disappearanceGraceSeconds, execute: workItem)
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
        let switchRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) storage-switch-local --no-refresh",
            ],
            timeout: 120
        )
        log("opsctl storage-switch-local --no-refresh local-after-eject rc=\(switchRC)")
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
        runPostTransitionRecovery(opsctl: opsctl, mode: "local-after-eject")
    }

    func restoreExternalCollection() {
        log("restoring external collection for mountRoot=\(mountRoot)")
        let opsctl = projectRoot.appendingPathComponent("scripts/ops/opsctl.sh").path
        let switchRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) storage-switch-external --no-refresh",
            ],
            timeout: 120
        )
        log("opsctl storage-switch-external --no-refresh rc=\(switchRC)")
        let refreshRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) feed-refresh --source all",
            ],
            timeout: 180
        )
        log("opsctl feed-refresh external-restore rc=\(refreshRC)")
        let coordinatorRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) storage-transition-coordinator --transition-mode external --apply --json >/dev/null 2>&1 || true",
            ],
            timeout: 180
        )
        log("opsctl storage-transition-coordinator external-restore rc=\(coordinatorRC)")
        runPostTransitionRecovery(opsctl: opsctl, mode: "external-restore")
    }

    func runPostTransitionRecovery(opsctl: String, mode: String) {
        let commands: [(String, String, TimeInterval)] = [
            ("split-brain-reconcile", "\(shellQuote(opsctl)) split-brain-reconcile --force-failback-if-hashes-match --json >/dev/null 2>&1 || true", 120),
            ("external-backlog-drain", "\(shellQuote(opsctl)) external-backlog-drain --apply --follow-through --poll-seconds 5 --wait-timeout-seconds 45 --json >/dev/null 2>&1 || true", 120),
            ("storage-pressure-clearance", "\(shellQuote(opsctl)) storage-pressure-clearance --apply --max-cycles 2 --poll-seconds 5 --wait-timeout-seconds 45 --json >/dev/null 2>&1 || true", 300),
            ("global-halt-refresh", "\(shellQuote(opsctl)) global-halt-refresh --json >/dev/null 2>&1 || true", 60),
            ("global-halt-auto-clear", "\(shellQuote(opsctl)) global-halt-auto-clear --json >/dev/null 2>&1 || true", 60),
            ("operator-cockpit", "\(shellQuote(opsctl)) operator-cockpit --json >/dev/null 2>&1 || true", 60),
            ("storage-reconnect-regression-guard", "\(shellQuote(opsctl)) storage-reconnect-regression-guard --json >/dev/null 2>&1 || true", 60),
        ]
        for (name, command, timeout) in commands {
            let rc = run(
                launchPath: "/bin/zsh",
                arguments: ["-lc", command],
                timeout: timeout
            )
            log("opsctl post-transition \(name) mode=\(mode) rc=\(rc)")
        }
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

    func currentStorageMode() -> String {
        let healthPaths = [
            projectRoot.appendingPathComponent("governance/health/storage_failback_sync_latest.json"),
            projectRoot.appendingPathComponent("governance/health/storage_mount_guard_latest.json"),
            projectRoot.appendingPathComponent("governance/health/process_watchdog_latest.json"),
        ]
        for path in healthPaths {
            guard let data = try? Data(contentsOf: path) else { continue }
            guard let obj = try? JSONSerialization.jsonObject(with: data, options: []),
                  let dict = obj as? [String: Any] else { continue }
            if let mode = dict["mode"] as? String, !mode.isEmpty {
                return mode
            }
            if let mode = dict["storage_mode"] as? String, !mode.isEmpty {
                return mode
            }
        }
        return localOverrideActive() ? "local_fallback" : "unknown"
    }

    func shouldRestoreExternalOnAppear() -> Bool {
        let mode = currentStorageMode()
        return mode.hasPrefix("local_fallback") || localOverrideActive()
    }

    func shouldRestartLocalOnDisappear(mode: String) -> Bool {
        if mode == "external" {
            return true
        }
        return localOverrideActive()
    }

    func confirmDisappearAndRestartLocal(originalMode: String, disappearedBSD: String) {
        pendingDisappearWorkItem = nil
        maybeMountTargetVolume(reason: "disappear_grace")
        if externalMountAvailableNow() {
            log("external_still_available_after_disappear disk=\(disappearedBSD) originalMode=\(originalMode) mountRoot=\(mountRoot); skipping local fallback")
            return
        }

        let mode = currentStorageMode()
        guard shouldRestartLocalOnDisappear(mode: originalMode) || shouldRestartLocalOnDisappear(mode: mode) else {
            log("disappearance verification skipped disk=\(disappearedBSD) originalMode=\(originalMode) currentMode=\(mode)")
            return
        }
        log("confirmed external unavailable after disappearance grace disk=\(disappearedBSD) originalMode=\(originalMode) currentMode=\(mode)")
        restartLocalCollectionAfterEject()
    }

    func externalMountAvailableNow() -> Bool {
        for candidate in candidateMountRoots {
            let volumeURL = URL(fileURLWithPath: candidate)
            if StorageEjectGuard.projectRootExists(on: volumeURL, projectDir: expectedProjectDir) {
                mountRoot = candidate
                return true
            }
        }
        guard let target = discoverTargetVolume(), target.isMounted else {
            return false
        }
        if let mountPoint = target.mountPoint, !mountPoint.isEmpty {
            mountRoot = mountPoint
        }
        return true
    }

    func externalPreferredByConfig() -> Bool {
        let raw = ProcessInfo.processInfo.environment["BOT_LOGS_PREFER_EXTERNAL"] ?? "1"
        return !["0", "false", "no", "off"].contains(raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    func startMountPollTimer() {
        let timer = DispatchSource.makeTimerSource(queue: serial)
        timer.schedule(deadline: .now() + 10.0, repeating: 20.0)
        timer.setEventHandler { [weak self] in
            self?.maybeMountTargetVolume(reason: "poll")
        }
        timer.resume()
        mountPollTimer = timer
    }

    func maybeMountTargetVolume(reason: String) {
        guard externalPreferredByConfig() else { return }
        guard !FileManager.default.fileExists(atPath: configuredMountRoot) else { return }

        let now = Date()
        if now.timeIntervalSince(lastMountAttemptAt) < 8.0 {
            return
        }
        lastMountAttemptAt = now

        guard let target = discoverTargetVolume() else {
            log("auto-mount skipped reason=\(reason) target_volume_not_found volumeName=\(targetVolumeName)")
            return
        }

        if target.isMounted {
            if let mountPoint = target.mountPoint, !mountPoint.isEmpty {
                mountRoot = mountPoint
            }
            log("auto-mount skipped reason=\(reason) identifier=\(target.deviceIdentifier) already_mounted=\(target.mountPoint ?? "unknown")")
            return
        }

        let mountRC = run(
            launchPath: "/usr/sbin/diskutil",
            arguments: [
                "mount",
                target.deviceIdentifier,
            ],
            timeout: 90
        )
        log("diskutil mount reason=\(reason) identifier=\(target.deviceIdentifier) volumeName=\(target.volumeName) rc=\(mountRC)")
    }

    func discoverTargetVolume() -> TargetVolume? {
        let plist = diskutilListPlist()
        let rows = plist["AllDisksAndPartitions"] as? [[String: Any]] ?? []
        var bestScore = Int.min
        var bestMatch: TargetVolume?

        func consider(_ row: [String: Any]) {
            guard let identifier = row["DeviceIdentifier"] as? String, !identifier.isEmpty else {
                return
            }
            let volumeName = (row["VolumeName"] as? String) ?? ""
            let volumeUUID = (row["VolumeUUID"] as? String) ?? ((row["DiskUUID"] as? String) ?? "")
            let mountPoint = row["MountPoint"] as? String

            var score = 0
            if !targetDiskIdentifierHint.isEmpty && identifier == targetDiskIdentifierHint {
                score += 100
            }
            if !targetVolumeUUIDHint.isEmpty && volumeUUID.caseInsensitiveCompare(targetVolumeUUIDHint) == .orderedSame {
                score += 80
            }
            if volumeName == targetVolumeName {
                score += 40
            }
            guard score > 0 else {
                return
            }
            if score <= bestScore {
                return
            }
            bestScore = score
            bestMatch = TargetVolume(
                deviceIdentifier: identifier,
                volumeName: volumeName,
                volumeUUID: volumeUUID,
                mountPoint: mountPoint
            )
        }

        for row in rows {
            consider(row)
            for key in ["Partitions", "APFSVolumes"] {
                guard let children = row[key] as? [[String: Any]] else {
                    continue
                }
                for child in children {
                    consider(child)
                }
            }
        }
        return bestMatch
    }

    func diskutilListPlist() -> [String: Any] {
        let result = runCapture(
            launchPath: "/usr/sbin/diskutil",
            arguments: [
                "list",
                "-plist",
                "external",
            ],
            timeout: 45
        )
        guard result.rc == 0 else {
            log("diskutil list -plist external rc=\(result.rc)")
            return [:]
        }
        guard let plist = try? PropertyListSerialization.propertyList(from: result.stdout, options: [], format: nil),
              let dict = plist as? [String: Any] else {
            log("diskutil list -plist external parse_failed")
            return [:]
        }
        return dict
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
            if candidateMountRoots.contains(url.path) {
                return true
            }
            if StorageEjectGuard.projectRootExists(on: url, projectDir: expectedProjectDir) {
                return true
            }
        }
        if let name = description[kDADiskDescriptionVolumeNameKey as String] as? String {
            return candidateVolumeNames.contains(name)
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
        return runCapture(launchPath: launchPath, arguments: arguments, timeout: timeout).rc
    }

    func runCapture(launchPath: String, arguments: [String], timeout: TimeInterval) -> (rc: Int32, stdout: Data, stderr: Data) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        process.currentDirectoryURL = projectRoot
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            log("failed to run \(launchPath): \(error)")
            return (-1, Data(), Data())
        }

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }

        if process.isRunning {
            process.terminate()
            return (-2, Data(), Data())
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        return (process.terminationStatus, stdoutData, stderrData)
    }

    static func bsdName(for disk: DADisk) -> String? {
        guard let ptr = DADiskGetBSDName(disk) else {
            return nil
        }
        return String(cString: ptr)
    }

    static func volumeURL(for disk: DADisk) -> URL? {
        guard let description = DADiskCopyDescription(disk) as? [String: Any] else {
            return nil
        }
        return description[kDADiskDescriptionVolumePathKey as String] as? URL
    }

    static func projectRootExists(on volumeURL: URL, projectDir: String) -> Bool {
        let candidate = volumeURL.appendingPathComponent(projectDir, isDirectory: true)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory) else {
            return false
        }
        return isDirectory.boolValue
    }

    static func resolveCandidateMountRoots(primary: String) -> [String] {
        let envRaw = ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES"] ?? ""
        var out: [String] = []
        var seen = Set<String>()

        func appendUnique(_ value: String) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            guard !seen.contains(trimmed) else { return }
            seen.insert(trimmed)
            out.append(trimmed)
        }

        appendUnique(primary)
        for token in envRaw.split(separator: ",") {
            appendUnique(String(token))
        }
        return out
    }

    static func envTimeInterval(_ name: String, defaultValue: TimeInterval) -> TimeInterval {
        let raw = ProcessInfo.processInfo.environment[name] ?? ""
        guard let parsed = Double(raw), parsed >= 0 else {
            return defaultValue
        }
        return parsed
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
    guardInstance.handleObservedDiskAppeared(disk)
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
