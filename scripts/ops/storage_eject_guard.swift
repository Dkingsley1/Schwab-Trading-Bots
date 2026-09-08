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
    let disappearanceDuplicateCollapseSeconds: TimeInterval
    let flapWindowSeconds: TimeInterval
    let flapThreshold: Int
    let flapCooldownSeconds: TimeInterval
    let mountAttemptBaseBackoffSeconds: TimeInterval
    let mountAttemptMaxBackoffSeconds: TimeInterval
    let mountStabilizationMinIntervalSeconds: TimeInterval
    let spotlightDisableTimeoutSeconds: TimeInterval
    let denyUnsafeEject: Bool
    let disableSpotlightOnMount: Bool
    let maxEventLedgerBytes: UInt64
    let logPath: URL
    let overridePath: URL
    let statePath: URL
    let eventLedgerPath: URL
    let serial = DispatchQueue(label: "com.dankingsley.storage_eject_guard")
    var mountRoot: String
    var targetVolumeBSDName: String?
    var targetWholeBSDName: String?
    var lastEjectHandledAt = Date.distantPast
    var lastRestoreHandledAt = Date.distantPast
    var lastMountAttemptAt = Date.distantPast
    var lastMountStabilizedAt = Date.distantPast
    var lastDisappearRecordedAt = Date.distantPast
    var recentDisappearances: [Date] = []
    var externalFailbackCooldownUntil: Date?
    var mountFailureCount = 0
    var lastMountFailureReason = ""
    var lastSpotlightDisableRC: Int32?
    var lastMetadataNeverIndexWriteOK: Bool?
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
        self.disappearanceDuplicateCollapseSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_DISAPPEAR_DUPLICATE_COLLAPSE_SECONDS", defaultValue: 3.0)
        self.flapWindowSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_FLAP_WINDOW_SECONDS", defaultValue: 600.0)
        self.flapThreshold = StorageEjectGuard.envInt("BOT_LOGS_FLAP_THRESHOLD", defaultValue: 2)
        self.flapCooldownSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_FLAP_COOLDOWN_SECONDS", defaultValue: 900.0)
        self.mountAttemptBaseBackoffSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_MOUNT_ATTEMPT_BASE_BACKOFF_SECONDS", defaultValue: 15.0)
        self.mountAttemptMaxBackoffSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_MOUNT_ATTEMPT_MAX_BACKOFF_SECONDS", defaultValue: 300.0)
        self.mountStabilizationMinIntervalSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_MOUNT_STABILIZATION_MIN_INTERVAL_SECONDS", defaultValue: 300.0)
        self.spotlightDisableTimeoutSeconds = StorageEjectGuard.envTimeInterval("BOT_LOGS_SPOTLIGHT_DISABLE_TIMEOUT_SECONDS", defaultValue: 8.0)
        self.denyUnsafeEject = StorageEjectGuard.envBool("BOT_LOGS_DENY_UNSAFE_EJECT", defaultValue: true)
        self.disableSpotlightOnMount = StorageEjectGuard.envBool("BOT_LOGS_DISABLE_SPOTLIGHT_ON_MOUNT", defaultValue: true)
        self.maxEventLedgerBytes = UInt64(max(StorageEjectGuard.envInt("BOT_LOGS_EJECT_EVENT_LEDGER_MAX_BYTES", defaultValue: 5_000_000), 1))
        self.mountRoot = mountRoot
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logDir = home.appendingPathComponent("Library/Logs/schwab_trading_bot", isDirectory: true)
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        self.logPath = logDir.appendingPathComponent("storage_eject_guard.log")
        self.overridePath = projectRoot.appendingPathComponent("config/.env.storage_override")
        self.statePath = projectRoot.appendingPathComponent("governance/health/storage_eject_guard_latest.json")
        self.eventLedgerPath = projectRoot.appendingPathComponent("governance/health/storage_eject_guard_events.jsonl")
    }

    func run() {
        log("starting configuredMountRoot=\(configuredMountRoot) targetVolumeName=\(targetVolumeName) candidateMountRoots=\(candidateMountRoots.joined(separator: ",")) projectRoot=\(projectRoot.path)")
        guard let session = DASessionCreate(kCFAllocatorDefault) else {
            log("failed to create DiskArbitration session")
            writeTransitionState(status: "blocked", event: "startup_failed", detail: "disk_arbitration_session_unavailable")
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
            self.stabilizeMountedVolume(reason: "startup")
            self.writeTransitionState(
                status: "ready",
                event: "monitoring",
                detail: "disk arbitration callbacks and mount polling active",
                externalAvailable: self.externalMountAvailableNow()
            )
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
            stabilizeMountedVolume(reason: "disk_appeared")

            if failbackCooldownActive() {
                writeTransitionState(
                    status: "ready",
                    event: "external_available_flap_cooldown",
                    detail: "external storage is mounted, but recent disappearances are cooling down; active hot routing remains unchanged",
                    externalAvailable: true,
                    stackRestartRequired: false
                )
                return
            }

            guard shouldRestoreExternalOnAppear() else {
                writeTransitionState(
                    status: "ready",
                    event: "external_available_standby",
                    detail: "external storage is available; active hot routing remains unchanged pending explicit certification",
                    externalAvailable: true
                )
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
            let flapActive = self.recordDisappearance(disk: disappearedBSD, mode: mode)
            self.targetVolumeBSDName = nil
            self.targetWholeBSDName = nil

            guard self.shouldRestartLocalOnDisappear(mode: mode) else {
                self.writeTransitionState(
                    status: "ready",
                    event: flapActive ? "external_disconnected_standby_flap_cooldown" : "external_disconnected_standby",
                    detail: flapActive
                        ? "external storage disconnected while local hot storage was active; flap cooldown is armed and stack restart is suppressed"
                        : "external storage disconnected while local hot storage was already active; stack restart suppressed",
                    externalAvailable: false,
                    stackRestartRequired: false
                )
                return
            }

            self.writeTransitionState(
                status: "degraded",
                event: "external_disconnect_failover_pending",
                detail: "active external route disappeared; waiting through bounded grace before local failover",
                externalAvailable: false,
                stackRestartRequired: true
            )

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
            let mode = currentStorageMode()
            log("handling \(action) for disk=\(diskName) mountRoot=\(mountRoot) mode=\(mode)")

            var switchRC: Int32 = 0
            if shouldRestartLocalOnDisappear(mode: mode) {
                switchRC = prepareLocalFallbackForEject()
                log("prepare-local-for-eject rc=\(switchRC)")
            } else {
                log("approved \(action) is standby-only; stack restart suppressed")
                writeTransitionState(
                    status: "ready",
                    event: "standby_eject_approved",
                    detail: "external storage was not the active hot route",
                    externalAvailable: true,
                    stackRestartRequired: false
                )
            }

            let released = releaseExternalMountBlockers(timeout: 12.0)
            log("external_mount_release ok=\(released)")
            if !released || switchRC != 0 {
                writeTransitionState(
                    status: "blocked",
                    event: "eject_preflight_failed",
                    detail: "local failover or external handle release did not complete",
                    externalAvailable: true,
                    stackRestartRequired: switchRC != 0,
                    transitionRC: switchRC
                )
                if denyUnsafeEject {
                    log("denying \(action) for disk=\(diskName) because fallback_or_handle_release_failed rc=\(switchRC) released=\(released)")
                    return unsafeEjectDissenter(
                        message: "BOT_LOGS eject denied: local failover or handle release did not complete"
                    )
                }
            }
            return nil
        }
    }

    func prepareLocalFallbackForEject() -> Int32 {
        return restartLocalCollectionAfterEject(reason: "approved-eject")
    }

    func unsafeEjectDissenter(message: String) -> Unmanaged<DADissenter>? {
        let dissenter = DADissenterCreate(kCFAllocatorDefault, DAReturn(kDAReturnBusy), message as CFString)
        return Unmanaged.passRetained(dissenter)
    }

    @discardableResult
    func restartLocalCollectionAfterEject(reason: String = "surprise-disconnect") -> Int32 {
        let opsctl = projectRoot.appendingPathComponent("scripts/ops/opsctl.sh").path
        log("activating local collection reason=\(reason) mountRoot=\(mountRoot)")
        let switchRC = run(
            launchPath: "/bin/zsh",
            arguments: [
                "-lc",
                "\(shellQuote(opsctl)) storage-switch-local --no-refresh",
            ],
            timeout: 150
        )
        log("opsctl storage-switch-local --no-refresh local-after-eject rc=\(switchRC)")
        guard switchRC == 0 else {
            writeTransitionState(
                status: "blocked",
                event: "local_failover_failed",
                detail: "storage switch to local did not complete",
                externalAvailable: false,
                stackRestartRequired: true,
                transitionRC: switchRC
            )
            return switchRC
        }
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
        let transitionRC = refreshRC != 0 ? refreshRC : coordinatorRC
        writeTransitionState(
            status: transitionRC == 0 ? "ready" : "degraded",
            event: "local_failover_complete",
            detail: "external route was replaced by verified local storage and collection was refreshed",
            externalAvailable: false,
            stackRestartRequired: false,
            transitionRC: transitionRC
        )
        return transitionRC
    }

    func restoreExternalCollection() {
        log("restoring external collection for mountRoot=\(mountRoot)")
        guard externalMountAvailableNow(), externalWriteProbeReady() else {
            log("external restore certification failed mountRoot=\(mountRoot)")
            writeTransitionState(
                status: "blocked",
                event: "external_failback_certification_failed",
                detail: "mount identity, project root, or write probe was not ready",
                externalAvailable: externalMountAvailableNow(),
                stackRestartRequired: false
            )
            return
        }
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
        let transitionRC = switchRC != 0 ? switchRC : (refreshRC != 0 ? refreshRC : coordinatorRC)
        writeTransitionState(
            status: transitionRC == 0 ? "ready" : "degraded",
            event: "external_failback_complete",
            detail: "external route passed certification, reconciliation, and collection refresh",
            externalAvailable: true,
            stackRestartRequired: false,
            transitionRC: transitionRC
        )
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
        if localOverrideActive() {
            return "local_fallback"
        }
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
        let autoFailback = ProcessInfo.processInfo.environment["BOT_LOGS_AUTO_FAILBACK_ON_APPEAR"] ?? "0"
        guard ["1", "true", "yes", "on"].contains(
            autoFailback.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        ) else {
            log("automatic external failback suppressed; explicit storage-switch-external certification required")
            return false
        }
        let mode = currentStorageMode()
        return mode.hasPrefix("local_fallback") || localOverrideActive()
    }

    func shouldRestartLocalOnDisappear(mode: String) -> Bool {
        if mode.hasPrefix("external") {
            return true
        }
        if mode == "unknown" && !localOverrideActive() {
            return true
        }
        return false
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
        restartLocalCollectionAfterEject(reason: "surprise-disconnect")
    }

    func externalWriteProbeReady() -> Bool {
        let root = URL(fileURLWithPath: mountRoot).appendingPathComponent(expectedProjectDir, isDirectory: true)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        let probe = root.appendingPathComponent(".storage_eject_guard_write_probe.\(ProcessInfo.processInfo.processIdentifier)")
        do {
            try "probe\n".write(to: probe, atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(at: probe)
            return true
        } catch {
            log("external write probe failed root=\(root.path) error=\(error)")
            try? FileManager.default.removeItem(at: probe)
            return false
        }
    }

    @discardableResult
    func recordDisappearance(disk: String, mode: String) -> Bool {
        let now = Date()
        if now.timeIntervalSince(lastDisappearRecordedAt) < disappearanceDuplicateCollapseSeconds {
            log("collapsed duplicate disappearance disk=\(disk) mode=\(mode)")
            pruneRecentDisappearances(now: now)
            return failbackCooldownActive(now: now)
        }
        lastDisappearRecordedAt = now
        recentDisappearances.append(now)
        pruneRecentDisappearances(now: now)
        if recentDisappearances.count >= max(flapThreshold, 1) {
            externalFailbackCooldownUntil = now.addingTimeInterval(flapCooldownSeconds)
            log("external flap cooldown armed count=\(recentDisappearances.count) window_seconds=\(flapWindowSeconds) cooldown_seconds=\(flapCooldownSeconds) disk=\(disk) mode=\(mode)")
        }
        return failbackCooldownActive(now: now)
    }

    func pruneRecentDisappearances(now: Date = Date()) {
        recentDisappearances = recentDisappearances.filter { now.timeIntervalSince($0) <= flapWindowSeconds }
        if let cooldownUntil = externalFailbackCooldownUntil, cooldownUntil <= now {
            externalFailbackCooldownUntil = nil
        }
    }

    func failbackCooldownActive(now: Date = Date()) -> Bool {
        pruneRecentDisappearances(now: now)
        if let cooldownUntil = externalFailbackCooldownUntil, cooldownUntil > now {
            return true
        }
        return recentDisappearances.count >= max(flapThreshold, 1)
    }

    func currentMountBackoffSeconds() -> TimeInterval {
        let failureExponent = min(max(mountFailureCount, 0), 6)
        let multiplier = pow(2.0, Double(failureExponent))
        return min(mountAttemptMaxBackoffSeconds, max(mountAttemptBaseBackoffSeconds * multiplier, 1.0))
    }

    func stabilizeMountedVolume(reason: String) {
        guard externalMountAvailableNow() else { return }
        let now = Date()
        let forceReason = reason.contains("auto_mount_success")
        if !forceReason && now.timeIntervalSince(lastMountStabilizedAt) < mountStabilizationMinIntervalSeconds {
            return
        }
        lastMountStabilizedAt = now
        let volumeURL = URL(fileURLWithPath: mountRoot, isDirectory: true)
        var markerOK = false
        let marker = volumeURL.appendingPathComponent(".metadata_never_index")
        do {
            if !FileManager.default.fileExists(atPath: marker.path) {
                try Data().write(to: marker, options: .atomic)
            }
            markerOK = true
        } catch {
            markerOK = false
            log("metadata_never_index write failed reason=\(reason) path=\(marker.path) error=\(error)")
        }
        lastMetadataNeverIndexWriteOK = markerOK

        if disableSpotlightOnMount {
            let rc = run(
                launchPath: "/usr/bin/mdutil",
                arguments: ["-i", "off", mountRoot],
                timeout: spotlightDisableTimeoutSeconds
            )
            lastSpotlightDisableRC = rc
            log("mount-stabilization reason=\(reason) mountRoot=\(mountRoot) metadata_never_index=\(markerOK) mdutil_disable_rc=\(rc)")
        } else {
            lastSpotlightDisableRC = nil
            log("mount-stabilization reason=\(reason) mountRoot=\(mountRoot) metadata_never_index=\(markerOK) mdutil_disable=disabled_by_env")
        }
    }

    func flapControlPayload(now: Date = Date()) -> [String: Any] {
        pruneRecentDisappearances(now: now)
        let cooldownUntil = externalFailbackCooldownUntil
        let cooldownSecondsRemaining = cooldownUntil.map { max($0.timeIntervalSince(now), 0.0) } ?? 0.0
        return [
            "active": failbackCooldownActive(now: now),
            "recent_disappear_count": recentDisappearances.count,
            "window_seconds": flapWindowSeconds,
            "threshold": max(flapThreshold, 1),
            "cooldown_seconds": flapCooldownSeconds,
            "cooldown_until_utc": cooldownUntil.map { StorageEjectGuard.iso8601String(from: $0) } ?? "",
            "cooldown_seconds_remaining": round(cooldownSecondsRemaining),
            "duplicate_collapse_seconds": disappearanceDuplicateCollapseSeconds,
            "policy": "collapse whole-disk plus volume disappear bursts, keep local fallback during flap cooldown, and require stable remount before external failback",
        ]
    }

    func mountControlPayload() -> [String: Any] {
        return [
            "failure_count": mountFailureCount,
            "current_backoff_seconds": currentMountBackoffSeconds(),
            "last_failure_reason": lastMountFailureReason,
            "last_mount_attempt_utc": lastMountAttemptAt == Date.distantPast ? "" : StorageEjectGuard.iso8601String(from: lastMountAttemptAt),
            "base_backoff_seconds": mountAttemptBaseBackoffSeconds,
            "max_backoff_seconds": mountAttemptMaxBackoffSeconds,
            "policy": "remount attempts use bounded exponential backoff and do not run while local override suppresses external hot routing",
        ]
    }

    func volumeIdentityPayload() -> [String: Any] {
        return [
            "configured_mount_root": configuredMountRoot,
            "active_mount_root": mountRoot,
            "candidate_mount_roots": candidateMountRoots,
            "target_volume_name": targetVolumeName,
            "target_volume_uuid_hint": targetVolumeUUIDHint,
            "target_disk_identifier_hint": targetDiskIdentifierHint,
            "target_volume_bsd": targetVolumeBSDName ?? "",
            "target_whole_bsd": targetWholeBSDName ?? "",
        ]
    }

    func mountStabilizationPayload() -> [String: Any] {
        return [
            "spotlight_disable_enabled": disableSpotlightOnMount,
            "last_mdutil_disable_rc": lastSpotlightDisableRC.map { Int($0) } ?? NSNull(),
            "metadata_never_index_ready": lastMetadataNeverIndexWriteOK ?? NSNull(),
            "last_stabilized_utc": lastMountStabilizedAt == Date.distantPast ? "" : StorageEjectGuard.iso8601String(from: lastMountStabilizedAt),
            "min_interval_seconds": mountStabilizationMinIntervalSeconds,
            "spotlight_disable_timeout_seconds": spotlightDisableTimeoutSeconds,
            "policy": "mark BOT_LOGS as metadata-never-index and disable Spotlight indexing on mount to reduce avoidable external-disk churn",
        ]
    }

    func transitionRecommendedActions(event: String) -> [String] {
        var actions: [String] = []
        if failbackCooldownActive() {
            actions.append("keep external failback suppressed until BOT_LOGS remains mounted beyond the flap cooldown")
            actions.append("check USB cable, hub, port, enclosure power, and macOS disk sleep settings if disappearances continue")
        }
        if event == "eject_preflight_failed" {
            actions.append("keep the eject denied until local failover and external handle release both complete")
        }
        if event == "auto_mount_failed" {
            actions.append("let bounded remount backoff continue; avoid launching duplicate diskutil mount attempts")
        }
        if lastSpotlightDisableRC == -2 {
            actions.append("Spotlight disable timed out; keep metadata-never-index marker and avoid blocking storage recovery on mdutil")
        }
        if currentStorageMode().hasPrefix("local_fallback") && externalMountAvailableNow() {
            actions.append("after the volume stays stable, run storage-switch-external before pruning local standby")
        }
        return actions
    }

    func writeTransitionState(
        status: String,
        event: String,
        detail: String,
        externalAvailable: Bool? = nil,
        stackRestartRequired: Bool = false,
        transitionRC: Int32 = 0
    ) {
        var payload: [String: Any] = [
            "timestamp_utc": StorageEjectGuard.iso8601Now(),
            "schema_version": 1,
            "ok": status == "ready",
            "overall_status": status,
            "event": event,
            "detail": detail,
            "storage_mode": currentStorageMode(),
            "mount_root": mountRoot,
            "external_available": externalAvailable ?? externalMountAvailableNow(),
            "local_override_active": localOverrideActive(),
            "stack_restart_required": stackRestartRequired,
            "transition_rc": Int(transitionRC),
            "guard_pid": ProcessInfo.processInfo.processIdentifier,
            "live_execution_authority": "none",
            "policy": "standby disconnects never restart the stack; active-route loss fails over once to local and external failback requires identity plus write certification",
        ]
        payload["volume_identity"] = volumeIdentityPayload()
        payload["flap_control"] = flapControlPayload()
        payload["mount_control"] = mountControlPayload()
        payload["mount_stabilization"] = mountStabilizationPayload()
        payload["recommended_actions"] = transitionRecommendedActions(event: event)
        do {
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try FileManager.default.createDirectory(at: statePath.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: statePath, options: .atomic)
            appendTransitionEvent(payload)
        } catch {
            log("failed to write transition state: \(error)")
        }
    }

    func appendTransitionEvent(_ payload: [String: Any]) {
        do {
            try FileManager.default.createDirectory(at: eventLedgerPath.deletingLastPathComponent(), withIntermediateDirectories: true)
            rotateEventLedgerIfNeeded()
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            if FileManager.default.fileExists(atPath: eventLedgerPath.path), let handle = try? FileHandle(forWritingTo: eventLedgerPath) {
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
                try handle.write(contentsOf: Data("\n".utf8))
                try handle.close()
            } else {
                var line = data
                line.append(Data("\n".utf8))
                try line.write(to: eventLedgerPath, options: .atomic)
            }
        } catch {
            log("failed to append transition event: \(error)")
        }
    }

    func rotateEventLedgerIfNeeded() {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: eventLedgerPath.path),
              let size = attrs[.size] as? NSNumber,
              size.uint64Value > maxEventLedgerBytes else {
            return
        }
        let rotated = eventLedgerPath.deletingPathExtension().appendingPathExtension("previous.jsonl")
        try? FileManager.default.removeItem(at: rotated)
        try? FileManager.default.moveItem(at: eventLedgerPath, to: rotated)
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
        if localOverrideActive() {
            return false
        }
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
        if externalMountAvailableNow() {
            stabilizeMountedVolume(reason: "\(reason)_already_available")
            return
        }

        let now = Date()
        let backoff = currentMountBackoffSeconds()
        if now.timeIntervalSince(lastMountAttemptAt) < backoff {
            log("auto-mount backoff active reason=\(reason) failure_count=\(mountFailureCount) wait_seconds=\(backoff)")
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
        if mountRC == 0 {
            mountFailureCount = 0
            lastMountFailureReason = ""
            stabilizeMountedVolume(reason: "auto_mount_success_\(reason)")
        } else {
            mountFailureCount += 1
            lastMountFailureReason = "diskutil_mount_rc_\(mountRC)"
            writeTransitionState(
                status: "degraded",
                event: "auto_mount_failed",
                detail: "diskutil mount failed; bounded remount backoff is active",
                externalAvailable: false,
                stackRestartRequired: false,
                transitionRC: mountRC
            )
        }
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

    static func envInt(_ name: String, defaultValue: Int) -> Int {
        let raw = ProcessInfo.processInfo.environment[name] ?? ""
        guard let parsed = Int(raw), parsed >= 0 else {
            return defaultValue
        }
        return parsed
    }

    static func envBool(_ name: String, defaultValue: Bool) -> Bool {
        let raw = ProcessInfo.processInfo.environment[name] ?? ""
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["1", "true", "yes", "on"].contains(normalized) {
            return true
        }
        if ["0", "false", "no", "off"].contains(normalized) {
            return false
        }
        return defaultValue
    }

    static func iso8601String(from date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    static func iso8601Now() -> String {
        iso8601String(from: Date())
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
