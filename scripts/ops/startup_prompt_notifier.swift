import AppKit
import Darwin
import Foundation
import UserNotifications

private let startActionID = "START_GUARDED_STACK"
private let declineActionID = "DECLINE_GUARDED_STACK"
private let categoryID = "SCHWAB_STARTUP_CONTROL"
private let notificationID = "schwab-trading-bot-startup-control"

private func argumentValue(_ name: String, default defaultValue: String = "") -> String {
    guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count else {
        return defaultValue
    }
    return CommandLine.arguments[index + 1]
}

private final class PromptNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    var onDecision: ((String) -> Void)?

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .list, .sound])
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let decision: String
        switch response.actionIdentifier {
        case startActionID:
            decision = "Yes"
        case declineActionID, UNNotificationDismissActionIdentifier:
            decision = "No"
        case UNNotificationDefaultActionIdentifier:
            decision = "fallback"
        default:
            decision = "No"
        }
        onDecision?(decision)
        completionHandler()
    }
}

let timeoutSeconds = max(Double(argumentValue("--timeout-seconds", default: "600")) ?? 600.0, 1.0)
let resultPath = argumentValue("--result-file")
if CommandLine.arguments.contains("--self-test") {
    if !resultPath.isEmpty {
        let resultURL = URL(fileURLWithPath: resultPath)
        try? FileManager.default.createDirectory(
            at: resultURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? "self_test_ready\n".write(to: resultURL, atomically: true, encoding: .utf8)
    } else {
        print("self_test_ready")
    }
    exit(EXIT_SUCCESS)
}
let center = UNUserNotificationCenter.current()
private let delegate = PromptNotificationDelegate()
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
center.delegate = delegate

var finished = false
func finish(_ decision: String) {
    DispatchQueue.main.async {
        guard !finished else { return }
        finished = true
        center.removePendingNotificationRequests(withIdentifiers: [notificationID])
        center.removeDeliveredNotifications(withIdentifiers: [notificationID])
        if !resultPath.isEmpty {
            let resultURL = URL(fileURLWithPath: resultPath)
            try? FileManager.default.createDirectory(
                at: resultURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? (decision + "\n").write(to: resultURL, atomically: true, encoding: .utf8)
        } else {
            print(decision)
            fflush(stdout)
        }
        app.terminate(nil)
    }
}

delegate.onDecision = finish

center.requestAuthorization(options: [.alert, .sound]) { granted, error in
    DispatchQueue.main.async {
        guard error == nil else {
            finish("unavailable:authorization_error")
            return
        }
        guard granted else {
            finish("unavailable:notification_permission_denied")
            return
        }

        let startAction = UNNotificationAction(
            identifier: startActionID,
            title: "Start",
            options: [.foreground]
        )
        let declineAction = UNNotificationAction(
            identifier: declineActionID,
            title: "Not Now",
            options: []
        )
        let category = UNNotificationCategory(
            identifier: categoryID,
            actions: [startAction, declineAction],
            intentIdentifiers: [],
            options: [.customDismissAction]
        )
        center.setNotificationCategories([category])

        let content = UNMutableNotificationContent()
        content.title = "Schwab Trading Bot"
        content.subtitle = "Start guarded paper trading?"
        content.body = "Choose Start or Not Now. No response leaves the system off."
        content.categoryIdentifier = categoryID
        content.sound = .default

        let request = UNNotificationRequest(identifier: notificationID, content: content, trigger: nil)
        center.removePendingNotificationRequests(withIdentifiers: [notificationID])
        center.removeDeliveredNotifications(withIdentifiers: [notificationID])
        center.add(request) { addError in
            if addError != nil {
                finish("unavailable:notification_delivery_error")
            }
        }
    }
}

DispatchQueue.main.asyncAfter(deadline: .now() + timeoutSeconds) {
    finish("timeout")
}

app.run()
