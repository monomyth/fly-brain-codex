import AppKit
import ApplicationServices

let pid = pid_t(CommandLine.arguments[1])!
let wanted = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : ""
let app = AXUIElementCreateApplication(pid)
AXUIElementSetMessagingTimeout(app, 2)
func attribute(_ element: AXUIElement, _ key: String) -> CFTypeRef? {
    var value: CFTypeRef?
    return AXUIElementCopyAttributeValue(element, key as CFString, &value) == .success ? value : nil
}
func children(_ element: AXUIElement) -> [AXUIElement] { attribute(element, kAXChildrenAttribute) as? [AXUIElement] ?? [] }
func labels(_ element: AXUIElement, depth: Int = 0) -> [String] {
    var result = [kAXTitleAttribute, kAXValueAttribute, kAXDescriptionAttribute, kAXHelpAttribute].compactMap { attribute(element, $0) as? String }
    if depth < 3 { for child in children(element) { result += labels(child, depth: depth + 1) } }
    return result
}
let windows = attribute(app, kAXWindowsAttribute) as? [AXUIElement] ?? []
var queue = windows
var count = 0
var found = false
while !queue.isEmpty && count < 4000 {
    let element = queue.removeFirst(); count += 1
    if attribute(element, kAXRoleAttribute) as? String == kAXButtonRole {
        let text = labels(element)
        if wanted.isEmpty { print(text.joined(separator: " | ")) }
        else if text.contains(wanted) {
            let enabled = attribute(element, kAXEnabledAttribute) as? Bool ?? false
            guard enabled else { fputs("Button is disabled\n", stderr); exit(2) }
            let result = AXUIElementPerformAction(element, kAXPressAction as CFString)
            print("Pressed \(wanted): \(result.rawValue)")
            found = result == .success
            break
        }
    }
    queue += children(element)
}
if !wanted.isEmpty && !found { fputs("Button not found or action failed\n", stderr); exit(1) }
