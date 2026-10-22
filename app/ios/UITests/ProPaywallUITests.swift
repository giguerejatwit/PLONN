import XCTest

final class ProPaywallUITests: XCTestCase {
    func testSignedOutPreviewRequiresSignInAndCannotRestore() throws {
        #if PLONN_STOREKIT_LAB
        throw XCTSkip("The local StoreKit lab uses its own synthetic account.")
        #else
        let app = XCUIApplication()
        app.launchEnvironment["PLONN_PRO_PREVIEW"] = "1"
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(app.staticTexts["pro-paywall-title"].waitForExistence(timeout: 20))
        let signIn = app.buttons["Sign in to subscribe"]
        for _ in 0..<5 where !signIn.isHittable { app.swipeUp() }
        XCTAssertTrue(signIn.isHittable, app.debugDescription)
        XCTAssertFalse(app.buttons["pro-subscribe"].exists)
        let restore = app.buttons["Restore purchases"]
        for _ in 0..<7 where !restore.isHittable { app.swipeUp() }
        XCTAssertTrue(restore.exists, app.debugDescription)
        XCTAssertFalse(restore.isEnabled)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Signed-out paywall disables restore"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        #endif
    }
}
