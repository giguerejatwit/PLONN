import XCTest

final class ProPaywallUITests: XCTestCase {
    func testUnconfiguredBetaProIsReadOnly() {
        let app = XCUIApplication()
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-appearanceMode", "light", "-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
        app.launch()
        defer { app.terminate() }
        let settings = app.buttons["Account and settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 30), app.debugDescription)
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: settings)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed)
        settings.tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 10))
        let pro = app.buttons["pro-account-plans"]
        for _ in 0..<6 where !pro.isHittable { scroll(app) }
        XCTAssertTrue(pro.waitForExistence(timeout: 5), app.debugDescription)
        pro.tap()
        XCTAssertTrue(app.staticTexts["pro-paywall-title"].waitForExistence(timeout: 10))
        XCTAssertEqual(app.staticTexts["pro-paywall-unavailable"].label, "Pro subscriptions are unavailable in this beta.")
        XCTAssertTrue(app.staticTexts["pro-paywall-read-only"].label.contains("Purchases are unavailable"))
        XCTAssertEqual(app.staticTexts["pro-paywall-server-state"].label, "Plan status cannot be confirmed in this beta.")
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Pro beta unavailable and server verification state"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        let plans = app.staticTexts["pro-paywall-plans-unavailable"]
        for _ in 0..<6 where !plans.isHittable { scroll(app) }
        XCTAssertEqual(plans.label, "Subscription plans are not available yet.")
        XCTAssertTrue(app.staticTexts["pro-paywall-allowance-state"].label.contains("still being finalized"))
        XCTAssertFalse(app.staticTexts["pro-paywall-product-price"].exists)
        XCTAssertFalse(app.buttons["Purchase"].exists)
        XCTAssertFalse(app.buttons["Checkout"].exists)
        XCTAssertFalse(app.buttons["Restore purchases"].exists)
        XCTAssertFalse(app.buttons["Manage App Store subscriptions"].exists)
        XCTAssertFalse(app.staticTexts["pro-paywall-device-state"].exists)
    }
    private func scroll(_ app:XCUIApplication) {
        app.coordinate(withNormalizedOffset:CGVector(dx:0.85,dy:0.70))
            .press(forDuration:0.05,thenDragTo:app.coordinate(withNormalizedOffset:CGVector(dx:0.85,dy:0.40)))
    }
}
