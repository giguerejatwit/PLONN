import XCTest

/// Launch-only, synthetic contract responses. Never uses the signed-in simulator or a live transport.
final class SharedMatchupReportUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    private enum State: String { case ready, stale, missing, authenticationRequired }

    func testNFLReady() throws { try verify("NFL", .ready) }
    func testNFLStale() throws { try verify("NFL", .stale) }
    func testNFLMissing() throws { try verify("NFL", .missing) }
    func testNFLAuthenticationRequired() throws { try verify("NFL", .authenticationRequired) }
    func testNBAReady() throws { try verify("NBA", .ready) }
    func testNBAStale() throws { try verify("NBA", .stale) }
    func testNBAMissing() throws { try verify("NBA", .missing) }
    func testNBAAuthenticationRequired() throws { try verify("NBA", .authenticationRequired) }
    func testMLBReady() throws { try verify("MLB", .ready) }
    func testMLBStale() throws { try verify("MLB", .stale) }
    func testMLBMissing() throws { try verify("MLB", .missing) }
    func testMLBAuthenticationRequired() throws { try verify("MLB", .authenticationRequired) }
    func testNHLReady() throws { try verify("NHL", .ready) }
    func testNHLStale() throws { try verify("NHL", .stale) }
    func testNHLMissing() throws { try verify("NHL", .missing) }
    func testNHLAuthenticationRequired() throws { try verify("NHL", .authenticationRequired) }

    private func example(_ sport: String) throws -> [String: Any] {
        let derived = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("contracts/shared-insights-v1.json")
        let explicit = ProcessInfo.processInfo.environment["PLONN_SHARED_REPORT_CONTRACT_PATH"]
        let paths = [explicit, derived.path, "/Users/indoorgeo/Dev/PLONN/app/contracts/shared-insights-v1.json"].compactMap { $0 }
        let path = try XCTUnwrap(paths.first { FileManager.default.fileExists(atPath: $0) }, "Exact shared report contract fixture must exist on this Mac")
        let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any])
        let examples = try XCTUnwrap(root["examples"] as? [String: Any])
        return try XCTUnwrap(examples[sport] as? [String: Any])
    }

    private func verify(_ sport: String, _ state: State) throws {
        var response = try example(sport)
        let originalReport = try XCTUnwrap(response["report"] as? [String: Any])
        var now = "2026-10-13T13:00:00Z"
        if state == .stale {
            response["status"] = "stale"
            response["reason"] = "generation_expired"
            now = "2026-10-14T13:00:00Z" // More than 24 hours old; still before the Oct 15 event.
        } else if state == .missing {
            response["status"] = "missing"
            response["reason"] = "cache_miss"
            for key in ["report", "snapshot", "generatedAt"] { response[key] = NSNull() }
        }
        let fixture: [String: Any] = ["response": response,
            "statusCode": state == .authenticationRequired ? 401 : 200, "now": now]
        let app = XCUIApplication()
        app.launchArguments = ["-appearanceMode", "light", "-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launchEnvironment["PLONN_SHARED_REPORT_FIXTURE"] = String(decoding: try JSONSerialization.data(withJSONObject: fixture), as: UTF8.self)
        app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(app.staticTexts["shared-report-offline-banner"].waitForExistence(timeout: 15), app.debugDescription)
        XCTAssertTrue(app.descendants(matching: .any)["shared-report-card"].exists)
        XCTAssertEqual(app.staticTexts.matching(identifier: "matchup-outlook-heading").count, 1)
        if state == .ready {
            XCTAssertTrue(app.staticTexts.matching(identifier: "shared-report-outlook").firstMatch.waitForExistence(timeout: 15), app.debugDescription)
        } else {
            let expected: String
            switch state {
            case .stale: expected = "This outlook is out of date. Current conditions may differ."
            case .missing: expected = "No outlook is available for this matchup yet."
            case .authenticationRequired: expected = "Sign in to view this matchup’s outlook."
            case .ready: fatalError("Handled above")
            }
            let settled = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", expected), object: app.staticTexts["shared-report-status"])
            XCTAssertEqual(XCTWaiter.wait(for: [settled], timeout: 15), .completed, app.debugDescription)
        }
        for id in ["matchup-insights-refresh-ai", "matchup-insights-enable-ai"] {
            XCTAssertFalse(app.buttons[id].exists, "Shared reads must not expose private generation/consent controls")
        }
        XCTAssertEqual(app.buttons.matching(NSPredicate(format: "label CONTAINS[c] %@ OR label CONTAINS[c] %@", "Generate", "Enable Gemma")).count, 0)
        let outlooks = app.staticTexts.matching(identifier: "shared-report-outlook")
        let observations = app.staticTexts.matching(identifier: "shared-report-observation")
        XCTAssertFalse(app.buttons["shared-report-details"].exists)
        XCTAssertFalse(app.buttons["Sources and evidence"].exists)
        for id in ["shared-report-observation", "shared-report-updated-at", "shared-report-sample", "matchup-narrative-recorded", "shared-report-generated-at", "shared-report-source-as-of", "shared-report-retrieved-at", "shared-report-checked-at"] {
            XCTAssertFalse(app.staticTexts[id].exists, "Internal source diagnostics must not be displayed: " + id)
        }
        if state == .ready || state == .stale {
            let expectedOutlooks = try XCTUnwrap(originalReport["outlook"] as? [[String: Any]])
            XCTAssertEqual(outlooks.count, expectedOutlooks.count)
            for outlook in outlooks.allElementsBoundByIndex {
                XCTAssertFalse(outlook.label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            XCTAssertFalse(app.staticTexts["shared-report-updated-at"].exists)
            let card = app.descendants(matching: .any)["shared-report-card"]
            for label in card.staticTexts.allElementsBoundByIndex.map(\.label) {
                XCTAssertFalse(label.contains("ESPN"))
                XCTAssertFalse(label.contains("Sample window"))
                XCTAssertFalse(label.contains("Sources checked"))
            }
            XCTAssertEqual(app.staticTexts["This outlook is out of date. Current conditions may differ."].exists, state == .stale)
            XCTAssertEqual(observations.count, 0)
            capture(app, "Offline fixture · \(sport) · \(state.rawValue) · outlook without source or sample metadata")
        } else {
            XCTAssertEqual(outlooks.count, 0)
            XCTAssertEqual(observations.count, 0)
            XCTAssertFalse(app.staticTexts["shared-report-updated-at"].exists)
            capture(app, "Offline fixture · \(sport) · \(state.rawValue) · honest unavailable state")
        }
    }

    private func capture(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
