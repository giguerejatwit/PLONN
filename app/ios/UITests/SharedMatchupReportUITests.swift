import XCTest

/// Launch-only, synthetic contract responses. Never uses the signed-in simulator or a live transport.
final class SharedMatchupReportUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testLiveMatchupStatsAcrossSportsWithoutSignInOrPregameReports() throws {
        for (sport,key) in [("NFL","totalYards"),("NBA","totalRebounds"),("MLB","batting.hits"),("NHL","shotsTotal"),("Soccer:eng.1","shotsOnTarget")] {
            let app = try launchLiveMatchup(sport:sport,key:key)
            XCTAssertTrue(app.staticTexts["live-matchup-heading"].waitForExistence(timeout:10))
            XCTAssertEqual(app.staticTexts["live-matchup-heading"].label,"Live matchup")
            XCTAssertTrue(app.descendants(matching:.any)["live-matchup-stat-"+key].firstMatch.exists)
            XCTAssertTrue(app.staticTexts["live-matchup-odds-unavailable"].exists)
            XCTAssertFalse(app.staticTexts["matchup-outlook-selection"].exists)
            app.terminate()
        }
    }

    func testFinalAndStaleMatchupsKeepTheirStatsVisible() throws {
        let final = try launchLiveMatchup(sport:"NFL",key:"totalYards",final:true)
        XCTAssertTrue(final.staticTexts["live-matchup-heading"].waitForExistence(timeout:10))
        XCTAssertEqual(final.staticTexts["live-matchup-heading"].label,"Final matchup")
        XCTAssertTrue(final.staticTexts["live-matchup-score"].label.contains("won"))
        XCTAssertFalse(final.staticTexts["live-matchup-odds-unavailable"].exists)
        final.terminate()
        let stale = try launchLiveMatchup(sport:"NFL",key:"totalYards",failed:true)
        XCTAssertTrue(stale.staticTexts["live-matchup-freshness"].waitForExistence(timeout:10))
        XCTAssertTrue(stale.staticTexts["live-matchup-freshness"].label.contains("Updates paused"))
        XCTAssertEqual(stale.staticTexts["live-matchup-heading"].label,"Matchup update")
        XCTAssertTrue(stale.descendants(matching:.any)["live-matchup-stat-totalYards"].firstMatch.exists)
        stale.terminate()
    }

    private func launchLiveMatchup(sport:String,key:String,final:Bool=false,failed:Bool=false) throws -> XCUIApplication {
        let event:[String:Any] = ["sport":sport,"eventID":"espn:100","homeTeam":"Home Team","awayTeam":"Away Team","startISO":"2026-10-19T00:00:00Z"]
        let stat:[String:Any] = key == "batting.hits" ? ["name":"batting","stats":[["name":"hits","displayValue":"4"]]] : ["name":key,"displayValue":"4"]
        let summary:[String:Any] = ["header":["id":"100","competitions":[["id":"100","status":["type":["state":final ? "post":"in","completed":final,"detail":final ? "Final":"In progress"]],"competitors":[["homeAway":"home","score":"7","team":["id":"1","displayName":"Home Team","abbreviation":"HOME"]],["homeAway":"away","score":"0","team":["id":"2","displayName":"Away Team","abbreviation":"AWAY"]]]]]],"boxscore":["teams":[["team":["id":"1"],"statistics":[stat]],["team":["id":"2"],"statistics":[stat]]]]]
        let fixture:[String:Any] = ["event":event,"summary":summary,"failed":failed,"checkedAt":"2026-10-19T01:00:00Z"]
        let app=XCUIApplication()
        app.launchEnvironment["PLONN_LIVE_MATCHUP_PREVIEW"] = String(data:try JSONSerialization.data(withJSONObject:fixture),encoding:.utf8)!
        app.launchArguments += ["-backendURL","https://127.0.0.1:9"]
        app.launch()
        return app
    }

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

    func testStrongestEdgeLeadsWithLineAndOddsAcrossSports() throws {
        for (sport, prop) in [("NFL", "player_rush_yds"), ("NBA", "player_rebounds"), ("MLB", "batter_hits"), ("NHL", "player_shots_on_goal")] {
            let response = try example(sport)
            let event = try XCTUnwrap(response["event"] as? [String: Any])
            let home = try XCTUnwrap(event["homeTeam"] as? String), away = try XCTUnwrap(event["awayTeam"] as? String)
            let start = try XCTUnwrap(event["startISO"] as? String)
            let market: [String: Any] = ["id": "edge-fixture", "event": "provider-fixture", "sport": sport, "kind": "prop", "name": "Fixture Player", "opponent": away, "label": prop, "aliases": "", "p": 0.68, "price": 125, "market": prop, "book": "Fixture Book", "quotes": [], "updated": "2026-10-13T12:59:30Z", "mode": "live", "line": 2.5, "eventLabel": away + " at " + home, "modelValidated": true, "modelUpdated": "2026-10-13T12:59:30Z", "modelVersion": "synthetic-test-only", "eventState": "pre", "commence": start, "side": "under"]
            let fixture: [String: Any] = ["response": response, "statusCode": 200, "now": "2026-10-13T13:00:00Z", "markets": [market]]
            let app = XCUIApplication()
            app.launchEnvironment["PLONN_SHARED_REPORT_FIXTURE"] = String(decoding: try JSONSerialization.data(withJSONObject: fixture), as: UTF8.self)
            app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
            app.launch()
            let selection = app.staticTexts["matchup-outlook-selection"]
            XCTAssertTrue(selection.waitForExistence(timeout: 15), app.debugDescription)
            XCTAssertTrue(selection.label.contains("Under 2.5")); XCTAssertTrue(selection.label.contains("+125"))
            XCTAssertTrue(selection.label.contains("Fixture Book"))
            XCTAssertEqual(app.staticTexts["matchup-outlook-lead-title"].label, "Strongest model edge")
            XCTAssertFalse(app.staticTexts["matchup-outlook-alternatives-heading"].exists, "A single supported angle must not add filler")
            XCTAssertFalse(app.staticTexts["shared-report-outlook"].exists, "Supporting narrative starts collapsed")
            let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = sport + " outlook edge"; shot.lifetime = .keepAlways; add(shot)
            app.terminate()
        }
    }

    func testMultipleSupportedDirectionsAcrossSports() throws {
        for (sport, prop) in [("NFL", "player_rush_yds"), ("NBA", "player_rebounds"), ("MLB", "batter_hits"), ("NHL", "player_shots_on_goal")] {
            let response = try example(sport)
            let event = try XCTUnwrap(response["event"] as? [String: Any])
            let home = try XCTUnwrap(event["homeTeam"] as? String), away = try XCTUnwrap(event["awayTeam"] as? String)
            let start = try XCTUnwrap(event["startISO"] as? String)
            func row(_ id: String, kind: String, name: String, key: String, line: Double, probability: Double, price: Double = -110, book: String = "Fixture Book") -> [String: Any] {
                ["id": id, "event": "provider-fixture", "sport": sport, "kind": kind, "name": name, "opponent": away, "label": key, "aliases": "", "p": probability, "price": price, "market": key, "book": book, "quotes": [], "updated": "2026-10-13T12:59:30Z", "mode": "live", "line": line, "eventLabel": away + " at " + home, "modelValidated": true, "modelUpdated": "2026-10-13T12:59:30Z", "modelVersion": "synthetic-test-only", "eventState": "pre", "commence": start, "side": "over"]
            }
            let rows = [
                row("player", kind: "prop", name: "Fixture Player", key: prop, line: 2.5, probability: 0.68, price: 125),
                row("duplicate-book", kind: "prop", name: "Fixture Player", key: prop, line: 2.5, probability: 0.68, book: "Other Book"),
                row("team-total", kind: "team", name: home, key: "team_totals", line: 20.5, probability: 0.62),
                row("total", kind: "team", name: "Over", key: "totals", line: 40.5, probability: 0.58),
                row("fourth", kind: "prop", name: "Another Player", key: prop, line: 3.5, probability: 0.55)]
            let fixture: [String: Any] = ["response": response, "statusCode": 200, "now": "2026-10-13T13:00:00Z", "markets": rows]
            let app = XCUIApplication()
            app.launchEnvironment["PLONN_SHARED_REPORT_FIXTURE"] = String(decoding: try JSONSerialization.data(withJSONObject: fixture), as: UTF8.self)
            app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
            app.launch()
            XCTAssertTrue(app.staticTexts["matchup-outlook-selection"].waitForExistence(timeout: 15), app.debugDescription)
            XCTAssertTrue(app.staticTexts["matchup-outlook-selection"].label.contains("Fixture Player · Over 2.5"))
            XCTAssertTrue(app.staticTexts["matchup-outlook-selection"].label.contains("+125"))
            let alternatives = app.staticTexts.matching(identifier: "matchup-outlook-alternative")
            XCTAssertEqual(alternatives.count, 2)
            XCTAssertTrue(alternatives.element(boundBy: 0).label.contains("2. " + home + " · Over 20.5"))
            XCTAssertTrue(alternatives.element(boundBy: 1).label.contains("3. Over 40.5"))
            XCTAssertTrue(alternatives.allElementsBoundByIndex.allSatisfy { $0.label.contains("-110 · Fixture Book") })
            XCTAssertEqual(app.staticTexts.matching(identifier: "matchup-outlook-alternative-reason").count, 2)
            if !alternatives.element(boundBy: 1).isHittable { app.swipeUp() }
            XCTAssertTrue(app.staticTexts.matching(identifier: "matchup-outlook-correlation").firstMatch.exists)
            XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Another Player")).firstMatch.exists)
            let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = sport + " multiple directions"; shot.lifetime = .keepAlways; add(shot)
            app.terminate()
        }
    }

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
            XCTAssertTrue(app.staticTexts["matchup-outlook-no-edge"].waitForExistence(timeout: 15) || app.staticTexts["matchup-outlook-selection"].exists)
            app.buttons["Supporting context"].tap()
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
