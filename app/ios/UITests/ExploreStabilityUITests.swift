import XCTest

#if DEBUG
final class ExploreStabilityUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    private let player = "Fixture Player"
    private let team = "Fixture Team"
    private let game = "Fixture Game Team"
    private let statsOnly = "Fixture Stats Only"

    private func candidateID(_ name: String, sport: String, subject: String, key: String) -> String {
        [sport, subject, name.lowercased(), key].map { "\($0.utf8.count):\($0)" }.joined(separator: "|")
    }
    private var playerID: String { candidateID(player, sport: "NHL", subject: "player", key: "player_shots_on_goal") }
    private var teamID: String { candidateID(team, sport: "NFL", subject: "team", key: "team_totals") }
    private var gameID: String { candidateID(game, sport: "NFL", subject: "team", key: "totals") }
    private var statsID: String { candidateID(statsOnly, sport: "NBA", subject: "team", key: "team_totals") }

    private func launch(delay: Int = 0, rich: Bool = false, noOdds: Bool = false, allSports: Bool = false) throws -> XCUIApplication {
        let app = XCUIApplication(), now = Date(), formatter = ISO8601DateFormatter()
        let stamp = formatter.string(from: now.addingTimeInterval(-60))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now))!
        let start = formatter.string(from: now.addingTimeInterval(min(3600, endOfDay.timeIntervalSince(now) / 2)))
        func market(_ id: String, _ name: String, _ sport: String, _ kind: String, _ key: String, _ line: Double) -> [String:Any] {
            ["id": id, "event": "event-" + id, "sport": sport, "kind": kind,
             "name": name, "opponent": key == "totals" ? name : "Fixture Rivals", "label": "Over \(line)", "aliases": "",
             "price": -110, "market": key, "book": "Fixture Book", "quotes": [],
             "updated": stamp, "mode": "live", "line": line, "side": "over", "eventState": "pre",
             "commence": start, "eventLabel": "Fixture Rivals at " + (kind == "prop" ? "Fixture Player Team" : name)]
        }
        var stats: [String:Any] = market("ui-stats", statsOnly, "NBA", "team", "team_totals", 20)
        stats["price"] = 0; stats["book"] = ""; stats.removeValue(forKey: "line"); stats.removeValue(forKey: "updated")
        var markets = [market("ui-player", player, "NHL", "prop", "player_shots_on_goal", 20),
                       market("ui-team", team, "NFL", "team", "team_totals", 24.5),
                       market("ui-game", game, "NFL", "game", "totals", 40), stats]
        // Extra subjects are confined to this explicit offline UI fixture.
        let additions: [(sport: String, name: String, subject: String, key: String, id: String)] = allSports ? [
            ("NFL", "Fixture NFL Player", "player", "player_receptions", "ui-nfl-player"),
            ("NHL", "Fixture NHL Team", "team", "team_totals", "ui-nhl-team"),
            ("NBA", "Fixture NBA Player", "player", "player_points", "ui-nba-player"),
            ("NBA", "Fixture NBA Team", "team", "team_totals", "ui-nba-team"),
            ("MLB", "Fixture MLB Player", "player", "batter_hits", "ui-mlb-player"),
            ("MLB", "Fixture MLB Team", "team", "team_totals", "ui-mlb-team")
        ] : []
        markets += additions.map { market($0.id, $0.name, $0.sport, $0.subject == "player" ? "prop" : "team", $0.key, 20) }
        if noOdds {
            markets = markets.map { original in
                var row = original
                row["price"] = 0; row["book"] = ""
                row.removeValue(forKey: "line"); row.removeValue(forKey: "updated")
                return row
            }
        }
        func proof(_ name: String, _ sport: String, _ subject: String, _ key: String, _ quoteID: String) -> [String:Any] {
            let isPlayer = subject == "player", home = isPlayer ? "Fixture Player Team" : name
            let path = ["NHL": "hockey/nhl", "NFL": "football/nfl", "NBA": "basketball/nba", "MLB": "baseball/mlb"][sport]!
            return ["sport": sport, "subject": subject, "name": name, "providerID": isPlayer ? "100" : "200",
                    "teamName": home, "teamID": "200", "marketKeys": [key], "verifiedAt": stamp,
                    "sourceURL": "https://site.api.espn.com/apis/site/v2/sports/" + path + (isPlayer ? "/teams/200/roster" : "/scoreboard"),
                    "event": ["sport": sport, "providerID": "event-" + quoteID, "home": home, "away": "Fixture Rivals", "start": start, "homeID": "200", "awayID": "201"]]
        }
        let subjects = [proof(player, "NHL", "player", "player_shots_on_goal", "ui-player"),
                        proof(team, "NFL", "team", "team_totals", "ui-team"),
                        proof(game, "NFL", "team", "totals", "ui-game")] + additions.map { proof($0.name, $0.sport, $0.subject, $0.key, $0.id) }
        func observations(playerRows: Bool = false, sport: String = "NFL") -> [[String:Any]] {
            if rich {
                let year = calendar.component(.year, from: now)
                let month = calendar.component(.month, from: now)
                let season = ["NBA", "NHL"].contains(sport) && month >= 9 ? year + 1 : (sport == "NFL" && month <= 2 ? year - 1 : year)
                return (0..<24).map { index in
                    let daysAgo = index < 15 ? index + 2 : 366 + index - 15
                    var row: [String:Any] = ["id": "rich-earlier-\(index)",
                        "date": formatter.string(from: now.addingTimeInterval(-Double(daysAgo) * 86400)),
                        "opponent": index.isMultiple(of: 3) ? "Fixture Rivals" : "Recorded Rival \(index)",
                        "status": "completed", "homeAway": index.isMultiple(of: 2) ? "home" : "away",
                        "season": index < 15 ? season : season - 1]
                    if playerRows { row["value"] = [2, 3, 1, 4][index % 4] }
                    else { row["scored"] = [21, 28, 14, 35][index % 4]; row["allowed"] = [12, 21, 17, 10][index % 4] }
                    return row
                }
            }
            return (0..<6).map { index in
                var row: [String:Any] = ["id": "earlier-\(index)",
                    "date": formatter.string(from: now.addingTimeInterval(-Double(index + 2) * 86400)),
                    "opponent": "Recorded Rival \(index)", "status": "completed"]
                if index == 4 { row["status"] = "void" }
                else if index == 5 { row["status"] = "missing" }
                else if playerRows { row["value"] = [21, 19, 20, 25][index] }
                else { row["scored"] = [28, 20, 10, 24][index]; row["allowed"] = [7, 20, 14, 10][index] }
                return row
            }
        }
        var histories = [player: observations(playerRows: true, sport: "NHL"), team: observations(), game: observations(), statsOnly: observations(sport: "NBA")]
        for row in additions { histories[row.name] = observations(playerRows: row.subject == "player", sport: row.sport) }
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-appearanceMode", "light", "-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launchEnvironment["PLONN_ANALYTICS_MARKETS_FIXTURE"] = String(decoding: try JSONSerialization.data(withJSONObject: markets), as: UTF8.self)
        app.launchEnvironment["PLONN_ANALYTICS_HISTORY_FIXTURE"] = String(decoding: try JSONSerialization.data(withJSONObject: histories), as: UTF8.self)
        app.launchEnvironment["PLONN_ANALYTICS_SUBJECTS_FIXTURE"] = String(decoding: try JSONSerialization.data(withJSONObject: subjects), as: UTF8.self)
        app.launchEnvironment["PLONN_ANALYTICS_HISTORY_DELAY_MS"] = String(delay)
        app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
        app.launch()
        openAnalytics(app)
        return app
    }
    private func openAnalytics(_ app: XCUIApplication) {
        let tab = app.tabBars.buttons["Analytics"]
        XCTAssertTrue(tab.waitForExistence(timeout: 30), app.debugDescription)
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: tab)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed)
        tab.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let selected = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isSelected == true"), object: tab)
        if XCTWaiter.wait(for: [selected], timeout: 3) != .completed {
            tab.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        }
        let confirmed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isSelected == true"), object: tab)
        XCTAssertEqual(XCTWaiter.wait(for: [confirmed], timeout: 5), .completed, app.debugDescription)
        guard tab.isSelected else {
            XCTFail("Analytics tab was not selected; refusing to scroll another tab.\n" + app.debugDescription)
            return
        }
        for _ in 0..<8 where !app.staticTexts["analytics-heading"].isHittable { app.swipeDown() }
        XCTAssertTrue(app.staticTexts["analytics-heading"].waitForExistence(timeout: 10), app.debugDescription)
    }
    private func selectLeague(_ sport: String, in app: XCUIApplication) {
        let button = app.buttons["analytics-league-" + sport]
        XCTAssertTrue(button.waitForExistence(timeout: 10), app.debugDescription)
        button.tap()
        XCTAssertTrue(button.isSelected, "Selected league must be exposed to accessibility.")
    }
    private func search(_ query: String, in app: XCUIApplication, sport: String? = nil) {
        if let sport { selectLeague(sport, in: app) }
        let field = app.textFields["analytics-search"]
        for _ in 0..<8 where !field.isHittable { app.swipeDown() }
        XCTAssertTrue(field.waitForExistence(timeout: 10), app.debugDescription)
        field.tap()
        let count = (field.value as? String)?.count ?? 0
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: count) + query + "\n")
    }
    private func text(_ prefix: String, _ id: String, in app: XCUIApplication) -> XCUIElement {
        app.staticTexts[prefix + id]
    }
    private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        for _ in 0..<8 where !element.isHittable { app.swipeUp() }
        XCTAssertTrue(element.waitForExistence(timeout: 10), app.debugDescription)
    }
    private func capture(_ app: XCUIApplication, name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
    }
    @discardableResult
    private func assertMissingImagery(_ pageID: String, subject: String, in app: XCUIApplication) -> CGRect {
        let image = app.descendants(matching: .any).matching(NSPredicate(format: "identifier == %@", "analytics-insight-imagery-" + pageID)).firstMatch
        XCTAssertTrue(image.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(image.isHittable, app.debugDescription)
        XCTAssertTrue(image.label.contains(subject), "A reused image must retain the correct subject: " + image.label)
        XCTAssertEqual(image.value as? String, "Image unavailable", "Missing source images must expose a clear accessible fallback")
        XCTAssertTrue(image.label.hasSuffix(subject == player ? "headshot" : "team logo"), image.label)
        XCTAssertEqual(image.frame.width, 56, accuracy: 0.5)
        XCTAssertEqual(image.frame.height, 56, accuracy: 0.5)
        return image.frame
    }

    private func assertNoManualGate(_ app: XCUIApplication) {
        XCTAssertFalse(app.textFields["analytics-manual-name"].exists)
        XCTAssertFalse(app.textFields["analytics-selected-line"].exists)
        XCTAssertFalse(app.buttons["analytics-load-history"].exists)
        XCTAssertFalse(app.buttons["analytics-match-odds"].exists)
        XCTAssertFalse(app.segmentedControls.buttons["Players"].exists)
        XCTAssertFalse(app.segmentedControls.buttons["Teams"].exists)
        for sport in ["NFL", "NBA", "MLB", "NHL"] { XCTAssertTrue(app.buttons["analytics-league-" + sport].exists) }
        XCTAssertEqual(app.scrollViews["analytics-leagues"].buttons.count, 4)
        XCTAssertFalse(app.popUpButtons["League"].exists)
    }

    func testFeedAutomaticallyPopulatesMixedSubjectsAndExactCachedMarkets() throws {
        let app = try launch(); defer { app.terminate() }
        assertNoManualGate(app)
        search(player, in: app, sport: "NHL")
        XCTAssertTrue(text("analytics-sample-", playerID, in: app).waitForExistence(timeout: 15), app.debugDescription)
        let playerDenominator = text("analytics-denominator-", "ui-player", in: app)
        reveal(playerDenominator, in: app)
        XCTAssertEqual(playerDenominator.label, "2 hits / 3 decisions")
        // SwiftUI exposes the single quote's children under the outer odds container.
        let quote = app.descendants(matching: .any)["analytics-odds-" + playerID].firstMatch
        XCTAssertTrue(quote.exists, app.debugDescription)
        XCTAssertTrue(quote.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Fixture Book")).firstMatch.exists)
        XCTAssertTrue(quote.staticTexts["Over 20"].exists)
        XCTAssertTrue(quote.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "-110")).firstMatch.exists)
        XCTAssertEqual(quote.staticTexts["analytics-denominator-ui-player"].label, "2 hits / 3 decisions")
        XCTAssertTrue(text("analytics-provenance-", playerID, in: app).exists)
        search("team_totals", in: app, sport: "NFL")
        let teamDenominator = text("analytics-denominator-", "ui-team", in: app)
        reveal(teamDenominator, in: app)
        XCTAssertEqual(teamDenominator.label, "1 hits / 4 decisions", "A team total uses the selected team's score, not both teams' combined score.")
        search(game, in: app, sport: "NFL")
        let gameDenominator = text("analytics-denominator-", "ui-game", in: app)
        reveal(gameDenominator, in: app)
        XCTAssertEqual(gameDenominator.label, "0 hits / 3 decisions", "A game total retains both teams' scores and an exact-line push.")
        capture(app, name: "Full-game total remains visible with zero historical hits")
    }

    func testResultsCarouselsKeepAllFourSportsAndSearchInTheCorrectSubjectRow() throws {
        let app = try launch(allSports: true); defer { app.terminate() }
        let teams = app.scrollViews["analytics-team-props-carousel"]
        let players = app.scrollViews["analytics-player-props-carousel"]
        for (sport, teamName, playerName, playerKey) in [
            ("NFL", team, "Fixture NFL Player", "player_receptions"),
            ("NBA", "Fixture NBA Team", "Fixture NBA Player", "player_points"),
            ("NHL", "Fixture NHL Team", player, "player_shots_on_goal"),
            ("MLB", "Fixture MLB Team", "Fixture MLB Player", "batter_hits")
        ] {
            search("", in: app, sport: sport)
            let teamCandidate = candidateID(teamName, sport: sport, subject: "team", key: "team_totals")
            let playerCandidate = candidateID(playerName, sport: sport, subject: "player", key: playerKey)
            reveal(app.staticTexts["analytics-team-props-heading"], in: app)
            XCTAssertEqual(app.staticTexts["analytics-team-props-heading"].label, "Team Props")
            XCTAssertTrue(teams.staticTexts["analytics-sample-" + teamCandidate].waitForExistence(timeout: 15), app.debugDescription)
            XCTAssertFalse(teams.descendants(matching: .any)["analytics-card-" + playerCandidate].firstMatch.exists)
            reveal(app.staticTexts["analytics-player-props-heading"], in: app)
            XCTAssertEqual(app.staticTexts["analytics-player-props-heading"].label, "Player Props")
            XCTAssertTrue(players.staticTexts["analytics-sample-" + playerCandidate].waitForExistence(timeout: 15), app.debugDescription)
            XCTAssertFalse(players.descendants(matching: .any)["analytics-card-" + teamCandidate].firstMatch.exists)
            XCTAssertTrue(players.descendants(matching: .any)["analytics-card-imagery-" + playerCandidate].firstMatch.exists)
        }
        search("player_shots_on_goal", in: app, sport: "NHL")
        reveal(app.staticTexts["analytics-team-props-empty"], in: app)
        XCTAssertEqual(app.staticTexts["analytics-team-props-empty"].label, "No team results match this search.")
        XCTAssertFalse(teams.exists, "An empty category must not manufacture cards.")
        reveal(players.staticTexts["analytics-denominator-ui-player"], in: app)
        XCTAssertEqual(players.staticTexts["analytics-denominator-ui-player"].label, "2 hits / 3 decisions")
        search("Fixture MLB Player", in: app, sport: "NFL")
        XCTAssertFalse(players.exists, "Search must not escape the selected NFL league.")
        search("", in: app, sport: "NFL")
        reveal(app.staticTexts["analytics-team-props-heading"], in: app)
        XCTAssertTrue(teams.exists)
        let firstTeam = teams.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH %@", "analytics-card-")).allElementsBoundByIndex.first { $0.identifier == "analytics-card-" + self.teamID || $0.identifier == "analytics-card-" + self.gameID }!
        XCTAssertTrue(firstTeam.exists, app.debugDescription)
        reveal(firstTeam.staticTexts.matching(NSPredicate(format: "identifier BEGINSWITH %@", "analytics-sample-")).firstMatch, in: app)
        let firstX = firstTeam.frame.minX
        teams.swipeLeft()
        XCTAssertLessThan(firstTeam.frame.minX, firstX - 30, "The team results row must scroll horizontally.")
        reveal(app.staticTexts["analytics-player-props-heading"], in: app)
        XCTAssertTrue(players.exists)
        XCTAssertFalse(app.staticTexts[statsOnly].exists, "Clearing search restores exact recorded-odds eligibility.")
        capture(app, name: "Offline fixture: separate Team Props and Player Props rows across four sports")
    }

    func testStatsOnlyKeepsRecordedHistoryAndShowsUnavailableOdds() throws {
        let app = try launch(); defer { app.terminate() }
        search(statsOnly, in: app, sport: "NBA")
        XCTAssertTrue(text("analytics-sample-", statsID, in: app).waitForExistence(timeout: 15), app.debugDescription)
        XCTAssertTrue(text("analytics-mean-", statsID, in: app).exists)
        let odds = app.descendants(matching: .any)["analytics-odds-" + statsID].firstMatch
        reveal(odds, in: app)
        XCTAssertTrue(odds.isHittable, app.debugDescription)
        XCTAssertTrue(odds.label.localizedCaseInsensitiveContains("unavailable"), odds.label)
        XCTAssertTrue(odds.label.localizedCaseInsensitiveContains("recorded statistics remain available"), odds.label)
        XCTAssertFalse(odds.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Fixture Book")).firstMatch.exists)
        XCTAssertFalse(app.descendants(matching: .any)["analytics-quote-ui-stats"].firstMatch.exists)
        XCTAssertFalse(text("analytics-denominator-", "ui-stats", in: app).exists)
        XCTAssertTrue(text("analytics-provenance-", statsID, in: app).exists)
        capture(app, name: "Recorded team statistics with unavailable cached odds")
    }

    func testSearchClearingAndReturningReuseRecordedResults() throws {
        let app = try launch(); defer { app.terminate() }
        search(player, in: app, sport: "NHL")
        let sample = text("analytics-sample-", playerID, in: app)
        XCTAssertTrue(sample.waitForExistence(timeout: 15))
        let recorded = sample.label
        search("No such fixture subject", in: app)
        XCTAssertFalse(sample.exists)
        search("", in: app)
        XCTAssertFalse(app.buttons["Clear search"].exists, "Clearing returns the unfiltered feed.")
        XCTAssertTrue(text("analytics-sample-", playerID, in: app).waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertFalse(app.staticTexts[statsOnly].exists, "Clearing search restores only subjects with matching odds.")
        search(player, in: app, sport: "NHL")
        XCTAssertTrue(sample.waitForExistence(timeout: 5))
        XCTAssertEqual(sample.label, recorded)
        let detail = app.descendants(matching: .any)["analytics-history-detail-" + playerID].firstMatch
        reveal(detail, in: app)
        XCTAssertTrue(detail.isHittable); detail.tap()
        let recordedOpponent = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Recorded Rival 0")).firstMatch
        reveal(recordedOpponent, in: app)
        app.tabBars.buttons["LLM"].tap(); openAnalytics(app)
        search(player, in: app, sport: "NHL")
        XCTAssertEqual(sample.label, recorded)
        capture(app, name: "Search reuses retained recorded player history")
    }

    func testLeavingDuringHistoryLoadAndRelaunchRepresentsFreshFeed() throws {
        let app = try launch(delay: 2000); defer { app.terminate() }
        app.tabBars.buttons["LLM"].tap()
        openAnalytics(app)
        search(player, in: app, sport: "NHL")
        XCTAssertTrue(text("analytics-sample-", playerID, in: app).waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertEqual(app.state, .runningForeground)
        app.terminate(); app.launch(); openAnalytics(app)
        search("team_totals", in: app, sport: "NFL")
        XCTAssertTrue(text("analytics-sample-", teamID, in: app).waitForExistence(timeout: 20), app.debugDescription)
        assertNoManualGate(app)
        XCTAssertEqual(app.state, .runningForeground)
        capture(app, name: "Automatic feed survives leaving during load and reopening")
    }

    func testTwelveTabAndSearchCyclesPreserveMixedFeedStability() throws {
        let app = try launch(); defer { app.terminate() }
        for index in 0..<12 {
            let subject = index.isMultiple(of: 2) ? player : team
            let id = index.isMultiple(of: 2) ? playerID : teamID
            search(subject, in: app, sport: index.isMultiple(of: 2) ? "NHL" : "NFL")
            XCTAssertTrue(text("analytics-sample-", id, in: app).waitForExistence(timeout: 15), "History missing during cycle \(index)")
            search("", in: app)
            app.tabBars.buttons[["LLM", "Models", "My Models"][index % 3]].tap()
            openAnalytics(app)
            XCTAssertEqual(app.state, .runningForeground, "Feed stopped during cycle \(index)")
        }
        search(game, in: app, sport: "NFL")
        let denominator = text("analytics-denominator-", "ui-game", in: app)
        reveal(denominator, in: app)
        XCTAssertEqual(denominator.label, "0 hits / 3 decisions")
        assertNoManualGate(app)
        capture(app, name: "Mixed feed after twelve tab and search cycles")
    }

    func testDefaultNFLAndSelectedNHLKeepTheirOwnTeamAndPlayerRows() throws {
        let app = try launch(); defer { app.terminate() }
        assertNoManualGate(app)
        XCTAssertTrue(app.buttons["analytics-league-NFL"].isSelected)
        let carousel = app.scrollViews["analytics-trend-carousel"]
        XCTAssertTrue(carousel.waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertFalse(carousel.staticTexts[player].exists)
        search("", in: app, sport: "NHL")
        XCTAssertTrue(carousel.staticTexts[player].waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertFalse(carousel.staticTexts[team].exists)
        XCTAssertTrue(app.textFields["analytics-search"].isHittable)
        XCTAssertTrue(app.staticTexts["analytics-fixture-banner"].isHittable)
        capture(app, name: "Offline fixture: single league row and NHL player results")
    }

    func testNoMatchingOddsShowsEmptyDefaultFeedButSearchRetainsGeneralStatistics() throws {
        let app = try launch(noOdds: true); defer { app.terminate() }
        let empty = app.staticTexts["analytics-no-matching-odds"]
        XCTAssertTrue(empty.waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertFalse(app.scrollViews["analytics-trend-carousel"].exists)
        XCTAssertFalse(app.scrollViews["analytics-team-props-carousel"].exists)
        XCTAssertFalse(app.scrollViews["analytics-player-props-carousel"].exists)
        for id in [playerID, teamID, gameID, statsID] {
            XCTAssertFalse(app.descendants(matching: .any)["analytics-card-" + id].firstMatch.exists)
        }
        XCTAssertTrue(app.textFields["analytics-search"].isHittable)
        search(statsOnly, in: app, sport: "NBA")
        XCTAssertTrue(text("analytics-sample-", statsID, in: app).waitForExistence(timeout: 15), app.debugDescription)
        XCTAssertTrue(app.scrollViews["analytics-trend-carousel"].exists)
        XCTAssertTrue(app.scrollViews["analytics-team-props-carousel"].exists)
        XCTAssertFalse(app.scrollViews["analytics-player-props-carousel"].exists)
        XCTAssertFalse(empty.exists)
        let odds = app.descendants(matching: .any)["analytics-odds-" + statsID].firstMatch
        reveal(odds, in: app)
        XCTAssertTrue(odds.label.localizedCaseInsensitiveContains("unavailable"), odds.label)
        search("", in: app)
        XCTAssertTrue(empty.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertFalse(app.scrollViews["analytics-trend-carousel"].exists)
        capture(app, name: "Offline fixture: honest empty default feed when matching odds are absent")
    }

    func testMissingImagesRemainMatchedAndStableAfterSearchReuse() throws {
        // Deliberately omit image URLs: fake fixture IDs must never download someone else's photo/logo.
        let app = try launch(); defer { app.terminate() }
        var initialSizes: [String: CGSize] = [:]
        for _ in 0..<3 {
            // Search includes team/context terms, so "Fixture Team" intentionally matches
            // several cards. Use unique sport + market queries to bring each subject first.
            for (name, id, query) in [(player, playerID, "NHL"), (team, teamID, "NFL team_totals"), (statsOnly, statsID, "NBA")] {
                search(query, in: app, sport: String(query.prefix(3)))
                XCTAssertTrue(text("analytics-sample-", id, in: app).waitForExistence(timeout: 15), app.debugDescription)
                let frame = assertMissingImagery(id + "|recorded-statistics", subject: name, in: app)
                if let size = initialSizes[id] {
                    XCTAssertEqual(frame.width, size.width, accuracy: 0.5, "Image slot width changed after card reuse")
                    XCTAssertEqual(frame.height, size.height, accuracy: 0.5, "Image slot height changed after card reuse")
                } else { initialSizes[id] = frame.size }
                XCTAssertTrue(app.staticTexts["analytics-fixture-banner"].exists)
            }
        }
        capture(app, name: "Offline fixture: missing imagery stays matched and stable after search reuse")
    }

    func testCarouselPreviewShowsEvidenceVariety() throws {
        let app = try launch(rich: true); defer { app.terminate() }
        search("team_totals", in: app, sport: "NFL")
        let carousel = app.scrollViews["analytics-trend-carousel"]
        XCTAssertTrue(carousel.waitForExistence(timeout: 20), app.debugDescription)
        let firstPage = teamID + "|" + teamID + "|recentBaseline"
        let firstSample = carousel.staticTexts["analytics-insight-sample-" + firstPage]
        XCTAssertTrue(firstSample.waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertEqual(firstSample.label, "24 games in this comparison")
        assertMissingImagery(firstPage, subject: team, in: app)
        XCTAssertTrue(app.textFields["analytics-search"].isHittable)
        capture(app, name: "Offline fixture: last-five-games comparison within NFL")
        for kind in ["consistency", "homeAway", "seasonBaseline"] {
            carousel.coordinate(withNormalizedOffset: CGVector(dx: 0.80, dy: 0.50))
                .press(forDuration: 0.1, thenDragTo: carousel.coordinate(withNormalizedOffset: CGVector(dx: 0.20, dy: 0.50)), withVelocity: .slow, thenHoldForDuration: 0)
            let page = teamID + "|" + teamID + "|" + kind
            let headline = carousel.staticTexts["analytics-insight-headline-" + page]
            XCTAssertTrue(headline.waitForExistence(timeout: 5), app.debugDescription)
            XCTAssertTrue(headline.isHittable, app.debugDescription)
            XCTAssertFalse(headline.label.contains("IQR")); XCTAssertFalse(headline.label.contains("Median"))
            assertMissingImagery(page, subject: team, in: app)
        }
        capture(app, name: "Offline fixture: plain-language NFL comparisons")
    }
}
#endif
