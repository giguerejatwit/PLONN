import XCTest

#if !DEBUG
final class ReleaseReadinessUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    private func launchRelease() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-appearanceMode", "light"]
        // A Release app must ignore development-only input, even if supplied.
        app.launchEnvironment["PLONN_ASK_RECIPES_FIXTURE"] = "[{\"name\":\"Release debug sentinel\"}]"
        app.launchEnvironment["PLONN_ASK_MARKETS_FIXTURE"] = "[]"
        app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
        app.launch()
        return app
    }

    func testReleaseChatHasHonestAvailabilityAndNoRuntimeDiagnostics() {
        let app = launchRelease()
        defer { app.terminate() }
        let tab = app.tabBars.buttons["Ask PLONN"]
        XCTAssertTrue(tab.waitForExistence(timeout: 30))
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: tab)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed)
        tab.tap()
        let status = app.staticTexts["ask-plonn-provider-status"]
        XCTAssertTrue(status.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(status.label.contains("Sign in to use Gemma4 26B"))
        XCTAssertFalse(status.label.contains("Local prototype"))
        app.buttons["ask-plonn-start-chat"].tap()
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "AI chat is unavailable")).firstMatch.exists)
        app.buttons["ask-plonn-send"].tap()
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "AI chat is unavailable")).firstMatch.waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts["Release debug sentinel"].exists)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Release chat availability"; screenshot.lifetime = .keepAlways; add(screenshot)
    }

    func testReleaseGuestCanReachAccountWithoutUnconfiguredSocialButtons() {
        let app = launchRelease()
        defer { app.terminate() }
        let tab = app.tabBars.buttons["Ask PLONN"]
        XCTAssertTrue(tab.waitForExistence(timeout:30))
        let ready = XCTNSPredicateExpectation(predicate:NSPredicate(format:"isHittable == true"),object:tab)
        XCTAssertEqual(XCTWaiter.wait(for:[ready],timeout:30),.completed)
        tab.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
        let signIn = app.buttons["ask-plonn-sign-in"]
        if !signIn.waitForExistence(timeout:3) {tab.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()}
        XCTAssertTrue(signIn.waitForExistence(timeout:10),app.debugDescription)
        signIn.tap()
        XCTAssertTrue(app.textFields["Email"].waitForExistence(timeout:5),app.debugDescription)
        XCTAssertTrue(app.secureTextFields["Password"].exists)
        XCTAssertFalse(app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Google")).firstMatch.exists)
        XCTAssertFalse(app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Apple")).firstMatch.exists)
        app.buttons["ask-plonn-account-done"].tap()
        XCTAssertTrue(signIn.waitForExistence(timeout:5))
        XCTAssertFalse(app.buttons["gemma-consent-allow"].exists)
    }

    func testReleaseSettingsHidesDevelopmentLedgerAndKeepsAccountControls() {
        let app = launchRelease()
        defer { app.terminate() }
        let settings = app.buttons["Account and settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 30), app.debugDescription)
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: settings)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed)
        settings.tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["Recorded results"].exists)
        XCTAssertFalse(app.buttons["Purchases & access"].exists)
        XCTAssertTrue(app.buttons["Account"].exists)
        XCTAssertTrue(app.buttons["Privacy & data"].exists)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Release settings"; screenshot.lifetime = .keepAlways; add(screenshot)
    }
}
#endif

final class AskPLONNUITests: XCTestCase {
    private func openAsk(recipeFixture: String? = nil, marketFixture: String? = nil) -> XCUIApplication {
        let app = XCUIApplication()
        // The temporary argument domain keeps odds refreshes away from paid providers.
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-appearanceMode", "light"]
        app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
        if let recipeFixture { app.launchEnvironment["PLONN_ASK_RECIPES_FIXTURE"] = recipeFixture }
        if let marketFixture { app.launchEnvironment["PLONN_ASK_MARKETS_FIXTURE"] = marketFixture }
        app.launch()
        let tab = app.tabBars.buttons["Ask PLONN"]
        XCTAssertTrue(tab.waitForExistence(timeout: 30), app.debugDescription)
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: tab)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed)
        tab.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let heading = app.staticTexts["Ask PLONN"]
        if !heading.waitForExistence(timeout: 3) {
            // The existing launch transition can expose tab geometry before it settles.
            tab.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        }
        XCTAssertTrue(heading.waitForExistence(timeout: 10), app.debugDescription)
        return app
    }

    private func capture(_ app: XCUIApplication, name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func revealTopic(_ app: XCUIApplication) {
        let topic = app.buttons["Explain model"]
        for _ in 0..<8 where !topic.isHittable { app.swipeUp() }
        XCTAssertTrue(topic.isHittable)
    }

    private func priceFixture(validated: Bool = true, probability: Double = 0.6, quoteAge: TimeInterval = 0, closedGame: Bool = false) throws -> String {
        let now = Date(), stamp = ISO8601DateFormatter().string(from: now.addingTimeInterval(-quoteAge))
        let selection: [String: Any] = ["id":"ui-price-edge", "event":"ui-price-event", "sport":"NBA", "kind":"prop",
            "name":"Price Fixture Player", "opponent":"Fixture Opponent", "label":"Over 20.5 points", "aliases":"",
            "p":probability, "price":-110, "market":"player_points", "book":"Fixture Book A",
            "quotes":[["book":"Fixture Book A","price":-110,"updated":stamp],
                      ["book":"Fixture Book B","price":120,"updated":stamp],
                      ["book":"Untimestamped Book","price":5000]],
            "updated":stamp,"mode":"live","line":20.5,"side":"over",
            "modelValidated":validated,"modelVersion":"fixture-v1","modelUpdated":stamp,"eventState":"pre",
            "commence":ISO8601DateFormatter().string(from: now.addingTimeInterval(3600))]
        var differentLine = selection
        differentLine["id"] = "ui-other-line"; differentLine["name"] = "Different Fixture Player"
        differentLine["line"] = 21.5; differentLine["price"] = 9000; differentLine["quotes"] = []
        var rows = [selection,differentLine]
        if closedGame {
            var closed = selection
            closed["id"] = "ui-closed-under"; closed["side"] = "under"; closed["label"] = "Under 20.5 points"
            closed["eventState"] = "final"; rows.append(closed)
        }
        let data = try JSONSerialization.data(withJSONObject:rows,options:[.sortedKeys])
        return String(decoding:data,as:UTF8.self)
    }
    private func attachPriceFixture(_ app: XCUIApplication) {
        app.buttons["ask-plonn-chat-context"].tap()
        app.buttons["ask-plonn-saved-contexts"].tap()
        app.buttons["ask-plonn-context"].tap()
        let selection = app.buttons.matching(NSPredicate(format:"label CONTAINS %@ AND label CONTAINS %@","Price Fixture Player","Over 20.5")).firstMatch
        XCTAssertTrue(selection.waitForExistence(timeout:5),app.debugDescription)
        selection.tap(); app.buttons["ask-plonn-context-done"].tap()
        // Typing an edge request exercises routing without selecting a topic.
        let question = app.textFields["ask-plonn-question"]
        question.tap(); question.typeText("What is the edge here?")
        app.buttons["ask-plonn-send"].tap()
    }
    private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        for _ in 0..<6 where !element.isHittable { app.swipeDown() }
        XCTAssertTrue(element.isHittable,app.debugDescription)
    }
    func testBestPriceAndPositiveModelEdgeWithTemporaryQuotes() throws {
        let app = openAsk(marketFixture:try priceFixture())
        attachPriceFixture(app)
        let best = app.staticTexts["sportsbook-best-price"]
        XCTAssertTrue(best.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertEqual(best.label,"+120")
        XCTAssertEqual(app.staticTexts["sportsbook-best-book"].label,"Fixture Book B")
        XCTAssertEqual(app.staticTexts["sportsbook-model-edge"].label,"+14.55 pp")
        XCTAssertEqual(app.staticTexts["sportsbook-implied-probability"].label,"Price implied probability: 45.45%")
        XCTAssertEqual(app.staticTexts["sportsbook-model-probability"].label,"Validated model probability: 60.00%")
        XCTAssertTrue(app.staticTexts["sportsbook-coverage"].label.contains("2 timestamped books"))
        reveal(best,in:app); capture(app,name:"Best timestamped book price and positive validated model edge")
        app.buttons["Recorded comparison summary"].tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout:5))
        XCTAssertTrue(answer.label.contains("+14.55 percentage points"),answer.label)
        XCTAssertTrue(answer.label.contains("Fixture Book B"),answer.label)
    }
    func testNegativeModelEdgeRemainsSigned() throws {
        let app = openAsk(marketFixture:try priceFixture(probability:0.4))
        attachPriceFixture(app)
        let edge = app.staticTexts["sportsbook-model-edge"]
        XCTAssertTrue(edge.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertEqual(edge.label,"-5.45 pp")
        XCTAssertEqual(app.staticTexts["sportsbook-best-price"].label,"+120")
        reveal(edge,in:app); capture(app,name:"Negative model edge stays visible alongside the best price")
    }
    func testPriceAvailableWithoutValidatedModelEdge() throws {
        let app = openAsk(marketFixture:try priceFixture(validated:false,probability:0.99))
        attachPriceFixture(app)
        let best = app.staticTexts["sportsbook-best-price"]
        XCTAssertTrue(best.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertEqual(best.label,"+120")
        XCTAssertEqual(app.staticTexts["sportsbook-model-edge"].label,"Unavailable")
        XCTAssertTrue(app.staticTexts["sportsbook-edge-unavailable"].label.contains("validated"))
        XCTAssertFalse(app.staticTexts["sportsbook-model-probability"].exists,"An unvalidated probability must not look like a live edge.")
        reveal(best,in:app); capture(app,name:"Best price available while unvalidated model edge is withheld")
    }
    func testDefiningModelEdgeNeedsNoResearchAttachment() {
        let app = openAsk()
        let question = app.textFields["ask-plonn-question"]
        question.tap(); question.typeText("What is Model Edge?")
        app.buttons["ask-plonn-send"].tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertTrue(answer.label.contains("hypothetical"),answer.label)
        XCTAssertTrue(answer.label.contains("percentage points"),answer.label)
        XCTAssertFalse(app.buttons["ask-plonn-choose-context"].exists)
        XCTAssertFalse(app.staticTexts["sportsbook-best-price"].exists,"An educational example must not look like a current price.")
        capture(app,name:"Model-edge definitions remain ordinary chat without a research attachment")
    }
    func testStaleQuotesHaveNoBestPriceOrEdge() throws {
        let app = openAsk(marketFixture:try priceFixture(quoteAge:300))
        attachPriceFixture(app)
        let reason = app.staticTexts["sportsbook-price-unavailable"]
        XCTAssertTrue(reason.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertFalse(app.staticTexts["sportsbook-best-price"].exists)
        XCTAssertEqual(app.staticTexts["sportsbook-model-edge"].label,"Model edge unavailable")
        XCTAssertTrue(reason.label.contains("No current timestamped prices"),reason.label)
        reveal(reason,in:app); capture(app,name:"Stale sportsbook quotes cannot produce a best price or model edge")
    }
    func testAnotherOutcomeReportingGameClosedWithholdsComparison() throws {
        let app = openAsk(marketFixture:try priceFixture(closedGame:true))
        attachPriceFixture(app)
        let reason = app.staticTexts["sportsbook-price-unavailable"]
        XCTAssertTrue(reason.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertFalse(app.staticTexts["sportsbook-best-price"].exists)
        XCTAssertTrue(reason.label.contains("closed"),reason.label)
        XCTAssertEqual(app.staticTexts["sportsbook-model-edge"].label,"Model edge unavailable")
        reveal(reason,in:app); capture(app,name:"A closed game observation on another outcome prevents a price or edge")
    }
    func testDisplayedComparisonExpiresWithoutRefreshingOdds() throws {
        let app = openAsk(marketFixture:try priceFixture(quoteAge:90))
        attachPriceFixture(app)
        XCTAssertTrue(app.staticTexts["sportsbook-best-price"].waitForExistence(timeout:10),app.debugDescription)
        let reason = app.staticTexts["sportsbook-price-unavailable"]
        XCTAssertTrue(reason.waitForExistence(timeout:40),app.debugDescription)
        XCTAssertTrue(reason.label.contains("expired"),reason.label)
        XCTAssertFalse(app.staticTexts["sportsbook-best-price"].exists)
        XCTAssertEqual(app.staticTexts["sportsbook-model-edge"].label,"Model edge unavailable")
        reveal(reason,in:app); capture(app,name:"A visible price and model edge expire without a provider refresh")
    }

    func testAskTabAndMissingContext() {
        let app = openAsk()
        XCTAssertFalse(app.tabBars.buttons["Builder"].exists)
        XCTAssertTrue(app.staticTexts["ask-plonn-provider-status"].label.contains("aren’t connected"))
        let send = app.buttons["ask-plonn-send"]
        XCTAssertTrue(send.exists)
        XCTAssertTrue(send.isHittable,"The composer should be visible without scrolling.")
        XCTAssertFalse(send.isEnabled)
        XCTAssertFalse(app.textFields["ask-plonn-player-name"].exists, "Setup belongs in a sheet, not above the chat.")
        capture(app, name: "LLM introduction")
        let question = app.textFields["ask-plonn-question"]
        question.tap(); question.typeText("What does my model setup tell me?")
        for _ in 0..<4 where !send.isHittable { app.swipeUp() }
        XCTAssertTrue(send.isEnabled)
        send.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(answer.label.lowercased().contains("model"), answer.label)
        XCTAssertTrue(answer.label.lowercased().contains("optional"), answer.label)
        XCTAssertTrue(app.buttons["ask-plonn-choose-context"].exists, "Specific saved results can offer an optional attachment.")
        XCTAssertFalse(app.staticTexts["ask-plonn-welcome"].exists)
        app.buttons["ask-plonn-choose-context"].tap()
        XCTAssertTrue(app.staticTexts["Chat context"].waitForExistence(timeout: 5))
        app.buttons["ask-plonn-context-done"].tap()
        XCTAssertEqual(question.value as? String, "What does my model setup tell me?", "Choosing context should retain the question.")
        capture(app, name: "Missing model evidence")
    }

    func testGeneralQuestionNeedsNoAttachmentWhenModelIsOffline() {
        let app = openAsk()
        XCTAssertTrue(app.buttons["ask-plonn-chat-context"].label.contains("optional"))
        app.buttons["ask-plonn-start-chat"].tap()
        XCTAssertFalse(app.staticTexts["ask-plonn-answer"].exists)
        app.buttons["ask-plonn-send"].tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertFalse(app.buttons["ask-plonn-choose-context"].exists, "An offline model must not turn a general question into mandatory setup.")
        XCTAssertFalse(app.textFields["ask-plonn-player-name"].exists)
        XCTAssertFalse(app.buttons["ask-plonn-sources"].exists)
        XCTAssertTrue(app.staticTexts["ask-plonn-answer-provider"].label.contains("unavailable"))
        XCTAssertTrue(app.buttons["ask-plonn-follow-up"].exists)
        capture(app, name: "General chat offline disclosure without mandatory setup")
    }

    func testChatGreetingAndNewChat() {
        let app = openAsk()
        let question = app.textFields["ask-plonn-question"]
        question.tap(); question.typeText("Hello")
        app.buttons["ask-plonn-send"].tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(answer.label.hasPrefix("Hi!"), answer.label)
        XCTAssertTrue(app.staticTexts["ask-plonn-user-message"].label == "Hello")
        XCTAssertFalse(app.staticTexts["ask-plonn-answer-provider"].exists, "A local greeting must not claim model inference.")
        XCTAssertFalse(app.textFields["ask-plonn-player-name"].exists)
        capture(app, name: "Conversation starts with a greeting")
        app.buttons["ask-plonn-new-chat"].tap()
        XCTAssertTrue(app.staticTexts["ask-plonn-welcome"].waitForExistence(timeout: 5))
        XCTAssertFalse(answer.exists)
        XCTAssertFalse(app.buttons["ask-plonn-send"].isEnabled)
    }

    func testExplainModelStarterOpensSavedContextWithoutPlayerForm() {
        let app = openAsk()
        app.buttons["ask-plonn-start-model"].tap()
        XCTAssertTrue(app.staticTexts["Chat context"].waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertFalse(app.textFields["ask-plonn-player-name"].exists)
        XCTAssertTrue(app.buttons["ask-plonn-context"].exists, "The model starter should expose saved contexts directly.")
        app.buttons["ask-plonn-context-done"].tap()
        XCTAssertTrue(app.staticTexts["ask-plonn-welcome"].exists)
        capture(app, name: "Message-first chat welcome")
    }

    func testSelectionCalculatorRemainsAvailable() {
        let app = openAsk()
        app.buttons["Selection calculator"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        XCTAssertTrue(app.staticTexts["Selection calculator"].waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(app.staticTexts["Research draft · No bets are placed"].exists)
        XCTAssertTrue(app.staticTexts["Your parlay"].exists)
        capture(app, name: "Preserved selection calculator")
        // Do not modify or clear any existing simulator draft, models or watchlist.
    }

    func testSavedConfigurationIsVerifiedWithoutTraining() throws {
        let app = openAsk()
        app.buttons["ask-plonn-chat-context"].tap()
        app.buttons["ask-plonn-saved-contexts"].tap()
        app.buttons["ask-plonn-context"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let savedSetup = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "Model setup")).firstMatch
        guard savedSetup.waitForExistence(timeout: 5) else {
            throw XCTSkip("No device model recipe exists; this read-only test does not create one.")
        }
        savedSetup.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        app.buttons["ask-plonn-context-done"].tap()
        revealTopic(app)
        app.buttons["Explain model"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let send = app.buttons["ask-plonn-send"]
        for _ in 0..<4 where !send.isHittable { app.swipeUp() }
        send.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 10), app.debugDescription)
        if answer.label.contains("configuration only") {
            XCTAssertTrue(answer.label.contains("does not prove training"), answer.label)
        } else {
            // Older device recipes can omit sport/stat metadata. Preserve them and
            // require an explicit explanation rather than infer a new configuration.
            XCTAssertTrue(answer.label.contains("unavailable"), answer.label)
            XCTAssertTrue(app.staticTexts["The saved recipe has missing or invalid configuration fields."].exists)
        }
        capture(app, name: "Existing device model configuration")
    }

    func testValidConfigurationSummaryWithTemporaryFixture() throws {
        let fixture: [String: Any] = ["id": "00000000-0000-0000-0000-000000000001",
            "name": "UI fixture model", "player": "Fixture Player", "quarterback": "", "opponent": "Fixture Opponent",
            "features": ["recent", "season"], "created": Date().addingTimeInterval(-3600).timeIntervalSinceReferenceDate,
            "line": 20.5, "side": "Over", "sport": "NBA", "stat": "Points", "targetKind": "Player",
            "modelType": "logistic_regression", "advanced": false]
        let data = try JSONSerialization.data(withJSONObject: [fixture], options: [.sortedKeys])
        // DEBUG environment fixture is temporary; it does not rewrite the existing recipes.
        let app = openAsk(recipeFixture: String(decoding: data, as: UTF8.self))
        app.buttons["ask-plonn-chat-context"].tap()
        app.buttons["ask-plonn-saved-contexts"].tap()
        app.buttons["ask-plonn-context"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let setup = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "UI fixture model")).firstMatch
        XCTAssertTrue(setup.waitForExistence(timeout: 5), app.debugDescription)
        setup.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        app.buttons["ask-plonn-context-done"].tap()
        revealTopic(app)
        app.buttons["Explain model"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let send = app.buttons["ask-plonn-send"]
        for _ in 0..<4 where !send.isHittable { app.swipeUp() }
        send.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(answer.label.contains("configuration only"), answer.label)
        XCTAssertTrue(answer.label.contains("does not prove training"), answer.label)
        XCTAssertTrue(answer.label.contains("NBA Player target Fixture Player, Points"), answer.label)
        capture(app, name: "Verified configuration summary with temporary fixture")

        // Switching modes must reset the default question and allow this same
        // saved model to be selected again without changing persisted recipes.
        app.buttons["ask-plonn-chat-context"].tap()
        let researchPlayer = app.buttons["ask-plonn-new-player"]
        for _ in 0..<8 where !researchPlayer.isHittable { app.swipeDown() }
        XCTAssertTrue(researchPlayer.isHittable, app.debugDescription)
        researchPlayer.tap()
        let name = app.textFields["ask-plonn-player-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5), app.debugDescription)
        name.tap(); name.typeText("Aaron Judge\n")
        app.segmentedControls["ask-plonn-player-sport"].buttons["MLB"].tap()
        app.buttons["ask-plonn-player-stat"].tap()
        app.buttons["Hits"].tap()
        app.buttons["ask-plonn-context-done"].tap()
        XCTAssertTrue(send.isEnabled)
        send.tap()
        let playerAnswer = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 1)
        XCTAssertTrue(playerAnswer.waitForExistence(timeout: 60), app.debugDescription)
        XCTAssertTrue(app.staticTexts["What do the recent games show?"].exists, app.debugDescription)
        XCTAssertTrue(playerAnswer.label.contains("Aaron Judge averaged"), playerAnswer.label)

        app.buttons["ask-plonn-chat-context"].tap()
        let savedContexts = app.buttons["ask-plonn-saved-contexts"]
        for _ in 0..<8 where !savedContexts.isHittable { app.swipeDown() }
        // This group stays expanded after the first model selection.
        app.buttons["ask-plonn-context"].tap()
        XCTAssertTrue(setup.waitForExistence(timeout: 5), app.debugDescription)
        setup.tap()
        XCTAssertFalse(name.exists, "Selecting the same model must leave player mode.")
        app.buttons["ask-plonn-context-done"].tap()
        XCTAssertTrue(send.isEnabled)
        send.tap()
        let reopenedModel = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 2)
        XCTAssertTrue(reopenedModel.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(reopenedModel.label.contains("NBA Player target Fixture Player, Points"), reopenedModel.label)
        capture(app, name: "Saved model reopens after player research")
    }
}
