import XCTest

final class AskPLONNLocalUITests: XCTestCase {
    func testOfflineHostedMatchupParaphraseRendersThroughRealGenerationFlow() throws {
        let app = launchOfflineMatchup("success")
        XCTAssertTrue(app.staticTexts["matchup-offline-fixture-banner"].waitForExistence(timeout: 15))
        let generate = app.buttons["matchup-insights-refresh-ai"]
        XCTAssertTrue(generate.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertFalse(app.staticTexts["matchup-insights-gemma-summary"].exists)
        generate.tap()
        let summary = app.staticTexts["matchup-insights-gemma-summary"]
        XCTAssertTrue(summary.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertEqual(summary.label, "Seattle Seahawks passing production contrasts with Denver Broncos allowance. These observations describe a limited sample.")
        XCTAssertTrue(app.staticTexts["Observed matchup data"].exists)
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "240 passing yards")).firstMatch.exists)
        XCTAssertFalse(app.staticTexts["matchup-insights-failure-category"].exists)
        capture(app, name: "Offline hosted contract: rendered Seahawks Broncos answer")
    }

    func testOfflineHostedMatchupInvalidAnswerShowsFixedCategoryWithoutPayload() throws {
        let app = launchOfflineMatchup("invalid")
        let generate = app.buttons["matchup-insights-refresh-ai"]
        XCTAssertTrue(generate.waitForExistence(timeout: 15), app.debugDescription)
        generate.tap()
        let category = app.staticTexts["matchup-insights-failure-category"]
        XCTAssertTrue(category.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(category.label.contains("answer_contract"), category.label)
        XCTAssertTrue(app.staticTexts["matchup-insights-ai-error"].label.contains("did not match the recorded evidence"))
        XCTAssertFalse(app.staticTexts["matchup-insights-gemma-summary"].exists)
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "99 points")).firstMatch.exists)
        capture(app, name: "Offline hosted contract: safe fixed failure category")
    }

    private func launchOfflineMatchup(_ mode: String) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-appearanceMode", "light", "-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launchEnvironment["PLONN_MATCHUP_INSIGHTS_FIXTURE"] = mode
        app.launchEnvironment["PLONN_ASK_LOCAL_MODEL_DISABLED"] = "1"
        app.launch()
        return app
    }

    func testNHLMatchupShowsGeneratedGemmaExplanationAboveRecordedObservations() throws {
        guard ProcessInfo.processInfo.environment["PLONN_RUN_NHL_LIVE_UI"] == "1" else {
            throw XCTSkip("Opt-in local Gemma4 and public NHL research check; never calls the paid odds provider.")
        }
        let app = XCUIApplication()
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launch()
        let games = app.tabBars.buttons["Games"]
        XCTAssertTrue(games.waitForExistence(timeout: 45), app.debugDescription)
        games.tap()
        let nhl = app.buttons["NHL"]
        XCTAssertTrue(nhl.waitForExistence(timeout: 10), app.debugDescription)
        nhl.tap()
        // Reopen the NHL matchup reported by the user, from the preceding day.
        let calendar = Calendar.current
        let yesterday = calendar.date(byAdding: .day, value: -1, to: Date())!
        let day = app.buttons[yesterday.formatted(date: .complete, time: .omitted)]
        if !day.exists {
            let dates = app.descendants(matching: .any).matching(identifier: "games-week-dates").firstMatch
            XCTAssertTrue(dates.waitForExistence(timeout: 5), app.debugDescription)
            dates.swipeRight()
        }
        XCTAssertTrue(day.waitForExistence(timeout: 10), app.debugDescription)
        day.tap()
        let search = app.textFields["today-search"]
        XCTAssertTrue(search.waitForExistence(timeout: 5), app.debugDescription)
        search.tap(); search.typeText("Calgary\n")
        let calgary = app.staticTexts["Calgary Flames"].firstMatch
        XCTAssertTrue(calgary.waitForExistence(timeout: 45), app.debugDescription)
        calgary.tap()
        let source = app.staticTexts["matchup-insights-source"]
        guard source.waitForExistence(timeout: 45) else {
            XCTFail(app.debugDescription); return
        }
        tap(app.buttons["matchup-insights-refresh-ai"], in: app)
        let settled = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label CONTAINS %@ OR label CONTAINS %@", "Gemma4 explanation", "AI unavailable"), object: source)
        XCTAssertEqual(XCTWaiter.wait(for: [settled], timeout: 60), .completed)
        guard source.label.contains("Gemma4 explanation") else {
            XCTFail(source.label + "\n" + app.debugDescription); return
        }
        let summary = app.staticTexts["matchup-insights-gemma-summary"]
        guard summary.waitForExistence(timeout: 60) else {
            XCTFail(source.label + "\n" + app.debugDescription); return
        }
        XCTAssertTrue(source.label.contains("Gemma4 explanation"), source.label)
        XCTAssertFalse(summary.label.isEmpty)
        XCTAssertFalse(summary.label.contains(where: \.isNumber), summary.label)
        XCTAssertTrue(summary.label.contains("Calgary") || summary.label.contains("Anaheim"), summary.label)
        XCTAssertTrue(app.staticTexts["Observed matchup data"].exists, app.debugDescription)
        let evidenceRows = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Team history, not a player forecast."))
        XCTAssertEqual(evidenceRows.count, 4, "The complete supplied goals/shots basis remains visible.")
        print("NHL live generated summary: " + summary.label)
        let text = XCTAttachment(string: summary.label)
        text.name = "Live NHL Gemma4 generated summary"; text.lifetime = .keepAlways; add(text)
        reveal(summary, in: app)
        for _ in 0..<3 where summary.frame.maxY > app.frame.maxY - 110 {
            app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.7))
                .press(forDuration: 0.01, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.4)))
        }
        capture(app, name: "Live NHL Gemma4 explanation above recorded observations")
        tap(app.buttons["matchup-insights-refresh-ai"], in: app)
        let refreshed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label CONTAINS %@ OR label CONTAINS %@", "Gemma4 explanation", "AI unavailable"), object: source)
        XCTAssertEqual(XCTWaiter.wait(for: [refreshed], timeout: 60), .completed)
        XCTAssertTrue(source.label.contains("Gemma4 explanation"), source.label)
        XCTAssertEqual(evidenceRows.count, 4)
        app.buttons["BackButton"].tap()
        XCTAssertTrue(calgary.waitForExistence(timeout: 10)); calgary.tap()
        XCTAssertTrue(source.waitForExistence(timeout: 45))
        tap(app.buttons["matchup-insights-refresh-ai"], in: app)
        let reopened = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label CONTAINS %@ OR label CONTAINS %@", "Gemma4 explanation", "AI unavailable"), object: source)
        XCTAssertEqual(XCTWaiter.wait(for: [reopened], timeout: 60), .completed)
        XCTAssertTrue(source.label.contains("Gemma4 explanation"), source.label)
        XCTAssertTrue(summary.exists); XCTAssertEqual(evidenceRows.count, 4)
        reveal(summary, in: app)
        for _ in 0..<3 where summary.frame.maxY > app.frame.maxY - 110 {
            app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.7))
                .press(forDuration: 0.01, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.4)))
        }
        capture(app, name: "Live NHL Gemma4 explanation after refresh and reopening")
    }

    private func openAsk(researchDelayMilliseconds: Int? = nil, marketFixture: String? = nil) -> XCUIApplication {
        let app = XCUIApplication()
        if let milliseconds = researchDelayMilliseconds {
            app.launchEnvironment["PLONN_ASK_RESEARCH_DELAY_MS"] = String(milliseconds)
        }
        if let marketFixture { app.launchEnvironment["PLONN_ASK_MARKETS_FIXTURE"] = marketFixture }
        app.launchArguments = ["-backendURL", "https://127.0.0.1:9", "-appearanceMode", "light"]
        app.launch()
        let tab = app.tabBars.buttons["Ask PLONN"]
        XCTAssertTrue(tab.waitForExistence(timeout: 30))
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: tab)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed)
        tab.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        if !app.staticTexts["Ask PLONN"].waitForExistence(timeout: 3) {
            tab.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        }
        XCTAssertTrue(app.staticTexts["Ask PLONN"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["ask-plonn-provider-status"].label.contains("Local prototype"))
        return app
    }
    private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        let suggestions = app.scrollViews["ask-plonn-suggestions"]
        if suggestions.exists {
            for _ in 0..<3 where !element.isHittable { suggestions.swipeLeft() }
            for _ in 0..<3 where !element.isHittable { suggestions.swipeRight() }
        }
        for _ in 0..<12 where !element.isHittable { app.swipeUp() }
        for _ in 0..<12 where !element.isHittable { app.swipeDown() }
        XCTAssertTrue(element.isHittable, app.debugDescription)
    }
    private func tap(_ element: XCUIElement, in app: XCUIApplication) {
        reveal(element, in: app)
        element.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    }
    private func useJudgeHits(_ app: XCUIApplication) {
        tap(app.buttons["ask-plonn-chat-context"], in: app)
        let name = app.textFields["ask-plonn-player-name"]
        tap(name, in: app); name.typeText("Aaron Judge\n")
        tap(app.segmentedControls["ask-plonn-player-sport"].buttons["MLB"], in: app)
        tap(app.buttons["ask-plonn-player-stat"], in: app)
        let hits = app.buttons["Hits"]
        XCTAssertTrue(hits.waitForExistence(timeout: 5), app.debugDescription)
        hits.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        XCTAssertEqual(name.value as? String,"Aaron Judge")
        tap(app.buttons["ask-plonn-context-done"], in: app)
        XCTAssertTrue(app.buttons["ask-plonn-send"].isHittable,"Send stays visible after entering a player.")
        XCTAssertTrue(app.buttons["ask-plonn-send"].isEnabled,"A blank question defaults to recent form.")
    }
    private func capture(_ app: XCUIApplication, name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
    }

    func testBestBookAndValidatedEdgeThroughLocalGemma4() throws {
        let now = Date(), formatter = ISO8601DateFormatter()
        let stamp = formatter.string(from:now)
        let fixture: [String:Any] = ["id":"gemma-price-fixture", "event":"gemma-fixture-event", "sport":"NBA", "kind":"prop",
            "name":"Gemma Price Fixture", "opponent":"Fixture Opponent", "label":"Over 20.5 points", "aliases":"",
            "p":0.6,"price":-110,"market":"player_points","book":"Fixture Book A",
            "quotes":[["book":"Fixture Book A","price":-110,"updated":stamp],["book":"Fixture Book B","price":120,"updated":stamp]],
            "updated":stamp,"mode":"live","line":20.5,"side":"over","eventState":"pre",
            "modelValidated":true,"modelVersion":"fixture-v1","modelUpdated":stamp,
            "commence":formatter.string(from:now.addingTimeInterval(3600))]
        let data = try JSONSerialization.data(withJSONObject:[fixture],options:[.sortedKeys])
        let app = openAsk(marketFixture:String(decoding:data,as:UTF8.self))
        tap(app.buttons["ask-plonn-chat-context"],in:app)
        tap(app.buttons["ask-plonn-saved-contexts"],in:app)
        tap(app.buttons["ask-plonn-context"],in:app)
        let selection = app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Gemma Price Fixture")).firstMatch
        XCTAssertTrue(selection.waitForExistence(timeout:5),app.debugDescription)
        tap(selection,in:app); tap(app.buttons["ask-plonn-context-done"],in:app)
        tap(app.buttons["Compare prices"],in:app); tap(app.buttons["ask-plonn-send"],in:app)
        let best = app.staticTexts["sportsbook-best-price"]
        XCTAssertTrue(best.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertEqual(best.label,"+120")
        XCTAssertEqual(app.staticTexts["sportsbook-model-edge"].label,"+14.55 pp")
        let provider = app.staticTexts["ask-plonn-answer-provider"].firstMatch
        XCTAssertTrue(provider.waitForExistence(timeout:90),app.debugDescription)
        XCTAssertTrue(provider.label.contains("Local Gemma4 26B"),provider.label)
        tap(app.buttons["Recorded comparison summary"],in:app)
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout:5),app.debugDescription)
        XCTAssertTrue(answer.label.contains("Fixture Book B"),answer.label)
        XCTAssertTrue(answer.label.contains("+14.55 percentage points"),answer.label)
        reveal(best,in:app); capture(app,name:"Exact recorded best-price and model-edge evidence through local Gemma4")
    }

    func testGeneralConversationAndAssistantFollowUpWithoutAttachment() {
        let app = openAsk()
        XCTAssertTrue(app.buttons["ask-plonn-chat-context"].label.contains("optional"))
        let question = app.textFields["ask-plonn-question"]
        tap(question, in: app)
        question.typeText("Explain implied probability in two numbered points.")
        tap(app.buttons["ask-plonn-send"], in: app)
        let first = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 0)
        XCTAssertTrue(first.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(first.label.lowercased().contains("probability"), first.label)
        XCTAssertTrue(app.staticTexts["ask-plonn-answer-provider"].firstMatch.label.contains("Local Gemma4 26B"))
        XCTAssertFalse(app.buttons["ask-plonn-choose-context"].exists)
        XCTAssertFalse(app.buttons["ask-plonn-sources"].exists, "General knowledge must not masquerade as sourced player data.")
        XCTAssertFalse(app.textFields["ask-plonn-player-name"].exists)
        capture(app, name: "Gemma4 general conversation without an attachment")

        tap(question, in: app); question.typeText("Explain the second point in simpler words.")
        tap(app.buttons["ask-plonn-send"], in: app)
        let second = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 1)
        XCTAssertTrue(second.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(app.staticTexts.matching(identifier: "ask-plonn-answer-provider").element(boundBy: 1).label.contains("Local Gemma4 26B"))
        XCTAssertFalse(app.buttons["ask-plonn-choose-context"].exists, second.label)
        XCTAssertFalse(second.label.lowercased().contains("unavailable"), second.label)
        reveal(second, in: app)
        capture(app, name: "General follow-up refers to the previous assistant explanation")
    }

    func testCurrentPlayerDataOffersOptionalResearchWithoutInventedStatistics() {
        let app = openAsk()
        let question = app.textFields["ask-plonn-question"]
        tap(question, in: app); question.typeText("how many hits did aaron judge have in his last game?")
        tap(app.buttons["ask-plonn-send"], in: app)
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(app.buttons["ask-plonn-choose-context"].exists, answer.label)
        XCTAssertFalse(app.buttons["ask-plonn-sources"].exists)
        XCTAssertTrue(app.staticTexts["ask-plonn-answer-provider"].label.contains("Local Gemma4 26B"))
        XCTAssertNil(answer.label.rangeOfCharacter(from: .decimalDigits), "No observed player statistic is available to general chat: " + answer.label)
        capture(app, name: "Current player statistics ask for research rather than invent a value")
    }

    func testGeneralQuestionWithAttachmentAndRemovingResearch() {
        let app = openAsk()
        useJudgeHits(app)
        let question = app.textFields["ask-plonn-question"]
        tap(question, in: app); question.typeText("What is overfitting in a prediction model?")
        tap(app.buttons["ask-plonn-send"], in: app)
        let first = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(first.label.lowercased().contains("overfit"), first.label)
        XCTAssertTrue(app.staticTexts["ask-plonn-answer-provider"].label.contains("Local Gemma4 26B"))
        XCTAssertFalse(app.buttons["ask-plonn-sources"].exists)
        tap(question, in: app); question.typeText("show me his averages")
        tap(app.buttons["ask-plonn-send"], in: app)
        let research = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 1)
        XCTAssertTrue(research.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(research.label.contains("Aaron Judge averaged"), research.label)
        XCTAssertTrue(app.staticTexts.matching(identifier: "ask-plonn-answer-provider").element(boundBy: 1).label.contains("Local Gemma4 26B"))
        XCTAssertFalse(app.buttons["ask-plonn-sources"].exists)
        XCTAssertFalse(app.buttons["ask-plonn-choose-context"].exists, "An existing attachment should supply research after general discussion.")
        capture(app, name: "Existing attachment supplies research after a general explanation")
        tap(app.buttons["ask-plonn-chat-context"], in: app)
        tap(app.buttons["ask-plonn-remove-context"], in: app)
        XCTAssertTrue(app.buttons["ask-plonn-chat-context"].label.contains("optional"))
        XCTAssertFalse(app.buttons["ask-plonn-send"].isEnabled)
        app.buttons["ask-plonn-new-chat"].tap()
        XCTAssertTrue(app.buttons["ask-plonn-start-chat"].waitForExistence(timeout: 5))
        app.buttons["ask-plonn-start-chat"].tap()
        let second = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(second.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(app.staticTexts["ask-plonn-answer-provider"].label.contains("Local Gemma4 26B"))
        XCTAssertFalse(app.buttons["ask-plonn-choose-context"].exists)
        capture(app, name: "Research attachment removed and ordinary chat resumes")
    }

    func testNewChatCancelsGeneralQuestionWithoutLateAnswer() {
        let app = openAsk(researchDelayMilliseconds: 5000)
        app.buttons["ask-plonn-start-chat"].tap()
        XCTAssertTrue(app.buttons["ask-plonn-stop"].waitForExistence(timeout: 5))
        app.buttons["ask-plonn-new-chat"].tap()
        XCTAssertTrue(app.staticTexts["ask-plonn-welcome"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["ask-plonn-answer"].waitForExistence(timeout: 6))
        XCTAssertFalse(app.buttons["ask-plonn-send"].isEnabled)
        capture(app, name: "New chat cancels a general question")
    }

    func testActualPlayerHistoryThroughLocalModelAndUnsupportedQuestion() {
        let app = openAsk()
        useJudgeHits(app)
        tap(app.buttons["ask-plonn-send"], in: app)
        let first = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 0)
        XCTAssertTrue(first.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(first.label.contains("Aaron Judge averaged"), first.label)
        XCTAssertTrue(first.label.contains("not a forecast"), first.label)
        let provider = app.staticTexts["ask-plonn-answer-provider"].firstMatch
        XCTAssertTrue(provider.label.contains("Local Gemma4 26B"), provider.label)
        XCTAssertFalse(app.buttons["ask-plonn-sources"].exists)
        XCTAssertFalse(app.buttons["ask-plonn-source-link"].exists)
        reveal(first, in: app)
        capture(app, name: "Real MLB history through local Gemma4")
        let question = app.textFields["ask-plonn-question"]
        tap(question, in: app)
        question.typeText("Give me a guaranteed winning bet for tomorrow.")
        tap(app.buttons["ask-plonn-send"], in: app)
        let second = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 1)
        XCTAssertTrue(second.waitForExistence(timeout: 60), app.debugDescription)
        XCTAssertTrue(second.label.contains("cannot be answered"), second.label)
        XCTAssertFalse(app.keyboards.firstMatch.exists)
        reveal(second, in: app)
        capture(app, name: "Unsupported betting claim refused")
    }

    func testTypedPriceQuestionRoutesWithoutChoosingATopic() {
        let app=openAsk()
        useJudgeHits(app)
        let question=app.textFields["ask-plonn-question"]
        tap(question,in:app);question.typeText("How do the recorded sportsbook prices compare?")
        tap(app.buttons["ask-plonn-send"],in:app)
        let answer=app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout:10),app.debugDescription)
        XCTAssertTrue(answer.label.contains("unavailable"),answer.label)
        XCTAssertTrue(app.staticTexts["An exact event, start time, player, stat, side and line are required."].exists)
        XCTAssertEqual(app.staticTexts["ask-plonn-answer-provider"].firstMatch.label,"Verified local summary")
        capture(app,name:"Typed price question uses the correct topic")
    }

    func testFollowUpKeepsModelContextWithoutSetupOnScreen() {
        let app = openAsk()
        useJudgeHits(app)
        tap(app.buttons["ask-plonn-send"], in: app)
        let first = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 0)
        XCTAssertTrue(first.waitForExistence(timeout: 90), app.debugDescription)
        XCTAssertTrue(first.label.contains("Aaron Judge averaged"), first.label)
        XCTAssertFalse(app.textFields["ask-plonn-player-name"].exists)
        tap(app.buttons["ask-plonn-follow-up"], in: app)
        XCTAssertEqual(app.textFields["ask-plonn-question"].value as? String, "What does that mean?")
        tap(app.buttons["ask-plonn-send"], in: app)
        let followUp = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 1)
        XCTAssertTrue(followUp.waitForExistence(timeout: 60), app.debugDescription)
        XCTAssertTrue(followUp.label.contains("Aaron Judge averaged"), followUp.label)
        XCTAssertFalse(followUp.label.contains("cannot be answered"), followUp.label)
        let provider = app.staticTexts.matching(identifier: "ask-plonn-answer-provider").element(boundBy: 1)
        XCTAssertTrue(provider.label.contains("Local Gemma4 26B"), provider.label)
        capture(app, name: "Gemma4 follows the same conversation")
        app.buttons["ask-plonn-new-chat"].tap()
        XCTAssertFalse(followUp.exists)
        XCTAssertTrue(app.buttons["ask-plonn-chat-context"].label.contains("Aaron Judge"))
        XCTAssertTrue(app.buttons["ask-plonn-send"].isEnabled, "New chat keeps the explicitly selected research context.")
    }

    func testMissingMatchupAndPricesStayUnavailable() {
        let app = openAsk()
        useJudgeHits(app)
        tap(app.buttons["Against opponent"], in: app)
        tap(app.buttons["ask-plonn-send"], in: app)
        let first = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 0)
        XCTAssertTrue(first.waitForExistence(timeout: 10))
        XCTAssertTrue(first.label.contains("unavailable"), first.label)
        XCTAssertEqual(app.staticTexts["ask-plonn-answer-provider"].firstMatch.label, "Verified local summary")
        tap(app.buttons["Compare prices"], in: app)
        tap(app.buttons["ask-plonn-send"], in: app)
        let second = app.staticTexts.matching(identifier: "ask-plonn-answer").element(boundBy: 1)
        XCTAssertTrue(second.waitForExistence(timeout: 10))
        XCTAssertTrue(second.label.contains("unavailable"), second.label)
        XCTAssertTrue(app.staticTexts["An exact event, start time, player, stat, side and line are required."].exists)
        reveal(second, in: app)
        capture(app, name: "No invented matchup or sportsbook prices")
    }

    func testStopDiscardsPendingResearch() {
        let app = openAsk(researchDelayMilliseconds: 5000)
        useJudgeHits(app)
        tap(app.buttons["ask-plonn-send"], in: app)
        let stop = app.buttons["ask-plonn-stop"]
        XCTAssertTrue(stop.waitForExistence(timeout: 5), app.debugDescription)
        stop.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let answer = app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout: 5))
        XCTAssertTrue(answer.label.contains("Research stopped"), answer.label)
        XCTAssertFalse(stop.exists)
        XCTAssertFalse(app.staticTexts["ask-plonn-answer-provider"].exists)
        tap(app.buttons["ask-plonn-retry"].firstMatch,in:app)
        XCTAssertEqual(app.textFields["ask-plonn-question"].value as? String,"What do the recent games show?")
        XCTAssertTrue(app.buttons["ask-plonn-send"].isEnabled)
        capture(app, name: "Cancelled research")
    }

    func testNewChatClearsPendingResearchWithoutLateAnswer() {
        let app = openAsk(researchDelayMilliseconds: 5000)
        useJudgeHits(app)
        tap(app.buttons["ask-plonn-send"], in: app)
        XCTAssertTrue(app.buttons["ask-plonn-stop"].waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(app.staticTexts["ask-plonn-typing"].exists, app.debugDescription)
        app.buttons["ask-plonn-new-chat"].tap()
        XCTAssertTrue(app.staticTexts["ask-plonn-welcome"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["ask-plonn-stop"].exists)
        XCTAssertFalse(app.staticTexts["ask-plonn-answer"].waitForExistence(timeout: 6), "Cancelled research must not reappear in a new chat.")
        XCTAssertTrue(app.buttons["ask-plonn-chat-context"].label.contains("Aaron Judge"))
        capture(app, name: "New chat cancels pending work")
    }

    func testLeavingLLMStopsPendingResearchAndReopensUsable() {
        let app=openAsk(researchDelayMilliseconds:5000)
        useJudgeHits(app)
        tap(app.buttons["ask-plonn-send"],in:app)
        XCTAssertTrue(app.buttons["ask-plonn-stop"].waitForExistence(timeout:5))
        app.tabBars.buttons["Games"].tap()
        app.tabBars.buttons["Ask PLONN"].tap()
        let answer=app.staticTexts["ask-plonn-answer"].firstMatch
        XCTAssertTrue(answer.waitForExistence(timeout:5),app.debugDescription)
        XCTAssertTrue(answer.label.contains("left Ask PLONN"),answer.label)
        XCTAssertFalse(app.buttons["ask-plonn-stop"].exists)
        tap(app.buttons["ask-plonn-retry"].firstMatch,in:app)
        XCTAssertTrue(app.buttons["ask-plonn-send"].isEnabled)
        capture(app,name:"LLM reopens after pending research is cancelled")
    }
}
