import XCTest
final class PLONNUITests: XCTestCase {
 private func openSettings(_ app: XCUIApplication) { app.tabBars.buttons["Today"].tap(); if app.buttons["Account and settings"].exists { app.buttons["Account and settings"].tap() } }
 override func setUpWithError() throws { continueAfterFailure = false }
 func testGamesShowsQuoteTimeWithoutRoutineCacheWarning() {
  let app=XCUIApplication()
  app.launchArguments=["-backendURL","http://127.0.0.1:8766"]
  app.launch()
  defer {app.terminate()}
  let settings=app.buttons["Account and settings"]
  let ready=expectation(for:NSPredicate(format:"hittable == true"),evaluatedWith:settings)
  wait(for:[ready],timeout:40)
  app.buttons["NBA"].tap()
  let asOf=app.staticTexts.matching(identifier:"odds-as-of").firstMatch
  let recorded=expectation(for:NSPredicate(format:"label CONTAINS %@", "Quote as of"),evaluatedWith:asOf)
  wait(for:[recorded],timeout:30)
  XCTAssertTrue(asOf.label.contains("EDT"),asOf.label)
  XCTAssertFalse(app.staticTexts.matching(identifier:"odds-feed-issue").firstMatch.exists)
  let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Games shows quote time without a cache warning";screenshot.lifetime = .keepAlways
  add(screenshot)
 }
 func testGamesShowsOfflineOddsSourceWithoutSpendingProviderCredits() {
  let app=XCUIApplication()
  app.launchArguments=["-backendURL","https://127.0.0.1:9"]
  app.launch()
  defer {app.terminate()}
  let settings=app.buttons["Account and settings"]
  let ready=expectation(for:NSPredicate(format:"hittable == true"),evaluatedWith:settings)
  wait(for:[ready],timeout:40)
  let issue=app.staticTexts.matching(identifier:"odds-feed-issue").firstMatch
  XCTAssertTrue(issue.waitForExistence(timeout:30),"The Games screen must explain an odds feed failure.")
  XCTAssertTrue(issue.label.contains("Local odds service is offline"),issue.label)
  let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Games explains unavailable odds";screenshot.lifetime = .keepAlways
  add(screenshot)
 }
 func testReferralSignupEntryIsOptionalAndClearsWithMode() throws {
  let app = XCUIApplication(); app.launch()
  let settings = app.buttons["Account and settings"]
  let ready = expectation(for: NSPredicate(format: "hittable == true"), evaluatedWith: settings)
  wait(for: [ready], timeout: 40) // launch intro waits for initial data or its timeout
  settings.tap()
  app.buttons["Account"].tap()
  XCTAssertTrue(app.textFields["Email"].waitForExistence(timeout: 10))
  XCTAssertFalse(app.textFields["signup-referral-code"].exists)
  app.segmentedControls.buttons["Sign up"].tap()
  let referral = app.textFields["signup-referral-code"]
  func reveal(_ element: XCUIElement) {
   for _ in 0..<5 { if element.isHittable { return }; app.swipeUp() }
   XCTAssertTrue(element.isHittable)
  }
  let name = app.textFields["Display name"]
  reveal(name); name.tap(); name.typeText("Referral test")
  let email = app.textFields["Email"]
  reveal(email); email.tap(); email.typeText("referral-test@example.invalid")
  let password = app.secureTextFields["Password"]
  reveal(password); password.tap(); password.typeText("local-test-password")
  let confirmation = app.secureTextFields["Confirm password"]
  reveal(confirmation); confirmation.tap(); confirmation.typeText("local-test-password")
  reveal(referral)
  reveal(app.buttons["Create account"])
  XCTAssertTrue(app.buttons["Create account"].isEnabled, "Referral code must be optional")
  reveal(referral); referral.tap(); referral.typeText(" partner ")
  XCTAssertTrue(app.buttons["Create account"].isEnabled)
  for _ in 0..<5 { if app.segmentedControls.buttons["Sign in"].isHittable { break }; app.swipeDown() }
  app.segmentedControls.buttons["Sign in"].tap()
  XCTAssertFalse(referral.exists)
  app.segmentedControls.buttons["Sign up"].tap()
  reveal(referral)
  XCTAssertTrue(referral.exists)
  XCTAssertEqual(referral.value as? String, "Referral code (optional)")
  // Do not submit: this smoke test never creates a hosted account.
 }

 func testHostedAccountSyncAndDeletion() throws {
  let env = ProcessInfo.processInfo.environment
  guard let email = env["PLONN_TEST_EMAIL"], let password = env["PLONN_TEST_PASSWORD"] else {
   throw XCTSkip("Requires a disposable hosted test account injected into the test runner.")
  }
  let app = XCUIApplication(); app.launch(); app.tabBars.buttons["Search"].tap()
  openSettings(app)
  app.swipeUp()
  app.buttons["Clear saved watchlist"].tap()
  app.tabBars.buttons["Search"].tap()
  app.buttons["Save Boston Celtics"].tap()
  openSettings(app)
  app.swipeDown()
  app.buttons["Account"].tap()
  if app.buttons["Sign out"].exists { app.buttons["Sign out"].tap() }
  let emailField = app.textFields["Email"]
  XCTAssertTrue(emailField.waitForExistence(timeout: 10))
  emailField.tap(); emailField.typeText(email)
  let passwordField = app.secureTextFields["Password"]
  passwordField.tap(); passwordField.typeText(password)
  app.buttons["Sign in"].tap()
  XCTAssertTrue(app.buttons["Sign out"].waitForExistence(timeout: 20))
  app.buttons["Copy device watchlist to cloud"].tap()
  XCTAssertTrue(app.staticTexts["Device selections copied to your account. Refresh to view them."].waitForExistence(timeout: 20), app.staticTexts["cloud-status"].label)
  app.buttons["Refresh cloud watchlist"].tap()
  XCTAssertTrue(app.staticTexts["Cloud watchlist refreshed."].waitForExistence(timeout: 20))
  XCTAssertTrue(app.staticTexts["Boston Celtics"].exists)
  app.terminate(); app.launch(); app.tabBars.buttons["Search"].tap()
  openSettings(app)
  app.swipeDown()
  app.buttons["Account"].tap()
  XCTAssertTrue(app.buttons["Sign out"].waitForExistence(timeout: 20))
  app.buttons["Refresh cloud watchlist"].tap()
  XCTAssertTrue(app.staticTexts["Cloud watchlist refreshed."].waitForExistence(timeout: 20))
  XCTAssertTrue(app.staticTexts["Boston Celtics"].exists)
  app.swipeUp()
  app.buttons["Delete account"].tap()
  app.buttons["Delete permanently"].tap()
  XCTAssertTrue(app.staticTexts["Account deleted. Device-only watchlist is unchanged."].waitForExistence(timeout: 20))
 }
 func testAccountScreenRequiresCredentials() {
  let app = XCUIApplication(); app.launch(); app.tabBars.buttons["Search"].tap()
  openSettings(app)
  app.swipeDown()
  app.buttons["Account"].tap()
  XCTAssertTrue(app.textFields["Email"].waitForExistence(timeout: 5))
  XCTAssertTrue(app.secureTextFields["Password"].exists)
  XCTAssertFalse(app.buttons["Sign in"].isEnabled)
  XCTAssertFalse(app.buttons["Create account"].isEnabled)
 }
 func testUnconfiguredLiveModeAndOfflineRecovery() {
  let app = XCUIApplication()
  // Argument-domain override isolates this check without changing the saved backend.
  app.launchArguments = ["-backendURL", ""]
  app.launch(); app.tabBars.buttons["Search"].tap()
  XCTAssertTrue(app.buttons["Sportsbook odds"].waitForExistence(timeout: 10))
  app.buttons["Sportsbook odds"].tap()
  XCTAssertTrue(app.staticTexts["Sign in under Settings > Account to view sportsbook odds. Sample research works offline."].waitForExistence(timeout: 5))
  XCTAssertTrue(app.staticTexts["No matching markets"].exists)
  XCTAssertFalse(app.staticTexts["Boston Celtics"].exists)
  XCTAssertFalse(app.progressIndicators.firstMatch.exists)
  app.buttons["Sample research"].tap()
  XCTAssertTrue(app.staticTexts["Boston Celtics"].waitForExistence(timeout: 5))
  XCTAssertTrue(app.staticTexts["60.0%"].exists)
 }
 func testSearchWatchlistParlayAndResults() {
  let app = XCUIApplication()
  app.launch(); app.tabBars.buttons["Search"].tap()
  XCTAssertTrue(app.staticTexts["Boston Celtics"].waitForExistence(timeout: 10))
  app.buttons["+ Parlay"].firstMatch.tap()
  let search = app.searchFields.firstMatch
  search.tap(); search.typeText("Judge bases\n")
  XCTAssertTrue(app.staticTexts["Aaron Judge"].waitForExistence(timeout: 5))
  app.buttons["+ Parlay"].firstMatch.tap()
  app.tabBars.buttons["Builder"].tap()
  XCTAssertTrue(app.staticTexts["33.0%"].waitForExistence(timeout: 5))
  XCTAssertTrue(app.staticTexts["Model fair odds: +203"].exists)
  let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = "Independent parlay"; shot.lifetime = .keepAlways; add(shot)
  app.tabBars.buttons["Search"].tap()
  search.tap()
  app.buttons["Clear text"].tap()
  search.typeText("Tatum\n")
  app.buttons["+ Parlay"].firstMatch.tap()
  app.tabBars.buttons["Builder"].tap()
  XCTAssertTrue(app.staticTexts["Same-game legs require a validated joint model. Probability unavailable."].waitForExistence(timeout: 5))
  openSettings(app)
  app.swipeUp()
  app.buttons["Clear saved watchlist"].tap()
  app.tabBars.buttons["Search"].tap()
  search.tap(); app.buttons["Clear text"].tap(); search.typeText("Judge bases\n")
  app.buttons["Save Aaron Judge"].tap()
  app.terminate(); app.launch(); app.tabBars.buttons["Search"].tap()
  app.tabBars.buttons["Saved"].tap()
  XCTAssertTrue(app.staticTexts["Aaron Judge"].waitForExistence(timeout: 5))
  app.buttons["Save Aaron Judge"].tap()
  XCTAssertTrue(app.staticTexts["Nothing saved yet"].waitForExistence(timeout: 5))
  openSettings(app); app.buttons["Recorded results"].tap()
  XCTAssertTrue(app.staticTexts["272"].waitForExistence(timeout: 5))
  XCTAssertTrue(app.staticTexts["112"].exists)
 }
 func testLiveGameCenterDirectory() throws {
  let app = XCUIApplication(); app.launch(); app.tabBars.buttons["Search"].tap()
  app.buttons["Game center · Scores, teams & players"].tap()
  XCTAssertTrue(app.navigationBars["Game center"].waitForExistence(timeout: 10))
  app.segmentedControls.buttons["Teams"].tap()
  let search = app.textFields["directory-search"]
  search.tap(); search.typeText("Yankees")
  XCTAssertTrue(app.staticTexts["New York Yankees"].waitForExistence(timeout: 30))
  search.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: 7))
  app.segmentedControls.buttons["Players"].tap()
  search.tap(); search.typeText("Aaron Judge")
  XCTAssertTrue(app.staticTexts["Aaron Judge"].waitForExistence(timeout: 30))
  let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = "Fuchsia MLB player directory"; shot.lifetime = .keepAlways; add(shot)
 }

 func testAILabComparison() throws {
  let app = XCUIApplication(); app.launch(); app.tabBars.buttons["Search"].tap()
  app.buttons["AI Lab · Compare prediction models"].tap()
  XCTAssertTrue(app.navigationBars["AI Lab"].waitForExistence(timeout: 10))
  app.buttons["Load illustrative example"].tap()
  app.swipeUp()
  app.buttons["Run comparison"].tap()
  app.swipeUp()
  XCTAssertTrue(app.staticTexts["Comparison"].waitForExistence(timeout: 5))
  XCTAssertTrue(app.staticTexts["60.0%"].exists)
  let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = "AI Lab comparison"; shot.lifetime = .keepAlways; add(shot)
 }

 func testTodayNavigation() throws {
  let app = XCUIApplication(); app.launch()
  XCTAssertTrue(app.tabBars.buttons["Today"].exists)
  XCTAssertTrue(app.textFields["today-search"].exists)
  app.buttons["NBA"].tap()
  app.buttons["MLB"].tap()
  app.buttons["Choose date"].tap()
  XCTAssertTrue(app.buttons["Done"].waitForExistence(timeout:5))
  app.buttons["Done"].tap()
  app.buttons["Account and settings"].tap()
  XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout:5))
  app.tabBars.buttons["AI Lab"].tap()
  XCTAssertTrue(app.navigationBars["AI Lab"].waitForExistence(timeout:5))
  app.tabBars.buttons["Today"].tap()
 }

 func testLiveGameMarketsProvider() throws {
  let app=XCUIApplication();app.launch()
  let search=app.textFields["today-search"]
  XCTAssertTrue(search.waitForExistence(timeout:10));search.tap();search.typeText("Pittsburgh")
  let game=app.staticTexts["Pittsburgh Pirates"]
  XCTAssertTrue(game.waitForExistence(timeout:30));game.tap()
  XCTAssertTrue(app.navigationBars["Game markets"].waitForExistence(timeout:10))
  XCTAssertTrue(app.staticTexts["Game odds and available player props · 2-minute snapshots · Model estimates unavailable."].waitForExistence(timeout:45))
  app.segmentedControls.buttons["Player props"].tap()
  app.swipeUp()
  XCTAssertTrue(app.staticTexts["ODDS SNAPSHOT"].firstMatch.waitForExistence(timeout:10))
  let analysis = try XCTUnwrap(app.buttons.matching(identifier:"Analysis").allElementsBoundByIndex.first { $0.isHittable && $0.frame.midY > 220 && $0.frame.midY < 700 })
  XCTAssertTrue(analysis.waitForExistence(timeout:10));analysis.tap()
  XCTAssertTrue(app.staticTexts["Recent performance"].waitForExistence(timeout:10))
  XCTAssertTrue(app.staticTexts["MLB Stats • Earlier dates only • Includes previous season when needed."].waitForExistence(timeout:40))
  app.segmentedControls.buttons["Last 10"].tap()
  XCTAssertTrue(app.staticTexts["Oldest → newest · 10 games available"].exists)
  app.segmentedControls.buttons["Last 5"].tap()
  XCTAssertTrue(app.staticTexts["Oldest → newest · 5 games available"].exists)

 }

 func testGuidedModelRestoresLegacySetupWithoutHostedRuns() throws {
  let app=XCUIApplication()
  let supports=["NBA","MLB","NFL","NHL"].flatMap { league in
   ["Game total","Spread","Moneyline"].map { stat -> [String:Any] in
    ["league":league,"stat":stat,"target_kind":"team","features":["recent","season","homeAway","rest"],
     "model_types":["logistic_regression","decision_tree","random_forest","neural_network"],
     "evaluation_mode":"fixed_threshold_retrospective","historical_event_required":true]
   }
  }
  app.launchEnvironment["PLONN_LAB_CAPABILITIES_FIXTURE"]=String(data:try JSONSerialization.data(withJSONObject:["supported":[],"supported_data_targets":supports]),encoding:.utf8)!
  let seedRecipe:[String:Any] = ["id":UUID().uuidString,"name":"Pittsburgh Steelers · Spread","player":"Pittsburgh Steelers","quarterback":"","opponent":"Cincinnati Bengals","features":["recent","season"],"created":0,"line":-3.5,"side":"Over","sport":"NFL","stat":"Spread","targetKind":"Team","advanced":false,"modelType":"logistic_regression","book":"DraftKings","historicalEventID":"401671627"]
  let recipes=String(data:try JSONSerialization.data(withJSONObject:[seedRecipe]),encoding:.utf8)!
  app.launchArguments=["-preferredSportsbook","DraftKings","-labFeatureRecipes",recipes]
  func reveal(_ element:XCUIElement) {
   for _ in 0..<10 {if element.isHittable{return};app.swipeUp()}
   for _ in 0..<10 {if element.isHittable{return};app.swipeDown()}
   XCTAssertTrue(element.isHittable,app.debugDescription)
  }
  func openCreate() {
   let tab=app.tabBars.buttons["Models"];XCTAssertTrue(tab.waitForExistence(timeout:20))
   // The launch animation exposes tab accessibility before enabling its hit testing.
   let ready=XCTNSPredicateExpectation(predicate:NSPredicate(format:"isHittable == true"),object:tab)
   XCTAssertEqual(XCTWaiter.wait(for:[ready],timeout:20),.completed)
   tab.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
   let create=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Custom model")).firstMatch
   XCTAssertTrue(create.waitForExistence(timeout:10),app.debugDescription)
   create.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
  }
  app.launch()
  guard app.tabBars.buttons["Models"].waitForExistence(timeout:10) else {throw XCTSkip("Requires a signed-in test account; the picker itself has an independent offline UI test.")}
  openCreate()
  let more=app.buttons["More options"];reveal(more);more.tap()
  let existing=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Pittsburgh Steelers · Spread")).firstMatch
  reveal(existing);existing.tap()
  XCTAssertTrue(app.staticTexts["Review model"].waitForExistence(timeout:5))
  XCTAssertTrue(app.staticTexts["Spread -3.5 · NFL"].exists)
  XCTAssertFalse(app.textFields["Enter the exact historical game ID"].exists)
  let create=app.buttons["lab-create-model"]
  XCTAssertTrue(create.exists)
  XCTAssertEqual(create.label,"Create model")
  XCTAssertFalse(app.buttons["Save on this device"].exists)
  XCTAssertFalse(app.staticTexts["Experimental historical estimate"].exists)
  let advanced=app.buttons["lab-advanced"];reveal(advanced);advanced.tap()
  let forest=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Random forest")).firstMatch
  reveal(forest);XCTAssertTrue(forest.isEnabled);forest.tap()
  // The disabled service still permits reviewing and editing. Nothing is submitted.
  let image=XCTAttachment(screenshot:app.screenshot());image.name="Guided model with advanced settings";image.lifetime = .keepAlways;add(image)
  app.terminate();app.launch();openCreate();reveal(more);more.tap();reveal(existing);existing.tap()
  XCTAssertTrue(app.staticTexts["Spread -3.5 · NFL"].waitForExistence(timeout:5))
  let game=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","2, Game")).firstMatch
  if game.exists {game.tap()} else {app.buttons["Game"].tap()}
  XCTAssertTrue(app.buttons["lab-choose-game"].exists)
  XCTAssertTrue(app.staticTexts["Saved matchup"].exists)
  XCTAssertEqual(app.textFields["e.g. -2.5"].value as? String,"-3.5")
  // No Create model action: this test never saves to an account or consumes training quota.

 }

 func testCompletedGamePickerShowsTeamsAndSelectsOpponentWithoutIDs() {
  let app=XCUIApplication()
  app.launchEnvironment["PLONN_LAB_GAME_PICKER_FIXTURE"]="1"
  app.launch()
  defer {app.terminate()}
  XCTAssertTrue(app.navigationBars["Choose NFL game"].waitForExistence(timeout:15))
  XCTAssertTrue(app.staticTexts["Pittsburgh Steelers at Cincinnati Bengals"].waitForExistence(timeout:10))
  XCTAssertFalse(app.textFields["Enter the exact historical game ID"].exists)
  XCTAssertFalse(app.staticTexts["2024_13_PIT_CIN"].exists)
  XCTAssertFalse(app.links["Find game IDs"].exists)
  let image=XCTAttachment(screenshot:app.screenshot());image.name="Completed game picker with matchup names";image.lifetime = .keepAlways;add(image)
  app.buttons["lab-game-2024_13_PIT_CIN-PIT"].tap()
  let selected=app.staticTexts["lab-picker-selection"]
  XCTAssertTrue(selected.waitForExistence(timeout:5))
  XCTAssertEqual(selected.label,"Team: Pittsburgh Steelers\nOpponent: Cincinnati Bengals")
  app.buttons["Choose completed game"].tap()
  let home=app.buttons["lab-game-2024_13_PIT_CIN-CIN"]
  XCTAssertTrue(home.waitForExistence(timeout:5));home.tap()
  XCTAssertEqual(selected.label,"Team: Cincinnati Bengals\nOpponent: Pittsburgh Steelers")
 }

}

final class PreviewAccessUITests: XCTestCase {
 override func setUpWithError() throws {continueAfterFailure=false}
 private func launch(statisticsFixture:Bool=false) -> XCUIApplication {
  let app=XCUIApplication()
  app.launchArguments=["-plonn.previewWelcomeSeen","YES","-backendURL","https://127.0.0.1:9","-AppleLanguages","(en)","-AppleLocale","en_US"]
  if statisticsFixture {
   app.launchEnvironment["PLONN_ANALYTICS_MARKETS_FIXTURE"]="[]"
   app.launchEnvironment["PLONN_ANALYTICS_SUBJECTS_FIXTURE"]="[]"
   app.launchEnvironment["PLONN_ANALYTICS_HISTORY_FIXTURE"]="{}"
  }
  app.launch();return app
 }
 private func openAndCancelSignIn(_ app:XCUIApplication,feature:String) {
  let button=app.buttons["sign-in-for-"+feature]
  XCTAssertTrue(button.waitForExistence(timeout:15),app.debugDescription)
  button.tap()
  XCTAssertTrue(app.textFields["Email"].waitForExistence(timeout:10),app.debugDescription)
  XCTAssertTrue(app.secureTextFields["Password"].exists)
  let cancel=app.buttons["account-access-cancel"]
  XCTAssertTrue(cancel.exists);cancel.tap()
  XCTAssertTrue(button.waitForExistence(timeout:10),"Cancelling must return to the original feature.")
 }
 func testWelcomeRequiresSignInEvenWithLegacyPreviewPreference() {
  let app=launch();defer{app.terminate()}
  XCTAssertTrue(app.staticTexts["preview-welcome-heading"].waitForExistence(timeout:20),app.debugDescription)
  XCTAssertFalse(app.tabBars.firstMatch.exists)
  app.buttons["preview-create-account"].tap()
  XCTAssertTrue(app.textFields["Display name"].waitForExistence(timeout:10))
  XCTAssertTrue(app.secureTextFields["Confirm password"].exists)
  app.buttons["account-access-cancel"].tap()
  app.buttons["preview-sign-in"].tap()
  XCTAssertTrue(app.textFields["Email"].waitForExistence(timeout:10))
  XCTAssertFalse(app.textFields["Display name"].exists)
  app.buttons["account-access-cancel"].tap()
  XCTAssertTrue(app.staticTexts["preview-welcome-heading"].waitForExistence(timeout:10))
  XCTAssertFalse(app.buttons["preview-explore"].exists)
  XCTAssertFalse(app.tabBars.firstMatch.exists)
  app.terminate();app.launch()
  XCTAssertTrue(app.staticTexts["preview-welcome-heading"].waitForExistence(timeout:20))
  XCTAssertFalse(app.buttons["preview-explore"].exists)
  XCTAssertFalse(app.tabBars.firstMatch.exists)
  let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Dark welcome requires sign-in with no Explore first";screenshot.lifetime = .keepAlways;add(screenshot)
 }
 func testPreviewGatesAIModelsSavedResearchAndOddsWithoutLosingNavigation() {
  let app=launch(statisticsFixture:true);defer{app.terminate()}
  let games=app.tabBars.buttons["Games"]
  let ready=expectation(for:NSPredicate(format:"hittable == true"),evaluatedWith:games);wait(for:[ready],timeout:35)
  for (tab,feature) in [("Ask PLONN","aiResearch"),("Models","privateModels"),("My Models","privateModels")] {
   app.tabBars.buttons[tab].tap()
   XCTAssertFalse(app.textFields["ask-plonn-question"].exists)
   XCTAssertFalse(app.buttons["Save private model"].exists)
   openAndCancelSignIn(app,feature:feature)
   XCTAssertTrue(app.tabBars.buttons[tab].isSelected)
  }
  games.tap();app.buttons["Account and settings"].tap()
  let saved=app.buttons["Saved research"]
  for _ in 0..<5 where !saved.isHittable {app.swipeUp()}
  XCTAssertTrue(saved.exists);saved.tap()
  openAndCancelSignIn(app,feature:"savedResearch")
  app.tabBars.buttons["Analytics"].tap()
  let signIn=app.buttons["analytics-sign-in"]
  XCTAssertTrue(signIn.waitForExistence(timeout:15));signIn.tap()
  XCTAssertTrue(app.staticTexts["Sign in to continue with odds."].waitForExistence(timeout:10))
  app.buttons["account-access-cancel"].tap()
  XCTAssertTrue(app.textFields["analytics-search"].waitForExistence(timeout:10))
  XCTAssertTrue(app.tabBars.buttons["Analytics"].isSelected)
  let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Preview remains on Analytics after cancelling odds sign-in";screenshot.lifetime = .keepAlways;add(screenshot)
 }
}
