import XCTest
final class PLONNUITests: XCTestCase {
 private func openSettings(_ app: XCUIApplication) { app.tabBars.buttons["Today"].tap(); if app.buttons["Account and settings"].exists { app.buttons["Account and settings"].tap() } }
 override func setUpWithError() throws { continueAfterFailure = false }
 func testGamesShowsCachedOddsSourceTimestamp() {
  let app=XCUIApplication()
  app.launchArguments=["-backendURL","http://127.0.0.1:8766"]
  app.launch()
  defer {app.terminate()}
  let settings=app.buttons["Account and settings"]
  let ready=expectation(for:NSPredicate(format:"hittable == true"),evaluatedWith:settings)
  wait(for:[ready],timeout:40)
  app.buttons["NBA"].tap()
  let issue=app.staticTexts.matching(identifier:"odds-feed-issue").firstMatch
  let cached=expectation(for:NSPredicate(format:"label CONTAINS %@", "Cached odds · As of"),evaluatedWith:issue)
  wait(for:[cached],timeout:30)
  XCTAssertTrue(issue.label.contains("EDT"),issue.label)
  let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Games shows cached odds and source timestamp";screenshot.lifetime = .keepAlways
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
  app.buttons["Account & sync"].tap()
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
  app.buttons["Account & sync"].tap()
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
  app.buttons["Account & sync"].tap()
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
  app.buttons["Account & sync"].tap()
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
  XCTAssertTrue(app.staticTexts["Sign in under Settings > Account & sync to view sportsbook odds. Sample research works offline."].waitForExistence(timeout: 5))
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

 func testFullGameCreationAndDeviceReopenWithoutHostedRuns() throws {
  let app=XCUIApplication()
  let supports=["NBA","MLB","NFL","NHL"].flatMap { league in
   ["Game total","Spread","Moneyline"].map { stat -> [String:Any] in
    ["league":league,"stat":stat,"target_kind":"team","features":["recent","season","homeAway","rest"],
     "model_types":["logistic_regression","decision_tree","random_forest","neural_network"],
     "evaluation_mode":"fixed_threshold_retrospective","historical_event_required":true]
   }
  }
  app.launchEnvironment["PLONN_LAB_CAPABILITIES_FIXTURE"]=String(data:try JSONSerialization.data(withJSONObject:["supported":[],"supported_data_targets":supports]),encoding:.utf8)!
  app.launchArguments=["-preferredSportsbook","DraftKings"]
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
   let create=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Create a model")).firstMatch
   XCTAssertTrue(create.waitForExistence(timeout:10),app.debugDescription)
   create.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
  }
  app.launch();openCreate()
  app.segmentedControls.buttons["Team / full game"].tap()
  app.buttons["Spread"].firstMatch.tap()
  let example=app.buttons["Use a completed-game example"];reveal(example);example.tap()
  let next=app.buttons["Choose features"];reveal(next);XCTAssertTrue(next.isEnabled);next.tap()
  let forest=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Random forest")).firstMatch
  reveal(forest);XCTAssertTrue(forest.isEnabled);forest.tap()
  let review=app.buttons["Review setup"];reveal(review);review.tap()
  let saved=app.buttons["Save on this device"];reveal(saved);saved.tap()
  XCTAssertTrue(app.buttons["Saved on device"].waitForExistence(timeout:5))
  let image=XCTAttachment(screenshot:app.screenshot());image.name="Full-game spread review";image.lifetime = .keepAlways;add(image)
  app.terminate();app.launch();openCreate()
  let recipe=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","Pittsburgh Steelers · Spread")).firstMatch
  reveal(recipe);recipe.tap()
  // Reopening retains the signed handicap and explicit historical game.
  for _ in 0..<8 {app.swipeDown()}
  XCTAssertEqual(app.textFields["Enter the exact historical game ID"].value as? String,"401671836")
  XCTAssertEqual(app.textFields["e.g. -1.5 or +1.5"].value as? String,"-3.5")
  let moneyline=app.buttons["Moneyline"].firstMatch;XCTAssertTrue(moneyline.exists);moneyline.tap()
  reveal(app.buttons["Use a completed-game example"]);app.buttons["Use a completed-game example"].tap()
  let noLine=app.staticTexts["Moneyline · Selected team to win. No line required."];reveal(noLine)
  XCTAssertTrue(noLine.exists)
  XCTAssertFalse(app.textFields["e.g. -1.5 or +1.5"].exists)
  reveal(next);next.tap();reveal(review);review.tap()
  reveal(app.buttons["Save on this device"]);app.buttons["Save on this device"].tap()
  XCTAssertTrue(app.buttons["Saved on device"].waitForExistence(timeout:5))
  // No Train model button is tapped: this test consumes no hosted account quota.
 }

}
