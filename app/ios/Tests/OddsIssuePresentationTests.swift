import XCTest
@testable import PLONNCore

final class OddsIssuePresentationTests: XCTestCase {
 func testDailyRecordedMessageShowsOnlyOriginalQuoteTime() {
  let time="Quote as of Oct 15, 8:00:00 AM EDT"
  let result=OddsIssuePresentation("Cached odds · "+time+" · Daily recorded sportsbook odds · Prices may have changed. See each quote's as-of time.")
  XCTAssertEqual(result.asOf,time)
  XCTAssertNil(result.warning)
  XCTAssertNil(result.status)
 }
 func testPriorRecordedMessageAndPausedRefreshRemainInformational() {
  for message in ["Recorded sportsbook odds · Refresh unavailable · Prices may have changed. See each quote's as-of time.","Cached odds. Provider refreshes paused until midnight Eastern.","Provider refreshes paused until midnight Eastern."] {
   let result=OddsIssuePresentation("Cached odds · Quote as of Oct 14, 1:00:00 PM EDT · "+message)
   XCTAssertNil(result.warning)
   XCTAssertNotNil(result.asOf)
  }
 }
 func testMissingTimestampDoesNotInventFreshness() {
  let result=OddsIssuePresentation("Cached odds · As-of time unavailable · Daily recorded sportsbook odds · Prices may have changed. See each quote's as-of time.")
  XCTAssertEqual(result.asOf,"As-of time unavailable")
  XCTAssertNil(result.warning)
 }
 func testActualErrorsRemainVisibleWithTimestamp() {
  for error in ["Odds service unavailable. Pull to retry.","Player props unavailable. The connection failed.","Unknown service failure"] {
   let result=OddsIssuePresentation("Cached odds · Quote as of yesterday · "+error)
   XCTAssertEqual(result.warning,error)
   XCTAssertEqual(result.asOf,"Quote as of yesterday")
   XCTAssertEqual(OddsIssuePresentation(error).warning,error)
  }
 }
 func testMissingPricesRemainVisible() {
  let result=OddsIssuePresentation("Cached odds · As-of time unavailable · No cached markets for this matchup.")
  XCTAssertEqual(result.status,"Prices unavailable for this matchup.")
 }
 func testArchiveQualificationAndUnknownMessagesAreNotSilenced() {
  let archive="Archived recorded sportsbook odds · Not live prices. Event status must be verified separately. See each quote's original as-of time."
  XCTAssertEqual(OddsIssuePresentation("Cached odds · Quote as of yesterday · "+archive).warning,archive)
  XCTAssertEqual(OddsIssuePresentation("Cached odds · malformed").warning,"Cached odds · malformed")
 }
}
