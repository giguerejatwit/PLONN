import XCTest
@testable import PLONNCore
final class RecordedStartPolicyTests:XCTestCase {
 func testNHLUniqueBoundedFutureCandidate() {
  let start=Date(timeIntervalSince1970:1791930000),now=Date(timeIntervalSince1970:1791920000)
  func accepts(_ offset:Double,_ candidates:[(String,Date)],sport:String="NHL",clock:Date?=nil)->Bool {RecordedStartPolicy.accepts(eventID:"one",quoteStart:start.addingTimeInterval(offset),start:start,sport:sport,candidates:candidates,now:clock ?? now)}
  let rows=[("one",start.addingTimeInterval(600))]
  XCTAssertTrue(accepts(600,rows));XCTAssertFalse(accepts(601,[("one",start.addingTimeInterval(601))]))
  XCTAssertFalse(accepts(600,rows+[("two",start)]));XCTAssertFalse(accepts(600,[]))
  XCTAssertFalse(accepts(600,rows,clock:start));XCTAssertFalse(accepts(600,rows,sport:"NFL"))
  XCTAssertTrue(accepts(0,rows,sport:"MLB"));XCTAssertFalse(accepts(600,rows,sport:"NBA"))
 }
 func testCrossUTCDayRejected() {
  let start=Date(timeIntervalSince1970:86400-300),quote=Date(timeIntervalSince1970:86400+300)
  XCTAssertFalse(RecordedStartPolicy.accepts(eventID:"one",quoteStart:quote,start:start,sport:"NHL",candidates:[("one",quote)],now:Date(timeIntervalSince1970:0)))
 }
}
