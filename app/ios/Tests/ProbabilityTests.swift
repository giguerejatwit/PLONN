import XCTest
@testable import PLONNCore
final class ProbabilityTests: XCTestCase {
 func leg(_ event: String, _ p: Double?, _ mode: String = "demo") -> ProbabilityLeg { ProbabilityLeg(event: event, probability: p, mode: mode) }
 func testIndependentExample() throws {
  let result = parlayEstimate([leg("boston", 0.6), leg("judge", 0.55)])
  XCTAssertEqual(try XCTUnwrap(result.0), 0.33, accuracy: 1e-12)
  XCTAssertEqual(try XCTUnwrap(fair(result.0)), 203.030303, accuracy: 0.000001)
  XCTAssertTrue(result.1.contains("Independence assumption"))
 }
 func testCorrelatedAndUnknownEventsUnavailable() {
  XCTAssertNil(parlayEstimate([leg("boston", 0.6), leg("boston", 0.55)]).0)
  XCTAssertNil(parlayEstimate([leg("", 0.6), leg("judge", 0.55)]).0)
 }
 func testMissingInvalidAndMixedEstimatesUnavailable() {
  for p: Double? in [nil, 0, 1, -0.1, 1.1, .nan, .infinity] {
   XCTAssertNil(parlayEstimate([leg("a", p), leg("b", 0.6)]).0)
   XCTAssertNil(fair(p))
  }
  XCTAssertNil(parlayEstimate([leg("a", 0.6), leg("b", 0.55, "live")]).0)
  XCTAssertNil(parlayEstimate([leg("a", 0.6)]).0)
 }
 func testAmericanOdds() throws {
  XCTAssertEqual(try XCTUnwrap(fair(0.6)), -150, accuracy: 1e-10)
  XCTAssertEqual(try XCTUnwrap(fair(0.5)), -100, accuracy: 1e-10)
  XCTAssertEqual(implied(-125), 5.0/9, accuracy: 1e-10)
  XCTAssertEqual(implied(150), 0.4, accuracy: 1e-10)
 }
}
