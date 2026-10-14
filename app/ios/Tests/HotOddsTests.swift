import XCTest
@testable import PLONNCore
final class HotOddsTests: XCTestCase {
 func testProviderDatePrecision() { XCTAssertEqual(sportsDate("2026-09-25T22:40Z"),sportsDate("2026-09-25T22:40:00Z")) }
 func testRequiresFreshValidatedLivePositiveValue() throws {
  let now = Date(timeIntervalSince1970: 1800000000)
  let date = ISO8601DateFormatter().string(from: now)
  var m = Market(id:"x",event:"g",sport:"NBA",kind:"team",name:"A",opponent:"B",label:"Win",aliases:"A",p:0.6,price:100,market:"h2h",book:"Book",quotes:[],updated:date,mode:"live",modelValidated:true,modelUpdated:date,modelVersion:"v1",eventState:"in")
  XCTAssertEqual(try XCTUnwrap(hotValue(m,now:now)),0.2,accuracy:0.00001)
  XCTAssertNil(hotValue(m,now:now.addingTimeInterval(121)))
  XCTAssertNil(hotValue(m,now:now.addingTimeInterval(-1)))
  m.modelValidated = nil; XCTAssertNil(hotValue(m,now:now))
  m.modelValidated = true; m.mode = "demo"; XCTAssertNil(hotValue(m,now:now))
  m.mode = "live"; m.eventState = "pre"; XCTAssertNil(hotValue(m,now:now))
  m.eventState = "in"; m.p = 0.4; XCTAssertNil(hotValue(m,now:now))
 }
}
