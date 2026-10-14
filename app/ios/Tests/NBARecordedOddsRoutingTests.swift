import XCTest
@testable import PLONNCore

final class NBARecordedOddsRoutingTests:XCTestCase {
 func testOnlyVerifiedNumericEventSeasonTypeIsAccepted() throws {
  func event(_ json:String)throws->[String:Any] {try JSONSerialization.jsonObject(with:Data(json.utf8)) as! [String:Any]}
  XCTAssertEqual(NBARecordedOddsRouting.seasonType(event:try event(#"{"season":{"type":1}}"#)),1)
  for raw in [#"{"season":{"type":true}}"#,#"{"season":{"type":"1"}}"#,#"{"season":{"type":1.5}}"#,#"{"season":{"type":4}}"#,#"{"status":{"type":{"state":"pre"}},"date":"2026-10-13"}"#] {
   XCTAssertNil(NBARecordedOddsRouting.seasonType(event:try event(raw)))
  }
 }
 func testSlatePhaseIsExplicitAndMixedOrUnknownFailsClosed() {
  XCTAssertEqual(NBARecordedOddsRouting.phase(sport:"NBA",seasonTypes:[1,1]),.preseason)
  XCTAssertEqual(NBARecordedOddsRouting.phase(sport:"NBA",seasonTypes:[2,3]),.regular)
  for values:[Int?] in [[],[nil],[1,nil],[1,2],[4]] {XCTAssertEqual(NBARecordedOddsRouting.phase(sport:"NBA",seasonTypes:values),.unknown)}
  XCTAssertNil(NBARecordedOddsRouting.phase(sport:"NFL",seasonTypes:[1]))
  XCTAssertNotEqual(NBARecordedOddsPhase.preseason,NBARecordedOddsPhase.regular)
 }
 func testPreseasonRequiresCacheOnlyAndVerifiedHandlerCapability() {
  XCTAssertFalse(NBARecordedOddsRouting.canRead(.preseason,cacheOnly:true,preseasonHandlerEnabled:false))
  XCTAssertFalse(NBARecordedOddsRouting.canRead(.preseason,cacheOnly:false,preseasonHandlerEnabled:true))
  XCTAssertTrue(NBARecordedOddsRouting.canRead(.preseason,cacheOnly:true,preseasonHandlerEnabled:true))
  XCTAssertFalse(NBARecordedOddsRouting.canRead(.unknown,cacheOnly:true,preseasonHandlerEnabled:true))
  XCTAssertTrue(NBARecordedOddsRouting.canRead(.regular,cacheOnly:true,preseasonHandlerEnabled:false))
  XCTAssertTrue(NBARecordedOddsRouting.canRead(nil,cacheOnly:true,preseasonHandlerEnabled:false))
 }
 func testDefaultRefreshCannotEraseNBASeasonEvidence() {
  for types:[Int?] in [[1],[],[nil],[2]] {
   let phase=NBARecordedOddsRouting.phase(sport:"NBA",seasonTypes:types)
   XCTAssertNotNil(phase)
   XCTAssertFalse(NBARecordedOddsRouting.canRead(phase,cacheOnly:false,preseasonHandlerEnabled:false))
  }
 }
 func testRecordedBodyUsesExactContractAndLegacyRegularOmitsPhase() throws {
  let preseason=OddsRequestBody(sport:"NBA",event:nil,cacheOnly:true,phase:"preseason")
  let payload=try JSONSerialization.jsonObject(with:JSONEncoder().encode(preseason)) as! [String:Any]
  XCTAssertEqual(Set(payload.keys),["sport","cacheOnly","phase"])
  XCTAssertEqual(payload["phase"] as? String,"preseason");XCTAssertEqual(payload["cacheOnly"] as? Bool,true)
  for sport in ["NBA","NFL"] {
   let body=OddsRequestBody(sport:sport,event:nil,cacheOnly:true)
   let payload=try JSONSerialization.jsonObject(with:JSONEncoder().encode(body)) as! [String:Any]
   XCTAssertNil(payload["phase"]);XCTAssertEqual(payload["sport"] as? String,sport)
  }
 }
}
