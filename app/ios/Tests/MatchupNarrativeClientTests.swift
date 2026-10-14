import XCTest
@testable import PLONNCore

final class MatchupNarrativeClientTests:XCTestCase {
 private var event:SharedMatchupEvent {.init(sport:"NFL",eventID:"espn:401873001",homeTeam:"New England Patriots",awayTeam:"New York Jets",startISO:"2026-10-18T17:00:00Z")}
 func testMissingArchiveStillRequiresAuthentication() async throws {
  let client=MatchupNarrativeClient(tokenProvider:{""},reader:{_,_ in XCTFail("Unauthenticated read");return nil})
  do {_ = try await client.fetch(event:event);XCTFail("Expected auth rejection")}
  catch {XCTAssertEqual(error as? SharedMatchupReportError,.authenticationRequired)}
 }
 func testValidSessionCanReadMissingWithoutTransport() async throws {
  let client=MatchupNarrativeClient(tokenProvider:{"test-session"},reader:{_,_ in nil})
  let result=try await client.fetch(event:event);XCTAssertNil(result)
 }
 func testArchiveRejectsUnboundedOrMalformedData() {
  XCTAssertThrowsError(try MatchupNarrativeArchive.read(bundle:Data("{}".utf8),event:event,now:Date(),versions:[:]))
  XCTAssertThrowsError(try MatchupNarrativeArchive.read(bundle:Data(repeating:32,count:MatchupNarrativeArchive.maximumBytes+1),event:event,now:Date(),versions:[:]))
 }
 func testEmptyExactEventInventoryReturnsMissing() throws {
  XCTAssertNil(try MatchupNarrativeArchive.read(bundle:Data(#"{"schemaVersion":1,"reports":[]}"#.utf8),event:event,now:Date(),versions:[:]))
 }
 func testSyntheticCorpusIsNotAnActivatedArchive() {
  // Actual publication must be separately pinned; the parity corpus is not a resource.
  XCTAssertFalse(Bundle.main.url(forResource:"matchup-narrative-v4",withExtension:"json") != nil)
 }
 func testActualBundleReadsExactEventsAndRejectsDuplicates() throws {
  let ios=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
  let bytes=try Data(contentsOf:ios.appendingPathComponent("PLONN/MatchupNarrativeReports-v4.json"))
  let receipt=try JSONSerialization.jsonObject(with:Data(contentsOf:ios.appendingPathComponent("verification/narrative-v4-20261014/actual/receipt.json"))) as! [String:Any]
  let verified=try SharedMatchupReportValidator.date(receipt["verifiedAt"] as! String)
  let pins=try XCTUnwrap(MatchupNarrativeArchive.trustedVersions)
  var parser=RichContextJSONParser(bytes);let bundle=try parser.parse()
  XCTAssertEqual(bundle["reports"].array.count,4)
  for entry in bundle["reports"].array {
   let raw=try XCTUnwrap(Data(base64Encoded:entry.string));let result=try MatchupNarrativeV4.validate(raw,now:verified,versions:pins)
   let read=try XCTUnwrap(MatchupNarrativeArchive.read(bundle:bytes,event:result.event,now:verified,versions:pins))
   XCTAssertEqual(read.rawData,raw)
  }
  let duplicate=bundle.setting("reports",.list([bundle["reports"].array[0],bundle["reports"].array[0]]))
  XCTAssertThrowsError(try MatchupNarrativeArchive.read(bundle:Data(duplicate.canonical.utf8),event:event,now:verified,versions:pins))
  let absent=SharedMatchupEvent(sport:"NFL",eventID:"espn:99999",homeTeam:event.homeTeam,awayTeam:event.awayTeam,startISO:event.startISO)
  XCTAssertNil(try MatchupNarrativeArchive.read(bundle:bytes,event:absent,now:verified,versions:pins))
 }

}
