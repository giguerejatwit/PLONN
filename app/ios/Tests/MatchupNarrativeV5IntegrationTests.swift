import XCTest
@testable import PLONNCore

final class MatchupNarrativeV5IntegrationTests: XCTestCase {
 typealias J = RichContextJSON
 func fixture() throws -> J {
  let app=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  var p=RichContextJSONParser(try Data(contentsOf:app.appendingPathComponent("contracts/matchup-narrative-v5.json")))
  return try p.parse()
 }
 func testEveryAcceptedWireThroughExactArchiveAndMergedPresentation() throws {
  let f=try fixture()
  guard case .object(let versions)=f["versions"] else {return XCTFail("versions")}
  for c in f["cases"].array where c["canonicalUTF8"] != .null {
   let raw=Data(c["canonicalUTF8"].string.utf8), now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
   let event=try JSONDecoder().decode(SharedMatchupEvent.self,from:Data(c["envelope"]["event"].canonical.utf8))
   let bundle=J.object(["schemaVersion":.number("1"),"reports":.list([.string(raw.base64EncodedString())])])
   let r=try XCTUnwrap(MatchupNarrativeArchive.read(bundle:Data(bundle.canonical.utf8),event:event,now:now,versions:versions.mapValues(\.string),version5:true))
   XCTAssertEqual(r.rawData,raw)
   XCTAssertNoThrow(try MatchupNarrativeCurrentState.validate(r,now:now))
   let check=try MatchupNarrativeCurrentState.nextCheck(r,now:now)
   XCTAssertGreaterThan(check,now)
   XCTAssertLessThanOrEqual(check,now.addingTimeInterval(60))
   XCTAssertThrowsError(try MatchupNarrativeCurrentState.validate(r,now:try SharedMatchupReportValidator.date(event.startISO)))
   let rows=MatchupOutlookPresentation.injuryObservations(r)
   XCTAssertEqual(Set(rows.map{$0["claimID"].string}).count,rows.count)
   for status in r.roleStatusEvidence.array { XCTAssertTrue(rows.contains(status)) }
   for claim in r.claims where !claim.roleEvidenceIDs.isEmpty {
    XCTAssertEqual(MatchupOutlookPresentation.text(for:claim,sport:event.sport),claim.text)
    XCTAssertTrue(claim.injuryIDs.allSatisfy { id in rows.contains{$0["claimID"].string==id} })
    XCTAssertFalse(claim.uncertainty.isEmpty)
   }
   if !r.injuryRoleGaps.array.isEmpty { XCTAssertFalse(MatchupOutlookPresentation.roleLimitations(r).isEmpty) }
   let absent=SharedMatchupEvent(sport:event.sport,eventID:"espn:999999999",homeTeam:event.homeTeam,awayTeam:event.awayTeam,startISO:event.startISO)
   XCTAssertNil(try MatchupNarrativeArchive.read(bundle:Data(bundle.canonical.utf8),event:absent,now:now,versions:versions.mapValues(\.string),version5:true))
  }
 }
 func testNoSyntheticPublicationGate() {
  XCTAssertFalse(Bundle.main.url(forResource:"matchup-narrative-v5",withExtension:"json") != nil)
 }
}
