import XCTest
import CryptoKit
@testable import PLONNCore
final class MatchupNarrativeV5ActualTests:XCTestCase {
 func testPinnedBundleContainsOnlyExactOriginals() throws {
  let ios=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
  let bytes=try Data(contentsOf:ios.appendingPathComponent("PLONN/MatchupNarrativeReports-v5.json"))
  XCTAssertEqual(SHA256.hash(data:bytes).map{String(format:"%02x",$0)}.joined(),MatchupNarrativeV5Archive.trustedBundleSHA256)
  var parser=RichContextJSONParser(bytes);let root=try parser.parse()
  XCTAssertEqual(root["reports"].array.count,4)
  for row in root["reports"].array {
   let raw=try XCTUnwrap(Data(base64Encoded:row.string));var p=RichContextJSONParser(raw);let wire=try p.parse()
   let event=try JSONDecoder().decode(SharedMatchupEvent.self,from:Data(wire["event"].canonical.utf8))
   let original=try Data(contentsOf:ios.appendingPathComponent("verification/narrative-v5-20261014/actual/"+event.sport.lowercased()+"-response.json"))
   XCTAssertEqual(raw,original)
   let read=try XCTUnwrap(MatchupNarrativeArchive.read(bundle:bytes,event:event,now:try RichSportsContext.time(wire["generatedAt"]),versions:MatchupNarrativeV5Archive.trustedVersions,version5:true))
   XCTAssertEqual(read.rawData,raw)
  }
 }
 func testFourActualOriginalEnvelopesAndExpiry() throws {
  let dir=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("verification/narrative-v5-20261014/actual")
  let expiry=["nfl":"2026-10-14T17:10:00Z","nba":"2026-10-14T18:19:18.257Z","mlb":"2026-10-14T18:20:47.553Z","nhl":"2026-10-14T19:14:00Z"]
  for sport in ["nfl","nba","mlb","nhl"] {
   let raw=try Data(contentsOf:dir.appendingPathComponent(sport+"-response.json"))
   var parser=RichContextJSONParser(raw);let wire=try parser.parse()
   let generated=try RichSportsContext.time(wire["generatedAt"]),boundary=try SharedMatchupReportValidator.date(expiry[sport]!)
   let result=try MatchupNarrativeV5.validate(raw,now:generated,versions:MatchupNarrativeV5Archive.trustedVersions)
   XCTAssertEqual(result.rawData,raw)
   XCTAssertNoThrow(try MatchupNarrativeCurrentState.validate(result,now:boundary.addingTimeInterval(-0.001)))
   XCTAssertThrowsError(try MatchupNarrativeCurrentState.validate(result,now:boundary))
   XCTAssertEqual(try MatchupNarrativeCurrentState.nextCheck(result,now:boundary.addingTimeInterval(-0.5)),boundary)
   if sport == "nba" { XCTAssertTrue(MatchupOutlookPresentation.roleLimitations(result).contains("Current player availability could not be verified; earlier reports are out of date.")) }
   XCTAssertTrue(result.roleStatusEvidence.array.allSatisfy{$0["eventStatusConfirmed"] == .bool(false)})
   XCTAssertEqual(result.claims.filter{!$0.roleEvidenceIDs.isEmpty}.count,sport=="nba" ? 0:sport=="mlb" ? 1:2)
   if Date() < boundary {
    _ = try MatchupNarrativeV5.validate(raw,now:Date(),versions:MatchupNarrativeV5Archive.trustedVersions)
   } else { XCTAssertThrowsError(try MatchupNarrativeV5.validate(raw,now:Date(),versions:MatchupNarrativeV5Archive.trustedVersions)) }
   print("ACTUAL_V5_ACCEPTED",sport,raw.count,RichSportsContext.digest(wire))
  }
 }
}
