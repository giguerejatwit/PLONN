import XCTest
import CryptoKit
@testable import PLONNCore
final class MatchupNarrativePublishedTests:XCTestCase {
 func testActualOriginalEnvelopesAtGenerationAndVerifiedTime() throws {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("verification/narrative-v4-20261014/actual")
  let paths=try FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:nil).filter{$0.lastPathComponent.hasSuffix("-response.json")}
  XCTAssertFalse(paths.isEmpty)
  let receipt=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("receipt.json"))) as! [String:Any]
  let verified=try SharedMatchupReportValidator.date(receipt["verifiedAt"] as! String)
  let pins=try XCTUnwrap(MatchupNarrativeArchive.trustedVersions)
  for path in paths {
   do {
   let raw=try Data(contentsOf:path);var parser=RichContextJSONParser(raw);let wire=try parser.parse()
   let generated=try SharedMatchupReportValidator.date(wire["generatedAt"].string)
   let historic=try MatchupNarrativeV4.validate(raw,now:generated,versions:pins)
   let current=try MatchupNarrativeV4.validate(raw,now:verified,versions:pins)
   XCTAssertEqual(current.rawData,raw);XCTAssertEqual(historic.rawData,raw)
   XCTAssertEqual(current.event,historic.event);XCTAssertFalse(current.claims.isEmpty)
   XCTAssertNotEqual(current.injuryContext,.null)
   print("NARRATIVE_ACTUAL_ACCEPTED \(current.event.sport) \(current.event.eventID) \(raw.count) \(SHA256.hash(data:raw).map{String(format:"%02x",$0)}.joined())")
   } catch {XCTFail("\(path.lastPathComponent): \(error)")}
  }
 }
 func testPublishedReportsRejectChangedCurrentEligibility() throws {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("verification/narrative-v4-20261014/actual")
  let boundaries=["nfl":"2026-10-14T14:26:00.000Z","nba":"2026-10-14T18:19:18.257Z","mlb":"2026-10-14T18:20:47.553Z","nhl":"2026-10-14T18:19:18.764Z"]
  let pins=try XCTUnwrap(MatchupNarrativeArchive.trustedVersions)
  for (sport,stamp) in boundaries {
   let raw=try Data(contentsOf:root.appendingPathComponent("\(sport)-response.json"))
   let boundary=try SharedMatchupReportValidator.date(stamp)
   XCTAssertNoThrow(try MatchupNarrativeV4.validate(raw,now:boundary.addingTimeInterval(-0.001),versions:pins),sport)
   XCTAssertThrowsError(try MatchupNarrativeV4.validate(raw,now:boundary.addingTimeInterval(0.001),versions:pins),sport)
  }
 }

}
