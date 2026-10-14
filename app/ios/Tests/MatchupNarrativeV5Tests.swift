import XCTest
@testable import PLONNCore

final class MatchupNarrativeV5Tests: XCTestCase {
 typealias J=RichContextJSON
 func corpus()throws->J {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  var parser=RichContextJSONParser(try Data(contentsOf:root.appendingPathComponent("contracts/matchup-narrative-v5.json")))
  return try parser.parse()
 }
 func versions(_ j:J)->[String:String] {guard case .object(let fields)=j else{return [:]};return fields.mapValues(\.string)}
 func date(_ c:J)->Date {Date(timeIntervalSince1970:Double(c["now"].number!)!)}
 func testAll48CanonicalEnvelopesAndIndependentClaims()throws {
  let fixture=try corpus();XCTAssertEqual(fixture["cases"].array.count,48)
  for c in fixture["cases"].array {
   if c["expectedError"] != .null {
    XCTAssertThrowsError(try MatchupNarrativeV5.prepare(c["context"],now:date(c),injuryContext:c["injuryContext"],injuryRoleContext:c["injuryRoleContext"]),c["id"].string)
    continue
   }
   do {
    let plan=try MatchupNarrativeV5.prepare(c["context"],now:date(c),injuryContext:c["injuryContext"],injuryRoleContext:c["injuryRoleContext"])
    XCTAssertEqual(plan["claims"],c["expectedClaims"],c["id"].string)
    XCTAssertEqual(plan["requiredDimensions"],c["requiredDimensions"],c["id"].string)
    let raw=Data(c["canonicalUTF8"].string.utf8)
    XCTAssertEqual(raw,Data(c["envelope"].canonical.utf8))
    let result=try MatchupNarrativeV5.validate(raw,now:date(c),versions:versions(fixture["versions"]))
    XCTAssertEqual(result.rawData,raw)
    XCTAssertEqual(result.report,c["envelope"]["report"],c["id"].string)
    XCTAssertEqual(result.claims.map(\.id),c["selection"]["claimIDs"].array.map(\.string))
   } catch { XCTFail("V5 case \(c["id"].string) rejected: \(error)") }
  }
 }
 func testAllCorpusMutationCategoriesFailClosed()throws {
  let fixture=try corpus(),pins=versions(fixture["versions"])
  for c in fixture["cases"].array where c["expectedError"] == .null {
   let original=c["envelope"]
   for mutation in fixture["rejectionMutations"].array.map(\.string) {
    var wire=original,now=date(c)
    switch mutation {
    case "roleEvidenceIDs","claimText":
     var claims=wire["report"]["claims"].array
     claims[0]=claims[0].setting(mutation=="claimText" ? "text":"roleEvidenceIDs",mutation=="claimText" ? .string("Unsupported claim"):.list([.string("unbound-role-evidence")]))
     wire=wire.setting("report",wire["report"].setting("claims",.list(claims)))
    case "roleStatusEvidence": wire=wire.setting("report",wire["report"].setting("roleStatusEvidence",.list([.object(["claimID":.string("unbound-status")])])))
    case "sourceFingerprint":wire=wire.setting("sourceFingerprint",.string(String(repeating:"0",count:64)))
    case "versions":wire=wire.setting("versions",wire["versions"].setting("rules",.string("unknown")))
    case "staleRead":now=try RichSportsContext.time(wire["event"]["startISO"]).addingTimeInterval(1)
    case "differentEvent":wire=wire.setting("event",wire["event"].setting("homeTeam",.string("Different Team")))
    default:XCTFail("Unknown mutation \(mutation)");continue
    }
    XCTAssertThrowsError(try MatchupNarrativeV5.validate(Data(wire.canonical.utf8),now:now,versions:pins),"\(c["id"].string): \(mutation)")
   }
  }
 }
 func testCanonicalAndVersionIsolation()throws {
  let fixture=try corpus(),c=fixture["cases"].array[0],raw=c["canonicalUTF8"].string,pins=versions(fixture["versions"])
  XCTAssertThrowsError(try MatchupNarrativeV5.validate(Data((raw+" ").utf8),now:date(c),versions:pins))
  XCTAssertThrowsError(try MatchupNarrativeV5.validate(Data(("{\"contractVersion\":\"matchup-narrative-v5\","+raw.dropFirst()).utf8),now:date(c),versions:pins))
  XCTAssertThrowsError(try MatchupNarrativeV5.validate(Data(raw.utf8),now:date(c).addingTimeInterval(-1),versions:pins))
  XCTAssertThrowsError(try MatchupNarrativeV5.validate(Data(raw.utf8),now:date(c),versions:[:]))
  XCTAssertThrowsError(try MatchupNarrativeV4.validate(Data(raw.utf8),now:date(c),versions:pins))
 }
 func testAvailabilitySelectionCannotDropBoundStatusOrRequiredSide()throws {
  let fixture=try corpus(),c=fixture["cases"].array[0],pins=versions(fixture["versions"]),wire=c["envelope"]
  let ids=wire["selection"]["claimIDs"].array
  for selection in [J.object(["claimIDs":.list(ids.filter{!$0.string.contains("availability")} )]),
                    J.object(["claimIDs":.list(ids+[.string("unknown-claim")])]),
                    J.object(["claimIDs":.list(ids),"extra":.bool(true)])] {
   XCTAssertThrowsError(try MatchupNarrativeV5.validate(Data(wire.setting("selection",selection).canonical.utf8),now:date(c),versions:pins))
  }
 }

}
