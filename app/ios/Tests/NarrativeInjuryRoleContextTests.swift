import XCTest
@testable import PLONNCore
final class NarrativeInjuryRoleContextTests:XCTestCase {
 typealias R = NarrativeInjuryRoleContext
 typealias J = RichContextJSON
 func cases()throws->[J] {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  var parser=RichContextJSONParser(try Data(contentsOf:root.appendingPathComponent("contracts/matchup-narrative-v5.json")))
  return try parser.parse()["cases"].array
 }
 func testAllFortyEightRoleCasesIndependentProjection()throws {
  let fixtures=try cases();XCTAssertEqual(fixtures.count,48)
  for c in fixtures {
   let now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
   if c["envelope"] == .null {XCTAssertThrowsError(try R.validate(c["injuryRoleContext"],now:now),c["id"].string);continue}
   let result=try R.auditedClaims(injury:c["injuryContext"],roles:c["injuryRoleContext"],now:now)
   let expected=c["envelope"]["report"]
   XCTAssertEqual(result["roleContext"],expected["injuryRoleContext"],c["id"].string)
   XCTAssertEqual(result["gaps"],expected["injuryRoleGaps"],c["id"].string)
   XCTAssertEqual(result["claims"],.list(c["expectedClaims"].array.filter{!$0["roleEvidenceIDs"].array.isEmpty}),c["id"].string)
  }
 }
 func testSizeShapesProviderAndFactIDReject()throws {
  let c=try cases()[0],p=c["injuryRoleContext"],now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
  XCTAssertThrowsError(try R.validate(p.setting("extra",.bool(true)),now:now))
  var ss=p["sources"].array;ss[0]=ss[0].setting("provider",.string("NFL"));XCTAssertThrowsError(try R.validate(p.setting("sources",.list(ss)),now:now))
  var facts=p["facts"].array;facts[0]=facts[0].setting("role",.string("goalie"));XCTAssertThrowsError(try R.validate(p.setting("facts",.list(facts)),now:now))
  XCTAssertThrowsError(try R.model(p,now:now,maxFacts:7));XCTAssertThrowsError(try R.model(p,now:now,maxBytes:4097))
 }
 func testAgingDoesNotRenewSemanticFingerprint()throws {
  let c=try cases()[0],p=c["injuryRoleContext"],now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
  let initial=try R.project(p,now:now),aged=try R.project(p,now:now.addingTimeInterval(21600))
  XCTAssertEqual(initial["fingerprint"],aged["fingerprint"]);XCTAssertTrue(aged["facts"].array.isEmpty);XCTAssertNotEqual(initial["eligibilityFingerprint"],aged["eligibilityFingerprint"])
 }
}
