import XCTest
@testable import PLONNCore

final class MatchupNarrativeV4Tests: XCTestCase {
 typealias J=RichContextJSON
 func corpus()throws->J {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  var parser=RichContextJSONParser(try Data(contentsOf:root.appendingPathComponent("contracts/matchup-narrative-v4.json")))
  return try parser.parse()
 }
 func versions(_ j:J)->[String:String] {guard case .object(let fields)=j else{return [:]};return fields.mapValues(\.string)}
 func testAllSyntheticClaimsAndCanonicalEnvelopes()throws {
  let fixture=try corpus()
  for c in fixture["cases"].array {
   let now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
   let plan=try MatchupNarrativeV4.prepare(c["context"],now:now,injuryContext:c["injuryContext"])
   XCTAssertEqual(plan["claims"],c["expectedClaims"],c["id"].string)
   XCTAssertEqual(plan["requiredDimensions"],c["requiredDimensions"],c["id"].string)
   let raw=Data(c["envelope"].canonical.utf8)
   do {
    let result=try MatchupNarrativeV4.validate(raw,now:now,versions:versions(fixture["versions"]))
    XCTAssertEqual(result.rawData,raw)
    XCTAssertEqual(result.claims.map(\.id),c["selection"]["claimIDs"].array.map(\.string))
   } catch { XCTFail("Envelope rejected: \(c["id"].string): \(error)") }
  }
 }
 func testCorpusRejections()throws {
  let fixture=try corpus()
  for mutation in fixture["rejectionMutations"].array {
   let c=try XCTUnwrap(fixture["cases"].array.first{$0["id"]==mutation["case"]})
   var wire=c["envelope"],now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
   switch mutation["target"].string {
   case "selection":wire=wire.setting("selection",mutation["value"])
   case "versions.rules":wire=wire.setting("versions",wire["versions"].setting("rules",mutation["value"]))
   case "report.claims.0.text":var claims=wire["report"]["claims"].array;claims[0]=claims[0].setting("text",mutation["value"]);wire=wire.setting("report",wire["report"].setting("claims",.list(claims)))
   case "now":now=Date(timeIntervalSince1970:Double(mutation["value"].number!)!)
   default:XCTFail("Unhandled corpus mutation");continue
   }
   XCTAssertThrowsError(try MatchupNarrativeV4.validate(Data(wire.canonical.utf8),now:now,versions:versions(fixture["versions"])),mutation["reason"].string)
  }
 }
 func testCanonicalWhitespaceAndDuplicateKeysRejected()throws {
  let fixture=try corpus(),c=fixture["cases"].array[0],raw=c["envelope"].canonical,now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
  XCTAssertThrowsError(try MatchupNarrativeV4.validate(Data((raw+" ").utf8),now:now,versions:versions(fixture["versions"])))
  let duplicate="{\"contractVersion\":\"matchup-narrative-v4\","+raw.dropFirst()
  XCTAssertThrowsError(try MatchupNarrativeV4.validate(Data(duplicate.utf8),now:now,versions:versions(fixture["versions"])))
 }
 func testDecimalUsesPythonHalfEven() {
  XCTAssertEqual(MatchupNarrativeV4.number(.number("1.225")),"1.22")
  XCTAssertEqual(MatchupNarrativeV4.number(.number("1.235")),"1.24")
  XCTAssertEqual(MatchupNarrativeV4.number(.number("250.0")),"250")
 }
 func testSelectionCannotOmitConditionalOrDuplicateDimension()throws {
  func claim(_ id:String,_ kind:String)->J {.object(["id":.string(id),"dimension":.string("passing"),"kind":.string(kind)])}
  let plan=J.object(["claims":.list([claim("signal","conditional"),claim("plain","descriptive")]),"requiredDimensions":.list([.string("passing")])])
  XCTAssertThrowsError(try MatchupNarrativeV4.validateSelection(.object(["claimIDs":.list([.string("plain")])]),plan:plan))
  XCTAssertThrowsError(try MatchupNarrativeV4.validateSelection(.object(["claimIDs":.list([.string("signal"),.string("plain")])]),plan:plan))
  XCTAssertNoThrow(try MatchupNarrativeV4.validateSelection(.object(["claimIDs":.list([.string("signal")])]),plan:plan))
 }

}
