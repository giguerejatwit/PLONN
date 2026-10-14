import XCTest
@testable import PLONNCore

final class RichSportsContextTests:XCTestCase {
 func cases()throws->[RichContextJSON] {
  let app=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  var p=RichContextJSONParser(try Data(contentsOf:app.appendingPathComponent("contracts/matchup-narrative-v4.json")))
  return try p.parse()["cases"].array
 }
 func testNamedPythonProjectionDigests()throws {
  let hashes:[String:String]=[
   "nfl_neutral_scoring_rich_dimensions": "950fefa19e7021a4d192cc82cda0cf631d5b2574053ce07079d2ef6562ed257e",
   "nfl_attempt_mix_excludes_sacks": "67bc8138f29fa268c5e84f78869d0b5fc013b39f658d40feb6f1eeafd959dcbd",
   "nfl_without_reference_descriptive": "165af0ab9bf606f4808b3054c64226f9a7c92b9e935f4fe28e65398963c1c879",
   "nfl_balanced_attempt_mix": "5783b18116f4943a7af02af980a6ade91c52e3de06121df432536b680a6a5d7a",
   "mlb_hits_and_batting_strikeouts": "3f94eb5472cf8e851a3f61c713f5d5f47cc89c0fb02b2270118f0d4e488c42db",
   "nhl_shots_and_special_teams": "c001b39260d8fc5795479021d7ec01d3586e82ce73e4c2893ec12a4f13f64ab3",
   "nba_recorded_pace_and_efficiency": "6ceef144a0447cf356f7d8db1febaa6f3adbba407b63f5a22a3a53dd58788d82",
   "nba_preseason_limited_no_pace": "253a1e8ad8c136fb0b0c58f29d72764edd2296b4d2bce5bcbab3e2052da1d4e9",
  ]
  let fixtures=try cases(),names=fixtures.map{$0["id"].string}
  XCTAssertEqual(fixtures.count,hashes.count,"Fixture count changed; update reviewed named projection expectations.")
  XCTAssertEqual(Set(names),Set(hashes.keys),"Fixture identities changed.")
  XCTAssertEqual(Set(names).count,names.count,"Duplicate fixture identities.")
  for c in fixtures {
   let name=c["id"].string
   guard let expected=hashes[name] else {XCTFail("Unexpected fixture: \(name)");continue}
   let clock=try XCTUnwrap(c["now"].number.flatMap(Double.init),"Missing clock: \(name)")
   let p=try RichSportsContext.project(Data(c["context"].canonical.utf8),now:Date(timeIntervalSince1970:clock))
   XCTAssertEqual(RichSportsContext.digest(p),expected,name)
  }
 }
 func testInvalidDerivationAndSourceRejected()throws {
  let c=try cases()[0],b=c["context"],now=Date(timeIntervalSince1970:Double(c["now"].number!)!);var fs=b["facts"].array
  fs[0]=fs[0].setting("value",.number("999"));XCTAssertThrowsError(try RichSportsContext.validate(Data(b.setting("facts",.list(fs)).canonical.utf8),now:now))
  var ss=b["sources"].array;ss[0]=ss[0].setting("url",.string("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?token=secret"));XCTAssertThrowsError(try RichSportsContext.validate(Data(b.setting("sources",.list(ss)).canonical.utf8),now:now))
 }
 func testStartedSuppressesAndAgingDoesNotChangeFingerprint()throws {
  let c=try cases()[0],raw=Data(c["context"].canonical.utf8),now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
  let a=try RichSportsContext.project(raw,now:now),b=try RichSportsContext.project(raw,now:RichSportsContext.time(c["context"]["event"]["startISO"]))
  XCTAssertEqual(a["fingerprint"],b["fingerprint"]);XCTAssertTrue(b["facts"].array.isEmpty);XCTAssertEqual(b["state"].string,"missing");XCTAssertTrue(b["excluded"].array.allSatisfy{$0["reason"].string=="event_started"})
 }
 func testProviderSpoofAndCoverageInflationRejected()throws {
  let c=try cases()[0],b=c["context"],now=Date(timeIntervalSince1970:Double(c["now"].number!)!)
  var sources=b["sources"].array;sources[0]=sources[0].setting("provider",.string("NBA"))
  XCTAssertThrowsError(try RichSportsContext.validate(Data(b.setting("sources",.list(sources)).canonical.utf8),now:now))
  var coverage=b["coverage"].array;let i=try XCTUnwrap(coverage.firstIndex{$0["state"].string=="missing"});coverage[i]=coverage[i].setting("state",.string("available")).setting("sourceIDs",b["facts"].array[0]["sourceIDs"])
  XCTAssertThrowsError(try RichSportsContext.validate(Data(b.setting("coverage",.list(coverage)).canonical.utf8),now:now))
 }
 func testPartialVolumeWorkloadAndSpecialTeamsCoverage()throws {
  let fixtures=try cases()
  func projectedCoverage(sport:String,metrics:[String])throws->[RichContextJSON] {
   let b=try XCTUnwrap(fixtures.first{$0["context"]["event"]["sport"].string==sport})["context"]
   let template=try XCTUnwrap(b["facts"].array.first)
   let facts=metrics.map {template.setting("metric",.string($0))}
   return try RichSportsContext.coverage(b,facts:facts,designations:[]).filter{$0["team"]==template["team"]}
  }
  let nfl=try projectedCoverage(sport:"NFL",metrics:["offensivePlaysPerGame","secondsPerOffensivePlay"])
  XCTAssertEqual(nfl.first{$0["topic"].string=="pace"}?["state"].string,"partial")
  let mlb=try projectedCoverage(sport:"MLB",metrics:["pitchesLast3Days","outsLast3Days"])
  XCTAssertEqual(mlb.first{$0["topic"].string=="bullpenWorkload"}?["state"].string,"partial")
  XCTAssertEqual(mlb.first{$0["topic"].string=="bullpenAvailability"}?["state"].string,"missing")
  let nhl=try projectedCoverage(sport:"NHL",metrics:["powerPlayGoalsPerGame","shortHandedGoalsPerGame"])
  XCTAssertEqual(nhl.first{$0["topic"].string=="specialTeams"}?["state"].string,"partial")
 }
 func testDuplicateJSONKeysRejected() {var parser=RichContextJSONParser(Data(#"{"x":1,"x":2}"#.utf8));XCTAssertThrowsError(try parser.parse())}
}
