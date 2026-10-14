import XCTest
@testable import PLONNCore
final class NarrativeInjuryContextTests:XCTestCase {
 let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("backend/local_lab/verification/injuries-20261014")
 func read(_ name:String)throws->RichContextJSON {var p=RichContextJSONParser(try Data(contentsOf:root.appendingPathComponent(name)));return try p.parse()}
 func clock()throws->Date {Date(timeIntervalSince1970:try NarrativeInjuryContext.time(read("source-and-projection-receipt.json")["evaluationClock"]))}
 func testActualFivePacketsExactPythonProjectionAndBoundedModel()throws {
  let hashes=["mlb-401908032":"f05d31a6474bdba32d2ab83a2045e4da83473095775ea658765ef585575ff54c","nba-401898404":"73e084bbe72d1d32c8188d1c781bcc3a4221a6cb7a4be77fcdb1b21a9603af27","nfl-401872995":"3ec35bf5e2f89fc02e34ca81cd437e36a4c6c92eb699cbaa7ac940e11b7e0a80","nfl-401873001":"a3c270168bb866c106c6388ad7637846e7747948460c6b5a4e8ffe82cfd4cdbd","nhl-401892504":"a86a2375b5c0fadc765d392cbff3411a355c13a101f21ef34ff79209d92f18ab"]
  for (name,hash) in hashes {
   let packet=try read(name+"-packet.json"),now=try clock()
   XCTAssertEqual(try NarrativeInjuryContext.project(packet,now:now),try read(name+"-projection.json"),name)
   let model=try NarrativeInjuryContext.model(packet,now:now)
   XCTAssertEqual(NarrativeInjuryContext.hash(model),hash,name);XCTAssertLessThanOrEqual(model.canonical.utf8.count,6144);XCTAssertLessThanOrEqual(model["observations"].array.count,6)
  }
 }
 func testStatusAndBindingTamperingRejected()throws {
  let p=try read("nfl-401873001-packet.json"),now=try clock();var rows=p["observations"].array
  rows[0]=rows[0].setting("statusKind",.string("available"));XCTAssertThrowsError(try NarrativeInjuryContext.validate(p.setting("observations",.list(rows)),now:now))
  rows=p["observations"].array;rows[0]=rows[0].setting("eventID",p["event"]["eventID"]);rows[0]=rows[0].setting("id",.string(NarrativeInjuryContext.id(rows[0])))
  XCTAssertThrowsError(try NarrativeInjuryContext.validate(p.setting("observations",.list(rows)),now:now))
 }
 func testFreshnessNotRenewedByFeedAndStartSuppresses()throws {
  let p=try read("nfl-401873001-packet.json"),now=try clock(),initial=try NarrativeInjuryContext.project(p,now:now)
  let aged=try NarrativeInjuryContext.project(p,now:now.addingTimeInterval(21600))
  XCTAssertEqual(initial["injuryFingerprint"],aged["injuryFingerprint"]);XCTAssertEqual(aged["state"],.string("stale"));XCTAssertTrue(aged["observations"].array.isEmpty)
  let started=try NarrativeInjuryContext.project(p,now:Date(timeIntervalSince1970:NarrativeInjuryContext.time(p["event"]["startISO"])))
  XCTAssertTrue(started["observations"].array.isEmpty)
 }
 func testStrictShapeSourceAndCoverage()throws {
  let p=try read("mlb-401908032-packet.json"),now=try clock()
  XCTAssertThrowsError(try NarrativeInjuryContext.validate(p.setting("extra",.bool(true)),now:now))
  var sources=p["sources"].array;sources[0]=sources[0].setting("url",.string("https://evil.example/injuries"));XCTAssertThrowsError(try NarrativeInjuryContext.validate(p.setting("sources",.list(sources)),now:now))
  var coverage=p["coverage"].array;coverage[0]=coverage[0].setting("excludedRows",.bool(false));XCTAssertThrowsError(try NarrativeInjuryContext.validate(p.setting("coverage",.list(coverage)),now:now))
 }
 func testFourSyntheticPromptViews()throws {
  let f=try read("synthetic-contract-fixtures.json"),now=Date(timeIntervalSince1970:try NarrativeInjuryContext.time(f["evaluationClock"]))
  for c in f["fixtures"].array {XCTAssertEqual(try NarrativeInjuryContext.model(c["packet"],now:now,maxObservations:12,maxBytes:12288),c["promptView"],c["sport"].string)}
 }
 func testConflictingFreshStatusExcluded()throws {
  let p=try read("nfl-401873001-packet.json"),now=try clock()
  let projection=try NarrativeInjuryContext.project(p,now:now)
  let original=try XCTUnwrap(projection["observations"].array.first).removing(["assertion","eventStatusConfirmed","freshness","claimID"])
  let source=p["sources"].array[0].setting("id",.string("second-source"))
  var extra=original.setting("sourceID",.string("second-source")).setting("reportedStatus",.string(original["reportedStatus"] == .string("Out") ? "Questionable":"Out")).setting("statusKind",.string(original["reportedStatus"] == .string("Out") ? "questionable":"out")).setting("scope",.string("general"))
  extra=extra.setting("id",.string(NarrativeInjuryContext.id(extra)))
  let coverage=p["coverage"].array.map{$0["teamID"] == extra["teamID"] ? $0.setting("sourceIDs",.list($0["sourceIDs"].array+[.string("second-source")])):$0}
  let changed=p.setting("sources",.list(p["sources"].array+[source])).setting("observations",.list(p["observations"].array+[extra])).setting("coverage",.list(coverage))
  let result=try NarrativeInjuryContext.project(changed,now:now)
  XCTAssertFalse(result["observations"].array.contains{$0["playerID"]==original["playerID"] && $0["teamID"]==original["teamID"]})
  XCTAssertTrue(result["excluded"].array.contains{$0["reason"] == .string("conflicting_status")})
 }
 func testOfficialSourceURLAndStatusBoundaries()throws {
  try NarrativeInjuryContext.sourceURL(.string("https://statsapi.mlb.com/api/v1/teams/147/roster?rosterType=fullRoster&hydrate=person(currentTeam)"),"MLB")
  for url in ["https://statsapi.mlb.com/api/v1/teams/147/roster?rosterType=fullRoster&hydrate=person(currentTeam)&token=", "https://statsapi.mlb.com:443/api/v1/teams/147/roster?rosterType=fullRoster&hydrate=person(currentTeam)","https://user@www.nfl.com/injuries/"] {XCTAssertThrowsError(try NarrativeInjuryContext.sourceURL(.string(url),url.contains("nfl") ? "NFL":"MLB"))}
  XCTAssertEqual(try NarrativeInjuryContext.official("NFL","Limited participation","practice"),"practice")
  XCTAssertThrowsError(try NarrativeInjuryContext.official("NBA","Limited participation","practice"))
  XCTAssertEqual(try NarrativeInjuryContext.official("MLB","Injured 10-Day","roster"),"il_10")
 }

}
