import XCTest
@testable import PLONNCore

final class CurrentMatchupOutlookTests:XCTestCase {
 let now=ISO8601DateFormatter().date(from:"2026-10-19T14:00:00Z")!
 func event(_ sport:String)->SharedMatchupEvent { .init(sport:sport,eventID:"espn:12345",homeTeam:"Home",awayTeam:"Away",startISO:"2026-10-20T20:00:00Z") }
 func profile(_ team:String,_ sport:String,_ metric:String,_ value:Double)->InsightTeamProfile {
  .init(team:team,sample:12,offense:[metric:value],allowed:[metric:value+1],source:"Verified provider season statistics",periodKey:"\(sport)|\(["NBA","NHL"].contains(sport) ? 2027:2026)|regular",through:now.addingTimeInterval(-30))
 }
 func testCurrentStatsWorkAcrossLaunchSportsWithoutArchiveOrAI() {
  for (sport,key,metric) in [("NFL","totalPointsPerGame","pointsPerGame"),("NBA","avgPoints","pointsPerGame"),("MLB","runsPerGame","runsPerGame"),("NHL","avgGoals","goalsPerGame")] {
   let rows=[profile("Home",sport,key,12),profile("Away",sport,key,10)]
   let facts=MatchupOutlookSelection.currentFacts(event:event(sport),profiles:rows,now:now)
   XCTAssertEqual(facts.count,4,sport);XCTAssertEqual(Set(facts.map(\.metric)),[metric])
   XCTAssertTrue(facts.allSatisfy {$0.cohort != nil && $0.expiresAt != nil})
   XCTAssertEqual(facts,MatchupOutlookSelection.currentFacts(event:event(sport),profiles:rows.reversed(),now:now),"Input ordering must not restart the outlook task")
  }
 }
 func testStaleFutureDuplicateAndMixedPeriodProfilesAreRejected() {
  let a=profile("Home","NFL","rushingYardsPerGame",120),b=profile("Away","NFL","rushingYardsPerGame",100)
  var stale=b;stale.through=now.addingTimeInterval(-901)
  var future=b;future.through=now.addingTimeInterval(1)
  var preseason=b;preseason.periodKey="NFL|2026|preseason"
  for rows in [[a,stale],[a,future],[a,preseason],[a,a,b]] {
   XCTAssertTrue(MatchupOutlookSelection.currentFacts(event:event("NFL"),profiles:rows,now:now).isEmpty)
  }
 }
 func testLeagueBenchmarkRequiresCompleteSamePeriodAndWeightsGames() {
  var rows=[profile("Home","NFL","rushingYardsPerGame",120),profile("Away","NFL","rushingYardsPerGame",100)]
  XCTAssertFalse(MatchupOutlookSelection.currentFacts(event:event("NFL"),profiles:rows,now:now).contains{$0.role=="reference"})
  rows += (0..<30).map {profile("Team\($0)","NFL","rushingYardsPerGame",110)}
  let ref=MatchupOutlookSelection.currentFacts(event:event("NFL"),profiles:rows,now:now).filter{$0.role=="reference"}
  XCTAssertEqual(ref.count,1);XCTAssertEqual(ref.first?.value,110);XCTAssertEqual(ref.first?.sample,384)
 }
 func testSearchMatchesNameOrTeamWithAccentsAndEveryWord() {
  XCTAssertTrue(LabNameSearch.matches(query:"  jose  ",name:"José Ramírez"))
  XCTAssertTrue(LabNameSearch.matches(query:"dallas",name:"Dak Prescott",detail:"Dallas Cowboys · QB"))
  XCTAssertFalse(LabNameSearch.matches(query:"dak dolphins",name:"Dak Prescott",detail:"Dallas Cowboys"))
  XCTAssertFalse(LabNameSearch.matches(query:"  ",name:"Dak Prescott"))
  XCTAssertLessThan(LabNameSearch.rank(query:"Dak",name:"Dak Prescott"),LabNameSearch.rank(query:"Dak",name:"Something Dak"))
 }
}
