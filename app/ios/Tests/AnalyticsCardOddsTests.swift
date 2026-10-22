import XCTest
@testable import PLONNCore

final class AnalyticsCardOddsTests:XCTestCase {
 private let now=sportsDate("2026-10-15T12:00:00Z")!
 private func quote(_ key:String="spreads",sport:String="NBA",name:String="Fixture Home")->Market {
  let player=key.hasPrefix("player_") || key.hasPrefix("batter_")
  return Market(id:key+"-"+sport,event:"provider-event",sport:sport,kind:player ? "prop":key == "totals" ? "game":"team",name:key == "totals" ? "Over":name,opponent:key == "totals" ? "Fixture Home":"Fixture Away",label:"Full game",aliases:"",p:nil,price:-110,market:key,book:"Fixture Book",quotes:[],updated:ISO8601DateFormatter().string(from:now.addingTimeInterval(-1200)),mode:"live",line:key == "h2h" ? nil:player ? 20.5:key == "spreads" ? -2.5:210.5,eventLabel:"Fixture Away at Fixture Home",commence:ISO8601DateFormatter().string(from:now.addingTimeInterval(3600)),side:player || key.contains("totals") ? "over":"home")
 }
 private func candidate(_ key:String,rows:[Market]?=nil,sport:String="NBA") throws -> AnalyticsFeedCandidate {
  let player=key.hasPrefix("player_") || key.hasPrefix("batter_")
  let name=player ? "Fixture Player":"Fixture Home"
  let event=AnalyticsEvent(sport:sport,providerID:"public-event",home:"Fixture Home",away:"Fixture Away",start:now.addingTimeInterval(3600),homeID:"1",awayID:"2")
  let proof=AnalyticsFeedVerifiedSubject(sport:sport,subject:player ? .player:.team,name:name,providerID:player ? "77":"1",teamName:player ? "Fixture Home":nil,teamID:player ? "1":nil,marketKeys:[key],event:event,sourceURL:"https://site.api.espn.com/apis/site/v2/sports/"+PlayerHistoryService.leaguePath(sport)+(player ? "/teams/1/roster":"/scoreboard"),verifiedAt:now)
  return try XCTUnwrap(AnalyticsFeedCore.build(snapshots:rows ?? [quote(key,sport:sport,name:name)],verifiedSubjects:[proof],now:now,recordedSnapshot:true).candidates.first{$0.market.market == key})
 }
 private func history(_ c:AnalyticsFeedCandidate)->AnalyticsHistoryResult {
  var h=AnalyticsHistoryEngine.statistics(market:c.market,observations:[AnalyticsObservation(id:"past",date:now.addingTimeInterval(-86400),opponent:"Fixture Away",value:25,scored:110,allowed:100)],now:now)
  h.verifiedPlayerID=c.subject == .player ? "77":nil
  return h
 }
 func testScoreDifferenceUsesSpreadsInsteadOfMoneylineInEverySport() throws {
  for sport in ["NFL","NBA","MLB","NHL"] {
   let spread=quote(sport:sport),moneyline=quote("h2h",sport:sport)
   let c=try candidate("h2h",rows:[moneyline,spread],sport:sport)
   XCTAssertEqual(AnalyticsFeedCore.cardOdds(candidate:c,history:history(c),now:now),[spread])
   XCTAssertEqual(AnalyticsFeedCore.eligibleComparisons(candidate:c,history:history(c),now:now),[moneyline],"Original source contract unchanged")
   XCTAssertEqual(AnalyticsFeedCore.cardOddsTitle(c),"Spread odds")
   XCTAssertEqual(AnalyticsFeedCore.cardOddsContract(spread),"Fixture Home -2.5")
   XCTAssertEqual(c.quotes.first{$0.market == "spreads"}?.updated,spread.updated)
  }
 }
 func testMissingSpreadDoesNotSubstituteMoneylineOrDifferentTeam() throws {
  var other=quote();other.name="Fixture Away";other.id="away"
  let c=try candidate("h2h",rows:[quote("h2h"),other])
  XCTAssertTrue(AnalyticsFeedCore.cardOdds(candidate:c,history:history(c),now:now).isEmpty)
 }
 func testTotalsTeamTotalsAndPlayerStatsUseTheirExactMarkets() throws {
  for key in ["spreads","totals","team_totals","player_points","player_points_alternate","player_rebounds"] {
   let c=try candidate(key),prices=AnalyticsFeedCore.cardOdds(candidate:c,history:history(c),now:now)
   XCTAssertEqual(prices.count,1,key);XCTAssertEqual(prices.first?.market,key)
   XCTAssertEqual(prices.first?.line,c.quotes.first?.line);XCTAssertEqual(prices.first?.price,-110)
  }
 }
 func testDifferentLineAndBookRemainPairedAndUnchanged() throws {
  var alternate=quote();alternate.id="other";alternate.line = -3.5;alternate.price=105;alternate.book="Other Book"
  let c=try candidate("h2h",rows:[quote("h2h"),quote(),alternate])
  let prices=AnalyticsFeedCore.cardOdds(candidate:c,history:history(c),now:now)
  XCTAssertEqual(Set(prices),Set([quote(),alternate]))
  XCTAssertEqual(prices.first{$0.book == "Other Book"}?.price,105)
  XCTAssertEqual(prices.first{$0.book == "Other Book"}?.line,-3.5)
 }
 func testWrongEventPeriodPlayerAndExpiredEvidenceCannotSupplyPrices() throws {
  for mutation in 0..<5 {
   var row=quote()
   switch mutation {
   case 0:row.eventLabel="Different Away at Different Home"
   case 1:row.commence=ISO8601DateFormatter().string(from:now.addingTimeInterval(7200))
   case 2:row.label="First half"
   case 3:row.line=nil
   default:row.updated=ISO8601DateFormatter().string(from:now.addingTimeInterval(1))
   }
   let c=try candidate("h2h",rows:[quote("h2h"),row])
   XCTAssertTrue(AnalyticsFeedCore.cardOdds(candidate:c,history:history(c),now:now).isEmpty)
  }
  let player=try candidate("player_points");var wrong=history(player);wrong.verifiedPlayerID="999"
  XCTAssertTrue(AnalyticsFeedCore.cardOdds(candidate:player,history:wrong,now:now).isEmpty)
  let c=try candidate("h2h",rows:[quote("h2h"),quote()])
  XCTAssertTrue(AnalyticsFeedCore.cardOdds(candidate:c,history:history(c),now:now.addingTimeInterval(901)).isEmpty)
 }
 func testStartedAndFinalEventsCannotShowPregamePricesWithoutFreshArchiveStatus() throws {
  let c=try candidate("h2h",rows:[quote("h2h"),quote()])
  var proof=try XCTUnwrap(c.verifiedSubject);var event=try XCTUnwrap(proof.event)
  event.lifecycle = .final;proof.event=event
  let ended=AnalyticsFeedCandidate(id:c.id,subject:c.subject,market:c.market,quotes:c.quotes,verifiedSubject:proof,recordedSnapshot:true,recordedArchive:true)
  XCTAssertTrue(AnalyticsFeedCore.cardOdds(candidate:ended,history:history(ended),now:now).isEmpty)
 }
}
