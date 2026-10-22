import XCTest
@testable import PLONNCore

@MainActor final class AnalyticsPageCacheTests:XCTestCase {
 private let date=sportsDate("2026-10-15T12:00:00Z")!
 private func candidate(eventID:String="event",price:Double = -110)->AnalyticsFeedCandidate {
  let event=AnalyticsEvent(sport:"NBA",providerID:eventID,home:"Home",away:"Away",start:date.addingTimeInterval(3600),homeID:"1",awayID:"2")
  let proof=AnalyticsFeedVerifiedSubject(sport:"NBA",subject:.team,name:"Home",providerID:"1",marketKeys:["spreads"],event:event,sourceURL:"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",verifiedAt:date)
  let market=Market(id:"team",event:"",sport:"NBA",kind:"team",name:"Home",opponent:"",label:"",aliases:"",p:nil,price:0,market:"spreads",book:"",quotes:[],updated:nil,mode:"analytics")
  var quote=market;quote.id="quote";quote.event=eventID;quote.price=price;quote.mode="live";quote.book="Book";quote.line = -2.5;quote.side="Home";quote.updated=ISO8601DateFormatter().string(from:date);quote.commence=ISO8601DateFormatter().string(from:event.start);quote.eventLabel="Away at Home"
  return AnalyticsFeedCandidate(id:market.id,subject:.team,market:market,quotes:[quote],verifiedSubject:proof,recordedSnapshot:true)
 }
 private func history(_ c:AnalyticsFeedCandidate)->AnalyticsHistoryResult {
  AnalyticsHistoryEngine.statistics(market:c.market,observations:[AnalyticsObservation(id:"past",date:date.addingTimeInterval(-86400),opponent:"Other",scored:100,allowed:95)],now:date)
 }
 private func snapshot(_ c:AnalyticsFeedCandidate,discovered:Date?=nil,historyDate:Date?=nil)->AnalyticsPageCache.Snapshot {
  .init(candidates:[c],results:[c.id:history(c)],historyDates:[c.id:historyDate ?? date],warnings:[],discoveredAt:discovered ?? date)
 }
 private func configure(_ controller:AnalyticsFeedController,_ scope:String="account|backend") {
  controller.configure(scopeID:scope,snapshots:[],read:{_,_,_ in throw URLError(.resourceUnavailable)})
 }
 private func settle(_ ready:()->Bool) async {
  for _ in 0..<1000 {if ready(){return};await Task.yield()}
  XCTAssertTrue(ready(),"Controlled async work did not settle")
 }
 func testSelectedStatGetsNextHistorySlotWithoutExceedingConcurrency() async {
  let original=candidate()
  let candidates=(0..<5).map { index in
   AnalyticsFeedCandidate(id: "stat-\(index)", subject: original.subject, market: original.market, quotes: original.quotes, verifiedSubject: original.verifiedSubject)
  }
  var started:[String]=[]
  var pending:[String:CheckedContinuation<AnalyticsHistoryResult,Never>]=[:]
  let controller=AnalyticsFeedController(cache:AnalyticsPageCache(),now:{self.date},load:{_,_ in .init(candidates:candidates,warnings:[])},history:{ c in
   started.append(c.id)
   return await withCheckedContinuation { pending[c.id]=$0 }
  })
  configure(controller);controller.resume()
  await settle{started.count == 2}
  controller.prioritize("stat-4")
  XCTAssertEqual(started.count,2)
  pending.removeValue(forKey:"stat-0")?.resume(returning:history(original))
  await settle{started.count == 3}
  XCTAssertEqual(started,["stat-0","stat-1","stat-4"])
  XCTAssertEqual(controller.loadingIDs.count,2)
  controller.pause()
  for continuation in pending.values { continuation.resume(returning:history(original)) }
 }
 func testCarouselKeepsLargeRosterButOnlyLoadsSelectedHistories() async {
  let original=candidate(),cache=AnalyticsPageCache()
  let candidates=(0..<80).map { index in
   AnalyticsFeedCandidate(id:"stat-\(index)",subject:original.subject,market:original.market,quotes:[],verifiedSubject:original.verifiedSubject)
  }
  var loaded:[String]=[]
  let controller=AnalyticsFeedController(cache:cache,now:{self.date},load:{_,_ in .init(candidates:candidates,warnings:[])},history:{ c in
   loaded.append(c.id);return self.history(c)
  })
  configure(controller);controller.resume()
  await settle{controller.results.count == 2}
  for _ in 0..<30 {await Task.yield()}
  XCTAssertEqual(controller.candidates.count,80)
  XCTAssertEqual(Set(loaded),["stat-0","stat-1"])
  controller.prioritize("stat-79")
  await settle{controller.results["stat-79"] != nil}
  XCTAssertEqual(loaded.count,3)
  controller.pause()
  let restored=AnalyticsFeedController(cache:cache,now:{self.date},load:{_,_ in XCTFail("Roster should be cached");return .init(candidates:[],warnings:[])},history:{self.history($0)})
  configure(restored)
  XCTAssertEqual(restored.candidates.count,80)
  XCTAssertNotNil(restored.results["stat-79"])
 }
 func testGuestLoadsWithoutOddsReaderAndAccountChangesClearPriorCards() async {
  let priced=candidate()
  let publicCandidate=AnalyticsFeedCandidate(id:priced.id,subject:priced.subject,market:priced.market,quotes:[],verifiedSubject:priced.verifiedSubject)
  var authenticatedReads:[Bool]=[]
  let controller=AnalyticsFeedController(cache:AnalyticsPageCache(),now:{self.date},load:{_,read in
   authenticatedReads.append(read != nil)
   return .init(candidates:[read == nil ? publicCandidate:priced],warnings:[])
  },history:{self.history($0)})
  controller.configure(scopeID:"guest|statistics",snapshots:[]);controller.resume()
  await settle{controller.results[priced.id] != nil}
  XCTAssertEqual(authenticatedReads,[false]);XCTAssertTrue(controller.candidates[0].quotes.isEmpty)
  configure(controller,"account|odds");XCTAssertTrue(controller.candidates.isEmpty);controller.resume()
  await settle{controller.candidates.first?.quotes.isEmpty == false}
  controller.configure(scopeID:"guest|statistics",snapshots:[])
  XCTAssertTrue(controller.candidates.isEmpty);XCTAssertTrue(controller.results.isEmpty)
  controller.resume();await settle{controller.results[priced.id] != nil}
  XCTAssertEqual(authenticatedReads,[false,true,false]);XCTAssertTrue(controller.candidates[0].quotes.isEmpty)
  controller.pause()
 }
 func testNewControllerReusesCompletedPageWithoutDiscoveryOrHistoryRead() async throws {
  let cache=AnalyticsPageCache(),c=candidate(),h=history(candidate());var loads=0,histories=0
  let load:AnalyticsFeedController.Loader={_,_ in loads+=1;return .init(candidates:[c],warnings:[])}
  let history:AnalyticsFeedController.HistoryLoader={_ in histories+=1;return h}
  let first=AnalyticsFeedController(cache:cache,now:{self.date},load:load,history:history)
  configure(first);first.resume();await settle{first.results[c.id] != nil};first.pause()
  let second=AnalyticsFeedController(cache:cache,now:{self.date.addingTimeInterval(100)},load:load,history:history)
  configure(second)
  XCTAssertEqual(second.candidates.map(\.id),[c.id]);XCTAssertNotNil(second.results[c.id])
  second.resume();for _ in 0..<20 {await Task.yield()}
  XCTAssertEqual(loads,1);XCTAssertEqual(histories,1)
  XCTAssertEqual(second.candidates[0].quotes,c.quotes)
  XCTAssertEqual(second.candidates[0].verifiedSubject?.verifiedAt,c.verifiedSubject?.verifiedAt)
  second.pause()
 }
 func testFiveMinuteRefreshUpdatesQuotesWithoutReloadingUnchangedHistory() async throws {
  let cache=AnalyticsPageCache();var current=date,loads=0,histories=0
  let old=candidate(),new=candidate(price:105),h=history(candidate())
  let controller=AnalyticsFeedController(cache:cache,now:{current},load:{_,_ in loads+=1;return .init(candidates:[loads == 1 ? old:new],warnings:[])},history:{_ in histories+=1;return h})
  configure(controller);controller.resume();await settle{controller.results[old.id] != nil}
  current=date.addingTimeInterval(300);controller.refreshIfNeeded()
  await settle{controller.candidates.first?.quotes.first?.price == 105}
  XCTAssertEqual(loads,2);XCTAssertEqual(histories,1);XCTAssertNotNil(controller.results[old.id])
  XCTAssertEqual(controller.candidates[0].quotes[0].updated,old.quotes[0].updated)
  controller.pause()
 }
 func testAccountAndBackendSwitchClearPageAndOldLeaseCannotRepopulate() async throws {
  let cache=AnalyticsPageCache(),c=candidate(),page=snapshot(candidate())
  let a=cache.activate(scopeID:"alice|one");cache.save(page,scopeID:"alice|one",lease:a)
  XCTAssertNotNil(cache.load(scopeID:"alice|one",lease:a,now:date))
  let b=cache.activate(scopeID:"bob|one")
  XCTAssertNil(cache.load(scopeID:"bob|one",lease:b,now:date))
  cache.save(page,scopeID:"alice|one",lease:a)
  let newA=cache.activate(scopeID:"alice|one")
  cache.save(page,scopeID:"alice|one",lease:a)
  XCTAssertNil(cache.load(scopeID:"alice|one",lease:newA,now:date))
  cache.save(page,scopeID:"alice|one",lease:newA)
  let otherBackend=cache.activate(scopeID:"alice|two")
  XCTAssertNil(cache.load(scopeID:"alice|two",lease:otherBackend,now:date))
  let controller=AnalyticsFeedController(cache:cache,now:{self.date},load:{_,_ in .init(candidates:[c],warnings:[])},history:{_ in self.history(c)})
  configure(controller);controller.resume();await settle{controller.results[c.id] != nil}
  configure(controller,"guest|backend")
  XCTAssertTrue(controller.candidates.isEmpty);XCTAssertTrue(controller.results.isEmpty)
  controller.pause()
 }
 func testCacheExpiresExactlyAtFiveMinutesAndCannotRenewItsClock() async throws {
  let cache=AnalyticsPageCache(),page=snapshot(candidate()),lease=cache.activate(scopeID:"a")
  cache.save(page,scopeID:"a",lease:lease)
  XCTAssertNotNil(cache.load(scopeID:"a",lease:lease,now:date.addingTimeInterval(299)))
  cache.save(page,scopeID:"a",lease:lease)
  XCTAssertNil(cache.load(scopeID:"a",lease:lease,now:date.addingTimeInterval(300)))
  XCTAssertNil(cache.load(scopeID:"a",lease:lease,now:date.addingTimeInterval(-1)))
 }
 func testHistoryReuseExpiresAndRejectsNewEventOrIdentity() async throws {
  let c=candidate(),page=snapshot(candidate())
  XCTAssertNotNil(AnalyticsPageCache.retained(from:page,candidates:[c],now:date.addingTimeInterval(899))[c.id])
  XCTAssertTrue(AnalyticsPageCache.retained(from:page,candidates:[c],now:date.addingTimeInterval(900)).isEmpty)
  XCTAssertTrue(AnalyticsPageCache.retained(from:page,candidates:[candidate(eventID:"second-game")],now:date).isEmpty)
  var market=c.market;market.name="Other"
  let changed=AnalyticsFeedCandidate(id:c.id,subject:c.subject,market:market,quotes:c.quotes,verifiedSubject:c.verifiedSubject)
  XCTAssertTrue(AnalyticsPageCache.retained(from:page,candidates:[changed],now:date).isEmpty)
 }
 func testRestoreDoesNotResurrectExpiredHistoryEvenWithNewDiscoveryTime() async throws {
  let cache=AnalyticsPageCache(),lease=cache.activate(scopeID:"a")
  cache.save(snapshot(candidate(),discovered:date.addingTimeInterval(850)),scopeID:"a",lease:lease)
  let restored=try XCTUnwrap(cache.load(scopeID:"a",lease:lease,now:date.addingTimeInterval(900)))
  XCTAssertTrue(restored.results.isEmpty)
 }
 func testManualHistoryRetryAndStatusRefreshStillWorkAfterRestore() async throws {
  let cache=AnalyticsPageCache(),c=candidate(),h=history(candidate()),lease=cache.activate(scopeID:"account|backend")
  cache.save(snapshot(c),scopeID:"account|backend",lease:lease)
  var histories=0,statuses=0
  let controller=AnalyticsFeedController(cache:cache,now:{self.date},load:{_,_ in XCTFail("Cached page rediscovered");return .init(candidates:[],warnings:[])},history:{_ in histories+=1;return h},status:{items in statuses+=1;return .init(candidates:items,warnings:[])})
  configure(controller);controller.resume();controller.refreshStatusIfNeeded()
  await settle{statuses == 1}
  controller.retry(c.id);await settle{histories == 1 && controller.results[c.id] != nil}
  controller.pause();controller.refreshStatusIfNeeded()
  XCTAssertEqual(statuses,1)
 }
 func testCancelledDiscoveryCannotOverwriteNewScope() async throws {
  let cache=AnalyticsPageCache(),c=candidate();var continuation:CheckedContinuation<AnalyticsFeedBuildResult,Never>?,calls=0
  let controller=AnalyticsFeedController(cache:cache,now:{self.date},load:{_,_ in
   calls+=1
   if calls == 1 {return await withCheckedContinuation{continuation=$0}}
   return .init(candidates:[],warnings:["new scope"])
  },history:{_ in self.history(c)})
  configure(controller,"a");controller.resume();await settle{continuation != nil}
  configure(controller,"b");controller.resume();await settle{controller.warnings == ["new scope"]}
  continuation?.resume(returning:.init(candidates:[c],warnings:["old scope"]))
  for _ in 0..<30 {await Task.yield()}
  XCTAssertEqual(controller.scopeID,"b");XCTAssertTrue(controller.candidates.isEmpty);XCTAssertEqual(controller.warnings,["new scope"])
  controller.pause()
 }
 func testLeagueSwitchRestoresItsOwnPageWithoutReloadingAndAccountChangeClearsAllLeagues() async {
  let cache=AnalyticsPageCache(),nba=candidate()
  var nflMarket=nba.market;nflMarket.sport="NFL";nflMarket.id="nfl-team"
  let nfl=AnalyticsFeedCandidate(id:nflMarket.id,subject:.team,market:nflMarket,quotes:[],verifiedSubject:nil)
  var loads=0
  let controller=AnalyticsFeedController(cache:cache,now:{self.date},load:{_,_ in
   loads+=1;return .init(candidates:[loads == 1 ? nba:nfl],warnings:[])
  },history:{self.history($0)})
  controller.configure(scopeID:"account",snapshots:[],sport:"NBA");controller.resume()
  await settle{controller.results[nba.id] != nil}
  controller.configure(scopeID:"account",snapshots:[],sport:"NFL")
  XCTAssertTrue(controller.candidates.isEmpty);controller.resume()
  await settle{controller.results[nfl.id] != nil}
  controller.configure(scopeID:"account",snapshots:[],sport:"NBA");controller.resume()
  XCTAssertEqual(controller.candidates.map(\.id),[nba.id]);XCTAssertNotNil(controller.results[nba.id])
  for _ in 0..<20 {await Task.yield()}
  XCTAssertEqual(loads,2)
  controller.configure(scopeID:"guest",snapshots:[],sport:"NBA")
  XCTAssertTrue(controller.candidates.isEmpty)
  controller.configure(scopeID:"account",snapshots:[],sport:"NFL")
  XCTAssertTrue(controller.candidates.isEmpty,"Signing out must clear every cached league")
  controller.pause()
 }
 func testLateLeagueDiscoveryCannotReplaceTheNewSelection() async {
  var pending:CheckedContinuation<AnalyticsFeedBuildResult,Never>?,calls=0
  let nba=candidate()
  let controller=AnalyticsFeedController(cache:AnalyticsPageCache(),now:{self.date},load:{_,_ in
   calls+=1
   if calls == 1 {return await withCheckedContinuation{pending=$0}}
   return .init(candidates:[nba],warnings:[])
  },history:{self.history($0)})
  controller.configure(scopeID:"same-account",snapshots:[],sport:"NFL");controller.resume()
  await settle{pending != nil}
  controller.configure(scopeID:"same-account",snapshots:[],sport:"NBA");controller.resume()
  await settle{controller.results[nba.id] != nil}
  pending?.resume(returning:.init(candidates:[],warnings:["Old NFL response"]))
  for _ in 0..<30 {await Task.yield()}
  XCTAssertEqual(controller.sport,"NBA");XCTAssertEqual(controller.candidates.map(\.id),[nba.id]);XCTAssertTrue(controller.warnings.isEmpty)
  controller.pause()
 }
}
