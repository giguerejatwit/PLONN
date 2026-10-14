#if ODDS_STANDALONE_TESTS
import XCTest
@testable import OddsRefreshCore

@MainActor final class OddsRefreshTests: XCTestCase {
    func market(_ id: String = "one", sport: String = "MLB", event: String = "event", kind: String = "team") -> Market {
        Market(id:id,event:event,sport:sport,kind:kind,name:"Home",opponent:"Away",label:"Home to win",aliases:"",price:-110,market:"h2h",book:"Book",quotes:[],mode:"live",eventLabel:"Away at Home",commence:"2026-09-30T22:00:00Z")
    }
    func account() -> AccountStore { AccountStore(owner: UUID()) }
    func store() -> Store { let store=Store();store.backendURL="";return store }

    func testFailedRefreshRetainsSnapshotAndTimestampThenRecoversEmpty() async {
        let account=account(), store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        let received=store.lastOddsRefresh
        account.client.functions.operation={ _ in throw URLError(.notConnectedToInternet) }
        await store.load("MLB",account:account)
        XCTAssertEqual(store.live.map(\.id),["one"])
        XCTAssertTrue(store.oddsStale)
        XCTAssertTrue(store.status.contains("Stale"))
        XCTAssertEqual(store.oddsIssue,store.status)
        XCTAssertEqual(store.lastOddsRefresh,received)
        XCTAssertFalse(store.loading)
        account.client.functions.operation={ _ in OddsResponse(markets:[],message:"No games offered") }
        await store.load("MLB",account:account)
        XCTAssertTrue(store.live.isEmpty)
        XCTAssertFalse(store.oddsStale)
        XCTAssertTrue(store.oddsLoadSucceeded)
        XCTAssertNil(store.oddsIssue)
    }

    func testFailedDifferentSportNeverReusesPreviousSport() async {
        let account=account(), store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        account.client.functions.operation={ _ in throw URLError(.timedOut) }
        await store.load("NBA",account:account)
        XCTAssertTrue(store.live.isEmpty)
        XCTAssertNil(store.lastOddsRefresh)
    }

    func testNewStoreCanRecoverSameScopeStaleSnapshot() async {
        let account=account(), first=store(), second=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await first.load("MLB",account:account)
        account.client.functions.operation={ _ in throw URLError(.timedOut) }
        await second.load("MLB",account:account)
        XCTAssertEqual(second.live.map(\.id),["one"])
        XCTAssertTrue(second.oddsStale)
    }

    func testAuthorizationFailureClearsRetainedPrices() async {
        let account=account(), store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        account.client.functions.operation={ _ in throw FunctionsError.httpError(401,Data()) }
        await store.load("MLB",account:account)
        XCTAssertTrue(store.live.isEmpty)
        XCTAssertTrue(store.oddsUnauthorized)
        XCTAssertEqual(store.oddsIssue,store.status)
        XCTAssertFalse(store.oddsStale)
    }

    func testConcurrentStoresShareOneTransportRequest() async {
        let account=account(), first=store(), second=store()
        var gate: CheckedContinuation<OddsResponse,Error>?
        var calls=0
        account.client.functions.operation={ _ in
            calls += 1
            return try await withCheckedThrowingContinuation { gate=$0 }
        }
        let a=Task { await first.load("MLB",account:account) }
        while gate == nil { await Task.yield() }
        let b=Task { await second.load("MLB",account:account) }
        while !second.loading { await Task.yield() }
        gate?.resume(returning:OddsResponse(markets:[market()],message:"Fresh"))
        await a.value;await b.value
        XCTAssertEqual(calls,1)
        XCTAssertEqual(first.live,second.live)
        XCTAssertTrue(second.oddsLoadSucceeded)
    }

    func testAccountSwitchImmediatelyClearsAndRejectsLateResponse() async {
        let account=account(), store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        var gate: CheckedContinuation<OddsResponse,Error>?
        account.client.functions.operation={ _ in try await withCheckedThrowingContinuation { gate=$0 } }
        let request=Task { await store.load("MLB",account:account) }
        while gate == nil { await Task.yield() }
        let original=account.user
        account.user=nil
        XCTAssertTrue(store.live.isEmpty)
        account.user=original // A -> signed out -> A must still reject the original request.
        gate?.resume(returning:OddsResponse(markets:[market("late")],message:"Late"))
        await request.value
        XCTAssertTrue(store.live.isEmpty)
        XCTAssertFalse(store.loading)
        XCTAssertNil(store.oddsIssue)
    }

    func testBackendChangeRejectsLateResponseAndClearsPreviousSource() async {
        let account=account(), store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        var gate: CheckedContinuation<OddsResponse,Error>?
        account.client.functions.operation={ _ in try await withCheckedThrowingContinuation { gate=$0 } }
        let request=Task { await store.load("MLB",account:account) }
        while gate == nil { await Task.yield() }
        store.backendURL="https://new-source.example"
        gate?.resume(returning:OddsResponse(markets:[market("late")],message:"Old backend"))
        await request.value
        XCTAssertTrue(store.live.isEmpty)
        XCTAssertNil(store.lastOddsRefresh)
        XCTAssertFalse(store.loading)
        store.backendURL=""
    }

    func testSlowOldSportCannotOverwriteNewSport() async {
        let account=account(), store=store()
        var gate: CheckedContinuation<OddsResponse,Error>?
        account.client.functions.operation={ body in
            if body["sport"] == "MLB" { return try await withCheckedThrowingContinuation { gate=$0 } }
            return OddsResponse(markets:[self.market("nba",sport:"NBA")],message:"NBA")
        }
        let old=Task { await store.load("MLB",account:account) }
        while gate == nil { await Task.yield() }
        await store.load("NBA",account:account,cacheOnly:true,nbaSeasonTypes:[1])
        gate?.resume(returning:OddsResponse(markets:[market()],message:"MLB"))
        await old.value
        XCTAssertEqual(store.live.map(\.sport),["NBA"])
        XCTAssertEqual(store.status,"NBA")
    }

    func testEventFilteringAndFailedPropsKeepOnlySameEvent() async {
        let account=account(), store=store()
        account.client.functions.operation={ body in
            if body["event"] != nil {
                return OddsResponse(markets:[self.market("prop",kind:"prop"),self.market("wrong-event",event:"other",kind:"prop"),self.market("wrong-sport",sport:"NBA",kind:"prop")],message:"Props")
            }
            return OddsResponse(markets:[self.market(),self.market("wrong",sport:"NBA")],message:"Slate")
        }
        await store.load("MLB",account:account,home:"Home",away:"Away",start:"2026-09-30T22:00:00Z")
        XCTAssertEqual(Set(store.live.map(\.id)),["one","prop"])
        account.client.functions.operation={ body in
            if body["event"] != nil { throw URLError(.timedOut) }
            return OddsResponse(markets:[self.market("updated")],message:"Slate")
        }
        await store.load("MLB",account:account,home:"Home",away:"Away",start:"2026-09-30T22:00:00Z")
        XCTAssertEqual(Set(store.live.map(\.id)),["updated","prop"])
        XCTAssertTrue(store.oddsStale)
        XCTAssertTrue(store.status.contains("Stale player props"))
        XCTAssertEqual(store.oddsIssue,store.status)
    }

    func testConfiguredServiceFailureIsVisibleAndRecoveryClearsIt() async {
        let account=account(),store=store()
        account.client.functions.operation={ _ in
            throw FunctionsError.httpError(503,Data(#"{"error":"Live odds are awaiting provider configuration"}"#.utf8))
        }
        await store.load("MLB",account:account)
        XCTAssertEqual(store.oddsIssue,"Live odds are temporarily unavailable. Try again later.")
        XCTAssertTrue(store.live.isEmpty)
        XCTAssertFalse(store.oddsLoadSucceeded)
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        XCTAssertNil(store.oddsIssue)
        XCTAssertTrue(store.oddsLoadSucceeded)
    }
    func testCachedOnlyQuotesKeepOriginalTimeAndCannotBeCurrentModelInput() async {
        let account=account(),store=store()
        let time="2026-10-09T19:52:46.682428Z"
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Provider refreshes paused until midnight Eastern.",cacheOnly:true,stale:true,fetchedAt:time,cacheOnlyUntil:"2026-10-10T04:00:00Z",quoteUpdatedAt:time) }
        await store.load("MLB",account:account)
        XCTAssertEqual(store.live.count,1)
        XCTAssertTrue(store.oddsCachedOnly);XCTAssertTrue(store.oddsStale);XCTAssertFalse(store.oddsLoadSucceeded)
        XCTAssertEqual(store.lastOddsRefresh,sportsDate(time))
        XCTAssertTrue(store.status.contains("Oct 9, 3:52:46 PM EDT"),store.status)
        let before=store.lastOddsRefresh
        await store.load("MLB",account:account)
        XCTAssertEqual(store.lastOddsRefresh,before)
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Fresh") }
        await store.load("MLB",account:account)
        XCTAssertFalse(store.oddsCachedOnly);XCTAssertFalse(store.oddsStale);XCTAssertNil(store.oddsIssue)
    }
    func testCacheMissRemainsUnavailableWithoutInventingAsOfTime() async {
        let account=account(),store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[],message:"No cached odds available.",cacheOnly:true,stale:true) }
        await store.load("NHL",account:account)
        XCTAssertTrue(store.live.isEmpty);XCTAssertNil(store.lastOddsRefresh)
        XCTAssertTrue(store.oddsCachedOnly);XCTAssertFalse(store.oddsLoadSucceeded)
        XCTAssertTrue(store.status.contains("As-of time unavailable"))
    }

    func testRecordedGameLinesSurviveMissingPropsWithoutFreshClaim() async {
        let account=account(),store=store()
        let stamp="2026-10-09T19:52:46Z"
        account.client.functions.operation={ body in
            if body["event"] != nil {throw URLError(.timedOut)}
            return OddsResponse(markets:[self.market()],message:"Recorded snapshot",cacheOnly:true,stale:true,fetchedAt:stamp,quoteUpdatedAt:stamp)
        }
        await store.load("MLB",account:account,home:"Home",away:"Away",start:"2026-09-30T22:00:00Z")
        XCTAssertEqual(store.live.map(\.id),["one"])
        XCTAssertEqual(store.lastOddsRefresh,sportsDate(stamp))
        XCTAssertTrue(store.oddsCachedOnly);XCTAssertTrue(store.oddsStale)
        XCTAssertTrue(store.status.contains("Recorded game odds retained"))
        XCTAssertFalse(store.status.contains("Game odds refreshed"))
    }

    func testRecordedGameLinesSurviveEmptyCachedProps() async {
        let account=account(),store=store()
        let stamp="2026-10-09T19:52:46Z"
        account.client.functions.operation={ body in
            if body["event"] != nil {return OddsResponse(markets:[],message:"No recorded props",cacheOnly:true,stale:true)}
            return OddsResponse(markets:[self.market()],message:"Recorded snapshot",cacheOnly:true,stale:true,fetchedAt:stamp,quoteUpdatedAt:stamp)
        }
        await store.load("MLB",account:account,home:"Home",away:"Away",start:"2026-09-30T22:00:00Z")
        XCTAssertEqual(store.live.map(\.id),["one"])
        XCTAssertEqual(store.lastOddsRefresh,sportsDate(stamp))
        XCTAssertTrue(store.oddsCachedOnly);XCTAssertTrue(store.oddsStale)
        XCTAssertFalse(store.oddsLoadSucceeded)
    }

    func testRecordedQuoteTimestampNeverUsesFetchOrServedTime() throws {
        let data=Data(#"{"markets":[],"message":"Recorded","cacheOnly":true,"stale":true,"fetchedAt":"2026-10-13T12:00:00Z","servedAt":"2026-10-13T13:00:00Z","quoteUpdatedAt":"2026-10-12T12:00:00Z"}"#.utf8)
        let response=try JSONDecoder().decode(OddsResponse.self,from:data)
        XCTAssertEqual(Store.recordedQuoteTime(response),sportsDate("2026-10-12T12:00:00Z"))
        var noQuote=response;noQuote.quoteUpdatedAt=nil
        XCTAssertNil(Store.recordedQuoteTime(noQuote))
        var row=market();row.updated="2026-10-11T12:00:00Z"
        noQuote.markets=[row]
        XCTAssertEqual(Store.recordedQuoteTime(noQuote),sportsDate("2026-10-11T12:00:00Z"))
        XCTAssertTrue(Store.oddsAsOf(Store.recordedQuoteTime(noQuote)!).hasPrefix("Quote as of "))
    }

    func testCachedFetchAndServedDatesCannotInventQuoteTime() async {
        let account=account(),store=store()
        account.client.functions.operation={ _ in OddsResponse(markets:[self.market()],message:"Recorded",cacheOnly:true,stale:true,fetchedAt:"2026-10-13T12:00:00Z",servedAt:"2026-10-13T13:00:00Z") }
        await store.load("MLB",account:account)
        XCTAssertEqual(store.live.count,1)
        XCTAssertNil(store.lastOddsRefresh)
        XCTAssertTrue(store.status.contains("As-of time unavailable"))
    }

    func testCacheOnlyBodyEncodesBooleanAndDoesNotRequestProps() async throws {
        let body=OddsRequestBody(sport:"NFL",event:nil,cacheOnly:true)
        let encoded=try JSONSerialization.jsonObject(with:JSONEncoder().encode(body)) as! [String:Any]
        XCTAssertEqual(encoded["cacheOnly"] as? Bool,true)
        XCTAssertNil(encoded["event"])
        let account=account(),store=store()
        var calls=0
        account.client.functions.operation={ request in
            calls += 1
            XCTAssertEqual(request["cacheOnly"],"true")
            XCTAssertNil(request["event"])
            return OddsResponse(markets:[self.market()],message:"Recorded",cacheOnly:true,stale:true,quoteUpdatedAt:"2026-10-09T19:52:46Z")
        }
        await store.load("MLB",account:account,home:"Home",away:"Away",start:"2026-09-30T22:00:00Z",cacheOnly:true)
        XCTAssertEqual(calls,1)
        XCTAssertEqual(store.live.map(\.id),["one"])
        XCTAssertFalse(store.status.contains("No cached markets"))
        XCTAssertTrue(store.status.contains("Quote as of"))
    }

    func testCacheOnlyAndRefreshFlightsHaveDifferentKeys() {
        let scope=OddsScope(backend:"",owner:UUID())
        XCTAssertNotEqual(OddsRequestKey(scope:scope,sport:"NFL",event:nil),OddsRequestKey(scope:scope,sport:"NFL",event:nil,cacheOnly:true))
    }

    func testRecordedEventRequiresWholeTeamsAndExactStart() {
        let start=sportsDate("2026-09-30T22:00:00Z")!
        XCTAssertTrue(Store.matchesRecordedEvent(market(),sport:"MLB",home:"Home",away:"Away",start:start))
        XCTAssertFalse(Store.matchesRecordedEvent(market(),sport:"MLB",home:"Hom",away:"Away",start:start))
        XCTAssertFalse(Store.matchesRecordedEvent(market(),sport:"NBA",home:"Home",away:"Away",start:start))
        XCTAssertFalse(Store.matchesRecordedEvent(market(),sport:"MLB",home:"Away",away:"Home",start:start))
        XCTAssertFalse(Store.matchesRecordedEvent(market(),sport:"MLB",home:"Home",away:"Away",start:start.addingTimeInterval(600)))
    }

    func testPropsWithWrongScheduledStartCannotJoinMatchingGame() async {
        let account=account(),store=store()
        account.client.functions.operation={ body in
            if body["event"] != nil {
                var wrong=self.market("bad-prop",kind:"prop");wrong.commence="2026-09-30T22:10:00Z"
                return OddsResponse(markets:[wrong],message:"Props")
            }
            return OddsResponse(markets:[self.market()],message:"Game")
        }
        await store.load("MLB",account:account,home:"Home",away:"Away",start:"2026-09-30T22:00:00Z")
        XCTAssertEqual(store.live.map(\.id),["one"])
    }

    func testLocalServerFailureExplainsHowToRecover() {
        XCTAssertEqual(Store.failureMessage(URLError(.cannotConnectToHost),backend:"http://127.0.0.1:8766"),"Local odds service is offline. Start the local odds server, then pull to retry.")
        XCTAssertEqual(Store.failureMessage(URLError(.notConnectedToInternet),backend:"https://service.example"),"You are offline. Reconnect, then pull to refresh odds.")
        XCTAssertEqual(Store.failureMessage(URLError(.cannotConnectToHost),backend:"https://service.example"),"Odds service unavailable. Pull to retry.")
    }

    func testQuotaAndProviderCredentialErrorsRemainDistinct() {
        let invalid=OddsServiceError(status:502,body:Data(#"{"error":"Odds provider returned HTTP 401. Check plan, quota, and key configuration."}"#.utf8))
        XCTAssertEqual(Store.failureMessage(invalid,backend:"http://127.0.0.1:8766"),"Live odds are temporarily unavailable. Try again later.")
        let limit=FunctionsError.httpError(429,Data(#"{"error":"Today's beta odds refresh allowance has been reached"}"#.utf8))
        XCTAssertEqual(Store.failureMessage(limit,backend:""),"Today's odds refresh allowance has been reached.")
        let localLimit=OddsServiceError(status:429,body:Data(#"{"error":"Local odds refresh allowance has been reached. Existing snapshots may be stale."}"#.utf8))
        XCTAssertEqual(Store.failureMessage(localLimit,backend:"http://127.0.0.1:8766"),"Odds refresh allowance reached. Existing snapshots may be stale.")
        XCTAssertEqual(Store.failureMessage(FunctionsError.httpError(429,Data()),backend:""),"Odds refresh limit reached. Wait a minute, then pull to retry.")
    }

    func testUnrecognizedServerTextNeverAppearsInProduct() {
        for data in [Data(#"{"error":"private API key: secret; <script>alert(1)</script>"}"#.utf8),Data("<html>private failure</html>".utf8)] {
            XCTAssertEqual(Store.failureMessage(OddsServiceError(status:503,body:data),backend:""),"Odds service unavailable. Pull to retry.")
        }
    }

    func testCacheForceExpiryAndScopeSeparation() async throws {
        let cache=SharedOddsCache<Int>()
        let owner=UUID()
        let scope=OddsScope(backend:"https://one.example",owner:owner)
        let key=OddsRequestKey(scope:scope,sport:"MLB",event:nil)
        var calls=0
        let operation: @MainActor () async throws -> Int = { calls += 1;return calls }
        let first=try await cache.fetch(key,force:false,operation:operation)
        _=try await cache.fetch(key,force:false,operation:operation)
        XCTAssertEqual(calls,1)
        _=try await cache.fetch(key,force:true,operation:operation)
        XCTAssertEqual(calls,2)
        _=try await cache.fetch(key,force:false,now:first.receivedAt.addingTimeInterval(60),operation:operation)
        XCTAssertEqual(calls,3)
        let others=[OddsRequestKey(scope:scope,sport:"NFL",event:nil),OddsRequestKey(scope:scope,sport:"MLB",event:"event"),OddsRequestKey(scope:OddsScope(backend:scope.backend,owner:UUID()),sport:"MLB",event:nil),OddsRequestKey(scope:OddsScope(backend:"https://two.example",owner:owner),sport:"MLB",event:nil)]
        for other in others { XCTAssertNil(cache.cached(other));_=try await cache.fetch(other,force:false,operation:operation) }
        XCTAssertEqual(calls,7)
        cache.invalidate(scope)
        XCTAssertNil(cache.cached(key))
        XCTAssertNotNil(cache.cached(others.last!))
    }
}
#endif
