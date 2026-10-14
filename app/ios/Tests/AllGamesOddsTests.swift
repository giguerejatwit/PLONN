#if ODDS_STANDALONE_TESTS
import XCTest
@testable import OddsRefreshCore

@MainActor final class AllGamesOddsTests:XCTestCase {
    func row(_ sport:String)->Market {
        Market(id:sport,event:sport,sport:sport,kind:"team",name:"Home",opponent:"Away",label:"Home moneyline",aliases:"",price:-110,market:"h2h",book:"Book",quotes:[],mode:"live",eventLabel:"Away at Home",commence:"2099-10-15T00:00:00Z")
    }
    func feed(_ sport:String)->([SportsItem],String) {
        ([SportsItem(id:sport,start:"2099-10-15T00:00:00Z",seasonType:sport == "NBA" ? 1:2)],"ESPN")
    }
    func setup()->AccountStore {
        UserDefaults.standard.removeObject(forKey:"backendURL")
        let account=AccountStore(owner:UUID())
        account.client.functions.operation={ request in
            XCTAssertEqual(request["cacheOnly"],"true")
            XCTAssertNil(request["event"])
            XCTAssertNil(request["frozenRun"])
            XCTAssertEqual(request["phase"],request["sport"] == "NBA" ? "preseason":nil)
            return OddsResponse(markets:[self.row(request["sport"]!)],message:"Recorded",cacheOnly:true,stale:true,quoteUpdatedAt:"2026-10-13T23:10:00Z")
        }
        return account
    }
    func assertAllFour(_ slate:AllGamesStore,file:StaticString=#filePath,line:UInt=#line) {
        XCTAssertEqual(Set(slate.markets.keys),Set(LaunchSports.supported),file:file,line:line)
        for sport in LaunchSports.supported {XCTAssertEqual(slate.markets[sport]?.map(\.sport),[sport],file:file,line:line)}
        XCTAssertEqual(slate.cachedLeagues,Set(LaunchSports.supported),file:file,line:line)
        XCTAssertEqual(slate.quoteTimes.count,4,file:file,line:line)
    }
    func testSupersededScheduleCannotEraseNewerAllSportPrices() async {
        let account=setup(),slate=AllGamesStore()
        let oldDate=Date(timeIntervalSince1970:1_800_000_000),newDate=Date(timeIntervalSince1970:1_800_086_400)
        var gates:[String:CheckedContinuation<([SportsItem],String),Never>]=[:]
        SportsDirectory.operation={sport,date in
            if date==oldDate {return await withCheckedContinuation {gates[sport]=$0}}
            return self.feed(sport)
        }
        let old=Task {await slate.load(oldDate,account:account)}
        while gates.count<4 {await Task.yield()}
        await slate.load(newDate,account:account)
        assertAllFour(slate)
        for (sport,gate) in gates {gate.resume(returning:feed(sport))}
        await old.value
        assertAllFour(slate)
        XCTAssertFalse(slate.loading)
    }
    func testCancelledScheduleRetainsAllSportRecordedPrices() async {
        let account=setup(),slate=AllGamesStore(),date=Date(timeIntervalSince1970:1_800_000_000)
        SportsDirectory.operation={sport,_ in self.feed(sport)}
        await slate.load(date,account:account)
        assertAllFour(slate)
        var gates:[String:CheckedContinuation<([SportsItem],String),Never>]=[:]
        SportsDirectory.operation={sport,_ in await withCheckedContinuation {gates[sport]=$0}}
        let cancelled=Task {await slate.load(date,account:account)}
        while gates.count<4 {await Task.yield()}
        cancelled.cancel()
        for (sport,gate) in gates {gate.resume(returning:feed(sport))}
        await cancelled.value
        assertAllFour(slate)
        XCTAssertFalse(slate.loading)
    }
    func testScopeChangeDuringScheduleCannotStartOddsRequests() async {
        let account=setup(),slate=AllGamesStore(),date=Date(timeIntervalSince1970:1_800_000_000)
        var gates:[String:CheckedContinuation<([SportsItem],String),Never>]=[:],calls=0
        SportsDirectory.operation={sport,_ in await withCheckedContinuation {gates[sport]=$0}}
        account.client.functions.operation={_ in calls += 1;return OddsResponse(markets:[],message:"")}
        let pending=Task {await slate.load(date,account:account)}
        while gates.count<4 {await Task.yield()}
        account.user=TestUser(id:UUID())
        for (sport,gate) in gates {gate.resume(returning:feed(sport))}
        await pending.value
        XCTAssertEqual(calls,0)
        XCTAssertTrue(slate.markets.isEmpty)
    }
}
#endif
