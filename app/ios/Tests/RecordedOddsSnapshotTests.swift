#if ODDS_STANDALONE_TESTS
import XCTest
@testable import OddsRefreshCore

@MainActor final class RecordedOddsSnapshotTests:XCTestCase {
    let reviewTime=sportsDate("2026-10-14T12:19:23Z")!
    func snapshot(_ sport:String)throws->[Market] {
        let folder=try XCTUnwrap(ProcessInfo.processInfo.environment["PLONN_ODDS_FIXTURES"])
        return try JSONDecoder().decode(OddsResponse.self,from:Data(contentsOf:URL(fileURLWithPath:folder).appendingPathComponent(sport.lowercased()+"-snapshot-fixture.json"))).markets
    }
    func prices(_ sport:String,away:String,home:String,start:String,now:Date?=nil)throws->[Market] {
        let rows=try snapshot(sport),time=try XCTUnwrap(sportsDate(start))
        return [away,home].compactMap {Store.recordedMoneyline($0,markets:rows,sport:sport,home:home,away:away,start:time,now:now ?? reviewTime)}
    }
    func testRecordedMLBPricesStillMatchEvenWhenCacheIsOlderThanTwoMinutes() throws {
        let rows=try prices("MLB",away:"Milwaukee Brewers",home:"Los Angeles Dodgers",start:"2026-10-15T00:00Z")
        XCTAssertEqual(rows.map(\.price),[154,-170])
        XCTAssertEqual(Set(rows.map(\.event)),["622fd31d2c8178b025455656e0d2fb89"])
        XCTAssertTrue(rows.allSatisfy{$0.updated == "2026-10-13T23:10:36Z"})
        XCTAssertTrue(try prices("MLB",away:"Milwaukee Brewers",home:"Los Angeles Dodgers",start:"2026-10-16T00:00Z").isEmpty)
    }
    func testBothRecordedNHLGamesMatchESPNIncludingMontrealAccent() throws {
        let montreal=try prices("NHL",away:"Montreal Canadiens",home:"Washington Capitals",start:"2026-10-14T23:30Z")
        XCTAssertEqual(montreal.map(\.price),[122,-129])
        XCTAssertEqual(montreal.first?.name,"Montréal Canadiens") // Original provider record is untouched.
        let dallas=try prices("NHL",away:"Dallas Stars",home:"Colorado Avalanche",start:"2026-10-15T00:30Z")
        XCTAssertEqual(dallas.map(\.price),[126,-135])
        XCTAssertTrue(try prices("NHL",away:"Montreal Canadiens",home:"Washington Capitals",start:"2026-10-14T23:30Z",now:sportsDate("2026-10-14T23:30Z")).isEmpty)
    }
    func testAllThirteenFutureNFLIdentitiesMatchWithoutDateWidening() throws {
        let rows=try snapshot("NFL"),events=Dictionary(grouping:rows,by:\.event)
        XCTAssertEqual(events.count,13)
        for values in events.values {
            let pair=try XCTUnwrap(values.first?.eventLabel).components(separatedBy:" at ")
            XCTAssertEqual(pair.count,2)
            let start=try XCTUnwrap(values.first?.commence.flatMap(sportsDate))
            for team in pair {
                XCTAssertNotNil(Store.recordedMoneyline(team,markets:rows,sport:"NFL",home:pair[1],away:pair[0],start:start,now:reviewTime))
                XCTAssertNil(Store.recordedMoneyline(team,markets:rows,sport:"NFL",home:pair[1],away:pair[0],start:start.addingTimeInterval(600),now:reviewTime))
            }
        }
    }
    func testNBAPreseasonHasYesterdayPricesButNoneOfTodaysSevenEvents() throws {
        let yesterday=try prices("NBA",away:"Indiana Pacers",home:"Oklahoma City Thunder",start:"2026-10-14T00:00Z",now:sportsDate("2026-10-13T23:13Z"))
        XCTAssertEqual(yesterday.map(\.price),[205,-225])
        let current:[(String,String,String)]=[
            ("Brooklyn Nets","Miami Heat","2026-10-14T23:30Z"),
            ("Charlotte Hornets","Boston Celtics","2026-10-14T23:30Z"),
            ("Phoenix Suns","San Antonio Spurs","2026-10-15T00:00Z"),
            ("New Orleans Pelicans","Dallas Mavericks","2026-10-15T00:00Z"),
            ("Chicago Bulls","Milwaukee Bucks","2026-10-15T00:00Z"),
            ("Detroit Pistons","Minnesota Timberwolves","2026-10-15T00:00Z"),
            ("Denver Nuggets","LA Clippers","2026-10-15T02:30Z")]
        for (away,home,start) in current {XCTAssertTrue(try prices("NBA",away:away,home:home,start:start).isEmpty)}
    }
    func testNormalizationPreservesWholeTeamIdentityAndNHLUniqueEventGuard() throws {
        XCTAssertTrue(RecordedTeamIdentity.matches("  MONTRÉAL  Canadiens ","Montreal Canadiens"))
        for pair in [("Los Angeles Lakers","Los Angeles Clippers"),("New York Jets","New York Giants"),("New York Mets","New York Yankees"),("Montreal","Montreal Canadiens"),("","")] {
            XCTAssertFalse(RecordedTeamIdentity.matches(pair.0,pair.1))
        }
        let rows=try snapshot("NHL"),start=sportsDate("2026-10-14T23:30Z")!
        var duplicate=try XCTUnwrap(rows.first{$0.name == "Montréal Canadiens" && $0.event == "11b474310b2b60cbae546d9c517444c6"})
        duplicate.event="different-provider-event"
        XCTAssertNil(Store.recordedMoneyline("Montreal Canadiens",markets:rows+[duplicate],sport:"NHL",home:"Washington Capitals",away:"Montreal Canadiens",start:start,now:reviewTime))
        XCTAssertNil(Store.recordedMoneyline("Washington Capitals",markets:rows,sport:"NHL",home:"Montreal Canadiens",away:"Washington Capitals",start:start,now:reviewTime))
    }
}
#endif
