import XCTest
@testable import PLONNCore

final class LiveMatchupContextTests: XCTestCase {
    private let start = Date(timeIntervalSince1970: 1_792_368_000)
    private func event(_ sport: String = "NFL") -> SharedMatchupEvent {
        .init(sport: sport, eventID: "espn:401873007", homeTeam: "Green Bay Packers", awayTeam: "Dallas Cowboys", startISO: ISO8601DateFormatter().string(from: start))
    }
    private func summary(state: String = "in", completed: Bool = false, stats: [[String:Any]] = []) -> [String:Any] {
        ["header": ["id":"401873007", "competitions": [["id":"401873007",
          "status":["type":["state":state,"completed":completed,"name":completed ? "STATUS_FINAL" : "STATUS_IN_PROGRESS", "detail": completed ? "Final" : "10:37 - 2nd Quarter"]],
          "competitors":[["homeAway":"home","score":"7","team":["id":"9","displayName":"Green Bay Packers","abbreviation":"GB"]],
                         ["homeAway":"away","score":"0","team":["id":"6","displayName":"Dallas Cowboys","abbreviation":"DAL"]]]]]],
         "boxscore":["teams":[["team":["id":"6"],"statistics":stats],["team":["id":"9"],"statistics":stats]]]]
    }
    private func parse(_ root: [String:Any], sport: String = "NFL") throws -> LiveMatchupSnapshot {
        try LiveMatchupClient.parse(JSONSerialization.data(withJSONObject: root), event: event(sport), checkedAt: start)
    }
    func testLiveScoreAndFinalResultRemainAvailableWithoutPregameReportOrOdds() throws {
        let live = try parse(summary(stats: [["name":"totalYards","displayValue":"194"]]))
        XCTAssertEqual(live.phase, .live)
        XCTAssertEqual(live.scoreSummary, "Green Bay Packers leads 7–0.")
        XCTAssertEqual(live.stats.first?.label, "Total yards")
        let final = try parse(summary(state:"post",completed:true))
        XCTAssertEqual(final.phase, .final)
        XCTAssertEqual(final.scoreSummary, "Green Bay Packers won 7–0.")
        XCTAssertFalse(final.isStale(at:start.addingTimeInterval(86400)))
        XCTAssertNil(LiveMatchupPolicy.nextRefresh(event:event(),snapshot:final,now:start,failed:false))
    }
    func testEveryTeamSportUsesItsOwnMetricsAndMissingValuesStayMissing() throws {
        let cases = [("NFL","thirdDownEff"),("NCAAF","thirdDownEff"),("NBA","totalRebounds"),("WNBA","totalRebounds"),("NCAAM","totalRebounds"),("NCAAW","totalRebounds"),("NHL","shotsTotal"),("Soccer","shotsOnTarget"),("Soccer:eng.1","shotsOnTarget")]
        for (sport,key) in cases {
            let value = try parse(summary(stats:[["name":key,"displayValue":"0"],["name":"totalYards","displayValue":"--"]]),sport:sport)
            XCTAssertEqual(value.stats.count,1,sport)
            XCTAssertEqual(value.stats.first?.home,"0",sport)
            XCTAssertNotNil(LiveMatchupClient.feedPath(sport))
        }
        XCTAssertNil(LiveMatchupClient.feedPath("unknown"))
        XCTAssertNil(LiveMatchupClient.feedPath("Soccer:../../nfl"))
    }
    func testBaseballKeepsBattingSeparateFromPitching() throws {
        let value = try parse(summary(stats:[
            ["name":"batting","stats":[["name":"hits","displayValue":"12"],["name":"walks","displayValue":"3"]]],
            ["name":"pitching","stats":[["name":"hits","displayValue":"4"],["name":"walks","displayValue":"1"]]],
            ["name":"fielding","stats":[["name":"errors","displayValue":"0"]]]]),sport:"MLB")
        XCTAssertEqual(value.stats.first(where:{$0.id == "batting.hits"})?.home,"12")
        XCTAssertEqual(value.stats.first(where:{$0.id == "batting.walks"})?.home,"3")
        XCTAssertEqual(value.stats.first(where:{$0.id == "fielding.errors"})?.home,"0")
    }
    func testWrongEventOrTeamsAreRejected() throws {
        var root=summary()
        var header=root["header"] as! [String:Any]; header["id"]="other"; root["header"]=header
        XCTAssertThrowsError(try parse(root))
        var other=event(); other = .init(sport:other.sport,eventID:other.eventID,homeTeam:"Another team",awayTeam:other.awayTeam,startISO:other.startISO)
        XCTAssertThrowsError(try LiveMatchupClient.parse(JSONSerialization.data(withJSONObject:summary()),event:other,checkedAt:start))
        let previous = try parse(summary(state:"post",completed:true))
        XCTAssertFalse(LiveMatchupPolicy.showsGameContext(event:other,state:"pre",snapshot:previous,now:start.addingTimeInterval(-10)))
        XCTAssertNotNil(LiveMatchupPolicy.nextRefresh(event:other,snapshot:previous,now:start,failed:false))
    }
    func testRecordedProviderSummariesAcrossSportsIncludingCowboysPackers() throws {
        let path=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("contracts/live-matchup-summaries-v1.json")
        let rows=try JSONSerialization.jsonObject(with:Data(contentsOf:path)) as! [[String:Any]]
        XCTAssertEqual(rows.count,5)
        for row in rows {
            let event=try JSONDecoder().decode(SharedMatchupEvent.self,from:JSONSerialization.data(withJSONObject:row["event"]!))
            let value=try LiveMatchupClient.parse(JSONSerialization.data(withJSONObject:row["summary"]!),event:event,checkedAt:try XCTUnwrap(sportsDate(row["checkedAt"] as! String)))
            XCTAssertEqual(value.phase,.final,event.sport)
            XCTAssertGreaterThanOrEqual(value.stats.count,5,event.sport)
            XCTAssertTrue(value.stats.allSatisfy{$0.home != nil && $0.away != nil},event.sport)
            if event.sport == "NFL" {
                XCTAssertEqual(value.scoreSummary,"Green Bay Packers won 23–13.")
                XCTAssertEqual(value.stats.first(where:{$0.id == "totalYards"})?.away,"303")
                XCTAssertEqual(value.stats.first(where:{$0.id == "totalYards"})?.home,"318")
            }
        }
    }
    func testMissingBoxscoreDoesNotHideScoreAndMissingScoreIsNotZero() throws {
        var root=summary();root.removeValue(forKey:"boxscore")
        XCTAssertEqual(try parse(root).scoreSummary,"Green Bay Packers leads 7–0.")
        XCTAssertEqual(try parse(root).stats,[])
        var header=root["header"] as! [String:Any], comps=header["competitions"] as! [[String:Any]]
        var teams=comps[0]["competitors"] as! [[String:Any]]; teams[0].removeValue(forKey:"score")
        comps[0]["competitors"]=teams;header["competitions"]=comps;root["header"]=header
        XCTAssertNil(try parse(root).scoreSummary)
    }
    func testKickoffTransitionsWithoutNavigationAndDoesNotInventLiveState() throws {
        let pre = try parse(summary(state:"pre",stats:[["name":"totalYards","displayValue":"194"]]))
        XCTAssertEqual(pre.phase,.scheduled)
        XCTAssertTrue(pre.stats.isEmpty)
        XCTAssertNil(pre.scoreSummary)
        XCTAssertFalse(LiveMatchupPolicy.showsGameContext(event:event(),state:"pre",snapshot:pre,now:start.addingTimeInterval(-1)))
        XCTAssertTrue(LiveMatchupPolicy.showsGameContext(event:event(),state:"pre",snapshot:pre,now:start))
        XCTAssertEqual(LiveMatchupPolicy.nextRefresh(event:event(),snapshot:pre,now:start.addingTimeInterval(-2),failed:false),2)
        XCTAssertTrue(LiveMatchupPolicy.showsGameContext(event:event(),state:"in",snapshot:nil,now:start.addingTimeInterval(-10)))
    }
    func testOutageFreshnessBoundaryAndRetry() throws {
        let value=try parse(summary())
        XCTAssertFalse(value.isStale(at:start.addingTimeInterval(89)))
        XCTAssertTrue(value.isStale(at:start.addingTimeInterval(90)))
        XCTAssertTrue(value.isStale(at:start.addingTimeInterval(-1)))
        XCTAssertEqual(LiveMatchupPolicy.nextRefresh(event:event(),snapshot:value,now:start,failed:true),30)
        XCTAssertEqual(LiveMatchupPolicy.nextRefresh(event:event(),snapshot:nil,now:start,failed:true),30)
    }
    func testDelayedAndCanceledGamesAreNotPresentedAsLiveOrFinal() throws {
        for name in ["STATUS_DELAYED","STATUS_POSTPONED","STATUS_CANCELED","STATUS_SUSPENDED"] {
            var root=summary(state:"post",completed:true)
            var header=root["header"] as! [String:Any], comps=header["competitions"] as! [[String:Any]]
            comps[0]["status"]=["type":["state":"post","completed":true,"name":name,"detail":"Delayed"]]
            header["competitions"]=comps;root["header"]=header
            let value=try parse(root)
            XCTAssertEqual(value.phase,.interrupted)
            XCTAssertNil(value.scoreSummary)
            XCTAssertNotNil(LiveMatchupPolicy.nextRefresh(event:event(),snapshot:value,now:start,failed:false))
        }
    }
    func testTransportUsesExactEventAndDoesNotAcceptHTTPFailures() async throws {
        let data=try JSONSerialization.data(withJSONObject:summary())
        let client=LiveMatchupClient(transport:{ request in
            XCTAssertEqual(request.url?.absoluteString,"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=401873007")
            XCTAssertEqual(request.cachePolicy,.reloadIgnoringLocalCacheData)
            XCTAssertNil(request.value(forHTTPHeaderField:"Authorization"))
            return (data,HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:nil,headerFields:nil)!)
        },now:{self.start})
        let fetched=try await client.fetch(event:event())
        XCTAssertEqual(fetched.phase,.live)
        let failure=LiveMatchupClient(transport:{ request in
            (data,HTTPURLResponse(url:request.url!,statusCode:503,httpVersion:nil,headerFields:nil)!)
        })
        do { _ = try await failure.fetch(event:event());XCTFail("Must reject an HTTP error") } catch {}
    }
}
