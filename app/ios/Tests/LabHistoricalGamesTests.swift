import XCTest
@testable import PLONNCore

final class LabHistoricalGamesTests: XCTestCase {
    private let date = ISO8601DateFormatter().date(from: "2024-12-01T18:00:00Z")!
    private func event(id: String = "401671627", completed: Bool = true, season: Int = 2, start: String = "2024-12-01T18:00Z") -> [String:Any] {
        ["id":id,"date":start,"season":["type":season],"competitions":[[
            "status":["type":["completed":completed,"state":"post","name":"STATUS_FINAL"]],"competitors":[
                ["homeAway":"home","team":["displayName":"Cincinnati Bengals"],"score":"38"],
                ["homeAway":"away","team":["displayName":"Pittsburgh Steelers"],"score":"44"]]]]]
    }
    private func data(_ value: Any) throws -> Data {try JSONSerialization.data(withJSONObject:value)}
    func testProviderSelectionPreservesBackendContractsAcrossSports() {
        for league in ["NFL","NBA","NHL","MLB"] {
            XCTAssertEqual(LabHistoricalGames.Source.source(league:league,fullGame:true),.espn)
        }
        XCTAssertEqual(LabHistoricalGames.Source.source(league:"NFL",fullGame:false),.nflverse)
        XCTAssertEqual(LabHistoricalGames.Source.source(league:"MLB",fullGame:false),.mlb)
        for league in ["NBA","NHL"] {XCTAssertEqual(LabHistoricalGames.Source.source(league:league,fullGame:false),.espn)}
    }
    func testOnlyCompletedRegularSeasonGamesOnChosenDateAppear() throws {
        let games=try LabHistoricalGames.parseESPN(data(["events":[event(),event(completed:false),event(season:3),event(season:1),event(start:"2024-12-02T18:00Z")]]),date:date)
        XCTAssertEqual(games.count,1)
        XCTAssertEqual(games[0].title,"Pittsburgh Steelers at Cincinnati Bengals")
        XCTAssertEqual(games[0].id,"401671627")
        XCTAssertEqual(games[0].opponent(for:games[0].away)?.name,"Cincinnati Bengals")
        XCTAssertEqual(games[0].opponent(for:games[0].home)?.name,"Pittsburgh Steelers")
        XCTAssertNil(games[0].opponent(for:.init(name:"Unrelated",requestName:"XXX")))
    }
    func testEveningGamesUseScoreboardDayInsteadOfUTCDay() throws {
        let games=try LabHistoricalGames.parseESPN(data(["events":[event(start:"2024-12-02T01:20Z")]]),date:date)
        XCTAssertEqual(games.count,1)
    }
    func testHistoricalReadinessUsesTheSameUTCBoundaryAsTraining() {
        let formatter=ISO8601DateFormatter()
        let now=formatter.date(from:"2026-10-17T22:00:00Z")!
        XCTAssertTrue(LabHistoricalGames.isEligibleForHistoricalTest(formatter.date(from:"2026-10-15T23:59:59Z")!,now:now))
        XCTAssertFalse(LabHistoricalGames.isEligibleForHistoricalTest(formatter.date(from:"2026-10-16T00:00:00Z")!,now:now))
        XCTAssertFalse(LabHistoricalGames.isEligibleForHistoricalTest(formatter.date(from:"2026-10-16T00:15:00Z")!,now:now))
        XCTAssertEqual(LabServiceContract.userFacingMessage(["training":["reason":"game_not_completed_historical_target"]]),"This game is too recent for historical testing. Choose an earlier completed game.")
    }
    func testNFLPlayerSelectionUsesExactScheduleMappingAndOpponentCodes() throws {
        let games=try LabHistoricalGames.parseESPN(data(["events":[event()]]),date:date)
        let csv="espn,game_id,game_type,gameday,home_team,away_team,home_score,away_score,stadium\r\n401671627,2024_13_PIT_CIN,REG,2024-12-01,CIN,PIT,38,44,\"Stadium, Cincinnati\"\r\n"
        let result=try LabHistoricalGames.nflGames(games,csv:Data(csv.utf8),date:date)
        XCTAssertEqual(result.first?.id,"2024_13_PIT_CIN")
        XCTAssertEqual(result.first?.home.name,"Cincinnati Bengals")
        XCTAssertEqual(result.first?.home.requestName,"CIN")
        XCTAssertEqual(result.first?.away.requestName,"PIT")
        for change in [csv.replacingOccurrences(of:"401671627",with:"999"),csv.replacingOccurrences(of:",38,",with:",37,"),csv.replacingOccurrences(of:",REG,",with:",POST,"),csv.replacingOccurrences(of:"2024-12-01",with:"2024-12-02")] {
            XCTAssertTrue(try LabHistoricalGames.nflGames(games,csv:Data(change.utf8),date:date).isEmpty)
        }
    }
    func testMLBUsesNativeIDsAndDistinguishesDoubleheaders() throws {
        func game(_ id:Int,_ number:Int,_ type:String="R",_ status:String="Final") -> [String:Any] {
            ["gamePk":id,"gameNumber":number,"doubleHeader":"Y","gameType":type,"officialDate":"2024-12-01","gameDate":"2024-12-01T18:00:00Z","status":["abstractGameState":status],"teams":["home":["team":["name":"Home"],"score":2],"away":["team":["name":"Away"],"score":3]]]
        }
        let result=try LabHistoricalGames.parseMLB(data(["dates":[["games":[game(700001,1),game(700002,2),game(700003,1,"S"),game(700004,1,"R","Live")]]]]),date:date)
        XCTAssertEqual(result.map(\.id),["700001","700002"])
        XCTAssertEqual(result.map(\.detail),["Final · Game 1","Final · Game 2"])
    }
    func testMalformedSchedulesFailInsteadOfPresentingEmptySuccess() {
        XCTAssertThrowsError(try LabHistoricalGames.parseESPN(Data("{}".utf8),date:date))
        XCTAssertThrowsError(try LabHistoricalGames.parseMLB(Data("{}".utf8),date:date))
        XCTAssertThrowsError(try LabHistoricalGames.nflGames([],csv:Data("wrong,header\n".utf8),date:date))
        XCTAssertThrowsError(try LabHistoricalGames.nflGames([],csv:Data("espn,espn\n".utf8),date:date))
    }
}
