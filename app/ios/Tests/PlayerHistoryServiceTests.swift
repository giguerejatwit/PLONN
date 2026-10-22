import XCTest
@testable import PLONNCore

private actor HistoryReadCounter {
    private var counts: [String: Int] = [:]
    func read(_ url: URL) async throws -> Data {
        counts[url.absoluteString, default: 0] += 1
        try await Task.sleep(nanoseconds: 20_000_000)
        return Data("{}".utf8)
    }
    func count(_ url: URL) -> Int { counts[url.absoluteString, default: 0] }
}
private final class HistoryTestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var time = Date(timeIntervalSince1970: 1_000_000)
    func now() -> Date { lock.lock(); defer { lock.unlock() }; return time }
    func advance(_ seconds: Double) { lock.lock(); defer { lock.unlock() }; time = time.addingTimeInterval(seconds) }
}

final class PlayerHistoryServiceTests: XCTestCase {
    private let cutoff = sportsDate("2026-10-06T00:00:00Z")!
    private func event(_ id: String, sport: String = "NFL", day: String = "2026-09-01T20:00:00Z", opponent: String = "2") -> [String: Any] {
        ["id": id, "gameDate": day, "leagueAbbreviation": sport, "homeTeamId": "1", "awayTeamId": opponent,
         "homeTeamScore": "20", "awayTeamScore": "10", "gameResult": "W", "team": ["id": "1", "isAllStar": false],
         "opponent": ["id": opponent, "displayName": "Team \(opponent)"]]
    }
    private func espn(_ values: [String], names: [String] = ["passingYards", "passingAttempts"], sport: String = "NFL", season: Int = 2026) -> [String: Any] {
        let prefix = sport == "NFL" ? "\(season)" : "\(season-1)-\(String(season).suffix(2))"
        return ["filters": [["name": "season", "value": String(season)]], "names": names,
            "events": ["101": event("101", sport: sport)], "seasonTypes": [["displayName": "\(prefix) Regular Season",
             "categories": [["displayName": "September", "events": [["eventId": "101", "stats": values]]]]]]]
    }
    private func parse(_ payload: [String: Any], sport: String = "NFL", market: String = "player_pass_yds") throws -> [PlayerGame] {
        try PlayerHistoryService.espnGames(payload: payload, sport: sport, playerID: "99", marketKey: market, season: 2026, cutoff: cutoff)
    }

    func testCapturedPublicProviderRowsForAllFourSports() throws {
        do {
            let data = #"{"filters":[{"displayName":"League","name":"league","value":"nba","options":[{"value":"nba","displayValue":"National Basketball Association","shortDisplayName":"NBA"}]},{"displayName":"Season","name":"season","value":"2026","options":[{"value":"2027","displayValue":"2026-27"},{"value":"2026","displayValue":"2025-26"},{"value":"2025","displayValue":"2024-25"},{"value":"2024","displayValue":"2023-24"},{"value":"2023","displayValue":"2022-23"},{"value":"2022","displayValue":"2021-22"},{"value":"2021","displayValue":"2020-21"},{"value":"2020","displayValue":"2019-20"},{"value":"2019","displayValue":"2018-19"},{"value":"2018","displayValue":"2017-18"},{"value":"2017","displayValue":"2016-17"},{"value":"2016","displayValue":"2015-16"},{"value":"2015","displayValue":"2014-15"},{"value":"2014","displayValue":"2013-14"},{"value":"2013","displayValue":"2012-13"},{"value":"2012","displayValue":"2011-12"},{"value":"2011","displayValue":"2010-11"},{"value":"2010","displayValue":"2009-10"},{"value":"2009","displayValue":"2008-09"},{"value":"2008","displayValue":"2007-08"},{"value":"2007","displayValue":"2006-07"},{"value":"2006","displayValue":"2005-06"},{"value":"2005","displayValue":"2004-05"},{"value":"2004","displayValue":"2003-04"}]}],"names":["minutes","fieldGoalsMade-fieldGoalsAttempted","fieldGoalPct","threePointFieldGoalsMade-threePointFieldGoalsAttempted","threePointPct","freeThrowsMade-freeThrowsAttempted","freeThrowPct","totalRebounds","assists","blocks","steals","fouls","turnovers","points"],"events":{"401871329":{"id":"401871329","atVs":"vs","gameDate":"2026-05-12T02:30:00.000+00:00","score":"115-110","homeTeamId":"13","awayTeamId":"25","homeTeamScore":"110","awayTeamScore":"115","gameResult":"L","opponent":{"id":"25","displayName":"Oklahoma City Thunder"},"leagueName":"National Basketball Association","leagueAbbreviation":"NBA","leagueShortName":"NBA","eventNote":"West Semifinals - Game 4","team":{"id":"13","isAllStar":false}},"401871328":{"id":"401871328","atVs":"vs","gameDate":"2026-05-10T00:30:00.000+00:00","score":"131-108","homeTeamId":"13","awayTeamId":"25","homeTeamScore":"108","awayTeamScore":"131","gameResult":"L","opponent":{"id":"25","displayName":"Oklahoma City Thunder"},"leagueName":"National Basketball Association","leagueAbbreviation":"NBA","leagueShortName":"NBA","eventNote":"West Semifinals - Game 3","team":{"id":"13","isAllStar":false}}},"seasonTypes":[{"displayName":"2025-26 Postseason","categories":[{"displayName":"Conference Semifinals","events":[{"eventId":"401871329","stats":["40","8-18","44.4","2-6","33.3","6-8","75.0","12","3","1","0","1","4","24"]},{"eventId":"401871328","stats":["37","7-19","36.8","2-6","33.3","3-4","75.0","6","8","0","1","1","3","19"]}]}]}]}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let games = try PlayerHistoryService.espnGames(payload: payload, sport: "NBA", playerID: "1966", marketKey: "player_threes", season: 2026, cutoff: cutoff)
            XCTAssertEqual(games.count, 2)
            XCTAssertEqual(games.first?.sport, "NBA")
            XCTAssertEqual(games.first?.playerID, "1966")
            XCTAssertNotNil(games.first?.opponentID)
        }
        do {
            let data = #"{"filters":[{"displayName":"League","name":"league","value":"nfl","options":[{"value":"college-football","displayValue":"NCAA - Football","shortDisplayName":"NCAAF"},{"value":"nfl","displayValue":"National Football League","shortDisplayName":"NFL"}]},{"displayName":"Season","name":"season","value":"2026","options":[{"value":"2026","displayValue":"2026"},{"value":"2025","displayValue":"2025"},{"value":"2024","displayValue":"2024"},{"value":"2023","displayValue":"2023"},{"value":"2022","displayValue":"2022"},{"value":"2021","displayValue":"2021"},{"value":"2020","displayValue":"2020"},{"value":"2019","displayValue":"2019"},{"value":"2018","displayValue":"2018"},{"value":"2017","displayValue":"2017"}]}],"names":["completions","passingAttempts","passingYards","completionPct","yardsPerPassAttempt","passingTouchdowns","interceptions","longPassing","sacks","QBRating","adjQBR","rushingAttempts","rushingYards","yardsPerRushAttempt","rushingTouchdowns","longRushing"],"events":{"401872976":{"id":"401872976","week":4,"atVs":"@","gameDate":"2026-10-04T20:25:00.000+00:00","score":"30-27","homeTeamId":"13","awayTeamId":"12","homeTeamScore":"27","awayTeamScore":"30","gameResult":"W","opponent":{"id":"13","displayName":"Las Vegas Raiders"},"leagueName":"National Football League","leagueAbbreviation":"NFL","leagueShortName":"NFL","team":{"id":"12","isAllStar":false}},"401872952":{"id":"401872952","week":3,"atVs":"@","gameDate":"2026-09-27T17:00:00.000+00:00","score":"24-10","homeTeamId":"15","awayTeamId":"12","homeTeamScore":"10","awayTeamScore":"24","gameResult":"W","opponent":{"id":"15","displayName":"Miami Dolphins"},"leagueName":"National Football League","leagueAbbreviation":"NFL","leagueShortName":"NFL","team":{"id":"12","isAllStar":false}}},"seasonTypes":[{"displayName":"2026 Regular Season","categories":[{"displayName":"Regular Season Stats","events":[{"eventId":"401872976","stats":["15","30","225","50.0","7.5","2","0","55","2","97.2","63.6","2","0","0.0","0","1"]},{"eventId":"401872952","stats":["20","24","246","83.3","10.3","2","1","48","0","119.8","68.5","1","1","1.0","0","1"]}]}]}]}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let games = try PlayerHistoryService.espnGames(payload: payload, sport: "NFL", playerID: "3139477", marketKey: "player_pass_yds", season: 2026, cutoff: cutoff)
            XCTAssertEqual(games.count, 2)
            XCTAssertEqual(games.first?.sport, "NFL")
            XCTAssertEqual(games.first?.playerID, "3139477")
            XCTAssertNotNil(games.first?.opponentID)
        }
        do {
            let data = #"{"filters":[{"displayName":"Season","name":"season","value":"2026","options":[{"value":"2027","displayValue":"2026-27"},{"value":"2026","displayValue":"2025-26"},{"value":"2025","displayValue":"2024-25"},{"value":"2024","displayValue":"2023-24"},{"value":"2023","displayValue":"2022-23"},{"value":"2022","displayValue":"2021-22"},{"value":"2021","displayValue":"2020-21"},{"value":"2020","displayValue":"2019-20"},{"value":"2019","displayValue":"2018-19"},{"value":"2018","displayValue":"2017-18"},{"value":"2017","displayValue":"2016-17"},{"value":"2016","displayValue":"2015-16"}]}],"names":["goals","assists","points","plusMinus","penaltyMinutes","shotsTotal","shootingPct","powerPlayGoals","powerPlayAssists","shortHandedGoals","shortHandedAssists","gameWinningGoals","timeOnIcePerGame","production"],"events":{"401869776":{"id":"401869776","atVs":"@","gameDate":"2026-05-01T02:00:00.000+00:00","score":"5-2","homeTeamId":"25","awayTeamId":"6","homeTeamScore":"5","awayTeamScore":"2","gameResult":"L","opponent":{"id":"25","displayName":"Anaheim Ducks"},"leagueName":"National Hockey League","leagueAbbreviation":"NHL","leagueShortName":"NHL","eventNote":"West 1st Round - Game 6","team":{"id":"6","isAllStar":false},"type":{"id":"14","text":"Round of 16","abbreviation":"RD16","type":"round-of-16","slug":"round-of-16"}},"401869774":{"id":"401869774","atVs":"vs","gameDate":"2026-04-29T02:00:00.000+00:00","score":"4-1","homeTeamId":"6","awayTeamId":"25","homeTeamScore":"4","awayTeamScore":"1","gameResult":"W","opponent":{"id":"25","displayName":"Anaheim Ducks"},"leagueName":"National Hockey League","leagueAbbreviation":"NHL","leagueShortName":"NHL","eventNote":"West 1st Round - Game 5","team":{"id":"6","isAllStar":false},"type":{"id":"14","text":"Round of 16","abbreviation":"RD16","type":"round-of-16","slug":"round-of-16"}}},"seasonTypes":[{"displayName":"2025-26 Postseason","categories":[{"displayName":"Postseason","events":[{"eventId":"401869776","stats":["0","0","0","-3","0","3","0.0","0","0","0","0","0","24:49","0:00"],"type":{"id":"14","text":"Round of 16","abbreviation":"RD16","type":"round-of-16","slug":"round-of-16"}},{"eventId":"401869774","stats":["0","2","2","1","0","4","0.0","0","1","0","0","0","24:09","12:04"],"type":{"id":"14","text":"Round of 16","abbreviation":"RD16","type":"round-of-16","slug":"round-of-16"}}]}]}]}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let games = try PlayerHistoryService.espnGames(payload: payload, sport: "NHL", playerID: "3895074", marketKey: "player_shots_on_goal", season: 2026, cutoff: cutoff)
            XCTAssertEqual(games.count, 2)
            XCTAssertEqual(games.first?.sport, "NHL")
            XCTAssertEqual(games.first?.playerID, "3895074")
            XCTAssertNotNil(games.first?.opponentID)
        }
        do {
            let data = #"{"stats":[{"type":{"displayName":"gameLog"},"group":{"displayName":"hitting"},"splits":[{"season":"2026","stat":{"summary":"0-5 | 4 K","gamesPlayed":1,"flyOuts":0,"groundOuts":1,"airOuts":0,"runs":0,"doubles":0,"triples":0,"homeRuns":0,"strikeOuts":4,"baseOnBalls":0,"intentionalWalks":0,"hits":0,"hitByPitch":0,"avg":".000","atBats":5,"obp":".000","slg":".000","ops":".000","caughtStealing":0,"stolenBases":0,"stolenBasePercentage":".---","caughtStealingPercentage":".---","groundIntoDoublePlay":0,"groundIntoTriplePlay":0,"numberOfPitches":24,"plateAppearances":5,"totalBases":0,"rbi":0,"leftOnBase":4,"sacBunts":0,"sacFlies":0,"babip":".000","groundOutsToAirouts":"1.00","catchersInterference":0,"atBatsPerHomeRun":"-.--"},"team":{"id":147,"name":"New York Yankees","link":"/api/v1/teams/147"},"player":{"id":592450,"fullName":"Aaron Judge","link":"/api/v1/people/592450"},"league":{"id":103,"name":"American League","link":"/api/v1/league/103"},"sport":{"id":1,"link":"/api/v1/sports/1","abbreviation":"MLB"},"opponent":{"id":137,"name":"San Francisco Giants","link":"/api/v1/teams/137"},"date":"2026-03-25","gameType":"R","isHome":false,"isWin":true,"positionsPlayed":[{"code":"9","name":"Outfielder","type":"Outfielder","abbreviation":"RF"}],"game":{"gamePk":823244,"link":"/api/v1.1/game/823244/feed/live","content":{"link":"/api/v1/game/823244/content"},"gameNumber":1,"dayNight":"day"}},{"season":"2026","stat":{"summary":"1-4 | HR, 2 K, 2 RBI","gamesPlayed":1,"flyOuts":1,"groundOuts":0,"airOuts":1,"runs":1,"doubles":0,"triples":0,"homeRuns":1,"strikeOuts":2,"baseOnBalls":0,"intentionalWalks":0,"hits":1,"hitByPitch":0,"avg":".111","atBats":4,"obp":".111","slg":".444","ops":".555","caughtStealing":0,"stolenBases":0,"stolenBasePercentage":".---","caughtStealingPercentage":".---","groundIntoDoublePlay":0,"groundIntoTriplePlay":0,"numberOfPitches":19,"plateAppearances":4,"totalBases":4,"rbi":2,"leftOnBase":0,"sacBunts":0,"sacFlies":0,"babip":".000","groundOutsToAirouts":"0.00","catchersInterference":0,"atBatsPerHomeRun":"4.00"},"team":{"id":147,"name":"New York Yankees","link":"/api/v1/teams/147"},"player":{"id":592450,"fullName":"Aaron Judge","link":"/api/v1/people/592450"},"league":{"id":103,"name":"American League","link":"/api/v1/league/103"},"sport":{"id":1,"link":"/api/v1/sports/1","abbreviation":"MLB"},"opponent":{"id":137,"name":"San Francisco Giants","link":"/api/v1/teams/137"},"date":"2026-03-27","gameType":"R","isHome":false,"isWin":true,"positionsPlayed":[{"code":"9","name":"Outfielder","type":"Outfielder","abbreviation":"RF"}],"game":{"gamePk":823243,"link":"/api/v1.1/game/823243/feed/live","content":{"link":"/api/v1/game/823243/content"},"gameNumber":1,"dayNight":"day"}}]}]}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let games = try PlayerHistoryService.mlbGames(payload: payload, playerID: "592450", marketKey: "batter_hits", season: 2026, cutoff: cutoff)
            XCTAssertEqual(games.count, 2)
            XCTAssertEqual(games.first?.sport, "MLB")
            XCTAssertEqual(games.first?.playerID, "592450")
            XCTAssertNotNil(games.first?.opponentID)
        }
    }

    func testCapturedGoalieLogUsesActualSavesAndIceTime() throws {
        let data = #"{"filters":[{"name":"season","value":"2026"}],"names":["gameStarted","timeOnIcePerGame","wins","losses","ties","overtimeLosses","goalsAgainst","avgGoalsAgainst","shotsAgainst","saves","savePct","shutouts"],"events":{"401803641":{"id":"401803641","atVs":"@","gameDate":"2026-04-14T02:00:00.000+00:00","score":"6-2","homeTeamId":"37","awayTeamId":"28","homeTeamScore":"6","awayTeamScore":"2","gameResult":"L","opponent":{"id":"37","displayName":"Vegas Golden Knights"},"leagueName":"National Hockey League","leagueAbbreviation":"NHL","leagueShortName":"NHL","team":{"id":"28","isAllStar":false},"type":{"id":"1","text":"Standard","abbreviation":"STD","type":"standard","slug":"standard"}},"401803623":{"id":"401803623","atVs":"vs","gameDate":"2026-04-11T23:00:00.000+00:00","score":"7-1","homeTeamId":"28","awayTeamId":"15","homeTeamScore":"1","awayTeamScore":"7","gameResult":"L","opponent":{"id":"15","displayName":"Philadelphia Flyers"},"leagueName":"National Hockey League","leagueAbbreviation":"NHL","leagueShortName":"NHL","team":{"id":"28","isAllStar":false},"type":{"id":"1","text":"Standard","abbreviation":"STD","type":"standard","slug":"standard"}}},"seasonTypes":[{"displayName":"2025-26 Regular Season","categories":[{"displayName":"April","events":[{"eventId":"401803641","stats":["1","60:00","0","1","0","0","6","6.00","38","32",".842","0"],"type":{"id":"1","text":"Standard","abbreviation":"STD","type":"standard","slug":"standard"}},{"eventId":"401803623","stats":["1","40:00","0","1","0","0","5","7.50","20","15",".750","0"],"type":{"id":"1","text":"Standard","abbreviation":"STD","type":"standard","slug":"standard"}}]}]}]}"#.data(using: .utf8)!
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let games = try PlayerHistoryService.espnGames(payload: payload, sport: "NHL", playerID: "3020225", marketKey: "player_total_saves", season: 2026, cutoff: cutoff)
        XCTAssertEqual(games.count, 2)
        XCTAssertEqual(games.first?.value, 32)
        XCTAssertEqual(games.first?.workload, "60:00 TOI")
    }

    func testSignedNFLYardsAndWorkloadAreNotMadeAttemptPairs() throws {
        XCTAssertEqual(try parse(espn(["-8", "2"])).first?.value, -8)
        XCTAssertTrue(try parse(espn(["0", "0"])).isEmpty)
        XCTAssertTrue(try parse(espn(["10", "--"])).isEmpty)
        XCTAssertTrue(try parse(espn(["NaN", "2"])).isEmpty)
    }
    func testNFLZeroRoleStatRemainsObservedWhenOtherOffensiveWorkloadProvesAppearance() throws {
        let activeQuarterback = espn(["0", "0", "25"], names: ["rushingYards", "rushingAttempts", "passingAttempts"])
        XCTAssertEqual(try parse(activeQuarterback, market: "player_rush_yds").first?.value, 0)
        let missingRushingStat = espn(["0", "25"], names: ["rushingAttempts", "passingAttempts"])
        XCTAssertThrowsError(try parse(missingRushingStat, market: "player_rush_yds"))
    }

    func testNBAThreesAndNHLClockUseTypedFields() throws {
        let basketball = espn(["2-6", "30"], names: ["threePointFieldGoalsMade-threePointFieldGoalsAttempted", "minutes"], sport: "NBA")
        XCTAssertEqual(try parse(basketball, sport: "NBA", market: "player_threes").first?.value, 2)
        XCTAssertTrue(try parse(espn(["6-2", "30"], names: ["threePointFieldGoalsMade-threePointFieldGoalsAttempted", "minutes"], sport: "NBA"), sport: "NBA", market: "player_threes").isEmpty)
        let hockey = espn(["0", "2", "24:49"], names: ["powerPlayGoals", "powerPlayAssists", "timeOnIcePerGame"], sport: "NHL")
        let game = try XCTUnwrap(parse(hockey, sport: "NHL", market: "player_power_play_points").first)
        XCTAssertEqual(game.value, 2); XCTAssertEqual(game.workload, "24:49 TOI")
        XCTAssertTrue(try parse(espn(["3", "0:00"], names: ["shotsTotal", "timeOnIcePerGame"], sport: "NHL"), sport: "NHL", market: "player_shots_on_goal").isEmpty)
    }
    func testESPNRejectsAmbiguousHeadersWrongSeasonSportAndRaggedRows() throws {
        XCTAssertThrowsError(try parse(espn(["2", "3"], names: ["passingYards", "passingYards"])))
        XCTAssertThrowsError(try parse(espn(["2", "3"], season: 2025)))
        XCTAssertThrowsError(try parse(espn(["2", "3"]), sport: "NBA"))
        XCTAssertTrue(try parse(espn(["2"])).isEmpty)
        var wrong = espn(["2", "3"]); wrong["events"] = ["101": event("101", sport: "NHL")]
        XCTAssertTrue(try parse(wrong).isEmpty)
        var future = espn(["2", "3"]); future["events"] = ["101": event("101", day: "2026-10-07T20:00:00Z")]
        XCTAssertTrue(try parse(future).isEmpty)
        var preseason = espn(["2", "3"])
        preseason["seasonTypes"] = [["displayName": "2026 Preseason", "categories": [["events": [["eventId": "101", "stats": ["2", "3"]]]]]]]
        XCTAssertTrue(try parse(preseason).isEmpty)
    }
    func testESPNConflictingDuplicatesAreQuarantined() throws {
        var payload = espn(["2", "3"])
        payload["seasonTypes"] = [["displayName": "2026 Regular Season", "categories": [["events": [
            ["eventId": "101", "stats": ["2", "3"]], ["eventId": "101", "stats": ["4", "3"]]]]]]]
        var events = payload["events"] as! [String: [String: Any]]; events["102"] = event("102")
        payload["events"] = events
        var types = payload["seasonTypes"] as! [[String: Any]]
        var categories = types[0]["categories"] as! [[String: Any]]
        var rows = categories[0]["events"] as! [[String: Any]]
        rows.append(["eventId": "102", "stats": ["9", "3"]]); categories[0]["events"] = rows
        types[0]["categories"] = categories; payload["seasonTypes"] = types
        let games = try parse(payload)
        XCTAssertEqual(games.map(\.id), ["102"])
        XCTAssertEqual(games.first?.value, 9)
        rows[1] = ["eventId": "101", "stats": ["malformed"]]
        categories[0]["events"] = rows; types[0]["categories"] = categories; payload["seasonTypes"] = types
        XCTAssertEqual(try parse(payload).map(\.id), ["102"])
    }
    private func mlbRow(pitching: Bool = false) -> [String: Any] {
        let stat: [String: Any] = pitching ? ["strikeOuts": 0, "inningsPitched": "0.0", "battersFaced": 2] : ["hits": 0, "plateAppearances": 4]
        return ["season": "2026", "sport": ["id": 1], "player": ["id": 99], "team": ["id": 1],
         "opponent": ["id": 2, "name": "Team Two"], "date": "2026-09-01", "gameType": "R", "game": ["gamePk": 101],
         "stat": stat]
    }
    private func mlbPayload(_ row: [String: Any], pitching: Bool = false) -> [String: Any] {
        ["stats": [["type": ["displayName": "gameLog"], "group": ["displayName": pitching ? "pitching" : "hitting"], "splits": [row]]]]
    }
    func testMLBZeroStatIsObservedOnlyWithRealWorkloadAndCorrectIdentities() throws {
        func read(_ row: [String: Any], pitching: Bool = false) throws -> [PlayerGame] {
            try PlayerHistoryService.mlbGames(payload: mlbPayload(row, pitching: pitching), playerID: "99",
                marketKey: pitching ? "pitcher_strikeouts" : "batter_hits", season: 2026, cutoff: cutoff)
        }
        XCTAssertEqual(try read(mlbRow()).first?.value, 0)
        XCTAssertEqual(try read(mlbRow(pitching: true), pitching: true).first?.value, 0)
        for (key, value) in [("sport", ["id": 2] as Any), ("player", ["id": 88] as Any), ("opponent", ["id": 1, "name": "Own"] as Any), ("gameType", "S" as Any), ("season", "2025" as Any), ("stat", ["hits": 0] as Any), ("stat", ["hits": 0, "plateAppearances": 0] as Any)] {
            var row = mlbRow(); row[key] = value; XCTAssertTrue(try read(row).isEmpty)
        }
        var invalidPitching = mlbRow(pitching: true); invalidPitching["stat"] = ["strikeOuts": 0, "inningsPitched": "1.3", "battersFaced": 2]
        XCTAssertTrue(try read(invalidPitching, pitching: true).isEmpty)
    }
    func testSeasonYearUsesLeagueSeasonRatherThanCalendarYear() {
        XCTAssertEqual(PlayerHistoryService.firstSeason(sport: "NFL", year: 2027, month: 1), 2026)
        XCTAssertEqual(PlayerHistoryService.firstSeason(sport: "NFL", year: 2026, month: 10), 2026)
        XCTAssertEqual(PlayerHistoryService.firstSeason(sport: "NBA", year: 2026, month: 10), 2027)
        XCTAssertEqual(PlayerHistoryService.firstSeason(sport: "NHL", year: 2026, month: 5), 2026)
    }

    private func finalScheduleRow(id: Int = 101, year: Int = 2026, date: String = "2026-09-01", home: Int = 1, away: Int = 2,
                                  state: String = "Final", code: String = "F") -> [String: Any] {
        ["gamePk": id, "season": String(year), "officialDate": date, "gameType": "R",
         "status": ["abstractGameState": state, "codedGameState": code],
         "teams": ["home": ["team": ["id": home]], "away": ["team": ["id": away]]]]
    }
    func testCapturedMLBFinalScheduleUsesOfficialDateNotUTCMidnight() throws {
        let logData = #"{"stats":[{"type":{"displayName":"gameLog"},"group":{"displayName":"hitting"},"splits":[{"season":"2026","stat":{"summary":"0-5 | 4 K","gamesPlayed":1,"flyOuts":0,"groundOuts":1,"airOuts":0,"runs":0,"doubles":0,"triples":0,"homeRuns":0,"strikeOuts":4,"baseOnBalls":0,"intentionalWalks":0,"hits":0,"hitByPitch":0,"avg":".000","atBats":5,"obp":".000","slg":".000","ops":".000","caughtStealing":0,"stolenBases":0,"stolenBasePercentage":".---","caughtStealingPercentage":".---","groundIntoDoublePlay":0,"groundIntoTriplePlay":0,"numberOfPitches":24,"plateAppearances":5,"totalBases":0,"rbi":0,"leftOnBase":4,"sacBunts":0,"sacFlies":0,"babip":".000","groundOutsToAirouts":"1.00","catchersInterference":0,"atBatsPerHomeRun":"-.--"},"team":{"id":147,"name":"New York Yankees","link":"/api/v1/teams/147"},"player":{"id":592450,"fullName":"Aaron Judge","link":"/api/v1/people/592450"},"league":{"id":103,"name":"American League","link":"/api/v1/league/103"},"sport":{"id":1,"link":"/api/v1/sports/1","abbreviation":"MLB"},"opponent":{"id":137,"name":"San Francisco Giants","link":"/api/v1/teams/137"},"date":"2026-03-25","gameType":"R","isHome":false,"isWin":true,"positionsPlayed":[{"code":"9","name":"Outfielder","type":"Outfielder","abbreviation":"RF"}],"game":{"gamePk":823244,"link":"/api/v1.1/game/823244/feed/live","content":{"link":"/api/v1/game/823244/content"},"gameNumber":1,"dayNight":"day"}}]}]}"#.data(using: .utf8)!
        let scheduleData = #"{"dates":[{"games":[{"gamePk":823244,"season":"2026","gameType":"R","gameDate":"2026-03-26T00:05:00Z","officialDate":"2026-03-25","status":{"abstractGameState":"Final","codedGameState":"F","detailedState":"Final","statusCode":"F","startTimeTBD":false,"abstractGameCode":"F"},"teams":{"away":{"team":{"id":147}},"home":{"team":{"id":137}}}}]}]}"#.data(using: .utf8)!
        let log = try XCTUnwrap(JSONSerialization.jsonObject(with: logData) as? [String: Any])
        let schedule = try XCTUnwrap(JSONSerialization.jsonObject(with: scheduleData) as? [String: Any])
        let games = try PlayerHistoryService.mlbGames(payload: log, playerID: "592450", marketKey: "batter_hits", season: 2026, cutoff: cutoff)
        XCTAssertEqual(PlayerHistoryService.finalizedMLBGames(games: games, payload: schedule).map(\.id), ["823244"])
    }

    func testMLBFinalStatusMustMatchExactAppearanceDateTeamsAndSeason() throws {
        let appearance = try XCTUnwrap(PlayerHistoryService.mlbGames(payload: mlbPayload(mlbRow()), playerID: "99", marketKey: "batter_hits", season: 2026, cutoff: cutoff).first)
        func verified(_ metadata: [[String: Any]]) -> [PlayerGame] {
            PlayerHistoryService.finalizedMLBGames(games: [appearance], payload: ["dates": [["games": metadata]]])
        }
        let final = finalScheduleRow()
        XCTAssertEqual(verified([final]).count, 1)
        var homeAppearance = appearance; homeAppearance.homeAway = "home"
        for (flag, expected): (Any, String?) in [(true, nil), (false, "home"), (1, nil), ("false", nil)] {
            var metadata = final; metadata["neutralSite"] = flag
            let parsed = PlayerHistoryService.finalizedMLBGames(games: [homeAppearance], payload: ["dates": [["games": [metadata]]]])
            XCTAssertEqual(parsed.count, 1)
            XCTAssertEqual(parsed.first?.homeAway, expected)
        }
        for invalid in [finalScheduleRow(state: "Live", code: "U"), finalScheduleRow(state: "Final", code: "C"),
                        finalScheduleRow(date: "2026-09-02"), finalScheduleRow(home: 3), finalScheduleRow(year: 2025), finalScheduleRow(id: 102)] {
            XCTAssertTrue(verified([invalid]).isEmpty)
        }
        XCTAssertTrue(verified([]).isEmpty)
        XCTAssertTrue(verified([final, finalScheduleRow(state: "Live", code: "U")]).isEmpty)
        var missing = final; missing.removeValue(forKey: "status")
        XCTAssertTrue(verified([final, missing]).isEmpty)
    }
    func testMLBLoadVerifiesFinalStatusBeforeRecentAndOpponentHistory() async throws {
        func encoded(_ object: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: object) }
        let directory = try encoded(["teams": [["id": 1, "name": "Team One", "sport": ["id": 1]], ["id": 2, "name": "Team Two", "sport": ["id": 1]]]])
        let players = try encoded(["people": [["id": 99, "fullName": "Fixture Player"]]])
        let firstRoster = try encoded(["roster": [["person": ["id": 99, "fullName": "Fixture Player"]]]])
        let secondRoster = try encoded(["roster": [["person": ["id": 88, "fullName": "Other Player"]]]])
        let finalRow = mlbRow()
        var suspendedRow = mlbRow(); suspendedRow["game"] = ["gamePk": 102]
        let log = try encoded(["stats": [["type": ["displayName": "gameLog"], "group": ["displayName": "hitting"], "splits": [finalRow, suspendedRow]]]])
        let schedule = try encoded(["dates": [["games": [finalScheduleRow(), finalScheduleRow(id: 102, state: "Live", code: "U")]]]])
        var priorRow = mlbRow(); priorRow["season"] = "2025"; priorRow["date"] = "2025-09-01"; priorRow["game"] = ["gamePk": 2025101]
        let priorLog = try encoded(mlbPayload(priorRow))
        let transport: PlayerHistoryService.Transport = { url in
            if url.path.hasSuffix("/teams") { return directory }
            if url.path.hasSuffix("/players") { return players }
            if url.path.hasSuffix("/teams/1/roster") { return firstRoster }
            if url.path.hasSuffix("/teams/2/roster") { return secondRoster }
            if url.path.hasSuffix("/schedule") {
                if URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.contains(where: { $0.name == "gamePks" && ($0.value ?? "").contains("2025101") }) == true { throw URLError(.badServerResponse) }
                return schedule
            }
            if URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.contains(where: { $0.name == "season" && $0.value == "2026" }) == true { return log }
            if URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.contains(where: { $0.name == "season" && $0.value == "2025" }) == true { return priorLog }
            throw URLError(.badServerResponse)
        }
        let service = PlayerHistoryService(transport: transport)
        var market = historyMarket(); market.sport = "MLB"; market.market = "batter_hits"
        let result = await service.load(market: market, includeOpponent: true, now: cutoff)
        XCTAssertEqual(result.games.map(\.id), ["101"])
        XCTAssertEqual(result.opponentGames.map(\.id), ["101"])
        XCTAssertTrue(result.warnings.contains { $0.contains("final game status") })
        XCTAssertTrue(result.warnings.contains { $0.contains("2025 season") })
        // A failed status request cannot turn an unverified appearance into opponent membership.
        let failedStatus = PlayerHistoryService(transport: { url in
            if url.path.hasSuffix("/schedule") { throw URLError(.badServerResponse) }
            return try await transport(url)
        })
        market.commence = "2026-09-01T20:00:00Z"
        let failed = await failedStatus.load(market: market, includeOpponent: true, now: cutoff)
        XCTAssertTrue(failed.games.isEmpty)
        XCTAssertNil(failed.context)
        XCTAssertTrue(failed.warnings.contains { $0.contains("2026 season") })
    }

    func testSearchNeedsExactUniqueSportAndCompletePage() {
        let item: [String: Any] = ["id": "99", "displayName": "José Player", "sport": "hockey", "league": "nhl", "type": "player"]
        let valid: [String: Any] = ["count": 1, "pageCount": 1, "items": [item]]
        XCTAssertEqual(PlayerHistoryService.espnSearchPlayerID(payload: valid, sport: "NHL", playerName: "Jose Player"), "99")
        XCTAssertNil(PlayerHistoryService.espnSearchPlayerID(payload: valid, sport: "NBA", playerName: "Jose Player"))
        XCTAssertNil(PlayerHistoryService.espnSearchPlayerID(payload: valid, sport: "NHL", playerName: "Jose"))
        var duplicate = valid; duplicate["count"] = 2; duplicate["items"] = [item, item]
        XCTAssertNil(PlayerHistoryService.espnSearchPlayerID(payload: duplicate, sport: "NHL", playerName: "Jose Player"))
        var paged = valid; paged["pageCount"] = 2
        XCTAssertNil(PlayerHistoryService.espnSearchPlayerID(payload: paged, sport: "NHL", playerName: "Jose Player"))
    }
    private func fixtureTransport() throws -> PlayerHistoryService.Transport {
        func encoded(_ value: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: value) }
        let directory = try encoded(["sports": [["slug": "football", "leagues": [["slug": "nfl", "teams": [
            ["team": ["id": "1", "displayName": "Team One"]], ["team": ["id": "2", "displayName": "Team Two"]]]]]]]])
        let firstRoster = try encoded(["athletes": [["id": "99", "displayName": "Fixture Player"]]])
        let secondRoster = try encoded(["athletes": [["id": "88", "displayName": "Other Player"]]])
        let search = try encoded(["count": 1, "pageCount": 1, "items": [["id": "99", "displayName": "Fixture Player", "type": "player", "sport": "football", "league": "nfl"]]])
        func seasonLog(_ season: Int) throws -> Data {
            var events: [String: Any] = [:], rows: [[String: Any]] = []
            let count = season == 2026 ? 12 : 8
            for index in 1...count {
                let id = String(season * 100 + index), opponent = season == 2026 && index > 2 ? "3" : "2"
                events[id] = event(id, day: "\(season)-09-\(String(format: "%02d", index))T20:00:00Z", opponent: opponent)
                rows.append(["eventId": id, "stats": [String(index), "2"]])
            }
            return try encoded(["filters": [["name": "season", "value": String(season)]], "names": ["passingYards", "passingAttempts"],
                "events": events, "seasonTypes": [["displayName": "\(season) Regular Season", "categories": [["events": rows]]]]])
        }
        let season2026 = try seasonLog(2026), season2025 = try seasonLog(2025), season2024 = try seasonLog(2024)
        return { url in
            if url.path.hasSuffix("/teams") { return directory }
            if url.path.hasSuffix("/teams/1/roster") { return firstRoster }
            if url.path.hasSuffix("/teams/2/roster") { return secondRoster }
            if url.path.hasSuffix("/search") { return search }
            let season = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "season" })?.value
            if season == "2026" { return season2026 }
            if season == "2025" { return season2025 }
            if season == "2024" { return season2024 }
            throw URLError(.badServerResponse)
        }
    }
    private func historyMarket(start: String = "2026-10-08T20:00:00Z", label: String? = "Team One at Team Two") -> Market {
        Market(id: "fixture", event: "target", sport: "NFL", kind: "prop", name: "Fixture Player", opponent: "Team One",
            label: "Over 10.5", aliases: "", p: nil, price: -110, market: "player_pass_yds", book: "Fixture", quotes: [], mode: "live",
            line: 10.5, eventLabel: label, commence: start, side: "Over")
    }
    func testAnalyticsCanRetainTwoSeasonsWhileExistingConsumersKeepTenGames() async throws {
        let service = PlayerHistoryService(transport: try fixtureTransport())
        let recent = await service.load(market: historyMarket(), seasonCount: 2, now: cutoff)
        XCTAssertEqual(recent.games.count, 10)
        let research = await service.load(market: historyMarket(), seasonCount: 2, now: cutoff, maximumGames: 2_000)
        XCTAssertEqual(research.games.count, 20)
        XCTAssertEqual(Set(research.games.map(\.season)), [2026, 2025])
        XCTAssertEqual(research.games.prefix(10).map(\.id), recent.games.map(\.id))
        XCTAssertEqual(research.games.prefix(10).map(\.value), recent.games.map(\.value))
        let stillRecent = await service.load(market: historyMarket(), seasonCount: 2, now: cutoff)
        XCTAssertEqual(stillRecent.games.map(\.id), recent.games.map(\.id), "Cached raw provider data must not change each caller’s requested window")
        let firstOnly = await service.load(market: historyMarket(), seasonCount: 2, now: cutoff, maximumGames: 0)
        XCTAssertEqual(firstOnly.games.count, 1)
    }
    func testOpponentHistoryFetchesOlderSeasonsRatherThanFilteringRecentTen() async throws {
        let service = PlayerHistoryService(transport: try fixtureTransport())
        let result = await service.load(market: historyMarket(), includeOpponent: true, now: cutoff)
        XCTAssertEqual(result.games.count, 10)
        XCTAssertTrue(result.games.allSatisfy { $0.opponentID == "3" })
        XCTAssertEqual(result.opponentGames.count, 10)
        XCTAssertTrue(result.opponentGames.allSatisfy { $0.opponentID == "2" })
        XCTAssertEqual(result.context?.opponentID, "2") // Market.opponent incorrectly names the player's own team.
        XCTAssertEqual(result.context?.loadedSeasons, [2026, 2025, 2024])
        XCTAssertEqual(result.context?.requestedSeasons, [2026, 2025, 2024])
        XCTAssertTrue(result.warnings.isEmpty)
    }
    func testPastMatchupNeedsAppearanceAndUnverifiedOpponentKeepsRecentGames() async throws {
        let service = PlayerHistoryService(transport: try fixtureTransport())
        let past = await service.load(market: historyMarket(start: "2026-09-01T20:00:00Z"), includeOpponent: true, now: cutoff)
        XCTAssertEqual(past.context?.opponentID, "2")
        XCTAssertTrue(past.games.allSatisfy { $0.date < sportsDate("2026-09-01T04:00:00Z")! })
        let noAppearance = await service.load(market: historyMarket(start: "2026-09-20T20:00:00Z"), includeOpponent: true, now: cutoff)
        XCTAssertNil(noAppearance.context)
        XCTAssertEqual(noAppearance.games.count, 10)
        XCTAssertTrue(noAppearance.warnings.contains { $0.contains("Opponent is unverified") })
        let noLabel = await service.load(market: historyMarket(label: nil), includeOpponent: true, now: cutoff)
        XCTAssertNil(noLabel.context)
        XCTAssertEqual(noLabel.games.count, 10)
    }
    func testFailedSeasonRetainsVerifiedRecentGamesAndWarning() async throws {
        let successful = try fixtureTransport()
        let service = PlayerHistoryService(transport: { url in
            if URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.contains(where: { $0.name == "season" && $0.value == "2024" }) == true { throw URLError(.badServerResponse) }
            return try await successful(url)
        })
        let result = await service.load(market: historyMarket(label: nil), includeOpponent: true, now: cutoff)
        XCTAssertEqual(result.games.count, 10)
        XCTAssertTrue(result.warnings.contains { $0.contains("2024 season") })
    }

    func testCacheCoalescesAndExpires() async throws {
        let counter = HistoryReadCounter(), clock = HistoryTestClock()
        let service = PlayerHistoryService(transport: { try await counter.read($0) }, clock: { clock.now() })
        let url = URL(string: "https://site.api.espn.com/apis/common/v3/search?query=Player")!
        async let first = service.read(url); async let second = service.read(url)
        _ = try await (first, second)
        _ = try await service.read(url)
        let initial = await counter.count(url); XCTAssertEqual(initial, 1)
        clock.advance(901); _ = try await service.read(url)
        let refreshed = await counter.count(url); XCTAssertEqual(refreshed, 2)
    }
    func testCacheBoundsEvictOldPublicReadsAndCancelledWaiterLeavesOtherConsumer() async throws {
        let counter = HistoryReadCounter(), clock = HistoryTestClock()
        let service = PlayerHistoryService(transport: { try await counter.read($0) }, clock: { clock.now() })
        let firstURL = URL(string: "https://site.api.espn.com/x?id=0")!
        for index in 0...64 {
            _ = try await service.read(URL(string: "https://site.api.espn.com/x?id=\(index)")!)
            clock.advance(1)
        }
        _ = try await service.read(firstURL)
        let calls = await counter.count(firstURL); XCTAssertEqual(calls, 2)
        let sharedURL = URL(string: "https://site.api.espn.com/shared")!
        let cancelled = Task { try await service.read(sharedURL) }
        async let surviving = service.read(sharedURL)
        try await Task.sleep(nanoseconds: 2_000_000)
        cancelled.cancel()
        let data = try await surviving; XCTAssertEqual(data, Data("{}".utf8))
        do { _ = try await cancelled.value; XCTFail("Cancelled consumer received a result") }
        catch is CancellationError { }
        let sharedCalls = await counter.count(sharedURL); XCTAssertEqual(sharedCalls, 1)
    }

    func testInvalidURLAndOversizedResponseAreNotAccepted() async throws {
        let service = PlayerHistoryService(transport: { _ in Data(repeating: 0, count: 4 * 1024 * 1024 + 1) })
        for text in ["http://site.api.espn.com/x", "https://site.api.espn.com.evil.test/x", "https://user:secret@site.api.espn.com/x", "https://statsapi.mlb.com/private"] {
            XCTAssertFalse(PlayerHistoryService.allowedURL(URL(string: text)!))
        }
        do { _ = try await service.read(URL(string: "https://site.api.espn.com/x")!); XCTFail("Accepted oversized data") }
        catch PlayerHistoryError.oversizedResponse { }
        catch { XCTFail("Unexpected error: \(error)") }
    }
    func testCancelledConsumerDoesNotReceiveCompletedData() async throws {
        let counter = HistoryReadCounter()
        let service = PlayerHistoryService(transport: { try await counter.read($0) })
        let task = Task { try await service.read(URL(string: "https://site.api.espn.com/x")!) }
        task.cancel()
        do { _ = try await task.value; XCTFail("Cancelled request returned data") }
        catch is CancellationError { }
    }
}

extension PlayerHistoryServiceTests {
    func testPlayerVenueNeedsExplicitOwnCompetitorAndStrictMLBBoolean() throws {
        func espnVenue(_ venue: Any?, duplicate: Bool = false) throws -> String? {
            var payload = espn(["200", "20"])
            var row = event("101")
            var own: [String: Any] = ["team": ["id": "1"]]
            own["homeAway"] = venue
            row["competitions"] = [["competitors": duplicate ? [own, own] : [own, ["team": ["id": "2"], "homeAway": "away"]]]]
            payload["events"] = ["101": row]
            return try parse(payload).first?.homeAway
        }
        XCTAssertEqual(try espnVenue("home"), "home")
        XCTAssertNil(try espnVenue("away"))
        XCTAssertNil(try espnVenue("neutral")); XCTAssertEqual(try espnVenue(nil), "home"); XCTAssertNil(try espnVenue("home", duplicate: true))
        XCTAssertEqual(try parse(espn(["200", "20"])).first?.homeAway, "home")
        var awayPayload = espn(["200", "20"])
        var awayEvent = event("101"); awayEvent["homeTeamId"] = "2"; awayEvent["awayTeamId"] = "1"
        awayEvent["homeTeamScore"] = "10"; awayEvent["awayTeamScore"] = "20"
        awayEvent["atVs"] = "vs" // Display notation cannot override verified source team IDs.
        awayPayload["events"] = ["101": awayEvent]
        XCTAssertEqual(try parse(awayPayload).first?.homeAway, "away")
        awayEvent["team"] = ["id": "1", "homeAway": "home"]
        awayPayload["events"] = ["101": awayEvent]
        XCTAssertNil(try parse(awayPayload).first?.homeAway)
        awayEvent["homeTeamId"] = "1"; awayPayload["events"] = ["101": awayEvent]
        XCTAssertTrue(try parse(awayPayload).isEmpty)
        for (proof, expected): (Any, String?) in [(true, "home"), (false, "away"), (1, nil), (0, nil), ("true", nil)] {
            var row = mlbRow(); row["isHome"] = proof
            let parsed = try PlayerHistoryService.mlbGames(payload: mlbPayload(row), playerID: "99", marketKey: "batter_hits", season: 2026, cutoff: cutoff)
            XCTAssertEqual(parsed.first?.homeAway, expected)
            XCTAssertEqual(parsed.first?.season, 2026)
        }
    }
    func testPlayerGameVenueConflictQuarantinesDuplicate() throws {
        let first = try XCTUnwrap(parse(espn(["200", "20"])).first)
        var home = first; home.homeAway = "home"
        var away = first; away.homeAway = "away"
        XCTAssertTrue(try PlayerHistoryService.uniqueGames([home, away]).isEmpty)
        XCTAssertEqual(try PlayerHistoryService.uniqueGames([home, home]).count, 1)
    }
}

extension PlayerHistoryServiceTests {
    func testNeutralAndMalformedNeutralSiteSuppressPlayerVenueOnly() throws {
        for competitionFlag in [false, true] {
            for (flag, expected): (Any, String?) in [(true, nil), (false, "home"), (1, nil), (0, nil), ("false", nil), (NSNull(), nil)] {
                var payload = espn(["200", "20"])
                var row = event("101")
                var competition: [String: Any] = ["competitors": [["team": ["id": "1"], "homeAway": "home"], ["team": ["id": "2"], "homeAway": "away"]]]
                if competitionFlag { competition["neutralSite"] = flag } else { row["neutralSite"] = flag }
                row["competitions"] = [competition]; payload["events"] = ["101": row]
                let parsed = try XCTUnwrap(parse(payload).first)
                XCTAssertEqual(parsed.homeAway, expected)
                XCTAssertEqual(parsed.value, 200); XCTAssertEqual(parsed.season, 2026)
            }
        }
        for (flag, expected): (Any, String?) in [(true, nil), (false, "home"), (1, nil), ("false", nil)] {
            var row = mlbRow(); row["isHome"] = true; row["neutralSite"] = flag
            let parsed = try XCTUnwrap(PlayerHistoryService.mlbGames(payload: mlbPayload(row), playerID: "99", marketKey: "batter_hits", season: 2026, cutoff: cutoff).first)
            XCTAssertEqual(parsed.homeAway, expected)
            XCTAssertEqual(parsed.value, 0)
        }
    }
}
