import XCTest
@testable import PLONNCore

final class PlayerSimulationTests: XCTestCase {
    let now = sportsDate("2026-10-23T21:00:00Z")!
    func history(_ sport: InterestSport = .nba, id: String = "1", sparse: Bool = false) -> PlayerSimulation.History {
        let team = InterestTeam(sport: sport, providerID: id, name: "Team " + id, abbreviation: "T" + id)
        let playerID = id == "1" ? "11" : "12"
        let roster = [PlayerMatchupPlayer(id: playerID, name: "Player " + id, teamID: id, team: team.name, position: "G")]
        let boxes = (0..<6).map { i -> PlayerSimulation.Box in
            let points = Double(10 + i * 2)
            let line = PlayerSimulation.Line(playerID: playerID, name: roster[0].name, group: "players", values: ["points":points, "rebounds":Double(i), "assists":Double(5 - i)])
            let lines = sparse && i < 4 ? [PlayerSimulation.Line(playerID: "22", name: "Other", group: "players", values: ["points":points])] : [line]
            return .init(game: .init(id: String(100 + i), date: now.addingTimeInterval(-Double(i + 1) * 86400), season: 2027, scored: Int(points), allowed: 11, home: true), lines: lines)
        }
        return .init(team: team, roster: roster, boxes: boxes, partial: false)
    }
    func testWholeBoxScoresAndPlayerCombinationsStayTogether() throws {
        let result = try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123)
        XCTAssertEqual(result.trials.count, 5000)
        XCTAssertEqual(result.awayDrawCounts.reduce(0, +), 5000)
        XCTAssertEqual(result.trials, try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123).trials)
        for i in result.trials.indices {
            let box = result.box(homeSide: false, trial: i)
            XCTAssertEqual(box.game.scored, result.scores.scores[i].away)
            XCTAssertEqual(box.lines.reduce(0) { $0 + ($1.values["points"] ?? 0) }, Double(result.scores.scores[i].away))
            XCTAssertEqual(SimulationOdds.value(box: box, playerID: "11", market: "player_rebounds_assists", sport: "NBA"), 5)
        }
    }
    func testRangesRequireDistinctAppearancesAndNeverImputeAbsence() throws {
        let result = try PlayerSimulation.run(away: history(sparse: true), home: history(id: "2"), now: now, seed: 123)
        XCTAssertNil(result.range(homeSide: false, lineID: "players:11", stat: "points"))
        let full = try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123)
        let range = try XCTUnwrap(full.range(homeSide: false, lineID: "players:11", stat: "points"))
        XCTAssertEqual(range.appearances, 6)
        XCTAssertEqual(range.low, 10); XCTAssertEqual(range.high, 20)
        XCTAssertEqual(range.average, full.scores.awaySummary.average, accuracy: 1e-10)
        XCTAssertNil(full.range(homeSide: false, lineID: "players:11", stat: "missing"))
        XCTAssertNil(SimulationOdds.value(box: full.away.boxes[0], playerID: "11", market: "player_frees_made", sport: "NBA"))
    }
    func testSportUnitsAndUnsupportedStatsAreNotInvented() {
        XCTAssertEqual(PlayerSimulationSource.number("4.2", kind: .outs), 14)
        XCTAssertNil(PlayerSimulationSource.number("4.3", kind: .outs))
        XCTAssertEqual(PlayerSimulationSource.number("19:13", kind: .seconds), 1153)
        XCTAssertNil(PlayerSimulationSource.number("19:70", kind: .seconds))
        XCTAssertEqual(PlayerSimulation.Stat(id: "ip", label: "IP", title: "Innings", kind: .outs).display(14), "4.2")
        XCTAssertEqual(PlayerSimulation.groups(.nfl).count, 9)
        XCTAssertEqual(PlayerSimulation.groups(.nhl).map(\.id), ["skaters", "goalies"])
        XCTAssertEqual(PlayerSimulation.groups(.mlb).map(\.id), ["batting", "pitching"])
        let box = history().boxes[0]
        XCTAssertNil(SimulationOdds.value(box: box, playerID: "11", market: "player_fantasy_points", sport: "NBA"))
        XCTAssertNil(SimulationOdds.value(box: box, playerID: "99", market: "player_points", sport: "NBA"))
    }
    func testRejectsWrongSportSparseStaleFutureAndDuplicateGames() throws {
        let good = history()
        XCTAssertThrowsError(try PlayerSimulation.run(away: good, home: history(.nfl, id: "2"), now: now, seed: 1))
        XCTAssertThrowsError(try PlayerSimulation.run(away: good, home: good, now: now, seed: 1))
        for boxes in [Array(good.boxes.prefix(4)), good.boxes + [good.boxes[0]]] {
            let invalid = PlayerSimulation.History(team: good.team, roster: good.roster, boxes: boxes, partial: false)
            XCTAssertThrowsError(try PlayerSimulation.run(away: invalid, home: history(id: "2"), now: now, seed: 1))
        }
        XCTAssertThrowsError(try PlayerSimulation.run(away: good, home: history(id: "2"), now: now.addingTimeInterval(-20 * 86400), seed: 1))
        XCTAssertThrowsError(try PlayerSimulation.run(away: good, home: history(id: "2"), now: now.addingTimeInterval(300 * 86400), seed: 1))
        XCTAssertThrowsError(try PlayerSimulation.run(away: good, home: history(id: "2"), now: now, seed: 1, count: 0))
    }
    func event() -> PlayerMatchupEvent {
        .init(game: .init(league: "NBA", eventID: "401999999", home: "Team 2", away: "Team 1", startsAt: "2026-10-24T21:00:00Z"), homeID: "2", awayID: "1", seasonType: 2)
    }
    func quote(_ id: String = "q1", key: String = "spreads", name: String = "Team 1", line: Double? = 0) -> Market {
        Market(id: id, event: "odds1", sport: "NBA", kind: key.hasPrefix("player") ? "prop" : "team", name: name, opponent: "Team 2", label: "Line", aliases: "", p: nil, price: -110, market: key, book: "Example book", quotes: [], updated: "2026-10-23T20:55:00Z", mode: "live", line: line, eventLabel: "Team 1 at Team 2", eventState: "pre", commence: "2026-10-24T21:00:00Z", side: key.hasPrefix("player") || key == "totals" ? "Over" : nil)
    }
    func testOddsPushesMissingAppearancesAndImpliedProbability() throws {
        let result = try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123)
        let quotes = [quote(), quote("q2", key: "h2h", line: nil), quote("q3", key: "player_points", name: "Player 1", line: 14)]
        let values = SimulationOdds.compare(result, event: event(), markets: quotes, now: now)
        XCTAssertEqual(values.count, 3)
        let spread = try XCTUnwrap(values.first { $0.id == "q1" })
        XCTAssertGreaterThan(spread.push, 0)
        XCTAssertEqual(spread.hit + spread.miss + spread.push + spread.unavailable, 5000)
        XCTAssertEqual(try XCTUnwrap(spread.implied), 110 / 210.0, accuracy: 1e-10)
        let moneyline = try XCTUnwrap(values.first { $0.id == "q2" })
        XCTAssertEqual(moneyline.push, 0); XCTAssertEqual(moneyline.unavailable, spread.push)
        let prop = try XCTUnwrap(values.first { $0.id == "q3" })
        XCTAssertEqual(prop.playerAppearances, 6); XCTAssertGreaterThan(prop.push, 0)
        let sparse = try PlayerSimulation.run(away: history(sparse: true), home: history(id: "2"), now: now, seed: 123)
        let noRate = try XCTUnwrap(SimulationOdds.compare(sparse, event: event(), markets: [quotes[2]], now: now).first)
        XCTAssertNil(noRate.hitRate); XCTAssertEqual(noRate.unavailable, 5000)
        XCTAssertEqual(noRate.missingEstimate, .limitedHistory(2))
        XCTAssertEqual(noRate.missingEstimate?.title, "Limited history")
        XCTAssertNil(prop.missingEstimate)
        // A withheld simulation estimate must not prevent saving the real quote.
        XCTAssertNil(SimulationOdds.builderRejection(for: noRate.quote, event: event(), existing: [], now: now))
    }
    func testSimulatorBuilderPreservesContractAndRejectsConflictingOrExpiredPicks() throws {
        let result = try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123)
        let original = quote("prop", key: "player_points", name: "Player 1", line: 14)
        let comparison = try XCTUnwrap(SimulationOdds.compare(result, event: event(), markets: [original], now: now).first)
        XCTAssertEqual(comparison.quote, original)
        XCTAssertNil(comparison.quote.p); XCTAssertNotEqual(comparison.quote.modelValidated, true)
        XCTAssertNotNil(SimulationOdds.builderRejection(for: original, event: event(), existing: [original], now: now))
        var opposite = original; opposite.id = "under"; opposite.side = "Under"
        XCTAssertNotNil(SimulationOdds.builderRejection(for: opposite, event: event(), existing: [original], now: now))
        XCTAssertNotNil(SimulationOdds.builderRejection(for: original, event: event(), existing: [], now: event().game.start!))
        var stale = original; stale.updated = "2026-10-21T20:00:00Z"
        XCTAssertNotNil(SimulationOdds.builderRejection(for: stale, event: event(), existing: [], now: now))
        var future = original; future.updated = "2026-10-23T22:00:00Z"
        XCTAssertNotNil(SimulationOdds.builderRejection(for: future, event: event(), existing: [], now: now))
        var invalid = original; invalid.price = 0
        XCTAssertNotNil(SimulationOdds.builderRejection(for: invalid, event: event(), existing: [], now: now))
        XCTAssertNotNil(SimulationOdds.builderRejection(for: original, event: nil, existing: [], now: now))
        let drafts = (0..<40).map { index -> Market in
            var item = original; item.id = "pick-\(index)"; item.event = "game-\(index)"; item.eventLabel = "Different game \(index)"; return item
        }
        XCTAssertNil(SimulationOdds.builderRejection(for: original, event: event(), existing: drafts, now: now))
    }
    func testUnsupportedPropExplainsMissingEstimateInsteadOfImplyingZeroChance() throws {
        let result = try PlayerSimulation.run(away: history(.nhl), home: history(.nhl,id: "2"), now: now, seed: 123)
        var contract = quote("power-play", key: "player_power_play_points", name: "Player 1", line: 0.5)
        contract.sport = "NHL"
        let scheduled = PlayerMatchupEvent(game: .init(league:"NHL",eventID:"401999999",home:"Team 2",away:"Team 1",startsAt:"2026-10-24T21:00:00Z"),homeID:"2",awayID:"1",seasonType:2)
        let comparison = try XCTUnwrap(SimulationOdds.compare(result, event: scheduled, markets: [contract], now: now).first)
        XCTAssertNil(comparison.hitRate)
        XCTAssertEqual(comparison.missingEstimate, .notModeled)
        XCTAssertNil(SimulationOdds.builderRejection(for: contract, event: scheduled, existing: [], now: now))
    }
    func testOddsRankByScenarioRateWithStableTiesAndMissingLast() {
        func row(_ id:String, _ hit:Int, _ miss:Int, _ push:Int = 0) -> SimulationOdds.Comparison {
            .init(quote:quote(id),hit:hit,miss:miss,push:push,unavailable:0,playerAppearances:nil)
        }
        let rows = [row("missing",0,0),row("zero",0,100),row("highest",95,5),row("tie-b",50,50),row("tie-a",25,25),row("pushes",80,0,20)]
        let expected = ["highest","pushes","tie-a","tie-b","zero","missing"]
        XCTAssertEqual(SimulationOdds.ranked(rows).map(\.id),expected)
        XCTAssertEqual(SimulationOdds.ranked(rows.reversed()).map(\.id),expected)
        // Book filtering preserves the same descending ordering.
        XCTAssertEqual(SimulationOdds.ranked(rows.filter { $0.id != "highest" }).first?.id,"pushes")
    }
    func testCapturedPlayersWithTwoCurrentTeamAppearancesExplainMissingRange() async throws {
        guard let path = ProcessInfo.processInfo.environment["PLONN_MARKET_DIAGNOSTIC"] else { throw XCTSkip("Opt-in captured current-team history") }
        let root = URL(fileURLWithPath: path)
        let index = try JSONDecoder().decode([String:String].self, from: Data(contentsOf: root.appendingPathComponent("response-index.json")))
        let reader: MatchupSimulationSource.Reader = { url in
            guard let file = index[url.absoluteString] else { throw MatchupSimulation.Failure.source }
            return try Data(contentsOf: root.appendingPathComponent("responses/" + file))
        }
        let team = InterestTeam(sport: .nba, providerID: "20", name: "Philadelphia 76ers", abbreviation: "PHI")
        let roster = try await reader(URL(string:"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/20/roster")!)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with:roster) as? [String:Any])
        let checked = try XCTUnwrap((body["timestamp"] as? String).flatMap(sportsDate))
        let schedule = try await MatchupSimulationSource(read:reader).history(team, cutoff:checked)
        let history = try await PlayerSimulationSource(read:reader).history(schedule, cutoff:checked)
        for playerName in ["Jaylen Brown", "LeBron James"] {
            let player = try XCTUnwrap(history.roster.first { $0.name == playerName })
            let appearances = history.boxes.filter { SimulationOdds.value(box: $0, playerID: player.id, market: "player_points", sport: "NBA") != nil }
            XCTAssertEqual(appearances.count, 2, playerName)
            let result = try PlayerSimulation.run(away:history, home: self.history(), now:now, seed:7, count:100)
            XCTAssertNil(result.range(homeSide:false, lineID:"players:" + player.id, stat:"points"))
            XCTAssertEqual(result.rangeUnavailableReason(homeSide:false, lineID:"players:" + player.id, stat:"points"), "2 recorded games · 3 needed for a range.")
        }
    }
    func testOddsRejectCustomWrongEventStaleAndConflict() throws {
        let result = try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123)
        XCTAssertTrue(SimulationOdds.compare(result, event: nil, markets: [quote()], now: now).isEmpty)
        XCTAssertTrue(SimulationOdds.compare(result, event: event(), markets: [quote()], now: now.addingTimeInterval(901)).isEmpty)
        var stale = quote(); stale.updated = "2026-10-21T20:55:00Z"
        var wrong = quote(); wrong.eventLabel = "Team 2 at Team 1"
        var future = quote(); future.updated = "2026-10-23T21:01:00Z"
        for value in [stale, wrong, future] { XCTAssertTrue(SimulationOdds.compare(result, event: event(), markets: [value], now: now).isEmpty) }
        var conflict = quote(); conflict.price = 150
        XCTAssertTrue(SimulationOdds.compare(result, event: event(), markets: [quote(), conflict], now: now).isEmpty)
    }
    func testNBATenMinuteTipOffOffsetMatchesTeamAndPlayerPrices() throws {
        let result = try PlayerSimulation.run(away: history(), home: history(id: "2"), now: now, seed: 123)
        var team = quote(), player = quote("prop", key: "player_points", name: "Player 1", line: 14)
        team.commence = "2026-10-24T21:10:00Z"; player.commence = team.commence
        let rows = [team, player]
        XCTAssertEqual(Set(SimulationOdds.compare(result, event: event(), markets: rows, now: now).map(\.id)), [team.id, player.id])
        var other = team; other.event = "another-game"; other.id = "other"
        XCTAssertTrue(SimulationOdds.compare(result, event: event(), markets: rows + [other], now: now).isEmpty)
    }
    func testLargeRecordedSlateRetainsEveryDistinctEventIdentity() throws {
        var selected = quote(); selected.commence = "2026-10-24T21:10:00Z"
        let otherContracts = (0..<2400).map { index -> Market in
            var row = selected; row.id = "other-\(index)"; row.market = "player_points_alternate"; return row
        }
        let scheduled = event()
        let proof = AnalyticsEvent(sport: "NBA", providerID: scheduled.game.eventID, home: scheduled.game.home, away: scheduled.game.away, start: scheduled.game.start!, homeID: scheduled.homeID, awayID: scheduled.awayID, seasonType: 2)
        let target = AnalyticsStatsTarget(sport: "NBA", kind: "team", name: selected.name, marketKey: "spreads", teamName: selected.name)
        let rows = [selected] + otherContracts
        XCTAssertEqual(AnalyticsOddsMatcher.matchCached(target: target, event: proof, markets: rows, now: now, recordedSnapshot: true).markets, [selected])
        var conflicting = otherContracts[0]; conflicting.event = "second-event"; conflicting.id = "conflicting"
        XCTAssertTrue(AnalyticsOddsMatcher.matchCached(target: target, event: proof, markets: rows + [conflicting], now: now, recordedSnapshot: true).markets.isEmpty)
    }
    func fixtureDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["PLONN_PLAYER_SIM_FIXTURES"] else { throw XCTSkip("Set PLONN_PLAYER_SIM_FIXTURES for captured four-sport box scores") }
        return URL(fileURLWithPath: path)
    }
    func parseFixture(_ data: Data, sport: InterestSport, teamID: String) throws -> (PlayerSimulation.Box, InterestTeam) {
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let header = try XCTUnwrap(root["header"] as? [String: Any]), competition = try XCTUnwrap((header["competitions"] as? [[String: Any]])?.first)
        let competitors = try XCTUnwrap(competition["competitors"] as? [[String: Any]])
        let own = try XCTUnwrap(competitors.first { $0["id"] as? String == teamID }), other = try XCTUnwrap(competitors.first { $0["id"] as? String != teamID })
        let identity = try XCTUnwrap(own["team"] as? [String: Any])
        let team = InterestTeam(sport: sport, providerID: teamID, name: identity["displayName"] as! String, abbreviation: identity["abbreviation"] as! String)
        let game = MatchupSimulation.Game(id: header["id"] as! String, date: sportsDate(competition["date"] as! String)!, season: (header["season"] as! [String: Any])["year"] as! Int,
                                         scored: Int(own["score"] as! String)!, allowed: Int(other["score"] as! String)!, home: own["homeAway"] as? String == "home")
        return (try PlayerSimulationSource.parse(data, team: team, game: game, cutoff: now), team)
    }
    func testCapturedFourSportBoxesAndJointScenarios() throws {
        let directory = try fixtureDirectory()
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as! [[String: Any]]
        var histories: [InterestSport: [PlayerSimulation.History]] = [:]
        for item in manifest {
            let sport = InterestSport(rawValue: item["sport"] as! String)!, rawTeam = item["team"] as! [String: Any], id = rawTeam["id"] as! String
            let team = InterestTeam(sport: sport, providerID: id, name: rawTeam["displayName"] as! String, abbreviation: rawTeam["abbreviation"] as! String)
            let roster = try PlayerMatchupSource.players(Data(contentsOf: directory.appendingPathComponent("\(sport.rawValue)-\(id)-roster.json")), teamID: id, teamName: team.name)
            var boxes: [PlayerSimulation.Box] = []
            for event in item["events"] as! [[String: Any]] {
                let eventID = event["id"] as! String
                do { boxes.append(try parseFixture(Data(contentsOf: directory.appendingPathComponent("\(sport.rawValue)-\(eventID)-box.json")), sport: sport, teamID: id).0) }
                catch { XCTFail("\(sport.rawValue) \(id) \(eventID): \(error)") }
            }
            XCTAssertGreaterThanOrEqual(boxes.count, 5)
            histories[sport, default: []].append(.init(team: team, roster: roster, boxes: boxes, partial: false))
        }
        for sport in InterestSport.allCases {
            let rows = try XCTUnwrap(histories[sport]); XCTAssertEqual(rows.count, 2)
            let result = try PlayerSimulation.run(away: rows[0], home: rows[1], now: now, seed: 55)
            XCTAssertEqual(result.trials.count, 5000)
            XCTAssertFalse(result.away.boxes[0].lines.isEmpty)
            XCTAssertFalse(result.home.boxes[0].lines.isEmpty)
            let counts = Set(result.away.boxes.flatMap(\.lines).map(\.group))
            XCTAssertTrue(counts.contains(PlayerSimulation.groups(sport)[0].id))
        }
    }
    func testCapturedParserRejectsTamperingAndKeepsMissingFieldsMissing() throws {
        let directory = try fixtureDirectory(), data = try Data(contentsOf: directory.appendingPathComponent("NBA-summary.json"))
        let (box, team) = try parseFixture(data, sport: .nba, teamID: "1")
        var root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var header = root["header"] as! [String: Any]; header["id"] = "999"; root["header"] = header
        XCTAssertThrowsError(try PlayerSimulationSource.parse(JSONSerialization.data(withJSONObject: root), team: team, game: box.game, cutoff: now))
        XCTAssertThrowsError(try PlayerSimulationSource.parse(data, team: team, game: box.game, cutoff: box.game.date))
        root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var score = root["boxscore"] as! [String: Any], sides = score["players"] as! [[String: Any]]
        var categories = sides[0]["statistics"] as! [[String: Any]], players = categories[0]["athletes"] as! [[String: Any]]
        var stats = players[0]["stats"] as! [String]; stats[1] = "999"; players[0]["stats"] = stats; categories[0]["athletes"] = players; sides[0]["statistics"] = categories; score["players"] = sides; root["boxscore"] = score
        XCTAssertThrowsError(try PlayerSimulationSource.parse(JSONSerialization.data(withJSONObject: root), team: team, game: box.game, cutoff: now))
    }
}
