import XCTest
@testable import PLONNCore

final class SimulationLiveTests: XCTestCase {
    func snapshot(_ teams: [SportPlaySimulation.Team], game: SportPlaySimulation.Game, phase: SimulationLiveSnapshot.Phase = .live, period: Int = 3, remaining: Int = 300, outs: Int = 0) -> SimulationLiveSnapshot {
        var live = SimulationLiveSnapshot(eventID: "401999999", sport: teams[0].identity.sport, teamIDs: teams.map { $0.identity.providerID }, start: Date().addingTimeInterval(-3600), checked: Date(), phase: phase, status: "Test checkpoint", scores: [game.awayScore, game.homeScore], period: period, remaining: remaining, possession: 0, down: 2, distance: 5, yard: 45, battingSide: 0, outs: outs, bases: [nil,nil,nil], batter: nil, pitcher: nil, battingOrders: teams.map { team in Array(team.profiles.filter { $0.mean("batting", "atBats") > 0 }.sorted { $0.mean("batting", "atBats") > $1.mean("batting", "atBats") }.prefix(9)).map(\.id) }, lines: [game.away, game.home], events: [], fingerprint: "synthetic-checkpoint")
        live.activePitchers = teams.map { $0.profiles.filter { $0.mean("pitching", "fullInnings.partInnings") > 0 }.max { $0.mean("pitching", "fullInnings.partInnings") < $1.mean("pitching", "fullInnings.partInnings") }?.id }
        return live
    }
    func testAllSportsResumeObservedLedgerAndFinalCollapses() throws {
        let fixtures = try CurrentRosterSimulationTests().fixtures()
        for sport in InterestSport.allCases {
            let teams = try fixtures.filter { $0.0.team.sport == sport }.map { try SportPlaySimulation.team($0.0, availability: $0.1) }
            let full = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 91)
            let prefix = full.through(max(1, full.plays.count / 4))
            let live = snapshot(teams, game: prefix, period: sport == .mlb ? 7 : sport == .nhl ? 3 : 4)
            for seed in 1...8 {
                let game = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: UInt64(seed), live: live)
                XCTAssertGreaterThanOrEqual(game.awayScore, prefix.awayScore, sport.rawValue)
                XCTAssertGreaterThanOrEqual(game.homeScore, prefix.homeScore, sport.rawValue)
                let start = game.through(0), end = game.through(game.plays.count)
                XCTAssertEqual(start.awayScore, prefix.awayScore); XCTAssertEqual(start.homeScore, prefix.homeScore)
                for (side, lines) in [prefix.away, prefix.home].enumerated() { for line in lines { for (stat, value) in line.values {
                    XCTAssertEqual((side == 0 ? start.away : start.home).first { $0.id == line.id }?.values[stat], value)
                } } }
                XCTAssertEqual(end.away, game.away); XCTAssertEqual(end.home, game.home)
                XCTAssertEqual(end.awayScore, game.awayScore); XCTAssertEqual(end.homeScore, game.homeScore)
                XCTAssertTrue(game.plays.allSatisfy { $0.period >= live.period })
                let fast = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: UInt64(seed), recordPlays: false, live: live)
                XCTAssertEqual(fast.away, game.away); XCTAssertEqual(fast.home, game.home)
            }
            let final = snapshot(teams, game: full, phase: .final)
            let collapsed = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 999, live: final)
            XCTAssertEqual(collapsed.awayScore, full.awayScore); XCTAssertEqual(collapsed.homeScore, full.homeScore)
            XCTAssertEqual(collapsed.away, full.away); XCTAssertEqual(collapsed.home, full.home)
            XCTAssertTrue(collapsed.plays.isEmpty)
            XCTAssertFalse(final.isStale(at: Date().addingTimeInterval(1000)))
            XCTAssertTrue(live.isStale(at: Date().addingTimeInterval(1000)))
            let overtime = snapshot(teams, game: prefix, period: 10)
            XCTAssertNotNil(overtime.forecastIssue)
            XCTAssertThrowsError(try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 1, live: overtime))
        }
    }
    func testOutOfOrderStatesCannotRewindObservedGame() throws {
        let helper = SimulationScoringTests(), teams = [helper.team("1", offensiveRebounds:3),helper.team("2", offensiveRebounds:3)]
        let game = try SportPlaySimulation.run(away:teams[0],home:teams[1],seed:2)
        let earlier = snapshot(teams,game:game,period:3,remaining:600)
        let later = snapshot(teams,game:game,period:3,remaining:500)
        XCTAssertTrue(later.supersedes(earlier))
        let delayed = snapshot(teams,game:game,period:3,remaining:550)
        XCTAssertFalse(delayed.supersedes(later))
        let final = snapshot(teams,game:game,phase:.final)
        XCTAssertTrue(final.supersedes(later))
        XCTAssertFalse(snapshot(teams,game:game,period:4,remaining:0).supersedes(final))
    }
    func testZeroClockRequiresConfirmedFinalBeforeCollapsing() throws {
        let helper = SimulationScoringTests(), teams = [helper.team("1", offensiveRebounds: 3), helper.team("2", offensiveRebounds: 3)]
        let original = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 71)
        let live = snapshot(teams, game: original, period: 4, remaining: 0)
        XCTAssertNotNil(live.forecastIssue)
        XCTAssertThrowsError(try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 12, live: live))
        let final = snapshot(teams, game: original, phase: .final, period: 4, remaining: 0)
        let game = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 12, live: final)
        XCTAssertEqual(game.awayScore, original.awayScore); XCTAssertEqual(game.homeScore, original.homeScore); XCTAssertTrue(game.plays.isEmpty)
    }
    func fixture(_ sport: InterestSport) throws -> (Data, PlayerMatchupEvent) {
        let directory = try PlayerSimulationTests().fixtureDirectory()
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as! [[String: Any]]
        let row = try XCTUnwrap(manifest.first { $0["sport"] as? String == sport.rawValue })
        let id = (row["events"] as! [[String: Any]])[0]["id"]!
        let data = try Data(contentsOf: directory.appendingPathComponent("\(sport.rawValue)-\(id)-box.json"))
        let body = try JSONSerialization.jsonObject(with: data) as! [String: Any], header = body["header"] as! [String: Any], c = (header["competitions"] as! [[String: Any]])[0], sides = c["competitors"] as! [[String: Any]]
        let a = sides.first { $0["homeAway"] as? String == "away" }!, h = sides.first { $0["homeAway"] as? String == "home" }!
        let event = PlayerMatchupEvent(game: .init(league: sport.rawValue, eventID: header["id"] as! String, home: "Home", away: "Away", startsAt: c["date"] as! String), homeID: h["id"] as! String, awayID: a["id"] as! String, seasonType: 2)
        return (data, event)
    }
    func testRealFourSportPayloadsIdentityRevisionAndDeduplication() throws {
        for sport in InterestSport.allCases {
            let (data, event) = try fixture(sport)
            let now = event.game.start!.addingTimeInterval(20_000)
            let parsed = try SimulationLiveSource.parse(data, event: event, now: now)
            XCTAssertEqual(parsed.phase, .final); XCTAssertEqual(parsed.lines.count, 2)
            XCTAssertTrue(parsed.lines.allSatisfy { !$0.isEmpty }, sport.rawValue)
            XCTAssertEqual(Set(parsed.events.map(\.id)).count, parsed.events.count)
            if let last = parsed.events.last { XCTAssertEqual([last.awayScore,last.homeScore], parsed.scores, sport.rawValue) }
            let later = try SimulationLiveSource.parse(data, event: event, now: now.addingTimeInterval(60))
            XCTAssertEqual(parsed.fingerprint, later.fingerprint)
            var body = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            if let plays = body["plays"] as? [[String: Any]], let last = plays.last {
                body["plays"] = plays + [last]
                let duplicate = try SimulationLiveSource.parse(JSONSerialization.data(withJSONObject: body), event: event, now: now)
                XCTAssertEqual(duplicate.fingerprint, parsed.fingerprint)
            }
            var header = body["header"] as! [String: Any]; header["id"] = "bad-game"; body["header"] = header
            XCTAssertThrowsError(try SimulationLiveSource.parse(JSONSerialization.data(withJSONObject: body), event: event, now: now))
        }
    }
    func testLiveClockCorrectionsAndIncompleteState() throws {
        let (data, event) = try fixture(.nba)
        var body = try JSONSerialization.jsonObject(with: data) as! [String: Any], header = body["header"] as! [String: Any]
        var competitions = header["competitions"] as! [[String: Any]]
        // Synthetic live revision for parser validation, never a forecast/backtest input.
        competitions[0]["status"] = ["period":4,"clock":90,"displayClock":"1:30","type":["name":"STATUS_IN_PROGRESS","state":"in","completed":false]]
        header["competitions"] = competitions; body["header"] = header
        let now = event.game.start!.addingTimeInterval(10_000)
        let live = try SimulationLiveSource.parse(JSONSerialization.data(withJSONObject: body), event: event, now: now)
        XCTAssertEqual(live.phase, .live); XCTAssertEqual(live.remaining,90); XCTAssertNil(live.forecastIssue)
        var plays = body["plays"] as! [[String: Any]]
        plays[0]["text"] = "Official play description corrected"; body["plays"] = plays
        let corrected = try SimulationLiveSource.parse(JSONSerialization.data(withJSONObject: body), event: event, now: now)
        XCTAssertNotEqual(corrected.fingerprint, live.fingerprint)
        competitions[0]["status"] = ["period":4,"clock":-1,"type":["state":"in","completed":false]]
        header["competitions"] = competitions; body["header"] = header
        XCTAssertThrowsError(try SimulationLiveSource.parse(JSONSerialization.data(withJSONObject: body), event: event, now: now))
    }
}
