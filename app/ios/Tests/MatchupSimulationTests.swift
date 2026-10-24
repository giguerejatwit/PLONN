import XCTest
@testable import PLONNCore

final class MatchupSimulationTests: XCTestCase {
    let now = ISO8601DateFormatter().date(from: "2026-10-23T19:00:00Z")!
    func team(_ sport: InterestSport, _ id: String = "1") -> InterestTeam {
        .init(sport: sport, providerID: id, name: "Team " + id, abbreviation: "T" + id)
    }
    func history(_ sport: InterestSport, _ id: String, scored: Int, allowed: Int, count: Int = 20) -> MatchupSimulation.History {
        var games: [MatchupSimulation.Game] = []
        for i in 0..<count {
            let elapsed = Double(i + 1) * -86400.0
            let date = now.addingTimeInterval(elapsed)
            let game = MatchupSimulation.Game(id: String(i + 100), date: date, season: 2026, scored: scored, allowed: allowed, home: i.isMultiple(of: 2))
            games.append(game)
        }
        return .init(team: team(sport, id), games: games, partial: false)
    }
    func testAllSportsUseSelectedTeamsAndTheirScores() throws {
        for (sport, score) in [(InterestSport.nfl, 21), (.nba, 110), (.mlb, 4), (.nhl, 3)] {
            let a = history(sport, "1", scored: score, allowed: score + 1)
            let h = history(sport, "2", scored: score + 1, allowed: score)
            let result = try MatchupSimulation.run(away: a, home: h, now: now, seed: 987)
            XCTAssertEqual(result.scores.count, 5000)
            XCTAssertTrue(result.scores.allSatisfy { $0.away == score && $0.home == score + 1 })
            XCTAssertEqual(result.homeAhead, 1)
            XCTAssertEqual(result.awayAhead + result.homeAhead + result.level, 1, accuracy: 1e-12)
            XCTAssertEqual(result.away.team.sport, sport)
            XCTAssertEqual(result.away.team.providerID, "1")
            XCTAssertEqual(result.home.team.providerID, "2")
            XCTAssertEqual(result.awaySummary.low, score)
            XCTAssertEqual(result.homeSummary.high, score + 1)
            let replacement = history(sport, "3", scored: score + 5, allowed: score + 8)
            let changed = try MatchupSimulation.run(away: a, home: replacement, now: now, seed: 987)
            XCTAssertNotEqual(result.scores, changed.scores)
            XCTAssertEqual(changed.home.team.providerID, "3")
            XCTAssertGreaterThan(changed.awaySummary.average, result.awaySummary.average)
        }
        XCTAssertEqual(MatchupSimulation.unit(.nhl), "goals")
        XCTAssertEqual(MatchupSimulation.unit(.mlb), "runs")
    }
    func testMixturePreservesVarianceAndLevelScoresAreNotAssignedAWinner() throws {
        let a = history(.nfl, "1", scored: 10, allowed: 20)
        let h = history(.nfl, "2", scored: 10, allowed: 20)
        let result = try MatchupSimulation.run(away: a, home: h, now: now, seed: 123)
        XCTAssertEqual(Set(result.scores.map(\.away)), [10, 20])
        XCTAssertEqual(result.awaySummary.average, 15, accuracy: 0.3)
        XCTAssertEqual(result.level, 0.5, accuracy: 0.03)
        XCTAssertEqual(result.homeAhead + result.awayAhead + result.level, 1, accuracy: 1e-12)
        XCTAssertEqual(result.scores, try MatchupSimulation.run(away: a, home: h, now: now, seed: 123).scores)
    }
    func testSelectionInvalidatesOldResultsAndCannotPairAcrossSports() {
        var selection = SimulationSelection()
        for sport in InterestSport.allCases {
            let before = selection.revision
            selection.changeSport(sport)
            XCTAssertNotEqual(selection.revision, before)
            XCTAssertNil(selection.away); XCTAssertNil(selection.home); XCTAssertNil(selection.event)
            XCTAssertFalse(selection.ready)
            selection.choose(team(sport), homeSide: false)
            selection.choose(team(sport, "2"), homeSide: true)
            XCTAssertTrue(selection.ready)
            let revision = selection.revision
            selection.choose(team(sport, "3"), homeSide: false)
            XCTAssertNotEqual(revision, selection.revision)
            XCTAssertEqual(selection.away?.providerID, "3")
            selection.swap()
            XCTAssertEqual(selection.home?.providerID, "3")
            XCTAssertEqual(selection.away?.providerID, "2")
            selection.choose(team(sport, "2"), homeSide: true)
            XCTAssertNotEqual(selection.home?.id, selection.away?.id)
        }
        selection.changeSport(.nba)
        selection.choose(team(.nfl), homeSide: true)
        XCTAssertNil(selection.home)
    }
    func testUpcomingSelectionSetsBothTeamsAndCustomEditClearsSchedule() {
        var selection = SimulationSelection()
        let event = PlayerMatchupEvent(game: .init(league: "NFL", eventID: "401000001", home: "Team 2", away: "Team 1", startsAt: "2026-10-24T20:00Z"), homeID: "2", awayID: "1")
        selection.choose(event, teams: [team(.nfl), team(.nfl, "2")])
        XCTAssertEqual(selection.event, event); XCTAssertTrue(selection.ready)
        selection.choose(team(.nfl, "3"), homeSide: true)
        XCTAssertNil(selection.event)
        selection.changeSport(.nhl)
        selection.choose(event, teams: [team(.nfl), team(.nfl, "2")])
        XCTAssertFalse(selection.ready); XCTAssertNil(selection.event)
    }
    func testInvalidSparseStaleAndFutureHistoryCannotProduceResults() throws {
        let a = history(.nfl, "1", scored: 14, allowed: 21)
        XCTAssertThrowsError(try MatchupSimulation.run(away: a, home: a, now: now, seed: 0))
        XCTAssertThrowsError(try MatchupSimulation.run(away: a, home: history(.nba, "2", scored: 100, allowed: 90), now: now, seed: 0))
        XCTAssertThrowsError(try MatchupSimulation.run(away: a, home: history(.nfl, "2", scored: 14, allowed: 21, count: 4), now: now, seed: 0))
        let h = history(.nfl, "2", scored: 14, allowed: 21)
        for delta in [300.0, -600.0] {
            let shifted = MatchupSimulation.History(team: h.team, games: h.games.map { .init(id: $0.id, date: now.addingTimeInterval(delta * 86400), season: 2026, scored: $0.scored, allowed: $0.allowed, home: $0.home) }, partial: false)
            XCTAssertThrowsError(try MatchupSimulation.run(away: a, home: shifted, now: now, seed: 0))
        }
        let duplicate = MatchupSimulation.History(team: h.team, games: h.games + [h.games[0]], partial: false)
        XCTAssertThrowsError(try MatchupSimulation.run(away: a, home: duplicate, now: now, seed: 0))
        XCTAssertThrowsError(try MatchupSimulation.run(away: a, home: h, now: now, seed: 0, count: 100_000))
    }
    func fixture(completed: Bool = true, state: String = "post", date: String = "2026-10-20T20:00Z", score: Any = 21) -> [String: Any] {
        ["team": ["id": "1"], "requestedSeason": ["year": 2026, "type": 2], "events": [
            ["id": "401000001", "date": date, "seasonType": ["type": 2], "competitions": [
                ["id": "401000001", "status": ["type": ["completed": completed, "state": state]], "competitors": [
                    ["id": "1", "homeAway": "away", "team": ["id": "1"], "score": ["value": score]],
                    ["id": "2", "homeAway": "home", "team": ["id": "2"], "score": ["value": 14]]
                ]]
            ]]
        ]]
    }
    func testSourceRejectsWrongIdentitySeasonInvalidScoresAndUnfinishedGames() throws {
        func parse(_ object: [String: Any]) throws -> [MatchupSimulation.Game] {
            try MatchupSimulationSource.parse(JSONSerialization.data(withJSONObject: object), team: team(.nfl), season: 2026, phase: 2, cutoff: now)
        }
        XCTAssertEqual(try parse(fixture()).first?.scored, 21)
        XCTAssertTrue(try parse(fixture(completed: false, state: "in")).isEmpty)
        XCTAssertTrue(try parse(fixture(date: "2026-10-24T20:00Z")).isEmpty)
        for score: Any in [-1, 2.5, true, 999] { XCTAssertThrowsError(try parse(fixture(score: score))) }
        var invalid = fixture(); invalid["team"] = ["id": "2"]
        XCTAssertThrowsError(try parse(invalid))
        invalid = fixture(); invalid["requestedSeason"] = ["year": 2025, "type": 2]
        XCTAssertThrowsError(try parse(invalid))
        invalid = fixture(); invalid["events"] = (invalid["events"] as! [[String: Any]]) + (invalid["events"] as! [[String: Any]])
        XCTAssertThrowsError(try parse(invalid))
    }
    func testCapturedProviderDataLoadsAllTeamsAndRunsTwoTeamsInEverySport() async throws {
        guard let path = ProcessInfo.processInfo.environment["PLONN_SIMULATOR_FIXTURE_DIRECTORY"] else { throw XCTSkip("Set PLONN_SIMULATOR_FIXTURE_DIRECTORY to the captured public schedules") }
        let root = URL(fileURLWithPath: path)
        for sport in InterestSport.allCases {
            let teams = try InterestsTeamDirectory.parse(Data(contentsOf: root.appendingPathComponent(sport.rawValue + "-teams.json")), sport: sport)
            XCTAssertEqual(teams.count, [.nfl, .nhl].contains(sport) ? 32 : 30)
            let source = MatchupSimulationSource(read: { url in
                let parts = url.path.components(separatedBy: "/")
                let id = parts[parts.count - 2]
                let query = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
                let season = query.first { $0.name == "season" }!.value!
                let phase = query.first { $0.name == "seasontype" }!.value!
                return try Data(contentsOf: root.appendingPathComponent("\(sport.rawValue)-\(id)-\(season)-\(phase).json"))
            })
            let pair: [InterestTeam]
            // Fixtures capture source directory order; the production directory is sorted.
            let ids: [String] = sport == .nfl ? ["22", "1"] : sport == .nba ? ["1", "2"] : sport == .mlb ? ["29", "11"] : ["25", "1"]
            pair = try ids.map { id in try XCTUnwrap(teams.first { $0.providerID == id }) }
            let a = try await source.history(pair[0], cutoff: now)
            let h = try await source.history(pair[1], cutoff: now)
            XCTAssertFalse(a.partial, "\(sport) away history unexpectedly incomplete")
            XCTAssertFalse(h.partial, "\(sport) home history unexpectedly incomplete")
            let result = try MatchupSimulation.run(away: a, home: h, now: now, seed: 23)
            XCTAssertEqual(result.scores.count, 5000)
            XCTAssertTrue(result.away.games.allSatisfy { $0.date < now })
            XCTAssertEqual(result.away.team, pair[0]); XCTAssertEqual(result.home.team, pair[1])
            XCTAssertGreaterThan(result.awaySummary.average, 0)
            print("SIMULATION_ACCEPTANCE \(sport.rawValue) \(pair[0].name) \(result.awaySummary.average) / \(pair[1].name) \(result.homeSummary.average) history=\(result.away.games.count),\(result.home.games.count)")
        }
    }
}
