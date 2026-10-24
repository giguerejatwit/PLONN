import XCTest
@testable import PLONNCore

final class SimulationScoringTests: XCTestCase {
    func team(_ id: String, offensiveRebounds: Double) -> SportPlaySimulation.Team {
        let identity = InterestTeam(sport: .nba, providerID: id, name: "Team " + id, abbreviation: "T" + id)
        let players = (1...5).map { PlayerMatchupPlayer(id: id + String($0), name: "Player \($0)", teamID: id, team: identity.name, position: "G") }
        let means = ["minutes":48.0, "fieldGoalsAttempted":20, "fieldGoalsMade":10, "threePointFieldGoalsAttempted":0, "threePointFieldGoalsMade":0, "freeThrowsAttempted":4, "freeThrowsMade":3, "turnovers":3, "offensiveRebounds":offensiveRebounds, "rebounds":10, "assists":5, "steals":1, "blocks":1, "points":23]
        let stats = Dictionary(uniqueKeysWithValues: means.map { ("players." + $0.key, $0.value) })
        let profiles = players.map { SportPlaySimulation.Profile(player: $0, means: stats, totals: stats.mapValues { $0 * 10 }, samples: stats.mapValues { _ in 10 }) }
        return .init(identity: identity, profiles: profiles, availability: .init(roster: players, observations: [], checked: Date(), injuryFeedUpdated: nil, injuriesVerified: false), missingHistory: [], possessions: 5 * (20 + 0.44 * 4 + 3 - offensiveRebounds))
    }
    func testScoringPreservesSecondChancesAndFreeThrowVolume() throws {
        for boards in [0.0, 3.0] {
            let away = team("1", offensiveRebounds: boards), home = team("2", offensiveRebounds: boards)
            var points = 0.0, attempts = 0.0, freeThrows = 0.0, rebounds = 0.0
            for index in 0..<1000 {
                let game = try SportPlaySimulation.run(away: away, home: home, seed: PlayerSimulation.trialSeed(456, index), recordPlays: false)
                points += Double(game.awayScore + game.homeScore)
                for line in game.away + game.home {
                    attempts += line.values["fieldGoalsAttempted"] ?? 0
                    freeThrows += line.values["freeThrowsAttempted"] ?? 0
                    rebounds += line.values["offensiveRebounds"] ?? 0
                }
            }
            // Independent expectation from the input rates, not a fitted game score.
            XCTAssertEqual(points / 2000, 115, accuracy: 4)
            XCTAssertEqual(attempts / 2000, 100, accuracy: 4)
            XCTAssertEqual(freeThrows / 2000, 20, accuracy: 2)
            if boards > 0 { XCTAssertGreaterThan(rebounds, 20_000) }
            else { XCTAssertEqual(rebounds, 0) }
        }
    }
    func testTrialStreamsDoNotOverlapByOneRandomDraw() {
        var previous = SportPlaySimulation.Random(state: PlayerSimulation.trialSeed(22, 0))
        _ = previous.uniform()
        var next = SportPlaySimulation.Random(state: PlayerSimulation.trialSeed(22, 1))
        XCTAssertNotEqual(previous.uniform(), next.uniform())
        XCTAssertEqual(Set((0..<5000).map { PlayerSimulation.trialSeed(22, $0) }).count, 5000)
    }
}
