import XCTest
@testable import PLONNCore

final class RepresentativeScenarioTests: XCTestCase {
    func testMostFrequentOutcomeSelectsItsRepresentativeNotAnAlternateWinner() throws {
        let scores: [MatchupSimulation.Score] = [.init(away: 1, home: 2), .init(away: 3, home: 5), .init(away: 2, home: 3), .init(away: 100, home: 0)]
        let scenario = try XCTUnwrap(MatchupSimulation.representative(in: scores))
        XCTAssertEqual(scenario.outcome, .home)
        XCTAssertEqual(scenario.share, 0.75)
        XCTAssertEqual(scenario.index, 2)
        XCTAssertFalse(scenario.tiedForTop)
    }
    func testMissingReplayForTopOutcomeDoesNotSubstituteLessLikelyWinner() {
        let scores = [MatchupSimulation.Score(away: 1, home: 3), .init(away: 2, home: 4)]
        XCTAssertNil(MatchupSimulation.representative(in: scores, candidates: [.init(away: 3, home: 1)]))
        XCTAssertNil(MatchupSimulation.representative(in: []))
    }
    func testTiesDoNotInventAnAdvantageOrOvertimeWinner() throws {
        let even = try XCTUnwrap(MatchupSimulation.representative(in: [.init(away: 3, home: 1), .init(away: 1, home: 3)]))
        XCTAssertTrue(even.tiedForTop)
        XCTAssertEqual(even.share, 0.5)
        let level = try XCTUnwrap(MatchupSimulation.representative(in: [.init(away: 1, home: 1), .init(away: 2, home: 2), .init(away: 1, home: 3)]))
        XCTAssertEqual(level.outcome, .level)
        XCTAssertEqual(level.share, 2.0 / 3.0)
        XCTAssertFalse(level.tiedForTop)
    }
    func testNBAReplayRepresentsHighestSampledWinShareWithoutChangingItsEvents() throws {
        let model = try BasketballSimulation.bundled()
        let scenario = try XCTUnwrap(model.representativeScenario)
        XCTAssertEqual(scenario.outcome, .home)
        XCTAssertEqual(scenario.share, 0.6374, accuracy: 1e-8)
        let replay = model.replays[scenario.index]
        XCTAssertGreaterThan(replay.score[1], replay.score[0])
        XCTAssertEqual(replay.frame(at: 1e6).score, replay.score)
        XCTAssertEqual(model.replays.count, 8) // Preserve the supplied source; only the UI is narrowed.
    }
}
