import XCTest
@testable import PLONNCore

final class ReplayPresentationTests: XCTestCase {
    func testFourSportsKeepVerifiedIdentitiesAndContinuousPositions() throws {
        let fixtures = try CurrentRosterSimulationTests().fixtures()
        for sport in InterestSport.allCases {
            let teams = try fixtures.filter { $0.0.team.sport == sport }.map { try SportPlaySimulation.team($0.0, availability: $0.1) }
            let game = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: 22)
            var previous: SportPlaySimulation.Play?
            var last: SportReplayMotion.Scene?
            for play in game.plays {
                func scene(_ t: Double, reduced: Bool = false) -> SportReplayMotion.Scene {
                    SportReplayMotion.scene(sport: sport, teams: teams, play: play, previous: previous, progress: t, reduceMotion: reduced)
                }
                let start = scene(0)
                if let last {
                    for marker in start.markers {
                        if let old = last.markers.first(where: { $0.id == marker.id }) {
                            XCTAssertEqual(marker.point.x, old.point.x, accuracy: 1e-8)
                            XCTAssertEqual(marker.point.y, old.point.y, accuracy: 1e-8)
                        }
                    }
                    let startBall = try XCTUnwrap(start.ball), lastBall = try XCTUnwrap(last.ball)
                    XCTAssertEqual(startBall.x, lastBall.x, accuracy: 1e-8)
                    XCTAssertEqual(startBall.y, lastBall.y, accuracy: 1e-8)
                }
                for t in [0.0, 0.25, 0.55, 0.8, 1.0] {
                    let frame = scene(t)
                    XCTAssertNotNil(frame.ball)
                    XCTAssertEqual(Set(frame.markers.map(\.id)).count, frame.markers.count)
                    if sport == .nba { XCTAssertEqual(frame.markers.count, 10) }
                    for marker in frame.markers {
                        XCTAssertTrue((0...1).contains(marker.point.x) && (0...1).contains(marker.point.y))
                        if let id = marker.playerID { XCTAssertTrue(teams[marker.team].availability.eligible.contains { $0.id == id }) }
                    }
                    XCTAssertTrue(frame.ball?.x.isFinite == true && frame.ball?.y.isFinite == true)
                }
                XCTAssertEqual(scene(0, reduced: true), scene(0.8, reduced: true), "Reduce Motion must remain still within each play")
                last = scene(1); previous = play
            }
            // Rendering and seeking cannot change the ledger that drives scores.
            let completed = game.through(game.plays.count)
            XCTAssertEqual(completed.away, game.away); XCTAssertEqual(completed.home, game.home)
            XCTAssertEqual(completed.awayScore, game.awayScore); XCTAssertEqual(completed.homeScore, game.homeScore)
        }
    }
    func testHistogramPreservesAllRunsAndMedianIncludingLowScoringSports() {
        for margins in [[], [0], [-1, 0, 1], [-2, -2, -1, 0, 1, 3], Array(-100...100), Array(repeating: 14, count: 5000)] {
            let histogram = MatchupSimulation.MarginHistogram(margins)
            XCTAssertEqual(histogram.bins.reduce(0) { $0 + $1.count }, margins.count)
            XCTAssertLessThanOrEqual(histogram.bins.count, 27)
            XCTAssertTrue(histogram.bins.allSatisfy { abs($0.center) < histogram.extent })
        }
        XCTAssertEqual(MatchupSimulation.MarginHistogram([-2, -1, 0, 1]).median, -0.5)
        XCTAssertEqual(MatchupSimulation.MarginHistogram([-2, -1, 0, 1, 2]).width, 1)
    }
}
