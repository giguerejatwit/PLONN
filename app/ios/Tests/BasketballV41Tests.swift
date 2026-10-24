import XCTest
@testable import PLONNCore

final class BasketballV41Tests: XCTestCase {
    private func fixture() throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("PLONN/CelticsPistonsSimulation.json")
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func decode(_ fixture: [String: Any]) throws -> BasketballSimulation {
        try BasketballSimulation.decode(JSONSerialization.data(withJSONObject: fixture))
    }
    func testExactV41PayloadProfilesAndOptionalHeadToHead() throws {
        let model = try BasketballSimulation.bundled()
        XCTAssertEqual(model.meta.modelVersion, "v4.1 experimental")
        XCTAssertEqual(model.meta.zoneOrder, ["rim", "mid", "three", "unlocated_two"])
        XCTAssertEqual(model.data.teams.map { $0.players.count }, [11, 11])
        XCTAssertEqual(model.replays.reduce(0) { $0 + $1.events.count }, 1860)
        XCTAssertEqual(model.meta.openingNight.currentSeasonGamesUsed, 0)
        XCTAssertEqual(model.meta.openingNight.residualCalibrationGames, 0)
        XCTAssertEqual(model.meta.headToHead?.usable_meeting_count, 11)
        XCTAssertEqual(model.meta.headToHead?.adjustment_by_team["2"] ?? 0, -0.6828395275, accuracy: 0.00001)
        for player in model.data.teams.flatMap(\.players) {
            XCTAssertEqual(player.zones.reduce(0, +), 1, accuracy: 1e-8)
            XCTAssertEqual(player.accuracy.count, 4)
        }
        var source = try fixture()
        var meta = try XCTUnwrap(source["meta"] as? [String: Any])
        meta.removeValue(forKey: "headToHead"); source["meta"] = meta
        let withoutDetails = try decode(source)
        XCTAssertNil(withoutDetails.meta.headToHead)
        XCTAssertEqual(withoutDetails.result.score_pairs, model.result.score_pairs)
        XCTAssertEqual(withoutDetails.replays.map(\.score), model.replays.map(\.score))
    }
    func testAllEightFinalBoxScoresAndMinutesMatchSuppliedRecords() throws {
        let model = try BasketballSimulation.bundled()
        for replay in model.replays {
            let live = replay.counts(completed: replay.events.count)
            for (id, exported) in replay.box {
                let line = live[id, default: .init()]
                XCTAssertEqual(line.points, exported.pts)
                XCTAssertEqual(line.rebounds, exported.reb)
                XCTAssertEqual(line.assists, exported.ast)
                XCTAssertEqual(line.turnovers, exported.tov)
                XCTAssertEqual(line.made, exported.fgm)
                XCTAssertEqual(line.attempts, exported.fga)
                XCTAssertEqual(line.threesMade, exported.tpm)
                XCTAssertEqual(line.threesAttempted, exported.tpa)
                XCTAssertEqual(line.freeThrowsMade, exported.ftm)
                XCTAssertEqual(line.freeThrowsAttempted, exported.fta)
                XCTAssertEqual(line.minutes, exported.minutes, accuracy: 0.0051, "\(replay.label) / \(id)")
            }
            for team in model.data.teams {
                XCTAssertEqual(team.players.reduce(0) { $0 + live[$1.id, default: .init()].minutes }, 240, accuracy: 1e-8)
            }
        }
    }
    func testPartialPlaybackCommitsScoreBoxAndMinutesAtSameBoundary() throws {
        let replay = try BasketballSimulation.bundled().replays[0]
        let index = try XCTUnwrap(replay.events.firstIndex { $0.points > 0 })
        let event = replay.events[index]
        let before = replay.frame(at: Double(index) + 0.999)
        let after = replay.frame(at: Double(index + 1))
        let countsBefore = replay.counts(completed: before.completed)
        let countsAfter = replay.counts(completed: after.completed)
        XCTAssertEqual(before.score, event.score_before)
        XCTAssertEqual(after.score, event.score)
        XCTAssertEqual(countsAfter[event.player, default: .init()].points - countsBefore[event.player, default: .init()].points, event.points)
        for id in event.lineups.flatMap({ $0 }) {
            XCTAssertEqual(countsAfter[id, default: .init()].seconds - countsBefore[id, default: .init()].seconds, event.duration)
        }
        // Repeated forward/backward seeks and restart are pure reconstruction, with no doubled counts.
        for position in [Double(replay.events.count), 0.0, Double(index + 1), Double(index) + 0.5] {
            _ = replay.counts(completed: replay.frame(at: position).completed)
        }
        XCTAssertEqual(replay.counts(completed: before.completed), countsBefore)
        XCTAssertEqual(replay.counts(completed: 0), [:])
    }
    func testUnlocatedShotsAndAggregateFreeThrowsHaveNoFabricatedShotPath() throws {
        let model = try BasketballSimulation.bundled()
        let events = model.replays.flatMap(\.events)
        let unknown = events.filter { $0.zone == "unlocated_two" }
        XCTAssertFalse(unknown.isEmpty)
        for event in unknown {
            XCTAssertFalse(event.hasIllustratedShotLocation)
            XCTAssertEqual(event.points, event.made ? 2 : 0)
            XCTAssertTrue(model.description(event).contains("location unknown"))
        }
        for event in events where event.kind == "free throws" {
            XCTAssertFalse(event.hasIllustratedShotLocation)
            XCTAssertTrue(model.description(event).contains("/2 free throws"))
            XCTAssertEqual(event.made, event.points > 0)
        }
    }
    func testOvertimeClockUsesFiveMinutesAndCompletedPrefix() throws {
        let original = try BasketballSimulation.bundled().replays[0]
        let source = original.events[0]
        let overtime = BasketballSimulation.Event(period: 5, clock: 291, duration: 9, team: source.team, points: 0,
            player: source.player, kind: "turnover", zone: nil, made: false, assist: nil, rebound: nil,
            offensive_rebound: false, score_before: [0, 0], score: [0, 0], lineups: source.lineups, free_throws: nil)
        let replay = BasketballSimulation.Replay(label: "OT clock test", seed: 0, score: [0, 0], events: [overtime], box: [:])
        XCTAssertEqual(replay.frame(at: 0).clock, "5:00")
        XCTAssertEqual(replay.frame(at: 0.5).seconds, 295.5)
        XCTAssertEqual(replay.frame(at: 0.5).periodLabel, "OT1")
        XCTAssertEqual(replay.frame(at: 0.5).completed, 0)
    }
    func testObservedResultCannotInfluenceSimulationAndUnknownMetadataSurvives() throws {
        let original = try BasketballSimulation.bundled()
        var source = try fixture(), meta = try XCTUnwrap(try fixture()["meta"] as? [String: Any])
        meta["actualScore"] = [777, 888]
        meta["futureAudit"] = ["nested": ["value": NSNull(), "note": "retained"]]
        source["meta"] = meta
        let changed = try decode(source)
        XCTAssertEqual(changed.result.score_pairs, original.result.score_pairs)
        XCTAssertEqual(changed.replays.map(\.score), original.replays.map(\.score))
        XCTAssertEqual(changed.replays[0].frame(at: 0).score, [0, 0])
        XCTAssertEqual(changed.meta.provenance["futureAudit"], .object(["nested": .object(["value": .null, "note": .string("retained")])]))
        XCTAssertEqual(changed.data.teams[0].players[0].adaptiveHistory["historical_prior_source_sha256"], .null)
    }
    func testBadZonesBoxMinutesReboundTeamsAndFreeThrowCountsAreRejected() throws {
        for mutation in ["zone", "minutes", "rebound", "freeThrows", "clockGap", "shotPoints"] {
            var source = try fixture()
            var replays = try XCTUnwrap(source["replays"] as? [[String: Any]])
            var events = try XCTUnwrap(replays[0]["events"] as? [[String: Any]])
            switch mutation {
            case "zone": events[0]["zone"] = "invented-zone"
            case "minutes":
                var box = try XCTUnwrap(replays[0]["box"] as? [String: [String: Any]])
                let id = try XCTUnwrap(box.keys.first); box[id]?["minutes"] = 200
                replays[0]["box"] = box
            case "rebound":
                let index = try XCTUnwrap(events.firstIndex { $0["rebound"] is String })
                let team = try XCTUnwrap(events[index]["team"] as? Int)
                let offensive = try XCTUnwrap(events[index]["offensive_rebound"] as? Bool)
                let lineups = try XCTUnwrap(events[index]["lineups"] as? [[String]])
                events[index]["rebound"] = lineups[offensive ? 1 - team : team][0]
            case "freeThrows":
                let index = try XCTUnwrap(events.firstIndex { $0["kind"] as? String == "free throws" })
                events[index]["free_throws"] = 3
            case "clockGap": events[0]["clock"] = 710
            default: events[0]["points"] = 1
            }
            replays[0]["events"] = events; source["replays"] = replays
            XCTAssertThrowsError(try decode(source), mutation)
        }
    }
}
