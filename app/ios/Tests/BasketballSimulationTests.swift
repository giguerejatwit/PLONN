import XCTest
@testable import PLONNCore

final class BasketballSimulationTests:XCTestCase {
    func testBundledScenariosAndEnsembleAgree() throws {
        let model=try BasketballSimulation.bundled()
        XCTAssertEqual(model.replays.count,8)
        XCTAssertEqual(model.result.runs,5000)
        XCTAssertEqual(model.data.teams.map(\.abbrev),["BOS","DET"])
        let scores=model.result.score_pairs
        XCTAssertEqual(Double(scores.filter{$0[0]>$0[1]}.count)/Double(scores.count),model.result.win_probability,accuracy:1e-8)
        for team in 0...1 {
            XCTAssertEqual(Double(scores.reduce(0){$0+$1[team]})/Double(scores.count),model.result.mean_score[team],accuracy:1e-8)
        }
        for replay in model.replays {
            let counts=replay.counts(completed:replay.events.count)
            XCTAssertEqual(replay.frame(at:Double(replay.events.count)).score,replay.score)
            for team in 0...1 {
                XCTAssertEqual(model.data.teams[team].players.reduce(0){$0+counts[$1.id,default:.init()].points},replay.score[team],replay.label)
            }
        }
    }
    func testTimelineNeverLeaksFutureScoresAndSupportsBackwardScrubbing() throws {
        let replay=try BasketballSimulation.bundled().replays[0]
        XCTAssertEqual(replay.frame(at:-100).score,[0,0])
        XCTAssertEqual(replay.frame(at:.nan).completed,0)
        XCTAssertEqual(replay.frame(at:.infinity).completed,0)
        for index in replay.events.indices {
            let before=replay.frame(at:Double(index)+0.9)
            XCTAssertEqual(before.completed,index)
            XCTAssertEqual(before.score,replay.events[index].score_before)
            let after=replay.frame(at:Double(index)+1)
            XCTAssertEqual(after.score,replay.events[index].score)
        }
        XCTAssertTrue(replay.frame(at:1e9).finished)
        XCTAssertEqual(replay.frame(at:1e9).clock,"Final")
        let first=replay.events[0]
        _=replay.counts(completed:replay.events.count)
        XCTAssertEqual(replay.counts(completed:1)[first.player]?.points,first.points)
        XCTAssertEqual(replay.counts(completed:0),[:])
        XCTAssertEqual(replay.frame(at:0).clock,"12:00")
    }
    func testBoxScoreTracksOnlyRecordedEvents() throws {
        let replay=try BasketballSimulation.bundled().replays[0]
        let counts=replay.counts(completed:replay.events.count)
        XCTAssertEqual(counts.values.reduce(0){$0+$1.assists},replay.events.filter{$0.assist != nil}.count)
        XCTAssertEqual(counts.values.reduce(0){$0+$1.rebounds},replay.events.filter{$0.rebound != nil}.count)
        XCTAssertEqual(counts.values.reduce(0){$0+$1.turnovers},replay.events.filter{$0.kind == "turnover"}.count)
        XCTAssertEqual(counts.values.reduce(0){$0+$1.attempts},replay.events.filter{$0.kind == "shot"}.count)
    }
    func testBrokenScoreRosterOrTimelineIsRejected() throws {
        let url=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("PLONN/CelticsPistonsSimulation.json")
        let original=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
        for key in ["score","score_before","player","lineups","clock"] {
            var fixture=original
            var replays=try XCTUnwrap(fixture["replays"] as? [[String:Any]])
            var events=try XCTUnwrap(replays[0]["events"] as? [[String:Any]])
            switch key {
            case "score","score_before":events[0][key]=[900,900]
            case "player":events[0][key]="unknown player"
            case "lineups":events[0][key]=[["unknown"],["unknown"]]
            default:events[0][key]=900
            }
            replays[0]["events"]=events;fixture["replays"]=replays
            XCTAssertThrowsError(try BasketballSimulation.decode(JSONSerialization.data(withJSONObject:fixture)),key)
        }
    }
}
