import XCTest
@testable import PLONNCore

final class UpcomingModelTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-10-19T12:00:00Z")!
    private func setup(_ league: String = "NFL") -> UpcomingGameSetup {
        .init(league: league, eventID: "123", home: "Home Team", away: "Away Team", startsAt: "2026-10-20T19:30:00Z")
    }
    private func event() -> [String: Any] {
        ["id": "123", "season": ["type": 2], "competitions": [["id": "123", "date": setup().startsAt,
          "status": ["type": ["state": "pre", "completed": false, "name": "STATUS_SCHEDULED"]],
          "competitors": [["homeAway": "home", "team": ["displayName": "Home Team"]],
                          ["homeAway": "away", "team": ["displayName": "Away Team"]]]]]]
    }
    private func result(_ setup: UpcomingGameSetup) throws -> [String: Any] {
        var body: [String: Any] = ["status": "ok", "evaluation_mode": setup.evaluationMode, "upcoming_prediction": true, "forecast_kind": "home_win",
         "setup": try setup.payload(), "generated_at": "2026-10-19T11:00:00Z",
         "inference": ["status": "ok", "probabilities": ["win": 0.55, "not_win": 0.45],
            "target": ["event_id": setup.eventID, "league": setup.league, "home_team_name": setup.home,
                       "away_team_name": setup.away, "selected_team": "home", "starts_at": setup.startsAt, "season_type": 2]],
         "training": ["status": "ok", "sample_size": 200],
         "validation": ["status": "ok", "sample_size": 60,
            "metrics": ["accuracy": 0.6, "multiclass_brier": 0.4, "rating_baseline": ["accuracy": 0.55, "multiclass_brier": 0.45], "baseline": ["accuracy": 0.5, "multiclass_brier": 0.5]]]]
        if setup.evaluationMode == UpcomingGameSetup.mode {
            body["availability"] = ["version": 1, "event_id": setup.eventID, "state": "limited", "probability_adjusted": false,
                "historical_training_available": false, "checked_at": "2026-10-19T11:00:00Z", "expires_at": "2026-10-19T11:30:00Z",
                "injuries": [setup.home, setup.away].map { ["team": $0, "state": "missing", "players": [], "excluded": 0] as [String: Any] },
                "lineups": [setup.home, setup.away].map { ["team": $0, "state": "unknown", "players": [], "detail": "Not confirmed"] as [String: Any] }]
            body["opponent_strength"] = ["home": ["team": setup.home, "rating": 1520, "games": 40, "last_game": "2026-10-15"],
                                          "away": ["team": setup.away, "rating": 1480, "games": 40, "last_game": "2026-10-15"]]
        }
        return body
    }
    func testAllSportsPayloadHasNoHistoryOrUserSelectedTraining() throws {
        for sport in ["NBA", "NFL", "MLB", "NHL"] {
            let game = setup(sport), payload = try game.payload()
            XCTAssertEqual(Set(payload.keys), ["evaluation_mode", "league", "event_id", "home_team", "away_team", "starts_at"])
            XCTAssertEqual(try JSONDecoder().decode(UpcomingGameSetup.self, from: JSONEncoder().encode(game)), game)
            XCTAssertEqual(try game.forecast(result(game), now: now).win, 0.55)
        }
    }
    func testScheduledGamesRejectStartedLiveAndPreseason() throws {
        let good = event()
        func parsed(_ event: [String: Any], now: Date? = nil) throws -> [UpcomingGameSetup] {
            try UpcomingGameSetup.scheduledGames(JSONSerialization.data(withJSONObject: ["events": [event]]), league: "NFL", now: now ?? self.now)
        }
        XCTAssertEqual(try parsed(good).count, 1)
        XCTAssertTrue(try parsed(good, now: now.addingTimeInterval(3 * 86400)).isEmpty)
        var preseason = good; preseason["season"] = ["type": 1]
        XCTAssertTrue(try parsed(preseason).isEmpty)
        var live = good
        var competitions = live["competitions"] as! [[String: Any]]
        competitions[0]["status"] = ["type": ["state": "in", "name": "STATUS_IN_PROGRESS", "completed": false]]
        live["competitions"] = competitions
        XCTAssertTrue(try parsed(live).isEmpty)
        competitions[0]["status"] = ["type": ["state": "pre", "name": "STATUS_POSTPONED", "completed": false]]
        live["competitions"] = competitions
        XCTAssertTrue(try parsed(live).isEmpty)
    }
    func testMismatchedAndMalformedForecastsNeverDisplay() throws {
        let game = setup()
        var wrong = try result(game); wrong["setup"] = try setup("NBA").payload()
        XCTAssertThrowsError(try game.forecast(wrong, now: now))
        wrong = try result(game); wrong["generated_at"] = "2026-10-21T11:00:00Z"
        XCTAssertThrowsError(try game.forecast(wrong, now: now))
        wrong = try result(game); wrong["evaluation_mode"] = "fixed_threshold_retrospective"
        XCTAssertThrowsError(try game.forecast(wrong, now: now))
        for probabilities in [["win": 0.6, "not_win": 0.6], ["win": -0.1, "not_win": 1.1], ["win": 0.55, "loss": 0.45]] {
            wrong = try result(game)
            var inference = wrong["inference"] as! [String: Any]; inference["probabilities"] = probabilities; wrong["inference"] = inference
            XCTAssertThrowsError(try game.forecast(wrong, now: now))
        }
        wrong = try result(game)
        var validation = wrong["validation"] as! [String: Any]; validation["target_observed_outcome"] = "win"; wrong["validation"] = validation
        XCTAssertThrowsError(try game.forecast(wrong, now: now))
    }
    func testSavedForecastPreservesOriginalTimestampAfterStart() throws {
        let game = setup()
        let saved = try game.forecast(result(game), now: now.addingTimeInterval(4 * 86400))
        XCTAssertEqual(saved.generated, now.addingTimeInterval(-3600))
    }
    func testLegacyForecastCanStillOpenWithoutAvailability() throws {
        var old = setup(); old.evaluationMode = "upcoming_game_v1"
        XCTAssertNil(try old.forecast(result(old), now: now).availability)
        XCTAssertEqual(try old.payload()["evaluation_mode"] as? String, "upcoming_game_v1")
    }
    func testEnhancedForecastRejectsMissingMismatchedOrOverstatedAvailability() throws {
        let game = setup()
        var bad = try result(game); bad.removeValue(forKey: "availability")
        XCTAssertThrowsError(try game.forecast(bad, now: now))
        for (key,value) in [("event_id", "999" as Any), ("probability_adjusted", true as Any), ("checked_at", "2026-10-19T12:00:00Z" as Any)] {
            bad = try result(game)
            var context = bad["availability"] as! [String: Any]; context[key] = value; bad["availability"] = context
            XCTAssertThrowsError(try game.forecast(bad, now: now))
        }
        let current = try game.forecast(result(game), now: now)
        XCTAssertLessThan(current.availability!.expires!, now)
        XCTAssertEqual(current.strengths.count, 2)
    }
    func testUpcomingErrorsUseActionableLanguageWithoutInternalIDs() {
        XCTAssertTrue(UpcomingGameSetup.failure(["reason": "upcoming_game_changed"]).contains("Refresh"))
        XCTAssertTrue(UpcomingGameSetup.failure(["reason": "game_no_longer_upcoming"]).contains("Choose another"))
        XCTAssertTrue(UpcomingGameSetup.failure(["reason": "insufficient_game_training_or_holdout_rows"]).contains("verified history"))
    }
}
